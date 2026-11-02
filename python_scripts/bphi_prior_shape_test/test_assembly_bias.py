"""Assembly adapters preserve old science code, paired draws, and covariance."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import assembly_bias as ab


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.config = ab.inputs.read_config(ab.HERE / 'configs/lrg3_s1_empirical_k010.json')

    def fixture(self):
        source = self.root / 'source'
        source.mkdir()
        rows, pairs, table = [], {}, []
        for i in range(100):
            params = dict(logM_cut=12.6, logM1=13.6, sigma=.3, alpha=.7, kappa=.5,
                          alpha_c=0., alpha_s=1., ic=.19)
            row = dict(z='1.1', ensemble='chain_s1', draw=i+10, source_row=i//2,
                       params=params, rejection='')
            rows.append(row)
            counts = (240000+100*i, 200000)
            pairs[i+10] = [dict(parameters=dict(params, Acent=0., Asat=0., Bcent=0., Bsat=0.),
                               ic=.19, ngal=n, status='ok') for n in counts]
            table.append(dict(z=1.1, ensemble='chain_s1', draw=i+10, source_row=i//2,
                n000=counts[0], n004=counts[1], bphi=2*np.log(counts[0]/counts[1])/np.log(.807952/.75),
                passes_count_cut=True))
        configs = {c: dict(sim_params=dict(sim_name=f'AbacusSummit_base_c{c}_ph000', z_mock=1.1),
                           HOD_params=dict(want_AB=True, LRG_params=dict(Bcent=0., Bsat=0.)),
                           clustering_params={}) for c in ('000', '004')}
        manifest = dict(settings={'ngal_min': 100000, 'reference_nbar': {'1.1': .000184}},
                        snapshots={'1.1': dict(configs=configs)}, rows=rows)
        ab.freeze_json(source / 'manifest.json', manifest)
        ab.freeze_json(source / 'topup_manifest.json', dict(rows=[]))
        (source / 'tables').mkdir()
        with (source / 'tables/bphi_samples.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        return source, rows, pairs

    @contextlib.contextmanager
    def prepared(self):
        source, rows, pairs = self.fixture()
        accepted = [(row, pairs[row['draw']]) for row in rows]
        with patch.object(ab, 'SOURCE_RUN', source), \
             patch.object(ab.results, 'load_plan', return_value={'rows': []}), \
             patch.object(ab.results, 'accepted_rows', return_value=accepted):
            directory, manifest = ab.prepare_responses(self.config, self.root / 'runs')
            yield directory, manifest, rows, pairs

    def test_reuses_vectors_and_fixed_amplitudes_without_deduplication(self):
        with self.prepared() as (directory, manifest, original, pairs):
            self.assertEqual(len(manifest['rows']), 100)
            self.assertEqual(len({r['source_row'] for r in manifest['rows']}), 50)
            for new, old in zip(manifest['rows'], original):
                self.assertEqual(new['params'], dict(old['params'], **ab.AB))
                self.assertEqual(new['draw'], old['draw'])
                self.assertNotIn('Bcent', old['params'])
            for c in ('000', '004'):
                cfg = ab.run_counts.load_snapshot_config(directory, manifest, '1.1', c)
                self.assertEqual(cfg['HOD_params']['LRG_params'], ab.AB)
            changed = copy.deepcopy(self.config)
            changed['tracers']['LRG3'].update(kmax=.11, prior='gaussian')
            self.assertEqual(ab.prepare_responses(changed, self.root / 'runs'), (directory, manifest))

    def test_modified_frozen_input_refused(self):
        with self.prepared():
            (self.root / 'source/tables/bphi_samples.csv').write_text('changed\n')
            with self.assertRaisesRegex(RuntimeError, 'Frozen input'):
                ab.prepare_responses(self.config, self.root / 'runs')

    def test_pilot_uses_first_accepted_draws_not_draw_ids(self):
        with self.prepared() as (directory, manifest, _, _):
            args = ab.parse_args(['counts', '--root', str(self.root / 'runs'), '--limit', '1', '--dry-run'])
            with patch.object(ab.run_counts, 'run') as run, patch.object(ab.signal, 'signal'):
                ab.run_responses(args, self.config)
            worker, selected, digest = run.call_args.args
            self.assertEqual([r['draw'] for r in selected['rows']], [10])
            self.assertIsNone(worker.limit)
            self.assertEqual(digest, ab.sha256(directory / 'manifest.json'))

    def write_pairs(self, directory, manifest, pairs, missing=False, bad=False):
        signature = 'test-signature'
        for i, row in enumerate(manifest['rows']):
            for c, saved in zip(('000', '004'), pairs[row['draw']]):
                if missing and i == 0 and c == '004':
                    continue
                record = dict(saved, parameters=dict(saved['parameters'], **ab.AB), row=row, cosmo=c,
                              signature=signature, manifest_sha256=ab.sha256(directory / 'manifest.json'))
                if bad and i == 0:
                    record['ngal'] = 1
                ab.run_counts.save_json(ab.run_counts.count_path(directory, row, c), record)

    def test_exact_response_selection_and_no_gaussian_substitution(self):
        with self.prepared() as (directory, manifest, _, pairs):
            self.write_pairs(directory, manifest, pairs)
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, payload = ab.summarize_responses(self.config, self.root / 'runs', True)
            priors = ab.response_priors(self.config, manifest, payload)
            np.testing.assert_allclose(priors['LRG3']['values'], manifest['baseline']['LRG3']['values'])
            self.assertEqual(priors['LRG3']['prior'], 'empirical')
            self.assertEqual(priors['LRG3']['draws'], manifest['baseline']['LRG3']['draws'])

    def test_pending_and_invalid_draws_block_prior_and_are_reported(self):
        with self.prepared() as (directory, manifest, _, pairs):
            self.write_pairs(directory, manifest, pairs, missing=True)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'incomplete'):
                ab.summarize_responses(self.config, self.root / 'runs', True)
            self.assertIn('pending', (directory / 'tables/invalid_or_pending.csv').read_text())
            self.assertFalse((directory / 'responses.json').exists())
            self.write_pairs(directory, manifest, pairs, bad=True)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'invalid'):
                ab.summarize_responses(self.config, self.root / 'runs', True)
            self.assertIn('low_counts', (directory / 'tables/invalid_or_pending.csv').read_text())

    def test_fit_cases_preserve_covariance_and_select_correct_data(self):
        import bphi_prior_pipeline as pipeline
        old = dict(schema=1, config=self.config, files={}, sources={},
                   data={'LRG3': {c: 'standard' for c in ab.inputs.COSMOS}},
                   priors={'LRG3': {'values': [1., 2.]}}, covariances={'LRG3': {'path': 'standard_c000'}},
                   hod_settings={})
        pk = self.root / 'pk.npy'
        pk.write_bytes(b'AB data placeholder')
        with patch.object(ab.inputs, 'prepare', return_value=old), \
             patch.object(pipeline, 'default_pk_path', return_value=pk):
            _, _, standard = ab.prepare_fit(self.config, self.root, 'standard')
            _, _, contaminated = ab.prepare_fit(self.config, self.root, 'ab_data')
        self.assertEqual(standard['covariances'], contaminated['covariances'])
        self.assertEqual(standard['priors'], contaminated['priors'])
        self.assertEqual(standard['data']['LRG3']['000'], 'standard')
        self.assertEqual(contaminated['data']['LRG3']['000'], str(pk))
        self.assertNotEqual(standard['config']['name'], contaminated['config']['name'])
        self.assertNotIn('assembly_test', old)

    def test_ab_prior_case_uses_new_responses_only(self):
        import bphi_prior_pipeline as pipeline
        pk = self.root / 'pk.npy'
        pk.write_bytes(b'AB data placeholder')
        with self.prepared() as (directory, manifest, _, pairs):
            self.write_pairs(directory, manifest, pairs)
            old = dict(schema=1, config=self.config, files={}, sources={},
                data={'LRG3': {c: 'standard' for c in ab.inputs.COSMOS}},
                priors={}, covariances={'LRG3': {'path': 'standard_c000'}}, hod_settings={})
            with patch.object(ab.inputs, 'prepare', return_value=old), \
                 patch.object(pipeline, 'default_pk_path', return_value=pk), contextlib.redirect_stdout(io.StringIO()):
                _, _, new = ab.prepare_fit(self.config, self.root / 'runs', 'ab_prior')
            self.assertEqual(new['priors']['LRG3']['ensemble'], 'chain_s1')
            self.assertEqual(len(new['priors']['LRG3']['values']), 100)
            self.assertIn(str(directory / 'responses.json'), new['files'])
            self.assertEqual(new['covariances'], old['covariances'])

    def test_defaults_and_login_guards(self):
        args = ab.parse_args(['fit'])
        self.assertEqual(args.cases, list(ab.CASES))
        self.assertEqual(args.spec['tracers']['LRG3']['ensemble'], 'chain_s1')
        self.assertEqual(args.nwalkers, 32)
        for mode in ('fit', 'counts'):
            with patch.dict(ab.os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, 'compute node'):
                ab.main([mode])
        joint = ab.parse_args(['prepare', '--config', str(ab.HERE / 'configs/joint_empirical.json')])
        self.assertEqual([s['ensemble'] for s in joint.spec['tracers'].values()], ['chain_s3', 'chain_s3', 'chain_s1'])
        self.assertEqual([s['kmax'] for s in joint.spec['tracers'].values()], [.08, .09, .1])


if __name__ == '__main__':
    unittest.main()
