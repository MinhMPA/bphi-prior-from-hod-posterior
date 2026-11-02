"""Unified config, per-tracer prior, checkpoint and read-only comparison tests."""
import contextlib
import copy
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from desilike.samples import Chain
from desilike.jax import numpy as jnp
from desilike.likelihoods import BaseLikelihood

import inference as driver
import inference_inputs as inputs
import inference_plots
from prior_models import PriorModel, freeze_json, sha256


class LRG3ToyLikelihood(BaseLikelihood):
    _calculate_with_namespace = False
    _params = dict(fnl_loc=dict(prior={'limits': [-300., 300.]}, ref={'limits': [-3., 3.]}),
                   bphi_selector=dict(prior={'limits': [0., 1.]}, ref={'limits': [0., 1.]}),
                   bphi=dict(derived=True),
                   b1=dict(prior={'limits': [.1, 10.]}, ref={'limits': [1., 3.]}),
                   sigmas=dict(prior={'limits': [0., 10.]}, ref={'limits': [1., 3.]}),
                   sn0=dict(prior={'dist': 'norm', 'loc': 0., 'scale': 1000.},
                            ref={'dist': 'norm', 'loc': 0., 'scale': 1.}))

    def calculate(self, fnl_loc=0., bphi_selector=.5, b1=2., sigmas=2., sn0=0.):
        self.bphi = PriorModel(np.linspace(3., 5., 100)).mapped('empirical', bphi_selector, 0, xp=jnp)
        self.loglikelihood = -.5*((fnl_loc-self.bphi)**2 + (b1-2)**2 + (sigmas-2)**2 + sn0**2)


def toy_likelihood():
    like = LRG3ToyLikelihood()
    for name in ('bphi', 'bphi_selector', 'b1', 'sigmas', 'sn0'):
        like.init.params[name].update(namespace='lrgz1p100')
    like()
    return like


class UnifiedTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.config_path = self.root / 'config.json'
        self.raw = dict(name='toy', tracers={'LRG3': dict(ensemble='chain_s1', prior='empirical', kmax=.1)})
        self.config_path.write_text(json.dumps(self.raw))

    def test_per_tracer_config_and_walker_requirement(self):
        raw = copy.deepcopy(self.raw)
        raw['tracers'].update(LRG1=dict(ensemble='chain_s3', prior='gaussian', kmax=.06),
                              LRG2=dict(ensemble='chain_s1', prior='kde', kmax=.07, kde_scale=1.5))
        self.config_path.write_text(json.dumps(raw))
        config = inputs.read_config(self.config_path)
        self.assertEqual(list(config['tracers']), ['LRG1', 'LRG2', 'LRG3'])
        self.assertEqual(config['tracers']['LRG2']['kde_scale'], 1.5)
        args = driver.parse_args(['fit', '--config', str(self.config_path)])
        self.assertEqual(args.nwalkers, 32)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            driver.parse_args(['fit', '--config', str(self.config_path), '--nwalkers', '10'])

    def test_invalid_config_rejected(self):
        for key, value in [('ensemble', 'chain_s2'), ('prior', 'flat'), ('kmax', .001),
                           ('kmax', float('nan')), ('kmax', .071), ('kde_scale', 0), ('typo', True)]:
            raw = copy.deepcopy(self.raw)
            raw['tracers']['LRG3'][key] = value
            self.config_path.write_text(json.dumps(raw))
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                inputs.read_config(self.config_path)

    def test_joint_scale_cut_configs(self):
        configs = Path(__file__).parent / 'configs'
        choices = [('joint_empirical', (.08, .09, .10)),
                   ('joint_empirical_k09_09_10', (.09, .09, .10)),
                   ('joint_empirical_k08_08_10', (.08, .08, .10))]
        matched = None
        for name, cuts in choices:
            args = driver.parse_args(['fit', '--config', str(configs / f'{name}.json')])
            self.assertEqual(args.spec['name'], name)
            self.assertEqual(tuple(s['kmax'] for s in args.spec['tracers'].values()), cuts)
            self.assertEqual([s['ensemble'] for s in args.spec['tracers'].values()],
                             ['chain_s3', 'chain_s3', 'chain_s1'])
            self.assertTrue(all(s['prior'] == 'empirical' for s in args.spec['tracers'].values()))
            common = {sample: {k: v for k, v in spec.items() if k != 'kmax'}
                      for sample, spec in args.spec['tracers'].items()}
            if matched is not None:
                self.assertEqual(common, matched)
            matched = common
            self.assertEqual((args.sampler, args.nwalkers, args.seeds), ('emcee', 32, [42]))
            self.assertEqual(args.cosmos, ['000', '300', '301'])
            self.assertEqual(args.output.name, name)

    def test_s1_s3_selection_and_physical_cut(self):
        table = []
        for ensemble, offset in [('chain_s1', 1000), ('chain_s3', 3000)]:
            for i in range(100):
                n0, n4 = 200000+offset+100*i, 200000
                table.append(dict(z='1.1', ensemble=ensemble, passes_count_cut='true', draw=str(i),
                    source_row=str(i), n000=str(n0), n004=str(n4),
                    bphi=str(2*np.log(n0/n4)/np.log(.807952/.75))))
        spec = inputs.read_config(self.config_path)['tracers']['LRG3']
        a = inputs.select_prior(table, spec, {'ngal_min': 100000})
        b = inputs.select_prior(table, dict(spec, ensemble='chain_s3'), {'ngal_min': 100000})
        self.assertGreater(b['mean'], a['mean'])
        table[0]['n004'] = '10'
        with self.assertRaises(ValueError):
            inputs.select_prior(table, spec, {'ngal_min': 100000})

    def test_mixed_mapping(self):
        specs = {s: dict(z=z, prior=kind) for (s, z), kind in
                 zip(inputs.ZS.items(), ['empirical', 'kde', 'gaussian'])}
        models = {s: PriorModel(np.linspace(i+1, i+3, 100)) for i, s in enumerate(specs)}
        values = {}
        from bphi_prior_pipeline import tracer_z_label
        for s, spec in specs.items():
            label = tracer_z_label('LRG', spec['z'])
            values[label+'.bphi_selector'] = np.array([.12, .56, .91])
            values[label+'.bphi_noise'] = np.array([-.7, 0., .8])
            values[label+'.bphi'] = models[s].mapped(spec['prior'], values[label+'.bphi_selector'], values[label+'.bphi_noise'])
        driver.validate_mapping(values, specs, models)
        values['lrgz0p800.bphi'] += 1
        with self.assertRaises(AssertionError):
            driver.validate_mapping(values, specs, models)

    def test_no_sampler_on_login_node(self):
        with patch.dict(driver.os.environ, {}, clear=True), self.assertRaises(RuntimeError):
            driver.main(['fit', '--config', str(self.config_path)])

    def test_plot_export_option(self):
        for command in ('compare', 'prior-plot'):
            argv = [command, '--config', str(self.config_path), '--cosmos', '000']
            self.assertFalse(inference_plots.parse_args(argv).pdf)
            args = inference_plots.parse_args(argv + ['--pdf'])
            self.assertTrue(args.pdf)
            self.assertEqual(args.cosmos, ['000'])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            inference_plots.parse_args(['fit', '--config', str(self.config_path), '--pdf'])
        argv = ['compare', '--config', str(self.config_path), '--kmax-summary']
        self.assertTrue(inference_plots.parse_args(argv).kmax_summary)
        args = inference_plots.parse_args(argv + ['--plot-output', str(self.root / 'figures'),
            '--references-only', '--reference', str(self.root / 'saved')])
        self.assertEqual(args.plot_output, self.root / 'figures')
        self.assertNotEqual(args.output, args.plot_output)
        self.assertTrue(args.references_only)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            inference_plots.parse_args(argv + ['--references-only'])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            inference_plots.parse_args(argv + ['--references-only', '--reference', str(self.root),
                '--comparison-config', str(self.config_path)])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            inference_plots.parse_args(['prior-plot', '--config', str(self.config_path), '--kmax-summary'])

    def test_kmax_summary_common_rules(self):
        rows = []
        for cut, width, factor in [(.06, 10., 1.), (.07, 8., .5)]:
            for c, truth, offset in [('000', 0., 4.), ('300', 30., 6.), ('301', -30., 8.)]:
                spec = inputs.read_config(self.config_path)['tracers']
                spec['LRG3']['kmax'] = cut
                rows.append(dict(cosmo=c, converged=True, mean=truth+factor*offset,
                    fnl_true=truth, width68=width, sampler='emcee', seed=42, nwalkers=32,
                    tracers=json.dumps(spec)))
        _, summary = inference_plots.kmax_rows(rows, ['000', '300', '301'])
        self.assertEqual([r['mean_width68'] for r in summary], [10., 8.])
        self.assertEqual(summary[0]['worst_abs_mean_offset'], 8.)
        self.assertEqual(summary[0]['worst_offset_per_halfwidth68'], 1.6)
        self.assertAlmostEqual(summary[1]['width_reduction_vs_smallest_cut_pct'], 20.)
        with contextlib.redirect_stdout(io.StringIO()):
            _, partial = inference_plots.kmax_rows(rows[:-1], ['000', '300', '301'])
        self.assertEqual(len(partial), 1)
        with self.assertRaises(ValueError):
            inference_plots.kmax_rows(rows+[rows[0]], ['000', '300', '301'])
        changed = copy.deepcopy(rows)
        spec = json.loads(changed[-1]['tracers'])
        spec['LRG3']['prior'] = 'gaussian'
        changed[-1]['tracers'] = json.dumps(spec)
        with self.assertRaises(ValueError):
            inference_plots.kmax_rows(changed, ['000', '300', '301'])

    def test_prior_plot_png_default_and_pdf_opt_in(self):
        with patch.object(inputs, 'OUTPUT', self.root / 'runs'), contextlib.redirect_stdout(io.StringIO()):
            argv = ['prior-plot', '--config', str(self.config_path)]
            args = inference_plots.parse_args(argv)
            manifest = dict(schema=1, config=args.spec, files={}, sources={},
                            priors={'LRG3': dict(values=np.linspace(3., 5., 100).tolist(), kde_scale=1.)})
            freeze_json(args.output / 'manifest.json', manifest)
            inference_plots.main(argv)
            png, pdf = [args.output / 'plots' / f'bphi_priors.{ext}' for ext in ('png', 'pdf')]
            self.assertTrue(png.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
            self.assertFalse(pdf.exists())
            inference_plots.main(argv + ['--pdf'])
            self.assertTrue(png.exists())
            self.assertTrue(pdf.read_bytes().startswith(b'%PDF-'))
            inference_plots.main(argv + ['--plot-output', str(self.root / 'separate')])
            self.assertTrue((self.root / 'separate/plots/bphi_priors.png').exists())
            self.assertFalse((self.root / 'separate/manifest.json').exists())

    def test_reference_only_without_config_manifest(self):
        with patch.object(inputs, 'OUTPUT', self.root / 'missing-runs'):
            argv = ['compare', '--config', str(self.config_path), '--references-only',
                    '--reference', str(self.root / 'saved'), '--cosmos', '000',
                    '--plot-output', str(self.root / 'diagnostic')]
            config = inputs.read_config(self.config_path)
            result = dict(settings=dict(cosmo='000', fnl_limits=[-300, 300], sampler='emcee',
                          seed=42, nwalkers=32), status=dict(converged=True),
                          chain={'fnl_loc': np.linspace(-20, 40, 1000)}, specs=config['tracers'])
            with patch.object(driver, 'load_run', return_value=result) as read, \
                 patch.object(inputs, 'load_manifest', side_effect=AssertionError('No config manifest needed')), \
                 contextlib.redirect_stdout(io.StringIO()):
                inference_plots.main(argv + ['--kmax-summary'])
            read.assert_called_once_with(self.root / 'saved', False)
            self.assertFalse((self.root / 'missing-runs').exists())
            self.assertEqual(len(list((self.root / 'diagnostic/plots').glob('*.png'))), 2)
            self.assertEqual(len(list((self.root / 'diagnostic/tables').glob('*.csv'))), 2)

    def test_emcee_resume_and_compare_preserves_chains(self):
        self.check_resume_and_compare('emcee')

    def test_zeus_resume_and_compare_preserves_chains(self):
        self.check_resume_and_compare('zeus')

    def check_resume_and_compare(self, sampler):
        with patch.object(inputs, 'OUTPUT', self.root / 'runs'), contextlib.redirect_stdout(io.StringIO()):
            args = driver.parse_args(['fit', '--config', str(self.config_path), '--cosmos', '000',
                '--sampler', sampler, '--nwalkers', '10', '--min-steps', '100', '--max-steps', '100',
                '--check-every', '50', '--allow-unconverged'])
            model = PriorModel(np.linspace(3., 5., 100))
            manifest = dict(schema=1, config=args.spec, priors={'LRG3': dict(values=model.values.tolist(), kde_scale=1.)},
                            files={}, sources={})
            freeze_json(args.output / 'manifest.json', manifest)
            like = toy_likelihood()
            driver.run_sampler(args, args.spec, manifest, '000', 42, like)
            directory = inputs.run_directory(args.output, '000', sampler, 10, 42)
            first = np.asarray(Chain.load(directory / 'chain_1.npy')['fnl_loc']).copy()
            args.min_steps = args.max_steps = 200
            driver.run_sampler(args, args.spec, manifest, '000', 42, toy_likelihood())
            chain = Chain.load(directory / 'chain_1.npy')
            self.assertEqual(len(chain), 200)
            np.testing.assert_array_equal(first, chain['fnl_loc'][:100])
            loaded = driver.load_run(directory, True)
            self.assertEqual(len(loaded['chain']), 1000)
            before = {p: sha256(p) for p in directory.iterdir()}
            inference_plots.compare(args)
            self.assertEqual(len(list((args.output / 'plots').glob('*.png'))), 1)
            self.assertFalse(list((args.output / 'plots').glob('*.pdf')))
            args.pdf = True
            inference_plots.compare(args)
            self.assertEqual(len(list((args.output / 'plots').glob('*.pdf'))), 1)
            with next((args.output / 'tables').glob('*.csv')).open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(int(rows[0]['n_states']), 1000)
            self.assertAlmostEqual(float(rows[0]['mean']), np.asarray(loaded['chain']['fnl_loc']).mean())
            self.assertAlmostEqual(float(rows[0]['width68']), float(rows[0]['q84'])-float(rows[0]['q16']))
            for p, digest in before.items():
                self.assertEqual(sha256(p), digest)
            with self.assertRaises(RuntimeError):
                inputs.load_manifest(dict(args.spec, name='changed'), args.output)
            freeze = json.loads((directory / 'settings.json').read_text())
            self.assertEqual(freeze['tracers']['LRG3']['ensemble'], 'chain_s1')


if __name__ == '__main__':
    unittest.main()
