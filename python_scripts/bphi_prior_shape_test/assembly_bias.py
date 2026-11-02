#!/usr/bin/env python3
"""Prepare fixed-assembly-bias inputs; reuse the existing HOD worker and inference engine."""
import argparse
import copy
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import numpy as np
import yaml

import inference
import inference_inputs as inputs
import inference_plots
import prior_models
import study
from prior_models import SOURCE_RUN, freeze_json, sha256

HERE = Path(__file__).resolve().parent
HOD = HERE.parent / 'hod_correlation_test'
sys.path.insert(0, str(HOD))
import ensemble_results as results
import run_counts

AB = dict(Bcent=-.04, Bsat=-.17)
TAG = 'Bcentm0p040_Bsatm0p170'
ROOT = Path(os.environ.get('BPHI_RESULTS_ROOT', HERE.parent.parent / 'outputs')).expanduser() / 'assembly_bias_test'
CASES = ('standard', 'ab_data', 'ab_prior')
LABELS = ('Standard P(k) + standard prior', 'AB P(k) + standard prior', 'AB P(k) + AB prior')


def verify(hashes):
    for name, expected in hashes.items():
        if sha256(name) != expected:
            raise RuntimeError(f'Frozen input/source changed: {name}')


def response_directory(config, root):
    selection = '_'.join(f"{s.lower()}_{v['ensemble'].replace('chain_', '')}" for s, v in config['tracers'].items())
    return root / 'responses' / f'{selection}_{TAG}'


def case_config(config, case):
    return dict(name=f"{config['name']}_{TAG}_{case}", tracers=copy.deepcopy(config['tracers']))


def fit_directory(config, root, case):
    return root / 'fits' / case_config(config, case)['name']


def prepare_responses(config, root):
    directory = response_directory(config, root)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = directory / 'manifest.json'
        selection = {s: dict(z=v['z'], ensemble=v['ensemble']) for s, v in config['tracers'].items()}
        if path.exists():
            manifest = json.loads(path.read_text())
            if manifest['assembly_bias'] != AB or manifest['selection'] != selection:
                raise RuntimeError('Response selection differs from the frozen run.')
            verify(manifest['input_hashes'])
            return directory, manifest
        original = json.loads((SOURCE_RUN / 'manifest.json').read_text())
        original_sha = sha256(SOURCE_RUN / 'manifest.json')
        plan = results.load_plan(SOURCE_RUN, original_sha)
        table_path = SOURCE_RUN / 'tables/bphi_samples.csv'
        with table_path.open() as stream:
            table = list(csv.DictReader(stream))
        files = [SOURCE_RUN / 'manifest.json', SOURCE_RUN / 'topup_manifest.json', table_path,
                 Path(__file__), Path(results.__file__), Path(run_counts.__file__), Path(inputs.__file__)]
        cache = SOURCE_RUN / 'existing_export_pairs.json'
        if cache.exists():
            files.append(cache)
            files.extend(sorted((SOURCE_RUN / 'original_exports').glob('*')))
        hashes = {str(p): sha256(p) for p in files if p.is_file()}
        snapshots, rows, baseline = {}, [], {}
        for sample, spec in config['tracers'].items():
            z = str(spec['z'])
            prior = inputs.select_prior(table, spec, original['settings'])
            pairs = results.accepted_rows(SOURCE_RUN, original, plan, z, spec['ensemble'], original_sha)
            by_draw = {r['draw']: (r, pair) for r, pair in pairs}
            baseline[sample] = prior
            for draw, source_row, value in zip(prior['draws'], prior['source_rows'], prior['values']):
                row, pair = by_draw[draw]
                if row['source_row'] != source_row or pair[0]['parameters'] != pair[1]['parameters']:
                    raise RuntimeError('Source draw or paired HOD parameters disagree.')
                np.testing.assert_allclose(value, 2*np.log(pair[0]['ngal']/pair[1]['ngal'])/np.log(.807952/.75),
                                           rtol=1e-12, atol=1e-12)
                for cosmo, record in zip(('000', '004'), pair):
                    if any(record['parameters'].get(k, 0.) != 0. for k in ('Acent', 'Asat', 'Bcent', 'Bsat')):
                        raise RuntimeError('The reference HOD already contains assembly bias.')
                    checkpoint = run_counts.count_path(SOURCE_RUN, row, cosmo)
                    if checkpoint.exists():
                        hashes[str(checkpoint)] = sha256(checkpoint)
                shifted = copy.deepcopy(row)
                shifted['params'].update(AB)
                rows.append(shifted)
            snapshots[z] = copy.deepcopy(original['snapshots'][z])
            for cosmo, cfg in snapshots[z]['configs'].items():
                if not cfg['HOD_params'].get('want_AB'):
                    raise RuntimeError('Source configuration must load the halo assembly-bias ranks.')
                cfg['HOD_params']['LRG_params'].update(AB)
                cfg['sim_params']['output_dir'] = str(directory / 'unused_catalog_output')
                target = directory / 'configs' / f'abacushod_c{cosmo}_ph000_z{float(z):.3f}.yaml'
                target.parent.mkdir(exist_ok=True)
                if target.exists():
                    if yaml.safe_load(target.read_text()) != cfg:
                        raise RuntimeError(f'Existing YAML differs: {target}')
                else:
                    target.write_text(yaml.safe_dump(cfg, sort_keys=False))
        # Response reuse must not depend on the fitting cut or Gaussian/KDE/empirical choice.
        for p in baseline.values():
            for key in ('prior', 'kmin', 'kmax', 'kde_scale', 'kde_bandwidth'):
                p.pop(key)
        manifest = dict(settings=original['settings'], snapshots=snapshots, rows=rows,
            published={}, selection=selection, assembly_bias=AB, baseline=baseline, input_hashes=hashes,
            completeness='Unchanged rule: derive once in AB c000 if not stored, then hold fixed in c004.',
            rejection_policy='Same accepted standard draws; report AB failures, no automatic removal or top-up.')
        verify(hashes)
        freeze_json(path, manifest)
        return directory, manifest


def run_responses(args, config):
    directory, manifest = prepare_responses(config, args.root)
    selected = copy.deepcopy(manifest)
    if args.limit is not None:
        selected['rows'] = [r for z in manifest['snapshots']
                            for r in [r for r in manifest['rows'] if r['z'] == z][:args.limit]]
    worker = SimpleNamespace(run_dir=directory, zsnaps=list(manifest['snapshots']), limit=None,
        threads=args.threads, stop_file=args.stop_file, dry_run=args.dry_run, require_approval=False)
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, run_counts.request_stop)
    with (directory / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_counts.run(worker, selected, sha256(directory / 'manifest.json'))


def summarize_responses(config, root, require_complete=False):
    directory, manifest = prepare_responses(config, root)
    path = directory / 'manifest.json'
    records, frozen_files = [], {str(path): sha256(path)}
    for row in manifest['rows']:
        pair = results.read_pair(directory, row, sha256(path))
        status, reasons = results.classify(row, pair, manifest['settings']['ngal_min'])
        if all(p is not None and p['status'] == 'ok' for p in pair):
            if pair[0]['parameters'] != pair[1]['parameters'] or pair[0]['ic'] != pair[1]['ic']:
                raise RuntimeError('c000/c004 must use identical HOD and assembly-bias parameters.')
            if any(pair[0]['parameters'].get(k) != v for k, v in AB.items()):
                raise RuntimeError('The recorded assembly-bias amplitudes differ.')
        counts = [p.get('ngal') if p else None for p in pair]
        bphi = 2*np.log(counts[0]/counts[1])/np.log(.807952/.75) if status == 'accepted' else None
        records.append(dict(z=float(row['z']), ensemble=row['ensemble'], draw=row['draw'],
            source_row=row['source_row'], n000=counts[0], n004=counts[1], bphi=bphi,
            passes_count_cut=status == 'accepted', status=status, reason='; '.join(reasons),
            parameters=json.dumps(row['params'], sort_keys=True)))
        for cosmo in ('000', '004'):
            p = run_counts.count_path(directory, row, cosmo)
            if p.exists():
                frozen_files[str(p)] = sha256(p)
    target = directory / 'tables'
    target.mkdir(exist_ok=True)
    for name, selected in [('bphi_samples', records), ('invalid_or_pending', [r for r in records if r['status'] != 'accepted'])]:
        with (target / f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(selected)
    complete = all(r['status'] == 'accepted' for r in records)
    for sample, spec in config['tracers'].items():
        values = [r['bphi'] for r in records if r['z'] == spec['z'] and r['status'] == 'accepted']
        extra = f'; mean={np.mean(values):.6f}, std={np.std(values):.6f}' if values else ''
        print(f'{sample}: {len(values)}/100 valid paired AB responses{extra}', flush=True)
    payload = dict(complete=complete, assembly_bias=AB, records=records, files=frozen_files)
    if complete:
        freeze_json(directory / 'responses.json', payload)
    elif require_complete:
        raise RuntimeError(f'AB prior is incomplete or has invalid draws. Review {target}/invalid_or_pending.csv; '
                           'no thinning, replacement, or fallback to the old LHC prior is allowed.')
    return directory, manifest, payload


def response_priors(config, response_manifest, payload):
    if not payload['complete']:
        raise RuntimeError('Need all 100 paired responses per selected sample.')
    table = [{k: str(v) for k, v in row.items()} for row in payload['records']]
    return {s: inputs.select_prior(table, spec, response_manifest['settings']) for s, spec in config['tracers'].items()}


def prepare_fit(config, root, case):
    import bphi_prior_pipeline as pipeline
    directory = fit_directory(config, root, case)
    spec = case_config(config, case)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (directory / 'manifest.json').exists():
            return spec, directory, inputs.load_manifest(spec, directory)
        # Prepare the standard inputs with the existing implementation, including
        # its c000 covariance equality check, then change only explicitly selected inputs.
        control = inputs.prepare(config, root / 'controls' / config['name'])
        manifest = copy.deepcopy(control)
        manifest['config'] = spec
        manifest['assembly_test'] = dict(case=case, amplitudes=AB, base_config=config,
                                        covariance='Standard c000 Veff, no assembly bias.')
        manifest['sources'][str(Path(__file__).resolve())] = sha256(__file__)
        if case != 'standard':
            if pipeline.ASSEMBLY_BIAS_SUFFIX != '_' + TAG:
                raise RuntimeError('Pipeline AB filename tag differs from this fixed test.')
            for sample, tracer in config['tracers'].items():
                for cosmo in inputs.COSMOS:
                    p = pipeline.default_pk_path('LRG', tracer['z'], cosmo=cosmo, simtype='PNG',
                                                 abias_suffix=pipeline.ASSEMBLY_BIAS_SUFFIX)
                    manifest['files'][str(p)] = sha256(p)
                    manifest['data'][sample][cosmo] = str(p)
        if case == 'ab_prior':
            response_dir, response_manifest, payload = summarize_responses(config, root, require_complete=True)
            manifest['priors'] = response_priors(config, response_manifest, payload)
            manifest['files'].update(payload['files'])
            manifest['files'].update(response_manifest['input_hashes'])
            manifest['files'][str(response_dir / 'responses.json')] = sha256(response_dir / 'responses.json')
        verify(manifest['files'])
        verify(manifest['sources'])
        freeze_json(directory / 'manifest.json', manifest)
        return spec, directory, manifest


def check_fit(config, manifest, cosmo, like):
    import bphi_prior_pipeline as pipeline
    study.check_likelihood(like)
    base = manifest['assembly_test']['base_config']
    # Reuse the frozen legacy check with the requested data substituted into a
    # comparison copy. No change to the actual model, priors, or observed data.
    reference = copy.deepcopy(manifest)
    for sample, spec in base['tracers'].items():
        reference['data'][sample][cosmo] = str(pipeline.default_pk_path('LRG', spec['z'], cosmo=cosmo, simtype='PNG'))
    standard_like = inference.build_likelihood(config, reference, cosmo)
    audit = inference.check_likelihood(config, reference, cosmo, standard_like)
    errors = []
    for fnl in (0., 30., -30.):
        point = {p.name: np.atleast_1d(p.value) for p in like.varied_params}
        point['fnl_loc'] = np.array([fnl])
        actual = float(np.asarray(like(point)).ravel()[0]) - inference.logprior(like, point)
        standard_like(point)
        expected = 0.
        for obs, old_obs in zip(like.observables, standard_like.observables):
            np.testing.assert_allclose(obs.flattheory, old_obs.flattheory, rtol=1e-10, atol=1e-7)
            np.testing.assert_allclose(obs.covariance, old_obs.covariance, rtol=0, atol=0)
            delta = np.asarray(obs.flatdata) - np.asarray(obs.flattheory)
            expected -= .5*float(delta @ np.linalg.solve(obs.covariance, delta))
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-8)
        errors.append(abs(actual-expected))
    audit.update(case=manifest['assembly_test']['case'], max_direct_loglike_error=max(errors),
                 standard_c000_covariance=True)
    return audit


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('command', choices=('prepare-responses', 'counts', 'summarize', 'prepare', 'check', 'fit', 'compare', 'prior-plot'))
    parser.add_argument('--config', type=Path, default=HERE / 'configs/lrg3_s1_empirical_k010.json')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 32)))
    parser.add_argument('--limit', type=int, help='First N accepted draws per selected tracer for a manual pilot.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--cosmos', nargs='+', choices=inputs.COSMOS, default=list(inputs.COSMOS))
    parser.add_argument('--sampler', choices=('emcee', 'zeus'), default='emcee')
    parser.add_argument('--nwalkers', type=int, default=32)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--min-steps', type=int, default=2000)
    parser.add_argument('--max-steps', type=int, default=40000)
    parser.add_argument('--check-every', type=int, default=500)
    parser.add_argument('--allow-unconverged', action='store_true')
    parser.add_argument('--pdf', action='store_true')
    args = parser.parse_args(argv)
    args.spec = inputs.read_config(args.config)
    ndim = 1+sum(5 if s['prior'] == 'kde' else 4 for s in args.spec['tracers'].values())
    if args.nwalkers % 2 or args.nwalkers < 2*ndim or min(args.seeds) < 0:
        parser.error(f'Use even nwalkers >= {2*ndim} and nonnegative seeds.')
    if args.threads < 1 or args.limit is not None and args.limit < 1:
        parser.error('threads/limit must be positive.')
    if not 20 <= args.min_steps <= args.max_steps or args.check_every < 20:
        parser.error('Require 20 <= min-steps <= max-steps and check-every >=20.')
    for key in ('cases', 'cosmos', 'seeds'):
        if len(set(getattr(args, key))) != len(getattr(args, key)):
            parser.error(f'Duplicate {key}.')
    if args.dry_run and args.command != 'counts' or args.limit is not None and args.command != 'counts':
        parser.error('--dry-run/--limit are only for counts.')
    return args


def main(argv=None):
    args = parse_args(argv)
    config = args.spec
    if args.command in ('counts', 'fit') and not args.dry_run and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run HOD evaluations/sampling on an allocated CPU compute node, not a login node.')
    if args.command == 'prepare-responses':
        directory, _ = prepare_responses(config, args.root)
        print(f'Frozen response inputs: {directory}', flush=True)
        return
    if args.command == 'counts':
        run_responses(args, config)
        return
    if args.command == 'summarize':
        summarize_responses(config, args.root)
        return
    if args.command == 'compare':
        for case in args.cases:
            inputs.load_manifest(case_config(config, case), fit_directory(config, args.root, case))
        args.references_only, args.comparison_config = True, []
        args.reference = [str(inputs.run_directory(fit_directory(config, args.root, case), '{cosmo}',
                                                  args.sampler, args.nwalkers, '{seed}')) for case in args.cases]
        args.reference_label = [LABELS[CASES.index(case)] for case in args.cases]
        args.output = args.root / 'comparisons' / config['name']
        inference_plots.compare(args)
        return
    if args.command in ('check', 'fit'):
        from desilike import setup_logging
        setup_logging('info')
    for case in args.cases:
        if args.command in ('prepare', 'check'):
            spec, directory, manifest = prepare_fit(config, args.root, case)
        else:
            spec, directory = case_config(config, case), fit_directory(config, args.root, case)
            manifest = inputs.load_manifest(spec, directory)
        args.output, args.spec = directory, spec
        print(f'{case}: {directory}', flush=True)
        if args.command == 'prepare':
            continue
        if args.command == 'prior-plot':
            inference_plots.plot_priors(args, manifest)
            continue
        for cosmo in args.cosmos:
            like = inference.build_likelihood(spec, manifest, cosmo)
            audit = check_fit(spec, manifest, cosmo, like)
            if args.command == 'check':
                (directory / f'check_c{cosmo}.json').write_text(json.dumps(audit, indent=2)+'\n')
                print(f'CHECK PASSED: {case} c{cosmo}; no sampler started.', flush=True)
            else:
                for seed in args.seeds:
                    inference.run_sampler(args, spec, manifest, cosmo, seed, like)


if __name__ == '__main__':
    main()
