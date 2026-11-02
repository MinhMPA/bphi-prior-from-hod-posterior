#!/usr/bin/env python3
"""Shared single-bin/joint PNG inference with independent per-tracer settings."""
import argparse
import fcntl
import inspect
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import inference_inputs as inputs
import prior_models
import study
from prior_models import freeze_json, sha256


def build_likelihood(config, manifest, cosmo, samples=None):
    import bphi_prior_pipeline as pipeline
    from scipy.linalg import block_diag
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate
    models = inputs.models_from_manifest(manifest)
    observables, covariances = [], []
    Theory = study.theory_class()
    for sample in samples or config['tracers']:
        spec = config['tracers'][sample]
        covinfo = manifest['covariances'][sample]
        pk = pipeline.load_png_rebinned_pk(manifest['data'][sample][cosmo],
                                           kmin=0., kmax=covinfo['kmax'], dk=.002, ells=(0, 2))
        with np.load(covinfo['path']) as saved:
            np.testing.assert_array_equal(saved['k'], pk['k'])
            index = np.flatnonzero((pk['k'] >= spec['kmin']) & (pk['k'] < spec['kmax']))
            selected = np.concatenate([index, index+len(pk['k'])])
            cov = saved['covariance'][np.ix_(selected, selected)]
        if not len(index) or not np.all(np.isfinite(cov)):
            raise ValueError(f'{sample}: empty bins or nonfinite covariance.')
        label = pipeline.tracer_z_label('LRG', spec['z'])
        theory = Theory(template=FixedPowerSpectrumTemplate(z=spec['z'], fiducial='DESI'), mode='bphi',
                        ells=(0, 2), prior_model=models[sample], prior_kind=spec['prior'])
        for name in ('b1', 'sn0', 'sigmas', 'bphi', 'bphi_selector', 'bphi_noise'):
            theory.init.params[name].update(namespace=label)
        theory.init.params['fnl_loc'].update(prior={'limits': [-300., 300.]})
        obs = TracerPowerSpectrumMultipolesObservable(
            data=np.concatenate([pk[f'pk{ell}'][index] for ell in (0, 2)]), covariance=cov,
            k=[pk['k'][index]]*2, ells=(0, 2), theory=theory)
        obs.name = f'{label}_prior_shape'
        observables.append(obs)
        covariances.append(cov)
    like = ObservablesGaussianLikelihood(observables=observables, covariance=block_diag(*covariances))
    if like.varied_params.names(basename='fnl_loc') != ['fnl_loc']:
        raise RuntimeError('Exactly one shared fnl_loc is required.')
    print(f"{config['name']} c{cosmo}: initial log posterior {float(like()):.8g}; "
          f'{len(like.varied_params)} varied parameters', flush=True)
    return like


def validate_mapping(chain, specs, models):
    from bphi_prior_pipeline import tracer_z_label
    for sample, spec in specs.items():
        label, kind = tracer_z_label('LRG', spec['z']), spec['prior']
        selector = np.asarray(chain[f'{label}.bphi_selector']) if kind != 'gaussian' else 0.
        noise = np.asarray(chain[f'{label}.bphi_noise']) if kind != 'empirical' else 0.
        np.testing.assert_allclose(chain[f'{label}.bphi'], models[sample].mapped(kind, selector, noise),
                                   rtol=1e-12, atol=1e-12)


def logprior(like, point):
    return sum(float(np.asarray(p.prior(point[p.name])).ravel()[0]) for p in like.varied_params)


def check_likelihood(config, manifest, cosmo, like):
    study.check_likelihood(like)
    models = inputs.models_from_manifest(manifest)
    singles, audits = [], []
    for sample, spec in config['tracers'].items():
        single = build_likelihood(config, manifest, cosmo, [sample])
        singles.append(single)
        old_cut = min(.1, spec['kmax'])
        reference, _ = study.build_likelihood(SimpleNamespace(kmin=spec['kmin'], kmax=old_cut),
            [spec['z']], cosmo, spec['prior'], {spec['z']: models[sample]})
        if study.parameter_states(single.all_params) != study.parameter_states(reference.all_params):
            raise RuntimeError(f'{sample}: parameter definitions differ from study.py.')
        obs, old_obs = single.observables[0], reference.observables[0]
        idx = np.flatnonzero(obs.k[0] < old_cut)
        selected = np.concatenate([idx, idx+len(obs.k[0])])
        np.testing.assert_array_equal(obs.k[0][idx], old_obs.k[0])
        np.testing.assert_array_equal(obs.flatdata[selected], old_obs.flatdata)
        np.testing.assert_allclose(single.covariance[np.ix_(selected, selected)], reference.covariance,
                                   rtol=1e-12, atol=1e-8)
        for fnl in (0., 30., -30.):
            point = {p.name: np.atleast_1d(p.value) for p in single.varied_params}
            point['fnl_loc'] = np.array([fnl])
            new_value, old_value = single(point), reference(point)
            np.testing.assert_allclose(np.asarray(obs.flattheory).ravel()[selected],
                                       np.asarray(old_obs.flattheory).ravel(), rtol=1e-10, atol=1e-7)
            if old_cut == spec['kmax']:
                np.testing.assert_allclose(new_value, old_value, rtol=1e-10, atol=1e-9)
        audits.append(dict(sample=sample, **spec, ndata=len(obs.flatdata), legacy_overlap_checked=True))
    # Compare the joint data likelihood with the sum of its single-bin pieces;
    # the shared fNL prior is counted once, not once per redshift.
    joint_errors = []
    for fnl in (0., 30., -30.):
        point = {p.name: np.atleast_1d(p.value) for p in like.varied_params}
        point['fnl_loc'] = np.array([fnl])
        actual = float(np.asarray(like(point)).ravel()[0])-logprior(like, point)
        expected = 0.
        for single in singles:
            values = {p.name: point[p.name] for p in single.varied_params}
            expected += float(np.asarray(single(values)).ravel()[0])-logprior(single, values)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-8)
        joint_errors.append(abs(actual-expected))
    return dict(tracers=audits, shared_fnl=True, max_joint_loglike_error=max(joint_errors))


def check_reference(directory, config, manifest, like, cosmo):
    from desilike.samples import Chain
    settings = json.loads((directory / 'settings.json').read_text())
    if settings['cosmo'] != cosmo or settings.get('fnl_limits') != [-300, 300]:
        raise RuntimeError('Reference cosmology or fNL support differs.')
    specs, models = inputs.prior_context(directory, settings)
    current = inputs.models_from_manifest(manifest)
    if specs != config['tracers']:
        raise RuntimeError('Regression reference must have the same per-tracer settings.')
    for sample in specs:
        np.testing.assert_array_equal(np.sort(models[sample].values), np.sort(current[sample].values))
        np.testing.assert_allclose(models[sample].bandwidth, current[sample].bandwidth, rtol=0, atol=0)
    if settings['parameters'] != json.loads(json.dumps(study.parameter_states(like.all_params))):
        raise RuntimeError('Regression reference has different parameter definitions.')
    for p, digest in settings['sources'].items():
        if sha256(p) != digest:
            raise RuntimeError(f'Reference source changed: {p}')
    chain = Chain.load(directory / 'chain_1.npy').remove_burnin(.5)
    validate_mapping(chain, specs, models)
    errors = []
    for i in np.linspace(0, len(chain)-1, 5, dtype=int):
        point = {p.name: np.asarray(chain[p.name])[i:i+1] for p in like.varied_params}
        actual = float(np.asarray(like(point)).ravel()[0])
        stored = float(np.asarray(chain.logposterior)[i])
        np.testing.assert_allclose(actual, stored, rtol=1e-10, atol=1e-9)
        errors.append(abs(actual-stored))
    result = dict(directory=str(directory), settings_sha256=sha256(directory / 'settings.json'),
                  max_logposterior_error=max(errors))
    print('REFERENCE CHECK PASSED:', result, flush=True)
    return result


def checkpoint(directory, walkers):
    expected = {directory / f'chain_{i+1}.npy' for i in range(walkers)}
    found = set(directory.glob('chain_*.npy'))
    states = [directory / 'checks.npy', directory / 'rng.json']
    if (found and (found != expected or not all(p.exists() for p in states))
            or not found and any(p.exists() for p in states)):
        raise RuntimeError(f'Incomplete checkpoint: {directory}')
    return bool(found)


def run_sampler(args, config, manifest, cosmo, seed, like):
    import bphi_prior_pipeline as pipeline
    from desilike.samplers import EmceeSampler, ZeusSampler
    from desilike.samples import diagnostics
    from desilike.theories.galaxy_clustering import PNGTracerPowerSpectrumMultipoles
    if args.nwalkers < 2*len(like.varied_params):
        raise ValueError(f'Need >= {2*len(like.varied_params)} walkers for this joint/prior configuration.')
    cls = EmceeSampler if args.sampler == 'emcee' else ZeusSampler
    directory = inputs.run_directory(args.output, cosmo, args.sampler, args.nwalkers, seed)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = [Path(__file__), Path(inputs.__file__), Path(study.__file__), Path(prior_models.__file__),
                   Path(pipeline.__file__), Path(inspect.getfile(PNGTracerPowerSpectrumMultipoles)), Path(inspect.getfile(cls))]
        settings = dict(schema=1, sampler=args.sampler, nwalkers=args.nwalkers, seed=seed, cosmo=cosmo,
                        tracers=config['tracers'], manifest_sha256=sha256(args.output / 'manifest.json'),
                        parameters=study.parameter_states(like.all_params), fnl_limits=[-300, 300],
                        sources={str(p.resolve()): sha256(p) for p in sources},
                        checks=dict(burnin=.5, ess_per_walker=100, checks_passed=2, gelman_rubin=None))
        freeze_json(directory / 'settings.json', settings)
        resumed = checkpoint(directory, args.nwalkers)
        sampler = cls(like, n_walkers=args.nwalkers, rng=None if resumed else seed, directory=directory)
        if resumed and len({len(c) for c in sampler.chains}) != 1:
            raise RuntimeError(f'Unequal checkpoint lengths: {directory}')
        print(f"{'Resuming' if resumed else 'Starting'} {args.sampler}: {directory}", flush=True)
        start = time.perf_counter()
        sampler.run(burn_in=.5, min_steps=args.min_steps, max_steps=args.max_steps, check_every=args.check_every,
                    save_every=args.check_every, checks_passed=2, gelman_rubin=None, ess=100)
        elapsed = time.perf_counter()-start
        models = inputs.models_from_manifest(manifest)
        kept = [c.remove_burnin(.5) for c in sampler.chains]
        for chain in kept:
            validate_mapping(chain, config['tracers'], models)
        tau = np.asarray(diagnostics.integrated_autocorrelation_time(kept, check_valid='ignore'))
        valid = bool(np.all(np.isfinite(tau)) and np.all(tau > 0))
        ess = float(len(kept[0])/tau.max()) if valid else None
        passed = (len(sampler.chains[0]) >= args.min_steps and len(sampler.checks) >= 2
                  and all(sampler.checks[-2:]) and ess is not None and ess >= 100)
        visited = {}
        for sample, spec in config['tracers'].items():
            if spec['prior'] != 'gaussian':
                label = pipeline.tracer_z_label('LRG', spec['z'])
                selector = np.concatenate([np.asarray(c[f'{label}.bphi_selector']) for c in kept])
                visited[sample] = len(np.unique(np.clip((selector*100).astype(int), 0, 99)))
        path = directory / 'status.json'
        previous = json.loads(path.read_text()) if path.exists() else {}
        status = dict(converged=bool(passed), steps_per_walker=len(sampler.chains[0]), nwalkers=args.nwalkers,
                      burnin=.5, min_ess_per_walker=ess, empirical_atoms_visited=visited,
                      sampling_seconds_completed_calls=previous.get('sampling_seconds_completed_calls', 0.)+elapsed,
                      checks=[bool(v) for v in sampler.checks],
                      note='Interacting walkers are not independent chains. Check independent seeds and latent mixing; timing excludes interrupted calls and initialization.')
        path.write_text(json.dumps(status, indent=2, allow_nan=False)+'\n')
        print(status, flush=True)
        if not passed:
            print('Checks not passed. Increase --max-steps and repeat the command to resume.', flush=True)


def load_run(directory, allow_unconverged=False):
    from desilike.samples import Chain
    status_path, settings_path = directory / 'status.json', directory / 'settings.json'
    if not status_path.exists():
        print(f'Missing completed run: {directory}', flush=True)
        return None
    status = json.loads(status_path.read_text())
    if not status.get('converged') and not allow_unconverged:
        print(f'Skipping unconfirmed sampler checks: {directory}', flush=True)
        return None
    settings = json.loads(settings_path.read_text())
    specs, models = inputs.prior_context(directory, settings)
    paths = [directory / f'chain_{i+1}.npy' for i in range(settings['nwalkers'])]
    if set(directory.glob('chain_*.npy')) != set(paths):
        raise RuntimeError(f'Missing/extra walker files: {directory}')
    watched = paths + [status_path, settings_path]
    stamps = [(p.stat().st_mtime_ns, p.stat().st_size) for p in watched]
    chains = [Chain.load(p) for p in paths]
    if {len(c) for c in chains} != {status['steps_per_walker']}:
        raise RuntimeError(f'Status/checkpoint mismatch: {directory}')
    for p, digest in settings['sources'].items():
        if sha256(p) != digest:
            raise RuntimeError(f'Frozen fit source changed: {p}')
    kept = [c.remove_burnin(.5) for c in chains]
    for chain in kept:
        validate_mapping(chain, specs, models)
        if not np.all(np.asarray(chain.weight) == 1):
            raise RuntimeError('Expected equal-weight posterior samples.')
    if stamps != [(p.stat().st_mtime_ns, p.stat().st_size) for p in watched]:
        raise RuntimeError('Run changed while reading. Compare after sampling completes.')
    chain = Chain.concatenate(kept)
    if not np.all(np.isfinite(np.asarray(chain['fnl_loc']))):
        raise RuntimeError('Nonfinite fNL samples.')
    return dict(chain=chain, settings=settings, status=status, specs=specs,
                directory=directory, settings_sha256=sha256(settings_path), status_sha256=sha256(status_path))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('command', choices=('prepare', 'check', 'fit', 'compare', 'prior-plot'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--cosmos', nargs='+', choices=inputs.COSMOS, default=list(inputs.COSMOS))
    parser.add_argument('--sampler', choices=('emcee', 'zeus'), default='emcee')
    parser.add_argument('--nwalkers', type=int, default=32)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--min-steps', type=int, default=2000)
    parser.add_argument('--max-steps', type=int, default=40000, help='Total steps per walker, including saved steps')
    parser.add_argument('--check-every', type=int, default=500)
    parser.add_argument('--reference', action='append', default=[], help='Saved run directory, optionally containing {cosmo} and {seed}')
    parser.add_argument('--reference-label', action='append', default=[])
    parser.add_argument('--comparison-config', type=Path, action='append', default=[])
    parser.add_argument('--allow-unconverged', action='store_true')
    args = parser.parse_args(argv)
    if args.nwalkers < 10 or args.nwalkers % 2 or min(args.seeds) < 0:
        parser.error('Use an even walker count >=10 and nonnegative seeds.')
    if not 20 <= args.min_steps <= args.max_steps or args.check_every < 20:
        parser.error('Require 20 <= min-steps <= max-steps and check-every >=20.')
    if any(len(set(getattr(args, k))) != len(getattr(args, k)) for k in ('cosmos', 'seeds')):
        parser.error('Duplicate cosmologies or seeds.')
    if args.reference_label and len(args.reference_label) != len(args.reference):
        parser.error('Provide one label per reference.')
    if args.comparison_config and args.command != 'compare':
        parser.error('--comparison-config is only for compare.')
    if args.reference and args.command not in ('check', 'compare'):
        parser.error('--reference is only for check or compare; it never resumes a legacy chain.')
    args.spec = inputs.read_config(args.config)
    args.output = inputs.OUTPUT / args.spec['name']
    ndim = 1+sum(5 if s['prior'] == 'kde' else 4 for s in args.spec['tracers'].values())
    if args.nwalkers < 2*ndim:
        parser.error(f'This configuration has {ndim} varied parameters; use >= {2*ndim} walkers.')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == 'fit' and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run sampling manually inside your allocated interactive CPU-node shell.')
    manifest = (inputs.prepare(args.spec, args.output) if args.command in ('prepare', 'check')
                else inputs.load_manifest(args.spec, args.output))
    if args.command == 'prepare':
        print(f'Frozen inputs: {args.output / "manifest.json"}', flush=True)
        return
    if args.command in ('compare', 'prior-plot'):
        import inference_plots
        if args.command == 'compare':
            inference_plots.compare(args)
        else:
            inference_plots.plot_priors(args, manifest)
        return
    from desilike import setup_logging
    setup_logging('info')
    for cosmo in args.cosmos:
        like = build_likelihood(args.spec, manifest, cosmo)
        audit = check_likelihood(args.spec, manifest, cosmo, like)
        if args.command == 'check':
            audit['references'] = [check_reference(Path(ref.format(cosmo=cosmo, seed=args.seeds[0])),
                args.spec, manifest, like, cosmo) for ref in args.reference]
            (args.output / f'check_c{cosmo}.json').write_text(json.dumps(audit, indent=2)+'\n')
            print(f'CHECK PASSED: {args.spec["name"]} c{cosmo}', flush=True)
        else:
            for seed in args.seeds:
                run_sampler(args, args.spec, manifest, cosmo, seed, like)
    if args.command == 'check':
        print('All requested checks passed. No sampler started.', flush=True)


if __name__ == '__main__':
    main()
