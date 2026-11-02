#!/usr/bin/env python3
"""Focused Gaussian/empirical and bphi/bfnl checks at the adopted scale cuts."""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import inspect
import json
import os
from pathlib import Path
import shlex
import subprocess

import numpy as np
from scipy.special import logsumexp

import weak_prior_validation as base
import inference
import inference_inputs as inputs
from prior_models import freeze_json, sha256
import gaussian_product_prior as product
from weak_prior_status import inspect_run

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('BPHI_RESULTS_ROOT', HERE.parent / 'outputs')).expanduser() / 'prior_mode_consistency'
RUNS = inputs.OUTPUT.parent
CASES = ('LRG1', 'LRG2', 'LRG3', 'joint')
MODES = ('bphi', 'bfnl')
FIDUCIAL = HERE / 'bphi_prior_shape_test/configs/joint_empirical_k09_09_10.json'


def config_for(case):
    config = inputs.read_config(FIDUCIAL)
    config['name'] = f'gaussian_modecheck_{case}'
    config['tracers'] = {s: dict(v, prior='gaussian') for s, v in config['tracers'].items()
                         if case == 'joint' or s == case}
    return config


def empirical_directory(case, cosmo):
    if case == 'joint':
        return RUNS / f'unified_v1/joint_empirical_k09_09_10/fit/c{cosmo}/emcee/walkers32/seed42'
    if case == 'LRG3':
        return RUNS / f'lrg3_chain_s1_empirical_v2/fit/c{cosmo}/LRG3/empirical/seed42'
    return RUNS / f'emcee_lrg12_kmax_v2/fit/c{cosmo}/{case}/kmax0p09/walkers32/seed42'


def fit_directory(args, case, mode, cosmo):
    return ROOT / case / mode / f'c{cosmo}' / args.sampler / f'walkers{args.nwalkers}' / f'seed{args.seed}'


def load_control(case):
    config = config_for(case)
    directory = ROOT / 'inputs' / case
    control = inputs.prepare(config, directory)
    return config, control


def moments(config, control):
    models = inputs.models_from_manifest(control)
    return ([models[s].mean for s in config['tracers']], [models[s].std for s in config['tracers']])


def check_reference_inputs(case, cosmo, config, control):
    directory = empirical_directory(case, cosmo)
    settings = json.loads((directory / 'settings.json').read_text())
    status = json.loads((directory / 'status.json').read_text())
    if not status.get('converged') or settings['cosmo'] != cosmo or settings.get('fnl_limits') != [-300, 300]:
        raise RuntimeError(f'Reference convergence, cosmology or fNL support mismatch: {directory}')
    specs, models = inputs.prior_context(directory, settings)
    expected = {s: dict(v, prior='empirical') for s, v in config['tracers'].items()}
    if specs != expected:
        raise RuntimeError(f'Reference settings differ beyond prior shape: {directory}')
    for sample, info in control['priors'].items():
        np.testing.assert_array_equal(np.sort(models[sample].values), np.sort(info['values']))
    for name, digest in settings['sources'].items():
        if sha256(name) != digest:
            raise RuntimeError(f'Frozen empirical source changed: {name}')
    return dict(directory=str(directory), settings_sha256=sha256(directory / 'settings.json'),
                status_sha256=sha256(directory / 'status.json'))


def build_pair(config, control, cosmo):
    from desilike.theories.galaxy_clustering import PNGTracerPowerSpectrumMultipoles, FixedPowerSpectrumTemplate
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    direct = inference.build_likelihood(config, control, cosmo)
    direct()
    base.pipeline.patch_png_bfnl_calculate(PNGTracerPowerSpectrumMultipoles)
    observables, names = [], []
    for (sample, spec), old in zip(config['tracers'].items(), direct.observables):
        label = base.pipeline.tracer_z_label('LRG', spec['z'])
        theory = PNGTracerPowerSpectrumMultipoles(mode='bfnl', ells=(0, 2),
            template=FixedPowerSpectrumTemplate(z=spec['z'], fiducial='DESI'))
        for name in ('b1', 'sigmas', 'sn0', 'bfnl_loc'):
            theory.init.params[name].update(namespace=label)
        # Q(B) is a proper induced prior. No extra product truncation is imposed.
        theory.init.params[f'{label}.bfnl_loc'].update(
            prior={'limits': [-np.inf, np.inf]}, ref={'dist': 'norm', 'loc': 0., 'scale': 50.})
        obs = TracerPowerSpectrumMultipolesObservable(data=old.flatdata.copy(), covariance=old.covariance.copy(),
            k=old.k, ells=(0, 2), theory=theory)
        obs.name = f'{label}_gaussian_product_check'
        observables.append(obs)
        names.append(f'{label}.bfnl_loc')
    data = ObservablesGaussianLikelihood(observables=observables, covariance=direct.covariance.copy(),
                                        name='modecheck_data')
    means, stds = moments(config, control)
    prior = product.make_prior(names, means, stds)
    marginal = data + prior
    marginal()
    if len(marginal.varied_params) != 4*len(config['tracers']) or marginal.varied_params.names(basename='fnl_loc'):
        raise RuntimeError('bfnl must sample independent products, not a free fNL or empirical selector.')
    return direct, marginal, data, prior


def quadrature_check(means, stds):
    from scipy.integrate import quad
    means, stds = np.asarray(means), np.asarray(stds)
    points = [means*f for f in (-299., -30., -.01, .0001, .01, 30., 299.)]
    points += [(means+stds*np.linspace(-1, 1, len(means)))*f for f in (-30., 30.)]
    refinement, independent = [], []
    grid, weights = product.quadrature()
    for beta in points:
        actual = float(product.log_induced(beta, means, stds))
        refined = float(product.log_induced(beta, means, stds, npositive=2*product.NPOSITIVE))
        refinement.append(abs(actual-refined))
        # Independent adaptive integration in log |f|, centered on each conditional peak.
        parts = []
        for sign in (-1, 1):
            half = grid[grid*sign > 0]
            logs = np.log(np.abs(half))
            score = product.log_kernel(beta, half, means, stds)+logs-np.log(2*product.FMAX)
            peak = float(np.max(score))
            center = float(logs[np.argmax(score)])
            def integrand(t):
                return np.exp(float(product.log_kernel(beta, np.array([sign*np.exp(t)]), means, stds)[0])
                              +t-np.log(2*product.FMAX)-peak)
            lo, hi = np.log(product.FMIN), np.log(product.FMAX)
            split = sorted(set([lo, hi]+[max(lo, min(hi, center+d)) for d in (-1., -.1, 0., .1, 1.)]))
            value = sum(quad(integrand, a, b, epsabs=1e-10, epsrel=1e-10)[0]
                        for a, b in zip(split[:-1], split[1:]))
            parts.append(peak+np.log(value))
        expected = float(logsumexp(parts))
        independent.append(abs(actual-expected))
    if max(refinement+independent) > 5e-4:
        raise RuntimeError(f'Induced-prior quadrature is insufficient: {max(refinement+independent)}')
    return dict(max_grid_refinement_logprior_error=max(refinement),
                max_adaptive_integral_logprior_error=max(independent), points=len(points))


def check_pair(config, control, cosmo, pair):
    import jax
    direct, marginal, data, prior = pair
    means, stds = moments(config, control)
    names = [base.pipeline.tracer_z_label('LRG', s['z']) for s in config['tracers'].values()]
    np.testing.assert_array_equal(direct.flatdata, data.flatdata)
    np.testing.assert_array_equal(direct.covariance, data.covariance)
    for p in data.varied_params:
        if p.basename != 'bfnl_loc':
            assert p.prior.__getstate__() == direct.varied_params[p.name].prior.__getstate__()
    errors, power_errors, prior_errors = [], [], []
    for fnl, noise in ((0., 0.), (30., 0.), (-30., 0.), (30., -1.), (-30., 1.), (.01, .5)):
        point = {p.name: np.atleast_1d(0. if p.value is None else p.value) for p in direct.varied_params}
        point['fnl_loc'] = np.array([fnl])
        beta = []
        for i, label in enumerate(names):
            point[f'{label}.bphi_noise'] = np.array([noise])
            point[f'{label}.sigmas'] = np.array([1.7])
            point[f'{label}.sn0'] = np.array([12.])
            beta.append((means[i]+stds[i]*noise)*fnl)
        direct_value = float(np.asarray(direct(point)).ravel()[0]) - inference.logprior(direct, point)
        values = {p.name: np.array([beta[names.index(p.namespace)]]) if p.basename == 'bfnl_loc'
                  else point[p.name] for p in marginal.varied_params}
        evaluated, derived = marginal(values, return_derived=True)
        actual = float(np.asarray(evaluated).ravel()[0])
        expected_prior = float(product.log_induced(beta, means, stds))
        actual_data = actual-inference.logprior(marginal, values)-float(np.asarray(prior.loglikelihood).ravel()[0])
        np.testing.assert_allclose(np.asarray(derived['loglikelihood']).ravel()[0],
                                   actual_data+expected_prior, rtol=1e-10, atol=1e-8)
        errors.append(abs(actual_data-direct_value))
        prior_errors.append(abs(float(np.asarray(prior.loglikelihood).ravel()[0])-expected_prior))
        np.testing.assert_allclose(actual_data, direct_value, rtol=1e-10, atol=1e-8)
        for a, b in zip(direct.observables, data.observables):
            power_errors.append(float(np.max(np.abs(a.flattheory-b.flattheory))))
            np.testing.assert_allclose(a.flattheory, b.flattheory, rtol=1e-10, atol=1e-7)
    compiled = jax.jit(marginal, static_argnames=['return_derived'])
    np.testing.assert_allclose(compiled(values), marginal(values), rtol=1e-10, atol=1e-8)
    report = dict(passed=True, cosmo=cosmo, tracers=config['tracers'], ndata=len(data.flatdata),
        bphi_nvaried=len(direct.varied_params), bfnl_nvaried=len(marginal.varied_params),
        max_data_loglike_error=max(errors), max_power_error=max(power_errors), max_prior_error=max(prior_errors),
        **quadrature_check(means, stds))
    print(f'CHECK PASSED {config["name"]} c{cosmo}: {report}', flush=True)
    return report


def freeze_manifest(case, mode, cosmo, config, control, like):
    from desilike.theories.galaxy_clustering import PNGTracerPowerSpectrumMultipoles
    payload = copy.deepcopy(control)
    payload.update(mode=mode, cosmo=cosmo, parameters=base.parameter_states(like.all_params),
        fnl_limits=[-300., 300.], product_limits=None, quadrature=dict(kind='log_midpoint',
        npositive=product.NPOSITIVE, fmin=product.FMIN, fmax=product.FMAX),
        note='Independent B_i plus Gaussian-induced joint prior for bfnl; direct Gaussian latents for bphi.')
    for path in (Path(__file__), Path(product.__file__), Path(base.__file__), Path(inference.__file__),
                 Path(inputs.__file__), Path(inspect.getfile(PNGTracerPowerSpectrumMultipoles))):
        payload['sources'][str(path.resolve())] = sha256(path)
    root = ROOT / case / mode / f'c{cosmo}'
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        freeze_json(root / 'manifest.json', payload)
    return payload


def verify_manifest(path):
    manifest = json.loads(path.read_text())
    for group in ('sources', 'files'):
        for name, digest in manifest[group].items():
            if sha256(name) != digest:
                raise RuntimeError(f'Frozen {group} changed: {name}')
    return manifest


def run(args):
    if args.command == 'fit' and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use submit or run fit inside an allocated CPU node.')
    from desilike import setup_logging
    setup_logging('info')
    for case in args.cases:
        config, control = load_control(case)
        for cosmo in args.cosmos:
            reference = check_reference_inputs(case, cosmo, config, control)
            pair = build_pair(config, control, cosmo)
            report = check_pair(config, control, cosmo, pair)
            report['empirical_reference'] = reference
            for mode in args.modes:
                like = pair[MODES.index(mode)]
                freeze_manifest(case, mode, cosmo, config, control, like)
                root = ROOT / case / mode / f'c{cosmo}'
                (root / 'check.json').write_text(json.dumps(report, indent=2)+'\n')
                if args.command == 'fit':
                    directory = fit_directory(args, case, mode, cosmo)
                    state = inspect_run(directory, read_steps=False)['state']
                    if state == 'PASSED':
                        print(f'Skipping passed fit: {directory}', flush=True)
                        continue
                    if state in ('RUNNING', 'CHECKPOINT_BUSY_OR_INVALID'):
                        raise RuntimeError(f'{state}: {directory}')
                    original = base.OUTPUT
                    base.OUTPUT = ROOT / case / mode
                    try:
                        directory.mkdir(parents=True, exist_ok=True)
                        with (directory / 'execution.lock').open('a') as lock:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            status_path = directory / 'status.json'
                            if status_path.exists():
                                saved = json.loads(status_path.read_text())
                                if saved.get('converged') and any(
                                        p.stat().st_mtime_ns > status_path.stat().st_mtime_ns
                                        for p in directory.glob('chain_*.npy')):
                                    raise RuntimeError('Completed status is older than chains; inspect before resuming.')
                            (directory / 'execution.json').write_text(json.dumps(dict(job_id=os.environ.get('SLURM_JOB_ID'),
                                started=datetime.now(timezone.utc).isoformat(), mode=mode, case=case), indent=2)+'\n')
                            base.fit(args, cosmo, like)
                    finally:
                        base.OUTPUT = original


def status(args):
    for case in args.cases:
        for cosmo in args.cosmos:
            ref = empirical_directory(case, cosmo)
            saved = json.loads((ref / 'status.json').read_text()) if (ref / 'status.json').exists() else {}
            print(f'{case} c{cosmo} empirical reference: converged={saved.get("converged", False)}')
            for mode in args.modes:
                row = inspect_run(fit_directory(args, case, mode, cosmo))
                print(f'  {mode:5s} {row["state"]:28s} steps={row["steps"]} ESS={row["ess"]} job={row["job"]}')


def submit(args):
    launches = ROOT / 'launches'
    launches.mkdir(parents=True, exist_ok=True)
    (ROOT / 'logs').mkdir(exist_ok=True)
    with (launches / 'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = set()
        records = list(launches.glob('submission_*.json'))
        if records:
            queue = subprocess.check_output(['squeue', '--noheader', '--array', '--user', os.environ['USER'],
                                             '--format=%F|%K'], text=True)
            active_ids = {tuple(v.strip().split('|')) for v in queue.splitlines()}
            for record_path in records:
                record = json.loads(record_path.read_text())
                oldplan = json.loads(Path(record['plan']).read_text())
                for i, row in enumerate(oldplan['rows']):
                    if (record['job_id'], str(i)) in active_ids:
                        active.add(row['directory'])
        rows = []
        for case in args.cases:
            for cosmo in args.cosmos:
                for mode in args.modes:
                    directory = fit_directory(args, case, mode, cosmo)
                    state = inspect_run(directory, read_steps=False)['state']
                    if str(directory) in active or state in ('PASSED', 'RUNNING'):
                        print(f'Skipping queued/running/passed: {directory}')
                        continue
                    if state == 'CHECKPOINT_BUSY_OR_INVALID':
                        raise RuntimeError(f'Inspect checkpoint: {directory}')
                    path = ROOT / case / mode / f'c{cosmo}' / 'manifest.json'
                    verify_manifest(path)
                    report = json.loads(path.with_name('check.json').read_text())
                    if not report.get('passed'):
                        raise RuntimeError(f'Run check first: {path.parent}')
                    rows.append(dict(case=case, cosmo=cosmo, mode=mode, directory=str(directory),
                                     manifest_sha256=sha256(path)))
        if not rows:
            print('No pending fits selected.')
            return
        plan = dict(rows=rows, options={k: getattr(args, k) for k in
            ('sampler', 'nwalkers', 'seed', 'min_steps', 'max_steps')}, driver_sha256=sha256(__file__))
        import hashlib
        tag = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
        path = launches / f'tasks_{tag}.json'
        freeze_json(path, plan)
        script = HERE / 'run_prior_mode_consistency.sh'
        command = ['sbatch', '--parsable', '--time=11:00:00', f'--array=0-{len(rows)-1}%{args.max_parallel}',
            '--export=ALL,BPHI_MODECHECK_CODE='+str(HERE), str(script), 'task', '--plan', str(path)]
        print(shlex.join(command), flush=True)
        if args.dry_run:
            print(f'Dry run: {len(rows)} Gaussian fits; empirical references unchanged.')
            return
        job = subprocess.check_output(command, text=True).strip().split(';')[0]
        if not job.isdigit():
            raise RuntimeError(f'Unexpected job ID: {job}')
        freeze_json(launches / f'submission_{job}.json', dict(job_id=job, plan=str(path), ntasks=len(rows)))
        print(f'Submitted {job}: {len(rows)} Gaussian fits, max {args.max_parallel} concurrent tasks.', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('command', choices=('check', 'fit', 'submit', 'task', 'status', 'compare'))
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--cosmos', nargs='+', choices=inputs.COSMOS, default=list(inputs.COSMOS))
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--sampler', choices=('emcee', 'zeus'), default='emcee')
    parser.add_argument('--nwalkers', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-steps', type=int, default=2000)
    parser.add_argument('--max-steps', type=int, default=160000)
    parser.add_argument('--max-parallel', type=int, default=3)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--max-reconstruct', type=int, default=50000)
    parser.add_argument('--pdf', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'task':
        if not args.plan or 'SLURM_ARRAY_TASK_ID' not in os.environ:
            parser.error('task requires --plan and SLURM_ARRAY_TASK_ID.')
        plan = json.loads(args.plan.read_text())
        if sha256(__file__) != plan['driver_sha256']:
            raise RuntimeError('Driver changed after submission.')
        row = plan['rows'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
        for key, value in plan['options'].items():
            setattr(args, key, value)
        path = ROOT / row['case'] / row['mode'] / f'c{row["cosmo"]}' / 'manifest.json'
        if sha256(path) != row['manifest_sha256']:
            raise RuntimeError('Manifest changed after submission.')
        args.command, args.cases, args.cosmos, args.modes = 'fit', [row['case']], [row['cosmo']], [row['mode']]
    if (args.nwalkers < 26 or args.nwalkers % 2 or args.seed < 0 or args.max_parallel < 1
            or not 20 <= args.min_steps <= args.max_steps or args.max_reconstruct < 100):
        parser.error('Require even walkers>=26, valid step limits, and >=100 reconstruction samples.')
    if any(len(set(getattr(args, key))) != len(getattr(args, key)) for key in ('cases', 'cosmos', 'modes')):
        parser.error('Duplicate selections.')
    if args.dry_run and args.command != 'submit':
        parser.error('--dry-run is for submit only.')
    if args.command == 'submit':
        submit(args)
    elif args.command == 'status':
        status(args)
    elif args.command == 'compare':
        from prior_mode_compare import compare
        compare(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
