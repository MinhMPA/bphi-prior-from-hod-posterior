#!/usr/bin/env python3
"""Fiducial weak-prior controls, reusing the validated likelihood and sampler."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess

import numpy as np

import weak_prior_validation as base
from weak_prior_status import ROOT, VALIDATION, inspect_run

CASES = {'LRG1': [0], 'LRG2': [1], 'LRG3': [2], 'LRG23': [1, 2], 'joint': [0, 1, 2]}
COSMOS = ('000', '300', '301')


def spec_for(case, cuts):
    return {f'LRG{i+1}': dict(z=base.ZS[i], kmin=.006, kmax=cuts[i]) for i in CASES[case]}


def case_root(case, cuts):
    tag = '_'.join(base.pipeline.ktag(cuts[i]) for i in CASES[case])
    return ROOT / f'{case}_bfnl_weak_k{tag}'


@contextmanager
def output_at(path):
    original = base.OUTPUT
    base.OUTPUT = path
    try:
        yield
    finally:
        base.OUTPUT = original


def validation_ready():
    directory = VALIDATION / 'c000/emcee/walkers32/seed42'
    if inspect_run(directory, read_steps=False)['state'] != 'PASSED':
        raise RuntimeError('Wait for the c000 kmax=0.10 validation to finish and pass its checks.')
    path = VALIDATION / 'c000/manifest.json'
    manifest = json.loads(path.read_text())
    for group in ('sources', 'files'):
        for name, digest in manifest[group].items():
            if base.sha256(name) != digest:
                raise RuntimeError(f'Validation input/source changed: {name}')
    check = json.loads((path.parent / 'check.json').read_text())
    if not check.get('passed'):
        raise RuntimeError('The saved numerical replay did not pass.')
    candidates = [directory / 'comparison' / f'provenance{s}.json' for s in ('_with_fnl', '')]
    for candidate in candidates:
        if candidate.exists():
            report = json.loads(candidate.read_text())
            if (report.get('status_sha256') == base.sha256(directory / 'status.json')
                    and report.get('settings_sha256') == base.sha256(directory / 'settings.json')
                    and report.get('manifest_sha256') == base.sha256(path)):
                return dict(manifest_sha256=base.sha256(path), comparison_sha256=base.sha256(candidate))
    raise RuntimeError('Run the validation compare command and inspect its posterior comparison before submission.')


def build_likelihood(spec, cosmo):
    from scipy.linalg import block_diag
    from desilike.likelihoods import ObservablesGaussianLikelihood
    observables, covariances = [], []
    for sample, row in spec.items():
        single, obs = base.pipeline.build_joint_png_likelihood(
            'LRG', [row['z']], False, mode='bfnl', simtype='PNG', cosmo=cosmo,
            kmin=row['kmin'], kmax=row['kmax'], dk=.002, ells=(0, 2), cov_veff=True)
        for observable in obs:
            if 'name' in observable.init:
                del observable.init['name']
        single()
        observables.extend(obs)
        covariances.append(np.asarray(single.covariance))
    like = ObservablesGaussianLikelihood(observables=observables, covariance=block_diag(*covariances))
    like()
    if like.varied_params.names(basename='fnl_loc') or len(like.varied_params) != 4*len(spec):
        raise RuntimeError('Unexpected parameterization for the weak bfnl control.')
    return like


def check_likelihood(like, spec, cosmo):
    offset, errors = 0, []
    for (sample, row), obs in zip(spec.items(), like.observables):
        reference, ref_obs = base.legacy_likelihood([row['z']], cosmo)
        reference()
        old = ref_obs[0]
        indices = np.flatnonzero((old.k[0] >= row['kmin']) & (old.k[0] < row['kmax']))
        selected = np.concatenate([indices, indices+len(old.k[0])])
        np.testing.assert_array_equal(obs.k[0], old.k[0][indices])
        np.testing.assert_array_equal(obs.flatdata, old.flatdata[selected])
        n = len(obs.flatdata)
        np.testing.assert_allclose(like.covariance[offset:offset+n, offset:offset+n],
                                   reference.covariance[np.ix_(selected, selected)], rtol=1e-12, atol=1e-8)
        label = base.pipeline.tracer_z_label('LRG', row['z'])
        for amplitude in (0., 100., -100.):
            point = {p.name: np.array([0. if p.value is None else p.value]) for p in like.varied_params}
            point.update({f'{label}.bfnl_loc': np.array([amplitude]),
                          f'{label}.sigmas': np.array([1.7]), f'{label}.sn0': np.array([12.])})
            like(point)
            reference({p.name: point[p.name] for p in reference.varied_params})
            np.testing.assert_allclose(obs.flattheory, old.flattheory[selected], rtol=1e-10, atol=1e-7)
        offset += n
    # Check the complete Gaussian data likelihood directly, excluding parameter priors.
    for amplitude in (0., 100., -100.):
        point = {p.name: np.array([0. if p.value is None else p.value]) for p in like.varied_params}
        for p in like.varied_params.select(basename='bfnl_loc'):
            point[p.name] = np.array([amplitude])
        posterior = float(np.asarray(like(point)).ravel()[0])
        prior = sum(float(np.asarray(p.prior(point[p.name])).ravel()[0]) for p in like.varied_params)
        residual = like.flatdata - np.concatenate([np.asarray(o.flattheory).ravel() for o in like.observables])
        expected = -.5*float(residual @ np.linalg.solve(like.covariance, residual))
        np.testing.assert_allclose(posterior-prior, expected, rtol=1e-10, atol=1e-8)
        errors.append(abs(posterior-prior-expected))
    return dict(passed=True, ndata=len(like.flatdata), nvaried=len(like.varied_params),
                max_loglike_error=max(errors), note='Matched legacy overlap and direct Gaussian likelihood; no old c301 chain is required.')


def manifest(like, spec, cosmo, gate):
    import inference
    import study
    import prior_models
    from desilike.theories.galaxy_clustering import PNGTracerPowerSpectrumMultipoles
    sources = [Path(__file__), Path(base.__file__), Path(base.pipeline.__file__),
               Path(inference.__file__), Path(study.__file__), Path(prior_models.__file__),
               Path(inspect.getfile(PNGTracerPowerSpectrumMultipoles))]
    files = []
    for row in spec.values():
        files.extend([base.pipeline.default_pk_path('LRG', row['z'], cosmo=cosmo, simtype='PNG'),
                      base.pipeline.default_cov_path('LRG', row['z'], cosmo='000', simtype='PNG',
                                                     ells=(0, 2), cov_veff=True)])
    return dict(schema=1, tracers=spec, cosmo=cosmo, mode='bfnl', prior='flat product',
                bfnl_limits=[-3000., 3000.], cov_cosmo='000', cov_veff=True, assembly_bias=False,
                dk=.002, ells=[0, 2], parameters=base.parameter_states(like.all_params),
                validation_manifest_sha256=gate['manifest_sha256'],
                sources={str(p.resolve()): base.sha256(p) for p in sources},
                files={str(p): base.sha256(p) for p in files})


def run(args):
    if args.command == 'fit' and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run fit on an allocated interactive CPU node or use submit.')
    gate = validation_ready()
    from desilike import setup_logging
    setup_logging('info')
    for case in args.cases:
        spec = spec_for(case, args.kmaxes)
        with output_at(case_root(case, args.kmaxes)):
            for cosmo in args.cosmos:
                like = build_likelihood(spec, cosmo)
                report = check_likelihood(like, spec, cosmo)
                root = base.OUTPUT / f'c{cosmo}'
                root.mkdir(parents=True, exist_ok=True)
                with (root / 'prepare.lock').open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    base.freeze_json(root / 'manifest.json', manifest(like, spec, cosmo, gate))
                (root / 'check.json').write_text(json.dumps(report, indent=2)+'\n')
                print(f'{case} c{cosmo}: {report}; cuts={spec}', flush=True)
                if args.command == 'fit':
                    directory = base.run_directory(args, cosmo)
                    directory.mkdir(parents=True, exist_ok=True)
                    with (directory / 'execution.lock').open('a') as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        if inspect_run(directory, read_steps=False)['state'] == 'CHECKPOINT_BUSY_OR_INVALID':
                            raise RuntimeError('Inspect the inconsistent checkpoint before resuming.')
                        if (directory / 'status.json').exists():
                            status = json.loads((directory / 'status.json').read_text())
                            if status.get('converged') and any(
                                    p.stat().st_mtime_ns > (directory / 'status.json').stat().st_mtime_ns
                                    for p in directory.glob('chain_*.npy')):
                                raise RuntimeError('Completed status is older than chain files; inspect before reuse.')
                        (directory / 'execution.json').write_text(json.dumps(dict(
                            job_id=os.environ.get('SLURM_JOB_ID'), array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
                            array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'), host=socket.gethostname(),
                            pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
                            log=os.environ.get('BPHI_WEAK_LOG')), indent=2)+'\n')
                        base.fit(args, cosmo, like)
    if args.command == 'check':
        print('All selected production checks passed; no sampler started.', flush=True)


def active_directories(launches):
    records = list(launches.glob('submission_*.json'))
    if not records:
        return set()
    queue = subprocess.check_output(
        ['squeue', '--noheader', '--array', '--user', os.environ['USER'], '--format=%F|%K|%T'], text=True)
    active = {tuple(line.strip().split('|')[:2]) for line in queue.splitlines()}
    directories = set()
    for path in records:
        record = json.loads(path.read_text())
        plan = json.loads(Path(record['plan']).read_text())
        config = plan['config']
        for index, row in enumerate(plan['rows']):
            if (record['job_id'], str(index)) in active:
                directories.add(case_root(row['case'], config['kmaxes']) / f"c{row['cosmo']}" /
                                config['sampler'] / f"walkers{config['nwalkers']}" / f"seed{config['seed']}")
    return directories


def submit(args):
    launches = ROOT / 'launches'
    launches.mkdir(parents=True, exist_ok=True)
    with (launches / 'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        submit_locked(args, launches)


def submit_locked(args, launches):
    gate = validation_ready()
    active = active_directories(launches)
    rows = []
    for case in args.cases:
        for cosmo in args.cosmos:
            with output_at(case_root(case, args.kmaxes)):
                directory = base.run_directory(args, cosmo)
                state = inspect_run(directory, read_steps=False)['state']
            if directory in active:
                state = 'QUEUED_OR_RUNNING'
            if state in ('PASSED', 'RUNNING', 'QUEUED_OR_RUNNING'):
                print(f'Skipping {state}: {directory}', flush=True)
            elif state == 'CHECKPOINT_BUSY_OR_INVALID':
                raise RuntimeError(f'Inspect checkpoint before submitting: {directory}')
            else:
                rows.append(dict(case=case, cosmo=cosmo))
    if not rows:
        print('No pending selected fits to submit.', flush=True)
        return
    config = {key: getattr(args, key) for key in ('kmaxes', 'sampler', 'nwalkers', 'seed', 'min_steps', 'max_steps')}
    plan = dict(schema=1, config=config, rows=rows, validation=gate,
                driver_sha256=base.sha256(__file__))
    tag = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
    (ROOT / 'logs').mkdir(exist_ok=True)
    path = launches / f'tasks_{tag}.json'
    base.freeze_json(path, plan)
    script = Path(__file__).with_name('run_paper_weak_priors.sh')
    command = ['sbatch', '--parsable', '--time=11:00:00', f'--array=0-{len(rows)-1}%{args.max_parallel}',
               '--export=ALL,BPHI_WEAK_CODE='+str(script.parent), str(script), 'task', '--plan', str(path)]
    print(f'{len(rows)} tasks; at most {args.max_parallel} simultaneous nodes. Default cuts: LRG1/2/3={args.kmaxes}.')
    print(shlex.join(command), flush=True)
    if args.dry_run:
        print('Dry run only; no job submitted.', flush=True)
        return
    job = subprocess.check_output(command, text=True).strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError(f'Unexpected sbatch job ID: {job}')
    base.freeze_json(launches / f'submission_{job}.json', dict(job_id=job, ntasks=len(rows), plan=str(path)))
    print(f'Submitted array {job}; logs: {ROOT / "logs"}/bphi_weak_{job}_<task>.out', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('command', choices=('check', 'fit', 'submit', 'task'))
    parser.add_argument('--cases', nargs='+', choices=tuple(CASES), default=list(CASES))
    parser.add_argument('--cosmos', nargs='+', choices=COSMOS, default=list(COSMOS))
    parser.add_argument('--kmaxes', nargs=3, type=float, default=[.09, .09, .1], metavar=('LRG1', 'LRG2', 'LRG3'))
    parser.add_argument('--sampler', choices=('emcee', 'zeus'), default='emcee')
    parser.add_argument('--nwalkers', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-steps', type=int, default=2000)
    parser.add_argument('--max-steps', type=int, default=120000)
    parser.add_argument('--max-parallel', type=int, default=3)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'task':
        if not args.plan or 'SLURM_ARRAY_TASK_ID' not in os.environ:
            parser.error('task requires --plan and SLURM_ARRAY_TASK_ID.')
        plan = json.loads(args.plan.read_text())
        if plan['driver_sha256'] != base.sha256(__file__):
            raise RuntimeError('Driver changed after submission; prepare a new task plan.')
        row = plan['rows'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
        for key, value in plan['config'].items():
            setattr(args, key, value)
        args.cases, args.cosmos, args.command = [row['case']], [row['cosmo']], 'fit'
    if (any(not .006 < cut <= .1 or not np.isclose(cut/.002, round(cut/.002), rtol=0, atol=1e-9)
            for cut in args.kmaxes) or args.nwalkers % 2 or args.nwalkers < 8*max(len(CASES[c]) for c in args.cases)
            or args.seed < 0 or args.max_parallel < 1 or not 20 <= args.min_steps <= args.max_steps):
        parser.error('Use cuts on the dk=0.002 grid in (0.006,0.10], enough even walkers (>=2*ndim), and valid step limits.')
    if len(set(args.cases)) != len(args.cases) or len(set(args.cosmos)) != len(args.cosmos):
        parser.error('Duplicate cases or cosmologies.')
    if args.dry_run and args.command != 'submit':
        parser.error('--dry-run is for submit only.')
    if args.command == 'submit':
        submit(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
