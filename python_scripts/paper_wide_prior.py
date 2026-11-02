#!/usr/bin/env python3
"""Joint weak-bfnl support check at the paper's fiducial per-tracer cuts."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

import numpy as np

import paper_weak_priors as weak
from weak_prior_status import inspect_run

base = weak.base
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('BPHI_RESULTS_ROOT', HERE.parent / 'outputs')).expanduser() / 'paper_wide_prior'


@contextmanager
def prior_support(bphi_limit):
    old = dict(base.pipeline.PRIORS_FLAT)
    base.pipeline.PRIORS_FLAT.update(bphi=float(bphi_limit), fnl_loc=300.)
    try:
        yield
    finally:
        base.pipeline.PRIORS_FLAT.clear()
        base.pipeline.PRIORS_FLAT.update(old)


def case_root(args):
    tag = '_'.join(base.pipeline.ktag(v) for v in args.kmaxes)
    return ROOT / f'joint_bfnl_weak3_k{tag}'


def directory(args, cosmo, wide=True):
    root = case_root(args) if wide else weak.case_root('joint', args.kmaxes)
    return root / f'c{cosmo}' / args.sampler / f'walkers{args.nwalkers}' / f'seed{args.seed}'


def verify_manifest(path):
    payload = json.loads(path.read_text())
    for group in ('sources', 'files'):
        for name, digest in payload[group].items():
            if base.sha256(name) != digest:
                raise RuntimeError(f'Frozen {group} changed: {name}')
    return payload


def build_likelihood(spec, cosmo):
    with prior_support(30.):
        like = weak.build_likelihood(spec, cosmo)
    for param in like.varied_params.select(basename='bfnl_loc'):
        if tuple(param.prior.limits) != (-9000., 9000.):
            raise RuntimeError('Expected the three-times wider product support.')
    return like


def run(args):
    if args.command == 'fit' and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use submit or fit inside an allocated CPU node.')
    from desilike import setup_logging
    setup_logging('info')
    gate = weak.validation_ready()
    spec = weak.spec_for('joint', args.kmaxes)
    for cosmo in args.cosmos:
        like = build_likelihood(spec, cosmo)
        report = weak.check_likelihood(like, spec, cosmo)
        reference = weak.build_likelihood(spec, cosmo)
        np.testing.assert_array_equal(like.flatdata, reference.flatdata)
        np.testing.assert_array_equal(like.covariance, reference.covariance)
        for param in like.varied_params:
            if param.basename != 'bfnl_loc':
                assert param.prior.__getstate__() == reference.varied_params[param.name].prior.__getstate__()
        payload = weak.manifest(like, spec, cosmo, gate)
        payload.update(bfnl_limits=[-9000., 9000.], reconstruction_bphi_limits=[-30., 30.],
                       reconstruction_fnl_limits=[-300., 300.], baseline_bfnl_limits=[-3000., 3000.])
        payload['sources'][str(Path(__file__).resolve())] = base.sha256(__file__)
        root = case_root(args) / f'c{cosmo}'
        root.mkdir(parents=True, exist_ok=True)
        with (root / 'prepare.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            base.freeze_json(root / 'manifest.json', payload)
            (root / 'check.json').write_text(json.dumps(report, indent=2)+'\n')
        print(f'CHECK PASSED joint c{cosmo}: {report}; product support [-9000,9000]', flush=True)
        if args.command == 'fit':
            target = directory(args, cosmo)
            state = inspect_run(target, read_steps=False)['state']
            if state == 'PASSED':
                print(f'Skipping passed fit: {target}', flush=True)
                continue
            if state in ('RUNNING', 'CHECKPOINT_BUSY_OR_INVALID'):
                raise RuntimeError(f'{state}: {target}')
            target.mkdir(parents=True, exist_ok=True)
            with (target / 'execution.lock').open('a') as lock, weak.output_at(case_root(args)):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                status = target / 'status.json'
                if status.exists() and json.loads(status.read_text()).get('converged') and any(
                        p.stat().st_mtime_ns > status.stat().st_mtime_ns for p in target.glob('chain_*.npy')):
                    raise RuntimeError('Saved converged status is older than chains; inspect before reuse.')
                (target / 'execution.json').write_text(json.dumps(dict(job_id=os.environ.get('SLURM_JOB_ID'),
                    started=datetime.now(timezone.utc).isoformat()), indent=2)+'\n')
                base.fit(args, cosmo, like)


def submit(args):
    weak.validation_ready()
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
            ids = {tuple(line.strip().split('|')) for line in queue.splitlines()}
            for path in records:
                record = json.loads(path.read_text())
                for i, row in enumerate(json.loads(Path(record['plan']).read_text())['rows']):
                    if (record['job_id'], str(i)) in ids:
                        active.add(row['directory'])
        rows = []
        for cosmo in args.cosmos:
            target = directory(args, cosmo)
            state = inspect_run(target, read_steps=False)['state']
            if str(target) in active or state in ('PASSED', 'RUNNING'):
                print(f'Skipping queued/running/passed: {target}')
                continue
            if state == 'CHECKPOINT_BUSY_OR_INVALID':
                raise RuntimeError(f'Inspect checkpoint: {target}')
            path = case_root(args) / f'c{cosmo}' / 'manifest.json'
            verify_manifest(path)
            if not json.loads(path.with_name('check.json').read_text()).get('passed'):
                raise RuntimeError('Run check before submit.')
            rows.append(dict(cosmo=cosmo, directory=str(target), manifest_sha256=base.sha256(path)))
        if not rows:
            print('No unfinished selected wide-prior fits to submit.')
            return
        plan = dict(rows=rows, driver_sha256=base.sha256(__file__), options={key: getattr(args, key) for key in
            ('kmaxes', 'sampler', 'nwalkers', 'seed', 'min_steps', 'max_steps')})
        tag = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
        path = launches / f'tasks_{tag}.json'
        base.freeze_json(path, plan)
        command = ['sbatch', '--parsable', '--time=11:00:00', f'--array=0-{len(rows)-1}%{args.max_parallel}',
                   '--export=ALL,BPHI_WIDE_CODE='+str(HERE), str(HERE / 'run_paper_wide_prior.sh'),
                   'task', '--plan', str(path)]
        print(shlex.join(command), flush=True)
        if args.dry_run:
            print(f'Dry run: {len(rows)} wide-prior fits; no baseline chains changed.')
            return
        job = subprocess.check_output(command, text=True).strip().split(';')[0]
        if not job.isdigit():
            raise RuntimeError(f'Unexpected job ID: {job}')
        base.freeze_json(launches / f'submission_{job}.json', dict(job_id=job, plan=str(path), ntasks=len(rows)))
        print(f'Submitted array {job}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('command', choices=('check', 'fit', 'submit', 'task', 'status', 'compare'))
    parser.add_argument('--cosmos', nargs='+', choices=weak.COSMOS, default=list(weak.COSMOS))
    parser.add_argument('--kmaxes', nargs=3, type=float, default=[.09, .09, .1])
    parser.add_argument('--sampler', choices=('emcee', 'zeus'), default='emcee')
    parser.add_argument('--nwalkers', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-steps', type=int, default=2000)
    parser.add_argument('--max-steps', type=int, default=120000)
    parser.add_argument('--max-parallel', type=int, default=3)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--pdf', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'task':
        if not args.plan or 'SLURM_ARRAY_TASK_ID' not in os.environ:
            parser.error('task requires --plan and SLURM_ARRAY_TASK_ID.')
        plan = json.loads(args.plan.read_text())
        if base.sha256(__file__) != plan['driver_sha256']:
            raise RuntimeError('Driver changed after submission.')
        for name, value in plan['options'].items():
            setattr(args, name, value)
        row = plan['rows'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
        path = case_root(args) / f'c{row["cosmo"]}' / 'manifest.json'
        if base.sha256(path) != row['manifest_sha256']:
            raise RuntimeError('Manifest changed after submission.')
        args.command, args.cosmos = 'fit', [row['cosmo']]
    if (any(not .006 < k <= .1 or not np.isclose(k/.002, round(k/.002), rtol=0, atol=1e-9) for k in args.kmaxes)
            or args.nwalkers < 24 or args.nwalkers % 2 or args.seed < 0 or args.max_parallel < 1
            or not 20 <= args.min_steps <= args.max_steps or len(set(args.cosmos)) != len(args.cosmos)):
        parser.error('Invalid cuts, walker count, seed, step limits or duplicate cosmologies.')
    if args.dry_run and args.command != 'submit':
        parser.error('--dry-run is for submit only.')
    if args.command == 'submit':
        submit(args)
    elif args.command == 'status':
        for cosmo in args.cosmos:
            for wide in (False, True):
                target = directory(args, cosmo, wide)
                print(f'c{cosmo} {"wide" if wide else "baseline"}: {inspect_run(target)}', flush=True)
    elif args.command == 'compare':
        from wide_prior_compare import compare
        compare(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
