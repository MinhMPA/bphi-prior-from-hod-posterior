#!/usr/bin/env python3
"""Read-only checkpoint status, usable without sourcing the cosmology stack."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(os.environ.get('BPHI_RESULTS_ROOT', Path(__file__).resolve().parent.parent / 'outputs')).expanduser() / 'paper_weak_priors'
VALIDATION = ROOT.parent / 'weak_prior_validation'


def locked(path):
    if not path.exists():
        return False
    with path.open('r') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream, fcntl.LOCK_UN)
    return False


def inspect_run(directory, read_steps=True):
    directory = Path(directory)
    result = dict(state='NOT_STARTED', steps=None, ess=None, updated=None, job='-', directory=str(directory))
    try:
        execution = directory / 'execution.json'
        if execution.exists():
            result['job'] = json.loads(execution.read_text()).get('job_id') or '-'
        files = sorted(directory.glob('chain_*.npy'))
        live = locked(directory / 'execution.lock') or locked(directory / 'run.lock')
        status_path = directory / 'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        settings_path = directory / 'settings.json'
        settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
        expected = {f'chain_{i+1}.npy' for i in range(settings.get('nwalkers', 0))}
        complete = (bool(files) and {p.name for p in files} == expected
                    and all((directory / name).is_file() for name in ('checks.npy', 'rng.json')))
        if files:
            newest = max(p.stat().st_mtime_ns for p in files)
            result['updated'] = datetime.fromtimestamp(newest / 1e9, timezone.utc).astimezone(
                ZoneInfo('Asia/Tokyo')).strftime('%m-%d %H:%M:%S JST')
            fresh = status_path.exists() and status_path.stat().st_mtime_ns >= newest
            if fresh:
                result.update(steps=status.get('steps_per_walker'), ess=status.get('min_ess_per_walker'))
            elif read_steps:
                import numpy as np
                saved = np.load(directory / 'chain_1.npy', allow_pickle=True).item()
                result['steps'] = saved['data'][0]['value'].shape[0]
            result['state'] = ('CHECKPOINT_BUSY_OR_INVALID' if not complete else
                               'PASSED' if fresh and status.get('converged') else 'RESUME')
        elif status:
            result['state'] = 'CHECKPOINT_BUSY_OR_INVALID'
        if live:
            result['state'] = 'RUNNING'
    except (OSError, ValueError, EOFError, KeyError) as exc:
        result.update(state='CHECKPOINT_BUSY_OR_INVALID', error=str(exc))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', action='store_true', help='Inspect the original kmax=0.10 regression instead.')
    parser.add_argument('--root', type=Path)
    args = parser.parse_args(argv)
    root = args.root or (VALIDATION if args.validation else ROOT)
    pattern = 'c*/**/settings.json' if args.validation else '*/c*/**/settings.json'
    directories = sorted({p.parent for p in root.glob(pattern)})
    directories = sorted(set(directories) | {p.parent for p in root.glob(pattern.replace('settings.json', 'execution.json'))})
    print(f'Root: {root}')
    print('State                          Steps/walker    Min ESS  Job                 Last checkpoint')
    for directory in directories:
        row = inspect_run(directory)
        ess = '-' if row['ess'] is None else f"{row['ess']:.1f}"
        print(f"{row['state']:30s} {str(row['steps'] or '-'):>12s} {ess:>10s} "
              f"{str(row['job']):19s} {row['updated'] or '-'}\n  {directory.relative_to(root)}")
    if not directories:
        print('No started runs found. Pending Slurm tasks do not yet have checkpoints.')
    for path in sorted((root / 'launches').glob('submission_*.json')):
        record = json.loads(path.read_text())
        print(f"Submitted job {record['job_id']}: {record['ntasks']} tasks, plan {record['plan']}")
    print('PASSED means saved sampler checks passed, not proof of unbiasedness; RUNNING means a worker holds its lock.')


if __name__ == '__main__':
    main()
