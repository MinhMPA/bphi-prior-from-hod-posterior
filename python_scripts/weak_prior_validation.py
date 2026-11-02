#!/usr/bin/env python3
"""Replay and rerun the paper's joint weak-prior bfnl likelihood, without HOD priors."""
import argparse
import csv
import fcntl
import inspect
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'python_scripts'))
sys.path.insert(0, str(HERE / 'bphi_prior_shape_test'))
import bphi_prior_pipeline as pipeline
from inference import checkpoint
from prior_models import freeze_json, sha256
from study import parameter_states

OUTPUT = Path(os.environ.get('BPHI_RESULTS_ROOT', HERE.parent / 'outputs')).expanduser() / 'weak_prior_validation'
LEGACY = pipeline.ROOT / 'bphi_prior/fit'
ZS = [0.5, 0.8, 1.1]


def reference_path(cosmo):
    name = pipeline.fit_prefix('LRG', ZS, 'bfnl', 'PNG', cosmo, False,
                               ells=(0, 2), kmin=.006, kmax=.1).name
    return LEGACY / f'{name}_chain.npy'


def legacy_likelihood(zsnaps, cosmo):
    like, observables = pipeline.build_joint_png_likelihood(
        'LRG', zsnaps, False, mode='bfnl', simtype='PNG', cosmo=cosmo,
        kmin=.006, kmax=.1, dk=.002, ells=(0, 2), cov_veff=True)
    # The pinned stack accepts obs.name, not a name keyword forwarded to its window.
    for obs in observables:
        if 'name' in obs.init:
            del obs.init['name']
    return like, observables


def build_likelihood(cosmo):
    from scipy.linalg import block_diag
    from desilike.likelihoods import ObservablesGaussianLikelihood
    observables, covariances = [], []
    for z in ZS:
        single, obs = legacy_likelihood([z], cosmo)
        single()
        observables.extend(obs)
        covariances.append(np.asarray(single.covariance))
    like = ObservablesGaussianLikelihood(observables=observables, covariance=block_diag(*covariances))
    like()
    if like.varied_params.names(basename='fnl_loc') or len(like.varied_params) != 12:
        raise RuntimeError('Expected twelve varied parameters, including three products, not a sampled fnl.')
    return like


def run_directory(args, cosmo):
    return OUTPUT / f'c{cosmo}' / args.sampler / f'walkers{args.nwalkers}' / f'seed{args.seed}'


def manifest(like, cosmo):
    from desilike.theories.galaxy_clustering import PNGTracerPowerSpectrumMultipoles
    import inference
    import study
    import prior_models
    sources = [Path(__file__), Path(pipeline.__file__), Path(inference.__file__),
               Path(study.__file__), Path(prior_models.__file__),
               Path(inspect.getfile(PNGTracerPowerSpectrumMultipoles))]
    files = [reference_path(cosmo)]
    for z in ZS:
        files.extend([pipeline.default_pk_path('LRG', z, cosmo=cosmo, simtype='PNG'),
                      pipeline.default_cov_path('LRG', z, cosmo='000', simtype='PNG',
                                                ells=(0, 2), cov_veff=True)])
    return dict(mode='bfnl', prior='flat product', bfnl_limits=[-3000., 3000.],
                zsnaps=ZS, kmin=.006, kmax=.1, dk=.002, ells=[0, 2],
                cov_cosmo='000', cov_veff=True, assembly_bias=False, cosmo=cosmo,
                parameters=parameter_states(like.all_params),
                sources={str(p.resolve()): sha256(p) for p in sources},
                files={str(p): sha256(p) for p in files})


def check_reference(like, cosmo):
    from desilike.samples import Chain
    original, old_obs = legacy_likelihood(ZS, cosmo)
    original()
    np.testing.assert_array_equal(like.flatdata, original.flatdata)
    np.testing.assert_allclose(like.covariance, original.covariance, rtol=1e-12, atol=1e-8)
    chain = Chain.load(reference_path(cosmo)).remove_burnin(.5)
    if int(chain.attrs['size']) != len(like.flatdata):
        raise RuntimeError('Reference and new likelihood have different data-vector lengths.')
    for param in like.varied_params:
        saved = chain[param.name].param
        if param.prior.__getstate__() != saved.prior.__getstate__():
            raise RuntimeError(f'Reference prior differs for {param.name}.')
    stored = np.asarray(chain.logposterior).ravel()
    errors = []
    for index in np.linspace(0, len(stored)-1, 12, dtype=int):
        point = {p.name: np.array([np.asarray(chain[p.name]).ravel()[index]]) for p in like.varied_params}
        value = float(np.asarray(like(point)).ravel()[0])
        old_value = float(np.asarray(original(point)).ravel()[0])
        np.testing.assert_allclose(value, old_value, rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(value, stored[index], rtol=1e-9, atol=1e-7)
        for obs, old in zip(like.observables, old_obs):
            np.testing.assert_allclose(obs.flattheory, old.flattheory, rtol=1e-10, atol=1e-7)
        errors.append(abs(value-stored[index]))
    report = dict(reference=str(reference_path(cosmo)), reference_sha256=sha256(reference_path(cosmo)),
                  ndata=len(like.flatdata), nvaried=len(like.varied_params), points_checked=len(errors),
                  max_logposterior_error=max(errors), passed=True,
                  note='Numerical replay, not a convergence test. No HOD prior or fnl reconstruction is used.')
    print(f'c{cosmo} REFERENCE CHECK PASSED: {report}', flush=True)
    return report


def fit(args, cosmo, like):
    from desilike.samplers import EmceeSampler, ZeusSampler
    from desilike.samples import diagnostics
    directory = run_directory(args, cosmo)
    directory.mkdir(parents=True, exist_ok=True)
    cls = EmceeSampler if args.sampler == 'emcee' else ZeusSampler
    settings = dict(manifest_sha256=sha256(OUTPUT / f'c{cosmo}' / 'manifest.json'),
                    sampler=args.sampler, nwalkers=args.nwalkers, seed=args.seed,
                    sampler_source=str(Path(inspect.getfile(cls)).resolve()),
                    sampler_sha256=sha256(inspect.getfile(cls)))
    with (directory / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze_json(directory / 'settings.json', settings)
        resumed = checkpoint(directory, args.nwalkers)
        if (directory / 'status.json').exists():
            status = json.loads((directory / 'status.json').read_text())
            if status.get('converged') and resumed:
                print(f'Already passed checks; leaving chains unchanged: {directory}', flush=True)
                return
        sampler = cls(like, n_walkers=args.nwalkers, rng=None if resumed else args.seed, directory=directory)
        if resumed and len({len(c) for c in sampler.chains}) != 1:
            raise RuntimeError('Unequal checkpoint lengths.')
        print(f"{'Resuming' if resumed else 'Starting'} {directory}", flush=True)
        start = time.perf_counter()
        sampler.run(burn_in=.5, min_steps=args.min_steps, max_steps=args.max_steps,
                    check_every=500, save_every=500, checks_passed=2, gelman_rubin=None, ess=100)
        kept = [c.remove_burnin(.5) for c in sampler.chains]
        tau = np.asarray(diagnostics.integrated_autocorrelation_time(kept, check_valid='ignore'))
        valid = bool(np.all(np.isfinite(tau)) and np.all(tau > 0))
        ess = float(len(kept[0])/tau.max()) if valid else None
        passed = (len(sampler.chains[0]) >= args.min_steps and len(sampler.checks) >= 2
                  and all(sampler.checks[-2:]) and ess is not None and ess >= 100)
        status = dict(converged=bool(passed), steps_per_walker=len(sampler.chains[0]),
                      min_ess_per_walker=ess, checks=[bool(v) for v in sampler.checks],
                      elapsed_seconds_this_call=time.perf_counter()-start, burnin=.5,
                      note='Walkers interact; these checks do not replace an independent-seed test.')
        (directory / 'status.json').write_text(json.dumps(status, indent=2, allow_nan=False)+'\n')
        print(status, flush=True)
        if not passed:
            print('Not converged. Repeat with a larger --max-steps total cap to resume.', flush=True)


def compare(args, cosmo):
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from desilike.samples import Chain
    directory = run_directory(args, cosmo)
    status = json.loads((directory / 'status.json').read_text())
    if not status.get('converged'):
        raise RuntimeError('Wait for the new run to pass its configured checks before comparing.')
    settings = json.loads((directory / 'settings.json').read_text())
    if settings['manifest_sha256'] != sha256(OUTPUT / f'c{cosmo}' / 'manifest.json'):
        raise RuntimeError('Fit/manifest mismatch.')
    if settings['sampler_sha256'] != sha256(settings['sampler_source']):
        raise RuntimeError('Sampler source changed.')
    checkpoint(directory, args.nwalkers)
    paths = [directory / f'chain_{i+1}.npy' for i in range(args.nwalkers)]
    watched = paths + [directory / 'status.json', reference_path(cosmo)]
    stamps = [(p.stat().st_mtime_ns, p.stat().st_size) for p in watched]
    new = [Chain.load(p) for p in paths]
    if {len(c) for c in new} != {status['steps_per_walker']}:
        raise RuntimeError('Status and saved chain lengths differ.')
    chains = [Chain.load(reference_path(cosmo)).remove_burnin(.5).ravel(),
              Chain.concatenate([c.remove_burnin(.5) for c in new])]
    labels = ['Original Zeus', f'New {args.sampler}']
    names = [f'{pipeline.tracer_z_label("LRG", z)}.bfnl_loc' for z in ZS]
    fig, axes = plt.subplots(1, 3 + int(args.fnl), figsize=(4*(3+int(args.fnl)), 3.5))
    rows = []
    for i, (label, chain) in enumerate(zip(labels, chains)):
        if not np.all(np.asarray(chain.weight) == 1):
            raise RuntimeError('Expected equally weighted MCMC states.')
        values = [np.asarray(chain[name]).ravel() for name in names]
        if args.fnl:
            from bphi_prior_plot_fnl import description_fnl_samples_from_bfnl_chain
            values.append(description_fnl_samples_from_bfnl_chain(
                chain, 'LRG', ZS, bphi_prior='flat', bfnl_chain_prior='flat',
                max_samples=args.max_fnl_samples, seed=42))
        for j, data in enumerate(values):
            if not np.all(np.isfinite(data)) or len(data) < 2:
                raise RuntimeError('Empty or nonfinite comparison samples.')
            low, median, high = np.quantile(data, [.16, .5, .84])
            name = names[j] if j < 3 else 'fnl_weak_prior_postprocessed'
            row = dict(cosmo=cosmo, label=label, parameter=name, mean=float(np.mean(data)),
                       q16=low, median=median, q84=high, width68=high-low, n_states=len(data))
            rows.append(row)
            print(f'{label} c{cosmo} {name}: mean={row["mean"]:.4f}, '
                  f'median={median:.4f} +{high-median:.4f} -{median-low:.4f}, W68={high-low:.4f}', flush=True)
            display = data[np.linspace(0, len(data)-1, min(len(data), 12000), dtype=int)]
            lo, hi = np.quantile(data, [.001, .999])
            grid = np.linspace(lo, hi, 600)
            axes[j].plot(grid, gaussian_kde(display)(grid), label=label, color=f'C{i}')
            axes[j].set_xlabel(rf'$B_{{{j+1}}}=b_{{\phi,{j+1}}}f_{{\rm NL}}$' if j < 3 else r'$f_{\rm NL}$ (weak prior)')
            axes[j].set_ylim(bottom=0)
    if stamps != [(p.stat().st_mtime_ns, p.stat().st_size) for p in watched]:
        raise RuntimeError('A chain changed during comparison; wait for sampling to finish.')
    axes[0].set_ylabel('Posterior density')
    axes[0].legend(frameon=False)
    fig.tight_layout()
    target = directory / 'comparison'
    target.mkdir(exist_ok=True)
    suffix = '_with_fnl' if args.fnl else ''
    for ext in (('png', 'pdf') if args.pdf else ('png',)):
        fig.savefig(target / f'posteriors{suffix}.{ext}', dpi=180)
    plt.close(fig)
    with (target / f'summary{suffix}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (target / f'provenance{suffix}.json').write_text(json.dumps(dict(
        manifest_sha256=settings['manifest_sha256'], settings_sha256=sha256(directory / 'settings.json'),
        status_sha256=sha256(directory / 'status.json'), reference_sha256=sha256(reference_path(cosmo)),
        burnin=.5, fnl_method='description' if args.fnl else None, max_fnl_samples=args.max_fnl_samples,
        note='B summaries use all retained states; KDE thinning is display-only. Optional fnl uses the same '
             'flat bphi [-10,10], fnl [-300,300] reconstruction on both chains, not a HOD prior. '
             'No calibrated Monte Carlo significance is inferred.'), indent=2)+'\n')
    print(f'Saved comparisons: {target}', flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('command', choices=('check', 'fit', 'compare'))
    parser.add_argument('--cosmos', nargs='+', choices=('000', '300', '301'), default=['000'])
    parser.add_argument('--sampler', choices=('emcee', 'zeus'), default='emcee')
    parser.add_argument('--nwalkers', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-steps', type=int, default=2000)
    parser.add_argument('--max-steps', type=int, default=120000)
    parser.add_argument('--fnl', action='store_true', help='Also apply the same paper weak-prior fnl reconstruction to both runs.')
    parser.add_argument('--max-fnl-samples', type=int, default=20000)
    parser.add_argument('--pdf', action='store_true')
    args = parser.parse_args(argv)
    if args.nwalkers < 24 or args.nwalkers % 2 or args.seed < 0:
        parser.error('Use an even walker count >=24 and nonnegative seed.')
    if not 20 <= args.min_steps <= args.max_steps or args.max_fnl_samples < 2:
        parser.error('Require 20 <= min-steps <= max-steps and max-fnl-samples >=2.')
    if len(set(args.cosmos)) != len(args.cosmos):
        parser.error('Duplicate cosmologies.')
    if (args.fnl or args.pdf) and args.command != 'compare':
        parser.error('--fnl/--pdf are comparison options.')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == 'fit' and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run fit manually inside your allocated interactive CPU node.')
    missing = [str(reference_path(c)) for c in args.cosmos if not reference_path(c).is_file()]
    if missing:
        raise FileNotFoundError('Missing current reference chain(s): '+', '.join(missing)+
                                '. Restore the intended reference or select an available cosmology; '
                                'archived bad_z1.1 chains are not substituted.')
    from desilike import setup_logging
    setup_logging('info')
    for cosmo in args.cosmos:
        like = build_likelihood(cosmo)
        directory = OUTPUT / f'c{cosmo}'
        directory.mkdir(parents=True, exist_ok=True)
        report = check_reference(like, cosmo)
        with (directory / 'prepare.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            freeze_json(directory / 'manifest.json', manifest(like, cosmo))
        (directory / 'check.json').write_text(json.dumps(report, indent=2)+'\n')
        if args.command == 'fit':
            fit(args, cosmo, like)
        elif args.command == 'compare':
            compare(args, cosmo)
    if args.command == 'check':
        print('All requested reference checks passed. No sampler started.', flush=True)


if __name__ == '__main__':
    main()
