"""Compare frozen empirical references with the matched Gaussian bphi/bfnl runs."""
import csv
import json
from pathlib import Path

import numpy as np

import prior_mode_consistency as driver
import gaussian_product_prior as product
from prior_models import sha256


def sample_summary(values):
    q025, q16, median, q84, q975 = np.quantile(values, [.025, .16, .5, .84, .975])
    return dict(mean=float(np.mean(values)), median=float(median), q16=float(q16), q84=float(q84),
                width68=float(q84-q16), q025=float(q025), q975=float(q975), width95=float(q975-q025))


def load_gaussian(args, case, mode, cosmo):
    from desilike.samples import Chain
    directory = driver.fit_directory(args, case, mode, cosmo)
    state = driver.inspect_run(directory, read_steps=False)['state']
    if state != 'PASSED':
        raise RuntimeError(f'{state}: {directory}')
    manifest_path = driver.ROOT / case / mode / f'c{cosmo}' / 'manifest.json'
    manifest = driver.verify_manifest(manifest_path)
    settings = json.loads((directory / 'settings.json').read_text())
    if settings['manifest_sha256'] != sha256(manifest_path):
        raise RuntimeError('Gaussian fit/manifest mismatch.')
    if settings['sampler_sha256'] != sha256(settings['sampler_source']):
        raise RuntimeError('Sampler source changed.')
    driver.base.checkpoint(directory, args.nwalkers)
    status = json.loads((directory / 'status.json').read_text())
    paths = [directory / f'chain_{i+1}.npy' for i in range(args.nwalkers)]
    watched = paths+[directory / 'status.json', directory / 'settings.json']
    stamps = [(p.stat().st_mtime_ns, p.stat().st_size) for p in watched]
    chains = [Chain.load(path) for path in paths]
    if {len(c) for c in chains} != {status['steps_per_walker']}:
        raise RuntimeError('Status/checkpoint lengths disagree.')
    kept = [c.remove_burnin(.5) for c in chains]
    if any(not np.all(np.asarray(c.weight) == 1) for c in kept):
        raise RuntimeError('Expected equally weighted MCMC states.')
    if mode == 'bphi':
        for c in kept:
            driver.inference.validate_mapping(c, manifest['config']['tracers'],
                                               driver.inputs.models_from_manifest(manifest))
        values = np.concatenate([np.asarray(c['fnl_loc']).ravel() for c in kept])
    else:
        names = [driver.base.pipeline.tracer_z_label('LRG', spec['z'])+'.bfnl_loc'
                 for spec in manifest['config']['tracers'].values()]
        values = np.concatenate([np.column_stack([np.asarray(c[n]).ravel() for n in names]) for c in kept])
    if not np.all(np.isfinite(values)):
        raise RuntimeError('Nonfinite posterior samples.')
    if stamps != [(p.stat().st_mtime_ns, p.stat().st_size) for p in watched]:
        raise RuntimeError('Fit changed during comparison.')
    provenance = dict(directory=str(directory), manifest_sha256=sha256(manifest_path),
        settings_sha256=sha256(directory / 'settings.json'), status_sha256=sha256(directory / 'status.json'),
        chain_files=[dict(path=str(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in paths])
    return values, manifest, provenance


def compare(args):
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    output = driver.ROOT / 'comparison'
    output.mkdir(parents=True, exist_ok=True)
    missing = []
    for case in args.cases:
        for cosmo in args.cosmos:
            for mode in driver.MODES:
                directory = driver.fit_directory(args, case, mode, cosmo)
                state = driver.inspect_run(directory, read_steps=False)['state']
                if state != 'PASSED':
                    missing.append(dict(case=case, cosmo=cosmo, mode=mode, state=state, directory=str(directory)))
    (output / 'pending.json').write_text(json.dumps(missing, indent=2)+'\n')
    if missing:
        for row in missing:
            print('PENDING:', row, flush=True)
        raise RuntimeError('Comparison awaits all requested Gaussian fits passing checks; see comparison/pending.json.')
    figure, axes = plt.subplots(len(args.cosmos), len(args.cases), squeeze=False,
                               figsize=(4*len(args.cases), 3*len(args.cosmos)))
    rows, provenance, contrasts = [], [], []
    display_grid = np.linspace(-300, 300, 1200)
    for col, case in enumerate(args.cases):
        config, control = driver.load_control(case)
        means, stds = driver.moments(config, control)
        for row_index, cosmo in enumerate(args.cosmos):
            ax = axes[row_index, col]
            reference = driver.inference.load_run(driver.empirical_directory(case, cosmo))
            if reference is None:
                raise RuntimeError('Missing or unconfirmed empirical reference.')
            expected = {s: dict(v, prior='empirical') for s, v in config['tracers'].items()}
            if reference['specs'] != expected:
                raise RuntimeError('Empirical/Gaussian tracer settings differ beyond prior shape.')
            _, models = driver.inputs.prior_context(reference['directory'], reference['settings'])
            for sample, info in control['priors'].items():
                np.testing.assert_array_equal(np.sort(models[sample].values), np.sort(info['values']))
            empirical = np.asarray(reference['chain']['fnl_loc']).ravel()
            gaussian, _, gaussian_prov = load_gaussian(args, case, 'bphi', cosmo)
            products, _, product_prov = load_gaussian(args, case, 'bfnl', cosmo)
            rng = np.random.default_rng(42)
            selection = rng.choice(len(products), min(args.max_reconstruct, len(products)), replace=False)
            selected = products[selection]
            split = len(selected)//2
            grid, mass1 = product.reconstruct(selected[:split], means, stds)
            _, mass2 = product.reconstruct(selected[split:], means, stds)
            mass = (split*mass1+(len(selected)-split)*mass2)/len(selected)
            summary1, summary2 = product.summarize_grid(grid, mass1), product.summarize_grid(grid, mass2)
            stability = dict(split_mean_difference=summary1['mean']-summary2['mean'],
                             split_width68_fraction=summary1['width68']/summary2['width68']-1)
            summaries = {'empirical_bphi': sample_summary(empirical), 'gaussian_bphi': sample_summary(gaussian),
                         'gaussian_bfnl': product.summarize_grid(grid, mass)}
            for label, summary in summaries.items():
                rows.append(dict(case=case, cosmo=cosmo, method=label, **summary,
                    reconstruction_states=len(selected) if label == 'gaussian_bfnl' else 0))
                print(f'{case} c{cosmo} {label}: mean={summary["mean"]:.4f}, median={summary["median"]:.4f}, '
                      f'W68={summary["width68"]:.4f}, W95={summary["width95"]:.4f}', flush=True)
            for name, left, right in [('prior_shape', 'gaussian_bphi', 'empirical_bphi'),
                                      ('parameterization', 'gaussian_bfnl', 'gaussian_bphi')]:
                a, b = summaries[left], summaries[right]
                contrast = dict(case=case, cosmo=cosmo, comparison=name, delta_mean=a['mean']-b['mean'],
                    delta_median=a['median']-b['median'], median_shift_per_reference_halfwidth=
                    2*(a['median']-b['median'])/b['width68'], width68_change_pct=100*(a['width68']/b['width68']-1),
                    width95_change_pct=100*(a['width95']/b['width95']-1))
                contrasts.append(contrast)
                print('CONTRAST:', contrast, flush=True)
            for values, color, label in [(empirical, 'C0', 'Empirical bphi'), (gaussian, 'C1', 'Gaussian bphi')]:
                sample = values[np.linspace(0, len(values)-1, min(len(values), 12000), dtype=int)]
                ax.plot(display_grid, gaussian_kde(sample)(display_grid), color=color, label=label)
            _, logweights = product.quadrature()
            widths = np.exp(logweights)*(2*(product.FMAX-product.FMIN))
            ax.plot(grid, mass/widths, color='C2', ls='--', label='Gaussian bfnl')
            truth = driver.base.pipeline.true_fnl_loc_for_cosmo(cosmo)
            ax.axvline(truth, color='black', ls=':', lw=1)
            ax.set(xlim=(-200, 200), ylim=(0, None), title=f'{case}, input fNL={truth:+g}', xlabel=r'$f_{\rm NL}$')
            if col == 0:
                ax.set_ylabel('Posterior density')
            provenance.append(dict(case=case, cosmo=cosmo, gaussian_bphi=gaussian_prov, gaussian_bfnl=product_prov,
                empirical_directory=str(reference['directory']), empirical_settings_sha256=reference['settings_sha256'],
                empirical_status_sha256=reference['status_sha256'], reconstruction_seed=42,
                reconstruction_total_states=len(products), reconstruction_used_states=len(selected),
                reconstruction_stability=stability))
            print('RECONSTRUCTION SPLIT CHECK (not a calibrated MC error):', stability, flush=True)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    for ext in (('png', 'pdf') if args.pdf else ('png',)):
        figure.savefig(output / f'fnl_prior_mode_comparison.{ext}', dpi=180)
    plt.close(figure)
    for name, data in [('constraints', rows), ('contrasts', contrasts)]:
        with (output / f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (output / 'provenance.json').write_text(json.dumps(dict(inputs=provenance,
        plot_script_sha256=sha256(__file__), kernel_sha256=sha256(product.__file__),
        note='Direct fnl samples use all retained states; bfnl uses normalized conditional masses W/Q '
             'with fitting quadrature weights, not the old stats file. 50% burn-in. KDE is display-only. '
             'Split subsample checks are not MCSE or independent-seed tests; no equivalence is asserted automatically.'), indent=2)+'\n')
    print(f'Saved {output}', flush=True)
