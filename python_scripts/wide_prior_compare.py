"""Compare joint product chains and the paper's flat-bphi Bayesian marginalization."""
import csv
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import paper_wide_prior as driver

RESPONSE_REFERENCE = Path(os.environ.get('BPHI_RESULTS_ROOT', Path(__file__).resolve().parent.parent / 'outputs')).expanduser() / 'fnl_offset_diagnostic/tables/data_hod_response.json'


def product_reference(manifest, cosmo):
    if not RESPONSE_REFERENCE.exists():
        print('No default-HOD response file; nonzero product truth is not assumed.', flush=True)
        return None
    payload = json.loads(RESPONSE_REFERENCE.read_text())
    values = []
    for z in driver.base.ZS:
        row = payload['snapshots'][str(z)]
        for name in ('SU_c000', 'SU_c004', f'PNG_c{cosmo}'):
            path = row['catalogues'][name]['path']
            digest = payload['sources_sha256'][path]
            if driver.base.sha256(path) != digest:
                raise RuntimeError(f'HOD-response input changed: {path}')
            if name.startswith('PNG') and manifest['files'].get(path) != digest:
                raise RuntimeError('HOD-response reference does not match the fitted PNG data.')
        values.append(float(row['bphi'])*driver.base.pipeline.true_fnl_loc_for_cosmo(cosmo))
    return dict(path=str(RESPONSE_REFERENCE), sha256=driver.base.sha256(RESPONSE_REFERENCE),
                products=values, note=payload['note'])


def add_reference(ax, value, label):
    ax.axvline(value, color='black', ls=':', lw=1.8, zorder=6)
    ax.text(.98, 1.02, label, transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            clip_on=False, zorder=7)


class FlatJointPosterior:
    """Exact integral of the empirical sum 1[max|B| <= a|f|] / |f|^3."""

    def __init__(self, products, bphi_limit, fnl_limit=300.):
        products = np.asarray(products)
        if products.ndim != 2 or products.shape[1] != 3 or not len(products) or not np.all(np.isfinite(products)):
            raise ValueError('Need finite joint (samples, 3) products.')
        if bphi_limit <= 0 or fnl_limit <= 0:
            raise ValueError('Prior limits must be positive.')
        thresholds = np.max(np.abs(products), axis=1)/bphi_limit
        if np.any(thresholds == 0):
            raise ValueError('Exactly zero product vector has a singular importance weight; inspect the chain.')
        self.t = np.sort(thresholds[thresholds < fnl_limit])
        if not len(self.t):
            raise ValueError('No states contribute inside the requested reconstruction support.')
        self.fmax = fnl_limit
        self.prefix = np.r_[0., np.cumsum(self.t**-2)]
        weights = .5*(self.t**-2-fnl_limit**-2)
        self.norm = float(weights.sum())
        self.importance_ess = float(weights.sum()**2/(weights @ weights))
        self.max_weight_fraction = float(weights.max()/weights.sum())
        self.contributing_states = len(self.t)

    def abs_cdf(self, x):
        if x <= self.t[0]:
            return 0.
        if x >= self.fmax:
            return 1.
        n = np.searchsorted(self.t, x, side='right')
        return float(np.clip(.5*(self.prefix[n]-n*x**-2)/self.norm, 0., 1.))

    def density(self, grid):
        absolute = np.abs(np.asarray(grid))
        result = np.zeros_like(absolute, dtype=float)
        valid = (absolute > 0) & (absolute <= self.fmax)
        counts = np.searchsorted(self.t, absolute[valid], side='right')
        result[valid] = counts/absolute[valid]**3/(2*self.norm)
        return result

    def summary(self):
        r68, r95 = [brentq(lambda x: self.abs_cdf(x)-q, self.t[0], self.fmax) for q in (.68, .95)]
        return dict(mean=0., median=0., q16=-r68, q84=r68, width68=2*r68,
                    q025=-r95, q975=r95, width95=2*r95)


def load_products(args, cosmo, wide):
    from desilike.samples import Chain
    directory = driver.directory(args, cosmo, wide)
    if driver.inspect_run(directory, read_steps=False)['state'] != 'PASSED':
        raise RuntimeError(f'Wait for passed sampler checks: {directory}')
    root = driver.case_root(args) if wide else driver.weak.case_root('joint', args.kmaxes)
    path = root / f'c{cosmo}' / 'manifest.json'
    manifest = driver.verify_manifest(path)
    expected = [-9000., 9000.] if wide else [-3000., 3000.]
    if manifest['bfnl_limits'] != expected or manifest['tracers'] != driver.weak.spec_for('joint', args.kmaxes):
        raise RuntimeError('Product support or scale-cut mismatch.')
    settings = json.loads((directory / 'settings.json').read_text())
    if settings['manifest_sha256'] != driver.base.sha256(path):
        raise RuntimeError('Fit/manifest mismatch.')
    if settings['sampler_sha256'] != driver.base.sha256(settings['sampler_source']):
        raise RuntimeError('Sampler implementation changed.')
    driver.base.checkpoint(directory, args.nwalkers)
    status = json.loads((directory / 'status.json').read_text())
    paths = [directory / f'chain_{i+1}.npy' for i in range(args.nwalkers)]
    watched = paths + [directory / 'status.json', directory / 'settings.json']
    stamps = [(p.stat().st_size, p.stat().st_mtime_ns) for p in watched]
    names = [driver.base.pipeline.tracer_z_label('LRG', z)+'.bfnl_loc' for z in driver.base.ZS]
    arrays = []
    for path in paths:
        chain = Chain.load(path)
        if len(chain) != status['steps_per_walker']:
            raise RuntimeError('Status and checkpoint lengths differ.')
        chain = chain.remove_burnin(.5)
        if not np.all(np.asarray(chain.weight) == 1):
            raise RuntimeError('Expected equally weighted states.')
        for name in names:
            if list(chain[name].param.prior.limits) != expected:
                raise RuntimeError('Chain product prior differs from manifest.')
        arrays.append(np.column_stack([np.asarray(chain[name]).ravel() for name in names]))
    if stamps != [(p.stat().st_size, p.stat().st_mtime_ns) for p in watched]:
        raise RuntimeError('Chain changed while reading.')
    result = np.concatenate(arrays)
    if not np.all(np.isfinite(result)):
        raise RuntimeError('Nonfinite retained products.')
    return result, manifest, dict(directory=str(directory), manifest_sha256=settings['manifest_sha256'],
        seed=settings['seed'], sampler=settings['sampler'],
        settings_sha256=driver.base.sha256(directory / 'settings.json'),
        status_sha256=driver.base.sha256(directory / 'status.json'),
        chain_stamps=[dict(path=str(p), size=s[0], mtime_ns=s[1]) for p, s in zip(watched, stamps)])


def compare(args):
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    output = driver.case_root(args) / 'comparison'
    output.mkdir(parents=True, exist_ok=True)
    pending = []
    for cosmo in args.cosmos:
        for wide in (False, True):
            path = driver.directory(args, cosmo, wide)
            state = driver.inspect_run(path, read_steps=False)['state']
            if state != 'PASSED':
                pending.append(dict(path=str(path), state=state))
    (output / 'pending.json').write_text(json.dumps(pending, indent=2)+'\n')
    if pending:
        raise RuntimeError(f'Comparison awaits {len(pending)} fits; see {output / "pending.json"}.')
    figure, axes = plt.subplots(len(args.cosmos), 4, squeeze=False, figsize=(16, 3.5*len(args.cosmos)))
    rows, provenance, contrasts = [], [], []
    for row, cosmo in enumerate(args.cosmos):
        old, old_manifest, old_prov = load_products(args, cosmo, False)
        new, new_manifest, new_prov = load_products(args, cosmo, True)
        if old_manifest['files'] != new_manifest['files']:
            raise RuntimeError('Baseline and wide data/covariance input hashes differ.')
        reference = product_reference(old_manifest, cosmo)
        identical = bool(np.array_equal(old, new))
        print(f'c{cosmo}: baseline={old_prov.get("directory")}; wide={new_prov.get("directory")}; '
              f'identical retained product states={identical}', flush=True)
        for i, (label, products) in enumerate([('Baseline B +/-3000', old), ('Wide B +/-9000', new)]):
            for j in range(3):
                values = products[:, j]
                q025, q16, median, q84, q975 = np.quantile(values, [.025, .16, .5, .84, .975])
                rows.append(dict(cosmo=cosmo, result=label, parameter=f'B_LRG{j+1}', mean=float(values.mean()),
                    median=median, q16=q16, q84=q84, width68=q84-q16, q025=q025, q975=q975, width95=q975-q025))
                display = values[np.linspace(0, len(values)-1, min(12000, len(values)), dtype=int)]
                grid = np.linspace(*np.quantile(values, [.001, .999]), 500)
                axes[row, j].plot(grid, gaussian_kde(display)(grid), color=f'C{i}', label=label,
                                  ls='-' if i == 0 else '--', lw=2.8 if i == 0 else 1.8, zorder=2+i)
                axes[row, j].set(xlabel=rf'$B_{{{j+1}}}=b_{{\phi,{j+1}}}f_{{\rm NL}}$', ylim=(0, None))
        truth = driver.base.pipeline.true_fnl_loc_for_cosmo(cosmo)
        for j in range(3):
            if truth == 0:
                add_reference(axes[row, j], 0., r'True $B=0$')
            elif reference is not None:
                value = reference['products'][j]
                add_reference(axes[row, j], value, rf'HOD ref. $B={value:+.1f}$')
        if identical:
            axes[row, 0].text(.03, .04, 'Baseline and wide B overlap', transform=axes[row, 0].transAxes,
                              fontsize=8, bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=2))
        summaries, diagnostics = {}, {}
        for i, (label, products, limit) in enumerate([
                ('Baseline: bphi +/-10', old, 10.), ('Wide B only: bphi +/-10', new, 10.),
                ('Wide B and bphi +/-30', new, 30.)]):
            model = FlatJointPosterior(products, limit)
            summary = model.summary()
            summaries[label] = summary
            rows.append(dict(cosmo=cosmo, result=label, parameter='fnl_reconstructed', **summary))
            split = len(products)//2
            halves = [FlatJointPosterior(p, limit).summary() for p in (products[:split], products[split:])]
            diagnostics[label] = dict(importance_ess=model.importance_ess, max_weight_fraction=model.max_weight_fraction,
                contributing_states=model.contributing_states, total_states=len(products),
                split_width68_difference=halves[0]['width68']-halves[1]['width68'])
            unstable = (model.importance_ess < 100 or model.max_weight_fraction > .05
                        or abs(halves[0]['width68']-halves[1]['width68']) > .1*summary['width68'])
            diagnostics[label]['stability_warning'] = bool(unstable)
            if unstable:
                print(f'WARNING c{cosmo} {label}: importance concentration or split instability; '
                      'do not interpret this as a settled prior-width effect.', flush=True)
            print(f'c{cosmo} {label}: {summary}; diagnostics: {diagnostics[label]}', flush=True)
            positive = np.geomspace(model.t[0]*.5, 300., 2000)
            grid = np.r_[-positive[::-1], 0., positive]
            axes[row, 3].plot(grid, model.density(grid), color=f'C{i}', label=label,
                              ls=('-', '--', '-.')[i], lw=2.8 if i == 0 else 1.8, zorder=2+i)
        for label in list(summaries)[1:]:
            a, b = summaries[label], summaries['Baseline: bphi +/-10']
            contrasts.append(dict(cosmo=cosmo, result=label,
                width68_change_pct=100*(a['width68']/b['width68']-1), width95_change_pct=100*(a['width95']/b['width95']-1)))
        axes[row, 0].set_ylabel(f'c{cosmo}, input fNL={truth:+g}\nPosterior density')
        axes[row, 3].set(xlabel=r'$f_{\rm NL}$', xlim=(-300, 300), ylim=(0, None))
        add_reference(axes[row, 3], truth, rf'True $f_{{\rm NL}}={truth:+g}$')
        provenance.append(dict(cosmo=cosmo, baseline=old_prov, wide=new_prov, diagnostics=diagnostics,
                               identical_product_states=identical, product_reference=reference, true_fnl=truth))
    figure.legend(*axes[0, 0].get_legend_handles_labels(), fontsize=9, frameon=False,
                  loc='upper left', bbox_to_anchor=(.06, .995), ncol=2)
    figure.legend(*axes[0, 3].get_legend_handles_labels(), fontsize=9, frameon=False,
                  loc='upper right', bbox_to_anchor=(.995, .995), ncol=3)
    figure.tight_layout(rect=(0., 0., 1., .94))
    for ext in (('png', 'pdf') if args.pdf else ('png',)):
        figure.savefig(output / f'joint_wide_prior.{ext}', dpi=180)
    plt.close(figure)
    for name, data in [('constraints', rows), ('contrasts', contrasts)]:
        with (output / f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
        for result in data:
            print(result, flush=True)
    (output / 'provenance.json').write_text(json.dumps(dict(inputs=provenance,
        script_sha256=driver.base.sha256(__file__), burnin=.5,
        method='All retained states; analytic integral of the paper flat-product/flat-bphi importance sum. '
        'No per-state normalization, induced-Q division or random fNL draws. Symmetric bphi support forces '
        'symmetric reconstructed fNL; zero mean is not evidence of unbiasedness. Importance ESS ignores '
        'autocorrelation; split checks are not calibrated Monte Carlo errors.'), indent=2)+'\n')
    print(f'Saved {output}', flush=True)
