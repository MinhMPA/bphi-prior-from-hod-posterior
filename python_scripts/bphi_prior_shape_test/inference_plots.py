"""Read-only plots for unified or frozen legacy posterior chains."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import inference as driver
import inference_inputs as inputs


def compare(args):
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from bphi_prior_pipeline import true_fnl_loc_for_cosmo
    configs = ([] if getattr(args, 'references_only', False) else
               [args.spec] + [inputs.read_config(p) for p in args.comparison_config])
    output = getattr(args, 'plot_output', None) or args.output
    for config in configs:
        inputs.load_manifest(config, inputs.OUTPUT / config['name'])
    fig, axes = plt.subplots(len(args.cosmos), 1, squeeze=False, figsize=(7, 3.2*len(args.cosmos)))
    rows, provenance = [], []
    for ax, cosmo in zip(axes.flat, args.cosmos):
        truth = true_fnl_loc_for_cosmo(cosmo)
        runs = []
        for i, ref in enumerate(args.reference):
            label = args.reference_label[i] if args.reference_label else f'Reference {i+1}'
            seeds = args.seeds if '{seed}' in ref else [args.seeds[0]]
            for seed in seeds:
                run_label = f'{label}, seed {seed}' if len(seeds) > 1 else label
                runs.append((run_label, Path(ref.format(cosmo=cosmo, seed=seed)), True))
        for config in configs:
            for seed in args.seeds:
                label = f"{config['name']}, {args.sampler}"
                if len(args.seeds) > 1:
                    label += f', seed {seed}'
                runs.append((label, inputs.run_directory(inputs.OUTPUT / config['name'], cosmo,
                                                        args.sampler, args.nwalkers, seed), False))
        if len({p for _, p, _ in runs}) != len(runs):
            raise ValueError('Duplicate comparison directories.')
        for i, (label, directory, is_reference) in enumerate(runs):
            result = driver.load_run(directory, args.allow_unconverged)
            if result is None:
                continue
            settings, status = result['settings'], result['status']
            if settings['cosmo'] != cosmo or settings.get('fnl_limits') != [-300, 300]:
                raise ValueError('Comparison requires the requested cosmology and unchanged fNL support.')
            values = np.asarray(result['chain']['fnl_loc']).ravel()
            low, median, high = np.quantile(values, [.16, .5, .84])
            mean = float(np.mean(values))
            rows.append(dict(cosmo=cosmo, label=label, sampler=settings.get('sampler', 'legacy'),
                seed=settings['seed'], nwalkers=settings['nwalkers'], converged=status['converged'],
                n_states=len(values), mean=mean, q16=low, median=median, q84=high, width68=high-low,
                mean_minus_true=mean-truth,
                fnl_true=truth, truth_in_68=bool(low <= truth <= high), directory=str(directory),
                tracers=json.dumps(result['specs'], sort_keys=True)))
            provenance.append({k: str(v) for k, v in result.items() if k.endswith('sha256') or k == 'directory'})
            print(f'c{cosmo} {label}: mean={mean:.4f}; median={median:.4f} '
                  f'+{high-median:.4f} -{median-low:.4f}; W68={high-low:.4f}; '
                  f'mean-input={mean-truth:+.4f}; input_in_68={low <= truth <= high}', flush=True)
            display = values[np.linspace(0, len(values)-1, min(len(values), 12000), dtype=int)]
            grid = np.linspace(-300, 300, 1000)
            if not status['converged']:
                label += ' (unconfirmed)'
            ax.plot(grid, gaussian_kde(display)(grid), color=f'C{i % 10}', ls='--' if is_reference else '-', label=label)
        ax.axvline(truth, color='k', ls=':', lw=1)
        ax.set(title=rf'$f_{{\rm NL}}^{{\rm input}}={truth:+g}$', xlabel=r'$f_{\rm NL}^{\rm loc}$',
               ylabel='Posterior density', xlim=(-300, 300), ylim=(0, None))
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    if rows:
        settings = dict(configs=configs, cosmos=args.cosmos, sampler=args.sampler, walkers=args.nwalkers,
                        seeds=args.seeds, references=args.reference, labels=args.reference_label)
        tag = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
        for name in ('plots', 'tables'):
            (output / name).mkdir(parents=True, exist_ok=True)
        for ext in (('png', 'pdf') if getattr(args, 'pdf', False) else ('png',)):
            path = output / 'plots' / f'fnl_comparison_{tag}.{ext}'
            fig.savefig(path, dpi=180)
            print(f'Saved {path}', flush=True)
        with (output / 'tables' / f'fnl_comparison_{tag}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (output / 'tables' / f'fnl_comparison_{tag}.json').write_text(
            json.dumps(dict(settings=settings, inputs=provenance,
                note='Direct marginalized fNL samples, 50% burn-in, no reweighting; KDE thinning is display-only.'), indent=2)+'\n')
        if getattr(args, 'kmax_summary', False):
            summarize_kmax(rows, args, tag)
    else:
        print('No completed fits available; no comparison written.', flush=True)
    plt.close(fig)


def kmax_rows(rows, cosmos):
    """Summarize matched single-tracer cuts, not independent-mock significances."""
    groups, signature = {}, None
    for row in rows:
        specs = json.loads(row['tracers'])
        if len(specs) != 1 or not row['converged']:
            raise ValueError('The kmax summary requires passed single-tracer fits.')
        sample, spec = next(iter(specs.items()))
        current = dict(sample=sample, spec={k: v for k, v in spec.items() if k != 'kmax'},
                       sampler=row['sampler'], seed=row['seed'], nwalkers=row['nwalkers'])
        if signature is not None and current != signature:
            raise ValueError('Keep tracer, ensemble, prior, kmin and sampler settings matched across cuts.')
        signature = current
        group = groups.setdefault(spec['kmax'], {})
        if row['cosmo'] in group:
            raise ValueError('Select one run per cosmology and cut; do not count duplicate controls.')
        if row['width68'] <= 0:
            raise ValueError('Nonpositive credible interval width.')
        group[row['cosmo']] = row
    summaries = []
    for cut, group in sorted(groups.items()):
        if set(group) != set(cosmos):
            print(f'Incomplete cosmology set at kmax={cut:g}; excluded from the all-cosmology summary.', flush=True)
            continue
        selected = [group[c] for c in cosmos]
        offsets = [r['mean']-r['fnl_true'] for r in selected]
        summaries.append(dict(kmax=cut, n_cosmos=len(selected),
            mean_width68=float(np.mean([r['width68'] for r in selected])),
            worst_abs_mean_offset=max(abs(v) for v in offsets),
            worst_offset_per_halfwidth68=max(2*abs(v)/r['width68'] for v, r in zip(offsets, selected))))
    if summaries:
        baseline = summaries[0]['mean_width68']
        for row in summaries:
            row['width_reduction_vs_smallest_cut_pct'] = 100*(1-row['mean_width68']/baseline)
    return groups, summaries


def summarize_kmax(rows, args, tag):
    import matplotlib.pyplot as plt
    output = getattr(args, 'plot_output', None) or args.output
    groups, summaries = kmax_rows(rows, args.cosmos)
    if not summaries:
        print('No complete matched-cosmology scale-cut summaries.', flush=True)
        return
    cuts = [s['kmax'] for s in summaries]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, cosmo in enumerate(args.cosmos):
        selected = [groups[k][cosmo] for k in cuts]
        truth = selected[0]['fnl_true']
        label = rf'$f_{{\rm NL}}^{{\rm input}}={truth:+g}$'
        axes[0].plot(cuts, [r['mean']-truth for r in selected], 'o-', color=f'C{i}', label=label)
        axes[1].plot(cuts, [r['width68'] for r in selected], 'o-', color=f'C{i}', label=label)
    axes[0].axhline(0, color='k', ls=':', lw=1)
    axes[0].set_ylabel(r'Posterior mean $- f_{\rm NL}^{\rm input}$')
    axes[1].set_ylabel(r'$W_{68}=q_{84}-q_{16}$')
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xlabel(r'$k_{\max}\ [h/{\rm Mpc}]$')
        ax.set_xticks(cuts)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in (('png', 'pdf') if getattr(args, 'pdf', False) else ('png',)):
        path = output / 'plots' / f'fnl_kmax_summary_{tag}.{ext}'
        fig.savefig(path, dpi=180)
        print(f'Saved {path}', flush=True)
    plt.close(fig)
    path = output / 'tables' / f'fnl_kmax_summary_{tag}.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print('Matched-cosmology diagnostics (equal case weights; not independent-realization bias or Gaussian significance):', flush=True)
    for row in summaries:
        print(f"kmax={row['kmax']:.2f}: average W68={row['mean_width68']:.4f}; "
              f"max |mean-input|={row['worst_abs_mean_offset']:.4f}; "
              f"max |mean-input|/(W68/2)={row['worst_offset_per_halfwidth68']:.4f}", flush=True)
    path.with_suffix('.json').write_text(json.dumps(dict(cosmos=args.cosmos, cuts=cuts,
        comparison_csv=f'fnl_comparison_{tag}.csv', comparison_provenance=f'fnl_comparison_{tag}.json',
        note='W68 is the full central 68% credible interval width, not a standard deviation. '
             'The maximum normalized offset is descriptive, not a sigma significance or a calibrated cutoff criterion. '
             'These same-data scale-cut fits are correlated; no independent-error test or preferred cut is inferred. '
             'Width reductions use the smallest complete cut shown as the reference.'), indent=2)+'\n')


def plot_priors(args, manifest):
    import matplotlib.pyplot as plt
    output = getattr(args, 'plot_output', None) or args.output
    models = inputs.models_from_manifest(manifest)
    fig, axes = plt.subplots(2, len(models), squeeze=False, figsize=(4*len(models), 6))
    for col, (sample, model) in enumerate(models.items()):
        spec = args.spec['tracers'][sample]
        lo = min(model.mean-4*model.std, min(model.values)-4*model.bandwidth)
        hi = max(model.mean+4*model.std, max(model.values)+4*model.bandwidth)
        grid = np.linspace(lo, hi, 1000)
        upper, lower = axes[:, col]
        upper.hist(model.values, bins='fd', density=True, histtype='step', color='C1', label='100 HOD responses')
        for kind, color in [('gaussian', 'C0'), ('kde', 'C2')]:
            style = '-' if spec['prior'] == kind else '--'
            upper.plot(grid, model.pdf(kind, grid), color=color, ls=style, label=kind)
            lower.plot(grid, model.cdf(kind, grid), color=color, ls=style)
        lower.step(np.r_[lo, np.sort(model.values), hi], np.r_[0, np.arange(1, 101)/100, 1],
                   where='post', color='C1')
        upper.set(title=f"{sample}: {spec['ensemble']} ({spec['prior']})", ylim=(0, None), xlim=(lo, hi))
        upper.legend(frameon=False, fontsize=8)
        lower.set(xlabel=r'$b_\phi$', ylim=(0, 1), xlim=(lo, hi))
    axes[0, 0].set_ylabel('Probability density')
    axes[1, 0].set_ylabel('Cumulative probability')
    fig.tight_layout()
    (output / 'plots').mkdir(parents=True, exist_ok=True)
    for ext in (('png', 'pdf') if getattr(args, 'pdf', False) else ('png',)):
        path = output / 'plots' / f'bphi_priors.{ext}'
        fig.savefig(path, dpi=180)
        print(f'Saved {path}', flush=True)
    plt.close(fig)


def parse_args(argv=None):
    # Keep export options outside inference.py: saved fits freeze its source hash.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--pdf', action='store_true', help='Also save PDF figures (default: PNG only).')
    parser.add_argument('--kmax-summary', action='store_true', help='Also summarize matched single-tracer scale cuts.')
    parser.add_argument('--plot-output', type=Path, help='Write plots/tables here without changing fit/input paths.')
    parser.add_argument('--references-only', action='store_true', help='Compare only saved --reference runs; no config manifest required.')
    export, remaining = parser.parse_known_args(argv)
    if '--help' in remaining or '-h' in remaining:
        print('Plot export option: --pdf also saves PDF figures; PNG is always saved.\n')
        print('Comparison option: --kmax-summary adds posterior-mean and 68%-width diagnostics.\n')
        print('Output: --plot-output PATH; --references-only compares saved references without a new fit.\n')
    args = driver.parse_args(remaining)
    if args.command not in ('compare', 'prior-plot'):
        parser.error('This plotting entry point only supports compare and prior-plot.')
    if export.kmax_summary and args.command != 'compare':
        parser.error('--kmax-summary is only supported for compare.')
    if export.references_only and (args.command != 'compare' or not args.reference or args.comparison_config):
        parser.error('--references-only requires compare with --reference and no --comparison-config.')
    args.pdf = export.pdf
    args.kmax_summary = export.kmax_summary
    args.plot_output = export.plot_output
    args.references_only = export.references_only
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == 'compare':
        compare(args)
    else:
        manifest = inputs.load_manifest(args.spec, args.output)
        plot_priors(args, manifest)


if __name__ == '__main__':
    main()
