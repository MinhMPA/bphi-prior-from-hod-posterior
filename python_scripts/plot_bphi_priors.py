#!/usr/bin/env python3
"""Plot the four released empirical priors and moment-matched Gaussians."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SAMPLES = ('LRG1', 'LRG2', 'LRG3', 'LRG3_AB')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(__file__).resolve().parents[1] /
                        'data/bphi_prior_samples.csv')
    parser.add_argument('--output', type=Path, default=Path('bphi_prior_distributions.png'))
    args = parser.parse_args()
    with args.input.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    groups = {sample: np.array([float(r['bphi']) for r in rows if r['sample'] == sample])
              for sample in SAMPLES}
    for sample, values in groups.items():
        if len(values) != 100 or not np.isfinite(values).all() or values.std() <= 0:
            raise ValueError(f'{sample}: expected 100 finite, nonconstant responses')
    fig, axes = plt.subplots(1, 4, figsize=(15.3, 3.5))
    lrg3 = np.concatenate([groups['LRG3'], groups['LRG3_AB']])
    common_edges = np.linspace(lrg3.min(), lrg3.max(), 13)
    for ax, sample, color in zip(axes, SAMPLES, ('#D55E00', '#D55E00', '#0072B2', '#009E73')):
        values = groups[sample]
        edges = common_edges if sample.startswith('LRG3') else np.linspace(values.min(), values.max(), 13)
        mean, std = values.mean(), values.std(ddof=0)
        grid = np.linspace(min(edges[0], mean-3.5*std), max(edges[-1], mean+3.5*std), 500)
        ax.hist(values, bins=edges, density=True, histtype='stepfilled', facecolor=color, alpha=.12)
        ax.hist(values, bins=edges, density=True, histtype='step', color=color, lw=1.7, label='Empirical')
        pdf = np.exp(-.5*((grid-mean)/std)**2)/(std*np.sqrt(2*np.pi))
        ax.plot(grid, pdf, color='black', ls='--', lw=1.5, label='Gaussian')
        ax.plot(values, np.full(len(values), .025), '|', color=color, alpha=.55,
                markersize=6, transform=ax.get_xaxis_transform())
        title = sample.replace('_AB', ' AB') + (': $s=3$' if sample in ('LRG1', 'LRG2') else ': $s=1$')
        ax.set(title=title, xlabel=r'$b_\phi$', ylim=(0, None))
        ax.tick_params(direction='in', top=True, right=True)
        print(f'{sample}: mean={mean:.8f}, variance={std**2:.8f}, std={std:.8f}')
    xlim = (min(axes[2].get_xlim()[0], axes[3].get_xlim()[0]),
            max(axes[2].get_xlim()[1], axes[3].get_xlim()[1]))
    ymax = max(axes[2].get_ylim()[1], axes[3].get_ylim()[1])
    for ax in axes[2:]:
        ax.set_xlim(*xlim)
        ax.set_ylim(0, ymax)
    axes[0].set_ylabel('Probability density')
    axes[0].legend(frameon=False, fontsize=9)
    fig.subplots_adjust(left=.07, right=.99, bottom=.19, top=.86, wspace=.25)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
