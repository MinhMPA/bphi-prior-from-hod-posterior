#!/usr/bin/env python3
"""Reproduce the power-spectrum comparison from the released CSV files."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'data' / 'paper_pk')
    parser.add_argument('--output', type=Path, default=Path('released_pk.pdf'))
    args = parser.parse_args()
    fits = json.loads((args.data_dir / 'manifest.json').read_text())['fits']
    records = {}
    for fit in fits:
        data = np.genfromtxt(args.data_dir / fit['spectrum_file'], delimiter=',', names=True)
        cov = np.loadtxt(args.data_dir / fit['covariance_file'], delimiter=',')
        n = len(data)
        if cov.shape != (2*n, 2*n):
            raise ValueError(f'Unexpected covariance shape for {fit["sample"]} c{fit["cosmo"]}')
        np.testing.assert_allclose(np.diag(cov), np.r_[data['sigma_P0']**2, data['sigma_P2']**2], rtol=1e-12)
        records[(fit['sample'], fit['cosmo'])] = (fit, data)

    plt.rcParams.update({'font.size': 11})
    fig = plt.figure(figsize=(15, 9.5))
    outer = fig.add_gridspec(2, 3, left=.105, right=.98, bottom=.10, top=.88, wspace=0, hspace=.05)
    colors = ('#0072B2', '#D55E00', '#8E5AA7')
    markers = ('o', 's', '^')
    cosmos = ('000', '300', '301')
    truths = (0, 30, -30)
    first_res = None
    for row, ell in enumerate((0, 2)):
        first_top = None
        for col, sample in enumerate(('LRG1', 'LRG2', 'LRG3')):
            inner = outer[row, col].subgridspec(2, 1, height_ratios=(2, 1), hspace=0)
            top = fig.add_subplot(inner[0], sharey=first_top)
            res = fig.add_subplot(inner[1], sharex=top, sharey=first_res)
            first_top = top if first_top is None else first_top
            first_res = res if first_res is None else first_res
            res.axhspan(-1, 1, color='0.93', lw=0)
            res.axhline(0, color='0.3', lw=.8)
            for cosmo, truth, color, marker in zip(cosmos, truths, colors, markers):
                fit, data = records[(sample, cosmo)]
                k = data['k_h_Mpc']
                pk = data[f'P{ell}']
                model = data[f'bestfit_P{ell}']
                sigma = data[f'sigma_P{ell}']
                offset = np.sign(truth) * (200 if ell == 0 else 400)
                top.errorbar(k, k*pk+offset, yerr=k*sigma, fmt=marker, color=color,
                             ms=3, elinewidth=.7, capsize=0, alpha=.8)
                top.plot(k, k*model+offset, color=color, lw=1.35)
                res.plot(k, (pk-model)/sigma, color=color, marker=marker, lw=.65, ms=3, alpha=.8)
            res.set_xlim(fit['kmin'], fit['kmax'])
            res.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
            for ax in (top, res):
                ax.tick_params(direction='in', top=True, right=True, labelsize=10, labelleft=(col == 0))
            top.tick_params(labelbottom=False)
            if ell == 0:
                top.set_title(rf'{sample} ($k_{{\max}}={fit["kmax"]:.2f}$)', fontsize=13)
                res.tick_params(labelbottom=False)
            else:
                res.set_xlabel(r'$k\ [h\,{\rm Mpc}^{-1}]$', fontsize=14)
            if col == 0:
                top.set_ylabel(rf'$kP_{ell}$ + offset' + '\n' + r'$[h^{-2}{\rm Mpc}^2]$', fontsize=13)
                res.set_ylabel(r'$(P_\ell-P_\ell^{\rm model})/\sigma_\ell$', fontsize=12)
        if ell == 0:
            first_top.set_ylim(bottom=-100)
    handles = [Line2D([], [], color=c, marker=m, lw=1.3, ms=4,
                      label=rf'$f_{{\rm NL}}^{{\rm true}}={truth:+g}$' if truth
                      else r'$f_{\rm NL}^{\rm true}=0$')
               for truth, c, m in zip(truths, colors, markers)]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(.5, .965), fontsize=13)
    fig.text(.5, .915, r'Display offsets for $f_{\rm NL}=\pm30$: '
             r'$\Delta(kP_0)=\pm200$, $\Delta(kP_2)=\pm400$', ha='center', va='center', fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
