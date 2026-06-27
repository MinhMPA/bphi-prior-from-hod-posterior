#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np

from bphi_prior_pipeline import (
    PNG_MODES,
    ell_tag,
    ensure_dir,
    flat_prior_abs,
    k_range_tag,
    normalize_ells,
)
from bphi_prior_plot_fnl import (
    clean_cosmo,
    cosmo_label,
    fnl_samples_from_chain,
    fnl_truth_label,
    infer_bfnl_chain_prior,
    load_chain,
    plot_hist_pair,
    redshift_label,
    summarize_samples,
    true_fnl_loc_for_cosmo,
    use_grid_plot_style,
    zset_tag,
    zsnaps_label,
)


def kmax_label(kmax):
    return rf"$k_{{\max}}={float(kmax):g}$"


def kmax_chain_path(args, zsnaps, cosmo, kmax):
    prior = "with_bphi_prior" if args.with_bphi_prior else "no_bphi_prior"
    abias = "_Bcentm0p040_Bsatm0p170" if args.assembly_bias else ""
    ztag = zset_tag(zsnaps)
    ktag = k_range_tag(args.kmin, kmax)
    return (
        Path(args.fit_dir)
        / f"png_fit_{args.simtype}_{args.tracer}_z{ztag}_c{clean_cosmo(cosmo)}_"
        f"{args.mode}{ell_tag(args.ells)}{abias}{ktag}_{prior}_chain.npy"
    )


def fnl_settings(args, chain_path):
    if args.mode == "bfnl":
        bphi_prior = args.bphi_prior
        if bphi_prior == "auto":
            bphi_prior = "gaussian" if args.with_bphi_prior else "flat"
        bfnl_chain_prior = args.bfnl_chain_prior
        if bfnl_chain_prior == "auto":
            bfnl_chain_prior = infer_bfnl_chain_prior(chain_path)
        return bphi_prior, bfnl_chain_prior
    return "direct", "flat"


def print_summary_table(rows):
    if not rows:
        return
    print("\nMarginalized fNL summary")
    print("cosmo z kmax q16 q50 q84 -1sigma +1sigma chain")
    for row in rows:
        print(
            f"{row['cosmo']} {row['zlabel']} {row['kmax']:.3f} "
            f"{row['q16']:.3f} {row['q50']:.3f} {row['q84']:.3f} "
            f"{row['q50'] - row['q16']:.3f} {row['q84'] - row['q50']:.3f} {row['path']}"
        )


def command_plot(args):
    import matplotlib.pyplot as plt

    use_grid_plot_style(plt)
    args.ells = normalize_ells(args.ells)
    cosmos = [clean_cosmo(cosmo) for cosmo in args.cosmos]
    zsets = [[float(zsnap)] for zsnap in args.zsnaps]
    kmax_values = [float(kmax) for kmax in args.kmax]
    output = args.output or Path(args.plot_dir) / "kmax_fnl_comparison.pdf"
    ensure_dir(output.parent)

    fig, axes = plt.subplots(
        len(cosmos),
        len(zsets),
        figsize=(3.6 * len(zsets), 2.7 * len(cosmos)),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    bins = np.linspace(-flat_prior_abs("fnl_loc"), flat_prior_abs("fnl_loc"), int(args.bins) + 1)
    stats_rows = []

    for irow, cosmo in enumerate(cosmos):
        for icol, zsnaps in enumerate(zsets):
            ax = axes[irow, icol]
            entries = []
            for ik, kmax in enumerate(kmax_values):
                path = kmax_chain_path(args, zsnaps, cosmo, kmax)
                if not path.exists():
                    message = f"Missing chain for {cosmo_label(cosmo)} {zsnaps_label(zsnaps)} {kmax_label(kmax)}: {path}"
                    if args.skip_missing:
                        print(message)
                        continue
                    raise FileNotFoundError(message)

                chain = load_chain(path, args.mode, zsnaps, args.tracer, burnin=args.burnin, thin=args.thin)
                bphi_prior, bfnl_chain_prior = fnl_settings(args, path)
                samples = fnl_samples_from_chain(
                    chain,
                    args.tracer,
                    zsnaps,
                    mode=args.mode,
                    bphi_prior=bphi_prior,
                    seed=args.seed + ik,
                    max_samples=args.max_samples,
                    fnl_method=args.fnl_method,
                    bfnl_chain_prior=bfnl_chain_prior,
                )
                label = kmax_label(kmax)
                q16, q50, q84 = summarize_samples(
                    f"{cosmo_label(cosmo)} {zsnaps_label(zsnaps)} {label}",
                    samples,
                    quantity="fNL",
                )
                stats_rows.append({
                    "cosmo": cosmo_label(cosmo),
                    "zlabel": zsnaps_label(zsnaps),
                    "kmax": kmax,
                    "q16": q16,
                    "q50": q50,
                    "q84": q84,
                    "path": path,
                })
                entries.append({
                    "label": label,
                    "samples": samples,
                    "color": f"C{ik}",
                    "linestyle": "-",
                })

            if entries:
                plot_hist_pair(
                    ax,
                    entries,
                    bins,
                    xlabel=r"$f_{\mathrm{NL}}^{\mathrm{loc}}$" if irow == len(cosmos) - 1 else None,
                    truth=true_fnl_loc_for_cosmo(cosmo),
                    title=zsnaps_label(zsnaps) if irow == 0 else None,
                    show_ylabel=icol == 0,
                )
            else:
                ax.text(0.5, 0.5, "missing chains", transform=ax.transAxes, ha="center", va="center")
            if icol == 0:
                ax.text(0.03, 0.88, fnl_truth_label(cosmo), transform=ax.transAxes, fontsize=11, weight="bold")
            if irow == 0 and icol == len(zsets) - 1 and entries:
                ax.legend(frameon=False, fontsize=9)

    fig.tight_layout(pad=0.2, h_pad=0.0, w_pad=0.35)
    fig.subplots_adjust(hspace=0.0)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print_summary_table(stats_rows)
    print(f"Saved {output}")


def build_parser():
    parser = argparse.ArgumentParser(description="Compare marginalized fNL posteriors across kmax choices.")
    parser.add_argument("--fit-dir", type=Path, default=Path("bphi_prior/fit"))
    parser.add_argument("--plot-dir", type=Path, default=Path("bphi_prior/plots"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--simtype", default="PNG", choices=["PNG", "SU"])
    parser.add_argument("--tracer", default="LRG", choices=["LRG"])
    parser.add_argument("--cosmos", nargs="+", default=["301"])
    parser.add_argument("--zsnaps", nargs="+", type=float, default=[0.5, 0.8])
    parser.add_argument("--kmin", type=float, default=0.006)
    parser.add_argument("--kmax", nargs="+", type=float, default=[0.06, 0.08, 0.1])
    parser.add_argument("--ells", nargs="+", type=int, default=[0, 2])
    parser.add_argument("--mode", default="bfnl", choices=PNG_MODES)
    parser.add_argument("--with-bphi-prior", dest="with_bphi_prior", action="store_true", default=True)
    parser.add_argument("--no-bphi-prior", dest="with_bphi_prior", action="store_false")
    parser.add_argument("--assembly-bias", action="store_true")
    parser.add_argument("--bphi-prior", choices=["auto", "gaussian", "flat", "direct"], default="auto")
    parser.add_argument("--bfnl-chain-prior", choices=["auto", "flat", "induced"], default="auto")
    parser.add_argument("--fnl-method", choices=["conditional", "description"], default="description")
    parser.add_argument("--burnin", type=float, default=0.5)
    parser.add_argument("--thin", type=int, default=5)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-missing", action="store_true")
    parser.set_defaults(func=command_plot)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
