#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
from desilike.samples import Chain

from bphi_prior_pipeline import (
    BPHI_BFNL_PRIOR_NGRID,
    DEFAULT_ZS,
    PNG_MODES,
    _broadcast_plot_arg,
    add_bfnl_loc_to_chain,
    add_fnl_loc_to_bfnl_chain,
    ensure_dir,
    find_param_key,
    flat_prior_abs,
    load_assembly_bias_bphi_prior,
    load_bphi_stats,
    logsumexp,
    parse_zsnaps_text,
    tracer_z_label,
    true_fnl_loc_for_cosmo,
)


def cosmo_label(cosmo):
    return f"c{str(cosmo).lower().removeprefix('c').zfill(3)}"


def fnl_truth_label(cosmo):
    truth = true_fnl_loc_for_cosmo(cosmo)
    if truth is None:
        return cosmo_label(cosmo)
    return rf"$f_{{\rm NL}}={truth:+.0f}$".replace("=+0", "=0")


def clean_cosmo(cosmo):
    return str(cosmo).lower().removeprefix("c").zfill(3)


def ztag(zsnap):
    return f"{float(zsnap):.3f}".replace(".", "p")


def redshift_label(zsnap):
    labels = {0.5: "LRG1", 0.8: "LRG2", 1.1: "LRG3"}
    zsnap = float(zsnap)
    for key, label in labels.items():
        if np.isclose(zsnap, key):
            return label
    return f"z={zsnap:.1f}"


def zsnaps_label(zsnaps):
    zsnaps = [float(zsnap) for zsnap in zsnaps]
    if len(zsnaps) == 1:
        return redshift_label(zsnaps[0])
    if len(zsnaps) == 3 and all(np.isclose(zsnap, target) for zsnap, target in zip(zsnaps, [0.5, 0.8, 1.1])):
        return "joint LRGs"
    return ",".join(redshift_label(zsnap) for zsnap in zsnaps)


def zset_tag(zsnaps):
    return "-".join(ztag(zsnap) for zsnap in zsnaps)


def bfnl_chain_path_for_zsnaps(fit_dir, zsnaps, cosmo, with_bphi_prior=True, assembly_bias=False, bphi_prior_ab=False):
    prior = "with_bphi_prior_AB" if bphi_prior_ab else ("with_bphi_prior" if with_bphi_prior else "no_bphi_prior")
    abias = "_Bcentm0p040_Bsatm0p170" if assembly_bias else ""
    return Path(fit_dir) / f"png_fit_PNG_LRG_z{zset_tag(zsnaps)}_c{clean_cosmo(cosmo)}_bfnl_ells0-2{abias}_{prior}_chain.npy"


def bfnl_chain_path(fit_dir, zsnap, cosmo, with_bphi_prior=True, assembly_bias=False, bphi_prior_ab=False):
    return bfnl_chain_path_for_zsnaps(
        fit_dir, [zsnap], cosmo, with_bphi_prior=with_bphi_prior, assembly_bias=assembly_bias, bphi_prior_ab=bphi_prior_ab,
    )


def summarize_samples(label, samples, quantity="fNL"):
    q16, q50, q84 = np.percentile(np.asarray(samples, dtype="f8"), [16, 50, 84])
    print(f"{label}: {quantity} = {q50:.3f} +{q84 - q50:.3f} -{q50 - q16:.3f}")
    return q16, q50, q84


def logsumexp_axis(values, axis=None):
    values = np.asarray(values, dtype="f8")
    vmax = np.max(values, axis=axis, keepdims=True)
    finite = np.isfinite(vmax)
    shifted_sum = np.sum(np.exp(np.where(finite, values - vmax, -np.inf)), axis=axis, keepdims=True)
    out = vmax + np.log(shifted_sum)
    out = np.where(finite, out, -np.inf)
    return np.squeeze(out, axis=axis) if axis is not None else float(np.squeeze(out))


def load_chain(path, mode, zsnaps, tracer, burnin=0.5, thin=5):
    chain = Chain.load(path).remove_burnin(burnin)[::thin]
    if mode in ("bphi", "b-p"):
        add_bfnl_loc_to_chain(chain, mode=mode)
    elif mode == "bfnl":
        add_fnl_loc_to_bfnl_chain(chain, tracer, zsnaps)
    return chain


def b1_samples_from_chain(chain, tracer, zsnap, max_samples=None, seed=42):
    name = f"{tracer_z_label(tracer, float(zsnap))}.b1"
    samples = np.ravel(np.asarray(chain[name]))
    if max_samples is not None and samples.size > max_samples:
        rng = np.random.default_rng(seed)
        samples = samples[rng.choice(samples.size, size=int(max_samples), replace=False)]
    return samples


def bfnl_values_from_chain(chain, tracer, zsnaps, max_samples=None, seed=42):
    params = {str(param) for param in chain.params()}
    names = [f"{tracer_z_label(tracer, z)}.bfnl_loc" for z in zsnaps]
    missing = [name for name in names if name not in params]
    if missing:
        raise ValueError(f"Missing bfnl parameters in chain: {missing}")

    bfnl = np.vstack([np.ravel(np.asarray(chain[name])) for name in names]).T
    if max_samples is not None and bfnl.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        bfnl = bfnl[rng.choice(bfnl.shape[0], size=int(max_samples), replace=False)]
    return bfnl


def fnl_grid():
    limit = flat_prior_abs("fnl_loc")
    ngrid = int(os.environ.get("BPHI_BFNL_PRIOR_NGRID", BPHI_BFNL_PRIOR_NGRID))
    if ngrid % 2:
        ngrid += 1
    edges = np.linspace(-limit, limit, ngrid + 1)
    return 0.5 * (edges[1:] + edges[:-1])


def bphi_stats_for_prior(tracer, zsnaps, bphi_prior="gaussian"):
    stats = load_bphi_stats()
    if str(bphi_prior).lower() == "assembly":
        if len(zsnaps) != 1 or not np.isclose(float(zsnaps[0]), 1.1):
            raise ValueError("assembly bphi prior is only available for z=1.1.")
        stats[(tracer, 1.1)] = load_assembly_bias_bphi_prior(tracer=tracer, zsnap=1.1)
    return stats


def bfnl_log_kernel(bfnl, fgrid, tracer, zsnaps, bphi_prior="gaussian"):
    bfnl = np.asarray(bfnl, dtype="f8")
    fgrid = np.asarray(fgrid, dtype="f8")
    bphi_prior = str(bphi_prior).lower()
    if bphi_prior == "auto":
        bphi_prior = "gaussian"
    stats = bphi_stats_for_prior(tracer, zsnaps, bphi_prior=bphi_prior)
    if bphi_prior == "assembly":
        bphi_prior = "gaussian"

    bphi = bfnl[:, :, None] / fgrid[None, None, :]
    logp = -len(zsnaps) * np.log(np.abs(fgrid))[None, :]
    if bphi_prior == "gaussian":
        mu = np.array([stats[(tracer, z)]["mean"] for z in zsnaps])
        sig = np.array([stats[(tracer, z)]["std"] for z in zsnaps])
        logp = logp + np.sum(
            -0.5 * ((bphi - mu[None, :, None]) / sig[None, :, None])**2
            - np.log(sig[None, :, None]),
            axis=1,
        )
    elif bphi_prior == "flat":
        valid = np.all(np.abs(bphi) <= flat_prior_abs("bphi"), axis=1)
        logp = np.where(valid, logp, -np.inf)
    else:
        raise ValueError("bfnl chains require bphi_prior='gaussian', 'assembly', 'flat', or 'auto'.")
    return logp


def infer_bfnl_chain_prior(path):
    name = Path(path).name
    if "_with_bphi_prior" in name:
        return "induced"
    if "_no_bphi_prior" in name:
        return "flat"
    return "flat"


def marginal_fnl_samples_from_bfnl_chain(chain, tracer, zsnaps, bphi_prior="gaussian", seed=42, max_samples=None):
    bfnl = bfnl_values_from_chain(chain, tracer, zsnaps, max_samples=max_samples, seed=seed)
    rng = np.random.default_rng(seed)
    fgrid = fnl_grid()

    fnl_samples = []
    for beta in bfnl:
        logp = bfnl_log_kernel(beta[None, :], fgrid, tracer, zsnaps, bphi_prior=bphi_prior)[0]
        finite = np.isfinite(logp)
        if not np.any(finite):
            continue
        p = np.zeros_like(logp)
        p[finite] = np.exp(logp[finite] - logsumexp(logp[finite]))
        fnl_samples.append(rng.choice(fgrid, p=p / p.sum()))
    return np.asarray(fnl_samples)


def description_fnl_samples_from_bfnl_chain(
    chain,
    tracer,
    zsnaps,
    bphi_prior="gaussian",
    bfnl_chain_prior="flat",
    seed=42,
    max_samples=None,
    chunk_size=512,
):
    bfnl = bfnl_values_from_chain(chain, tracer, zsnaps, max_samples=max_samples, seed=seed)
    fgrid = fnl_grid()
    log_density = np.full(fgrid.size, -np.inf)
    bfnl_chain_prior = str(bfnl_chain_prior).lower()
    if bfnl_chain_prior == "auto":
        bfnl_chain_prior = "flat"
    if bfnl_chain_prior not in ("flat", "induced"):
        raise ValueError("bfnl_chain_prior must be 'flat', 'induced', or 'auto'.")

    for start in range(0, bfnl.shape[0], int(chunk_size)):
        beta = bfnl[start:start + int(chunk_size)]
        log_kernel = bfnl_log_kernel(beta, fgrid, tracer, zsnaps, bphi_prior=bphi_prior)
        if bfnl_chain_prior == "induced":
            log_proposal = logsumexp_axis(log_kernel, axis=1)
            log_terms = log_kernel - log_proposal[:, None]
        else:
            log_terms = log_kernel
        log_density = np.logaddexp(log_density, logsumexp_axis(log_terms, axis=0))

    finite = np.isfinite(log_density)
    if not np.any(finite):
        return np.asarray([])
    prob = np.zeros_like(log_density)
    prob[finite] = np.exp(log_density[finite] - logsumexp(log_density[finite]))
    rng = np.random.default_rng(seed)
    size = bfnl.shape[0] if max_samples is None else min(int(max_samples), bfnl.shape[0])
    return rng.choice(fgrid, size=size, p=prob / prob.sum())


def fnl_samples_from_chain(
    chain,
    tracer,
    zsnaps,
    mode="bfnl",
    bphi_prior="auto",
    seed=42,
    max_samples=None,
    fnl_method="description",
    bfnl_chain_prior="flat",
):
    fnl_key = find_param_key(chain.params(), "fnl_loc")
    if mode in ("b-p", "bphi") or bphi_prior == "direct" or (bphi_prior == "auto" and fnl_key is not None):
        if fnl_key is None:
            raise ValueError("Requested direct fNL samples, but the chain has no fnl_loc parameter.")
        samples = np.ravel(np.asarray(chain[fnl_key]))
        if max_samples is not None and samples.size > max_samples:
            rng = np.random.default_rng(seed)
            samples = samples[rng.choice(samples.size, size=int(max_samples), replace=False)]
        return samples
    if fnl_method == "description":
        return description_fnl_samples_from_bfnl_chain(
            chain,
            tracer,
            zsnaps,
            bphi_prior=bphi_prior,
            bfnl_chain_prior=bfnl_chain_prior,
            seed=seed,
            max_samples=max_samples,
        )
    if fnl_method != "conditional":
        raise ValueError("fnl_method must be 'conditional' or 'description'.")
    return marginal_fnl_samples_from_bfnl_chain(
        chain, tracer, zsnaps, bphi_prior=bphi_prior, seed=seed, max_samples=max_samples,
    )


def plot_fnl_marginal_comparison(entries, output, truth_fnl=None, bins=80, text=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    bin_edges = np.linspace(-flat_prior_abs("fnl_loc"), flat_prior_abs("fnl_loc"), int(bins) + 1)
    truth_labels = set()
    for i, entry in enumerate(entries):
        samples = np.asarray(entry["samples"], dtype="f8")
        if samples.size == 0:
            continue
        color = entry.get("color", f"C{i}")
        ax.hist(
            samples,
            bins=bin_edges,
            density=True,
            histtype=entry.get("histtype", "stepfilled" if i == 0 else "step"),
            linewidth=1.8,
            alpha=entry.get("alpha", 0.25 if i == 0 else 1.0),
            color=color,
            linestyle=entry.get("linestyle", "-"),
            label=entry["label"],
        )
        entry_truth = entry.get("truth_fnl")
        if entry_truth is not None:
            truth_label = entry.get("truth_label", "truth")
            truth_key = (truth_label, float(entry_truth))
            if truth_key not in truth_labels:
                truth_labels.add(truth_key)
                ax.axvline(float(entry_truth), color=color, linestyle=":", linewidth=1.2, label=truth_label)
        summarize_samples(entry["label"], samples, quantity="fNL")
    if truth_fnl is not None:
        ax.axvline(float(truth_fnl), color="black", linestyle="--", linewidth=1.2, label="truth")
    ax.set_xlabel(r"$f_{\mathrm{NL}}^{\mathrm{loc}}$")
    ax.set_ylabel("posterior density")
    if text:
        ax.set_title(text)
    ax.legend(frameon=False)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_hist_pair(ax, entries, bins, xlabel=None, truth=None, truth_label="truth", title=None, show_ylabel=False):
    for entry in entries:
        samples = np.asarray(entry["samples"], dtype="f8")
        ax.hist(
            samples,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.4,
            color=entry.get("color", None),
            linestyle=entry.get("linestyle", "-"),
            label=entry["label"],
        )
    if truth is not None:
        ax.axvline(float(truth), color="black", linestyle=":", linewidth=1.0, label=truth_label)
    if title:
        ax.set_title(title, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel("density")


def use_grid_plot_style(plt):
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
    })


def command_snapshot_grid(args):
    import matplotlib.pyplot as plt

    use_grid_plot_style(plt)
    cosmos = [clean_cosmo(cosmo) for cosmo in args.cosmos]
    zsnaps = [float(zsnap) for zsnap in args.grid_zsnaps]
    fit_dir = Path(args.fit_dir)
    output = args.output or Path(args.plot_dir) / "validation_snapshot_grid.pdf"
    ensure_dir(output.parent)

    fig, axes = plt.subplots(
        len(cosmos), len(zsnaps),
        figsize=(3.5 * len(zsnaps), 2.7 * len(cosmos)),
        sharex=True,
        sharey=False,
    )
    axes = np.atleast_2d(axes)
    bins = np.linspace(-flat_prior_abs("fnl_loc"), flat_prior_abs("fnl_loc"), int(args.bins) + 1)

    for irow, cosmo in enumerate(cosmos):
        for icol, zsnap in enumerate(zsnaps):
            ax = axes[irow, icol]
            entries = []
            for with_prior, label, color, linestyle in [
                (True, "with prior", "C0", "-"),
                (False, "weak prior", "C1", "--"),
            ]:
                path = bfnl_chain_path(fit_dir, zsnap, cosmo, with_bphi_prior=with_prior)
                chain = load_chain(path, "bfnl", [zsnap], args.tracer, burnin=args.burnin, thin=args.thin)
                samples = fnl_samples_from_chain(
                    chain,
                    args.tracer,
                    [zsnap],
                    mode="bfnl",
                    bphi_prior="gaussian" if with_prior else "flat",
                    seed=args.seed,
                    max_samples=args.max_samples,
                    fnl_method=args.fnl_method,
                    bfnl_chain_prior="induced" if with_prior else "flat",
                )
                summarize_samples(f"{cosmo_label(cosmo)} {redshift_label(zsnap)} {label}", samples, quantity="fNL")
                entries.append({"label": label, "samples": samples, "color": color, "linestyle": linestyle})
            plot_hist_pair(
                ax,
                entries,
                bins,
                xlabel=r"$f_{\mathrm{NL}}^{\mathrm{loc}}$" if irow == len(cosmos) - 1 else None,
                truth=true_fnl_loc_for_cosmo(cosmo),
                title=redshift_label(zsnap) if irow == 0 else None,
                show_ylabel=icol == 0,
            )
            if icol == 0:
                ax.text(0.03, 0.88, fnl_truth_label(cosmo), transform=ax.transAxes, fontsize=12, weight="bold")
            else:
                ax.set_yticks([])
            if irow == 0 and icol == len(zsnaps) - 1:
                ax.legend(frameon=False, fontsize=12)
            ax.set_ylim(0.0, 0.0135)

    fig.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def command_assembly_grid(args):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    use_grid_plot_style(plt)
    cosmos = [clean_cosmo(cosmo) for cosmo in args.cosmos]
    zsnaps = [float(zsnap) for zsnap in (args.assembly_zsnaps or [args.assembly_zsnap])]
    zlabel = zsnaps_label(zsnaps)
    fit_dir = Path(args.fit_dir)
    default_name = "validation_assembly_b1_fnl_grid_joint_lrgs.pdf"
    if len(zsnaps) == 1:
        default_name = f"validation_assembly_b1_fnl_grid_z{ztag(zsnaps[0])}.pdf"
    output = args.output or Path(args.plot_dir) / default_name
    ensure_dir(output.parent)

    fig, axes = plt.subplots(len(cosmos), 2, figsize=(7.2, 2.7 * len(cosmos)), sharey=False)
    axes = np.atleast_2d(axes)
    fnl_bins = np.linspace(-flat_prior_abs("fnl_loc"), flat_prior_abs("fnl_loc"), int(args.bins) + 1)

    all_b1 = []
    cache = {}
    assembly_cases = [
        {"assembly_bias": False, "bphi_prior_ab": False, "label": r"$P(k) + b_\phi$ prior", "linestyle": "-", "color": "C0", "bphi_prior": "gaussian"},
        {"assembly_bias": True, "bphi_prior_ab": False, "label": r"contaminated $P(k) + b_\phi$ prior", "linestyle": "--", "color": "C1", "bphi_prior": "gaussian"},
    ]
    if len(zsnaps) == 1 and np.isclose(zsnaps[0], 1.1):
        assembly_cases.append({
            "assembly_bias": True,
            "bphi_prior_ab": True,
            "label": r"contaminated $P(k)$ + contaminated $b_\phi$ prior",
            "linestyle": "-.",
            "color": "C2",
            "bphi_prior": "assembly",
        })
    for cosmo in cosmos:
        for case in assembly_cases:
            path = bfnl_chain_path_for_zsnaps(
                fit_dir, zsnaps, cosmo, with_bphi_prior=True,
                assembly_bias=case["assembly_bias"], bphi_prior_ab=case["bphi_prior_ab"],
            )
            chain = load_chain(path, "bfnl", zsnaps, args.tracer, burnin=args.burnin, thin=args.thin)
            cache[(cosmo, case["assembly_bias"], case["bphi_prior_ab"])] = chain
            for zsnap in zsnaps:
                all_b1.append(b1_samples_from_chain(chain, args.tracer, zsnap, max_samples=args.max_samples, seed=args.seed))
    b1_min = min(float(np.min(samples)) for samples in all_b1)
    b1_max = max(float(np.max(samples)) for samples in all_b1)
    b1_pad = 0.08 * (b1_max - b1_min)
    b1_bins = np.linspace(b1_min - b1_pad, b1_max + b1_pad, int(args.bins) + 1)
    joint_lrg_colors = ["cyan", "purple", "magenta"]

    for irow, cosmo in enumerate(cosmos):
        entries_b1 = []
        entries_fnl = []
        for case in assembly_cases:
            chain = cache[(cosmo, case["assembly_bias"], case["bphi_prior_ab"])]
            for iz, zsnap in enumerate(zsnaps):
                b1_samples = b1_samples_from_chain(chain, args.tracer, zsnap, max_samples=args.max_samples, seed=args.seed)
                b1_label = case["label"] if len(zsnaps) == 1 else f"{case['label']} {redshift_label(zsnap)}"
                summarize_samples(f"{cosmo_label(cosmo)} {redshift_label(zsnap)} {case['label']}", b1_samples, quantity="b1")
                b1_color = case["color"] if len(zsnaps) == 1 else joint_lrg_colors[iz % len(joint_lrg_colors)]
                entries_b1.append({
                    "label": b1_label,
                    "samples": b1_samples,
                    "color": b1_color,
                    "linestyle": case["linestyle"],
                })
            fnl_samples = fnl_samples_from_chain(
                chain,
                args.tracer,
                zsnaps,
                mode="bfnl",
                bphi_prior=case["bphi_prior"],
                seed=args.seed,
                max_samples=args.max_samples,
                fnl_method=args.fnl_method,
                bfnl_chain_prior="induced",
            )
            summarize_samples(f"{cosmo_label(cosmo)} {zlabel} {case['label']}", fnl_samples, quantity="fNL")
            entries_fnl.append({
                "label": case["label"],
                "samples": fnl_samples,
                "color": case["color"],
                "linestyle": case["linestyle"],
            })

        plot_hist_pair(
            axes[irow, 0],
            entries_b1,
            b1_bins,
            xlabel=r"$b_1$" if irow == len(cosmos) - 1 else None,
            show_ylabel=True,
        )
        plot_hist_pair(
            axes[irow, 1],
            entries_fnl,
            fnl_bins,
            xlabel=r"$f_{\mathrm{NL}}^{\mathrm{loc}}$" if irow == len(cosmos) - 1 else None,
            truth=true_fnl_loc_for_cosmo(cosmo),
        )
        axes[irow, 0].text(0.03, 0.88, fnl_truth_label(cosmo), transform=axes[irow, 0].transAxes, fontsize=11, weight="bold")
        if irow == 0 and len(zsnaps) == 1:
            axes[irow, 1].legend(frameon=False, fontsize=9)

    if len(zsnaps) > 1:
        standard_handles = [
            Line2D([0], [0], color=joint_lrg_colors[iz], linestyle="-", linewidth=1.4)
            for iz, _ in enumerate(zsnaps)
        ]
        assembly_handles = [
            Line2D([0], [0], color=joint_lrg_colors[iz], linestyle="--", linewidth=1.4)
            for iz, _ in enumerate(zsnaps)
        ]
        lrg_labels = [redshift_label(zsnap) for zsnap in zsnaps]
        fig.legend(
            standard_handles,
            [f"standard {label}" for label in lrg_labels],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=3,
            frameon=False,
            fontsize=8,
        )
        fig.legend(
            assembly_handles,
            [f"assembly bias {label}" for label in lrg_labels],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.97),
            ncol=3,
            frameon=False,
            fontsize=8,
        )

    fig.tight_layout(pad=0.2, h_pad=0.0, w_pad=0.35, rect=[0.0, 0.0, 1.0, 0.94] if len(zsnaps) > 1 else None)
    fig.subplots_adjust(hspace=0.0)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def command_plot(args):
    if args.snapshot_grid:
        command_snapshot_grid(args)
        return
    if args.assembly_grid:
        command_assembly_grid(args)
        return

    if args.reference_chain is None or not args.comparison_chain:
        raise ValueError("--reference-chain and --comparison-chain are required unless using --snapshot-grid or --assembly-grid.")
    if args.output is None:
        raise ValueError("--output is required unless using a grid mode with the default output.")

    labels = [args.reference_label]
    comparison_labels = args.comparison_label
    if comparison_labels is None:
        comparison_labels = [f"comparison {i + 1}" for i in range(len(args.comparison_chain))]
    if len(comparison_labels) != len(args.comparison_chain):
        raise ValueError("--comparison-label must have the same length as --comparison-chain.")
    labels += list(comparison_labels)

    paths = [args.reference_chain] + list(args.comparison_chain)
    modes = [args.reference_mode or args.mode] + _broadcast_plot_arg(
        args.comparison_mode, len(args.comparison_chain), args.mode, "--comparison-mode"
    )
    bphi_priors = [args.reference_bphi_prior] + _broadcast_plot_arg(
        args.comparison_bphi_prior, len(args.comparison_chain), args.reference_bphi_prior, "--comparison-bphi-prior"
    )
    fnl_methods = [args.reference_fnl_method or args.fnl_method] + _broadcast_plot_arg(
        args.comparison_fnl_method, len(args.comparison_chain), args.fnl_method, "--comparison-fnl-method"
    )
    bfnl_chain_priors = [args.reference_bfnl_chain_prior] + _broadcast_plot_arg(
        args.comparison_bfnl_chain_prior, len(args.comparison_chain), args.reference_bfnl_chain_prior, "--comparison-bfnl-chain-prior"
    )
    cosmos = [args.reference_cosmo or args.cosmo] + _broadcast_plot_arg(
        args.comparison_cosmo, len(args.comparison_chain), args.reference_cosmo or args.cosmo, "--comparison-cosmo"
    )
    zsnaps_by_chain = [parse_zsnaps_text(args.reference_zsnaps, default=args.zsnaps)]
    zsnaps_by_chain += [
        parse_zsnaps_text(item, default=args.zsnaps)
        for item in _broadcast_plot_arg(args.comparison_zsnaps, len(args.comparison_chain), None, "--comparison-zsnaps")
    ]

    unique_cosmos = [cosmo for cosmo in dict.fromkeys(cosmo for cosmo in cosmos if cosmo is not None)]
    color_by_cosmo = {cosmo: f"C{i}" for i, cosmo in enumerate(unique_cosmos)}
    color_by_chain = len(unique_cosmos) <= 1

    entries = []
    for path, label, mode, zsnaps, bphi_prior, fnl_method, bfnl_chain_prior, cosmo in zip(
        paths, labels, modes, zsnaps_by_chain, bphi_priors, fnl_methods, bfnl_chain_priors, cosmos,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing chain file: {path}")
        if bfnl_chain_prior == "auto":
            bfnl_chain_prior = infer_bfnl_chain_prior(path)
        chain = load_chain(path, mode, zsnaps, args.tracer, burnin=args.burnin, thin=args.thin)
        samples = fnl_samples_from_chain(
            chain,
            args.tracer,
            zsnaps,
            mode=mode,
            bphi_prior=bphi_prior,
            seed=args.seed,
            max_samples=args.max_samples,
            fnl_method=fnl_method,
            bfnl_chain_prior=bfnl_chain_prior,
        )
        entry = {"label": label, "samples": samples}
        if cosmo is not None:
            entry["truth_fnl"] = true_fnl_loc_for_cosmo(cosmo)
            entry["truth_label"] = f"{cosmo_label(cosmo)} truth"
            entry["color"] = f"C{len(entries)}" if color_by_chain else color_by_cosmo.get(cosmo, f"C{len(entries)}")
            entry["histtype"] = "step"
            entry["alpha"] = 1.0
            entry["linestyle"] = ":" if fnl_method == "description" else ("-" if bphi_prior in ("gaussian", "direct") else "--")
        entries.append(entry)

    truth_fnl = args.truth_fnl
    if truth_fnl is None and args.cosmo is not None and args.reference_cosmo is None and args.comparison_cosmo is None:
        truth_fnl = true_fnl_loc_for_cosmo(args.cosmo)

    ensure_dir(args.output.parent)
    plot_fnl_marginal_comparison(entries, args.output, truth_fnl=truth_fnl, bins=args.bins, text=args.text)
    print(f"Saved {args.output}")


def build_parser():
    parser = argparse.ArgumentParser(description="Plot 1D marginalized fNL posteriors from PNG chains.")
    parser.add_argument("--snapshot-grid", action="store_true", help="Make a 3x3 grid comparing snapshot consistency across cosmologies.")
    parser.add_argument("--assembly-grid", action="store_true", help="Make a grid comparing b1 and fNL with/without assembly bias.")
    parser.add_argument("--fit-dir", type=Path, default=Path("bphi_prior/fit"))
    parser.add_argument("--plot-dir", type=Path, default=Path("bphi_prior/plots"))
    parser.add_argument("--cosmos", nargs="+", default=["000", "300", "301"])
    parser.add_argument("--grid-zsnaps", nargs="+", type=float, default=[0.5, 0.8, 1.1])
    parser.add_argument("--assembly-zsnap", type=float, default=1.1)
    parser.add_argument("--assembly-zsnaps", nargs="+", type=float, default=None)
    parser.add_argument("--reference-chain", "--standard-chain", dest="reference_chain", type=Path, default=None)
    parser.add_argument("--reference-label", "--standard-label", dest="reference_label", default="reference")
    parser.add_argument("--comparison-chain", nargs="+", type=Path, default=None)
    parser.add_argument("--comparison-label", nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--burnin", type=float, default=0.5)
    parser.add_argument("--thin", type=int, default=5)
    parser.add_argument("--mode", default="bfnl", choices=PNG_MODES)
    parser.add_argument("--reference-mode", choices=PNG_MODES, default=None)
    parser.add_argument("--comparison-mode", nargs="+", choices=PNG_MODES, default=None)
    parser.add_argument("--reference-zsnaps", default=None)
    parser.add_argument("--comparison-zsnaps", nargs="+", default=None)
    parser.add_argument("--reference-bphi-prior", choices=["auto", "gaussian", "assembly", "flat", "direct"], default="auto")
    parser.add_argument("--comparison-bphi-prior", nargs="+", choices=["auto", "gaussian", "assembly", "flat", "direct"], default=None)
    parser.add_argument("--fnl-method", choices=["conditional", "description"], default="description",
                        help="For bfnl chains, use current conditional resampling or description.bib-style importance marginalization.")
    parser.add_argument("--reference-fnl-method", choices=["conditional", "description"], default=None)
    parser.add_argument("--comparison-fnl-method", nargs="+", choices=["conditional", "description"], default=None)
    parser.add_argument("--reference-bfnl-chain-prior", choices=["auto", "flat", "induced"], default="auto",
                        help="Proposal prior used by the reference bfnl chain; 'auto' infers it from the filename.")
    parser.add_argument("--comparison-bfnl-chain-prior", nargs="+", choices=["auto", "flat", "induced"], default=None)
    parser.add_argument("--cosmo", default=None)
    parser.add_argument("--reference-cosmo", default=None)
    parser.add_argument("--comparison-cosmo", nargs="+", default=None)
    parser.add_argument("--truth-fnl", type=float, default=None)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tracer", default="LRG", choices=["LRG"])
    parser.add_argument("--zsnaps", nargs="+", type=float, default=DEFAULT_ZS)
    parser.set_defaults(func=command_plot)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
