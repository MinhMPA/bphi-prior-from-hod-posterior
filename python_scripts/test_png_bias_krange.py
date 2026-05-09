#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cosmoprimo import Fourier
from cosmoprimo.fiducial import AbacusSummit
from pypower import PowerSpectrumMultipoles


def load_pk0(path):
    result = PowerSpectrumMultipoles.load(str(path))
    k, pk0 = result(ell=0, return_k=True, complex=False)
    return np.asarray(k), np.asarray(pk0)


def build_file(root, tracer, zsnap, cosmo):
    ztag = f"{zsnap:.3f}"
    base = f"{tracer}_z{ztag}_c{cosmo}_default_pk_real.npy"
    return Path(root) / base


def fit_const(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return np.nan, np.nan
    yv = y[mask]
    mu = np.mean(yv)
    rms = np.sqrt(np.mean((yv - mu) ** 2))
    return float(mu), float(rms)


def main():
    parser = argparse.ArgumentParser(description="PNG bias-range diagnostic using real-space P(k).")
    parser.add_argument("--tracer", default="LRG")
    parser.add_argument("--zsnap", type=float, required=True)
    parser.add_argument("--root", default="/pscratch/sd/j/jiaxi/b_phi/mocks_lhc_PNG")
    parser.add_argument("--outdir", default="/pscratch/sd/j/jiaxi/b_phi/bphi_prior/plots")
    parser.add_argument("--kmin-grid", type=float, default=0.005)
    parser.add_argument("--kmax-grid", type=float, default=0.080)
    parser.add_argument("--dk-grid", type=float, default=0.002)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = {cosmo: build_file(args.root, args.tracer, args.zsnap, cosmo) for cosmo in ("000", "300", "301")}
    missing = [f"{c}:{p}" for c, p in files.items() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing pk_real files: " + ", ".join(missing))

    data = {}
    for cosmo, path in files.items():
        k, pk0 = load_pk0(path)
        data[cosmo] = (k, pk0)

    k = data["000"][0]
    if not (np.allclose(k, data["300"][0]) and np.allclose(k, data["301"][0])):
        raise ValueError("k-grid mismatch across c000/c300/c301.")

    pk_interp = Fourier(AbacusSummit(), engine="class").pk_interpolator()
    pm = np.asarray(pk_interp(k, z=args.zsnap))

    ratio = {cosmo: data[cosmo][1] / pm for cosmo in ("000", "300", "301")}
    bias = {cosmo: np.sqrt(np.clip(ratio[cosmo], 0.0, np.inf)) for cosmo in ("000", "300", "301")}

    kmins = np.arange(args.kmin_grid, min(args.kmax_grid, 0.06), args.dk_grid)
    kmaxs = np.arange(0.02, args.kmax_grid + 1e-12, args.dk_grid)
    stats = []
    for kmin in kmins:
        for kmax in kmaxs:
            if kmax <= kmin + 0.004:
                continue
            mask = (k >= kmin) & (k <= kmax)
            if np.count_nonzero(mask) < 4:
                continue
            b000, s000 = fit_const(k[mask], bias["000"][mask])
            b300, s300 = fit_const(k[mask], bias["300"][mask])
            b301, s301 = fit_const(k[mask], bias["301"][mask])
            stats.append((kmin, kmax, b000, s000, b300, s300, b301, s301))

    stats = np.array(stats, dtype=float)
    if stats.size == 0:
        raise RuntimeError("No valid k-ranges found.")

    table_path = outdir / f"png_bias_kscan_{args.tracer}_z{args.zsnap:.3f}.txt"
    header = "kmin kmax b1_c000 scatter_c000 b1_c300 scatter_c300 b1_c301 scatter_c301"
    np.savetxt(table_path, stats, header=header)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for cosmo, color in (("000", "k"), ("300", "C0"), ("301", "C3")):
        axes[0].plot(k, ratio[cosmo], color=color, lw=1.3, label=f"c{cosmo}")
    axes[0].set_xlim(0.001, 0.12)
    axes[0].set_ylim(bottom=0)
    axes[0].set_xlabel("k [h/Mpc]")
    axes[0].set_ylabel("P_g(k) / P_m(k)")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.3)

    for cosmo, color in (("000", "k"), ("300", "C0"), ("301", "C3")):
        axes[1].plot(k, bias[cosmo], color=color, lw=1.3, label=f"sqrt(P_g/P_m) c{cosmo}")
    axes[1].set_xlim(0.001, 0.12)
    axes[1].set_xlabel("k [h/Mpc]")
    axes[1].set_ylabel("effective bias")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"{args.tracer} z={args.zsnap:.3f} PNG real-space bias diagnostic")
    fig.tight_layout()
    png_path = outdir / f"png_bias_ratio_{args.tracer}_z{args.zsnap:.3f}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.plot(stats[:, 1], stats[:, 3], "k-", lw=1.2, label="scatter c000")
    ax2.plot(stats[:, 1], stats[:, 5], "C0-", lw=1.2, label="scatter c300")
    ax2.plot(stats[:, 1], stats[:, 7], "C3-", lw=1.2, label="scatter c301")
    ax2.set_xlabel("kmax [h/Mpc] (all scanned kmin combined)")
    ax2.set_ylabel("const-fit RMS in selected window")
    ax2.grid(alpha=0.3)
    ax2.legend(frameon=False)
    fig2.tight_layout()
    png_path2 = outdir / f"png_bias_scatter_{args.tracer}_z{args.zsnap:.3f}.png"
    fig2.savefig(png_path2, dpi=150)
    plt.close(fig2)

    print(f"Saved: {table_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {png_path2}")


if __name__ == "__main__":
    main()
