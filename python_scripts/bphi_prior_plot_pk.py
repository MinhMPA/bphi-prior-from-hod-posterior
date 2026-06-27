#!/usr/bin/env python3
import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bphi_prior_pipeline import *

ROOT = Path(__file__).resolve().parent
PK_DIR = ROOT / "bphi_prior" / "pk"
COV_DIR = ROOT / "bphi_prior" / "cov"
PLOT_DIR = ROOT / "bphi_prior" / "plots"
BAOCOV_FILE = ROOT.parent / "PFS_mock-tests" / "baocov.py"
PNG_MOCK_DIR = ROOT / "mocks_lhc_PNG"
SIMTYPES = ("SU", "PNG")
BOXSIZE = 2000.0

def load_pk_var():
    spec = importlib.util.spec_from_file_location("baocov_module", BAOCOV_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pk_var


def plot_one(tracer, zsnap, cosmo="000", simtype="SU", ells=(0,), skip_first_bin=True, volume=None, cov_veff=False, abias_suffix=""):
    ensure_dir(PLOT_DIR)
    pk_data = load_pk(tracer, zsnap, cosmo=cosmo, simtype=simtype, ells=ells, abias_suffix=abias_suffix)
    covariance = np.load(default_cov_path(tracer, zsnap, cosmo=cosmo, simtype=simtype, ells=ells, cov_veff=cov_veff))
    pk_var = load_pk_var()
    k = np.asarray(pk_data["k"], dtype="f8")
    pk0 = np.asarray(pk_data["pk0"], dtype="f8")
    nmodes = np.asarray(pk_data["nmodes"], dtype="f8")
    nbar = float(pk_data["nbar"])
    boxsize = float(pk_data["boxsize"])
    volume = float(pk_data.get("covariance_volume", boxsize**3)) if volume is None else float(volume)
    var_boxsize = volume ** (1.0 / 3.0)
    var_nmodes = nmodes * volume / boxsize**3
    dk=0.002

    analytic_var = pk_var(nbar=nbar, kmid=k, P0=pk0, boxsize=var_boxsize, dk=dk)
    measured_var = np.diag(covariance)
    exact_bin_var = 2.0 * (pk0 + 1.0 / nbar) ** 2 / var_nmodes
    plot_slice = slice(1, None) if skip_first_bin else slice(None)
    k = k[plot_slice]
    pk0 = pk0[plot_slice]
    analytic_var = analytic_var[plot_slice]
    measured_var = measured_var[plot_slice]
    exact_bin_var = exact_bin_var[plot_slice]

    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

    axes[0].plot(k, k * pk0, color="black", label=r"$k P_0(k)$")
    print(k)
    axes[0].set_ylabel(r"$k P_0(k)$")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(k, measured_var[:len(k)], color="C0", label="thecov diag")
    axes[1].plot(k, analytic_var, color="C3", ls="--", label="baocov.pk_var")
    axes[1].plot(k, exact_bin_var, color="C2", ls="-.", label="2(P0 + 1/nbar)^2 / nmodes")
    axes[1].set_xlabel(r"$k$ [$h/\mathrm{Mpc}$]")
    axes[1].set_ylabel(r"$\mathrm{Var}[P_0(k)]$")
    axes[1].set_yscale("log")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(k, measured_var[:len(k)] / analytic_var, color="C3", ls="--", label="thecov / pk_var")
    axes[2].plot(k, measured_var[:len(k)] / exact_bin_var, color="C2", ls="-.", label="thecov / exact-bin")
    axes[2].axhline(1.0, color="black", lw=1)
    axes[2].set_xlabel(r"$k$ [$h/\mathrm{Mpc}$]")
    axes[2].set_ylabel("ratio")
    axes[2].grid(True)
    axes[2].legend()

    title = f"{simtype} {tracer} z={zsnap:.3f} cosmo={cosmo}"
    fig.suptitle(title)
    fig.tight_layout()
    veff_tag = "_veff" if cov_veff else ""
    output = PLOT_DIR / f"pk_var_compare_{simtype}_{tracer}_z{ztag(zsnap)}_c{cosmo}{abias_suffix}_ells{'_'.join(map(str, ells))}{veff_tag}.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description="Plot P(k) and compare variance with baocov.pk_var.")
    parser.add_argument("--tracer", default="LRG")
    parser.add_argument("--zsnaps", nargs="*", type=float, default=[0.5, 0.8, 1.1])
    parser.add_argument("--simtype", default="SU", choices=SIMTYPES)
    parser.add_argument("--ells", nargs="+", type=int, default=[0])
    parser.add_argument("--cosmo", default="000")
    parser.add_argument("--include-first-bin", action="store_true")
    parser.add_argument("--volume", nargs="+", type=float, default=None, help="Covariance volume(s) in (Mpc/h)^3; one value or one per zsnap.")
    parser.add_argument("--cov-veff", action="store_true", help="Use DR1 n(z)-based effective volumes for variance comparison.")
    parser.add_argument("--assembly-bias", action="store_true", help="Use assembly-bias P(k) data-vector files.")
    args = parser.parse_args()
    abias_suffix = abias_suffix_from_args(args)

    for iz, zsnap in enumerate(args.zsnaps):
        volume = volume_for_index(args.volume, args.zsnaps, iz, cov_veff=args.cov_veff)
        if volume is not None:
            print(f"Using variance comparison volume for z={zsnap:.3f}: {volume:.6e} (Mpc/h)^3")
        output = plot_one(
            args.tracer,
            zsnap,
            cosmo=args.cosmo,
            simtype=args.simtype,
            ells=args.ells,
            skip_first_bin=not args.include_first_bin,
            volume=volume,
            cov_veff=args.cov_veff,
            abias_suffix=abias_suffix,
        )
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
