#!/usr/bin/env python3
import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
PK_DIR = ROOT / "bphi_prior" / "pk"
COV_DIR = ROOT / "bphi_prior" / "cov"
PLOT_DIR = ROOT / "bphi_prior" / "plots"
BAOCOV_FILE = ROOT.parent / "PFS_mock-tests" / "baocov.py"


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def ztag(zsnap):
    return f"{zsnap:.3f}"


def pk_path(tracer, zsnap, cosmo="000"):
    return PK_DIR / f"pk_{tracer}_z{ztag(zsnap)}_c{cosmo}.npy"


def cov_path(tracer, zsnap, cosmo="000"):
    return COV_DIR / f"cov_{tracer}_z{ztag(zsnap)}_c{cosmo}.npy"


def load_pk(tracer, zsnap, cosmo="000"):
    from pypower import MeshFFTPower, PowerSpectrumMultipoles

    state = np.load(pk_path(tracer, zsnap, cosmo=cosmo), allow_pickle=True)[()]
    if "_projs" in state:
        poles = PowerSpectrumMultipoles.from_state(state)
    else:
        result = MeshFFTPower.from_state(state)
        poles = result.poles if hasattr(result, "poles") else PowerSpectrumMultipoles.from_state(state)

    attrs = getattr(poles, "attrs", {})
    poles = poles.select((0.0, 0.1))
    data_size = float(attrs.get("data_size1", attrs.get("data_size", 0.0)))
    return {
        "k": np.asarray(poles.modeavg(), dtype="f8"),
        "pk0": np.asarray(poles(ell=0, complex=False), dtype="f8"),
        "nmodes": np.asarray(poles.nmodes, dtype="f8"),
        "shotnoise": float(poles.shotnoise),
        "nbar": data_size / 2000.0**3,
        "boxsize": 2000.0,
    }


def load_pk_var():
    spec = importlib.util.spec_from_file_location("baocov_module", BAOCOV_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pk_var


def plot_one(tracer, zsnap, cosmo="000"):
    ensure_dir(PLOT_DIR)
    pk_data = load_pk(tracer, zsnap, cosmo=cosmo)
    covariance = np.load(cov_path(tracer, zsnap, cosmo=cosmo))
    pk_var = load_pk_var()
    k = np.asarray(pk_data["k"], dtype="f8")
    pk0 = np.asarray(pk_data["pk0"], dtype="f8")
    nmodes = np.asarray(pk_data["nmodes"], dtype="f8")
    nbar = float(pk_data["nbar"])
    boxsize = float(pk_data["boxsize"])
    dk=0.002

    analytic_var = pk_var(nbar=nbar, kmid=k, P0=pk0, boxsize=boxsize, dk=dk)
    measured_var = np.diag(covariance)
    exact_bin_var = 2.0 * (pk0 + 1.0 / nbar) ** 2 / nmodes

    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

    axes[0].plot(k, k * pk0, color="black", label=r"$k P_0(k)$")
    axes[0].set_ylabel(r"$k P_0(k)$")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(k, measured_var, color="C0", label="thecov diag")
    axes[1].plot(k, analytic_var, color="C3", ls="--", label="baocov.pk_var")
    axes[1].plot(k, exact_bin_var, color="C2", ls="-.", label="2(P0 + 1/nbar)^2 / nmodes")
    axes[1].set_xlabel(r"$k$ [$h/\mathrm{Mpc}$]")
    axes[1].set_ylabel(r"$\mathrm{Var}[P_0(k)]$")
    axes[1].set_yscale("log")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(k, measured_var / analytic_var, color="C3", ls="--", label="thecov / pk_var")
    axes[2].plot(k, measured_var / exact_bin_var, color="C2", ls="-.", label="thecov / exact-bin")
    axes[2].axhline(1.0, color="black", lw=1)
    axes[2].set_xlabel(r"$k$ [$h/\mathrm{Mpc}$]")
    axes[2].set_ylabel("ratio")
    axes[2].grid(True)
    axes[2].legend()

    title = f"{tracer} z={zsnap:.3f} cosmo={cosmo}"
    fig.suptitle(title)
    fig.tight_layout()
    output = PLOT_DIR / f"pk_var_compare_{tracer}_z{ztag(zsnap)}_c{cosmo}.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description="Plot P(k) and compare variance with baocov.pk_var.")
    parser.add_argument("--tracer", default="LRG")
    parser.add_argument("--zsnaps", nargs="*", type=float, default=[0.5, 0.8, 1.1])
    parser.add_argument("--cosmo", default="000")
    args = parser.parse_args()

    for zsnap in args.zsnaps:
        output = plot_one(args.tracer, zsnap, cosmo=args.cosmo)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
