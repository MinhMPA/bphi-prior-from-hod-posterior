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
PNG_MOCK_DIR = ROOT / "mocks_lhc_PNG"
SIMTYPES = ("SU", "PNG")
BOXSIZE = 2000.0


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def ztag(zsnap):
    return f"{zsnap:.3f}"


def su_text_pk_path(tracer, zsnap, cosmo="000"):
    return PK_DIR / f"pk_{tracer}_z{ztag(zsnap)}_c{cosmo}.txt_pypower"


def png_mock_pk_path(tracer, zsnap, cosmo="000"):
    return PNG_MOCK_DIR / f"{tracer}_z{ztag(zsnap)}_c{cosmo}_default_pk_RSD.npy"


def pk_path(tracer, zsnap, cosmo="000", simtype="SU"):
    if simtype == "SU":
        return su_text_pk_path(tracer, zsnap, cosmo=cosmo)
    if simtype == "PNG":
        return png_mock_pk_path(tracer, zsnap, cosmo=cosmo)
    raise ValueError(f"Unknown simtype '{simtype}'. Choices are {SIMTYPES}.")


def cov_path(tracer, zsnap, cosmo="000", simtype="SU"):
    if simtype == "SU":
        return COV_DIR / f"cov_{tracer}_z{ztag(zsnap)}_c{cosmo}.npy"
    if simtype == "PNG":
        return COV_DIR / f"cov_PNG_{tracer}_z{ztag(zsnap)}_c{cosmo}.npy"
    raise ValueError(f"Unknown simtype '{simtype}'. Choices are {SIMTYPES}.")


def load_su_text_pk(path):
    header = {}
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            content = line[1:].strip()
            if "=" in content:
                key, value = [item.strip() for item in content.split("=", 1)]
                header[key] = value

    nmodes, _, k, pk0 = np.loadtxt(path, unpack=True)
    data_size = float(header.get("data_size1", header.get("data_size", 0.0)))
    shotnoise = float(header.get("shotnoise", 0.0))
    return {
        "k": np.asarray(k, dtype="f8"),
        "pk0": np.asarray(pk0, dtype="f8"),
        "nmodes": np.asarray(nmodes, dtype="f8"),
        "shotnoise": shotnoise,
        "nbar": data_size / BOXSIZE**3,
        "boxsize": BOXSIZE,
    }


def load_png_rebinned_pk(path, kmin=0.0, kmax=0.1, dk=0.002):
    state = np.load(path, allow_pickle=True)[()]
    poles = state["poles"]
    attrs = poles["attrs"]
    edges = np.asarray(poles["edges"][0], dtype="f8")
    nmodes_fine = np.asarray(poles["nmodes"], dtype="f8")
    power_nonorm = np.asarray(poles["power_nonorm"], dtype=np.complex128)
    ells = tuple(poles["ells"])
    wnorm = float(poles["wnorm"])
    shotnoise = float(poles["shotnoise_nonorm"]) / wnorm

    try:
        ell0_index = ells.index(0)
    except ValueError as exc:
        raise ValueError(f"Could not find ell=0 in PNG P(k) file {path}. Available ells are {ells}.") from exc

    pk0_fine = power_nonorm[ell0_index].real / wnorm
    fine_centers = 0.5 * (edges[:-1] + edges[1:])
    target_edges = np.arange(kmin, kmax + dk / 2.0, dk)
    target_k = np.full(target_edges.size - 1, np.nan, dtype="f8")
    target_pk0 = np.full(target_edges.size - 1, np.nan, dtype="f8")
    target_nmodes = np.zeros(target_edges.size - 1, dtype="f8")

    for ibin in range(target_edges.size - 1):
        lo, hi = target_edges[ibin], target_edges[ibin + 1]
        mask = (fine_centers >= lo) & (fine_centers < hi) & np.isfinite(pk0_fine) & (nmodes_fine > 0)
        if not np.any(mask):
            continue
        weights = nmodes_fine[mask]
        target_k[ibin] = np.average(fine_centers[mask], weights=weights)
        target_pk0[ibin] = np.average(pk0_fine[mask], weights=weights)
        target_nmodes[ibin] = np.sum(weights)

    if np.any(~np.isfinite(target_pk0)):
        missing = np.where(~np.isfinite(target_pk0))[0]
        raise ValueError(f"Missing rebinned PNG P(k) bins for {path}: target bin indices {missing.tolist()}")

    data_size = float(attrs.get("data_size1", attrs.get("data_size", 0.0)))
    return {
        "k": target_k,
        "pk0": target_pk0,
        "nmodes": target_nmodes,
        "shotnoise": shotnoise,
        "nbar": data_size / BOXSIZE**3,
        "boxsize": BOXSIZE,
    }


def load_pk(tracer, zsnap, cosmo="000", simtype="SU"):
    path = pk_path(tracer, zsnap, cosmo=cosmo, simtype=simtype)
    if simtype == "SU":
        return load_su_text_pk(path)
    return load_png_rebinned_pk(path)


def load_pk_var():
    spec = importlib.util.spec_from_file_location("baocov_module", BAOCOV_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pk_var


def plot_one(tracer, zsnap, cosmo="000", simtype="SU", skip_first_bin=True):
    ensure_dir(PLOT_DIR)
    pk_data = load_pk(tracer, zsnap, cosmo=cosmo, simtype=simtype)
    covariance = np.load(cov_path(tracer, zsnap, cosmo=cosmo, simtype=simtype))
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
    plot_slice = slice(1, None) if skip_first_bin else slice(None)
    k = k[plot_slice]
    pk0 = pk0[plot_slice]
    analytic_var = analytic_var[plot_slice]
    measured_var = measured_var[plot_slice]
    exact_bin_var = exact_bin_var[plot_slice]

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

    title = f"{simtype} {tracer} z={zsnap:.3f} cosmo={cosmo}"
    fig.suptitle(title)
    fig.tight_layout()
    output = PLOT_DIR / f"pk_var_compare_{simtype}_{tracer}_z{ztag(zsnap)}_c{cosmo}.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description="Plot P(k) and compare variance with baocov.pk_var.")
    parser.add_argument("--tracer", default="LRG")
    parser.add_argument("--zsnaps", nargs="*", type=float, default=[0.5, 0.8, 1.1])
    parser.add_argument("--simtype", default="SU", choices=SIMTYPES)
    parser.add_argument("--cosmo", default="000")
    parser.add_argument("--include-first-bin", action="store_true")
    args = parser.parse_args()

    for zsnap in args.zsnaps:
        output = plot_one(
            args.tracer,
            zsnap,
            cosmo=args.cosmo,
            simtype=args.simtype,
            skip_first_bin=not args.include_first_bin,
        )
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
