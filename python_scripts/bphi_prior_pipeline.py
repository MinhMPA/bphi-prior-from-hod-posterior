#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import numpy as np

from HOD_utils import (
    annotate_joint_triangle_map,
    overlay_markers_on_triangle,
    overlay_truth_on_triangle,
    overlay_truth_values_on_triangle,
    plot_contours,
    triangle_params_with_bfnl_loc,
)


ROOT = Path(os.environ.get('BPHI_DATA_ROOT', Path(__file__).resolve().parent)).expanduser()
PAPER_DATA = Path(os.environ.get('BPHI_PAPER_DATA', Path(__file__).resolve().parent)).expanduser()
BAOCOV_FILE = ROOT.parent / "PFS_mock-tests" / "baocov.py"
MOCK_DIRS = {
    "SU": ROOT / "mocks_lhc_SU",
    "PNG": ROOT / "mocks_lhc_PNG",
}
BPHI_STATS_FILE = ROOT / "backup" / "bphi_AbacusHOD_LRG.txt"
BPHI_STATS_UPDATE_FILES = (ROOT / "backup" / "bphi_AbacusHOD_z1.1_updated.txt",)
BPHI_STATS_ASSEMBLY_FILE = PAPER_DATA / "bphi_results_abacushod_Bcentm0p040_Bsatm0p170.txt"
BPHI_BESTFIT_B1_FILE = ROOT / "bphi_true_value_PNGmock_fit" / "b1_with-sn0.txt"
BPHI_SU_DENOM = np.log(0.807952 / 0.75)
BPHI_NGAL_MIN = 100000
BOXSIZE = 2000.0
DEFAULT_ZS = [0.5, 0.8, 1.1]
ASSEMBLY_BIAS_SUFFIX = "_Bcentm0p040_Bsatm0p170"
COV_VEFF_P0 = 10000.0
COV_VEFF_CONFIG = {
    0.5: (0.4, 0.6, 17000.0),
    0.8: (0.6, 0.8, 17000.0),
    1.1: (0.8, 1.1, 14000.0),
}
PNG_MODES = ("bphi", "b-p", "bfnl")
PRIORS_FLAT = {"bphi": 10, "fnl_loc": 300}
BPHI_BFNL_PRIOR_NGRID = 4096
SIMTYPES = ("SU", "PNG")
TRUE_FNL_LOC_BY_COSMO = {
    "000": 0.0,
    "300": 30.0,
    "301": -30.0,
    "302": 100.0,
    "303": -100.0,
}


def env_path(name, default):
    value = os.environ.get(name)
    if not value:
        return default
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


FIT_DIR = env_path("BPHI_FIT_DIR", ROOT / "bphi_prior" / "fit")


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def ztag(zsnap):
    return f"{zsnap:.3f}"


def fundamental_k():
    return 2.0 * np.pi / BOXSIZE


def zset_tag(zsnaps):
    return "-".join(ztag(zsnap).replace(".", "p") for zsnap in zsnaps)


def ktag(k):
    return f"{float(k):.3f}".replace("-", "m").replace(".", "p")


def k_range_tag(kmin=None, kmax=None):
    if kmax is None or np.isclose(float(kmax), 0.1):
        return ""
    kmin = 0.0 if kmin is None else float(kmin)
    return f"_k{ktag(kmin)}-{ktag(kmax)}"


def normalize_ells(ells):
    if ells is None:
        return (0,)
    if isinstance(ells, str):
        ells = ells.replace(",", " ").split()
    elif np.isscalar(ells):
        ells = (ells,)
    return tuple(int(ell) for ell in ells)


def ell_tag(ells):
    ells = normalize_ells(ells)
    return "" if ells == (0,) else "_ells" + "-".join(str(ell) for ell in ells)


def abias_suffix_from_args(args):
    return ASSEMBLY_BIAS_SUFFIX if getattr(args, "assembly_bias", False) else ""


def mock_dir(simtype):
    if simtype not in MOCK_DIRS:
        raise ValueError(f"Unknown simtype '{simtype}'. Choices are {SIMTYPES}.")
    return MOCK_DIRS[simtype]


def mock_pk_path(tracer, zsnap, cosmo="000", simtype="SU", abias_suffix=""):
    return mock_dir(simtype) / f"{tracer}_z{ztag(zsnap)}_c{cosmo}_default{abias_suffix}_pk_RSD.npy"


def load_bestfit_b1(zsnap, path=BPHI_BESTFIT_B1_FILE):
    data = np.loadtxt(path, comments="#", ndmin=2)
    matches = data[np.isclose(data[:, 0], float(zsnap))]
    if matches.size == 0:
        raise ValueError(f"No best-fit b1 entry for zsnap={zsnap} in {path}")
    return float(matches[0, 1])


def default_pk_path(tracer, zsnap, cosmo="000", simtype="SU", abias_suffix=""):
    return mock_pk_path(tracer, zsnap, cosmo=cosmo, simtype=simtype, abias_suffix=abias_suffix)


def default_cov_path(tracer, zsnap, cosmo="000", simtype="SU", ells=(0,), cov_veff=False):
    veff_tag = "_veff" if cov_veff else ""
    return mock_dir(simtype) / f"{tracer}_z{ztag(zsnap)}_c{cosmo}_default_cov{ell_tag(ells)}{veff_tag}_RSD.npy"


def load_png_rebinned_pk(path, kmin=0.0, kmax=0.1, dk=0.002, ells=(0,)):
    from pypower import PowerSpectrumMultipoles

    ells = normalize_ells(ells)
    power = PowerSpectrumMultipoles.load(path)
    native_dk = float(np.mean(np.diff(power.edges[0])))
    rebin = int(round(dk / native_dk))
    if not np.isclose(rebin * native_dk, dk, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"Requested dk={dk:g} is not an integer multiple of native dk={native_dk:g} for {path}.")
    if rebin > 1:
        power = power[:(power.shape[0] // rebin) * rebin].rebin(factor=rebin)
    target_k, _ = power(return_k=True, complex=False, ell=ells[0])
    mask = (target_k >= kmin) & (target_k < kmax)
    data_size = float(power.attrs.get("data_size1", power.attrs.get("data_size", 0.0)))

    data = {
        "k": target_k[mask],
        "nmodes": power.nmodes[mask],
        "shotnoise": float(power.shotnoise),
        "nbar": data_size / BOXSIZE**3,
        "boxsize": BOXSIZE,
        "ells": ells,
    }
    for ell in ells:
        _, pkell = power(return_k=True, complex=False, ell=ell)
        data[f"pk{ell}"] = pkell[mask]
    return data


def load_pk(tracer, zsnap, cosmo="000", simtype="SU", kmin=0.0, kmax=0.1, dk=0.002, ells=(0,), abias_suffix=""):
    pk_path = default_pk_path(tracer, zsnap, cosmo=cosmo, simtype=simtype, abias_suffix=abias_suffix)
    return load_png_rebinned_pk(pk_path, kmin=kmin, kmax=kmax, dk=dk, ells=ells)


def import_lsstypes():
    try:
        from lsstypes import CovarianceMatrix, Mesh2SpectrumPole, Mesh2SpectrumPoles
    except ImportError as exc:
        raise ImportError(
            "lsstypes is required for covariance loading. Source "
            "/global/common/software/desi/users/adematti/cosmodesi_environment.sh main first."
        ) from exc
    return CovarianceMatrix, Mesh2SpectrumPole, Mesh2SpectrumPoles


def make_lsstype_pk_observable(pk_data, indices=None):
    _, Mesh2SpectrumPole, Mesh2SpectrumPoles = import_lsstypes()

    ells = normalize_ells(pk_data.get("ells", (0,)))
    k = np.asarray(pk_data["k"], dtype="f8")
    nmodes = np.asarray(pk_data.get("nmodes", np.ones_like(k)), dtype="i8")
    if indices is not None:
        k = k[indices]
        nmodes = nmodes[indices]

    poles = []
    for ell in ells:
        pkell = np.asarray(pk_data[f"pk{ell}"], dtype="f8")
        if indices is not None:
            pkell = pkell[indices]
        poles.append(Mesh2SpectrumPole(
            k=k,
            num_raw=pkell,
            num_shotnoise=np.zeros_like(pkell),
            norm=np.ones_like(pkell),
            nmodes=nmodes,
            ell=ell,
        ))
    return Mesh2SpectrumPoles(poles, ells=ells)


def load_covariance(tracer, zsnap, cosmo="000", simtype="SU", kmin=0.0, kmax=0.1, dk=0.002, ells=(0,), cov_veff=False, cov_cosmo=None, abias_suffix=""):
    CovarianceMatrix, _, _ = import_lsstypes()
    ells = normalize_ells(ells)
    cov_cosmo = cosmo if cov_cosmo is None else cov_cosmo
    label = f"{simtype} {tracer} z={zsnap:.3f} data c{cosmo} cov c{cov_cosmo}"
    covariance_path = default_cov_path(tracer, zsnap, cosmo=cov_cosmo, simtype=simtype, ells=ells, cov_veff=cov_veff)
    covariance = np.asarray(np.load(covariance_path), dtype="f8")
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(
            f"Covariance for {label} must be a square matrix, but {covariance_path} has shape "
            f"{covariance.shape}."
        )
    if covariance.shape[0] % len(ells):
        raise ValueError(
            f"Covariance shape mismatch for {label}: {covariance_path} has {covariance.shape[0]} "
            f"rows, which is not divisible by the requested ells={ells}."
        )

    # The .npy covariance stores only the matrix.  Match it to the P(k) grid by
    # assuming each ell block has N bins and corresponds to the last N available k bins.
    covariance_nbins = covariance.shape[0] // len(ells)
    reference_kmax = max(float(kmax), covariance_nbins * float(dk))
    pk_data = load_pk(
        tracer, zsnap, cosmo=cosmo, simtype=simtype,
        kmin=0.0, kmax=reference_kmax, dk=dk, ells=ells, abias_suffix=abias_suffix,
    )
    k = np.asarray(pk_data["k"], dtype="f8")
    if covariance_nbins > k.size:
        raise ValueError(
            f"Covariance shape mismatch for {label}: {covariance_path} has {covariance_nbins} "
            f"bins, but the P(k) file has only {k.size} bins up to kmax={reference_kmax:g}."
        )

    covariance_pk_indices = np.arange(k.size - covariance_nbins, k.size)
    covariance_observable = make_lsstype_pk_observable(pk_data, indices=covariance_pk_indices)
    if covariance_observable.size != covariance.shape[0]:
        raise ValueError(
            f"Covariance shape mismatch for {label}: observable for ells={ells} has "
            f"{covariance_observable.size} data points, but {covariance_path} has "
            f"shape {covariance.shape}."
        )
    covariance_lsstype = CovarianceMatrix(
        covariance,
        observable=covariance_observable,
        attrs={"path": str(covariance_path)},
    )

    expected_pk_indices = np.flatnonzero((k >= kmin) & (k < kmax))
    if expected_pk_indices.size == 0:
        raise ValueError(
            f"No P(k) bins selected for {label}: requested {kmin:g} <= k < {kmax:g}; "
            f"available range is {np.nanmin(k):g} <= k <= {np.nanmax(k):g}."
        )

    covariance_lsstype = covariance_lsstype.at.observable.select(k=(kmin, np.nextafter(kmax, -np.inf)))
    covariance_fit = np.asarray(covariance_lsstype.value(), dtype="f8")
    k_fit = np.asarray(covariance_lsstype.observable.get(ells[0]).k, dtype="f8")
    data_fit = np.asarray(covariance_lsstype.observable.value(), dtype="f8")

    expected_k = k[expected_pk_indices]
    expected_size = expected_k.size * len(ells)
    if covariance_fit.shape != (expected_size, expected_size) or not np.array_equal(k_fit, expected_k):
        covariance_k = k[covariance_pk_indices]
        cov_range = f"{np.nanmin(covariance_k):g} <= k <= {np.nanmax(covariance_k):g}"
        req_range = f"{expected_k[0]:g} <= k <= {expected_k[-1]:g}"
        raise ValueError(
            f"Covariance k-range mismatch for {label}: {covariance_path} covers {cov_range}, "
            f"but the requested fit bins cover {req_range}. Recompute the covariance with "
            f"compatible --kmin/--kmax, or use a fit k-range covered by the saved covariance."
        )

    return {
        "covariance": covariance_fit,
        "k": k_fit,
        "data": data_fit,
        "pk0": np.asarray(covariance_lsstype.observable.get(0).value(), dtype="f8") if 0 in ells else data_fit,
        "ells": ells,
        "observable": covariance_lsstype.observable,
        "covariance_lsstype": covariance_lsstype,
        "pk_data": pk_data,
        "pk_indices": expected_pk_indices,
        "path": covariance_path,
    }


def compute_thecov_gaussian_covariance(pk_data, kmin=0.0, kmax=0.1, dk=0.002, ells=(0,), volume=None):
    import thecov.covariance
    import thecov.geometry

    ells = normalize_ells(ells)
    k = np.asarray(pk_data["k"], dtype="f8")
    expected_nbins = int(np.rint((kmax - kmin) / dk))
    mask = (k >= kmin) & (k < kmax)
    if np.count_nonzero(mask) != expected_nbins:
        mask = (k >= kmin) & (k <= kmax)

    volume = float(pk_data.get("covariance_volume", float(pk_data["boxsize"]) ** 3)) if volume is None else float(volume)
    geometry = thecov.geometry.BoxGeometry(
        volume=volume,
        nbar=float(pk_data["nbar"]),
    )
    gaussian = thecov.covariance.GaussianCovariance(geometry)
    gaussian.set_kbins(kmin, kmax, dk)
    for ell in ells:
        pkell = np.asarray(pk_data[f"pk{ell}"], dtype="f8")
        if pkell.size != expected_nbins:
            pkell = pkell[mask]
        if pkell.size != expected_nbins:
            raise ValueError(
                f"P_{ell}(k) bin count mismatch for covariance: expected {expected_nbins} bins from "
                f"kmin={kmin}, kmax={kmax}, dk={dk}, but found {pkell.size} bins in the loaded spectrum."
            )
        gaussian.set_galaxy_pk_multipole(pkell, ell, has_shotnoise=False)
    gaussian.compute_covariance()
    return np.block([
        [np.asarray(gaussian.get_ell_cov(ell1, ell2).cov, dtype="f8") for ell2 in ells]
        for ell1 in ells
    ])


def save_covariance(tracer, zsnap, covariance, cosmo="000", simtype="SU", ells=(0,), cov_veff=False):
    ensure_dir(mock_dir(simtype))
    output = default_cov_path(tracer, zsnap, cosmo=cosmo, simtype=simtype, ells=ells, cov_veff=cov_veff)
    np.save(output, covariance)
    return output


def cov_veff_for_zsnap(zsnap):
    config = None
    for zref, item in COV_VEFF_CONFIG.items():
        if np.isclose(float(zsnap), zref, rtol=0.0, atol=1.0e-8):
            config = item
            break
    if config is None:
        raise ValueError(f"No --cov-veff configuration for zsnap={zsnap:g}.")

    from effective_volume import read_nz, veff_from_file

    zmin, zmax, area_total = config
    ngc_file = PAPER_DATA / "LRG_NGC_nz.txt"
    sgc_file = PAPER_DATA / "LRG_SGC_nz.txt"
    ngc_area0 = read_nz(ngc_file)["area"]
    sgc_area0 = read_nz(sgc_file)["area"]
    area_ngc = area_total * ngc_area0 / (ngc_area0 + sgc_area0)
    area_sgc = area_total * sgc_area0 / (ngc_area0 + sgc_area0)
    ngc = veff_from_file(ngc_file, zmin, zmax, COV_VEFF_P0, area_ngc)
    sgc = veff_from_file(sgc_file, zmin, zmax, COV_VEFF_P0, area_sgc)
    veff = ngc[0] + sgc[0]
    print(
        f"Computed --cov-veff for z={zsnap:.3f}: {zmin:g}<z<{zmax:g}, "
        f"area={area_total:g} deg^2, P0={COV_VEFF_P0:g}, "
        f"V_eff={veff:.6e} (Mpc/h)^3 = {veff / 1.0e9:.6f} (Gpc/h)^3"
    )
    return veff


def volume_for_index(volumes, zsnaps, index, cov_veff=False):
    if cov_veff:
        if volumes is not None:
            raise ValueError("Use either --volume or --cov-veff, not both.")
        return cov_veff_for_zsnap(zsnaps[index])
    if volumes is None:
        return None
    if len(volumes) == 1:
        return float(volumes[0])
    if len(volumes) != len(zsnaps):
        raise ValueError("--volume must have either one value or one value per --zsnaps entry.")
    return float(volumes[index])


def fit_prefix(tracer, zsnaps, mode, simtype, cosmo, use_bphi_prior, ells=(0,), abias_suffix="", kmin=None, kmax=None, bphi_prior_ab=False):
    prior_tag = "with_bphi_prior_AB" if bphi_prior_ab else ("with_bphi_prior" if use_bphi_prior else "no_bphi_prior")
    ztag_set = zset_tag(zsnaps)
    return FIT_DIR / f"png_fit_{simtype}_{tracer}_z{ztag_set}_c{cosmo}_{mode}{ell_tag(ells)}{abias_suffix}{k_range_tag(kmin, kmax)}_{prior_tag}"


def flat_prior_abs(name):
    return abs(float(PRIORS_FLAT[name]))


def bfnl_flat_limit():
    return flat_prior_abs("bphi") * flat_prior_abs("fnl_loc")


def scalar_float(value):
    if value is None:
        return None
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.size != 1:
        return None
    return float(array.reshape(-1)[0])


def point_to_float_dict(point):
    if point is None:
        return None
    values = {}
    for key, value in point.items():
        value = scalar_float(value)
        if value is not None:
            values[str(key)] = value
    return values or None


def save_point(path, point):
    values = point_to_float_dict(point) or {}
    np.savez(path, **values)
    return values


def load_point(path):
    if not Path(path).exists():
        return None
    with np.load(path) as data:
        values = {key: float(data[key]) for key in data.files}
    return values or None


def true_fnl_loc_for_cosmo(cosmo):
    tag = str(cosmo).lower().removeprefix("c").zfill(3)
    return TRUE_FNL_LOC_BY_COSMO.get(tag)


def tracer_z_label(tracer, zsnap):
    return f"{tracer.lower()}z{ztag(zsnap).replace('.', 'p')}"


def find_param_key(params, basename):
    for param in params:
        name = str(param)
        if name == basename or getattr(param, "basename", None) == basename:
            return name
    return None


def bfnl_loc_name_from_bphi_name(name):
    suffix = ".bphi"
    if str(name).endswith(suffix):
        return f"{str(name)[:-len(suffix)]}.bfnl_loc"
    return "bfnl_loc"


def add_bfnl_loc_to_point(point, params, mode="bphi"):
    if point is None:
        return point
    fnl_key = find_param_key(params, "fnl_loc")
    if fnl_key is None:
        return point
    fnl_value = scalar_float(point.get(fnl_key))
    if fnl_value is None:
        return point
    if mode == "bphi":
        for param in params:
            if getattr(param, "basename", None) != "bphi":
                continue
            bphi_name = str(param)
            bphi_value = scalar_float(point.get(bphi_name))
            if bphi_value is None:
                continue
            point[bfnl_loc_name_from_bphi_name(bphi_name)] = bphi_value * fnl_value
    elif mode == "b-p":
        for param in params:
            if getattr(param, "basename", None) != "b1":
                continue
            b1_name = str(param)
            b1_value = scalar_float(point.get(b1_name))
            if b1_value is None:
                continue
            p_name = b1_name.replace(".b1", ".p")
            p_value = scalar_float(point.get(p_name))
            if p_value is None:
                p_value = 1.0
            bphi_value = 2.0 * 1.686 * (b1_value - p_value)
            point[b1_name.replace(".b1", ".bfnl_loc")] = bphi_value * fnl_value
    return point


def add_bfnl_loc_to_chain(chain, mode="bphi"):
    fnl_key = find_param_key(chain.params(), "fnl_loc")
    if fnl_key is None:
        return chain
    if mode == "bphi":
        for param in list(chain.params(basename="bphi")):
            bphi_name = str(param)
            bfnl_name = bfnl_loc_name_from_bphi_name(bphi_name)
            chain.set_derived(
                bfnl_name,
                np.asarray(chain[bphi_name]) * np.asarray(chain[fnl_key]),
                latex=r"b_{\phi}f_{\mathrm{NL}}^{\mathrm{loc}}",
            )
    elif mode == "b-p":
        for param in list(chain.params(basename="b1")):
            b1_name = str(param)
            p_name = b1_name.replace(".b1", ".p")
            bfnl_name = b1_name.replace(".b1", ".bfnl_loc")
            b1_values = np.asarray(chain[b1_name])
            if p_name in [str(pp) for pp in chain.params()]:
                p_values = np.asarray(chain[p_name])
            else:
                p_values = 1.0
            bphi_values = 2.0 * 1.686 * (b1_values - p_values)
            chain.set_derived(
                bfnl_name,
                bphi_values * np.asarray(chain[fnl_key]),
                latex=r"b_{\phi}f_{\mathrm{NL}}^{\mathrm{loc}}",
            )
    return chain


def bfnl_fnl_loc_from_values(values, tracer, zsnaps, bphi_stats=None):
    bphi_stats = load_bphi_stats() if bphi_stats is None else bphi_stats
    numerator = None
    denominator = 0.0
    for zsnap in zsnaps:
        name = f"{tracer_z_label(tracer, zsnap)}.bfnl_loc"
        if name not in values:
            continue
        bfnl = np.asarray(values[name], dtype="f8")
        bphi = scalar_float(bphi_stats.get((tracer, zsnap), {}).get("mean"))
        if bphi is None or not np.isfinite(bphi) or bphi == 0.0:
            continue
        numerator = bphi * bfnl if numerator is None else numerator + bphi * bfnl
        denominator += bphi**2
    if numerator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def add_fnl_loc_to_bfnl_point(point, tracer, zsnaps, bphi_stats=None):
    if point is None or scalar_float(point.get("fnl_loc")) is not None:
        return point
    fnl_loc = bfnl_fnl_loc_from_values(point, tracer, zsnaps, bphi_stats=bphi_stats)
    if fnl_loc is not None:
        point["fnl_loc"] = float(np.asarray(fnl_loc))
    return point


def add_fnl_loc_to_bfnl_chain(chain, tracer, zsnaps, bphi_stats=None):
    if find_param_key(chain.params(), "fnl_loc") is not None:
        return chain
    values = {str(param): np.asarray(chain[str(param)]) for param in chain.params(basename="bfnl_loc")}
    fnl_loc = bfnl_fnl_loc_from_values(values, tracer, zsnaps, bphi_stats=bphi_stats)
    if fnl_loc is not None:
        chain.set_derived(
            "fnl_loc",
            fnl_loc,
            latex=r"f_{\mathrm{NL}}^{\mathrm{loc}}",
        )
    return chain


def triangle_chain_selection(
    cosmo,
    use_bphi_prior,
    default_burnin,
    default_thin,
    burnin_env=None,
    thin_env=None,
):
    burnin = float(os.environ.get(burnin_env, default_burnin)) if burnin_env else default_burnin
    thin = int(os.environ.get(thin_env, default_thin)) if thin_env else default_thin
    tag = str(cosmo).lower().removeprefix("c").zfill(3)
    if tag == "301" and not use_bphi_prior:
        burnin = float(os.environ.get("BPHI_C301_NO_PRIOR_BURNIN", "0.75"))
        thin = int(os.environ.get("BPHI_C301_NO_PRIOR_THIN", "1"))
    return burnin, thin


def joint_triangle_chain_selection(cosmo, use_bphi_prior):
    return triangle_chain_selection(
        cosmo,
        use_bphi_prior,
        default_burnin=0.6,
        default_thin=5,
        burnin_env="BPHI_JOINT_TRIANGLE_BURNIN",
        thin_env="BPHI_JOINT_TRIANGLE_THIN",
    )


def single_triangle_chain_selection(cosmo, use_bphi_prior):
    return triangle_chain_selection(
        cosmo,
        use_bphi_prior,
        default_burnin=0.5,
        default_thin=5,
        burnin_env="BPHI_TRIANGLE_BURNIN",
        thin_env="BPHI_TRIANGLE_THIN",
    )


def maybe_plot_joint_triangle(tracer, zsnaps, mode, simtype, cosmo, ells=(0,), abias_suffix="", kmin=None, kmax=None):
    from desilike.samples import Chain, Profiles, plotting

    ells = normalize_ells(ells)
    bphi_stats = load_bphi_stats()
    entries = []
    for use_bphi_prior, color, ls, label in [
        (False, "C1", "--", "No bphi prior"),
        (True, "C0", "-", "With bphi prior"),
    ]:
        prefix = fit_prefix(
            tracer, zsnaps, mode, simtype, cosmo, use_bphi_prior,
            ells=ells, abias_suffix=abias_suffix, kmin=kmin, kmax=kmax,
        )
        chain_fn = Path(f"{prefix}_chain.npy")
        if not chain_fn.exists():
            continue
        burnin, thin = joint_triangle_chain_selection(cosmo, use_bphi_prior)
        chain = Chain.load(chain_fn).remove_burnin(burnin)[::thin]
        if mode in ("bphi", "b-p"):
            add_bfnl_loc_to_chain(chain, mode=mode)
        elif mode == "bfnl":
            add_fnl_loc_to_bfnl_chain(chain, tracer, zsnaps, bphi_stats=bphi_stats)
        chain_map = point_to_float_dict(chain.choice(index="argmax", input=True))
        chain_map = add_bfnl_loc_to_point(chain_map, chain.params(), mode=mode)
        if mode == "bfnl":
            chain_map = add_fnl_loc_to_bfnl_point(chain_map, tracer, zsnaps, bphi_stats=bphi_stats)
        if chain_map is not None:
            save_point(f"{prefix}_chain_map.npz", chain_map)
        profiler_bestfit = load_point(f"{prefix}_profiler_bestfit.npz")
        if profiler_bestfit is None:
            profiles_fn = Path(f"{prefix}_profiles.npy")
            if profiles_fn.exists():
                profiles = Profiles.load(profiles_fn)
                profiler_bestfit = point_to_float_dict(profiles.bestfit.choice(input=True))
                if profiler_bestfit is not None:
                    save_point(f"{prefix}_profiler_bestfit.npz", profiler_bestfit)
        profiler_bestfit = add_bfnl_loc_to_point(profiler_bestfit, chain.params(), mode=mode)
        if mode == "bfnl":
            profiler_bestfit = add_fnl_loc_to_bfnl_point(profiler_bestfit, tracer, zsnaps, bphi_stats=bphi_stats)
        entries.append({
            "chain": chain,
            "label": label,
            "color": color,
            "ls": ls,
            "bestfit": profiler_bestfit,
            "map": chain_map,
        })

    if len(entries) < 2:
        return None

    chains = [entry["chain"] for entry in entries]
    labels = [entry["label"] for entry in entries]
    colors = [entry["color"] for entry in entries]
    linestyles = [entry["ls"] for entry in entries]
    params = chains[0].params(varied=True, input=True)
    params = triangle_params_with_bfnl_loc(chains[0], params, mode)
    true_fnl_loc = true_fnl_loc_for_cosmo(cosmo)
    truth_values = triangle_truth_values(params, tracer, zsnaps, cosmo, bphi_stats=bphi_stats)
    g = plotting.plot_triangle(
        chains,
        params=params,
        labels=labels,
        contour_colors=colors,
        contour_ls=linestyles,
        filled=False,
    )
    for entry in entries:
        overlay_markers_on_triangle(
            g,
            params,
            entry["bestfit"],
            color=entry["color"],
            marker="*",
            label=f"{entry['label']} maximum likelihood",
            diagonal_ls="--",
            size=72,
        )
        overlay_markers_on_triangle(
            g,
            params,
            entry["map"],
            color=entry["color"],
            marker="o",
            label=f"{entry['label']} MAP",
            diagonal_ls=":",
            size=42,
        )
    overlay_truth_on_triangle(
        g,
        params,
        "fnl_loc",
        true_fnl_loc,
        color="black",
        linestyle="-",
    )
    overlay_truth_values_on_triangle(g, params, truth_values)
    annotate_joint_triangle_map(g, params, entries, true_fnl_loc)

    output = FIT_DIR / f"png_fit_{simtype}_{tracer}_z{zset_tag(zsnaps)}_c{cosmo}_{mode}{ell_tag(ells)}{abias_suffix}{k_range_tag(kmin, kmax)}_joint_triangle.png"
    g.export(str(output))
    return output

def parse_zsnaps_text(text, default=None):
    if text is None:
        return list(default if default is not None else DEFAULT_ZS)
    if isinstance(text, (list, tuple)):
        text = ",".join(str(item) for item in text)
    return [float(item) for item in str(text).replace(",", " ").split()]


def _broadcast_plot_arg(values, nitems, default, name):
    if values is None:
        return [default] * nitems
    values = list(values)
    if len(values) == 1 and nitems > 1:
        return values * nitems
    if len(values) != nitems:
        raise ValueError(f"{name} must have one value or {nitems} values.")
    return values


def load_bphi_stats(path=BPHI_STATS_FILE, update_paths=BPHI_STATS_UPDATE_FILES):
    stats = {}
    b1_stats = {}
    raw_counts = {}
    raw_b1 = {}
    base_path = Path(path)
    paths = [base_path] + [Path(path) for path in update_paths if Path(path).exists()]
    update_keys = set()

    def add_bphi(tracer, zsnap, hod_id, bphi, b1=None):
        key = (tracer, float(zsnap))
        hod_id = int(hod_id)
        stats.setdefault(key, {})[hod_id] = float(bphi)
        if b1 is not None and np.isfinite(float(b1)):
            b1_stats.setdefault(key, {})[hod_id] = float(b1)

    def mark_update_key(tracer, zsnap):
        key = (tracer, float(zsnap))
        if key not in update_keys:
            update_keys.add(key)
            stats[key] = {}
            b1_stats.pop(key, None)

    for path in paths:
        is_update = path != base_path
        with open(path, "r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    if "bphi=ln(Ngal_c000/Ngal_c004)" in line:
                        raise ValueError(
                            f"{path} uses the old SU bphi convention without the factor of 2. "
                            "Regenerate it with HOD_bphi.py so bphi=2*ln(Ngal_c000/Ngal_c004)/ln(0.807952/0.75)."
                        )
                    continue
                parts = line.split()
                if len(parts) >= 5 and parts[2] in ("000", "004"):
                    tracer, zsnap, cosmo, hod_id, ngal = parts[:5]
                    if is_update:
                        mark_update_key(tracer, zsnap)
                    raw_key = (tracer, float(zsnap), int(hod_id))
                    raw_counts.setdefault(raw_key, {})[cosmo] = float(ngal)
                    if cosmo == "000" and len(parts) >= 6:
                        raw_b1[raw_key] = float(parts[5])
                    continue
                tracer, zsnap, hod_id, _, _, bphi = parts[:6]
                if is_update:
                    mark_update_key(tracer, zsnap)
                b1 = float(parts[6]) if len(parts) >= 7 else None
                add_bphi(tracer, zsnap, hod_id, bphi, b1=b1)

    for (tracer, zsnap, hod_id), counts in raw_counts.items():
        if "000" not in counts or "004" not in counts:
            continue
        bphi = 2.0 * np.log(counts["000"] / counts["004"]) / BPHI_SU_DENOM
        add_bphi(tracer, zsnap, hod_id, bphi, b1=raw_b1.get((tracer, zsnap, hod_id)))

    summary = {}
    for key, values in stats.items():
        hod_ids = sorted(values)
        arr = np.asarray([values[hod_id] for hod_id in hod_ids], dtype="f8")
        item = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "n": int(arr.size),
        }
        if key in b1_stats:
            b1 = np.asarray([b1_stats[key][hod_id] for hod_id in sorted(b1_stats[key])], dtype="f8")
            item["b1"] = float(b1[0])
            item["b1_std"] = float(np.std(b1))
            item["b1_n"] = int(b1.size)
        summary[key] = item
    return summary


def load_assembly_bias_bphi_prior(path=BPHI_STATS_ASSEMBLY_FILE, tracer="LRG", zsnap=1.1):
    counts = {}
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            tracer_i, zsnap_i, cosmo, hod_id, ngal = parts[:5]
            if tracer_i != tracer or not np.isclose(float(zsnap_i), float(zsnap)):
                continue
            counts.setdefault(int(hod_id), {})[cosmo] = float(ngal)

    bphi = []
    invalid = []
    for hod_id in range(0, 101):
        item = counts.get(hod_id, {})
        ngal_000 = item.get("000")
        ngal_004 = item.get("004")
        if ngal_000 is None or ngal_004 is None or ngal_000 <= BPHI_NGAL_MIN or ngal_004 <= BPHI_NGAL_MIN:
            invalid.append(hod_id)
            continue
        bphi.append(2.0 * np.log(ngal_000 / ngal_004) / BPHI_SU_DENOM)

    if not bphi:
        raise ValueError(f"No valid assembly-bias bphi realizations found in {path}.")
    bphi = np.asarray(bphi, dtype="f8")
    return {
        "mean": float(np.mean(bphi)),
        "std": float(np.std(bphi)),
        "n": int(bphi.size),
        "invalid_hod_ids": invalid,
        "path": Path(path),
    }


def load_png_mock_b1(tracer, zsnap, cosmo="000", b1_cosmo="000"):
    path = BPHI_BESTFIT_B1_FILE
    b1 = load_bestfit_b1(zsnap, path=path)
    bphi = 2.0 * 1.686 * (b1 - 1.0)
    true_fnl_loc = true_fnl_loc_for_cosmo(cosmo)

    return {
        "b1": b1,
        "bphi": bphi,
        "bfnl_loc": None if true_fnl_loc is None else bphi * true_fnl_loc,
        "path": path,
        "b1_cosmo": b1_cosmo,
        "fnl_cosmo": cosmo,
    }

def triangle_truth_values(params, tracer, zsnaps, cosmo, bphi_stats=None):
    true_fnl_loc = true_fnl_loc_for_cosmo(cosmo)
    truth = {}
    names = {str(param) for param in params}

    for zsnap in zsnaps:
        label = tracer_z_label(tracer, zsnap)

        b1_name = f"{label}.b1"
        bfnl_name = f"{label}.bfnl_loc"

        png_truth = load_png_mock_b1(tracer, zsnap, cosmo=cosmo, b1_cosmo="000")

        if b1_name in names:
            truth[b1_name] = png_truth["b1"]

        if true_fnl_loc is not None and bfnl_name in names:
            truth[bfnl_name] = png_truth["bfnl_loc"]

    return truth


def logsumexp(values):
    values = np.asarray(values, dtype="f8")
    vmax = np.max(values)
    if not np.isfinite(vmax):
        return -np.inf
    return float(vmax + np.log(np.sum(np.exp(values - vmax))))


def make_bfnl_bphi_prior_likelihood(
    bfnl_param_names,
    bphi_locs,
    bphi_scales,
    fnl_limit=None,
    bfnl_limit=None,
    ngrid=None,
):
    # Marginalize p(bfnl | bphi prior, flat fnl) over one common fnl for all z bins.
    from desilike.likelihoods import BaseLikelihood
    from desilike.parameter import Parameter, ParameterCollection

    bfnl_param_names = list(bfnl_param_names)
    bphi_locs = np.asarray(bphi_locs, dtype="f8")
    bphi_scales = np.asarray(bphi_scales, dtype="f8")
    fnl_limit = flat_prior_abs("fnl_loc") if fnl_limit is None else abs(float(fnl_limit))
    bfnl_limit = bfnl_flat_limit() if bfnl_limit is None else abs(float(bfnl_limit))
    ngrid = int(os.environ.get("BPHI_BFNL_PRIOR_NGRID", ngrid or BPHI_BFNL_PRIOR_NGRID))

    if len(bfnl_param_names) != bphi_locs.size or bphi_locs.size != bphi_scales.size:
        raise ValueError("bfnl parameter names, bphi means and bphi stds must have the same length.")
    if fnl_limit <= 0:
        raise ValueError("The flat fnl_loc prior half-width must be positive.")
    if np.any(bphi_scales <= 0):
        raise ValueError("All bphi prior standard deviations must be positive.")
    if ngrid < 128:
        raise ValueError("BPHI_BFNL_PRIOR_NGRID must be at least 128.")
    if ngrid % 2:
        ngrid += 1

    prior_param_names = [f"prior_bfnl_loc_{i}" for i in range(len(bfnl_param_names))]

    def _params(params, bfnl_param_names=None, prior_param_names=None, bfnl_limit=None):
        params = ParameterCollection(params)
        for source_name, prior_name in zip(bfnl_param_names or [], prior_param_names or []):
            params.set(Parameter(
                prior_name,
                derived="{" + str(source_name) + "}",
                fixed=True,
                latex=r"b_{\phi}f_{\mathrm{NL}}^{\mathrm{loc}}",
            ))
        return params

    class BfnlBphiPriorLikelihood(BaseLikelihood):

        def initialize(
            self,
            bfnl_param_names=None,
            prior_param_names=None,
            bphi_locs=None,
            bphi_scales=None,
            fnl_limit=None,
            bfnl_limit=None,
            ngrid=None,
            name="bfnl_bphi_prior",
        ):
            super().initialize(name=name)
            self.bfnl_param_names = list(bfnl_param_names)
            self.prior_param_names = list(prior_param_names)
            self.bphi_locs = np.asarray(bphi_locs, dtype="f8")
            self.bphi_scales = np.asarray(bphi_scales, dtype="f8")
            self.fnl_limit = abs(float(fnl_limit))
            self.bfnl_limit = abs(float(bfnl_limit))
            self.ngrid = int(ngrid)
            edges = np.linspace(-self.fnl_limit, self.fnl_limit, self.ngrid + 1)
            self.fnl_grid = 0.5 * (edges[:-1] + edges[1:])
            self.log_fnl_weight = np.log(edges[1] - edges[0]) - np.log(2.0 * self.fnl_limit)
            self.log_norm = -np.log(self.bphi_scales) - 0.5 * np.log(2.0 * np.pi)

        def calculate(self, **params):
            bfnl = np.asarray([params[name] for name in self.prior_param_names], dtype="f8")
            if np.any(np.abs(bfnl) > self.bfnl_limit):
                self.loglikelihood = -np.inf
                return
            fnl = self.fnl_grid
            bphi = bfnl[:, None] / fnl[None, :]
            chi = (bphi - self.bphi_locs[:, None]) / self.bphi_scales[:, None]
            logpdf = (
                self.log_norm[:, None]
                - 0.5 * chi**2
                - np.log(np.abs(fnl))[None, :]
            )
            self.loglikelihood = logsumexp(self.log_fnl_weight + np.sum(logpdf, axis=0))

    BfnlBphiPriorLikelihood._params = staticmethod(_params)

    return BfnlBphiPriorLikelihood(
        bfnl_param_names=bfnl_param_names,
        prior_param_names=prior_param_names,
        bfnl_limit=bfnl_limit,
        bphi_locs=bphi_locs,
        bphi_scales=bphi_scales,
        fnl_limit=fnl_limit,
        ngrid=ngrid,
    )

def patch_png_bfnl_calculate(PNGTracerPowerSpectrumMultipoles):
    if getattr(PNGTracerPowerSpectrumMultipoles, "_bfnl_tuple_patch", False):
        return

    from scipy import constants
    from desilike.jax import interp1d
    from desilike.jax import numpy as jnp

    def calculate(self, **kwargs):
        if hasattr(self, "pack_input_bias_params"):
            bias_params = self.pack_input_bias_params(
                kwargs,
                defaults=dict(b1=1.0, sigmas=0.0, sn0=0.0, bphi=1.0, p=1.0, bfnl_loc=0.0),
            )
            (b1X, b1Y), (sigmasX, sigmasY), sn0 = [bias_params[name] for name in ["b1", "sigmas", "sn0"]]
        else:
            # Backward-compatible path for local desilike versions where PNG is single-tracer.
            b1 = kwargs.get("b1", 1.0)
            sigmas = kwargs.get("sigmas", 0.0)
            sn0 = kwargs.get("sn0", 0.0)
        self.z = self.template.z
        jac, kap, muap = self.template.ap_k_mu(self.k, self.mu)
        pk_dd = self.template.pk_dd
        kin = self.template.k
        cosmo = self.template.cosmo
        f = self.template.f
        pk_prim = cosmo.get_primordial(mode="scalar").pk_interpolator()(kin)
        if self.method == "prim":
            pphi_prim = 9.0 / 25.0 * 2.0 * np.pi**2 / kin**3 * pk_prim / cosmo.h**3
            alpha = 1.0 / (pk_dd / pphi_prim) ** 0.5
        else:
            tk = (pk_dd / pk_prim / kin / (pk_dd[0] / pk_prim[0] / kin[0])) ** 0.5
            znorm = 10.0
            normalized_growth_factor = cosmo.growth_factor(self.template.z) / cosmo.growth_factor(znorm) / (1.0 + znorm)
            alpha = 3.0 * cosmo.Omega0_m * 100**2 / (2.0 * (constants.c / 1e3) ** 2 * kin**2 * tk * normalized_growth_factor)
        kin, pk_dd, alpha = kin[1:], pk_dd[1:], alpha[1:]
        alpha = interp1d(jnp.log10(kap), np.log10(kin), alpha)
        if hasattr(self, "pack_input_bias_params"):
            if self.mode == "bphi":
                fnl_loc = kwargs["fnl_loc"]
                bphiX, bphiY = bias_params["bphi"]
                bfnl_locX, bfnl_locY = bphiX * fnl_loc, bphiY * fnl_loc
            elif self.mode == "b-p":
                fnl_loc = kwargs["fnl_loc"]
                pX, pY = bias_params["p"]
                bfnl_locX, bfnl_locY = [2.0 * 1.686 * (b1 - p) * fnl_loc for b1, p in [(b1X, pX), (b1Y, pY)]]
            else:
                bfnl_locX, bfnl_locY = bias_params["bfnl_loc"]
            bX, bY = b1X + bfnl_locX * alpha, b1Y + bfnl_locY * alpha
            fog = 1.0 / ((1.0 + sigmasX**2 * kap**2 * muap**2 / 2.0) * (1.0 + sigmasY**2 * kap**2 * muap**2 / 2.0))
            pkmu = jac * fog * (bX + f * muap**2) * (bY + f * muap**2) * interp1d(jnp.log10(kap), np.log10(kin), pk_dd) + sn0 / self.nd
        else:
            if self.mode == "bphi":
                fnl_loc = kwargs["fnl_loc"]
                bphi = kwargs["bphi"]
                bfnl_loc = bphi * fnl_loc
            elif self.mode == "b-p":
                fnl_loc = kwargs["fnl_loc"]
                p = kwargs.get("p", 1.0)
                bfnl_loc = 2.0 * 1.686 * (b1 - p) * fnl_loc
            else:
                bfnl_loc = kwargs["bfnl_loc"]
            bias = b1 + bfnl_loc * alpha
            fog = 1.0 / (1.0 + sigmas**2 * kap**2 * muap**2 / 2.0) ** 2
            pkmu = jac * fog * (bias + f * muap**2) ** 2 * interp1d(jnp.log10(kap), np.log10(kin), pk_dd) + sn0 / self.nd
        self.power = self.to_poles(pkmu)

    PNGTracerPowerSpectrumMultipoles.calculate = calculate
    PNGTracerPowerSpectrumMultipoles._bfnl_tuple_patch = True

def build_joint_png_likelihood(
    tracer,
    zsnaps,
    use_bphi_prior,
    mode="bphi",
    simtype="SU",
    cosmo="000",
    kmin=0.005,
    kmax=0.1,
    dk=0.002,
    ells=(0,),
    cov_veff=False,
    abias_suffix="",
    bphi_prior_ab=False,
):
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles

    if use_bphi_prior and mode == "b-p":
        raise ValueError("bphi priors can only be applied when mode='bphi' or 'bfnl'.")
    if bphi_prior_ab:
        if not use_bphi_prior:
            raise ValueError("--bphi-prior-ab requires --with-bphi-prior.")
        if len(zsnaps) != 1 or not np.isclose(float(zsnaps[0]), 1.1):
            raise ValueError("WARNING: assembly bias bphi prior not ready for this setting; use --zsnaps 1.1 only.")
    ells = normalize_ells(ells)
    likelihood_kmin = max(kmin, fundamental_k())

    if mode == "bfnl":
        patch_png_bfnl_calculate(PNGTracerPowerSpectrumMultipoles)
    
    observables = []
    covariance_blocks = []
    bphi_stats = load_bphi_stats()
    if bphi_prior_ab:
        ab_summary = load_assembly_bias_bphi_prior(tracer=tracer, zsnap=1.1)
        bphi_stats[(tracer, 1.1)] = ab_summary
        print(
            f"Using assembly-bias bphi prior for {tracer} z=1.100: "
            f"mean={ab_summary['mean']:.8f}, std={ab_summary['std']:.8f}, "
            f"n={ab_summary['n']} from {ab_summary['path']}"
        )
    bfnl_prior_param_names = []
    bfnl_prior_locs = []
    bfnl_prior_scales = []
    for zsnap in zsnaps:
        bphi_summary = bphi_stats[(tracer, zsnap)]
        covariance_data = load_covariance(
            tracer, zsnap, cosmo=cosmo, simtype=simtype,
            kmin=likelihood_kmin, kmax=kmax, dk=dk, ells=ells, cov_veff=cov_veff, cov_cosmo="000",
            abias_suffix=abias_suffix,
        )
        k_fit = np.asarray(covariance_data["k"], dtype="f8")
        data_fit = np.asarray(covariance_data["data"], dtype="f8")
        covariance_fit = covariance_data["covariance"]
        if data_fit.size != k_fit.size * len(ells):
            raise ValueError(
                f"P(k) shape mismatch for {simtype} {tracer} z={zsnap:.3f} c{cosmo}: "
                f"k has {k_fit.size} bins and ells={ells}, but data has {data_fit.size} values."
            )
        template = FixedPowerSpectrumTemplate(z=zsnap, fiducial="DESI")
        theory = PNGTracerPowerSpectrumMultipoles(template=template, mode=mode, ells=ells)
        label = f"{tracer.lower()}z{ztag(zsnap).replace('.', 'p')}"

        theory.init.params["b1"].update(namespace=label)
        if len(ells) == 1:
            theory.init.params["sn0"].update(namespace=label, fixed=True, value=0.0)
            theory.init.params["sigmas"].update(namespace=label, fixed=True, value=0.0)
        else:
            theory.init.params["sn0"].update(namespace=label)
            theory.init.params["sigmas"].update(namespace=label)
        if mode == "bphi":
            theory.init.params["bphi"].update(namespace=label)
            if use_bphi_prior:
                bphi_summary = bphi_stats[(tracer, zsnap)]
                theory.init.params[f"{label}.bphi"].update(
                    prior={"dist": "norm", "loc": bphi_summary["mean"], "scale": bphi_summary["std"]},
                    ref={"dist": "norm", "loc": bphi_summary["mean"], "scale": max(0.2 * bphi_summary["std"], 0.05)},
                ) 
            else:
                theory.init.params[f"{label}.bphi"].update(
                    prior={"limits": [-flat_prior_abs("bphi"), flat_prior_abs("bphi")]}
                )

        elif mode == "bfnl":
            theory.init.params["bfnl_loc"].update(
                namespace=label,
                prior={"limits": [-bfnl_flat_limit(), bfnl_flat_limit()]},
                ref={"dist": "norm", "loc": 0.0, "scale": 50.0},
            )
            if use_bphi_prior:
                bfnl_prior_param_names.append(f"{label}.bfnl_loc")
                bfnl_prior_locs.append(bphi_summary["mean"])
                bfnl_prior_scales.append(bphi_summary["std"])
            

        elif mode == "b-p":
            theory.init.params["p"].update(fixed=True, value=1.0)

        observable = TracerPowerSpectrumMultipolesObservable(
            data=data_fit,
            covariance=covariance_fit,
            k=[k_fit] * len(ells),
            ells=ells,
            theory=theory,
        )
        unique_obs_name = f"{label}_spectrum2poles"
        observable.name = unique_obs_name
        # For the /global/homes/j/jiaxi/codes/desilike stack, keep the name in init as well
        # to avoid duplicated default labels when ObservablesGaussianLikelihood builds the tree.
        try:
            observable.init.update(name=unique_obs_name)
        except Exception:
            pass
        observables.append(observable)
        covariance_blocks.append(covariance_fit)

    total_size = sum(block.shape[0] for block in covariance_blocks)
    covariance = np.zeros((total_size, total_size), dtype="f8")
    offset = 0
    for block in covariance_blocks:
        size = block.shape[0]
        covariance[offset:offset + size, offset:offset + size] = block
        offset += size

    likelihood = ObservablesGaussianLikelihood(observables=observables, covariance=covariance)
    if mode == "bfnl" and use_bphi_prior:
        bfnl_bphi_prior = make_bfnl_bphi_prior_likelihood(
            bfnl_param_names=bfnl_prior_param_names,
            bphi_locs=bfnl_prior_locs,
            bphi_scales=bfnl_prior_scales,
            fnl_limit=flat_prior_abs("fnl_loc"),
            bfnl_limit=bfnl_flat_limit(),
        )
        likelihood = likelihood + bfnl_bphi_prior
    return likelihood, observables


def run_fit(
    tracer,
    zsnaps,
    use_bphi_prior,
    mode="bphi",
    simtype="SU",
    cosmo="000",
    kmin=0.005,
    kmax=0.1,
    dk=0.002,
    ells=(0,),
    run_sampler=False,
    cov_veff=False,
    abias_suffix="",
    bphi_prior_ab=False,
):
    if bphi_prior_ab:
        if not use_bphi_prior:
            raise ValueError("--bphi-prior-ab requires --with-bphi-prior.")
        if len(zsnaps) != 1 or not np.isclose(float(zsnaps[0]), 1.1):
            raise ValueError("WARNING: assembly bias bphi prior not ready for this setting; use --zsnaps 1.1 only.")

    # Respect user environment/PYTHONPATH by default.
    # Only override desilike import path when DESILIKE_LOCAL_PATH is explicitly set.
    local_desilike = os.environ.get("DESILIKE_LOCAL_PATH")
    if local_desilike and os.path.isdir(local_desilike) and local_desilike not in sys.path:
        sys.path.insert(0, local_desilike)

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    ells = normalize_ells(ells)

    from desilike import setup_logging
    from desilike.profilers import MinuitProfiler
    from desilike.samples import Profiles

    setup_logging("info")
    likelihood, observables = build_joint_png_likelihood(
        tracer=tracer,
        zsnaps=zsnaps,
        use_bphi_prior=use_bphi_prior,
        mode=mode,
        simtype=simtype,
        cosmo=cosmo,
        kmin=kmin,
        kmax=kmax,
        dk=dk,
        ells=ells,
        cov_veff=cov_veff,
        abias_suffix=abias_suffix,
        bphi_prior_ab=bphi_prior_ab,
    )

    ensure_dir(FIT_DIR)
    prefix = fit_prefix(
        tracer, zsnaps, mode, simtype, cosmo, use_bphi_prior,
        ells=ells, abias_suffix=abias_suffix, kmin=kmin, kmax=kmax, bphi_prior_ab=bphi_prior_ab,
    )

    profiles_fn = Path(f"{prefix}_profiles.npy")
    if profiles_fn.exists():
        profiles = Profiles.load(profiles_fn)
    else:
        profiler = MinuitProfiler(likelihood, seed=42, save_fn=str(profiles_fn))
        profiles = profiler.maximize(niterations=1)
        profiles = profiler.profiles if profiles is None else profiles

        fnl_grid = np.linspace(-flat_prior_abs("fnl_loc"), flat_prior_abs("fnl_loc"), 121)
        if mode == "bphi":
            fnl_params = [str(param) for param in profiler.varied_params if getattr(param, "basename", None) == "fnl_loc"]
            if fnl_params:
                profiles = profiler.profile(params=fnl_params, grid=[fnl_grid] * len(fnl_params)) or profiler.profiles
        elif mode == "bfnl":
            bfnl_grid = np.linspace(-bfnl_flat_limit(), bfnl_flat_limit(), 121)
            bfnl_params = [str(param) for param in profiler.varied_params if getattr(param, "basename", None) == "bfnl_loc"]
            if bfnl_params:
                profiles = profiler.profile(params=bfnl_params, grid=[bfnl_grid] * len(bfnl_params)) or profiler.profiles
        profiles.to_stats(tablefmt="pretty", fn=f"{prefix}_profiler_stats.txt")
    profiler_bestfit = profiles.bestfit.choice(input=True)
    likelihood(**profiler_bestfit)
    profiler_bestfit_point = point_to_float_dict(profiler_bestfit)
    if mode == "bfnl":
        profiler_bestfit_point = add_fnl_loc_to_bfnl_point(profiler_bestfit_point, tracer, zsnaps)
    save_point(f"{prefix}_profiler_bestfit.npz", profiler_bestfit_point)

    for zsnap, observable in zip(zsnaps, observables):
        observable.plot(fn=f"{prefix}_z{ztag(zsnap)}_bestfit.png")

    if run_sampler:
        print("Preparing sampler...", flush=True)
        from desilike.samplers import ZeusSampler
        from desilike.samples import plotting

        chain_fn = Path(f"{prefix}_chain.npy")
        if chain_fn.exists():
            print(f"Resuming sampler from {chain_fn}", flush=True)
            sampler = ZeusSampler(likelihood, seed=42, save_fn=f"{prefix}_chain", nwalkers=60, chains=chain_fn)
        else:
            print(f"Starting new sampler; no existing chain found at {chain_fn}", flush=True)
            sampler = ZeusSampler(likelihood, seed=42, save_fn=f"{prefix}_chain", nwalkers=60)
        sampler.run(check_every=30, check={"max_eigen_gr": 0.01})
        print("Sampler run finished; making summary plots.", flush=True)
        chain = sampler.chains[0]
        burnin, thin = single_triangle_chain_selection(cosmo, use_bphi_prior)
        chain = chain.remove_burnin(burnin)[::thin]
        if mode in ("bphi", "b-p"):
            add_bfnl_loc_to_chain(chain, mode=mode)
        elif mode == "bfnl":
            add_fnl_loc_to_bfnl_chain(chain, tracer, zsnaps)
        chain_map = chain.choice(index="argmax", input=True)
        chain_map = add_bfnl_loc_to_point(point_to_float_dict(chain_map), chain.params(), mode=mode)
        if mode == "bfnl":
            chain_map = add_fnl_loc_to_bfnl_point(chain_map, tracer, zsnaps)
        save_point(f"{prefix}_chain_map.npz", chain_map)
        chain.to_stats(tablefmt="pretty", fn=f"{prefix}_chain_stats.txt")
        params = chain.params(varied=True, input=True)
        params = triangle_params_with_bfnl_loc(chain, params, mode)
        truth_values = triangle_truth_values(params, tracer, zsnaps, cosmo)
        g = plotting.plot_triangle(chain, params=params, filled=False)
        profiler_bestfit = add_bfnl_loc_to_point(point_to_float_dict(profiler_bestfit), chain.params(), mode=mode)
        if mode == "bfnl":
            profiler_bestfit = add_fnl_loc_to_bfnl_point(profiler_bestfit, tracer, zsnaps)
        overlay_markers_on_triangle(g, params, profiler_bestfit, color="C0", marker="*",
                                    label="maximum likelihood", diagonal_ls="--", size=64)
        overlay_markers_on_triangle(g, params, chain_map, color="C0", marker="o",
                                    label="MAP", diagonal_ls=":", size=36)
        overlay_truth_on_triangle(
            g,
            params,
            "fnl_loc",
            true_fnl_loc_for_cosmo(cosmo),
            color="black",
            linestyle="-",
        )
        overlay_truth_values_on_triangle(g, params, truth_values)
        annotate_joint_triangle_map(g, params, [{"bestfit": profiler_bestfit, "map": chain_map}], true_fnl_loc_for_cosmo(cosmo))
        g.export(f"{prefix}_triangle.png")
        joint_output = maybe_plot_joint_triangle(
            tracer, zsnaps, mode, simtype, cosmo,
            ells=ells, abias_suffix=abias_suffix, kmin=kmin, kmax=kmax,
        )
        if joint_output is not None:
            print(f"Saved {joint_output}")


def command_pk(args):
    ells = normalize_ells(args.ells)
    abias_suffix = abias_suffix_from_args(args)
    directory = mock_dir(args.simtype)
    print(f"Using existing {args.simtype} P(k) directory: {directory}")
    for zsnap in args.zsnaps:
        input_pk = default_pk_path(args.tracer, zsnap, cosmo=args.cosmo, simtype=args.simtype, abias_suffix=abias_suffix)
        if not input_pk.exists():
            raise FileNotFoundError(f"Expected {args.simtype} P(k) file not found: {input_pk}")
        pk_data = load_pk(
            args.tracer, zsnap, cosmo=args.cosmo, simtype=args.simtype,
            kmin=args.kmin, kmax=args.kmax, dk=args.dk, ells=ells, abias_suffix=abias_suffix,
        )
        print(f"Using existing {args.simtype} P(k) file {input_pk} with ells={ells}, nbins={pk_data['k'].size}")


def command_cov(args):
    ells = normalize_ells(args.ells)
    ensure_dir(mock_dir(args.simtype))
    for iz, zsnap in enumerate(args.zsnaps):
        pk_data = load_pk(
            args.tracer, zsnap, cosmo=args.cosmo, simtype=args.simtype,
            kmin=args.kmin, kmax=args.kmax, dk=args.dk, ells=ells,
        )
        volume = volume_for_index(args.volume, args.zsnaps, iz, cov_veff=args.cov_veff)
        if volume is not None:
            print(f"Using covariance volume for z={zsnap:.3f}: {volume:.6e} (Mpc/h)^3")
        covariance = compute_thecov_gaussian_covariance(
            pk_data=pk_data,
            kmin=args.kmin,
            kmax=args.kmax,
            dk=args.dk,
            ells=ells,
            volume=volume,
        )
        output = save_covariance(args.tracer, zsnap, covariance, cosmo=args.cosmo, simtype=args.simtype, ells=ells, cov_veff=args.cov_veff)
        print(f"Saved covariance to {output}")


def command_fit(args):
    ells = normalize_ells(args.ells)
    abias_suffix = abias_suffix_from_args(args)
    if args.bphi_prior_ab:
        if not args.with_bphi_prior:
            print("WARNING: --bphi-prior-ab requires --with-bphi-prior.", file=sys.stderr)
            raise SystemExit(1)
        if len(args.zsnaps) != 1 or not np.isclose(float(args.zsnaps[0]), 1.1):
            print("WARNING: assembly bias bphi prior not ready for this setting; use --zsnaps 1.1 only.", file=sys.stderr)
            raise SystemExit(1)
    run_fit(
        tracer=args.tracer,
        zsnaps=args.zsnaps,
        use_bphi_prior=args.with_bphi_prior,
        mode=args.mode,
        simtype=args.simtype,
        cosmo=args.cosmo,
        kmin=args.kmin,
        kmax=args.kmax,
        dk=args.dk,
        ells=ells,
        run_sampler=args.run_sampler,
        cov_veff=args.cov_veff,
        abias_suffix=abias_suffix,
        bphi_prior_ab=args.bphi_prior_ab,
    )


def command_plot(args):
    from desilike.samples import Chain

    reference_label = args.reference_label
    comparison_labels = args.comparison_label
    if comparison_labels is None:
        comparison_labels = [f"comparison {i + 1}" for i in range(len(args.comparison_chain))]
    if len(comparison_labels) != len(args.comparison_chain):
        raise ValueError("--comparison-label must have the same length as --comparison-chain.")

    paths = [args.reference_chain] + list(args.comparison_chain)
    labels = [reference_label] + list(comparison_labels)
    modes = [args.reference_mode or args.mode] + _broadcast_plot_arg(
        args.comparison_mode, len(args.comparison_chain), args.mode, "--comparison-mode"
    )
    zsnaps_by_chain = [parse_zsnaps_text(args.reference_zsnaps, default=args.zsnaps)]
    zsnaps_by_chain += [
        parse_zsnaps_text(item, default=args.zsnaps)
        for item in _broadcast_plot_arg(args.comparison_zsnaps, len(args.comparison_chain), None, "--comparison-zsnaps")
    ]

    chains = []
    for path, mode, zsnaps in zip(paths, modes, zsnaps_by_chain):
        if not path.exists():
            raise FileNotFoundError(f"Missing chain file: {path}")
        chain = Chain.load(path).remove_burnin(args.burnin)[::args.thin]
        if mode in ("bphi", "b-p"):
            add_bfnl_loc_to_chain(chain, mode=mode)
        elif mode == "bfnl":
            add_fnl_loc_to_bfnl_chain(chain, args.tracer, zsnaps)
        chains.append(chain)

    output = args.output or FIT_DIR / "chain_comparison_triangle.png"
    ensure_dir(output.parent)

    params = args.params
    if params is None:
        params = chains[0].params(varied=True, input=True)
        params = triangle_params_with_bfnl_loc(chains[0], params, modes[0])

    plot_contours(
        chains,
        params=params,
        items=labels,
        item_std=reference_label,
        texts=args.text,
        fn=output,
        show=False,
        filled=[True] + [False] * len(args.comparison_chain),
    )
    print(f"Saved {output}")


def build_parser():
    parser = argparse.ArgumentParser(description="Pipeline for bphi-prior P(k), covariance and PNG fits.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pk_parser = subparsers.add_parser("pk", help="Validate external P(k) inputs for SU or PNG runs.")
    pk_parser.add_argument("--tracer", default="LRG", choices=["LRG"])
    pk_parser.add_argument("--zsnaps", nargs="+", type=float, default=DEFAULT_ZS)
    pk_parser.add_argument("--simtype", default="SU", choices=SIMTYPES)
    pk_parser.add_argument("--cosmo", default="000")
    pk_parser.add_argument("--kmin", type=float, default=0.0)
    pk_parser.add_argument("--kmax", type=float, default=0.1)
    pk_parser.add_argument("--dk", type=float, default=0.002)
    pk_parser.add_argument("--ells", nargs="+", type=int, default=[0])
    pk_parser.set_defaults(func=command_pk)

    cov_parser = subparsers.add_parser("cov", help="Compute Gaussian P(k) covariance with thecov from external SU/PNG P(k).")
    cov_parser.add_argument("--tracer", default="LRG", choices=["LRG"])
    cov_parser.add_argument("--zsnaps", nargs="+", type=float, default=DEFAULT_ZS)
    cov_parser.add_argument("--simtype", default="SU", choices=SIMTYPES)
    cov_parser.add_argument("--cosmo", default="000")
    cov_parser.add_argument("--kmin", type=float, default=0.0)
    cov_parser.add_argument("--kmax", type=float, default=0.1)
    cov_parser.add_argument("--dk", type=float, default=0.002)
    cov_parser.add_argument("--ells", nargs="+", type=int, default=[0])
    cov_parser.add_argument("--volume", nargs="+", type=float, default=None, help="Covariance volume(s) in (Mpc/h)^3; one value or one per zsnap.")
    cov_parser.add_argument("--cov-veff", action="store_true", help="Use DR1 n(z)-based effective volumes for covariance.")
    cov_parser.set_defaults(func=command_cov)

    fit_parser = subparsers.add_parser("fit", help="Run joint PNG fits with or without bphi priors.")
    fit_parser.add_argument("--tracer", default="LRG", choices=["LRG"])
    fit_parser.add_argument("--zsnaps", nargs="+", type=float, default=DEFAULT_ZS)
    fit_parser.add_argument("--simtype", default="SU", choices=SIMTYPES)
    fit_parser.add_argument("--cosmo", default="000")
    fit_parser.add_argument("--kmin", type=float, default=0.0)
    fit_parser.add_argument("--kmax", type=float, default=0.1)
    fit_parser.add_argument("--dk", type=float, default=0.002)
    fit_parser.add_argument("--ells", nargs="+", type=int, default=[0])
    fit_parser.add_argument("--mode", default="bphi", choices=PNG_MODES)
    fit_parser.add_argument("--with-bphi-prior", action="store_true")
    fit_parser.add_argument("--bphi-prior-ab", "--bphi-prior-AB", dest="bphi_prior_ab", action="store_true",
                            help="Use the z=1.1 assembly-bias bphi prior; only valid for --zsnaps 1.1.")
    fit_parser.add_argument("--run-sampler", action="store_true")
    fit_parser.add_argument("--cov-veff", action="store_true", help="Load DR1 n(z)-based effective-volume covariance files.")
    fit_parser.add_argument("--assembly-bias", action="store_true", help="Use assembly-bias P(k) data-vector files.")
    fit_parser.set_defaults(func=command_fit)

    plot_parser = subparsers.add_parser("plot", help="Compare posterior chains against a reference chain.")
    plot_parser.add_argument("--reference-chain", "--standard-chain", dest="reference_chain", type=Path, required=True, help="Reference/standard chain .npy file.")
    plot_parser.add_argument("--reference-label", "--standard-label", dest="reference_label", default="reference")
    plot_parser.add_argument("--comparison-chain", nargs="+", type=Path, required=True, help="One or more comparison chain .npy files.")
    plot_parser.add_argument("--comparison-label", nargs="+", default=None, help="Labels for comparison chains; must match --comparison-chain length.")
    plot_parser.add_argument("--params", nargs="+", default=None, help="Parameters to include in the triangle plot.")
    plot_parser.add_argument("--text", default=None, help="Optional figure title/text.")
    plot_parser.add_argument("--output", type=Path, default=None, help="Output triangle plot filename.")
    plot_parser.add_argument("--burnin", type=float, default=0.5)
    plot_parser.add_argument("--thin", type=int, default=5)
    plot_parser.add_argument("--mode", default="bfnl", choices=PNG_MODES, help="Add derived parameters consistently with this fit mode.")
    plot_parser.add_argument("--reference-mode", choices=PNG_MODES, default=None, help="Mode for the reference chain; defaults to --mode.")
    plot_parser.add_argument("--comparison-mode", nargs="+", choices=PNG_MODES, default=None, help="Mode(s) for comparison chains.")
    plot_parser.add_argument("--reference-zsnaps", default=None, help="Comma-separated redshifts for the reference chain, e.g. 0.5,0.8,1.1.")
    plot_parser.add_argument("--comparison-zsnaps", nargs="+", default=None, help="Comma-separated redshift sets for comparison chains.")
    plot_parser.add_argument("--tracer", default="LRG", choices=["LRG"])
    plot_parser.add_argument("--zsnaps", nargs="+", type=float, default=DEFAULT_ZS)
    plot_parser.set_defaults(func=command_plot)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
