#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
BAOCOV_FILE = ROOT.parent / "PFS_mock-tests" / "baocov.py"
PK_DIR = ROOT / "bphi_prior" / "pk"
COV_DIR = ROOT / "bphi_prior" / "cov"
PNG_MOCK_DIR = ROOT / "mocks_lhc_PNG"
BPHI_STATS_FILE = ROOT / "backup" / "bphi_AbacusHOD_LRG.txt"
BOXSIZE = 2000.0
DEFAULT_NTHREAD = 32
DEFAULT_ZS = [0.5, 0.8, 1.1]
PNG_MODES = ("bphi", "b-p", "bfnl")
PRIORS_FLAT = {"bphi": 10, "fnl_loc": 300}
BPHI_BFNL_PRIOR_NGRID = 4096
SIMTYPES = ("SU", "PNG")
TRUE_FNL_LOC_BY_COSMO = {
    "000": 0.0,
    "300": 30.0,
    "302": -30.0,
    "303": 100.0,
    "304": -100.0,
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


def default_config_path(zsnap, cosmo="000"):
    return ROOT / "HOD_config" / f"abacushod_z{ztag(zsnap)}_c{cosmo}.yaml"


def su_text_pk_path(tracer, zsnap, cosmo="000"):
    return PK_DIR / f"pk_{tracer}_z{ztag(zsnap)}_c{cosmo}.txt_pypower"


def png_mock_pk_path(tracer, zsnap, cosmo="000"):
    return PNG_MOCK_DIR / f"{tracer}_z{ztag(zsnap)}_c{cosmo}_default_pk_RSD.npy"


def default_pk_path(tracer, zsnap, cosmo="000", simtype="SU"):
    if simtype == "SU":
        return su_text_pk_path(tracer, zsnap, cosmo=cosmo)
    if simtype == "PNG":
        return png_mock_pk_path(tracer, zsnap, cosmo=cosmo)
    raise ValueError(f"Unknown simtype '{simtype}'. Choices are {SIMTYPES}.")


def default_cov_path(tracer, zsnap, cosmo="000", simtype="SU"):
    if simtype == "SU":
        return COV_DIR / f"cov_{tracer}_z{ztag(zsnap)}_c{cosmo}.npy"
    if simtype == "PNG":
        return COV_DIR / f"cov_PNG_{tracer}_z{ztag(zsnap)}_c{cosmo}.npy"
    raise ValueError(f"Unknown simtype '{simtype}'. Choices are {SIMTYPES}.")


def load_yaml(path):
    import yaml

    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_abacushod(tracer, zsnap, cosmo="000", nthread=DEFAULT_NTHREAD):
    from abacusnbody.hod.abacus_hod import AbacusHOD

    config = load_yaml(default_config_path(zsnap, cosmo=cosmo))
    sim_params = config.get("sim_params", {})
    hod_params = config.get("HOD_params", {})
    clustering_params = config.get("clustering_params", {})
    tracer_flags = hod_params.setdefault("tracer_flags", {})
    tracer_flags["LRG"] = tracer == "LRG"
    tracer_flags["ELG"] = tracer == "ELG"
    tracer_flags["QSO"] = tracer == "QSO"
    hod_params["want_rsd"] = True
    ball = AbacusHOD(sim_params, hod_params, clustering_params)
    mock = ball.run_hod(
        tracers=ball.tracers,
        want_rsd=ball.want_rsd,
        Nthread=int(config.get("nthread", nthread)),
        verbose=True,
        write_to_disk=False,
    )
    return mock


def compute_pk_from_mock(mock, tracer, kmin=0.0, kmax=0.1, dk=0.002, nmesh=256):
    from pypower import CatalogFFTPower

    data_positions = np.array(
        [
            mock[tracer]["x"] % BOXSIZE,
            mock[tracer]["y"] % BOXSIZE,
            mock[tracer]["z"] % BOXSIZE,
        ]
    ).T
    result = CatalogFFTPower(
        data_positions1=data_positions,
        position_type="pos",
        resampler="tsc",
        interlacing=3,
        edges={"min": kmin, "max": kmax, "step": dk},
        boxsize=BOXSIZE,
        nmesh=nmesh,
        ells=(0,),
        los="z",
        mpiroot=0,
    )
    return result


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
    pk_path = default_pk_path(tracer, zsnap, cosmo=cosmo, simtype=simtype)
    if simtype == "SU":
        return load_su_text_pk(pk_path)
    return load_png_rebinned_pk(pk_path)


def compute_thecov_gaussian_covariance(pk_data, kmin=0.0, kmax=0.1, dk=0.002):
    import thecov.covariance
    import thecov.geometry

    k = np.asarray(pk_data["k"], dtype="f8")
    pk0 = np.asarray(pk_data["pk0"], dtype="f8")
    expected_nbins = int(np.rint((kmax - kmin) / dk))

    if pk0.size != expected_nbins:
        mask = (k >= kmin) & (k < kmax)
        if np.count_nonzero(mask) != expected_nbins:
            mask = (k >= kmin) & (k <= kmax)
        pk0 = pk0[mask]

    if pk0.size != expected_nbins:
        raise ValueError(
            f"P(k) bin count mismatch for covariance: expected {expected_nbins} bins from "
            f"kmin={kmin}, kmax={kmax}, dk={dk}, but found {pk0.size} bins in the loaded spectrum."
        )

    geometry = thecov.geometry.BoxGeometry(
        volume=float(pk_data["boxsize"]) ** 3,
        nbar=float(pk_data["nbar"]),
    )
    gaussian = thecov.covariance.GaussianCovariance(geometry)
    gaussian.set_kbins(kmin, kmax, dk)
    gaussian.set_galaxy_pk_multipole(pk0, 0, has_shotnoise=False)
    gaussian.compute_covariance()
    return np.asarray(gaussian.get_ell_cov(0, 0).cov, dtype="f8")


def save_covariance(tracer, zsnap, covariance, cosmo="000", simtype="SU"):
    ensure_dir(COV_DIR)
    output = default_cov_path(tracer, zsnap, cosmo=cosmo, simtype=simtype)
    np.save(output, covariance)
    return output


def fit_prefix(tracer, zsnaps, mode, simtype, cosmo, use_bphi_prior):
    prior_tag = "with_bphi_prior" if use_bphi_prior else "no_bphi_prior"
    ztag_set = zset_tag(zsnaps)
    return FIT_DIR / f"png_fit_{simtype}_{tracer}_z{ztag_set}_c{cosmo}_{mode}_{prior_tag}"


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


def point_param_value(point, params, basename):
    if point is None:
        return None
    key = find_param_key(params, basename)
    if key is None:
        return None
    return scalar_float(point.get(key))


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


def triangle_params_with_bfnl_loc(chain, params, mode):
    if mode not in ("bphi", "b-p"):
        return params
    bfnl_params = chain.params(basename="bfnl_loc")
    if not bfnl_params:
        return params
    return params + bfnl_params


def overlay_truth_on_triangle(g, params, basename, value, color="0.25", linestyle="-.", label=None):
    key = find_param_key(params, basename)
    if key is None or value is None:
        return
    axes = g.subplots
    legend_done = False
    for i, param_i in enumerate(params):
        if str(param_i) == key:
            this_label = label if not legend_done else None
            axes[i, i].axvline(value, color=color, linestyle=linestyle, linewidth=1.0, alpha=0.9, label=this_label)
            legend_done = True
        for j in range(i):
            ax = axes[i, j]
            if str(params[j]) == key:
                this_label = label if not legend_done else None
                ax.axvline(value, color=color, linestyle=linestyle, linewidth=1.0, alpha=0.9, label=this_label)
                legend_done = True
            if str(param_i) == key:
                this_label = label if not legend_done else None
                ax.axhline(value, color=color, linestyle=linestyle, linewidth=1.0, alpha=0.9, label=this_label)
                legend_done = True


def overlay_truth_values_on_triangle(g, params, values, color="black", linestyle="-", linewidth=1.1):
    if not values:
        return
    axes = g.subplots
    values = {str(name): scalar_float(value) for name, value in values.items()}
    values = {name: value for name, value in values.items() if value is not None}
    for i, param_i in enumerate(params):
        name_i = str(param_i)
        value_i = values.get(name_i)
        if value_i is not None:
            axes[i, i].axvline(value_i, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.95)
        for j in range(i):
            ax = axes[i, j]
            name_j = str(params[j])
            value_j = values.get(name_j)
            if value_j is not None:
                ax.axvline(value_j, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.95)
            if value_i is not None:
                ax.axhline(value_i, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.95)


def annotate_joint_triangle_map(g, params, entries, true_fnl_loc):
    return

def overlay_markers_on_triangle(g, params, point, color, marker, label=None, diagonal_ls="--", size=48):
    if point is None:
        return
    axes = g.subplots
    legend_done = False
    for i, param_i in enumerate(params):
        value_i = scalar_float(point.get(str(param_i)))
        if value_i is None:
            continue
        ax_diag = axes[i, i]
        ax_diag.axvline(value_i, color=color, linestyle=diagonal_ls, linewidth=1.0, alpha=0.9)
        this_label = label if not legend_done else None
        ax_diag.scatter(
            [value_i],
            [0.92],
            transform=ax_diag.get_xaxis_transform(),
            color=color,
            marker=marker,
            s=size,
            zorder=10,
            label=this_label,
            edgecolors="black" if marker not in [".", ",", "|", "_"] else None,
            linewidths=0.4,
        )
        legend_done = True
        for j in range(i):
            value_j = scalar_float(point.get(str(params[j])))
            if value_j is None:
                continue
            ax = axes[i, j]
            this_label = label if not legend_done else None
            ax.scatter(
                value_j,
                value_i,
                color=color,
                marker=marker,
                s=size,
                zorder=10,
                label=this_label,
                edgecolors="black" if marker not in [".", ",", "|", "_"] else None,
                linewidths=0.4,
            )
            legend_done = True


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


def maybe_plot_joint_triangle(tracer, zsnaps, mode, simtype, cosmo):
    from desilike.samples import Chain, Profiles, plotting

    bphi_stats = load_bphi_stats()
    entries = []
    for use_bphi_prior, color, ls, label in [
        (False, "C1", "--", "No bphi prior"),
        (True, "C0", "-", "With bphi prior"),
    ]:
        prefix = fit_prefix(tracer, zsnaps, mode, simtype, cosmo, use_bphi_prior)
        chain_fn = Path(f"{prefix}_chain.npy")
        if not chain_fn.exists():
            continue
        burnin, thin = joint_triangle_chain_selection(cosmo, use_bphi_prior)
        chain = Chain.load(chain_fn).remove_burnin(burnin)[::thin]
        if mode in ("bphi", "b-p"):
            add_bfnl_loc_to_chain(chain, mode=mode)
        chain_map = point_to_float_dict(chain.choice(index="argmax", input=True))
        chain_map = add_bfnl_loc_to_point(chain_map, chain.params(), mode=mode)
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
            marker="o",
            diagonal_ls="--",
            size=72,
        )
        overlay_markers_on_triangle(
            g,
            params,
            entry["map"],
            color=entry["color"],
            marker="o",
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

    output = FIT_DIR / f"png_fit_{simtype}_{tracer}_z{zset_tag(zsnaps)}_c{cosmo}_{mode}_joint_triangle.png"
    g.export(str(output))
    return output


def load_bphi_stats(path=BPHI_STATS_FILE):
    stats = {}
    b1_stats = {}
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tracer, zsnap, _, _, _, bphi = parts[:6]
            zsnap = float(zsnap)
            key = (tracer, zsnap)
            bphi = float(bphi)
            stats.setdefault(key, []).append(bphi)
            if len(parts) >= 7:
                b1 = float(parts[6])
                if np.isfinite(b1):
                    b1_stats.setdefault(key, []).append(b1)
    summary = {}
    for key, values in stats.items():
        arr = np.asarray(values, dtype="f8")
        item = {
            "mean": float(arr[0]),
            "std": float(np.std(arr)),
            "n": int(arr.size),
        }
        if key in b1_stats:
            b1 = np.asarray(b1_stats[key], dtype="f8")
            item["b1"] = float(b1[0])
            item["b1_std"] = float(np.std(b1))
            item["b1_n"] = int(b1.size)
        summary[key] = item
    return summary


def triangle_truth_values(params, tracer, zsnaps, cosmo, bphi_stats=None):
    bphi_stats = load_bphi_stats() if bphi_stats is None else bphi_stats
    true_fnl_loc = true_fnl_loc_for_cosmo(cosmo)
    truth = {}
    names = {str(param) for param in params}
    for zsnap in zsnaps:
        label = tracer_z_label(tracer, zsnap)
        summary = bphi_stats.get((tracer, zsnap), {})
        if true_fnl_loc is not None and "mean" in summary:
            bfnl_name = f"{label}.bfnl_loc"
            if bfnl_name in names:
                truth[bfnl_name] = summary["mean"] * true_fnl_loc
        if "b1" in summary:
            b1_name = f"{label}.b1"
            if b1_name in names:
                truth[b1_name] = summary["b1"]
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
):
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles

    if use_bphi_prior and mode == "b-p":
        raise ValueError("bphi priors can only be applied when mode='bphi' or 'bfnl'.")
    likelihood_kmin = max(kmin, fundamental_k())

    if mode == "bfnl":
        patch_png_bfnl_calculate(PNGTracerPowerSpectrumMultipoles)
    
    observables = []
    covariance_blocks = []
    bphi_stats = load_bphi_stats()
    bfnl_prior_param_names = []
    bfnl_prior_locs = []
    bfnl_prior_scales = []
    for zsnap in zsnaps:
        bphi_summary = bphi_stats[(tracer, zsnap)]
        pk_data = load_pk(tracer, zsnap, cosmo=cosmo, simtype=simtype)
        covariance = np.load(default_cov_path(tracer, zsnap, cosmo=cosmo, simtype=simtype))
        k = np.asarray(pk_data["k"], dtype="f8")
        pk0 = np.asarray(pk_data["pk0"], dtype="f8")
        covariance = np.asarray(covariance, dtype="f8")
        if covariance.shape != (k.size, k.size):
            raise ValueError(
                f"Covariance shape mismatch for {simtype} {tracer} z={zsnap:.3f} c{cosmo}: "
                f"covariance has shape {covariance.shape}, but P(k) has {k.size} bins."
            )
        template = FixedPowerSpectrumTemplate(z=zsnap, fiducial="DESI")
        theory = PNGTracerPowerSpectrumMultipoles(template=template, mode=mode, ells=(0,))
        label = f"{tracer.lower()}z{ztag(zsnap).replace('.', 'p')}"
        kmask = (k >= likelihood_kmin) & (k < kmax)
        if not np.any(kmask):
            raise ValueError(
                f"No P(k) bins selected for {simtype} {tracer} z={zsnap:.3f} c{cosmo}: "
                f"requested {likelihood_kmin:g} <= k < {kmax:g}; available range is "
                f"{np.nanmin(k):g} <= k <= {np.nanmax(k):g}."
            )
        k_fit = k[kmask]
        pk0_fit = pk0[kmask]
        covariance_fit = covariance[np.ix_(kmask, kmask)]

        theory.init.params["b1"].update(namespace=label)
        theory.init.params["sigmas"].update(namespace=label, fixed=True, value=0.0)
        theory.init.params["sn0"].update(namespace=label)
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
            theory.init.params["p"].update(namespace=label, fixed=True, value=1.0)

        observable = TracerPowerSpectrumMultipolesObservable(
            data=pk0_fit,
            covariance=covariance_fit,
            k=k_fit,
            ells=(0,),
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
    run_sampler=False,
):
    # Respect user environment/PYTHONPATH by default.
    # Only override desilike import path when DESILIKE_LOCAL_PATH is explicitly set.
    local_desilike = os.environ.get("DESILIKE_LOCAL_PATH")
    if local_desilike and os.path.isdir(local_desilike) and local_desilike not in sys.path:
        sys.path.insert(0, local_desilike)

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    from desilike import setup_logging
    from desilike.profilers import MinuitProfiler

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
    )

    ensure_dir(FIT_DIR)
    prefix = fit_prefix(tracer, zsnaps, mode, simtype, cosmo, use_bphi_prior)

    profiler = MinuitProfiler(likelihood, seed=42, save_fn=f"{prefix}_profiles.npy")
    profiles = profiler.maximize(niterations=1)
    profiles = profiler.profiles if profiles is None else profiles
    profiler_bestfit = profiles.bestfit.choice(input=True)
    likelihood(**profiler_bestfit)
    save_point(f"{prefix}_profiler_bestfit.npz", profiler_bestfit)

    for zsnap, observable in zip(zsnaps, observables):
        observable.plot(fn=f"{prefix}_z{ztag(zsnap)}_bestfit.png")

    fnl_grid = np.linspace(-flat_prior_abs("fnl_loc"), flat_prior_abs("fnl_loc"), 121)
    if mode == "bphi":
        fnl_params = [str(param) for param in profiler.varied_params if getattr(param, "basename", None) == "fnl_loc"]
        if fnl_params:
            profiles = profiler.profile(params=fnl_params, grid=[fnl_grid] * len(fnl_params))
    elif mode == "bfnl":
        bfnl_grid = np.linspace(-bfnl_flat_limit(), bfnl_flat_limit(), 121)
        bfnl_params = [str(param) for param in profiler.varied_params if getattr(param, "basename", None) == "bfnl_loc"]
        if bfnl_params:
            profiles = profiler.profile(params=bfnl_params, grid=[bfnl_grid] * len(bfnl_params))
    profiles.to_stats(tablefmt="pretty", fn=f"{prefix}_profiler_stats.txt")

    if run_sampler:
        from desilike.samplers import ZeusSampler
        from desilike.samples import Chain, plotting

        chain_fn = f"{prefix}_chain.npy"
        if not Path(chain_fn).exists():
            sampler = ZeusSampler(likelihood, seed=42, save_fn=f"{prefix}_chain", nwalkers=60)
            sampler.run(check={"max_eigen_gr": 0.05})
            chain = sampler.chains[0]
        else:
            chain = Chain.load(chain_fn)
        burnin, thin = single_triangle_chain_selection(cosmo, use_bphi_prior)
        chain = chain.remove_burnin(burnin)[::thin]
        if mode in ("bphi", "b-p"):
            add_bfnl_loc_to_chain(chain, mode=mode)
        chain_map = chain.choice(index="argmax", input=True)
        chain_map = add_bfnl_loc_to_point(point_to_float_dict(chain_map), chain.params(), mode=mode)
        save_point(f"{prefix}_chain_map.npz", chain_map)
        chain.to_stats(tablefmt="pretty", fn=f"{prefix}_chain_stats.txt")
        params = chain.params(varied=True, input=True)
        params = triangle_params_with_bfnl_loc(chain, params, mode)
        truth_values = triangle_truth_values(params, tracer, zsnaps, cosmo)
        g = plotting.plot_triangle(chain, params=params, filled=False)
        profiler_bestfit = add_bfnl_loc_to_point(point_to_float_dict(profiler_bestfit), chain.params(), mode=mode)
        overlay_markers_on_triangle(g, params, profiler_bestfit, color="C0", marker="o",
                                    diagonal_ls="--", size=64)
        overlay_markers_on_triangle(g, params, chain_map, color="C0", marker="o",
                                    diagonal_ls=":", size=36)
        overlay_truth_on_triangle(
            g,
            params,
            "fnl_loc",
            true_fnl_loc_for_cosmo(cosmo),
            color="black",
            linestyle="-",
        )
        overlay_truth_values_on_triangle(g, params, truth_values)
        g.export(f"{prefix}_triangle.png")
        joint_output = maybe_plot_joint_triangle(tracer, zsnaps, mode, simtype, cosmo)
        if joint_output is not None:
            print(f"Saved {joint_output}")


def command_pk(args):
    for zsnap in args.zsnaps:
        input_pk = default_pk_path(args.tracer, zsnap, cosmo=args.cosmo, simtype=args.simtype)
        if not input_pk.exists():
            raise FileNotFoundError(f"Expected {args.simtype} P(k) file not found: {input_pk}")
        print(f"Using existing {args.simtype} P(k) file {input_pk}")


def command_cov(args):
    ensure_dir(COV_DIR)
    for zsnap in args.zsnaps:
        pk_data = load_pk(args.tracer, zsnap, cosmo=args.cosmo, simtype=args.simtype)
        covariance = compute_thecov_gaussian_covariance(
            pk_data=pk_data,
            kmin=args.kmin,
            kmax=args.kmax,
            dk=args.dk,
        )
        output = save_covariance(args.tracer, zsnap, covariance, cosmo=args.cosmo, simtype=args.simtype)
        print(f"Saved covariance to {output}")


def command_fit(args):
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
        run_sampler=args.run_sampler,
    )


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
    pk_parser.add_argument("--nmesh", type=int, default=256)
    pk_parser.add_argument("--nthread", type=int, default=DEFAULT_NTHREAD)
    pk_parser.set_defaults(func=command_pk)

    cov_parser = subparsers.add_parser("cov", help="Compute Gaussian P(k) covariance with thecov from external SU/PNG P(k).")
    cov_parser.add_argument("--tracer", default="LRG", choices=["LRG"])
    cov_parser.add_argument("--zsnaps", nargs="+", type=float, default=DEFAULT_ZS)
    cov_parser.add_argument("--simtype", default="SU", choices=SIMTYPES)
    cov_parser.add_argument("--cosmo", default="000")
    cov_parser.add_argument("--kmin", type=float, default=0.0)
    cov_parser.add_argument("--kmax", type=float, default=0.1)
    cov_parser.add_argument("--dk", type=float, default=0.002)
    cov_parser.set_defaults(func=command_cov)

    fit_parser = subparsers.add_parser("fit", help="Run joint PNG fits with or without bphi priors.")
    fit_parser.add_argument("--tracer", default="LRG", choices=["LRG"])
    fit_parser.add_argument("--zsnaps", nargs="+", type=float, default=DEFAULT_ZS)
    fit_parser.add_argument("--simtype", default="SU", choices=SIMTYPES)
    fit_parser.add_argument("--cosmo", default="000")
    fit_parser.add_argument("--kmin", type=float, default=0.0)
    fit_parser.add_argument("--kmax", type=float, default=0.1)
    fit_parser.add_argument("--dk", type=float, default=0.002)
    fit_parser.add_argument("--mode", default="bphi", choices=PNG_MODES)
    fit_parser.add_argument("--with-bphi-prior", action="store_true")
    fit_parser.add_argument("--run-sampler", action="store_true")
    fit_parser.set_defaults(func=command_fit)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
