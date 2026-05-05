#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
BAOCOV_FILE = ROOT.parent / "PFS_mock-tests" / "baocov.py"
PK_DIR = ROOT / "bphi_prior" / "pk"
COV_DIR = ROOT / "bphi_prior" / "cov"
PNG_MOCK_DIR = ROOT / "mocks_lhc_PNG"
BPHI_STATS_FILE = ROOT / "bphi_AbacusHOD_LRG.txt"
BOXSIZE = 2000.0
DEFAULT_NTHREAD = 32
DEFAULT_ZS = [0.5, 0.8, 1.1]
PNG_MODES = ("bphi", "b-p", "bfnl")
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
    tag = str(cosmo).removeprefix("c")
    return TRUE_FNL_LOC_BY_COSMO.get(tag)


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


def annotate_joint_triangle_map(g, params, entries, true_fnl_loc):
    lines = []
    if true_fnl_loc is not None:
        lines.append(f"truth fnl_loc = {true_fnl_loc:g}")
    for entry in entries:
        ml_value = point_param_value(entry["bestfit"], params, "fnl_loc")
        map_value = point_param_value(entry["map"], params, "fnl_loc")
        if ml_value is not None:
            lines.append(f'{entry["label"]} ML fnl_loc = {ml_value:.2f}')
        if map_value is not None:
            lines.append(f'{entry["label"]} MAP fnl_loc = {map_value:.2f}')
    if not lines:
        return
    ax = np.ravel(g.subplots)[0]
    r"""
    ax.text(
        0.98,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.85},
    )
    """

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


def collect_legend_handles(g):
    handles, labels = [], []
    seen = set()
    for ax in np.ravel(g.subplots):
        if ax is None:
            continue
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label and label not in seen:
                handles.append(handle)
                labels.append(label)
                seen.add(label)
    return handles, labels


def maybe_plot_joint_triangle(tracer, zsnaps, mode, simtype, cosmo):
    from desilike.samples import Chain, Profiles, plotting

    entries = []
    for use_bphi_prior, color, ls, label in [
        (False, "C1", "--", "No bphi prior"),
        (True, "C0", "-", "With bphi prior"),
    ]:
        prefix = fit_prefix(tracer, zsnaps, mode, simtype, cosmo, use_bphi_prior)
        chain_fn = Path(f"{prefix}_chain.npy")
        if not chain_fn.exists():
            continue
        chain = Chain.load(chain_fn).remove_burnin(0.5)[::5]
        chain_map = point_to_float_dict(chain.choice(index="argmax", input=True))
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
    true_fnl_loc = true_fnl_loc_for_cosmo(cosmo)
    g = plotting.plot_triangle(
        chains,
        params=params,
        labels=labels,
        contour_colors=colors,
        contour_ls=linestyles,
        filled=False,
    )
    for entry in entries:
        ml_fnl_loc = point_param_value(entry["bestfit"], params, "fnl_loc")
        ml_label = f'{entry["label"]} maximum likelihood'
        if ml_fnl_loc is not None:
            ml_label += f" (fnl_loc={ml_fnl_loc:.2f})"
        overlay_markers_on_triangle(
            g,
            params,
            entry["bestfit"],
            color=entry["color"],
            marker="*",
            label=ml_label,
            diagonal_ls="--",
            size=72,
        )
        map_fnl_loc = point_param_value(entry["map"], params, "fnl_loc")
        map_label = f'{entry["label"]} chain MAP'
        if map_fnl_loc is not None:
            map_label += f" (fnl_loc={map_fnl_loc:.2f})"
        overlay_markers_on_triangle(
            g,
            params,
            entry["map"],
            color=entry["color"],
            marker="o",
            label=map_label,
            diagonal_ls=":",
            size=42,
        )
    overlay_truth_on_triangle(
        g,
        params,
        "fnl_loc",
        true_fnl_loc,
        label=f"truth fnl_loc={true_fnl_loc:g}" if true_fnl_loc is not None else None,
    )
    annotate_joint_triangle_map(g, params, entries, true_fnl_loc)

    legend_ax = g.subplots[1, 0] if g.subplots.shape[0] > 1 else g.subplots[0, 0]
    handles, labels = collect_legend_handles(g)
    if handles:
        legend_ax.legend(handles, labels, loc="best", fontsize=8)

    output = FIT_DIR / f"png_fit_{simtype}_{tracer}_z{zset_tag(zsnaps)}_c{cosmo}_{mode}_joint_triangle.png"
    g.export(str(output))
    return output


def load_bphi_stats(path=BPHI_STATS_FILE):
    stats = {}
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tracer, zsnap, _, _, _, bphi = line.split()[:6]
            zsnap = float(zsnap)
            stats.setdefault((tracer, zsnap), []).append(float(bphi))
    summary = {}
    for key, values in stats.items():
        arr = np.asarray(values, dtype="f8")
        summary[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "n": int(arr.size),
        }
    return summary

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

    if use_bphi_prior and mode != "bphi":
        raise ValueError("bphi priors can only be applied when mode='bphi'.")
    likelihood_kmin = max(kmin, fundamental_k())

    observables = []
    covariance_blocks = []
    bphi_stats = load_bphi_stats()
    for zsnap in zsnaps:
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
        if use_bphi_prior and mode == "bphi":
            bphi_summary = bphi_stats[(tracer, zsnap)]
            theory.init.params[f"{label}.bphi"].update(
                prior={"dist": "norm", "loc": bphi_summary["mean"], "scale": bphi_summary["std"]},
                ref={"dist": "norm", "loc": bphi_summary["mean"], "scale": max(0.2 * bphi_summary["std"], 0.05)},
            ) #value=bphi_summary["mean"],
        elif mode == "bphi":
            theory.init.params[f"{label}.bphi"].update(prior={"limits": [-10.0, 20.0]})

        observable = TracerPowerSpectrumMultipolesObservable(
            data=pk0_fit,
            covariance=covariance_fit,
            k=k_fit,
            ells=(0,),
            theory=theory,
            name=f"{label}_pk",
        )
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

    with open(f"{prefix}_bestfit.txt", "w", encoding="utf-8") as stream:
        stream.write(str(profiles.bestfit))
        stream.write("\n")

    fnl_grid = np.linspace(-300.0, 300.0, 121)
    profiles = profiler.profile(params=["fnl_loc"], grid=fnl_grid)
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
        chain = chain.remove_burnin(0.5)[::5]
        chain_map = chain.choice(index="argmax", input=True)
        save_point(f"{prefix}_chain_map.npz", chain_map)
        chain.to_stats(tablefmt="pretty", fn=f"{prefix}_chain_stats.txt")
        params = chain.params(varied=True, input=True)
        g = plotting.plot_triangle(chain, params=params, filled=False)
        overlay_markers_on_triangle(g, params, profiler_bestfit, color="C0", marker="X",
                                    label="Profiler best-fit", diagonal_ls="--")
        overlay_markers_on_triangle(g, params, chain_map, color="C0", marker="o",
                                    label="Chain MAP", diagonal_ls=":")
        single_legend_ax = g.subplots[1, 0] if g.subplots.shape[0] > 1 else g.subplots[0, 0]
        handles, labels = single_legend_ax.get_legend_handles_labels()
        if handles:
            single_legend_ax.legend(handles, labels, loc="best", fontsize=8)
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
