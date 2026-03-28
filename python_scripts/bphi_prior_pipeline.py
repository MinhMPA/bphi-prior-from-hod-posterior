#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
BAOCOV_FILE = ROOT.parent / "PFS_mock-tests" / "baocov.py"
PK_DIR = ROOT / "bphi_prior" / "pk"
COV_DIR = ROOT / "bphi_prior" / "cov"
FIT_DIR = ROOT / "bphi_prior" / "fit"
BPHI_STATS_FILE = ROOT / "bphi_AbacusHOD_LRG.txt"
BOXSIZE = 2000.0
DEFAULT_NTHREAD = 32
DEFAULT_ZS = [0.5, 0.8, 1.1]
PNG_MODES = ("bphi", "b-p", "bfnl")


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def ztag(zsnap):
    return f"{zsnap:.3f}"


def fundamental_k():
    return 2.0 * np.pi / BOXSIZE


def default_config_path(zsnap, cosmo="000"):
    return ROOT / "HOD_config" / f"abacushod_z{ztag(zsnap)}_c{cosmo}.yaml"


def default_pk_path(tracer, zsnap, cosmo="000"):
    return PK_DIR / f"pk_{tracer}_z{ztag(zsnap)}_c{cosmo}.npy"


def default_cov_path(tracer, zsnap, cosmo="000"):
    return COV_DIR / f"cov_{tracer}_z{ztag(zsnap)}_c{cosmo}.npy"


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

def load_pk(tracer, zsnap, cosmo="000"):
    from pypower import MeshFFTPower, PowerSpectrumMultipoles

    state = np.load(default_pk_path(tracer, zsnap, cosmo=cosmo), allow_pickle=True)[()]
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
        "nbar": data_size / BOXSIZE**3,
        "boxsize": BOXSIZE,
    }


def compute_thecov_gaussian_covariance(pk_data, kmin=0.0, kmax=0.1, dk=0.002):
    import thecov.covariance
    import thecov.geometry

    geometry = thecov.geometry.BoxGeometry(
        volume=float(pk_data["boxsize"]) ** 3,
        nbar=float(pk_data["nbar"]),
    )
    gaussian = thecov.covariance.GaussianCovariance(geometry)
    gaussian.set_kbins(kmin, kmax, dk)
    gaussian.set_galaxy_pk_multipole(np.asarray(pk_data["pk0"], dtype="f8"), 0, has_shotnoise=False)
    gaussian.compute_covariance()
    return np.asarray(gaussian.get_ell_cov(0, 0).cov, dtype="f8")


def save_covariance(tracer, zsnap, covariance, cosmo="000"):
    ensure_dir(COV_DIR)
    output = default_cov_path(tracer, zsnap, cosmo=cosmo)
    np.save(output, covariance)
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
    bphi_stats = load_bphi_stats()
    for zsnap in zsnaps:
        pk_data = load_pk(tracer, zsnap, cosmo=cosmo)
        covariance = np.load(default_cov_path(tracer, zsnap, cosmo=cosmo))
        template = FixedPowerSpectrumTemplate(z=zsnap, fiducial="DESI")
        theory = PNGTracerPowerSpectrumMultipoles(template=template, mode=mode, ells=(0,))
        label = f"{tracer.lower()}z{ztag(zsnap).replace('.', 'p')}"
        kmask = np.asarray(pk_data["k"], dtype="f8") >= likelihood_kmin

        theory.init.params["b1"].update(namespace=label)
        theory.init.params["sigmas"].update(namespace=label, fixed=True, value=0.0)
        theory.init.params["sn0"].update(namespace=label, fixed=True, value=0.0)
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
            data=np.asarray(pk_data["pk0"], dtype="f8")[kmask],
            covariance=covariance[np.ix_(kmask, kmask)],
            klim={0: [likelihood_kmin, kmax, dk]},
            k=np.asarray(pk_data["k"], dtype="f8")[kmask],
            ells=(0,),
            theory=theory,
        )
        observables.append(observable)

    likelihood = ObservablesGaussianLikelihood(observables=observables)
    return likelihood, observables


def run_fit(
    tracer,
    zsnaps,
    use_bphi_prior,
    mode="bphi",
    cosmo="000",
    kmin=0.005,
    kmax=0.1,
    dk=0.002,
    run_sampler=False,
):
    from desilike import setup_logging
    from desilike.profilers import MinuitProfiler

    setup_logging("info")
    likelihood, observables = build_joint_png_likelihood(
        tracer=tracer,
        zsnaps=zsnaps,
        use_bphi_prior=use_bphi_prior,
        mode=mode,
        cosmo=cosmo,
        kmin=kmin,
        kmax=kmax,
        dk=dk,
    )

    ensure_dir(FIT_DIR)
    tag = "with_bphi_prior" if use_bphi_prior else "no_bphi_prior"
    prefix = FIT_DIR / f"png_fit_{tracer}_{mode}_{tag}"

    profiler = MinuitProfiler(likelihood, seed=42, save_fn=f"{prefix}_profiles.npy")
    profiles = profiler.maximize(niterations=1)
    profiles = profiler.profiles if profiles is None else profiles
    likelihood(**profiles.bestfit.choice(input=True))

    for zsnap, observable in zip(zsnaps, observables):
        observable.plot(fn=f"{prefix}_z{ztag(zsnap)}_bestfit.png")

    with open(f"{prefix}_bestfit.txt", "w", encoding="utf-8") as stream:
        stream.write(str(profiles.bestfit))
        stream.write("\n")

    fnl_grid = np.linspace(-300.0, 300.0, 121)
    profiler.profile(params=["fnl_loc"], grid=fnl_grid)

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
        chain.to_stats(tablefmt="pretty", fn=f"{prefix}_chain_stats.txt")
        plotting.plot_triangle(chain, fn=f"{prefix}_triangle.png", dpi=100)


def command_pk(args):
    ensure_dir(PK_DIR)
    for zsnap in args.zsnaps:
        print(f"Generating {args.tracer} mock and P(k) at z={zsnap:.3f}")
        mock = load_abacushod(args.tracer, zsnap, cosmo=args.cosmo, nthread=args.nthread)
        payload = compute_pk_from_mock(
            mock=mock,
            tracer=args.tracer,
            kmin=args.kmin,
            kmax=args.kmax,
            dk=args.dk,
            nmesh=args.nmesh,
        )
        output_npy = default_pk_path(args.tracer, zsnap, cosmo="000")
        output_txt = output_npy.with_suffix(".txt_pypower")
        payload.save(output_npy)
        payload.poles.save_txt(output_txt, complex=False)
        print(f"Saved {output_npy} and {output_txt}")


def command_cov(args):
    ensure_dir(COV_DIR)
    for zsnap in args.zsnaps:
        pk_data = load_pk(args.tracer, zsnap, cosmo=args.cosmo)
        covariance = compute_thecov_gaussian_covariance(
            pk_data=pk_data,
            kmin=args.kmin,
            kmax=args.kmax,
            dk=args.dk,
        )
        output = save_covariance(args.tracer, zsnap, covariance, cosmo=args.cosmo)
        print(f"Saved covariance to {output}")


def command_fit(args):
    run_fit(
        tracer=args.tracer,
        zsnaps=args.zsnaps,
        use_bphi_prior=args.with_bphi_prior,
        mode=args.mode,
        cosmo=args.cosmo,
        kmin=args.kmin,
        kmax=args.kmax,
        dk=args.dk,
        run_sampler=args.run_sampler,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Pipeline for bphi-prior P(k), covariance and PNG fits.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pk_parser = subparsers.add_parser("pk", help="Generate LRG HOD mock P(k) from AbacusHOD config.")
    pk_parser.add_argument("--tracer", default="LRG", choices=["LRG"])
    pk_parser.add_argument("--zsnaps", nargs="+", type=float, default=DEFAULT_ZS)
    pk_parser.add_argument("--cosmo", default="000")
    pk_parser.add_argument("--kmin", type=float, default=0.0)
    pk_parser.add_argument("--kmax", type=float, default=0.1)
    pk_parser.add_argument("--dk", type=float, default=0.002)
    pk_parser.add_argument("--nmesh", type=int, default=256)
    pk_parser.add_argument("--nthread", type=int, default=DEFAULT_NTHREAD)
    pk_parser.set_defaults(func=command_pk)

    cov_parser = subparsers.add_parser("cov", help="Compute Gaussian P(k) covariance with thecov.")
    cov_parser.add_argument("--tracer", default="LRG", choices=["LRG"])
    cov_parser.add_argument("--zsnaps", nargs="+", type=float, default=DEFAULT_ZS)
    cov_parser.add_argument("--cosmo", default="000")
    cov_parser.add_argument("--kmin", type=float, default=0.0)
    cov_parser.add_argument("--kmax", type=float, default=0.1)
    cov_parser.add_argument("--dk", type=float, default=0.002)
    cov_parser.set_defaults(func=command_cov)

    fit_parser = subparsers.add_parser("fit", help="Run joint PNG fits with or without bphi priors.")
    fit_parser.add_argument("--tracer", default="LRG", choices=["LRG"])
    fit_parser.add_argument("--zsnaps", nargs="+", type=float, default=DEFAULT_ZS)
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
