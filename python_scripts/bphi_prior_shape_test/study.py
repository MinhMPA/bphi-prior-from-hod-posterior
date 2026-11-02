#!/usr/bin/env python3
"""Isolated prior-shape PNG fits; sampling must be launched by the user."""
import argparse
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np

from prior_models import HERE, KINDS, LABELS, SOURCE_RUN, freeze_json, load_models, plot_priors, prepare, sha256

ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / 'python_scripts'))
ZSETS = {"LRG1": [0.5], "LRG2": [0.8], "LRG3": [1.1], "joint": [0.5, 0.8, 1.1]}
PRIOR_MAPPING_VERSION = 2
DEFAULT_OUTPUT = Path(os.environ.get('BPHI_RESULTS_ROOT', ROOT / 'outputs')).expanduser() / 'bphi_prior_shape_test/runs/namespaced_v2'


def parameter_states(params):
    states = []
    for param in params:
        state = param.__getstate__()
        for key in ("prior", "ref"):
            state[key]["limits"] = [float(v) if np.isfinite(v) else None for v in state[key]["limits"]]
        states.append(state)
    return states


def comparison_settings(config):
    keys = ("prior_mapping_version", "cosmo", "zsnaps", "kmin", "kmax", "dk", "ells", "fnl_limits", "cov_cosmo", "cov_veff",
            "assembly_bias", "inputs", "sources")
    shared = {key: config[key] for key in keys}
    shared["parameters"] = [p for p in config["parameters"] if p["basename"] not in ("bphi_selector", "bphi_noise")]
    return shared


def check_likelihood(likelihood):
    import jax
    compiled = jax.jit(likelihood, static_argnames=["return_derived"])
    theories = [c for c in likelihood.runtime_info.pipeline.calculators if hasattr(c, "prior_model")]
    loglikes = []
    for offset in (0., .35):
        values = {p.name: np.atleast_1d(p.value) for p in likelihood.varied_params}
        values["fnl_loc"] = np.array([30.])
        for name in values:
            if name.endswith("bphi_selector"):
                values[name] = np.array([.125 + offset])
            elif name.endswith("bphi_noise"):
                values[name] = np.array([.7 + offset])
        expected, direct_derived = likelihood(values, return_derived=True)
        actual, jit_derived = compiled(values, return_derived=True)
        np.testing.assert_allclose(actual, expected, rtol=1e-10)
        if not np.all(np.isfinite(actual)):
            raise RuntimeError("The nonzero-fnl likelihood check was nonfinite")
        for theory in theories:
            names = {p.basename: p.name for p in theory.runtime_info.params}
            bphi = theory.prior_model.mapped(theory.prior_kind,
                    values.get(names["bphi_selector"], np.array([.5])),
                    values.get(names["bphi_noise"], np.array([0.])))
            for derived in (direct_derived, jit_derived):
                np.testing.assert_allclose(derived[names["bphi"]], bphi, rtol=1e-10)
        logprior = sum(float(p.prior(values[p.name][0])) for p in likelihood.varied_params)
        loglikes.append(float(np.asarray(actual).ravel()[0]) - logprior)
    if abs(loglikes[1] - loglikes[0]) < 1e-8:
        raise RuntimeError("Changing bphi latents did not change the nonzero-fnl likelihood")
    print("Namespaced bphi mapping and nonzero-fnl likelihood response passed, including JAX.", flush=True)


def theory_class():
    from desilike.theories.galaxy_clustering import PNGTracerPowerSpectrumMultipoles
    from desilike.jax import numpy as jnp

    class PriorShapePNG(PNGTracerPowerSpectrumMultipoles):
        config_fn = PNGTracerPowerSpectrumMultipoles
        # Each instance models one bin; calculate() expects parameter basenames.
        _calculate_with_namespace = False

        @classmethod
        def _params(cls, params, tracers=None, mode="bphi", prior_kind="gaussian"):
            params = super()._params(params, tracers=tracers, mode=mode)
            params["bphi"].update(derived=True, prior=None, ref=None)
            params.extend({
                "bphi_selector": dict(value=.5, prior={"limits": [0., 1.]},
                                      ref={"limits": [0., 1.]}, fixed=prior_kind == "gaussian"),
                "bphi_noise": dict(value=0., prior={"dist": "norm", "loc": 0., "scale": 1.},
                                   ref={"dist": "norm", "loc": 0., "scale": 1.}, fixed=prior_kind == "empirical"),
            })
            return params

        def initialize(self, *args, prior_model=None, prior_kind="gaussian", **kwargs):
            self.prior_model, self.prior_kind = prior_model, prior_kind
            super().initialize(*args, **kwargs)

        def calculate(self, bphi_selector=.5, bphi_noise=0., **kwargs):
            self.bphi = self.prior_model.mapped(self.prior_kind, bphi_selector, bphi_noise, xp=jnp)
            super().calculate(bphi=self.bphi, **kwargs)

    return PriorShapePNG


def build_likelihood(args, zs, cosmo, kind, models):
    import bphi_prior_pipeline as pipeline
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate
    from scipy.linalg import block_diag
    Theory = theory_class()
    observables, covariances, inputs = [], [], []
    for z in zs:
        data = pipeline.load_covariance("LRG", z, cosmo=cosmo, simtype="PNG", kmin=args.kmin,
                                        kmax=args.kmax, dk=.002, ells=(0, 2), cov_veff=True, cov_cosmo="000")
        theory = Theory(template=FixedPowerSpectrumTemplate(z=z, fiducial="DESI"), mode="bphi", ells=(0, 2),
                        prior_model=models[z], prior_kind=kind)
        label = pipeline.tracer_z_label("LRG", z)
        for name in ("b1", "sn0", "sigmas", "bphi", "bphi_selector", "bphi_noise"):
            theory.init.params[name].update(namespace=label)
        theory.init.params["fnl_loc"].update(prior={"limits": [-300., 300.]})
        observable = TracerPowerSpectrumMultipolesObservable(data=data["data"], covariance=data["covariance"],
                                                              k=[data["k"]] * 2, ells=(0, 2), theory=theory)
        observable.name = f"{label}_prior_shape"
        observables.append(observable)
        covariances.append(data["covariance"])
        paths = [pipeline.default_pk_path("LRG", z, cosmo=cosmo, simtype="PNG"),
                 pipeline.default_cov_path("LRG", z, cosmo="000", simtype="PNG", ells=(0, 2), cov_veff=True)]
        inputs.extend(dict(path=str(path), sha256=sha256(path)) for path in paths)
    likelihood = ObservablesGaussianLikelihood(observables=observables, covariance=block_diag(*covariances))
    assert likelihood.varied_params.names(basename="fnl_loc") == ["fnl_loc"], "Exactly one shared fnl_loc is required"
    print("Initial log posterior:", float(likelihood()), flush=True)
    return likelihood, inputs


def fit(args):
    if not args.check_only and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Launch sampling yourself inside an allocated interactive CPU-node shell")
    from desilike import setup_logging
    from desilike.samplers import ZeusSampler
    from desilike.theories.galaxy_clustering import PNGTracerPowerSpectrumMultipoles
    import bphi_prior_pipeline as pipeline
    setup_logging("info")
    if "directory" not in inspect.signature(ZeusSampler).parameters:
        raise RuntimeError("This driver requires the installed directory/n_walkers Zeus API; source ../environment.sh")
    models = load_models(args.output)
    for cosmo in args.cosmos:
        for zset in args.zsets:
            for kind in args.priors:
                directory = args.output / "fit" / f"c{cosmo}" / zset / kind / f"seed{args.seed}"
                print(f"c{cosmo} {zset} prior={kind}: {directory}", flush=True)
                likelihood, inputs = build_likelihood(args, ZSETS[zset], cosmo, kind, models)
                if args.nwalkers < 2 * len(likelihood.varied_params):
                    raise ValueError("Use at least twice as many walkers as varied parameters")
                config = dict(prior=kind, priors_sha256=sha256(args.output / "priors.json"),
                              prior_mapping_version=PRIOR_MAPPING_VERSION,
                              cosmo=cosmo, zsnaps=ZSETS[zset], kmin=args.kmin, kmax=args.kmax,
                              dk=.002, ells=[0, 2], fnl_limits=[-300, 300], cov_cosmo="000", cov_veff=True,
                              assembly_bias=False, nwalkers=args.nwalkers, seed=args.seed, inputs=inputs,
                              sources={str(p): sha256(p) for p in (Path(__file__), HERE / "prior_models.py",
                                       Path(pipeline.__file__), Path(inspect.getfile(PNGTracerPowerSpectrumMultipoles)))},
                              parameters=parameter_states(likelihood.all_params))
                json.dumps(config, allow_nan=False)
                if args.check_only:
                    check_likelihood(likelihood)
                    print("CHECK ONLY: no sampler started.", flush=True)
                    continue
                freeze_json(directory / "settings.json", config)
                chains = sorted(directory.glob("chain_*.npy"))
                if chains and (len(chains) != args.nwalkers or not (directory / "checks.npy").exists()):
                    raise RuntimeError(f"Incomplete sampler checkpoint in {directory}; inspect before resuming")
                print("Resuming saved walkers." if chains else "Starting new walkers.", flush=True)
                np.random.seed(args.seed)
                sampler = ZeusSampler(likelihood, n_walkers=args.nwalkers, rng=None if chains else args.seed,
                                      directory=directory)
                sampler.run(burn_in=.5, min_steps=args.min_steps, max_steps=args.max_steps, check_every=args.check_every,
                            save_every=args.check_every, checks_passed=2, gelman_rubin=1.01, ess=100)
                converged = len(sampler.checks) >= 2 and all(sampler.checks[-2:])
                report = dict(converged=bool(converged), steps_per_walker=len(sampler.chains[0]),
                              checks=[bool(v) for v in sampler.checks], burnin=.5,
                              note="Ensemble walkers are not independent chains; confirm stability with another seed")
                (directory / "status.json").write_text(json.dumps(report, indent=2) + "\n")
                print(report, flush=True)
                if not converged:
                    print("WARNING: convergence not established; increase --max-steps and resume this setting.", flush=True)


def compare(args):
    import csv
    import matplotlib.pyplot as plt
    from desilike.samples import Chain
    from bphi_prior_plot_fnl import fnl_samples_from_chain, summarize_samples
    from bphi_prior_pipeline import true_fnl_loc_for_cosmo
    from scipy.stats import gaussian_kde
    records = []
    plots, tables = args.output / "plots", args.output / "tables"
    plots.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    for cosmo in args.cosmos:
        columns = min(2, len(args.zsets))
        rows = (len(args.zsets) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.25 * rows), squeeze=False)
        for ax in list(axes.flat)[len(args.zsets):]:
            ax.set_visible(False)
        present = False
        for ax, zset in zip(axes.flat, args.zsets):
            width_ref, median_ref, shared_settings = None, None, None
            for kind in KINDS:
                if kind not in args.priors:
                    continue
                directory = args.output / "fit" / f"c{cosmo}" / zset / kind / f"seed{args.seed}"
                paths = sorted(directory.glob("chain_*.npy"))
                if not paths:
                    print(f"Missing fit: {directory}", flush=True)
                    continue
                status = json.loads((directory / "status.json").read_text()) if (directory / "status.json").exists() else {}
                if not status.get("converged", False) and not args.allow_unconverged:
                    print(f"Skipping unconfirmed convergence: {directory}", flush=True)
                    continue
                config = json.loads((directory / "settings.json").read_text())
                if config.get("prior_mapping_version") != PRIOR_MAPPING_VERSION:
                    raise RuntimeError(f"Invalid legacy bphi mapping: {directory}. "
                                       "These fits held bphi fixed; rerun in runs/namespaced_v2.")
                if config["priors_sha256"] != sha256(args.output / "priors.json"):
                    raise RuntimeError(f"Fit used different bphi samples: {directory}")
                if len(paths) != config["nwalkers"]:
                    raise RuntimeError(f"Missing walkers: {directory}")
                shared = comparison_settings(config)
                if shared_settings is not None and shared != shared_settings:
                    raise RuntimeError(f"Cannot compare fits with different data/model settings: {directory}")
                shared_settings = shared
                chain = Chain.concatenate([Chain.load(path).remove_burnin(args.burnin)[::args.thin] for path in paths])
                samples = fnl_samples_from_chain(chain, "LRG", ZSETS[zset], mode="bphi", bphi_prior="direct")
                q16, median, q84 = summarize_samples(f"c{cosmo} {zset} {kind}", samples)
                width = q84 - q16
                if kind == "gaussian":
                    width_ref, median_ref = width, median
                truth = true_fnl_loc_for_cosmo(cosmo)
                records.append(dict(cosmo=cosmo, sample=zset, prior=kind, seed=args.seed, converged=status.get("converged", False),
                                    n_samples=len(samples), mean=float(samples.mean()), std=float(samples.std()),
                                    q16=q16, median=median, q84=q84, width68=width, fnl_true=truth,
                                    truth_in_68=bool(q16 <= truth <= q84),
                                    delta_median_vs_gaussian=None if median_ref is None else median - median_ref,
                                    width_ratio_vs_gaussian=None if width_ref is None else width / width_ref))
                grid = np.linspace(-300, 300, 1000)
                ax.plot(grid, gaussian_kde(samples)(grid), color=f"C{KINDS.index(kind)}", label=kind.capitalize())
                present = True
            ax.axvline(true_fnl_loc_for_cosmo(cosmo), color="black", linestyle="--", linewidth=1)
            ax.set(title="Joint LRGs" if zset == "joint" else zset,
                   xlabel=r"$f_{\rm NL}^{\rm loc}$", ylabel="Posterior density", xlim=(-300, 300), ylim=(0, None))
            if ax.get_legend_handles_labels()[0]:
                ax.legend(frameon=False, fontsize=9)
        for ax in list(axes.flat)[len(args.zsets):]:
            ax.set_visible(False)
        fig.tight_layout()
        if present:
            for ext in ("png", "pdf"):
                fig.savefig(plots / f"fnl_prior_shapes_c{cosmo}_seed{args.seed}.{ext}", dpi=180)
        plt.close(fig)
    if records:
        with (tables / f"fnl_constraints_seed{args.seed}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    else:
        print("No completed fits available; no fnl comparison written.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "fit", "compare"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-run", type=Path, default=SOURCE_RUN)
    parser.add_argument("--kde-scale", type=float, default=1.)
    parser.add_argument("--cosmos", nargs="+", choices=("000", "300", "301"), default=["000", "300", "301"])
    parser.add_argument("--zsets", nargs="+", choices=tuple(ZSETS), default=list(ZSETS))
    parser.add_argument("--priors", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--kmin", type=float, default=.006)
    parser.add_argument("--kmax", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nwalkers", type=int, default=60)
    parser.add_argument("--min-steps", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--check-every", type=int, default=200)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--burnin", type=float, default=.5)
    parser.add_argument("--thin", type=int, default=5)
    parser.add_argument("--allow-unconverged", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not 0 <= args.burnin < 1 or args.thin < 1 or args.nwalkers < 2 or args.check_every < 1:
        parser.error("Invalid burn-in, thinning, walker count, or check interval")
    if not 0 < args.kmin < args.kmax or not 0 <= args.min_steps <= args.max_steps:
        parser.error("Require 0 < kmin < kmax and 0 <= min-steps <= max-steps")
    if args.command == "prepare":
        prepare(args.output, args.source_run, args.kde_scale)
        plot_priors(args.output)
    elif args.command == "fit":
        fit(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
