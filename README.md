# bphi Prior from HOD Posterior

Constructing `b_phi` priors from posterior constraints on HOD parameters, and
using them to constrain local `fnl`.

Jiaxi Yu and Nhat-Minh Nguyen (2026), arXiv:26XX.XXXXX —
[github.com/MinhMPA/bphi-prior-from-hod-posterior](https://github.com/MinhMPA/bphi-prior-from-hod-posterior).

## Overview

Local primordial non-Gaussianity (PNG) imprints a scale-dependent term in the
galaxy power spectrum proportional to `fnl * b_phi`,

```text
P_gg(k,z) = [ b_1(z) + b_phi(z) fnl / M(k,z) ]^2 P_mm(k,z) + P_shot(k,z)
```

so the uncertainty on `b_phi` is the dominant limitation when inferring `fnl`.
This repository (i) builds DESI-like galaxy mocks by drawing HOD parameters from
the DESI One-Percent posteriors, (ii) propagates those small-scale constraints
into a posterior-predictive prior `P(b_phi | d_small-scale, theta_HOD)` via a
separate-universe estimator, and (iii) feeds that prior into a local-PNG
likelihood to constrain `fnl`. The code in `python_scripts/` is organized below
to follow this pipeline, with each stage mapped to the corresponding section of
`paper/main.tex`.

## Code Map

The pipeline runs in four stages. Each script is listed under the paper section
it implements.

### 1. Galaxy mocks from AbacusSummit + HOD — paper Sec. 3

Populate Gaussian (`AbacusSummit`) and local-PNG (`AbacusSummitPNG`) halo
catalogs with galaxies using HOD parameters sampled from the DESI 1% posteriors.

| Script | Role |
|--------|------|
| `preproc_parallel.sh` | Preprocess `AbacusSummit` Gaussian halo catalogs (fiducial `c000`, separate-universe `c004`) into AbacusHOD inputs |
| `preproc_parallel_PNG.sh` | Preprocess `AbacusSummitPNG` catalogs (`c000`, `c300`–`c303`) |
| `HOD_LHC.py` | Generate Latin-hypercube HOD parameter samples spanning the posterior |
| `HOD_one_lhc.py` (+ `HOD_one_lhc.sh`) | Populate one realization (a LHC row or the YAML default) into a mock; optionally compute real-space `P(k)` and linear bias `b_1` |
| `bphi_prior_HOD.py` | Run HOD over the full LHC sample set (the posterior-predictive mock ensemble) |
| `HOD_utils.py` | Shared utilities: catalog population (`compute_all`), `wp`/multipoles, real-space `P(k)`, `b_1` estimation, and triangle/contour plotting |
| `AbacusHOD_test.py` | Validation of the AbacusHOD population path |
| `AbacusHOD.sh`, `HOD_separate_universe.sh` | Orchestration launchers documenting the mock-building sequence |

### 2. `b_phi` estimation and prior construction — paper Sec. 2

Estimate `b_phi` and its scatter at fixed small-scale constraints, yielding the
HOD-informed prior. Two independent estimators are provided as a cross-check.

| Script | Role |
|--------|------|
| `HOD_bphi.py` (+ `HOD_bphi.sh`) | Separate-universe estimator: SU mode (`c000` vs `c004`) or PNG mode (`c000` vs a PNG cosmology) |
| `bphi_dz.py` | Abundance/growth-derivative estimator, `b_phi ~ 2 (dln n / dz) / (dln sigma_8 / dz)`, the universality-relation form |
| `bphi_report.py` | Mean and error bars of the `b_phi` posterior-predictive |
| `bphi_plot.py` | Plot `b_phi` distributions |

### 3. Local `fnl` inference — paper Sec. 4–5

Validate the data vector and covariance, then run the joint redshift-space
multipole (`P_0`, `P_2`) PNG fit. The sampling **mode** selects the benchmark
scenario (see below).

| Script | Role |
|--------|------|
| `bphi_prior_pipeline.py` (+ `bphi_prior_pipeline.sh`) | **Production pipeline.** Subcommands `pk` (validate external `P(k)`), `cov` (Gaussian covariance via `thecov`), `fit` (joint `P_0`+`P_2` PNG fit). Modes `bphi`/`b-p`/`bfnl`; multipoles selected with `--ells 0 2` |
| `bphi_prior_pipeline_bfnl.py` (+ `bphi_prior_pipeline_bfnl.sh`) | Earlier monopole-only variant focused on the `bfnl` baseline (`bphi`/`bfnl` modes only, no `--ells` routing) |

### 4. Validation, diagnostics, and plots — paper Sec. 5

| Script | Role |
|--------|------|
| `bphi_prior_plot_pk.py` | Plot `P(k)` and compare its variance against `baocov.pk_var` |
| `test_png_bias_krange.py` | PNG scale-dependent-bias `k`-range diagnostic from real-space `P(k)` |
| `lowS8_HOD.py`, `AbacusHOD_lowS8.py`, `lowS8_comparison.py` (+ `bphi_HOD.sh`) | Low-`S8` cross-check: tune and create low-`S8` Abacus mocks, then recompute bias and `b_phi` for comparison |

## Benchmark Scenarios

The paper notation for the product amplitude is `B_i = b_{phi,i} fnl_loc`. In
code and desilike configs this product is named `bfnl_z` or `bfnl_loc`. The
pipeline `--mode` flag selects the scenario:

| `--mode` | Scenario | Description |
|----------|----------|-------------|
| `bfnl` | **A — bfnl baseline** | Sample one product amplitude `B_i = b_{phi,i} fnl_loc` per independent tracer/redshift bin; no `bphi_i` prior, no universal `fnl_loc` |
| `bphi` | **B — HOD-informed Gaussian `bphi` prior** | Infer one universal `fnl_loc` using a Gaussian HOD prior `pi_i(beta_i) = Normal(beta_i; mu_bphi_i, sigma_bphi_i^2)` per `bphi_i`. Validate B-direct (`mode="bphi"`) against the B-marginal custom likelihood before promoting either |
| `b-p` | **C — fixed-`p` universality** | Fix `bphi_i = 2 delta_c (b1_i - p_i)` with tracer/sample-specific `p_i` for LRG/ELG/QSO, then infer one universal `fnl_loc` |

## Marginal Likelihood for `fnl`

The marginalized Scenario B likelihood follows the appendix formalism in
`paper/main.tex` (Sec. "Marginal likelihood for fnl"):

```text
L_i^marg(f) = integral d beta_i L_i(B_i = f beta_i) pi_i(beta_i)
P(f | all data) proportional to pi(f) product_i L_i^marg(f)
```

If both the product-amplitude likelihood `L_i(B_i)` and the HOD-informed
`bphi_i` prior are Gaussian, the marginal likelihood is analytic and uses only
`Bhat_i`, `sigma_B_i`, `mu_bphi_i`, and `sigma_bphi_i`. Gauss-Hermite quadrature
is only needed when the `bphi_i` prior is Gaussian but `L_i(B_i)` is kept as a
numerical 1D likelihood. Non-Gaussian `bphi_i` priors use their own
tabulated-grid or weighted-sample integration weights.

## Analysis Contract

The target PNG likelihood fits the joint redshift-space galaxy power-spectrum
multipole data vector

```text
d_z = [P_0^g(k), P_2^g(k)]
```

for each tracer/redshift sample. The monopole (`ell=0`) and quadrupole
(`ell=2`) are fit jointly, using the matching Gaussian covariance supplied by
`thecov`. The quadrupole is part of the production data vector, not a
diagnostic-only quantity.

The HOD catalog-space conventions (correctness-critical — violating them double
shifts the catalog) are:

- AbacusHOD LRG catalogs generated with `want_rsd: True` are already shifted to
  redshift space by AbacusHOD. Downstream clustering or power-spectrum code must
  not apply a second RSD shift to those LRG positions.
- LRG centrals are placed at halo-center positions, but their velocities include
  the configured central velocity-bias/scatter contribution, e.g.
  `vel + alpha_c * vdev`.
- LRG satellites are not generally placed at halo centers. In the primary
  `use_particles: True` path, satellite positions follow particle/subsample
  positions and velocities include a 1-halo contribution controlled by
  `alpha_s`.
- AbacusHOD secondary-redshift fallback catalogs use NFW satellite placement,
  which has different velocity-bias behavior from the particle-subsample path
  and must be documented separately in validation outputs.
- ELG/HODDIES catalogs use a separate velocity/RSD convention. Before ELG
  benchmarks are promoted to production validation, audit whether HODDIES has
  already applied RSD so the pipeline does not double shift the catalog.

## Installation

This code runs on the DESI `cosmodesi` stack. On NERSC it is pre-built — the
`.sh` launchers source it directly, no install needed:

```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
```

To build the stack elsewhere, follow the upstream installer:
[cosmodesi/cosmodesiconda](https://github.com/cosmodesi/cosmodesiconda).

Mock-building additionally requires the separate `abacushod` conda environment
(`abacusnbody`/AbacusHOD, `HODDIES`); activate it with `conda activate abacushod`.

## Dependencies

The code targets the DESI `cosmodesi` software stack on NERSC; there is no
standalone `requirements.txt`. The `.sh` launchers source the environment for
each stage.

**DESI clustering + inference stack** (provided by `cosmodesi_environment.sh`):

| Package | Used for |
|---------|----------|
| `desilike` | Likelihood/inference framework: power-spectrum observables, templates, samplers (Minuit, Zeus), profilers, `Chain` I/O |
| `pypower` | Power-spectrum estimation (`CatalogFFTPower`, `PowerSpectrumMultipoles`) |
| `pycorr` | Two-point correlation functions (`wp`, multipoles) |
| `cosmoprimo` | Cosmology primitives and the `Fourier` transfer-function interface |
| `thecov` | Analytic Gaussian covariance (`CovarianceMatrix`) |
| `lsstypes` | Covariance loading / effective-volume utilities |

**Mock building** (the separate `abacushod` conda environment):

| Package | Used for |
|---------|----------|
| `abacusnbody` (AbacusHOD) | Populate `AbacusSummit`/`AbacusSummitPNG` halo catalogs with galaxies |
| `HODDIES` | ELG HOD population |

**General scientific Python:** `numpy`, `scipy` (`interpolate.interp1d`,
`stats.qmc`, `constants`), `pandas`, `matplotlib`, `corner`, `PyYAML`.

Standard-library only otherwise (`argparse`, `os`, `sys`, `pathlib`, `re`,
`functools`, `importlib`). Stages also run under SLURM (`srun`/`sbatch`).

## Example Usage

The five stages below follow the analysis of `paper/main.tex`, shown for one LRG
sample (`LRG`, `z=1.1`) validated against the `fnl=+30` PNG mock (`c300`). The
`.sh` launchers source the `cosmodesi` environment and wrap each step in the
appropriate `srun`/`sbatch` allocation.

```bash
# 1. Build HOD mocks (paper "b_phi from separate-universe mocks")
#    Preprocess the AbacusSummit base (c000) and low-sigma8 (c004) halo pairs,
#    plus the AbacusSummitPNG validation boxes (c000, c300=fnl+30, c301=fnl-30).
sbatch preproc_parallel.sh 000          # repeat for 004
sbatch preproc_parallel_PNG.sh 0        # SLURM array over PNG cosmologies
#    Draw N=100 Latin-hypercube HOD samples from the DESI 1% posteriors, then
#    populate the 100 base/low-sigma8 pairs (and the PNG validation mock).
python HOD_LHC.py LRG
sbatch HOD_one_lhc.sh LRG SU

# 2. Construct the b_phi prior (paper "b_phi as a separate-universe response")
#    HOD_one_lhc.py records N_gal for each HOD point and cosmology; the SU
#    response b_phi = 2 ln(N_gal^c000 / N_gal^c004) / ln(0.807952 / 0.75) is
#    tabulated into backup/bphi_AbacusHOD_*.txt, which the fit auto-loads as the
#    Gaussian prior N(<b_phi>, var) per redshift bin.

# 3. Build the P0+P2 data vector and covariance (paper "Data Vectors and Covariances")
#    Validate the pypower multipoles, then the thecov Gaussian covariance scaled
#    to the DESI effective volume.
python bphi_prior_pipeline.py pk  --simtype PNG --tracer LRG --zsnaps 1.1 --cosmo 300 --ells 0 2 --kmin 0.006
python bphi_prior_pipeline.py cov --simtype PNG --tracer LRG --zsnaps 1.1 --cosmo 300 --ells 0 2 --kmin 0.006 --cov-veff

# 4. Run the joint PNG fit with the HOD b_phi prior (paper "Inference")
#    Default bfnl mode: sample B_i = b_phi_i * fnl under the Gaussian b_phi prior,
#    fitting P0+P2 jointly with the zeus sampler.
python bphi_prior_pipeline.py fit \
    --simtype PNG --tracer LRG --zsnaps 1.1 --mode bfnl \
    --kmin 0.006 --ells 0 2 --cov-veff --cosmo 300 \
    --with-bphi-prior --run-sampler

# 5. Marginalize bfnl -> fnl and plot (paper "Results")
#    Reweight the bfnl chain over the b_phi prior to obtain the fnl posterior.
python bphi_prior_plot_fnl.py --mode bfnl --fnl-method description --cosmo 300

# Convenience launcher (mode = bfnl | bphi | b-p)
bash bphi_prior_pipeline.sh bfnl
```
