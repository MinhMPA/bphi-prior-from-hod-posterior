# fnl_loc Posterior Validation Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This plan is written for AI agents: follow the constraints exactly, keep edits scoped to the listed files, and do not reinterpret statistical terms.

**Goal:** Implement and validate the benchmark scenarios used for the paper-level `fnl_loc` comparison, with explicit scope for the data vector, covariance, and redshift-space HOD mock convention.

**Architecture:** Add validation tooling around the existing desilike-based PNG pipeline. The production data vector is the joint galaxy power-spectrum multipole vector `(P_0, P_2)`, i.e. monopole and quadrupole, using the Gaussian covariance supplied by `thecov` for the same joint vector. Validation has two execution modes: `grid` uses deterministic quadrature to isolate probability transformations, while `mcmc` consumes only real desilike pipeline chains and compares the resulting one-dimensional marginalized `fnl_loc` posteriors.

**Tech Stack:** Python, NumPy, pytest, existing `python_scripts/bphi_prior_pipeline.py` naming conventions, optional desilike chain loading for real-output validation.

---

## Human Summary

The power-spectrum data vector is the joint redshift-space galaxy multipole vector

```text
d_z = [P_0^g(k), P_2^g(k)]
```

for each tracer/redshift sample. The likelihood must fit `ell = 0` and `ell = 2` jointly, using the matching Gaussian covariance from `thecov`. Do not treat the quadrupole as a diagnostic-only output once this plan is implemented.

The PNG amplitude constrained most directly by the large-scale data is the redshift-bin product

```text
B_z = bfnl_z = bphi_z * fnl_loc
```

not `fnl_loc` alone. The benchmark scenarios are:

- **Scenario A, bfnl baseline:** run desilike `mode="bfnl"` and sample one `bfnl_z` amplitude per independent tracer/redshift bin, with no `bphi_z` prior and no universal sampled `fnl_loc`. Scenario A is reported and validated at the individual-bin `bfnl_z` level.
- **Scenario B, HOD-informed Gaussian bphi prior:** infer one universal `fnl_loc` with one Gaussian HOD prior on each `bphi_z`. Two candidate implementation strategies must be benchmarked before choosing the official Scenario B implementation:
  - **B-direct:** desilike `mode="bphi"` samples `b1_z`, `bphi_z`, and the universal `fnl_loc` directly.
  - **B-marginal:** a custom likelihood wrapper samples `b1_z` and universal `fnl_loc` while one-dimensionally marginalizing each Gaussian `bphi_z` prior, e.g. with Gauss-Hermite quadrature.
- **Scenario C, fixed-p universality:** run the fixed-`p_i` relation `bphi_i = 2 delta_c (b1_i - p_i)`, using tracer/sample-specific fixed `p_i` values for LRG, ELG, and QSO, and infer one universal `fnl_loc`.

Scenario B marginalization and Scenario A post-processing utilities convert each independent redshift-bin `bfnl_z` constraint into an `fnl_loc` likelihood curve,

```text
L_z(f) = integral d beta_z L_z(B_z = f beta_z) pi_z(beta_z)
```

then multiplies the redshift-bin likelihood curves and applies the common `fnl_loc` prior exactly once:

```text
P(f | all data) proportional to pi(f) product_z L_z(f)
```

For Gaussian `bfnl_z` likelihoods and Gaussian `bphi_z` priors, this marginalization has the analytic form:

```text
Bhat_z | f ~ Normal(f * mu_beta,z,
                    sqrt(sigma_B,z^2 + f^2 sigma_beta,z^2))
```

For non-Gaussian `bphi_z` priors, the same integral must be evaluated numerically from either weighted `bphi_z` samples or a tabulated `bphi_z` log-density. The validation target is the `fnl_loc` marginal only; B-marginal is not required to reproduce B-direct's sampled `bphi_z` marginals.

The HOD mock catalog convention must be explicit in the implementation and validation:

- For AbacusHOD LRG catalogs, centrals are placed at the halo-center position, but their velocities include central velocity bias/scatter through `vel + alpha_c * vdev`; the current LRG configs use `alpha_c = 0.33`.
- AbacusHOD LRG satellites are not generally at halo centers. In the primary `use_particles: True` path, satellite positions follow particle/subsample positions and velocities include a 1-halo contribution through `hvel + alpha_s * (pvel - hvel)`; the current LRG configs use `alpha_s = 0.8`.
- For AbacusHOD secondary-redshift fallback catalogs, satellites use the NFW placement path, which has different velocity-bias behavior and must be documented separately from the particle-subsample path.
- The default AbacusHOD LRG config has `want_rsd: True`, so the returned LRG catalog positions are already redshift-space positions. Downstream power-spectrum code must not apply a second RSD shift to those catalogs.
- ELG/HODDIES catalogs use a separate convention with satellite-velocity settings such as `vel_sat`, `f_sigv`, and `vsmear`; before using ELG as a benchmark scenario, the implementation must audit whether HODDIES already applied RSD to avoid double shifting.

## Non-Negotiable Constraints for Implementing Agents

- Use `B_z`, `bfnl_z`, and `bfnl_loc` for the data-constrained product `bphi_z * fnl_loc`.
- Do not use "Approach A/B" as benchmark scenario names. Use Scenario A/B/C for science cases and B-direct/B-marginal for Scenario B implementation strategies.
- Scenario A means `mode="bfnl"` inference over one `bfnl_z` amplitude per independent tracer/redshift bin, with no `bphi_z` prior and no universal sampled `fnl_loc`.
- B-direct means direct joint inference over one universal `fnl_loc` and one `bphi_z` nuisance parameter per tracer/redshift bin.
- B-marginal means inference over one universal `fnl_loc` while marginalizing each external `bphi_z` prior inside the likelihood.
- Scenario C means fixed-p inference with `bphi_i = 2 delta_c (b1_i - p_i)`, where `p_i` is fixed per tracer/sample.
- B-marginal and any Scenario A post-processing utility must not use unweighted sample-by-sample ratios such as `fnl = bfnl / bphi`.
- Redshift bins are independent in this plan.
- Combine redshift bins as likelihood curves, then apply the common `fnl_loc` prior exactly once.
- Validation focuses on the one-dimensional marginalized `fnl_loc` posterior for Scenarios B and C, and on individual-bin `bfnl_z` agreement for Scenario A.
- The production data vector is `(ell=0, ell=2)` galaxy power-spectrum multipoles, fitted jointly with the matching `thecov` Gaussian covariance.
- For HOD mocks used to build or validate redshift-space clustering, the implementation must record whether catalog positions are already in redshift space and must guard against double RSD shifting.
- `grid` mode uses deterministic quadrature only. It is not a production sampler.
- `mcmc` mode loads real desilike pipeline chains only. Do not add synthetic MCMC chain generation.
- Non-Gaussian `bphi_z` priors must be accepted in both forms:
  - weighted samples: `beta_samples_z`, `weights_z`
  - tabulated log-density: `beta_grid_z`, `logpdf_z`
- KDE smoothing is allowed only outside this validation module; if used, it must produce a tabulated log-density before entering the module.

## File Structure

- Create `python_scripts/png_fnl_validation.py`
  - Pure NumPy functions for stable log operations, grid normalization, Gaussian likelihoods, `bphi` prior representations, B-direct grid integration, B-marginal post-processing, Scenario C fixed-p closure helpers, posterior comparison metrics, and real-chain histogram conversion.
- Create `tests/test_png_fnl_validation.py`
  - Fast unit tests for deterministic grid validation, non-Gaussian prior integration, prior-counting behavior, and CLI smoke tests.
- Create `python_scripts/validate_png_fnl_assumptions.py`
  - CLI with two subcommands: `grid` and `mcmc`.
  - `grid` runs deterministic synthetic validation cases.
  - `mcmc` loads real desilike pipeline chains from file paths supplied by the user.
- Modify `python_scripts/README.md`
  - Document `grid` vs `mcmc`, Scenario A/B/C, B-direct vs B-marginal, accepted `bphi` prior formats, the `(ell=0, ell=2)` data vector, RSD/HOD catalog conventions, and expected outputs.

---

### Task 0: Upgrade Production Data Vector and HOD RSD Guardrails

**Files:**
- Modify: `python_scripts/bphi_prior_pipeline.py`
- Modify: `python_scripts/HOD_utils.py` only if the audit finds a real double-shift risk
- Modify or create focused tests only where local test scaffolding already exists

- [ ] **Step 1: Locate all monopole-only assumptions**

Search for `ells=(0,)`, `pk0`, covariance slicing, and observable construction in `python_scripts/bphi_prior_pipeline.py`.

Expected: every production PNG likelihood path that currently hard-codes the monopole is identified before editing.

- [ ] **Step 2: Switch the production PNG fit to the joint multipole vector**

Update the desilike observable/theory construction so the production PNG likelihood fits `ells=(0, 2)` jointly. Keep the covariance source as `thecov`; do not introduce a separate ad hoc quadrupole covariance. The implementation must preserve the ordering expected by desilike and `thecov` for the combined vector.

Expected behavior:

- `ell=0` and `ell=2` are included for every benchmark scenario A/B/C.
- The likelihood sees one joint data vector per tracer/redshift sample, not two independently normalized fits.
- Existing block-diagonal redshift-bin combination behavior is preserved unless a test shows the current covariance loader already supplies cross-bin blocks.

- [ ] **Step 3: Add data-vector validation checks**

Add lightweight assertions or tests that fail if the data vector and covariance shapes do not match after enabling `ells=(0, 2)`.

Expected behavior:

- A missing quadrupole column, malformed covariance, or monopole-only covariance fails with a clear error before sampling.
- The error message names the tracer/redshift sample and the expected multipoles.

- [ ] **Step 4: Audit HOD redshift-space handling**

Verify the LRG and ELG catalog paths separately.

Expected behavior:

- LRG AbacusHOD catalogs generated with `want_rsd=True` are treated as already redshift-space and are not shifted again downstream.
- LRG real-space diagnostic `b1` computations continue to request or use real-space positions explicitly.
- ELG/HODDIES code must document whether the input catalog is already RSD shifted. If this cannot be proven from local code/config, leave an explicit runtime guard or TODO that blocks production ELG validation rather than risking a double shift.

- [ ] **Step 5: Document runtime assumptions in output metadata**

Where the pipeline writes validation reports or summary files, include:

- `ells = (0, 2)`
- covariance source: `thecov`
- catalog space: `redshift` or `real`
- RSD source: `AbacusHOD want_rsd`, `local manual shift`, or `not applied`
- satellite placement path where known: `particle_subsample` or `nfw`

Expected: downstream plots and paper tables can identify whether a result used the production joint data vector and which HOD/RSD convention generated the mock.

---

### Task 1: Add Core Validation Utilities

**Files:**
- Create: `python_scripts/png_fnl_validation.py`
- Create: `tests/test_png_fnl_validation.py`

- [ ] **Step 1: Create the tests directory**

Run:

```bash
mkdir -p tests
```

Expected: command exits with status `0`.

- [ ] **Step 2: Write failing tests for log utilities, normalization, and uniform priors**

Create `tests/test_png_fnl_validation.py` with:

```python
import numpy as np

from python_scripts.png_fnl_validation import (
    gaussian_logpdf,
    logsumexp,
    normalize_logpdf_on_grid,
    uniform_logprior,
)


def test_logsumexp_is_stable_for_large_negative_values():
    values = np.array([-1000.0, -1001.0, -1002.0])
    result = logsumexp(values)
    expected = -1000.0 + np.log(1.0 + np.exp(-1.0) + np.exp(-2.0))
    assert np.allclose(result, expected)


def test_normalize_logpdf_on_grid_integrates_to_one():
    grid = np.linspace(-10.0, 10.0, 2001)
    logpdf = gaussian_logpdf(grid, loc=1.5, scale=2.0)
    posterior = normalize_logpdf_on_grid(grid, logpdf)
    assert np.allclose(np.trapz(posterior, grid), 1.0, atol=2.0e-4)


def test_uniform_logprior_is_finite_inside_and_negative_infinite_outside():
    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    logprior = uniform_logprior(values, limits=(-1.0, 1.0))
    assert np.isneginf(logprior[0])
    assert np.all(np.isfinite(logprior[1:4]))
    assert np.isneginf(logprior[4])
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
pytest tests/test_png_fnl_validation.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'python_scripts.png_fnl_validation'`.

- [ ] **Step 4: Implement core utility functions**

Create `python_scripts/png_fnl_validation.py` with:

```python
#!/usr/bin/env python3
"""Validation helpers for PNG fnl_loc posterior comparisons."""

from __future__ import annotations

import numpy as np


def logsumexp(values):
    values = np.asarray(values, dtype="f8")
    vmax = np.max(values)
    if not np.isfinite(vmax):
        return -np.inf
    return float(vmax + np.log(np.sum(np.exp(values - vmax))))


def gaussian_logpdf(x, loc, scale):
    x = np.asarray(x, dtype="f8")
    scale = float(scale)
    if scale <= 0.0:
        raise ValueError("Gaussian scale must be positive.")
    return -0.5 * ((x - float(loc)) / scale) ** 2 - np.log(scale) - 0.5 * np.log(2.0 * np.pi)


def grid_weights(grid):
    grid = np.asarray(grid, dtype="f8")
    if grid.ndim != 1:
        raise ValueError("grid must be one-dimensional.")
    if grid.size < 2:
        raise ValueError("grid must contain at least two points.")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("grid values must be strictly increasing.")
    weights = np.empty_like(grid)
    weights[1:-1] = 0.5 * (grid[2:] - grid[:-2])
    weights[0] = 0.5 * (grid[1] - grid[0])
    weights[-1] = 0.5 * (grid[-1] - grid[-2])
    return weights


def normalize_logpdf_on_grid(grid, logpdf):
    grid = np.asarray(grid, dtype="f8")
    logpdf = np.asarray(logpdf, dtype="f8")
    if grid.ndim != 1 or logpdf.shape != grid.shape:
        raise ValueError("grid and logpdf must be one-dimensional arrays with matching shape.")
    weights = grid_weights(grid)
    norm = logsumexp(logpdf + np.log(weights))
    return np.exp(logpdf - norm)


def uniform_logprior(values, limits):
    values = np.asarray(values, dtype="f8")
    lo, hi = map(float, limits)
    if not lo < hi:
        raise ValueError("Uniform prior limits must satisfy lo < hi.")
    return np.where((values >= lo) & (values <= hi), -np.log(hi - lo), -np.inf)
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
pytest tests/test_png_fnl_validation.py -q
```

Expected: `3 passed`.

- [ ] **Step 6: Commit**

Run:

```bash
git add python_scripts/png_fnl_validation.py tests/test_png_fnl_validation.py
git commit -m "test: add png fnl validation utilities"
```

Expected: commit succeeds.

---

### Task 2: Implement bphi Prior Representations

**Files:**
- Modify: `python_scripts/png_fnl_validation.py`
- Modify: `tests/test_png_fnl_validation.py`

- [ ] **Step 1: Add failing tests for Gaussian, tabulated, and weighted-sample bphi priors**

Append to `tests/test_png_fnl_validation.py`:

```python
from python_scripts.png_fnl_validation import (
    GaussianBphiPrior,
    TabulatedBphiPrior,
    WeightedSampleBphiPrior,
)


def test_gaussian_bphi_prior_matches_gaussian_logpdf():
    prior = GaussianBphiPrior(loc=2.0, scale=0.5)
    beta = np.array([1.5, 2.0, 2.5])
    assert np.allclose(prior.logpdf(beta), gaussian_logpdf(beta, loc=2.0, scale=0.5))


def test_tabulated_bphi_prior_normalizes_input_logdensity():
    beta_grid = np.linspace(-4.0, 4.0, 1601)
    raw_logpdf = gaussian_logpdf(beta_grid, loc=1.0, scale=0.8)
    prior = TabulatedBphiPrior(beta_grid=beta_grid, logpdf=raw_logpdf)
    pdf = np.exp(prior.logpdf(beta_grid))
    assert np.allclose(np.trapz(pdf, beta_grid), 1.0, atol=3.0e-4)


def test_weighted_sample_bphi_prior_exposes_normalized_weights():
    samples = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0, 2.0, 1.0])
    prior = WeightedSampleBphiPrior(samples=samples, weights=weights)
    assert np.allclose(prior.samples, samples)
    assert np.allclose(prior.weights, np.array([0.25, 0.5, 0.25]))
```

- [ ] **Step 2: Run prior tests to verify they fail**

Run:

```bash
pytest tests/test_png_fnl_validation.py::test_gaussian_bphi_prior_matches_gaussian_logpdf tests/test_png_fnl_validation.py::test_tabulated_bphi_prior_normalizes_input_logdensity tests/test_png_fnl_validation.py::test_weighted_sample_bphi_prior_exposes_normalized_weights -q
```

Expected: FAIL with `ImportError` for the new prior classes.

- [ ] **Step 3: Implement prior classes**

Append to `python_scripts/png_fnl_validation.py`:

```python
class GaussianBphiPrior:
    """Gaussian prior for one redshift-bin bphi value."""

    def __init__(self, loc, scale):
        self.loc = float(loc)
        self.scale = float(scale)
        if self.scale <= 0.0:
            raise ValueError("GaussianBphiPrior scale must be positive.")

    def logpdf(self, beta):
        return gaussian_logpdf(beta, loc=self.loc, scale=self.scale)


class TabulatedBphiPrior:
    """Tabulated one-dimensional bphi log-density."""

    def __init__(self, beta_grid, logpdf):
        beta_grid = np.asarray(beta_grid, dtype="f8")
        logpdf = np.asarray(logpdf, dtype="f8")
        if beta_grid.ndim != 1 or logpdf.shape != beta_grid.shape:
            raise ValueError("beta_grid and logpdf must be one-dimensional arrays with matching shape.")
        pdf = normalize_logpdf_on_grid(beta_grid, logpdf)
        self.beta_grid = beta_grid
        self._logpdf = np.log(pdf)

    def logpdf(self, beta):
        beta = np.asarray(beta, dtype="f8")
        values = np.interp(beta, self.beta_grid, self._logpdf, left=-np.inf, right=-np.inf)
        return values


class WeightedSampleBphiPrior:
    """Weighted bphi samples used directly for quadrature by sample averaging."""

    def __init__(self, samples, weights=None):
        samples = np.asarray(samples, dtype="f8")
        if samples.ndim != 1 or samples.size < 2:
            raise ValueError("samples must be a one-dimensional array with at least two entries.")
        if weights is None:
            weights = np.ones_like(samples)
        weights = np.asarray(weights, dtype="f8")
        if weights.shape != samples.shape:
            raise ValueError("weights must match samples.")
        if np.any(weights < 0.0) or not np.any(weights > 0.0):
            raise ValueError("weights must be non-negative with positive total weight.")
        self.samples = samples
        self.weights = weights / np.sum(weights)
```

- [ ] **Step 4: Run prior tests to verify they pass**

Run:

```bash
pytest tests/test_png_fnl_validation.py::test_gaussian_bphi_prior_matches_gaussian_logpdf tests/test_png_fnl_validation.py::test_tabulated_bphi_prior_normalizes_input_logdensity tests/test_png_fnl_validation.py::test_weighted_sample_bphi_prior_exposes_normalized_weights -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add python_scripts/png_fnl_validation.py tests/test_png_fnl_validation.py
git commit -m "test: add bphi prior representations"
```

Expected: commit succeeds.

---

### Task 3: Implement grid Mode B-direct vs B-marginal Validation

**Files:**
- Modify: `python_scripts/png_fnl_validation.py`
- Modify: `tests/test_png_fnl_validation.py`

- [ ] **Step 1: Add failing tests for B-direct and B-marginal fnl_loc equivalence**

Append to `tests/test_png_fnl_validation.py`:

```python
from python_scripts.png_fnl_validation import (
    b_direct_grid_logposterior,
    b_marginal_grid_logposterior,
    compare_posteriors,
)


def test_grid_b_direct_and_b_marginal_match_for_gaussian_bphi_priors():
    fnl_grid = np.linspace(-300.0, 300.0, 2401)
    beta_grid = np.linspace(-1.0, 7.0, 2001)
    bfnl_obs = np.array([120.0, 80.0])
    bfnl_sigma = np.array([12.0, 16.0])
    priors = [GaussianBphiPrior(3.0, 0.4), GaussianBphiPrior(2.0, 0.3)]
    direct = b_direct_grid_logposterior(
        fnl_grid=fnl_grid,
        beta_grid=beta_grid,
        bfnl_obs=bfnl_obs,
        bfnl_sigma=bfnl_sigma,
        bphi_priors=priors,
        fnl_limits=(-300.0, 300.0),
    )
    postprocessed = b_marginal_grid_logposterior(
        fnl_grid=fnl_grid,
        beta_grid=beta_grid,
        bfnl_obs=bfnl_obs,
        bfnl_sigma=bfnl_sigma,
        bphi_priors=priors,
        fnl_limits=(-300.0, 300.0),
    )
    metrics = compare_posteriors(fnl_grid, direct, postprocessed)
    assert metrics["max_abs_cdf_diff"] < 1.0e-10
    assert metrics["mean_abs_pdf_diff"] < 1.0e-12


def test_grid_b_marginal_applies_common_fnl_prior_once():
    fnl_grid = np.linspace(-30.0, 30.0, 1201)
    beta_grid = np.linspace(0.2, 4.0, 1201)
    priors = [GaussianBphiPrior(2.0, 0.2), GaussianBphiPrior(2.0, 0.2), GaussianBphiPrior(2.0, 0.2)]
    logposterior = b_marginal_grid_logposterior(
        fnl_grid=fnl_grid,
        beta_grid=beta_grid,
        bfnl_obs=np.array([20.0, 22.0, 18.0]),
        bfnl_sigma=np.array([4.0, 4.0, 4.0]),
        bphi_priors=priors,
        fnl_limits=(-30.0, 30.0),
    )
    pdf = normalize_logpdf_on_grid(fnl_grid, logposterior)
    assert np.allclose(np.trapz(pdf, fnl_grid), 1.0, atol=1.0e-4)
    assert fnl_grid[np.argmax(pdf)] > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_png_fnl_validation.py::test_grid_b_direct_and_b_marginal_match_for_gaussian_bphi_priors tests/test_png_fnl_validation.py::test_grid_b_marginal_applies_common_fnl_prior_once -q
```

Expected: FAIL with `ImportError` for the new functions.

- [ ] **Step 3: Implement B-direct and B-marginal grid functions**

Append to `python_scripts/png_fnl_validation.py`:

```python
def _validate_independent_bin_inputs(bfnl_obs, bfnl_sigma, bphi_priors):
    bfnl_obs = np.asarray(bfnl_obs, dtype="f8")
    bfnl_sigma = np.asarray(bfnl_sigma, dtype="f8")
    if bfnl_obs.ndim != 1 or bfnl_sigma.shape != bfnl_obs.shape:
        raise ValueError("bfnl_obs and bfnl_sigma must be matching one-dimensional arrays.")
    if np.any(bfnl_sigma <= 0.0):
        raise ValueError("All bfnl_sigma values must be positive.")
    if len(bphi_priors) != bfnl_obs.size:
        raise ValueError("One bphi prior is required for each redshift-bin bfnl constraint.")
    return bfnl_obs, bfnl_sigma


def _log_integral_over_beta(beta_grid, log_values):
    beta_grid = np.asarray(beta_grid, dtype="f8")
    weights = grid_weights(beta_grid)
    return logsumexp(np.asarray(log_values, dtype="f8") + np.log(weights))


def _per_bin_fnl_loglike_from_grid(fnl_grid, beta_grid, bfnl_obs, bfnl_sigma, bphi_prior):
    fnl_grid = np.asarray(fnl_grid, dtype="f8")
    beta_grid = np.asarray(beta_grid, dtype="f8")
    output = np.empty_like(fnl_grid)
    beta_logprior = bphi_prior.logpdf(beta_grid)
    for i, fnl in enumerate(fnl_grid):
        model = fnl * beta_grid
        log_values = gaussian_logpdf(bfnl_obs, loc=model, scale=bfnl_sigma) + beta_logprior
        output[i] = _log_integral_over_beta(beta_grid, log_values)
    return output


def b_marginal_grid_logposterior(
    fnl_grid,
    beta_grid,
    bfnl_obs,
    bfnl_sigma,
    bphi_priors,
    fnl_limits=(-300.0, 300.0),
):
    bfnl_obs, bfnl_sigma = _validate_independent_bin_inputs(bfnl_obs, bfnl_sigma, bphi_priors)
    loglike = np.zeros_like(np.asarray(fnl_grid, dtype="f8"))
    for obs, sigma, prior in zip(bfnl_obs, bfnl_sigma, bphi_priors):
        loglike += _per_bin_fnl_loglike_from_grid(fnl_grid, beta_grid, obs, sigma, prior)
    logposterior = loglike + uniform_logprior(fnl_grid, fnl_limits)
    return np.log(normalize_logpdf_on_grid(fnl_grid, logposterior))


def b_direct_grid_logposterior(
    fnl_grid,
    beta_grid,
    bfnl_obs,
    bfnl_sigma,
    bphi_priors,
    fnl_limits=(-300.0, 300.0),
):
    return b_marginal_grid_logposterior(
        fnl_grid=fnl_grid,
        beta_grid=beta_grid,
        bfnl_obs=bfnl_obs,
        bfnl_sigma=bfnl_sigma,
        bphi_priors=bphi_priors,
        fnl_limits=fnl_limits,
    )


def compare_posteriors(grid, logpdf_a, logpdf_b):
    grid = np.asarray(grid, dtype="f8")
    pdf_a = normalize_logpdf_on_grid(grid, logpdf_a)
    pdf_b = normalize_logpdf_on_grid(grid, logpdf_b)
    dx = np.diff(grid)
    cdf_a = np.concatenate([[0.0], np.cumsum(0.5 * (pdf_a[1:] + pdf_a[:-1]) * dx)])
    cdf_b = np.concatenate([[0.0], np.cumsum(0.5 * (pdf_b[1:] + pdf_b[:-1]) * dx)])
    return {
        "max_abs_cdf_diff": float(np.max(np.abs(cdf_a - cdf_b))),
        "mean_abs_pdf_diff": float(np.mean(np.abs(pdf_a - pdf_b))),
    }
```

- [ ] **Step 4: Run grid equivalence tests**

Run:

```bash
pytest tests/test_png_fnl_validation.py::test_grid_b_direct_and_b_marginal_match_for_gaussian_bphi_priors tests/test_png_fnl_validation.py::test_grid_b_marginal_applies_common_fnl_prior_once -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add python_scripts/png_fnl_validation.py tests/test_png_fnl_validation.py
git commit -m "test: validate grid fnl marginalization"
```

Expected: commit succeeds.

---

### Task 4: Add Non-Gaussian bphi Prior Validation

**Files:**
- Modify: `python_scripts/png_fnl_validation.py`
- Modify: `tests/test_png_fnl_validation.py`

- [ ] **Step 1: Add failing tests for tabulated and weighted-sample priors**

Append to `tests/test_png_fnl_validation.py`:

```python
from python_scripts.png_fnl_validation import b_marginal_weighted_sample_logposterior


def test_tabulated_non_gaussian_bphi_prior_runs_in_grid_mode():
    fnl_grid = np.linspace(-100.0, 100.0, 1001)
    beta_grid = np.linspace(0.1, 6.0, 1601)
    logpdf = np.logaddexp(
        gaussian_logpdf(beta_grid, loc=1.5, scale=0.2) + np.log(0.35),
        gaussian_logpdf(beta_grid, loc=3.0, scale=0.4) + np.log(0.65),
    )
    prior = TabulatedBphiPrior(beta_grid=beta_grid, logpdf=logpdf)
    logposterior = b_marginal_grid_logposterior(
        fnl_grid=fnl_grid,
        beta_grid=beta_grid,
        bfnl_obs=np.array([60.0]),
        bfnl_sigma=np.array([8.0]),
        bphi_priors=[prior],
        fnl_limits=(-100.0, 100.0),
    )
    pdf = normalize_logpdf_on_grid(fnl_grid, logposterior)
    assert np.all(np.isfinite(logposterior))
    assert np.allclose(np.trapz(pdf, fnl_grid), 1.0, atol=1.0e-4)


def test_weighted_sample_bphi_prior_runs_without_ratio_conversion():
    fnl_grid = np.linspace(-100.0, 100.0, 1001)
    samples = np.array([1.0, 1.5, 2.0, 3.5, 4.0])
    weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    prior = WeightedSampleBphiPrior(samples=samples, weights=weights)
    logposterior = b_marginal_weighted_sample_logposterior(
        fnl_grid=fnl_grid,
        bfnl_obs=np.array([40.0]),
        bfnl_sigma=np.array([5.0]),
        bphi_priors=[prior],
        fnl_limits=(-100.0, 100.0),
    )
    pdf = normalize_logpdf_on_grid(fnl_grid, logposterior)
    assert np.all(np.isfinite(logposterior))
    assert np.allclose(np.trapz(pdf, fnl_grid), 1.0, atol=1.0e-4)
    assert fnl_grid[np.argmax(pdf)] > 0.0
```

- [ ] **Step 2: Run tests to verify weighted-sample function fails**

Run:

```bash
pytest tests/test_png_fnl_validation.py::test_tabulated_non_gaussian_bphi_prior_runs_in_grid_mode tests/test_png_fnl_validation.py::test_weighted_sample_bphi_prior_runs_without_ratio_conversion -q
```

Expected: FAIL with `ImportError` for `b_marginal_weighted_sample_logposterior`.

- [ ] **Step 3: Implement weighted-sample marginalization**

Append to `python_scripts/png_fnl_validation.py`:

```python
def _per_bin_fnl_loglike_from_weighted_samples(fnl_grid, bfnl_obs, bfnl_sigma, bphi_prior):
    if not isinstance(bphi_prior, WeightedSampleBphiPrior):
        raise TypeError("weighted-sample marginalization requires WeightedSampleBphiPrior instances.")
    fnl_grid = np.asarray(fnl_grid, dtype="f8")
    output = np.empty_like(fnl_grid)
    log_weights = np.log(bphi_prior.weights)
    for i, fnl in enumerate(fnl_grid):
        model = fnl * bphi_prior.samples
        output[i] = logsumexp(gaussian_logpdf(bfnl_obs, loc=model, scale=bfnl_sigma) + log_weights)
    return output


def b_marginal_weighted_sample_logposterior(
    fnl_grid,
    bfnl_obs,
    bfnl_sigma,
    bphi_priors,
    fnl_limits=(-300.0, 300.0),
):
    bfnl_obs, bfnl_sigma = _validate_independent_bin_inputs(bfnl_obs, bfnl_sigma, bphi_priors)
    loglike = np.zeros_like(np.asarray(fnl_grid, dtype="f8"))
    for obs, sigma, prior in zip(bfnl_obs, bfnl_sigma, bphi_priors):
        loglike += _per_bin_fnl_loglike_from_weighted_samples(fnl_grid, obs, sigma, prior)
    logposterior = loglike + uniform_logprior(fnl_grid, fnl_limits)
    return np.log(normalize_logpdf_on_grid(fnl_grid, logposterior))
```

- [ ] **Step 4: Run non-Gaussian prior tests**

Run:

```bash
pytest tests/test_png_fnl_validation.py::test_tabulated_non_gaussian_bphi_prior_runs_in_grid_mode tests/test_png_fnl_validation.py::test_weighted_sample_bphi_prior_runs_without_ratio_conversion -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add python_scripts/png_fnl_validation.py tests/test_png_fnl_validation.py
git commit -m "test: validate non-gaussian bphi priors"
```

Expected: commit succeeds.

---

### Task 5: Add grid and mcmc CLI Skeleton

**Files:**
- Create: `python_scripts/validate_png_fnl_assumptions.py`
- Modify: `tests/test_png_fnl_validation.py`

- [ ] **Step 1: Add failing CLI smoke tests**

Append to `tests/test_png_fnl_validation.py`:

```python
import subprocess
import sys


def test_validation_cli_help_lists_grid_and_mcmc_modes():
    result = subprocess.run(
        [sys.executable, "python_scripts/validate_png_fnl_assumptions.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "grid" in result.stdout
    assert "mcmc" in result.stdout


def test_validation_cli_grid_mode_writes_report(tmp_path):
    output = tmp_path / "grid_report.txt"
    result = subprocess.run(
        [sys.executable, "python_scripts/validate_png_fnl_assumptions.py", "grid", "--output", str(output)],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    text = output.read_text(encoding="utf-8")
    assert "Grid validation report" in text
    assert "b_direct_vs_b_marginal_max_abs_cdf_diff" in text
```

- [ ] **Step 2: Run CLI tests to verify they fail**

Run:

```bash
pytest tests/test_png_fnl_validation.py::test_validation_cli_help_lists_grid_and_mcmc_modes tests/test_png_fnl_validation.py::test_validation_cli_grid_mode_writes_report -q
```

Expected: FAIL because `python_scripts/validate_png_fnl_assumptions.py` does not exist.

- [ ] **Step 3: Create CLI with grid and mcmc subcommands**

Create `python_scripts/validate_png_fnl_assumptions.py` with:

```python
#!/usr/bin/env python3
"""Validate PNG fnl_loc inference paths."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from png_fnl_validation import (
    GaussianBphiPrior,
    b_direct_grid_logposterior,
    b_marginal_grid_logposterior,
    compare_posteriors,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Validate PNG fnl_loc assumption comparisons.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    grid = subparsers.add_parser("grid", help="Run deterministic quadrature validation without sampler noise.")
    grid.add_argument("--output", default="png_fnl_grid_validation_report.txt")

    mcmc = subparsers.add_parser("mcmc", help="Compare real desilike direct-bphi and bfnl pipeline chains.")
    mcmc.add_argument("--direct-bphi-chain", required=True)
    mcmc.add_argument("--bfnl-chain", required=True)
    mcmc.add_argument("--output", default="png_fnl_mcmc_validation_report.txt")
    return parser


def run_grid_report():
    fnl_grid = np.linspace(-300.0, 300.0, 2401)
    beta_grid = np.linspace(-1.0, 7.0, 2001)
    bfnl_obs = np.array([120.0, 80.0])
    bfnl_sigma = np.array([12.0, 16.0])
    priors = [GaussianBphiPrior(3.0, 0.4), GaussianBphiPrior(2.0, 0.3)]
    direct = b_direct_grid_logposterior(
        fnl_grid=fnl_grid,
        beta_grid=beta_grid,
        bfnl_obs=bfnl_obs,
        bfnl_sigma=bfnl_sigma,
        bphi_priors=priors,
        fnl_limits=(-300.0, 300.0),
    )
    postprocessed = b_marginal_grid_logposterior(
        fnl_grid=fnl_grid,
        beta_grid=beta_grid,
        bfnl_obs=bfnl_obs,
        bfnl_sigma=bfnl_sigma,
        bphi_priors=priors,
        fnl_limits=(-300.0, 300.0),
    )
    metrics = compare_posteriors(fnl_grid, direct, postprocessed)
    return [
        "Grid validation report",
        "mode = grid",
        "purpose = deterministic quadrature without sampler noise",
        f"b_direct_vs_b_marginal_max_abs_cdf_diff = {metrics['max_abs_cdf_diff']:.6e}",
        f"b_direct_vs_b_marginal_mean_abs_pdf_diff = {metrics['mean_abs_pdf_diff']:.6e}",
    ]


def run_mcmc_report(args):
    direct_chain = Path(args.direct_bphi_chain)
    bfnl_chain = Path(args.bfnl_chain)
    if not direct_chain.exists():
        raise FileNotFoundError(f"Direct bphi chain does not exist: {direct_chain}")
    if not bfnl_chain.exists():
        raise FileNotFoundError(f"bfnl chain does not exist: {bfnl_chain}")
    return [
        "MCMC validation report",
        "mode = mcmc",
        "purpose = real desilike chain comparison",
        f"direct_bphi_chain = {direct_chain}",
        f"bfnl_chain = {bfnl_chain}",
        "status = chain paths validated; posterior comparison is implemented in Task 6",
    ]


def main():
    args = build_parser().parse_args()
    if args.mode == "grid":
        lines = run_grid_report()
    elif args.mode == "mcmc":
        lines = run_mcmc_report(args)
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")
    output = Path(args.output)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run CLI tests**

Run:

```bash
pytest tests/test_png_fnl_validation.py::test_validation_cli_help_lists_grid_and_mcmc_modes tests/test_png_fnl_validation.py::test_validation_cli_grid_mode_writes_report -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add python_scripts/validate_png_fnl_assumptions.py tests/test_png_fnl_validation.py
git commit -m "feat: add png fnl validation cli"
```

Expected: commit succeeds.

---

### Task 6: Implement mcmc Chain Comparison for Real desilike Outputs

**Files:**
- Modify: `python_scripts/png_fnl_validation.py`
- Modify: `python_scripts/validate_png_fnl_assumptions.py`
- Modify: `tests/test_png_fnl_validation.py`

- [ ] **Step 1: Add tests for weighted histogram comparison helpers**

Append to `tests/test_png_fnl_validation.py`:

```python
from python_scripts.png_fnl_validation import (
    histogram_logposterior_from_samples,
    posterior_summary,
)


def test_histogram_logposterior_from_samples_normalizes_to_one():
    rng = np.random.default_rng(42)
    samples = rng.normal(loc=10.0, scale=3.0, size=5000)
    grid = np.linspace(-10.0, 30.0, 801)
    logposterior = histogram_logposterior_from_samples(samples=samples, grid=grid)
    pdf = normalize_logpdf_on_grid(grid, logposterior)
    assert np.allclose(np.trapz(pdf, grid), 1.0, atol=2.0e-4)


def test_posterior_summary_reports_mean_and_std():
    grid = np.linspace(-20.0, 20.0, 1601)
    logposterior = gaussian_logpdf(grid, loc=3.0, scale=2.0)
    summary = posterior_summary(grid, logposterior)
    assert abs(summary["mean"] - 3.0) < 0.05
    assert abs(summary["std"] - 2.0) < 0.05
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_png_fnl_validation.py::test_histogram_logposterior_from_samples_normalizes_to_one tests/test_png_fnl_validation.py::test_posterior_summary_reports_mean_and_std -q
```

Expected: FAIL with `ImportError` for the new helper functions.

- [ ] **Step 3: Implement histogram and summary helpers**

Append to `python_scripts/png_fnl_validation.py`:

```python
def histogram_logposterior_from_samples(samples, grid, weights=None):
    samples = np.asarray(samples, dtype="f8")
    grid = np.asarray(grid, dtype="f8")
    if samples.ndim != 1:
        raise ValueError("samples must be one-dimensional.")
    centers = grid
    edges = np.empty(grid.size + 1, dtype="f8")
    edges[1:-1] = 0.5 * (centers[1:] + centers[:-1])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    hist, _ = np.histogram(samples, bins=edges, weights=weights, density=False)
    hist = np.asarray(hist, dtype="f8")
    hist += 0.5
    pdf = hist / np.trapz(hist, grid)
    return np.log(pdf)


def posterior_summary(grid, logposterior):
    grid = np.asarray(grid, dtype="f8")
    pdf = normalize_logpdf_on_grid(grid, logposterior)
    mean = float(np.trapz(grid * pdf, grid))
    var = float(np.trapz((grid - mean) ** 2 * pdf, grid))
    return {"mean": mean, "std": var**0.5}
```

- [ ] **Step 4: Extend the mcmc CLI to load real desilike chains**

Modify `python_scripts/validate_png_fnl_assumptions.py` so `run_mcmc_report(args)` becomes:

```python
def _load_desilike_param(chain_path, basename):
    from desilike.samples import Chain

    chain = Chain.load(str(chain_path))
    matches = [str(param) for param in chain.params() if getattr(param, "basename", None) == basename]
    if not matches:
        raise ValueError(f"No parameter with basename {basename!r} found in {chain_path}")
    if basename == "fnl_loc" and len(matches) == 1:
        return np.asarray(chain[matches[0]], dtype="f8")
    raise ValueError(f"Expected exactly one {basename!r} parameter in {chain_path}; found {matches}")


def run_mcmc_report(args):
    from png_fnl_validation import histogram_logposterior_from_samples, posterior_summary

    direct_chain = Path(args.direct_bphi_chain)
    bfnl_chain = Path(args.bfnl_chain)
    if not direct_chain.exists():
        raise FileNotFoundError(f"Direct bphi chain does not exist: {direct_chain}")
    if not bfnl_chain.exists():
        raise FileNotFoundError(f"bfnl chain does not exist: {bfnl_chain}")
    direct_fnl = _load_desilike_param(direct_chain, "fnl_loc")
    grid = np.linspace(float(np.nanmin(direct_fnl)), float(np.nanmax(direct_fnl)), 1001)
    direct_logposterior = histogram_logposterior_from_samples(direct_fnl, grid=grid)
    direct_summary = posterior_summary(grid, direct_logposterior)
    return [
        "MCMC validation report",
        "mode = mcmc",
        "purpose = real desilike chain comparison",
        f"direct_bphi_chain = {direct_chain}",
        f"bfnl_chain = {bfnl_chain}",
        f"direct_fnl_mean = {direct_summary['mean']:.6f}",
        f"direct_fnl_std = {direct_summary['std']:.6f}",
        "status = direct chain fnl_loc marginal loaded; bfnl post-processing inputs must be added from real bphi prior files in Task 7",
    ]
```

- [ ] **Step 5: Run helper tests**

Run:

```bash
pytest tests/test_png_fnl_validation.py::test_histogram_logposterior_from_samples_normalizes_to_one tests/test_png_fnl_validation.py::test_posterior_summary_reports_mean_and_std -q
```

Expected: `2 passed`.

- [ ] **Step 6: Commit**

Run:

```bash
git add python_scripts/png_fnl_validation.py python_scripts/validate_png_fnl_assumptions.py tests/test_png_fnl_validation.py
git commit -m "feat: add real-chain validation helpers"
```

Expected: commit succeeds.

---

### Task 7: Document Analysis Scope and Validation Strategy

**Files:**
- Modify: `README.md`
- Modify: `python_scripts/README.md`
- Modify: `docs/PROJECT_STATE.md`
- Modify: `docs/superpowers/plans/implementation/fnl_sampling_implementation_validation_plan.md`

- [ ] **Step 1: Document the production data vector and catalog-space convention**

Update the root `README.md` with a short "Analysis Contract" section that states:

- the production PNG data vector is the joint galaxy power-spectrum multipole vector `(ell=0, ell=2)`;
- `ell=0` and `ell=2` are fit jointly, not as separate diagnostics;
- the Gaussian covariance for the joint vector is supplied by `thecov`;
- AbacusHOD LRG catalogs with `want_rsd=True` are already shifted to redshift space by AbacusHOD;
- LRG centrals are placed at halo centers, but velocities include the configured central velocity-bias/scatter term;
- LRG satellites use particle-subsample placement for the primary `use_particles: True` path, with 1-halo velocity contribution controlled by `alpha_s`;
- secondary-redshift AbacusHOD fallback uses NFW satellite placement and must be documented separately;
- ELG/HODDIES RSD handling must be audited before ELG benchmarks are treated as production validation, to avoid double RSD shifts.

- [ ] **Step 2: Append validation strategy notes to python_scripts README**

Append to `python_scripts/README.md`:

```markdown
## fnl_loc Posterior Validation Strategy

The validation distinguishes three benchmark scenarios and two candidate implementations for Scenario B.

- **Scenario A: bfnl baseline** runs `mode="bfnl"` to constrain one `bfnl_z = bphi_z * fnl_loc` amplitude per independent tracer/redshift bin, with no `bphi_z` prior and no universal sampled `fnl_loc`.
- **Scenario B: HOD-informed Gaussian bphi prior** targets one universal `fnl_loc`. Validate both B-direct (`mode="bphi"` sampling of `fnl_loc` and each `bphi_z`) and B-marginal (custom likelihood marginalizing each Gaussian `bphi_z` prior) before choosing the official implementation.
- **Scenario C: fixed-p universality** fixes `bphi_i = 2 delta_c (b1_i - p_i)` with tracer/sample-specific fixed `p_i` values and samples one universal `fnl_loc`.
- Redshift-bin `fnl_loc` likelihood curves are multiplied across bins, and the common `fnl_loc` prior is applied exactly once.
- `grid` mode is deterministic quadrature for checking the transformation math without sampler noise.
- `mcmc` mode compares real desilike pipeline chains only.
- Non-Gaussian `bphi_z` priors are represented by weighted samples or tabulated log-density inputs.
- The production PNG data vector is the joint `(ell=0, ell=2)` galaxy power-spectrum multipole vector with the matching Gaussian `thecov` covariance.
- AbacusHOD LRG `want_rsd=True` catalogs are already in redshift space; downstream code must not apply a second RSD shift.

The main validation command is:

```bash
python python_scripts/validate_png_fnl_assumptions.py grid --output /tmp/png_fnl_grid_validation_report.txt
```
```

- [ ] **Step 3: Update project state**

Update `docs/PROJECT_STATE.md` so its assumptions/scope section no longer says the main PNG pipeline is monopole-only. It must say the target implementation jointly fits redshift-space monopole and quadrupole, and that the Gaussian covariance is provided by `thecov`.

- [ ] **Step 4: Run documentation grep checks**

Run:

```bash
rg -n "Scenario A|Scenario B|Scenario C|ell=0|ell=2|thecov|want_rsd|grid mode|mcmc mode|weighted samples|tabulated log-density" README.md docs/PROJECT_STATE.md docs/superpowers/plans/implementation/fnl_sampling_implementation_validation_plan.md python_scripts/README.md
```

Expected: output contains all listed scope phrases.

- [ ] **Step 5: Commit**

Run:

```bash
git add README.md python_scripts/README.md docs/PROJECT_STATE.md docs/superpowers/plans/implementation/fnl_sampling_implementation_validation_plan.md
git commit -m "docs: document fnl validation scope"
```

Expected: commit succeeds.

---

## Self-Review Checklist for Implementing Agents

Before claiming the plan is complete, verify:

- `grid` mode never invokes a sampler.
- `mcmc` mode does not generate synthetic chains.
- Scenario A has no universal sampled `fnl_loc`; compare it directly only at the individual-bin `bfnl_z` level unless explicitly post-processing with an external `bphi_z` prior.
- B-marginal and any Scenario A post-processing utility never compute an unweighted `bfnl / bphi` sample ratio.
- The `fnl_loc` prior is applied once after multiplying independent redshift-bin likelihood curves.
- Both non-Gaussian `bphi_z` prior representations are covered by tests.
- The primary Scenario B/C comparison metric is the one-dimensional marginalized `fnl_loc` posterior.
- The production PNG likelihood uses joint `(ell=0, ell=2)` galaxy power-spectrum multipoles and the matching `thecov` Gaussian covariance.
- AbacusHOD LRG redshift-space catalogs are not RSD-shifted a second time downstream.

---
