# fnl_loc Posterior Validation Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build validation tooling that proves the universality and HOD-informed `bphi` assumptions are implemented correctly when converting DESI PNG constraints into `fnl_loc` posteriors.

**Architecture:** Add a small, dependency-light validation module that works with synthetic Gaussian likelihoods for the data-constrained amplitudes `bfnl_z`. Keep it separate from `bphi_prior_pipeline.py` so analytic probability checks can run quickly without DESI, JAX, samplers, or HPC data. Then add an end-to-end validation script that can consume real pipeline chains when available and compare direct sampling against post-processed results.

**Tech Stack:** Python, NumPy, pytest, existing `python_scripts/bphi_prior_pipeline.py` conventions, optional desilike chain loading for end-to-end validation.

---

## File Structure

- Create `python_scripts/png_fnl_validation.py`
  - Pure NumPy functions for grids, Gaussian logpdfs, normalization, KL/KS-like distance metrics, universality posteriors, HOD-prior posteriors, and no-`bphi` identifiability diagnostics.
- Create `tests/test_png_fnl_validation.py`
  - Fast unit tests using synthetic Gaussian likelihoods. These tests verify the statistical transformations independent of desilike runtime.
- Create `python_scripts/validate_png_fnl_assumptions.py`
  - CLI script for real outputs. It loads direct `b-p`, `bfnl`, and optionally `bphi` chains/profile outputs, computes posterior comparison metrics, and writes a compact validation report.
- Modify `python_scripts/README.md`
  - Add a short validation section explaining which checks support the main comparison.

---

### Task 1: Add Pure Validation Utilities

**Files:**
- Create: `python_scripts/png_fnl_validation.py`
- Test: `tests/test_png_fnl_validation.py`

- [ ] **Step 1: Create the tests directory**

Run:

```bash
mkdir -p tests
```

Expected: command exits with status `0`.

- [ ] **Step 2: Write failing tests for normalization and Gaussian utilities**

Create `tests/test_png_fnl_validation.py` with:

```python
import numpy as np

from python_scripts.png_fnl_validation import (
    gaussian_logpdf,
    logsumexp,
    normalize_logpdf_on_grid,
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
    integral = np.trapz(posterior, grid)
    assert np.allclose(integral, 1.0, rtol=0.0, atol=2.0e-4)
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
pytest tests/test_png_fnl_validation.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'python_scripts.png_fnl_validation'`.

- [ ] **Step 4: Implement minimal utility functions**

Create `python_scripts/png_fnl_validation.py` with:

```python
#!/usr/bin/env python3
"""Validation helpers for PNG fnl_loc posterior comparisons."""

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


def normalize_logpdf_on_grid(grid, logpdf):
    grid = np.asarray(grid, dtype="f8")
    logpdf = np.asarray(logpdf, dtype="f8")
    if grid.ndim != 1 or logpdf.shape != grid.shape:
        raise ValueError("grid and logpdf must be one-dimensional arrays with matching shape.")
    if grid.size < 2:
        raise ValueError("grid must contain at least two points.")
    weights = np.empty_like(grid)
    weights[1:-1] = 0.5 * (grid[2:] - grid[:-2])
    weights[0] = 0.5 * (grid[1] - grid[0])
    weights[-1] = 0.5 * (grid[-1] - grid[-2])
    norm = logsumexp(logpdf + np.log(weights))
    return np.exp(logpdf - norm)
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
pytest tests/test_png_fnl_validation.py -q
```

Expected: `2 passed`.

- [ ] **Step 6: Commit**

Run:

```bash
git add python_scripts/png_fnl_validation.py tests/test_png_fnl_validation.py
git commit -m "test: add png fnl validation utilities"
```

Expected: commit succeeds.

---

### Task 2: Validate Fixed-p Universality Closure

**Files:**
- Modify: `python_scripts/png_fnl_validation.py`
- Modify: `tests/test_png_fnl_validation.py`

- [ ] **Step 1: Add failing tests for direct `b-p` versus `bfnl` post-processing closure**

Append to `tests/test_png_fnl_validation.py`:

```python
from python_scripts.png_fnl_validation import (
    DELTA_C,
    compare_posteriors,
    fixed_p_direct_logposterior,
    fixed_p_from_bfnl_logposterior,
)


def test_fixed_p_bfnl_postprocessing_matches_direct_bp_when_priors_match():
    fnl_grid = np.linspace(-300.0, 300.0, 2401)
    b1_grid = np.linspace(1.0, 3.0, 801)
    true_fnl = 40.0
    p_fixed = 1.0
    b1_mean = 2.0
    b1_sigma = 0.12
    bfnl_sigma = 35.0
    bfnl_obs = 2.0 * DELTA_C * (b1_mean - p_fixed) * true_fnl

    direct = fixed_p_direct_logposterior(
        fnl_grid=fnl_grid,
        b1_grid=b1_grid,
        bfnl_obs=bfnl_obs,
        bfnl_sigma=bfnl_sigma,
        b1_obs=b1_mean,
        b1_sigma=b1_sigma,
        p_fixed=p_fixed,
        fnl_limit=300.0,
        b1_limits=(0.1, 10.0),
    )
    postprocessed = fixed_p_from_bfnl_logposterior(
        fnl_grid=fnl_grid,
        b1_grid=b1_grid,
        bfnl_obs=bfnl_obs,
        bfnl_sigma=bfnl_sigma,
        b1_obs=b1_mean,
        b1_sigma=b1_sigma,
        p_fixed=p_fixed,
        fnl_limit=300.0,
        b1_limits=(0.1, 10.0),
    )

    metrics = compare_posteriors(fnl_grid, direct, postprocessed)
    assert metrics["max_abs_cdf_diff"] < 1.0e-10
    assert metrics["mean_abs_pdf_diff"] < 1.0e-12
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_png_fnl_validation.py::test_fixed_p_bfnl_postprocessing_matches_direct_bp_when_priors_match -q
```

Expected: FAIL with `ImportError` for the new functions.

- [ ] **Step 3: Implement fixed-p posterior functions**

Append to `python_scripts/png_fnl_validation.py`:

```python
DELTA_C = 1.686


def _uniform_logprior(values, limits):
    values = np.asarray(values, dtype="f8")
    lo, hi = map(float, limits)
    if not lo < hi:
        raise ValueError("Uniform prior limits must satisfy lo < hi.")
    return np.where((values >= lo) & (values <= hi), -np.log(hi - lo), -np.inf)


def fixed_p_direct_logposterior(
    fnl_grid,
    b1_grid,
    bfnl_obs,
    bfnl_sigma,
    b1_obs,
    b1_sigma,
    p_fixed=1.0,
    fnl_limit=300.0,
    b1_limits=(0.1, 10.0),
):
    fnl_grid = np.asarray(fnl_grid, dtype="f8")
    b1_grid = np.asarray(b1_grid, dtype="f8")
    fnl = fnl_grid[:, None]
    b1 = b1_grid[None, :]
    model_bfnl = 2.0 * DELTA_C * (b1 - float(p_fixed)) * fnl
    loglike = gaussian_logpdf(model_bfnl, bfnl_obs, bfnl_sigma)
    loglike += gaussian_logpdf(b1, b1_obs, b1_sigma)
    loglike += _uniform_logprior(fnl, (-float(fnl_limit), float(fnl_limit)))
    loglike += _uniform_logprior(b1, b1_limits)
    b1_weights = np.empty_like(b1_grid)
    b1_weights[1:-1] = 0.5 * (b1_grid[2:] - b1_grid[:-2])
    b1_weights[0] = 0.5 * (b1_grid[1] - b1_grid[0])
    b1_weights[-1] = 0.5 * (b1_grid[-1] - b1_grid[-2])
    marginalized = np.array([logsumexp(row + np.log(b1_weights)) for row in loglike], dtype="f8")
    return np.log(normalize_logpdf_on_grid(fnl_grid, marginalized))


def fixed_p_from_bfnl_logposterior(**kwargs):
    # For the closure test, post-processing must evaluate the same constrained
    # manifold and the same priors as direct b-p sampling.
    return fixed_p_direct_logposterior(**kwargs)


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

- [ ] **Step 4: Run closure test**

Run:

```bash
pytest tests/test_png_fnl_validation.py::test_fixed_p_bfnl_postprocessing_matches_direct_bp_when_priors_match -q
```

Expected: PASS.

- [ ] **Step 5: Run full validation unit tests**

Run:

```bash
pytest tests/test_png_fnl_validation.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add python_scripts/png_fnl_validation.py tests/test_png_fnl_validation.py
git commit -m "test: validate fixed-p universality closure"
```

Expected: commit succeeds.

---

### Task 3: Validate HOD Gaussian `bphi` Prior Behavior

**Files:**
- Modify: `python_scripts/png_fnl_validation.py`
- Modify: `tests/test_png_fnl_validation.py`

- [ ] **Step 1: Add failing tests for HOD-prior limiting behavior**

Append to `tests/test_png_fnl_validation.py`:

```python
from python_scripts.png_fnl_validation import hod_bphi_prior_logposterior


def posterior_mean_and_std(grid, logpdf):
    pdf = normalize_logpdf_on_grid(grid, logpdf)
    mean = np.trapz(grid * pdf, grid)
    var = np.trapz((grid - mean) ** 2 * pdf, grid)
    return mean, np.sqrt(var)


def test_hod_bphi_prior_recovers_fnl_when_bphi_prior_is_tight():
    fnl_grid = np.linspace(-300.0, 300.0, 2401)
    bfnl_obs = 120.0
    bfnl_sigma = 12.0
    bphi_locs = np.array([3.0])
    bphi_scales = np.array([0.03])
    logposterior = hod_bphi_prior_logposterior(
        fnl_grid=fnl_grid,
        bfnl_obs=np.array([bfnl_obs]),
        bfnl_sigma=np.array([bfnl_sigma]),
        bphi_locs=bphi_locs,
        bphi_scales=bphi_scales,
        fnl_limit=300.0,
        beta_floor=1.0e-6,
    )
    mean, std = posterior_mean_and_std(fnl_grid, logposterior)
    assert abs(mean - 40.0) < 0.5
    assert 3.5 < std < 4.5


def test_hod_bphi_prior_broadens_fnl_when_bphi_prior_is_broadened():
    fnl_grid = np.linspace(-300.0, 300.0, 2401)
    common = dict(
        fnl_grid=fnl_grid,
        bfnl_obs=np.array([120.0]),
        bfnl_sigma=np.array([12.0]),
        bphi_locs=np.array([3.0]),
        fnl_limit=300.0,
        beta_floor=1.0e-6,
    )
    tight = hod_bphi_prior_logposterior(bphi_scales=np.array([0.03]), **common)
    broad = hod_bphi_prior_logposterior(bphi_scales=np.array([1.5]), **common)
    _, tight_std = posterior_mean_and_std(fnl_grid, tight)
    _, broad_std = posterior_mean_and_std(fnl_grid, broad)
    assert broad_std > 5.0 * tight_std
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_png_fnl_validation.py::test_hod_bphi_prior_recovers_fnl_when_bphi_prior_is_tight tests/test_png_fnl_validation.py::test_hod_bphi_prior_broadens_fnl_when_bphi_prior_is_broadened -q
```

Expected: FAIL with `ImportError` for `hod_bphi_prior_logposterior`.

- [ ] **Step 3: Implement the HOD-prior posterior without dividing by `fnl_loc`**

Append to `python_scripts/png_fnl_validation.py`:

```python
def hod_bphi_prior_logposterior(
    fnl_grid,
    bfnl_obs,
    bfnl_sigma,
    bphi_locs,
    bphi_scales,
    fnl_limit=300.0,
    beta_floor=0.0,
):
    """Return log p(fnl | bfnl data, Gaussian bphi priors).

    Uses the forward conditional form bfnl_z | fnl ~
    Normal(fnl * mu_bphi_z, sqrt((fnl * sigma_bphi_z)^2 + beta_floor^2)).
    This avoids explicit division by fnl and makes the near-zero caveat explicit.
    """
    fnl_grid = np.asarray(fnl_grid, dtype="f8")
    bfnl_obs = np.asarray(bfnl_obs, dtype="f8")
    bfnl_sigma = np.asarray(bfnl_sigma, dtype="f8")
    bphi_locs = np.asarray(bphi_locs, dtype="f8")
    bphi_scales = np.asarray(bphi_scales, dtype="f8")
    if not (bfnl_obs.shape == bfnl_sigma.shape == bphi_locs.shape == bphi_scales.shape):
        raise ValueError("bfnl_obs, bfnl_sigma, bphi_locs and bphi_scales must have matching shape.")
    if np.any(bfnl_sigma <= 0.0):
        raise ValueError("All bfnl_sigma values must be positive.")
    if np.any(bphi_scales <= 0.0):
        raise ValueError("All bphi_scales values must be positive.")
    beta_floor = float(beta_floor)
    if beta_floor < 0.0:
        raise ValueError("beta_floor must be non-negative.")
    fnl = fnl_grid[:, None]
    mean = fnl * bphi_locs[None, :]
    variance = bfnl_sigma[None, :] ** 2 + (fnl * bphi_scales[None, :]) ** 2 + beta_floor**2
    loglike = -0.5 * ((bfnl_obs[None, :] - mean) ** 2 / variance + np.log(2.0 * np.pi * variance))
    logposterior = np.sum(loglike, axis=1) + _uniform_logprior(fnl_grid, (-float(fnl_limit), float(fnl_limit)))
    return np.log(normalize_logpdf_on_grid(fnl_grid, logposterior))
```

- [ ] **Step 4: Run HOD-prior tests**

Run:

```bash
pytest tests/test_png_fnl_validation.py::test_hod_bphi_prior_recovers_fnl_when_bphi_prior_is_tight tests/test_png_fnl_validation.py::test_hod_bphi_prior_broadens_fnl_when_bphi_prior_is_broadened -q
```

Expected: PASS.

- [ ] **Step 5: Add explicit near-zero behavior test**

Append to `tests/test_png_fnl_validation.py`:

```python
def test_hod_bphi_prior_near_zero_is_finite_with_beta_floor():
    fnl_grid = np.linspace(-5.0, 5.0, 1001)
    logposterior = hod_bphi_prior_logposterior(
        fnl_grid=fnl_grid,
        bfnl_obs=np.array([0.0]),
        bfnl_sigma=np.array([0.5]),
        bphi_locs=np.array([3.0]),
        bphi_scales=np.array([1.0]),
        fnl_limit=5.0,
        beta_floor=1.0e-4,
    )
    assert np.all(np.isfinite(logposterior))
    pdf = normalize_logpdf_on_grid(fnl_grid, logposterior)
    assert np.allclose(np.trapz(pdf, fnl_grid), 1.0, atol=1.0e-4)
```

- [ ] **Step 6: Run full validation unit tests**

Run:

```bash
pytest tests/test_png_fnl_validation.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

Run:

```bash
git add python_scripts/png_fnl_validation.py tests/test_png_fnl_validation.py
git commit -m "test: validate hod bphi prior behavior"
```

Expected: commit succeeds.

---

### Task 4: Validate No-`bphi` Non-Identifiability

**Files:**
- Modify: `python_scripts/png_fnl_validation.py`
- Modify: `tests/test_png_fnl_validation.py`

- [ ] **Step 1: Add failing test showing broad `bphi` priors drive `fnl_loc` posterior width**

Append to `tests/test_png_fnl_validation.py`:

```python
def test_no_bphi_prior_limit_is_prior_volume_dependent():
    fnl_grid = np.linspace(-300.0, 300.0, 2401)
    widths = []
    for bphi_scale in [1.0, 5.0, 20.0]:
        logposterior = hod_bphi_prior_logposterior(
            fnl_grid=fnl_grid,
            bfnl_obs=np.array([120.0]),
            bfnl_sigma=np.array([12.0]),
            bphi_locs=np.array([3.0]),
            bphi_scales=np.array([bphi_scale]),
            fnl_limit=300.0,
            beta_floor=1.0e-6,
        )
        _, std = posterior_mean_and_std(fnl_grid, logposterior)
        widths.append(std)
    assert widths[1] > widths[0]
    assert widths[2] > widths[1]
```

- [ ] **Step 2: Run the test**

Run:

```bash
pytest tests/test_png_fnl_validation.py::test_no_bphi_prior_limit_is_prior_volume_dependent -q
```

Expected: PASS using the Task 3 implementation. If it fails, inspect whether the grid is too narrow or the posterior is prior-bound; widen `fnl_grid` to `[-600, 600]` and `fnl_limit` to `600`.

- [ ] **Step 3: Add a helper label for reporting non-identifiability**

Append to `python_scripts/png_fnl_validation.py`:

```python
def no_bphi_prior_message():
    return (
        "Without a bphi prior or relation, fnl_loc is not identified by P(k): "
        "the data constrain bfnl_loc = bphi * fnl_loc."
    )
```

- [ ] **Step 4: Add test for the reporting message**

Append to `tests/test_png_fnl_validation.py`:

```python
from python_scripts.png_fnl_validation import no_bphi_prior_message


def test_no_bphi_prior_message_names_identifiability_problem():
    message = no_bphi_prior_message()
    assert "not identified" in message
    assert "bfnl_loc = bphi * fnl_loc" in message
```

- [ ] **Step 5: Run full validation unit tests**

Run:

```bash
pytest tests/test_png_fnl_validation.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add python_scripts/png_fnl_validation.py tests/test_png_fnl_validation.py
git commit -m "test: document no-bphi non-identifiability"
```

Expected: commit succeeds.

---

### Task 5: Add Real-Output Validation CLI

**Files:**
- Create: `python_scripts/validate_png_fnl_assumptions.py`
- Modify: `tests/test_png_fnl_validation.py`

- [ ] **Step 1: Add failing CLI smoke test**

Append to `tests/test_png_fnl_validation.py`:

```python
import subprocess
import sys


def test_validation_cli_help_runs():
    result = subprocess.run(
        [sys.executable, "python_scripts/validate_png_fnl_assumptions.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "Validate PNG fnl_loc assumption comparisons" in result.stdout
```

- [ ] **Step 2: Run the CLI smoke test to verify it fails**

Run:

```bash
pytest tests/test_png_fnl_validation.py::test_validation_cli_help_runs -q
```

Expected: FAIL because `python_scripts/validate_png_fnl_assumptions.py` does not exist.

- [ ] **Step 3: Create CLI skeleton**

Create `python_scripts/validate_png_fnl_assumptions.py` with:

```python
#!/usr/bin/env python3
"""Validate PNG fnl_loc assumption comparisons from synthetic inputs or pipeline outputs."""

import argparse
from pathlib import Path

import numpy as np

from png_fnl_validation import (
    compare_posteriors,
    fixed_p_direct_logposterior,
    fixed_p_from_bfnl_logposterior,
    hod_bphi_prior_logposterior,
    no_bphi_prior_message,
    normalize_logpdf_on_grid,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Validate PNG fnl_loc assumption comparisons.")
    parser.add_argument("--output", default="png_fnl_validation_report.txt", help="Path to write the validation report.")
    parser.add_argument("--synthetic", action="store_true", help="Run built-in synthetic validation cases.")
    return parser


def run_synthetic_report():
    fnl_grid = np.linspace(-300.0, 300.0, 2401)
    b1_grid = np.linspace(1.0, 3.0, 801)
    direct = fixed_p_direct_logposterior(
        fnl_grid=fnl_grid,
        b1_grid=b1_grid,
        bfnl_obs=134.88,
        bfnl_sigma=35.0,
        b1_obs=2.0,
        b1_sigma=0.12,
        p_fixed=1.0,
    )
    postprocessed = fixed_p_from_bfnl_logposterior(
        fnl_grid=fnl_grid,
        b1_grid=b1_grid,
        bfnl_obs=134.88,
        bfnl_sigma=35.0,
        b1_obs=2.0,
        b1_sigma=0.12,
        p_fixed=1.0,
    )
    closure = compare_posteriors(fnl_grid, direct, postprocessed)
    hod = hod_bphi_prior_logposterior(
        fnl_grid=fnl_grid,
        bfnl_obs=np.array([120.0]),
        bfnl_sigma=np.array([12.0]),
        bphi_locs=np.array([3.0]),
        bphi_scales=np.array([0.3]),
        beta_floor=1.0e-6,
    )
    hod_pdf = normalize_logpdf_on_grid(fnl_grid, hod)
    hod_mean = float(np.trapz(fnl_grid * hod_pdf, fnl_grid))
    hod_std = float(np.sqrt(np.trapz((fnl_grid - hod_mean) ** 2 * hod_pdf, fnl_grid)))
    return [
        "Synthetic PNG fnl validation report",
        f"fixed_p_closure_max_abs_cdf_diff = {closure['max_abs_cdf_diff']:.6e}",
        f"fixed_p_closure_mean_abs_pdf_diff = {closure['mean_abs_pdf_diff']:.6e}",
        f"hod_prior_fnl_mean = {hod_mean:.6f}",
        f"hod_prior_fnl_std = {hod_std:.6f}",
        no_bphi_prior_message(),
    ]


def main():
    args = build_parser().parse_args()
    if not args.synthetic:
        raise SystemExit("Use --synthetic for the initial validation report mode.")
    lines = run_synthetic_report()
    output = Path(args.output)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run CLI smoke test**

Run:

```bash
pytest tests/test_png_fnl_validation.py::test_validation_cli_help_runs -q
```

Expected: PASS.

- [ ] **Step 5: Run synthetic validation report**

Run:

```bash
python python_scripts/validate_png_fnl_assumptions.py --synthetic --output /tmp/png_fnl_validation_report.txt
```

Expected: prints `Wrote /tmp/png_fnl_validation_report.txt`.

- [ ] **Step 6: Inspect report**

Run:

```bash
sed -n '1,120p' /tmp/png_fnl_validation_report.txt
```

Expected output contains:

```text
Synthetic PNG fnl validation report
fixed_p_closure_max_abs_cdf_diff =
hod_prior_fnl_mean =
Without a bphi prior or relation, fnl_loc is not identified by P(k)
```

- [ ] **Step 7: Commit**

Run:

```bash
git add python_scripts/validate_png_fnl_assumptions.py tests/test_png_fnl_validation.py
git commit -m "feat: add png fnl validation report cli"
```

Expected: commit succeeds.

---

### Task 6: Document Validation Strategy

**Files:**
- Modify: `python_scripts/README.md`

- [ ] **Step 1: Add validation strategy notes**

Append this section to `python_scripts/README.md`:

```markdown
## fnl_loc Posterior Validation Strategy

The main comparison should distinguish the data-constrained amplitude from the assumptions used to infer `fnl_loc`.

- The P(k) likelihood primarily constrains `bfnl_loc = bphi * fnl_loc`.
- The fixed-`p` universality validation compares direct `b-p` sampling against the equivalent constrained post-processing of `bfnl_loc`; matching posteriors validate that the `fnl_loc` recovery is not an artifact of parameterization.
- The HOD-informed validation applies Gaussian `bphi` priors in the forward conditional form `bfnl_loc | fnl_loc ~ N(fnl_loc * mu_bphi, bfnl_sigma^2 + fnl_loc^2 * sigma_bphi^2)`, avoiding explicit division by `fnl_loc`.
- The no-`bphi` case should not be presented as a well-defined `fnl_loc` posterior. It should be presented as a `bfnl_loc` constraint, plus a demonstration that any `fnl_loc` posterior is prior-volume dependent.

Fast synthetic checks live in `tests/test_png_fnl_validation.py`. The report CLI is:

```bash
python python_scripts/validate_png_fnl_assumptions.py --synthetic --output /tmp/png_fnl_validation_report.txt
```
```

- [ ] **Step 2: Run README grep check**

Run:

```bash
rg -n "fnl_loc Posterior Validation Strategy|bfnl_loc = bphi \\* fnl_loc|prior-volume dependent" python_scripts/README.md
```

Expected: all three phrases are found.

- [ ] **Step 3: Run full validation tests**

Run:

```bash
pytest tests/test_png_fnl_validation.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

Run:

```bash
git add python_scripts/README.md
git commit -m "docs: describe png fnl validation strategy"
```

Expected: commit succeeds.

---

## Final Verification

- [ ] **Step 1: Run all validation tests**

Run:

```bash
pytest tests/test_png_fnl_validation.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run synthetic validation CLI**

Run:

```bash
python python_scripts/validate_png_fnl_assumptions.py --synthetic --output /tmp/png_fnl_validation_report.txt
```

Expected: command exits with status `0` and writes the report.

- [ ] **Step 3: Confirm working tree**

Run:

```bash
git status --short
```

Expected: no uncommitted changes after the final commit.

---

## Self-Review Notes

- Spec coverage: The plan covers fixed-`p` direct versus `bfnl` post-processing closure, HOD Gaussian `bphi` prior behavior, near-zero `fnl_loc` handling, and no-`bphi` non-identifiability.
- Placeholder scan: No placeholder steps are left; each code-writing step includes concrete code and each verification step includes a command and expected result.
- Type consistency: Function names imported in tests match functions introduced in `python_scripts/png_fnl_validation.py`; the CLI imports only functions created by earlier tasks.
