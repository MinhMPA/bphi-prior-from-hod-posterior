# python_scripts Refactor and Hygiene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `python_scripts/` less brittle by centralizing paths/configuration, removing duplicated pipeline logic, converting raw `sys.argv` entrypoints to `argparse`, and adding fast tests that validate behavior without requiring NERSC data.

**Architecture:** Keep Slurm/HPC shell files as thin entrypoints and keep research workflows script-friendly. Add a small `python_scripts/pnglib/` helper package with pure functions and dataclasses for paths, tracer/redshift metadata, PNG file IO, and bphi stats parsing. Avoid a class-heavy redesign; use modules, dataclasses, and tested functions so existing scripts can migrate incrementally.

**Tech Stack:** Python standard library, NumPy, pytest, bash syntax checks, existing desilike/HOD scripts.

---

## Scope Boundaries

This plan changes structure and hygiene only. It must not change the scientific definitions of `bphi`, `bfnl`, `b-p`, HOD LHC priors, covariance construction, or plotting outputs except where explicitly called out as a bug fix.

This plan deliberately does not create a full installable package. The code remains runnable from `python_scripts/` on NERSC, while tests can import helper modules locally.

---

## File Structure

- Create `python_scripts/pnglib/__init__.py`
  - Marks the helper directory as an importable package.
- Create `python_scripts/pnglib/paths.py`
  - Centralized path/config dataclass and environment overrides.
- Create `python_scripts/pnglib/tracers.py`
  - Centralized tracer choices, redshift-bin metadata, cosmology tag normalization, and simtype defaults.
- Create `python_scripts/pnglib/pk_io.py`
  - Shared `ztag`, P(k) path builders, SU text P(k) loader, PNG rebinned P(k) loader, covariance path builder.
- Create `python_scripts/pnglib/bphi_stats.py`
  - Shared bphi stats parser with explicit missing-file behavior.
- Create `tests/test_pnglib_paths.py`
  - Unit tests for path overrides and path defaults.
- Create `tests/test_pnglib_tracers.py`
  - Unit tests for redshift metadata, choices, and cosmology normalization.
- Create `tests/test_pnglib_pk_io.py`
  - Unit tests for text P(k) loading and path generation.
- Create `tests/test_pnglib_bphi_stats.py`
  - Unit tests for bphi stats parsing and missing-file behavior.
- Create `tests/test_cli_hygiene.py`
  - Static/import tests ensuring converted scripts have `build_parser()`/`main()` and do not execute on import.
- Modify `python_scripts/bphi_prior_pipeline.py`
  - Use shared modules, avoid requiring bphi stats for no-prior runs, and remove duplicated path/IO code.
- Modify `python_scripts/bphi_prior_plot_pk.py`
  - Use shared P(k)/covariance path and loader functions.
- Modify `python_scripts/bphi_prior_pipeline_bfnl.py`
  - Replace duplicated implementation with a compatibility wrapper around `bphi_prior_pipeline.py`.
- Modify `python_scripts/bphi_report.py`
  - Convert raw `sys.argv` to `argparse`.
- Modify `python_scripts/lowS8_comparison.py`
  - Convert raw `sys.argv` to `argparse` and centralize default paths.
- Modify `python_scripts/HOD_LHC.py`
  - Add `argparse`, raise unsupported-input errors correctly, keep `HOD_lhc()` importable.
- Modify `python_scripts/bphi_prior_HOD.py`
  - Convert raw `sys.argv`, centralize path defaults, and raise unsupported-input errors correctly.
- Modify `python_scripts/AbacusHOD_lowS8.py`
  - Convert raw `sys.argv`, centralize HODDIES path default.
- Modify `python_scripts/bphi_dz.py`
  - Move top-level execution into `main()` and add parser for input/output roots.
- Modify `python_scripts/HOD_one_lhc.py`
  - Add explicit `--allow-partial` behavior so clustering failures fail by default.
- Modify `python_scripts/preproc_parallel.sh`
  - Fix `$NEWCFGT` typo and centralize `BPHI_ROOT`.
- Modify `python_scripts/bphi_prior_pipeline.sh`
  - Use environment-overridable paths and run both with/without bphi priors unless intentionally disabled.
- Modify `python_scripts/README.md`
  - Document path environment variables, CLI conventions, and the test suite.

---

### Task 1: Create Importable Helper Package and Path Configuration

**Files:**
- Create: `python_scripts/pnglib/__init__.py`
- Create: `python_scripts/pnglib/paths.py`
- Create: `tests/test_pnglib_paths.py`

- [ ] **Step 1: Create the test file first**

Create `tests/test_pnglib_paths.py`:

```python
from pathlib import Path

from python_scripts.pnglib.paths import PathConfig, default_path_config, resolve_env_path


def test_resolve_env_path_uses_default_when_env_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("BPHI_TEST_PATH", raising=False)
    default = tmp_path / "default"
    assert resolve_env_path("BPHI_TEST_PATH", default, base=tmp_path) == default


def test_resolve_env_path_expands_relative_env_against_base(monkeypatch, tmp_path):
    monkeypatch.setenv("BPHI_TEST_PATH", "relative/output")
    assert resolve_env_path("BPHI_TEST_PATH", tmp_path / "fallback", base=tmp_path) == tmp_path / "relative/output"


def test_resolve_env_path_keeps_absolute_env(monkeypatch, tmp_path):
    absolute = tmp_path / "absolute-output"
    monkeypatch.setenv("BPHI_TEST_PATH", str(absolute))
    assert resolve_env_path("BPHI_TEST_PATH", tmp_path / "fallback", base=tmp_path) == absolute


def test_default_path_config_uses_script_dir_defaults(monkeypatch):
    monkeypatch.delenv("BPHI_ROOT", raising=False)
    monkeypatch.delenv("BPHI_FIT_DIR", raising=False)
    config = default_path_config()
    assert config.script_dir.name == "python_scripts"
    assert config.pk_dir == config.script_dir / "bphi_prior" / "pk"
    assert config.cov_dir == config.script_dir / "bphi_prior" / "cov"
    assert config.fit_dir == config.script_dir / "bphi_prior" / "fit"
    assert config.bphi_stats_file == config.script_dir / "backup" / "bphi_AbacusHOD_LRG.txt"


def test_path_config_from_env_overrides_roots(monkeypatch, tmp_path):
    monkeypatch.setenv("BPHI_ROOT", str(tmp_path))
    monkeypatch.setenv("BPHI_FIT_DIR", "custom-fit")
    config = default_path_config()
    assert config.root == tmp_path
    assert config.fit_dir == tmp_path / "custom-fit"
    assert config.pk_dir == tmp_path / "bphi_prior" / "pk"
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:

```bash
pytest tests/test_pnglib_paths.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'python_scripts.pnglib'`.

- [ ] **Step 3: Create package marker**

Create `python_scripts/pnglib/__init__.py`:

```python
"""Shared helpers for local PNG / bphi analysis scripts."""
```

- [ ] **Step 4: Implement centralized path config**

Create `python_scripts/pnglib/paths.py`:

```python
from dataclasses import dataclass
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_DIR.parent


def resolve_env_path(name, default, base=None):
    value = os.environ.get(name)
    default = Path(default)
    if not value:
        return default
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(base or SCRIPT_DIR) / path
    return path


@dataclass(frozen=True)
class PathConfig:
    script_dir: Path
    repo_root: Path
    root: Path
    pk_dir: Path
    cov_dir: Path
    plot_dir: Path
    fit_dir: Path
    png_mock_dir: Path
    bphi_stats_file: Path
    hod_config_dir: Path
    scratch_root: Path
    desilike_local_path: Path | None = None
    hoddies_path: Path | None = None


def default_path_config():
    root = resolve_env_path("BPHI_ROOT", SCRIPT_DIR, base=SCRIPT_DIR)
    scratch_root = resolve_env_path("BPHI_SCRATCH_ROOT", "/pscratch/sd/j/jiaxi/b_phi", base=root)
    desilike_value = os.environ.get("DESILIKE_LOCAL_PATH")
    hoddies_value = os.environ.get("HODDIES_PATH", "/global/homes/j/jiaxi/codes/HODDIES")
    return PathConfig(
        script_dir=SCRIPT_DIR,
        repo_root=REPO_ROOT,
        root=root,
        pk_dir=resolve_env_path("BPHI_PK_DIR", root / "bphi_prior" / "pk", base=root),
        cov_dir=resolve_env_path("BPHI_COV_DIR", root / "bphi_prior" / "cov", base=root),
        plot_dir=resolve_env_path("BPHI_PLOT_DIR", root / "bphi_prior" / "plots", base=root),
        fit_dir=resolve_env_path("BPHI_FIT_DIR", root / "bphi_prior" / "fit", base=root),
        png_mock_dir=resolve_env_path("BPHI_PNG_MOCK_DIR", root / "mocks_lhc_PNG", base=root),
        bphi_stats_file=resolve_env_path(
            "BPHI_STATS_FILE",
            root / "backup" / "bphi_AbacusHOD_LRG.txt",
            base=root,
        ),
        hod_config_dir=resolve_env_path("BPHI_HOD_CONFIG_DIR", root / "HOD_config", base=root),
        scratch_root=scratch_root,
        desilike_local_path=Path(desilike_value).expanduser() if desilike_value else None,
        hoddies_path=Path(hoddies_value).expanduser() if hoddies_value else None,
    )
```

- [ ] **Step 5: Run path tests**

Run:

```bash
pytest tests/test_pnglib_paths.py -q
```

Expected: all tests in `tests/test_pnglib_paths.py` pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add python_scripts/pnglib/__init__.py python_scripts/pnglib/paths.py tests/test_pnglib_paths.py
git commit -m "refactor: centralize python_scripts path config"
```

Expected: commit succeeds.

---

### Task 2: Centralize Tracer, Redshift, and Cosmology Metadata

**Files:**
- Create: `python_scripts/pnglib/tracers.py`
- Create: `tests/test_pnglib_tracers.py`

- [ ] **Step 1: Write failing metadata tests**

Create `tests/test_pnglib_tracers.py`:

```python
import pytest

from python_scripts.pnglib.tracers import (
    PNG_MODES,
    SIMTYPES,
    redshift_bins,
    simtype_cosmos,
    normalize_cosmo,
)


def test_known_mode_and_simtype_lists_are_stable():
    assert PNG_MODES == ("bphi", "b-p", "bfnl")
    assert SIMTYPES == ("SU", "PNG")


def test_lrg_redshift_bins_include_primary_bins_and_secondary_bin():
    bins = redshift_bins("LRG")
    assert [(item.zmin, item.zmax, item.zsnap) for item in bins] == [
        (0.4, 0.6, 0.5),
        (0.6, 0.8, 0.8),
        (0.8, 1.1, 1.1),
        (0.8, 1.1, 0.95),
    ]


def test_unknown_tracer_raises_value_error():
    with pytest.raises(ValueError, match="Unknown tracer"):
        redshift_bins("BAD")


def test_cosmology_normalization():
    assert normalize_cosmo("c4") == "004"
    assert normalize_cosmo("004") == "004"
    assert normalize_cosmo(300) == "300"


def test_simtype_cosmos_defaults():
    assert simtype_cosmos("SU") == ("000", "004")
    assert simtype_cosmos("PNG") == ("000", "300", "301")
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:

```bash
pytest tests/test_pnglib_tracers.py -q
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `python_scripts.pnglib.tracers`.

- [ ] **Step 3: Implement metadata module**

Create `python_scripts/pnglib/tracers.py`:

```python
from dataclasses import dataclass


TRACERS = ("LRG", "ELG", "QSO")
SIMTYPES = ("SU", "PNG")
PNG_MODES = ("bphi", "b-p", "bfnl")


@dataclass(frozen=True)
class RedshiftBin:
    zmin: float
    zmax: float
    zsnap: float


_REDSHIFT_BINS = {
    "LRG": (
        RedshiftBin(0.4, 0.6, 0.5),
        RedshiftBin(0.6, 0.8, 0.8),
        RedshiftBin(0.8, 1.1, 1.1),
        RedshiftBin(0.8, 1.1, 0.95),
    ),
    "ELG": (
        RedshiftBin(0.8, 1.1, 0.95),
        RedshiftBin(1.1, 1.6, 1.325),
    ),
    "QSO": (
        RedshiftBin(0.8, 1.1, 1.1),
        RedshiftBin(1.1, 1.6, 1.325),
        RedshiftBin(1.6, 2.1, 1.85),
        RedshiftBin(2.1, 3.5, 2.5),
    ),
}


def redshift_bins(tracer):
    tracer = str(tracer).upper()
    try:
        return _REDSHIFT_BINS[tracer]
    except KeyError as exc:
        raise ValueError(f"Unknown tracer '{tracer}'. Choices are {TRACERS}.") from exc


def normalize_cosmo(cosmo):
    tag = str(cosmo).lower().removeprefix("c")
    if not tag.isdigit():
        raise ValueError(f"Cosmology tag must be numeric after optional 'c' prefix, got {cosmo!r}.")
    return tag.zfill(3)


def simtype_cosmos(simtype):
    simtype = str(simtype).upper()
    if simtype == "SU":
        return ("000", "004")
    if simtype == "PNG":
        return ("000", "300", "301")
    raise ValueError(f"Unknown simtype '{simtype}'. Choices are {SIMTYPES}.")
```

- [ ] **Step 4: Run metadata tests**

Run:

```bash
pytest tests/test_pnglib_tracers.py -q
```

Expected: all tests in `tests/test_pnglib_tracers.py` pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add python_scripts/pnglib/tracers.py tests/test_pnglib_tracers.py
git commit -m "refactor: centralize tracer metadata"
```

Expected: commit succeeds.

---

### Task 3: Extract Shared P(k), Covariance, and bphi Stats IO

**Files:**
- Create: `python_scripts/pnglib/pk_io.py`
- Create: `python_scripts/pnglib/bphi_stats.py`
- Create: `tests/test_pnglib_pk_io.py`
- Create: `tests/test_pnglib_bphi_stats.py`

- [ ] **Step 1: Write failing P(k) IO tests**

Create `tests/test_pnglib_pk_io.py`:

```python
import numpy as np

from python_scripts.pnglib.paths import PathConfig
from python_scripts.pnglib.pk_io import (
    default_cov_path,
    default_pk_path,
    load_su_text_pk,
    ztag,
)


def make_config(tmp_path):
    return PathConfig(
        script_dir=tmp_path,
        repo_root=tmp_path,
        root=tmp_path,
        pk_dir=tmp_path / "pk",
        cov_dir=tmp_path / "cov",
        plot_dir=tmp_path / "plots",
        fit_dir=tmp_path / "fit",
        png_mock_dir=tmp_path / "png",
        bphi_stats_file=tmp_path / "stats.txt",
        hod_config_dir=tmp_path / "HOD_config",
        scratch_root=tmp_path,
    )


def test_ztag_formats_three_decimal_places():
    assert ztag(0.5) == "0.500"
    assert ztag(0.95) == "0.950"


def test_default_paths_use_config_roots(tmp_path):
    config = make_config(tmp_path)
    assert default_pk_path(config, "LRG", 0.5, cosmo="000", simtype="SU") == tmp_path / "pk" / "pk_LRG_z0.500_c000.txt_pypower"
    assert default_pk_path(config, "LRG", 0.5, cosmo="300", simtype="PNG") == tmp_path / "png" / "LRG_z0.500_c300_default_pk_RSD.npy"
    assert default_cov_path(config, "LRG", 0.5, cosmo="300", simtype="PNG") == tmp_path / "cov" / "cov_PNG_LRG_z0.500_c300.npy"


def test_load_su_text_pk_reads_header_and_columns(tmp_path):
    path = tmp_path / "pk.txt"
    path.write_text(
        "# data_size1 = 8000\n"
        "# shotnoise = 123.5\n"
        "10 0.0 0.01 1000.0\n"
        "20 0.0 0.02 1100.0\n",
        encoding="utf-8",
    )
    data = load_su_text_pk(path, boxsize=20.0)
    assert np.allclose(data["k"], [0.01, 0.02])
    assert np.allclose(data["pk0"], [1000.0, 1100.0])
    assert np.allclose(data["nmodes"], [10.0, 20.0])
    assert data["shotnoise"] == 123.5
    assert data["nbar"] == 1.0
```

- [ ] **Step 2: Write failing bphi stats tests**

Create `tests/test_pnglib_bphi_stats.py`:

```python
import numpy as np
import pytest

from python_scripts.pnglib.bphi_stats import load_bphi_stats


def test_load_bphi_stats_uses_first_row_as_reference_and_std_as_width(tmp_path):
    path = tmp_path / "bphi.txt"
    path.write_text(
        "# tracer zsnap HOD_id Ngal_c000 Ngal_c004 bphi b1\n"
        "LRG 0.500 0 100 90 1.5 2.1\n"
        "LRG 0.500 1 101 91 2.5 2.3\n",
        encoding="utf-8",
    )
    stats = load_bphi_stats(path, required=True)
    item = stats[("LRG", 0.5)]
    assert item["mean"] == 1.5
    assert np.allclose(item["std"], 0.5)
    assert item["n"] == 2
    assert item["b1"] == 2.1


def test_missing_bphi_stats_can_be_optional(tmp_path):
    assert load_bphi_stats(tmp_path / "missing.txt", required=False) == {}


def test_missing_bphi_stats_required_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing bphi stats file"):
        load_bphi_stats(tmp_path / "missing.txt", required=True)
```

- [ ] **Step 3: Run new tests and confirm they fail**

Run:

```bash
pytest tests/test_pnglib_pk_io.py tests/test_pnglib_bphi_stats.py -q
```

Expected: FAIL with import errors for `pk_io` and `bphi_stats`.

- [ ] **Step 4: Implement shared P(k) IO module**

Create `python_scripts/pnglib/pk_io.py`:

```python
import numpy as np

from .tracers import normalize_cosmo


BOXSIZE = 2000.0


def ztag(zsnap):
    return f"{float(zsnap):.3f}"


def default_pk_path(config, tracer, zsnap, cosmo="000", simtype="SU"):
    cosmo = normalize_cosmo(cosmo)
    if simtype == "SU":
        return config.pk_dir / f"pk_{tracer}_z{ztag(zsnap)}_c{cosmo}.txt_pypower"
    if simtype == "PNG":
        return config.png_mock_dir / f"{tracer}_z{ztag(zsnap)}_c{cosmo}_default_pk_RSD.npy"
    raise ValueError("simtype must be 'SU' or 'PNG'.")


def default_cov_path(config, tracer, zsnap, cosmo="000", simtype="SU"):
    cosmo = normalize_cosmo(cosmo)
    if simtype == "SU":
        return config.cov_dir / f"cov_{tracer}_z{ztag(zsnap)}_c{cosmo}.npy"
    if simtype == "PNG":
        return config.cov_dir / f"cov_PNG_{tracer}_z{ztag(zsnap)}_c{cosmo}.npy"
    raise ValueError("simtype must be 'SU' or 'PNG'.")


def load_su_text_pk(path, boxsize=BOXSIZE):
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
        "nbar": data_size / float(boxsize) ** 3,
        "boxsize": float(boxsize),
    }


def load_png_rebinned_pk(path, kmin=0.0, kmax=0.1, dk=0.002, boxsize=BOXSIZE):
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
        if np.any(mask):
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
        "nbar": data_size / float(boxsize) ** 3,
        "boxsize": float(boxsize),
    }


def load_pk(config, tracer, zsnap, cosmo="000", simtype="SU"):
    path = default_pk_path(config, tracer, zsnap, cosmo=cosmo, simtype=simtype)
    if simtype == "SU":
        return load_su_text_pk(path)
    return load_png_rebinned_pk(path)
```

- [ ] **Step 5: Implement shared bphi stats module**

Create `python_scripts/pnglib/bphi_stats.py`:

```python
import numpy as np


def load_bphi_stats(path, required=True):
    path = path if hasattr(path, "exists") else __import__("pathlib").Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing bphi stats file: {path}")
        return {}
    stats = {}
    b1_stats = {}
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"Expected at least 6 columns in {path}, got line: {line}")
            tracer, zsnap, _, _, _, bphi = parts[:6]
            key = (tracer, float(zsnap))
            stats.setdefault(key, []).append(float(bphi))
            if len(parts) >= 7:
                b1 = float(parts[6])
                if np.isfinite(b1):
                    b1_stats.setdefault(key, []).append(b1)
    summary = {}
    for key, values in stats.items():
        arr = np.asarray(values, dtype="f8")
        item = {"mean": float(arr[0]), "std": float(np.std(arr)), "n": int(arr.size)}
        if key in b1_stats:
            b1 = np.asarray(b1_stats[key], dtype="f8")
            item.update({"b1": float(b1[0]), "b1_std": float(np.std(b1)), "b1_n": int(b1.size)})
        summary[key] = item
    return summary
```

- [ ] **Step 6: Run IO tests**

Run:

```bash
pytest tests/test_pnglib_pk_io.py tests/test_pnglib_bphi_stats.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

Run:

```bash
git add python_scripts/pnglib/pk_io.py python_scripts/pnglib/bphi_stats.py tests/test_pnglib_pk_io.py tests/test_pnglib_bphi_stats.py
git commit -m "refactor: extract shared png io helpers"
```

Expected: commit succeeds.

---

### Task 4: Refactor Main PNG Pipeline to Shared IO and Fix No-Prior Stats Dependency

**Files:**
- Modify: `python_scripts/bphi_prior_pipeline.py`
- Modify: `tests/test_pnglib_bphi_stats.py`

- [ ] **Step 1: Add regression test for optional bphi stats**

Append to `tests/test_pnglib_bphi_stats.py`:

```python
from python_scripts.pnglib.bphi_stats import stats_required_for_fit


def test_stats_required_for_fit_only_when_prior_or_truth_needs_it():
    assert stats_required_for_fit(mode="bphi", use_bphi_prior=True)
    assert stats_required_for_fit(mode="bfnl", use_bphi_prior=True)
    assert not stats_required_for_fit(mode="bphi", use_bphi_prior=False)
    assert not stats_required_for_fit(mode="bfnl", use_bphi_prior=False)
    assert not stats_required_for_fit(mode="b-p", use_bphi_prior=False)
```

- [ ] **Step 2: Run regression test and confirm it fails**

Run:

```bash
pytest tests/test_pnglib_bphi_stats.py::test_stats_required_for_fit_only_when_prior_or_truth_needs_it -q
```

Expected: FAIL with `ImportError` for `stats_required_for_fit`.

- [ ] **Step 3: Implement stats requirement helper**

Append to `python_scripts/pnglib/bphi_stats.py`:

```python
def stats_required_for_fit(mode, use_bphi_prior):
    return bool(use_bphi_prior and mode in {"bphi", "bfnl"})
```

- [ ] **Step 4: Replace imports and constants in `bphi_prior_pipeline.py`**

Modify the import/constant section of `python_scripts/bphi_prior_pipeline.py` to use shared modules:

```python
#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import numpy as np

from pnglib.bphi_stats import load_bphi_stats as load_bphi_stats_file
from pnglib.bphi_stats import stats_required_for_fit
from pnglib.paths import default_path_config
from pnglib.pk_io import BOXSIZE, default_cov_path, default_pk_path, load_pk, ztag
from pnglib.tracers import PNG_MODES, SIMTYPES, normalize_cosmo


CONFIG = default_path_config()
ROOT = CONFIG.root
PK_DIR = CONFIG.pk_dir
COV_DIR = CONFIG.cov_dir
PNG_MOCK_DIR = CONFIG.png_mock_dir
BPHI_STATS_FILE = CONFIG.bphi_stats_file
BOXSIZE = BOXSIZE
DEFAULT_NTHREAD = 32
DEFAULT_ZS = [0.5, 0.8, 1.1]
PRIORS_FLAT = {"bphi": 10, "fnl_loc": 300}
BPHI_BFNL_PRIOR_NGRID = 4096
TRUE_FNL_LOC_BY_COSMO = {
    "000": 0.0,
    "300": 30.0,
    "302": -30.0,
    "303": 100.0,
    "304": -100.0,
}


FIT_DIR = CONFIG.fit_dir
```

Remove duplicate local definitions of `ztag`, `su_text_pk_path`, `png_mock_pk_path`, `default_pk_path`, `default_cov_path`, `load_su_text_pk`, `load_png_rebinned_pk`, and `load_pk` from `bphi_prior_pipeline.py`.

- [ ] **Step 5: Update path function calls in `bphi_prior_pipeline.py`**

Replace calls of the old signatures:

```python
default_pk_path(args.tracer, zsnap, cosmo=args.cosmo, simtype=args.simtype)
default_cov_path(tracer, zsnap, cosmo=cosmo, simtype=simtype)
load_pk(tracer, zsnap, cosmo=cosmo, simtype=simtype)
```

with:

```python
default_pk_path(CONFIG, args.tracer, zsnap, cosmo=args.cosmo, simtype=args.simtype)
default_cov_path(CONFIG, tracer, zsnap, cosmo=cosmo, simtype=simtype)
load_pk(CONFIG, tracer, zsnap, cosmo=cosmo, simtype=simtype)
```

- [ ] **Step 6: Replace local `load_bphi_stats()` wrapper**

Replace the body of `load_bphi_stats()` in `bphi_prior_pipeline.py` with:

```python
def load_bphi_stats(path=BPHI_STATS_FILE, required=True):
    return load_bphi_stats_file(path, required=required)
```

- [ ] **Step 7: Fix no-prior stats dependency in `build_joint_png_likelihood()`**

In `build_joint_png_likelihood()`, replace:

```python
bphi_stats = load_bphi_stats()
```

with:

```python
bphi_stats = load_bphi_stats(required=stats_required_for_fit(mode, use_bphi_prior))
```

Then inside the redshift loop, replace:

```python
bphi_summary = bphi_stats[(tracer, zsnap)]
```

with:

```python
bphi_summary = bphi_stats.get((tracer, zsnap))
if stats_required_for_fit(mode, use_bphi_prior) and bphi_summary is None:
    raise KeyError(f"Missing bphi stats for tracer={tracer} zsnap={zsnap:.3f}")
```

Only use `bphi_summary["mean"]` and `bphi_summary["std"]` inside branches where stats are required.

- [ ] **Step 8: Run focused tests**

Run:

```bash
pytest tests/test_pnglib_bphi_stats.py tests/test_pnglib_pk_io.py -q
```

Expected: all tests pass.

- [ ] **Step 9: Compile the pipeline**

Run:

```bash
python3 -m py_compile python_scripts/bphi_prior_pipeline.py
```

Expected: command exits with status `0`.

- [ ] **Step 10: Commit**

Run:

```bash
git add python_scripts/bphi_prior_pipeline.py python_scripts/pnglib/bphi_stats.py tests/test_pnglib_bphi_stats.py
git commit -m "fix: make bphi stats optional for no-prior fits"
```

Expected: commit succeeds.

---

### Task 5: Refactor Plot-Pk Script to Shared IO

**Files:**
- Modify: `python_scripts/bphi_prior_plot_pk.py`

- [ ] **Step 1: Replace duplicated path and P(k) helpers**

Modify the top of `python_scripts/bphi_prior_plot_pk.py` to import shared helpers:

```python
#!/usr/bin/env python3
import argparse
import importlib.util

import matplotlib.pyplot as plt
import numpy as np

from pnglib.paths import default_path_config
from pnglib.pk_io import default_cov_path, load_pk, ztag
from pnglib.tracers import SIMTYPES


CONFIG = default_path_config()
PLOT_DIR = CONFIG.plot_dir
BAOCOV_FILE = CONFIG.repo_root / "PFS_mock-tests" / "baocov.py"
```

Remove duplicate definitions of `ROOT`, `PK_DIR`, `COV_DIR`, `PNG_MOCK_DIR`, `SIMTYPES`, `BOXSIZE`, `ztag`, `su_text_pk_path`, `png_mock_pk_path`, `pk_path`, `cov_path`, `load_su_text_pk`, `load_png_rebinned_pk`, and `load_pk`.

- [ ] **Step 2: Update path calls in `plot_one()`**

Replace:

```python
pk_data = load_pk(tracer, zsnap, cosmo=cosmo, simtype=simtype)
covariance = np.load(cov_path(tracer, zsnap, cosmo=cosmo, simtype=simtype))
```

with:

```python
pk_data = load_pk(CONFIG, tracer, zsnap, cosmo=cosmo, simtype=simtype)
covariance = np.load(default_cov_path(CONFIG, tracer, zsnap, cosmo=cosmo, simtype=simtype))
```

- [ ] **Step 3: Compile script**

Run:

```bash
python3 -m py_compile python_scripts/bphi_prior_plot_pk.py
```

Expected: command exits with status `0`.

- [ ] **Step 4: Run shared tests**

Run:

```bash
pytest tests/test_pnglib_pk_io.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add python_scripts/bphi_prior_plot_pk.py
git commit -m "refactor: reuse shared pk io in plot script"
```

Expected: commit succeeds.

---

### Task 6: Replace Duplicate bfnl Pipeline with Compatibility Wrapper

**Files:**
- Modify: `python_scripts/bphi_prior_pipeline_bfnl.py`
- Modify: `python_scripts/bphi_prior_pipeline_bfnl.sh`
- Create: `tests/test_pipeline_wrapper.py`

- [ ] **Step 1: Write wrapper smoke test**

Create `tests/test_pipeline_wrapper.py`:

```python
import subprocess
import sys


def test_bfnl_pipeline_wrapper_help_runs():
    result = subprocess.run(
        [sys.executable, "python_scripts/bphi_prior_pipeline_bfnl.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "Pipeline for bphi-prior" in result.stdout
```

- [ ] **Step 2: Replace `_bfnl` duplicate with wrapper**

Replace the full contents of `python_scripts/bphi_prior_pipeline_bfnl.py` with:

```python
#!/usr/bin/env python3
"""Compatibility wrapper for the unified PNG/bphi pipeline.

Use python_scripts/bphi_prior_pipeline.py directly for new runs.
"""

from bphi_prior_pipeline import main


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Update shell wrapper to call unified pipeline**

In `python_scripts/bphi_prior_pipeline_bfnl.sh`, replace both occurrences of:

```bash
python bphi_prior_pipeline_bfnl.py fit
```

with:

```bash
python bphi_prior_pipeline.py fit
```

- [ ] **Step 4: Run wrapper smoke test**

Run:

```bash
pytest tests/test_pipeline_wrapper.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Compile both pipeline files**

Run:

```bash
python3 -m py_compile python_scripts/bphi_prior_pipeline.py python_scripts/bphi_prior_pipeline_bfnl.py
```

Expected: command exits with status `0`.

- [ ] **Step 6: Commit**

Run:

```bash
git add python_scripts/bphi_prior_pipeline_bfnl.py python_scripts/bphi_prior_pipeline_bfnl.sh tests/test_pipeline_wrapper.py
git commit -m "refactor: replace duplicate bfnl pipeline with wrapper"
```

Expected: commit succeeds.

---

### Task 7: Convert `bphi_report.py` to argparse and Import-Safe Main

**Files:**
- Modify: `python_scripts/bphi_report.py`
- Create: `tests/test_cli_hygiene.py`

- [ ] **Step 1: Add CLI hygiene tests**

Create `tests/test_cli_hygiene.py`:

```python
import importlib
import subprocess
import sys


def test_bphi_report_import_is_safe():
    module = importlib.import_module("python_scripts.bphi_report")
    assert hasattr(module, "build_parser")
    assert hasattr(module, "main")


def test_bphi_report_help_runs():
    result = subprocess.run(
        [sys.executable, "python_scripts/bphi_report.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "Summarize bphi result tables" in result.stdout
```

- [ ] **Step 2: Run tests and confirm import-safety failure**

Run:

```bash
pytest tests/test_cli_hygiene.py::test_bphi_report_import_is_safe -q
```

Expected: FAIL because importing `bphi_report.py` currently reads `sys.argv`.

- [ ] **Step 3: Refactor `bphi_report.py`**

Replace `python_scripts/bphi_report.py` with:

```python
#!/usr/bin/env python3
import argparse

import numpy as np
from astropy.table import Table
from cosmoprimo.fiducial import AbacusSummit
from scipy.special import hyp2f1


def growth_factor_approx(z):
    cosmo = AbacusSummit()
    return 1.0 / (z + 1.0) * hyp2f1(1.0 / 3.0, 1.0, 11.0 / 6.0, (1.0 - 1.0 / cosmo.Omega0_m) / (z + 1.0) ** 3)


def summarize_table(task, table_path):
    taskname = "bphilowS8" if task == "bphilowS8dz" else task
    datatot = Table.read(table_path or f"bphi_results_{taskname}.txt", format="ascii.no_header")
    lines = []
    for tracer in np.unique(datatot["col1"]):
        data = datatot[datatot["col1"] == tracer]
        if task in {"bphilowS8", "bphiHOD"}:
            for zsnap in np.unique(data["col4"]):
                bphi = data["col7"][data["col4"] == zsnap]
                p_val = data["col8"][data["col4"] == zsnap]
                b1 = data["col10"][data["col4"] == zsnap] / 2.0 / 1.686 + 1.0
                lines.append(
                    f"{task}: {tracer} z={zsnap}, "
                    f"2*dlog(n)/dlog(sigma8)={np.mean(bphi):.2f}+-{np.std(bphi):.2f}, "
                    f"p value={np.mean(p_val):.2f}+-{np.std(p_val):.2f}, "
                    f"b1={np.mean(b1):.2f}+-{np.std(b1):.2f}"
                )
        elif task == "bphilowS8dz":
            phs = ["000", "001", "002", "003", "004", "005"]
            for k in range(len(np.unique(data["col4"])) - 1):
                dlog_ndz = np.log(data["col5"][len(phs) * k:len(phs) * (k + 1)] / data["col5"][len(phs) * (k + 1):len(phs) * (k + 2)])
                dlog_sigma8dz = np.log(growth_factor_approx(data["col4"][len(phs) * k]) / growth_factor_approx(data["col4"][len(phs) * (k + 1)]))
                bphidz = 2.0 * dlog_ndz / dlog_sigma8dz
                lines.append(
                    f"{tracer} dz={data['col4'][len(phs)*(k+1)]}-{data['col4'][len(phs)*k]:.3f}="
                    f"{data['col4'][len(phs)*(k+1)]-data['col4'][len(phs)*k]:.3f}, "
                    f"2*(dlog(n)/dz)/(dlog(sigma8)/dz)={np.mean(bphidz):.2f}+-{np.std(bphidz):.2f}"
                )
        else:
            raise ValueError("task must be one of bphilowS8, bphilowS8dz, bphiHOD")
    return lines


def build_parser():
    parser = argparse.ArgumentParser(description="Summarize bphi result tables.")
    parser.add_argument("task", choices=["bphilowS8", "bphilowS8dz", "bphiHOD"])
    parser.add_argument("--table", default=None, help="Input table path. Defaults to bphi_results_<task>.txt.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    for line in summarize_table(args.task, args.table):
        print(line)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run bphi_report CLI tests**

Run:

```bash
pytest tests/test_cli_hygiene.py::test_bphi_report_import_is_safe tests/test_cli_hygiene.py::test_bphi_report_help_runs -q
```

Expected: both tests pass.

- [ ] **Step 5: Compile script**

Run:

```bash
python3 -m py_compile python_scripts/bphi_report.py
```

Expected: command exits with status `0`.

- [ ] **Step 6: Commit**

Run:

```bash
git add python_scripts/bphi_report.py tests/test_cli_hygiene.py
git commit -m "refactor: convert bphi_report to argparse"
```

Expected: commit succeeds.

---

### Task 8: Convert `lowS8_comparison.py` to argparse and Centralized Paths

**Files:**
- Modify: `python_scripts/lowS8_comparison.py`
- Modify: `tests/test_cli_hygiene.py`

- [ ] **Step 1: Add import and help tests**

Append to `tests/test_cli_hygiene.py`:

```python
def test_lowS8_comparison_import_is_safe():
    module = importlib.import_module("python_scripts.lowS8_comparison")
    assert hasattr(module, "build_parser")
    assert hasattr(module, "main")


def test_lowS8_comparison_help_runs():
    result = subprocess.run(
        [sys.executable, "python_scripts/lowS8_comparison.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "Compute bphi from lowS8 or HOD number-count comparisons" in result.stdout
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
pytest tests/test_cli_hygiene.py::test_lowS8_comparison_import_is_safe -q
```

Expected: FAIL because the script currently executes on import.

- [ ] **Step 3: Refactor lowS8 comparison entrypoint**

Modify `python_scripts/lowS8_comparison.py` so all top-level runtime code moves into `main(argv=None)` and argument parsing is:

```python
def build_parser():
    parser = argparse.ArgumentParser(description="Compute bphi from lowS8 or HOD number-count comparisons.")
    parser.add_argument("task", choices=["bphilowS8", "bphiHOD"])
    parser.add_argument("tracer", choices=["LRG", "ELG", "QSO"])
    parser.add_argument("--mockdir", default=None)
    parser.add_argument("--pkdir", default=None)
    parser.add_argument("--output", default=None)
    return parser
```

At runtime, use:

```python
from pnglib.paths import default_path_config

config = default_path_config()
if args.task == "bphilowS8":
    mockdir = args.mockdir or str(config.scratch_root / "mocks")
    pkdir = args.pkdir or str(config.scratch_root / "pk")
else:
    mockdir = args.mockdir or str(config.scratch_root / "mocks_lhc")
    pkdir = args.pkdir or str(config.scratch_root / "pk_lhc")
output = args.output or f"bphi_results_{args.task}.txt"
```

Keep the existing numerical loop unchanged inside a function named:

```python
def compute_bphi_table(task, tracer, mockdir, pkdir, output):
    ...
```

- [ ] **Step 4: Run lowS8 CLI tests**

Run:

```bash
pytest tests/test_cli_hygiene.py::test_lowS8_comparison_import_is_safe tests/test_cli_hygiene.py::test_lowS8_comparison_help_runs -q
```

Expected: both tests pass.

- [ ] **Step 5: Compile script**

Run:

```bash
python3 -m py_compile python_scripts/lowS8_comparison.py
```

Expected: command exits with status `0`.

- [ ] **Step 6: Commit**

Run:

```bash
git add python_scripts/lowS8_comparison.py tests/test_cli_hygiene.py
git commit -m "refactor: convert lowS8 comparison to argparse"
```

Expected: commit succeeds.

---

### Task 9: Harden HOD LHC Generation and CLI

**Files:**
- Modify: `python_scripts/HOD_LHC.py`
- Modify: `python_scripts/bphi_prior_HOD.py`
- Modify: `tests/test_cli_hygiene.py`

- [ ] **Step 1: Add HOD LHC validation tests**

Append to `tests/test_cli_hygiene.py`:

```python
import pytest


def test_hod_lhc_import_is_safe_and_rejects_bad_tracer():
    module = importlib.import_module("python_scripts.HOD_LHC")
    assert hasattr(module, "HOD_lhc")
    with pytest.raises(ValueError, match="Unsupported tracer"):
        module.HOD_lhc("BAD", 0.5, 10)


def test_bphi_prior_hod_import_is_safe():
    module = importlib.import_module("python_scripts.bphi_prior_HOD")
    assert hasattr(module, "build_parser")
    assert hasattr(module, "main")
```

- [ ] **Step 2: Run tests and confirm failures**

Run:

```bash
pytest tests/test_cli_hygiene.py::test_hod_lhc_import_is_safe_and_rejects_bad_tracer tests/test_cli_hygiene.py::test_bphi_prior_hod_import_is_safe -q
```

Expected: FAIL because unsupported `HOD_lhc()` does not raise and `bphi_prior_HOD.py` executes path setup at import.

- [ ] **Step 3: Fix unsupported tracer handling in `HOD_LHC.py`**

Replace:

```python
else:
    bounds = {
        }
    ValueError("a general case has not been implemented")
```

with:

```python
else:
    raise ValueError(f"Unsupported tracer '{tracer}'. Choices are LRG, ELG, QSO.")
```

Also add a redshift validation after tracer branches:

```python
if "bounds" not in locals():
    raise ValueError(f"Unsupported redshift {redshift} for tracer {tracer}.")
```

- [ ] **Step 4: Add parser to `HOD_LHC.py`**

At the bottom of `HOD_LHC.py`, replace raw `sys.argv` handling with:

```python
def build_parser():
    parser = argparse.ArgumentParser(description="Generate HOD LHC parameter tables.")
    parser.add_argument("tracer", choices=["LRG", "ELG", "QSO"])
    parser.add_argument("--n-lhc", type=int, default=100)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    for item in redshift_bins(args.tracer):
        HOD_lhc(args.tracer, item.zsnap, args.n_lhc)


if __name__ == "__main__":
    main()
```

Add imports:

```python
import argparse
from pnglib.tracers import redshift_bins
```

- [ ] **Step 5: Convert `bphi_prior_HOD.py` to import-safe parser**

Move the `sys.path.append()` for HODDIES into `main()` and add:

```python
def build_parser():
    parser = argparse.ArgumentParser(description="Generate HOD mocks for bphi prior construction.")
    parser.add_argument("tracer", choices=["LRG", "ELG", "QSO"])
    parser.add_argument("--n-lhc", type=int, default=24)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--param-file", default=None)
    parser.add_argument("--abacus-root", default="/global/cfs/cdirs/desi/cosmosim/Abacus")
    return parser
```

Then define:

```python
def main(argv=None):
    args = build_parser().parse_args(argv)
    from pnglib.paths import default_path_config
    config = default_path_config()
    if config.hoddies_path is not None:
        sys.path.append(str(config.hoddies_path))
    from HODDIES import HOD
    outdir = args.outdir or str(config.scratch_root / "mocks_lhc")
    param_file = args.param_file or str(config.scratch_root / "conf" / "lowS8_HOD.yaml")
    run_bphi_prior_hod(args.tracer, args.n_lhc, outdir, param_file, args.abacus_root)
```

Wrap the existing runtime loop in:

```python
def run_bphi_prior_hod(tracer, n_lhc, outdir, param_file, path_to_abacus_sim):
    ...
```

- [ ] **Step 6: Run HOD CLI tests**

Run:

```bash
pytest tests/test_cli_hygiene.py::test_hod_lhc_import_is_safe_and_rejects_bad_tracer tests/test_cli_hygiene.py::test_bphi_prior_hod_import_is_safe -q
```

Expected: both tests pass.

- [ ] **Step 7: Compile scripts**

Run:

```bash
python3 -m py_compile python_scripts/HOD_LHC.py python_scripts/bphi_prior_HOD.py
```

Expected: command exits with status `0`.

- [ ] **Step 8: Commit**

Run:

```bash
git add python_scripts/HOD_LHC.py python_scripts/bphi_prior_HOD.py tests/test_cli_hygiene.py
git commit -m "refactor: harden hod lhc cli entrypoints"
```

Expected: commit succeeds.

---

### Task 10: Harden `HOD_one_lhc.py` Failure Semantics

**Files:**
- Modify: `python_scripts/HOD_one_lhc.py`
- Modify: `tests/test_cli_hygiene.py`

- [ ] **Step 1: Add parser import test for allow-partial**

Append to `tests/test_cli_hygiene.py`:

```python
def test_hod_one_lhc_parser_has_allow_partial():
    module = importlib.import_module("python_scripts.HOD_one_lhc")
    parser = module.build_parser()
    help_text = parser.format_help()
    assert "--allow-partial" in help_text
```

- [ ] **Step 2: Run test and confirm failure**

Run:

```bash
pytest tests/test_cli_hygiene.py::test_hod_one_lhc_parser_has_allow_partial -q
```

Expected: FAIL because `HOD_one_lhc.py` does not expose `build_parser()`.

- [ ] **Step 3: Extract parser in `HOD_one_lhc.py`**

Move parser construction from `main()` into:

```python
def build_parser():
    parser = argparse.ArgumentParser(description="Run one HOD LHC/default realization.")
    parser.add_argument("tracer", choices=["LRG", "ELG"])
    parser.add_argument("hod_id", type=int, help="0 for YAML defaults, 1-100 for LHC rows")
    parser.add_argument("zmin", type=float)
    parser.add_argument("zmax", type=float)
    parser.add_argument("zsnap", type=float)
    parser.add_argument("simtype", choices=["SU", "PNG"])
    parser.add_argument("--Acent", type=float, default=None, help="Optional AbacusHOD Acent value.")
    parser.add_argument("--Asat", type=float, default=None, help="Optional AbacusHOD Asat value.")
    parser.add_argument("--Bcent", type=float, default=None, help="Optional AbacusHOD Bcent value.")
    parser.add_argument("--Bsat", type=float, default=None, help="Optional AbacusHOD Bsat value.")
    parser.add_argument("--cosmos", nargs="+", default=None, help="Cosmology tags to run, e.g. 000 300.")
    parser.add_argument("--config", default=None, help="Use one explicit LRG AbacusHOD YAML config.")
    parser.add_argument("--output-root", default=None, help="Directory root for mock outputs.")
    parser.add_argument("--force", action="store_true", help="Run even if the done flag already exists.")
    parser.add_argument("--compute-b1", action="store_true", help="For c000 mocks, compute real-space P(k) and linear bias b1.")
    parser.add_argument("--b1-kmin", type=float, default=0.01, help="Minimum k for b1 estimate.")
    parser.add_argument("--b1-kmax", type=float, default=0.05, help="Maximum k for b1 estimate.")
    parser.add_argument("--allow-partial", action="store_true", help="Do not fail the job if b1, clustering, or plotting fails.")
    return parser
```

Change `main()` to:

```python
def main(argv=None):
    args = build_parser().parse_args(argv)
    ...
```

- [ ] **Step 4: Make clustering failures fatal by default**

Replace the clustering exception block:

```python
except Exception as exc:
    print(..., file=sys.stderr)
```

with:

```python
except Exception as exc:
    print(
        f"Clustering/plot failed for tracer={tracer} z={zsnap:.3f} id={hod_id} cosmo={cosmo} Ngal={Ngal}: {exc}",
        file=sys.stderr,
    )
    if not args.allow_partial:
        raise
```

For the b1 exception block, keep nonfatal default only if the science workflow treats b1 as optional. Otherwise use the same `if not args.allow_partial: raise` behavior. If keeping b1 nonfatal, add this comment above the block:

```python
# b1 is a diagnostic/truth marker output; clustering products remain the primary job output.
```

- [ ] **Step 5: Run parser test**

Run:

```bash
pytest tests/test_cli_hygiene.py::test_hod_one_lhc_parser_has_allow_partial -q
```

Expected: PASS.

- [ ] **Step 6: Compile script**

Run:

```bash
python3 -m py_compile python_scripts/HOD_one_lhc.py
```

Expected: command exits with status `0`.

- [ ] **Step 7: Commit**

Run:

```bash
git add python_scripts/HOD_one_lhc.py tests/test_cli_hygiene.py
git commit -m "fix: make hod clustering failures explicit"
```

Expected: commit succeeds.

---

### Task 11: Convert Remaining Raw-argv Scripts to Import-Safe CLIs

**Files:**
- Modify: `python_scripts/AbacusHOD_lowS8.py`
- Modify: `python_scripts/bphi_dz.py`
- Modify: `tests/test_cli_hygiene.py`

- [ ] **Step 1: Add import-safety tests**

Append to `tests/test_cli_hygiene.py`:

```python
def test_abacushod_lowS8_import_is_safe():
    module = importlib.import_module("python_scripts.AbacusHOD_lowS8")
    assert hasattr(module, "build_parser")
    assert hasattr(module, "main")


def test_bphi_dz_import_is_safe():
    module = importlib.import_module("python_scripts.bphi_dz")
    assert hasattr(module, "build_parser")
    assert hasattr(module, "main")
```

- [ ] **Step 2: Run tests and confirm failures**

Run:

```bash
pytest tests/test_cli_hygiene.py::test_abacushod_lowS8_import_is_safe tests/test_cli_hygiene.py::test_bphi_dz_import_is_safe -q
```

Expected: FAIL because both scripts execute on import or lack parser functions.

- [ ] **Step 3: Convert `AbacusHOD_lowS8.py`**

Add:

```python
def build_parser():
    parser = argparse.ArgumentParser(description="Run lowS8 AbacusHOD mocks.")
    parser.add_argument("tracer", choices=["LRG", "ELG", "QSO"])
    parser.add_argument("--hoddies-path", default=None)
    parser.add_argument("--output-root", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from pnglib.paths import default_path_config
    config = default_path_config()
    hoddies_path = args.hoddies_path or str(config.hoddies_path)
    sys.path.append(hoddies_path)
    run_lowS8(args.tracer, output_root=args.output_root)


if __name__ == "__main__":
    main()
```

Move existing runtime code into:

```python
def run_lowS8(tracer, output_root=None):
    ...
```

- [ ] **Step 4: Convert `bphi_dz.py`**

Add:

```python
def build_parser():
    parser = argparse.ArgumentParser(description="Compute bphi from redshift evolution.")
    parser.add_argument("--mockdir", default=None)
    parser.add_argument("--pkdir", default=None)
    parser.add_argument("--output", default="bphi_results_bphilowS8dz.txt")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from pnglib.paths import default_path_config
    config = default_path_config()
    mockdir = args.mockdir or str(config.scratch_root / "mocks")
    pkdir = args.pkdir or str(config.scratch_root / "pk")
    run_bphi_dz(mockdir=mockdir, pkdir=pkdir, output=args.output)


if __name__ == "__main__":
    main()
```

Move existing runtime code into:

```python
def run_bphi_dz(mockdir, pkdir, output):
    ...
```

- [ ] **Step 5: Run import-safety tests**

Run:

```bash
pytest tests/test_cli_hygiene.py::test_abacushod_lowS8_import_is_safe tests/test_cli_hygiene.py::test_bphi_dz_import_is_safe -q
```

Expected: both tests pass.

- [ ] **Step 6: Compile scripts**

Run:

```bash
python3 -m py_compile python_scripts/AbacusHOD_lowS8.py python_scripts/bphi_dz.py
```

Expected: command exits with status `0`.

- [ ] **Step 7: Commit**

Run:

```bash
git add python_scripts/AbacusHOD_lowS8.py python_scripts/bphi_dz.py tests/test_cli_hygiene.py
git commit -m "refactor: make remaining bphi scripts import safe"
```

Expected: commit succeeds.

---

### Task 12: Shell Script Hygiene and Path Overrides

**Files:**
- Modify: `python_scripts/preproc_parallel.sh`
- Modify: `python_scripts/preproc_parallel_PNG.sh`
- Modify: `python_scripts/bphi_prior_pipeline.sh`
- Modify: `python_scripts/HOD_one_lhc.sh`
- Create: `tests/test_shell_hygiene.py`

- [ ] **Step 1: Add shell hygiene tests**

Create `tests/test_shell_hygiene.py`:

```python
from pathlib import Path
import subprocess


SHELL_SCRIPTS = [
    "python_scripts/preproc_parallel.sh",
    "python_scripts/preproc_parallel_PNG.sh",
    "python_scripts/bphi_prior_pipeline.sh",
    "python_scripts/HOD_one_lhc.sh",
]


def test_shell_scripts_parse_with_bash_n():
    for script in SHELL_SCRIPTS:
        result = subprocess.run(["bash", "-n", script], text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stderr


def test_preproc_parallel_uses_newcfg_variable():
    text = Path("python_scripts/preproc_parallel.sh").read_text(encoding="utf-8")
    assert "NEWCFGT" not in text
    assert 'if [ -f "$NEWCFG" ]; then' in text
```

- [ ] **Step 2: Run shell tests and confirm typo failure**

Run:

```bash
pytest tests/test_shell_hygiene.py -q
```

Expected: FAIL because `preproc_parallel.sh` contains `NEWCFGT`.

- [ ] **Step 3: Fix `NEWCFGT` typo**

In `python_scripts/preproc_parallel.sh`, replace:

```bash
if [ -f "$NEWCFGT" ]; then
```

with:

```bash
if [ -f "$NEWCFG" ]; then
```

- [ ] **Step 4: Add path override variables to shell scripts**

In both `preproc_parallel.sh` and `preproc_parallel_PNG.sh`, replace:

```bash
dir_source="/pscratch/sd/j/jiaxi/b_phi"
```

with:

```bash
dir_source="${BPHI_SCRATCH_ROOT:-/pscratch/sd/j/jiaxi/b_phi}"
```

In `HOD_one_lhc.sh`, replace:

```bash
cd $SCRATCH/b_phi
```

with:

```bash
cd "${BPHI_ROOT:-$SCRATCH/b_phi}"
```

In `bphi_prior_pipeline.sh`, replace:

```bash
export DESILIKE_LOCAL_PATH=/global/homes/j/jiaxi/codes/desilike
```

with:

```bash
export DESILIKE_LOCAL_PATH="${DESILIKE_LOCAL_PATH:-/global/homes/j/jiaxi/codes/desilike}"
```

- [ ] **Step 5: Run shell tests**

Run:

```bash
pytest tests/test_shell_hygiene.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Run bash syntax checks directly**

Run:

```bash
bash -n python_scripts/preproc_parallel.sh
bash -n python_scripts/preproc_parallel_PNG.sh
bash -n python_scripts/bphi_prior_pipeline.sh
bash -n python_scripts/HOD_one_lhc.sh
```

Expected: all commands exit with status `0`.

- [ ] **Step 7: Commit**

Run:

```bash
git add python_scripts/preproc_parallel.sh python_scripts/preproc_parallel_PNG.sh python_scripts/bphi_prior_pipeline.sh python_scripts/HOD_one_lhc.sh tests/test_shell_hygiene.py
git commit -m "fix: centralize shell path overrides"
```

Expected: commit succeeds.

---

### Task 13: Add Documentation for New Hygiene Conventions

**Files:**
- Modify: `python_scripts/README.md`

- [ ] **Step 1: Append CLI and path convention documentation**

Append to `python_scripts/README.md`:

```markdown
## Refactor Hygiene Conventions

Shared script logic lives in `python_scripts/pnglib/`.

- `pnglib.paths`: central path configuration and environment overrides.
- `pnglib.tracers`: tracer, redshift-bin, mode, simtype, and cosmology-tag metadata.
- `pnglib.pk_io`: P(k), covariance, and path helpers.
- `pnglib.bphi_stats`: bphi stats parsing and prior requirement helpers.

CLI entrypoints should define `build_parser()` and `main(argv=None)`, and should only execute inside:

```python
if __name__ == "__main__":
    main()
```

Path environment overrides:

- `BPHI_ROOT`: local script data root. Defaults to `python_scripts/`.
- `BPHI_SCRATCH_ROOT`: NERSC scratch root. Defaults to `/pscratch/sd/j/jiaxi/b_phi`.
- `BPHI_PK_DIR`: overrides P(k) input directory.
- `BPHI_COV_DIR`: overrides covariance directory.
- `BPHI_PLOT_DIR`: overrides diagnostic plot directory.
- `BPHI_FIT_DIR`: overrides fit output directory.
- `BPHI_PNG_MOCK_DIR`: overrides PNG mock P(k) directory.
- `BPHI_STATS_FILE`: overrides bphi prior stats file.
- `BPHI_HOD_CONFIG_DIR`: overrides HOD config directory.
- `HODDIES_PATH`: overrides HODDIES import path.
- `DESILIKE_LOCAL_PATH`: overrides local desilike import path.

Fast local checks:

```bash
pytest tests/test_pnglib_paths.py tests/test_pnglib_tracers.py tests/test_pnglib_pk_io.py tests/test_pnglib_bphi_stats.py tests/test_cli_hygiene.py tests/test_shell_hygiene.py -q
python3 -m py_compile python_scripts/*.py python_scripts/pnglib/*.py
bash -n python_scripts/preproc_parallel.sh
```
```

- [ ] **Step 2: Verify documentation phrases**

Run:

```bash
rg -n "Refactor Hygiene Conventions|BPHI_ROOT|build_parser\\(\\)" python_scripts/README.md
```

Expected: all three phrases are found.

- [ ] **Step 3: Commit**

Run:

```bash
git add python_scripts/README.md
git commit -m "docs: document python_scripts hygiene conventions"
```

Expected: commit succeeds.

---

## Final Verification

- [ ] **Step 1: Run all local unit tests**

Run:

```bash
pytest tests/test_pnglib_paths.py tests/test_pnglib_tracers.py tests/test_pnglib_pk_io.py tests/test_pnglib_bphi_stats.py tests/test_pipeline_wrapper.py tests/test_cli_hygiene.py tests/test_shell_hygiene.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Compile Python scripts**

Run:

```bash
python3 -m py_compile python_scripts/*.py python_scripts/pnglib/*.py
```

Expected: command exits with status `0`. Existing invalid-escape warnings in labels are acceptable only if they existed before this refactor and do not fail compilation.

- [ ] **Step 3: Syntax-check shell scripts**

Run:

```bash
bash -n python_scripts/*.sh
```

Expected: command exits with status `0`.

- [ ] **Step 4: Confirm duplicate pipeline is retired**

Run:

```bash
wc -l python_scripts/bphi_prior_pipeline.py python_scripts/bphi_prior_pipeline_bfnl.py
```

Expected: `bphi_prior_pipeline_bfnl.py` is under 20 lines and delegates to `bphi_prior_pipeline.py`.

- [ ] **Step 5: Confirm no raw argv remains in converted scripts**

Run:

```bash
rg -n "sys\\.argv\\[" python_scripts/bphi_report.py python_scripts/lowS8_comparison.py python_scripts/HOD_LHC.py python_scripts/bphi_prior_HOD.py python_scripts/AbacusHOD_lowS8.py python_scripts/bphi_dz.py
```

Expected: no matches.

- [ ] **Step 6: Confirm working tree**

Run:

```bash
git status --short
```

Expected: no uncommitted changes after the final commit, except any pre-existing untracked plan documents intentionally left uncommitted by the user.

---

## Self-Review Notes

- Spec coverage: The plan covers central path/config handling, shared tracer metadata, shared P(k)/bphi IO, no-prior bphi stats robustness, duplicate pipeline retirement, raw `sys.argv` conversion, shell hygiene, explicit HOD failure semantics, tests, validation commands, and documentation.
- Placeholder scan: This plan contains no placeholder instructions. Each implementation task names exact files, commands, expected results, and concrete code blocks for the key changes.
- Type consistency: The helper names used by tests and later tasks match the modules introduced in earlier tasks: `PathConfig`, `default_path_config`, `redshift_bins`, `default_pk_path`, `load_bphi_stats`, and `stats_required_for_fit`.
