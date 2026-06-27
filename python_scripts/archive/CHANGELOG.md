# Changelog

All notable changes to Python scripts in this repository should be recorded here.

Use the `Unreleased` section for work in progress, then move entries under a
dated heading when changes are committed or released. Keep entries concise and
focus on behavior, inputs, outputs, dependencies, and reproducibility.

## Unreleased

### Added

- Add new Python scripts, analysis entry points, or reusable utilities here.

### Changed

- Record updates to existing pipeline behavior, model assumptions, plotting, or
  default parameter choices here.

### Fixed

- Note bug fixes, validation corrections, and reproducibility fixes here.

## Historical Baseline

### Added

- Added the `bphi_prior_pipeline.py` workflow for constructing data vectors,
  covariance estimates, and MCMC inputs for bphi-fNL fitting.
- Added bphi prior workflows, including prior construction, PNG fitting, and
  plotting support.
- Added HOD sampling utilities and validation scripts for AbacusHOD workflows.
- Added Latin hypercube sampling support and deterministic HOD mock seeding for
  matching tracers across runs.
- Added MPI and one-tracer HOD mock workflows for clustering and bphi analysis.
- Added low-S8 comparison and HOD analysis scripts.
- Added `test_png_bias_krange.py` for PNG bias k-range checks.

### Changed

- Updated the bfnl pipeline scripts and shell wrappers.
- Reorganized the repository into the current script-focused layout.
- Corrected configuration directory handling for script execution.

## Maintenance Notes

- Track Python script changes in this file when editing `*.py` files.
- Include shell wrapper changes when they alter how Python scripts are launched.
- Prefer entries that explain user-visible effects, for example:
  `Changed bphi_prior_pipeline.py to write covariance outputs per tracer.`
- Avoid listing formatting-only edits unless they affect reproducibility or
  execution.
