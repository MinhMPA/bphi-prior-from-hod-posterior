# b_phi Work Summary (through 19 May 2026)

## Goal
Focused on debugging and stabilizing the PNG/bphi prior fitting workflow, improving diagnostics/plots, and preparing a clean path to include physically motivated `b1` truth markers in triangle plots.

## Main Pipeline and Fitting Updates
- Simplified `bphi_prior_pipeline.sh` so it is mode-driven and automatically runs both **with** and **without** bphi priors (except unsupported combinations).
- Added/cleaned debug-fit behavior for separate redshift fits (`z = 0.5, 0.8, 1.1`) and separate output directories via `BPHI_FIT_DIR`.
- Fixed data/covariance truncation handling in `bphi_prior_pipeline.py` where fitting slice logic was inconsistent.
- Ensured debug runs now produce the same analysis artifacts as ordinary runs:
  - triangle plots
  - profiler stats
  - chain stats

## Priors and Likelihood Logic
- Reviewed desilike PNG parameter priors for `b-p`, `bfnl`, and `bphi` modes from source.
- Updated flat-prior handling to use absolute limits consistently (`FLAT_PRIORS` usage cleanup).
- Corrected the SU abundance-response convention to include the PNG factor of 2:
  - `bphi = 2 * ln(Ngal_c000 / Ngal_c004) / ln(0.807952 / 0.75)`.
- Current LRG bphi prior centers/widths are:
  - z=0.500: `N(3.10823245, 1.8469^2)`
  - z=0.800: `N(3.86419230, 2.1879^2)`
  - z=1.100: `N(3.98423919, 2.5797^2)`
- Implemented combined-prior treatment for the required case:
  - `mode == "bfnl"` and `use_bphi_prior == True`.
- Added a custom combined prior term that supports `bphi * fnl` consistency in the `bfnl` workflow.
- Changed `bphi_best` behavior to use the **first LHC bphi entry** (central/reference entry), not the mean over all LHC bphi values.

## Triangle Plot and Diagnostics Improvements
- Added MAP and ML extraction/saving helpers for robust reuse in plots.
- Joint and single triangle plots now overlay both:
  - ML point
  - MAP point
  with distinct markers.
- Maintained consistent posterior colors for with/without bphi prior while using different marker styles for MAP/ML.
- Removed non-essential text annotations on triangle figures (kept contour labels only, as requested).
- Added black truth lines for:
  - true `fnl_loc` (from `TRUE_FNL_LOC_BY_COSMO`)
  - derived `bphi * fnl_loc`
  - `b1` (when provided in bphi stats file)
- Improved problematic `c301` no-prior plotting behavior by using more conservative chain selection defaults (burn-in/thinning controls) to stabilize contour appearance without degrading `c000`/`c300`.

## Desilike Source/Model Understanding
- Located and checked desilike PNG implementation and defaults:
  - PNG model mode behavior (`bphi`, `b-p`, `bfnl`)
  - parameter prior defaults (`fnl_loc`, `bphi`, `bfnl_loc`, `p`, `b1`, `sigmas`, `sn0`)
- Clarified `sigmas` and `sn0` roles in the PNG tracer model context for fit interpretation.

## HOD / b1 Preparation Work
- Added a debug script path for:
  - `HOD_one_lhc.sh`
  - `tracer == LRG`
  - `zsnaps=(0.95)`
  - config `/pscratch/sd/j/jiaxi/b_phi/HOD_config/abacushod_z0.950_c000.yaml`
- Implemented optional `b1`-computation support integrated into HOD workflow:
  - real-space `P(k)` used for linear-bias estimate
  - RSD clustering products (`wp`, `xi0`, `xi2`, and RSD `Pk`) kept for standard outputs
- Extended bphi table parsing/writing to accept an optional `b1` column and made pipeline loading robust to missing/NaN entries.

## AbacusHOD Runtime Update
- Diagnosed the `bash HOD_one_lhc.sh 0 SU` failure from AbacusHOD:
  - `z_mock = 0.950` is treated as a secondary redshift by AbacusHOD.
  - secondary redshift snapshots do not have particle position/velocity subsample outputs.
  - AbacusHOD therefore requires NFW satellite placement for this redshift.
- Updated `HOD_utils.compute_all()` so secondary-redshift `AbacusHOD` objects automatically run with:
  - `want_nfw=True`
  - `NFW_draw` loaded from `archive/chains_ELG/data/nfw.npy`
- Added `ABACUSHOD_NFW_DRAW` as an environment override for using a different `nfw.npy`.
- Kept the user-facing HOD command unchanged:
  - `bash HOD_one_lhc.sh 0 SU`
- Verified the wrapper behavior with `py_compile` and a lightweight mock `run_hod` test; full `srun` production rerun is still the next runtime validation.

## Files Most Actively Updated
- `b_phi/bphi_true_value/*`
- `b_phi/bphi_prior_pipeline.py`
- `b_phi/bphi_prior_pipeline.sh`
- `b_phi/bphi_prior_fit_debug.sh`
- `b_phi/HOD_one_lhc.py`
- `b_phi/HOD_one_lhc.sh`
- `b_phi/HOD_utils.py`
- `b_phi/HOD_bphi.py`

## Current Status: 19 May 2026
- Core fitting/debug pipeline is functional with enhanced diagnostics and clearer prior-handling behavior.
  - Combined `bfnl + bphi-prior` logic is in place for your required simultaneous mode/prior configuration.
    - `ell = 0,2` suffers from projection effect
  - desilike `bfnl` pipeline still needs follow-up validation/fixing.
    - solving problems of biased `b1` measurement: use the real-space Pk of PNG HOD catalogues to fit `b1` true values with an extra `sn0` term
    - `ell=0` should fix `sn0=sigmas=0` for unbiased `fnl`; `ell=0,2` should free all parameters 
  - Joint/single posterior visualization now includes ML/MAP markers and truth-reference overlays.
