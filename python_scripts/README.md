# b_phi Work Summary (through 10 May 2026)

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
- Updated z=1.1 bphi prior center/width in the pipeline:
  - `N(1.99211960, 1.2898^2)` for bphi prior usage.
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

## PNG Data Vector and HOD Redshift-Space Convention

- The target PNG fitting data vector is the joint redshift-space galaxy
  power-spectrum multipole vector `(ell=0, ell=2)`. The monopole and quadrupole
  must be fit together with the matching Gaussian covariance supplied by
  `thecov`.
- AbacusHOD LRG mocks generated with `want_rsd: True` already contain
  redshift-space positions. Do not manually shift those LRG positions again
  before measuring `P_0` and `P_2`.
- LRG centrals are placed at halo-center positions, but their velocities include
  the configured central velocity-bias/scatter term.
- LRG satellites are not generally halo-center objects. For the primary
  `use_particles: True` path, their positions follow particle/subsample
  positions and their velocities include a 1-halo contribution controlled by
  `alpha_s`.
- AbacusHOD secondary-redshift fallback uses NFW satellite placement; validation
  reports should distinguish this from the particle-subsample satellite path.
- ELG/HODDIES mocks use separate RSD and satellite-velocity settings
  (`vel_sat`, `f_sigv`, `vsmear`). Audit whether HODDIES has already applied RSD
  before applying any local ELG redshift-space shift.

## fnl_loc Validation Strategy

- **Scenario A: bfnl baseline** runs desilike `mode="bfnl"` and samples one
  product amplitude `B_i = b_{\phi,i} fnl_loc` per independent tracer/redshift
  bin, with no `bphi_i` prior and no universal sampled `fnl_loc`. In code this
  product amplitude is usually named `bfnl_z` or `bfnl_loc`.
- **Scenario B: HOD-informed Gaussian bphi prior** targets one universal
  `fnl_loc` with one Gaussian HOD prior
  `pi_i(beta_i) = Normal(beta_i; mu_bphi_i, sigma_bphi_i^2)` for each
  `bphi_i`. Validate both B-direct (`mode="bphi"` sampling of `fnl_loc` and
  each `bphi_i`) and B-marginal (custom likelihood marginalizing each Gaussian
  `bphi_i` prior) before selecting the official implementation.
- **Scenario C: fixed-p universality** fixes
  `bphi_i = 2 delta_c (b1_i - p_i)` with tracer/sample-specific fixed `p_i`
  values and samples one universal `fnl_loc`.
- B-marginal follows the `paper/main.tex` appendix formalism:
  `L_i^marg(f) = integral d beta_i L_i(B_i = f beta_i) pi_i(beta_i)`, followed
  by one common `fnl_loc` prior after multiplying independent bin likelihoods.
- If both `L_i(B_i)` and the Gaussian `bphi_i` prior are Gaussian, use the
  analytic Gaussian-Gaussian expression from `Bhat_i`, `sigma_B_i`,
  `mu_bphi_i`, and `sigma_bphi_i`; Gauss-Hermite quadrature is not needed.
- If the Gaussian `bphi_i` prior is kept but `L_i(B_i)` is numerical,
  Gauss-Hermite quadrature implements the same marginalization integral.
- `grid` mode is deterministic quadrature for checking transformation math
  without sampler noise.
- `mcmc` mode compares real desilike pipeline chains only.
- Non-Gaussian `bphi_i` priors are represented by weighted samples or tabulated
  log-density inputs and use their own integration weights, not Gauss-Hermite
  weights.

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

## Current Status: 10 May 2026
- Core fitting/debug pipeline is functional with enhanced diagnostics and clearer prior-handling behavior.
  - Combined `bfnl + bphi-prior` logic is in place for your required simultaneous mode/prior configuration.
  - desilike `bfnl` pipeline still needs follow-up validation/fixing.
  - Joint/single posterior visualization now includes ML/MAP markers and truth-reference overlays.
- HOD-side `b1` integration path is implemented and ready for production compute runs.
  - ~AbacusHOD secondary-redshift handling is now patched for `z = 0.950` via automatic NFW fallback: the HOD parameters in the published paper doesn't work on this snapshot~
  - rerun the SU/PNG HOD catalog production and collect the new `b1` outputs.
  - ~use the real-space Pk of PNG HOD catalogues to fit bphi true values: the first 2 point dominates the fitting. Moreoever, either or not including these two cannot reproduce similar bphi from the universality relation~
  - 
