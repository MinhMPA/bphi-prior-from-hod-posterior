# Project State

Last updated: 2026-05-09
Updated by: Codex agent
Project: Data-driven b_phi priors for local PNG
Paper target: PRD-style working draft; final venue not fixed

## 1. One-paragraph project summary

This project develops data-driven priors on the local primordial non-Gaussianity response parameter `b_phi` for DESI-like galaxy samples. The method propagates small-scale HOD constraints into `b_phi` using AbacusSummit HOD mocks, separate-universe number-count responses, and validation against AbacusSummitPNG mocks with nonzero `f_NL`. The current paper-level conclusion is preliminary: the analysis is organized around the fact that the large-scale power spectrum constrains `b_phi * f_NL` more directly than `f_NL` alone, so a defensible `f_NL` constraint requires either a universality assumption or an external HOD-informed `b_phi` prior.

## 2. Current core claim

We show that local-PNG power-spectrum fits constrain the redshift-dependent amplitude `bfnl_z = b_phi,z * f_NL^loc`, and that HOD-informed `b_phi` priors provide a more transparent route from this measured amplitude to `f_NL^loc` than an unconstrained `b_phi` fit or a purely universality-based assumption, using DESI-like LRG HOD mocks built from AbacusSummit and AbacusSummitPNG simulations.

Status: preliminary. The claim is well supported as the intended narrative and workflow, but final quantitative constraints are not present in the tracked repository.

Supporting paths: `paper/main.tex`, `docs/superpowers/plans/fnl_sampling_implementation_validation_plan.md`, `python_scripts/bphi_prior_pipeline.py`, `python_scripts/README.md`

## 3. Scientific motivation

- Problem: local PNG induces a scale-dependent galaxy-bias contribution proportional to `f_NL * b_phi`, making `b_phi` uncertainty a limiting systematic for galaxy-survey `f_NL` inference. Supporting paths: `paper/main.tex`
- Existing insufficiency: the universality relation `b_phi = 2 delta_c (b_1 - 1)` is only appropriate for idealized mass-selected tracers and is known to fail for realistic galaxy selections. Supporting paths: `paper/main.tex`, `paper/references.bib`
- New ingredient: use DESI One-Percent/HOD-constrained galaxy-halo connection models to build posterior predictive `b_phi` priors and test them with SU and PNG Abacus workflows. Supporting paths: `paper/main.tex`, `python_scripts/HOD_bphi.py`, `python_scripts/bphi_prior_pipeline.py`
- Interpretation improvement: report `bfnl_z` when no `b_phi` assumption is used; only convert to `f_NL^loc` under an explicit `b_phi` relation or prior. Supporting paths: `docs/superpowers/plans/fnl_sampling_implementation_validation_plan.md`

## 4. Current results snapshot

| Result | Status | Location | Notes |
|---|---|---|---|
| Power-spectrum data should be interpreted as constraining `bfnl_z = b_phi,z * f_NL^loc` | robust conceptual decision | `docs/superpowers/plans/fnl_sampling_implementation_validation_plan.md` | Use `bfnl_z` as the common diagnostic across prior assumptions. |
| No free-`b_phi` assumption gives no well-identified `f_NL^loc` posterior | robust conceptual decision | `docs/superpowers/plans/fnl_sampling_implementation_validation_plan.md` | Paper should report `bfnl_z` rather than a prior-volume-driven `f_NL` posterior in this case. |
| Combined `bfnl` mode plus HOD `b_phi` prior is implemented | preliminary | `python_scripts/bphi_prior_pipeline.py`, `python_scripts/README.md` | Uses a custom likelihood that marginalizes over one common `f_NL`; desilike `bfnl` workflow still needs follow-up validation. |
| LRG `z=1.1` `b_phi` prior center/width updated to `N(1.99211960, 1.2898^2)` | preliminary | `python_scripts/README.md` | Tracked repo does not include the referenced final stats file, so treat this as workflow state rather than final result. |
| HOD-side real-space `b1` computation path is implemented | preliminary | `python_scripts/HOD_utils.py`, `python_scripts/HOD_one_lhc.py`, `python_scripts/HOD_bphi.py`, `python_scripts/README.md` | Intended for truth markers and diagnostics; production rerun remains pending. |
| AbacusHOD secondary-redshift handling for `z=0.950` now uses NFW satellite placement fallback | preliminary implementation result | `python_scripts/HOD_utils.py`, `python_scripts/HOD_one_lhc.sh`, `python_scripts/README.md` | Lightweight validation noted; full `srun` production rerun remains open. |
| Final `f_NL` constraints and paper-ready `b_phi` prior tables | missing / not yet in repo | `python_scripts/bphi_prior_pipeline.py`, `paper/main.tex` | No tracked final chains, stats tables, or figure PDFs found. |

## 5. Figures and tables for the paper

| Figure/Table | Purpose | Source script/notebook | Status | Needed changes |
|---|---|---|---|---|
| Fig. 1 | Pipeline overview: HOD posterior to SU/PNG `b_phi` calibration to `f_NL` forecast | `paper/main.tex` | not created | Draft schematic and align notation with analysis scripts. |
| Fig. 2 | Compare `bfnl_z` constraints and `f_NL^loc` posteriors under universality, HOD prior, and no-`b_phi` assumptions | `python_scripts/bphi_prior_pipeline.py`, `docs/superpowers/plans/fnl_sampling_implementation_validation_plan.md` | preliminary tooling | Rerun final chains and decide whether `bfnl` mode is primary. |
| Fig. 3 | Triangle plots with/without HOD `b_phi` prior, including ML/MAP and truth overlays | `python_scripts/bphi_prior_pipeline.py` | current tooling | Regenerate after final bphi stats and final chain settings. |
| Fig. 4 | P(k) and covariance variance diagnostics | `python_scripts/bphi_prior_plot_pk.py` | diagnostic tooling | Decide whether this belongs in paper or appendix. |
| Fig. 5 | PNG real-space bias k-range diagnostic | `python_scripts/test_png_bias_krange.py` | diagnostic tooling | Run for final redshifts/cosmologies; summarize chosen `b1` window. |
| Fig. 6 | HOD clustering validation against DESI-like measurements | `python_scripts/HOD_utils.py`, `python_scripts/HOD_one_lhc.py`, `jupyter_notebooks/Fit_b1_to_AbacusSUmmit_HOD_mocks.ipynb` | draft / exploratory | Replace notebook output with reproducible script if used in paper. |
| Table 1 | Final redshift-dependent HOD `b_phi` priors | `python_scripts/HOD_bphi.py` | missing final table | Produce tracked or archived table with means, widths, sample counts, and caveats. |
| Table 2 | Key `f_NL` forecast or recovery metrics | `python_scripts/bphi_prior_pipeline.py` | missing final table | Export final constraints from chain/profile stats. |

## 6. Key decisions made

| Date | Decision | Rationale | Alternatives considered | Consequence |
|---|---|---|---|---|
| 2026-05-09 | Treat `bfnl_z = b_phi,z * f_NL^loc` as the common diagnostic. | It is the amplitude directly constrained by the large-scale P(k). | Direct `f_NL`, direct `b_phi`, fixed universality only. | Paper narrative should separate measurement of `bfnl_z` from interpretation as `f_NL`. |
| 2026-05-09 | Present no-`b_phi` runs as `bfnl_z` constraints, not as physical `f_NL` constraints. | Without a `b_phi` prior or relation, `f_NL` is not identifiable. | Show broad `f_NL` posterior from implicit priors. | Avoids prior-volume artifacts in the main claim. |
| 2026-05-09 | Use fixed `p=1` in `b-p` universality comparisons. | Matches standard universality-style analyses. | Free `p`; HOD prior only. | Universality comparison remains a deterministic relation to `b1`. |
| 2026-05-09 | Apply HOD `b_phi` information in `bfnl` mode through a combined prior likelihood over one common `f_NL`. | Keeps the data-constrained amplitude separate from the external `b_phi` prior. | Directly sample every `b_phi,z` with `f_NL`. | Needs validation near `f_NL = 0` and against direct `b_phi` sampling. |
| 2026-05-09 | Use the first LHC `b_phi` entry as the central/reference `b_phi` value in stats loading. | Current workflow treats first entry as the default/reference HOD. | Mean over all LHC `b_phi` values. | Must document exactly what "central" means in paper tables. |
| 2026-05-09 | Use flat prior half-widths `|b_phi| < 10` and `|f_NL^loc| < 300` in current pipeline defaults. | Stable broad defaults for fits. | Narrower or survey-specific priors. | Final posteriors should report sensitivity to these bounds if no HOD prior is used. |
| 2026-05-09 | For AbacusHOD secondary redshifts, enable NFW satellite placement fallback by default. | Secondary snapshots lack particle subsample outputs needed by default satellite placement. | Restrict to primary snapshots only. | Enables `z=0.950` debug path but depends on a valid `nfw.npy`. |

## 7. Assumptions and scope

- Cosmology: AbacusSummit fiducial cosmology is used in templates and matter P(k); `TRUE_FNL_LOC_BY_COSMO` maps `c000 -> 0`, `c300 -> 30`, `c302 -> -30`, `c303 -> 100`, and `c304 -> -100`. Supporting paths: `python_scripts/bphi_prior_pipeline.py`, `python_scripts/test_png_bias_krange.py`
- Simulations: SU comparison uses `c000` versus `c004`; PNG workflows use AbacusSummitPNG-style cosmology tags such as `c000`, `c300`, `c301`, `c302`, and `c303`. Supporting paths: `python_scripts/HOD_bphi.py`, `python_scripts/HOD_one_lhc.py`, `python_scripts/bphi_prior_pipeline_bfnl.sh`
- Tracers/redshifts: primary paper pipeline is currently LRG at `z = 0.5, 0.8, 1.1`; ELG support exists in some HOD scripts at `z = 0.95, 1.325` but is not the current paper baseline. Supporting paths: `python_scripts/bphi_prior_pipeline.py`, `python_scripts/HOD_bphi.py`, `python_scripts/HOD_one_lhc.sh`
- Noise model: current P(k) fits use a Gaussian covariance from `thecov`, with block-diagonal combination across redshift bins. Supporting paths: `python_scripts/bphi_prior_pipeline.py`, `python_scripts/bphi_prior_plot_pk.py`
- Survey geometry: current covariance and mock measurements assume a periodic box with `BOXSIZE = 2000.0`; real survey window effects are outside the tracked current pipeline. Supporting paths: `python_scripts/bphi_prior_pipeline.py`, `python_scripts/bphi_prior_plot_pk.py`
- Bias model: fits vary `b1`, `sn0`, and PNG response parameters depending on mode; `sigmas` is fixed to `0.0`; `b-p` mode fixes `p=1`. Supporting paths: `python_scripts/bphi_prior_pipeline.py`
- RSD treatment: P(k) fitting uses redshift-space monopoles for the main PNG pipeline; real-space P(k) is used for diagnostic `b1` estimates. Supporting paths: `python_scripts/HOD_utils.py`, `python_scripts/bphi_prior_pipeline.py`
- Priors: current defaults are `|b_phi| < 10`, `|f_NL^loc| < 300`, and `|bfnl_loc| < 3000`; HOD-informed `b_phi` priors are Gaussian when available. Supporting paths: `python_scripts/bphi_prior_pipeline.py`
- Likelihood approximation: `bfnl` plus `b_phi` prior mode marginalizes over a grid of common `f_NL` values with default `BPHI_BFNL_PRIOR_NGRID = 4096`. Supporting paths: `python_scripts/bphi_prior_pipeline.py`

## 8. Known limitations

- Final quantitative results are not yet tracked: no final `b_phi` stats table, final chain files, final stats exports, or paper-ready figures are present in the repository. Supporting paths: `python_scripts/README.md`, `paper/main.tex`
- The paper draft has placeholders in the simulation/mocks sections (`XX`, `YY`, and unfinished statistic descriptions). Supporting paths: `paper/main.tex`
- The current workflow relies on external NERSC paths and data products under `/pscratch`, `/global`, or local desilike/HODDIES installations. Supporting paths: `python_scripts/HOD_one_lhc.py`, `python_scripts/bphi_prior_pipeline.sh`, `HOD_config/*.yaml`
- The `bfnl` plus HOD-prior implementation needs explicit validation against fixed-p closure, limiting behavior, and direct `b_phi` sampling. Supporting paths: `docs/superpowers/plans/fnl_sampling_implementation_validation_plan.md`, `python_scripts/README.md`
- The current covariance model is Gaussian and box-based; survey window, selection, and non-Gaussian covariance effects are not yet represented in the tracked pipeline. Supporting paths: `python_scripts/bphi_prior_pipeline.py`, `python_scripts/bphi_prior_plot_pk.py`
- Assembly-bias and conformity impacts are part of the intended paper scope, but final runs and quantitative conclusions are not present in tracked files. Supporting paths: `paper/main.tex`, `python_scripts/HOD_one_lhc.py`
- Some diagnostics are exploratory scripts or notebooks rather than fully reproducible paper pipelines. Supporting paths: `jupyter_notebooks/Fit_b1_to_AbacusSUmmit_HOD_mocks.ipynb`, `python_scripts/bphi_report.py`

## 9. Open questions

| Question | Importance | Owner | Current best answer | Next action |
|---|---|---|---|---|
| Does the combined `bfnl + b_phi prior` likelihood reproduce fixed-p and direct-sampling closure tests? | high | Minh / Jiaxi | unknown; validation plan exists | Implement or run tests in `docs/superpowers/plans/fnl_sampling_implementation_validation_plan.md`. |
| What are the final redshift-dependent `b_phi` prior means and widths for LRG? | high | Jiaxi / Minh | preliminary `z=1.1` value noted; full table absent | Regenerate `python_scripts/HOD_bphi.py` outputs and archive final stats table. |
| Does the main result survive assembly-bias and conformity variations? | high | Jiaxi / Minh | intended but not quantified in tracked repo | Run controlled HOD variations and add rows to results table. |
| Which PNG cosmologies should define validation figures? | medium | Minh | scripts mention `c000`, `c300`, `c301`, `c302`, `c303`; truth map lacks `c301` | Decide final cosmology set and document `f_NL` values. |
| Is `z=0.950` LRG debug path part of the main paper or a runtime validation only? | medium | Minh | runtime validation path | Complete production rerun or remove from paper narrative. |
| Should covariance diagnostics appear in the paper or only internal validation? | medium | Minh | undecided | Decide after final P(k)/cov reruns. |
| How should external DESI One-Percent HOD posterior constraints be documented and cited? | high | Jiaxi / Minh | cited conceptually in draft | Add exact data/vector description and references in methods. |

## 10. To-do list before paper submission

### Analysis

- [ ] Produce final tracked or archived `b_phi` prior table for all paper redshifts.
- [ ] Run final SU HOD production with optional `b1` outputs.
- [ ] Run PNG validation mocks for the selected cosmology tags.
- [ ] Validate `bfnl` likelihood closure and HOD-prior limiting behavior.
- [ ] Quantify assembly-bias and conformity effects on `b_phi` mean and scatter.
- [ ] Decide final `kmin`, `kmax`, and covariance settings.
- [ ] Export final chain/profile stats for all main comparisons.

### Figures

- [ ] Create pipeline overview figure.
- [ ] Regenerate triangle plots with final bphi stats and final chain settings.
- [ ] Make final `bfnl_z` and `f_NL^loc` comparison figure.
- [ ] Generate final `b_phi` prior table/plot across redshift.
- [ ] Decide whether P(k), covariance, and k-range diagnostics belong in appendix.

### Writing

- [ ] Fill in AbacusSummit simulation details.
- [ ] Replace `XX` and `YY` placeholders in the HOD mock section.
- [ ] Clarify DESI One-Percent/HOD posterior inputs.
- [ ] Write measurement and validation sections.
- [ ] Add limitations section covering universality, assembly bias, covariance, and external-data dependence.
- [ ] Fix notation and possible typo in the universality-relation equation in `paper/main.tex`.

## 11. Paper narrative

Current proposed story:

1. Problem: local PNG produces a large-scale `k^-2` bias signal, but galaxy-survey inference is limited by uncertainty in `b_phi`.
2. Method: build DESI-like HOD mocks on AbacusSummit, estimate `b_phi` from SU number-count response, and validate response interpretation with PNG simulations.
3. Main result: the power spectrum measures `bfnl_z`; HOD-informed `b_phi` priors translate that amplitude into `f_NL^loc` more transparently than unconstrained `b_phi` fits.
4. Why it matters: survey `f_NL` constraints need explicit, sample-specific `b_phi` information rather than relying only on the universality relation.
5. Caveats: final quantitative claims depend on HOD posterior coverage, assembly bias, covariance approximations, external HPC products, and validation of the `bfnl` implementation.
6. Future direction: extend to additional tracers, survey-window likelihoods, multi-tracer analyses, and more complete galaxy-formation systematics.

## 12. Relevant files

| Purpose | Path |
|---|---|
| Main paper draft | `paper/main.tex` |
| Bibliography | `paper/references.bib` |
| Main PNG/bphi fitting pipeline | `python_scripts/bphi_prior_pipeline.py` |
| Current pipeline wrapper | `python_scripts/bphi_prior_pipeline.sh` |
| Older/parallel bfnl pipeline wrapper | `python_scripts/bphi_prior_pipeline_bfnl.py`, `python_scripts/bphi_prior_pipeline_bfnl.sh` |
| HOD number-count to `b_phi` calculator | `python_scripts/HOD_bphi.py` |
| One-HOD production workflow | `python_scripts/HOD_one_lhc.py`, `python_scripts/HOD_one_lhc.sh` |
| HOD utilities, clustering, b1 estimation, NFW fallback | `python_scripts/HOD_utils.py` |
| P(k)/covariance diagnostic plotting | `python_scripts/bphi_prior_plot_pk.py` |
| PNG bias k-range diagnostic | `python_scripts/test_png_bias_krange.py` |
| Current work summary | `python_scripts/README.md` |
| Script changelog | `python_scripts/CHANGELOG.md` |
| HOD configs | `HOD_config/abacushod_z0.5_c000.yaml`, `HOD_config/abacushod_z0.8_c000.yaml`, `HOD_config/abacushod_z1.1_c000.yaml`, `HOD_config/lowS8_HOD_ELG.yaml` |
| FNL interpretation plan | `docs/superpowers/plans/fnl_sampling_implementation_validation_plan.md` |
| FNL validation plan | `docs/superpowers/plans/fnl_sampling_implementation_validation_plan.md` |
| Exploratory b1 notebook | `jupyter_notebooks/Fit_b1_to_AbacusSUmmit_HOD_mocks.ipynb` |

## 13. Changelog

### 2026-05-09

- Added initial project-state document from repository scan.
- Recorded the current paper narrative around `bfnl_z = b_phi,z * f_NL^loc`.
- Marked quantitative `b_phi` priors, `f_NL` constraints, and final paper figures as preliminary or missing where tracked artifacts are absent.
- Added open questions for `bfnl` validation, final bphi prior tables, assembly-bias robustness, and paper methods completion.
