# Paper TO DO

This checklist is for deliverables that directly determine whether the project becomes a credible paper. The near-term paper should not depend on the claim that our exact HOD-informed `b_phi` priors become community defaults. It should stand on a validated workflow, transparent assumptions, and useful empirical trends for DESI-like tracers.

## Scientific Products

- [ ] Produce a tracked table of `b_phi` summary statistics for the paper tracer/redshift samples.
- [ ] Include at minimum the LRG redshift sequence used in the main analysis.
- [ ] Add QSO and ELG summaries if their HOD/RSD conventions are audited well enough to avoid a caveated comparison dominating the paper.
- [ ] Report both the prior mean and width, with the exact definition of the central/reference HOD sample.
- [ ] Add a `b_phi(z)` trend plot for the available DESI-like samples.
- [ ] Add a `b_phi(b_1)` or `b_phi` versus universality-comparison plot.
- [ ] Mark the fixed-`p` universality prediction on the trend plots.
- [ ] State clearly that the HOD posterior choice is model-dependent and scientifically debatable.

## PNG Inference Validation

- [ ] Validate that the `B_i = b_{\phi,i} f_NL` baseline reports product-amplitude constraints without implying an identified `f_NL` posterior.
- [ ] Validate the HOD-informed Gaussian-prior case against direct `b_phi` sampling for the same likelihood and priors.
- [ ] Validate the fixed-`p` universality case against the expected deterministic relation `b_phi_i = 2 delta_c (b_1,i - p_i)`.
- [ ] Check limiting behavior near `f_NL = 0`.
- [ ] Check that the common `f_NL` prior is applied exactly once after multiplying independent bin likelihood curves.
- [ ] Document when the analytic Gaussian-Gaussian marginal likelihood is used.
- [ ] Document when numerical quadrature is used, and distinguish Gauss-Hermite from grid/sample integration for non-Gaussian priors.

## Pipeline And Reproducibility

- [ ] Finalize the joint `(ell=0, ell=2)` PNG likelihood path with the matching `thecov` Gaussian covariance.
- [ ] Add runtime checks that catch monopole-only data vectors or mismatched covariance shapes.
- [ ] Audit RSD handling for AbacusHOD LRG catalogs with `want_rsd=True`.
- [ ] Audit ELG/HODDIES RSD handling before promoting ELG to a main-paper benchmark.
- [ ] Record the HOD satellite-placement path in outputs where it affects interpretation.
- [ ] Export final chain/profile statistics used in paper figures and tables.
- [ ] Archive enough metadata to reproduce the final plots without reading private scratch paths by hand.

## Figures And Tables

- [ ] Create the paper pipeline schematic: HOD posterior to SU response to `b_phi` prior to PNG inference.
- [ ] Create the `b_phi(z)` and `b_phi(b_1)` trend figure.
- [ ] Create the `B_i`/`f_NL` comparison figure across Scenario A, Scenario B, and Scenario C.
- [ ] Regenerate triangle plots with final prior statistics and final chain settings.
- [ ] Produce a final prior table with tracer, redshift, `b_1`, `mu_bphi_i`, `sigma_bphi_i`, sample count, and caveats.
- [ ] Produce a final inference table with `B_i` constraints and `f_NL` constraints under the stated prior assumptions.

## Writing

- [ ] Fill in the simulation and HOD mock details that remain skeletal in `paper/main.tex`.
- [ ] Write the separate-universe `b_phi` measurement section.
- [ ] Write the AbacusSummitPNG validation section.
- [ ] Write the inference validation section.
- [ ] State the paper's actual deliverable precisely: a validated HOD-to-`b_phi` prior workflow plus preliminary DESI-like `b_phi(z,b_1)` trends.
- [ ] Avoid overclaiming that the released priors are definitive or likely to become community defaults.
- [ ] Add a limitations section covering HOD posterior dependence, assembly bias, covariance approximations, RSD conventions, and survey-window omissions.
- [ ] Make the abstract quantitative once final results exist.

## Submission Readiness

- [ ] Remove placeholders and commented result claims from the draft.
- [ ] Ensure all figures and tables are referenced in the text.
- [ ] Rebuild the PDF after every `main.tex` edit.
- [ ] Commit and push `main.tex` edits to the Overleaf repo.
- [ ] Run a final paper audit before submission.
