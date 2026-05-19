# Strategic TO CONSIDER

This checklist is for non-blocking ideas that could make the work more useful beyond the paper. These items should not block submission unless they become easy wins. The strongest long-term adoption path may be a clean reusable wrapper around `desilike` and `thecov`, not the exact HOD prior numbers released with this first paper.

## Reusable Package Direction

- [ ] Factor the marginal-likelihood logic into a small, documented module that can be used independently of the paper scripts.
- [ ] Provide a clean API for `L_i(B_i)` inputs, Gaussian `b_phi_i` priors, tabulated priors, and weighted-sample priors.
- [ ] Include analytic Gaussian-Gaussian marginalization as the fast path.
- [ ] Include Gauss-Hermite quadrature for Gaussian `b_phi_i` priors with numerical `L_i(B_i)`.
- [ ] Include grid/sample integration for non-Gaussian `b_phi_i` priors.
- [ ] Make prior application explicit so the common `f_NL` prior cannot be accidentally applied once per bin.
- [ ] Add examples that use synthetic likelihoods before requiring DESI-scale inputs.

## DESI / cosmodesi Adoption Path

- [ ] Identify the smallest interface needed for a `cosmodesi` or DESI-internal user to plug in product-amplitude likelihoods.
- [ ] Keep dependencies close to the existing DESI stack: `desilike`, `thecov`, NumPy, and standard chain formats.
- [ ] Avoid requiring this repository's HOD generation machinery for the reusable inference component.
- [ ] Write a short design note for how this wrapper could sit on top of `desilike` likelihood outputs.
- [ ] Prepare one end-to-end example using `thecov` covariance and a joint `(ell=0, ell=2)` data vector.
- [ ] Decide whether the wrapper belongs in this repo, a separate small repo, or as a PR/contribution to a cosmodesi-adjacent package.

## Prior Release Strategy

- [ ] Treat released HOD-informed priors as example products, not universal defaults.
- [ ] Provide machine-readable prior tables with clear provenance and caveats.
- [ ] Record which HOD posterior, tracer selection, redshift, RSD convention, and satellite-placement path produced each prior.
- [ ] Support alternative prior inputs so collaborators can substitute their own HOD posterior products.
- [ ] Separate "recommended for this paper" priors from "format example" priors.
- [ ] Consider releasing only trend summaries if the exact HOD posterior choices remain too collaboration-specific.

## Longer-Term Science Extensions

- [ ] Extend the `b_phi(z,b_1)` trend comparison across LRG, QSO, and ELG samples.
- [ ] Quantify whether the redshift evolution is mainly driven by tracer bias, HOD selection, or sample definition.
- [ ] Compare HOD-informed trends against fixed-`p` universality over the same `b_1` range.
- [ ] Test assembly-bias and conformity variants as systematic envelopes rather than as fiducial priors.
- [ ] Explore whether multi-tracer combinations change which `b_phi` prior details matter most.
- [ ] Revisit survey-window and non-Gaussian covariance effects once the baseline workflow is stable.

## Communication And Memory Value

- [ ] Frame the paper as a validated workflow plus empirical DESI-like trends, not as the final word on `b_phi` priors.
- [ ] Make the reusable wrapper the thing collaborators can remember and reuse after the specific priors are debated.
- [ ] Keep one figure that communicates the practical inference problem: data constrain `B_i`, external information turns it into `f_NL`.
- [ ] Keep one figure that communicates the science trend: how `b_phi` varies with redshift and `b_1` for DESI-like samples.
- [ ] Prepare a short "how to use your own `b_phi` prior" example for future collaborators.
