# bphi Prior from HOD Posterior

This context describes how PNG power-spectrum amplitudes are interpreted as `fnl_loc` constraints using assumptions about the PNG bias response `bphi`.

## Language

**bfnl_z**:
The redshift-bin PNG amplitude constrained by the power spectrum, equal to `bphi_z * fnl_loc`.
_Avoid_: `bphi*fnl_loc posterior` when referring to the sampled amplitude; use `bfnl_z` or `bfnl_loc`.

**Direct bphi sampling**:
An inference path that samples `fnl_loc` and one or more `bphi_z` nuisance parameters jointly with explicit priors on each.
_Avoid_: Treating direct sampling as prior-free.

**bfnl post-processing**:
An inference path that samples data-constrained `bfnl_z` amplitudes first, then obtains `fnl_loc` by marginalizing over an external `bphi_z` prior.
_Avoid_: Sample-by-sample ratio conversion.

**Non-Gaussian bphi prior**:
An external `bphi_z` prior represented either by weighted samples or by a tabulated one-dimensional log-density.
_Avoid_: Assuming every `bphi_z` prior can be compressed to a mean and standard deviation.

**Grid validation mode**:
A deterministic quadrature validation that evaluates posterior densities on explicit `fnl_loc` and `bphi_z` grids to test the probability transformations without sampler noise.
_Avoid_: Treating grid validation as a production sampler.

**MCMC validation mode**:
A chain-based validation that compares actual sampled `fnl_loc` marginals from direct `bphi` sampling against post-processed `bfnl_z` chains.
_Avoid_: Using MCMC validation as the only proof of mathematical equivalence.

## Relationships

- The power spectrum constrains **bfnl_z** directly, not **fnl_loc** alone.
- **Direct bphi sampling** and **bfnl post-processing** should agree when they use the same likelihood, `fnl_loc` prior, `bphi_z` prior, and numerical integration assumptions.
- **bfnl post-processing** integrates over **bphi_z**; it is not the unweighted ratio `bfnl_z / bphi_z`.
- A **Non-Gaussian bphi prior** can be integrated numerically from weighted samples or from a tabulated log-density.
- The primary equivalence diagnostic between **Direct bphi sampling** and **bfnl post-processing** is the one-dimensional marginalized `fnl_loc` posterior.
- **bfnl post-processing** converts each redshift-bin **bfnl_z** constraint into an `fnl_loc` likelihood curve, multiplies those likelihood curves across bins, and applies the common `fnl_loc` prior once.
- Current validation treats redshift-bin **bfnl_z** constraints as independent.
- **Grid validation mode** proves the transformation math; **MCMC validation mode** checks the real sampler and pipeline outputs.

## Example Dialogue

> **Dev:** "Can I convert a `bfnl_z` chain to `fnl_loc` by dividing each sample by a `bphi_z` sample?"
> **Domain expert:** "No. For approach B, compute the marginal `fnl_loc` posterior by integrating the `bfnl_z` likelihood against the external `bphi_z` prior."

## Flagged Ambiguities

- "sample `bphi*fnl_loc`" means sample **bfnl_z** amplitudes in `mode="bfnl"`, not sample an already-converted **fnl_loc** posterior.
