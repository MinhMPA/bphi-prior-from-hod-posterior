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

**Benchmark Scenario**:
A science-analysis case to compare in the paper or validation report, independent of the implementation strategy used to compute it.
_Avoid_: Reusing "Approach A/B" for implementation methods when the same scenario has multiple candidate implementations.

**Scenario B implementation strategy**:
One of the candidate computational routes for the HOD-informed Gaussian `bphi_z` prior scenario.
_Avoid_: Treating the direct-sampling and marginalized-likelihood routes as different science scenarios before validation selects the official implementation.

**B-direct**:
The Scenario B implementation strategy that samples `fnl_loc` and each `bphi_z` directly using desilike `mode="bphi"` with Gaussian `bphi_z` priors.
_Avoid_: Calling this "B-desilike", because B-marginal also builds on desilike components.

**B-marginal**:
The Scenario B implementation strategy that samples `fnl_loc` while marginalizing each Gaussian `bphi_z` prior with Gauss-Hermite quadrature in a custom likelihood wrapper.
_Avoid_: Calling this "B-marglike" or implying it is an upstream desilike mode.

**Scenario A bfnl baseline**:
The no-`bphi`-prior benchmark that samples one `bfnl_z` amplitude per sample/bin and does not sample a universal `fnl_loc`.
_Avoid_: Comparing Scenario A to `fnl_loc` posteriors directly.

**Scenario C fixed-p closure**:
The universality benchmark that validates fixed-`p_i` inference both per sample/bin and after combining into one universal `fnl_loc` posterior.
_Avoid_: Treating per-bin implied `fnl_i` ratios as the same object as the direct jointly sampled universal `fnl_loc` posterior.

**PNG production data vector**:
The joint redshift-space galaxy power-spectrum multipole vector `(ell=0, ell=2)`, fit with the matching Gaussian covariance from `thecov`.
_Avoid_: Treating the quadrupole as a diagnostic-only quantity once the production plan is implemented.

**Redshift-space HOD catalog**:
An HOD mock whose positions have already been shifted to redshift space by the catalog generator, such as AbacusHOD LRG catalogs generated with `want_rsd=True`.
_Avoid_: Applying a second downstream RSD shift without proving the input catalog is still in real space.

**HOD satellite placement path**:
The rule used to place satellite galaxies in halos. For AbacusHOD LRG primary `use_particles=True` catalogs, satellites follow particle/subsample positions and include 1-halo velocity contribution; secondary-redshift fallback uses NFW placement.
_Avoid_: Saying all HOD galaxies sit at halo centers.

## Relationships

- The power spectrum constrains **bfnl_z** directly, not **fnl_loc** alone.
- **Direct bphi sampling** and **bfnl post-processing** should agree when they use the same likelihood, `fnl_loc` prior, `bphi_z` prior, and numerical integration assumptions.
- **bfnl post-processing** integrates over **bphi_z**; it is not the unweighted ratio `bfnl_z / bphi_z`.
- A **Non-Gaussian bphi prior** can be integrated numerically from weighted samples or from a tabulated log-density.
- The primary equivalence diagnostic between **Direct bphi sampling** and **bfnl post-processing** is the one-dimensional marginalized `fnl_loc` posterior.
- **bfnl post-processing** converts each redshift-bin **bfnl_z** constraint into an `fnl_loc` likelihood curve, multiplies those likelihood curves across bins, and applies the common `fnl_loc` prior once.
- Current validation treats redshift-bin **bfnl_z** constraints as independent.
- **Grid validation mode** proves the transformation math; **MCMC validation mode** checks the real sampler and pipeline outputs.
- **Scenario B implementation strategies** are benchmarked for convergence, bias, and runtime before choosing the official **Benchmark Scenario** implementation.
- **B-direct** and **B-marginal** should target the same Scenario B posterior over one universal `fnl_loc`.
- **Scenario A bfnl baseline** can only be cross-validated directly at the individual sample/bin **bfnl_z** level.
- **Scenario C fixed-p closure** compares per-bin implied `fnl_i = bfnl_i / [2 delta_c (b1_i - p_i)]` diagnostics and a combined universal `fnl_loc` posterior against direct `b-p` sampling.
- The **PNG production data vector** is `(ell=0, ell=2)` and uses the matching `thecov` Gaussian covariance.
- AbacusHOD LRG centrals are placed at halo centers, but their velocities can include central velocity bias/scatter.
- AbacusHOD LRG satellites are not generally halo-center objects; primary catalogs use particle/subsample placement, while secondary fallback catalogs use NFW placement.
- A **Redshift-space HOD catalog** must not be shifted to redshift space again downstream.

## Example Dialogue

> **Dev:** "Can I convert a `bfnl_z` chain to `fnl_loc` by dividing each sample by a `bphi_z` sample?"
> **Domain expert:** "No. For approach B, compute the marginal `fnl_loc` posterior by integrating the `bfnl_z` likelihood against the external `bphi_z` prior."

## Flagged Ambiguities

- "sample `bphi*fnl_loc`" means sample **bfnl_z** amplitudes in `mode="bfnl"`, not sample an already-converted **fnl_loc** posterior.
