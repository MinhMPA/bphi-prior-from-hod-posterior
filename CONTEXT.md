# bphi Prior from HOD Posterior

This context describes how PNG power-spectrum amplitudes are interpreted as `fnl_loc` constraints using assumptions about the PNG bias response `bphi`.

## Language

**bfnl_z**:
The code-facing name for the redshift-bin product amplitude constrained by the power spectrum. In the paper formalism this is `B_i = b_{\phi,i} fnl_loc`; in desilike configs and validation code it is usually `bfnl_z` or `bfnl_loc`.
_Avoid_: `bphi*fnl_loc posterior` when referring to the sampled amplitude; use `bfnl_z` or `bfnl_loc`.

**Direct bphi sampling**:
An inference path that samples `fnl_loc` and one or more `bphi_i` nuisance parameters jointly with explicit priors on each.
_Avoid_: Treating direct sampling as prior-free.

**bfnl post-processing**:
An inference path that samples data-constrained `B_i`/`bfnl_z` amplitudes first, then obtains `fnl_loc` by marginalizing over an external `bphi_i` prior.
_Avoid_: Sample-by-sample ratio conversion.

**Non-Gaussian bphi prior**:
An external `bphi_i` prior represented either by weighted samples or by a tabulated one-dimensional log-density.
_Avoid_: Assuming every `bphi_i` prior can be compressed to a mean and standard deviation.

**Gaussian HOD-informed bphi prior**:
The fiducial external prior in the paper, written as `pi_i(beta_i) = Normal(beta_i; mu_bphi_i, sigma_bphi_i^2)` where `beta_i` is the integration variable for `b_{\phi,i}`.
_Avoid_: Writing the prior width as `sigma_phi` or the mean as `mu_phi`, because `phi` can be confused with the primordial potential.

**Product-amplitude likelihood**:
The one-dimensional likelihood `L_i(B_i)` for the data-constrained product amplitude after marginalizing over the other nuisance parameters in bin `i`.
_Avoid_: Calling `L_i(B_i)` the marginalized likelihood; the marginalized object is `L_i^marg(fnl_loc)` after integrating over `beta_i`.

**Gaussian-Gaussian marginal likelihood**:
The analytic case where both `L_i(B_i)` and the Gaussian `bphi_i` prior are approximated as Gaussian. It uses `Bhat_i`, `sigma_B_i`, `mu_bphi_i`, and `sigma_bphi_i` directly.
_Avoid_: Using Gauss-Hermite quadrature for this analytic limit.

**Gauss-Hermite bphi marginalization**:
The numerical implementation for a Gaussian `bphi_i` prior when `L_i(B_i)` is kept as a numerical one-dimensional likelihood rather than replaced by a Gaussian approximation.
_Avoid_: Using Gauss-Hermite weights for non-Gaussian `bphi_i` priors.

**Grid validation mode**:
A deterministic quadrature validation that evaluates posterior densities on explicit `fnl_loc` and `bphi_i` grids to test the probability transformations without sampler noise.
_Avoid_: Treating grid validation as a production sampler.

**MCMC validation mode**:
A chain-based validation that compares actual sampled `fnl_loc` marginals from direct `bphi` sampling against post-processed `bfnl_z` chains.
_Avoid_: Using MCMC validation as the only proof of mathematical equivalence.

**Benchmark Scenario**:
A science-analysis case to compare in the paper or validation report, independent of the implementation strategy used to compute it.
_Avoid_: Reusing "Approach A/B" for implementation methods when the same scenario has multiple candidate implementations.

**Scenario B implementation strategy**:
One of the candidate computational routes for the HOD-informed Gaussian `bphi_i` prior scenario.
_Avoid_: Treating the direct-sampling and marginalized-likelihood routes as different science scenarios before validation selects the official implementation.

**B-direct**:
The Scenario B implementation strategy that samples `fnl_loc` and each `bphi_i` directly using desilike `mode="bphi"` with Gaussian `bphi_i` priors.
_Avoid_: Calling this "B-desilike", because B-marginal also builds on desilike components.

**B-marginal**:
The Scenario B implementation strategy that samples `fnl_loc` while marginalizing each Gaussian `bphi_i` prior in a custom likelihood wrapper. If `L_i(B_i)` is Gaussian, use the analytic Gaussian-Gaussian expression; if `L_i(B_i)` is numerical, use Gauss-Hermite quadrature for the Gaussian prior.
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

- The power spectrum constrains **B_i** (`bfnl_z` in code) directly, not **fnl_loc** alone.
- **Direct bphi sampling** and **bfnl post-processing** should agree when they use the same likelihood, `fnl_loc` prior, `bphi_i` prior, and numerical integration assumptions.
- **bfnl post-processing** integrates over **bphi_i**; it is not the unweighted ratio `bfnl_z / bphi_i`.
- The paper formalism writes this marginalization as `L_i^marg(f) = integral d beta_i L_i(B_i = f beta_i) pi_i(beta_i)`, then multiplies independent bin likelihood curves and applies the common `fnl_loc` prior once.
- In the Gaussian-Gaussian limit, evaluate the analytic expression directly from `Bhat_i`, `sigma_B_i`, `mu_bphi_i`, and `sigma_bphi_i`.
- Gauss-Hermite quadrature applies only when the `bphi_i` prior is Gaussian and `L_i(B_i)` is numerical.
- A **Non-Gaussian bphi prior** can be integrated numerically from weighted samples or from a tabulated log-density.
- The primary equivalence diagnostic between **Direct bphi sampling** and **bfnl post-processing** is the one-dimensional marginalized `fnl_loc` posterior.
- **bfnl post-processing** converts each redshift-bin **B_i**/`bfnl_z` constraint into an `fnl_loc` likelihood curve, multiplies those likelihood curves across bins, and applies the common `fnl_loc` prior once.
- Current validation treats redshift-bin **B_i**/`bfnl_z` constraints as independent.
- **Grid validation mode** proves the transformation math; **MCMC validation mode** checks the real sampler and pipeline outputs.
- **Scenario B implementation strategies** are benchmarked for convergence, bias, and runtime before choosing the official **Benchmark Scenario** implementation.
- **B-direct** and **B-marginal** should target the same Scenario B posterior over one universal `fnl_loc`.
- **Scenario A bfnl baseline** can only be cross-validated directly at the individual sample/bin **B_i**/`bfnl_z` level.
- **Scenario C fixed-p closure** compares per-bin implied `fnl_i = bfnl_i / [2 delta_c (b1_i - p_i)]` diagnostics and a combined universal `fnl_loc` posterior against direct `b-p` sampling.
- The **PNG production data vector** is `(ell=0, ell=2)` and uses the matching `thecov` Gaussian covariance.
- AbacusHOD LRG centrals are placed at halo centers, but their velocities can include central velocity bias/scatter.
- AbacusHOD LRG satellites are not generally halo-center objects; primary catalogs use particle/subsample placement, while secondary fallback catalogs use NFW placement.
- A **Redshift-space HOD catalog** must not be shifted to redshift space again downstream.

## Example Dialogue

> **Dev:** "Can I convert a `bfnl_z` chain to `fnl_loc` by dividing each sample by a `bphi_z` sample?"
> **Domain expert:** "No. For Scenario B, compute the marginal `fnl_loc` posterior by integrating the `B_i`/`bfnl_z` likelihood against the external `bphi_i` prior."

> **Dev:** "Eq. A.6 is analytic. Do I still need Gauss-Hermite quadrature?"
> **Domain expert:** "No, not when both `L_i(B_i)` and the `bphi_i` prior are Gaussian. Use Gauss-Hermite only for a Gaussian `bphi_i` prior with a numerical `L_i(B_i)`."

## Flagged Ambiguities

- "sample `bphi*fnl_loc`" means sample **bfnl_z** amplitudes in `mode="bfnl"`, not sample an already-converted **fnl_loc** posterior.
- `mu_beta` and `sigma_beta` in older notes mean the Gaussian `bphi_i` prior parameters now written as `mu_bphi_i` and `sigma_bphi_i`.
