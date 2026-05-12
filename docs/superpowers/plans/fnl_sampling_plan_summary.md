# Current Human Summary: fnl_loc Posterior Comparison and Validation Plan

## Purpose

The study compares two valid ways to infer the one-dimensional marginalized `fnl_loc` posterior from power-spectrum data and external information about the PNG bias response `bphi`.

The central quantity constrained by the power spectrum is the redshift-bin PNG amplitude

```text
B_z = bfnl_z = bphi_z * fnl_loc
```

The power spectrum does not identify `fnl_loc` by itself. Any `fnl_loc` posterior is therefore a statement about both the data-constrained `bfnl_z` amplitudes and the assumed prior or relation for `bphi_z`.

## Active Comparison Setup

The active validation compares two inference paths under matched assumptions.

### Approach A: Direct bphi Sampling

Sample `fnl_loc` and the redshift-bin `bphi_z` parameters directly in the chain:

```text
P(f, beta_z | data) proportional to P(f) product_z P(beta_z) L_z(B_z = f beta_z)
```

where

```text
f = fnl_loc
beta_z = bphi_z
B_z = bfnl_z
```

This approach gives direct sampled one-dimensional marginals for both `fnl_loc` and each `bphi_z`. It is statistically valid when the priors on `fnl_loc` and `bphi_z` are explicit. Its main practical weakness is that the likelihood mostly depends on the product `f * beta_z`, so the joint posterior can have a long multiplicative degeneracy.

### Approach B: bfnl Sampling plus bphi Marginalization

Run the pipeline in `mode="bfnl"` so each redshift bin has its own data-constrained amplitude:

```text
B_z = bfnl_z
```

Then convert each redshift-bin `bfnl_z` constraint into an `fnl_loc` likelihood curve by marginalizing over that bin's external `bphi_z` prior:

```text
L_z(f) = integral d beta_z L_z(B_z = f beta_z) pi_z(beta_z)
```

Combine redshift bins by multiplying these likelihood curves and applying the common `fnl_loc` prior exactly once:

```text
P(f | all data) proportional to pi(f) product_z L_z(f)
```

Do not multiply per-bin `fnl_loc` posteriors directly if each one already includes `pi(f)`, because that counts the `fnl_loc` prior multiple times.

### Gaussian bphi Priors

For Gaussian compressed `bfnl_z` likelihoods and Gaussian `bphi_z` priors,

```text
Bhat_z | B_z ~ Normal(B_z, sigma_B,z)
beta_z ~ Normal(mu_beta,z, sigma_beta,z)
```

the marginalization is analytic:

```text
Bhat_z | f ~ Normal(f * mu_beta,z,
                    sqrt(sigma_B,z^2 + f^2 sigma_beta,z^2))
```

This is the fast path for initial validation.

### Non-Gaussian bphi Priors

For non-Gaussian `bphi_z` priors, evaluate the same marginalization numerically:

```text
L_z(f) = integral d beta_z L_z(B_z = f beta_z) pi_z(beta_z)
```

Validation should support both prior representations:

```text
1. weighted bphi samples: beta_samples_z, weights_z
2. tabulated bphi log-density: beta_grid_z, logpdf_z
```

KDE smoothing may be used outside the validation module to create a tabulated log-density, but KDE bandwidth should not be a hidden part of the canonical validation definition.

## Validation Modes

### grid Mode

`grid` mode is deterministic quadrature. It evaluates posterior densities on explicit `fnl_loc` and `bphi_z` grids and integrates numerically. It is for proving the probability transformations without sampler noise.

Use `grid` mode to verify:

- Approach A and Approach B produce matching `fnl_loc` marginals under matched priors.
- The Gaussian analytic formula matches numerical integration.
- Weighted-sample and tabulated non-Gaussian `bphi_z` priors are integrated consistently.
- Independent redshift-bin likelihood curves combine with the `fnl_loc` prior applied once.

### mcmc Mode

`mcmc` mode is chain-based validation using chains produced by the real desilike pipeline only.

Use `mcmc` mode to compare:

```text
direct bphi chain fnl_loc marginal
vs
bfnl-mode chain post-processed fnl_loc marginal
```

The goal is not a mathematically exact proof; that is the job of `grid` mode. The goal is to check actual sampler behavior, chain processing, and pipeline outputs.

## Fixed-p Universality Case

The fixed-p universality relation remains a separate validation and presentation case:

```text
bphi_z = 2 * delta_c * (b1_z - p)
```

with `p` fixed, usually `p = 1`.

Direct `b-p` sampling should be compared against a constrained `bfnl_z` post-processing calculation only when priors and measures are matched:

```text
B_z = 2 * delta_c * (b1_z - p) * fnl_loc
```

Agreement validates the parameterization; disagreement points to prior mismatch, Jacobian/measure mismatch, sampler failure, or implementation differences.

## Recommended Presentation

The cleanest paper or internal-note structure is:

1. Show `bfnl_z` constraints from the power spectrum.
2. Show `fnl_loc` posterior from direct `bphi` sampling with explicit `bphi_z` priors.
3. Show `fnl_loc` posterior from `bfnl` sampling plus post-processing marginalization over the same `bphi_z` priors.
4. Demonstrate that the two `fnl_loc` marginals agree under matched assumptions.
5. Show fixed-p universality as a separate deterministic-relation case.
6. State that without a `bphi_z` prior or relation, only `bfnl_z` is identifiable.
