# Human Summary: fnl_loc Posterior Comparison and Validation Plan

## Purpose

The study asks how the recovered `fnl_loc` posterior changes when we make different assumptions about the PNG bias response `bphi`.

The central point is that the power spectrum does not directly measure `fnl_loc` alone. To leading order, it measures the scale-dependent bias amplitude

```text
bfnl_z = bphi_z * fnl_loc
```

for each redshift bin or tracer sample. Therefore, any `fnl_loc` posterior is partly a statement about the data and partly a statement about the assumed relation or prior for `bphi_z`. The comparison should make that separation explicit.

## Main Comparison Setup

The proposed main comparison has three cases.

### 1. Universality Relation, Fixed p

This corresponds to the common analysis assumption

```text
bphi_z = 2 * delta_c * (b1_z - p)
```

with `p` fixed, usually `p = 1`.

This should be presented using direct `b-p` sampling because that is closest to how standard analyses are implemented in practice. It answers the question: "What `fnl_loc` posterior would we report if we assumed the universal mass-function relation?"

This is not just a prior on `bphi`; it is a deterministic relation tying `bphi_z` to the fitted linear bias `b1_z`. Because `b1_z` is constrained by the broadband clustering, this assumption can strongly shape the inferred `fnl_loc` posterior.

### 2. HOD-Informed Gaussian Prior on bphi

This case uses external HOD/separate-universe information:

```text
bphi_z ~ Normal(mu_z, sigma_z)
```

It is better to implement this around the data-constrained amplitude `bfnl_z`, then infer `fnl_loc` through the relation `bfnl_z = bphi_z * fnl_loc`.

This avoids forcing the sampler to explore the long multiplicative degeneracy between `fnl_loc` and `bphi_z`. It also keeps the interpretation clean:

```text
P(k) data constrain bfnl_z.
HOD information maps bfnl_z into fnl_loc.
```

The Gaussian `bphi` prior should be applied in a forward conditional form:

```text
bfnl_z | fnl_loc ~ Normal(fnl_loc * mu_z, sqrt(data_variance_z + fnl_loc^2 * sigma_z^2))
```

This avoids explicit division by `fnl_loc`, which is awkward because `fnl_loc = 0` is allowed. A small numerical floor may be used only to keep the likelihood finite in edge cases; any such floor should be tested for sensitivity.

### 3. No Prior on bphi

This case should not be presented as a well-defined `fnl_loc` posterior.

Without a prior or relation for `bphi_z`, `fnl_loc` is not identified by the power spectrum. The data constrain only `bfnl_z = bphi_z * fnl_loc`. Any apparent `fnl_loc` posterior in this case would come from an implicit prior volume choice on `bphi_z`.

The transparent presentation is:

```text
No bphi assumption: report bfnl_z constraints.
```

Then explicitly demonstrate that converting this to `fnl_loc` requires a `bphi` assumption and that broadening the `bphi` prior changes the `fnl_loc` posterior.

## Why Use bfnl_z as the Common Diagnostic

Using `bfnl_z` as the common diagnostic gives an apples-to-apples comparison because it is the quantity most directly constrained by the data.

The three science cases can then be described as different interpretations of the same measured amplitude:

```text
Universality: bphi_z is fixed by b1_z and p.
HOD prior: bphi_z is externally constrained with a Gaussian prior.
No bphi prior: bphi_z is unconstrained, so fnl_loc is not identifiable.
```

This framing avoids confusing a sampling parameterization with a physical assumption. It also makes clear that the difference between posteriors is caused by the assumption on `bphi`, not by a hidden change in the likelihood.

## Validation Tests

The validation tests are designed to show that the reported `fnl_loc` posteriors are not artifacts of parameterization or implementation.

### Validation 1: Fixed-p Closure

Compare:

```text
direct b-p sampling
```

against:

```text
bfnl_z likelihood/posterior + fixed-p post-processing
```

with matched priors and the same constrained relation

```text
bfnl_z = 2 * delta_c * (b1_z - p) * fnl_loc
```

These should agree. If they do not, the difference likely comes from prior mismatch, Jacobian/measure mismatch, sampler failure, or an implementation difference. This test supports the validity of using `bfnl_z` as the common comparison object.

### Validation 2: HOD Prior Limiting Behavior

Check that the HOD-informed posterior behaves sensibly:

- A tight `bphi` prior should recover approximately `fnl_loc = bfnl_z / mu_bphi`.
- A broader `bphi` prior should broaden the `fnl_loc` posterior.
- Multiple redshift bins should combine through one common `fnl_loc`.
- The posterior should remain finite and stable near `fnl_loc = 0`.

This validates that the Gaussian `bphi` prior is being applied as intended and that the zero-crossing caveat is under numerical control.

### Validation 3: No-bphi Non-Identifiability

Demonstrate that as the `bphi` prior is broadened, the inferred `fnl_loc` posterior becomes increasingly prior-volume dependent.

This is not a failure. It is the expected behavior and supports the claim that the no-`bphi` case should be reported as a `bfnl_z` constraint rather than as an `fnl_loc` posterior.

### Validation 4: Optional bphi-Mode Cross-Check

Direct `bphi` sampling can be run as a cross-check for the HOD-prior case, but it should not be the primary implementation. It samples `fnl_loc` and `bphi_z` separately even though the data mostly constrain their product, which can make sampling inefficient and prior-volume effects harder to diagnose.

If direct `bphi` sampling and the `bfnl_z`-based HOD implementation agree under matched priors, that is useful supporting evidence. If they disagree, the `bfnl_z` implementation should be easier to debug because it isolates the data-constrained amplitude from the external prior.

## Recommended Presentation

The cleanest paper or internal-note structure is:

1. Show the `bfnl_z` constraints from the power spectrum.
2. Show `fnl_loc` posterior under fixed-`p` universality.
3. Show `fnl_loc` posterior under the HOD-informed Gaussian `bphi` prior.
4. Show that without a `bphi` assumption, only `bfnl_z` is identifiable.
5. Include validation plots or tables showing fixed-`p` closure and HOD-prior limiting behavior.

This makes the scientific message transparent:

```text
The data measure the PNG bias amplitude.
The recovered fnl_loc posterior depends on how bphi is assumed or constrained.
```
