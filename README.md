# bphi Prior from HOD Posterior

Constructing `b_phi` priors from posterior constraints on HOD parameters.

Jiaxi Yu and Nhat-Minh Nguyen (2026), arXiv:26XX.XXXXX.

## Analysis Contract

The target PNG likelihood fits the joint redshift-space galaxy power-spectrum
multipole data vector

```text
d_z = [P_0^g(k), P_2^g(k)]
```

for each tracer/redshift sample. The monopole (`ell=0`) and quadrupole
(`ell=2`) are fit jointly, using the matching Gaussian covariance supplied by
`thecov`. The quadrupole is part of the production data vector, not a
diagnostic-only quantity.

The HOD catalog-space convention is:

- AbacusHOD LRG catalogs generated with `want_rsd: True` are already shifted to
  redshift space by AbacusHOD. Downstream clustering or power-spectrum code must
  not apply a second RSD shift to those LRG positions.
- LRG centrals are placed at halo-center positions, but their velocities include
  the configured central velocity-bias/scatter contribution, e.g.
  `vel + alpha_c * vdev`.
- LRG satellites are not generally placed at halo centers. In the primary
  `use_particles: True` path, satellite positions follow particle/subsample
  positions and velocities include a 1-halo contribution controlled by
  `alpha_s`.
- AbacusHOD secondary-redshift fallback catalogs use NFW satellite placement,
  which has different velocity-bias behavior from the particle-subsample path
  and must be documented separately in validation outputs.
- ELG/HODDIES catalogs use a separate velocity/RSD convention. Before ELG
  benchmarks are promoted to production validation, audit whether HODDIES has
  already applied RSD so the pipeline does not double shift the catalog.

## Benchmark Scenarios

- **Scenario A, bfnl baseline:** sample one `bfnl_z = bphi_z * fnl_loc`
  amplitude per independent tracer/redshift bin in desilike `mode="bfnl"`,
  with no `bphi_z` prior and no universal sampled `fnl_loc`.
- **Scenario B, HOD-informed Gaussian bphi prior:** infer one universal
  `fnl_loc` using one Gaussian HOD prior for each `bphi_z`. Validate both
  B-direct (`mode="bphi"` sampling) and B-marginal (custom marginal likelihood)
  before choosing the official implementation.
- **Scenario C, fixed-p universality:** fix
  `bphi_i = 2 delta_c (b1_i - p_i)` using tracer/sample-specific fixed `p_i`
  values for LRG, ELG, and QSO, then infer one universal `fnl_loc`.
