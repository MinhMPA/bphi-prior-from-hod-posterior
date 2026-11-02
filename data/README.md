# Released Data

## Empirical b_phi Samples

[`bphi_prior_samples.csv`](bphi_prior_samples.csv) contains four groups of
100 accepted HOD realizations, one row per realization. Each row has equal
weight within its group. The file is plain UTF-8 CSV and can be read with
Python's `csv` module, NumPy, pandas, or spreadsheet software.

| Column | Meaning |
|---|---|
| `sample` | `LRG1`, `LRG2`, `LRG3`, or `LRG3_AB` |
| `z_snapshot` | Simulation snapshot redshift |
| `ensemble` | `chain_s1` or broadened `chain_s3` HOD ensemble |
| `draw_id` | Identifier of the accepted realization within that ensemble; gaps can occur |
| `source_row` | Zero-based row index in the processed source HOD posterior |
| `n_c000`, `n_c004` | Galaxy counts in the paired fiducial and separate-universe simulations |
| `bphi` | Separate-universe response computed from those counts |
| `Bcent`, `Bsat` | Fixed AbacusHOD assembly-bias parameters |

The response is `2 * ln(n_c000/n_c004) / ln(0.807952/0.75)`.
Both counts are strictly greater than 100000. LRG1 and LRG2 use
`chain_s3`, produced by scaling sampled full HOD vectors about the
posterior center by three and accepting physical realizations; LRG3 uses
the unbroadened `chain_s1`. Posterior vectors are drawn with replacement:
repeated source rows are valid, independently counted draws. The two LRG3
groups use the same underlying HOD draw IDs and source rows, with assembly
bias switched on only for `LRG3_AB`. Do not interpret matching row IDs
across distinct redshift bins as a joint HOD correlation.

For example:

```python
import csv

with open("data/bphi_prior_samples.csv", newline="") as stream:
    rows = list(csv.DictReader(stream))
lrg1 = [float(row["bphi"]) for row in rows if row["sample"] == "LRG1"]
```

The associated [exporter](../python_scripts/export_bphi_samples.py) checks
the raw input rows and response formula before writing this file, and the
[plotting script](../python_scripts/plot_bphi_priors.py) reproduces simple
distributions and Gaussian moment summaries.

The same 400 rows are the data behind the four panels of
`paper_bphi_empirical_distributions.pdf`. The overlaid Gaussian curves use
the corresponding sample mean and population standard deviation (`ddof=0`);
the assembly-bias panel is a separate test, not the fiducial LRG3 prior.

## Reference HOD Chains

[`hod_reference_chains/`](hod_reference_chains/) contains copies of the three
LRG HOD posterior chains used to construct the ensembles above. They are from
Yuan et al., [The DESI One-Percent Survey: Exploring the Halo Occupation
Distribution of Luminous Red Galaxies and Quasi-Stellar Objects with
AbacusSummit](https://arxiv.org/abs/2306.06314), and their original public
release is [Zenodo 7972386](https://doi.org/10.5281/zenodo.7972386). Please
cite the paper and original data release when using these chains. These are
reference HOD chains, not the PNG-fit chains generated in this work; the
`b_phi` responses and broadened samples in this repository are our derived
products, not files from the original release.
The chain checksums below were verified against our saved analysis manifest;
the original Zenodo archive and its redistribution terms were not independently
checked during this packaging.

| HOD sample | Chain file | Snapshot used here |
|---|---|---:|
| LRG, `0.4 < z < 0.6` | `LRG_main_z0406_chains_xi2d.npz` | 0.5 |
| LRG, `0.6 < z < 0.8` | `LRG_main_z0608_chains_xi2d.npz` | 0.8 |
| LRG, `0.95 < z < 1.1` | `LRGzgt08Chains/lrg_z09511.txt` | 1.1 |

The two low-redshift files contain weighted Dynesty nested-sampling results;
their inferred parameter order and reader are documented in
[`extract_hod_covariances.py`](../python_scripts/extract_hod_covariances.py).
The NPZ files contain pickled objects, so load them only from a trusted source.
The high-redshift file is a weighted text chain, accompanied by its original
`.paramnames`, `.ranges`, and `.properties.ini` files; its first two columns
are weight and negative log likelihood, and its burn-in was already removed.
Our analysis also applies its recorded parameter conversions, physical and
galaxy-count selection, and, for LRG1/2, threefold broadening. Raw chain rows
are therefore not themselves the 100 `b_phi` values. Verify the copied chain
files with `cd data/hod_reference_chains && sha256sum -c SHA256SUMS`.

## Mock Power Spectra And Covariances

[`paper_pk/`](paper_pk/) contains the data behind
`paper_pk_stacked_vertical_offsets.pdf`. Each `LRG{1,2,3}_c{000,300,301}_pk.csv`
contains the fitted-bin data for one simulation and snapshot:

| Column | Meaning |
|---|---|
| `k_h_Mpc` | Bin wavenumber, in `h/Mpc` |
| `P0`, `P2` | Mock power-spectrum monopole and quadrupole, in `(Mpc/h)^3` |
| `bestfit_P0`, `bestfit_P2` | Model multipoles at the maximum-likelihood retained chain state, in `(Mpc/h)^3` |
| `sigma_P0`, `sigma_P2` | Square roots of the corresponding covariance diagonal, in `(Mpc/h)^3` |

The matching `_covariance.csv` is the full numerical covariance in
`(Mpc/h)^6`, including the monopole--quadrupole blocks. Its row and column
order is all `P0` bins in CSV order, followed by all `P2` bins in the same
order. It has shape `84 x 84` for LRG1/2 and `94 x 94` for LRG3. For a given
sample the covariance is identical across the three cosmologies: all use the
standard c000 analytic Gaussian covariance with effective-volume scaling,
not a covariance estimated separately for each mock.

The input simulations have `f_NL = 0, +30, -30` for `c000`, `c300`, `c301`,
respectively. LRG1/2 use `kmax = 0.09 h/Mpc` and LRG3 `0.10 h/Mpc`; the
lower selection is `k >= 0.006 h/Mpc`, with nominal bin width `0.002 h/Mpc`.
The input monopole uses pypower's default shot-noise subtraction. The files
contain unshifted `P_ell`, not the display-offset `k P_ell` values. The figure
plots `k P_ell + offset` and uses errors `k sigma_Pell`; for `f_NL=+30/-30`
the offsets are `+/-200` for the monopole and `+/-400` for the quadrupole,
in `(Mpc/h)^2`. The plotted residual is `(P_ell - bestfit_Pell)/sigma_Pell`.
The `manifest.json` records fit settings and source-input hashes.

Recreate the figure using only NumPy and Matplotlib:

```bash
python python_scripts/plot_released_pk.py --output released_pk.pdf
```

The [`export_paper_pk_data.py`](../python_scripts/export_paper_pk_data.py)
script records how these files were extracted. Rerunning that exporter is
optional and requires the original analysis installation, fit chains, data,
covariances, and scientific environment; the released CSVs are sufficient to
reproduce the figure. They are simulated mock measurements, not observed DESI
galaxy power spectra.
