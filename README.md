# bphi Prior from HOD Posterior

This repository provides HOD-derived samples of the local-PNG response
`b_phi` and scripts for using them in galaxy power-spectrum fits. The
released empirical priors are in [data/bphi_prior_samples.csv](data/bphi_prior_samples.csv);
their format and provenance are described in [data/README.md](data/README.md).
The same data directory also contains the reference DESI One-Percent LRG HOD
chains and the mock power spectra, full covariances, and best-fit model values
used in the paper figures.
The HOD chains are from Yuan et al.'s [DESI One-Percent LRG and QSO HOD
analysis](https://arxiv.org/abs/2306.06314); their original public data release
is [Zenodo 7972386](https://doi.org/10.5281/zenodo.7972386).

## Released Priors

| Sample | Snapshot redshift | HOD ensemble | Number of values | Mean | Standard deviation |
|---|---:|---|---:|---:|---:|
| LRG1 | 0.5 | `chain_s3` | 100 | 2.5561 | 0.4234 |
| LRG2 | 0.8 | `chain_s3` | 100 | 3.5396 | 0.4176 |
| LRG3 | 1.1 | `chain_s1` | 100 | 3.9469 | 0.1884 |
| LRG3_AB | 1.1 | `chain_s1`, assembly bias | 100 | 3.9454 | 0.1877 |

The quoted standard deviations use `ddof=0`. The assembly-bias sample is
a separate test with `Bcent=-0.04` and `Bsat=-0.17`; it is not the fiducial
LRG3 prior. These are empirical samples, not Gaussian draws. For a quick
plot and moment-matched Gaussian comparison, run:

```bash
python python_scripts/plot_bphi_priors.py --output bphi_priors.pdf
```

This requires a compatible installation of NumPy and Matplotlib. The script
also accepts a PNG output filename and prints the means, variances and standard
deviations.

## Current Analysis

The fiducial informed analysis uses empirical `b_phi` priors in `bphi`
mode, with `chain_s3` for LRG1 and LRG2 and `chain_s1` for LRG3.
The fiducial maximum wavenumbers are 0.09, 0.09 and 0.10 h/Mpc, respectively.
The joint fit uses a common `f_NL` and independent per-bin prior measures;
a row number does not imply a cross-redshift HOD pairing. Alternative
configurations support Gaussian and KDE approximations and other scale cuts.

Each response is computed from separate-universe galaxy counts as

```text
b_phi = 2 ln(N_c000 / N_c004) / ln(0.807952 / 0.75).
```

The released draws satisfy the galaxy-count selection in both simulations.
The `chain_s3` ensemble is constructed by scaling complete HOD posterior
vectors about their center by three, followed by physical and count
selection; it does not assume a Gaussian HOD posterior. See the data README
for the meaning of the draw identifiers and assembly-bias pairing.

## Repository Guide

- [data/](data/): response samples, original reference HOD chains, paper
  mock power-spectrum data and covariances, and field definitions.
- [python_scripts/plot_bphi_priors.py](python_scripts/plot_bphi_priors.py):
  standalone plot and summary of the released data.
- [python_scripts/plot_released_pk.py](python_scripts/plot_released_pk.py):
  recreate the paper power-spectrum figure from the released CSV files.
- [python_scripts/export_bphi_samples.py](python_scripts/export_bphi_samples.py):
  validates and exports the samples from the original analysis products.
- [python_scripts/bphi_prior_shape_test/](python_scripts/bphi_prior_shape_test/):
  current configurable empirical, Gaussian and KDE `bphi`-mode fits,
  configurations, and assembly-bias machinery.
- [python_scripts/hod_correlation_test/](python_scripts/hod_correlation_test/):
  HOD ensemble construction, separate-universe counts and summary utilities.
- [python_scripts/](python_scripts/): weak-prior and Gaussian consistency
  checks, plus the original LHC-based mock and `bfnl`/`bphi`/`b-p` analysis
  and plotting utilities; see its
  [README](python_scripts/README.md) for the distinction from the updated
  empirical-prior workflow. Files in `archive/` are historical.

## Running Fits

The CSV and its plotting script are self-contained apart from Python packages.
Full fits and HOD calculations additionally require a compatible scientific
environment, halo catalogues, and the original analysis input formats. The
reference HOD chains and figure-ready mock spectra and covariances are included,
but the halo catalogues, original power-spectrum objects, and PNG-fit MCMC
chains are not. Several validation comparisons also require historical fit
chains. Data locations and cluster settings must be configured for the
installation; see [python_scripts/README.md](python_scripts/README.md).

With those inputs and dependencies available, an example joint fit is:

```bash
export BPHI_DATA_ROOT=/path/to/original/b_phi_inputs
export BPHI_RESULTS_ROOT=/path/to/output
cd python_scripts/bphi_prior_shape_test
bash run_inference.sh check --config configs/joint_empirical_k09_09_10.json --cosmos 000
# Run on an allocated interactive CPU node, after the check passes:
bash run_inference.sh fit --config configs/joint_empirical_k09_09_10.json --cosmos 000
```

The check prepares a frozen input manifest; the fit resumes compatible saved
checkpoints and reports its convergence status. Do not run concurrent writers
for the same output. Use separate output directories or configurations for
distinct experiments. The Slurm launchers reflect the original NERSC setup and
may require account, allocation and environment changes.
