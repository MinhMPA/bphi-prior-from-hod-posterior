# bphi Prior From HOD Posterior

This directory contains the active scripts used to build HOD mock data vectors, construct abundance-response `b_phi` priors, run PNG power-spectrum fits with `desilike`, and make the validation plots used in the paper.  Files under `archive/` are old development copies and are not part of the current workflow. All the chains and contours are available on idark/igpu at 
```bash
/lustre/work/jiaxiyu/bphi/fit
```

## Environment

Most scripts assume the Perlmutter/cosmodesi stack:

```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
cd /pscratch/sd/j/jiaxi/b_phi/bphi-prior-from-hod-posterior/python_scripts
```

For plotting on compute/login nodes without a display, the plotting shell script also sets:

```bash
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl-bphi}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/mpl-bphi}
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
export JAX_PLATFORM_NAME=${JAX_PLATFORM_NAME:-cpu}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
```

## Directory Layout Expected by the Scripts

The scripts read and write paths relative to this directory:

```text
mocks_lhc_SU/                         SU HOD P(k), 2pt and covariance files
mocks_lhc_PNG/                        PNG HOD P(k), 2pt and covariance files
backup/bphi_AbacusHOD_LRG.txt         default b_phi prior table
backup/bphi_AbacusHOD_z1.1_updated.txt
                                       updated z=1.1 b_phi prior table
bphi_results_abacushod_Bcentm0p040_Bsatm0p170.txt
                                       assembly-bias SU abundance table
bphi_true_value_PNGmock_fit/b1_with-sn0.txt
                                       best-fit b1 values used as truth markers
bphi_prior/fit/                       chain/profile/triangle outputs
bphi_prior/plots/                     validation and summary plots
```

If these products live elsewhere, either run from the production `b_phi` directory or create links/copies before running the scripts.

## Main Scripts

`bphi_prior_pipeline.py` is the main fit pipeline.  It has four subcommands:

```bash
python bphi_prior_pipeline.py pk   --help
python bphi_prior_pipeline.py cov  --help
python bphi_prior_pipeline.py fit  --help
python bphi_prior_pipeline.py plot --help
```

`bphi_prior_pipeline.sh` in this snapshot mirrors the Python source rather than acting as the driver shell script, so use `bphi_prior_pipeline.py` directly or use the smaller driver scripts listed below.

`HOD_LHC.py` defines the HOD Latin-hypercube parameter samples.  The parameter means and 1-sigma errors are taken from the reference HOD constraints; where errors were asymmetric, the larger 1-sigma side was used.  The current LRG implementation samples a Gaussian Latin hypercube with `N=100`; z=0.5 and z=0.8 use a 3-sigma scale, while z=1.1 uses the quoted 1-sigma scale.

`HOD_one_lhc.py` runs one HOD point, writes clustering products, and optionally estimates real-space `b1`.  The shell wrapper `HOD_one_lhc.sh` is configured for LRG z=1.1 in this snapshot and supports assembly-bias mocks through:

```bash
bash HOD_one_lhc.sh LRG SU 1
```

which passes:

```bash
--Bcent -0.04 --Bsat -0.17
```

Assembly-bias mock outputs get the suffix:

```text
_Bcentm0p040_Bsatm0p170
```

so they do not overwrite standard HOD files.

## b_phi Priors

The SU abundance-response prior is computed from HOD number counts as:

```text
b_phi = 2 ln(Ngal_c000 / Ngal_c004) / ln(0.807952 / 0.75)
```

Invalid realizations are removed by requiring both `Ngal_c000` and `Ngal_c004` to be greater than `100000`.  The pipeline loads the default prior table from `backup/bphi_AbacusHOD_LRG.txt` and always replaces z=1.1 with the updated file `backup/bphi_AbacusHOD_z1.1_updated.txt` when it exists.

For the recent assembly-bias validation, the z=1.1 assembly-bias prior is loaded from:

```text
bphi_results_abacushod_Bcentm0p040_Bsatm0p170.txt
```

The validated assembly-bias prior is:

```text
b_phi(z=1.1, assembly) = 3.91707660 +/- 1.75616090
```

computed from 90 valid HOD realizations after removing:

```text
[0, 13, 38, 42, 49, 64, 66, 74, 78, 82, 87]
```

To use this prior in a fit, pass:

```bash
--bphi-prior-ab
```

This option is intentionally restricted to `--zsnaps 1.1`; joint fits with the assembly-bias prior are blocked with a warning because assembly-bias priors for LRG1/LRG2 have not been prepared.

## Covariances and Effective Volume

Gaussian covariances are built with `thecov` using box geometry.  By default, fits load the covariance volume stored in the P(k) file metadata or fall back to the cubic box volume.  Passing `--cov-veff` uses the DR1-like `n(z)` effective volume computed from `LRG_NGC_nz.txt` and `LRG_SGC_nz.txt`.

The hard-coded effective-volume setup is:

```text
LRG1: 0.4 < z < 0.6, area = 17000 deg^2
LRG2: 0.6 < z < 0.8, area = 17000 deg^2
LRG3: 0.8 < z < 1.1, area = 14000 deg^2
```

You can also call:

```bash
python effective_volume.py --zmin 0.8 --zmax 1.1 --area-total 14000
```

## Running Fits

The main fit command is:

```bash
python bphi_prior_pipeline.py fit \
    --simtype PNG \
    --tracer LRG \
    --zsnaps 1.1 \
    --mode bfnl \
    --kmin 0.006 \
    --run-sampler \
    --cov-veff \
    --ells 0 2 \
    --cosmo 300 \
    --with-bphi-prior
```

Supported PNG parameterizations are:

```text
bfnl   fit b_phi f_NL directly as bfnl_loc, then marginalize to f_NL
bphi   fit b_phi and f_NL separately
b-p    impose universality, b_phi = 2 delta_c (b1 - p), with p fixed to 1
```

For assembly-bias data vectors with the standard b_phi prior:

```bash
python bphi_prior_pipeline.py fit \
    --simtype PNG --tracer LRG --zsnaps 1.1 --mode bfnl \
    --kmin 0.006 --run-sampler --cov-veff --ells 0 2 \
    --assembly-bias --cosmo 300 --with-bphi-prior
```

For assembly-bias data vectors with the assembly-bias b_phi prior:

```bash
python bphi_prior_pipeline.py fit \
    --simtype PNG --tracer LRG --zsnaps 1.1 --mode bfnl \
    --kmin 0.006 --run-sampler --cov-veff --ells 0 2 \
    --assembly-bias --bphi-prior-ab --cosmo 300 --with-bphi-prior
```

`bphi_prior_assembly.sh` runs the z=1.1 assembly-bias fits for `c000`, `c300`, and `c301`; its third argument controls whether the assembly-bias b_phi prior is used:

```bash
bash bphi_prior_assembly.sh bfnl 1 1   # d_AB + prior_AB
bash bphi_prior_assembly.sh bfnl 1 0   # d_AB + standard prior
```

`bphi_prior_kmax.sh` runs the kmax tests for `c301`, LRG1/LRG2:

```bash
BPHI_KMAX=0.06 bash bphi_prior_kmax.sh
BPHI_KMAX=0.08 bash bphi_prior_kmax.sh
```

Non-default k-ranges are encoded in output filenames, for example:

```text
_k0p006-0p060
```

so kmax scans do not overwrite the default `kmax=0.1` fits.

## Output Naming

Fit products are written under `bphi_prior/fit` with names like:

```text
png_fit_PNG_LRG_z1p100_c300_bfnl_ells0-2_with_bphi_prior_chain.npy
png_fit_PNG_LRG_z1p100_c300_bfnl_ells0-2_Bcentm0p040_Bsatm0p170_with_bphi_prior_chain.npy
png_fit_PNG_LRG_z1p100_c300_bfnl_ells0-2_Bcentm0p040_Bsatm0p170_with_bphi_prior_AB_chain.npy
```

The `_with_bphi_prior_AB` suffix is reserved for assembly-bias-prior fits and prevents overwriting older assembly-bias chains that used the standard b_phi prior.

## Plotting and f_NL Marginalization

`bphi_prior_plot_fnl.py` makes 1D marginalized `f_NL` plots and validation grids.  The default method is:

```text
--fnl-method description
```

which performs the Bayesian post-processing marginalization described in the project notes: samples in `bfnl` space are reweighted/integrated over the b_phi prior to obtain a marginalized `f_NL` posterior.  The older conditional draw method remains available with:

```bash
--fnl-method conditional
```

Summary validation figures:

```bash
bash bphi_prior_plot_fnl.sh
```

Direct comparison of standard vs assembly-bias z=1.1 priors:

```bash
python bphi_prior_plot_fnl.py \
    --mode bfnl \
    --fnl-method description \
    --cosmo 300 \
    --reference-chain bphi_prior/fit/png_fit_PNG_LRG_z1p100_c300_bfnl_ells0-2_Bcentm0p040_Bsatm0p170_with_bphi_prior_chain.npy \
    --reference-label "standard bphi prior" \
    --reference-zsnaps 1.1 \
    --reference-bphi-prior gaussian \
    --reference-bfnl-chain-prior induced \
    --comparison-chain bphi_prior/fit/png_fit_PNG_LRG_z1p100_c300_bfnl_ells0-2_Bcentm0p040_Bsatm0p170_with_bphi_prior_AB_chain.npy \
    --comparison-label "assembly-bias bphi prior" \
    --comparison-zsnaps 1.1 \
    --comparison-bphi-prior assembly \
    --comparison-bfnl-chain-prior induced \
    --output bphi_prior/plots/assembly_bphi_prior_AB_vs_standard_fnl_z1p100_c300.png
```

`bphi_prior_plot_kmax.py` compares marginalized `f_NL` constraints for different `kmax` choices:

```bash
python bphi_prior_plot_kmax.py \
    --cosmos 301 \
    --zsnaps 0.5 0.8 \
    --kmax 0.06 0.08 0.1 \
    --output bphi_prior/plots/kmax_fnl_comparison_c301.pdf
```

`bphi_prior_plot_pk.py` compares P(k) variance estimates from `thecov`, `baocov.pk_var`, and the simple Gaussian-bin expression.

## Recent Validation Numbers

The current LRG3 assembly-bias table used in the paper is:

```text
fNL_true   standard                 d_AB                     d_AB + prior_AB
0          -1.5 +42.2 -46.3         -1.5 +41.0 -44.5         -0.3 +41.6 -45.1
+30        37.2 +63.9 -29.9         38.4 +63.9 -28.7         40.1 +63.9 -30.5
-30        -30.8 +32.8 -63.3        -34.3 +32.2 -65.0        -36.0 +35.7 -64.8
```

For the `c301` kmax tests, using the average 68 percent half-width as the effective uncertainty, the reduced kmax runs lost approximately:

```text
LRG1 kmax=0.06: 25.3 percent inverse-variance constraining power
LRG1 kmax=0.08: 24.7 percent inverse-variance constraining power
LRG2 kmax=0.06: 21.7 percent inverse-variance constraining power
LRG2 kmax=0.08: 26.9 percent inverse-variance constraining power
```

relative to `kmax=0.1`.

## Notes and Pitfalls

- Always use the updated z=1.1 prior file when standard z=1.1 priors are needed; the pipeline does this automatically through `BPHI_STATS_UPDATE_FILES`.
- `--bphi-prior-ab` is z=1.1 only and writes distinct `_with_bphi_prior_AB` outputs.
- Assembly-bias data-vector files and kmax-scan files use suffixes in their filenames; this is deliberate to avoid overwriting existing chains.
- `bfnl` triangle plots include a diagnostic derived `fnl_loc`, but the publication 1D `f_NL` constraints should come from `bphi_prior_plot_fnl.py` with `--fnl-method description`.
- The code prints nonfatal JAX/CUDA and MUNGE warnings on some Perlmutter nodes; plots and fits can still complete successfully.
