# Analysis Scripts

The released empirical priors are in
[../data/bphi_prior_samples.csv](../data/bphi_prior_samples.csv). Their
provenance, selection and columns are documented in
[../data/README.md](../data/README.md). The current fiducial fitting driver is
[bphi_prior_shape_test/run_inference.sh](bphi_prior_shape_test/run_inference.sh),
not the older LHC driver in this directory.

## Plotting the Released Samples

From the repository root:

```bash
python python_scripts/plot_bphi_priors.py --output bphi_priors.pdf
```

The histogram and marks show the 100 empirical values; the dashed curve is a
Gaussian matched to their mean and population standard deviation. The output
may instead be `.png`. The script prints the four means, variances and
standard deviations.

The figure-ready mock multipoles, best-fit models and full covariance matrices
are in [../data/paper_pk/](../data/paper_pk/). With NumPy and Matplotlib,
`python python_scripts/plot_released_pk.py --output released_pk.pdf`
recreates the paper comparison directly from these files. See the data README
for units, matrix ordering, source hashes and reference HOD chain provenance.

`export_bphi_samples.py` creates the CSV from the original
`paper_hod_priors/bphi_figure_samples.csv` and the assembly-bias
`bphi_samples.csv` using explicit `--standard` and `--assembly` paths.
It validates counts, response values, fixed assembly-bias parameters and
matching LRG3 HOD draws. Run `python python_scripts/export_bphi_samples.py
--help` for arguments. The original input tables are not bundled.

## Current Informed Fits

The configurable driver uses one common `f_NL` and a separate `b_phi`
prior for each fitted tracer. The fiducial joint configuration is
`configs/joint_empirical_k09_09_10.json`: LRG1 and LRG2 use
`chain_s3` with `kmax=0.09`, and LRG3 uses `chain_s1` with
`kmax=0.10`, in h/Mpc. Empirical priors consist of 100 equally weighted
point masses; a histogram is only a visualization. Gaussian and KDE
configurations use the same selected response samples.

By default, the current fitter reads the standard LRG1/2/3 values from the
released CSV. `BPHI_PRIOR_CSV` can specify a replacement CSV with the same
format. To use other HOD ensembles, set `BPHI_HOD_RUN` to a complete
original HOD run containing its manifests and response tables; this switches
the input source away from the released table. The assembly-bias fitting
adapter also requires the full original HOD run and assembly count products;
the released four-group CSV alone is not sufficient to reproduce those fits.

Configure and run from the repository root:

```bash
export BPHI_DATA_ROOT=/path/to/original/b_phi_inputs
export BPHI_RESULTS_ROOT=/path/to/output
# Optional on NERSC, or activate a compatible environment yourself:
export COSMODESI_ENVIRONMENT=/path/to/cosmodesi_environment.sh
cd python_scripts/bphi_prior_shape_test
bash run_inference.sh check --config configs/joint_empirical_k09_09_10.json --cosmos 000
bash run_inference.sh fit --config configs/joint_empirical_k09_09_10.json --cosmos 000
```

The fit command requires an allocated Slurm CPU node. Input preparation
expects the mock and covariance layout used by the original analysis, including
the standard c000 covariance. Missing files are not downloaded by this
repository. Runs use a frozen input manifest and resume matching checkpoints;
do not run two processes writing the same run simultaneously. Verify sampler
convergence before using posterior summaries.

Dependencies for the full workflow include compatible versions of `desilike`,
`thecov`, `abacusnbody`, `pypower`, `pycorr`, NumPy, SciPy, Matplotlib,
JAX and the sampler packages; HOD calculations additionally require the
simulation catalogs and preprocessed halo subsamples. The exact working
environment is not distributed here. `DESILIKE_SITE_PACKAGES` can point to
a compatible local desilike installation when using the provided environment
helper.

The snapshot-specific YAML files in
`hod_correlation_test/configs/` are templates. Review
their simulation paths, subsample directories and output directories before
running. The original reference HOD chains and figure-ready mock spectra and
covariances are included in `data/`; original power-spectrum objects, halo
catalogues and pre-existing PNG-fit chains are not included. The Slurm
wrappers use NERSC-specific settings; adapt them to the target cluster.

## Earlier LHC and Product-Amplitude Workflow

`HOD_LHC.py`, `HOD_one_lhc.py`, `HOD_bphi.py`,
`bphi_prior_pipeline.py` and `bphi_prior_plot_fnl.py` are retained for the
earlier independent-LHC and product-amplitude analyses. These scripts are not
automatically switched to the released empirical fiducial priors. In
particular, the old assembly-bias LHC prior is a different sample from the
released `LRG3_AB` values. Old plots and results should not be presented as
outputs of the updated empirical workflow without a new calculation.

`bphi_prior_pipeline.sh` is a historical copy of Python source, not a shell
driver; call `python bphi_prior_pipeline.py --help` instead. For old mock
production, set `BPHI_DATA_ROOT` to the input/output tree containing the
required `HOD_config` and mock directories. HODDIES installations can be
selected with `HODDIES_PATH`; the preprocessing launchers require appropriate
simulation and cluster settings.

`bfnl` samples product amplitudes `B_i=b_{phi,i} f_NL`, whereas `bphi`
mode samples a common `f_NL` together with prior variables for `b_phi`.
Converting product-amplitude chains to `f_NL` requires an explicit
`b_phi` prior and the appropriate prior/parameterization treatment; a
derived ratio alone is not a general substitute for Bayesian marginalization.
The weak-prior and Gaussian mode-consistency utilities in
`python_scripts/` are separate checks, and some comparisons need
historical chains that are not bundled.

For halo copying, provide `--source-root` and `--destination` or set
`BPHI_HALO_SOURCE` and `BPHI_HALO_DEST`; inspect the paths before running.
