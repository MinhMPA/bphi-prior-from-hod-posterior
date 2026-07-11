# bphi Prior from HOD Posterior

Build `b_phi` priors from HOD posteriors and use them to constrain local `f_NL`.

Jiaxi Yu and Nhat-Minh Nguyen (2026), [arXiv:2607.01314](https://arxiv.org/abs/2607.01314) (submitted to PRD) —
[github.com/MinhMPA/bphi-prior-from-hod-posterior](https://github.com/MinhMPA/bphi-prior-from-hod-posterior).

Local PNG adds a scale-dependent term `~ f_NL * b_phi` to the galaxy power
spectrum, so the uncertainty on `b_phi` sets the floor on `f_NL` constraints.
This code builds DESI-like HOD mocks, measures `b_phi` with a separate-universe
estimator to form a Gaussian prior, and feeds that prior into a `desilike` PNG
fit for `f_NL`. The method is described in `paper/main.tex`.

## Install

The code runs on the DESI `cosmodesi` stack, which provides `desilike`,
`pypower`, `pycorr`, `cosmoprimo`, and `thecov` (plus `numpy`, `scipy`,
`matplotlib`). On NERSC it is pre-built:

```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
```

Elsewhere, build it with the upstream installer,
[cosmodesi/cosmodesiconda](https://github.com/cosmodesi/cosmodesiconda).

Building the mocks additionally needs the `abacushod` conda environment
(`abacusnbody`/AbacusHOD, and `HODDIES` for ELGs):

```bash
conda activate abacushod
```

## Run

All scripts live in `python_scripts/`. A minimal LRG example (`z=1.1`, validated
against the `f_NL=+30` mock `c300`):

```bash
# 1. Build HOD mocks: AbacusSummit base (c000) + low-sigma8 (c004) pairs, plus PNG boxes.
sbatch preproc_parallel.sh 000          # repeat for 004
sbatch preproc_parallel_PNG.sh 0
python HOD_LHC.py LRG                    # draw the Latin-hypercube HOD samples
sbatch HOD_one_lhc.sh LRG SU            # populate the mocks (records N_gal per cosmology)

# 2. Build the data vector and covariance.
python bphi_prior_pipeline.py pk  --simtype PNG --tracer LRG --zsnaps 1.1 --cosmo 300 --ells 0 2 --kmin 0.006
python bphi_prior_pipeline.py cov --simtype PNG --tracer LRG --zsnaps 1.1 --cosmo 300 --ells 0 2 --kmin 0.006 --cov-veff

# 3. Fit f_NL with the HOD b_phi prior.
python bphi_prior_pipeline.py fit --simtype PNG --tracer LRG --zsnaps 1.1 --cosmo 300 \
    --ells 0 2 --kmin 0.006 --cov-veff --mode bfnl --with-bphi-prior --run-sampler

# 4. Marginalize to f_NL and plot.
python bphi_prior_plot_fnl.py --mode bfnl --fnl-method description --cosmo 300
```

The `b_phi` prior is formed from the mock number counts via the separate-universe
response `b_phi = 2 ln(N_gal^c000 / N_gal^c004) / ln(0.807952/0.75)`, tabulated
under `backup/` and auto-loaded by the fit. `--mode` picks the PNG
parameterization: `bfnl` (default), `bphi`, or `b-p`.

## More

- `python_scripts/README.md`: full script reference, options, and output layout.
- `python bphi_prior_pipeline.py {pk,cov,fit,plot} --help`: all pipeline flags.
- [desilike documentation](https://desilike.readthedocs.io/) for the likelihood,
  templates, priors, and samplers used in the fit.
