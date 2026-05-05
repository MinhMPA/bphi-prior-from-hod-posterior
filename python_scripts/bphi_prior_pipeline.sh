#!/bin/bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

if false; then
# get the pk measurements, analytical covariance and visually confirm it
python bphi_prior_pipeline.py pk --simtype SU --tracer LRG --zsnaps 0.5 0.8 1.1
python bphi_prior_pipeline.py cov --simtype SU --tracer LRG --zsnaps 0.5 0.8 1.1
python bphi_prior_plot_pk.py --simtype SU --tracer LRG --zsnaps 0.5 0.8 1.1
# fit the PNG parameters with and without bphi priors
srun -N 1 -C cpu -n 1 -c 32 --mem=120G -t 02:00:00 --qos interactive --account desi python bphi_prior_pipeline.py fit --simtype SU --tracer LRG --zsnaps 0.5 0.8 1.1 --mode bphi --run-sampler --kmin 0.006 --with-bphi-prior --kmin 0.006
srun -N 1 -C cpu -n 1 -c 32 --mem=120G -t 02:00:00 --qos interactive --account desi python bphi_prior_pipeline.py fit --simtype SU --tracer LRG --zsnaps 0.5 0.8 1.1 --mode bphi --run-sampler --kmin 0.006
fi

# PNG mocks tests:
cosmos=(000 300 301 302 303)
for cosmo in "${cosmos[@]}"; do
    #python bphi_prior_pipeline.py cov --simtype PNG --tracer LRG --zsnaps 0.5 0.8 1.1 --cosmo ${cosmo}
    #python bphi_prior_plot_pk.py --simtype PNG --tracer LRG --zsnaps 0.5 0.8 1.1 --cosmo ${cosmo}
    srun -N 1 -C gpu -n 1 --gpus 4 --mem=120G -t 02:00:00 --qos interactive --account desi_g python bphi_prior_pipeline.py fit --simtype PNG --tracer LRG --zsnaps 0.5 0.8 1.1 --mode bphi --cosmo ${cosmo} --run-sampler --kmin 0.006 --with-bphi-prior
    srun -N 1 -C gpu -n 1 --gpus 4 --mem=120G -t 02:00:00 --qos interactive --account desi_g python bphi_prior_pipeline.py fit --simtype PNG --tracer LRG --zsnaps 0.5 0.8 1.1 --mode bphi --cosmo ${cosmo} --run-sampler --kmin 0.006
done
