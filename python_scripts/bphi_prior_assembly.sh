#!/bin/bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

mode=${1:-bfnl}
case "${mode}" in
    bphi|bfnl|b-p) ;;
    *)
        echo "Usage: bash bphi_prior_pipeline.sh [bphi|bfnl|b-p] [use_assembly_bias=0|1]"
        exit 1
        ;;
esac

USE_ASSEMBLY_BIAS=${2:-1}
USE_ASSEMBLY_BIAS_PRIOR=${3:-1}
ABIAS_ARGS=()
BPHI_PRIOR_ARGS=()
if [[ "${USE_ASSEMBLY_BIAS}" == "1" || "${USE_ASSEMBLY_BIAS}" == "true" || "${USE_ASSEMBLY_BIAS}" == "yes" ]]; then
    ABIAS_ARGS=(--assembly-bias)
fi
if [[ "${USE_ASSEMBLY_BIAS_PRIOR}" == "1" || "${USE_ASSEMBLY_BIAS_PRIOR}" == "true" || "${USE_ASSEMBLY_BIAS_PRIOR}" == "yes" ]]; then
    BPHI_PRIOR_ARGS=(--bphi-prior-ab)
fi
cosmos=(000 300 301) 
zsnaps=(1.1) #(0.5 0.8 1.1)

# do the fittings
if true; then
base_cmd=(
    python bphi_prior_pipeline.py fit
    --simtype PNG
    --tracer LRG
    --zsnaps "${zsnaps[@]}"
    --mode "${mode}"
    --kmin 0.006
    --run-sampler
    --cov-veff
    --ells 0 2
    "${ABIAS_ARGS[@]}"
    "${BPHI_PRIOR_ARGS[@]}"
)
runner=(
    srun -N 1 -C gpu -n 1 
    --gpus 2 --qos shared_interactive 
    -t 04:00:00 --account desi_g
)

for cosmo in "${cosmos[@]}"; do
    echo "=== c${cosmo} ${zsnaps[@]}: ${mode}, with bphi prior, ${ABIAS_ARGS[*]} ${BPHI_PRIOR_ARGS[*]} ==="
    "${runner[@]}" "${base_cmd[@]}" --cosmo "${cosmo}" --with-bphi-prior
done
fi
