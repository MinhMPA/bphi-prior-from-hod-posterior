#!/bin/bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

mode=${MODE:-bfnl}
case "${mode}" in
    bphi|bfnl|b-p) ;;
    *)
        echo "Usage: bash bphi_prior_pipeline.sh [bphi|bfnl|b-p] [use_assembly_bias=0|1]"
        exit 1
        ;;
esac

cosmos=(301) 
zsnaps=(0.5 0.8) 
kmin=${BPHI_KMIN:-0.006}
kmax=${BPHI_KMAX:-$1}
echo "Using k range: ${kmin} <= k < ${kmax}"

# do the fittings
if true; then
runner=(
    srun -N 1 -C gpu -n 1 
    --gpus 2 --qos shared_interactive 
    -t 04:00:00 --account desi_g
)

for cosmo in "${cosmos[@]}"; do
    for zsnap in "${zsnaps[@]}"; do
    base_cmd=(
        python bphi_prior_pipeline.py fit
        --simtype PNG
        --tracer LRG
        --zsnaps "${zsnap}"
        --mode "${mode}"
        --kmin "${kmin}"
        --kmax "${kmax}"
        --run-sampler
        --cov-veff
        --ells 0 2
    )
    echo "=== c${cosmo} ${zsnap}: ${mode}, with bphi prior ==="
    "${runner[@]}" "${base_cmd[@]}" --cosmo "${cosmo}" --with-bphi-prior
    done
done
fi
