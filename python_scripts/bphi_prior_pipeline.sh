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

USE_ASSEMBLY_BIAS=${2:-0}
ABIAS_ARGS=()
if [[ "${USE_ASSEMBLY_BIAS}" == "1" || "${USE_ASSEMBLY_BIAS}" == "true" || "${USE_ASSEMBLY_BIAS}" == "yes" ]]; then
    ABIAS_ARGS=(--assembly-bias)
fi

cosmos=(000 300 301)
zsnaps=(0.5 0.8 1.1)

# making sure the data vector and covariances are available
if false; then
python bphi_prior_pipeline.py cov --simtype PNG --ells 0 2 --cov-veff
python bphi_prior_plot_pk.py --simtype PNG --ells 0 2 --cov-veff
fi

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
        --kmin 0.006
        --run-sampler
        --cov-veff
        --ells 0 2
        "${ABIAS_ARGS[@]}"
    )
    echo "=== c${cosmo} ${zsnap}: ${mode}, with bphi prior ==="
    "${runner[@]}" "${base_cmd[@]}" --cosmo "${cosmo}" --with-bphi-prior
    done
done
fi
