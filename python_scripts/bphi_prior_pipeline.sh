#!/bin/bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
#source activate bfnl # to guarantee the bfnl mode is working properly
export PYTHONNOUSERSITE=1
#export DESILIKE_LOCAL_PATH=/global/homes/j/jiaxi/codes/desilike
#export PYTHONPATH=${DESILIKE_LOCAL_PATH}:$PYTHONPATH
# Guard against old sessions leaking the wrong local tree.
#if [[ "${PYTHONPATH}" == *"/global/homes/j/jiaxi/codes_mine/desilike"* ]]; then
#    export PYTHONPATH="${PYTHONPATH//:\/global\/homes\/j\/jiaxi\/codes_mine\/desilike/}"
#    export PYTHONPATH="${PYTHONPATH//\/global\/homes\/j\/jiaxi\/codes_mine\/desilike:/}"
#fi
#
set -euo pipefail

cd "$(dirname "$0")"

mode=${1:-bfnl}
case "${mode}" in
    bphi|bfnl|b-p) ;;
    *)
        echo "Usage: bash bphi_prior_pipeline.sh [bphi|bfnl|b-p]"
        exit 1
        ;;
esac

cosmos=(000 300 301)
zsnaps=(0.5 0.8 1.1)

# making sure the data vector and covariances are available
if false; then
for cosmo in "${cosmos[@]}"; do
    python bphi_prior_pipeline.py cov --simtype PNG --cosmo ${cosmo} --ells 0 2
done
fi

base_cmd=(
    python bphi_prior_pipeline.py fit
    --simtype PNG
    --tracer LRG
    --zsnaps "${zsnaps[@]}"
    --mode "${mode}"
    --kmin 0.006
    --run-sampler
    --ells 0 2
)
runner=(
    srun -N 1 -C gpu -n 1 
    --gpus 2 --qos shared_interactive 
    -t 04:00:00 --account desi_g
)

for cosmo in "${cosmos[@]}"; do
    #echo "=== c${cosmo}: ${mode}, no bphi prior ==="
    #"${runner[@]}" "${base_cmd[@]}" --cosmo "${cosmo}"

    if [[ "${mode}" == "b-p" ]]; then
        echo "=== c${cosmo}: ${mode} has no free bphi parameter; skipping --with-bphi-prior ==="
        continue
    fi

    echo "=== c${cosmo}: ${mode}, with bphi prior ==="
    "${runner[@]}" "${base_cmd[@]}" --cosmo "${cosmo}" --with-bphi-prior
done
