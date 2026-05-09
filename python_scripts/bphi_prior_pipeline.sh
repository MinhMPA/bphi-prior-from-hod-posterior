#!/bin/bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
source activate bfnl
export PYTHONNOUSERSITE=1
export DESILIKE_LOCAL_PATH=/global/homes/j/jiaxi/codes/desilike
export PYTHONPATH=${DESILIKE_LOCAL_PATH}:$PYTHONPATH
# Guard against old sessions leaking the wrong local tree.
if [[ "${PYTHONPATH}" == *"/global/homes/j/jiaxi/codes_mine/desilike"* ]]; then
    export PYTHONPATH="${PYTHONPATH//:\/global\/homes\/j\/jiaxi\/codes_mine\/desilike/}"
    export PYTHONPATH="${PYTHONPATH//\/global\/homes\/j\/jiaxi\/codes_mine\/desilike:/}"
fi
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

cosmos=(301) #(000 300 301)
zsnaps=(0.5 0.8 1.1)
base_cmd=(
    python bphi_prior_pipeline.py fit
    --simtype PNG
    --tracer LRG
    --zsnaps "${zsnaps[@]}"
    --mode "${mode}"
    --kmin 0.006
    --run-sampler
)
runner=(
    srun -N 1 -C gpu -n 1 
    --gpus 4 --qos interactive --mem=120G 
    -t 02:00:00 --account desi_g
)

for cosmo in "${cosmos[@]}"; do
    echo "=== c${cosmo}: ${mode}, no bphi prior ==="
    #"${runner[@]}" "${base_cmd[@]}" --cosmo "${cosmo}"

    if [[ "${mode}" == "b-p" ]]; then
        echo "=== c${cosmo}: ${mode} has no free bphi parameter; skipping --with-bphi-prior ==="
        continue
    fi

    echo "=== c${cosmo}: ${mode}, with bphi prior ==="
    "${runner[@]}" "${base_cmd[@]}" --cosmo "${cosmo}" --with-bphi-prior
done
