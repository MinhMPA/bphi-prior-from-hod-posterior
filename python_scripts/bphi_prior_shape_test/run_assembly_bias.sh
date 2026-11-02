#!/bin/bash
# The same launcher supports manual CPU-node runs and explicit sbatch submission.
#SBATCH --job-name=bphi_assembly
#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=11:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -eo pipefail
# Slurm copies the submitted script, so preserve its original location explicitly.
HERE=${BPHI_ASSEMBLY_CODE:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
mode=${1:-help}
if (( $# )); then shift; fi
case "${mode}" in
    submit-counts|submit-fit)
        mkdir -p "${BPHI_RESULTS_ROOT:-${HERE}/../../outputs}/assembly_bias_test/logs"
        export BPHI_ASSEMBLY_CODE="${HERE}"
        exec sbatch --export=ALL "${HERE}/run_assembly_bias.sh" "${mode#submit-}" "$@"
        ;;
    prepare-responses|counts|summarize|prepare|check|fit|compare|prior-plot) ;;
    *)
        echo 'Usage: bash run_assembly_bias.sh {prepare-responses|counts|summarize|prepare|check|fit|compare|prior-plot|submit-counts|submit-fit} [options]'
        echo 'Default: LRG3 chain_s1 empirical, kmax=0.10; fixed Bcent=-0.04, Bsat=-0.17.'
        echo 'Use --config configs/joint_empirical_k09_09_10.json for fiducial joint LRGs; --cosmos 000 selects one cosmology.'
        exit 0
        ;;
esac
source "${HERE}/../environment.sh"
if [[ -n "${LOCAL_ABACUSUTILS:-}" ]]; then
    export PYTHONPATH="${LOCAL_ABACUSUTILS}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [[ "${mode}" == counts ]]; then
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
fi
set -u
cd "${HERE}"
# Forward Slurm's early warning; counts stop between evaluations, fits keep their
# normal periodic sampler checkpoints until the allocation ends.
trap 'if [[ -n "${worker:-}" ]]; then kill -USR1 "${worker}" 2>/dev/null || true; fi' USR1
if [[ "${mode}" != counts ]]; then trap '' USR1; fi
python -u "${HERE}/assembly_bias.py" "${mode}" "$@" &
worker=$!
set +e
wait "${worker}"
status=$?
if kill -0 "${worker}" 2>/dev/null; then
    wait "${worker}"
    status=$?
fi
exit "${status}"
