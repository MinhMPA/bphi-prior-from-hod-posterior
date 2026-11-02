#!/bin/bash
#SBATCH --job-name=bphi_weak
#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=11:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -eo pipefail
HERE=${BPHI_WEAK_CODE:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
mode=${1:-help}
if (( $# )); then shift; fi
ROOT=${BPHI_RESULTS_ROOT:-${HERE}/../outputs}/paper_weak_priors
case "${mode}" in
    status) exec python -u "${HERE}/weak_prior_status.py" "$@" ;;
    queue) exec squeue -u "${USER}" -o '%.18i %.20j %.10T %.10M %.10l %R' ;;
    check|fit|submit|task) ;;
    *)
        echo 'Usage: bash run_paper_weak_priors.sh {check|fit|submit|status|queue} [options]'
        echo 'Defaults: LRG1 LRG2 LRG3 LRG23 joint; c000 c300 c301; kmax=0.09,0.09,0.10; emcee/32.'
        echo 'submit requests 11-hour regular CPU array tasks (max 3 simultaneous); add --dry-run to inspect.'
        echo 'fit runs inside your interactive CPU allocation. status --validation checks the kmax=0.10 pilot.'
        exit 0
        ;;
esac
source "${HERE}/environment.sh"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
set -u
mkdir -p "${ROOT}/logs"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
export BPHI_WEAK_LOG="${ROOT}/logs/${mode}_${SLURM_JOB_ID:-interactive}_${SLURM_ARRAY_TASK_ID:-manual}_${stamp}_$$.log"
# Batch stdout also goes to Slurm logs. Keep an append-only log for manual runs.
if [[ "${mode}" == task ]]; then
    srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-32}" --cpu-bind=cores \
        python -u "${HERE}/paper_weak_priors.py" "${mode}" "$@" 2>&1 | tee --output-error=warn -a "${BPHI_WEAK_LOG}"
else
    python -u "${HERE}/paper_weak_priors.py" "${mode}" "$@" 2>&1 | tee --output-error=warn -a "${BPHI_WEAK_LOG}"
fi
