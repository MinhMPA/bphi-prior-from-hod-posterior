#!/bin/bash
# Matched joint bfnl check: product +/-9000; reconstruction bphi +/-30, fnl +/-300.
#SBATCH --job-name=bphi_wide
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
HERE=${BPHI_WIDE_CODE:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
source "${HERE}/environment.sh"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
set -u
ROOT=${BPHI_RESULTS_ROOT:-${HERE}/../outputs}/paper_wide_prior
mkdir -p "${ROOT}/logs"
log="${ROOT}/logs/${1:-help}_${SLURM_JOB_ID:-manual}_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
if [[ "${1:-}" == task ]]; then
    srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-32}" --cpu-bind=cores \
        python -u "${HERE}/paper_wide_prior.py" "$@" 2>&1 | tee --output-error=warn -a "${log}"
else
    python -u "${HERE}/paper_wide_prior.py" "$@" 2>&1 | tee --output-error=warn -a "${log}"
fi
