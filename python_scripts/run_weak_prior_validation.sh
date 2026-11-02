#!/bin/bash
# Joint kmax=0.10 bfnl weak-prior regression; defaults to c000, no HOD prior.
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${HERE}/environment.sh"
export CUDA_VISIBLE_DEVICES=""
export JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1
set -u
if (( $# == 0 )); then set -- check; fi
exec python -u "${HERE}/weak_prior_validation.py" "$@"
