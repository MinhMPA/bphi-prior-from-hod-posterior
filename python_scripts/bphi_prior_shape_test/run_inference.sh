#!/bin/bash
# One driver for all per-tracer prior choices and scale cuts; no job submission.
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${HERE}/../environment.sh"
export CUDA_VISIBLE_DEVICES=""
export JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1
set -u
case "${1:-}" in
    compare|prior-plot) exec python -u "${HERE}/inference_plots.py" "$@" ;;
    *) exec python -u "${HERE}/inference.py" "$@" ;;
esac
