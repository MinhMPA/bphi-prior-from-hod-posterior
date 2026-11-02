#!/bin/bash
# Activate a compatible stack before running, or point this variable to its setup script.
if [[ -n "${COSMODESI_ENVIRONMENT:-}" ]]; then
    source "${COSMODESI_ENVIRONMENT}" main
fi
if [[ -n "${DESILIKE_SITE_PACKAGES:-}" ]]; then
    export PYTHONPATH="${DESILIKE_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl-bphi-referee}
