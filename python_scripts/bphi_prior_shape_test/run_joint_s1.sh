#!/bin/bash
# Joint chain_s1 empirical priors versus fiducial s3/s3/s1; no job submission.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mode=${1:-check}
if (( $# )); then shift; fi
config="${HERE}/configs/joint_s1_empirical_k09_09_10.json"
fiducial="${HERE}/configs/joint_empirical_k09_09_10.json"
case "${mode}" in
    prepare|check)
        if [[ -z "${BPHI_HOD_RUN:-}" && -z "${BPHI_PRIOR_CSV:-}" ]]; then
            echo 'Set BPHI_HOD_RUN to a complete HOD run, or BPHI_PRIOR_CSV to a file containing chain_s1 for all three LRGs.' >&2
            exit 1
        fi
        exec bash "${HERE}/run_inference.sh" "${mode}" --config "${config}" "$@" ;;
    prior-plot)
        exec bash "${HERE}/run_inference.sh" "${mode}" --config "${config}" "$@" ;;
    fit)
        exec bash "${HERE}/run_inference.sh" fit --config "${config}" --max-steps 100000 "$@" ;;
    compare)
        exec bash "${HERE}/run_inference.sh" compare --config "${config}" \
            --comparison-config "${fiducial}" "$@" ;;
    *) echo "Usage: bash $0 [prepare|check|fit|prior-plot|compare] [driver options]" >&2; exit 1 ;;
esac
