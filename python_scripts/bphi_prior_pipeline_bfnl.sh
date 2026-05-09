#!/bin/bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

mode="${1:-${MODE:-bphi}}"
case "${mode}" in
    bphi|bfnl) ;;
    *)
        echo "Unsupported mode '${mode}'. Use 'bphi' or 'bfnl'." >&2
        exit 1
        ;;
esac

prior_fnl_loc_args=()
if [[ "${mode}" == "bfnl" && -n "${PRIOR_FNL_LOC:-}" ]]; then
    prior_fnl_loc_args=(--prior-fnl-loc "${PRIOR_FNL_LOC}")
fi

fit_args=(--tracer LRG --zsnaps 0.5 0.8 1.1 --mode "${mode}" --kmin 0.006)

# PNG mocks tests:
cosmos=(000 300 301 302 303)
for cosmo in "${cosmos[@]}"; do
    #python bphi_prior_pipeline.py cov --simtype PNG --tracer LRG --zsnaps 0.5 0.8 1.1 --cosmo ${cosmo}
    #python bphi_prior_plot_pk.py --simtype PNG --tracer LRG --zsnaps 0.5 0.8 1.1 --cosmo ${cosmo}
    srun -N 1 -C gpu -n 1 --gpus 4 --mem=120G -t 02:00:00 --qos interactive --account desi_g python bphi_prior_pipeline_bfnl.py fit --simtype PNG "${fit_args[@]}" --cosmo "${cosmo}" "${prior_fnl_loc_args[@]}" --run-sampler --with-bphi-prior
    srun -N 1 -C gpu -n 1 --gpus 4 --mem=120G -t 02:00:00 --qos interactive --account desi_g python bphi_prior_pipeline_bfnl.py fit --simtype PNG "${fit_args[@]}" --cosmo "${cosmo}" "${prior_fnl_loc_args[@]}" --run-sampler
done
