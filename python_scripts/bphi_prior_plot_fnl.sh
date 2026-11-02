#!/bin/bash
set -eo pipefail

# Load the same cosmodesi environment used to run the PNG fits.
# set -u is enabled after sourcing because the environment script uses unset variables.
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u

cd "$(dirname "${BASH_SOURCE[0]}")"

# Configure matplotlib/JAX for non-interactive CPU plotting on Perlmutter.
mkdir -p bphi_prior/plots /tmp/mpl-bphi
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl-bphi}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/mpl-bphi}
export BPHI_BFNL_PRIOR_NGRID=${BPHI_BFNL_PRIOR_NGRID:-1024}
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
export JAX_PLATFORM_NAME=${JAX_PLATFORM_NAME:-cpu}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}

# Input/output locations and plotting defaults. These can be overridden
# from the shell without editing this script.
python_cmd=${PYTHON:-python}
script=${BPHI_FNL_PLOT_SCRIPT:-bphi_prior_plot_fnl.py}
fit_dir=${BPHI_FIT_DIR:-bphi_prior/fit}
plot_dir=${BPHI_PLOT_DIR:-bphi_prior/plots}
max_samples=${BPHI_PLOT_MAX_SAMPLES:-12000}
bins=${BPHI_PLOT_BINS:-80}
fnl_method=${BPHI_FNL_METHOD:-description}

common_args=(
    --mode bfnl
    --fnl-method "${fnl_method}"
    --max-samples "${max_samples}"
    --bins "${bins}"
)
if false; then
# Point 1: one-panel LRG1 consistency plot.
# Compare bfnl fits with and without the bphi prior for all PNG cosmologies.
"${python_cmd}" "${script}" "${common_args[@]}" \
    --reference-chain "${fit_dir}/png_fit_PNG_LRG_z0p500_c000_bfnl_ells0-2_with_bphi_prior_chain.npy" \
    --reference-label "c000 with bphi prior" --reference-zsnaps 0.5 --reference-bphi-prior gaussian --reference-cosmo 000 \
    --comparison-chain \
        "${fit_dir}/png_fit_PNG_LRG_z0p500_c000_bfnl_ells0-2_no_bphi_prior_chain.npy" \
        "${fit_dir}/png_fit_PNG_LRG_z0p500_c300_bfnl_ells0-2_with_bphi_prior_chain.npy" \
        "${fit_dir}/png_fit_PNG_LRG_z0p500_c300_bfnl_ells0-2_no_bphi_prior_chain.npy" \
        "${fit_dir}/png_fit_PNG_LRG_z0p500_c301_bfnl_ells0-2_with_bphi_prior_chain.npy" \
        "${fit_dir}/png_fit_PNG_LRG_z0p500_c301_bfnl_ells0-2_no_bphi_prior_chain.npy" \
    --comparison-label "c000 weak prior" "c300 with bphi prior" "c300 weak prior" "c301 with bphi prior" "c301 weak prior" \
    --comparison-zsnaps 0.5 0.5 0.5 0.5 0.5 \
    --comparison-bphi-prior flat gaussian flat gaussian flat \
    --comparison-cosmo 000 300 300 301 301 \
    --text "LRG1: HOD vs weak bphi prior" \
    --output "${plot_dir}/validation_with_without_bphi_prior_z0p500.png"

# Point 1: same consistency check for LRG2.
"${python_cmd}" "${script}" "${common_args[@]}" \
    --reference-chain "${fit_dir}/png_fit_PNG_LRG_z0p800_c000_bfnl_ells0-2_with_bphi_prior_chain.npy" \
    --reference-label "c000 with bphi prior" --reference-zsnaps 0.8 --reference-bphi-prior gaussian --reference-cosmo 000 \
    --comparison-chain \
        "${fit_dir}/png_fit_PNG_LRG_z0p800_c000_bfnl_ells0-2_no_bphi_prior_chain.npy" \
        "${fit_dir}/png_fit_PNG_LRG_z0p800_c300_bfnl_ells0-2_with_bphi_prior_chain.npy" \
        "${fit_dir}/png_fit_PNG_LRG_z0p800_c300_bfnl_ells0-2_no_bphi_prior_chain.npy" \
        "${fit_dir}/png_fit_PNG_LRG_z0p800_c301_bfnl_ells0-2_with_bphi_prior_chain.npy" \
        "${fit_dir}/png_fit_PNG_LRG_z0p800_c301_bfnl_ells0-2_no_bphi_prior_chain.npy" \
    --comparison-label "c000 weak prior" "c300 with bphi prior" "c300 weak prior" "c301 with bphi prior" "c301 weak prior" \
    --comparison-zsnaps 0.8 0.8 0.8 0.8 0.8 \
    --comparison-bphi-prior flat gaussian flat gaussian flat \
    --comparison-cosmo 000 300 300 301 301 \
    --text "LRG2: HOD vs weak bphi prior" \
    --output "${plot_dir}/validation_with_without_bphi_prior_z0p800.png"

# Point 1: same consistency check for LRG3.
"${python_cmd}" "${script}" "${common_args[@]}" \
    --reference-chain "${fit_dir}/png_fit_PNG_LRG_z1p100_c000_bfnl_ells0-2_with_bphi_prior_chain.npy" \
    --reference-label "c000 with bphi prior" --reference-zsnaps 1.1 --reference-bphi-prior gaussian --reference-cosmo 000 \
    --comparison-chain \
        "${fit_dir}/png_fit_PNG_LRG_z1p100_c000_bfnl_ells0-2_no_bphi_prior_chain.npy" \
        "${fit_dir}/png_fit_PNG_LRG_z1p100_c300_bfnl_ells0-2_with_bphi_prior_chain.npy" \
        "${fit_dir}/png_fit_PNG_LRG_z1p100_c300_bfnl_ells0-2_no_bphi_prior_chain.npy" \
        "${fit_dir}/png_fit_PNG_LRG_z1p100_c301_bfnl_ells0-2_with_bphi_prior_chain.npy" \
        "${fit_dir}/png_fit_PNG_LRG_z1p100_c301_bfnl_ells0-2_no_bphi_prior_chain.npy" \
    --comparison-label "c000 weak prior" "c300 with bphi prior" "c300 weak prior" "c301 with bphi prior" "c301 weak prior" \
    --comparison-zsnaps 1.1 1.1 1.1 1.1 1.1 \
    --comparison-bphi-prior flat gaussian flat gaussian flat \
    --comparison-cosmo 000 300 300 301 301 \
    --text "LRG3: HOD vs weak bphi prior" \
    --output "${plot_dir}/validation_with_without_bphi_prior_z1p100.png"

# Point 2: LRG3 model comparison for each cosmology.
# Compare bfnl+bphi-prior, direct bphi mode, universality p=1, and bfnl with no prior.
for cosmo in 000 300 301; do
    "${python_cmd}" "${script}" "${common_args[@]}" \
        --reference-chain "${fit_dir}/png_fit_PNG_LRG_z1p100_c${cosmo}_bfnl_ells0-2_with_bphi_prior_chain.npy" \
        --reference-label "bfnl + bphi prior" --reference-zsnaps 1.1 --reference-bphi-prior gaussian --reference-cosmo "${cosmo}" \
        --comparison-chain \
            "${fit_dir}/png_fit_PNG_LRG_z1p100_c${cosmo}_bphi_ells0-2_with_bphi_prior_chain.npy" \
            "${fit_dir}/png_fit_PNG_LRG_z1p100_c${cosmo}_b-p_ells0-2_no_bphi_prior_chain.npy" \
            "${fit_dir}/png_fit_PNG_LRG_z1p100_c${cosmo}_bfnl_ells0-2_no_bphi_prior_chain.npy" \
        --comparison-label "bphi + bphi prior" "universality p=1" "bfnl weak prior" \
        --comparison-mode bphi b-p bfnl \
        --comparison-zsnaps 1.1 1.1 1.1 \
        --comparison-bphi-prior direct direct flat \
        --comparison-cosmo "${cosmo}" "${cosmo}" "${cosmo}" \
        --text "LRG3 c${cosmo}: PNG-bias prior choices" \
        --output "${plot_dir}/validation_z1p1_methods_c${cosmo}.png"
done

# Point 3: extended footprint comparison.
# Compare LRG3-only against joint LRGs in both bfnl and direct bphi modes.
for cosmo in 000 300 301; do
    "${python_cmd}" "${script}" "${common_args[@]}" \
        --reference-chain "${fit_dir}/png_fit_PNG_LRG_z1p100_c${cosmo}_bfnl_ells0-2_with_bphi_prior_chain.npy" \
        --reference-label "bfnl LRG3" --reference-zsnaps 1.1 --reference-bphi-prior gaussian --reference-cosmo "${cosmo}" \
        --comparison-chain \
            "${fit_dir}/png_fit_PNG_LRG_z0p500-0p800-1p100_c${cosmo}_bfnl_ells0-2_with_bphi_prior_chain.npy" \
            "${fit_dir}/png_fit_PNG_LRG_z1p100_c${cosmo}_bphi_ells0-2_with_bphi_prior_chain.npy" \
            "${fit_dir}/png_fit_PNG_LRG_z0p500-0p800-1p100_c${cosmo}_bphi_ells0-2_with_bphi_prior_chain.npy" \
        --comparison-label "bfnl joint LRGs" "bphi LRG3" "bphi joint LRGs" \
        --comparison-mode bfnl bphi bphi \
        --comparison-zsnaps 0.5,0.8,1.1 1.1 0.5,0.8,1.1 \
        --comparison-bphi-prior gaussian direct direct \
        --comparison-cosmo "${cosmo}" "${cosmo}" "${cosmo}" \
        --text "c${cosmo}: extended LRG footprint" \
        --output "${plot_dir}/validation_extended_footprint_modes_c${cosmo}.png"
done

# Point 4: assembly-bias robustness in fNL.
# Compare standard and assembly-bias data vectors for LRG3-only and joint-LRG fits.
for cosmo in 000 300 301; do
    "${python_cmd}" "${script}" "${common_args[@]}" \
        --reference-chain "${fit_dir}/png_fit_PNG_LRG_z1p100_c${cosmo}_bfnl_ells0-2_with_bphi_prior_chain.npy" \
        --reference-label "standard LRG3" --reference-zsnaps 1.1 --reference-bphi-prior gaussian --reference-cosmo "${cosmo}" \
        --comparison-chain "${fit_dir}/png_fit_PNG_LRG_z1p100_c${cosmo}_bfnl_ells0-2_Bcentm0p040_Bsatm0p170_with_bphi_prior_chain.npy" \
        --comparison-label "assembly-bias LRG3" \
        --comparison-zsnaps 1.1 \
        --comparison-bphi-prior gaussian \
        --comparison-cosmo "${cosmo}" \
        --text "c${cosmo} LRG3: assembly-bias data vector" \
        --output "${plot_dir}/validation_assembly_z1p1_c${cosmo}.png"

    "${python_cmd}" "${script}" "${common_args[@]}" \
        --reference-chain "${fit_dir}/png_fit_PNG_LRG_z0p500-0p800-1p100_c${cosmo}_bfnl_ells0-2_with_bphi_prior_chain.npy" \
        --reference-label "standard joint LRGs" --reference-zsnaps 0.5,0.8,1.1 --reference-bphi-prior gaussian --reference-cosmo "${cosmo}" \
        --comparison-chain "${fit_dir}/png_fit_PNG_LRG_z0p500-0p800-1p100_c${cosmo}_bfnl_ells0-2_Bcentm0p040_Bsatm0p170_with_bphi_prior_chain.npy" \
        --comparison-label "assembly-bias joint LRGs" \
        --comparison-zsnaps 0.5,0.8,1.1 \
        --comparison-bphi-prior gaussian \
        --comparison-cosmo "${cosmo}" \
        --text "c${cosmo} joint LRGs: assembly-bias data vector" \
        --output "${plot_dir}/validation_assembly_allz_c${cosmo}.png"
done

# Point 1 summary figure: 3x3 grid.
# Rows are cosmologies and columns are LRG bins; each panel compares with/no bphi prior.
"${python_cmd}" "${script}" \
    --snapshot-grid \
    --fnl-method "${fnl_method}" \
    --fit-dir "${fit_dir}" \
    --plot-dir "${plot_dir}" \
    --max-samples "${max_samples}" \
    --bins "${bins}" \
    --output "${plot_dir}/validation_snapshot_grid.pdf"

# Point 4 summary figure: assembly-bias grid for LRG3.
# Rows are cosmologies; columns show b1 and fNL posteriors for:
# P(k)+bphi prior, contaminated P(k)+bphi prior, and contaminated P(k)+contaminated bphi prior.
"${python_cmd}" "${script}" \
    --assembly-grid \
    --fnl-method "${fnl_method}" \
    --fit-dir "${fit_dir}" \
    --plot-dir "${plot_dir}" \
    --max-samples "${max_samples}" \
    --bins "${bins}" \
    --output "${plot_dir}/validation_assembly_b1_fnl_grid_z1p100.pdf"

# Point 4 summary figure: same assembly-bias grid for the joint LRG fit.
# The b1 column overlays LRG1/LRG2/LRG3, while the fNL column shows the jointly marginalized result.
"${python_cmd}" "${script}" \
    --assembly-grid \
    --fnl-method "${fnl_method}" \
    --assembly-zsnaps 0.5 0.8 1.1 \
    --fit-dir "${fit_dir}" \
    --plot-dir "${plot_dir}" \
    --max-samples "${max_samples}" \
    --bins "${bins}" \
    --output "${plot_dir}/validation_assembly_b1_fnl_grid_joint_lrgs.pdf"
fi

# Publication-style summary grids.
"${python_cmd}" "${script}" \
    --snapshot-grid \
    --fnl-method "${fnl_method}" \
    --fit-dir "${fit_dir}" \
    --plot-dir "${plot_dir}" \
    --max-samples "${max_samples}" \
    --bins "${bins}" \
    --output "${plot_dir}/validation_snapshot_grid.pdf"

"${python_cmd}" "${script}" \
    --assembly-grid \
    --fnl-method "${fnl_method}" \
    --fit-dir "${fit_dir}" \
    --plot-dir "${plot_dir}" \
    --max-samples "${max_samples}" \
    --bins "${bins}" \
    --output "${plot_dir}/validation_assembly_b1_fnl_grid_z1p100.pdf"

"${python_cmd}" "${script}" \
    --assembly-grid \
    --fnl-method "${fnl_method}" \
    --assembly-zsnaps 0.5 0.8 1.1 \
    --fit-dir "${fit_dir}" \
    --plot-dir "${plot_dir}" \
    --max-samples "${max_samples}" \
    --bins "${bins}" \
    --output "${plot_dir}/validation_assembly_b1_fnl_grid_joint_lrgs.pdf"

# Point 3: extended footprint comparison.
# Compare LRG3-only against joint LRGs in both bfnl and direct bphi modes.
for cosmo in 000 300 301; do
    "${python_cmd}" "${script}" "${common_args[@]}" \
        --reference-chain "${fit_dir}/png_fit_PNG_LRG_z1p100_c${cosmo}_bfnl_ells0-2_with_bphi_prior_chain.npy" \
        --reference-label "bfnl joint" --reference-zsnaps 1.1 --reference-bphi-prior gaussian --reference-cosmo "${cosmo}" \
        --comparison-chain \
            "${fit_dir}/png_fit_PNG_LRG_z1p100_c${cosmo}_bfnl_ells0-2_with_bphi_prior_chain.npy" \
            "${fit_dir}/png_fit_PNG_LRG_z0p800-1p100_c${cosmo}_bfnl_ells0-2_with_bphi_prior_chain.npy" \
            "${fit_dir}/png_fit_PNG_LRG_z0p500-0p800-1p100_c${cosmo}_bfnl_ells0-2_with_bphi_prior_chain.npy" \
        --comparison-label "bfnl LRG3" "bfnl LRG2-LRG3" "bfnl LRG1-LRG2-LRG3" \
        --comparison-mode bfnl bfnl bfnl \
        --comparison-zsnaps 1.1 0.8,1.1 0.5,0.8,1.1 \
        --comparison-bphi-prior gaussian gaussian gaussian \
        --comparison-cosmo "${cosmo}" "${cosmo}" "${cosmo}" \
        --text "c${cosmo}: extended LRG footprint" \
        --output "${plot_dir}/validation_extended_footprint_modes_c${cosmo}.png"
done
