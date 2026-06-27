#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=120G
#SBATCH -t 00:30:00
#SBATCH --qos=regular
#SBATCH --account=desi
#SBATCH --array=2-100
#SBATCH -J lhc_abacushod
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

cd $SCRATCH/b_phi
mkdir -p logs
 
# -------- inputs ----------
tracer=${1:-LRG} #LRG
HOD_ID=${SLURM_ARRAY_TASK_ID} #$1 ${SLURM_ARRAY_TASK_ID} # if 0, the data vector of bphi_prior_pipeline.py will be provided
SIMTYPE=${2:-SU} # SU/PNG
USE_ASSEMBLY_BIAS=${3:-0}
COMPUTE_B1=${4:-1}

# confirmed that thw abacusHOD implementation has been corrected
ABIAS_ARGS=()
if [[ "${USE_ASSEMBLY_BIAS}" == "1" || "${USE_ASSEMBLY_BIAS}" == "true" || "${USE_ASSEMBLY_BIAS}" == "yes" ]]; then
    ABIAS_ARGS=(--Bcent -0.04 --Bsat -0.17)
fi

B1_ARGS=()
if [[ "${COMPUTE_B1}" == "1" || "${COMPUTE_B1}" == "true" || "${COMPUTE_B1}" == "yes" ]]; then
    B1_ARGS=(--compute-b1)
fi

# Avoid thread explosions / unstable malloc crashes
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# information: redshift bin
if [ "$tracer" = "LRG" ]; then
    zmins=(0.8) #(0.4 0.6 0.8) # 0.8)
    zmaxs=(1.1) #(0.6 0.8 1.1) # 1.1)
    zsnaps=(1.1) #(0.5 0.8 1.1) # 0.95)
elif [ "$tracer" = "ELG" ]; then
    zmins=(0.8 1.1)
    zmaxs=(1.1 1.6)
    zsnaps=(0.95 1.325)
    # Default: GPU-enabled ELG. To force CPU-JAX fallback for unstable runs:
    # export ELG_FORCE_CPU_JAX=1
fi
# 

calculator="srun -N 1 -n 1 -c 32 -C cpu" # --mem=120G --qos shared_interactive --account desi -t 02:00:00"
for i in "${!zmins[@]}"; do
    zmin=${zmins[$i]}
    zmax=${zmaxs[$i]}
    zsnap=${zsnaps[$i]}

    ${calculator} python HOD_one_lhc.py "${tracer}" "${HOD_ID}" "${zmin}" "${zmax}" "${zsnap}" "${SIMTYPE}" "${ABIAS_ARGS[@]}" "${B1_ARGS[@]}"
done
