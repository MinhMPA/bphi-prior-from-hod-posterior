#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH -t 04:00:00
#SBATCH --qos=regular
#SBATCH --account=desi_g
#SBATCH --array=0-99
#SBATCH -J lhc_abacushod
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
set -euo pipefail

cd $SCRATCH/b_phi
mkdir -p logs
 
# -------- inputs ----------
tracer=$1
#LRG
HOD_ID=${SLURM_ARRAY_TASK_ID}
# $1

# Avoid thread explosions / unstable malloc crashes
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [ "$tracer" = "LRG" ]; then
    zmins=(0.4 0.6 0.8) # 0.8)
    zmaxs=(0.6 0.8 1.1) # 1.1)
    zsnaps=(0.5 0.8 1.1) # 0.95)
elif [ "$tracer" = "ELG" ]; then
    zmins=(0.8 1.1)
    zmaxs=(1.1 1.6)
    zsnaps=(0.95 1.325)
fi

calculator="srun -N 1 -n 1 -C gpu --gpus=1 --qos interactive --account desi_g --mem=120G"
for i in "${!zmins[@]}"; do
    zmin=${zmins[$i]}
    zmax=${zmaxs[$i]}
    zsnap=${zsnaps[$i]}

    ${calculator} python HOD_one_lhc.py "${tracer}" "${HOD_ID}" "${zmin}" "${zmax}" "${zsnap}"
done
