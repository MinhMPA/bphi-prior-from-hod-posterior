#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH -t 04:00:00
#SBATCH --qos=regular
#SBATCH --account=desi
#SBATCH --array=0-19
#SBATCH -J abacushod_prep2
#SBATCH -o tests/logs/%x_%A_%a.out
#SBATCH -e tests/logs/%x_%A_%a.err

i=$1
dir_source="/pscratch/sd/j/jiaxi/b_phi"

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
module load python
conda activate abacushod

# Map array index to (i, snap)
NSNAPS=4
SNAPS=(0.500 0.800 0.950 1.100)
cosmos=(000 300 301 302 303)

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  calculator="srun -N 1 -n 1 -c 32 -C cpu --mem=200G --account desi "
  i=$(( SLURM_ARRAY_TASK_ID / NSNAPS ))
  snap_index=$(( SLURM_ARRAY_TASK_ID % NSNAPS ))

  snap=${SNAPS[$snap_index]}
  cosmo=${cosmos[$i]}
  SIMNAME="Abacus_pngbase_c${cosmo}_ph000"
  BASECFG="${dir_source}/HOD_config/abacushod_z${snap}_c000.yaml"
  NEWCFG="${dir_source}/HOD_config/tmp/abacushod_png_c${cosmo}_ph000_z${snap}.yaml"
  TARGET="${dir_source}/tests/${SIMNAME}/z${snap}/halos_xcom_16_seed600_abacushod_oldfenv_new.h5"

  if [ -f "$NEWCFGT" ]; then
    echo "YAML File exists — skipping."
  else
    # Generate config
    sed -E "
      s/^([[:space:]]*sim_name:).*/\1 ${SIMNAME}/;
      s/^([[:space:]]*z_mock:).*/\1 ${snap}/
    " "$BASECFG" > "$NEWCFG"
  fi

  if [ -f "$TARGET" ]; then
      echo "Abacus sbsamples File exists — skipping."
      exit 0
  fi
  ${calculator} python -m abacusnbody.hod.prepare_sim --path2config "$NEWCFG"
else
  #for i in {0..4}; do
    calculator="srun -N 1 -n 1 -c 32 -C cpu --mem=200G -t 04:00:00 --qos interactive --account desi "
    cosmo=${cosmos[$i]}
    SIMNAME="Abacus_pngbase_c${cosmo}_ph000"

    for snap in "${SNAPS[@]}"; do
      BASECFG="${dir_source}/HOD_config/abacushod_z${snap}_c000.yaml"
      NEWCFG="${dir_source}/HOD_config/tmp/abacushod_png_c${cosmo}_ph000_z${snap}.yaml"
      TARGET="${dir_source}/tests/${SIMNAME}/z${snap}/halos_xcom_16_seed600_abacushod_oldfenv_new.h5"

      if [ -f "$NEWCFG" ]; then
        echo "YAML File exists — skipping."
      else
        echo "Generating $NEWCFG"
        sed -E "
          s/^([[:space:]]*sim_name:).*/\1 ${SIMNAME}/;
          s/^([[:space:]]*z_mock:).*/\1 ${snap}/
        " "$BASECFG" > "$NEWCFG"
      fi
      if [ -f "$TARGET" ]; then
          echo "${SIMNAME} z${snap} sbsamples exist — skipping."
      else
          echo "${SIMNAME} z${snap} sbsamples are not ready."
          ${calculator} python -m abacusnbody.hod.prepare_sim --path2config "$NEWCFG"
      fi
    done
  #done

  echo "All YAMLs generated. No python executed."
fi