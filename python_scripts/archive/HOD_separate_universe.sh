#!/bin/bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1

# preprocess the Abacus catalogues for abacusHOD
# sbatch preproc_parallel.sh 000
# sbatch preproc_parallel.sh 004

# implement HOD with AbacusHOD or HODDIES

## validation of the code: AbacusHOD
# python AbacusHOD_test.py
## validation of the code: HODDIES
# python srun -N 1 -n 1 -C gpu --gpus=1 --qos shared_interactive --account desi_g python lowS8_HOD.py ELG

## generate the Latin Hyper cube parameters
# python HOD_LHC.py LRG
# python HOD_LHC.py ELG

## generate the Latin Hyper cube parameters
# sbatch HOD_one_lhc.sh LRG
# sbatch HOD_one_lhc.sh ELG

# compute the bphi
#python HOD_bphi.py LRG

