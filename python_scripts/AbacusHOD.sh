#!/bin/bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1

# preprocess the Abacus catalogues
# bash preproc_parallel.sh 000
# bash preproc_parallel.sh 004
# implement the Abacus
#srun -N 1 -n 1 -c 128 -C cpu --mem=120G -t 04:00:00 --qos shared_interactive --account desi 
#python AbacusHOD_test.py
python bphi_prior_AbacusHOD.py LRG
