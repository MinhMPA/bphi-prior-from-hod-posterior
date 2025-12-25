#!/bin/bash

task=$1
zmin=$2
zmax=$3

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

if [ "$task" = "HOD" ]; then
    # tune Abacus mocks
    #calculator="srun -N 1 -n 1 -c 32 -C cpu --mem=100G -t 01:00:00 --qos shared_interactive --account desi "
    calculator="srun -N 1 -n 1 -c 32 -C gpu -t 01:00:00 --qos shared_interactive --account desi_g "
    ${calculator} python lowS8_HOD.py ${zmin} ${zmax}
elif [ "$task" = "mocks" ]; then
    #create Abacus mocks
    calculator="srun -N 1 -n 1 -c 32 -C cpu --mem=70G -t 04:00:00 --qos interactive --account desi "
    ${calculator} python AbacusHOD_lowS8.py ${zmin} ${zmax}
elif [ "$task" = "bphi" ]; then
    # compute the bias and bphi
    calculator="srun -N 1 -n 1 -c 32 -C cpu --mem=70G -t 04:00:00 --qos shared_interactive --account desi "
    ${calculator} python lowS8_comparison.py ${zmin} ${zmax} >> bphi_results.txt
fi
