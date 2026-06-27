#!/bin/bash

task=$1
tracer=$2

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
calculator="srun -N 1 -n 1 -c 32 -C cpu --mem=100G -t 04:00:00 --qos shared_interactive --account desi "
#calculator="srun -N 1 -n 1 -c 32 -C gpu -t 01:00:00 --mem=50G --qos shared_interactive --account desi_g select_gpu_device"
if [ "$task" = "HOD" ]; then
    # tune Abacus mocks
    ${calculator} python lowS8_HOD.py ${tracer}
elif [ "$task" = "mocks" ]; then
    #create Abacus mocks
    ${calculator} python AbacusHOD_lowS8.py ${tracer}
elif [ "$task" = "lhcHOD" ]; then
    # create HOD mocks with parameters in Latin Hyper Cube
    calculator="srun -N 1 -n 1 -c 32 -C cpu --mem=180G -t 04:00:00 --qos shared_interactive --account desi"
    ${calculator} python bphi_prior_HOD.py ${tracer}
elif [ "$task" = "bphilowS8" ] || [ "$task" = "bphilowS8dz" ] || [ "$task" = "bphiHOD" ] ; then
    if [ "$tracer" != "report" ]; then
        # compute the bias and bphi
        calculator="srun -N 1 -n 1 -c 128 -C cpu --mem=70G -t 04:00:00 --qos shared_interactive --account desi "
        ${calculator} python lowS8_comparison.py ${task} ${tracer} 
    else
        # compute mean and error bars
        python bphi_report.py ${task} 
    fi
fi
