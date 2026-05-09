#!/bin/bash

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
set -euo pipefail

cd $SCRATCH/b_phi

tracer=LRG
SIMTYPE=PNG # SU/PNG

# compute the bphi prior
cosmos=(300 301 302 303)
for cosmo in "${cosmos[@]}"; do
    python HOD_bphi.py ${tracer} --mode ${SIMTYPE} --cosmo ${cosmo}
done 
