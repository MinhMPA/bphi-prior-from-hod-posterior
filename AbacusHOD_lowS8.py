import sys
dir_hoddies='/global/homes/j/jiaxi/codes/HODDIES'
sys.path.append(dir_hoddies)
from HODDIES import HOD
import numpy as np
import os, sys

# Initiate the HOD instance with the default parameter file default_HOD_parameters.yaml
phs=['000','001','002','003','004','005']
sourcedir="/pscratch/sd/j/jiaxi/b_phi/AbacusSummit_base_c{}_ph{}_HOD.fits"

for ph in phs:
    for cosmo in ['000','004']:
        if not os.path.exists(sourcedir.format(cosmo,ph)):
            print(f'creating {sourcedir.format(cosmo,ph)}')
            HOD_obj= HOD(param_file=f'{dir_hoddies}/HODDIES/lowS8_HOD_parameters.yaml', path_to_abacus_sim='/global/cfs/cdirs/desi/cosmosim/Abacus', **{'hcat':{'Abacus':{'sim_name':f'AbacusSummit_base_c{cosmo}_ph{ph}'}}})
            cats = HOD_obj.make_mock_cat(fix_seed=None)  
            cats.write(f"/pscratch/sd/j/jiaxi/b_phi/{HOD_obj.args['hcat']['sim_name']}_HOD.fits")
        else:
            print(f'file exists: {sourcedir.format(cosmo,ph)}')
