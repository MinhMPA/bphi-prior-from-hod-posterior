import sys
dir_hoddies='/global/homes/j/jiaxi/codes/HODDIES'
sys.path.append(dir_hoddies)
from HODDIES import HOD
import numpy as np
import os, sys

zmin= sys.argv[1]
zmax= sys.argv[2]

# Initiate the HOD instance with the default parameter file default_HOD_parameters.yaml
phs=['000','001','002','003','004','005']
sourcedir=f"/pscratch/sd/j/jiaxi/b_phi/mocks/AbacusSummit_base_c{{}}_ph{{}}_HOD_z{zmin}z{zmax}.fits"

for ph in phs:
    for cosmo in ['000','004']:
        if not os.path.exists(sourcedir.format(cosmo,ph)):
            print(f'creating {sourcedir.format(cosmo,ph)}')
            HOD_obj= HOD(param_file=f'/pscratch/sd/j/jiaxi/b_phi/conf/lowS8_HOD_z{zmin}z{zmax}.yaml', path_to_abacus_sim='/global/cfs/cdirs/desi/cosmosim/Abacus', **{'hcat':{'Abacus':{'sim_name':f'AbacusSummit_base_c{cosmo}_ph{ph}'}}})
            cats = HOD_obj.make_mock_cat(fix_seed=None)  
            cats.write(f"/pscratch/sd/j/jiaxi/b_phi/mocks/{HOD_obj.args['hcat']['sim_name']}_HOD_z{zmin}z{zmax}.fits")
            cats=[]
        else:
            print(f'file exists: {sourcedir.format(cosmo,ph)}')
