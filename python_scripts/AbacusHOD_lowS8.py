import sys
dir_hoddies='/global/homes/j/jiaxi/codes/HODDIES'
sys.path.append(dir_hoddies)
from HODDIES import HOD
import numpy as np
import os, sys

tracer = sys.argv[1]

if tracer == 'LRG':
    zmins = [0.4,0.6,0.8]
    zmaxs = [0.6,0.8,1.1]
    redshifts=[0.5,0.8,1.1]
elif tracer == 'ELG':
    zmins = [0.8,1.1]
    zmaxs = [1.1,1.6]
    redshifts=[1.1,1.325]
elif tracer == 'QSO':
    zmins = [0.8,1.1,1.6,2.1]
    zmaxs = [1.1,1.6,2.1,3.5]
    redshifts=[1.1,1.325,1.850,2.500]
    vsmears = [100,100,300,550]

# the Abacus simulation realisations
phs=['000','001','002','003','004','005']

for iz,(zmin,zmax,redshift) in enumerate(zip(zmins,zmaxs,redshifts)):
    sourcedir=f"/pscratch/sd/j/jiaxi/b_phi/mocks/AbacusSummit_base_c{{}}_ph{{}}_HOD_{tracer}_z{zmin}z{zmax}.fits"
    for ph in phs:
        for cosmo in ['000','004']:
            if not os.path.exists(sourcedir.format(cosmo,ph)):
                print(f'creating {sourcedir.format(cosmo,ph)}')
                if tracer == 'QSO':
                    HOD_obj= HOD(param_file=f'../HOD_config/lowS8_HOD_{tracer}.yaml', path_to_abacus_sim='/global/cfs/cdirs/desi/cosmosim/Abacus', **{'hcat':{'z_simu':redshift,'Abacus':{'sim_name':f'AbacusSummit_base_c{cosmo}_ph{ph}','z_simu':redshift}},'QSO':{'vsmear':vsmears[iz]}})
                else:
                    HOD_obj= HOD(param_file=f'../HOD_config/lowS8_HOD_{tracer}.yaml', path_to_abacus_sim='/global/cfs/cdirs/desi/cosmosim/Abacus', **{'hcat':{'z_simu':redshift,'Abacus':{'sim_name':f'AbacusSummit_base_c{cosmo}_ph{ph}','z_simu':redshift}}})

                cats = HOD_obj.make_mock_cat(fix_seed=None)  
                cats.write(sourcedir.format(cosmo,ph))
                cats=[]
            else:
                print(f'file exists: {sourcedir.format(cosmo,ph)}')
