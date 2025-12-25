import numpy as np
from astropy.table import Table
import glob,os,sys
import yaml
from cosmoprimo import *
from scipy.special import hyp2f1
from cosmoprimo.fiducial import AbacusSummit

phs=['000','001','002','003','004','005']
mockdir="/pscratch/sd/j/jiaxi/b_phi/mocks/"
pkdir="/pscratch/sd/j/jiaxi/b_phi/pk/"
Ngals={}
bias={}

# linear bias
def D(z):
    cosmo = AbacusSummit()
    return 1/(z+1) * hyp2f1(1./3.,1,11./6.,(1-1./cosmo.Omega0_m)/(z+1)**3)

cosmo  = '000'
for ph in phs:
    # (d(logn)/dz)/(d(logsigma8)/dz) between snapshots
    for tracer in ['LRG','ELG','QSO']:
        if tracer == 'LRG':
            zmins = [0.4,0.6,0.8]
            zmaxs = [0.6,0.8,1.1]
        elif tracer == 'ELG':
            zmins = [0.8,1.1]
            zmaxs = [1.1,1.6]
        elif tracer == 'QSO':
            zmins = [0.8,1.1,1.6,2.1]
            zmaxs = [1.1,1.6,2.1,3.5]

        for k in range(len(zmins)-1):
            zmin1,zmax1 = zmins[k],  zmaxs[k]
            zmin2,zmax2 = zmins[k+1],zmaxs[k+1]

            # load redshifts
            param_file=f'/pscratch/sd/j/jiaxi/b_phi/conf/lowS8_HOD_z{zmin1}z{zmax1}.yaml'
            with open(param_file, "r") as f:
                conf = yaml.safe_load(f)
            z1 = conf["hcat"].get("z_simu")
            fn = f"AbacusSummit_base_c{{}}_ph{{}}_HOD_z{zmin1}z{zmax1}.fits"
            tmp=Table.read(mockdir+fn.format(cosmo,ph))
            sel = tmp['TRACER']==tracer
            Ngals[f'z1_ph{ph}_{tracer}'] = np.sum(sel)

            param_file=f'/pscratch/sd/j/jiaxi/b_phi/conf/lowS8_HOD_z{zmin2}z{zmax2}.yaml'
            with open(param_file, "r") as f:
                conf = yaml.safe_load(f)
            z2 = conf["hcat"].get("z_simu")
            fn = f"AbacusSummit_base_c{{}}_ph{{}}_HOD_z{zmin2}z{zmax2}.fits"
            tmp=Table.read(mockdir+fn.format(cosmo,ph))
            sel = tmp['TRACER']==tracer
            Ngals[f'z2_ph{ph}_{tracer}'] = np.sum(sel)

            # print the dz results
            dlog_ndz=np.log(Ngals[f'z1_ph{ph}_{tracer}']/Ngals[f'z2_ph{ph}_{tracer}'])
            dlog_sigma8dz = np.log(D(z1)/D(z2))
            print(f"{fn.format(cosmo,ph)[27:]} {tracer} dz={z2-z1:.3f} for {zmin1}-{zmax2}, 2*(dlog(n)/dz)/(dlog(sigma8)/dz)={2*dlog_ndz/dlog_sigma8dz:.2f}")
