import numpy as np
from astropy.table import Table
import sys
from cosmoprimo import *
from scipy.special import hyp2f1
from cosmoprimo.fiducial import AbacusSummit

task = sys.argv[1]
taskname = task

if (task == 'bphilowS8')|(task == 'bphilowS8dz'):
    phs  = ['000','001','002','003','004','005']
    if task == 'bphilowS8dz':
        taskname = 'bphilowS8'
elif task == 'bphiHOD':
    Nmock  = 24
    ids    = [i for i in range(Nmock)]
Ngals= {}
bias = {}

# linear bias
def D(z):
    cosmo = AbacusSummit()
    return 1/(z+1) * hyp2f1(1./3.,1,11./6.,(1-1./cosmo.Omega0_m)/(z+1)**3)

# read the file
datatot = Table.read(f'bphi_results_{taskname}.txt',format='ascii.no_header')
for tracer in np.unique(datatot['col1']):
    sel= datatot['col1'] == tracer
    data = datatot[sel]
    if (task == 'bphilowS8')|(task == 'bphiHOD'):
        for zsnap in np.unique(data['col4']):
            # print the bphi results
            bphi = data['col7'][(data['col4']==zsnap)]
            p_val= data['col8'][(data['col4']==zsnap)]
            b1   = data['col10'][(data['col4']==zsnap)]/2/1.686+1
            print(f"{task}: {tracer} z={zsnap}, 2*dlog(n)/dlog(sigma8)={np.mean(bphi):.2f}+-{np.std(bphi):.2f}, p value={np.mean(p_val):.2f}+-{np.std(p_val):.2f}, b1={np.mean(b1):.2f}+-{np.std(b1):.2f}") 
    elif (task == 'bphilowS8dz'):
        # (d(logn)/dz)/(d(logsigma8)/dz) between snapshots
        snaps = np.unique(data['col4'])
        for k in range(len(snaps)-1):
            # print the dz results
            dlog_ndz=np.log(data['col5'][len(phs)*k:len(phs)*(k+1)]/data['col5'][len(phs)*(k+1):len(phs)*(k+2)])
            dlog_sigma8dz = np.log(D(data['col4'][len(phs)*k])/D(data['col4'][len(phs)*(k+1)]))
            bphidz = 2*dlog_ndz/dlog_sigma8dz
            print(f"{tracer} dz={data['col4'][len(phs)*(k+1)]}-{data['col4'][len(phs)*k]:.3f}={data['col4'][len(phs)*(k+1)]-data['col4'][len(phs)*k]:.3f}, 2*(dlog(n)/dz)/(dlog(sigma8)/dz)={np.mean(bphidz):.2f}+-{np.std(bphidz):.2f}")
