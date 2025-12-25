import numpy as np
from astropy.table import Table
import glob,os,sys
import yaml
from cosmoprimo import *
from cosmoprimo.fiducial import AbacusSummit
from pypower import CatalogFFTPower, PowerSpectrumMultipoles

zmin= sys.argv[1]#0.4,0.6,0.8,1.1
zmax= sys.argv[2]#0.6,0.8,1.1,1.6

phs=['000','001','002','003','004','005']
mockdir="/pscratch/sd/j/jiaxi/b_phi/mocks/"
pkdir="/pscratch/sd/j/jiaxi/b_phi/pk/"
fn = f"AbacusSummit_base_c{{}}_ph{{}}_HOD_z{zmin}z{zmax}.fits"
Ngals={}
bias={}

param_file=f'/pscratch/sd/j/jiaxi/b_phi/conf/lowS8_HOD_z{zmin}z{zmax}.yaml'
with open(param_file, "r") as f:
    data = yaml.safe_load(f)
z = data["hcat"].get("z_simu")
tracers=data["tracers"]

# linear bias
cosmo = AbacusSummit()
fo = Fourier(cosmo, engine='class')
pk = fo.pk_interpolator()
boxsize=2000

dlog_sigma8 = np.log(0.807952/0.75)
dlog_A = np.log(2.0830e-9/1.7949e-9)

for ph in phs:
    # d(logn)/d(logsigma8) for different cosmologies
    for cosmo in ['000','004']:
        # read mocks
        tmp=Table.read(mockdir+fn.format(cosmo,ph))
        for gal in tracers:
            sel = tmp['TRACER']==gal
            Ngals[f'c{cosmo}_ph{ph}_{gal}'] = np.sum(sel)
            fn_output = pkdir+fn.format(cosmo,ph)[:-5]+f'_{gal}_real.npy'
            if not os.path.exists(fn_output):
                from mpi4py import MPI
                mpicomm = MPI.COMM_WORLD
                inputs = np.array([tmp[sel]['x'],tmp[sel]['y'],tmp[sel]['z']]).T
                result = CatalogFFTPower(data_positions1=inputs%boxsize, 
                                        position_type='pos',resampler='tsc', interlacing=3, edges={'step': 0.005}, \
                                        boxsize=boxsize, cellsize=10, ells=(0, 2, 4), \
                                        mpicomm=mpicomm, mpiroot=None, los='z',
                                        ) #
                result.save(fn_output)

            power = PowerSpectrumMultipoles.load(fn_output)
            sep,tpt = power(return_k=True, complex=False,ell=0)
            tpt = tpt[(0.01<sep)&(sep<0.05)]
            sep = sep[(0.01<sep)&(sep<0.05)]
            bias[f'c{cosmo}_ph{ph}_{gal}'] = np.sqrt(np.mean(tpt/pk(sep, z=z)))

    for gal in tracers:
        ## print Delta n/Delta sigma8
        #print('Delta n/Delta sigma8 calculation')
        dlog_n=np.log(Ngals[f'c000_ph{ph}_{gal}']/Ngals[f'c004_ph{ph}_{gal}'])
    
        print(f"{fn.format(cosmo,ph)[27:]} {gal} z={z}, b_phi~2*Delta log(n)/Delta log(sigma8)={2*dlog_n/dlog_sigma8:.2f}")
        print(f"the Universality relation p=b1-b_phi/2/1.686={bias[f'c000_ph{ph}_{gal}']-dlog_n/dlog_sigma8/1.686:.2f}")
        #print(f"{mockdir.format(cosmo,ph)[27:]} {gal} z={z}, b_phi=4*Delta log(n)/Delta log(As) ={4*dlog_n/dlog_A:.2f}")
        if gal == 'QSO':
            p_val = 1.6
        else:
            p_val = 1
        print(f"the Universality relation is b_phi=2*1.686*(b1-{p_val})={2*1.686*(bias[f'c000_ph{ph}_{gal}']-p_val):.2f}")
        print("============================================== \n")
