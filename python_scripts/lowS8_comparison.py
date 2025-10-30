import numpy as np
from astropy.table import Table
import glob,os,sys
from scipy.interpolate import CubicSpline
from pypower import CatalogFFTPower, PowerSpectrumMultipoles

phs=['000','001','002','003','004','005']
z  = 1.1 
sourcedir="/pscratch/sd/j/jiaxi/b_phi/AbacusSummit_base_c{}_ph{}_HOD.fits"
Ngals={}
bias={}
# linear bias
Pklin = np.loadtxt(f'{os.environ["HOME"]}/fNL_design/pkdm_unitsim_FNL0.dat')
spl   = CubicSpline(Pklin[:,0],Pklin[:,1])
boxsize=2000

dlog_sigma8 = np.log(0.807952/0.75)
dlog_A = np.log(2.0830e-9/1.7949e-9)

for ph in phs:
    for cosmo in ['000','004']:
        # read mocks
        tmp=Table.read(sourcedir.format(cosmo,ph))
        for gal in ['LRG','ELG']:
            sel = tmp['TRACER']==gal
            Ngals[f'c{cosmo}_ph{ph}_{gal}'] = np.sum(sel)
            fn_output = sourcedir.format(cosmo,ph)[:-5]+f'_{gal}_real.npy'
            if not os.path.exists(fn_output):
                from mpi4py import MPI
                mpicomm = MPI.COMM_WORLD
                inputs = np.array([tmp[sel]['x'],tmp[sel]['y'],tmp[sel]['z']]).T
                result = CatalogFFTPower(data_positions1=inputs%boxsize, 
                                        position_type='pos',resampler='tsc', interlacing=3, edges={'step': 0.005}, \
                                        boxsize=boxsize, cellsize=10, ells=(0, 2, 4), \
                                        mpicomm=mpicomm, mpiroot=None, los='z',nmesh=128,
                                        ) #
                result.save(fn_output)

            power = PowerSpectrumMultipoles.load(fn_output)
            sep,tpt = power(return_k=True, complex=False,ell=0)
            tpt = tpt[(0.01<sep)&(sep<0.05)]
            sep = sep[(0.01<sep)&(sep<0.05)]
            bias[f'c{cosmo}_ph{ph}_{gal}'] = np.sqrt(np.mean(tpt/spl(sep)))

    for gal in ['LRG','ELG']:
        ## print Delta n/Delta sigma8
        print('Delta n/Delta sigma8 calculation')
        dlog_n=np.log(Ngals[f'c000_ph{ph}_{gal}']/Ngals[f'c004_ph{ph}_{gal}'])
    
        print(f"{sourcedir.format(cosmo,ph)[27:]} {gal} z={z}, b_phi~2*Delta log(n)/Delta log(sigma8)={2*dlog_n/dlog_sigma8:.2f}")
        print(f"{sourcedir.format(cosmo,ph)[27:]} {gal} z={z}, b_phi=4*Delta log(n)/Delta log(As) ={4*dlog_n/dlog_A:.2f}")
        print(f"the Universality relation is b_phi=2*1.686*(b1-1)={2*1.686*(bias[f'c000_ph{ph}_{gal}']-1):.2f}")

