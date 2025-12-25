import numpy as np
from astropy.table import Table
import glob,os,sys
import yaml
from cosmoprimo import *
from cosmoprimo.fiducial import AbacusSummit
from pypower import CatalogFFTPower, PowerSpectrumMultipoles

phs=['000','001','002','003','004','005']
mockdir="/pscratch/sd/j/jiaxi/b_phi/mocks/"
pkdir="/pscratch/sd/j/jiaxi/b_phi/pk/"

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

Ngals={}
bias={}

# linear bias
cosmo = AbacusSummit()
fo = Fourier(cosmo, engine='class')
pk = fo.pk_interpolator()
boxsize=2000

# sigma8 difference
dlog_sigma8 = np.log(0.807952/0.75)
dlog_A = np.log(2.0830e-9/1.7949e-9)

f = open('bphi_results.txt','a')
f.write("# tracer(0) zmin(1) zmax(2) zsnap(3) Ngal_c000(4) Ngal_c004(5) bphi=2*Delta log(n)/Delta log(sigma8)(6) p=b1-b_phi/2/1.686(7) b_phi=4*Delta log(n)/Delta log(As)(8) bphi=b_phi=2*1.686*(b1-1)(9) \n")

for iz,(zmin,zmax,redshift) in enumerate(zip(zmins,zmaxs,redshifts)):
    sourcedir=f"{{}}/AbacusSummit_base_c{{}}_ph{{}}_HOD_{tracer}_z{zmin}z{zmax}.fits"
    for ph in phs:
        # d(logn)/d(logsigma8) for different cosmologies
        for cosmo in ['000','004']:
            # read mocks
            tmp = Table.read(sourcedir.format(mockdir,cosmo,ph))
            Ngals[f'c{cosmo}_ph{ph}_{tracer}'] = len(tmp)
            fn_output = sourcedir.format(pkdir,cosmo,ph)[:-5]+'_real.npy'
            if not os.path.exists(fn_output):
                from mpi4py import MPI
                mpicomm = MPI.COMM_WORLD
                inputs = np.array([tmp['x'],tmp['y'],tmp['z']]).T
                result = CatalogFFTPower(data_positions1=inputs%boxsize, 
                                        position_type='pos',resampler='tsc', interlacing=3, edges={'step': 0.005}, \
                                        boxsize=boxsize, cellsize=10, ells=(0, 2, 4), \
                                        mpicomm=mpicomm, mpiroot=None, los='z',
                                        ) #
                result.save(fn_output)

            tmp=[]
            power = PowerSpectrumMultipoles.load(fn_output)
            sep,tpt = power(return_k=True, complex=False,ell=0)
            tpt = tpt[(0.01<sep)&(sep<0.05)]
            sep = sep[(0.01<sep)&(sep<0.05)]
            bias[f'c{cosmo}_ph{ph}_{tracer}'] = np.sqrt(np.mean(tpt/pk(sep, z=redshift)))

        ## print Delta n/Delta sigma8
        dlog_n=np.log(Ngals[f'c000_ph{ph}_{tracer}']/Ngals[f'c004_ph{ph}_{tracer}'])
        f.write(f"{tracer} {zmin} {zmax} {redshift} {Ngals[f'c000_ph{ph}_{tracer}']} {Ngals[f'c004_ph{ph}_{tracer}']} {2*dlog_n/dlog_sigma8} {bias[f'c000_ph{ph}_{tracer}']-dlog_n/dlog_sigma8/1.686} {4*dlog_n/dlog_A:.2f} {2*1.686*(bias[f'c000_ph{ph}_{tracer}']-1)} \n")
f.close()
