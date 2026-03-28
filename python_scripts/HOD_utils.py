import numpy as np
import yaml
from abacusnbody.hod.GRAND_HOD import *
from abacusnbody.hod.utils import setup_logging

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise RuntimeError(f"Error loading configuration file '{config_path}': {e}") 

def compute_all(Ball, nthread, out=False, verbose = False):
    setup_logging()
    mock_dict = Ball.run_hod(tracers=Ball.tracers, want_rsd=Ball.want_rsd, Nthread = nthread, verbose = verbose, write_to_disk=out)
    return mock_dict

def compute_wpmps(mock, bins, z, tracer='LRG', output='./', fn=None):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    boxsize = 2000
    mu_max  = 1
    nmu     = 200
    pimax   = 40
    # cosmological parameters
    Om = 0.3137721
    Ode = 1-Om
    H = 100*np.sqrt(Om*(1+z)**3+Ode)
    from pycorr import TwoPointCorrelationFunction, project_to_multipoles, project_to_wp

    if tracer == 'LRG':
        data_positions = np.array([mock['LRG']['x']%boxsize,mock['LRG']['y']%boxsize,(mock['LRG']['z'])%boxsize])
    elif tracer == 'ELG':
        data_positions = np.array([mock['x']%boxsize,mock['y']%boxsize,(mock['z']+mock['vz']*(1+z)/H)%boxsize])

    result_smu = TwoPointCorrelationFunction('smu', (bins,np.linspace(-mu_max,mu_max,nmu+1)), data_positions1=data_positions,
                            engine='corrfunc', position_type = 'xyz',D1D2 = None,R1R2 = None, los='z',nthreads=4,boxsize=boxsize)
    _,xi = project_to_multipoles(result_smu,ells=[0,2])
    result_rppi= TwoPointCorrelationFunction('rppi', (bins,np.linspace(-pimax,pimax,pimax*2+1)), data_positions1=data_positions,
                            engine='corrfunc', position_type = 'xyz',D1D2 = None,R1R2 = None, los='z',nthreads=4,boxsize=boxsize)
    _,wp = project_to_wp(result_rppi, pimax=40)
    np.savetxt(f'{output}/{fn}_2pt_RSD.txt',np.array([(bins[1:]+bins[:-1])/2,bins[:-1],bins[1:],xi[0],xi[1],wp]).T,header='smid smin smax xi0 xi2 wp')

    # compute real-space power spectra for galaxy bias
    if tracer == 'LRG':
        fn_pk = f'{output}/{fn}_pk_RSD'
    elif tracer == 'ELG':
        data_positions = np.array([mock['x']%boxsize,mock['y']%boxsize,(mock['z'])%boxsize])
        fn_pk = f'{output}/{fn}_pk_real'
    from pypower import CatalogFFTPower
    power = CatalogFFTPower(data_positions1=data_positions,
                            position_type='xyz',resampler='tsc', interlacing=3, edges={'step': 0.001}, \
                            boxsize=boxsize,cellsize=10, mpiroot=0, ells=(0,2), nmesh=128, los='z')
    power.save(fn_pk)

    return xi[0],xi[1],wp

def HOD_plot(tracer,zmin,zmax,sbins,output='./',ihod=''):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    origin  = f'/global/cfs/cdirs/desi/survey/catalogs/edav1/xi/sv3/{{}}'
    rebin   = 2
    galcomb = 'NScomb'
    weight  = 'default_angular_bitwise_FKP'
    njk     = 128  
    extra   = '_nran18_split20'
    smin    = 0.15
    smax    = 32
    smid  = (sbins[1:]+sbins[:-1])/2
    s_sel =(smin<smid)&(smid<smax)
    clustering = np.loadtxt(f'{output}/{ihod[1:]}_2pt_RSD.txt',usecols=(3,4,5),unpack=True)

    # plot the 2PCF multipoles
    if tracer == 'LRG':   
        obsdir = f'{origin}{tracer}_main_{galcomb}_{zmin}_{zmax}_{weight}_log{rebin}_njack{njk}{extra}.txt'
        fn_pk = f'{output}/{ihod[1:]}_pk_RSD'
    else:
        obsdir = f'{origin}{tracer}_{galcomb}_{zmin}_{zmax}_{weight}_log{rebin}_njack{njk}{extra}.txt'
        fn_pk = f'{output}/{ihod[1:]}_pk_real'

    plt.rc('font', family='serif', size=12) 
    fig = plt.figure(figsize=(12,4))
    spec = gridspec.GridSpec(nrows=1,ncols=3, left = 0.08,right = 0.98,bottom=0.15,top = 0.96, hspace=0.,wspace=0.28)
    ax = np.empty((1,3), dtype=type(plt.axes))

    ss,wp_obs,wp_err = np.loadtxt(obsdir.format("rppi/wp_"),usecols=(0,2,3),unpack=True)
    k_sel  = (smin<ss)&(ss<=smax)

    ax[0,0] = fig.add_subplot(spec[0,0])
    plt.errorbar(ss[k_sel],ss[k_sel]*wp_obs[k_sel],ss[k_sel]*wp_err[k_sel],ls='none',marker='o',color='k',ecolor='k')
    plt.plot(smid[s_sel],smid[s_sel]*clustering[-1][s_sel])

    plt.xscale('log')
    plt.xlim(smin,30)
    if tracer == 'LRG':
        plt.ylim(105,230)
    elif tracer == 'ELG':
        plt.ylim(18,85)
    plt.xlabel('$r_p$ [$\mathrm{{Mpc}}/h$]')
    plt.ylabel(r'$r_p \cdot w_p(r_p)$ [$\mathrm{{Mpc}}/h$]$^2$')

    xi_obs,xi_err = [0,1], [0,1]
    ss,xi_obs[0],xi_obs[1],xi_err[0],xi_err[1] = np.loadtxt(obsdir.format("smu/xipoles_"),usecols=(0,2,3,5,6),unpack=True)
    ss_sel  = (smin<ss)&(ss<=smax)

    ax[0,1] = fig.add_subplot(spec[0,1])
    plt.errorbar(ss[ss_sel],ss[ss_sel]*xi_obs[0][ss_sel],ss[ss_sel]*xi_err[0][ss_sel],ls='none',marker='o',color='k',ecolor='k')
    plt.plot(smid[s_sel],smid[s_sel]*clustering[0][s_sel], label='l=0')
    plt.xlim(0.1,30)
    plt.xscale('log')
    if tracer == 'QSO':plt.ylim(-1,15)
    plt.xlim(smin,30)
    plt.xlabel('$s$ [$\mathrm{{Mpc}}/h$]')
    plt.ylabel(r'$s \cdot \xi_{\ell}(s)$ [$\mathrm{{Mpc}}/h$]')
    plt.legend()

    ax[0,2] = fig.add_subplot(spec[0,2])
    plt.errorbar(ss[ss_sel],ss[ss_sel]*xi_obs[1][ss_sel],ss[ss_sel]*xi_err[1][ss_sel],ls='none',marker='o',color='k',ecolor='k')
    plt.plot(smid[s_sel],smid[s_sel]*clustering[1][s_sel], label='l=2')
    plt.xlim(smin,30)
    if tracer == 'QSO':plt.ylim(-10,40)
    plt.xscale('log')
    plt.xlabel('$s$ [$\mathrm{{Mpc}}/h$]')
    plt.ylabel(r'$s \cdot \xi_{\ell}(s)$ [$\mathrm{{Mpc}}/h$]')
    plt.legend()
    plt.savefig(f'{output}/xiwp_{tracer}_z{zmin}z{zmax}{ihod}.png')
    plt.close()

    from pypower import PowerSpectrumMultipoles
    result = PowerSpectrumMultipoles.load(fn_pk+'.npy')
    result = result.select((0.01,0.2))
    fig,ax=plt.subplots()
    for ill, ell in enumerate(result.ells):
        sep,tpt = result(ell=ell, return_k=True, complex=False)
        ax.plot(sep,sep*tpt, label=r'$\ell = {:d}$'.format(ell))
    ax.legend()
    ax.grid(True)
    ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
    ax.set_ylabel(r'$k*P(k)$')
    if tracer == 'LRG':
        plt.ylim(-200,2000)
    elif tracer == 'ELG':
        plt.ylim(-50,400)

    plt.savefig(f'{output}/pk_{tracer}_z{zmin}z{zmax}{ihod}.png')
    plt.close()