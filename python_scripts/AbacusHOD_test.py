import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import yaml
from abacusnbody.hod.abacus_hod import AbacusHOD
from abacusnbody.hod.GRAND_HOD import *
from abacusnbody.hod.utils import setup_logging

setup_logging()

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise RuntimeError(f"Error loading configuration file '{config_path}': {e}") 

def compute_all(Ball, nthread, out=False, verbose = False):
    mock_dict = Ball.run_hod(tracers=Ball.tracers, want_rsd=Ball.want_rsd, Nthread = nthread, verbose = verbose, write_to_disk=out)
    #clustering = Ball.compute_multipole(mock_dict, rpbins=Ball.rpbins, pimax=Ball.pimax, sbins=Ball.rpbins, nbins_mu=200, Nthread = nthread)    
    return mock_dict#,clustering

def compute_wpmps(mock,bins,z):
    boxsize = 2000
    mu_max  = 1
    nmu     = 200
    pimax   = 40
    # cosmological parameters
    Om = 0.3137721
    Ode = 1-Om
    H = 100*np.sqrt(Om*(1+z)**3+Ode)
    from pycorr import TwoPointCorrelationFunction, project_to_multipoles, project_to_wp

    data_positions = np.array([mock['LRG']['x']%boxsize,mock['LRG']['y']%boxsize,(mock['LRG']['z'])%boxsize])#+mock['LRG']['vz']*(1+z)/H)%boxsize])
    result_smu = TwoPointCorrelationFunction('smu', (bins,np.linspace(-mu_max,mu_max,nmu+1)), data_positions1=data_positions,
                            engine='corrfunc', position_type = 'xyz',D1D2 = None,R1R2 = None, los='z',nthreads=4,boxsize=boxsize)
    _,xi = project_to_multipoles(result_smu,ells=[0,2])
    result_rppi= TwoPointCorrelationFunction('rppi', (bins,np.linspace(-pimax,pimax,pimax*2+1)), data_positions1=data_positions,
                            engine='corrfunc', position_type = 'xyz',D1D2 = None,R1R2 = None, los='z',nthreads=4,boxsize=boxsize)
    _,wp = project_to_wp(result_rppi, pimax=40)
    return xi[0],xi[1],wp

def HOD_plot(tracer,zmin,zmax,sbins,clustering,ihod=''):
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
    # plot the 2PCF multipoles
    if tracer == 'LRG':   
        obsdir = f'{origin}{tracer}_main_{galcomb}_{zmin}_{zmax}_{weight}_log{rebin}_njack{njk}{extra}.txt'
    else:
        obsdir = f'{origin}{tracer}_{galcomb}_{zmin}_{zmax}_{weight}_log{rebin}_njack{njk}{extra}.txt'

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
    plt.savefig(f'xiwp_{tracer}_z{zmin}z{zmax}{ihod}.png')
    plt.close()

if __name__ == "__main__":
    # load the SV3 data
    tracer = 'LRG'
    rebin   = 2
    zmin    = 0.4#[0.4,0.6,0.8]
    zmax    = 0.6#[0.6,0.8,1.1]
    redshift= 0.5#[0.5,0.8,1.1]

    cosmos=['000']#,'004']
    Ngals={}
    dlog_sigma8= np.log(0.807952/0.75)
    dlog_A     = np.log(2.0830e-9/1.7949e-9)

    for cosmo in cosmos:
        # creating HOD mocks
        fn_config  = f'/pscratch/sd/j/jiaxi/b_phi/HOD_config/abacushod_z{redshift:.3f}_c{cosmo}.yaml'
        config     = load_config(fn_config)
        sim_params = config.get("sim_params", {}) #contains simulation info like with box/snapshot to use, where to read subsample file, where to output etc. 
        HOD_params = config.get("HOD_params", {}) #contains your default HOD parameter values. 
        clustering_params = config.get("clustering_params", {}) #contains rpbins, pimax etc.
        nthread    = config.get("nthread", 32)
        smin       = 0.15
        smax       = 32
        # only need to load once
        ball_profiles = AbacusHOD(sim_params, HOD_params, clustering_params) #loading an AbacusHOD object
        # concentration
        # ball_profiles.tracers['LRG']['Asat']=0
        # ball_profiles.tracers['LRG']['Acent']=0
        # environment
        # ball_profiles.tracers['LRG']['Bsat']=0
        # ball_profiles.tracers['LRG']['Bcent']=0    
        ### 
        sbins = np.geomspace(0.01, 100., 49)[::rebin]
        mock=compute_all(ball_profiles, nthread=nthread, out=False, verbose=True) # first run would need extra compile time.
        clustering=compute_wpmps(mock,sbins,redshift)

        Ngals[f'c{cosmo}_{tracer}'] = len(mock[tracer]['x'])

    # bphi values
    dlog_n=np.log(Ngals[f'c{cosmos[0]}_{tracer}']/Ngals[f'c{cosmo}_{tracer}'])
    print(f"Ngalc000:{Ngals[f'c{cosmos[0]}_{tracer}']} Ngalc004:{Ngals[f'c{cosmo}_{tracer}']} bphi_dsigma8:{2*dlog_n/dlog_sigma8} bphi_uni:{2*1.686*(1.93-1)} \n")
    HOD_plot(tracer,zmin,zmax,sbins,clustering)
