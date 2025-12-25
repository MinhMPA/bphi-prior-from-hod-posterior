import sys
sys.path.append('/global/homes/j/jiaxi/codes/HODDIES')
from HODDIES import HOD
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pycorr import TwoPointCorrelationFunction
from pycorr import project_to_multipoles, project_to_wp

# Initiate the HOD instance with default parameter file default_HOD_parameters.yaml
tracer = sys.argv[1]

# load the SV3 data
origin = f'/global/cfs/cdirs/desi/survey/catalogs/edav1/xi/sv3/{{}}'
rebin=2
extra = '_nran18_split20'
weight  = 'default_angular_bitwise_FKP'
njk     = 128  
galcomb = '_NScomb'

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

plt.rc('font', family='serif', size=12) 
for iz,(zmin,zmax,redshift) in enumerate(zip(zmins,zmaxs,redshifts)):
    if tracer == 'QSO':
        HOD_obj= HOD(param_file=f'/pscratch/sd/j/jiaxi/b_phi/conf/lowS8_HOD_{tracer}.yaml', path_to_abacus_sim='/global/cfs/cdirs/desi/cosmosim/Abacus', **{'hcat':{'z_simu':redshift,'Abacus':{'sim_name':'AbacusSummit_base_c000_ph000','z_simu':redshift}},'QSO':{'vsmear':vsmears[iz]}})
    else:
        HOD_obj= HOD(param_file=f'/pscratch/sd/j/jiaxi/b_phi/conf/lowS8_HOD_{tracer}.yaml', path_to_abacus_sim='/global/cfs/cdirs/desi/cosmosim/Abacus', **{'hcat':{'z_simu':redshift,'Abacus':{'sim_name':'AbacusSummit_base_c000_ph000','z_simu':redshift}}})

    # Create galaxy mock catalog, First evaluation is slower because it needs to compile jit functions
    cats = HOD_obj.make_mock_cat(fix_seed=None) #return 1 catalogue per tracer 

    # plot the 2PCF multipoles   
    fig = plt.figure(figsize=(12,4))
    spec = gridspec.GridSpec(nrows=1,ncols=3, left = 0.08,right = 0.98,bottom=0.15,top = 0.96, hspace=0.,wspace=0.28)
    ax = np.empty((1,3), dtype=type(plt.axes))

    rp, wp = HOD_obj.get_wp(cats, tracers=tracer)
    s, xi = HOD_obj.get_2PCF(cats, tracers=tracer)

    rpmin = 0.1
    obsdir = f'{origin.format("rppi/allcounts_")}{{}}_NScomb_{zmin}_{zmax}_{weight}_log_njack{njk}{extra}.npy'
    result = TwoPointCorrelationFunction.load(obsdir.format(tracer))
    result = result[::rebin,::]
    result.select((rpmin,30))
    ss, wp_obs, cov = project_to_wp(result)
    # chi2 computation is impossible as the binning are different
    #wp = wp[(rpmin<=rp)&(rp<30)]
    #chi2_wp = (wp_obs-wp) @ np.linalg.inv(cov) @ (wp_obs-wp)

    ax[0,0] = fig.add_subplot(spec[0,0])
    plt.errorbar(ss,ss*wp_obs,ss*np.sqrt(np.diag(cov)),ls='none',marker='o',color='k',ecolor='k')
    plt.plot(rp,rp*wp)

    plt.xscale('log')
    plt.xlim(rpmin,30)
    if tracer == 'LRG':
        plt.ylim(105,230)
    elif tracer == 'ELG':
        plt.ylim(18,85)
    plt.xlabel('$r_p$ [$\mathrm{{Mpc}}/h$]')
    plt.ylabel(r'$r_p \cdot w_p(r_p)$ [$\mathrm{{Mpc}}/h$]$^2$')

    if tracer == 'QSO':
        smin = 0.5
    else:
        smin = 0.1
    obsdir = f'{origin.format("smu/allcounts_")}{{}}_NScomb_{zmin}_{zmax}_{weight}_log_njack{njk}{extra}.npy'
    result = TwoPointCorrelationFunction.load(obsdir.format(tracer))
    result = result[::rebin,::]
    result.select((smin,30))
    ss,xi_obs, cov = project_to_multipoles(result,ells=[0,2],return_sep=True,return_cov=True)
    #
    #xi = [xi[k][(smin<=s)&(s<30)] for k in range(2)]
    #chi2_mps = (np.concatenate(xi_obs)-np.concatenate(xi)) @ np.linalg.inv(cov) @ (np.concatenate(xi_obs)-np.concatenate(xi))

    ax[0,1] = fig.add_subplot(spec[0,1])
    plt.errorbar(ss,ss*xi_obs[0],ss*np.sqrt(np.diag(cov))[:len(ss)],ls='none',marker='o',color='k',ecolor='k')
    plt.plot(s,s*xi[0], label='l=0')
    plt.xlim(0.1,30)
    plt.xscale('log')
    if tracer == 'QSO':plt.ylim(-1,15)
    plt.xlim(smin,30)
    plt.xlabel('$s$ [$\mathrm{{Mpc}}/h$]')
    plt.ylabel(r'$s \cdot \xi_{\ell}(s)$ [$\mathrm{{Mpc}}/h$]')
    plt.legend()

    ax[0,2] = fig.add_subplot(spec[0,2])
    plt.errorbar(ss,ss*xi_obs[1],ss*np.sqrt(np.diag(cov))[len(ss):],ls='none',marker='o',color='k',ecolor='k')
    plt.plot(s,s*xi[1], label='l=2')
    plt.xlim(smin,30)
    if tracer == 'QSO':plt.ylim(-10,40)
    plt.xscale('log')
    plt.xlabel('$s$ [$\mathrm{{Mpc}}/h$]')
    plt.ylabel(r'$s \cdot \xi_{\ell}(s)$ [$\mathrm{{Mpc}}/h$]')
    plt.legend()
    plt.savefig(f'xiwp_{tracer}_z{zmin}z{zmax}.png')
    plt.close()

    #print(f'fitting chi2 {chi2_wp+chi2_mps:.1f}')

