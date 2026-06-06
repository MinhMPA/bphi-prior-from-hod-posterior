import numpy as np
import yaml
import os
from functools import lru_cache
from pathlib import Path

try:
    from abacusnbody.hod.GRAND_HOD import *
    from abacusnbody.hod.utils import setup_logging
except ImportError:
    def setup_logging(*args, **kwargs):
        return None

DEFAULT_NFW_DRAW_PATH = Path(__file__).resolve().parent / "archive/chains_ELG/data/nfw.npy"


@lru_cache(maxsize=4)
def load_nfw_draw(nfw_draw_path=None):
    path = Path(
        nfw_draw_path
        or os.environ.get("ABACUSHOD_NFW_DRAW", DEFAULT_NFW_DRAW_PATH)
    ).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing NFW draw file: {path}. Set ABACUSHOD_NFW_DRAW to a valid nfw.npy."
        )
    return np.load(path)


def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise RuntimeError(f"Error loading configuration file '{config_path}': {e}") 

def compute_all(Ball, nthread, out=False, verbose=False, want_rsd=None,
                want_nfw=None, NFW_draw=None, nfw_draw_path=None):
    setup_logging()
    if want_rsd is None:
        want_rsd = Ball.want_rsd
    if want_nfw is None:
        want_nfw = getattr(Ball, "z_type", None) == "secondary"
    if want_nfw and NFW_draw is None:
        NFW_draw = load_nfw_draw(nfw_draw_path)
    mock_dict = Ball.run_hod(
        tracers=Ball.tracers,
        want_rsd=want_rsd,
        want_nfw=want_nfw,
        NFW_draw=NFW_draw,
        Nthread=nthread,
        verbose=verbose,
        write_to_disk=out,
    )
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

    # Existing diagnostic P(k) output used by the plotting helper below.
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

def mock_real_space_positions(mock, tracer='LRG', boxsize=2000):
    if isinstance(mock, dict) and tracer in mock and isinstance(mock[tracer], dict):
        catalog = mock[tracer]
    else:
        catalog = mock
    return np.array([
        np.asarray(catalog['x']) % boxsize,
        np.asarray(catalog['y']) % boxsize,
        np.asarray(catalog['z']) % boxsize,
    ])

def compute_real_space_pk(mock, tracer='LRG', output='./', fn=None, boxsize=2000,
                          edges_step=0.005, cellsize=10, nmesh=128):
    import os
    from pypower import CatalogFFTPower
    os.makedirs(output, exist_ok=True)
    if fn is None:
        fn = tracer
    fn_pk = f'{output}/{fn}_pk_real'
    data_positions = mock_real_space_positions(mock, tracer=tracer, boxsize=boxsize)
    power = CatalogFFTPower(data_positions1=data_positions,
                            position_type='xyz', resampler='tsc', interlacing=3,
                            edges={'step': edges_step}, boxsize=boxsize,
                            cellsize=cellsize, mpiroot=0, ells=(0,), nmesh=nmesh, los='z')
    power.save(fn_pk)
    return fn_pk

def estimate_linear_bias_from_pk(fn_pk, z, kmin=0.01, kmax=0.05):
    from cosmoprimo import Fourier
    from cosmoprimo.fiducial import AbacusSummit
    from pypower import PowerSpectrumMultipoles

    path = str(fn_pk)
    if not path.endswith('.npy'):
        path = path + '.npy'
    power = PowerSpectrumMultipoles.load(path)
    kvals, pk_gal = power(return_k=True, complex=False, ell=0)
    mask = (kmin < kvals) & (kvals < kmax)
    if not np.any(mask):
        raise ValueError(f'No P(k) bins in b1 range {kmin} < k < {kmax}.')

    pk_interp = Fourier(AbacusSummit(), engine='class').pk_interpolator()
    pk_matter = pk_interp(kvals[mask], z=z)
    ratio = np.asarray(pk_gal[mask]) / np.asarray(pk_matter)
    ratio = ratio[np.isfinite(ratio) & (ratio > 0.)]
    if ratio.size == 0:
        raise ValueError(f'No finite positive Pgal/Pmatter ratios in {kmin} < k < {kmax}.')
    return float(np.sqrt(np.mean(ratio)))

def compute_linear_bias(mock, z, tracer='LRG', output='./', fn=None,
                        kmin=0.01, kmax=0.05, boxsize=2000,
                        edges_step=0.005, cellsize=10, nmesh=128):
    fn_pk = compute_real_space_pk(mock, tracer=tracer, output=output, fn=fn,
                                  boxsize=boxsize, edges_step=edges_step,
                                  cellsize=cellsize, nmesh=nmesh)
    b1 = estimate_linear_bias_from_pk(fn_pk, z=z, kmin=kmin, kmax=kmax)
    with open(f'{output}/{fn}_b1.txt', 'w') as stream:
        stream.write('# tracer zsnap kmin kmax b1 pk_file\n')
        stream.write(f'{tracer} {z:.3f} {kmin:.6g} {kmax:.6g} {b1:.8f} {fn_pk}.npy\n')
    return b1

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


def _contour_scalar_float(value):
    if value is None:
        return None
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.size != 1:
        return None
    return float(array.reshape(-1)[0])


def _contour_find_param_key(params, basename):
    for param in params:
        name = str(param)
        if name == basename or getattr(param, "basename", None) == basename:
            return name
    return None


def triangle_params_with_bfnl_loc(chain, params, mode):
    if mode == "bfnl":
        fnl_params = chain.params(basename="fnl_loc")
        return params + fnl_params if fnl_params else params
    if mode not in ("bphi", "b-p"):
        return params
    bfnl_params = chain.params(basename="bfnl_loc")
    if not bfnl_params:
        return params
    return params + bfnl_params


def overlay_truth_on_triangle(g, params, basename, value, color="0.25", linestyle="-.", label=None):
    key = _contour_find_param_key(params, basename)
    if key is None or value is None:
        return
    axes = g.subplots
    legend_done = False
    for i, param_i in enumerate(params):
        if str(param_i) == key:
            this_label = label if not legend_done else None
            axes[i, i].axvline(value, color=color, linestyle=linestyle, linewidth=1.0, alpha=0.9, label=this_label)
            legend_done = True
        for j in range(i):
            ax = axes[i, j]
            if str(params[j]) == key:
                this_label = label if not legend_done else None
                ax.axvline(value, color=color, linestyle=linestyle, linewidth=1.0, alpha=0.9, label=this_label)
                legend_done = True
            if str(param_i) == key:
                this_label = label if not legend_done else None
                ax.axhline(value, color=color, linestyle=linestyle, linewidth=1.0, alpha=0.9, label=this_label)
                legend_done = True


def overlay_truth_values_on_triangle(g, params, values, color="black", linestyle="-", linewidth=1.1):
    if not values:
        return
    axes = g.subplots
    values = {str(name): _contour_scalar_float(value) for name, value in values.items()}
    values = {name: value for name, value in values.items() if value is not None}
    for i, param_i in enumerate(params):
        name_i = str(param_i)
        value_i = values.get(name_i)
        if value_i is not None:
            axes[i, i].axvline(value_i, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.95)
        for j in range(i):
            ax = axes[i, j]
            name_j = str(params[j])
            value_j = values.get(name_j)
            if value_j is not None:
                ax.axvline(value_j, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.95)
            if value_i is not None:
                ax.axhline(value_i, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.95)


def annotate_joint_triangle_map(g, params, entries, true_fnl_loc):
    axes = np.asarray(g.subplots)
    if axes.size == 0:
        return
    fig = axes.flat[0].figure
    fig.text(
        0.02,
        0.02,
        "Star: maximum likelihood. Circle: MAP. Black lines: true fNL and best-fit b1.",
        ha="left",
        va="bottom",
        fontsize=9,
    )


def overlay_markers_on_triangle(g, params, point, color, marker, label=None, diagonal_ls="--", size=48):
    if point is None:
        return
    axes = g.subplots
    legend_done = False
    for i, param_i in enumerate(params):
        value_i = _contour_scalar_float(point.get(str(param_i)))
        if value_i is None:
            continue
        ax_diag = axes[i, i]
        ax_diag.axvline(value_i, color=color, linestyle=diagonal_ls, linewidth=1.0, alpha=0.9)
        this_label = label if not legend_done else None
        ax_diag.scatter(
            [value_i],
            [0.92],
            transform=ax_diag.get_xaxis_transform(),
            color=color,
            marker=marker,
            s=size,
            zorder=10,
            label=this_label,
            edgecolors="black" if marker not in [".", ",", "|", "_"] else None,
            linewidths=0.4,
        )
        legend_done = True
        for j in range(i):
            value_j = _contour_scalar_float(point.get(str(params[j])))
            if value_j is None:
                continue
            ax = axes[i, j]
            this_label = label if not legend_done else None
            ax.scatter(
                value_j,
                value_i,
                color=color,
                marker=marker,
                s=size,
                zorder=10,
                label=this_label,
                edgecolors="black" if marker not in [".", ",", "|", "_"] else None,
                linewidths=0.4,
            )
            legend_done = True


def _product_chain(product, prefer="chain"):
    if hasattr(product, prefer):
        return getattr(product, prefer)
    other = "rchain" if prefer == "chain" else "chain"
    if hasattr(product, other):
        return getattr(product, other)
    return product


def _std_from_product(product, param):
    if hasattr(product, "std"):
        return float(product.std(param))
    chain = _product_chain(product)
    if hasattr(chain, "std"):
        return float(chain.std(param))
    return float(np.std(np.asarray(chain[param])))


def report_improvement(products, params=None, items=None, item_std="LL", print_result=True):
    if params is None:
        params = ["logA", "sigma8_cb", "omega_cdm", "Omega_m", "H0", "elg.b1"]
    if items is None:
        items = [item for item in products if item != item_std] if isinstance(products, dict) else []
    if not isinstance(products, dict):
        raise ValueError("report_improvement expects a dictionary of products/chains keyed by item name.")

    summary = {}
    for item in items:
        summary[item] = {}
        if print_result:
            print(f"{item:20}: ", end="")
        for param in params:
            fiducial = _std_from_product(products[item_std], param)
            value = 1.0 - _std_from_product(products[item], param) / fiducial
            summary[item][param] = value
            if print_result:
                print(f"{param}({value:.2%}) ", end="")
        if print_result:
            print()
    return summary


def plot_contours(
    chains,
    params=None,
    items=None,
    extra_info=None,
    item_std="LL",
    texts="data: complete abacus mocks; cov: holi-v3-altmtl",
    fn=None,
    show=True,
    filled=None,
    markers=None,
    truth=None,
    **kwargs,
):
    if params is None:
        params = ["logA", "sigma8_cb", "omega_cdm", "Omega_m", "H0", "elg.b1"]

    if isinstance(chains, dict):
        labels = [item_std] if item_std in chains else []
        labels += [item for item in (items or chains) if item != item_std]
        plot_chains = [
            _product_chain(chains[label], prefer="chain" if label == item_std else "rchain")
            for label in labels
        ]
    else:
        plot_chains = list(chains)
        labels = list(items) if items is not None else [f"chain {i}" for i in range(len(plot_chains))]
        if len(labels) != len(plot_chains):
            raise ValueError("items must have the same length as chains.")

    if extra_info is not None:
        if isinstance(extra_info, (list, tuple)):
            if len(extra_info) == len(labels):
                labels = [f"{label} {extra}" for label, extra in zip(labels, extra_info)]
            else:
                labels = [f"{label} {' '.join(str(extra) for extra in extra_info)}" for label in labels]
        else:
            labels = [f"{label} {extra_info}" for label in labels]

    if filled is None and isinstance(chains, dict) and labels and labels[0].startswith(str(item_std)):
        filled = [True] + [False] * (len(labels) - 1)

    from desilike.samples import plotting
    g = plotting.plot_triangle(
        plot_chains,
        params=params,
        labels=labels,
        filled=filled,
        markers=markers,
        **kwargs,
    )

    if truth:
        overlay_truth_values_on_triangle(g, params, truth)

    if texts:
        import matplotlib.pyplot as plt
        plt.suptitle(texts, x=0.7, y=0.7, fontsize="large")
    if fn is not None:
        g.export(str(fn))
    elif show:
        import matplotlib.pyplot as plt
        plt.show()

    return g
