import sys,os
dir_hoddies='/global/homes/j/jiaxi/codes/HODDIES'
sys.path.append(dir_hoddies)
from HODDIES import HOD
import numpy as np
import pandas as pd

names = ["Ac", "As", "log_Mcent", "M_1", "M_0", "alpha", "sigma_M", "f_sigv"]

def HOD_lhc(tracer, N_samples):
    from scipy.stats import qmc
    import corner
    import matplotlib.pyplot as plt
    # The priors of HOD parameters
    Nfix=1
    if tracer == 'LRG':
        bounds = {
            "Ac":        (0.0, 1.0),
            "As":        (1.0, 1.0),
            "log_Mcent": (12, 13.8),
            "M_0":       (9, 14.8), # (𝜅𝑀cen)
            "M_1":       (12.5, 15.5), 
            "alpha":     (0.0, 2.0),
            "sigma_M":   (0.0, 3.0),
            "f_sigv":    (0.0, 3.0),
        }
    elif tracer == 'ELG':
        bounds = {
            "Ac":        (0.0, 1.0),
            "As":        (0.0, 1.0),
            "log_Mcent": (11.0, 12.2),
            "M_0":       (11.0, 12.0), # (𝜅𝑀cen)
            "M_1":       (13.0, 13.0), 
            "alpha":     (-2.0, 0.0),
            "sigma_M":   (0.0, 3.0),
            "f_sigv":    (0.0, 3.0),
        }
    elif tracer == 'QSO':
        bounds = {
            "Ac":        (0.0, 1.0),
            "As":        (1.0, 1.0),
            "log_Mcent": (11.2, 14),
            "M_0":       (10.6, 14.5), # (𝜅𝑀cen)
            "M_1":       (12.0, 16.0), 
            "alpha":     (0.3, 2.0),
            "sigma_M":   (0.0, 3.0),
            "f_sigv":    (0.0, 3.0),
        }
    else:
        bounds = {
            "Ac":        (0.0, 1.0),
            "As":        (0.0, 1.0),
            "log_Mcent": (11.0, 14.0),
            "M_0":       (9.0, 14.8), 
            "M_1":       (12.0, 16.0), 
            "alpha":     (-2.0, 2.0),
            "sigma_M":   (0.0, 3.0),
            "f_sigv":    (0.0, 3.0),
        }
        Nfix = 0
        ValueError("a general case has not been implemented")

    # generate latin hyper cube parameters
    fn_lhc = f"bphi_HOD_LatinHyperCube_{tracer}_N{N_samples}.csv"
    if not os.path.exists(fn_lhc):
        d = len(names)-Nfix

        np.random.seed(42)
        sampler = qmc.LatinHypercube(d=d, seed=123, optimization="random-cd")
        u = sampler.random(n=N_samples)   # uniform in [0,1]^d

        samples = np.zeros((N_samples,len(names)))
        uid = 0
        for j, p in enumerate(names):
            lo, hi = bounds[p]
            if lo != hi:
                samples[:, j] = lo + u[:, uid] * (hi - lo)
                uid += 1
            else:
                samples[:, j] = lo

        df = pd.DataFrame(samples, columns=names)
        df.to_csv(fn_lhc, index=False)

    df = pd.read_csv(fn_lhc)

    # plot the latincube parameter space
    fig = corner.corner(
        df[names].values,
        labels=names,
        range=[bounds[p] for p in names],  # prevents "no dynamic range" errors

        # plot Scatter only
        plot_density=False,
        plot_contours=False,     # <-- important (some versions need this)
        fill_contours=False,
        no_fill_contours=True,
        levels=None,
        smooth=None,

        # adjust the figure
        data_kwargs={"ms": 5, "alpha": 1.0}
    
    )

    # --- Clear diagonal panels completely ---
    axes = np.array(fig.axes).reshape((len(names), len(names)))

    for i in range(len(names)):
        ax = axes[i, i]
        ax.cla()  # clear everything
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)  # remove box

    plt.savefig(f'HOD_latincube_{tracer}')
    plt.close()

    return df

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def run_one_theta(HOD_obj, fname, params):
    # set parameters
    if os.path.exists(fname):
        return 
    
    # create mocks if not exists
    tracer_args = HOD_obj.args["tracer"]
    for name, val in zip(names, params):
        tracer_args[name] = val

    cats = HOD_obj.make_mock_cat(fix_seed=None)

    cats.write(fname)
    return 


if __name__ == "__main__":
    tracer = sys.argv[1]

    # snapshot list (same logic as your original)
    if tracer == "LRG":
        zsnaps = [0.5, 0.8, 1.1]
    elif tracer == "ELG":
        zsnaps = [1.1, 1.325]
    elif tracer == "QSO":
        zsnaps = [1.1, 1.325, 1.8, 2.5]
    else:
        zsnaps = [0.5, 0.8, 1.1, 1.325, 1.8, 2.5]

    N_lhc = 24
    outdir = "/pscratch/sd/j/jiaxi/b_phi/mocks_lhc"
    param_file = "/pscratch/sd/j/jiaxi/b_phi/conf/lowS8_HOD.yaml"
    path_to_abacus_sim = "/global/cfs/cdirs/desi/cosmosim/Abacus"
    os.makedirs(outdir, exist_ok=True)

    # --- rank 0 builds df_lhc and broadcasts ---
    if rank == 0:
        df_lhc = HOD_lhc(tracer, N_lhc)  # your LHC generator
        theta_params = df_lhc.loc[:N_lhc-1, names].to_numpy()
    else:
        theta_params = None

    theta_params = comm.bcast(theta_params, root=0)  # shape (N_lhc, 8)

    # Split indices across ranks
    my_indices = np.array_split(np.arange(N_lhc), size)[rank]

    for zsnap in zsnaps:
        # Each rank builds its own HOD_obj ONCE per zsnap
        HOD_obj = HOD(
            param_file=param_file,
            path_to_abacus_sim=path_to_abacus_sim,
            **{"hcat": {"z_simu": zsnap, "Abacus": {"z_simu": zsnap}}}
        )

        # tracer-specific config once
        if tracer == "LRG":
            HOD_obj.args["tracer"]["link_sat_to_central"] = True
        elif tracer == "ELG":
            HOD_obj.args["tracer"]["HOD_model"] = "mHMQ"
            HOD_obj.args["tracer"]["gamma"] = 7.06
            HOD_obj.args["tracer"]["vel_sat"] = "rd_normal"

        # Run assigned realizations
        n_done_local = 0
        n_skip_local = 0

        for idx in my_indices:
            params = theta_params[idx]
            fn = f"{outdir}/AbacusSummit_base_c000_ph000_HOD_{tracer}_z{zsnap:.3f}_i{idx:03d}.fits"
            if os.path.exists(fn):
                n_skip_local += 1
                continue
            run_one_theta(HOD_obj, fn, params)
            n_done_local += 1

        n_done = comm.allreduce(n_done_local, op=MPI.SUM)
        n_skip = comm.allreduce(n_skip_local, op=MPI.SUM)

        comm.Barrier()
        if rank == 0:
            print(f"[z={zsnap:.3f}] wrote {n_done} new, skipped {n_skip}, total {n_done+n_skip}")
            
r"""
global _HOD_obj,tracer, zsnap
def HOD_mocks(*theta): 
    global _HOD_obj,tracer, zsnap

    #_HOD_obj  = theta[0]
    idx      = theta[0]

    # load specific parameters for each tracer 
    if tracer == 'LRG': 
        _HOD_obj.args['tracer']['link_sat_to_central']=True 
    elif tracer == 'ELG': 
        _HOD_obj.args['tracer']['HOD_model']= 'mHMQ' 
        _HOD_obj.args['tracer']['gamma'] = 7.06 
        _HOD_obj.args['tracer']['vel_sat'] = 'rd_normal' 
    
    # implement the HOD parameters in Latin Hyper Cube 
    for i,name in enumerate(names): 
        _HOD_obj.args['tracer'][name] = theta[1][i]
    
    # get the mock and 2PCF 
    fname  = f"/pscratch/sd/j/jiaxi/b_phi/mocks_lhc/AbacusSummit_base_c000_ph000_HOD_{tracer}_z{zsnap:.3f}_i{idx:03d}.fits.gz"
    cats   = _HOD_obj.make_mock_cat(fix_seed=42) 
    rp, wp = _HOD_obj.get_wp(cats, tracers=tracer)
    s, xi  = _HOD_obj.get_2PCF(cats, tracers=tracer)
    np.savetxt(fname[:-9]+'_wp.txt',np.array([rp,wp]).T)
    np.savetxt(fname[:-9]+'_mps.txt',np.array([s,xi[0],xi[1]]).T)
    cats.write(fname) 
    return

if __name__ == "__main__":
    from multiprocessing import Pool
    #from functools import partial
    tracer = sys.argv[1]

    if tracer == "LRG":
        zsnaps = [0.5, 0.8, 1.1]
    elif tracer == "ELG":
        zsnaps = [1.1, 1.325]
    elif tracer == "QSO":
        zsnaps = [1.1, 1.325, 1.8, 2.5]
    else:
        zsnaps = [0.5, 0.8, 1.1, 1.325, 1.8, 2.5]

    N_lhc = 24
    df_lhc = HOD_lhc(tracer, N_lhc)  # the Latin hypercube table

    param_file = "/pscratch/sd/j/jiaxi/b_phi/conf/lowS8_HOD.yaml"
    path_to_abacus_sim = "/global/cfs/cdirs/desi/cosmosim/Abacus"
    outdir = "/pscratch/sd/j/jiaxi/b_phi/mocks_lhc"
    os.makedirs(outdir, exist_ok=True)

    # Use fewer processes if you don't have 25 cores allocated
    nproc = min(N_lhc, os.cpu_count())

    for zsnap in zsnaps:
        HOD_obj= HOD(param_file=f'/pscratch/sd/j/jiaxi/b_phi/conf/lowS8_HOD.yaml', path_to_abacus_sim='/global/cfs/cdirs/desi/cosmosim/Abacus', **{'hcat':{'z_simu':zsnap,'Abacus':{'z_simu':zsnap}}}) 
        _HOD_obj = HOD_obj
        with Pool(processes=nproc) as p: 
            fnames = p.starmap(HOD_mocks,zip([HOD_obj]*N_lhc, [i for i in range(N_lhc)], df_lhc.loc[:N_lhc-1, names].to_numpy().tolist()))    
        
        print(f"Done z={zsnap}: wrote {len(fnames)} mocks")
"""
r"""
from multiprocessing import Pool, cpu_count

def HOD_mocks(HOD_obj,*theta):
    # load specific parameters for each tracer
    if tracer == 'LRG':
        HOD_obj.args['tracer']['link_sat_to_central']=True
    elif tracer == 'ELG':
        HOD_obj.args['tracer']['HOD_model']= 'mHMQ'
        HOD_obj.args['tracer']['gamma']    = 7.06
        HOD_obj.args['tracer']['vel_sat']  = 'rd_normal'
    # implement the HOD parameters in Latin Hyper Cube
    for i,name in enumerate(names):
        HOD_obj.args['tracer'][name] = theta[i+1]
    
    # get the mock and 2PCF
    cats = HOD_obj.make_mock_cat(fix_seed=None)  
    cats.write(f"/pscratch/sd/j/jiaxi/b_phi/mocks_lhc/AbacusSummit_base_c000_ph000_HOD_{tracer}_z{theta[0]:.3f}.fits.gz")

    return 

if __name__ == "__main__":
    # which tracer
    tracer = sys.argv[1]
    if tracer == 'LRG':
        zsnaps = [0.5,0.8,1.1]
    elif tracer == 'ELG':
        zsnaps = [1.1,1.325]
    elif tracer == 'QSO':
        zsnaps = [1.1,1.325,1.8,2.5]
    else:
        zsnaps = [0.5,0.8,1.1,1.1,1.325,1.1,1.325,1.8,2.5]
    # number of latin-cube samples
    N_lhc = 25
    nproc = min(N_lhc, cpu_count())
    df_lhc = HOD_lhc(tracer, N_lhc)

    # generate HOD mocks and bphi
    for zsnap in zsnaps:
        HOD_obj= HOD(param_file=f'/pscratch/sd/j/jiaxi/b_phi/conf/lowS8_HOD.yaml', path_to_abacus_sim='/global/cfs/cdirs/desi/cosmosim/Abacus', **{'hcat':{'z_simu':zsnap,'Abacus':{'z_simu':zsnap}}})

        thetas = np.column_stack([
            np.full(N_lhc, zsnap),                # zsnap column
            df_lhc.loc[:N_lhc-1, names].to_numpy()
        ])
        with Pool(processes=25) as pool:
            pool.map(HOD_mocks, thetas)

        # implement the HOD parameters in latin hyper cube
"""

