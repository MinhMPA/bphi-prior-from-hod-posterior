#!/usr/bin/env python3
import os, sys
import numpy as np

from HOD_LHC import HOD_lhc  
from HOD_utils import load_config, compute_all, compute_wpmps, HOD_plot

def main():
    if len(sys.argv) != 7:
        print("Usage: HOD_one_lhc.py tracer HOD_ID zmin zmax zsnap simtype", file=sys.stderr)
        sys.exit(2)

    tracer = sys.argv[1]
    hod_id = int(sys.argv[2])       # 0 or 1to100
    zmin   = float(sys.argv[3])
    zmax   = float(sys.argv[4])
    zsnap  = float(sys.argv[5])     # 0.5 / 0.8 / 0.95 / 1.1
    simtype= sys.argv[6]

    N_lhc = 100
    rebin = 2
    sbins = np.geomspace(0.01, 100., 49)[::rebin]
    if simtype == 'SU':
        cosmos     = ['000','004']
        abacustype = 'abacushod'
    elif simtype == 'PNG':
        cosmos     = ['000','300','301']#,'302','303']
        abacustype = 'abacushod_png'

    # Generate/load LHC table and parameter names
    df_lhc, names = HOD_lhc(tracer, zsnap, N_lhc)
    use_default_hod = (hod_id == 0)
    if not use_default_hod:
        theta = df_lhc.loc[hod_id - 1, names].to_numpy()

    dir_source = "/pscratch/sd/j/jiaxi/b_phi"
    zdir = f"{zsnap:.3f}"

    for cosmo in cosmos:
        # Make output unique per realization (so array jobs don't collide)
        if use_default_hod:
            outdir = f"{dir_source}/mocks_lhc_{simtype}/"
            default_prefix = f"{tracer}_z{zdir}_c{cosmo}_default"
            done_flag = os.path.join(outdir, f"{default_prefix}_2pt_RSD.txt")
        else:
            outdir = f"{dir_source}/mocks_lhc_{simtype}/{tracer}_z{zdir}_c{cosmo}"
            done_flag = os.path.join(outdir, f"id{hod_id:03d}_2pt_RSD.txt")
        os.makedirs(outdir, exist_ok=True)
            
        
        if os.path.exists(done_flag):
            print(f"Already done: {done_flag}")
            continue

        if tracer == 'LRG':
            from abacusnbody.hod.abacus_hod import AbacusHOD
            fn_config = f"{dir_source}/HOD_config/tmp/{abacustype}_c{cosmo}_ph000_z{zdir}.yaml"

            if not os.path.exists(fn_config):
                raise FileNotFoundError(f"Missing config yaml: {fn_config}")

            config = load_config(fn_config)
            sim_params = config.get("sim_params", {})
            HOD_params = config.get("HOD_params", {})
            clustering_params = config.get("clustering_params", {})
            nthread = int(config.get("nthread", 32))

            # Fresh process + fresh AbacusHOD per task = safest against segfault accumulation
            ball_profiles = AbacusHOD(sim_params, HOD_params, clustering_params)
            if not use_default_hod:
                # Apply LHC params
                for k, p in enumerate(theta):
                    if p <= 0.01:
                        p = 0.01
                    ball_profiles.tracers[tracer][names[k]] = p

            mock = compute_all(ball_profiles, nthread=nthread, out=False, verbose=True)
            Ngal = len(mock[tracer]['x'])
        elif tracer == 'ELG':
            # GPU by default; set ELG_FORCE_CPU_JAX=1 in job env to force CPU-JAX fallback.
            if os.environ.get('ELG_FORCE_CPU_JAX', '0') == '1':
                os.environ['JAX_PLATFORMS'] = 'cpu'
                os.environ['JAX_PLATFORM_NAME'] = 'cpu'
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
                print('ELG backend: forcing JAX CPU')
            else:
                print('ELG backend: JAX default (GPU-enabled env)')

            sys.path.append('/global/homes/j/jiaxi/codes/HODDIES')
            from HODDIES import HOD
            # Apply LHC params
            ELG_params = {}
            if not use_default_hod:
                ELG_params[tracer] = {names[k]: (0.01 if p <= 0.01 else p) for k, p in enumerate(theta)}
            ELG_params['hcat'] = {'z_simu':zsnap,
                                    'Abacus':{'sim_name':f'AbacusSummit_base_c{cosmo}_ph000','z_simu':zsnap}}
            HOD_obj= HOD(param_file=f'./HOD_config/hoddies_ELG_z0.950.yaml', path_to_abacus_sim='/global/cfs/cdirs/desi/cosmosim/Abacus', \
                        **ELG_params)
            # generate mock and clustering
            mock = HOD_obj.make_mock_cat(fix_seed=600) 
            Ngal = len(mock['x'])

        # Append Ngal even if clustering is skipped/fails.
        if use_default_hod:
            print(f"{tracer} {zsnap:.3f} {cosmo} {hod_id} {Ngal}\n")
        else:
            results_fn = f"{dir_source}/bphi_results_{abacustype}.txt"
            with open(results_fn, "a") as f:
                f.write(f"{tracer} {zsnap:.3f} {cosmo} {hod_id} {Ngal}\n")

        if Ngal <= 1:
            # Some extreme HOD points produce too few galaxies for correlation estimators.
            with open(done_flag, "w") as f:
                f.write(f"# Insufficient galaxies for 2pt (Ngal={Ngal})\n")
            print(f"Skip clustering: tracer={tracer} z={zsnap:.3f} id={hod_id} cosmo={cosmo} Ngal={Ngal}")
            continue

        try:
            if use_default_hod:
                compute_wpmps(mock, sbins, zsnap, tracer=tracer, output=outdir, fn=default_prefix)
                # Save plot/output (uses your helper)
                HOD_plot(tracer, zmin, zmax, sbins, output=outdir, ihod=f"_{default_prefix}")
            else:
                compute_wpmps(mock, sbins, zsnap, tracer=tracer, output=outdir, fn=f"id{hod_id:03d}")
                # Save plot/output (uses your helper)
                HOD_plot(tracer, zmin, zmax, sbins, output=outdir, ihod=f"_id{hod_id:03d}")
            print(f"Completed id={hod_id} cosmo={cosmo} Ngal={Ngal} -> {outdir}")
        except Exception as exc:
            print(
                f"Clustering/plot failed for tracer={tracer} z={zsnap:.3f} id={hod_id} cosmo={cosmo} Ngal={Ngal}: {exc}",
                file=sys.stderr,
            )

if __name__ == "__main__":
    main()
