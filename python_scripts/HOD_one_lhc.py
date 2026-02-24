#!/usr/bin/env python3
import os, sys
import numpy as np

from b_phi.HOD_LHC import HOD_lhc  
from HOD_utils import load_config, compute_all, compute_wpmps, HOD_plot

def main():
    if len(sys.argv) != 6:
        print("Usage: HOD_one_lhc.py tracer HOD_ID zmin zmax zsnap", file=sys.stderr)
        sys.exit(2)

    tracer = sys.argv[1]
    hod_id = int(sys.argv[2])       # 0to99
    zmin   = float(sys.argv[3])
    zmax   = float(sys.argv[4])
    zsnap  = float(sys.argv[5])     # 0.5 / 0.8 / 0.95 / 1.1

    N_lhc = 100
    rebin = 2
    sbins = np.geomspace(0.01, 100., 49)[::rebin]
    cosmos = ['000','004']

    # Generate/load LHC table and parameter names
    df_lhc, names = HOD_lhc(tracer, zsnap, N_lhc)
    theta = df_lhc.loc[hod_id, names].to_numpy()

    dir_source = "/pscratch/sd/j/jiaxi/b_phi"
    zdir = f"{zsnap:.3f}"

    for cosmo in cosmos:
        if tracer == 'LRG':
            from abacusnbody.hod.abacus_hod import AbacusHOD

            fn_config = f"{dir_source}/HOD_config/tmp/abacushod_c{cosmo}_ph000_z{zdir}.yaml"
            if not os.path.exists(fn_config):
                raise FileNotFoundError(f"Missing config yaml: {fn_config}")

            config = load_config(fn_config)
            sim_params = config.get("sim_params", {})
            HOD_params = config.get("HOD_params", {})
            clustering_params = config.get("clustering_params", {})
            nthread = int(config.get("nthread", 32))

            # Make output unique per realization (so array jobs don't collide)
            outdir = f"{dir_source}/mocks_lhc/{tracer}_z{zdir}_c{cosmo}"
            os.makedirs(outdir, exist_ok=True)

            done_flag = os.path.join(outdir, f"id{hod_id+1:03d}_pk_RSD.npy")
            
            if os.path.exists(done_flag):
                print(f"Already done: {done_flag}")
                continue

            # Fresh process + fresh AbacusHOD per task = safest against segfault accumulation
            ball_profiles = AbacusHOD(sim_params, HOD_params, clustering_params)

            # Apply LHC params
            for k, p in enumerate(theta):
                if p <= 0.01:
                    p = 0.01
                ball_profiles.tracers[tracer][names[k]] = p

            mock = compute_all(ball_profiles, nthread=nthread, out=False, verbose=True)
            compute_wpmps(mock, sbins, zsnap, outdir, f"id{hod_id+1:03d}")
        elif tracer == 'ELG':

        
            
        # Save plot/output (uses your helper)
        HOD_plot(tracer, zmin, zmax, sbins, outdir, f"_id{hod_id+1:03d}")

        # Append results (safe enough for small scale; for heavy concurrency use per-task files)
        results_fn = f"{dir_source}/bphi_results_AbacusHOD.txt"
        with open(results_fn, "a") as f:
            f.write(f"{tracer} {zsnap:.3f} {cosmo} {hod_id+1} {len(mock[tracer]['x'])}\n")

        print(f"Completed id={hod_id+1} -> {outdir}")

if __name__ == "__main__":
    main()