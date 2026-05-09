#!/usr/bin/env python3
import argparse
import os, sys
from pathlib import Path
import numpy as np

from HOD_LHC import HOD_lhc  
from HOD_utils import load_config, compute_all, compute_wpmps, HOD_plot, compute_linear_bias


def format_abias_tag(name, value):
    tag = f"{value:.3f}".replace("-", "m").replace(".", "p")
    return f"_{name}{tag}"


def main():
    parser = argparse.ArgumentParser(description="Run one HOD LHC/default realization.")
    parser.add_argument("tracer")
    parser.add_argument("hod_id", type=int, help="0 for YAML defaults, 1-100 for LHC rows")
    parser.add_argument("zmin", type=float)
    parser.add_argument("zmax", type=float)
    parser.add_argument("zsnap", type=float)
    parser.add_argument("simtype", choices=["SU", "PNG"])
    parser.add_argument("--Acent", type=float, default=None, help="Optional AbacusHOD Acent value.")
    parser.add_argument("--Asat", type=float, default=None, help="Optional AbacusHOD Asat value.")
    parser.add_argument("--Bcent", type=float, default=None, help="Optional AbacusHOD Bcent value.")
    parser.add_argument("--Bsat", type=float, default=None, help="Optional AbacusHOD Bsat value.")
    parser.add_argument("--cosmos", nargs="+", default=None, help="Cosmology tags to run, e.g. 000 300.")
    parser.add_argument("--config", default=None, help="Use one explicit LRG AbacusHOD YAML config.")
    parser.add_argument("--output-root", default=None, help="Directory root for mock outputs.")
    parser.add_argument("--force", action="store_true", help="Run even if the done flag already exists.")
    parser.add_argument("--compute-b1", action="store_true", help="For c000 mocks, compute real-space P(k) and linear bias b1.")
    parser.add_argument("--b1-kmin", type=float, default=0.01, help="Minimum k for b1 estimate.")
    parser.add_argument("--b1-kmax", type=float, default=0.05, help="Maximum k for b1 estimate.")
    args = parser.parse_args()

    tracer = args.tracer
    hod_id = args.hod_id       # 0 or 1to100
    zmin   = args.zmin
    zmax   = args.zmax
    zsnap  = args.zsnap        # 0.5 / 0.8 / 0.95 / 1.1
    simtype= args.simtype
    abias_params = {name: getattr(args, name) for name in ["Acent", "Asat", "Bcent", "Bsat"]}
    active_abias = {name: value for name, value in abias_params.items() if value is not None}
    abias_suffix = "".join(format_abias_tag(name, value) for name, value in active_abias.items())

    N_lhc = 100
    rebin = 2
    sbins = np.geomspace(0.01, 100., 49)[::rebin]
    if simtype == 'SU':
        cosmos     = ['000','004']
        abacustype = 'abacushod'
    elif simtype == 'PNG':
        cosmos     = ['000','300','301']#,'302','303']
        abacustype = 'abacushod_png'
    if args.cosmos is not None:
        cosmos = [str(cosmo).replace("c", "").zfill(3) for cosmo in args.cosmos]
    custom_config = Path(args.config).expanduser() if args.config else None
    if custom_config is not None:
        if tracer != "LRG" or len(cosmos) != 1:
            raise ValueError("--config is intended for a single LRG cosmology debug run.")
        if not custom_config.exists():
            raise FileNotFoundError(f"Missing config yaml: {custom_config}")

    # Generate/load LHC table and parameter names
    df_lhc, names = HOD_lhc(tracer, zsnap, N_lhc)
    use_default_hod = (hod_id == 0)
    if not use_default_hod:
        theta = df_lhc.loc[hod_id - 1, names].to_numpy()

    dir_source = "/pscratch/sd/j/jiaxi/b_phi"
    output_root = args.output_root or f"{dir_source}/mocks_lhc_{simtype}"
    results_root = args.output_root or dir_source
    zdir = f"{zsnap:.3f}"

    for cosmo in cosmos:
        # Make output unique per realization (so array jobs don't collide)
        if use_default_hod:
            outdir = output_root
            default_prefix = f"{tracer}_z{zdir}_c{cosmo}_default{abias_suffix}"
            prefix = default_prefix
            done_flag = os.path.join(outdir, f"{default_prefix}_2pt_RSD.txt")
        else:
            outdir = os.path.join(output_root, f"{tracer}_z{zdir}_c{cosmo}")
            id_prefix = f"id{hod_id:03d}{abias_suffix}"
            prefix = id_prefix
            done_flag = os.path.join(outdir, f"{id_prefix}_2pt_RSD.txt")
        os.makedirs(outdir, exist_ok=True)
            
        b1_done_flag = os.path.join(outdir, f"{prefix}_b1.txt")
        skip_existing_clustering = os.path.exists(done_flag) and not args.force
        needs_missing_b1 = args.compute_b1 and use_default_hod and not os.path.exists(b1_done_flag)

        if skip_existing_clustering and not needs_missing_b1:
            print(f"Already done: {done_flag}")
            continue
        if skip_existing_clustering and needs_missing_b1:
            print(f"Already done: {done_flag}; regenerating c000 mock to add missing b1.")

        if tracer == 'LRG':
            from abacusnbody.hod.abacus_hod import AbacusHOD
            fn_config = (
                str(custom_config)
                if custom_config is not None
                else f"{dir_source}/HOD_config/tmp/{abacustype}_c{cosmo}_ph000_z{zdir}.yaml"
            )

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
            for name, value in active_abias.items():
                ball_profiles.tracers[tracer][name] = value

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

        b1_value = None
        if args.compute_b1 and use_default_hod and Ngal > 1:
            try:
                mock_for_b1 = mock
                if tracer == "LRG":
                    mock_for_b1 = compute_all(ball_profiles, nthread=nthread, out=False, verbose=False, want_rsd=False)
                b1_value = compute_linear_bias(
                    mock_for_b1, zsnap, tracer=tracer, output=outdir, fn=prefix,
                    kmin=args.b1_kmin, kmax=args.b1_kmax,
                )
                print(
                    f"Computed b1: tracer={tracer} z={zsnap:.3f} id={hod_id} "
                    f"cosmo={cosmo} b1={b1_value:.8f}"
                )
            except Exception as exc:
                print(
                    f"b1 computation failed for tracer={tracer} z={zsnap:.3f} "
                    f"id={hod_id} cosmo={cosmo} Ngal={Ngal}: {exc}",
                    file=sys.stderr,
                )

        b1_suffix = "" if b1_value is None else f" {b1_value:.8f}"

        # Append Ngal even if clustering is skipped/fails.
        if use_default_hod:
            print(f"{tracer} {zsnap:.3f} {cosmo} {hod_id} {Ngal}{b1_suffix}\n")
        else:
            results_tag = abias_suffix
            os.makedirs(results_root, exist_ok=True)
            results_fn = os.path.join(results_root, f"bphi_results_{abacustype}{results_tag}.txt")
            with open(results_fn, "a") as f:
                f.write(f"{tracer} {zsnap:.3f} {cosmo} {hod_id} {Ngal}{b1_suffix}\n")

        if Ngal <= 1:
            # Some extreme HOD points produce too few galaxies for correlation estimators.
            with open(done_flag, "w") as f:
                f.write(f"# Insufficient galaxies for 2pt (Ngal={Ngal})\n")
            print(f"Skip clustering: tracer={tracer} z={zsnap:.3f} id={hod_id} cosmo={cosmo} Ngal={Ngal}")
            continue

        if skip_existing_clustering:
            print(f"Completed b1 update id={hod_id} cosmo={cosmo} Ngal={Ngal} -> {outdir}")
            continue

        try:
            if use_default_hod:
                compute_wpmps(mock, sbins, zsnap, tracer=tracer, output=outdir, fn=default_prefix)
                # Save plot/output (uses your helper)
                HOD_plot(tracer, zmin, zmax, sbins, output=outdir, ihod=f"_{default_prefix}")
            else:
                compute_wpmps(mock, sbins, zsnap, tracer=tracer, output=outdir, fn=id_prefix)
                # Save plot/output (uses your helper)
                HOD_plot(tracer, zmin, zmax, sbins, output=outdir, ihod=f"_{id_prefix}")
            print(f"Completed id={hod_id} cosmo={cosmo} Ngal={Ngal} -> {outdir}")
        except Exception as exc:
            print(
                f"Clustering/plot failed for tracer={tracer} z={zsnap:.3f} id={hod_id} cosmo={cosmo} Ngal={Ngal}: {exc}",
                file=sys.stderr,
            )

if __name__ == "__main__":
    main()
