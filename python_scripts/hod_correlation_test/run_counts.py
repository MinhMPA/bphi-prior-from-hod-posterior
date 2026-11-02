#!/usr/bin/env python3
"""Interactive pilot or sequential counts-only campaign; never submits a job."""
import argparse
import copy
import fcntl
import gc
import hashlib
import inspect
import json
import os
import signal
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
STOP = False


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def count_path(run_dir, row, cosmo):
    return run_dir / "counts" / row["ensemble"] / f"z{row['z']}" / f"draw{row['draw']:04d}_c{cosmo}.json"


def read_checkpoint(path, signature):
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value["signature"] != signature:
        raise RuntimeError(f"Checkpoint inputs/code differ: {path}. Use a new --run-dir; not overwriting.")
    return value


def load_snapshot_config(run_dir, manifest, z, cosmo):
    path = run_dir / "configs" / f"abacushod_c{cosmo}_ph000_z{float(z):.3f}.yaml"
    config = yaml.safe_load(path.read_text())
    if config != manifest["snapshots"][z]["configs"][cosmo]:
        raise RuntimeError(f"YAML differs from the frozen manifest: {path}. Prepare a new run for changed configurations.")
    sim = config["sim_params"]
    if sim["sim_name"] != f"AbacusSummit_base_c{cosmo}_ph000" or not np.isclose(sim["z_mock"], float(z)):
        raise ValueError(f"Wrong snapshot/cosmology in {path}")
    return config


def load_engine():
    import sys
    import importlib.metadata
    print("Importing local HOD implementation...", flush=True)
    import abacusnbody
    from abacusnbody.hod import abacus_hod, GRAND_HOD
    sys.path.insert(0, str(ROOT / 'python_scripts'))
    import HOD_utils
    checkout = Path(os.environ.get("LOCAL_ABACUSUTILS", Path(inspect.getfile(abacusnbody)).resolve().parent.parent)).resolve()
    module = Path(inspect.getfile(abacusnbody)).resolve()
    if not module.is_relative_to(checkout):
        raise RuntimeError(f"Expected local abacusutils under {checkout}, but imported {module}")
    paths = [Path(__file__), Path(abacus_hod.__file__), Path(GRAND_HOD.__file__), Path(HOD_utils.__file__)]
    runtime = dict(abacusnbody=str(module), sources={str(path): file_hash(path) for path in paths},
                   python=sys.version, packages={name: importlib.metadata.version(name) for name in ("numpy", "numba")})
    print(f"Using abacusnbody from {module}", flush=True)
    return abacus_hod.AbacusHOD, HOD_utils.compute_all, runtime


def stop_requested(args):
    return STOP or (args.stop_file is not None and args.stop_file.exists())


def request_stop(signum, frame):
    global STOP
    STOP = True
    print(f"Signal {signum}: finishing the current HOD evaluation, then stopping.", flush=True)


def resolve_ic(row, expected_complete, volume, nbar):
    if "ic" in row["params"]:
        return row["params"]["ic"], "stored Ac" if row["ensemble"] == "chain_s1" else "legacy LHC"
    if not np.isfinite(expected_complete) or expected_complete <= 0:
        raise ValueError("Expected complete-sample count must be finite and positive")
    return min(1.0, nbar * volume / expected_complete), "derived once in fiducial c000"


def approve(run_dir, manifest, manifest_sha):
    signatures, completed = set(), set()
    for row in manifest["rows"]:
        paths = [count_path(run_dir, row, cosmo) for cosmo in ("000", "004")]
        if not all(path.exists() for path in paths):
            continue
        pair = [json.loads(path.read_text()) for path in paths]
        if all(value["status"] == "ok" and value["manifest_sha256"] == manifest_sha
               and value["ngal"] > manifest["settings"]["ngal_min"] for value in pair):
            signatures.update(value["signature"] for value in pair)
            completed.add((row["z"], row["ensemble"]))
    required = {(z, ensemble) for z in manifest["snapshots"] for ensemble in ("chain_s1", "chain_s3", "legacy_check")}
    if completed != required or len(signatures) != 1:
        raise RuntimeError(f"Need successful paired pilot counts for every bin/ensemble under one code version. Missing: {sorted(required - completed)}")
    save_json(run_dir / "debug_approved.json", dict(manifest_sha256=manifest_sha, signature=signatures.pop(),
              statement="User manually reviewed the pilot, sigma convention, nbar calibration, and selection policy",
              approved_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    print("Recorded manual debug approval. No job was submitted.")


def run(args, manifest, manifest_sha):
    rows = [row for row in manifest["rows"] if row["z"] in args.zsnaps and
            (args.limit is None or row["draw"] <= args.limit)]
    if not rows:
        raise ValueError("No matching prepared draws")
    configs = {(z, cosmo): load_snapshot_config(args.run_dir, manifest, z, cosmo)
               for z in args.zsnaps for cosmo in ("000", "004")}
    if args.dry_run:
        for z in args.zsnaps:
            group = [row for row in rows if row["z"] == z]
            print(f"z={z}: {len(group)} rows, {sum(bool(row['rejection']) for row in group)} physical rejections; c000 then c004")
        print("Dry run only: no halo catalogue loaded and no job submitted.")
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run counts on an allocated interactive/batch compute node, not a login node")
    factory, compute_all, runtime = load_engine()
    signature = fingerprint(dict(manifest_sha256=manifest_sha, runtime=runtime, threads=args.threads))
    if args.require_approval:
        approval = args.run_dir / "debug_approved.json"
        if not approval.exists() or json.loads(approval.read_text())["signature"] != signature:
            raise RuntimeError("Matching manual pilot approval is required; review results and use --approve-debug first")
    save_json(args.run_dir / "runtime.json", dict(signature=signature, **runtime))
    for z in args.zsnaps:
        group = [row for row in rows if row["z"] == z]
        for cosmo in ("000", "004"):
            pending = []
            for row in group:
                target = count_path(args.run_dir, row, cosmo)
                if read_checkpoint(target, signature) is not None:
                    print(f"Reuse {target}", flush=True)
                    continue
                if row["rejection"]:
                    save_json(target, dict(signature=signature, manifest_sha256=manifest_sha,
                              status="rejected", reason=row["rejection"], row=row, cosmo=cosmo))
                    continue
                pending.append(row)
            if not pending:
                continue
            if stop_requested(args):
                print("Stopped between evaluations; resubmit the same run to resume.")
                return
            config = copy.deepcopy(configs[z, cosmo])
            config["sim_params"]["output_dir"] = str(args.run_dir / "unused_catalog_output")
            config_path = args.run_dir / "configs" / f"abacushod_c{cosmo}_ph000_z{float(z):.3f}.yaml"
            print(f"Loading {config_path}: {len(pending)} remaining HOD evaluations", flush=True)
            started = time.monotonic()
            ball = factory(config["sim_params"], config["HOD_params"], config["clustering_params"])
            baseline = copy.deepcopy(ball.tracers)
            if set(baseline) != {"LRG"}:
                raise ValueError("This test requires LRG-only configurations")
            volume = float(ball.params["Lbox"]) ** 3
            print(f"Loaded in {time.monotonic() - started:.1f}s; volume={volume:g}", flush=True)
            for row in pending:
                if stop_requested(args):
                    print("Stopped between evaluations; resubmit the same run to resume.")
                    return
                started = time.monotonic()
                ball.tracers = copy.deepcopy(baseline)
                ball.tracers["LRG"].update(row["params"])
                complete_count = None
                if cosmo == "000":
                    ball.tracers["LRG"]["ic"] = 1.0
                    expected, _ = ball.compute_ngal(Nthread=args.threads)
                    complete_count = float(expected["LRG"])
                    ic, ic_source = resolve_ic(row, complete_count, volume, manifest["settings"]["reference_nbar"][z])
                else:
                    fiducial = read_checkpoint(count_path(args.run_dir, row, "000"), signature)
                    if fiducial is None or fiducial["status"] != "ok":
                        raise RuntimeError("c004 requires the matching completed c000 count and fixed completeness")
                    ic, ic_source = fiducial["ic"], fiducial["ic_source"]
                ball.tracers["LRG"]["ic"] = ic
                print(f"z={z} c{cosmo} {row['ensemble']} draw={row['draw']} ic={ic:.8g} parameters={ball.tracers['LRG']}", flush=True)
                mock = compute_all(ball, nthread=args.threads, out=False, verbose=True)
                ngal = len(mock["LRG"]["x"])
                del mock
                value = dict(signature=signature, manifest_sha256=manifest_sha, status="ok", row=row,
                             cosmo=cosmo, ngal=ngal, ic=ic, ic_source=ic_source, volume=volume,
                             expected_complete_c000=complete_count, parameters=ball.tracers["LRG"],
                             elapsed_seconds=time.monotonic() - started,
                             randoms="Unchanged preprocessed AbacusHOD random arrays; no reseeding between draws")
                # Convert NumPy scalars returned by the engine before JSON serialization.
                value["parameters"] = {key: float(val) for key, val in value["parameters"].items()}
                save_json(count_path(args.run_dir, row, cosmo), value)
                print(f"Saved Ngal={ngal} in {value['elapsed_seconds']:.1f}s", flush=True)
                if cosmo == "004" and ngal > 0 and fiducial["ngal"] > 0:
                    bphi = 2 * np.log(fiducial["ngal"] / ngal) / np.log(0.807952 / 0.75)
                    print(f"PAIRED RESULT: z={z} {row['ensemble']} draw={row['draw']} bphi={bphi:.8f}", flush=True)
                    if row["ensemble"] == "legacy_check":
                        old = next((item for item in manifest["published"][z] if item["hod_id"] == row["draw"]), None)
                        print(f"Recorded legacy result for comparison: {old}", flush=True)
            del ball
            gc.collect()
    print("Selected counts complete. No clustering or catalogue files were written.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=HERE / "runs/default")
    parser.add_argument("--zsnaps", nargs="+", choices=["0.5", "0.8", "1.1"], default=["0.5", "0.8", "1.1"])
    parser.add_argument("--limit", type=int, help="Run draw IDs 1..limit per ensemble; omit for all prepared points")
    parser.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 32)))
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-approval", action="store_true")
    parser.add_argument("--approve-debug", action="store_true", help="Record YOUR manual approval after reviewing all bins; does not submit")
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    if args.threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error("threads and limit must be positive")
    path = args.run_dir / "manifest.json"
    if not path.exists():
        parser.error("Run prepare_ensembles.py first")
    manifest, manifest_sha = json.loads(path.read_text()), file_hash(path)
    with (args.run_dir / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.approve_debug:
            approve(args.run_dir, manifest, manifest_sha)
            return
        for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, request_stop)
        run(args, manifest, manifest_sha)


if __name__ == "__main__":
    main()
