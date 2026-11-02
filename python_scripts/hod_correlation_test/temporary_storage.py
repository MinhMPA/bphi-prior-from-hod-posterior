#!/usr/bin/env python3
"""Stage small HOD results on CFS and restore outputs after scratch recovers."""
import argparse
import csv
import fcntl
import io
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

from ensemble_results import PARAMETERS
from run_counts import count_path, file_hash, save_json

HERE = Path(__file__).resolve().parent
DEFAULT_RUN = Path("/global/cfs/cdirs/desi/users/jiaxiyu/b_phi_halo_inputs/runs/default")
LEGACY_HOME = Path.home() / "b_phi_hod_correlation_test"


def bounded_copy(source, destination, seconds=20):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".copying")
    try:
        subprocess.run(["timeout", "-k", "2s", f"{seconds}s", "cp", "--", str(source), str(temporary)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Copy failed or scratch read timed out: {source}") from error
    os.replace(temporary, destination)


def csv_records(path):
    return list(csv.DictReader(io.StringIO(path.read_text())))


def make_export_cache(manifest, manifest_sha, signature, count_rows, parameter_rows, hashes):
    parameters = {(row["z"], row["ensemble"], int(row["draw"])): row for row in parameter_rows}
    original = {(row["z"], row["ensemble"], row["draw"]): row for row in manifest["rows"]}
    records = {}
    for entry in count_rows:
        key = (entry["z"], entry["ensemble"], int(entry["draw"]))
        row = original[key]
        if row["source_row"] != int(entry["source_row"]) or row["rejection"]:
            raise RuntimeError(f"Export does not match the frozen draw: {key}")
        n000, n004 = int(entry["n000"]), int(entry["n004"])
        if min(n000, n004) <= 0:
            raise RuntimeError(f"Nonpositive exported galaxy counts: {key}")
        bphi = 2 * np.log(n000 / n004) / np.log(0.807952 / 0.75)
        if not np.isclose(bphi, float(entry["bphi"]), rtol=1e-12, atol=1e-12):
            raise RuntimeError(f"Exported bphi/counts disagree: {key}")
        ic = float(entry["ic"])
        if "ic" in row["params"] and not np.isclose(ic, row["params"]["ic"], rtol=1e-12, atol=1e-12):
            raise RuntimeError(f"Exported completeness differs from the frozen draw: {key}")
        params = dict(manifest["snapshots"][row["z"]]["configs"]["000"]["HOD_params"]["LRG_params"])
        params.update(row["params"])
        params["ic"] = ic
        if row["ensemble"] != "legacy_check":
            exported = parameters[key]
            if int(exported["source_row"]) != row["source_row"] or (int(exported["n000"]), int(exported["n004"])) != (n000, n004):
                raise RuntimeError(f"Count and parameter exports disagree: {key}")
            for name in PARAMETERS:
                value = float(exported[name])
                if not np.isclose(params[name], value, rtol=1e-12, atol=1e-12):
                    raise RuntimeError(f"Parameter export does not match {name} for {key}")
                params[name] = value
        for cosmo, ngal in (("000", n000), ("004", n004)):
            relative = str(count_path(Path("."), row, cosmo))
            records[relative] = dict(signature=signature, manifest_sha256=manifest_sha,
                                     status="ok", row=row, cosmo=cosmo, ngal=ngal, ic=ic,
                                     ic_source="Recorded in saved count/parameter exports", parameters=params,
                                     recovery_source="Validated original bphi_samples.csv and hod_parameter_samples.csv")
    for z in manifest["snapshots"]:
        ids = {record["row"]["draw"] for record in records.values()
               if record["row"]["z"] == z and record["row"]["ensemble"] == "chain_s1"}
        if len(ids) != manifest["settings"]["nsamples"]:
            raise RuntimeError(f"Saved exports do not contain all completed chain_s1 draws for z={z}")
    return dict(parent_manifest_sha256=manifest_sha, exports_sha256=hashes, records=records)


def stage(args):
    source, destination = args.scratch_run.resolve(), args.temporary_run.resolve()
    if source == destination:
        raise ValueError("Temporary and original directories must differ")
    if (destination / "manifest.json").exists() or (destination / "counts").exists():
        raise RuntimeError("A temporary run already exists; reuse it instead of staging again")
    for name in ("manifest.json", "topup_manifest.json", "debug_approved.json", "runtime.json"):
        bounded_copy(source / name, destination / name)
    manifest = json.loads((destination / "manifest.json").read_text())
    manifest_sha = file_hash(destination / "manifest.json")
    approval = json.loads((destination / "debug_approved.json").read_text())
    runtime = json.loads((destination / "runtime.json").read_text())
    if approval["manifest_sha256"] != manifest_sha or approval["signature"] != runtime["signature"]:
        raise RuntimeError("Original approval/runtime does not match the staged manifest")
    hashes = {}
    for name in ("bphi_samples.csv", "hod_parameter_samples.csv", "bphi_summary.csv"):
        path = destination / "original_exports" / name
        bounded_copy(source / "tables" / name, path)
        hashes[name] = file_hash(path)
    cache = make_export_cache(manifest, manifest_sha, approval["signature"],
                              csv_records(destination / "original_exports/bphi_samples.csv"),
                              csv_records(destination / "original_exports/hod_parameter_samples.csv"), hashes)
    save_json(destination / "existing_export_pairs.json", cache)
    aliases = {}
    for original, digest in manifest["sources_sha256"].items():
        path = Path(original)
        if "chains_LRGQSO" not in path.parts and path.parent.name != "LHC":
            continue
        target = destination / "inputs" / ("/".join(path.parts[path.parts.index("b_phi") + 1:]))
        bounded_copy(path, target)
        if file_hash(target) != digest:
            raise RuntimeError(f"Frozen input changed: {original}")
        aliases[original] = str(target)
    save_json(destination / "input_map.json", aliases)
    for z, snapshot in manifest["snapshots"].items():
        for cosmo, config in snapshot["configs"].items():
            path = destination / "configs" / f"abacushod_c{cosmo}_ph000_z{float(z):.3f}.yaml"
            path.parent.mkdir(exist_ok=True)
            path.write_text(yaml.safe_dump(config, sort_keys=False))
    (destination.parent.parent / "logs").mkdir(parents=True, exist_ok=True)
    save_json(destination / "temporary_storage.json", dict(original_run=str(source), temporary_run=str(destination),
              parent_manifest_sha256=manifest_sha, staged_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              cached_pairs=len(cache["records"]) // 2,
              note="Original completed pairs reused from their validated CSV exports; raw scratch checkpoints retained"))
    print(f"Staged {len(cache['records']) // 2} original count pairs at {destination}")
    print("No original HOD counts recalculated. New count checkpoints, tables, plots and logs use the temporary directory.")


def migrate_home(home_root, destination):
    home_root, destination = home_root.resolve(), destination.resolve()
    source = home_root / "runs/default"
    if not source.is_dir() or destination.exists():
        raise RuntimeError("Expected the existing HOME run and an unused CFS run destination")
    if set(path.name for path in home_root.iterdir()) - {"runs", "logs"} or set(
        path.name for path in (home_root / "runs").iterdir()
    ) != {"default"}:
        raise RuntimeError("HOME directory contains unrelated entries; refusing to remove it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (source / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        staging = Path(tempfile.mkdtemp(prefix=".migrate-home-", dir=destination.parent))
        try:
            shutil.copytree(source, staging, dirs_exist_ok=True)
            hashes = {}
            for path in source.rglob("*"):
                if path.is_file():
                    relative = path.relative_to(source)
                    hashes[str(relative)] = file_hash(path)
                    if file_hash(staging / relative) != hashes[str(relative)]:
                        raise RuntimeError(f"HOME/CFS migration checksum mismatch: {relative}")
            manifest = json.loads((staging / "manifest.json").read_text())
            aliases = json.loads((staging / "input_map.json").read_text())
            for original, current in aliases.items():
                relative = Path(current).resolve().relative_to(source)
                if file_hash(staging / relative) != manifest["sources_sha256"][original]:
                    raise RuntimeError(f"Migrated posterior input differs: {original}")
                aliases[original] = str(destination / relative)
            save_json(staging / "input_map.json", aliases)
            metadata = json.loads((staging / "temporary_storage.json").read_text())
            metadata.update(temporary_run=str(destination), migrated_from_home=str(home_root),
                            migrated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            save_json(staging / "temporary_storage.json", metadata)
            logs = destination.parent.parent / "logs"
            for path in (home_root / "logs").rglob("*"):
                if path.is_file():
                    target = logs / path.relative_to(home_root / "logs")
                    if target.exists() and file_hash(target) != file_hash(path):
                        raise RuntimeError(f"Existing CFS log differs: {target}")
                    bounded_copy(path, target)
                    if file_hash(target) != file_hash(path):
                        raise RuntimeError(f"Migrated log checksum mismatch: {path}")
            logs.mkdir(parents=True, exist_ok=True)
            save_json(staging / "home_migration.json", dict(original_home=str(home_root),
                      verified_original_file_hashes=hashes,
                      rebased_files=["input_map.json", "temporary_storage.json"]))
            staging.replace(destination)
            shutil.rmtree(home_root)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    print(f"Verified and migrated {len(hashes)} HOME run files to {destination}")
    print(f"Removed the old HOME directory: {home_root}")


def restore(args):
    source, destination = args.temporary_run.resolve(), args.scratch_run.resolve()
    bounded_copy(destination / "manifest.json", source / "restore_manifest_check.json", seconds=8)
    if file_hash(source / "manifest.json") != file_hash(source / "restore_manifest_check.json"):
        raise RuntimeError("Original and temporary manifests differ; refusing to merge")
    blocked = destination / "counts/chain_s1/z0.8/draw0035_c004.json"
    probe = subprocess.run(["timeout", "-k", "2s", "8s", "head", "-c", "1", str(blocked)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if probe.returncode:
        raise RuntimeError("Known affected scratch checkpoint is still unreadable; all temporary results remain on CFS")
    bounded_copy(destination / "topup_manifest.json", source / "restore_plan_check.json", seconds=8)
    old_plan = json.loads((source / "restore_plan_check.json").read_text())
    new_plan = json.loads((source / "topup_manifest.json").read_text())
    new_rows = {(row["z"], row["ensemble"], row["draw"]): row for row in new_plan["rows"]}
    if old_plan["parent_manifest_sha256"] != new_plan["parent_manifest_sha256"] or any(
        new_rows.get((row["z"], row["ensemble"], row["draw"])) != row for row in old_plan["rows"]
    ):
        raise RuntimeError("Additional draw plan conflicts with the original run")
    copied = 0
    for path in (source / "counts").rglob("*.json"):
        record = json.loads(path.read_text())
        if record["row"]["draw"] <= 100:
            raise RuntimeError(f"Temporary count would overwrite an original draw: {path}")
        target = destination / path.relative_to(source)
        exists = subprocess.run(["timeout", "-k", "2s", "8s", "test", "-e", str(target)])
        if exists.returncode not in (0, 1):
            raise RuntimeError(f"Scratch metadata lookup timed out: {target}")
        if exists.returncode == 0:
            existing = subprocess.run(["timeout", "-k", "2s", "8s", "cmp", "-s", str(path), str(target)])
            if existing.returncode:
                raise RuntimeError(f"Existing checkpoint differs or is unreadable: {target}")
        else:
            bounded_copy(path, target)
        copied += 1
    bounded_copy(source / "topup_manifest.json", destination / "topup_manifest.json")
    for directory in ("tables", "plots"):
        for path in (source / directory).glob("*"):
            if path.is_file():
                bounded_copy(path, destination / directory / path.name)
    for name in ("topup_runtime.json", "temporary_storage.json"):
        path = source / name
        if path.exists():
            bounded_copy(path, destination / name)
    for path in (source.parent.parent / "logs").glob("hod_s3_*"):
        bounded_copy(path, HERE / "logs" / path.name)
    print(f"Copied {copied} additional checkpoints and current tables/plots back to {destination}")
    print("Temporary copies retained for verification; original draw IDs 1..100 were preserved.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["stage", "restore", "migrate-home"])
    parser.add_argument("--scratch-run", type=Path, default=HERE / "runs/default")
    parser.add_argument("--temporary-run", "--home-run", dest="temporary_run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--try-restore", action="store_true", help="Keep temporary results and exit successfully if scratch remains unavailable")
    args = parser.parse_args()
    try:
        if args.mode == "migrate-home":
            migrate_home(LEGACY_HOME, args.temporary_run)
        else:
            (stage if args.mode == "stage" else restore)(args)
    except RuntimeError as error:
        if args.mode == "restore" and args.try_restore:
            print(f"Restore deferred: {error}")
            return
        raise


if __name__ == "__main__":
    main()
