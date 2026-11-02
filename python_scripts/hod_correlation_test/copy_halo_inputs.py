#!/usr/bin/env python3
"""Copy the six preprocessed HOD input directories to CFS without changing originals."""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

SOURCE = Path(os.environ.get("BPHI_HALO_SOURCE", "./abacus_hod_subsamples")).expanduser()
DESTINATION = Path(os.environ.get("BPHI_HALO_DEST", "./abacus_hod_copies")).expanduser()
ZSNAPS = ("0.800", "0.500", "1.100")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2 ** 21), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def inspect_pair(source, destination, copy):
    before = source.stat()
    if copy:
        checksum = hashlib.sha256()
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            for chunk in iter(lambda: incoming.read(2 ** 21), b""):
                outgoing.write(chunk)
                checksum.update(chunk)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        source_digest = checksum.hexdigest()
        os.chmod(destination, before.st_mode & 0o777)
        os.utime(destination, ns=(before.st_atime_ns, before.st_mtime_ns))
    else:
        source_digest = digest(source)
    destination_digest = digest(destination)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed during transfer: {source}")
    if source_digest != destination_digest or destination.stat().st_size != before.st_size:
        raise RuntimeError(f"Checksum/size mismatch: {source}")
    return dict(source=str(source), bytes=before.st_size, source_mtime_ns=before.st_mtime_ns,
                sha256=source_digest, verified_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def run_worker(mode, source, temporary=None, checksum=None, seconds=180):
    command = ["timeout", "-k", "5s", f"{seconds}s", sys.executable, str(Path(__file__).resolve()),
               "--worker", mode, "--source", str(source)]
    if temporary is not None:
        command.extend(["--temporary", str(temporary)])
    if checksum is not None:
        command.extend(["--sha256", checksum])
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    try:
        stdout, stderr = process.communicate(timeout=seconds + 10)
    except subprocess.TimeoutExpired:
        # A blocked descendant may retain the pipes after GNU timeout exits.
        if process.poll() is None:
            process.kill()
        process.stdout.close()
        process.stderr.close()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        raise RuntimeError(f"{mode} timed out waiting for worker output: {source}")
    if process.returncode:
        detail = stderr.strip()[-800:] or "read/copy timed out"
        raise RuntimeError(f"{mode} failed (exit {process.returncode}): {source}: {detail}")
    return json.loads(stdout)


def prepare(source_root, destination):
    path = destination / "transfer_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["source_root"] != str(source_root):
            raise RuntimeError("Existing inventory uses a different source root")
        return manifest
    groups = []
    for z in ZSNAPS:
        for cosmo in ("000", "004"):
            relative = f"AbacusSummit_base_c{cosmo}_ph000/z{z}"
            names = run_worker("list", source_root / relative, seconds=30)
            if len(names) != 102 or any(Path(name).name != name or not name.endswith(".h5") for name in names):
                raise RuntimeError(f"Expected 102 HDF5 subsamples in {relative}; got {len(names)}")
            groups.append(dict(directory=relative, filenames=names))
    manifest = dict(source_root=str(source_root), groups=groups,
                    note="Preprocessed halos, particles and environments only; no full simulation catalogue copy")
    save_json(path, manifest)
    return manifest


def copy_one(source_root, destination, relative, seconds):
    source, target = source_root / relative, destination / relative
    record_path = destination / "verification" / relative.with_suffix(".json")
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record["source"] != str(source):
            raise RuntimeError(f"Recorded source differs: {target}")
        run_worker("verify", target, checksum=record["sha256"], seconds=seconds)
        return record, "reused"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        # Recover an interrupted rename/checkpoint without overwriting an unverified file.
        record = run_worker("compare", source, temporary=target, seconds=seconds)
    else:
        temporary = target.parent / ".partial" / (target.name + "." + uuid.uuid4().hex)
        temporary.parent.mkdir(exist_ok=True)
        record = run_worker("copy", source, temporary=temporary, seconds=seconds)
        temporary.replace(target)
    save_json(record_path, record)
    return record, "copied"


def require_verified(destination, zsnaps):
    destination = Path(destination).resolve()
    manifest = json.loads((destination / "transfer_manifest.json").read_text())
    by_directory = {group["directory"]: group["filenames"] for group in manifest["groups"]}
    for z in zsnaps:
        for cosmo in ("000", "004"):
            directory = f"AbacusSummit_base_c{cosmo}_ph000/z{float(z):.3f}"
            names = by_directory.get(directory, [])
            if len(names) != 102:
                raise RuntimeError(f"Missing complete input inventory: {directory}")
            for name in names:
                relative = Path(directory) / name
                record_path = destination / "verification" / relative.with_suffix(".json")
                if not record_path.is_file():
                    raise RuntimeError(f"CFS input not yet checksum-verified: {relative}")
                record = json.loads(record_path.read_text())
                stat = (destination / relative).stat()
                if len(record.get("sha256", "")) != 64 or record["source"] != str(Path(manifest["source_root"]) / relative) or (
                    stat.st_size, stat.st_mtime_ns
                ) != (record["bytes"], record["source_mtime_ns"]):
                    raise RuntimeError(f"CFS input changed since verification: {relative}")
            print(f"Verified CFS input records: z={z} c{cosmo}, {len(names)} files", flush=True)


def transfer(manifest, destination, seconds, limit=None, skip_files=()):
    source_root = Path(manifest["source_root"])
    skip_files = set(skip_files)
    status = dict(state="running", groups=[], failures=[], skipped=[], newly_copied=0, reused=0)
    attempted = 0
    for group in manifest["groups"]:
        summary = dict(directory=group["directory"], expected=len(group["filenames"]), verified=0, bytes=0)
        status["groups"].append(summary)
        for name in group["filenames"]:
            if limit is not None and attempted >= limit:
                status["state"] = "limited"
                save_json(destination / "transfer_status.json", status)
                return status
            relative = Path(group["directory"]) / name
            if str(relative) in skip_files:
                status["skipped"].append(str(relative))
                print(f"SKIPPED {relative}: temporarily excluded; input remains incomplete", flush=True)
                attempted += 1
                save_json(destination / "transfer_status.json", status)
                continue
            print(f"COPY/CHECK {relative}", flush=True)
            try:
                record, outcome = copy_one(source_root, destination, relative, seconds)
                summary["verified"] += 1
                summary["bytes"] += record["bytes"]
                status["newly_copied" if outcome == "copied" else "reused"] += 1
                print(f"{outcome.upper()} {relative}: {record['bytes']} bytes, SHA256={record['sha256']}", flush=True)
            except (RuntimeError, OSError) as error:
                status["failures"].append(dict(file=str(relative), error=str(error)))
                print(f"FAILED {relative}: {error}", flush=True)
            attempted += 1
            save_json(destination / "transfer_status.json", status)
        print(f"GROUP {group['directory']}: {summary['verified']}/{summary['expected']} verified", flush=True)
    status["state"] = "partial" if status["failures"] or status["skipped"] else "complete"
    save_json(destination / "transfer_status.json", status)
    print(f"TRANSFER {status['state']}: copied={status['newly_copied']}, reused={status['reused']}, "
          f"failed={len(status['failures'])}, skipped={len(status['skipped'])}; scratch originals retained", flush=True)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--timeout", type=int, default=180, help="Time limit per file, including checksum verification")
    parser.add_argument("--limit", type=int, help="Optional small transfer pilot")
    parser.add_argument("--skip-file", action="append", default=[],
                        help="Temporarily skip an input path relative to the source root; repeatable")
    parser.add_argument("--worker", choices=["list", "copy", "compare", "verify"], help=argparse.SUPPRESS)
    parser.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--temporary", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.worker == "list":
            value = sorted(path.name for path in args.source.glob("*.h5"))
        elif args.worker == "verify":
            actual = digest(args.source)
            if actual != args.sha256:
                raise RuntimeError(f"Previously verified CFS file changed: {args.source}")
            value = dict(sha256=actual)
        else:
            value = inspect_pair(args.source, args.temporary, copy=args.worker == "copy")
        print(json.dumps(value))
        return
    if args.timeout <= 0 or args.limit is not None and args.limit <= 0:
        parser.error("timeout and limit must be positive")
    destination = args.destination.resolve()
    if destination == args.source_root.resolve() or args.source_root.resolve() in destination.parents:
        parser.error("Destination must not be inside the source tree")
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".copy.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = prepare(args.source_root, destination)
        inventory = {str(Path(group["directory"]) / name)
                     for group in manifest["groups"] for name in group["filenames"]}
        skipped = [str(Path(name)) for name in args.skip_file]
        if set(skipped) - inventory:
            parser.error("--skip-file must name an input in transfer_manifest.json")
        if args.prepare_only:
            print(f"Prepared {sum(len(g['filenames']) for g in manifest['groups'])} input files in {destination}")
            return
        status = transfer(manifest, destination, args.timeout, args.limit, skip_files=skipped)
    if status["state"] == "partial":
        sys.exit(1)


if __name__ == "__main__":
    main()
