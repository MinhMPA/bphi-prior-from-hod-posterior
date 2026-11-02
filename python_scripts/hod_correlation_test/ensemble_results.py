"""Read original and additional draws under the same frozen HOD inputs."""
import json
import os
import socket
import time
import uuid
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np

from run_counts import count_path, file_hash

PARAMETERS = ["logM_cut", "logM1", "sigma", "alpha", "kappa", "alpha_c", "alpha_s", "ic"]


@contextmanager
def exclusive_run_lock(run_dir):
    lock = Path(run_dir) / ".run.lock.d"
    owner_path = lock / "owner.json"
    try:
        lock.mkdir()
    except FileExistsError as error:
        try:
            owner = owner_path.read_text().strip()
        except OSError:
            owner = "owner metadata unavailable"
        raise RuntimeError(f"Run already locked: {lock}; {owner}. "
                           "Clear a stale lock only after confirming its job/process has stopped.") from error
    from run_counts import save_json
    owner = dict(token=uuid.uuid4().hex, pid=os.getpid(), host=socket.gethostname(),
                 job_id=os.environ.get("SLURM_JOB_ID"),
                 created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    save_json(owner_path, owner)
    try:
        yield
    finally:
        if json.loads(owner_path.read_text())["token"] != owner["token"]:
            raise RuntimeError(f"Run lock ownership changed; leaving {lock} untouched")
        owner_path.unlink()
        lock.rmdir()


def approved_runtime(run_dir, runtime):
    mapping = os.environ.get("HOD_CODE_MAP")
    if not mapping:
        return runtime
    aliases = json.loads(Path(mapping).read_text())
    inverse = {str(Path(copied).resolve()): original for original, copied in aliases.items()}
    sources = {inverse.get(str(Path(path).resolve()), path): digest
               for path, digest in runtime["sources"].items()}
    normalized = dict(runtime, sources=sources)
    original = json.loads((run_dir / "runtime.json").read_text())
    original.pop("signature")
    if len(sources) != len(runtime["sources"]) or normalized != original:
        raise RuntimeError("Relocated runtime differs from the approved code/packages, not just source paths")
    from run_counts import save_json
    save_json(run_dir / "cfs_code_runtime.json",
              dict(actual_runtime=runtime, approved_runtime=normalized, code_map=aliases,
                   code_map_sha256=file_hash(Path(mapping)),
                   note="Only source locations normalized; every engine hash and package version matches the approved run"))
    return normalized


@lru_cache(maxsize=4)
def export_cache(run_dir):
    path = run_dir / "existing_export_pairs.json"
    if not path.exists():
        return None
    cache = json.loads(path.read_text())
    for name, digest in cache["exports_sha256"].items():
        if file_hash(run_dir / "original_exports" / name) != digest:
            raise RuntimeError(f"Saved count/parameter export changed: {name}")
    return cache


def resolve_input(path):
    mapping = os.environ.get("HOD_INPUT_MAP")
    if mapping:
        aliases = json.loads(Path(mapping).read_text())
        return Path(aliases.get(str(path), str(path)))
    return path


def physical_violations(params):
    problems = []
    for name, value in params.items():
        if not np.isfinite(value):
            problems.append(f"{name}={value}: nonfinite")
        elif name == "sigma" and value <= 0:
            problems.append(f"sigma={value:.8g}: must be > 0")
        elif name in ("alpha", "kappa", "alpha_c", "alpha_s") and value < 0:
            problems.append(f"{name}={value:.8g}: must be >= 0")
        elif name == "ic" and not 0 < value <= 1:
            problems.append(f"ic={value:.8g}: must be in (0, 1]")
    return problems


def load_plan(run_dir, manifest_sha):
    path = run_dir / "topup_manifest.json"
    if not path.exists():
        return dict(parent_manifest_sha256=manifest_sha, rows=[])
    plan = json.loads(path.read_text())
    if plan["parent_manifest_sha256"] != manifest_sha:
        raise RuntimeError(f"Additional draws refer to different original inputs: {path}")
    return plan


def ensemble_rows(manifest, plan, z, ensemble):
    rows = [row for row in manifest["rows"] + plan["rows"]
            if row["z"] == z and row["ensemble"] == ensemble]
    rows.sort(key=lambda row: row["draw"])
    if len({row["draw"] for row in rows}) != len(rows):
        raise RuntimeError(f"Duplicate draw IDs in {ensemble} z={z}")
    return rows


def read_pair(run_dir, row, manifest_sha, signature=None, producer_sources=None):
    records = []
    cache = export_cache(run_dir)
    if cache is not None and cache["parent_manifest_sha256"] != manifest_sha:
        raise RuntimeError("Saved exports refer to a different original manifest")
    for cosmo in ("000", "004"):
        path = count_path(run_dir, row, cosmo)
        key = str(path.relative_to(run_dir))
        if cache is not None and key in cache["records"]:
            record = cache["records"][key]
        else:
            if not path.exists():
                records.append(None)
                continue
            record = json.loads(path.read_text())
        if record["manifest_sha256"] != manifest_sha or record["row"] != row or record["cosmo"] != cosmo:
            raise RuntimeError(f"Checkpoint does not match the frozen draw: {path}")
        if signature is not None and record["signature"] != signature:
            raise RuntimeError(f"Checkpoint inputs/code differ: {path}")
        if producer_sources is not None and "producer_sources" in record and record["producer_sources"] != producer_sources:
            raise RuntimeError(f"Additional-count producer changed: {path}")
        records.append(record)
    if all(record is not None for record in records) and records[0]["signature"] != records[1]["signature"]:
        raise RuntimeError(f"Paired checkpoints have different runtime signatures: {row}")
    return records


def classify(row, pair, minimum):
    problems = physical_violations(row["params"])
    if problems or row["rejection"]:
        return "physical_rejection", problems or [row["rejection"]]
    for record in pair:
        if record is not None and record["status"] != "ok":
            return "count_rejection", [record.get("reason", record["status"])]
    if any(record is None for record in pair):
        return "pending", []
    problems = physical_violations(pair[0]["parameters"])
    if problems:
        return "physical_rejection", problems
    reasons = [f"c{record['cosmo']} Ngal={record['ngal']} <= {minimum}"
               for record in pair if record["ngal"] <= minimum]
    return ("low_counts", reasons) if reasons else ("accepted", [])


def accepted_rows(run_dir, manifest, plan, z, ensemble, manifest_sha, signature=None, producer_sources=None):
    accepted = []
    for row in ensemble_rows(manifest, plan, z, ensemble):
        pair = read_pair(run_dir, row, manifest_sha, signature, producer_sources)
        status, _ = classify(row, pair, manifest["settings"]["ngal_min"])
        if status == "accepted":
            accepted.append((row, pair))
    return accepted


def producer_hashes():
    from pathlib import Path
    directory = Path(__file__).resolve().parent
    return {str(directory / name): file_hash(directory / name)
            for name in ("topup_counts.py", "ensemble_results.py", "copy_halo_inputs.py")}
