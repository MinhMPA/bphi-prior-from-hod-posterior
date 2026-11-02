#!/usr/bin/env python3
"""Continue the s=3 posterior stream until each snapshot has the target valid pairs."""
import argparse
import copy
import gc
import json
import os
import signal
import time
from pathlib import Path

import numpy as np

import prepare_ensembles as prepare
import run_counts as runner
from copy_halo_inputs import require_verified
from ensemble_results import (accepted_rows, approved_runtime, classify, ensemble_rows, exclusive_run_lock, load_plan,
                              physical_violations, producer_hashes, read_pair, resolve_input)

HERE = Path(__file__).resolve().parent


class PosteriorStream:
    def __init__(self, manifest, z, previous):
        original_path = prepare.SOURCES[z]
        path = resolve_input(original_path)
        if runner.file_hash(path) != manifest["sources_sha256"][str(original_path)]:
            raise RuntimeError(f"Original posterior chain changed: {path}")
        samples, weights, _ = prepare.read_nested(path) if path.suffix == ".npz" else prepare.read_text(path)
        self.weights = weights / weights.sum()
        self.names = manifest["snapshots"][z]["parameter_names"]
        self.ndim = len(prepare.LOW_NAMES) if path.suffix == ".npz" else len(prepare.COMMON)
        self.samples = samples.copy()
        self.samples[:, self.names.index("sigma")] *= manifest["settings"]["sigma_factor"][z]
        self.mean = np.asarray(manifest["snapshots"][z]["transformed_weighted_mean"])
        if not np.allclose(self.weights @ self.samples, self.mean, rtol=1e-12, atol=1e-12):
            raise RuntimeError(f"Posterior center changed for z={z}")
        self.z = z
        index = list(prepare.SOURCES).index(z)
        self.rng = np.random.default_rng(np.random.SeedSequence([manifest["settings"]["seed"], index]))
        original = ensemble_rows(manifest, {"rows": []}, z, "chain_s3")
        source_rows = self.rng.choice(len(samples), size=len(original), replace=True, p=self.weights)
        if source_rows.tolist() != [row["source_row"] for row in original]:
            raise RuntimeError(f"Cannot reproduce the original posterior draw stream for z={z}")
        self.draw = max(row["draw"] for row in original)
        for expected in previous:
            if self.next() != expected:
                raise RuntimeError(f"Additional posterior draw stream changed: z={z} draw={expected['draw']}")

    def next(self):
        source_row = int(self.rng.choice(len(self.samples), p=self.weights))
        self.draw += 1
        values = self.mean[:self.ndim] + 3 * (self.samples[source_row, :self.ndim] - self.mean[:self.ndim])
        params = dict(zip(self.names[:self.ndim], map(float, values)))
        return dict(z=self.z, draw=self.draw, ensemble="chain_s3", source_row=source_row,
                    source_weight=float(self.weights[source_row]), params=params,
                    rejection=prepare.physical_problem(params))


def prepare_more(run_dir, manifest, plan, manifest_sha, target, zsnaps, max_proposals):
    for z in zsnaps:
        accepted = len(accepted_rows(run_dir, manifest, plan, z, "chain_s3", manifest_sha))
        pending = []
        for row in ensemble_rows(manifest, plan, z, "chain_s3"):
            status, _ = classify(row, read_pair(run_dir, row, manifest_sha), manifest["settings"]["ngal_min"])
            if status == "pending":
                pending.append(row)
        needed = max(0, target - accepted - len(pending))
        if needed:
            previous = [row for row in plan["rows"] if row["z"] == z]
            stream = PosteriorStream(manifest, z, previous)
            while needed:
                if stream.draw - manifest["settings"]["nsamples"] >= max_proposals:
                    raise RuntimeError(f"Reached {max_proposals} additional proposals for z={z}; inspect the rejection report")
                row = stream.next()
                plan["rows"].append(row)
                problems = physical_violations(row["params"])
                if problems:
                    print(f"REJECT z={z} chain_s3 draw={row['draw']} source_row={row['source_row']}: "
                          + "; ".join(problems), flush=True)
                else:
                    needed -= 1
                    pending.append(row)
        print(f"z={z}: accepted s3={accepted}/{target}; parameter-valid pending={len(pending)}", flush=True)
    plan["target"] = target
    runner.save_json(run_dir / "topup_manifest.json", plan)


def evaluate(args, manifest, manifest_sha, signature, producer, factory, compute_all, configs, rows, z):
    for cosmo in ("000", "004"):
        if runner.stop_requested(args):
            return
        pending = [row for row in rows if read_pair(args.run_dir, row, manifest_sha, signature, producer)[int(cosmo == "004")] is None]
        if not pending:
            continue
        ball = None
        for row in pending:
            if runner.stop_requested(args):
                return
            started = time.monotonic()
            fiducial = None
            if cosmo == "004":
                fiducial = read_pair(args.run_dir, row, manifest_sha, signature, producer)[0]
                if fiducial is None:
                    raise RuntimeError("c004 requires the matching c000 checkpoint")
                if fiducial["status"] != "ok":
                    value = dict(signature=signature, manifest_sha256=manifest_sha, producer_sources=producer,
                                 status="count_rejection", reason=fiducial["reason"], row=row, cosmo=cosmo)
                    runner.save_json(runner.count_path(args.run_dir, row, cosmo), value)
                    continue
            if ball is None:
                config = copy.deepcopy(configs[z, cosmo])
                config["sim_params"]["output_dir"] = str(args.run_dir / "unused_catalog_output")
                print(f"Loading z={z} c{cosmo}: {len(pending)} remaining s3 evaluations", flush=True)
                ball = factory(config["sim_params"], config["HOD_params"], config["clustering_params"])
                baseline = copy.deepcopy(ball.tracers)
                if set(baseline) != {"LRG"}:
                    raise ValueError("This test requires LRG-only configurations")
                volume = float(ball.params["Lbox"]) ** 3
            ball.tracers = copy.deepcopy(baseline)
            ball.tracers["LRG"].update(row["params"])
            complete_count = None
            if cosmo == "000":
                ball.tracers["LRG"]["ic"] = 1.0
                expected, _ = ball.compute_ngal(Nthread=args.threads)
                complete_count = float(expected["LRG"])
                if not np.isfinite(complete_count) or complete_count <= 0:
                    reason = f"c000 expected complete-sample Ngal={complete_count}: must be finite and > 0"
                    print(f"REJECT z={z} draw={row['draw']} source_row={row['source_row']}: {reason}", flush=True)
                    runner.save_json(runner.count_path(args.run_dir, row, cosmo),
                                     dict(signature=signature, manifest_sha256=manifest_sha, producer_sources=producer,
                                          status="count_rejection", reason=reason, row=row, cosmo=cosmo))
                    continue
                ic, ic_source = runner.resolve_ic(row, complete_count, volume, manifest["settings"]["reference_nbar"][z])
            else:
                ic, ic_source = fiducial["ic"], fiducial["ic_source"]
            ball.tracers["LRG"]["ic"] = ic
            print(f"z={z} c{cosmo} chain_s3 draw={row['draw']} source_row={row['source_row']} "
                  f"parameters={ball.tracers['LRG']}", flush=True)
            mock = compute_all(ball, nthread=args.threads, out=False, verbose=True)
            ngal = len(mock["LRG"]["x"])
            del mock
            value = dict(signature=signature, manifest_sha256=manifest_sha, producer_sources=producer,
                         status="ok", row=row, cosmo=cosmo, ngal=ngal, ic=ic, ic_source=ic_source,
                         volume=volume, expected_complete_c000=complete_count,
                         parameters={key: float(val) for key, val in ball.tracers["LRG"].items()},
                         elapsed_seconds=time.monotonic() - started,
                         randoms="Unchanged preprocessed AbacusHOD random arrays; no reseeding between draws")
            runner.save_json(runner.count_path(args.run_dir, row, cosmo), value)
            print(f"Saved z={z} c{cosmo} draw={row['draw']} Ngal={ngal}", flush=True)
            if ngal <= manifest["settings"]["ngal_min"]:
                print(f"REJECT count cut: z={z} c{cosmo} draw={row['draw']} Ngal={ngal}", flush=True)
        del ball
        gc.collect()


def run(args, manifest, manifest_sha, plan):
    for z in args.zsnaps:
        s1 = len(accepted_rows(args.run_dir, manifest, plan, z, "chain_s1", manifest_sha))
        print(f"z={z}: reusing {s1} accepted chain_s1 pairs", flush=True)
        if s1 < args.target:
            raise RuntimeError(f"z={z} has only {s1} accepted s1 pairs; finish the original s1 run first")
    prepare_more(args.run_dir, manifest, plan, manifest_sha, args.target, args.zsnaps, args.max_proposals)
    if args.prepare_only:
        print("Saved additional draws. No halo data loaded.")
        return
    if all(len(accepted_rows(args.run_dir, manifest, plan, z, "chain_s3", manifest_sha)) >= args.target for z in args.zsnaps):
        print("The target is already complete; no HOD evaluations required.")
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run counts on an allocated interactive/batch compute node")
    configs = {(z, cosmo): runner.load_snapshot_config(args.run_dir, manifest, z, cosmo)
               for z in args.zsnaps for cosmo in ("000", "004")}
    subsample_root = getattr(args, "subsample_root", None)
    if subsample_root is not None:
        require_verified(subsample_root, args.zsnaps)
        for config in configs.values():
            config["sim_params"]["subsample_dir"] = str(subsample_root)
    factory, compute_all, runtime = runner.load_engine()
    runtime = approved_runtime(args.run_dir, runtime)
    signature = runner.fingerprint(dict(manifest_sha256=manifest_sha, runtime=runtime, threads=args.threads))
    if args.require_approval:
        approval = json.loads((args.run_dir / "debug_approved.json").read_text())
        if approval["manifest_sha256"] != manifest_sha or approval["signature"] != signature:
            raise RuntimeError("Runtime does not match the approved original run; use its thread count and environment")
    producer = producer_hashes()
    runner.save_json(args.run_dir / "topup_runtime.json", dict(signature=signature, producer_sources=producer,
                     threads=args.threads, subsample_root=str(subsample_root) if subsample_root else None, **runtime))
    for z in args.zsnaps:
        for ensemble in ("chain_s1", "chain_s3"):
            for row in ensemble_rows(manifest, plan, z, ensemble):
                read_pair(args.run_dir, row, manifest_sha, signature, producer)
        while not runner.stop_requested(args):
            accepted = len(accepted_rows(args.run_dir, manifest, plan, z, "chain_s3", manifest_sha, signature, producer))
            if accepted >= args.target:
                print(f"COMPLETE z={z}: chain_s1 reused; chain_s3 accepted={accepted}", flush=True)
                break
            prepare_more(args.run_dir, manifest, plan, manifest_sha, args.target, [z], args.max_proposals)
            pending = []
            for row in ensemble_rows(manifest, plan, z, "chain_s3"):
                status, _ = classify(row, read_pair(args.run_dir, row, manifest_sha, signature, producer),
                                     manifest["settings"]["ngal_min"])
                if status == "pending":
                    pending.append(row)
            evaluate(args, manifest, manifest_sha, signature, producer, factory, compute_all, configs,
                     pending[:args.target - accepted], z)
        if runner.stop_requested(args):
            print("Stopped between evaluations; resubmit the same top-up command to resume.")
            return
    print("All requested s3 targets complete.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=HERE / "runs/default")
    parser.add_argument("--subsample-root", type=Path, help="Use checksum-verified CFS copies of the HOD subsamples")
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--max-proposals", type=int, default=10000, help="Maximum additional posterior proposals per snapshot")
    parser.add_argument("--zsnaps", nargs="+", choices=["0.5", "0.8", "1.1"], default=["0.5", "0.8", "1.1"])
    parser.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 64)))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--require-approval", action="store_true")
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args()
    if min(args.target, args.max_proposals, args.threads) < 1:
        parser.error("target, max-proposals, and threads must be positive")
    args.run_dir = args.run_dir.resolve()
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    manifest_sha = runner.file_hash(args.run_dir / "manifest.json")
    with exclusive_run_lock(args.run_dir):
        plan = load_plan(args.run_dir, manifest_sha)
        for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, runner.request_stop)
        run(args, manifest, manifest_sha, plan)


if __name__ == "__main__":
    main()
