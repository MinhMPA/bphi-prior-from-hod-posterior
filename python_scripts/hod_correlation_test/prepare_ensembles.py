#!/usr/bin/env python3
"""Prepare paired posterior draws without loading halo catalogues or submitting jobs."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("HOD_SOURCE_ROOT", HERE.parent.parent))
sys.path.insert(0, str(HERE.parent))
from extract_hod_covariances import COMMON, LOW_NAMES, read_nested, read_text, sha256

SOURCES = {
    "0.5": ROOT / "archive/chains_LRGQSO/LRG_main_z0406_chains_xi2d.npz",
    "0.8": ROOT / "archive/chains_LRGQSO/LRG_main_z0608_chains_xi2d.npz",
    "1.1": ROOT / "archive/chains_LRGQSO/LRGzgt08Chains/lrg_z09511.txt",
}


def physical_problem(params):
    if not all(np.isfinite(value) for value in params.values()):
        return "nonfinite parameter"
    for name in ("sigma",):
        if params[name] <= 0:
            return f"{name} <= 0"
    for name in ("alpha", "kappa", "alpha_c", "alpha_s"):
        if name in params and params[name] < 0:
            return f"{name} < 0"
    if "ic" in params and not 0 < params["ic"] <= 1:
        return "ic outside (0,1]"
    return ""


def baseline_rows(z, path):
    counts, result = {}, {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        values = line.split()
        if values[0] != "LRG" or not np.isclose(float(values[1]), float(z)):
            continue
        if values[2] in ("000", "004"):
            counts.setdefault(int(values[3]), {})[values[2]] = float(values[4])
        else:
            hod_id = int(values[2])
            result[hod_id] = dict(hod_id=hod_id, n000=float(values[3]), n004=float(values[4]), bphi=float(values[5]))
    for hod_id, pair in counts.items():
        if set(pair) != {"000", "004"}:
            continue
        result[hod_id] = dict(hod_id=hod_id, n000=pair["000"], n004=pair["004"],
                              bphi=float(2 * np.log(pair["000"] / pair["004"]) / np.log(0.807952 / 0.75)))
    return [result[key] for key in sorted(result)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=HERE / "settings.json")
    parser.add_argument("--config-dir", type=Path, default=HERE / "configs")
    parser.add_argument("--run-dir", type=Path, default=HERE / "runs/default")
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    if not isinstance(settings["nsamples"], int) or settings["nsamples"] < 1:
        raise ValueError("nsamples must be a positive integer")
    rows, snapshots, hashes, published = [], {}, {}, {}
    for index, (z, path) in enumerate(SOURCES.items()):
        factor, nbar = settings["sigma_factor"][z], settings["reference_nbar"][z]
        if not np.isfinite(factor) or factor <= 0 or not np.isfinite(nbar) or nbar <= 0:
            raise ValueError(f"Set positive sigma_factor and reference_nbar for z={z}")
        hashes[str(path)] = sha256(path)
        samples, weights, info = read_nested(path) if path.suffix == ".npz" else read_text(path)
        weights = weights / weights.sum()
        if path.suffix == ".txt":
            for suffix in (".paramnames", ".properties.ini", ".ranges"):
                hashes[str(path.with_suffix(suffix))] = sha256(path.with_suffix(suffix))
        names = LOW_NAMES if path.suffix == ".npz" else COMMON + ["Ac"]
        native_mean = weights @ samples
        converted = samples.copy()
        converted[:, names.index("sigma")] *= factor
        mean = weights @ converted
        rng = np.random.default_rng(np.random.SeedSequence([settings["seed"], index]))
        source_rows = rng.choice(len(samples), size=settings["nsamples"], replace=True, p=weights)
        ndim = len(LOW_NAMES) if path.suffix == ".npz" else len(COMMON)
        for draw, source_row in enumerate(source_rows, start=1):
            for scale in (1, 3):
                values = mean[:ndim] + scale * (converted[source_row, :ndim] - mean[:ndim])
                params = dict(zip(names[:ndim], map(float, values)))
                if z == "1.1" and scale == 1:
                    params["ic"] = float(samples[source_row, -1])
                rows.append(dict(z=z, draw=draw, ensemble=f"chain_s{scale}", source_row=int(source_row),
                                 source_weight=float(weights[source_row]), params=params,
                                 rejection=physical_problem(params)))
        snapshots[z] = dict(parameter_names=names, native_weighted_mean=native_mean.tolist(),
                            transformed_weighted_mean=mean.tolist(), source_metadata=info, configs={})
        for cosmo in ("000", "004"):
            config_path = args.config_dir / f"abacushod_c{cosmo}_ph000_z{float(z):.3f}.yaml"
            hashes[str(config_path)] = sha256(config_path)
            config = yaml.safe_load(config_path.read_text())
            if config["sim_params"]["sim_name"] != f"AbacusSummit_base_c{cosmo}_ph000" or not np.isclose(config["sim_params"]["z_mock"], float(z)):
                raise ValueError(f"Snapshot/cosmology mismatch in {config_path}")
            snapshots[z]["configs"][cosmo] = config
        lhc_path = ROOT / f"LHC/bphi_AbacusHOD_LatinHyperCube_LRG_z{float(z):.3f}_N100.csv"
        hashes[str(lhc_path)] = sha256(lhc_path)
        with lhc_path.open() as stream:
            legacy = list(csv.DictReader(stream))
        # Keep a few exact historical LHC points for interactive reproducibility checks.
        for hod_id in (1, 2, 3):
            original = {key: float(value) for key, value in legacy[hod_id - 1].items()}
            params = {key: max(0.01, value) for key, value in original.items()}
            rows.append(dict(z=z, draw=hod_id, ensemble="legacy_check", source_row=hod_id - 1,
                             params=params, original_params=original, rejection=""))
        baseline = ROOT / "backup" / ("bphi_AbacusHOD_z1.1_updated.txt" if z == "1.1" else "bphi_AbacusHOD_LRG.txt")
        hashes[str(baseline)] = sha256(baseline)
        published[z] = baseline_rows(z, baseline)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise RuntimeError(f"Input changed while preparing: {path}")
    content = dict(settings=settings, sources_sha256=hashes, snapshots=snapshots, rows=rows, published=published)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    target = args.run_dir / "manifest.json"
    if target.exists():
        if json.loads(target.read_text()) != content:
            raise RuntimeError("Prepared inputs differ. Use a NEW --run-dir; existing results will not be overwritten.")
        print(f"Reusing unchanged {target}")
    else:
        target.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
        print(f"Prepared {target}")
    config_dir = args.run_dir / "configs"
    config_dir.mkdir(exist_ok=True)
    for z, snapshot in snapshots.items():
        for cosmo, config in snapshot["configs"].items():
            path = config_dir / f"abacushod_c{cosmo}_ph000_z{float(z):.3f}.yaml"
            if path.exists():
                if yaml.safe_load(path.read_text()) != config:
                    raise RuntimeError(f"Changed frozen configuration: {path}. Use a new --run-dir.")
            else:
                path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Six explicit snapshot/cosmology YAMLs: {config_dir}")
    for z in SOURCES:
        for ensemble in ("chain_s1", "chain_s3"):
            selected = [row for row in rows if row["z"] == z and row["ensemble"] == ensemble]
            print(f"z={z} {ensemble}: {len(selected)} draws; {sum(bool(row['rejection']) for row in selected)} rejected before HOD")
    print("Review before production:")
    for note in settings["review_before_production"]:
        print("  " + note)


if __name__ == "__main__":
    main()
