#!/usr/bin/env python3
"""Extract weighted HOD posterior covariances from the trusted local archive."""
import argparse
import csv
import hashlib
import json
import pickle
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent / "archive" / "chains_LRGQSO"
COMMON = ["logM_cut", "logM1", "sigma", "alpha", "kappa"]
LOW_NAMES = COMMON + ["alpha_c", "alpha_s"]
ALIASES = {"lgmc": "logM_cut", "lgm1": "logM1"}
LABELS = [r"$\log M_{\rm cut}$", r"$\log M_1$", r"$\sigma$", r"$\alpha$", r"$\kappa$"]


class LegacyResultsReader(pickle.Unpickler):
    def find_class(self, module, name):
        # Old Dynesty Results was a dict; current versions cannot unpickle it.
        if (module, name) == ("dynesty.results", "Results"):
            return dict
        return super().find_class(module, name)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_nested(path):
    with zipfile.ZipFile(path) as archive, archive.open("res.npy") as stream:
        version = np.lib.format.read_magic(stream)
        readers = {(1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0}
        shape, _, dtype = readers[version](stream)
        if shape != () or not dtype.hasobject:
            raise ValueError(f"Unexpected nested result format: {path}")
        result = LegacyResultsReader(stream).load().item()
    samples = np.asarray(result["samples"], dtype=float)
    logw = np.asarray(result["logwt"], dtype=float)
    if np.any(np.isnan(logw)) or np.any(np.isposinf(logw)) or not np.any(np.isfinite(logw)):
        raise ValueError(f"Invalid log weights: {path}")
    weights = np.exp(logw - np.max(logw))
    log_weight_sum = np.max(logw) + np.log(weights.sum())
    return samples, weights, {
        "format": "Dynesty nested sampling",
        "native_parameters": LOW_NAMES,
        "parameter_order_source": "Inferred from local HOD_LHC.py order and Yuan et al. Table 3; no names embedded in NPZ",
        "parameter_order_reference": "https://academic.oup.com/view-large/446843089",
        "nlive": int(result["nlive"]), "niter": int(result["niter"]),
        "burnin": "None: use all nested-sampling posterior weights, including final live points",
        "logsumexp_logwt_minus_final_logz": float(log_weight_sum - result["logz"][-1]),
        "derived_parameters": [],
    }


def read_text(path):
    entries = [line.split()[0] for line in path.with_suffix(".paramnames").read_text().splitlines() if line.strip()]
    native = [name.rstrip("*") for name in entries]
    names = [ALIASES.get(name, name) for name in native]
    properties = dict(line.split("=", 1) for line in path.with_suffix(".properties.ini").read_text().splitlines() if "=" in line)
    if properties.get("burn_removed", "").strip().upper() != "T":
        raise ValueError("The supplied high-z chain must already have burn-in removed")
    table = np.loadtxt(path)
    if table.shape[1] != len(names) + 2:
        raise ValueError("Text columns must be weight, negative log likelihood, then named parameters")
    order = [names.index(name) for name in COMMON + ["Ac"]]
    return table[:, 2:][:, order], table[:, 0], {
        "format": "GetDist-style weighted text",
        "native_parameters": native,
        "native_column_order": ["weight", "negative_log_likelihood"] + native,
        "parameter_order_source": str(path.with_suffix(".paramnames")),
        "burnin": "Already removed (burn_removed=T); no additional cut",
        "derived_parameters": [ALIASES.get(name.rstrip("*"), name.rstrip("*")) for name in entries if name.endswith("*")],
        "original_weight_sum": float(table[:, 0].sum()),
    }


def posterior_moments(samples, weights):
    if weights.shape != (len(samples),) or not np.all(np.isfinite(samples)):
        raise ValueError("Nonfinite samples or inconsistent sample/weight dimensions")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Posterior weights must be finite, nonnegative, and have positive sum")
    weights = weights / weights.sum()
    mean = weights @ samples
    centered = samples - mean
    cov = centered.T @ (weights[:, None] * centered)
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    np.testing.assert_allclose(cov, np.cov(samples, rowvar=False, aweights=weights, ddof=0), rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(cov, cov.T, atol=1e-14)
    np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-14)
    if np.linalg.eigvalsh(cov).min() <= 0:
        raise ValueError("Covariance is not positive definite")
    quantiles = []
    for values in samples.T:
        order = np.argsort(values[weights > 0])
        x, w = values[weights > 0][order], weights[weights > 0][order]
        quantiles.append(np.interp([0.16, 0.5, 0.84], np.cumsum(w) - w / 2, x))
    return mean, std, cov, corr, np.asarray(quantiles), weights


def write_matrix(path, names, matrix):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["parameter"] + names)
        writer.writerows([name] + [f"{x:.18e}" for x in row] for name, row in zip(names, matrix))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, default=HERE / "hod_parameter_covariances")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {
        "z0406": args.archive / "LRG_main_z0406_chains_xi2d.npz",
        "z0608": args.archive / "LRG_main_z0608_chains_xi2d.npz",
        "z09511": args.archive / "LRGzgt08Chains" / "lrg_z09511.txt",
    }
    inputs = list(sources.values()) + [sources["z09511"].with_suffix(suffix) for suffix in (".paramnames", ".properties.ini", ".ranges")]
    before = {str(path): sha256(path) for path in inputs}
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "ddof": 0,
                "covariance_definition": "sum(normalized_w * (theta - mean) (theta - mean)^T)",
                "quantile_definition": "Linear interpolation on midpoint weighted CDF",
                "note": "Posterior covariance, not covariance of the mean; no extra burn-in, clipping, resampling or thinning",
                "bins": {}}
    arrays, stats, notes = {}, [], []
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8), layout="constrained")
    for ax, (redshift, source) in zip(axes, sources.items()):
        samples, weights, info = read_nested(source) if source.suffix == ".npz" else read_text(source)
        names = LOW_NAMES if source.suffix == ".npz" else COMMON + ["Ac"]
        if samples.shape[1] != len(names):
            raise ValueError(f"Unexpected number of parameters for {redshift}")
        mean, std, cov, corr, quantiles, weights = posterior_moments(samples, weights)
        info.update(source=str(source), parameters=names, nrows=len(samples),
                    positive_weight_rows=int(np.count_nonzero(weights)),
                    weight_concentration_neff=float(1 / (weights @ weights)),
                    weight_concentration_note="Not an autocorrelation-corrected MCMC effective sample size",
                    smallest_covariance_eigenvalue=float(np.linalg.eigvalsh(cov).min()))
        metadata["bins"][redshift] = info
        for key, value in dict(parameters=names, mean=mean, std=std, covariance=cov,
                               correlation=corr, quantiles=quantiles,
                               common5_covariance=cov[:5, :5], common5_correlation=corr[:5, :5]).items():
            arrays[f"{redshift}_{key}"] = np.asarray(value)
        for key, matrix in (("covariance", cov), ("correlation", corr)):
            write_matrix(args.output / f"{redshift}_{key}.csv", names, matrix)
            write_matrix(args.output / f"{redshift}_common5_{key}.csv", COMMON, matrix[:5, :5])
        for name, mu, sig, (q16, med, q84) in zip(names, mean, std, quantiles):
            stats.append(dict(redshift=redshift, parameter=name, derived=name in info["derived_parameters"],
                              mean=mu, std=sig, q16=q16, median=med, q84=q84))
        i, j = np.triu_indices(len(names), k=1)
        strongest = np.argsort(np.abs(corr[i, j]))[-3:][::-1]
        notes.append(f"{redshift}: {len(samples)} stored rows; {len(names)} parameters")
        notes.append("  order: " + ", ".join(names))
        notes.append("  mean: " + np.array2string(mean, precision=8, max_line_width=150))
        notes.append("  std:  " + np.array2string(std, precision=8, max_line_width=150))
        for index in strongest:
            notes.append(f"  corr({names[i[index]]}, {names[j[index]]}) = {corr[i[index], j[index]]:+.6f}")
        labels = LABELS + ([r"$\alpha_c$", r"$\alpha_s$"] if len(names) == 7 else [r"$A_c$"])
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(names)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(names)), labels)
        ax.set_title(f"{redshift}: weighted posterior correlation", fontsize=11)
        for row in range(len(names)):
            for col in range(len(names)):
                ax.text(col, row, f"{corr[row, col]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(corr[row, col]) > 0.65 else "black")
    fig.colorbar(im, ax=axes, shrink=0.65, label="Correlation coefficient")
    fig.savefig(args.output / "hod_correlations.png", dpi=180)
    fig.savefig(args.output / "hod_correlations.pdf")
    plt.close(fig)
    after = {str(path): sha256(path) for path in inputs}
    if before != after:
        raise RuntimeError("An input file changed during extraction; rerun for a stable snapshot")
    metadata.update(inputs_unchanged=True, input_sha256=before)
    arrays["common5_parameters"] = np.asarray(COMMON)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez(args.output / "hod_covariances.npz", **arrays)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (args.output / "parameter_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)
    notes += ["", "All input hashes unchanged. Covariances match np.cov(..., aweights=w, ddof=0).",
              "All three matrices are symmetric and positive definite.",
              "Low-z parameter labels are inferred; the NPZ files contain no embedded names.",
              "Ac is a derived amplitude, not alpha_c. No missing parameters or cross-redshift covariances were invented."]
    (args.output / "extraction_summary.txt").write_text("\n".join(notes) + "\n")
    print("\n".join(notes))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
