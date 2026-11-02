#!/usr/bin/env python3
"""Print pilot/production bphi results; explicitly distinguish legacy selection policies."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from ensemble_results import (PARAMETERS, classify, ensemble_rows, load_plan,
                              read_pair, resolve_input)

HERE = Path(__file__).resolve().parent


def summarize(values):
    values = np.asarray(values)
    if not len(values):
        return dict(n=0, mean=None, std=None, q16=None, median=None, q84=None)
    q16, median, q84 = np.quantile(values, [0.16, 0.5, 0.84])
    return dict(n=len(values), mean=float(values.mean()), std=float(values.std()),
                q16=float(q16), median=float(median), q84=float(q84))


def write_table(path, records, fieldnames=None):
    if records or fieldnames:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames or list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def legacy_parameters(manifest, z, minimum):
    from run_counts import ROOT, file_hash
    original_root = Path(os.environ.get("HOD_SOURCE_ROOT", Path(__file__).resolve().parents[2]))
    original_path = original_root / f"LHC/bphi_AbacusHOD_LatinHyperCube_LRG_z{float(z):.3f}_N100.csv"
    path = resolve_input(original_path)
    if file_hash(path) != manifest["sources_sha256"][str(original_path)]:
        raise RuntimeError(f"Historical LHC parameter file changed: {path}")
    with path.open() as stream:
        parameters = list(csv.DictReader(stream))
    baseline = manifest["snapshots"][z]["configs"]["000"]["HOD_params"]["LRG_params"]
    records = []
    for row in manifest["published"][z]:
        if row["hod_id"] <= 0 or min(row["n000"], row["n004"]) <= minimum:
            continue
        params = dict(baseline)
        params.update({key: max(0.01, float(value)) for key, value in parameters[row["hod_id"] - 1].items()})
        records.append(dict(z=z, ensemble="existing_LHC", draw=row["hod_id"], source_row=row["hod_id"] - 1,
                            n000=row["n000"], n004=row["n004"], **{name: params[name] for name in PARAMETERS}))
    return records


def plot_parameters(parameter_records, snapshots, plot_dir):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    labels = [r"$\log_{10} M_{\rm cut}$", r"$\log_{10} M_1$", r"$\sigma$", r"$\alpha$",
              r"$\kappa$", r"$\alpha_c$", r"$\alpha_s$", r"$f_{\rm ic}$"]
    ensembles = [("chain_s1", "chain_s1", "C0"), ("chain_s3", "chain_s3", "C1"),
                 ("existing_LHC", "Existing LHC, common count cut", "C2")]
    fig, axes = plt.subplots(len(snapshots), len(PARAMETERS), figsize=(21, 8.5), squeeze=False)
    for i, z in enumerate(snapshots):
        groups = {ensemble: [row for row in parameter_records if row["z"] == z and row["ensemble"] == ensemble]
                  for ensemble, _, _ in ensembles}
        for j, (name, label) in enumerate(zip(PARAMETERS, labels)):
            ax = axes[i, j]
            joined = [row[name] for rows in groups.values() for row in rows]
            if not joined:
                continue
            bins = np.histogram_bin_edges(joined, bins=20)
            for (ensemble, _, color), linestyle in zip(ensembles, ("-", "--", ":")):
                values = [row[name] for row in groups[ensemble]]
                if not values:
                    continue
                if np.ptp(values) == 0:
                    ax.axvline(values[0], color=color, linewidth=1.5, linestyle=linestyle)
                else:
                    ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.5, color=color)
            ax.set_xlabel(label, fontsize=11)
            ax.tick_params(labelsize=9)
            if np.ptp(joined) == 0:
                ax.set_xlim(joined[0] - 0.1, joined[0] + 0.1)
                ax.set_yticks([])
                ax.text(0.04, 0.97, f"Fixed: {joined[0]:g}", transform=ax.transAxes, va="top", fontsize=9)
            if j == 0:
                ax.set_ylabel("Probability density", fontsize=10)
    fig.legend([Line2D([], [], color=color, linewidth=1.5) for _, _, color in ensembles],
               [label for _, label, _ in ensembles], loc="upper center", ncol=3, frameon=False, fontsize=11)
    fig.subplots_adjust(top=0.91, bottom=0.08, left=0.05, right=0.99, hspace=0.5, wspace=0.38)
    for i, z in enumerate(snapshots):
        counts = [sum(row["z"] == z and row["ensemble"] == ensemble for row in parameter_records)
                  for ensemble, _, _ in ensembles]
        fig.text(0.05, axes[i, 0].get_position().y1 + 0.015,
                 f"LRG{i + 1} (z={z}): chain_s1 N={counts[0]}, chain_s3 N={counts[1]}, existing LHC N={counts[2]}",
                 fontsize=10)
    for extension in ("png", "pdf"):
        fig.savefig(plot_dir / f"hod_parameter_distributions.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).resolve().parent / 'runs/default')
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    from run_counts import file_hash
    manifest_sha = file_hash(run_dir / "manifest.json")
    plan = load_plan(run_dir, manifest_sha)
    target = plan.get("target", manifest["settings"]["nsamples"])
    minimum = manifest["settings"]["ngal_min"]
    rows, summary, distributions, rejections, parameter_records = [], [], {}, [], []
    for z in manifest["snapshots"]:
        for ensemble in ("chain_s1", "chain_s3", "legacy_check"):
            selected = ensemble_rows(manifest, plan, z, ensemble)
            values, complete, physically_rejected, low_counts, count_rejected, pending = [], 0, 0, 0, 0, 0
            for row in selected:
                pair = read_pair(run_dir, row, manifest_sha)
                status, reasons = classify(row, pair, minimum)
                if status == "pending":
                    pending += 1
                    continue
                if status in ("physical_rejection", "count_rejection", "low_counts"):
                    physically_rejected += status == "physical_rejection"
                    count_rejected += status == "count_rejection"
                    low_counts += status == "low_counts"
                    params = dict(row["params"])
                    if pair[0] is not None and pair[0]["status"] == "ok":
                        params.update(pair[0]["parameters"])
                    rejections.append(dict(z=z, ensemble=ensemble, draw=row["draw"], source_row=row["source_row"],
                                           status=status, reasons="; ".join(reasons),
                                           n000=pair[0].get("ngal") if pair[0] else None,
                                           n004=pair[1].get("ngal") if pair[1] else None, ngal_min=minimum,
                                           **{name: params.get(name) for name in PARAMETERS}))
                if not all(record is not None and record["status"] == "ok" for record in pair):
                    continue
                complete += 1
                n000, n004 = [record["ngal"] for record in pair]
                valid = status == "accepted"
                bphi = float(2 * np.log(n000 / n004) / np.log(0.807952 / 0.75)) if min(n000, n004) > 0 else None
                rows.append(dict(z=z, ensemble=ensemble, draw=row["draw"], source_row=row["source_row"],
                                 n000=n000, n004=n004, ic=pair[0]["ic"], bphi=bphi, passes_count_cut=valid))
                if valid:
                    values.append(bphi)
                    if ensemble != "legacy_check":
                        parameter_records.append(dict(z=z, ensemble=ensemble, draw=row["draw"], source_row=row["source_row"],
                                                      n000=n000, n004=n004,
                                                      **{name: pair[0]["parameters"][name] for name in PARAMETERS}))
            distributions[z, ensemble] = values
            summary.append(dict(z=z, ensemble=ensemble, selection="Ngal > threshold in both cosmologies",
                                proposed=len(selected), physically_rejected=physically_rejected,
                                completed_pairs=complete, low_counts=low_counts,
                                count_rejected=count_rejected, pending=pending,
                                remaining=max(0, target - len(values)) if ensemble != "legacy_check" else 0,
                                **summarize(values)))
        published = manifest["published"][z]
        # Preserve the actual pipeline baseline, and expose a matched-selection diagnostic separately.
        for label, selected in (
            ("published_LHC_prior", published),
            ("existing_LHC_common_cut", [r for r in published if r["hod_id"] > 0 and min(r["n000"], r["n004"]) > minimum]),
        ):
            values = [row["bphi"] for row in selected]
            distributions[z, label] = values
            summary.append(dict(z=z, ensemble=label, selection="as stored (may include id0/low counts)" if label.startswith("published") else "LHC ids 1..100; common count cut",
                                proposed=len(published), physically_rejected=0, completed_pairs=len(selected),
                                low_counts=sum(min(r["n000"], r["n004"]) <= minimum for r in published),
                                count_rejected=0, pending=0, remaining=0, **summarize(values)))
        parameter_records.extend(legacy_parameters(manifest, z, minimum))
    table_dir = run_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, records in (("bphi_summary", summary), ("bphi_samples", rows)):
        write_table(table_dir / f"{name}.csv", records)
    rejection_fields = ["z", "ensemble", "draw", "source_row", "status", "reasons", "n000", "n004", "ngal_min"] + PARAMETERS
    write_table(table_dir / "hod_rejections.csv", rejections, rejection_fields)
    write_table(table_dir / "hod_parameter_samples.csv", parameter_records)
    rejection_summary = []
    for z in manifest["snapshots"]:
        rejected = [row for row in rejections if row["z"] == z and row["ensemble"] == "chain_s3"]
        rejection_summary.append(dict(z=z, ensemble="chain_s3", rejected_sets=len(rejected),
                                      low_counts=sum(row["status"] == "low_counts" for row in rejected),
                                      count_rejections=sum(row["status"] == "count_rejection" for row in rejected),
                                      **{name: sum(any(reason.startswith(f"{name}=") for reason in row["reasons"].split("; "))
                                                   for row in rejected) for name in PARAMETERS}))
    write_table(table_dir / "hod_rejection_summary.csv", rejection_summary)
    for row in summary:
        print(f"z={row['z']} {row['ensemble']}: n={row['n']} mean={row['mean']} std={row['std']} "
              f"median={row['median']} 68%=[{row['q16']}, {row['q84']}] pending={row['pending']} "
              f"physical_rejections={row['physically_rejected']} low_counts={row['low_counts']} "
              f"count_rejections={row['count_rejected']} remaining={row['remaining']}")
    for row in rejections:
        print(f"REJECT z={row['z']} {row['ensemble']} draw={row['draw']} source_row={row['source_row']}: {row['reasons']}")
    print(f"Rejected parameter sets and counts: {table_dir / 'hod_rejections.csv'}")
    print("WARNING: published_LHC_prior is the stored pipeline baseline, not necessarily identical to ids 1..100 with the common cut.")
    if not any(distributions[z, "chain_s1"] or distributions[z, "chain_s3"] for z in manifest["snapshots"]):
        print("No new valid chain pairs yet; no comparison figure generated.")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    plot_parameters(parameter_records, manifest["snapshots"], plot_dir)
    for legacy in ("published_LHC_prior", "existing_LHC_common_cut"):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
        for ax, z in zip(axes, manifest["snapshots"]):
            for ensemble, label, color in (("chain_s3", "Joint posterior, s=3", "C1"),
                                            (legacy, "Published LHC prior" if legacy.startswith("published") else "Existing LHC, common cut", "C2")):
                values = distributions[z, ensemble]
                if values:
                    ax.hist(values, bins="auto", density=True, histtype="step", color=color, label=f"{label} (N={len(values)})")
            ax.set(title=f"z={z}", xlabel=r"$b_\phi$")
            ax.legend(fontsize=7)
        axes[0].set_ylabel("Probability density")
        for extension in ("png", "pdf"):
            fig.savefig(plot_dir / f"bphi_comparison_{legacy}.{extension}", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
