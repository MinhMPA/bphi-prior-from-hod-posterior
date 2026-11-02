"""Three prior measures built from the same accepted chain_s3 count pairs."""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

KINDS = ("gaussian", "empirical", "kde")
ZS = (0.5, 0.8, 1.1)
LABELS = {0.5: "LRG1", 0.8: "LRG2", 1.1: "LRG3"}
HERE = Path(__file__).resolve().parent
SOURCE_RUN = Path(os.environ.get('BPHI_HOD_RUN', HERE.parent / 'hod_correlation_test/runs/default')).expanduser()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def freeze_json(path, value):
    path = Path(path)
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    value = json.loads(encoded)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f"Frozen settings differ: {path}. Use a new --output directory.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


class PriorModel:
    def __init__(self, values, kde_scale=1.0):
        self.values = np.asarray(values, dtype=float)
        if self.values.ndim != 1 or len(self.values) < 2 or not np.all(np.isfinite(self.values)):
            raise ValueError("Prior values must be a finite one-dimensional sample")
        self.mean = float(self.values.mean())
        self.std = float(self.values.std(ddof=0))
        if self.std <= 0 or not np.isfinite(kde_scale) or kde_scale <= 0:
            raise ValueError("Prior variance and KDE bandwidth scale must be positive")
        self.bandwidth = float(kde_scale * len(self.values)**(-0.2) * self.values.std(ddof=1))
        self.ordered_values = np.sort(self.values)

    def mapped(self, kind, selector, noise, xp=np):
        if kind not in KINDS:
            raise ValueError(f"Unknown prior: {kind}")
        if kind == "gaussian":
            return self.mean + self.std * noise
        index = xp.clip(xp.floor(selector * len(self.values)).astype(int), 0, len(self.values) - 1)
        value = xp.asarray(self.ordered_values)[index]
        return value if kind == "empirical" else value + self.bandwidth * noise

    def pdf(self, kind, x):
        x = np.asarray(x)
        if kind == "gaussian":
            return norm.pdf(x, loc=self.mean, scale=self.std)
        if kind == "kde":
            return np.mean(norm.pdf((x[..., None] - self.values) / self.bandwidth), axis=-1) / self.bandwidth
        raise ValueError("The empirical prior consists of point masses, not a continuous PDF")

    def cdf(self, kind, x):
        x = np.asarray(x)
        if kind == "gaussian":
            return ndtr((x - self.mean) / self.std)
        if kind == "kde":
            return np.mean(ndtr((x[..., None] - self.values) / self.bandwidth), axis=-1)
        if kind == "empirical":
            return np.mean(x[..., None] >= self.values, axis=-1)
        raise ValueError(f"Unknown prior: {kind}")


def prepare(output, source_run=SOURCE_RUN, kde_scale=1.0):
    source_run = Path(source_run).resolve()
    source = source_run / "tables/bphi_samples.csv"
    manifest = json.loads((source_run / "manifest.json").read_text())
    before = sha256(source)
    with source.open() as stream:
        table = list(csv.DictReader(stream))
    selected = {}
    for z in ZS:
        rows = [r for r in table if float(r["z"]) == z and r["ensemble"] == "chain_s3"
                and r["passes_count_cut"].lower() == "true"]
        rows.sort(key=lambda r: int(r["draw"]))
        if len(rows) != 100 or len({r["draw"] for r in rows}) != 100:
            raise RuntimeError(f"z={z}: expected exactly 100 distinct accepted chain_s3 draws")
        values = []
        for row in rows:
            n000, n004 = int(row["n000"]), int(row["n004"])
            if min(n000, n004) <= manifest["settings"]["ngal_min"]:
                raise RuntimeError(f"Count cut failed: {row}")
            value = float(2 * np.log(n000 / n004) / np.log(0.807952 / 0.75))
            if not np.isclose(value, float(row["bphi"]), rtol=1e-12, atol=1e-12):
                raise RuntimeError(f"Stored bphi disagrees with galaxy counts: {row}")
            values.append(value)
        model = PriorModel(values, kde_scale)
        selected[str(z)] = dict(values=values, draws=[int(r["draw"]) for r in rows],
                                source_rows=[int(r["source_row"]) for r in rows],
                                mean=model.mean, std=model.std, kde_bandwidth=model.bandwidth)
    if sha256(source) != before:
        raise RuntimeError("The HOD table changed during preparation")
    payload = dict(source_run=str(source_run), source_table_sha256=before,
                   source_manifest_sha256=sha256(source_run / "manifest.json"),
                   source_topup_manifest_sha256=sha256(source_run / "topup_manifest.json"),
                   hod_settings=manifest["settings"], kde_scale=float(kde_scale), snapshots=selected,
                   selection="100 accepted chain_s3 pairs per bin; Ngal > threshold in c000 and c004",
                   joint_prior="Independent per-bin prior measures; no cross-redshift row pairing")
    freeze_json(Path(output) / "priors.json", payload)
    return payload


def load_models(output):
    payload = json.loads((Path(output) / "priors.json").read_text())
    return {float(z): PriorModel(row["values"], payload["kde_scale"])
            for z, row in payload["snapshots"].items()}


def plot_priors(output):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    output = Path(output)
    models = load_models(output)
    plots, tables = output / "plots", output / "tables"
    plots.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.8), gridspec_kw={"height_ratios": [2, 1]}, sharex="col")
    records = []
    for col, (z, model) in enumerate(models.items()):
        lo = min(model.mean - 4 * model.std, model.values.min() - 4 * model.bandwidth)
        hi = max(model.mean + 4 * model.std, model.values.max() + 4 * model.bandwidth)
        grid = np.linspace(lo, hi, 1000)
        upper, lower = axes[:, col]
        upper.hist(model.values, bins="fd", density=True, histtype="step", color="C1", linewidth=1.6)
        for kind, color in (("gaussian", "C0"), ("kde", "C2")):
            upper.plot(grid, model.pdf(kind, grid), color=color, linewidth=1.6)
            lower.plot(grid, model.cdf(kind, grid), color=color, linewidth=1.6)
        lower.step(np.r_[lo, np.sort(model.values), hi], np.r_[0, np.arange(1, 101) / 100, 1],
                   where="post", color="C1", linewidth=1.4)
        upper.plot(model.values, np.zeros(100), "|", color="C1", alpha=0.4, clip_on=False)
        upper.set_title(LABELS[z])
        upper.set_ylim(bottom=0)
        lower.set(xlabel=r"$b_\phi$", ylim=(0, 1), xlim=(lo, hi))
        q16, median, q84 = np.quantile(model.values, [.16, .5, .84])
        records.append(dict(sample=LABELS[z], z=z, n=100, mean=model.mean, gaussian_std=model.std,
                            empirical_std=model.std, median=median, q16=q16, q84=q84,
                            kde_bandwidth=model.bandwidth, kde_std=np.hypot(model.std, model.bandwidth)))
    axes[0, 0].set_ylabel("Probability density")
    axes[1, 0].set_ylabel("Cumulative probability")
    fig.legend([Line2D([], [], color=f"C{i}") for i in range(3)],
               ["Moment-matched Gaussian", "Empirical: 100 values", "Gaussian KDE"],
               loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(top=.89, hspace=.12, wspace=.28, bottom=.1, left=.07, right=.99)
    for extension in ("png", "pdf"):
        fig.savefig(plots / f"bphi_prior_shapes.{extension}", dpi=180)
    plt.close(fig)
    with (tables / "prior_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for record in records:
        print(record, flush=True)
