"""Plot document-index candidate quality for saved FiQA EMDIVF centers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parents[1]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def series(data: list[dict[str, str]], key: str, scored: str, fraction: str):
    groups: dict[int, list[dict[str, str]]] = {}
    for row in data:
        groups.setdefault(int(row[key]), []).append(row)
    settings, x, y, lo, hi = [], [], [], [], []
    for setting, group in sorted(groups.items()):
        values = np.asarray([float(r[fraction]) for r in group])
        settings.append(setting)
        x.append(np.mean([float(r[scored]) for r in group]))
        y.append(values.mean())
        lo.append(np.quantile(values, .1))
        hi.append(np.quantile(values, .9))
    return np.asarray(settings), np.asarray(x), np.asarray(y), np.asarray(lo), np.asarray(hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "artifacts/archive/repo_exploratory/emdivf_doc_ann_pilot")
    parser.add_argument("--output", type=Path, default=HERE / "results/doc_ann_pilot")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    vamana = series(rows(args.results / "fiqa_vamana_bucket_pilot.csv"), "beam",
                    "exact_chamfer_cmps", "topb_found_fraction")
    mvivf = series(rows(args.results / "fiqa_mvivf_bucket_pilot.csv"), "nprobes",
                   "unique_candidates", "topb_selected_fraction")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for values, label, color, marker in ((vamana, "MV-Vamana document graph", "#0072B2", "o"),
                                          (mvivf, "MVIVF document leaves", "#D55E00", "s")):
        settings, x, y, lo, hi = values
        ax.plot(x, y, marker=marker, color=color, linewidth=2, label=label)
        ax.fill_between(x, lo, hi, color=color, alpha=.13, linewidth=0)
        for setting, xp, yp in zip(settings, x, y):
            ax.annotate(str(setting), (xp, yp), xytext=(4, -10),
                        textcoords="offset points", color=color, fontsize=8)
    ax.set(xlabel="Mean exact document Chamfer scores per prototype",
           ylabel="Exact top-512 bucket recovered",
           xlim=(0, 43000), ylim=(0, 1.02),
           title="FiQA: reuse document indexes to build EMDIVF buckets")
    ax.grid(alpha=.2)
    ax.legend(loc="lower right", frameon=False)
    fig.text(.5, .01,
             "32 EMDIVF centers; labels show beam L or MVIVF leaf probes; bands span p10–p90 centers",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(args.output / f"fiqa_bucket_generator_quality.{ext}", dpi=200)


if __name__ == "__main__":
    main()
