#!/usr/bin/env python3
"""Plot the saved FiQA parent-pool operating curves without rerunning search."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", type=Path, default=HERE / "artifacts/archive/repo_exploratory/hierarchical_bucket_ablation/results")
parser.add_argument("--output", type=Path, default=HERE / "results/build_ablation")
args = parser.parse_args()
RESULTS = args.results
args.output.mkdir(parents=True, exist_ok=True)


def curve(prefix, method):
    with (RESULTS / f"{prefix}_curve.csv").open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["method"] == method]
    rows.sort(key=lambda row: int(row["probes"]))
    return ([float(row["mean_candidates"]) for row in rows],
            [100 * float(row["candidate_recall_10"]) for row in rows])


fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), sharey=True)
baseline_x, baseline_y = curve("fiqa_m1024", "original")
for ax in axes:
    ax.plot(baseline_x, baseline_y, color="black", linewidth=2.2,
            label="Global top-512")
    ax.set(xlabel="Mean unique document point clouds scored per query",
           xlim=(0, 28000), ylim=(30, 100))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="lower right", fontsize=8)

single_parent = [
    (2048, "fiqa_m1024_p1"),
    (4096, "fiqa_m1024_p1_b4096"),
    (8192, "fiqa_m1024_p1_b8192"),
    (16384, "fiqa_m1024_p1_b16384"),
    (32768, "fiqa_m1024_p1_b32768"),
]
colors = ["#D55E00", "#E69F00", "#009E73", "#0072B2", "#CC79A7"]
for (pool, prefix), color in zip(single_parent, colors):
    x, y = curve(prefix, "parent_pool")
    axes[0].plot(x, y, marker="o", markersize=3, linewidth=1.5,
                 color=color, label=f"1 parent, B′={pool:,}")
axes[0].set_title("One parent: varying pool size B′")
axes[0].set_ylabel("Exact top-10 candidate recall (%)")
axes[0].legend(frameon=False, loc="lower right", fontsize=8)

multi_parent = [
    (2048, 2, "fiqa_m1024", "#E69F00", "--"),
    (2048, 4, "fiqa_m1024_p4", "#E69F00", ":"),
    (4096, 2, "fiqa_m1024_p2_b4096", "#009E73", "--"),
    (4096, 4, "fiqa_m1024_p4_b4096", "#009E73", ":"),
    (8192, 2, "fiqa_m1024_p2_b8192", "#0072B2", "--"),
]
for pool, parents, prefix, color, style in multi_parent:
    x, y = curve(prefix, "parent_pool")
    if prefix == "fiqa_m1024":
        # This file has several parent-pool configurations; select the 2-parent row.
        with (RESULTS / f"{prefix}_curve.csv").open(newline="") as stream:
            rows = [r for r in csv.DictReader(stream)
                    if r["method"] == "parent_pool"
                    and r["parent_pool"] == str(pool)
                    and r["parent_probes"] == str(parents)
                    and r["parents"] == "128"]
        rows.sort(key=lambda r: int(r["probes"]))
        x = [float(r["mean_candidates"]) for r in rows]
        y = [100 * float(r["candidate_recall_10"]) for r in rows]
    axes[1].plot(x, y, marker="o", markersize=3, linewidth=1.5,
                 color=color, linestyle=style,
                 label=f"{parents} parents, B′={pool:,}")
axes[1].set_title("Multiple parents: selected settings")
axes[1].legend(frameon=False, loc="lower right", fontsize=8)

fig.suptitle("FiQA dev · M=1,024 · B=512 · exact EMD routing")
fig.tight_layout()
for suffix in ("png", "pdf"):
    fig.savefig(args.output / f"fiqa_parent_pool_full_candidate_curves.{suffix}", dpi=220)
