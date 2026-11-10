#!/usr/bin/env python3
"""Summarize exact larger-parent-pool FiQA build ablations."""

import argparse
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from artifact_io import read_index  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", type=Path, default=HERE / "artifacts/archive/repo_exploratory/hierarchical_bucket_ablation/results")
parser.add_argument("--output", type=Path, default=HERE / "results/build_ablation")
args = parser.parse_args()
RESULTS = args.results
args.output.mkdir(parents=True, exist_ok=True)


def index_seconds(name, m, b):
    return read_index(RESULTS / name, m, b)[0]["bucket_seconds"]


def first(path):
    with path.open(newline="") as stream:
        return next(csv.DictReader(stream))


baseline_seconds = index_seconds("fiqa_tq8_full.idx", 1024, 512)
baseline_curve = RESULTS / "fiqa_m1024_curve.csv"
with baseline_curve.open(newline="") as stream:
    baseline = next(row for row in csv.DictReader(stream)
                    if row["method"] == "original" and row["probes"] == "32")

parent_files = {
    2048: "fiqa_tq8_parents128.idx",
    4096: "fiqa_tq8_parents128_b4096.idx",
    8192: "fiqa_tq8_parents128_b8192.idx",
    16384: "fiqa_tq8_parents128_b16384.idx",
    32768: "fiqa_exact_parents128_b32768.idx",
}
variants = [
    (2048, 1, "fiqa_m1024_p1"),
    (2048, 4, "fiqa_m1024_p4"),
    (4096, 1, "fiqa_m1024_p1_b4096"),
    (4096, 2, "fiqa_m1024_p2_b4096"),
    (4096, 4, "fiqa_m1024_p4_b4096"),
    (8192, 1, "fiqa_m1024_p1_b8192"),
    (8192, 2, "fiqa_m1024_p2_b8192"),
    (16384, 1, "fiqa_m1024_p1_b16384"),
    (32768, 1, "fiqa_m1024_p1_b32768"),
]
rows = []
for pool, probes, prefix in variants:
    parent_seconds = index_seconds(parent_files[pool], 128, pool)
    build = first(RESULTS / f"{prefix}_build.csv")
    with (RESULTS / f"{prefix}_curve.csv").open(newline="") as stream:
        quality = next(row for row in csv.DictReader(stream)
                       if row["method"] == "parent_pool" and row["probes"] == "32")
    child_seconds = float(build["shared_child_rescoring_seconds"])
    rows.append(dict(parent_pool=pool, parent_probes=probes,
                     parent_build_seconds=parent_seconds,
                     child_build_seconds=child_seconds,
                     total_build_seconds=parent_seconds + child_seconds,
                     build_speedup_vs_global=baseline_seconds / (parent_seconds + child_seconds),
                     candidate_recall_10=float(quality["candidate_recall_10"]),
                     mean_unique_candidates=float(quality["mean_candidates"]),
                     mean_child_bucket_overlap=float(build["mean_child_topb_overlap"]),
                     parent_method="exact" if pool == 32768 else "TQ8+exact refinement"))

summary = args.output / "fiqa_large_parent_pool_summary.csv"
with summary.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

fig, ax = plt.subplots(figsize=(7.3, 4.6))
styles = {1: ("o", "#0072B2"), 2: ("s", "#E69F00"), 4: ("D", "#009E73")}
for probes, (marker, color) in styles.items():
    points = sorted((r for r in rows if r["parent_probes"] == probes),
                    key=lambda r: r["parent_pool"])
    if not points:
        continue
    ax.plot([r["total_build_seconds"] for r in points],
            [100 * r["candidate_recall_10"] for r in points],
            marker=marker, color=color, linewidth=1.8,
            label=f"{probes} parent{'s' if probes > 1 else ''}")
    if probes == 1:
        for r in points:
            offset = (-56, 6) if r["parent_pool"] == 32768 else (5, 6)
            ax.annotate(f"B′={r['parent_pool']:,}",
                        (r["total_build_seconds"], 100 * r["candidate_recall_10"]),
                        xytext=offset, textcoords="offset points", fontsize=8)
ax.scatter([baseline_seconds], [100 * float(baseline["candidate_recall_10"])],
           color="black", marker="*", s=170, label="Global build", zorder=5)
ax.annotate("Global", (baseline_seconds, 100 * float(baseline["candidate_recall_10"])),
            xytext=(5, -14), textcoords="offset points", fontsize=9)
ax.set(xlabel="Bucket build time (s)", ylabel="Candidate Recall@10 (%)",
       title="FiQA M=1,024: larger parent pools", ylim=(88, 99))
ax.grid(alpha=.2)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(args.output / f"fiqa_large_parent_pool_tradeoff.{ext}", dpi=220)
