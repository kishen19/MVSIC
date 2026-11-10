#!/usr/bin/env python3
"""Regenerate the main EMDIVF comparisons using saved CSVs only."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent


def read(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved results: {path}. Set --artifacts to the local artifact directory.")
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def frontier(points):
    best = float("-inf")
    result = []
    for x, y in sorted(set(points), key=lambda point: (point[0], -point[1])):
        if y > best:
            result.append((x, y))
            best = y
    return result


def save(fig, output, name):
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=220)
    plt.close(fig)


def fiqa(artifacts, output, minimum_recall, include_copied):
    dynamic = artifacts / "results/fiqa_dynamic_scan"
    audit = artifacts / "results/fiqa_quantization/kernel_audit"
    series = []
    for policy, label, color in (
        ("direct512", "EMDIVF M=2048, B=512", "#0072B2"),
        ("adaptive75", "EMDIVF, copied buckets", "#56B4E9"),
    ):
        if policy == "adaptive75" and not include_copied:
            continue
        rows = [r for prefix in ("test", "test_128", "test_high")
                for r in read(dynamic / f"{prefix}_{policy}.csv")]
        series.append(([(float(r["total_ms"]), float(r["recall_10"])) for r in rows],
                       label, color, "-", "o"))
    mvivf = [r for path in sorted(audit.glob("numa0_mvivf_nr*.csv")) for r in read(path)]
    if not mvivf:
        raise FileNotFoundError(f"No MVIVF measurements found in {audit}")
    series.append(([(1000 / float(r["QPS_seq"]), float(r["recall_k_k"])) for r in mvivf
                    if float(r["QPS_seq"]) > 0], "MVIVF, max leaf size 500", "#CC79A7", "--", "s"))
    for subdir, reranks, label, color, style, marker in (
        ("vamana_corrected", (0, 20, 40), "MV-Vamana (TQ4)", "black", "-.", "D"),
        ("vamana_tq1", (20, 40), "MV-Vamana (TQ1)", "#D55E00", ":", "v"),
    ):
        rows = [r for nr in reranks for r in read(audit / subdir / f"numa0_vamana_nr{nr}.csv")]
        series.append(([(1000 / float(r["QPS_seq"]), float(r["recall_k_k"])) for r in rows
                        if float(r["QPS_seq"]) > 0], label, color, style, marker))
    fig, ax = plt.subplots(figsize=(7.8, 5))
    for points, label, color, style, marker in series:
        points = frontier(points)
        ax.plot(*zip(*points), color=color, linestyle=style, marker=marker,
                linewidth=1.8, markersize=4, label=label)
    ax.set(xlabel="Mean single-query latency (ms)", ylabel="Recall@10",
           xscale="log", ylim=(minimum_recall, 1.005), title="FiQA: quantized retrieval")
    ax.grid(alpha=.22)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, output, "fiqa_latency_recall")


def hotpotqa(artifacts, output, minimum_recall):
    groups = defaultdict(list)
    path = artifacts / "results/batch_overnight/point_cloud_score_comparison.csv"
    for row in read(path):
        groups[row["method"]].append((float(row["point_clouds_scored"]), float(row["recall_10"])))
    fig, ax = plt.subplots(figsize=(7.8, 5))
    palette = plt.get_cmap("tab10")
    for i, (method, points) in enumerate(groups.items()):
        points = frontier(points)
        style = "-" if "EMDIVF" in method else "--" if "MVIVF" in method else "-."
        ax.plot(*zip(*points), marker="o", markersize=4, linewidth=1.8,
                linestyle=style, color=palette(i % 10), label=method)
    ax.set(xlabel="# document point clouds scored", ylabel="Recall@10",
           xscale="log", ylim=(minimum_recall, 1.005), title="HotpotQA: document-scoring work")
    ax.grid(alpha=.22)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, output, "hotpotqa_point_clouds_recall")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=HERE / "artifacts")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--dataset", choices=("fiqa", "hotpotqa", "both"), default="both")
    parser.add_argument("--min-recall", type=float, default=.75)
    parser.add_argument("--include-copied", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.dataset in ("fiqa", "both"):
        fiqa(args.artifacts, args.output, args.min_recall, args.include_copied)
    if args.dataset in ("hotpotqa", "both"):
        hotpotqa(args.artifacts, args.output, args.min_recall)


if __name__ == "__main__":
    main()
