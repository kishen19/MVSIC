#!/usr/bin/env python3
"""Plot controlled FiQA float-to-TQ1 speedups against scored document clouds."""

import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "artifacts/archive/repo_exploratory/fiqa_quantization_investigation/results"


def emdivf_rows():
    with (RESULTS / "emdivf_float_score_0.csv").open(newline="") as handle:
        quant = {int(row["nprobes"]): row for row in csv.DictReader(handle)}
    with (RESULTS / "emdivf_float_score_1.csv").open(newline="") as handle:
        floats = {int(row["nprobes"]): row for row in csv.DictReader(handle)}
    result = []
    for probes in sorted(quant):
        q, f = quant[probes], floats[probes]
        if q["candidates_mean"] != f["candidates_mean"]:
            raise ValueError(f"EMDIVF candidate mismatch at {probes} probes")
        result.append(
            (
                float(q["candidates_mean"]),
                float(f["score_tq1_ms"]) / float(q["score_tq1_ms"]),
                float(f["total_ms"]) / float(q["total_ms"]),
            )
        )
    return result


def mvivf_blocks(path):
    text = path.read_text()
    blocks = re.findall(r"--- nprobes = (\d+) ---([\s\S]*?)(?=--- nprobes =|\Z)", text)

    def field(block, label):
        match = re.search(rf"^\s*{re.escape(label)}:\s+([\d.]+)", block, re.MULTILINE)
        if not match:
            raise ValueError(f"Missing {label} in {path}")
        return float(match.group(1))

    return {
        int(probes): {
            "clouds": field(block, "probe_cmps"),
            "score": field(block, "t_leaf_dists"),
            "total": field(block, "Total time"),
        }
        for probes, block in blocks
    }


def mvivf_rows():
    floats = mvivf_blocks(RESULTS / "mvivf_wards_None.log")
    quant = mvivf_blocks(RESULTS / "mvivf_wards_1BTQ.log")
    result = []
    for probes in sorted(quant):
        q, f = quant[probes], floats[probes]
        if q["clouds"] != f["clouds"]:
            raise ValueError(f"MVIVF leaf-count mismatch at {probes} probes")
        n_queries = 128
        result.append(
            (
                q["clouds"] / n_queries,
                f["score"] / q["score"],
                f["total"] / q["total"],
            )
        )
    return result


def main():
    global RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=HERE / "results/quantization_audit")
    args = parser.parse_args()
    RESULTS = args.results
    args.output.mkdir(parents=True, exist_ok=True)
    series = [
        ("EMDIVF", emdivf_rows(), "#1f77b4", "o"),
        ("MVIVF", mvivf_rows(), "#e76f51", "s"),
    ]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), sharex=True)
    for name, rows, color, marker in series:
        x = [row[0] for row in rows]
        for col, axis in [(1, axes[0]), (2, axes[1])]:
            axis.plot(x, [row[col] for row in rows], marker=marker, markersize=6,
                      linewidth=2.2, color=color, label=name)
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xlim(2500, 60000)
        axis.set_xticks([3000, 5000, 10000, 15000, 30000, 50000])
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1000:g}k"))
        axis.grid(axis="both", alpha=0.22)
        axis.set_xlabel("Document point clouds scored per query")
    axes[0].set_title("Candidate scoring stage")
    axes[1].set_title("Whole query")
    axes[0].set_ylabel("Float latency / TQ1 latency (×)")
    axes[1].set_ylabel("Float latency / quantized latency (×)")
    axes[0].set_ylim(0, 25)
    axes[1].set_ylim(0, 17)
    axes[0].legend(loc="lower right", frameon=False)
    fig.suptitle("FiQA: speedup from TQ1 candidate scoring", fontsize=13)
    fig.text(0.5, 0.005,
             "Paired within each method; EMDIVF times first 128 test queries, "
             "MVIVF uses a seeded random 128.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.065, 1, 0.95))
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"fiqa_quantization_speedups.{suffix}",
                    dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
