#!/usr/bin/env python3
"""Plot saved prefix-doubling build measurements; never run retrieval jobs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def number(row: dict[str, str], *names: str) -> float:
    for name in names:
        value = row.get(name, "")
        if value != "":
            return float(value)
    raise ValueError(f"Missing one of {names}: {row}")


def pareto(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    result = []
    best = float("-inf")
    for x, y in sorted(set(points), key=lambda point: (point[0], -point[1])):
        if y > best:
            result.append((x, y))
            best = y
    return result


def baseline(variant: str) -> bool:
    return any(token in variant.lower() for token in ("baseline", "global", "reference"))


def save(fig, destination: Path, name: str) -> None:
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(destination / f"{name}.{suffix}", dpi=220)
    plt.close(fig)


def plot_search_curves(destination, labels, colors, files, latency, minimum_recall):
    if not files:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.1))
    output = []
    for path in files:
        variant = path.stem.removesuffix("_latency" if latency else "_curve")
        data = rows(path)
        if not data:
            continue
        points = []
        for row in data:
            # Candidate-only curves and returned-result latency curves are kept
            # separate: candidate coverage is a ceiling, not final TQ8 recall.
            if latency:
                x = number(row, "total_ms", "total_ms_mean", "latency_ms")
                y = number(row, "recall_10", "final_recall_10", "mean_recall")
            else:
                x = number(row, "mean_candidates", "candidates_mean")
                y = number(row, "candidate_recall_10", "mean_recall")
            points.append((x, y))
        frontier = pareto(points)
        if not frontier:
            continue
        x, y = zip(*frontier)
        label = labels.get(variant, variant.replace("_", " "))
        ax.plot(x, y, label=label, color=colors[variant], linewidth=2 if baseline(variant) else 1.6,
                marker="o", markersize=3.2, linestyle="-" if baseline(variant) else "--")
        for px, py in frontier:
            output.append(dict(variant=variant, label=label, x=px, recall_10=py))
    ax.set(xlabel="End-to-end query latency (ms)" if latency else "# point clouds scored",
           ylabel="Recall@10" if latency else "Exact top-10 candidate recall",
           ylim=(minimum_recall, 1.005),
           title="FiQA dev: prefix-doubling bucket construction" +
           ("" if latency else "\nFixed exhaustive TQ8 Chamfer prototype order"))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    name = "latency_recall" if latency else "candidate_recall"
    save(fig, destination, name)
    if output:
        with (destination / f"{name}_frontiers.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)


def plot_build(destination, labels, colors, files):
    records = []
    for path in files:
        for row in rows(path):
            variant = row.get("variant", path.stem.removesuffix("_build"))
            records.append((variant, row))
    if not records:
        return
    records.sort(key=lambda record: (not baseline(record[0]), record[0]))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, max(3.8, 0.36 * len(records) + 1.1)),
                             sharey=True)
    names = [labels.get(variant, variant.replace("_", " ")) for variant, _ in records]
    for ax, column, divisor, xlabel in zip(
            axes,
            ("total_seconds", "mean_bucket_overlap", "peak_temporary_id_bytes"),
            (1, 0.01, 1024 ** 2),
            ("Bucket construction time (s)", "Mean exact top-512 bucket overlap (%)",
             "Peak temporary bucket IDs (MiB)")):
        values = [number(row, column) / divisor for _, row in records]
        ax.barh(range(len(records)), values,
                color=[colors.get(variant, "#0072B2") for variant, _ in records], height=0.7)
        ax.set_yticks(range(len(records)), names)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        ax.margins(x=0.2)
        for i, value in enumerate(values):
            ax.text(value, i, f" {value:.2f}", va="center", fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("FiQA: build cost, bucket fidelity, and temporary posting storage")
    save(fig, destination, "build_diagnostics")


def plot_rounds(destination, labels, colors, files):
    if not files:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0), sharex=True)
    metrics = [
        ("reservoir_size", 1, "Temporary bucket size"),
        ("reachable_fraction", 100, "Corpus represented by temporary buckets (%)"),
        ("mean_bucket_overlap", 100, "Mean exact top-512 bucket overlap (%)"),
        ("cumulative_seconds", 1, "Cumulative construction time (s)"),
    ]
    for path in files:
        variant = path.stem.removesuffix("_rounds")
        data = sorted(rows(path), key=lambda row: int(row["active_centers"]))
        if not data:
            continue
        for ax, (column, scale, ylabel) in zip(axes.flat, metrics):
            x = [int(row["active_centers"]) for row in data]
            y = [number(row, column) * scale for row in data]
            ax.plot(x, y, color=colors.get(variant, "#0072B2"), marker="o", markersize=3,
                    label=labels.get(variant, variant.replace("_", " ")))
            ax.set_ylabel(ylabel)
            ax.set_xscale("log", base=2)
            ax.grid(alpha=0.2)
    for ax in axes[1]:
        ax.set_xlabel("Number of built prototypes")
    handles, names = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, 0.96),
                   ncol=min(3, len(handles)), fontsize=8, frameon=False)
    fig.suptitle("FiQA prefix doubling: shrinking temporary buckets", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    for suffix in ("png", "pdf"):
        fig.savefig(destination / f"round_diagnostics.{suffix}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results/prefix_build")
    parser.add_argument("--min-recall", type=float, default=0.75)
    args = parser.parse_args()
    destination = args.results
    labels = {}
    manifest = destination / "variants.csv"
    if manifest.exists():
        labels = {row["variant"]: row["label"] for row in rows(manifest)}
    curve_files = sorted(destination.glob("*_curve.csv"))
    latency_files = sorted(destination.glob("*_latency.csv"))
    build_files = sorted(destination.glob("*_build.csv"))
    round_files = sorted(destination.glob("*_rounds.csv"))
    variants = sorted({path.stem.removesuffix("_curve") for path in curve_files} |
                      {path.stem.removesuffix("_latency") for path in latency_files} |
                      {path.stem.removesuffix("_build") for path in build_files} |
                      {path.stem.removesuffix("_rounds") for path in round_files})
    palette = plt.get_cmap("tab10")
    colors = {variant: "black" if baseline(variant) else palette(i % 10)
              for i, variant in enumerate(variants)}
    plot_search_curves(destination, labels, colors, curve_files, False, args.min_recall)
    plot_search_curves(destination, labels, colors, latency_files, True, args.min_recall)
    plot_build(destination, labels, colors, build_files)
    plot_rounds(destination, labels, colors, round_files)


if __name__ == "__main__":
    main()
