"""
Analyze within-point-cloud spread: pairwise distances, diameter, and related metrics.

For each point cloud (document/query), computes statistics over all pairwise L2
distances between its vectors. Since BEIR embeddings are unit-normalized,
L2 distance relates to angular distance via ||a-b||^2 = 2(1 - cos(theta)).

Usage:
    python data-tools/point_cloud_spread_stats.py \
        --datasets arguana scidocs fiqa \
        --data-root data/beir \
        --output data-tools/results/point_cloud_spread

    # Quick smoke test on a subset:
    python data-tools/point_cloud_spread_stats.py \
        --datasets arguana --splits points --max-clouds 200
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from tqdm import tqdm

from utils import ReadQueries


@dataclass
class CloudSpreadStats:
    cloud_idx: int
    num_vectors: int
    num_pairs: int
    # L2 pairwise distances (unit vectors on R^d)
    min_pairwise_l2: float
    mean_pairwise_l2: float
    median_pairwise_l2: float
    p90_pairwise_l2: float
    p10_pairwise_l2: float
    std_pairwise_l2: float
    diameter_l2: float
    # Angular distances in degrees (derived from L2 on unit sphere)
    min_pairwise_angle_deg: float
    mean_pairwise_angle_deg: float
    median_pairwise_angle_deg: float
    p90_pairwise_angle_deg: float
    diameter_angle_deg: float
    # Distances to the (unnormalized) centroid
    mean_radius_l2: float
    std_radius_l2: float
    max_radius_l2: float
    # Derived spread ratios
    diameter_over_mean_l2: float
    cv_pairwise_l2: float  # std / mean


def l2_to_angle_deg(d: float) -> float:
    """Convert L2 distance between unit vectors to angle in degrees."""
    cos_theta = np.clip(1.0 - 0.5 * d * d, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def compute_cloud_stats(vectors: np.ndarray, cloud_idx: int) -> CloudSpreadStats | None:
    n = vectors.shape[0]
    if n < 2:
        return None

    pairwise_l2 = pdist(vectors, metric="euclidean")
    mean_l2 = float(np.mean(pairwise_l2))
    std_l2 = float(np.std(pairwise_l2))
    diameter_l2 = float(np.max(pairwise_l2))

    centroid = vectors.mean(axis=0)
    radii = np.linalg.norm(vectors - centroid, axis=1)

    return CloudSpreadStats(
        cloud_idx=cloud_idx,
        num_vectors=n,
        num_pairs=len(pairwise_l2),
        min_pairwise_l2=float(np.min(pairwise_l2)),
        mean_pairwise_l2=mean_l2,
        median_pairwise_l2=float(np.median(pairwise_l2)),
        p90_pairwise_l2=float(np.percentile(pairwise_l2, 90)),
        p10_pairwise_l2=float(np.percentile(pairwise_l2, 10)),
        std_pairwise_l2=std_l2,
        diameter_l2=diameter_l2,
        min_pairwise_angle_deg=l2_to_angle_deg(float(np.min(pairwise_l2))),
        mean_pairwise_angle_deg=l2_to_angle_deg(mean_l2),
        median_pairwise_angle_deg=l2_to_angle_deg(float(np.median(pairwise_l2))),
        p90_pairwise_angle_deg=l2_to_angle_deg(float(np.percentile(pairwise_l2, 90))),
        diameter_angle_deg=l2_to_angle_deg(diameter_l2),
        mean_radius_l2=float(np.mean(radii)),
        std_radius_l2=float(np.std(radii)),
        max_radius_l2=float(np.max(radii)),
        diameter_over_mean_l2=diameter_l2 / mean_l2 if mean_l2 > 0 else float("nan"),
        cv_pairwise_l2=std_l2 / mean_l2 if mean_l2 > 0 else float("nan"),
    )


def analyze_file(
    path: str,
    max_clouds: int | None = None,
) -> tuple[list[CloudSpreadStats], dict]:
    clouds = ReadQueries(path)
    if max_clouds is not None:
        clouds = clouds[:max_clouds]

    stats: list[CloudSpreadStats] = []
    skipped_singleton = 0
    for i, vectors in enumerate(tqdm(clouds, desc=f"Analyzing {Path(path).name}")):
        row = compute_cloud_stats(np.asarray(vectors, dtype=np.float32), i)
        if row is None:
            skipped_singleton += 1
            continue
        stats.append(row)

    meta = {
        "path": path,
        "num_clouds_total": len(clouds),
        "num_clouds_analyzed": len(stats),
        "skipped_singleton": skipped_singleton,
    }
    return stats, meta


def write_per_cloud_csv(stats: list[CloudSpreadStats], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not stats:
        return
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(stats[0]).keys()))
        writer.writeheader()
        for row in stats:
            writer.writerow(asdict(row))


def summarize_metric(values: np.ndarray) -> dict[str, float]:
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def write_summary_markdown(
    all_results: dict[tuple[str, str], list[CloudSpreadStats]],
    out_path: Path,
) -> None:
    metrics = [
        ("diameter_l2", "Diameter (L2)"),
        ("mean_pairwise_l2", "Mean pairwise L2"),
        ("median_pairwise_l2", "Median pairwise L2"),
        ("p90_pairwise_l2", "P90 pairwise L2"),
        ("diameter_angle_deg", "Diameter (degrees)"),
        ("mean_pairwise_angle_deg", "Mean pairwise angle (deg)"),
        ("median_pairwise_angle_deg", "Median pairwise angle (deg)"),
        ("p90_pairwise_angle_deg", "P90 pairwise angle (deg)"),
        ("diameter_over_mean_l2", "Diameter / mean L2"),
        ("cv_pairwise_l2", "CV of pairwise L2"),
        ("mean_radius_l2", "Mean radius to centroid"),
        ("num_vectors", "Vectors per cloud"),
    ]

    lines = [
        "# Point Cloud Within-Cloud Spread Statistics\n",
        "Pairwise L2 distances between vectors **within** each point cloud. "
        "Embeddings are unit-normalized, so L2 distance maps to angular distance.\n",
    ]

    for (dataset, split), stats in sorted(all_results.items()):
        if not stats:
            continue
        lines.append(f"\n## {dataset} ({split})\n")
        lines.append(f"- Clouds analyzed: {len(stats)}\n")
        lines.append(
            f"- Avg vectors/cloud: {np.mean([s.num_vectors for s in stats]):.1f} "
            f"(min={min(s.num_vectors for s in stats)}, "
            f"max={max(s.num_vectors for s in stats)})\n"
        )
        lines.append("\n| Metric | Mean | Median | P90 | Min | Max |\n")
        lines.append("|--------|------|--------|-----|-----|-----|\n")
        for field, label in metrics:
            vals = np.array([getattr(s, field) for s in stats])
            sm = summarize_metric(vals)
            lines.append(
                f"| {label} | {sm['mean']:.4f} | {sm['median']:.4f} | "
                f"{sm['p90']:.4f} | {sm['min']:.4f} | {sm['max']:.4f} |\n"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines))


def plot_distributions(
    all_results: dict[tuple[str, str], list[CloudSpreadStats]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Per-dataset histograms for key metrics ---
    key_metrics = [
        ("diameter_l2", "Diameter (L2)"),
        ("mean_pairwise_l2", "Mean pairwise L2"),
        ("median_pairwise_l2", "Median pairwise L2"),
        ("diameter_angle_deg", "Diameter (degrees)"),
        ("diameter_over_mean_l2", "Diameter / mean"),
        ("num_vectors", "Vectors per cloud"),
    ]

    for field, label in key_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Within-cloud {label} across datasets", fontsize=13)

        # Points split
        ax = axes[0]
        for (dataset, split), stats in sorted(all_results.items()):
            if split != "points" or not stats:
                continue
            vals = [getattr(s, field) for s in stats]
            ax.hist(vals, bins=50, alpha=0.5, density=True, label=dataset)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.set_title("Documents (points)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Queries split
        ax = axes[1]
        for (dataset, split), stats in sorted(all_results.items()):
            if split != "queries" or not stats:
                continue
            vals = [getattr(s, field) for s in stats]
            ax.hist(vals, bins=50, alpha=0.5, density=True, label=dataset)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.set_title("Queries")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / f"hist_{field}.png", dpi=150)
        plt.close(fig)

    # --- Box plots comparing datasets ---
    compare_metrics = [
        ("diameter_l2", "Diameter (L2)"),
        ("mean_pairwise_l2", "Mean pairwise L2"),
        ("p90_pairwise_l2", "P90 pairwise L2"),
        ("diameter_angle_deg", "Diameter (degrees)"),
    ]

    for field, label in compare_metrics:
        for split in ("points", "queries"):
            groups = []
            labels = []
            for dataset in sorted({k[0] for k in all_results}):
                stats = all_results.get((dataset, split), [])
                if stats:
                    groups.append([getattr(s, field) for s in stats])
                    labels.append(dataset)
            if not groups:
                continue

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.boxplot(groups, labels=labels, showfliers=False)
            ax.set_ylabel(label)
            ax.set_title(f"{label} — {split}")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / f"box_{split}_{field}.png", dpi=150)
            plt.close(fig)

    # --- Scatter: cloud size vs spread ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cloud size vs within-cloud spread", fontsize=13)
    for ax, split, yfield, ylabel in [
        (axes[0], "points", "diameter_l2", "Diameter (L2)"),
        (axes[1], "points", "mean_pairwise_l2", "Mean pairwise L2"),
    ]:
        for (dataset, s), stats in sorted(all_results.items()):
            if s != split or not stats:
                continue
            xs = [st.num_vectors for st in stats]
            ys = [getattr(st, yfield) for st in stats]
            ax.scatter(xs, ys, alpha=0.15, s=8, label=dataset)
        ax.set_xlabel("Vectors per cloud")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Documents — {ylabel}")
        ax.legend(markerscale=3)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_size_vs_spread_points.png", dpi=150)
    plt.close(fig)

    # --- ECDF of diameter for cross-dataset comparison ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for (dataset, split), stats in sorted(all_results.items()):
        if split != "points" or not stats:
            continue
        vals = np.sort([s.diameter_l2 for s in stats])
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, label=dataset, linewidth=2)
    ax.set_xlabel("Diameter (L2)")
    ax.set_ylabel("ECDF")
    ax.set_title("ECDF of within-cloud diameter (documents)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "ecdf_diameter_points.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute within-point-cloud pairwise distance statistics."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["arguana", "scidocs", "fiqa"],
        help="Dataset names under data-root.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/beir",
        help="Root directory containing dataset folders.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["points", "queries"],
        choices=["points", "queries"],
        help="Which .pcs splits to analyze.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data-tools/results/point_cloud_spread",
        help="Output directory for CSVs, plots, and summary.",
    )
    parser.add_argument(
        "--max-clouds",
        type=int,
        default=None,
        help="Analyze at most this many clouds per file (for quick tests).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_root = repo_root / args.data_root
    out_dir = repo_root / args.output

    all_results: dict[tuple[str, str], list[CloudSpreadStats]] = {}

    for dataset in args.datasets:
        dataset_dir = data_root / dataset
        for split in args.splits:
            pcs_path = dataset_dir / f"{dataset}_{split}.pcs"
            if not pcs_path.exists():
                print(f"Skipping missing file: {pcs_path}", file=sys.stderr)
                continue

            print(f"\n=== {dataset} / {split} ===")
            stats, meta = analyze_file(str(pcs_path), max_clouds=args.max_clouds)
            all_results[(dataset, split)] = stats

            csv_path = out_dir / "per_cloud" / f"{dataset}_{split}.csv"
            write_per_cloud_csv(stats, csv_path)
            print(
                f"  Analyzed {meta['num_clouds_analyzed']} clouds "
                f"({meta['skipped_singleton']} singletons skipped) "
                f"-> {csv_path}"
            )

    write_summary_markdown(all_results, out_dir / "summary.md")
    plot_distributions(all_results, out_dir / "plots")
    print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
