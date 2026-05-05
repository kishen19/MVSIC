#!/usr/bin/env python3
"""Leaf-quantization comparison plots.

For each dataset under ``experiments/quantization_compare/results/<dataset>/``:

  ``<dataset>_pareto.pdf`` -- Recall (k@k) vs QPS Pareto curves, one per
  leaf-quantizer variant (currently ``tq1`` and ``fs``).

The MVIVF skeleton is identical across variants; only the leaf-scoring
quantizer changes. Styling follows ``experiments/builds/scripts/plot_stage.py``
(the same look as ``experiments/latency``, ``experiments/multi_latency``,
``experiments/batch``).

Usage:
    scripts/plot.py                     # every dataset under results/
    scripts/plot.py --datasets nq,fiqa
"""
from __future__ import annotations

import argparse
import pathlib
import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("pip install pandas matplotlib numpy to use this script")


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent


# Display order + pretty labels. tab10 colors keyed by position in this order
# so colors stay stable across datasets even when a variant is missing.
_VARIANT_ORDER = ["tq1", "fs"]
_VARIANT_LABELS = {
    "tq1": "TQ-1bit (OneBitTQ)",
    "fs":  "FastScan (block_size=8)",
}

# Mirrors plot_stage.py for the recall_k_k axis.
_XLIM_LEFT_RECALL_K_K = 0.82
_XLIM_RIGHT_RECALL = 1.002


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    if "qps_seq" in df.columns and "QPS_seq" not in df.columns:
        rename["qps_seq"] = "QPS_seq"
    if "qps_par" in df.columns and "QPS_par" not in df.columns:
        rename["qps_par"] = "QPS_par"
    return df.rename(columns=rename) if rename else df


def _load_dataset(results_root: pathlib.Path, dataset: str) -> pd.DataFrame:
    base = results_root / dataset
    if not base.exists():
        raise SystemExit(f"no results directory: {base}")
    files = sorted(base.rglob("latency_*.csv"))
    if not files:
        raise SystemExit(f"no latency_*.csv files under {base}")
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue
        df = _normalize_columns(df)
        rel = f.relative_to(base)
        parts = rel.parts
        # Layout: <method>/<build>/<variant>/<search>/<csv>
        if len(parts) < 5:
            continue
        df["index_name"] = parts[0]
        df["build"]      = parts[1]
        df["variant"]    = parts[2]
        df["search"]     = parts[3]
        rows.append(df)
    if not rows:
        raise SystemExit(f"all latency_*.csv files under {base} were empty")
    return pd.concat(rows, ignore_index=True)


def _pareto_front(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    # Maximize x (recall) and y (qps). Return mask of points on the front.
    order = np.argsort(-xs, kind="stable")
    keep: list[int] = []
    best_y = -np.inf
    for i in order:
        if ys[i] > best_y:
            keep.append(i)
            best_y = ys[i]
    mask = np.zeros_like(xs, dtype=bool)
    mask[keep] = True
    return mask


def _pareto_curve(xs: np.ndarray, ys: np.ndarray,
                  min_recall_spacing: float = 0.005) -> tuple[np.ndarray, np.ndarray]:
    """Same algorithm as ``plot_stage._pareto_curve``: dedupe-by-recall,
    Pareto front, then thin out near-duplicate recalls for cleaner curves.
    """
    if xs.size == 0:
        return xs, ys

    best_by_recall: dict[float, float] = {}
    for x, y in zip(xs, ys):
        cur = best_by_recall.get(float(x))
        if cur is None or float(y) > cur:
            best_by_recall[float(x)] = float(y)
    xs_u = np.array(sorted(best_by_recall.keys()), dtype=float)
    ys_u = np.array([best_by_recall[x] for x in xs_u], dtype=float)

    front_mask = _pareto_front(xs_u, ys_u)
    fx = xs_u[front_mask]
    fy = ys_u[front_mask]
    if fx.size == 0:
        return fx, fy
    order = np.argsort(fx)
    fx = fx[order]
    fy = fy[order]

    if min_recall_spacing <= 0 or fx.size <= 2:
        return fx, fy
    keep_idx = [0]
    for i in range(1, fx.size - 1):
        if abs(fx[i] - fx[keep_idx[-1]]) > min_recall_spacing:
            keep_idx.append(i)
    if keep_idx[-1] != fx.size - 1:
        keep_idx.append(fx.size - 1)
    keep_idx_arr = np.array(keep_idx, dtype=int)
    return fx[keep_idx_arr], fy[keep_idx_arr]


def _set_recall_xlim(ax, xs_all: np.ndarray) -> None:
    if xs_all.size:
        lo = float(np.min(xs_all))
        left = max(_XLIM_LEFT_RECALL_K_K, lo - 0.02)
    else:
        left = _XLIM_LEFT_RECALL_K_K
    ax.set_xlim(left, _XLIM_RIGHT_RECALL)


def _plot_pareto(df: pd.DataFrame, dataset: str, out_path: pathlib.Path) -> None:
    variants_present = [q for q in _VARIANT_ORDER if q in df["variant"].unique()]
    if not variants_present:
        print(f"  [skip pareto for {dataset}: no known quantizer variants]")
        return

    cmap = plt.colormaps.get_cmap("tab10")
    # Stable color per variant across datasets.
    color_for = {q: cmap(_VARIANT_ORDER.index(q) % 10) for q in variants_present}

    fig, ax = plt.subplots(figsize=(7.2, 5.2), layout="constrained")
    xs_all_parts: list[np.ndarray] = []

    for q in variants_present:
        sub = df[df["variant"] == q].dropna(subset=["recall_k_k", "QPS_seq"])
        if sub.empty:
            continue
        xs = sub["recall_k_k"].to_numpy(dtype=float)
        ys = sub["QPS_seq"].to_numpy(dtype=float)
        xs_all_parts.append(xs)

        color = color_for[q]
        ax.scatter(xs, ys, s=18, color=color, alpha=0.22, zorder=1)

        fx, fy = _pareto_curve(xs, ys)
        if fx.size == 0:
            continue
        ax.plot(
            fx,
            fy,
            marker="o",
            markersize=4.5,
            linewidth=1.9,
            color=color,
            label=_VARIANT_LABELS.get(q, q),
            zorder=3,
        )

    ax.set_xlabel("Recall (k@k)")
    ax.set_ylabel("QPS (per-query)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.legend(loc="best", fontsize=8, title="leaf quantizer")
    _set_recall_xlim(
        ax,
        np.concatenate(xs_all_parts) if xs_all_parts else np.array([], dtype=float),
    )

    fig.suptitle(f"{dataset}: leaf-quantization compare [latency]", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _list_datasets(results_root: pathlib.Path) -> list[str]:
    if not results_root.exists():
        return []
    return sorted([
        p.name for p in results_root.iterdir()
        if p.is_dir() and not p.name.startswith(".") and not p.name.startswith("_")
    ])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets", default=None,
        help="comma-separated dataset names; default: every dir under --results",
    )
    p.add_argument(
        "--results", type=pathlib.Path,
        default=REPO_ROOT / "experiments" / "quantization_compare" / "results",
    )
    p.add_argument(
        "--out-dir", type=pathlib.Path, default=None,
        help="default: <results>/_plots",
    )
    args = p.parse_args()

    out_dir = args.out_dir if args.out_dir else args.results / "_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = (
        [d.strip() for d in args.datasets.split(",") if d.strip()]
        if args.datasets else _list_datasets(args.results)
    )
    if not datasets:
        print(f"[warn] no datasets under {args.results}; nothing to plot.")
        return 0

    for ds in datasets:
        try:
            df = _load_dataset(args.results, ds)
        except SystemExit as e:
            print(f"[skip {ds}] {e}")
            continue
        n_rows = len(df)
        n_q = df["variant"].nunique()
        print(f"[{ds}] {n_rows} rows from {n_q} quantizer variants")
        _plot_pareto(df, ds, out_dir / f"{ds}_pareto.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
