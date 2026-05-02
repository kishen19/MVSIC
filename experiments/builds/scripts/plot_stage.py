#!/usr/bin/env python3
"""Per-stage Pareto + breakdown plots for the main experiments.

Reads CSVs under

    experiments/<stage>/results/<dataset>/<index_name>/<build_name>/<variant_name>/<search_name>/<prefix>_<sv>.csv

(prefix = `latency_` / `multi_latency_` / `batch_`) and emits two PDFs per
dataset:

    <out_dir>/<dataset>_pareto.pdf      recall_k_k  vs.  QPS (one curve per method x variant)
    <out_dir>/<dataset>_breakdown.pdf   stacked bar of timer labels at the
                                         best-recall sweep point per method x variant

The timer-label list per method comes from `benchmarks/methods.yaml`.

Usage (called by the per-stage plot.py wrappers):

    python3 experiments/builds/scripts/plot_stage.py \
        --stage latency \
        --results experiments/latency/results \
        --methods benchmarks/methods.yaml \
        --datasets nfcorpus,scifact,arguana,scidocs,fiqa
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import Optional

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import yaml
except ImportError:
    sys.exit("pip install pandas matplotlib pyyaml numpy to use this script")


_PREFIX_FOR_STAGE = {
    "latency":       "latency_",
    "multi_latency": "multi_latency_",
    "batch":         "batch_",
}

# Methods we expect to show up under <results>/<dataset>/. Order also drives
# legend ordering in the plot.
_METHOD_ORDER = ["mvivf", "muvera", "vamana", "svh_graph", "fastplaid"]

# Pretty labels for the legend / x-tick names.
_PRETTY_METHOD = {
    "mvivf": "MVIVF",
    "muvera": "MUVERA",
    "vamana": "MV-Vamana",
    "svh_graph": "SVH Graph",
    "fastplaid": "FastPlaid",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    if "qps_seq" in df.columns and "QPS_seq" not in df.columns:
        rename["qps_seq"] = "QPS_seq"
    if "qps_par" in df.columns and "QPS_par" not in df.columns:
        rename["qps_par"] = "QPS_par"
    return df.rename(columns=rename) if rename else df


def _annotate_path(results_root: pathlib.Path, csv_file: pathlib.Path) -> dict:
    """Recover dataset/method/build/variant/search from the file path.

    Layout: <results_root>/<dataset>/<method>/<build>/<variant>/<search>/<csv>
    Variant is optional; when missing, all parts shift left by one.
    """
    rel = csv_file.relative_to(results_root)
    parts = rel.parts  # final entry is the csv filename
    meta: dict = {"_csv": parts[-1]}
    if len(parts) >= 5:
        meta["dataset"] = parts[0]
        meta["method"] = parts[1]
        meta["build"] = parts[2]
        if len(parts) == 6:
            meta["variant"] = parts[3]
            meta["search"] = parts[4]
        else:
            meta["variant"] = "raw"
            meta["search"] = parts[3]
    return meta


def _load_csvs(results_root: pathlib.Path, prefix: str,
               dataset: Optional[str] = None) -> pd.DataFrame:
    base = results_root / dataset if dataset else results_root
    if not base.exists():
        raise SystemExit(f"no results directory found at {base}")
    files = sorted(base.rglob(f"{prefix}*.csv"))
    if not files:
        raise SystemExit(f"no {prefix}*.csv files under {base}")
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue
        df = _normalize_columns(df)
        meta = _annotate_path(results_root, f)
        for k, v in meta.items():
            df[k] = v
        rows.append(df)
    if not rows:
        raise SystemExit(f"all {prefix}*.csv files under {base} were empty")
    return pd.concat(rows, ignore_index=True)


def _qps_column(stage: str) -> str:
    # latency / multi_latency report per-query throughput in QPS_seq;
    # batch reports throughput from search_all in QPS_par.
    return "QPS_par" if stage == "batch" else "QPS_seq"


def _pareto_front(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    # Maximize x (recall) and y (qps). Return mask of points on the front.
    order = np.argsort(-xs, kind="stable")
    keep = []
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
    """Benchmark-style Pareto curve for recall-vs-QPS points.

    Steps:
      1) De-duplicate recall values by keeping max QPS at each recall.
      2) Keep non-dominated points (maximize recall and QPS).
      3) Optionally simplify jagged fronts with a minimum recall spacing.
    """
    if xs.size == 0:
        return xs, ys

    # 1) De-duplicate by recall: for each unique recall, keep highest QPS.
    best_by_recall: dict[float, float] = {}
    for x, y in zip(xs, ys):
        cur = best_by_recall.get(float(x))
        if cur is None or float(y) > cur:
            best_by_recall[float(x)] = float(y)
    xs_u = np.array(sorted(best_by_recall.keys()), dtype=float)
    ys_u = np.array([best_by_recall[x] for x in xs_u], dtype=float)

    # 2) True Pareto front.
    front_mask = _pareto_front(xs_u, ys_u)
    fx = xs_u[front_mask]
    fy = ys_u[front_mask]
    if fx.size == 0:
        return fx, fy
    order = np.argsort(fx)
    fx = fx[order]
    fy = fy[order]

    # 3) Simplify with min recall spacing.
    if min_recall_spacing <= 0 or fx.size <= 2:
        return fx, fy
    keep_idx = [0]
    for i in range(1, fx.size - 1):
        if abs(fx[i] - fx[keep_idx[-1]]) > min_recall_spacing:
            keep_idx.append(i)
    if keep_idx[-1] != fx.size - 1:
        keep_idx.append(fx.size - 1)
    keep_idx = np.array(keep_idx, dtype=int)
    return fx[keep_idx], fy[keep_idx]


def _plot_pareto(df: pd.DataFrame, stage: str, dataset: str,
                 out_path: pathlib.Path) -> None:
    qps = _qps_column(stage)
    if qps not in df.columns or "recall_k_k" not in df.columns:
        print(f"  [skip pareto for {dataset}: missing {qps} or recall_k_k]")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    methods_present = [m for m in _METHOD_ORDER if m in df["method"].unique()]
    cmap = plt.colormaps.get_cmap("tab10")
    color_for = {m: cmap(i % 10) for i, m in enumerate(methods_present)}
    for method in methods_present:
        sub_m = df[df["method"] == method]
        for vi, variant in enumerate(sorted(sub_m["variant"].unique())):
            sub = sub_m[sub_m["variant"] == variant].dropna(
                subset=["recall_k_k", qps]
            )
            if sub.empty:
                continue
            xs = sub["recall_k_k"].to_numpy(dtype=float)
            ys = sub[qps].to_numpy(dtype=float)
            ax.scatter(xs, ys, s=18, color=color_for[method], alpha=0.35)
            fx, fy = _pareto_curve(xs, ys)
            if fx.size == 0:
                continue
            label = f"{_PRETTY_METHOD.get(method, method)}"
            if variant and variant != "raw":
                label += f" / {variant}"
            ls = "-" if vi == 0 else "--"
            ax.plot(fx, fy, marker="o",
                    linestyle=ls, color=color_for[method], label=label)

    ax.set_xlabel("Recall@k")
    ax.set_ylabel("QPS (batch)" if stage == "batch" else "QPS (per-query)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"{dataset}: recall vs QPS [{stage}]")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _label_columns(df: pd.DataFrame, labels: list) -> list:
    # Search columns in df that match any of the per-method timer labels.
    # benchmark_search.py expands `avg_timings` into columns whose names match
    # the labels in methods.yaml directly.
    return [lab for lab in labels if lab in df.columns]


def _best_per_group(df: pd.DataFrame, recall_col: str = "recall_k_k") -> pd.DataFrame:
    if recall_col not in df.columns:
        return pd.DataFrame()
    # For each (method, variant) take the row with the highest recall.
    return df.loc[df.groupby(["method", "variant"])[recall_col].idxmax()]


def _plot_breakdown(df: pd.DataFrame, methods_yaml: dict, stage: str,
                    dataset: str, out_path: pathlib.Path) -> None:
    best = _best_per_group(df)
    if best.empty:
        print(f"  [skip breakdown for {dataset}: no recall_k_k rows]")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_x = []
    bar_labels = []
    bar_data = []  # list of dicts {label: value} for each bar
    for _, row in best.iterrows():
        method = row["method"]
        variant = row["variant"]
        meta = methods_yaml.get(method, {})
        labels = meta.get("labels", []) or []
        cols = _label_columns(pd.DataFrame([row]), labels)
        if not cols:
            continue
        bar_data.append({lab: float(row[lab]) for lab in cols})
        bar_x.append(len(bar_x))
        nice = _PRETTY_METHOD.get(method, method)
        if variant and variant != "raw":
            nice += f"\n{variant}"
        bar_labels.append(nice)

    if not bar_data:
        print(f"  [skip breakdown for {dataset}: no timer columns matched]")
        plt.close(fig)
        return

    all_segments = []
    for d in bar_data:
        for k in d:
            if k not in all_segments:
                all_segments.append(k)
    cmap = plt.colormaps.get_cmap("tab20")
    color_for = {seg: cmap(i % 20) for i, seg in enumerate(all_segments)}
    bottoms = np.zeros(len(bar_data), dtype=float)
    for seg in all_segments:
        heights = np.array([d.get(seg, 0.0) for d in bar_data], dtype=float)
        ax.bar(bar_x, heights, bottom=bottoms, label=seg,
               color=color_for[seg], width=0.6)
        bottoms += heights

    ax.set_xticks(bar_x)
    ax.set_xticklabels(bar_labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Time (s, summed across queries)")
    ax.set_title(f"{dataset}: per-stage timer breakdown @ best recall [{stage}]")
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
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
    p.add_argument("--stage", required=True, choices=list(_PREFIX_FOR_STAGE))
    p.add_argument("--results", required=True, type=pathlib.Path,
                   help="experiments/<stage>/results/")
    p.add_argument("--methods", default="benchmarks/methods.yaml", type=pathlib.Path)
    p.add_argument("--out-dir", type=pathlib.Path, default=None,
                   help="default: <results>/_plots")
    p.add_argument("--datasets", default=None,
                   help="comma-separated dataset names; default: all directories under --results")
    args = p.parse_args()

    prefix = _PREFIX_FOR_STAGE[args.stage]
    out_dir = args.out_dir if args.out_dir else args.results / "_plots"
    with open(args.methods, encoding="utf-8") as f:
        methods_yaml = yaml.safe_load(f) or {}

    datasets = (
        [d.strip() for d in args.datasets.split(",") if d.strip()]
        if args.datasets else _list_datasets(args.results)
    )
    if not datasets:
        print(f"[warn] no datasets under {args.results}; nothing to plot.")
        return 0

    for ds in datasets:
        try:
            df = _load_csvs(args.results, prefix, dataset=ds)
        except SystemExit as e:
            print(f"[skip {ds}] {e}")
            continue
        print(f"[{ds}] {len(df)} rows from {df['method'].nunique()} methods")
        _plot_pareto(df, args.stage, ds, out_dir / f"{ds}_pareto.pdf")
        _plot_breakdown(df, methods_yaml, args.stage, ds, out_dir / f"{ds}_breakdown.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
