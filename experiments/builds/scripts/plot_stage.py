#!/usr/bin/env python3
"""Per-stage Pareto + breakdown plots for the main experiments.

Reads CSVs under

    experiments/<stage>/results/<dataset>/<index_name>/<build_name>/<variant_name>/<search_name>/<prefix>_<sv>.csv

(prefix = `latency_` / `multi_latency_` / `batch_`) and emits PDFs per dataset:

    <out_dir>/<dataset>_pareto.pdf           Recall vs QPS (same as before).
    <out_dir>/<dataset>_pareto_latency_ms.pdf   (*latency* & *multi_latency* only)
                                             Recall vs latency ms (``1000 / QPS_seq``).
    <out_dir>/<dataset>_breakdown.pdf        stacked timer bars @ best recall.

For latency / multi_latency, x-axis limits match the QPS Pareto plots (left ≥ 0.88 / 0.82;
right ~ 1.002).

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
    "mvivf_spill": "MVIVF Spill",
    "mvivf_flat": "MVIVF Flat",
    "muvera": "MUVERA",
    "mpool": "Mean-Pool",
    "vamana": "MV-Vamana",
    "svh_ivf": "SVH IVF",
    "svh_graph": "SVH Graph",
    "fastplaid": "FastPlaid",
}

# Logical breakdown buckets per method: collapse the raw timer columns coming
# out of methods.yaml into 3-4 named segments so the per-dataset breakdown
# plot stays readable. Each value is an ordered list of CSV columns to sum
# (missing columns are ignored). The bucket order also drives stack order
# (bottom -> top) and legend order. Stays in sync with methods.yaml::labels
# and the search_with_stats implementations.
_BREAKDOWN_BUCKETS = {
    "mvivf": {
        "encode":     ["t_compress", "t_quant"],
        "greedy":     ["t_search_dists", "t_search_beam", "t_search_rest", "t_search_top_level"],
        "leaf_probe": ["t_leaf_dists", "t_leaf_rest"],
        "rerank":     ["t_rerank"],
    },
    "mvivf_spill": {
        "encode":     ["t_compress", "t_quant"],
        "greedy":     ["t_search_dists", "t_search_beam", "t_search_rest", "t_search_top_level"],
        "leaf_probe": ["t_leaf_dists", "t_leaf_dedup", "t_leaf_rest"],
        "rerank":     ["t_rerank"],
    },
    "mvivf_flat": {
        "encode":     ["t_compress"],
        "greedy":     ["t_search"],
        "leaf_probe": ["t_leaf_dists", "t_leaf_rest"],
        "rerank":     ["t_rerank"],
    },
    "muvera": {
        "encode":     ["t_compress", "t_fde", "t_quant"],
        "search":     ["t_search"],
        "rerank":     ["t_rerank"],
    },
    "mpool": {
        "encode":     ["t_compress", "t_mean_pooling", "t_quant"],
        "search":     ["t_search"],
        "rerank":     ["t_rerank"],
    },
    "vamana": {
        "encode":     ["t_compress", "t_quant"],
        "search":     ["t_search"],
        "rerank":     ["t_rerank"],
    },
    "svh_ivf": {
        "encode":     ["t_compress"],
        "search":     ["t_search_each_total"],
        "merge":      ["t_merge_dedup"],
        "rerank":     ["t_rerank"],
    },
    "svh_graph": {
        "encode":     ["t_compress"],
        "search":     ["t_graph_search"],
        "aggregate":  ["t_aggregate"],
        "rerank":     ["t_rerank"],
    },
    "fastplaid": {
        "search":     ["t_batch_search"],
    },
}

# Stable color per bucket (shared across methods so the legend is consistent
# across plots).
_BUCKET_COLORS = {
    "encode":     "#4C72B0",
    "greedy":     "#DD8452",
    "search":     "#DD8452",
    "leaf_probe": "#55A467",
    "merge":      "#A7B85A",
    "aggregate":  "#A7B85A",
    "rerank":     "#C44E52",
}


def _bucketize(row: pd.Series, method: str) -> dict:
    """Sum the per-row raw timer columns into the logical buckets defined for
    ``method``. Returns ``{bucket: seconds}`` (only buckets with non-zero
    contributions are included).
    """
    spec = _BREAKDOWN_BUCKETS.get(method)
    if not spec:
        return {}
    out = {}
    for bucket, cols in spec.items():
        total = 0.0
        any_present = False
        for c in cols:
            if c in row.index and pd.notna(row[c]):
                try:
                    total += float(row[c])
                    any_present = True
                except (TypeError, ValueError):
                    pass
        if any_present:
            out[bucket] = total
    return out

# Main experiment Pareto x-axis: zoom into high-recall region [left, right].
_XLIM_LEFT_RECALL_1_K = 0.88
_XLIM_LEFT_RECALL_K_K = 0.82
# Tiny pad past 1.0 so markers / ticks at recall == 1 are not clipped by the spine.
_XLIM_RIGHT_RECALL = 1.002


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


def _pareto_curve_recall_latency_ms(
    xs: np.ndarray,
    qps: np.ndarray,
    min_recall_spacing: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    """Same sweep points as QPS Pareto, but y = ``1000 / QPS`` ms.

    Per recall bucket keep **minimum** latency; frontier maximizes recall and
    minimizes latency (same undominated set as QPS Pareto when QPS > 0).
    """
    eps = 1e-12
    qps_f = np.maximum(np.asarray(qps, dtype=float), eps)
    lat = 1000.0 / qps_f

    best_by_recall: dict[float, float] = {}
    for x, y in zip(xs, lat):
        xf = float(x)
        yf = float(y)
        cur = best_by_recall.get(xf)
        if cur is None or yf < cur:
            best_by_recall[xf] = yf

    xs_u = np.array(sorted(best_by_recall.keys()), dtype=float)
    ys_u = np.array([best_by_recall[x] for x in xs_u], dtype=float)

    order = np.argsort(-xs_u, kind="stable")
    keep_i: list[int] = []
    best_lat = np.inf
    for i in order:
        if ys_u[i] < best_lat:
            keep_i.append(i)
            best_lat = ys_u[i]
    mask = np.zeros_like(xs_u, dtype=bool)
    mask[np.array(keep_i, dtype=int)] = True

    fx = xs_u[mask]
    fy = ys_u[mask]
    if fx.size == 0:
        return fx, fy
    ord_x = np.argsort(fx)
    fx = fx[ord_x]
    fy = fy[ord_x]

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


def _xlim_left_floor_for_recall_col(x_col: str) -> float:
    if x_col == "recall_1_k":
        return _XLIM_LEFT_RECALL_1_K
    if x_col == "recall_k_k":
        return _XLIM_LEFT_RECALL_K_K
    return 0.0


def _pareto_panel_xlim(ax, df: pd.DataFrame, methods_present: list[str],
                       x_col: str, qps: str) -> None:
    floor = _xlim_left_floor_for_recall_col(x_col)
    xs_parts: list[np.ndarray] = []
    for method in methods_present:
        sub_m = df[df["method"] == method]
        for variant in sorted(sub_m["variant"].unique()):
            sub = sub_m[sub_m["variant"] == variant].dropna(subset=[x_col, qps])
            if not sub.empty:
                xs_parts.append(sub[x_col].to_numpy(dtype=float))
    if xs_parts:
        lo = float(np.min(np.concatenate(xs_parts)))
        left = max(floor, lo - 0.02)
        ax.set_xlim(left, _XLIM_RIGHT_RECALL)
    else:
        ax.set_xlim(floor, _XLIM_RIGHT_RECALL)


def _plot_pareto_panel(
    ax,
    df: pd.DataFrame,
    stage: str,
    qps: str,
    x_col: str,
    xlabel: str,
    *,
    show_ylabel: bool,
    show_legend: bool,
) -> None:
    methods_present = [m for m in _METHOD_ORDER if m in df["method"].unique()]
    cmap = plt.colormaps.get_cmap("tab10")
    color_for = {m: cmap(i % 10) for i, m in enumerate(methods_present)}
    y_label = "QPS (batch)" if stage == "batch" else "QPS (per-query)"

    for method in methods_present:
        sub_m = df[df["method"] == method]
        for vi, variant in enumerate(sorted(sub_m["variant"].unique())):
            sub = sub_m[sub_m["variant"] == variant].dropna(subset=[x_col, qps])
            if sub.empty:
                continue
            xs = sub[x_col].to_numpy(dtype=float)
            ys = sub[qps].to_numpy(dtype=float)
            ax.scatter(xs, ys, s=18, color=color_for[method], alpha=0.22, zorder=1)
            fx, fy = _pareto_curve(xs, ys)
            if fx.size == 0:
                continue
            label = f"{_PRETTY_METHOD.get(method, method)}"
            if variant and variant != "raw":
                label += f" / {variant}"
            ls = "-" if vi == 0 else "--"
            ax.plot(
                fx,
                fy,
                marker="o",
                markersize=4.5,
                linewidth=1.9,
                linestyle=ls,
                color=color_for[method],
                label=label if show_legend else None,
                zorder=3,
            )

    ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel(y_label)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.spines["top"].set_visible(False)
    if show_legend:
        ax.legend(loc="best", fontsize=8)
    _pareto_panel_xlim(ax, df, methods_present, x_col, qps)


def _plot_pareto_latency_ms_panel(
    ax,
    df: pd.DataFrame,
    qps: str,
    x_col: str,
    xlabel: str,
    *,
    show_ylabel: bool,
    show_legend: bool,
) -> None:
    methods_present = [m for m in _METHOD_ORDER if m in df["method"].unique()]
    cmap = plt.colormaps.get_cmap("tab10")
    color_for = {m: cmap(i % 10) for i, m in enumerate(methods_present)}
    y_label = "Latency (ms)"

    for method in methods_present:
        sub_m = df[df["method"] == method]
        for vi, variant in enumerate(sorted(sub_m["variant"].unique())):
            sub = sub_m[sub_m["variant"] == variant].dropna(subset=[x_col, qps])
            if sub.empty:
                continue
            xs = sub[x_col].to_numpy(dtype=float)
            qv = sub[qps].to_numpy(dtype=float)
            lat_ms = 1000.0 / np.maximum(qv, 1e-12)
            ax.scatter(xs, lat_ms, s=18, color=color_for[method], alpha=0.22, zorder=1)
            fx, fy = _pareto_curve_recall_latency_ms(xs, qv)
            if fx.size == 0:
                continue
            label = f"{_PRETTY_METHOD.get(method, method)}"
            if variant and variant != "raw":
                label += f" / {variant}"
            ls = "-" if vi == 0 else "--"
            ax.plot(
                fx,
                fy,
                marker="o",
                markersize=4.5,
                linewidth=1.9,
                linestyle=ls,
                color=color_for[method],
                label=label if show_legend else None,
                zorder=3,
            )

    ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel(y_label)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.spines["top"].set_visible(False)
    if show_legend:
        ax.legend(loc="best", fontsize=8)
    _pareto_panel_xlim(ax, df, methods_present, x_col, qps)


def _plot_pareto(df: pd.DataFrame, stage: str, dataset: str,
                 out_path: pathlib.Path) -> None:
    qps = _qps_column(stage)
    if qps not in df.columns:
        print(f"  [skip pareto for {dataset}: missing {qps}]")
        return

    has_left = "recall_1_k" in df.columns and not df.dropna(
        subset=["recall_1_k", qps]
    ).empty
    has_right = "recall_k_k" in df.columns and not df.dropna(
        subset=["recall_k_k", qps]
    ).empty

    if not has_left and not has_right:
        print(
            f"  [skip pareto for {dataset}: no rows with recall_1_k/recall_k_k "
            f"and {qps}]"
        )
        return

    if has_left and has_right:
        fig, (ax_left, ax_right) = plt.subplots(
            1, 2, figsize=(12, 5), sharey=True, layout="constrained"
        )
        _plot_pareto_panel(
            ax_left,
            df,
            stage,
            qps,
            "recall_1_k",
            "Recall (1@k)",
            show_ylabel=True,
            show_legend=False,
        )
        _plot_pareto_panel(
            ax_right,
            df,
            stage,
            qps,
            "recall_k_k",
            "Recall (k@k)",
            show_ylabel=False,
            show_legend=True,
        )
    elif has_left:
        fig, ax_one = plt.subplots(figsize=(7.2, 5.2), layout="constrained")
        _plot_pareto_panel(
            ax_one,
            df,
            stage,
            qps,
            "recall_1_k",
            "Recall (1@k)",
            show_ylabel=True,
            show_legend=True,
        )
    else:
        fig, ax_one = plt.subplots(figsize=(7.2, 5.2), layout="constrained")
        _plot_pareto_panel(
            ax_one,
            df,
            stage,
            qps,
            "recall_k_k",
            "Recall (k@k)",
            show_ylabel=True,
            show_legend=True,
        )

    fig.suptitle(f"{dataset}: recall vs QPS [{stage}]", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _plot_pareto_latency_ms(df: pd.DataFrame, stage: str, dataset: str,
                          out_path: pathlib.Path) -> None:
    """Recall vs ``1000/QPS_seq`` (ms). Only for latency / multi_latency stages."""
    qps = _qps_column(stage)
    if qps not in df.columns:
        print(f"  [skip latency-ms plot for {dataset}: missing {qps}]")
        return

    has_left = "recall_1_k" in df.columns and not df.dropna(
        subset=["recall_1_k", qps]
    ).empty
    has_right = "recall_k_k" in df.columns and not df.dropna(
        subset=["recall_k_k", qps]
    ).empty

    if not has_left and not has_right:
        print(
            f"  [skip latency-ms plot for {dataset}: no rows with recall and {qps}]"
        )
        return

    if has_left and has_right:
        fig, (ax_left, ax_right) = plt.subplots(
            1, 2, figsize=(12, 5), sharey=True, layout="constrained"
        )
        _plot_pareto_latency_ms_panel(
            ax_left,
            df,
            qps,
            "recall_1_k",
            "Recall (1@k)",
            show_ylabel=True,
            show_legend=False,
        )
        _plot_pareto_latency_ms_panel(
            ax_right,
            df,
            qps,
            "recall_k_k",
            "Recall (k@k)",
            show_ylabel=False,
            show_legend=True,
        )
    elif has_left:
        fig, ax_one = plt.subplots(figsize=(7.2, 5.2), layout="constrained")
        _plot_pareto_latency_ms_panel(
            ax_one,
            df,
            qps,
            "recall_1_k",
            "Recall (1@k)",
            show_ylabel=True,
            show_legend=True,
        )
    else:
        fig, ax_one = plt.subplots(figsize=(7.2, 5.2), layout="constrained")
        _plot_pareto_latency_ms_panel(
            ax_one,
            df,
            qps,
            "recall_k_k",
            "Recall (k@k)",
            show_ylabel=True,
            show_legend=True,
        )

    fig.suptitle(
        f"{dataset}: recall vs latency (1000 / QPS_seq ms) [{stage}]",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _best_per_group(df: pd.DataFrame, recall_col: str = "recall_k_k") -> pd.DataFrame:
    if recall_col not in df.columns:
        return pd.DataFrame()
    # For each (method, variant) take the row with the highest recall.
    return df.loc[df.groupby(["method", "variant"])[recall_col].idxmax()]


def _row_total_seconds(row: pd.Series, method: str) -> float:
    """Sum every timer column the method exposes (across all buckets) for the
    purposes of estimating the per-query normalisation factor.
    """
    spec = _BREAKDOWN_BUCKETS.get(method) or {}
    total = 0.0
    for cols in spec.values():
        for c in cols:
            if c in row.index and pd.notna(row[c]):
                try:
                    total += float(row[c])
                except (TypeError, ValueError):
                    pass
    return total


def _per_query_factor(row: pd.Series, method: str) -> float:
    """Per-query normalisation: latency CSVs sum ``avg_timings`` across all
    queries in one repetition, so dividing by ``QPS_seq * total_time`` yields
    the per-query seconds. If we can't recover the query count (e.g. batch
    rows have empty avg_timings), fall back to 1.0 (=> raw summed seconds).
    """
    qps = float(row.get("QPS_seq", 0.0) or 0.0)
    if qps <= 0:
        return 1.0
    total = _row_total_seconds(row, method)
    if total <= 0:
        return 1.0
    n_q = qps * total
    return n_q if n_q > 0 else 1.0


def _plot_breakdown(df: pd.DataFrame, stage: str,
                    dataset: str, out_path: pathlib.Path) -> None:
    """Per-method, per-row breakdown bars at best recall for one dataset.

    Each bar collapses the raw timer columns into the 3-4 logical buckets
    declared in ``_BREAKDOWN_BUCKETS`` so the plot stays readable across
    methods that otherwise expose 5-12 timer columns.
    """
    best = _best_per_group(df)
    if best.empty:
        print(f"  [skip breakdown for {dataset}: no recall_k_k rows]")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_x = []
    bar_labels = []
    bar_data = []  # list of dicts {bucket: per-query seconds} for each bar

    # Method order matches _METHOD_ORDER so plots are visually consistent
    # across datasets even when only a subset of methods has CSVs.
    methods_in_order = [m for m in _METHOD_ORDER if m in best["method"].unique()]
    extras = sorted(set(best["method"]) - set(methods_in_order))
    methods_in_order += extras

    for method in methods_in_order:
        for _, row in best[best["method"] == method].iterrows():
            seg = _bucketize(row, method)
            if not seg:
                continue
            # Normalise to per-query seconds so bars are comparable to the
            # cross-dataset mvivf-only breakdown.
            factor = _per_query_factor(row, method)
            seg = {k: v / factor for k, v in seg.items()}
            bar_data.append(seg)
            variant = row.get("variant", "")
            nice = _PRETTY_METHOD.get(method, method)
            if variant and variant != "raw":
                nice += f"\n{variant}"
            bar_labels.append(nice)
            bar_x.append(len(bar_x))

    if not bar_data:
        print(f"  [skip breakdown for {dataset}: no timer columns matched]")
        plt.close(fig)
        return

    all_segments: list[str] = []
    for d in bar_data:
        for k in d:
            if k not in all_segments:
                all_segments.append(k)
    # Stable, semantic colors when possible; fall back to tab10 for unknown.
    cmap = plt.colormaps.get_cmap("tab10")
    color_for = {}
    palette_idx = 0
    for seg in all_segments:
        if seg in _BUCKET_COLORS:
            color_for[seg] = _BUCKET_COLORS[seg]
        else:
            color_for[seg] = cmap(palette_idx % 10)
            palette_idx += 1
    bottoms = np.zeros(len(bar_data), dtype=float)
    for seg in all_segments:
        heights = np.array([d.get(seg, 0.0) for d in bar_data], dtype=float)
        ax.bar(bar_x, heights, bottom=bottoms, label=seg,
               color=color_for[seg], width=0.6, edgecolor="white", linewidth=0.5)
        bottoms += heights

    ax.set_xticks(bar_x)
    ax.set_xticklabels(bar_labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Per-query time (s) @ best recall")
    ax.set_title(f"{dataset}: timer breakdown [{stage}]")
    ax.legend(loc="best", fontsize=8, ncol=1)
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
        if args.stage in ("latency", "multi_latency"):
            _plot_pareto_latency_ms(
                df, args.stage, ds, out_dir / f"{ds}_pareto_latency_ms.pdf"
            )
        _plot_breakdown(df, args.stage, ds, out_dir / f"{ds}_breakdown.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
