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
    import matplotlib.ticker as mticker
    import numpy as np
    import pandas as pd
    import yaml
except ImportError:
    sys.exit("pip install pandas matplotlib pyyaml numpy to use this script")

plt.rcParams.update({
    # These PDFs are often included as three panels in one NeurIPS-width row.
    # Use large source fonts so the final scaled figure remains readable.
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 17,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "lines.linewidth": 2.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


_PREFIX_FOR_STAGE = {
    "latency":       "latency_",
    "multi_latency": "multi_latency_",
    "batch":         "batch_",
}

# Methods we expect to show up under <results>/<dataset>/. Order also drives
# legend ordering in the plot.
#
# The default canonical plot family is the four "core" methods (mvivf gets
# legend ordering ahead of muvera / vamana / svh_graph); ``mvivf_spill`` is
# carried in ``_METHOD_ORDER`` only for stable color indexing. External
# baselines are opt-in via ``--include-external`` (alias ``--include-igp-fp``).
_METHOD_ORDER = ["mvivf", "mvivf_spill", "muvera", "vamana", "svh_graph",
                 "igp", "fastplaid", "gem", "hnswlib"]
_PLOTTED_METHOD_ORDER = [m for m in _METHOD_ORDER
                         if m not in ("mvivf_spill", "igp", "fastplaid", "gem", "hnswlib")]
_OPTIONAL_BASELINE_METHODS = ("igp", "fastplaid", "gem", "hnswlib")
# The external-baseline overlay pass only draws MV-IVF + SVH from the core
# methods (drops muvera / vamana) alongside the external baselines, so the
# comparison stays focused on the two strongest in-house methods.
_EXTERNAL_METHOD_ORDER = ["mvivf", "svh_graph", "igp", "fastplaid", "gem", "hnswlib"]
_EXTERNAL_SUFFIX = "_external"
_IGP_FP_SUFFIX = "_external"


def _methods_for_plot(include_external: bool) -> list[str]:
    """Return the ordered method list the panel renderers should iterate."""
    if include_external:
        return list(_EXTERNAL_METHOD_ORDER)
    return list(_PLOTTED_METHOD_ORDER)


def _suffix_for(include_external: bool) -> str:
    return _EXTERNAL_SUFFIX if include_external else ""

# Pretty labels for the legend / x-tick names.
_PRETTY_METHOD = {
    "mvivf": "MV-IVF",
    "mvivf_spill": "MV-IVF Spill",
    "mvivf_flat": "MV-IVF Flat",
    "muvera": "MUVERA",
    "mpool": "Mean-Pool",
    "vamana": "MV-Vamana",
    "svh_ivf": "SVH IVF",
    "svh_graph": "SVH",
    "fastplaid": "FastPlaid",
    "igp": "IGP",
    "gem": "GEM",
    "hnswlib": "HNSWlib",
}

_PRETTY_VARIANT = {
    "tq1": "Rand-1bit",
    "1btq": "Rand-1bit",
    "onebittq": "Rand-1bit",
    "tq8": "Rand-8bit",
    "8btq": "Rand-8bit",
    "eightbittq": "Rand-8bit",
    "tq4": "TQ-4bit",
    "4btq": "TQ-4bit",
    "turboquant4": "TQ-4bit",
    "fastscan": "FastScan",
    "raw": "",
}


def _pretty_variant(variant: str | None) -> str:
    if not variant or variant == "raw":
        return ""
    key = str(variant).strip().lower().replace("-", "").replace("_", "")
    return _PRETTY_VARIANT.get(key, str(variant))


def _pretty_label(method: str, variant: str | None) -> str:
    return _PRETTY_METHOD.get(method, method)


def _legend(ax) -> None:
    ax.legend(
        loc="best",
        fontsize=15,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.35",
        borderpad=0.3,
        handlelength=1.6,
        handletextpad=0.35,
        labelspacing=0.2,
    )


def _dedup_legend_handles(axes) -> tuple[list, list[str]]:
    """Collect legend entries once, preserving the order used by the panels."""
    handles = []
    labels = []
    seen = set()
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label in seen:
                continue
            seen.add(label)
            handles.append(handle)
            labels.append(label)
    return handles, labels


def _top_legend_ncols(n_labels: int) -> int:
    """Keep the top legend compact without hard-coding the method count."""
    if n_labels <= 0:
        return 1
    if n_labels <= 5:
        return n_labels
    return (n_labels + 1) // 2


def _show_y_tick_labels(axes) -> None:
    """With shared y-axes, Matplotlib hides labels except on the left panel."""
    for ax in axes:
        ax.tick_params(axis="y", which="both", left=True, labelleft=True)


def _stage_title(stage: str, *, latency_ms: bool = False) -> str:
    if stage == "batch":
        return "Recall vs Throughput (Batch)"
    if stage == "multi_latency":
        return "Recall vs Latency (all thrds)"
    if stage == "latency":
        return "Recall vs Latency (1 thrd)" if latency_ms else "Recall vs QPS (1 thrd)"
    return f"Recall vs {stage}"


def _stage_ylabel(stage: str, *, latency_ms: bool = False) -> str:
    if latency_ms:
        return "Latency (ms)"
    if stage == "batch":
        return "QPS"
    return "QPS"


def _recall_xlabel(kind: str, k: int | None) -> str:
    if k is None:
        return r"Recall-$1$@$k$" if kind == "1" else r"Recall-$k$@$k$"
    return f"Recall-${kind}$@${k}$" if kind == "1" else f"Recall-${k}$@${k}$"


def _single_k(df: pd.DataFrame) -> int | None:
    if "k" not in df.columns:
        return None
    vals = sorted({int(v) for v in df["k"].dropna().astype(int).unique()})
    return vals[0] if len(vals) == 1 else None


def _is_combined_k_structured_run(df: pd.DataFrame, search: str | None) -> bool:
    if search is not None or "search" not in df.columns:
        return False
    searches = {str(v) for v in df["search"].dropna().unique()}
    return any(s.startswith("k=") for s in searches)

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
    "igp": {
        "search":     ["t_retrieval"],
        "filter":     ["t_filter"],
        "decode":     ["t_decode"],
        "rerank":     ["t_refine"],
    },
    "gem": {
        "filter":     ["t_filter"],
        "search":     ["t_graph_search"],
        "rerank":     ["t_rerank"],
    },
    "hnswlib": {
        "search":     ["t_graph_search"],
        "rerank":     ["t_rerank"],
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
    "filter":     "#55A467",
    "decode":     "#A7B85A",
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
               dataset: Optional[str] = None,
               *, include_external: bool = False) -> pd.DataFrame:
    """Load every CSV under ``<results_root>/<dataset>/`` whose filename
    matches the stage's ``prefix*.csv`` glob.

    When ``include_external`` is True we additionally pick up ``nr*.csv``
    files but only under ``<dataset>/fastplaid/``: FastPlaid's batch driver
    writes files named ``nr<num_rerank>.csv`` rather than the canonical
    ``batch_results_*.csv``, so the standard glob silently skips them. We
    scope the auxiliary glob to the fastplaid sub-tree so we don't pick up
    e.g. mvivf's per-num_rerank shards twice.
    """
    base = results_root / dataset if dataset else results_root
    if not base.exists():
        raise SystemExit(f"no results directory found at {base}")
    files = list(sorted(base.rglob(f"{prefix}*.csv")))
    if include_external:
        # FastPlaid's batch CSVs live at <ds>/fastplaid/<build>/<search>/nr*.csv.
        # Restrict the auxiliary glob to that sub-tree so we don't accidentally
        # double-count `nr*.csv` shards that some other method might write.
        fp_root = base / "fastplaid" if dataset else None
        fp_iter = (fp_root.rglob("nr*.csv") if fp_root and fp_root.exists()
                   else base.rglob("fastplaid/**/nr*.csv"))
        seen = set(files)
        for f in sorted(fp_iter):
            if f not in seen:
                files.append(f)
                seen.add(f)
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
    methods_override: Optional[list[str]] = None,
) -> None:
    method_order = methods_override if methods_override is not None else _PLOTTED_METHOD_ORDER
    methods_present = [m for m in method_order if m in df["method"].unique()]
    cmap = plt.colormaps.get_cmap("tab10")
    # Key color by position in the full _METHOD_ORDER so colors are stable
    # across datasets even when some methods are absent.
    color_for = {m: cmap(_METHOD_ORDER.index(m) % 10) for m in methods_present}
    y_label = _stage_ylabel(stage, latency_ms=False)

    for method in methods_present:
        sub_m = df[df["method"] == method]
        for vi, variant in enumerate(sorted(sub_m["variant"].unique())):
            sub = sub_m[sub_m["variant"] == variant].dropna(subset=[x_col, qps])
            if sub.empty:
                continue
            xs = sub[x_col].to_numpy(dtype=float)
            ys = sub[qps].to_numpy(dtype=float)
            fx, fy = _pareto_curve(xs, ys)
            if fx.size == 0:
                continue
            label = _pretty_label(method, variant)
            ls = "-" if vi == 0 else "--"
            ax.plot(
                fx,
                fy,
                marker="o",
                markersize=5.4,
                linewidth=2.8,
                linestyle=ls,
                color=color_for[method],
                label=label,
                zorder=3,
            )

    ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel(y_label)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.spines["top"].set_visible(False)
    if show_legend:
        _legend(ax)
    _pareto_panel_xlim(ax, df, methods_present, x_col, qps)


def _plot_pareto_latency_ms_panel(
    ax,
    df: pd.DataFrame,
    stage: str,
    qps: str,
    x_col: str,
    xlabel: str,
    *,
    show_ylabel: bool,
    show_legend: bool,
    methods_override: Optional[list[str]] = None,
) -> None:
    method_order = methods_override if methods_override is not None else _PLOTTED_METHOD_ORDER
    methods_present = [m for m in method_order if m in df["method"].unique()]
    cmap = plt.colormaps.get_cmap("tab10")
    # Key color by position in the full _METHOD_ORDER so colors are stable
    # across datasets even when some methods are absent.
    color_for = {m: cmap(_METHOD_ORDER.index(m) % 10) for m in methods_present}
    y_label = _stage_ylabel(stage, latency_ms=True)

    for method in methods_present:
        sub_m = df[df["method"] == method]
        for vi, variant in enumerate(sorted(sub_m["variant"].unique())):
            sub = sub_m[sub_m["variant"] == variant].dropna(subset=[x_col, qps])
            if sub.empty:
                continue
            xs = sub[x_col].to_numpy(dtype=float)
            qv = sub[qps].to_numpy(dtype=float)
            lat_ms = 1000.0 / np.maximum(qv, 1e-12)
            fx, fy = _pareto_curve_recall_latency_ms(xs, qv)
            if fx.size == 0:
                continue
            label = _pretty_label(method, variant)
            ls = "-" if vi == 0 else "--"
            ax.plot(
                fx,
                fy,
                marker="o",
                markersize=5.4,
                linewidth=2.8,
                linestyle=ls,
                color=color_for[method],
                label=label,
                zorder=3,
            )

    ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel(y_label)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.spines["top"].set_visible(False)
    if show_legend:
        _legend(ax)
    _pareto_panel_xlim(ax, df, methods_present, x_col, qps)


def _plot_pareto(df: pd.DataFrame, stage: str, dataset: str,
                 out_path: pathlib.Path,
                 *, methods_override: Optional[list[str]] = None) -> None:
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
        k_eval = _single_k(df)
        fig, (ax_left, ax_right) = plt.subplots(
            1, 2, figsize=(12, 5), sharey=True, layout="constrained"
        )
        _plot_pareto_panel(
            ax_left,
            df,
            stage,
            qps,
            "recall_1_k",
            _recall_xlabel("1", k_eval),
            show_ylabel=True,
            show_legend=False,
            methods_override=methods_override,
        )
        _plot_pareto_panel(
            ax_right,
            df,
            stage,
            qps,
            "recall_k_k",
            _recall_xlabel("k", k_eval),
            show_ylabel=False,
            show_legend=True,
            methods_override=methods_override,
        )
    elif has_left:
        fig, ax_one = plt.subplots(figsize=(7.2, 5.2), layout="constrained")
        _plot_pareto_panel(
            ax_one,
            df,
            stage,
            qps,
            "recall_1_k",
            _recall_xlabel("1", _single_k(df)),
            show_ylabel=True,
            show_legend=True,
            methods_override=methods_override,
        )
    else:
        fig, ax_one = plt.subplots(figsize=(7.2, 5.2), layout="constrained")
        _plot_pareto_panel(
            ax_one,
            df,
            stage,
            qps,
            "recall_k_k",
            _recall_xlabel("k", _single_k(df)),
            show_ylabel=True,
            show_legend=True,
            methods_override=methods_override,
        )

    fig.suptitle(f"{dataset}: {_stage_title(stage)}", fontsize=17)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _plot_pareto_latency_ms(df: pd.DataFrame, stage: str, dataset: str,
                          out_path: pathlib.Path,
                          *, methods_override: Optional[list[str]] = None) -> None:
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
            stage,
            qps,
            "recall_1_k",
            _recall_xlabel("1", _single_k(df)),
            show_ylabel=True,
            show_legend=False,
            methods_override=methods_override,
        )
        _plot_pareto_latency_ms_panel(
            ax_right,
            df,
            stage,
            qps,
            "recall_k_k",
            _recall_xlabel("k", _single_k(df)),
            show_ylabel=False,
            show_legend=True,
            methods_override=methods_override,
        )
    elif has_left:
        fig, ax_one = plt.subplots(figsize=(7.2, 5.2), layout="constrained")
        _plot_pareto_latency_ms_panel(
            ax_one,
            df,
            stage,
            qps,
            "recall_1_k",
            _recall_xlabel("1", _single_k(df)),
            show_ylabel=True,
            show_legend=True,
            methods_override=methods_override,
        )
    else:
        fig, ax_one = plt.subplots(figsize=(7.2, 5.2), layout="constrained")
        _plot_pareto_latency_ms_panel(
            ax_one,
            df,
            stage,
            qps,
            "recall_k_k",
            _recall_xlabel("k", _single_k(df)),
            show_ylabel=True,
            show_legend=True,
            methods_override=methods_override,
        )

    fig.suptitle(
        f"{dataset}: {_stage_title(stage, latency_ms=True)}",
        fontsize=17,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _has_recall_data(df: pd.DataFrame, x_col: str, y_col: str) -> bool:
    return x_col in df.columns and y_col in df.columns and not df.dropna(
        subset=[x_col, y_col]
    ).empty


def _plot_paper_four_panel(
    df: pd.DataFrame,
    stage: str,
    dataset: str,
    out_path: pathlib.Path,
    *,
    latency_ms: bool,
    methods_override: Optional[list[str]] = None,
) -> None:
    """Paper-facing one-row plot:
    R1@10, R10@10, R100@100.

    Emitted only when both k=10 and k=100 are present.
    """
    qps = _qps_column(stage)
    if "k" not in df.columns or qps not in df.columns:
        return
    k_vals = {int(v) for v in df["k"].dropna().astype(int).unique()}
    if not {10, 100}.issubset(k_vals):
        return

    specs = [
        (10, "recall_1_k", _recall_xlabel("1", 10)),
        (10, "recall_k_k", _recall_xlabel("k", 10)),
        (100, "recall_k_k", _recall_xlabel("k", 100)),
    ]
    for k_eval, x_col, _ in specs:
        sub = df[df["k"].astype(int) == k_eval]
        if not _has_recall_data(sub, x_col, qps):
            return

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.4, 4.05),
        sharey=True,
    )
    for i, (k_eval, x_col, xlabel) in enumerate(specs):
        sub = df[df["k"].astype(int) == k_eval]
        if latency_ms:
            _plot_pareto_latency_ms_panel(
                axes[i],
                sub,
                stage,
                qps,
                x_col,
                xlabel,
                show_ylabel=(i == 0),
                show_legend=False,
                methods_override=methods_override,
            )
        else:
            _plot_pareto_panel(
                axes[i],
                sub,
                stage,
                qps,
                x_col,
                xlabel,
                show_ylabel=(i == 0),
                show_legend=False,
                methods_override=methods_override,
            )

    _show_y_tick_labels(axes)
    fig.suptitle(
        f"{dataset}: {_stage_title(stage, latency_ms=latency_ms)}",
        fontsize=17,
        y=0.985,
    )
    handles, labels = _dedup_legend_handles(axes)
    if handles:
        ncols = _top_legend_ncols(len(labels))
        legend_rows = (len(labels) + ncols - 1) // ncols
        axes_top = 0.81 if legend_rows == 1 else 0.70
        fig.subplots_adjust(
            left=0.08,
            right=0.975,
            bottom=0.16,
            top=axes_top,
            wspace=0.28,
        )
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            ncol=ncols,
            fontsize=13,
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="0.35",
            borderpad=0.3,
            handlelength=1.6,
            handletextpad=0.35,
            columnspacing=0.8,
            labelspacing=0.2,
        )
    else:
        fig.subplots_adjust(
            left=0.08,
            right=0.975,
            bottom=0.16,
            top=0.84,
            wspace=0.28,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _plot_paper_recall_k_individual(
    df: pd.DataFrame,
    stage: str,
    dataset: str,
    out_path: pathlib.Path,
    *,
    k_eval: int = 10,
    methods_override: Optional[list[str]] = None,
) -> None:
    """Single-panel 'paper' figure: Recall-``k_eval``@``k_eval`` vs QPS only,
    titled by the plain dataset name (no ``Recall vs Throughput`` suptitle).
    ``df`` may span multiple search configs (e.g. k=10 and k=100); rows are
    filtered down to ``k == k_eval`` here.
    """
    qps = _qps_column(stage)
    if qps not in df.columns or "recall_k_k" not in df.columns or "k" not in df.columns:
        print(f"  [skip paper recall-k@k for {dataset}: missing columns]")
        return
    sub = df[df["k"].astype(int) == k_eval].dropna(subset=["recall_k_k", qps])
    if sub.empty:
        print(f"  [skip paper recall-k@k for {dataset}: no k={k_eval} rows]")
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.2), layout="constrained")
    _plot_pareto_panel(
        ax, sub, stage, qps, "recall_k_k", _recall_xlabel("k", k_eval),
        show_ylabel=True, show_legend=True, methods_override=methods_override,
    )
    ax.set_title(dataset, fontsize=17)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _plot_paper_recall_k_combined(
    dfs: "dict[str, pd.DataFrame]",
    stage: str,
    out_path: pathlib.Path,
    *,
    k_eval: int = 10,
    methods_override: Optional[list[str]] = None,
) -> None:
    """Multi-panel 'paper' figure: one Recall-``k_eval``@``k_eval``-vs-QPS
    panel per dataset, each titled by its plain dataset name (no
    figure-level suptitle)."""
    qps = _qps_column(stage)
    datasets = list(dfs.keys())
    n = len(datasets)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(5.7 * n, 4.4), sharey=True)
    if n == 1:
        axes = [axes]
    for i, ds in enumerate(datasets):
        df = dfs[ds]
        sub = df[df["k"].astype(int) == k_eval] if "k" in df.columns else df.iloc[0:0]
        _plot_pareto_panel(
            axes[i], sub, stage, qps, "recall_k_k", _recall_xlabel("k", k_eval),
            show_ylabel=(i == 0), show_legend=False, methods_override=methods_override,
        )
        axes[i].xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune=None))
        axes[i].xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        axes[i].set_title(ds, fontsize=17, pad=10)
    _show_y_tick_labels(axes)
    handles, labels = _dedup_legend_handles(axes)
    if handles:
        ncols = _top_legend_ncols(len(labels))
        legend_rows = (len(labels) + ncols - 1) // ncols
        axes_top = 0.74 if legend_rows == 1 else 0.62
        fig.subplots_adjust(left=0.08, right=0.975, bottom=0.16, top=axes_top, wspace=0.10)
        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=ncols,
            fontsize=17,
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="0.35",
            borderpad=0.4,
            handlelength=2.0,
            handletextpad=0.5,
            columnspacing=1.1,
            labelspacing=0.3,
        )
    else:
        fig.subplots_adjust(left=0.08, right=0.975, bottom=0.16, top=0.88, wspace=0.28)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.10)
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
    methods_in_order = [m for m in _PLOTTED_METHOD_ORDER if m in best["method"].unique()]
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
            nice = _pretty_label(method, variant).replace(" / ", "\n")
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
    ax.set_xticklabels(bar_labels, rotation=15, ha="right", fontsize=13)
    ax.set_ylabel("Per-query time (s) @ best recall")
    ax.set_title(f"{dataset}: timer breakdown [{stage}]")
    ax.legend(
        loc="best",
        fontsize=13,
        ncol=1,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.35",
    )
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
                   help=("default: <results>/_plots, or <results>/_plots/<search> "
                         "when --search is given so different k values stay in "
                         "sibling subfolders."))
    p.add_argument("--datasets", default=None,
                   help="comma-separated dataset names; default: all directories under --results")
    p.add_argument(
        "--search", default=None,
        help=(
            "Filter rows to a single search-config name (the leaf directory "
            "under <variant>/, e.g. ``k=10`` or ``k=100``). When set and "
            "--out-dir is not given, plots land under <results>/_plots/<search>/ "
            "so per-k PDFs do not collide."
        ),
    )
    p.add_argument(
        "--include-external", "--include_external",
        action="store_true", dest="include_external",
        help=(
            "Generate an additional pass of every plot that includes external "
            "baselines (igp, fastplaid, gem, hnswlib). Writes "
            "``<dataset>_<plot>_external.pdf`` siblings beside the default "
            "PDFs. Skipped silently when no external baseline has rows."
        ),
    )
    p.add_argument(
        "--include-igp-fp", "--include_igp_fp",
        action="store_true", dest="include_external",
        help="Deprecated alias for --include-external.",
    )
    p.add_argument(
        "--paper-recall-k-pair", "--paper_recall_k_pair",
        dest="paper_recall_k_pair", default=None,
        help=(
            "Comma-separated dataset names (2+) to additionally render as a "
            "Recall-k@k-only 'paper' figure using the external method set "
            "(mvivf, svh_graph, igp, fastplaid): one single-panel PDF per "
            "dataset (``<ds>_paper_pareto_external.pdf``) plus one combined "
            "multi-panel PDF (``<ds1>_<ds2>..._paper_pareto_external.pdf``) "
            "with each panel titled by its plain dataset name. Requires "
            "--include-external and --stage batch; skipped otherwise."
        ),
    )
    args = p.parse_args()

    prefix = _PREFIX_FOR_STAGE[args.stage]
    if args.out_dir is not None:
        out_dir = args.out_dir
    elif args.search:
        out_dir = args.results / "_plots" / args.search
    else:
        out_dir = args.results / "_plots"
    with open(args.methods, encoding="utf-8") as f:
        methods_yaml = yaml.safe_load(f) or {}

    datasets = (
        [d.strip() for d in args.datasets.split(",") if d.strip()]
        if args.datasets else _list_datasets(args.results)
    )
    if not datasets:
        print(f"[warn] no datasets under {args.results}; nothing to plot.")
        return 0

    # Per-pass closures so the default + opt-in passes share rendering code.
    def _emit_plots(ds: str, df: pd.DataFrame, *,
                    methods_override: Optional[list[str]] = None,
                    file_suffix: str = "") -> None:
        combined_k_structured = _is_combined_k_structured_run(df, args.search)
        if combined_k_structured:
            print(
                f"  [skip legacy combined plots for {ds}: k-structured data; "
                "paper plots require both k=10 and k=100]"
            )
        else:
            _plot_pareto(
                df, args.stage, ds,
                out_dir / f"{ds}_pareto{file_suffix}.pdf",
                methods_override=methods_override,
            )
        if args.stage == "batch":
            _plot_paper_four_panel(
                df,
                args.stage,
                ds,
                out_dir / f"{ds}_paper_pareto{file_suffix}.pdf",
                latency_ms=False,
                methods_override=methods_override,
            )
        if args.stage in ("latency", "multi_latency"):
            if not combined_k_structured:
                _plot_pareto_latency_ms(
                    df, args.stage, ds,
                    out_dir / f"{ds}_pareto_latency_ms{file_suffix}.pdf",
                    methods_override=methods_override,
                )
            _plot_paper_four_panel(
                df,
                args.stage,
                ds,
                out_dir / f"{ds}_paper_pareto_latency_ms{file_suffix}.pdf",
                latency_ms=True,
                methods_override=methods_override,
            )

    def _filter_search(ds: str, df: pd.DataFrame) -> pd.DataFrame | None:
        if not args.search:
            return df
        available = sorted(df["search"].dropna().unique().tolist())
        df = df[df["search"] == args.search]
        if df.empty:
            print(f"[skip {ds}] no rows with search={args.search!r} "
                  f"(available under {ds}: {available or 'n/a'})")
            return None
        return df

    paper_recall_k_pair = (
        [d.strip() for d in args.paper_recall_k_pair.split(",") if d.strip()]
        if args.paper_recall_k_pair else []
    )
    paper_recall_k_dfs: dict[str, pd.DataFrame] = {}

    for ds in datasets:
        # ---- Default pass: canonical core methods only. -----------------
        try:
            df = _load_csvs(args.results, prefix, dataset=ds)
        except SystemExit as e:
            print(f"[skip {ds}] {e}")
        else:
            df_default = _filter_search(ds, df)
            if df_default is not None:
                print(f"[{ds}] {len(df_default)} rows from "
                      f"{df_default['method'].nunique()} methods"
                      + (f" (search={args.search})" if args.search else ""))
                _emit_plots(ds, df_default)

        # ---- Opt-in pass: add igp + fastplaid (skips silently if absent). ----
        if not args.include_external:
            continue
        try:
            df_ext = _load_csvs(
                args.results, prefix, dataset=ds, include_external=True,
            )
        except SystemExit as e:
            print(f"[skip {ds} +external] {e}")
            continue
        df_ext = _filter_search(ds, df_ext)
        if df_ext is None:
            continue
        present = set(df_ext.get("method", pd.Series(dtype=str)).unique())
        baselines_present = present & set(_OPTIONAL_BASELINE_METHODS)
        if not baselines_present:
            print(f"[skip {ds} +external] no external baseline rows present")
            continue
        print(f"[{ds} +external] {len(df_ext)} rows from "
              f"{df_ext['method'].nunique()} methods (added: "
              f"{sorted(baselines_present)})"
              + (f" (search={args.search})" if args.search else ""))
        _emit_plots(
            ds, df_ext,
            methods_override=_methods_for_plot(True),
            file_suffix=_EXTERNAL_SUFFIX,
        )
        if ds in paper_recall_k_pair:
            paper_recall_k_dfs[ds] = df_ext

    if paper_recall_k_pair:
        if args.stage != "batch" or not args.include_external:
            print(
                "[skip paper-recall-k-pair] requires --stage batch and "
                "--include-external"
            )
        else:
            missing = [d for d in paper_recall_k_pair if d not in paper_recall_k_dfs]
            if missing:
                print(
                    f"[skip paper-recall-k-pair] no external data for: {missing}"
                )
            else:
                methods_override = _methods_for_plot(True)
                for ds in paper_recall_k_pair:
                    _plot_paper_recall_k_individual(
                        paper_recall_k_dfs[ds], args.stage, ds,
                        out_dir / f"{ds}_paper_pareto_r10_external.pdf",
                        methods_override=methods_override,
                    )
                ordered = {d: paper_recall_k_dfs[d] for d in paper_recall_k_pair}
                combo_name = "_".join(paper_recall_k_pair)
                _plot_paper_recall_k_combined(
                    ordered, args.stage,
                    out_dir / f"{combo_name}_paper_pareto_r10_external.pdf",
                    methods_override=methods_override,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
