#!/usr/bin/env python3
"""MV-IVF stacked-bar timer breakdown across datasets.

For a given stage (`latency` / `multi_latency`), this script collects
the fastest MV-IVF row per dataset that reaches a fixed Recall-k@k target and
draws a single stacked bar chart
where each bar is one dataset and each colored segment is one logical stage:

    quantize/compress = query compression + query quantization
    greedy search     = full greedy routing/probe search wall time
    leaf probe/sort   = leaf scoring and leaf-local overhead
    rerank            = exact/asymmetric reranking

Pass ``--datasets`` to control which datasets appear. Without it, every
top-level dataset directory under ``--results`` is used.

Latency CSVs already contain per-stage columns because
``compute_stats_latency`` / ``compute_stats_multi_latency`` populate
``avg_timings`` according to the labels in ``benchmarks/methods.yaml``.

The batch path does not emit these substage timers, so batch is intentionally
unsupported here.

Output:

    <out_dir>/mvivf_breakdown_<stage>.pdf
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

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


_PREFIX_FOR_STAGE = {
    "latency":       "latency_",
    "multi_latency": "multi_latency_",
}


# Logical stage -> list of CSV columns to sum.  Keep the keys ordered; that
# order also drives the stacking order (bottom -> top).
_LATENCY_SEGMENTS: dict[str, list[str]] = {
    "Quantize+compress": ["t_compress", "t_quant"],
    "Greedy search":     ["t_greedy"],
    "Leaf probe+sort":   ["t_leaf_dists", "t_leaf_rest", "t_leaf_dedup"],
    "Rerank":            ["t_rerank"],
}

_GREEDY_FALLBACK = [
    "t_search_dists",
    "t_search_beam",
    "t_search_rest",
    "t_search_top_level",
    "t_search",
]

_VIDORE_DATASETS = {
    "docvqa",
    "infovqa",
    "arxivqa",
    "tabfquad",
    "chartqa",
    "shiftproject",
    "synth_ai",
    "synth_energy",
    "synth_gov",
    "synth_healthcare",
    "tatdqa",
}

_SEGMENT_COLORS: dict[str, str] = {
    "Quantize+compress": "#4C72B0",
    "Greedy search":     "#DD8452",
    "Leaf probe+sort":   "#55A467",
    "Rerank":            "#C44E52",
}

_BREAKDOWN_GROUPS: list[tuple[str, list[str]]] = [
    ("Small BEIR", ["arguana", "fiqa", "nfcorpus", "scidocs", "scifact"]),
    ("Large / pooled", ["hotpotqa", "msmarco", "nq", "lotte"]),
]


def _annotate_path(results_root: pathlib.Path, csv_file: pathlib.Path) -> dict:
    """Recover dataset/method/build/variant/search from path layout
    ``<root>/<dataset>/<method>/<build>/[<variant>/]<search>/<csv>``.
    """
    rel = csv_file.relative_to(results_root)
    parts = rel.parts
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


def _load_mvivf(results_root: pathlib.Path, prefix: str,
                dataset: str) -> Optional[pd.DataFrame]:
    base = results_root / dataset
    if not base.exists():
        print(f"  [skip {dataset}] no directory at {base}")
        return None
    files = sorted((base / "mvivf").rglob(f"{prefix}*.csv")) if (base / "mvivf").exists() else []
    if not files:
        print(f"  [skip {dataset}] no mvivf {prefix}*.csv under {base}")
        return None

    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue
        meta = _annotate_path(results_root, f)
        for k, v in meta.items():
            df[k] = v
        rows.append(df)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def _selected_row(df: pd.DataFrame, target_recall: float) -> Optional[pd.Series]:
    """Pick the fastest MV-IVF operating point at a fixed recall target.

    This is more informative for a breakdown plot than choosing the extreme
    highest-recall row, which often just selects the largest num_rerank file.
    If a future dataset misses the target, fall back to the highest-recall row.
    """
    if "recall_k_k" not in df.columns or df["recall_k_k"].dropna().empty:
        return None
    qps_col = "QPS_seq"
    recall = df["recall_k_k"].astype(float)
    candidates = df[recall >= target_recall]
    if not candidates.empty and qps_col in candidates.columns:
        return candidates.loc[candidates[qps_col].astype(float).idxmax()]
    return df.loc[recall.idxmax()]


def _segments_for_row(row: pd.Series, stage: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for seg, cols in _LATENCY_SEGMENTS.items():
        present = [c for c in cols if c in row.index]
        vals = [float(row[c]) for c in present if pd.notna(row[c])]
        if vals:
            out[seg] = sum(vals)
    if "Greedy search" not in out:
        vals = [
            float(row[c]) for c in _GREEDY_FALLBACK
            if c in row.index and pd.notna(row[c])
        ]
        if vals:
            out["Greedy search"] = sum(vals)
    return out


def _normalize_per_query(seg: dict[str, float], n_queries_estimate: float) -> dict[str, float]:
    """Latency CSVs sum ``avg_timings`` across all queries (one repetition).

    Convert to per-query seconds for readability so bars are comparable across
    datasets of different sizes.  We approximate the query count from
    ``QPS_seq * total_time`` if available; otherwise we leave the data as-is.
    """
    if n_queries_estimate <= 0:
        return {k: v * 1000.0 for k, v in seg.items()}
    return {k: (v / n_queries_estimate) * 1000.0 for k, v in seg.items()}


def _estimate_num_queries(row: pd.Series) -> float:
    qps = float(row.get("QPS_seq", 0.0) or 0.0)
    if qps <= 0:
        return 0.0
    total = sum(_segments_for_row(row, stage="latency").values())
    if total <= 0:
        return 0.0
    return qps * total


def _plot(stage: str, per_dataset: dict[str, dict[str, float]],
          out_path: pathlib.Path) -> None:
    if not per_dataset:
        print(f"  [skip plot] nothing to draw for stage={stage}")
        return

    seg_order = [s for s in _LATENCY_SEGMENTS.keys()
                 if any(s in per_dataset[d] for d in per_dataset)]
    group_datasets = [
        (label, [d for d in datasets if d in per_dataset])
        for label, datasets in _BREAKDOWN_GROUPS
    ]
    placed = {d for _, datasets in group_datasets for d in datasets}
    leftovers = [d for d in per_dataset if d not in placed]
    if leftovers:
        label, datasets = group_datasets[-1]
        group_datasets[-1] = (label, datasets + leftovers)
    group_datasets = [(label, datasets) for label, datasets in group_datasets if datasets]

    fig, axes = plt.subplots(
        1,
        len(group_datasets),
        figsize=(7.2, 3.4),
        width_ratios=[len(datasets) for _, datasets in group_datasets],
        layout="constrained",
        squeeze=False,
    )
    handles = labels = None
    for ax, (group_label, datasets) in zip(axes[0, :], group_datasets):
        x = np.arange(len(datasets))
        bottoms = np.zeros(len(datasets), dtype=float)
        for seg in seg_order:
            heights = np.array(
                [float(per_dataset[d].get(seg, 0.0)) for d in datasets], dtype=float
            )
            ax.bar(
                x,
                heights,
                bottom=bottoms,
                label=seg,
                color=_SEGMENT_COLORS.get(seg, None),
                width=0.6,
                edgecolor="white",
                linewidth=0.6,
            )
            bottoms += heights
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=25, ha="right", fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_ylabel("Latency (ms)", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()

    if handles and labels:
        legend_kwargs = dict(
            handles=handles,
            labels=labels,
            ncol=len(labels),
            fontsize=9,
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="0.35",
        )
        try:
            fig.legend(loc="outside lower center", **legend_kwargs)
        except ValueError:
            fig.set_layout_engine(None)
            fig.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 0.02),
                **legend_kwargs,
            )
            fig.subplots_adjust(
                left=0.10,
                right=0.98,
                top=0.82,
                bottom=0.34,
                wspace=0.28,
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _list_datasets(results_root: pathlib.Path, *, include_vidore: bool) -> list[str]:
    if not results_root.exists():
        return []
    out = []
    for p in sorted(results_root.iterdir()):
        if not p.is_dir() or p.name.startswith(".") or p.name.startswith("_"):
            continue
        if not include_vidore and p.name in _VIDORE_DATASETS:
            continue
        out.append(p.name)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=list(_PREFIX_FOR_STAGE))
    p.add_argument("--results", required=True, type=pathlib.Path,
                   help="experiments/<stage>/results/")
    p.add_argument("--datasets", default=None,
                   help="comma-separated dataset names; default: all non-ViDoRe datasets under --results")
    p.add_argument("--include-vidore", action="store_true",
                   help="include ViDoRe datasets when --datasets is omitted")
    p.add_argument("--target-recall", type=float, default=0.90,
                   help="select the highest-QPS MV-IVF row with recall_k_k at or above this value")
    p.add_argument("--out-dir", type=pathlib.Path, default=None,
                   help="default: <results>/_plots")
    p.add_argument("--methods", type=pathlib.Path, default=None,
                   help="(unused; kept for parity with plot_stage.py)")
    args = p.parse_args()

    prefix = _PREFIX_FOR_STAGE[args.stage]
    out_dir = args.out_dir if args.out_dir else args.results / "_plots"
    datasets = (
        [d.strip() for d in args.datasets.split(",") if d.strip()]
        if args.datasets else _list_datasets(args.results, include_vidore=args.include_vidore)
    )
    if not datasets:
        print("[warn] no datasets requested; nothing to plot.")
        return 0

    print(f"[mvivf-breakdown stage={args.stage}] datasets: {datasets}")
    print(f"[mvivf-breakdown stage={args.stage}] target recall_k_k: {args.target_recall:.3f}")
    per_dataset: dict[str, dict[str, float]] = {}
    for ds in datasets:
        df = _load_mvivf(args.results, prefix, ds)
        if df is None or df.empty:
            continue
        row = _selected_row(df, args.target_recall)
        if row is None:
            print(f"  [skip {ds}] no recall_k_k rows")
            continue
        seg = _segments_for_row(row, args.stage)
        if not seg:
            print(f"  [skip {ds}] no usable timer columns "
                  f"(stage={args.stage}; "
                  f"variant={row.get('variant')}, build={row.get('build')})")
            continue
        if args.stage != "batch":
            n_q = _estimate_num_queries(row)
            seg = _normalize_per_query(seg, n_q)
        per_dataset[ds] = seg
        recall = float(row.get("recall_k_k", float("nan")))
        qps = float(row.get("QPS_seq", float("nan")))
        nr = row.get("num_rerank", "n/a")
        csv = row.get("_csv", "n/a")
        print(f"  [{ds}] selected recall_k_k={recall:.3f}, QPS_seq={qps:.3f}, "
              f"num_rerank={nr}, file={csv}; "
              f"segments={ {k: round(v, 6) for k, v in seg.items()} }")

    out_path = out_dir / f"mvivf_breakdown_{args.stage}.pdf"
    _plot(args.stage, per_dataset, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
