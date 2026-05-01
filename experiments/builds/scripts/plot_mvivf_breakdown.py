#!/usr/bin/env python3
"""MVIVF stacked-bar timer breakdown across datasets.

For a given stage (`latency` / `multi_latency` / `batch`), this script collects
the **best-recall** MVIVF row per dataset and draws a single stacked bar chart
where each bar is one dataset and each colored segment is one logical stage:

    encode      = query compression + query quantization (= ``t_quant``)
    greedy      = greedy / flat probe search
                   (= ``t_search_dists`` + ``t_search_beam`` + ``t_search_rest``)
    leaf_probe  = leaf scoring (= ``t_leaf_dists`` + ``t_leaf_rest``)
    rerank      = ``t_rerank``

The set of datasets included in the plot is the constant ``DATASETS`` at the
top of this file -- edit that list (or pass ``--datasets``) to change which
datasets show up.  Defaults to ``beir5, nq, hotpotqa``.

Latency CSVs already contain per-stage columns because
``compute_stats_latency`` / ``compute_stats_multi_latency`` populate
``avg_timings`` according to the labels in ``benchmarks/methods.yaml``.

The batch path (``compute_stats_batch``) only times ``search_all`` end-to-end
(``avg_timings`` is empty), so for ``--stage batch`` we draw a single
``total`` bar per dataset using ``1.0 / QPS_par`` (mean per-query time at full
parallelism).  This keeps the script useful even though the batch path does
not yet emit a per-substage breakdown.

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


# Edit this to change the default set of datasets in the plot.
DATASETS: list[str] = ["beir5", "nq", "hotpotqa"]


_PREFIX_FOR_STAGE = {
    "latency":       "latency_",
    "multi_latency": "multi_latency_",
    "batch":         "batch_",
}


# Logical stage -> list of CSV columns to sum.  Keep the keys ordered; that
# order also drives the stacking order (bottom -> top).
_LATENCY_SEGMENTS: dict[str, list[str]] = {
    "encode":     ["t_quant"],  # query compression + quantization
    "greedy":     ["t_search_dists", "t_search_beam", "t_search_rest"],
    "leaf_probe": ["t_leaf_dists", "t_leaf_rest"],
    "rerank":     ["t_rerank"],
}

_SEGMENT_COLORS: dict[str, str] = {
    "encode":     "#4C72B0",
    "greedy":     "#DD8452",
    "leaf_probe": "#55A467",
    "rerank":     "#C44E52",
    "total":      "#8172B3",
}


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


def _best_row(df: pd.DataFrame) -> Optional[pd.Series]:
    if "recall_k_k" not in df.columns or df["recall_k_k"].dropna().empty:
        return None
    return df.loc[df["recall_k_k"].astype(float).idxmax()]


def _segments_for_row(row: pd.Series, stage: str) -> dict[str, float]:
    if stage == "batch":
        qps = float(row.get("QPS_par", 0.0) or 0.0)
        if qps <= 0:
            return {}
        # Per-query wall time at full parallelism. Approximation: total batch
        # time / num_queries == 1 / QPS_par.
        return {"total": 1.0 / qps}
    out: dict[str, float] = {}
    for seg, cols in _LATENCY_SEGMENTS.items():
        present = [c for c in cols if c in row.index]
        vals = [float(row[c]) for c in present if pd.notna(row[c])]
        if vals:
            out[seg] = sum(vals)
    return out


def _normalize_per_query(seg: dict[str, float], n_queries_estimate: float) -> dict[str, float]:
    """Latency CSVs sum ``avg_timings`` across all queries (one repetition).

    Convert to per-query seconds for readability so bars are comparable across
    datasets of different sizes.  We approximate the query count from
    ``QPS_seq * total_time`` if available; otherwise we leave the data as-is.
    """
    if n_queries_estimate <= 0:
        return seg
    return {k: v / n_queries_estimate for k, v in seg.items()}


def _estimate_num_queries(row: pd.Series) -> float:
    qps = float(row.get("QPS_seq", 0.0) or 0.0)
    if qps <= 0:
        return 0.0
    total = sum(
        float(row[c]) for cols in _LATENCY_SEGMENTS.values() for c in cols
        if c in row.index and pd.notna(row[c])
    )
    if total <= 0:
        return 0.0
    return qps * total


def _plot(stage: str, per_dataset: dict[str, dict[str, float]],
          out_path: pathlib.Path) -> None:
    if not per_dataset:
        print(f"  [skip plot] nothing to draw for stage={stage}")
        return

    datasets = list(per_dataset.keys())
    seg_order: list[str]
    if stage == "batch":
        seg_order = ["total"]
    else:
        seg_order = [s for s in _LATENCY_SEGMENTS.keys()
                     if any(s in per_dataset[d] for d in datasets)]

    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(datasets) + 2), 5))
    bottoms = np.zeros(len(datasets), dtype=float)
    x = np.arange(len(datasets))
    for seg in seg_order:
        heights = np.array(
            [float(per_dataset[d].get(seg, 0.0)) for d in datasets], dtype=float
        )
        ax.bar(x, heights, bottom=bottoms, label=seg,
               color=_SEGMENT_COLORS.get(seg, None), width=0.6,
               edgecolor="white", linewidth=0.6)
        bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Per-query time (s)" if stage != "batch"
                  else "Mean per-query time at QPS_par (s)")
    ax.set_title(f"MVIVF stage breakdown @ best recall [{stage}]")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=list(_PREFIX_FOR_STAGE))
    p.add_argument("--results", required=True, type=pathlib.Path,
                   help="experiments/<stage>/results/")
    p.add_argument("--datasets", default=None,
                   help="comma-separated dataset names; "
                        f"default from DATASETS in this file ({','.join(DATASETS)})")
    p.add_argument("--out-dir", type=pathlib.Path, default=None,
                   help="default: <results>/_plots")
    p.add_argument("--methods", type=pathlib.Path, default=None,
                   help="(unused; kept for parity with plot_stage.py)")
    args = p.parse_args()

    prefix = _PREFIX_FOR_STAGE[args.stage]
    out_dir = args.out_dir if args.out_dir else args.results / "_plots"
    datasets = (
        [d.strip() for d in args.datasets.split(",") if d.strip()]
        if args.datasets else list(DATASETS)
    )
    if not datasets:
        print("[warn] no datasets requested; nothing to plot.")
        return 0

    print(f"[mvivf-breakdown stage={args.stage}] datasets: {datasets}")
    per_dataset: dict[str, dict[str, float]] = {}
    for ds in datasets:
        df = _load_mvivf(args.results, prefix, ds)
        if df is None or df.empty:
            continue
        row = _best_row(df)
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
        print(f"  [{ds}] best recall_k_k={recall:.3f}; "
              f"segments={ {k: round(v, 6) for k, v in seg.items()} }")

    out_path = out_dir / f"mvivf_breakdown_{args.stage}.pdf"
    _plot(args.stage, per_dataset, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
