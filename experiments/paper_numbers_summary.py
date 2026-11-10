#!/usr/bin/env python3
"""Compute paper-ready experiment numbers from experiments/ results.

Outputs per-dataset and combined summaries for:
- latency      (latency-ms vs recall regime)
- batch        (QPS vs recall regime)
- multi_latency (optional appendix regime)

Also parses build/index-size report and computes relative factors.

Usage:
  python3 experiments/paper_numbers_summary.py
  python3 experiments/paper_numbers_summary.py --targets 0.9,0.95
  python3 experiments/paper_numbers_summary.py --stages latency,batch
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PREFIX_FOR_STAGE = {
    "latency": "latency_",
    "multi_latency": "multi_latency_",
    "batch": "batch_",
}

QPS_COL_FOR_STAGE = {
    "latency": "QPS_seq",
    "multi_latency": "QPS_par",
    "batch": "QPS_par",
}

BEIR_DATASETS = {
    "nfcorpus",
    "scifact",
    "arguana",
    "scidocs",
    "fiqa",
    "quora",
    "nq",
    "hotpotqa",
    "msmarco",
    "nq500k",
}

VIDORE_DATASETS = {
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
}


def dataset_group(ds: str) -> str:
    if ds in BEIR_DATASETS:
        return "beir"
    if ds in VIDORE_DATASETS:
        return "vidore"
    return "other"


def safe_geomean(vals: Iterable[float]) -> float:
    arr = np.asarray([v for v in vals if np.isfinite(v) and v > 0], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(arr))))


def read_stage_rows(
    repo_root: Path, stage: str, methods: set[str], recall_col: str = "recall_k_k"
) -> pd.DataFrame:
    base = repo_root / "experiments" / stage / "results"
    prefix = PREFIX_FOR_STAGE[stage]
    qps_col = QPS_COL_FOR_STAGE[stage]
    rows = []
    if not base.exists():
        return pd.DataFrame()

    for p in base.rglob(f"{prefix}*.csv"):
        parts = p.relative_to(base).parts
        if len(parts) < 6:
            continue
        ds, method = parts[0], parts[1]
        if methods and method not in methods:
            continue
        build, variant, search = parts[2], parts[3], parts[4]
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        if recall_col not in df.columns or qps_col not in df.columns:
            continue
        df = df.copy()
        df["stage"] = stage
        df["dataset"] = ds
        df["dataset_group"] = dataset_group(ds)
        df["method"] = method
        df["build_name"] = build
        df["variant"] = variant
        df["search_name"] = search
        df["source_csv"] = str(p)
        df["qps_effective"] = pd.to_numeric(df[qps_col], errors="coerce")
        df["latency_ms"] = np.where(df["qps_effective"] > 0, 1000.0 / df["qps_effective"], np.nan)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out[recall_col] = pd.to_numeric(out[recall_col], errors="coerce")
    return out.dropna(subset=[recall_col, "qps_effective"])


def per_dataset_summary(df: pd.DataFrame, targets: list[float], recall_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    g = df.groupby(["stage", "dataset_group", "dataset", "method"], dropna=False)
    base = g.agg(
        n_points=("qps_effective", "count"),
        best_recall=(recall_col, "max"),
        best_qps=("qps_effective", "max"),
        min_latency_ms=("latency_ms", "min"),
    ).reset_index()

    target_rows = []
    for (stage, dsg, ds, method), sub in g:
        for t in targets:
            good = sub[sub[recall_col] >= t]
            qps_at = float(good["qps_effective"].max()) if not good.empty else float("nan")
            lat_at = float(good["latency_ms"].min()) if not good.empty else float("nan")
            target_rows.append(
                {
                    "stage": stage,
                    "dataset_group": dsg,
                    "dataset": ds,
                    "method": method,
                    "recall_target": t,
                    "qps_at_target": qps_at,
                    "latency_ms_at_target": lat_at,
                }
            )
    targets_df = pd.DataFrame(target_rows)
    return base, targets_df


def per_dataset_at_k_targets(
    df: pd.DataFrame,
    *,
    k_value: int,
    targets: list[float],
    recall_col: str,
    stages: set[str],
) -> pd.DataFrame:
    """Best per-method operating point at fixed k and recall thresholds."""
    if df.empty or "k" not in df.columns:
        return pd.DataFrame()

    sub_df = df.copy()
    sub_df["k"] = pd.to_numeric(sub_df["k"], errors="coerce")
    sub_df = sub_df[(sub_df["k"] == k_value) & sub_df["stage"].isin(stages)]
    if sub_df.empty:
        return pd.DataFrame()

    rows = []
    keys = ["stage", "dataset_group", "dataset", "method"]
    for (stage, dsg, ds, method), sub in sub_df.groupby(keys, dropna=False):
        for target in targets:
            good = sub[sub[recall_col] >= target]
            rows.append(
                {
                    "stage": stage,
                    "dataset_group": dsg,
                    "dataset": ds,
                    "method": method,
                    "k": k_value,
                    "recall_col": recall_col,
                    "recall_target": target,
                    "qps_at_target": float(good["qps_effective"].max()) if not good.empty else float("nan"),
                    "latency_ms_at_target": float(good["latency_ms"].min()) if not good.empty else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _positive_finite(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return vals.where(vals > 0)


def paper_highlight_speedups(
    at_k_df: pd.DataFrame,
    *,
    anchor: str,
    baselines: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summaries for paper claims at fixed k/recall thresholds.

    For latency, factors are ``baseline_latency / anchor_latency``.
    For batch, factors are ``anchor_qps / baseline_qps``.
    In both cases, factors above one mean the anchor is better.
    """
    if at_k_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    detail_rows = []
    summary_rows = []
    group_keys = ["stage", "recall_target"]
    for (stage, target), sub in at_k_df.groupby(group_keys, dropna=False):
        qps = sub.pivot_table(
            index=["dataset_group", "dataset"],
            columns="method",
            values="qps_at_target",
            aggfunc="max",
        )
        lat = sub.pivot_table(
            index=["dataset_group", "dataset"],
            columns="method",
            values="latency_ms_at_target",
            aggfunc="min",
        )
        if anchor not in qps.columns and anchor not in lat.columns:
            continue

        methods = sorted(set(qps.columns).union(set(lat.columns)))
        comparisons = list(baselines)
        if len([m for m in methods if m != anchor]) > 0:
            comparisons.append(f"best_non_{anchor}")

        for comparison in comparisons:
            if stage == "batch":
                if anchor not in qps.columns:
                    continue
                anchor_vals = _positive_finite(qps[anchor])
                if comparison.startswith("best_non_"):
                    other_cols = [m for m in qps.columns if m != anchor]
                    if not other_cols:
                        continue
                    baseline_vals = qps[other_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)
                    competing_method = qps[other_cols].apply(pd.to_numeric, errors="coerce").idxmax(axis=1)
                    factor = anchor_vals / _positive_finite(baseline_vals)
                    metric = "qps_improvement_factor"
                else:
                    if comparison not in qps.columns:
                        continue
                    baseline_vals = _positive_finite(qps[comparison])
                    competing_method = pd.Series(comparison, index=qps.index)
                    factor = anchor_vals / baseline_vals
                    metric = "qps_improvement_factor"
            else:
                if anchor not in lat.columns:
                    continue
                anchor_vals = _positive_finite(lat[anchor])
                if comparison.startswith("best_non_"):
                    other_cols = [m for m in lat.columns if m != anchor]
                    if not other_cols:
                        continue
                    baseline_vals = lat[other_cols].apply(pd.to_numeric, errors="coerce").min(axis=1)
                    competing_method = lat[other_cols].apply(pd.to_numeric, errors="coerce").idxmin(axis=1)
                    factor = _positive_finite(baseline_vals) / anchor_vals
                    metric = "latency_speedup_factor"
                else:
                    if comparison not in lat.columns:
                        continue
                    baseline_vals = _positive_finite(lat[comparison])
                    competing_method = pd.Series(comparison, index=lat.index)
                    factor = baseline_vals / anchor_vals
                    metric = "latency_speedup_factor"

            factor = factor.replace([np.inf, -np.inf], np.nan).dropna()
            if factor.empty:
                continue

            for idx, value in factor.items():
                detail_rows.append(
                    {
                        "stage": stage,
                        "recall_target": target,
                        "dataset_group": idx[0],
                        "dataset": idx[1],
                        "anchor_method": anchor,
                        "comparison": comparison,
                        "competing_method": competing_method.loc[idx],
                        metric: float(value),
                    }
                )

            best_idx = factor.idxmax()
            summary_rows.append(
                {
                    "stage": stage,
                    "recall_target": target,
                    "anchor_method": anchor,
                    "comparison": comparison,
                    "metric": metric,
                    "datasets_compared": int(factor.shape[0]),
                    "geomean_factor": safe_geomean(factor.values),
                    "mean_factor": float(factor.mean()),
                    "best_factor": float(factor.max()),
                    "best_dataset": best_idx[1],
                    "best_competing_method": competing_method.loc[best_idx],
                    "win_rate": float((factor > 1.0).mean()),
                    "mean_improvement_pct": float((factor.mean() - 1.0) * 100.0),
                    "geomean_improvement_pct": float((safe_geomean(factor.values) - 1.0) * 100.0),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def method_threshold_export(at_k_df: pd.DataFrame) -> pd.DataFrame:
    if at_k_df.empty:
        return pd.DataFrame()
    out = at_k_df.copy()
    out["batch_qps_at_target"] = np.where(
        out["stage"] == "batch", out["qps_at_target"], np.nan
    )
    out["latency_ms_at_target"] = np.where(
        out["stage"] == "latency", out["latency_ms_at_target"], np.nan
    )
    cols = [
        "stage",
        "dataset_group",
        "dataset",
        "method",
        "k",
        "recall_col",
        "recall_target",
        "batch_qps_at_target",
        "latency_ms_at_target",
    ]
    return out[cols].sort_values(["stage", "recall_target", "dataset", "method"])


def pairwise_threshold_speedups(
    at_k_df: pd.DataFrame,
    comparisons: list[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise speedups at fixed k/recall thresholds.

    For batch rows, ``qps_improvement_factor = anchor_qps / baseline_qps``.
    For latency rows, ``latency_speedup_factor = baseline_latency / anchor_latency``.
    Factors above one mean the anchor method is better.
    """
    if at_k_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    keys = ["stage", "recall_target", "dataset_group", "dataset", "method"]
    lookup = at_k_df.set_index(keys)
    dataset_keys = (
        at_k_df[["stage", "recall_target", "dataset_group", "dataset"]]
        .drop_duplicates()
        .sort_values(["stage", "recall_target", "dataset", "dataset_group"])
    )

    def value_for(key: tuple, method: str, col: str) -> float:
        idx = (*key, method)
        if idx not in lookup.index:
            return float("nan")
        value = lookup.at[idx, col]
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        return float(value) if pd.notna(value) else float("nan")

    detail_rows = []
    for r in dataset_keys.itertuples(index=False):
        key = (r.stage, r.recall_target, r.dataset_group, r.dataset)
        for anchor, baseline in comparisons:
            a_qps = value_for(key, anchor, "qps_at_target")
            b_qps = value_for(key, baseline, "qps_at_target")
            a_lat = value_for(key, anchor, "latency_ms_at_target")
            b_lat = value_for(key, baseline, "latency_ms_at_target")

            qps_factor = float("nan")
            latency_factor = float("nan")
            anchor_qps = baseline_qps = float("nan")
            anchor_lat = baseline_lat = float("nan")
            if r.stage == "batch":
                anchor_qps, baseline_qps = a_qps, b_qps
                if a_qps > 0 and b_qps > 0 and np.isfinite(a_qps) and np.isfinite(b_qps):
                    qps_factor = a_qps / b_qps
            elif r.stage == "latency":
                anchor_lat, baseline_lat = a_lat, b_lat
                if a_lat > 0 and b_lat > 0 and np.isfinite(a_lat) and np.isfinite(b_lat):
                    latency_factor = b_lat / a_lat

            detail_rows.append(
                {
                    "stage": r.stage,
                    "recall_target": r.recall_target,
                    "dataset_group": r.dataset_group,
                    "dataset": r.dataset,
                    "anchor_method": anchor,
                    "baseline_method": baseline,
                    "comparison": f"{anchor}_vs_{baseline}",
                    "anchor_batch_qps_at_target": anchor_qps,
                    "baseline_batch_qps_at_target": baseline_qps,
                    "qps_improvement_factor": qps_factor,
                    "anchor_latency_ms_at_target": anchor_lat,
                    "baseline_latency_ms_at_target": baseline_lat,
                    "latency_speedup_factor": latency_factor,
                }
            )

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        return detail_df, pd.DataFrame()

    summary_rows = []
    group_keys = ["stage", "recall_target", "anchor_method", "baseline_method", "comparison"]
    for key, sub in detail_df.groupby(group_keys, dropna=False):
        stage = key[0]
        if stage == "batch":
            factor_col = "qps_improvement_factor"
            anchor_metric_col = "anchor_batch_qps_at_target"
            baseline_metric_col = "baseline_batch_qps_at_target"
            metric_name = "batch_qps"
        elif stage == "latency":
            factor_col = "latency_speedup_factor"
            anchor_metric_col = "anchor_latency_ms_at_target"
            baseline_metric_col = "baseline_latency_ms_at_target"
            metric_name = "latency_ms"
        else:
            continue

        compared = sub.dropna(subset=[factor_col])
        if compared.empty:
            summary_rows.append(
                {
                    "stage": stage,
                    "recall_target": key[1],
                    "anchor_method": key[2],
                    "baseline_method": key[3],
                    "comparison": key[4],
                    "metric": metric_name,
                    "datasets_with_anchor": int(sub[anchor_metric_col].notna().sum()),
                    "datasets_with_baseline": int(sub[baseline_metric_col].notna().sum()),
                    "datasets_compared": 0,
                }
            )
            continue

        factors = compared[factor_col].to_numpy(dtype=float)
        best_idx = compared[factor_col].idxmax()
        row = {
            "stage": stage,
            "recall_target": key[1],
            "anchor_method": key[2],
            "baseline_method": key[3],
            "comparison": key[4],
            "metric": metric_name,
            "datasets_with_anchor": int(sub[anchor_metric_col].notna().sum()),
            "datasets_with_baseline": int(sub[baseline_metric_col].notna().sum()),
            "datasets_compared": int(compared["dataset"].nunique()),
            "geomean_speedup": safe_geomean(factors),
            "mean_speedup": float(np.nanmean(factors)),
            "max_speedup": float(np.nanmax(factors)),
            "max_speedup_dataset": compared.at[best_idx, "dataset"],
            "win_rate": float((compared[factor_col] > 1.0).mean()),
        }
        if stage == "batch":
            row.update(
                {
                    "anchor_mean_qps": float(compared[anchor_metric_col].mean()),
                    "baseline_mean_qps": float(compared[baseline_metric_col].mean()),
                    "anchor_max_qps": float(compared[anchor_metric_col].max()),
                    "baseline_max_qps": float(compared[baseline_metric_col].max()),
                }
            )
        else:
            row.update(
                {
                    "anchor_mean_latency_ms": float(compared[anchor_metric_col].mean()),
                    "baseline_mean_latency_ms": float(compared[baseline_metric_col].mean()),
                    "anchor_min_latency_ms": float(compared[anchor_metric_col].min()),
                    "baseline_min_latency_ms": float(compared[baseline_metric_col].min()),
                }
            )
        summary_rows.append(row)

    return detail_df, pd.DataFrame(summary_rows)


def anchor_speedups(
    targets_df: pd.DataFrame, anchor: str
) -> pd.DataFrame:
    """Relative factors vs anchor method.

    - latency/multi_latency: latency factor = other_latency / anchor_latency.
      (>1 means anchor is faster/lower latency)
    - batch: qps factor = anchor_qps / other_qps.
      (>1 means anchor has higher throughput)
    """
    if targets_df.empty:
        return pd.DataFrame()

    rows = []
    keys = ["stage", "dataset_group", "dataset", "recall_target"]
    for key, sub in targets_df.groupby(keys, dropna=False):
        sub = sub.set_index("method")
        if anchor not in sub.index:
            continue
        a_qps = sub.at[anchor, "qps_at_target"]
        a_lat = sub.at[anchor, "latency_ms_at_target"]
        stage = key[0]
        for method, r in sub.iterrows():
            if method == anchor:
                continue
            qps = r["qps_at_target"]
            lat = r["latency_ms_at_target"]
            if stage == "batch":
                factor = a_qps / qps if qps and np.isfinite(qps) and qps > 0 else float("nan")
                factor_name = "anchor_qps_over_method_qps"
            else:
                factor = lat / a_lat if a_lat and np.isfinite(a_lat) and a_lat > 0 and np.isfinite(lat) else float("nan")
                factor_name = "method_latency_over_anchor_latency"
            rows.append(
                {
                    "stage": stage,
                    "dataset_group": key[1],
                    "dataset": key[2],
                    "recall_target": key[3],
                    "anchor_method": anchor,
                    "method": method,
                    factor_name: factor,
                }
            )
    return pd.DataFrame(rows)


def combined_target_summary(targets_df: pd.DataFrame, anchor: str) -> pd.DataFrame:
    if targets_df.empty:
        return pd.DataFrame()

    rows = []
    group_keys = ["stage", "dataset_group", "recall_target"]
    for key, sub in targets_df.groupby(group_keys, dropna=False):
        stage = key[0]
        for method, msub in sub.groupby("method", dropna=False):
            rows.append(
                {
                    "stage": stage,
                    "dataset_group": key[1],
                    "recall_target": key[2],
                    "method": method,
                    "datasets_covered": int(msub["dataset"].nunique()),
                    "macro_avg_qps_at_target": float(msub["qps_at_target"].mean(skipna=True)),
                    "macro_avg_latency_ms_at_target": float(msub["latency_ms_at_target"].mean(skipna=True)),
                }
            )

    # Add anchor-vs-method combined factors.
    speed_rows = []
    for key, sub in targets_df.groupby(group_keys, dropna=False):
        stage = key[0]
        piv = sub.pivot_table(
            index="dataset",
            columns="method",
            values=["qps_at_target", "latency_ms_at_target"],
            aggfunc="max",
        )
        if ("qps_at_target", anchor) not in piv.columns:
            continue
        methods = sorted({c[1] for c in piv.columns if c[1] != anchor})
        for m in methods:
            if stage == "batch":
                if ("qps_at_target", m) not in piv.columns:
                    continue
                facs = (
                    piv[("qps_at_target", anchor)] / piv[("qps_at_target", m)]
                ).replace([np.inf, -np.inf], np.nan).dropna()
                factor_name = "geomean_anchor_qps_over_method_qps"
            else:
                if ("latency_ms_at_target", m) not in piv.columns or ("latency_ms_at_target", anchor) not in piv.columns:
                    continue
                facs = (
                    piv[("latency_ms_at_target", m)] / piv[("latency_ms_at_target", anchor)]
                ).replace([np.inf, -np.inf], np.nan).dropna()
                factor_name = "geomean_method_latency_over_anchor_latency"
            speed_rows.append(
                {
                    "stage": stage,
                    "dataset_group": key[1],
                    "recall_target": key[2],
                    "anchor_method": anchor,
                    "method": m,
                    "datasets_compared": int(facs.shape[0]),
                    factor_name: safe_geomean(facs.values),
                    "anchor_win_rate": float((facs > 1.0).mean()) if facs.shape[0] > 0 else float("nan"),
                }
            )

    df_rows = pd.DataFrame(rows)
    df_speed = pd.DataFrame(speed_rows)
    if df_speed.empty:
        return df_rows
    return pd.concat([df_rows, df_speed], ignore_index=True, sort=False)


def parse_build_report_md(path: Path, methods: set[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    lines = path.read_text(encoding="utf-8").splitlines()
    data = []
    for ln in lines:
        ln = ln.strip()
        if not ln.startswith("|"):
            continue
        if ln.startswith("|---"):
            continue
        parts = [x.strip() for x in ln.strip("|").split("|")]
        if parts == ["dataset", "method", "build_name", "build_time_sec", "index_size_mb"]:
            continue
        if len(parts) != 5:
            continue
        ds, method, build_name, build_time_sec, index_size_mb = parts
        if methods and method not in methods:
            continue
        try:
            bt = float(build_time_sec)
            sz = float(index_size_mb)
        except ValueError:
            continue
        data.append(
            {
                "dataset": ds,
                "dataset_group": dataset_group(ds),
                "method": method,
                "build_name": build_name,
                "build_time_sec": bt,
                "index_size_mb": sz,
            }
        )
    return pd.DataFrame(data)


def build_speedups(build_df: pd.DataFrame, anchor: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if build_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    per_ds = []
    for (dsg, ds), sub in build_df.groupby(["dataset_group", "dataset"], dropna=False):
        # Some datasets can have multiple build variants for the same method
        # (e.g., legacy l100/l500 builds). Collapse to one row per method.
        sub = (
            sub.groupby("method", as_index=False)[["build_time_sec", "index_size_mb"]]
            .min()
            .set_index("method")
        )
        if anchor not in sub.index:
            continue
        a_time = float(sub.at[anchor, "build_time_sec"])
        a_size = float(sub.at[anchor, "index_size_mb"])
        for method, row in sub.iterrows():
            if method == anchor:
                continue
            m_time = float(row["build_time_sec"])
            m_size = float(row["index_size_mb"])
            per_ds.append(
                {
                    "dataset_group": dsg,
                    "dataset": ds,
                    "anchor_method": anchor,
                    "method": method,
                    "method_build_time_over_anchor": (m_time / a_time) if a_time > 0 else float("nan"),
                    "method_size_over_anchor": (m_size / a_size) if a_size > 0 else float("nan"),
                    "anchor_build_time_reduction_pct_vs_method": (1.0 - a_time / m_time) * 100.0 if m_time > 0 else float("nan"),
                    "anchor_size_reduction_pct_vs_method": (1.0 - a_size / m_size) * 100.0 if m_size > 0 else float("nan"),
                }
            )
    per_ds_df = pd.DataFrame(per_ds)

    comb = []
    if not per_ds_df.empty:
        for (dsg, m), sub in per_ds_df.groupby(["dataset_group", "method"], dropna=False):
            comb.append(
                {
                    "dataset_group": dsg,
                    "anchor_method": anchor,
                    "method": m,
                    "datasets_compared": int(sub["dataset"].nunique()),
                    "geomean_method_build_time_over_anchor": safe_geomean(sub["method_build_time_over_anchor"].values),
                    "geomean_method_size_over_anchor": safe_geomean(sub["method_size_over_anchor"].values),
                    "mean_anchor_build_time_reduction_pct_vs_method": float(sub["anchor_build_time_reduction_pct_vs_method"].mean()),
                    "mean_anchor_size_reduction_pct_vs_method": float(sub["anchor_size_reduction_pct_vs_method"].mean()),
                }
            )
    return per_ds_df, pd.DataFrame(comb)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty:
        pd.DataFrame().to_csv(path, index=False)
        return
    df.to_csv(path, index=False)


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "_No data._\n"
    show = df.head(max_rows).copy()
    cols = list(show.columns)
    out = []
    out.append("| " + " | ".join(cols) + " |")
    out.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in show.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                if math.isnan(v):
                    vals.append("NaN")
                elif abs(v) >= 1000:
                    vals.append(f"{v:.2f}")
                else:
                    vals.append(f"{v:.4g}")
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    if len(df) > max_rows:
        out.append(f"\n_Shown first {max_rows} of {len(df)} rows._")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <repo>/experiments/analysis/paper_numbers",
    )
    ap.add_argument("--methods", default="mvivf,muvera,svh_graph,vamana")
    ap.add_argument("--stages", default="latency,batch,multi_latency")
    ap.add_argument("--targets", default="0.8,0.9,0.95")
    ap.add_argument("--anchor", default="mvivf")
    ap.add_argument("--recall-col", default="recall_k_k")
    ap.add_argument("--paper-k", type=int, default=10)
    ap.add_argument("--paper-targets", default="0.88,0.9,0.95,0.99")
    ap.add_argument("--paper-baselines", default="muvera,svh_graph,vamana")
    ap.add_argument(
        "--pairwise-comparisons",
        default="mvivf:vamana,mvivf:muvera,mvivf:svh_graph,vamana:muvera,vamana:svh_graph",
        help="Comma-separated anchor:baseline method comparisons for paper speedup CSVs.",
    )
    ap.add_argument(
        "--exclude-dataset-groups",
        default="vidore",
        help="Comma-separated dataset groups to exclude from summary outputs.",
    )
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else repo_root / "experiments" / "analysis" / "paper_numbers"
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = {m.strip() for m in args.methods.split(",") if m.strip()}
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    targets = sorted({float(x) for x in args.targets.split(",") if x.strip()})
    paper_targets = sorted({float(x) for x in args.paper_targets.split(",") if x.strip()})
    paper_baselines = [m.strip() for m in args.paper_baselines.split(",") if m.strip()]
    pairwise_comparisons = []
    for item in args.pairwise_comparisons.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise SystemExit(f"invalid pairwise comparison {item!r}; expected anchor:baseline")
        anchor, baseline = [x.strip() for x in item.split(":", 1)]
        pairwise_comparisons.append((anchor, baseline))
    excluded_groups = {g.strip() for g in args.exclude_dataset_groups.split(",") if g.strip()}

    stage_frames = []
    for st in stages:
        if st not in PREFIX_FOR_STAGE:
            continue
        stage_frames.append(read_stage_rows(repo_root, st, methods, recall_col=args.recall_col))
    all_df = pd.concat([df for df in stage_frames if not df.empty], ignore_index=True) if stage_frames else pd.DataFrame()
    if excluded_groups and not all_df.empty:
        all_df = all_df[~all_df["dataset_group"].isin(excluded_groups)].copy()

    per_ds_df, targets_df = per_dataset_summary(all_df, targets=targets, recall_col=args.recall_col)
    speedups_df = anchor_speedups(targets_df, anchor=args.anchor)
    combined_df = combined_target_summary(targets_df, anchor=args.anchor)
    paper_at_k_df = per_dataset_at_k_targets(
        all_df,
        k_value=args.paper_k,
        targets=paper_targets,
        recall_col=args.recall_col,
        stages={"latency", "batch"},
    )
    paper_highlights_df, paper_highlight_detail_df = paper_highlight_speedups(
        paper_at_k_df,
        anchor=args.anchor,
        baselines=paper_baselines,
    )
    paper_method_points_df = method_threshold_export(paper_at_k_df)
    pairwise_speedups_df, pairwise_summary_df = pairwise_threshold_speedups(
        paper_at_k_df,
        comparisons=pairwise_comparisons,
    )

    build_df = parse_build_report_md(
        repo_root / "experiments" / "builds" / "results" / "build_report.md",
        methods=methods,
    )
    if excluded_groups and not build_df.empty:
        build_df = build_df[~build_df["dataset_group"].isin(excluded_groups)].copy()
    build_per_ds_df, build_combined_df = build_speedups(build_df, anchor=args.anchor)

    write_csv(all_df, out_dir / "raw_points_all.csv")
    write_csv(per_ds_df, out_dir / "per_dataset_bests.csv")
    write_csv(targets_df, out_dir / "per_dataset_at_recall_targets.csv")
    write_csv(speedups_df, out_dir / "speedups_vs_anchor_per_dataset.csv")
    write_csv(combined_df, out_dir / "combined_summary.csv")
    write_csv(paper_at_k_df, out_dir / f"paper_at_k{args.paper_k}_recall_targets.csv")
    write_csv(paper_method_points_df, out_dir / f"paper_method_points_k{args.paper_k}.csv")
    write_csv(paper_highlights_df, out_dir / f"paper_highlights_k{args.paper_k}.csv")
    write_csv(paper_highlight_detail_df, out_dir / f"paper_highlights_k{args.paper_k}_per_dataset.csv")
    write_csv(pairwise_speedups_df, out_dir / f"paper_pairwise_speedups_k{args.paper_k}.csv")
    write_csv(pairwise_summary_df, out_dir / f"paper_pairwise_speedup_summary_k{args.paper_k}.csv")
    for anchor, baseline in pairwise_comparisons:
        comp = f"{anchor}_vs_{baseline}"
        write_csv(
            pairwise_speedups_df[pairwise_speedups_df["comparison"] == comp],
            out_dir / f"paper_pairwise_{comp}_k{args.paper_k}.csv",
        )
    write_csv(build_df, out_dir / "build_raw.csv")
    write_csv(build_per_ds_df, out_dir / "build_speedups_per_dataset.csv")
    write_csv(build_combined_df, out_dir / "build_speedups_combined.csv")

    md = []
    md.append("# Paper Numbers Summary\n")
    md.append(f"- Anchor method: `{args.anchor}`")
    md.append(f"- Methods: `{', '.join(sorted(methods))}`")
    md.append(f"- Stages: `{', '.join(stages)}`")
    md.append(f"- Recall targets: `{', '.join(str(t) for t in targets)}`\n")
    md.append(f"- Excluded dataset groups: `{', '.join(sorted(excluded_groups)) or 'none'}`")
    md.append(
        f"- Paper highlights: `k={args.paper_k}`, `{args.recall_col}`, "
        f"targets `{', '.join(str(t) for t in paper_targets)}`\n"
    )

    md.append("## Per-dataset best operating points")
    md.append(markdown_table(per_ds_df, max_rows=60))

    md.append("## Per-dataset metrics at recall targets")
    md.append(markdown_table(targets_df, max_rows=80))

    md.append("## Speedups vs anchor at recall targets")
    md.append(markdown_table(speedups_df, max_rows=80))

    md.append("## Combined summaries (beir / vidore / other)")
    md.append(markdown_table(combined_df, max_rows=80))

    md.append(f"## Paper highlights at k={args.paper_k}")
    md.append(markdown_table(paper_highlights_df, max_rows=80))

    md.append(f"## Paper highlight per-dataset details at k={args.paper_k}")
    md.append(markdown_table(paper_highlight_detail_df, max_rows=120))

    md.append(f"## Paper pairwise speedup summary at k={args.paper_k}")
    md.append(markdown_table(pairwise_summary_df, max_rows=120))

    md.append("## Build time / index size speedups")
    md.append(markdown_table(build_per_ds_df, max_rows=80))

    md.append("## Build time / index size combined")
    md.append(markdown_table(build_combined_df, max_rows=60))

    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote summary files to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

