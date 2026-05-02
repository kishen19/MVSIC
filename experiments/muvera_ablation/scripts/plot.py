#!/usr/bin/env python3
"""MUVERA ablation Pareto plots (latency only, no breakdown bars)."""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("pip install pandas matplotlib numpy to use this script")


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent
DEFAULT_DATASETS = ["arguana", "fiqa", "nq500k", "nq"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    if "qps_seq" in df.columns and "QPS_seq" not in df.columns:
        rename["qps_seq"] = "QPS_seq"
    if "qps_par" in df.columns and "QPS_par" not in df.columns:
        rename["qps_par"] = "QPS_par"
    return df.rename(columns=rename) if rename else df


def _annotate_path(results_root: pathlib.Path, csv_file: pathlib.Path) -> dict:
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


def _load_csvs(results_root: pathlib.Path, dataset: str) -> pd.DataFrame:
    base = results_root / dataset
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
        meta = _annotate_path(results_root, f)
        for k, v in meta.items():
            df[k] = v
        rows.append(df)
    if not rows:
        raise SystemExit(f"all latency_*.csv files under {base} were empty")
    return pd.concat(rows, ignore_index=True)


def _pareto_curve(xs: np.ndarray, ys: np.ndarray,
                  min_recall_spacing: float = 0.005) -> tuple[np.ndarray, np.ndarray]:
    if xs.size == 0:
        return xs, ys
    best_by_recall: dict[float, float] = {}
    for x, y in zip(xs, ys):
        fx = float(x)
        fy = float(y)
        cur = best_by_recall.get(fx)
        if cur is None or fy > cur:
            best_by_recall[fx] = fy
    xs_u = np.array(sorted(best_by_recall.keys()), dtype=float)
    ys_u = np.array([best_by_recall[x] for x in xs_u], dtype=float)

    order = np.argsort(-xs_u, kind="stable")
    keep = []
    best_y = -np.inf
    for i in order:
        if ys_u[i] > best_y:
            keep.append(i)
            best_y = ys_u[i]
    if not keep:
        return np.array([]), np.array([])
    fx = xs_u[np.array(keep, dtype=int)]
    fy = ys_u[np.array(keep, dtype=int)]
    ord2 = np.argsort(fx)
    fx = fx[ord2]
    fy = fy[ord2]
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


def _fde_from_build(build: str) -> int:
    m = re.search(r"muvera_(\d+)_", build)
    return int(m.group(1)) if m else 10**9


def _build_label(build: str) -> str:
    d = _fde_from_build(build)
    return f"d_fde={d}" if d < 10**9 else build


def _plot_dataset(df: pd.DataFrame, dataset: str, out_path: pathlib.Path) -> None:
    sub = df[(df["method"] == "muvera")].dropna(subset=["recall_k_k", "QPS_seq"])
    if sub.empty:
        print(f"  [skip {dataset}] no MUVERA rows with recall/QPS")
        return

    builds = sorted(sub["build"].unique(), key=_fde_from_build)
    cmap = plt.colormaps.get_cmap("tab10")
    colors = {b: cmap(i % 10) for i, b in enumerate(builds)}

    fig, ax = plt.subplots(figsize=(7, 5))
    for b in builds:
        s = sub[sub["build"] == b]
        xs = s["recall_k_k"].to_numpy(dtype=float)
        ys = s["QPS_seq"].to_numpy(dtype=float)
        # Faded points behind the Pareto line.
        ax.scatter(xs, ys, s=18, color=colors[b], alpha=0.22, zorder=1)
        fx, fy = _pareto_curve(xs, ys)
        if fx.size == 0:
            continue
        ax.plot(
            fx, fy, marker="o", markersize=4.5, linewidth=1.9,
            color=colors[b], label=_build_label(b), zorder=3
        )

    ax.set_xlabel("Recall@k")
    ax.set_ylabel("QPS (per-query)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"{dataset}: MUVERA ablation (Pareto by d_fde)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="comma-separated dataset names (default: arguana,fiqa,nq500k,nq)",
    )
    p.add_argument(
        "--results",
        type=pathlib.Path,
        default=REPO_ROOT / "experiments" / "muvera_ablation" / "results",
    )
    p.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=None,
        help="default: <results>/_plots",
    )
    _ = p.add_argument("--methods", default=None)  # accepted for compatibility
    args = p.parse_args()

    out_dir = args.out_dir if args.out_dir else args.results / "_plots"
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for ds in datasets:
        try:
            df = _load_csvs(args.results, ds)
        except SystemExit as e:
            print(f"[skip {ds}] {e}")
            continue
        _plot_dataset(df, ds, out_dir / f"{ds}_pareto.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
