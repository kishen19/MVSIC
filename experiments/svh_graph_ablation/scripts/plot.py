#!/usr/bin/env python3
"""SVH ablation Pareto plots: side-by-side Recall (1@k) vs QPS and Recall (k@k) vs QPS."""
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

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "lines.linewidth": 2.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent
DEFAULT_DATASETS = [
    "arguana",
    "fiqa",
    "hotpotqa",
    "nfcorpus",
    "nq",
    "nq500k",
    "quora",
    "scidocs",
    "scifact",
]


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


def _alpha_from_build(build: str) -> float:
    """Parse trailing `_a<number>` as alpha (e.g. `..._a1.0`, `..._a0.95`)."""
    m = re.search(r"_a(\d+(?:\.\d+)?)", build)
    return float(m.group(1)) if m else 99.0


def _build_sort_key(build: str) -> float:
    return _alpha_from_build(build)


def _build_label(build: str) -> str:
    a = _alpha_from_build(build)
    return f"alpha={a:g}" if a < 99.0 else build


def _recall_xlabel(kind: str, df: pd.DataFrame) -> str:
    if "k" in df.columns:
        vals = sorted({int(v) for v in df["k"].dropna().astype(int).unique()})
        if len(vals) == 1:
            return f"Recall-$1$@${vals[0]}$" if kind == "1" else f"Recall-${vals[0]}$@${vals[0]}$"
    return r"Recall-$1$@$k$" if kind == "1" else r"Recall-$k$@$k$"


def _plot_pareto_panel(
    ax,
    sub: pd.DataFrame,
    builds: list[str],
    colors: dict[str, object],
    x_col: str,
    xlabel: str,
    *,
    show_ylabel: bool,
    show_legend: bool,
) -> None:
    for b in builds:
        s = sub[sub["build"] == b].dropna(subset=[x_col, "QPS_seq"])
        if s.empty:
            continue
        xs = s[x_col].to_numpy(dtype=float)
        ys = s["QPS_seq"].to_numpy(dtype=float)
        ax.scatter(xs, ys, s=20, color=colors[b], alpha=0.25, zorder=1)
        fx, fy = _pareto_curve(xs, ys)
        if fx.size == 0:
            continue
        ax.plot(
            fx,
            fy,
            marker="o",
            markersize=5.4,
            linewidth=2.8,
            color=colors[b],
            label=_build_label(b) if show_legend else None,
            zorder=3,
        )
    ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel("QPS (per-query)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    if show_legend:
        ax.legend(
            loc="best",
            fontsize=13,
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="0.35",
        )

    xs_parts: list[np.ndarray] = []
    for b in builds:
        s = sub[sub["build"] == b].dropna(subset=[x_col, "QPS_seq"])
        if not s.empty:
            xs_parts.append(s[x_col].to_numpy(dtype=float))
    if xs_parts:
        lo = float(np.min(np.concatenate(xs_parts)))
        ax.set_xlim(max(0.0, lo - 0.02), 1.0)
    else:
        ax.set_xlim(0.0, 1.0)


def _plot_dataset(df: pd.DataFrame, dataset: str, out_path: pathlib.Path) -> None:
    sub = df[(df["method"] == "svh_graph")]
    has_left = not sub.dropna(subset=["recall_1_k", "QPS_seq"]).empty
    has_right = not sub.dropna(subset=["recall_k_k", "QPS_seq"]).empty
    if not has_left and not has_right:
        print(f"  [skip {dataset}] no SVH rows with recall/QPS")
        return

    builds = sorted(sub["build"].unique(), key=_build_sort_key)
    cmap = plt.colormaps.get_cmap("tab10")
    colors = {b: cmap(i % 10) for i, b in enumerate(builds)}

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(12, 5), sharey=True, layout="constrained"
    )
    if has_left:
        _plot_pareto_panel(
            ax_left,
            sub,
            builds,
            colors,
            "recall_1_k",
            _recall_xlabel("1", sub),
            show_ylabel=True,
            show_legend=not has_right,
        )
    if has_right:
        _plot_pareto_panel(
            ax_right,
            sub,
            builds,
            colors,
            "recall_k_k",
            _recall_xlabel("k", sub),
            show_ylabel=not has_left,
            show_legend=has_right,
        )
    fig.suptitle(f"{dataset}: SVH ablation Pareto")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="comma-separated dataset names (default: all ablation BEIR suite)",
    )
    p.add_argument(
        "--results",
        type=pathlib.Path,
        default=REPO_ROOT / "experiments" / "svh_graph_ablation" / "results",
    )
    p.add_argument("--out-dir", type=pathlib.Path, default=None, help="default: <results>/_plots")
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
