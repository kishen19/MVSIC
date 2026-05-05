#!/usr/bin/env python3
"""Build-side reporting for the main experiments.

Walks `results/indexes/<dataset>/<method>/<build>/build_stats.json` and emits

    <out_dir>/build_report.md    per-(dataset, method, build) markdown table
    <out_dir>/build_time.pdf     bar plot of build time
    <out_dir>/index_size.pdf     bar plot of on-disk index size (MB)

Usage:
    experiments/builds/scripts/build_report.py
    experiments/builds/scripts/build_report.py --indexes results/indexes \
        --datasets nfcorpus,scifact,arguana,scidocs,fiqa
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("pip install pandas matplotlib numpy to use this script")


_METHOD_ORDER = ["mvivf", "muvera", "vamana", "svh_graph", "fastplaid", "igp"]
_PRETTY = {
    "mvivf": "MVIVF",
    "muvera": "MUVERA",
    "vamana": "MV-Vamana",
    "svh_graph": "SVH Graph",
    "fastplaid": "FastPlaid",
    "igp": "IGP",
}


def _walk_stats(indexes_root: pathlib.Path,
                datasets: list[str] | None) -> pd.DataFrame:
    rows = []
    if datasets is None:
        if not indexes_root.exists():
            return pd.DataFrame()
        # Discover shards. ViDoRe puts its shards one level deeper
        # (results/indexes/vidore/<sub>) so expand that container into its
        # children rather than treating "vidore" itself as a shard.
        datasets = []
        for p in sorted(indexes_root.iterdir()):
            if not p.is_dir():
                continue
            if p.name == "vidore":
                for sub in sorted(p.iterdir()):
                    if sub.is_dir():
                        datasets.append(sub.name)
            else:
                datasets.append(p.name)
    for ds in datasets:
        ds_dir = indexes_root / ds
        # vidore puts datasets one level deeper.
        if (indexes_root / "vidore" / ds).exists():
            ds_dir = indexes_root / "vidore" / ds
        if not ds_dir.exists():
            continue
        for stats_path in ds_dir.rglob("build_stats.json"):
            try:
                with open(stats_path) as f:
                    s = json.load(f)
            except Exception as e:
                print(f"[skip {stats_path}] {e}")
                continue
            rel = stats_path.relative_to(ds_dir)
            parts = rel.parts  # <method>/<build>/build_stats.json
            if len(parts) < 3:
                continue
            row = {
                "dataset": ds,
                "method": parts[0],
                "build_name": parts[1],
                "build_time_sec": s.get("build_time_sec"),
                "index_size_mb": s.get("index_size_mb"),
                "built_at": s.get("built_at"),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _bar_plot(df: pd.DataFrame, value_col: str, ylabel: str,
              out_path: pathlib.Path, log: bool = False) -> None:
    if df.empty or value_col not in df.columns or df[value_col].dropna().empty:
        return
    datasets = sorted(df["dataset"].unique())
    methods = [m for m in _METHOD_ORDER if m in df["method"].unique()]
    if not datasets or not methods:
        return
    cmap = plt.colormaps.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(datasets) + 2), 4.5))
    width = 0.8 / max(1, len(methods))
    x_centers = np.arange(len(datasets))
    for mi, method in enumerate(methods):
        heights = []
        for ds in datasets:
            sub = df[(df["dataset"] == ds) & (df["method"] == method)]
            v = sub[value_col].dropna()
            heights.append(float(v.iloc[0]) if not v.empty else np.nan)
        offsets = (mi - (len(methods) - 1) / 2.0) * width
        ax.bar(x_centers + offsets, heights, width=width,
               label=_PRETTY.get(method, method), color=cmap(mi % 10))
    ax.set_xticks(x_centers)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _markdown_table(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    if df.empty:
        out_path.write_text("# Build report\n\n_No build_stats.json files found._\n")
        print(f"  wrote {out_path} (empty)")
        return

    cols = [
        "dataset", "method", "build_name",
        "build_time_sec", "index_size_mb",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()
    fmt_map = {
        "build_time_sec": lambda v: "" if pd.isna(v) else f"{v:.1f}",
        "index_size_mb":  lambda v: "" if pd.isna(v) else f"{v:.2f}",
    }
    for c, fn in fmt_map.items():
        if c in df.columns:
            df[c] = df[c].map(fn)
    df = df.sort_values(["dataset", "method", "build_name"])

    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |"
                     for _, r in df.iterrows())
    md = "# Build report\n\n" + "\n".join([header, sep, body]) + "\n"
    out_path.write_text(md)
    print(f"  wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--indexes", type=pathlib.Path, default=pathlib.Path("results/indexes"),
                   help="Root of the index tree (default: results/indexes).")
    p.add_argument("--datasets", default=None,
                   help="Comma-separated dataset names; default: all.")
    p.add_argument("--out-dir", type=pathlib.Path, default=None,
                   help="Default: experiments/builds/results/")
    args = p.parse_args()

    out_dir = args.out_dir if args.out_dir else pathlib.Path("experiments/builds/results")
    datasets = (
        [d.strip() for d in args.datasets.split(",") if d.strip()]
        if args.datasets else None
    )
    df = _walk_stats(args.indexes, datasets)
    if df.empty:
        print(f"[warn] no build_stats.json under {args.indexes}; nothing to report.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    _markdown_table(df, out_dir / "build_report.md")
    _bar_plot(df, "build_time_sec", "Build time (s)",
              out_dir / "build_time.pdf", log=True)
    _bar_plot(df, "index_size_mb", "Index size on disk (MB)",
              out_dir / "index_size.pdf", log=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
