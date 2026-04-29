#!/usr/bin/env python3
"""Produce a latency-vs-recall Pareto plot from `latency_*.csv` files.

Usage:
    scripts/plot.py \
        --results experiments/mvivf_ablation/results/nq500k/mvivf/csv/ \
        --group-by k_per_level \
        --out experiments/mvivf_ablation/results/nq500k/mvivf/pareto.pdf

Expected CSV columns (produced by benchmark.py in latency mode):
    method_name, build_config, dataset, k, nprobes, num_rerank,
    qps_seq, qps_par, avg_cmps, recall_1_k, recall_k_k
Latency = 1 / qps_seq.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

try:
    import matplotlib.pyplot as plt  # noqa: F401
    import pandas as pd
except ImportError:
    sys.exit("pip install pandas matplotlib to use this script")


def load_csvs(path: pathlib.Path) -> pd.DataFrame:
    files = sorted(path.glob("latency_*.csv"))
    if not files:
        raise SystemExit(f"no latency_*.csv under {path}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def plot(df: pd.DataFrame, group_col: str, out: pathlib.Path) -> None:
    if "qps_seq" in df.columns:
        df = df.assign(latency_ms=1000.0 / df["qps_seq"])
    recall_col = "recall_1_k" if "recall_1_k" in df.columns else "recall_k_k"
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, sub in df.groupby(group_col):
        sub = sub.sort_values("latency_ms")
        ax.plot(sub["latency_ms"], sub[recall_col], marker="o", label=f"{group_col}={key}")
    ax.set_xscale("log")
    ax.set_xlabel("Sequential latency per query (ms)")
    ax.set_ylabel(recall_col)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title(f"Pareto: latency vs {recall_col} (grouped by {group_col})")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Wrote {out}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=pathlib.Path, required=True)
    p.add_argument("--group-by", default="build_config",
                   help="CSV column to group curves by.")
    p.add_argument("--out", type=pathlib.Path, required=True)
    args = p.parse_args()
    df = load_csvs(args.results)
    plot(df, args.group_by, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
