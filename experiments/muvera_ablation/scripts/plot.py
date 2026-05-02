#!/usr/bin/env python3
"""MUVERA ablation Pareto plots (latency stage).

Thin wrapper around `experiments/builds/scripts/plot_stage.py` so ablation
results get the same recall-vs-QPS Pareto plotting behavior as main
experiments (including multi-num_rerank fronts).
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent
PLOT_STAGE = REPO_ROOT / "experiments" / "builds" / "scripts" / "plot_stage.py"

# Edit this list if you want different default datasets for this ablation.
DEFAULT_DATASETS = ["arguana", "fiqa", "nq500k", "nq"]


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
    p.add_argument("--out-dir", type=pathlib.Path, default=None)
    p.add_argument(
        "--methods",
        type=pathlib.Path,
        default=REPO_ROOT / "benchmarks" / "methods.yaml",
    )
    args = p.parse_args()

    cmd = [
        sys.executable,
        str(PLOT_STAGE),
        "--stage",
        "latency",
        "--results",
        str(args.results),
        "--methods",
        str(args.methods),
    ]
    if args.datasets:
        cmd += ["--datasets", args.datasets]
    if args.out_dir:
        cmd += ["--out-dir", str(args.out_dir)]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
