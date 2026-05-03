#!/usr/bin/env python3
"""Latency-stage plots. Thin wrapper around builds/scripts/plot_stage.py.

Emits side-by-side Recall (1@k) and Recall (k@k) vs QPS Pareto PDFs (see
plot_stage.py), plus timer breakdown PDFs, under experiments/latency/results/_plots/.

Usage:
    experiments/latency/scripts/plot.py
    experiments/latency/scripts/plot.py --datasets nfcorpus,arguana
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent
PLOT_STAGE = REPO_ROOT / "experiments" / "builds" / "scripts" / "plot_stage.py"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=None, help="comma-separated dataset names")
    p.add_argument("--results", type=pathlib.Path,
                   default=REPO_ROOT / "experiments" / "latency" / "results")
    p.add_argument("--out-dir", type=pathlib.Path, default=None)
    p.add_argument("--methods", type=pathlib.Path,
                   default=REPO_ROOT / "benchmarks" / "methods.yaml")
    args = p.parse_args()

    cmd = [
        sys.executable, str(PLOT_STAGE),
        "--stage", "latency",
        "--results", str(args.results),
        "--methods", str(args.methods),
    ]
    if args.datasets:
        cmd += ["--datasets", args.datasets]
    if args.out_dir:
        cmd += ["--out-dir", str(args.out_dir)]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
