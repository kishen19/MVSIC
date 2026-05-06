#!/usr/bin/env python3
"""Latency-stage MVIVF breakdown plot. Wrapper around
``experiments/builds/scripts/plot_mvivf_breakdown.py``."""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent
PLOT = REPO_ROOT / "experiments" / "builds" / "scripts" / "plot_mvivf_breakdown.py"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=None, help="comma-separated dataset names")
    p.add_argument("--results", type=pathlib.Path,
                   default=REPO_ROOT / "experiments" / "latency" / "results")
    p.add_argument("--out-dir", type=pathlib.Path, default=None)
    p.add_argument("--target-recall", type=float, default=0.90)
    args = p.parse_args()

    cmd = [
        sys.executable, str(PLOT),
        "--stage", "latency",
        "--results", str(args.results),
    ]
    if args.datasets:
        cmd += ["--datasets", args.datasets]
    if args.out_dir:
        cmd += ["--out-dir", str(args.out_dir)]
    cmd += ["--target-recall", str(args.target_recall)]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
