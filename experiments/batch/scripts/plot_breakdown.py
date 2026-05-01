#!/usr/bin/env python3
"""Batch-stage MVIVF breakdown plot. Wrapper around
``experiments/builds/scripts/plot_mvivf_breakdown.py``.

Note: ``compute_stats_batch`` only times ``search_all`` end-to-end and does
not populate ``avg_timings``, so the batch breakdown collapses to a single
``total`` segment per dataset (mean per-query time computed as
``1 / QPS_par``).  When per-substage batch timers are added, the same
script will pick them up automatically.
"""
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
                   default=REPO_ROOT / "experiments" / "batch" / "results")
    p.add_argument("--out-dir", type=pathlib.Path, default=None)
    args = p.parse_args()

    cmd = [
        sys.executable, str(PLOT),
        "--stage", "batch",
        "--results", str(args.results),
    ]
    if args.datasets:
        cmd += ["--datasets", args.datasets]
    if args.out_dir:
        cmd += ["--out-dir", str(args.out_dir)]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
