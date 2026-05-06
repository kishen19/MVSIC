#!/usr/bin/env python3
"""Multi-latency-stage plots. Thin wrapper around builds/scripts/plot_stage.py.

Side-by-side Recall (1@k) / Recall (k@k) vs QPS and vs latency ms
(``{dataset}_pareto_latency_ms.pdf``) under experiments/multi_latency/results/_plots/.
Pass ``--k 10`` / ``--k 100`` to filter and route to ``_plots/k=<N>/``.
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
    p.add_argument("--datasets", default=None)
    p.add_argument("--results", type=pathlib.Path,
                   default=REPO_ROOT / "experiments" / "multi_latency" / "results")
    p.add_argument("--out-dir", type=pathlib.Path, default=None)
    p.add_argument("--methods", type=pathlib.Path,
                   default=REPO_ROOT / "benchmarks" / "methods.yaml")
    p.add_argument(
        "--k", type=int, default=None,
        help=("Filter to the search_config named ``k=<value>``. When set and "
              "--out-dir is unset, plots land under "
              "<results>/_plots/k=<value>/."),
    )
    args = p.parse_args()

    cmd = [
        sys.executable, str(PLOT_STAGE),
        "--stage", "multi_latency",
        "--results", str(args.results),
        "--methods", str(args.methods),
    ]
    if args.datasets:
        cmd += ["--datasets", args.datasets]
    if args.out_dir:
        cmd += ["--out-dir", str(args.out_dir)]
    if args.k is not None:
        cmd += ["--search", f"k={args.k}"]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
