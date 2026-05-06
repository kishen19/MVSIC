#!/usr/bin/env python3
"""Run builds, latency, batch, and multi_latency experiments for a list of datasets."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

DATASETS = [
    "hotpotqa"
]

METHODS = ["mvivf", "muvera", "vamana", "svh_graph"]

STEPS = [
    ("builds", "experiments/builds/scripts/run_builds.sh"),
    ("latency", "experiments/latency/scripts/run_latency.sh"),
    ("batch", "experiments/batch/scripts/run_batch.sh"),
    ("multi_latency", "experiments/multi_latency/scripts/run_multi_latency.sh"),
]
#    (multi_latency", "experiments/multi_latency/scripts/run_multi_latency.sh"),


def run_step(repo_root: Path, name: str, script_rel: str, dataset: str, method: str) -> int:
    script_path = repo_root / script_rel
    if not script_path.exists():
        print(f"  ERROR: script not found: {script_path}", flush=True)
        return 127
    cmd = [str(script_path), "--dataset", dataset, "--method", method]
    if name != "builds":
        cmd += ["--k", "100"]
    print(f"  $ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=repo_root)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        help=f"Datasets to run (default: {DATASETS})",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=[name for name, _ in STEPS],
        default=[name for name, _ in STEPS],
        help="Subset of steps to run, in the given order.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going if a step fails (default: stop the dataset on failure).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    selected_steps = [(n, s) for n, s in STEPS if n in args.steps]

    failures: list[tuple[str, str, str, int]] = []
    overall_start = time.time()

    for dataset in args.datasets:
        print(f"\n=== dataset: {dataset} ===", flush=True)
        stop_dataset = False
        for method in METHODS:
            if stop_dataset:
                break
            print(f"\n--- method: {method} ---", flush=True)
            for name, script in selected_steps:
                print(f"\n[{dataset}/{method}] step: {name}", flush=True)
                step_start = time.time()
                rc = run_step(repo_root, name, script, dataset, method)
                elapsed = time.time() - step_start
                print(f"[{dataset}/{method}] {name} -> rc={rc} ({elapsed:.1f}s)", flush=True)
                if rc != 0:
                    failures.append((dataset, method, name, rc))
                    if not args.continue_on_error:
                        print(f"[{dataset}/{method}] stopping further steps for this dataset", flush=True)
                        stop_dataset = True
                        break

    total = time.time() - overall_start
    print(f"\n=== done in {total:.1f}s ===", flush=True)
    if failures:
        print("Failures:", flush=True)
        for dataset, method, name, rc in failures:
            print(f"  - {dataset}/{method}/{name}: rc={rc}", flush=True)
        return 1
    print("All steps succeeded.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
