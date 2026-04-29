#!/usr/bin/env python3
"""Measure codebook training (and encoding) wall time for each quantizer.

Drives the ``//mvsic/tools:measure_codebook_time`` C++ binary over one or more
datasets and writes a markdown report.

Usage (from repo root):

    python3 data-tools/measure_codebook_time.py \
        --dataset nq500k:data/beir/nq500k/nq500k_points.pcs \
        --dataset fiqa:data/beir/fiqa/fiqa_points.pcs \
        --metric ip \
        --out experiments/codebook_timings.md

If ``--build`` is passed the script builds the C++ binary first via
``bazel build -c opt //mvsic/tools:measure_codebook_time``.

Each dataset is run sequentially through each quantizer and one JSON record is
emitted per (dataset, quantizer, metric) triple. The markdown report contains:

- a reproducibility block (command, commit, host),
- a single wide table (train seconds, encode seconds, n, dim),
- a short decision rule applied by Phase 2 (codebook caching vs re-train).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

QUANTIZERS = ("pq", "fastscan", "rabitq", "turboquant", "spqtq", "onebittq")

DEFAULT_BINARY = "bazel-bin/mvsic/tools/measure_codebook_time"


def run_quantizer(
    binary: Path,
    dataset_label: str,
    dataset_path: Path,
    quantizer: str,
    metric: str,
    block_size: int,
    rabitq_bits: int,
    extra: list[str],
) -> dict | None:
    cmd = [
        str(binary),
        "-d", str(dataset_path),
        "-metric", metric,
        "-quantizer", quantizer,
        "-block_size", str(block_size),
        "-rabitq_bits", str(rabitq_bits),
        *extra,
    ]
    print(f"\n>>> {dataset_label}/{quantizer}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"    [FAILED rc={proc.returncode}] stderr:\n{proc.stderr}",
              file=sys.stderr)
        return None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        rec["dataset_label"] = dataset_label
        return rec
    print(f"    [WARN] no JSON record in stdout", file=sys.stderr)
    return None


def git_sha(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def write_report(records: list[dict], out_path: Path, cmd: list[str],
                 root: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Codebook Training & Encoding Timings")
    lines.append("")
    lines.append("Wall-clock time to (a) train each quantizer's codebook on a "
                 "full .pcs file and (b) encode the same dataset.")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(f"- Command: `{' '.join(cmd)}`")
    lines.append(f"- Git SHA: `{git_sha(root)}`")
    lines.append(f"- Host: `{platform.node()}` (`{platform.platform()}`)")
    lines.append(f"- Python: `{sys.version.split()[0]}`")
    lines.append(f"- Timestamp (UTC): `{datetime.datetime.utcnow().isoformat()}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Dataset | Quantizer | Metric | N | Dim | Train (s) | Encode (s) |")
    lines.append("|---------|-----------|--------|---|-----|-----------|------------|")
    for r in records:
        lines.append(
            f"| {r.get('dataset_label','?')} "
            f"| {r.get('quantizer','?')} "
            f"| {r.get('metric','?')} "
            f"| {r.get('n','?')} "
            f"| {r.get('d','?')} "
            f"| {r.get('train_sec', float('nan')):.3f} "
            f"| {r.get('encode_sec', float('nan')):.3f} |"
        )
    lines.append("")
    lines.append("## Decision Rule")
    lines.append("")
    lines.append("If any `train_sec > 60` for the largest dataset of interest, "
                 "the codebook should be persisted as a sidecar file "
                 "(follow-up phase). Otherwise, the default is to re-train "
                 "on load inside `Quantizer::train(points)`.")
    lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"\nReport written to: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure codebook training/encoding wall time per quantizer.")
    parser.add_argument(
        "--dataset", action="append", default=[],
        help="label:path, e.g. nq500k:data/beir/nq500k/nq500k_points.pcs. "
             "Can be repeated.",
    )
    parser.add_argument("--metric", choices=("ip", "l2"), default="ip")
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--rabitq_bits", type=int, default=7)
    parser.add_argument(
        "--quantizer", choices=("all", *QUANTIZERS), default="all",
        help="restrict to a single quantizer (default: run all).",
    )
    parser.add_argument("--binary", default=DEFAULT_BINARY,
                        help="path to the measure_codebook_time binary.")
    parser.add_argument("--build", action="store_true",
                        help="bazel build -c opt the binary first.")
    parser.add_argument("--mmap", action="store_true",
                        help="pass -mmap to load the .pcs via mmap.")
    parser.add_argument("--out", default="experiments/codebook_timings.md")
    args = parser.parse_args()

    if not args.dataset:
        parser.error("at least one --dataset required")

    root = Path(__file__).resolve().parent.parent

    if args.build:
        print("$ bazel build -c opt //mvsic/tools:measure_codebook_time")
        subprocess.run(
            ["bazel", "build", "-c", "opt",
             "//mvsic/tools:measure_codebook_time"],
            cwd=root, check=True,
        )

    binary = Path(args.binary)
    if not binary.is_absolute():
        binary = root / binary
    if not binary.exists():
        parser.error(
            f"binary not found at {binary}; pass --build or set --binary")

    quantizers = QUANTIZERS if args.quantizer == "all" else (args.quantizer,)

    records: list[dict] = []
    for entry in args.dataset:
        if ":" not in entry:
            parser.error(f"--dataset must be label:path, got {entry!r}")
        label, path = entry.split(":", 1)
        ds_path = Path(path)
        if not ds_path.is_absolute():
            ds_path = root / ds_path
        if not ds_path.exists():
            print(f"[WARN] skipping {label}: {ds_path} not found", file=sys.stderr)
            continue
        for q in quantizers:
            extra: list[str] = []
            if args.mmap:
                extra.append("-mmap")
            rec = run_quantizer(
                binary=binary, dataset_label=label, dataset_path=ds_path,
                quantizer=q, metric=args.metric,
                block_size=args.block_size, rabitq_bits=args.rabitq_bits,
                extra=extra,
            )
            if rec is not None:
                records.append(rec)

    if not records:
        print("No successful runs -- not writing report.", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    write_report(records, out_path, cmd=sys.argv, root=root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
