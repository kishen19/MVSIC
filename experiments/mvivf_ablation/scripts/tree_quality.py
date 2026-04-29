#!/usr/bin/env python3
"""Wrap the leaf-diversity microbenchmark for use in the MVIVF ablation.

Runs ``bazel-bin/microbenchmark/mvivf/bench_mvivf_leaf_diversity`` on a
pre-built MVIVF index and parses the output into a JSON record for downstream
aggregation / plotting.

The underlying binary computes, for each query's top-k ground-truth
neighbors:

- **Distinct leaves per query** — lower = tighter tree (this is the
  "NN-spread" metric referenced in the ablation plan).
- **Distinct root children per query** — corresponds to the number of
  top-level subtrees the GT spans.
- **Greedy set-cover size** and the **best / second-best leaf counts** —
  how concentrated the top-k are in a single leaf vs. scattered.

Usage:
    scripts/tree_quality.py \\
        --index results/nq500k/mvivf/indexes/mvivf_winner.bin \\
        --db    data/beir/nq500k/nq500k_points.pcs \\
        --queries data/beir/nq500k/nq500k_queries.pcs \\
        --gt    data/beir/nq500k/nq500k_chamfer_neighbors.gt \\
        --k 10 \\
        --metric ip \\
        --out   results/nq500k/mvivf/tree_quality_winner.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

DEFAULT_BINARY = "bazel-bin/microbenchmark/mvivf/bench_mvivf_leaf_diversity"


def parse_output(text: str) -> dict:
    """Extract numeric fields from the free-form binary output."""
    out: dict = {"raw": text}

    # Collect a couple of straightforward lines.
    def _grab(pattern: str, cast=float) -> float | None:
        m = re.search(pattern, text)
        if not m:
            return None
        try:
            return cast(m.group(1))
        except (TypeError, ValueError):
            return None

    out["num_internal_nodes"] = _grab(r"num_internal_nodes\s*:\s*(\d+)", int)
    out["num_leaves"] = _grab(r"num_leaves\s*:\s*(\d+)", int)
    out["avg_leaf_size"] = _grab(r"avg_leaf_size\s*:\s*([-+\d.eE]+)")
    out["avg_internal_node_size"] = _grab(r"avg_internal_node_size\s*:\s*([-+\d.eE]+)")
    out["height"] = _grab(r"height\s*:\s*(\d+)", int)
    out["avg_child_fraction_imbalance"] = _grab(
        r"avg_child_fraction_imbalance\s*:\s*([-+\d.eE]+)")
    out["max_child_fraction_imbalance"] = _grab(
        r"max_child_fraction_imbalance\s*:\s*([-+\d.eE]+)")

    # Leaf-diversity aggregate block.
    leaf_block = re.search(
        r"Distinct leaves per query.*?avg\s*:\s*([-+\d.eE]+).*?"
        r"median\s*:\s*([-+\d.eE]+).*?min\s*:\s*(\d+).*?max\s*:\s*(\d+)",
        text, re.DOTALL)
    if leaf_block:
        out["leaf_avg"] = float(leaf_block.group(1))
        out["leaf_median"] = float(leaf_block.group(2))
        out["leaf_min"] = int(leaf_block.group(3))
        out["leaf_max"] = int(leaf_block.group(4))

    root_block = re.search(
        r"Distinct root children per query.*?avg\s*:\s*([-+\d.eE]+).*?"
        r"median\s*:\s*([-+\d.eE]+).*?min\s*:\s*(\d+).*?max\s*:\s*(\d+)",
        text, re.DOTALL)
    if root_block:
        out["root_child_avg"] = float(root_block.group(1))
        out["root_child_median"] = float(root_block.group(2))
        out["root_child_min"] = int(root_block.group(3))
        out["root_child_max"] = int(root_block.group(4))

    greedy_block = re.search(
        r"Greedy min-cover size \(leaves, estimate\):.*?avg\s*:\s*([-+\d.eE]+)"
        r".*?median:\s*([-+\d.eE]+).*?min:\s*(\d+).*?max:\s*(\d+)",
        text, re.DOTALL)
    if greedy_block:
        out["greedy_leaf_cover_avg"] = float(greedy_block.group(1))
        out["greedy_leaf_cover_median"] = float(greedy_block.group(2))
        out["greedy_leaf_cover_min"] = int(greedy_block.group(3))
        out["greedy_leaf_cover_max"] = int(greedy_block.group(4))

    best_block = re.search(
        r"Best leaf \(1st in greedy order\).*?avg\s*:\s*([-+\d.eE]+)",
        text, re.DOTALL)
    if best_block:
        out["best_leaf_count_avg"] = float(best_block.group(1))

    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True, type=pathlib.Path,
                   help="pre-built MVIVF .bin index")
    p.add_argument("--db", required=True, type=pathlib.Path,
                   help="points .pcs file")
    p.add_argument("--queries", required=True, type=pathlib.Path,
                   help="queries .pcs file")
    p.add_argument("--gt", required=True, type=pathlib.Path,
                   help="ground-truth file")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--metric", choices=("ip", "l2"), default="ip")
    p.add_argument("--binary", default=DEFAULT_BINARY,
                   help="path to bench_mvivf_leaf_diversity")
    p.add_argument("--out", type=pathlib.Path, default=None,
                   help="write JSON record to this file; otherwise stdout")
    p.add_argument("--build", action="store_true",
                   help="bazel build the microbenchmark first")
    args = p.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parent.parent.parent.parent
    binary = pathlib.Path(args.binary)
    if not binary.is_absolute():
        binary = repo_root / binary
    if args.build:
        subprocess.run(
            ["bazel", "build", "-c", "opt",
             "//microbenchmark/mvivf:bench_mvivf_leaf_diversity"],
            cwd=repo_root, check=True,
        )
    if not binary.exists():
        sys.exit(f"binary not found at {binary}; pass --build or --binary")

    dist_func = "L2" if args.metric == "l2" else "IP"
    cmd = [
        str(binary),
        "-i", str(args.db),
        "-q", str(args.queries),
        "-gt", str(args.gt),
        "-x", str(args.index),
        "-k", str(args.k),
        "-dist_func", dist_func,
    ]
    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        print(res.stdout, file=sys.stderr)
        print(res.stderr, file=sys.stderr)
        sys.exit(f"bench_mvivf_leaf_diversity failed (rc={res.returncode})")

    record = parse_output(res.stdout)
    record["metric"] = args.metric
    record["k"] = args.k
    record["index"] = str(args.index)

    # Drop raw output from persisted JSON (it can be large).
    record_no_raw = {k: v for k, v in record.items() if k != "raw"}
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record_no_raw, indent=2))
        print(f"Wrote {args.out}")
    else:
        json.dump(record_no_raw, sys.stdout, indent=2)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
