#!/usr/bin/env python3
"""Rewrite a search YAML's `results_dir` paths from one stage folder to another.

Used by run_multi_latency.sh / run_batch.sh so that the latency YAMLs under
experiments/latency/configs/ are the single source of truth for the search
sweeps; only the output directory changes per stage.
"""
from __future__ import annotations

import argparse
import sys

try:
    import yaml
except ImportError as e:
    print(
        "rewrite_results_dir.py requires PyYAML (install with: pip install pyyaml).",
        file=sys.stderr,
    )
    raise SystemExit(1) from e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--from-stage", required=True, help="e.g. latency")
    ap.add_argument("--to-stage", required=True, help="e.g. batch / multi_latency")
    args = ap.parse_args()

    src_token = f"experiments/{args.from_stage}/"
    dst_token = f"experiments/{args.to_stage}/"

    with open(args.in_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise SystemExit("config root must be a mapping")

    for d in cfg.get("datasets", []) or []:
        if not isinstance(d, dict):
            continue
        rd = d.get("results_dir")
        if isinstance(rd, str) and src_token in rd:
            d["results_dir"] = rd.replace(src_token, dst_token)

    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
