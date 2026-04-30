#!/usr/bin/env python3
"""Filter a benchmark build/search YAML to a single `datasets` entry.

Used by run_ablation.sh when --dataset is set and only a multi-dataset config exists.
Requires PyYAML (same dependency as benchmarks/benchmark_build.py).
"""
from __future__ import annotations

import argparse
import sys

try:
    import yaml
except ImportError as e:
    print(
        "filter_benchmark_config_dataset.py requires PyYAML "
        "(install with: pip install pyyaml).",
        file=sys.stderr,
    )
    raise SystemExit(1) from e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--dataset", required=True, help="Must match datasets[].name")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.in_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise SystemExit("config root must be a mapping")
    datasets = cfg.get("datasets")
    if not datasets:
        raise SystemExit("config has no non-empty 'datasets' key")

    sel = [
        d
        for d in datasets
        if isinstance(d, dict) and str(d.get("name")) == args.dataset
    ]
    if not sel:
        names = [d.get("name") for d in datasets if isinstance(d, dict)]
        raise SystemExit(
            f"dataset {args.dataset!r} not found under datasets; available: {names}"
        )

    cfg["datasets"] = sel
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            cfg,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
