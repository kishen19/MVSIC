#!/usr/bin/env python3
"""Filter a benchmark build/search YAML by `--dataset` and/or `--method`.

Used by experiments/builds/scripts/run_builds.sh and the per-stage runners
(run_latency.sh, run_multi_latency.sh, run_batch.sh).

Both filters pass through unchanged if not provided.

Optional ``--strip-indices`` removes named entries from ``indices`` after the
other filters (runners use this to scope FastPlaid without editing YAML files).
"""
from __future__ import annotations

import argparse
import sys

try:
    import yaml
except ImportError as e:
    print(
        "filter_config.py requires PyYAML (install with: pip install pyyaml).",
        file=sys.stderr,
    )
    raise SystemExit(1) from e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default=None, help="Match datasets[].name (single dataset).")
    ap.add_argument("--method", default=None, help="Match indices[].name (single method).")
    ap.add_argument(
        "--strip-indices",
        default=None,
        help=(
            "Comma-separated index names to remove from ``indices`` after other filters "
            "(used by run scripts to scope FastPlaid without editing checked-in YAML)."
        ),
    )
    args = ap.parse_args()

    with open(args.in_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise SystemExit("config root must be a mapping")

    if args.dataset:
        datasets = cfg.get("datasets") or []
        sel = [d for d in datasets if isinstance(d, dict) and str(d.get("name")) == args.dataset]
        if not sel:
            names = [d.get("name") for d in datasets if isinstance(d, dict)]
            raise SystemExit(
                f"dataset {args.dataset!r} not found; available: {names}"
            )
        cfg["datasets"] = sel

    if args.method:
        indices = cfg.get("indices") or []
        sel = [i for i in indices if isinstance(i, dict) and str(i.get("name")) == args.method]
        if not sel:
            names = [i.get("name") for i in indices if isinstance(i, dict)]
            raise SystemExit(
                f"method {args.method!r} not found; available: {names}"
            )
        cfg["indices"] = sel

    if args.strip_indices:
        drop = {x.strip() for x in args.strip_indices.split(",") if x.strip()}
        indices = cfg.get("indices") or []
        cfg["indices"] = [
            i
            for i in indices
            if isinstance(i, dict) and str(i.get("name")) not in drop
        ]

    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
