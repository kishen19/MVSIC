"""Command-line entry point for complete MVSIC dataset generation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import DEFAULT_CHECKPOINTS, DEFAULT_SPLITS, GenerationOptions, generate


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m embed",
        description=(
            "Generate a complete MVSIC dataset: embeddings, ID mappings, gold "
            "judgments when available, a manifest, and optionally exact Chamfer GT."
        ),
    )
    parser.add_argument("suite", choices=("beir", "lotte", "vidore"))
    parser.add_argument("dataset", help="local dataset name (for example scifact or pooled)")
    parser.add_argument(
        "--source",
        help="remote dataset name; primarily for ViDoRe (defaults to vidore/<dataset>)",
    )
    parser.add_argument(
        "--split",
        help="source split (defaults: BEIR=test, LoTTE=dev, ViDoRe=test)",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", help="model checkpoint override")
    parser.add_argument("--batch-size", type=_positive)
    parser.add_argument("--query-batch-size", type=_positive)
    parser.add_argument("--doc-maxlen", type=_positive, default=300)
    parser.add_argument("--query-maxlen", type=_positive, default=32)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--threads", type=_positive)
    parser.add_argument("--zero-atol", type=float, default=1e-9)
    parser.add_argument(
        "--limit",
        type=_positive,
        help="encode only the first N corpus/query records (smoke tests only)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compute-gt", action="store_true")
    parser.add_argument("--gt-k", type=_positive, default=2000)
    parser.add_argument("--metric", choices=("ip", "l2"), default="ip")
    parser.add_argument("--chunk-clouds", type=_positive, default=20_000)
    parser.add_argument("--batch-queries", type=_positive, default=64)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.zero_atol < 0:
        parser.error("--zero-atol must be non-negative")
    split = args.split or DEFAULT_SPLITS[args.suite]
    if args.suite == "lotte" and split not in ("dev", "test"):
        parser.error("LoTTE --split must be dev or test")
    default_batch = 8 if args.suite == "vidore" else 64
    default_query_batch = 64
    options = GenerationOptions(
        suite=args.suite,
        name=args.dataset,
        data_root=args.data_root,
        source=args.source,
        split=split,
        checkpoint=args.checkpoint or DEFAULT_CHECKPOINTS[args.suite],
        batch_size=args.batch_size or default_batch,
        query_batch_size=args.query_batch_size or default_query_batch,
        doc_maxlen=args.doc_maxlen,
        query_maxlen=args.query_maxlen,
        dtype=args.dtype,
        threads=args.threads,
        zero_atol=args.zero_atol,
        limit=args.limit,
        overwrite=args.overwrite,
        compute_gt=args.compute_gt,
        gt_k=args.gt_k,
        metric=args.metric,
        chunk_clouds=args.chunk_clouds,
        batch_queries=args.batch_queries,
        quiet=args.quiet,
    )
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    manifest = generate(options)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

