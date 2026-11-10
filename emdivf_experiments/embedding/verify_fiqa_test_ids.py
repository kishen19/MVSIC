"""Verify FiQA test query row/ID alignment and save a text ID sidecar.

This accompanies fiqa_splits.py for existing legacy test PCS.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from embed.encoders import ColBERTEncoder
from emdivf_experiments.embedding.fiqa_splits import load_split
from embed.formats import AtomicTextWriter, iter_pcs, read_pcs_layout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/beir/fiqa"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    rows = load_split(args.source, "test")
    pcs = args.output / "fiqa_queries.pcs"
    ids_path = args.output / "fiqa_query_ids.txt"
    if ids_path.exists():
        raise FileExistsError(ids_path)
    if read_pcs_layout(pcs).count != len(rows):
        raise ValueError("official test query count differs from existing PCS")

    references = iter_pcs(pcs)
    with ColBERTEncoder(
        "colbert-ir/colbertv2.0", doc_maxlen=300, query_maxlen=32,
        threads=args.threads, zero_atol=1e-9,
    ) as encoder:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            embeddings = encoder.encode_queries([text for _, text in batch])
            for offset, embedding in enumerate(embeddings):
                reference = next(references)
                if reference.shape != embedding.shape or not np.allclose(
                    reference, embedding, atol=2e-4, rtol=2e-4,
                ):
                    raise ValueError(f"test query row {start + offset} does not match its official ID")
            print(f"verified {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)

    writer = AtomicTextWriter(ids_path)
    for query_id, _ in rows:
        writer.write_id(query_id)
    writer.finalize()
    print(f"wrote {ids_path}", flush=True)


if __name__ == "__main__":
    main()
