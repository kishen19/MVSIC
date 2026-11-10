"""FiQA train/dev encoder for the EMDIVF experiment.

Run as python -m emdivf_experiments.embedding.fiqa_splits from the repo root.
Use the same ColBERT environment used by embed.
The shared FiQA corpus and existing test embeddings are never rewritten.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from embed.encoders import ColBERTEncoder
from embed.formats import AtomicTextWriter, StreamingPCSWriter, iter_pcs, read_pcs_layout


def load_split(source: Path, split: str) -> list[tuple[str, str]]:
    from beir.datasets.data_loader import GenericDataLoader

    _, queries, qrels = GenericDataLoader(data_folder=str(source)).load(split=split)
    positives = {
        str(query_id)
        for query_id, documents in qrels.items()
        if any(float(score) > 0 for score in documents.values())
    }
    result = [(str(query_id), text) for query_id, text in queries.items() if str(query_id) in positives]
    if len(result) != len(positives):
        raise ValueError(f"{split}: qrels reference IDs missing from queries.jsonl")
    if not result:
        raise ValueError(f"{split}: no positively judged queries")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="extracted BEIR FiQA directory")
    parser.add_argument("--output", type=Path, default=Path("data/beir/fiqa"))
    parser.add_argument("--checkpoint", default="colbert-ir/colbertv2.0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    split_rows = {split: load_split(args.source, split) for split in ("train", "dev", "test")}
    id_sets = {split: {item[0] for item in rows} for split, rows in split_rows.items()}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        print(f"{left}/{right} ID overlap: {len(id_sets[left] & id_sets[right])}", flush=True)
    # BEIR FiQA can reuse a query ID between qrels splits. This is not a leak
    # if each split is loaded separately, so record overlaps rather than reject.
    existing_test = read_pcs_layout(args.output / "fiqa_queries.pcs")
    if existing_test.count != len(split_rows["test"]) or existing_test.dim != 128:
        raise ValueError("existing FiQA test PCS does not match the official test split")
    print(f"test count verified: {existing_test.count}; train={len(split_rows['train'])}; dev={len(split_rows['dev'])}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    with ColBERTEncoder(
        args.checkpoint,
        doc_maxlen=300,
        query_maxlen=32,
        threads=args.threads,
        zero_atol=1e-9,
    ) as encoder:
        # Confirm the current model/preprocessing reproduces the test query
        # before writing either new split.
        reference = next(iter_pcs(args.output / "fiqa_queries.pcs"))
        reproduced = encoder.encode_queries([split_rows["test"][0][1]])[0]
        if reference.shape != reproduced.shape or not np.allclose(
            reference, reproduced, atol=2e-4, rtol=2e-4
        ):
            maximum_error = (
                float(np.max(np.abs(reference - reproduced)))
                if reference.shape == reproduced.shape
                else float("inf")
            )
            raise ValueError(
                "ColBERT output does not match existing FiQA test embeddings "
                f"(shape {reference.shape} vs {reproduced.shape}, max error {maximum_error}); "
                "check model and preprocessing before encoding train/dev"
            )
        print("existing test query reproduction verified", flush=True)
        for split in ("train", "dev"):
            rows = split_rows[split]
            pcs_path = args.output / f"fiqa_queries_{split}.pcs"
            ids_path = args.output / f"fiqa_query_ids_{split}.txt"
            if pcs_path.exists() or ids_path.exists():
                raise FileExistsError(f"{split}: output already exists; refusing to overwrite")
            writer = StreamingPCSWriter(pcs_path, len(rows))
            ids_writer = AtomicTextWriter(ids_path)
            try:
                for start in range(0, len(rows), args.batch_size):
                    batch = rows[start : start + args.batch_size]
                    embeddings = encoder.encode_queries([text for _, text in batch])
                    for (query_id, _), embedding in zip(batch, embeddings):
                        writer.append(embedding)
                        ids_writer.write_id(query_id)
                    if start % (args.batch_size * 20) == 0:
                        print(f"{split}: {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
                writer.finalize()
                ids_writer.finalize()
            except Exception:
                writer.abort()
                ids_writer.abort()
                raise
            print(f"wrote {pcs_path} and {ids_path}", flush=True)


if __name__ == "__main__":
    main()
