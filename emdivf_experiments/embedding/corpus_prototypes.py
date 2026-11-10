"""Corpus-only FiQA prototype pools for the isolated EMDIVF experiment.

Only corpus.jsonl and document PCS are inputs; no observed queries or qrels are
read. The paired controls use the same sampled source documents: a short text
span encoded as a ColBERT query, and 32 original document vectors selected for
diversity. Run with --dry-run to inspect sampling without encoding or writing.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

from embed.encoders import ColBERTEncoder
from embed.formats import (
    AtomicTextWriter,
    StreamingPCSWriter,
    mmap_pcs_offsets,
    mmap_pcs_values,
    read_pcs_layout,
    validate_pcs,
    write_json,
)


def load_corpus(path: Path) -> tuple[list[dict], str]:
    rows: list[dict] = []
    digest = hashlib.sha256()
    ids: set[str] = set()
    with path.open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            row = json.loads(raw)
            document_id = str(row["_id"])
            if document_id in ids:
                raise ValueError(f"duplicate corpus ID: {document_id}")
            ids.add(document_id)
            rows.append({"id": document_id, "title": str(row.get("title", "")),
                         "text": str(row.get("text", ""))})
    return rows, digest.hexdigest()


def sample_rows(rows: list[dict], sizes: np.ndarray, count: int, seed: int
                ) -> tuple[np.ndarray, dict]:
    """Proportional length/title strata; deterministic largest-remainder quotas."""
    eligible = np.asarray([i for i, row in enumerate(rows)
                           if sizes[i] >= 32 and (row["title"].strip() or row["text"].strip())],
                          dtype=np.int64)
    if not 0 < count <= len(eligible):
        raise ValueError(f"pool size must be in [1, {len(eligible)}], got {count}")
    boundaries = np.unique(np.quantile(sizes[eligible], [0.25, 0.5, 0.75]))
    strata: dict[tuple[int, bool], list[int]] = {}
    for i in eligible:
        key = (int(np.searchsorted(boundaries, sizes[i], side="right")),
               bool(rows[i]["title"].strip()))
        strata.setdefault(key, []).append(int(i))
    keys = sorted(strata)
    exact = {key: count * len(strata[key]) / len(eligible) for key in keys}
    quotas = {key: int(exact[key]) for key in keys}
    remaining = count - sum(quotas.values())
    for key in sorted(keys, key=lambda k: (-(exact[k] - quotas[k]), k))[:remaining]:
        quotas[key] += 1
    rng = np.random.default_rng(seed)
    selected = []
    for key in keys:
        selected.extend(rng.choice(strata[key], size=quotas[key], replace=False).tolist())
    selected = np.asarray(sorted(selected), dtype=np.int64)
    if len(selected) != count or len(np.unique(selected)) != count:
        raise AssertionError("sampling did not produce exactly the requested unique rows")
    return selected, {
        "eligible_documents": len(eligible), "excluded_documents": len(rows) - len(eligible),
        "eligibility": "at least 32 original document vectors and nonempty corpus text/title",
        "length_quartile_boundaries": boundaries.tolist(),
        "strata": [{"length_bin": key[0], "has_title": key[1],
                    "population": len(strata[key]), "sampled": quotas[key]} for key in keys],
    }


def pseudoquery(row: dict, seed: int, document_row: int) -> str:
    """Use a title and one short corpus span; no generated or observed questions."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, document_row]))
    title = " ".join(row["title"].split())
    sentences = [" ".join(s.split()) for s in re.split(r"(?<=[.!?])\s+|[\r\n]+", row["text"])]
    sentences = [s for s in sentences if len(s.split()) >= 5]
    text = sentences[int(rng.integers(len(sentences)))] if sentences else " ".join(row["text"].split())
    words = text.split()
    if len(words) > 24:
        start = int(rng.integers(len(words) - 24 + 1))
        words = words[start:start + 24]
    result = " ".join((title.split()[:8] + words)[:32]).strip()
    if not result:
        raise ValueError(f"empty pseudoquery from corpus row {document_row}")
    return result


def diverse_vector_indices(cloud: np.ndarray, count: int = 32) -> np.ndarray:
    """Farthest-first actual-vector subset; centroid-nearest initial vector."""
    cloud = np.asarray(cloud, dtype=np.float32)
    if len(cloud) < count or count <= 0:
        raise ValueError("cannot select distinct original vectors with this cardinality")
    if not np.isfinite(cloud).all():
        raise ValueError("nonfinite document vectors")
    centroid = cloud.mean(axis=0)
    first = int(np.square(cloud - centroid).sum(axis=1).argmin())
    selected = [first]
    # Squared Euclidean distances preserve the farthest-first ordering. Use the
    # original vectors without renormalizing, padding, repeating, or averaging.
    closest = np.square(cloud - cloud[first]).sum(axis=1)
    closest[first] = -np.inf
    while len(selected) < count:
        next_row = int(closest.argmax())
        selected.append(next_row)
        closest = np.minimum(closest, np.square(cloud - cloud[next_row]).sum(axis=1))
        closest[selected] = -np.inf
    return np.asarray(sorted(selected), dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/tmp/mvsic_emd_fiqa_source/fiqa/corpus.jsonl"))
    parser.add_argument("--docs", type=Path, default=Path("data/beir/fiqa/fiqa_points.pcs"))
    parser.add_argument("--output", type=Path,
                        default=Path("emdivf_experiments/artifacts/results/fiqa_corpus_prototypes"))
    parser.add_argument("--checkpoint", default="colbert-ir/colbertv2.0")
    parser.add_argument("--count", type=int, default=5500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.seed < 0 or args.batch_size <= 0 or args.threads <= 0:
        parser.error("seed must be nonnegative; batch-size and threads must be positive")

    rows, corpus_sha = load_corpus(args.source)
    layout = read_pcs_layout(args.docs)
    if layout.count != len(rows) or layout.dim != 128:
        raise ValueError("source corpus and document PCS counts/dimension differ")
    offsets = mmap_pcs_offsets(layout)
    values = mmap_pcs_values(layout)
    sizes = np.diff(offsets[:layout.count + 1]) // layout.dim
    selected, sampling = sample_rows(rows, sizes, args.count, args.seed)
    texts = [pseudoquery(rows[i], args.seed, int(i)) for i in selected]
    files = {
        "text": "fiqa_corpus_text_queries.pcs", "vectors": "fiqa_corpus_vector32_queries.pcs",
        "ids": "fiqa_corpus_prototype_ids.txt", "rows": "fiqa_corpus_prototype_rows.jsonl",
        "manifest": "manifest.json",
    }
    status = {
        "status": "sampled", "seed": args.seed, "pool_size": args.count,
        "source_corpus": str(args.source.resolve()), "source_corpus_sha256": corpus_sha,
        "source_document_pcs": str(args.docs.resolve()),
        "source_document_pcs_bytes": args.docs.stat().st_size,
        "source_document_count": layout.count, "source_dimension": layout.dim,
        "sampling": sampling,
        "source_query_files_read": [], "source_qrels_files_read": [],
        "pool_order": "ascending source corpus/PCS row after stratified sampling",
        "text_representation": "title plus deterministic random short sentence/span, encoded via ColBERT queryFromText",
        "vector_representation": "32 distinct original document vectors, farthest-first squared-L2 subset with centroid-nearest start",
        "text_duplicate_count": sum(n - 1 for n in Counter(texts).values()),
        "files": files,
    }
    if args.dry_run:
        print(json.dumps(status, indent=2))
        print("No files written; model not loaded.")
        return
    output = args.output.resolve()
    for name in files.values():
        if (output / name).exists():
            raise FileExistsError(f"refusing to overwrite {output / name}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest = output / files["manifest"]
    write_json(manifest, status)

    def cloud_at(i: int) -> np.ndarray:
        return values[int(offsets[i]) // layout.dim:int(offsets[i + 1]) // layout.dim]

    try:
        with ColBERTEncoder(args.checkpoint, doc_maxlen=300, query_maxlen=32,
                            threads=args.threads, zero_atol=1e-9) as encoder:
            # There is no legacy corpus ID sidecar. Confirm source order/model
            # on eight corpus rows before asserting paired source identities.
            checks = np.unique(np.linspace(0, layout.count - 1, 8, dtype=np.int64))
            old_texts = [(rows[i]["title"] + " " + rows[i]["text"]).strip() for i in checks]
            reproduced = encoder.encode_documents(old_texts)
            maximum_error = 0.0
            for i, observed in zip(checks, reproduced):
                reference = cloud_at(int(i))
                if reference.shape != observed.shape or not np.allclose(reference, observed, atol=2e-4, rtol=2e-4):
                    raise ValueError(f"legacy corpus reproduction/order check failed at row {i}; "
                                     f"shapes {reference.shape}/{observed.shape}")
                maximum_error = max(maximum_error, float(np.max(np.abs(reference - observed))))
            status.update(status="encoding", encoder=encoder.metadata(),
                          corpus_row_checks=checks.tolist(), corpus_check_max_error=maximum_error,
                          corpus_mapping_note="Eight source rows reproduced against PCS; no full-corpus embedding verification claimed")
            write_json(manifest, status)
            print(f"Verified source document rows/model on {len(checks)} rows; max error {maximum_error:g}", flush=True)
            with StreamingPCSWriter(output / files["text"], len(selected)) as text_writer, \
                 StreamingPCSWriter(output / files["vectors"], len(selected)) as vector_writer, \
                 AtomicTextWriter(output / files["ids"]) as id_writer, \
                 AtomicTextWriter(output / files["rows"]) as row_writer:
                for start in range(0, len(selected), args.batch_size):
                    batch_rows = selected[start:start + args.batch_size]
                    batch_texts = texts[start:start + args.batch_size]
                    encoded = encoder.encode_queries(batch_texts)
                    for i, text, embedding in zip(batch_rows, batch_texts, encoded):
                        if embedding.shape != (32, layout.dim):
                            raise ValueError(f"pseudoquery {i} has unexpected shape {embedding.shape}")
                        document = cloud_at(int(i))
                        subset = diverse_vector_indices(document)
                        prototype_id = "corpus:" + rows[i]["id"]
                        text_writer.append(embedding)
                        vector_writer.append(document[subset])
                        id_writer.write_id(prototype_id)
                        row_writer.write_id(json.dumps({"prototype_id": prototype_id,
                            "document_id": rows[i]["id"], "document_row": int(i),
                            "document_vectors": len(document), "text": text,
                            "selected_vector_rows": subset.tolist()}, ensure_ascii=False))
                    status.update(encoded=start + len(batch_rows), elapsed_seconds=time.perf_counter() - started)
                    write_json(manifest, status)
                    print(f"Encoded paired pools {start + len(batch_rows)}/{len(selected)} "
                          f"({status['elapsed_seconds']:.1f}s)", flush=True)
            status.update(status="complete", elapsed_seconds=time.perf_counter() - started,
                          validation={kind: validate_pcs(output / files[kind], norm_atol=2e-3)
                                      for kind in ("text", "vectors")})
            write_json(manifest, status)
            print(f"Completed paired corpus-only prototype pools in {output}", flush=True)
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}",
                      elapsed_seconds=time.perf_counter() - started)
        write_json(manifest, status)
        raise


if __name__ == "__main__":
    main()
