"""End-to-end dataset embedding pipeline."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from .datasets import DatasetBundle, QuerySet, Record, load_bundle
from .encoders import Encoder, create_encoder
from .formats import (
    AtomicTextWriter,
    StreamingPCSWriter,
    iter_ids,
    read_ids,
    read_pcs_layout,
    validate_exact_gt,
    write_gold_gt,
    write_json,
)
from .ground_truth import compute_ground_truth


LOGGER = logging.getLogger(__name__)
DEFAULT_SPLITS = {"beir": "test", "lotte": "dev", "vidore": "test"}
DEFAULT_CHECKPOINTS = {
    "beir": "colbert-ir/colbertv2.0",
    "lotte": "colbert-ir/colbertv2.0",
    "vidore": "vidore/colpali-v1.3-hf",
}


@dataclass(frozen=True)
class GenerationOptions:
    suite: str
    name: str
    data_root: Path
    source: str | None
    split: str
    checkpoint: str
    batch_size: int
    query_batch_size: int
    doc_maxlen: int
    query_maxlen: int
    dtype: str
    threads: int | None
    zero_atol: float
    limit: int | None
    overwrite: bool
    compute_gt: bool
    gt_k: int
    metric: str
    chunk_clouds: int
    batch_queries: int
    quiet: bool


@dataclass
class PreparedQueries:
    records: list[Record]
    gold_rows: list[list[int]] | None
    original_count: int
    filtered_count: int
    missing_references: int


def _progress(iterable: Iterable, *, total: int, description: str, quiet: bool):
    if quiet:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=description, unit="item")
    except ImportError:
        return iterable


def _batches(records: Iterable[Record], batch_size: int) -> Iterator[list[Record]]:
    batch: list[Record] = []
    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _encode_records(
    records: Iterable[Record],
    *,
    count: int,
    pcs_path: Path,
    ids_path: Path,
    encoder: Callable[[Sequence[Any]], list],
    batch_size: int,
    relevant_ids: set[str],
    description: str,
    quiet: bool,
) -> tuple[dict[str, int], dict[str, int]]:
    if count <= 0:
        raise ValueError(f"{description}: no records remain to encode")
    writer = StreamingPCSWriter(pcs_path, count)
    id_writer = AtomicTextWriter(ids_path)
    relevant_map: dict[str, int] = {}
    seen_ids: set[str] = set()
    emitted = 0
    pcs_committed = False
    ids_committed = False
    try:
        batches = _batches(records, batch_size)
        total_batches = (count + batch_size - 1) // batch_size
        for batch in _progress(
            batches, total=total_batches, description=description, quiet=quiet
        ):
            values = [record.value for record in batch]
            embeddings = encoder(values)
            if len(embeddings) != len(batch):
                raise ValueError(
                    f"{description}: encoder returned {len(embeddings)} items for {len(batch)} inputs"
                )
            for record, embedding in zip(batch, embeddings):
                if record.id in seen_ids:
                    raise ValueError(f"{description}: duplicate source ID {record.id!r}")
                seen_ids.add(record.id)
                if record.id in relevant_ids:
                    relevant_map[record.id] = emitted
                writer.append(embedding)
                id_writer.write_id(record.id)
                emitted += 1
        if emitted != count:
            raise ValueError(f"{description}: emitted {emitted} records, expected {count}")
        layout = writer.finalize()
        pcs_committed = True
        id_writer.finalize()
        ids_committed = True
    except Exception:
        writer.abort()
        id_writer.abort()
        # If either rename succeeded, remove both paths rather than leave a
        # new file paired with a stale file from an overwritten generation.
        if pcs_committed or ids_committed:
            pcs_path.unlink(missing_ok=True)
            ids_path.unlink(missing_ok=True)
        raise
    return (
        {
            "count": layout.count,
            "dim": layout.dim,
            "num_vectors": layout.num_vectors,
        },
        relevant_map,
    )


def _reuse_corpus(
    pcs_path: Path, ids_path: Path, relevant_ids: set[str]
) -> tuple[dict[str, int], dict[str, int]]:
    if pcs_path.exists() != ids_path.exists():
        raise ValueError(
            f"partial corpus output exists ({pcs_path}, {ids_path}); use --overwrite"
        )
    layout = read_pcs_layout(pcs_path)
    relevant_map: dict[str, int] = {}
    id_count = 0
    seen: set[str] = set()
    for row_id, external_id in enumerate(iter_ids(ids_path)):
        if external_id in seen:
            raise ValueError(f"{ids_path}: duplicate ID {external_id!r}")
        seen.add(external_id)
        if external_id in relevant_ids:
            relevant_map[external_id] = row_id
        id_count += 1
    if id_count != layout.count:
        raise ValueError(
            f"{ids_path}: contains {id_count} IDs for {layout.count} point clouds"
        )
    return (
        {"count": layout.count, "dim": layout.dim, "num_vectors": layout.num_vectors},
        relevant_map,
    )


def _prepare_queries(query_set: QuerySet, document_rows: dict[str, int]) -> PreparedQueries:
    if query_set.qrels is None:
        return PreparedQueries(
            records=query_set.records,
            gold_rows=None,
            original_count=len(query_set.records),
            filtered_count=0,
            missing_references=0,
        )

    records: list[Record] = []
    gold_rows: list[list[int]] = []
    missing_references = 0
    for record in query_set.records:
        source_documents = query_set.qrels.get(record.id, [])
        mapped: list[int] = []
        for document_id in source_documents:
            row = document_rows.get(document_id)
            if row is None:
                missing_references += 1
            else:
                mapped.append(row)
        mapped = sorted(set(mapped))
        if not mapped:
            continue
        records.append(record)
        gold_rows.append(mapped)
    return PreparedQueries(
        records=records,
        gold_rows=gold_rows,
        original_count=len(query_set.records),
        filtered_count=len(query_set.records) - len(records),
        missing_references=missing_references,
    )


def _corpus_stem(options: GenerationOptions) -> str:
    default_split = DEFAULT_SPLITS[options.suite]
    return options.name if options.split == default_split else f"{options.name}-{options.split}"


def _query_stem(corpus_stem: str, query_name: str) -> str:
    return corpus_stem if query_name in ("search", "default") else f"{corpus_stem}-{query_name}"


def _requested_encoder_metadata(options: GenerationOptions) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": "colpali" if options.suite == "vidore" else "colbert",
        "checkpoint": options.checkpoint,
        "padding": f"remove rows with L2 norm <= {options.zero_atol:g}",
        "normalization": "row-wise L2",
    }
    if options.suite == "vidore":
        result["dtype"] = options.dtype
    else:
        result["doc_maxlen"] = options.doc_maxlen
        result["query_maxlen"] = options.query_maxlen
    return result


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read existing manifest {path}; use --overwrite") from error
    if not isinstance(value, dict):
        raise ValueError(f"existing manifest {path} is not a JSON object; use --overwrite")
    return value


def _validate_reuse_manifest(
    manifest: dict[str, Any], options: GenerationOptions, bundle: DatasetBundle
) -> None:
    """Reject reuse when known provenance differs from this invocation."""
    dataset = manifest.get("dataset", {})
    encoder = manifest.get("encoder", {})
    previous_options = manifest.get("options", {})
    expected = {
        "dataset suite": (dataset.get("suite"), bundle.suite),
        "dataset name": (dataset.get("name"), bundle.name),
        "source": (dataset.get("source"), bundle.source),
        "split": (dataset.get("split"), bundle.split),
        "checkpoint": (encoder.get("checkpoint"), options.checkpoint),
        "limit": (previous_options.get("limit"), options.limit),
        "zero padding tolerance": (
            previous_options.get("zero_atol"),
            options.zero_atol,
        ),
    }
    if options.suite == "vidore":
        expected["dtype"] = (encoder.get("dtype"), options.dtype)
    else:
        expected["document max length"] = (
            encoder.get("doc_maxlen"),
            options.doc_maxlen,
        )
        expected["query max length"] = (
            encoder.get("query_maxlen"),
            options.query_maxlen,
        )
    mismatches = [
        f"{label}: existing={old!r}, requested={new!r}"
        for label, (old, new) in expected.items()
        if old != new
    ]
    if mismatches:
        detail = "; ".join(mismatches)
        raise ValueError(f"existing generated data is incompatible ({detail}); use --overwrite")


def generate(options: GenerationOptions) -> Path:
    if "/" in options.name or "\\" in options.name:
        raise ValueError("dataset name must be a single path component; use --source for remote names")
    output_dir = options.data_root / options.suite / options.name
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Loading %s dataset %s (%s)", options.suite, options.name, options.split)
    bundle: DatasetBundle = load_bundle(
        options.suite,
        options.name,
        output_dir,
        source=options.source,
        split=options.split,
        limit=options.limit,
    )

    relevant_ids = {
        document_id
        for query_set in bundle.query_sets
        if query_set.qrels is not None
        for documents in query_set.qrels.values()
        for document_id in documents
    }
    corpus_stem = _corpus_stem(options)
    points_path = output_dir / f"{corpus_stem}_points.pcs"
    point_ids_path = output_dir / f"{corpus_stem}_point_ids.txt"
    manifest_path = output_dir / f"{corpus_stem}_manifest.json"
    previous_manifest = _read_manifest(manifest_path) if not options.overwrite else None
    if previous_manifest is not None and (points_path.exists() or point_ids_path.exists()):
        _validate_reuse_manifest(previous_manifest, options, bundle)

    encoder_instance: Encoder | None = None

    def encoder() -> Encoder:
        nonlocal encoder_instance
        if encoder_instance is None:
            LOGGER.info("Loading embedding model %s", options.checkpoint)
            encoder_instance = create_encoder(
                options.suite,
                options.checkpoint,
                doc_maxlen=options.doc_maxlen,
                query_maxlen=options.query_maxlen,
                dtype=options.dtype,
                threads=options.threads,
                zero_atol=options.zero_atol,
            )
        return encoder_instance

    manifest_queries: dict[str, Any] = {}
    try:
        if points_path.exists() and point_ids_path.exists() and not options.overwrite:
            LOGGER.info("Reusing existing corpus embeddings: %s", points_path)
            corpus_stats, document_rows = _reuse_corpus(
                points_path, point_ids_path, relevant_ids
            )
        else:
            if (points_path.exists() or point_ids_path.exists()) and not options.overwrite:
                raise ValueError("partial corpus output exists; pass --overwrite to replace it")
            LOGGER.info("Encoding %d corpus items", bundle.corpus.count)
            corpus_stats, document_rows = _encode_records(
                bundle.corpus,
                count=bundle.corpus.count,
                pcs_path=points_path,
                ids_path=point_ids_path,
                encoder=encoder().encode_documents,
                batch_size=options.batch_size,
                relevant_ids=relevant_ids,
                description=f"{options.name} corpus",
                quiet=options.quiet,
            )

        missing_documents = relevant_ids - document_rows.keys()
        if missing_documents:
            LOGGER.warning(
                "%d judged document IDs are absent from the generated corpus",
                len(missing_documents),
            )

        for query_set in bundle.query_sets:
            prepared = _prepare_queries(query_set, document_rows)
            if not prepared.records:
                raise ValueError(
                    f"{query_set.name}: no queries remain after filtering against gold judgments"
                )
            stem = _query_stem(corpus_stem, query_set.name)
            queries_path = output_dir / f"{stem}_queries.pcs"
            query_ids_path = output_dir / f"{stem}_query_ids.txt"
            gold_path = output_dir / f"{stem}_gold_neighbors.csr"
            gt_path = output_dir / f"{stem}_chamfer_neighbors.gt"
            expected_query_ids = [record.id for record in prepared.records]

            if queries_path.exists() and query_ids_path.exists() and not options.overwrite:
                layout = read_pcs_layout(queries_path)
                existing_ids = read_ids(query_ids_path)
                if existing_ids != expected_query_ids or layout.count != len(expected_query_ids):
                    raise ValueError(
                        f"existing {query_set.name} queries do not match the filtered source; use --overwrite"
                    )
                query_stats = {
                    "count": layout.count,
                    "dim": layout.dim,
                    "num_vectors": layout.num_vectors,
                }
                LOGGER.info("Reusing existing %s queries: %s", query_set.name, queries_path)
            else:
                if (queries_path.exists() or query_ids_path.exists()) and not options.overwrite:
                    raise ValueError(
                        f"partial {query_set.name} query output exists; use --overwrite"
                    )
                query_stats, _ = _encode_records(
                    prepared.records,
                    count=len(prepared.records),
                    pcs_path=queries_path,
                    ids_path=query_ids_path,
                    encoder=encoder().encode_queries,
                    batch_size=options.query_batch_size,
                    relevant_ids=set(),
                    description=f"{options.name} {query_set.name} queries",
                    quiet=options.quiet,
                )
            if query_stats["dim"] != corpus_stats["dim"]:
                raise ValueError(
                    f"{query_set.name}: query dimension {query_stats['dim']} does not match "
                    f"corpus dimension {corpus_stats['dim']}"
                )

            if prepared.gold_rows is not None:
                write_gold_gt(gold_path, prepared.gold_rows)
                gold_file: str | None = gold_path.name
            else:
                if options.overwrite:
                    gold_path.unlink(missing_ok=True)
                gold_file = None

            exact_gt: dict[str, Any] | None = None
            if options.compute_gt:
                if gt_path.exists() and not options.overwrite:
                    if previous_manifest is None:
                        raise ValueError(
                            f"cannot verify the metric used by {gt_path} because the manifest "
                            "is missing; use --overwrite"
                        )
                    previous_metric = previous_manifest.get("metric")
                    if previous_metric != options.metric:
                        raise ValueError(
                            f"{gt_path}: existing metric is {previous_metric!r}, requested "
                            f"{options.metric!r}; use --overwrite"
                        )
                    exact_gt = validate_exact_gt(gt_path, len(prepared.records))
                    expected_k = min(options.gt_k, corpus_stats["count"])
                    if exact_gt["k"] != expected_k:
                        raise ValueError(
                            f"{gt_path}: contains k={exact_gt['k']}, requested k={expected_k}; "
                            "use --overwrite"
                        )
                    LOGGER.info("Reusing existing exact ground truth: %s", gt_path)
                else:
                    LOGGER.info("Computing exact %s ground truth for %s", options.metric, stem)
                    exact_gt = compute_ground_truth(
                        points_path,
                        queries_path,
                        gt_path,
                        k=options.gt_k,
                        metric=options.metric,
                        chunk_clouds=options.chunk_clouds,
                        batch_queries=options.batch_queries,
                        threads=options.threads,
                        quiet=options.quiet,
                    )
            elif options.overwrite:
                # Re-embedded point clouds invalidate exact neighbors from an
                # earlier generation unless they are recomputed in this run.
                gt_path.unlink(missing_ok=True)

            manifest_queries[query_set.name] = {
                **query_stats,
                "stem": stem,
                "queries": queries_path.name,
                "ids": query_ids_path.name,
                "gold_gt": gold_file,
                "exact_gt": gt_path.name if exact_gt is not None else None,
                "exact_gt_stats": exact_gt,
                "original_count": prepared.original_count,
                "filtered_no_gold": prepared.filtered_count,
                "missing_gold_document_references": prepared.missing_references,
            }

        encoder_metadata = (
            encoder_instance.metadata()
            if encoder_instance is not None
            else _requested_encoder_metadata(options)
        )
        manifest = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "command": [sys.executable, "-m", "embed", *sys.argv[1:]],
            "git_revision": _git_revision(),
            "dataset": {
                "suite": bundle.suite,
                "name": bundle.name,
                "source": bundle.source,
                "split": bundle.split,
                **bundle.metadata,
            },
            "encoder": encoder_metadata,
            "metric": options.metric,
            "corpus": {
                **corpus_stats,
                "stem": corpus_stem,
                "points": points_path.name,
                "ids": point_ids_path.name,
            },
            "queries": manifest_queries,
            "options": {
                "limit": options.limit,
                "zero_atol": options.zero_atol,
                "gt_k": options.gt_k,
                "chunk_clouds": options.chunk_clouds,
                "batch_queries": options.batch_queries,
            },
        }
        write_json(manifest_path, manifest)
        LOGGER.info("Dataset ready: %s", output_dir)
        return manifest_path
    finally:
        if encoder_instance is not None:
            encoder_instance.close()
