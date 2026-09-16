"""Exact, memory-bounded Chamfer ground-truth computation."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np

from .formats import (
    mmap_pcs_offsets,
    mmap_pcs_values,
    read_pcs_layout,
    write_exact_gt,
)


LOGGER = logging.getLogger(__name__)


def _segment_reduce(
    values: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    axis: int,
    operation: str,
) -> np.ndarray:
    if operation == "max":
        reducer = np.maximum
    elif operation == "min":
        reducer = np.minimum
    elif operation == "sum":
        reducer = np.add
    else:
        raise ValueError(f"unknown segment reduction {operation!r}")
    if np.any(starts == ends):
        raise ValueError("exact ground truth does not support empty point clouds")
    return reducer.reduceat(values, starts, axis=axis)


def _merge_topk(
    top_distances: np.ndarray,
    top_ids: np.ndarray,
    chunk_distances: np.ndarray,
    chunk_start: int,
) -> None:
    k = top_distances.shape[1]
    combined = np.concatenate((top_distances, chunk_distances), axis=1)
    selected = np.argpartition(combined, kth=k - 1, axis=1)[:, :k]
    selected_distances = np.take_along_axis(combined, selected, axis=1)
    from_chunk = selected >= k
    old_positions = np.where(from_chunk, 0, selected)
    selected_ids = np.take_along_axis(top_ids, old_positions, axis=1)
    chunk_ids = (selected - k + chunk_start).astype(np.uint32, copy=False)
    selected_ids = np.where(from_chunk, chunk_ids, selected_ids)
    top_distances[:] = selected_distances
    top_ids[:] = selected_ids


def _progress(iterable, *, total: int, quiet: bool):
    if quiet:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc="Exact GT chunks")
    except ImportError:
        return iterable


def compute_ground_truth(
    database: str | os.PathLike[str],
    queries: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    k: int = 2000,
    metric: str = "ip",
    chunk_clouds: int = 20_000,
    batch_queries: int = 64,
    threads: int | None = None,
    quiet: bool = False,
) -> dict[str, int | float | str]:
    """Compute exact top-k asymmetric Chamfer neighbors.

    The corpus is memory-mapped and visited in cloud chunks. Queries are kept
    in RAM because benchmark query sets are much smaller than their corpora.
    """
    metric = metric.lower()
    if metric not in ("ip", "l2"):
        raise ValueError("metric must be 'ip' or 'l2'")
    if k <= 0 or chunk_clouds <= 0 or batch_queries <= 0:
        raise ValueError("k, chunk_clouds, and batch_queries must be positive")

    thread_context = None
    if threads:
        try:
            from threadpoolctl import threadpool_limits

            thread_context = threadpool_limits(limits=threads)
        except ImportError:
            LOGGER.warning("threadpoolctl is unavailable; --threads may not limit BLAS")

    start_time = time.perf_counter()
    try:
        query_layout = read_pcs_layout(queries)
        database_layout = read_pcs_layout(database)
        if query_layout.dim != database_layout.dim:
            raise ValueError(
                f"dimension mismatch: queries={query_layout.dim}, corpus={database_layout.dim}"
            )
        if query_layout.count == 0 or database_layout.count == 0:
            raise ValueError("queries and corpus must both be non-empty")

        query_offsets_raw = np.asarray(
            mmap_pcs_offsets(query_layout)[: query_layout.count + 1], dtype=np.uint64
        )
        database_offsets_raw = np.asarray(
            mmap_pcs_offsets(database_layout)[: database_layout.count + 1], dtype=np.uint64
        )
        query_offsets = (query_offsets_raw // query_layout.dim).astype(np.int64)
        database_offsets = (database_offsets_raw // database_layout.dim).astype(np.int64)
        query_sizes = np.diff(query_offsets)
        database_sizes = np.diff(database_offsets)
        if np.any(query_sizes <= 0):
            bad = int(np.flatnonzero(query_sizes <= 0)[0])
            raise ValueError(f"query point cloud {bad} is empty")
        if np.any(database_sizes <= 0):
            bad = int(np.flatnonzero(database_sizes <= 0)[0])
            raise ValueError(f"corpus point cloud {bad} is empty")

        query_values = np.array(mmap_pcs_values(query_layout), dtype=np.float32, copy=True)
        database_values = mmap_pcs_values(database_layout)
        effective_k = min(k, database_layout.count)
        if effective_k != k:
            LOGGER.warning(
                "requested k=%d exceeds corpus size %d; using k=%d",
                k,
                database_layout.count,
                effective_k,
            )

        top_distances = np.full(
            (query_layout.count, effective_k), np.inf, dtype=np.float32
        )
        top_ids = np.zeros((query_layout.count, effective_k), dtype=np.uint32)
        query_norms = None
        if metric == "l2":
            query_norms = np.einsum("ij,ij->i", query_values, query_values).astype(np.float32)

        starts = range(0, database_layout.count, chunk_clouds)
        num_chunks = (database_layout.count + chunk_clouds - 1) // chunk_clouds
        for database_start in _progress(starts, total=num_chunks, quiet=quiet):
            database_end = min(database_start + chunk_clouds, database_layout.count)
            vector_start = int(database_offsets[database_start])
            vector_end = int(database_offsets[database_end])
            database_chunk = np.ascontiguousarray(
                database_values[vector_start:vector_end], dtype=np.float32
            )
            local_offsets = (
                database_offsets[database_start : database_end + 1] - vector_start
            ).astype(np.int64)
            document_starts = local_offsets[:-1]
            document_ends = local_offsets[1:]
            num_documents = database_end - database_start
            chunk_distances = np.empty(
                (query_layout.count, num_documents), dtype=np.float32
            )
            database_norms = None
            if metric == "l2":
                database_norms = np.einsum(
                    "ij,ij->i", database_chunk, database_chunk
                ).astype(np.float32)

            for query_start in range(0, query_layout.count, batch_queries):
                query_end = min(query_start + batch_queries, query_layout.count)
                row_start = int(query_offsets[query_start])
                row_end = int(query_offsets[query_end])
                query_batch = query_values[row_start:row_end]
                local_query_offsets = (
                    query_offsets[query_start : query_end + 1] - row_start
                ).astype(np.int64)
                query_starts = local_query_offsets[:-1]
                query_ends = local_query_offsets[1:]
                query_counts = np.diff(local_query_offsets).astype(np.float32)

                products = query_batch @ database_chunk.T
                if metric == "ip":
                    per_document = _segment_reduce(
                        products,
                        document_starts,
                        document_ends,
                        axis=1,
                        operation="max",
                    )
                    sums = _segment_reduce(
                        per_document,
                        query_starts,
                        query_ends,
                        axis=0,
                        operation="sum",
                    )
                    chunk_distances[query_start:query_end] = -sums / query_counts[:, None]
                else:
                    assert query_norms is not None and database_norms is not None
                    squared = (
                        query_norms[row_start:row_end, None]
                        + database_norms[None, :]
                        - 2.0 * products
                    )
                    np.maximum(squared, 0.0, out=squared)
                    per_document = _segment_reduce(
                        squared,
                        document_starts,
                        document_ends,
                        axis=1,
                        operation="min",
                    )
                    sums = _segment_reduce(
                        per_document,
                        query_starts,
                        query_ends,
                        axis=0,
                        operation="sum",
                    )
                    chunk_distances[query_start:query_end] = sums / query_counts[:, None]

            _merge_topk(top_distances, top_ids, chunk_distances, database_start)

        ordering = np.lexsort((top_ids, top_distances), axis=1)
        top_distances = np.take_along_axis(top_distances, ordering, axis=1)
        top_ids = np.take_along_axis(top_ids, ordering, axis=1)
        write_exact_gt(output, top_distances, top_ids)
        elapsed = time.perf_counter() - start_time
        return {
            "metric": metric,
            "k": effective_k,
            "num_queries": query_layout.count,
            "num_documents": database_layout.count,
            "seconds": elapsed,
        }
    finally:
        if thread_context is not None:
            thread_context.unregister()
