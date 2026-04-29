"""
Python port of data-tools/compute_ground_truth.cpp, optimized for very large
multi-vector databases on CPU-only machines with many cores (e.g. 192-core
servers), tuned for the common "small-queries / huge-database" regime:

    * queries: <= 10_000 point clouds, each padded to a *uniform* number of
      vectors per cloud (typically 32).  The whole query set fits in RAM.
    * database: potentially hundreds of GB, memory-mapped and streamed in
      chunks.

Fast paths enabled by the query assumptions:
    * The per-query axis reduction (sum of per-cloud-max across the Q vectors
      of a query) becomes a pure `reshape(B, Q, C).sum(axis=1)`, i.e. a
      contiguous stride-aware reduction, instead of the generic
      `np.add.reduceat` along a non-contiguous axis.
    * Query norms (for L2) are precomputed once, flat.
    * We accumulate a (n_queries, C_chunk) distance matrix per DB chunk and do
      a single argpartition-based top-k merge per chunk (rather than once per
      query batch), which also lets the merge avoid materializing a full
      (n_queries, C_chunk) id matrix (the new ids are always a contiguous
      range, so we can encode "new vs. already-in-top-k" implicitly).

Design goals:
  * Memory-safe on large datasets.  Peak RAM is bounded by the per-chunk
    scratch (the dense GEMM output).
  * Fast on many-core CPU.  The inner loop is a single dense GEMM of the form
    (sum-of-query-vectors x d) @ (d x sum-of-chunk-vectors).T, which is what
    OpenBLAS / MKL are good at and scales to hundreds of cores.
  * Same IO formats as the C++ tool, so the binary output is a drop-in
    replacement consumable by utils.ReadGT / stats::ReadGT:
        int32 k
        for each query i in [0, n_queries):
            for each neighbor j in [0, k):
                float32 distance
                uint32  id
  * Same Chamfer-* distance definitions as the C++ one_to_one.h:
        Chamfer-IP(Q, D) = - mean_{q in Q} max_{x in D} <q, x>
        Chamfer-L2(Q, D) =   mean_{q in Q} min_{x in D} ||q - x||^2

Usage (mirrors compute_ground_truth.cpp):
    python data-tools/compute_ground_truth.py \
        -i <database.pcs> -q <queries.pcs> -o <gt.bin> \
        [-k 2000] [-dist_func IP|L2] \
        [--chunk_clouds 20000] [--batch_queries 32] [--threads 64]

Notes on the .pcs format (see mvsic/core/types/point_cloud_set.h):
    uint64 dims
    uint64 n                 # number of point clouds
    uint64 num_vectors       # total vectors across all clouds
    float32[num_vectors * dims] values
    uint64 num_offsets       # == n + 1
    uint64[num_offsets] offsets  # cumulative *float* counts (i.e. vec_idx*dims)
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# .pcs layout helpers
# ---------------------------------------------------------------------------


@dataclass
class PCSLayout:
    path: str
    dims: int
    n: int  # number of point clouds
    num_vectors: int  # total vectors across all clouds
    values_off: int  # byte offset to the float32 values array
    offsets_off: int  # byte offset to the uint64 offsets array
    num_offsets: int  # should be n + 1


def _read_pcs_layout(path: str) -> PCSLayout:
    """Parse just the headers of a .pcs file; no big arrays are loaded."""
    with open(path, "rb") as f:
        header = f.read(24)
        if len(header) < 24:
            raise ValueError(f"{path}: file truncated before pcs header end")
        dims = int(np.frombuffer(header[0:8], dtype=np.uint64)[0])
        n = int(np.frombuffer(header[8:16], dtype=np.uint64)[0])
        num_vectors = int(np.frombuffer(header[16:24], dtype=np.uint64)[0])

    values_off = 24
    values_bytes = num_vectors * dims * 4
    num_offsets_off = values_off + values_bytes

    with open(path, "rb") as f:
        f.seek(num_offsets_off)
        buf = f.read(8)
        if len(buf) < 8:
            raise ValueError(f"{path}: file truncated before num_offsets")
        num_offsets = int(np.frombuffer(buf, dtype=np.uint64)[0])

    offsets_off = num_offsets_off + 8
    # The C++ PointCloudSet reader only uses offsets[0..n], so num_offsets is
    # allowed to be >= n + 1 (some serializers emit a few trailing offsets).
    # We only require there to be enough entries to cover every cloud.
    if num_offsets < n + 1:
        raise ValueError(
            f"{path}: num_offsets ({num_offsets}) < n + 1 ({n + 1})"
        )
    return PCSLayout(
        path=path,
        dims=dims,
        n=n,
        num_vectors=num_vectors,
        values_off=values_off,
        offsets_off=offsets_off,
        num_offsets=num_offsets,
    )


def _mmap_values(layout: PCSLayout) -> np.memmap:
    """Memory-map the float32 values array as (num_vectors, dims) read-only."""
    return np.memmap(
        layout.path,
        dtype=np.float32,
        mode="r",
        offset=layout.values_off,
        shape=(layout.num_vectors, layout.dims),
    )


def _read_offsets_vec(layout: PCSLayout) -> np.ndarray:
    """Read the first n+1 offsets, validate the .pcs format invariants, and
    return them converted to vector-index units.

    Format invariants (per mvsic/core/types/point_cloud_set.h):
      - num_offsets >= n + 1  (C++ loader only uses offsets[0..n]; extra
        trailing entries are ignored)
      - offsets[0] == 0
      - offsets[n] == num_vectors * dims
      - every offset is a multiple of dims
      - offsets are monotonically non-decreasing
    """
    raw = np.memmap(
        layout.path,
        dtype=np.uint64,
        mode="r",
        offset=layout.offsets_off,
        shape=(layout.num_offsets,),
    )
    needed = layout.n + 1
    offs_float = np.asarray(raw[:needed], dtype=np.uint64)

    errors = []

    if int(offs_float[0]) != 0:
        errors.append(f"offsets[0] = {int(offs_float[0])}, expected 0")

    expected_last = layout.num_vectors * layout.dims
    if int(offs_float[-1]) != expected_last:
        errors.append(
            f"offsets[n] = {int(offs_float[-1])}, expected "
            f"num_vectors*dims = {expected_last}"
        )

    bad_mod = offs_float % layout.dims != 0
    if np.any(bad_mod):
        n_bad = int(bad_mod.sum())
        first_bad = int(np.argmax(bad_mod))
        errors.append(
            f"{n_bad} of {needed} offsets are NOT multiples of dims={layout.dims} "
            f"(first bad: offsets[{first_bad}] = {int(offs_float[first_bad])})"
        )

    # Compare as signed to detect uint64 underflow caused by non-monotonic values.
    diffs = np.diff(offs_float.astype(np.int64, copy=False))
    dec = np.where(diffs < 0)[0]
    if dec.size:
        first = int(dec[0])
        errors.append(
            f"offsets not monotonically non-decreasing: {dec.size} decreasing "
            f"transition(s); first at i={first}: "
            f"offsets[{first}]={int(offs_float[first])} -> "
            f"offsets[{first+1}]={int(offs_float[first+1])} "
            f"(delta={int(offs_float[first+1]) - int(offs_float[first])})"
        )

    if errors:
        detail = "\n  - ".join(errors)
        raise ValueError(
            f"{layout.path}: malformed .pcs offsets array; format expected by "
            f"mvsic/core/types/point_cloud_set.h is violated.\n"
            f"  dims={layout.dims}, n={layout.n}, num_vectors={layout.num_vectors}, "
            f"num_offsets={layout.num_offsets}\n"
            f"  - {detail}\n"
            f"  Likely cause: the file was produced by concatenating multiple "
            f".pcs files without rebasing the offsets arrays, or by a writer "
            f"that stored offsets in wrong units.  Regenerate the file via a "
            f"correct writer (e.g. baselines/utils/colbert_embed_lotte.py)."
        )

    return (offs_float // layout.dims).astype(np.int64, copy=True)


def _load_pcs_fully(path: str):
    """Load a small .pcs file (e.g. the query set) fully into RAM as a
    contiguous stack of vectors + per-cloud offsets."""
    layout = _read_pcs_layout(path)
    values_mm = _mmap_values(layout)
    values = np.array(values_mm, dtype=np.float32, copy=True)
    offs_vec = _read_offsets_vec(layout)
    return layout, values, offs_vec


# ---------------------------------------------------------------------------
# Segment-reduce helpers
#
# offsets are contiguous by construction (they come from a prefix-sum), so
# np.maximum.reduceat / np.minimum.reduceat / np.add.reduceat give the right
# answers except for empty segments.  np.*.reduceat on a pair [i, i+1] where
# indices are equal returns arr[i] instead of "empty" -- we fix that up
# afterwards by masking empty-segment output with a neutral element.
# ---------------------------------------------------------------------------


def _segment_reduce(arr: np.ndarray, starts: np.ndarray, ends: np.ndarray,
                    axis: int, op: str, empty_value: float) -> np.ndarray:
    """Segment reduction along `axis`.  `starts`/`ends` define the (contiguous)
    segments (length C).  Handles empty segments (including a trailing empty
    segment where starts[-1] == arr.shape[axis]) by overwriting the output
    with `empty_value`."""
    if op == "max":
        ufunc = np.maximum
    elif op == "min":
        ufunc = np.minimum
    elif op == "sum":
        ufunc = np.add
    else:
        raise ValueError(f"Unknown reduce op {op!r}")
    axis_len = arr.shape[axis]
    # reduceat rejects indices == axis_len.  Any such indices can only be the
    # empty trailing segment(s); clamp them to axis_len - 1 so reduceat
    # succeeds, then overwrite those positions with the neutral value below.
    safe_starts = np.minimum(starts, max(axis_len - 1, 0)) if axis_len > 0 else starts
    out = ufunc.reduceat(arr, safe_starts, axis=axis)
    empty_mask = ends == starts
    if np.any(empty_mask):
        if axis == 0:
            out[empty_mask, ...] = empty_value
        elif axis == 1:
            out[..., empty_mask] = empty_value
        else:
            raise ValueError(f"Unsupported axis={axis}")
    return out


# ---------------------------------------------------------------------------
# Top-k maintenance
# ---------------------------------------------------------------------------


def _merge_topk_chunk(topk_dists: np.ndarray, topk_ids: np.ndarray,
                      chunk_dists: np.ndarray, db_start: int, k: int):
    """Merge a chunk-wide distance matrix (N x C) against the running
    (N x k) top-k tables.  The new DB cloud IDs are always the contiguous
    range [db_start, db_start + C), which lets us skip materializing a full
    (N x C) id matrix: for each surviving index i in [0, k+C), if i < k the
    source id is `topk_ids[row, i]`, else the source id is `db_start + i - k`.

    Updates topk_dists / topk_ids in place.

    Memory:  one (N, k + C) float32 scratch + one (N, k) int64 index array,
    both freed as soon as the merge returns.
    """
    N, C = chunk_dists.shape
    total = k + C
    combined = np.concatenate([topk_dists, chunk_dists], axis=1)  # (N, k+C)

    if total > k:
        # Smallest k per row (unsorted within).
        idx = np.argpartition(combined, k, axis=1)[:, :k]
    else:
        idx = np.broadcast_to(np.arange(total, dtype=np.int64), (N, total))

    new_dists = np.take_along_axis(combined, idx, axis=1)

    is_new = idx >= k
    # Gather the "old topk" ids -- harmless to look up at idx==0 for positions
    # that are actually new; those get overwritten by `where` below.
    old_idx = np.where(is_new, 0, idx)
    from_old = np.take_along_axis(topk_ids, old_idx, axis=1)
    from_new = (idx - k + db_start).astype(np.uint32, copy=False)
    new_ids = np.where(is_new, from_new, from_old)

    topk_dists[:] = new_dists
    topk_ids[:] = new_ids


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_ground_truth(db_file: str, query_file: str, out_file: str,
                         k: int, dist_func: str,
                         chunk_clouds: int, batch_queries: int,
                         verbose: bool = True):
    if dist_func not in ("IP", "L2"):
        raise ValueError(f"dist_func must be 'IP' or 'L2', got {dist_func!r}")

    # ---- Queries (small, fully in RAM) ----
    q_layout, Q_stack, q_offs = _load_pcs_fully(query_file)
    n_queries = q_layout.n
    d = q_layout.dims
    q_nq = (q_offs[1:] - q_offs[:-1]).astype(np.int64)  # vectors per query cloud

    if np.any(q_nq <= 0):
        bad = int(np.argmin(q_nq))
        raise ValueError(
            f"Query {bad} has {int(q_nq[bad])} vectors; drop empty clouds "
            f"(see data-tools/strip_zero_vectors.py) before running this."
        )

    uniform = bool((q_nq == q_nq[0]).all())
    Q_per_cloud = int(q_nq[0]) if uniform else 0
    if uniform:
        # Fast path: every query cloud has the same number of vectors, so we
        # can replace the generic `reduceat` along axis=0 (summing per-cloud
        # mins/maxes across the Q vectors of each query) with a contiguous
        # reshape-and-sum.
        Q_3d = Q_stack.reshape(n_queries, Q_per_cloud, d)  # (N, Q, d), view
        inv_Q = np.float32(1.0 / Q_per_cloud)
    else:
        Q_3d = None
        inv_Q = None

    if verbose:
        if uniform:
            print(f"[queries] {n_queries} clouds x {Q_per_cloud} vectors "
                  f"(uniform), dim={d} -- fast path enabled", flush=True)
        else:
            print(f"[queries] {n_queries} clouds, {q_layout.num_vectors} "
                  f"vectors, dim={d} -- NON-uniform: min={int(q_nq.min())}, "
                  f"max={int(q_nq.max())}; using slower fallback path.",
                  flush=True)

    q_norms_sq = None
    if dist_func == "L2":
        # (num_query_vectors,) flat, precomputed once.
        q_norms_sq = np.einsum("ij,ij->i", Q_stack, Q_stack).astype(np.float32)

    # ---- Database (mmap + chunked) ----
    db_layout = _read_pcs_layout(db_file)
    if db_layout.dims != d:
        raise ValueError(
            f"dim mismatch: queries d={d}, db d={db_layout.dims}")
    n_db = db_layout.n
    D_values = _mmap_values(db_layout)        # (num_vectors, d) mmap
    d_offs = _read_offsets_vec(db_layout)     # (n_db + 1,) vector indices

    if verbose:
        print(f"[database] {n_db} clouds, "
              f"{db_layout.num_vectors} vectors, dim={d} (mmap)",
              flush=True)
        print(f"[config] k={k} dist={dist_func} "
              f"chunk_clouds={chunk_clouds} batch_queries={batch_queries}",
              flush=True)

    # ---- Top-k tables ----
    # +inf initial distance ensures empty slots lose against any real candidate.
    topk_dists = np.full((n_queries, k), np.inf, dtype=np.float32)
    topk_ids = np.zeros((n_queries, k), dtype=np.uint32)

    # ---- Main loop: for each db chunk, batch queries, one merge per chunk ----
    t_gemm = 0.0
    t_reduce = 0.0
    t_merge = 0.0
    t_io = 0.0

    # Scratch for the per-chunk (N_queries x C) distance matrix -- reused
    # across chunks to keep allocator pressure low.  Allocated on first use
    # since C can vary on the final (partial) chunk.
    chunk_dists_scratch: np.ndarray | None = None

    n_db_chunks = (n_db + chunk_clouds - 1) // chunk_clouds
    pbar = tqdm(total=n_db_chunks, desc="DB chunks", disable=not verbose)

    for db_start in range(0, n_db, chunk_clouds):
        db_end = min(db_start + chunk_clouds, n_db)
        chunk_vec_start = int(d_offs[db_start])
        chunk_vec_end = int(d_offs[db_end])
        n_vec_chunk = chunk_vec_end - chunk_vec_start

        if n_vec_chunk == 0:
            # All clouds in this chunk are empty.  Nothing to do.
            pbar.update(1)
            continue

        # Materialize a contiguous float32 chunk of db vectors.  This is where
        # we accept a read-through-the-page-cache cost in exchange for a clean
        # BLAS-friendly layout.
        t0 = time.perf_counter()
        D_chunk = np.ascontiguousarray(
            D_values[chunk_vec_start:chunk_vec_end])  # (M, d)
        t_io += time.perf_counter() - t0

        # Local offsets into D_chunk, in vector units (length C+1).
        local_d_offs = (d_offs[db_start:db_end + 1] - d_offs[db_start]).astype(
            np.int64)
        C = db_end - db_start
        d_starts = local_d_offs[:-1]
        d_ends = local_d_offs[1:]

        if dist_func == "L2":
            d_norms_sq = np.einsum("ij,ij->i", D_chunk, D_chunk).astype(
                np.float32)

        # (Re-)allocate the per-chunk distance buffer if the chunk width
        # changed (only happens on the final partial chunk).
        if chunk_dists_scratch is None or chunk_dists_scratch.shape[1] != C:
            chunk_dists_scratch = np.empty((n_queries, C), dtype=np.float32)
        chunk_dists = chunk_dists_scratch

        # Iterate query batches over this db chunk, writing directly into
        # `chunk_dists[qb_start:qb_end]`.
        for qb_start in range(0, n_queries, batch_queries):
            qb_end = min(qb_start + batch_queries, n_queries)
            B = qb_end - qb_start

            if uniform:
                # Q_batch is a simple reshape of the 3d view.
                Q_batch = Q_3d[qb_start:qb_end].reshape(B * Q_per_cloud, d)
                R = B * Q_per_cloud
            else:
                q_row_start = int(q_offs[qb_start])
                q_row_end = int(q_offs[qb_end])
                Q_batch = Q_stack[q_row_start:q_row_end]
                R = q_row_end - q_row_start
                local_q_offs = (q_offs[qb_start:qb_end + 1]
                                - q_offs[qb_start]).astype(np.int64)
                q_starts = local_q_offs[:-1]
                q_ends = local_q_offs[1:]
                q_nq_batch = q_nq[qb_start:qb_end].astype(np.float32)

            # ---- The big GEMM ----
            # IP: (R x d) @ (d x M) = (R x M)
            t0 = time.perf_counter()
            IP = Q_batch @ D_chunk.T
            t_gemm += time.perf_counter() - t0

            t0 = time.perf_counter()
            if dist_func == "IP":
                # Chamfer-IP = - mean_{q in Q} max_{x in D_cloud} <q, x>
                per_cloud_max = _segment_reduce(
                    IP, d_starts, d_ends, axis=1, op="max",
                    empty_value=-np.inf)  # (R, C)
                if uniform:
                    # Contiguous reduction over the middle axis: faster than
                    # np.add.reduceat(..., axis=0) because the (Q, C) plane
                    # for each query is stride-1 in the C dimension and
                    # stride-C in the Q dimension -- exactly what numpy's
                    # sum prefers.
                    per_query_sum = per_cloud_max.reshape(
                        B, Q_per_cloud, C).sum(axis=1)  # (B, C)
                    np.multiply(per_query_sum, -inv_Q,
                                out=chunk_dists[qb_start:qb_end])
                else:
                    per_query_sum = _segment_reduce(
                        per_cloud_max, q_starts, q_ends, axis=0,
                        op="sum", empty_value=0.0)  # (B, C)
                    np.divide(per_query_sum, q_nq_batch[:, None],
                              out=chunk_dists[qb_start:qb_end])
                    np.negative(chunk_dists[qb_start:qb_end],
                                out=chunk_dists[qb_start:qb_end])
            else:
                # sq_dist = |q|^2 + |x|^2 - 2 <q, x>, clipped at 0.
                if uniform:
                    q_norms_rows = q_norms_sq[
                        qb_start * Q_per_cloud:qb_end * Q_per_cloud]
                else:
                    q_norms_rows = q_norms_sq[q_row_start:q_row_end]
                sq_dist = (q_norms_rows[:, None]
                           + d_norms_sq[None, :]
                           - 2.0 * IP)
                np.maximum(sq_dist, 0.0, out=sq_dist)
                per_cloud_min = _segment_reduce(
                    sq_dist, d_starts, d_ends, axis=1, op="min",
                    empty_value=np.inf)  # (R, C)
                if uniform:
                    per_query_sum = per_cloud_min.reshape(
                        B, Q_per_cloud, C).sum(axis=1)
                    np.multiply(per_query_sum, inv_Q,
                                out=chunk_dists[qb_start:qb_end])
                else:
                    per_query_sum = _segment_reduce(
                        per_cloud_min, q_starts, q_ends, axis=0,
                        op="sum", empty_value=0.0)
                    np.divide(per_query_sum, q_nq_batch[:, None],
                              out=chunk_dists[qb_start:qb_end])
            t_reduce += time.perf_counter() - t0

        # ---- One merge per chunk ----
        t0 = time.perf_counter()
        _merge_topk_chunk(topk_dists, topk_ids, chunk_dists, db_start, k)
        t_merge += time.perf_counter() - t0

        pbar.update(1)
    pbar.close()

    if verbose:
        print(f"[timing] gemm={t_gemm:.2f}s  reduce={t_reduce:.2f}s  "
              f"merge={t_merge:.2f}s  chunk_copy={t_io:.2f}s", flush=True)

    # ---- Final per-row sort of the top-k tables (ascending by distance) ----
    sort_idx = np.argsort(topk_dists, axis=1, kind="stable")
    topk_dists = np.take_along_axis(topk_dists, sort_idx, axis=1)
    topk_ids = np.take_along_axis(topk_ids, sort_idx, axis=1)

    # ---- Write output in the exact C++ format (int32 k, then (f32,u32) pairs) ----
    pair_dtype = np.dtype([("distance", np.float32), ("id", np.uint32)])
    out_arr = np.empty((n_queries, k), dtype=pair_dtype)
    out_arr["distance"] = topk_dists
    out_arr["id"] = topk_ids

    if verbose:
        print(f"[write] {out_file}", flush=True)
    with open(out_file, "wb") as f:
        f.write(np.array([k], dtype=np.int32).tobytes())
        # tofile respects the pair_dtype byte layout (float32, uint32 = 8B).
        out_arr.tofile(f)
    if verbose:
        print(f"[done] ground truth written to {out_file}", flush=True)


# ---------------------------------------------------------------------------
# CLI / thread pinning
# ---------------------------------------------------------------------------


def _configure_threads(n_threads: int):
    """Set BLAS / OpenMP thread counts.  Must run BEFORE importing numpy if you
    want to change from the default; we expose it as a hint for users that
    launch this with a wrapper.  When called after import it is still
    effective for OpenBLAS via the runtime handle (best-effort)."""
    if n_threads <= 0:
        return
    # These env vars are read at BLAS-init time; we set them here in case
    # the user calls compute_ground_truth from a long-lived process.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ.setdefault(var, str(n_threads))
    try:
        import threadpoolctl  # type: ignore
        threadpoolctl.threadpool_limits(n_threads)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Compute exact Chamfer ground truth for multi-vector "
            "PointCloudSet data.  Matches the binary IO format of "
            "data-tools/compute_ground_truth.cpp."
        ))
    ap.add_argument("-i", "--database", required=True,
                    help="Database .pcs file (mmap'd, may be very large).")
    ap.add_argument("-q", "--queries", required=True,
                    help="Query .pcs file (loaded fully into RAM).")
    ap.add_argument("-o", "--output", required=True,
                    help="Output ground-truth file.")
    ap.add_argument("-k", type=int, default=2000,
                    help="Number of nearest neighbors per query (default 2000).")
    ap.add_argument("-dist_func", "--dist_func", choices=["IP", "L2"],
                    default="IP",
                    help="Chamfer distance variant (default IP).")
    ap.add_argument("--chunk_clouds", type=int, default=20_000,
                    help="Number of DB clouds processed per chunk (default 20000). "
                         "Larger => bigger GEMM, more RAM for D_chunk + the "
                         "(N_queries x C) per-chunk distance buffer.")
    ap.add_argument("--batch_queries", type=int, default=64,
                    help="Number of query clouds per inner GEMM batch "
                         "(default 64).  With uniform 32-vectors-per-cloud "
                         "queries this gives R = 32*batch_queries rows into "
                         "the GEMM; pick it so R*M*4B fits in RAM.")
    ap.add_argument("--threads", type=int, default=0,
                    help="BLAS/OpenMP threads (0 = leave default, typically all cores).")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress progress output.")
    args = ap.parse_args()

    _configure_threads(args.threads)

    compute_ground_truth(
        db_file=args.database,
        query_file=args.queries,
        out_file=args.output,
        k=args.k,
        dist_func=args.dist_func,
        chunk_clouds=args.chunk_clouds,
        batch_queries=args.batch_queries,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
