#!/usr/bin/env python3
"""Convert MVSIC .pcs/.gt artifacts into GEM's gem_data tree.

GEM (sigmod26gem) wants its inputs as a specific .npy layout (centroids,
coarse centroids, per-token codes, FP16 doc shards, queries). This script
performs the heavy preprocessing -- two-stage k-means, per-token centroid
assignment, TF-IDF doc-to-coarse-cluster routing -- and writes the result
out so the gem_runner C++ binary can mmap it and build the HNSW graph on
top.

Standalone usage:

    python3 benchmarks/gem/gem_preprocess.py \
        --dataset arguana \
        --ds-path data/beir/arguana \
        --out-dir indices/arguana/gem/gem_k1024_k64/gem_data \
        --k1 1024 --k2 64 --top-r 4

It is also auto-invoked by ``GEMWrapper.build()`` in
``benchmarks/framework_utils.py``, so end users normally never run it
directly.

Cache marker: a ``.READY`` file is written into ``--out-dir`` containing
the canonical (k1, k2, top_r, n_doc, n_token, dim) tuple. Subsequent
invocations short-circuit when the marker matches the requested params,
unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# .pcs reader (kept independent of benchmarks/utils.py so this script can run
# without torch installed). Layout matches benchmarks/utils.py::load_point_clouds:
#
#   uint64 dim
#   uint64 n_doc
#   uint64 n_vec
#   float32[n_vec * dim] flat vectors
#   uint64 n_offsets   (== n_doc + 1)
#   uint64[n_offsets] offsets (in *floats*, not vectors)


def load_pcs_flat(path: Path):
    with open(path, "rb") as f:
        dim = struct.unpack("Q", f.read(8))[0]
        n_doc = struct.unpack("Q", f.read(8))[0]
        n_vec = struct.unpack("Q", f.read(8))[0]
        data = np.fromfile(f, dtype=np.float32, count=n_vec * dim).reshape(n_vec, dim)
        n_off = struct.unpack("Q", f.read(8))[0]
        offsets = np.fromfile(f, dtype=np.uint64, count=n_off)
    if n_off != n_doc + 1:
        raise ValueError(
            f"{path}: malformed pcs (n_doc={n_doc} but n_offsets={n_off})"
        )
    # Offsets are float-offsets, not vector-offsets.
    doclens = ((offsets[1 : n_doc + 1] - offsets[:n_doc]) // dim).astype(np.int64)
    return data, doclens, int(dim)


# ---------------------------------------------------------------------------
# k-means helpers (faiss-backed, with a sample-then-fit + batched assign
# pattern lifted from IGPWrapper to keep memory bounded on big datasets).


def fit_kmeans(
    vecs: np.ndarray,
    k: int,
    niter: int,
    seed: int,
    sample_max: int,
    verbose: bool,
) -> np.ndarray:
    try:
        import faiss
    except ImportError as e:
        raise RuntimeError(
            "benchmarks/gem/gem_preprocess.py requires faiss (faiss-cpu Python pkg)."
        ) from e

    if vecs.shape[0] > sample_max:
        rng = np.random.default_rng(seed)
        idx = rng.choice(vecs.shape[0], size=sample_max, replace=False)
        sample = vecs[idx]
    else:
        sample = vecs
    sample = np.ascontiguousarray(sample.astype(np.float32, copy=False))

    # min_points_per_centroid=1 lets us train on very small samples (small
    # datasets where N/k can drop below faiss's default 39 threshold).
    km = faiss.Kmeans(
        int(sample.shape[1]),
        int(k),
        niter=int(niter),
        seed=int(seed),
        verbose=bool(verbose),
        min_points_per_centroid=1,
    )
    km.train(sample)
    return np.asarray(km.centroids, dtype=np.float32).reshape(int(k), int(sample.shape[1]))


def assign_codes(vecs: np.ndarray, centroids: np.ndarray, batch: int = 200_000) -> np.ndarray:
    """argmax cosine/IP assignment in batches (memory-bounded)."""
    n = vecs.shape[0]
    codes = np.empty(n, dtype=np.int32)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        sims = vecs[s:e].astype(np.float32, copy=False) @ centroids.T
        codes[s:e] = np.argmax(sims, axis=1).astype(np.int32)
    return codes


# ---------------------------------------------------------------------------
# TF-IDF doc-to-coarse-cluster assignment. Each doc's token codes get mapped
# to coarse clusters via fine_to_coarse, then aggregated with an IDF weight
# computed over fine-cluster document frequencies. Each doc is assigned to
# its top-r coarse clusters by aggregate weight. This is the paper's
# Section 4.1.2 routing rule.


def tfidf_cluster_assign(
    token_codes_per_doc: list[np.ndarray],
    k1: int,
    k2: int,
    fine_to_coarse: np.ndarray,
    top_r: int,
) -> list[list[int]]:
    n_doc = len(token_codes_per_doc)

    # Document frequency over fine centroids.
    df = np.zeros(k1, dtype=np.float64)
    for codes in token_codes_per_doc:
        if codes.size == 0:
            continue
        df[np.unique(codes)] += 1.0
    idf = np.log((n_doc + 1.0) / (df + 1.0)) + 1.0  # smoothed IDF

    top_r = max(1, min(int(top_r), int(k2)))

    coarse_to_docs: list[list[int]] = [[] for _ in range(k2)]
    for d, codes in enumerate(token_codes_per_doc):
        if codes.size == 0:
            continue
        # tf over fine centroids
        tf = np.bincount(codes, minlength=k1).astype(np.float64)
        tf_idf = tf * idf
        # Aggregate fine -> coarse via fine_to_coarse map.
        coarse_scores = np.zeros(k2, dtype=np.float64)
        np.add.at(coarse_scores, fine_to_coarse, tf_idf)
        if top_r >= k2:
            top = np.arange(k2)
        else:
            top = np.argpartition(-coarse_scores, kth=top_r - 1)[:top_r]
        for c in top:
            coarse_to_docs[int(c)].append(d)
    return coarse_to_docs


# ---------------------------------------------------------------------------
# k1 / k2 defaults. The paper picks k1 ~ 16*sqrt(N_doc) for MS MARCO scale.
# For our smaller BEIR-style datasets we floor k1 at 1024 (otherwise the
# HNSW graph degenerates) and k2 at 64.


def default_k1(n_token: int, n_doc: int) -> int:
    target = max(16.0 * math.sqrt(max(n_doc, 1)), 1024.0)
    k = int(round(2 ** round(math.log2(target))))
    # Cap to something safe for the dataset size.
    return max(64, min(k, n_token))


def default_k2(n_doc: int, k1: int) -> int:
    target = max(n_doc / 5000.0, 64.0)
    k = int(round(2 ** round(math.log2(target))))
    return max(8, min(k, k1))


# ---------------------------------------------------------------------------
# Main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="dataset name (used only in logs/meta).")
    ap.add_argument("--ds-path", required=True, help="folder with <ds>_points.pcs / _queries.pcs")
    ap.add_argument(
        "--out-dir",
        required=True,
        help="gem_data root; writes cdata/, docdata/, qdata/ under here.",
    )
    ap.add_argument("--k1", type=int, default=0, help="fine clusters (0 -> auto, ~16*sqrt(N_doc))")
    ap.add_argument("--k2", type=int, default=0, help="coarse clusters (0 -> auto, ~N_doc/5000)")
    ap.add_argument("--top-r", type=int, default=4, help="coarse clusters per doc (TF-IDF top-r)")
    ap.add_argument("--num-shards", type=int, default=1, help="how many encoding<i>_float16 shards to write")
    ap.add_argument("--niter", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--kmeans-sample-max",
        type=int,
        default=2_000_000,
        help="cap on rows used to train fine k-means (sampling without replacement)",
    )
    ap.add_argument("--force", action="store_true", help="rebuild even if .READY matches")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    cd = out_dir / "cdata"
    dd = out_dir / "docdata"
    qd = out_dir / "qdata"
    for d in (cd, dd, qd):
        d.mkdir(parents=True, exist_ok=True)
    ready = out_dir / ".READY"

    pcs_path = Path(args.ds_path) / f"{args.dataset}_points.pcs"
    queries_path = Path(args.ds_path) / f"{args.dataset}_queries.pcs"
    if not pcs_path.exists():
        print(f"[gem_preprocess] missing {pcs_path}", file=sys.stderr)
        return 2
    if not queries_path.exists():
        print(f"[gem_preprocess] missing {queries_path}", file=sys.stderr)
        return 2

    print(f"[gem_preprocess] loading {pcs_path}", flush=True)
    t_load = time.time()
    data, doclens, dim = load_pcs_flat(pcs_path)
    n_doc = int(doclens.shape[0])
    n_tok = int(data.shape[0])
    print(
        f"[gem_preprocess] n_doc={n_doc} n_tok={n_tok} dim={dim} "
        f"(load {time.time() - t_load:.1f}s)",
        flush=True,
    )

    k1 = int(args.k1) if args.k1 > 0 else default_k1(n_tok, n_doc)
    k2 = int(args.k2) if args.k2 > 0 else default_k2(n_doc, k1)
    if k1 > n_tok:
        k1 = n_tok
    if k2 > k1:
        k2 = k1
    top_r = max(1, min(int(args.top_r), int(k2)))
    num_shards = max(int(args.num_shards), 1)
    print(
        f"[gem_preprocess] k1={k1} k2={k2} top_r={top_r} num_shards={num_shards}",
        flush=True,
    )

    cache_key = json.dumps(
        {
            "k1": k1,
            "k2": k2,
            "top_r": top_r,
            "num_shards": num_shards,
            "n_doc": n_doc,
            "n_token": n_tok,
            "dim": dim,
        },
        sort_keys=True,
    )
    if ready.exists() and not args.force:
        try:
            existing = ready.read_text().strip()
        except OSError:
            existing = ""
        if existing == cache_key:
            print(
                f"[gem_preprocess] {out_dir} already up to date "
                "(.READY matches; pass --force to rebuild)",
                flush=True,
            )
            return 0

    # ---- Fine centroids -----------------------------------------------------
    t0 = time.time()
    fine = fit_kmeans(
        data,
        k1,
        niter=args.niter,
        seed=args.seed,
        sample_max=int(args.kmeans_sample_max),
        verbose=args.verbose,
    )
    np.save(cd / "centroids.npy", fine.astype(np.float16))
    print(f"[gem_preprocess] fine centroids done in {time.time() - t0:.1f}s", flush=True)

    # ---- Per-token codes ----------------------------------------------------
    t0 = time.time()
    token_codes = assign_codes(data, fine)
    print(
        f"[gem_preprocess] per-token codes done in {time.time() - t0:.1f}s",
        flush=True,
    )

    # ---- Coarse centroids ---------------------------------------------------
    t0 = time.time()
    coarse = fit_kmeans(
        fine.astype(np.float32),
        k2,
        niter=args.niter,
        seed=args.seed,
        sample_max=int(args.kmeans_sample_max),
        verbose=args.verbose,
    )
    np.save(cd / "coarse_centroids.npy", coarse.astype(np.float32))
    fine_to_coarse = assign_codes(fine.astype(np.float32), coarse).astype(np.int32)
    print(
        f"[gem_preprocess] coarse centroids done in {time.time() - t0:.1f}s",
        flush=True,
    )

    # ---- TF-IDF doc routing -------------------------------------------------
    t0 = time.time()
    per_doc = []
    cursor = 0
    for dl in doclens.astype(int):
        per_doc.append(token_codes[cursor : cursor + dl])
        cursor += dl
    coarse_to_docs = tfidf_cluster_assign(per_doc, k1, k2, fine_to_coarse, top_r)
    with open(cd / "coarse_cluster_info.txt", "w") as f:
        for line in coarse_to_docs:
            f.write(" ".join(str(int(x)) for x in line))
            f.write("\n")
    nonempty = sum(1 for x in coarse_to_docs if x)
    avg = sum(len(x) for x in coarse_to_docs) / max(nonempty, 1)
    print(
        f"[gem_preprocess] TF-IDF routing done in {time.time() - t0:.1f}s "
        f"({nonempty}/{k2} non-empty clusters, avg docs/cluster={avg:.1f})",
        flush=True,
    )

    # ---- Shard the doc data -------------------------------------------------
    t0 = time.time()
    doc_chunks = np.array_split(np.arange(n_doc), num_shards)
    cum_starts = np.concatenate([[0], np.cumsum(doclens.astype(np.int64))]).astype(np.int64)
    for shard_i, doc_idx in enumerate(doc_chunks):
        if len(doc_idx) == 0:
            np.save(dd / f"encoding{shard_i}_float16.npy", np.zeros((0, dim), dtype=np.float16))
            np.save(dd / f"doc_codes_{shard_i}.npy", np.zeros((0,), dtype=np.int32))
            np.save(dd / f"doclens{shard_i}.npy", np.zeros((0,), dtype=np.int64))
            continue
        ranges = [(int(cum_starts[d]), int(cum_starts[d + 1])) for d in doc_idx]
        total = sum(e - s for s, e in ranges)
        emb = np.empty((total, dim), dtype=np.float16)
        cod = np.empty((total,), dtype=np.int32)
        off = 0
        for s, e in ranges:
            n = e - s
            emb[off : off + n] = data[s:e].astype(np.float16, copy=False)
            cod[off : off + n] = token_codes[s:e]
            off += n
        np.save(dd / f"encoding{shard_i}_float16.npy", emb)
        np.save(dd / f"doc_codes_{shard_i}.npy", cod)
        np.save(dd / f"doclens{shard_i}.npy", doclens[doc_idx].astype(np.int64))
    print(
        f"[gem_preprocess] {num_shards}-shard write done in {time.time() - t0:.1f}s",
        flush=True,
    )

    # ---- Queries ------------------------------------------------------------
    t0 = time.time()
    qdata, qlens, qdim = load_pcs_flat(queries_path)
    if qdim != dim:
        print(
            f"[gem_preprocess] WARN: query dim {qdim} != doc dim {dim}; "
            "the runner will reject this.",
            flush=True,
        )
    n_q = int(qlens.shape[0])
    q_max = int(qlens.max()) if n_q > 0 else 0
    # 3D layout [n_q, q_max, dim] (zero-padded) matches upstream's
    # load_from_evqa shape. qlens.npy lets the runner trim per-query.
    qembs = np.zeros((n_q, q_max, dim), dtype=np.float32)
    cursor = 0
    for j, ql in enumerate(qlens.astype(int)):
        qembs[j, :ql] = qdata[cursor : cursor + ql]
        cursor += ql
    np.save(qd / "qembs.npy", qembs)
    np.save(qd / "qlens.npy", qlens.astype(np.int64))
    print(
        f"[gem_preprocess] queries done in {time.time() - t0:.1f}s "
        f"(n_q={n_q}, max_len={q_max})",
        flush=True,
    )

    # ---- Meta + READY marker ------------------------------------------------
    meta = {
        "dataset": args.dataset,
        "n_doc": n_doc,
        "n_token": n_tok,
        "dim": dim,
        "k1": int(k1),
        "k2": int(k2),
        "top_r": int(top_r),
        "num_shards": int(num_shards),
        "n_query": n_q,
        "q_max_len": q_max,
        "source_pcs": str(pcs_path),
        "source_queries_pcs": str(queries_path),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    ready.write_text(cache_key)
    print(
        f"[gem_preprocess] wrote {out_dir} (k1={k1}, k2={k2}, shards={num_shards})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
