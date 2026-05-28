#!/usr/bin/env python3
"""Convert MVSIC .pcs artifacts into flat arrays for hnswlib_runner.

Used by HnswlibWrapper in benchmarks/framework_utils.py and //benchmarks/hnswlib:hnswlib_runner.

Writes under <out_dir>/:
  base_vecs.npy      float32 [n_tokens, dim]
  base_docids.npy    uint32  [n_tokens]
  doc_offsets.npy    uint64  [n_doc + 1]  (token index boundaries)
  q_vecs.npy         float32 [total_q_tokens, dim]
  q_lens.npy         int64   [n_query]
  .READY             JSON marker (inputs + shapes)
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


def load_pcs_flat(path: Path):
    with open(path, "rb") as f:
        dim = struct.unpack("Q", f.read(8))[0]
        n_doc = struct.unpack("Q", f.read(8))[0]
        n_vec = struct.unpack("Q", f.read(8))[0]
        data = np.fromfile(f, dtype=np.float32, count=n_vec * dim).reshape(n_vec, dim)
        n_off = struct.unpack("Q", f.read(8))[0]
        offsets = np.fromfile(f, dtype=np.uint64, count=n_off)
    if n_off != n_doc + 1:
        raise ValueError(f"{path}: malformed pcs (n_doc={n_doc} but n_offsets={n_off})")
    doclens = ((offsets[1 : n_doc + 1] - offsets[:n_doc]) // dim).astype(np.int64)
    token_offsets = (offsets[:n_doc] // dim).astype(np.uint64)
    return data, doclens, token_offsets, int(dim), int(n_doc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ds-path", required=True, help="folder with <ds>_points.pcs / _queries.pcs")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--metric", default="ip", choices=("ip", "l2"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ds_path = Path(args.ds_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ready = out_dir / ".READY"

    points_path = ds_path / f"{args.dataset}_points.pcs"
    queries_path = ds_path / f"{args.dataset}_queries.pcs"
    if not points_path.is_file():
        raise SystemExit(f"missing {points_path}")
    if not queries_path.is_file():
        raise SystemExit(f"missing {queries_path}")

    marker = {
        "dataset": args.dataset,
        "points_path": str(points_path.resolve()),
        "queries_path": str(queries_path.resolve()),
        "points_mtime": points_path.stat().st_mtime,
        "queries_mtime": queries_path.stat().st_mtime,
        "metric": args.metric,
    }
    if ready.is_file() and not args.force:
        try:
            prev = json.loads(ready.read_text())
            if prev == marker:
                print(f"[hnswlib_preprocess] up-to-date: {out_dir}", flush=True)
                return 0
        except Exception:
            pass

    print(f"[hnswlib_preprocess] reading {points_path}", flush=True)
    base_vecs, doclens, doc_token_off, dim, n_doc = load_pcs_flat(points_path)
    doc_offsets = np.zeros(n_doc + 1, dtype=np.uint64)
    doc_offsets[1:] = np.cumsum(doclens.astype(np.uint64))

    print(f"[hnswlib_preprocess] reading {queries_path}", flush=True)
    q_vecs, q_lens_arr, _, q_dim, n_query = load_pcs_flat(queries_path)
    if q_dim != dim:
        raise SystemExit(f"dim mismatch: points={dim} queries={q_dim}")

    base_docids = np.repeat(np.arange(n_doc, dtype=np.uint32), doclens.astype(np.int64))

    np.save(out_dir / "base_vecs.npy", np.ascontiguousarray(base_vecs, dtype=np.float32))
    np.save(out_dir / "base_docids.npy", base_docids)
    np.save(out_dir / "doc_offsets.npy", doc_offsets)
    np.save(out_dir / "q_vecs.npy", np.ascontiguousarray(q_vecs, dtype=np.float32))
    np.save(out_dir / "q_lens.npy", q_lens_arr.astype(np.int64))

    marker.update(
        {
            "dim": dim,
            "n_doc": n_doc,
            "n_tokens": int(base_vecs.shape[0]),
            "n_query": n_query,
        }
    )
    ready.write_text(json.dumps(marker, indent=2) + "\n")
    print(
        f"[hnswlib_preprocess] wrote {out_dir} "
        f"(n_doc={n_doc}, n_tokens={base_vecs.shape[0]}, n_query={n_query}, dim={dim})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
