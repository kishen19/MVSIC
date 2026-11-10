#!/usr/bin/env python3
"""Compare FiQA document sizes in EMDIVF postings with the whole corpus."""

import argparse
import json
import struct
from pathlib import Path


def read_sizes(path: Path) -> list[int]:
    with path.open("rb") as handle:
        dims, n_clouds, n_vectors = struct.unpack("<3Q", handle.read(24))
        handle.seek(24 + n_vectors * dims * 4)
        (n_offsets,) = struct.unpack("<Q", handle.read(8))
        if n_offsets != n_clouds + 1:
            raise ValueError("PCS offset count does not match cloud count")
        offsets = struct.unpack(f"<{n_offsets}Q", handle.read(n_offsets * 8))
    return [(offsets[i + 1] - offsets[i]) // dims for i in range(n_clouds)]


def summarize(values: list[int]) -> dict[str, float]:
    values = sorted(values)
    n = len(values)
    return {
        "count": n,
        "mean_vectors": sum(values) / n,
        "median_vectors": values[n // 2],
        "p90_vectors": values[int(0.9 * (n - 1))],
        "p99_vectors": values[int(0.99 * (n - 1))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--b", type=int, default=512)
    args = parser.parse_args()

    sizes = read_sizes(args.docs)
    payload = args.index.read_bytes()
    magic, dims, corpus, train, max_b, seed, random_centers, n_centers = struct.unpack_from(
        "<8s4IQBI", payload
    )
    if magic != b"EMDIVF1\0" or corpus != len(sizes):
        raise ValueError("Index/corpus mismatch")
    if args.m > n_centers or args.b > max_b:
        raise ValueError("Requested prefix exceeds saved index")

    offset = struct.calcsize("<8s4IQBI")
    members: list[int] = []
    for center in range(n_centers):
        offset += struct.calcsize("<I3d")
        if center < args.m:
            for rank in range(args.b):
                (doc,) = struct.unpack_from("<I", payload, offset + rank * 8)
                members.append(doc)
        offset += max_b * 8
    if offset != len(payload):
        raise ValueError("Index trailing bytes")

    result = {
        "corpus": summarize(sizes),
        "bucket_memberships": summarize([sizes[doc] for doc in members]),
        "unique_posting_docs": summarize([sizes[doc] for doc in set(members)]),
        "unique_posting_doc_fraction": len(set(members)) / len(sizes),
        "index_m": args.m,
        "index_b": args.b,
        "dimension": dims,
        "seed": seed,
        "random_centers": bool(random_centers),
        "train_size": train,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
