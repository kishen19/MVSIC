"""Read the portable EMDIVF1 index and existing exact-Chamfer GT artifacts."""

from pathlib import Path
import struct

import numpy as np


HEADER = struct.Struct("<8sIIIIQBI")
BUCKET_ENTRY = np.dtype([("id", "<u4"), ("distance", "<f4")])
GT_ENTRY = np.dtype([("distance", "<f4"), ("id", "<u4")])


def read_index(path: Path, m: int = 2048, b: int = 512) -> tuple[dict, np.ndarray]:
    with path.open("rb") as handle:
        raw = handle.read(HEADER.size)
    if len(raw) != HEADER.size:
        raise ValueError(f"truncated EMDIVF header: {path}")
    magic, dim, n, pool, stored_b, seed, random, stored_m = HEADER.unpack(raw)
    if magic != b"EMDIVF1\0" or not dim or not n or not pool:
        raise ValueError(f"invalid EMDIVF header: {path}")
    if not 0 < m <= stored_m <= pool or not 0 < b <= stored_b <= n or random not in (0, 1):
        raise ValueError(f"invalid requested/stored EMDIVF dimensions: {path}")
    record = np.dtype([("center", "<u4"), ("selection_seconds", "<f8"),
                       ("bucket_seconds", "<f8"), ("radius", "<f8"),
                       ("neighbors", BUCKET_ENTRY, (stored_b,))])
    if path.stat().st_size != HEADER.size + stored_m * record.itemsize:
        raise ValueError(f"truncated or extra EMDIVF data: {path}")
    data = np.memmap(path, dtype=record, mode="r", offset=HEADER.size, shape=(stored_m,))
    entries = data["neighbors"]
    if np.any(data["center"] >= pool) or len(np.unique(data["center"])) != stored_m:
        raise ValueError(f"invalid or repeated prototype rows: {path}")
    if np.any(entries["id"] >= n) or not np.isfinite(entries["distance"]).all():
        raise ValueError(f"invalid saved document IDs/scores: {path}")
    if np.any(entries["distance"][:, 1:] < entries["distance"][:, :-1]):
        raise ValueError(f"unsorted buckets: {path}")
    buckets = entries["id"][:m, :b].copy()
    sorted_ids = np.sort(buckets, axis=1)
    if np.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]):
        raise ValueError(f"duplicate document within a bucket: {path}")
    last = data[m - 1]
    metadata = dict(corpus_documents=int(n), dimension=int(dim), m=m, b=b,
                    prototype_pool_size=int(pool), stored_m=int(stored_m),
                    stored_b=int(stored_b), seed=int(seed),
                    selection_seconds=float(last["selection_seconds"]),
                    bucket_seconds=float(last["bucket_seconds"]))
    return metadata, buckets


def read_gt(path: Path, k: int = 10) -> np.ndarray:
    with path.open("rb") as handle:
        raw = handle.read(4)
        if len(raw) != 4:
            raise ValueError(f"truncated GT: {path}")
        stored_k, = struct.unpack("<I", raw)
        payload = handle.read()
    if not 0 < k <= stored_k or not payload or len(payload) % (stored_k * GT_ENTRY.itemsize):
        raise ValueError(f"invalid exact Chamfer GT dimensions: {path}")
    return np.frombuffer(payload, dtype=GT_ENTRY).reshape(-1, stored_k)["id"][:, :k].copy()
