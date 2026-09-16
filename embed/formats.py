"""Readers and writers for the binary formats consumed by MVSIC.

MVSIC serializes ``size_t`` values directly.  The supported build platform is
64-bit little-endian, so this module writes those fields as little-endian
``uint64`` values.  Point-cloud offsets count floats, not vector rows.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


_U64 = struct.Struct("<Q")
_PCS_HEADER = struct.Struct("<QQQ")
_GT_HEADER = struct.Struct("<i")
_PAIR_DTYPE = np.dtype([("distance", "<f4"), ("id", "<u4")])


def _check_platform() -> None:
    if struct.calcsize("P") != 8 or sys.byteorder != "little":
        raise RuntimeError(
            "MVSIC's current on-disk format requires a 64-bit little-endian host"
        )


@dataclass(frozen=True)
class PCSLayout:
    path: Path
    dim: int
    count: int
    num_vectors: int
    values_offset: int
    offsets_offset: int
    num_offsets: int
    trailing_bytes: int

    @property
    def expected_size_without_ids(self) -> int:
        return self.offsets_offset + self.num_offsets * 8


def read_pcs_layout(path: str | os.PathLike[str]) -> PCSLayout:
    """Read and validate the structural metadata of a ``.pcs`` file."""
    _check_platform()
    path = Path(path)
    file_size = path.stat().st_size
    if file_size < _PCS_HEADER.size:
        raise ValueError(f"{path}: truncated before the 24-byte PCS header")

    with path.open("rb") as handle:
        dim, count, num_vectors = _PCS_HEADER.unpack(handle.read(_PCS_HEADER.size))
        if dim == 0:
            raise ValueError(f"{path}: vector dimension is zero")
        values_bytes = num_vectors * dim * 4
        num_offsets_pos = _PCS_HEADER.size + values_bytes
        if num_offsets_pos + 8 > file_size:
            raise ValueError(f"{path}: truncated inside the vector payload")
        handle.seek(num_offsets_pos)
        (num_offsets,) = _U64.unpack(handle.read(8))

    if num_offsets < count + 1:
        raise ValueError(
            f"{path}: has {num_offsets} offsets for {count} clouds; expected at least {count + 1}"
        )
    offsets_offset = num_offsets_pos + 8
    expected_size = offsets_offset + num_offsets * 8
    if expected_size > file_size:
        raise ValueError(f"{path}: truncated inside the offsets array")
    trailing_bytes = file_size - expected_size
    # A few legacy writers append one uint32 ID per cloud.  MVSIC's native
    # file constructor ignores these IDs, but accepting them keeps inspection
    # and migration of existing files possible.
    if trailing_bytes not in (0, count * 4):
        raise ValueError(
            f"{path}: unexpected {trailing_bytes} trailing bytes after offsets"
        )

    layout = PCSLayout(
        path=path,
        dim=int(dim),
        count=int(count),
        num_vectors=int(num_vectors),
        values_offset=_PCS_HEADER.size,
        offsets_offset=offsets_offset,
        num_offsets=int(num_offsets),
        trailing_bytes=int(trailing_bytes),
    )
    offsets = mmap_pcs_offsets(layout)
    required = offsets[: layout.count + 1]
    if int(required[0]) != 0:
        raise ValueError(f"{path}: offsets[0] must be zero")
    if np.any(required[1:] < required[:-1]):
        raise ValueError(f"{path}: offsets are not monotonically non-decreasing")
    if np.any(required % layout.dim != 0):
        raise ValueError(f"{path}: offsets must be multiples of dim={layout.dim}")
    expected_last = layout.num_vectors * layout.dim
    if int(required[-1]) != expected_last:
        raise ValueError(
            f"{path}: final offset is {int(required[-1])}, expected {expected_last}"
        )
    return layout


def mmap_pcs_values(layout: PCSLayout) -> np.memmap:
    return np.memmap(
        layout.path,
        dtype="<f4",
        mode="r",
        offset=layout.values_offset,
        shape=(layout.num_vectors, layout.dim),
    )


def mmap_pcs_offsets(layout: PCSLayout) -> np.memmap:
    return np.memmap(
        layout.path,
        dtype="<u8",
        mode="r",
        offset=layout.offsets_offset,
        shape=(layout.num_offsets,),
    )


def iter_pcs(path: str | os.PathLike[str]) -> Iterator[np.ndarray]:
    layout = read_pcs_layout(path)
    values = mmap_pcs_values(layout)
    offsets = mmap_pcs_offsets(layout)
    for index in range(layout.count):
        start = int(offsets[index]) // layout.dim
        end = int(offsets[index + 1]) // layout.dim
        yield values[start:end]


def validate_pcs(
    path: str | os.PathLike[str],
    *,
    require_nonempty: bool = True,
    require_unit_norm: bool = True,
    norm_atol: float = 1e-4,
    chunk_vectors: int = 1_000_000,
) -> dict[str, int | float]:
    """Validate structural and numerical invariants without loading the file."""
    layout = read_pcs_layout(path)
    offsets = mmap_pcs_offsets(layout)[: layout.count + 1]
    sizes = np.diff(offsets) // layout.dim
    if require_nonempty and np.any(sizes == 0):
        first = int(np.flatnonzero(sizes == 0)[0])
        raise ValueError(f"{layout.path}: point cloud {first} is empty")

    values = mmap_pcs_values(layout)
    min_norm = float("inf")
    max_norm = 0.0
    for start in range(0, layout.num_vectors, chunk_vectors):
        block = np.asarray(values[start : start + chunk_vectors])
        if not np.isfinite(block).all():
            bad = np.argwhere(~np.isfinite(block))[0]
            raise ValueError(
                f"{layout.path}: non-finite value at vector {start + int(bad[0])}, coordinate {int(bad[1])}"
            )
        if block.size:
            norms = np.linalg.norm(block, axis=1)
            min_norm = min(min_norm, float(norms.min()))
            max_norm = max(max_norm, float(norms.max()))
            if require_unit_norm and not np.allclose(norms, 1.0, atol=norm_atol, rtol=0.0):
                local = int(np.flatnonzero(~np.isclose(norms, 1.0, atol=norm_atol, rtol=0.0))[0])
                raise ValueError(
                    f"{layout.path}: vector {start + local} has norm {float(norms[local]):.8g}, expected 1"
                )
    if layout.num_vectors == 0:
        min_norm = 0.0
    return {
        "dim": layout.dim,
        "count": layout.count,
        "num_vectors": layout.num_vectors,
        "min_vectors_per_cloud": int(sizes.min()) if sizes.size else 0,
        "max_vectors_per_cloud": int(sizes.max()) if sizes.size else 0,
        "min_norm": min_norm,
        "max_norm": max_norm,
    }


class StreamingPCSWriter:
    """Atomically stream variable-length point clouds into MVSIC PCS format."""

    def __init__(self, path: str | os.PathLike[str], expected_count: int):
        _check_platform()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.expected_count = int(expected_count)
        if self.expected_count <= 0:
            raise ValueError("cannot write an empty PointCloudSet")
        tmp = tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
            delete=False,
        )
        self._handle = tmp
        self._tmp_path = Path(tmp.name)
        self._handle.write(b"\0" * _PCS_HEADER.size)
        self._dim: int | None = None
        self._count = 0
        self._num_vectors = 0
        self._offsets = [0]
        self._finished = False

    def append(self, cloud: np.ndarray) -> None:
        if self._finished:
            raise RuntimeError("writer has already been finalized")
        array = np.asarray(cloud, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"embedding must be a 2-D array, got shape {array.shape}")
        if array.shape[0] == 0:
            raise ValueError(f"point cloud {self._count} is empty after padding removal")
        if self._dim is None:
            self._dim = int(array.shape[1])
            if self._dim <= 0:
                raise ValueError("embedding dimension must be positive")
        elif array.shape[1] != self._dim:
            raise ValueError(
                f"point cloud {self._count} has dim {array.shape[1]}, expected {self._dim}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"point cloud {self._count} contains non-finite values")
        if self._count >= self.expected_count:
            raise ValueError(f"received more than {self.expected_count} point clouds")
        array = np.ascontiguousarray(array, dtype="<f4")
        self._handle.write(array.tobytes(order="C"))
        self._count += 1
        self._num_vectors += int(array.shape[0])
        self._offsets.append(self._num_vectors * self._dim)

    def finalize(self) -> PCSLayout:
        if self._finished:
            return read_pcs_layout(self.path)
        try:
            if self._count != self.expected_count:
                raise ValueError(
                    f"received {self._count} point clouds, expected {self.expected_count}"
                )
            assert self._dim is not None
            self._handle.write(_U64.pack(len(self._offsets)))
            self._handle.write(np.asarray(self._offsets, dtype="<u8").tobytes())
            self._handle.seek(0)
            self._handle.write(
                _PCS_HEADER.pack(self._dim, self._count, self._num_vectors)
            )
            self._handle.flush()
            self._handle.close()
            os.replace(self._tmp_path, self.path)
            self._finished = True
            return read_pcs_layout(self.path)
        except Exception:
            self.abort()
            raise

    def abort(self) -> None:
        if not self._handle.closed:
            self._handle.close()
        self._tmp_path.unlink(missing_ok=True)

    def __enter__(self) -> "StreamingPCSWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.finalize()
        else:
            self.abort()


class AtomicTextWriter:
    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
            delete=False,
        )
        self._tmp_path = Path(self._handle.name)

    def write_id(self, value: str) -> None:
        value = str(value)
        if "\n" in value or "\r" in value:
            raise ValueError("IDs containing newlines cannot be stored in an ID sidecar")
        self._handle.write(value + "\n")

    def finalize(self) -> None:
        self._handle.flush()
        self._handle.close()
        os.replace(self._tmp_path, self.path)

    def abort(self) -> None:
        if not self._handle.closed:
            self._handle.close()
        self._tmp_path.unlink(missing_ok=True)

    def __enter__(self) -> "AtomicTextWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.finalize()
        else:
            self.abort()


def iter_ids(path: str | os.PathLike[str]) -> Iterator[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            yield line.rstrip("\r\n")


def read_ids(path: str | os.PathLike[str]) -> list[str]:
    return list(iter_ids(path))


def write_gold_gt(
    path: str | os.PathLike[str], rows: Sequence[Sequence[int]]
) -> None:
    """Write MVSIC's CSR gold-ground-truth representation atomically."""
    _check_platform()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.zeros(len(rows) + 1, dtype="<u8")
    for index, row in enumerate(rows):
        offsets[index + 1] = offsets[index] + len(row)
    if int(offsets[-1]) >= 2**64:
        raise ValueError("too many gold-ground-truth entries")
    tmp = tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    tmp_path = Path(tmp.name)
    try:
        tmp.write(_U64.pack(len(offsets)))
        tmp.write(_U64.pack(int(offsets[-1])))
        tmp.write(offsets.tobytes())
        for row in rows:
            if any(value < 0 or value >= 2**32 for value in row):
                raise ValueError("gold-ground-truth document IDs must fit uint32")
            ids = np.asarray(row, dtype="<u4")
            tmp.write(ids.tobytes())
        tmp.flush()
        tmp.close()
        os.replace(tmp_path, path)
    except Exception:
        tmp.close()
        tmp_path.unlink(missing_ok=True)
        raise


def read_gold_gt(path: str | os.PathLike[str], num_queries: int) -> list[np.ndarray]:
    path = Path(path)
    with path.open("rb") as handle:
        (num_offsets,) = _U64.unpack(handle.read(8))
        (num_entries,) = _U64.unpack(handle.read(8))
        offsets = np.fromfile(handle, dtype="<u8", count=num_offsets)
        entries = np.fromfile(handle, dtype="<u4", count=num_entries)
        trailing = handle.read(1)
    if num_offsets != num_queries + 1 or len(offsets) != num_offsets:
        raise ValueError(f"{path}: malformed gold-GT offsets")
    if len(entries) != num_entries or trailing:
        raise ValueError(f"{path}: malformed gold-GT entries")
    if int(offsets[0]) != 0 or int(offsets[-1]) != num_entries:
        raise ValueError(f"{path}: invalid gold-GT offset bounds")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError(f"{path}: gold-GT offsets are not monotonic")
    return [entries[int(offsets[i]) : int(offsets[i + 1])] for i in range(num_queries)]


def write_exact_gt(
    path: str | os.PathLike[str], distances: np.ndarray, ids: np.ndarray
) -> None:
    distances = np.asarray(distances, dtype=np.float32)
    ids = np.asarray(ids, dtype=np.uint32)
    if distances.ndim != 2 or distances.shape != ids.shape:
        raise ValueError("exact-GT distances and IDs must be equally shaped 2-D arrays")
    if not np.isfinite(distances).all():
        raise ValueError("exact-GT distances contain non-finite values")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    tmp_path = Path(tmp.name)
    try:
        tmp.write(_GT_HEADER.pack(distances.shape[1]))
        for start in range(0, distances.shape[0], 256):
            end = min(start + 256, distances.shape[0])
            block = np.empty((end - start, distances.shape[1]), dtype=_PAIR_DTYPE)
            block["distance"] = distances[start:end]
            block["id"] = ids[start:end]
            tmp.write(block.tobytes())
        tmp.flush()
        tmp.close()
        os.replace(tmp_path, path)
    except Exception:
        tmp.close()
        tmp_path.unlink(missing_ok=True)
        raise


def validate_exact_gt(path: str | os.PathLike[str], num_queries: int) -> dict[str, int]:
    path = Path(path)
    size = path.stat().st_size
    if size < 4:
        raise ValueError(f"{path}: truncated before exact-GT header")
    with path.open("rb") as handle:
        (k,) = _GT_HEADER.unpack(handle.read(4))
    if k <= 0:
        raise ValueError(f"{path}: invalid exact-GT k={k}")
    expected = 4 + num_queries * k * _PAIR_DTYPE.itemsize
    if size != expected:
        raise ValueError(f"{path}: has {size} bytes, expected {expected}")
    return {"num_queries": num_queries, "k": k}


def read_exact_gt(
    path: str | os.PathLike[str], num_queries: int
) -> tuple[np.ndarray, np.ndarray]:
    info = validate_exact_gt(path, num_queries)
    with Path(path).open("rb") as handle:
        handle.seek(4)
        pairs = np.fromfile(handle, dtype=_PAIR_DTYPE, count=num_queries * info["k"])
    pairs = pairs.reshape(num_queries, info["k"])
    return pairs["distance"].copy(), pairs["id"].copy()


def write_json(path: str | os.PathLike[str], value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    tmp_path = Path(tmp.name)
    try:
        json.dump(value, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp.flush()
        tmp.close()
        os.replace(tmp_path, path)
    except Exception:
        tmp.close()
        tmp_path.unlink(missing_ok=True)
        raise
