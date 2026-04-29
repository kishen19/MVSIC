"""
Convert old MVSIC index files to the new skeleton-only format.

Background
----------
As part of the quantization decoupling refactor, every index family now saves a
"skeleton" containing only the structural pieces that are expensive to rebuild
(graph topology, tree structure, raw centers, leaf point IDs, metadata):

    - quantizer codebooks are NOT saved.
    - encoded / quantized leaf data is NOT saved.
    - both are re-trained and re-encoded from the base points on `load()`.

Supported upgrades:

    v1 (pre-refactor) -> v2 (skeleton-only) for Vamana/MUVERA/MPool/SVH*.
    v2               -> v3 (MVIVF family)  drops the persisted
                        `params.quantize_centers` byte which is now a
                        compile-time `CompressCenters` template parameter.

The v2/v3 files start with a common header:

    magic    : uint32   family-specific sentinel
    version  : uint32
    class_id : uint32   family-specific class identifier, checked on load.

The table below summarises the v2/v3 bodies; per-family sections of this script
document the corresponding source layouts and the byte-level conversion rules.

    Family      | magic      | class_id derivation
    ------------|------------|-------------------------------------------------
    Vamana      | 0x56414D41 | LeafModel::kClassId
    MUVERA      | 0x4D555645 | static_cast<uint32>(kLeafMethod)
    MPool       | 0x4D504F4F | static_cast<uint32>(kLeafMethod)
    SVHGraph    | 0x53564847 | static_cast<uint32>(kLeafMethod)
    SVHIVF      | 0x53564849 | (kLeafMethod << 1) | kHasCenterQuant
    MVIVF       | 0x4D564946 | (kLeafModel::kClassId << 1) | kHasCenterQuant
    MVIVFFlat   | 0x4D464C46 | (kLeafModel::kClassId << 1) | kHasCenterQuant
    MVIVFSpill  | 0x4D53504C | (kLeafModel::kClassId << 1) | kHasCenterQuant

Recommended workflow
--------------------
The cleanest way to migrate an existing on-disk index is to rebuild it with the
new C++ benches: load the dataset, call `build()`, then `save()`.  The rebuild
typically dominates runtime only for very large graphs, and the resulting
v2 files are immediately compatible with the new search path.

If you still want to re-serialize an index *without* rebuilding, implement the
per-family converters below.  Each converter is a pure byte-level transform:
it skips the legacy codebook/encoded sections and rewrites the skeleton behind
the new v2 header.  The stubs indicate exactly which sections to keep and
which to drop; consult the `save()` / `load()` git history for the precise v1
layout of each family.

Usage
-----
    # v1 -> v2 for graph / SVH families:
    python data-tools/convert_index.py \\
        --family vamana \\
        --leaf-model None \\
        --input  /path/to/old/index.v1 \\
        --output /path/to/new/index.v2

    # v2 -> v3 for MVIVF family (drops persisted quantize_centers byte):
    python data-tools/convert_index.py \\
        --family mvivf_spill \\
        --leaf-model PQ \\
        --compress-centers \\
        --input  /path/to/old/index.v2 \\
        --output /path/to/new/index.v3

The `--leaf-model` flag determines `class_id` in the new header and must match
the concrete C++ template instantiation used at load time (e.g.
`IndexVamanaIP`, `IndexMVIVFCompressPQIP`, ...).
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import Callable, Dict


MAGIC = {
    "vamana": 0x56414D41,
    "muvera": 0x4D555645,
    "mpool": 0x4D504F4F,
    "svhgraph": 0x53564847,
    "svhivf": 0x53564849,
    "mvivf": 0x4D564946,
    "mvivf_flat": 0x4D464C46,
    "mvivf_spill": 0x4D53504C,
}

# Default output version per family (bump here when an on-disk format changes).
VERSION_BY_FAMILY = {
    "vamana": 2,
    "muvera": 2,
    "mpool": 2,
    "svhgraph": 2,
    "svhivf": 2,
    "mvivf": 3,
    "mvivf_flat": 3,
    "mvivf_spill": 3,
}


LEAF_METHOD_IDS = {
    "None": 0,
    "PQ": 1,
    "FastScan": 2,
    "RaBitQ": 3,
    "TurboQuant": 4,
    "SPQTQ": 5,
    "OneBitTQ": 6,
}


def _write_header(out, family: str, class_id: int) -> None:
    out.write(struct.pack("<III", MAGIC[family], VERSION_BY_FAMILY[family], class_id))


def convert_vamana(src: Path, dst: Path, class_id: int) -> None:
    """
    v1 Vamana layout (approximate, legacy):
        [start_point:uint32][graph...][optional quantizer codebook + encoded]
    v2 Vamana layout:
        [magic][version][class_id][start_point:uint32][graph...]

    The quantizer codebook and encoded set are discarded.  The graph section
    uses `parlayANN::vamana::Graph::save/load`; we copy its exact byte range
    from `src` into `dst`.  The exact end-offset of the graph depends on the
    graph size header embedded in the stream; implement the byte walk here.
    """
    raise NotImplementedError(
        "Vamana v1->v2 byte walk is not yet implemented.  The simplest option "
        "today is to rebuild via the C++ bench: "
        "`bazel run //mvsic/vamana:bench_build -- ...`."
    )


def convert_muvera(src: Path, dst: Path, class_id: int) -> None:
    """
    v1 MUVERA layout (legacy):
        [d_fde:uint32][graph...][points_fdes:PointRange...][quantizer...]
    v2 MUVERA layout:
        [magic][version][class_id][d_fde:uint32][graph...][points_fdes...]

    Strip the trailing quantizer codebook, but keep `d_fde`, the graph, and the
    raw `points_fdes` PointRange verbatim.
    """
    raise NotImplementedError(
        "MUVERA v1->v2 converter is a stub; rebuild via `bench_build` or port "
        "the graph/PointRange byte layout here."
    )


def convert_mpool(src: Path, dst: Path, class_id: int) -> None:
    """
    v1 MPool layout (legacy):
        [graph...][points_mp:PointRange...][quantizer...]
    v2 MPool layout:
        [magic][version][class_id][graph...][points_mp:PointRange...]
    """
    raise NotImplementedError(
        "MPool v1->v2 converter is a stub; rebuild via `bench_build` or port "
        "the graph/PointRange byte layout here."
    )


def convert_svhgraph(src: Path, dst: Path, class_id: int) -> None:
    """
    v1 SVHGraph layout (legacy):
        [d:uint32][map_sz:uint64][vector_to_id[...]][graph...][flattened_points:PointRange...][quantizer...]
    v2 SVHGraph layout:
        [magic][version][class_id][d:uint32][map_sz:uint64][vector_to_id[...]]
        [graph...][flattened_points:PointRange...]
    """
    raise NotImplementedError(
        "SVHGraph v1->v2 converter is a stub; rebuild via `bench_build` or "
        "port the structure verbatim."
    )


def convert_svhivf(src: Path, dst: Path, class_id: int) -> None:
    """
    v1 SVHIVF layout (legacy):
        [d:uint32][num_nodes:uint64]
        [center_offsets:uint64 x num_nodes][center_data:float x ...]
        [children_offsets:uint64 x num_nodes][children_indices:uint64 x ...]
        [point_id_offsets:uint64 x num_nodes][point_ids:{uint64,uint64} x ...]
        [int type_id][leaf quantizer codebook]
    v2 SVHIVF layout:
        [magic][version][class_id]
        [d:uint32][num_nodes:uint64]
        [center_offsets][center_data]
        [children_offsets][children_indices]
        [point_id_offsets][point_ids]
        (no codebook)
    """
    with src.open("rb") as fin, dst.open("wb") as fout:
        _write_header(fout, "svhivf", class_id)

        d = fin.read(4)
        fout.write(d)
        num_nodes_bytes = fin.read(8)
        fout.write(num_nodes_bytes)
        num_nodes = struct.unpack("<Q", num_nodes_bytes)[0]

        def copy_offsets_and_payload(elem_size: int) -> None:
            offsets_bytes = fin.read(8 * num_nodes)
            fout.write(offsets_bytes)
            total = sum(struct.unpack(f"<{num_nodes}Q", offsets_bytes))
            payload = fin.read(total * elem_size)
            fout.write(payload)

        d_val = struct.unpack("<I", d)[0]
        copy_offsets_and_payload(4 * d_val)
        copy_offsets_and_payload(8)
        copy_offsets_and_payload(16)


def _convert_mvivf_params_blob_v2_to_v3(fin, fout, params_u32_count: int) -> None:
    """
    MVIVF-family v2 params blob = `params_u32_count` little-endian u32 fields
    followed by a trailing `bool quantize_centers` byte. v3 drops that byte.
    Copy the u32 fields verbatim and skip the trailing byte.
    """
    payload = fin.read(4 * params_u32_count)
    fout.write(payload)
    tail = fin.read(1)
    if len(tail) != 1:
        raise ValueError("truncated v2 params blob: missing trailing quantize_centers byte")


def convert_mvivf(src: Path, dst: Path, class_id: int) -> None:
    """
    MVIVF v2 params blob (u32 fields, in order):
        k_per_level, max_leaf_size, max_depth, num_spill, s,
        mvclus.niters, mvclus.max_point_clouds_per_cluster,
        mvclus.max_points_per_centroid_inner_kmeans
      + trailing bool quantize_centers (v2 only).

    v3 drops the trailing byte; everything else is copied verbatim.
    """
    with src.open("rb") as fin, dst.open("wb") as fout:
        _read_v2_header_check(fin, "mvivf")
        _write_header(fout, "mvivf", class_id)
        _convert_mvivf_params_blob_v2_to_v3(fin, fout, params_u32_count=8)
        fout.write(fin.read())


def convert_mvivf_flat(src: Path, dst: Path, class_id: int) -> None:
    """
    MVIVFFlat v2 params blob (u32):
        k_per_level, s, mvclus.niters, mvclus.max_point_clouds_per_cluster,
        mvclus.max_points_per_centroid_inner_kmeans
      + trailing bool quantize_centers (v2 only).
    """
    with src.open("rb") as fin, dst.open("wb") as fout:
        _read_v2_header_check(fin, "mvivf_flat")
        _write_header(fout, "mvivf_flat", class_id)
        _convert_mvivf_params_blob_v2_to_v3(fin, fout, params_u32_count=5)
        fout.write(fin.read())


def convert_mvivf_spill(src: Path, dst: Path, class_id: int) -> None:
    """
    MVIVFSpill v2 params blob (u32):
        k_per_level, s, mvclus.niters, mvclus.max_point_clouds_per_cluster,
        mvclus.max_points_per_centroid_inner_kmeans, max_leaf_size, max_depth,
        num_spill, num_spill_l2
      + trailing bool quantize_centers (v2 only).
    """
    with src.open("rb") as fin, dst.open("wb") as fout:
        _read_v2_header_check(fin, "mvivf_spill")
        _write_header(fout, "mvivf_spill", class_id)
        _convert_mvivf_params_blob_v2_to_v3(fin, fout, params_u32_count=9)
        fout.write(fin.read())


def _read_v2_header_check(fin, family: str) -> int:
    """Consume the v2 magic/version/class_id triple and return class_id unchanged."""
    hdr = fin.read(12)
    if len(hdr) != 12:
        raise ValueError(f"{family}: source file too short for a v2 header")
    magic, version, class_id = struct.unpack("<III", hdr)
    if magic != MAGIC[family]:
        raise ValueError(
            f"{family}: unexpected magic 0x{magic:08X} (want 0x{MAGIC[family]:08X})"
        )
    if version != 2:
        raise ValueError(f"{family}: source version {version} (expected 2 for v2->v3 upgrade)")
    return class_id


CONVERTERS: Dict[str, Callable[[Path, Path, int], None]] = {
    "vamana": convert_vamana,
    "muvera": convert_muvera,
    "mpool": convert_mpool,
    "svhgraph": convert_svhgraph,
    "svhivf": convert_svhivf,
    "mvivf": convert_mvivf,
    "mvivf_flat": convert_mvivf_flat,
    "mvivf_spill": convert_mvivf_spill,
}


def compute_class_id(family: str, leaf_model: str, compress_centers: bool) -> int:
    leaf_id = LEAF_METHOD_IDS.get(leaf_model)
    if leaf_id is None:
        raise ValueError(
            f"Unknown --leaf-model {leaf_model!r}; valid choices: "
            f"{sorted(LEAF_METHOD_IDS)}"
        )
    if family in {"svhivf", "mvivf", "mvivf_flat", "mvivf_spill"}:
        return (leaf_id << 1) | (1 if compress_centers else 0)
    return leaf_id


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=sorted(CONVERTERS))
    parser.add_argument("--leaf-model", default="None", choices=sorted(LEAF_METHOD_IDS))
    parser.add_argument("--compress-centers", action="store_true",
                        help="SVHIVF / MVIVF family: center TurboQuant is enabled "
                             "in the concrete class.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    if not args.input.is_file():
        print(f"[convert] input not found: {args.input}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)

    class_id = compute_class_id(args.family, args.leaf_model, args.compress_centers)
    CONVERTERS[args.family](args.input, args.output, class_id)
    print(f"[convert] wrote {args.output} (family={args.family}, class_id={class_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
