#!/usr/bin/env python3
"""
Print MVIVF index tree stats (num internal nodes, leaves, avg sizes, height).

Usage:
  python index_tree_stats.py --index <path_to_index.bin> --points <path_to_points.pcs> [--metric IP|L2]

The points file must be the same dataset used to build the index (needed to load the index).
Default metric is IP (inner product); use --metric L2 for L2 indices.
"""

import argparse
import sys

import mvsic


def main():
    parser = argparse.ArgumentParser(
        description="Print MVIVF index tree stats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--index",
        required=True,
        help="Path to the index file (e.g. index_<hash>.bin)",
    )
    parser.add_argument(
        "--points",
        required=True,
        help="Path to the point cloud set file (e.g. <name>_points.pcs)",
    )
    parser.add_argument(
        "--metric",
        choices=("IP", "L2"),
        default="IP",
        help="Distance metric the index was built with",
    )
    args = parser.parse_args()

    # Load points (required to load the index)
    points = mvsic.PointCloudSetIP(args.points) if args.metric == "IP" else mvsic.PointCloudSetL2(args.points)
    if points.size() == 0:
        print("Error: points file is empty.", file=sys.stderr)
        sys.exit(1)
    dim = points[0].get_dims()

    # Build params don't matter for load(); we only need dim and metric
    params = mvsic.IndexParams.mvivf()
    if args.metric == "IP":
        index = mvsic.IndexMVIVFIP(dim, params)
    else:
        index = mvsic.IndexMVIVFL2(dim, params)

    index.load(args.index, points)
    stats = mvsic.get_mvivf_tree_stats(index)

    print("Tree stats for index:", args.index)
    print("  num_internal_nodes        :", stats["num_internal_nodes"])
    print("  num_leaves                :", stats["num_leaves"])
    print("  avg_leaf_size             :", f"{stats['avg_leaf_size']:.2f}")
    print("  avg_internal_node_size    :", f"{stats['avg_internal_node_size']:.2f}")
    print("  total_point_clouds_internal:", stats["total_point_clouds_internal"])
    print("  height                   :", stats["height"])


if __name__ == "__main__":
    main()
