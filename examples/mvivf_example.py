import mvsic
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file")
    parser.add_argument("--query_path", type=str, required=True, help="Path to the query file")
    parser.add_argument("--index_path", type=str, required=True, help="Path to save/load the index")
    parser.add_argument("--gt_path", type=str, required=True, help="Path to the ground truth file")
    parser.add_argument("--dim", type=int, required=True, help="Dimension of the vectors")
    parser.add_argument(
        "-k", type=int, default=10, help="Number of nearest neighbors to search for"
    )
    parser.add_argument("--nprobes", type=int, default=10, help="Number of probes for searching")
    parser.add_argument(
        "--num_rerank", type=int, default=100, help="Number of candidates to rerank"
    )
    args = parser.parse_args()

    print("Loading data...")
    points = mvsic.PointCloudSetIP(args.data_path)

    print("Loading queries...")
    queries = mvsic.PointCloudSetIP(args.query_path)

    print("Loading ground truth...")
    gt = mvsic.ReadGT(args.gt_path, queries.size())

    print("Setting up index...")
    index_params = mvsic.IndexParams.mvivf()
    index = mvsic.IndexMVIVFIP(args.dim, index_params)

    print("Building index...")
    index.build(points)

    print(f"Saving index to {args.index_path}...")
    index.save(args.index_path)

    # print("Loading index...")
    # index.load(args.index_path, points)

    print("Computing stats...")
    search_params = mvsic.SearchParams.mvivf(args.k, args.nprobes, args.num_rerank)
    stats = mvsic.compute_stats(index, points, queries, gt, search_params)

    print("Stats:")
    print(f"  Sequential QPS: {stats.QPS_seq}")
    print(f"  Parallel QPS: {stats.QPS_par}")
    print(f"  Average Comparisons: {stats.avg_cmps}")
    print(f"  Recall @ 1: {stats.recall_1_k}")
    print(f"  Recall @ k: {stats.recall_k_k}")


if __name__ == "__main__":
    main()
