import mvsic
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file")
    parser.add_argument("--query_path", type=str, required=True, help="Path to the query file")
    parser.add_argument("--index_path", type=str, required=True, help="Path to save/load the index")
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

    print("Setting up index...")
    index_params = mvsic.IndexParams.mvivf()
    index = mvsic.IndexMVIVFIP(args.dim, index_params)

    # print("Building index...")
    # index.build(points)

    # print(f"Saving index to {args.index_path}...")
    # index.save(args.index_path)

    print("Loading index...")
    index.load(args.index_path, points)

    print("Searching for a single query...")
    search_params = mvsic.SearchParams.mvivf(args.k, args.nprobes, args.num_rerank)

    query_point = queries[0]
    results, dist_cmps = index.search(query_point, points, search_params)

    print(f"Search complete. Performed {dist_cmps} distance comparisons.")
    print(f"Top {args.k} results for the first query:")
    for r in results:
        print(f"  ID: {r[0]}, Distance: {r[1]}")

    print("\nSearching for all queries...")
    all_results, total_dist_cmps = index.search_all(queries, points, search_params)
    print(f"Search complete. Performed {total_dist_cmps} distance comparisons.")
    print(f"Top {args.k} results for the first query (from search_all):")
    for r in all_results[0]:
        print(f"  ID: {r[0]}, Distance: {r[1]}")


if __name__ == "__main__":
    main()
