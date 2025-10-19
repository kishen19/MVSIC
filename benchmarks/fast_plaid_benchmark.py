import argparse
import time
import sys
import os
import csv

from fast_plaid.search import FastPlaid
import mvsic
from utils import load_point_clouds, load_gt, update_index_time_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--nbits", type=int, default=4, help="Product quantization bits")
    parser.add_argument("--build", action='store_true', help="Build the index")
    parser.add_argument(
        "--update_index_time", action='store_true', help="Update the index time in the CSV"
    )

    args = parser.parse_args()

    # Construct paths
    dataset_path = f"./data/{args.dataset}/{args.dataset}_points.pcs"
    query_path = f"./data/{args.dataset}/{args.dataset}_queries.pcs"
    gt_path = f"./data/{args.dataset}/{args.dataset}_chamfer_neighbors.gt"
    index_path = f"./results1/{args.dataset}/indices/fastplaid_nbits{args.nbits}"
    results_path = f"./results1/{args.dataset}/stats/fastplaid_nbits{args.nbits}.csv"

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    print("Loading data...")
    documents = load_point_clouds(dataset_path)
    print(f"Loaded {len(documents)} documents")

    print("Loading queries...")
    queries = load_point_clouds(query_path)
    print(f"Loaded {len(queries)} queries")

    print("Loading ground truth...")
    gt = load_gt(gt_path, len(queries))
    print("Loaded ground truth")

    # print("\n--- TEMPORARY: Using only the first 5 queries for testing ---\n")
    # queries = queries[:5]
    # gt = gt[:5]

    index = FastPlaid(index=index_path)

    if args.build:
        print("Building index...")
        start_time = time.time()
        index.create(
            documents_embeddings=documents,
            nbits=args.nbits,
        )
        build_time = time.time() - start_time
        print(f"Index built in {build_time:.2f} seconds")

        if args.update_index_time:
            method = f"fastplaid_nbits{args.nbits}"
            update_index_time_csv(args.dataset, method, build_time)
            print("Index time updated in ./results1/index_times.csv")
    else:
        # The index is loaded automatically by the constructor if it exists
        print(f"Index at {index_path} is ready.")

    ks = [10, 100]
    n_ivf_probes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    n_full_scores_multipliers = [1, 2, 4, 8]

    with open(results_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            ['k', 'nprobes', 'num_rerank', 'recall_1@k', 'recall_k@k', 'QPS', 'QPS_par', 'Avg_cmps']
        )

        for k in ks:
            for n_ivf_probe in n_ivf_probes:
                for multiplier in n_full_scores_multipliers:
                    n_full_scores = multiplier * k

                    # Individual search
                    start_time = time.time()
                    individual_results = []
                    for q in queries:
                        results = index.search(
                            queries_embeddings=[q],
                            top_k=k,
                            n_ivf_probe=n_ivf_probe,
                            n_full_scores=n_full_scores,
                        )
                        individual_results.append([res[0] for res in results[0]])
                    end_time = time.time()
                    qps = len(queries) / (end_time - start_time)

                    # Batch search
                    start_time = time.time()
                    batch_results_with_scores = index.search(
                        queries_embeddings=queries,
                        top_k=k,
                        n_ivf_probe=n_ivf_probe,
                        n_full_scores=n_full_scores,
                    )
                    end_time = time.time()
                    qps_par = len(queries) / (end_time - start_time)

                    recall_1, recall_k = mvsic.compute_stats(batch_results_with_scores, gt, k)

                    # Avg_cmps is not available in fast_plaid, so we put 0
                    avg_cmps = 0

                    print(
                        f"k={k}, n_ivf_probe={n_ivf_probe}, n_full_scores={n_full_scores}, recall 1@{k}={recall_1:.4f}, recall {k}@{k}={recall_k:.4f}, QPS={qps:.2f}, QPS_par={qps_par:.2f}"
                    )

                    csv_writer.writerow(
                        [k, n_ivf_probe, n_full_scores, recall_1, recall_k, qps, qps_par, avg_cmps]
                    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting by user request.")
        sys.exit(0)
