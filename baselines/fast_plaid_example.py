import argparse
import torch
import numpy as np
import struct
import time
import sys
from fast_plaid.search import FastPlaid

def load_point_clouds(file_path):
    point_clouds = []
    with open(file_path, 'rb') as f:
        dim = struct.unpack('Q', f.read(8))[0]
        n = struct.unpack('Q', f.read(8))[0]
        num_vectors = struct.unpack('Q', f.read(8))[0]
        
        data = np.fromfile(f, dtype=np.float32, count=num_vectors * dim)
        data = data.reshape(num_vectors, dim)
        
        num_offsets = struct.unpack('Q', f.read(8))[0]
        offsets = np.fromfile(f, dtype=np.uint64, count=num_offsets)
        
        for i in range(n):
            start = offsets[i] // dim
            end = offsets[i+1] // dim
            point_clouds.append(torch.from_numpy(data[start:end]))
            
    return point_clouds

def load_gt(file_path, num_points):
    with open(file_path, 'rb') as f:
        num_neighbors = struct.unpack('i', f.read(4))[0]
        gt = []
        for _ in range(num_points):
            neighbors = []
            for _ in range(num_neighbors):
                dist = struct.unpack('f', f.read(4))[0]
                doc_id = struct.unpack('I', f.read(4))[0]
                neighbors.append((dist, doc_id))
            gt.append(neighbors)
    return gt

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
    # Indexing parameters
    parser.add_argument("--n_samples_kmeans", type=int, default=None, help="Number of samples to compute centroids")
    parser.add_argument("--nbits", type=int, default=4, help="Product quantization bits")
    parser.add_argument("--kmeans_niters", type=int, default=4, help="K-means iterations")

    # Search parameters
    parser.add_argument("--n_ivf_probe", type=int, default=8, help="Cluster probes per query")
    parser.add_argument("--n_full_scores", type=int, default=4096, help="Candidates for full scoring")

    args = parser.parse_args()

    print("Loading data...")
    start_time = time.time()
    documents = load_point_clouds(args.data_path)
    print(f"Loaded {len(documents)} documents in {time.time() - start_time:.2f} seconds")

    print("Loading queries...")
    start_time = time.time()
    queries = load_point_clouds(args.query_path)
    print(f"Loaded {len(queries)} queries in {time.time() - start_time:.2f} seconds")

    print("Loading ground truth...")
    start_time = time.time()
    gt = load_gt(args.gt_path, len(queries))
    print(f"Loaded ground truth in {time.time() - start_time:.2f} seconds")

    print("Setting up index...")
    index = FastPlaid(index=args.index_path)

    print("Building index...")
    start_time = time.time()
    index.create(
        documents_embeddings=documents,
        n_samples_kmeans=args.n_samples_kmeans,
        nbits=args.nbits,
        kmeans_niters=args.kmeans_niters
    )
    print(f"Index built in {time.time() - start_time:.2f} seconds")

    print("Searching...")
    start_time = time.time()
    results = index.search(
        queries_embeddings=queries, 
        top_k=args.k,
        n_ivf_probe=args.n_ivf_probe,
        n_full_scores=args.n_full_scores
    )
    print(f"Search completed in {time.time() - start_time:.2f} seconds")

    print(f"Top {args.k} neighbors for the first query:")
    if results and results[0]:
        for doc_id, score in results[0]:
            print(f"  Document ID: {doc_id}, Score: {score}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting by user request.")
        sys.exit(0)
