"""
Script to filter queries that have golden gt values.
Input:
    --path <path>
    --dataset <dataset>
Output:
Generates the files:
    /<path>/<dataset>/<dataset>_reduced_queries.pcs
    /<path>/<dataset>/<dataset>_reduced_chamfer_neighbors.gt
"""

import argparse

import numpy as np
from utils import ReadGoldGT, ReadGT, ReadQueries

pair_dtype = np.dtype(
    [("distance", np.float32), ("id", np.uint32)]
)  # Each pair is 4 (float) + 4 (uint32) = 8 bytes


def WriteQueries(filename, queries):
    with open(filename, "wb") as f:
        dims = queries[0].shape[1]
        n = len(queries)
        num_vectors = sum([q.shape[0] for q in queries])
        f.write(np.array(dims, dtype=np.uint64).tobytes())
        f.write(np.array(n, dtype=np.uint64).tobytes())
        f.write(np.array(num_vectors, dtype=np.uint64).tobytes())
        vectors = np.ravel(queries)
        print(vectors.shape)
        f.write(vectors.astype(np.float32).tobytes())
        offsets = np.zeros(n + 1, dtype=np.uint64)
        f.write(np.array(n + 1, dtype=np.uint64).tobytes())
        offsets[0] = 0
        for i in range(1, n + 1):
            offsets[i] = offsets[i - 1] + queries[i - 1].shape[0] * dims
        f.write(offsets.astype(np.uint64).tobytes())


def WriteGT(file_path, gt):
    with open(file_path, "wb") as f:
        num_neighbors = len(gt[0])
        f.write(np.array(num_neighbors, dtype=np.int32).tobytes())
        for i in range(len(gt)):
            for j in range(num_neighbors):
                f.write(np.array((gt[i][j][0], gt[i][j][1]), dtype=pair_dtype).tobytes())


def reduce_queries(path, dataset):
    queries = ReadQueries(f"{path}/{dataset}/{dataset}_queries.pcs")
    print(f"Queried loaded. Number of queries: {len(queries)}")
    gt = ReadGT(f"{path}/{dataset}/{dataset}_chamfer_neighbors.gt", len(queries))
    print(f"GT loaded. Number of GT entries: {len(gt)}")
    gtgold = ReadGoldGT(f"{path}/{dataset}/{dataset}_gold_neighbors.csr", len(queries))
    print(f"Gold GT loaded. Number of Gold GT entries: {len(gtgold)}")
    indices = []
    for i in range(len(gtgold)):
        if len(gtgold[i]) > 0:
            indices.append(i)
    print(f"Number of queries with at least one Gold GT: {len(indices)}")
    reduced_queries = [queries[i] for i in indices]
    reduced_gt = [gt[i] for i in indices]
    WriteQueries(f"{path}/{dataset}/{dataset}_reduced_queries.pcs", reduced_queries)
    print("Reduced queries written.")
    WriteGT(f"{path}/{dataset}/{dataset}_reduced_chamfer_neighbors.gt", reduced_gt)
    print("Reduced GT written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduce queries and GT files.")
    parser.add_argument(
        "-p", "--path", type=str, required=True, help="Path to the dataset directory."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset (e.g., 'nq').",
    )
    args = parser.parse_args()
    path = args.path
    dataset = args.dataset
    reduce_queries(path, dataset)
    print("Queries reduced successfully.")
