import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing as mp
from itertools import chain
import time

import kmeans
from seeding import random_seeding
from utils import chamfer_distance, split_batches, shared_array_init


# Assign Clusters
def process_batch_assign(batch_indices, X, centers):
    results = []
    for i in batch_indices:
        dists = [chamfer_distance(X[i], c) for c in centers]
        results.append(np.argmin(dists))
    return results

def assign_clusters(X, centers, n_jobs=-1):
    n = len(X)
    cpu_count = mp.cpu_count() if n_jobs == -1 else n_jobs
    batch_size = max(250, n // (2 * cpu_count))
    batches = split_batches(n, batch_size)
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_batch_assign)(batch, X, centers)
        for batch in tqdm(batches, desc="Assigning clusters")
    )
    return np.fromiter(chain.from_iterable(results), dtype=np.int32, count=n)


# Compute Cost given Assignments
def process_batch_cost(batch_indices, X, centers, cluster_ids):
    cost = 0.0
    for i in batch_indices:
        cost += chamfer_distance(X[i], centers[cluster_ids[i]])
    return cost

def cost(X, centers, cluster_ids, n_jobs=-1):
    n = len(X)
    cpu_count = mp.cpu_count() if n_jobs == -1 else n_jobs
    batch_size = max(250, n // cpu_count)
    batches = split_batches(n, batch_size)
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_batch_cost)(batch, X, centers, cluster_ids)
        for batch in tqdm(batches, desc="Computing cost")
    )
    return sum(results)

# Multi-Vector K-Means
def mvkmeans(X, k, max_iter=5, n_jobs=-1):
    s, d = X[0].shape

    # Timing Utils
    init_time = 0.0
    assign_times = []
    cost_times = []
    centroid_times = []

    # Initialization
    print("Seeding...", end=" ", flush=True)
    t0 = time.perf_counter()
    centers = random_seeding(X, k)
    t1 = time.perf_counter()
    print(f"Done in {t1 - t0:.2f}s", flush=True)
    init_time = t1 - t0

    print("Initial Cluster IDs...", end=" ", flush=True)
    t0 = time.perf_counter()
    cluster_ids = assign_clusters(X, centers, n_jobs)
    t1 = time.perf_counter()
    print(f"Computed in {t1 - t0:.2f}s", flush=True)
    assign_times.append(t1 - t0)

    print("Seeding Cost...", end=" ", flush=True)
    t0 = time.perf_counter()
    seed_cost = cost(X, centers, cluster_ids, n_jobs)
    t1 = time.perf_counter()
    print(f"Computed in {t1 - t0:.2f}s", flush=True)
    cost_times.append(t1 - t0)
    print(f"Seeding Cost: {seed_cost:.4f}")

    for t in range(max_iter):
        print(f"\n--- Iteration {t} ---")

        # Group indices by cluster
        clusters = {i: np.where(cluster_ids == i)[0] for i in range(k)}

        # Recompute centers
        t0 = time.perf_counter()
        new_centers = []
        for i in range(k):
            indices = clusters[i]
            if len(indices) == 0:
                print(f"Cluster {i} is empty. Reseeding...")
                new_centers.append(random_seeding(X, 1)[0])
                continue
            data = X[indices].reshape(len(indices) * s, d)
            new_centers.append(kmeans.faiss_kmeans(data, s))
        centers = np.array(new_centers)
        t1 = time.perf_counter()
        centroid_times.append(t1 - t0)

        # Reassign Points
        t0 = time.perf_counter()
        cluster_ids = assign_clusters(X, centers, n_jobs)
        t1 = time.perf_counter()
        assign_times.append(t1 - t0)

        # Recompute cost
        t0 = time.perf_counter()
        round_cost = cost(X, centers, cluster_ids, n_jobs)
        t1 = time.perf_counter()
        cost_times.append(t1 - t0)

        print(f"Round {t} Cost: {round_cost:.4f}")
        print(f"[TIMER] assign_clusters: {assign_times[-1]:.2f}s, centroid update: {centroid_times[-1]:.2f}s, cost: {cost_times[-1]:.2f}s")

    # Timing Summary
    print("\n=== Timing Summary ===")
    print(f"Average assign_clusters time: {np.mean(assign_times):.2f}s")
    print(f"Average centroid update time: {np.mean(centroid_times):.2f}s")
    print(f"Average cost computation time: {np.mean(cost_times):.2f}s")


# ---------- Demo ----------
if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(100000, 10, 3).astype(np.float32)
    shm, shared_X = shared_array_init(X)
    print("Sample input shape:", shared_X.shape)
    mvkmeans(shared_X, k=500, max_iter=5, n_jobs=-1)
    shm.close()
    shm.unlink()
