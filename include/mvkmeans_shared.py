import numpy as np
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing as mp
from itertools import chain
import time
import statistics
from multiprocessing import shared_memory

import kmeans
from seeding import random_seeding
from utils import chamfer_distance


# ---------- Batching Helper ----------
def split_batches(n, batch_size):
    return [range(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]


# ---------- Shared Memory Setup ----------
def setup_shared_array(X):
    shm = shared_memory.SharedMemory(create=True, size=X.nbytes)
    shared_X = np.ndarray(X.shape, dtype=X.dtype, buffer=shm.buf)
    shared_X[:] = X[:]
    return shm, X.shape, X.dtype


# ---------- Assign Clusters Using Shared Memory ----------
def process_batch_assign(batch_indices, centroids, shape, dtype, shm_name):
    shm = shared_memory.SharedMemory(name=shm_name)
    X = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    results = []
    for i in batch_indices:
        dists = [chamfer_distance(X[i], c) for c in centroids]
        results.append(np.argmin(dists))
    shm.close()
    return results

def assign_clusters_chamfer_shared(X, centroids, n_jobs=-1):
    shm, shape, dtype = setup_shared_array(X)
    try:
        n = len(X)
        cpu_count = mp.cpu_count() if n_jobs == -1 else n_jobs
        batch_size = max(1, n // (2 * cpu_count))
        batches = split_batches(n, batch_size)

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(process_batch_assign)(batch, centroids, shape, dtype, shm.name)
            for batch in tqdm(batches, desc="Assigning clusters (shared)")
        )
        return np.fromiter(chain.from_iterable(results), dtype=np.int32, count=n)
    finally:
        shm.close()
        shm.unlink()


# ---------- Cost Computation Using Shared Memory ----------
def process_batch_cost(batch_indices, centroids, cluster_ids, shape, dtype, shm_name):
    shm = shared_memory.SharedMemory(name=shm_name)
    X = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    cost = 0.0
    for i in batch_indices:
        cost += chamfer_distance(X[i], centroids[cluster_ids[i]])
    shm.close()
    return cost

def cost_shared(X, centroids, cluster_ids, n_jobs=-1):
    shm, shape, dtype = setup_shared_array(X)
    try:
        n = len(X)
        cpu_count = mp.cpu_count() if n_jobs == -1 else n_jobs
        batch_size = max(1, n // cpu_count)
        batches = split_batches(n, batch_size)

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(process_batch_cost)(batch, centroids, cluster_ids, shape, dtype, shm.name)
            for batch in tqdm(batches, desc="Computing cost (shared)")
        )
        return sum(results)
    finally:
        shm.close()
        shm.unlink()


# ---------- Main Loop ----------
def mvkmeans(X, k, max_iter=5, n_jobs=-1):
    s, d = X[0].shape

    assign_times = []
    cost_times = []
    centroid_times = []

    centroids = random_seeding(X, k)

    # Initial cluster assignment
    t0 = time.perf_counter()
    cluster_ids = assign_clusters_chamfer_shared(X, centroids, n_jobs)
    t1 = time.perf_counter()
    assign_times.append(t1 - t0)
    print("Initial Cluster IDs Computed", flush=True)

    # Initial cost
    t0 = time.perf_counter()
    seed_cost = cost_shared(X, centroids, cluster_ids, n_jobs)
    t1 = time.perf_counter()
    cost_times.append(t1 - t0)
    print(f"Seeding Cost: {seed_cost:.4f}")
    print(f"[TIMER] assign_clusters: {assign_times[-1]:.2f}s, cost: {cost_times[-1]:.2f}s")

    for t in range(max_iter):
        print(f"\n--- Iteration {t} ---")

        # Group indices by cluster
        clusters = {i: np.where(cluster_ids == i)[0] for i in range(k)}

        # Recompute centroids
        t0 = time.perf_counter()
        new_centroids = []
        for i in range(k):
            indices = clusters[i]
            if len(indices) == 0:
                print(f"Cluster {i} is empty. Reseeding...")
                new_centroids.append(random_seeding(X, 1)[0])
                continue
            data = X[indices].reshape(len(indices) * s, d)
            new_centroids.append(kmeans.faiss_kmeans(data, s))
        centroids = np.array(new_centroids)
        t1 = time.perf_counter()
        centroid_times.append(t1 - t0)

        # Reassign
        t0 = time.perf_counter()
        cluster_ids = assign_clusters_chamfer_shared(X, centroids, n_jobs)
        t1 = time.perf_counter()
        assign_times.append(t1 - t0)

        # Compute cost
        t0 = time.perf_counter()
        round_cost = cost_shared(X, centroids, cluster_ids, n_jobs)
        t1 = time.perf_counter()
        cost_times.append(t1 - t0)

        print(f"Round {t} Cost: {round_cost:.4f}")
        print(f"[TIMER] assign_clusters: {assign_times[-1]:.2f}s, centroid update: {centroid_times[-1]:.2f}s, cost: {cost_times[-1]:.2f}s")

    # Timing Summary
    print("\n=== Timing Summary ===")
    print(f"Average assign_clusters time: {statistics.mean(assign_times):.2f}s")
    print(f"Average centroid update time: {statistics.mean(centroid_times):.2f}s")
    print(f"Average cost computation time: {statistics.mean(cost_times):.2f}s")


# ---------- Demo ----------
if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(100000, 10, 3).astype(np.float32)
    print("Sample input shape:", X.shape)
    mvkmeans(X, k=4, max_iter=5, n_jobs=8)
