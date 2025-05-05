import sys
import numpy as np
import fastcluster
from scipy.cluster.hierarchy import fcluster
from multiprocessing import Pool, cpu_count
from functools import partial
import time # Optional: for timing
from data_types import PointCloud

def _calculate_pooled_vectors(data, labels):
	unique_labels = np.unique(labels)
	pooled_vectors = np.array([
		np.mean(data[labels == label_id], axis=0) 
		for label_id in unique_labels
	])
	return pooled_vectors

def _process_single_point(data, pf, metric, linkage):
	results_for_point = {}
	n_vectors, n_dims = data.shape
			
	if n_vectors == 1:
		return data

	# 1. Calculate linkage matrix (dendrogram) - Done only ONCE per point
	# Ensure data is C-contiguous and float64 for fastcluster
	# data_contiguous = np.ascontiguousarray(data, dtype=np.float64)
	dend = fastcluster.linkage(data, method=linkage, metric=metric)

	# 2. Iterate through pooling factors and perform pooling
	# Calculate the target number of clusters
	# Ensure k is at least 1 and not more than n_vectors
	k = min(n_vectors, max(1, n_vectors // pf + 1)) 

	if k == 1:
		# If only one cluster, the result is the mean of all vectors
		pooled_vec = np.mean(data, axis=0, keepdims=True)
	elif k == n_vectors:
		# If k equals n, each original vector is its own cluster
		pooled_vec = data 
	else:
		# Get cluster assignments for k clusters
		labels = fcluster(dend, k, criterion='maxclust')
		# Calculate the mean vector for each cluster
		pooled_vec = _calculate_pooled_vectors(data, labels)
	return pooled_vec

def pooling(X, pooling_factor = 2, linkage="ward", metric="euclidean", num_workers=None):
	start_time = time.time() # Optional timing

	data_list = [X[i].to_nparray() for i in range(X.size())]
	if num_workers is None:
		num_workers = cpu_count()
	num_workers = min(num_workers, len(data_list))
	
	print(f"Starting pooling with {num_workers} workers for {len(data_list)} points...")

	# Create a partial function to fix the arguments for the worker
	# This is needed because pool.map passes only one argument from the iterable
	worker_func = partial(_process_single_point, 
												pf=pooling_factor,
												metric=metric,
												linkage=linkage)
	results_list = []
	with Pool(processes=num_workers) as pool:
		# Map the worker function over the data points
		results_list = pool.map(worker_func, data_list)

	print(f"Parallel processing finished. Aggregating results...")
	end_time = time.time() # Optional timing
	print(f"Pooling completed in {end_time - start_time:.2f} seconds.")
	print(results_list[0].shape)
	return PointCloud(results_list)

if __name__ == "__main__":
	np.random.seed(42)

	# Dummy Example
	# X0 = np.random.rand(100000, 10, 3).astype(np.float32)
	# X = PointCloud(X0)
	# print("Sample input shape:", X.size())
	# print("Sample input dims:", X.dims())
	# print("Sample input coords:", X.coords(0))
	# print("Sample input coords shape:", X.coords(0).shape)
	# mvkmeans(X, k=5, s=10, max_iter=5, n_jobs=8)
	# X.delete()
	filename = str(sys.argv[1]).strip()
	pooling_factor = int(sys.argv[2])
	outfile = str(sys.argv[3]).strip()
	X = PointCloud(filename ,np.float32)
	print("Input shape:", X.size())
	print("Input dims:", X.dims())
	print("Input coords:", X.coords(0))
	print("Input coords shape:", X.coords(0).shape)
	pooled_X = pooling(X, pooling_factor, linkage="ward", metric="cosine")
	print("Pooled. Writing to file...", flush=True)
	pooled_X.save(outfile)
	print("Wrote to file:", outfile, flush=True)
	X.delete()
	pooled_X.delete()
