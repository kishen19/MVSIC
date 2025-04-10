import faiss
import numpy as np
import os

def faiss_kmeans(data, k, n_iter=25, n_threads=None):
	if n_threads is None:
		n_threads = os.cpu_count()

	faiss.omp_set_num_threads(n_threads)

	d = data.shape[1]
	kmeans = faiss.Kmeans(d=d, k=k, niter=n_iter, verbose=False)
	kmeans.train(data)

	# distances, assignments = kmeans.index.search(data, 1)  # assignments: shape (n_samples, 1)
	return kmeans.centroids

if __name__ == "__main__":
	data = np.random.random((1000000, 3)).astype('float32')
	centers = faiss_kmeans(data, k=5)

	print("Centers shape:", centers.shape)       # (10, 128)
	# print("Labels shape:", labels.shape)         # (100000,)
	# print("First 5 labels:", labels[:5])

