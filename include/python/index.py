import faiss
import numpy as np

from mvkmeans import mvkmeans

class FaissIndex:
	def __init__(self, dimension):
		self.d = dimension
		self.index = faiss.IndexFlatL2(dimension)

	def add_data(self, data):
		if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] != self.d:
			raise ValueError(f"Data must be a 2D numpy array with shape (n_samples, {self.d})")
		self.index.add(data)

	def search(self, query, k=5):
		if not isinstance(query, np.ndarray) or query.ndim != 2 or query.shape[1] != self.d:
			raise ValueError(f"Query must be a 2D numpy array with shape (n_queries, {self.d})")
		distances, indices = self.index.search(query, k)
		return distances, indices

class MVIVF:
	def __init__(self, data):
		"""
		data: PointCloud object
		"""
		
	def build_index(self):
		centers, cluster_ids = mvkmeans(self.data, k=self.n/500)
		self.centers = centers
		self.clusters = {i: np.where(cluster_ids == i)[0] for i in range(k)}
		self.index = {i: FaissIndex(self.d) for i in range(k)}
		for i in range(k):
			data = np.concatenate([self.data.coords(j) for j in self.clusters[i]]).reshape(-1, self.d)
			self.index[i].add_data(self.data[self.clusters[i]])


if __name__ == "__main__":
	dimension = 128
	faiss_index = FaissIndex(dimension)
	
	data = np.random.random((100, dimension)).astype('float32')
	print(data.shape)
	faiss_index.add_data(data)

	query = np.random.random((5, dimension)).astype('float32')
	distances, indices = faiss_index.search(query, k=3)

	print("Distances:\n", distances)
	print("Indices:\n", indices)