import faiss
import numpy as np

from mvkmeans import mvkmeans

class MVIVF:
	def __init__(self, data):
		
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