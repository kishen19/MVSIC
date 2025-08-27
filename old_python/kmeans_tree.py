import numpy as np
import heapq

from kmeans import faiss_kmeans
from index import FaissIndex  


class KMeansTreeNode:
  def __init__(self, centroid, points=None):
    self.centroid = centroid
    self.points = points
    self.children = []
    self.faiss_index = None  # Per-level or leaf index


class KMeansTree:
  def __init__(self, l, m):
    self.l = l  # Number of clusters at each level
    self.m = m  # Minimum cluster size to stop recursion
    self.root = None
    self.global_faiss_index = FaissIndex()  # Single global Faiss index
    self.global_ids = 0  # Unique ID counter for points and centroids

  def build_tree(self, data):
    self.root = self._build_node(data)

  def _build_node(self, data):
    if len(data) <= self.m:
      # Add points to a leaf-level Faiss index
      point_ids = np.arange(self.global_ids, self.global_ids + len(data))
      faiss_index = FaissIndex()
      faiss_index.add(data, point_ids)
      self.global_ids += len(data)

      centroid = np.mean(data, axis=0)
      node = KMeansTreeNode(centroid, points=data)
      node.faiss_index = faiss_index
      return node

    # Perform k-means clustering using the implementation from kmeans.py
    centroids, labels = kmeans(data, self.l)

    # Add centroids to the global Faiss index
    centroid_ids = np.arange(self.global_ids, self.global_ids + len(centroids))
    self.global_faiss_index.add(centroids, centroid_ids)
    self.global_ids += len(centroids)

    # Create the current node
    node = KMeansTreeNode(centroid=np.mean(data, axis=0))

    # Create a per-level Faiss index for centroids
    faiss_index = FaissIndex()
    faiss_index.add(centroids, centroid_ids)
    node.faiss_index = faiss_index

    # Recursively build child nodes
    for i in range(self.l):
      cluster_points = data[labels == i]
      if len(cluster_points) > 0:
        child_node = self._build_node(cluster_points)
        node.children.append(child_node)

    return node

  def search(self, query_point, k, b):
    # Priority queue for internal nodes
    priority_queue = []
    heapq.heappush(priority_queue, (0, self.root))  # (distance, node)

    leaf_nodes = []
    while priority_queue and len(leaf_nodes) < b:
      dist, node = heapq.heappop(priority_queue)

      if node.faiss_index is not None and node.points is not None:
        # Leaf node
        leaf_nodes.append(node)
      else:
        # Internal node, push children into the queue
        for child in node.children:
          child_dist = np.linalg.norm(query_point - child.centroid)
          heapq.heappush(priority_queue, (child_dist, child))

    # Collect k nearest neighbors from the b leaf nodes
    all_distances = []
    all_indices = []
    for leaf in leaf_nodes:
      distances, indices = leaf.faiss_index.search(query_point, k)
      all_distances.extend(distances)
      all_indices.extend(indices)

    # Combine and sort results to get the top k neighbors
    combined = list(zip(all_distances, all_indices))
    combined.sort(key=lambda x: x[0])
    return combined[:k]
    return list(zip(distances, indices))
