#pragma once

#include "parlay/primitives.h"
#include "faiss/Clustering.h"
#include "faiss/IndexFlat.h"

template <typename Seq>
auto faiss_kmeans(const Seq& data, size_t d, size_t k, bool is_metric) {
  size_t n = data.size();

  // Flatten input into raw float array for FAISS
  auto flat_data = parlay::flatten(data);

  // Setup clustering parameters
  faiss::Clustering clus(d, k);
  clus.verbose = false;
  clus.min_points_per_centroid = 1;
  // clus.max_points_per_centroid = 1000000000;


  if (is_metric) {
    // Index used to assign points during clustering (L2 distance)
    faiss::IndexFlatL2 index(d);
    clus.train(n, flat_data.data(), index);
  } else{
    // Index used to assign points during clustering (cosine distance)
    faiss::IndexFlatIP index(d);
    clus.train(n, flat_data.data(), index);
  }

  // Extract centroids
  float* centroids_ptr = clus.centroids.data();
  parlay::sequence<parlay::sequence<float>> centroids(k, parlay::sequence<float>::uninitialized(d));
  parlay::parallel_for(0, k, [&](size_t i) {
    std::memcpy(centroids[i].begin(), centroids_ptr + i * d, d*sizeof(float));
  });
  return std::move(centroids);
}

template <typename Seq>
auto faiss_kmeans_assign(const Seq& data, size_t d, size_t k, bool is_metric) {
  size_t n = data.size();

  // Flatten input into raw float array for FAISS
  auto flat_data = parlay::flatten(data);

  // Setup clustering parameters
  faiss::Clustering clus(d, k);
  clus.verbose = false;
  clus.min_points_per_centroid = 1;
  // clus.max_points_per_centroid = 1000000000;

  parlay::sequence<faiss::idx_t> assignments(n); // To store cluster assignments
  parlay::sequence<float> distances(n);

  if (is_metric) {
    // Index used to assign points during clustering (L2 distance)
    faiss::IndexFlatL2 index(d);
    clus.train(n, flat_data.data(), index);
    faiss::IndexFlatL2 search_index(d);
    search_index.add(k, clus.centroids.data());
    search_index.search(n, flat_data.data(), 1, distances.data(), assignments.data());
  } else{
    // Index used to assign points during clustering (cosine distance)
    faiss::IndexFlatIP index(d);
    clus.train(n, flat_data.data(), index);
    faiss::IndexFlatIP search_index(d);
    search_index.add(k, clus.centroids.data());
    search_index.search(n, flat_data.data(), 1, distances.data(), assignments.data());
  }

  // Extract centroids
  float* centroids_ptr = clus.centroids.data();
  parlay::sequence<parlay::sequence<float>> centroids(k, parlay::sequence<float>::uninitialized(d));
  parlay::parallel_for(0, k, [&](size_t i) {
    std::memcpy(centroids[i].begin(), centroids_ptr + i * d, d*sizeof(float));
  });

  // Remove clusters with all almost-duplicates
  auto id_pt = parlay::delayed_tabulate(n, [&](size_t i) { return std::make_pair(assignments[i], i); });
  auto grouped = parlay::group_by_index(id_pt, k);
  parlay::sequence<bool> active(k,true);
  parlay::parallel_for(0, k, [&](size_t i) {
    // average distance to center
    auto dists = parlay::delayed_tabulate(grouped[i].size(), [&](size_t j) {
      return distances[grouped[i][j]];
    });
    float avg_dist = parlay::reduce(dists) / grouped[i].size();
    if (avg_dist < 1e-5) {
      active[i] = false;
    }
  });
  auto active_indices = parlay::pack_index(active);
  return std::make_tuple(std::move(centroids), std::move(assignments), std::move(active_indices));
}

template <typename Seq>
auto faiss_wgh_kmeans(const Seq& data, size_t d, size_t k, 
    parlay::sequence<float>& wghs, bool is_metric) {
  size_t n = data.size();

  // Flatten input into raw float array for FAISS
  auto flat_data = parlay::flatten(data);

  // Setup clustering parameters
  faiss::Clustering clus(d, k);
  clus.verbose = false;
  clus.min_points_per_centroid = 1;
  // clus.max_points_per_centroid = 1000000000;

  if (is_metric) {
    // Index used to assign points during clustering (L2 distance)
    faiss::IndexFlatL2 index(d);
    clus.train(n, flat_data.data(), index, wghs.data());
  } else{
    // Index used to assign points during clustering (cosine distance)
    faiss::IndexFlatIP index(d);
    clus.train(n, flat_data.data(), index, wghs.data());
  }

  // Extract centroids
  float* centroids_ptr = clus.centroids.data();
  parlay::sequence<parlay::sequence<float>> centroids(k, parlay::sequence<float>::uninitialized(d));
  parlay::parallel_for(0, k, [&](size_t i) {
    std::memcpy(centroids[i].begin(), centroids_ptr + i * d, d*sizeof(float));
  });

  return centroids;
}

template <typename Seq>
auto faiss_kmeans_cost(const Seq& data, size_t d, size_t k, 
    parlay::sequence<float>& wghs={}, bool is_metric=false) {
  size_t n = data.size();

  if (wghs.size() == 0) {
    wghs = parlay::sequence<float>::from_function(n, [&](size_t i) { return 1; });
  }

  // Flatten input into raw float array for FAISS
  auto flat_data = parlay::flatten(data);

  // Setup clustering parameters
  faiss::Clustering clus(d, k);
  clus.verbose = false;
  clus.min_points_per_centroid = 1;
  // clus.max_points_per_centroid = 1000000000;

  if (is_metric) {
    // Index used to assign points during clustering (L2 distance)
    faiss::IndexFlatL2 index(d);
    clus.train(n, flat_data.data(), index, wghs.data());
  } else{
    // Index used to assign points during clustering (cosine distance)
    faiss::IndexFlatIP index(d);
    clus.train(n, flat_data.data(), index, wghs.data());
  }
  const faiss::ClusteringIterationStats& final_stats = clus.iteration_stats.back();
  float final_cost = final_stats.obj;

  return final_cost;
}