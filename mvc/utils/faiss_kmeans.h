#pragma once

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "faiss/Clustering.h"
#include "faiss/IndexFlat.h"

template <typename Seq>
auto faiss_kmeans(const Seq& data, size_t d, uint32_t k, bool is_metric) {
  size_t n = data.size();

  // Flatten input into raw float array for FAISS
  auto flat_data = parlay::map(parlay::flatten(data), [](auto x) { return static_cast<float>(x); });

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
  parlay::sequence<parlay::sequence<float>> centroids(k, parlay::sequence<float>(d));
  parlay::parallel_for(0, k, [&](size_t i) {
    std::memcpy(centroids[i].begin(), centroids_ptr + i * d, d*sizeof(float)); // TODO: optimize
  });
  return centroids;
}

template <typename Seq>
auto faiss_kmeans_assign(const Seq& data, size_t d, uint32_t k, bool is_metric) {
  size_t n = data.size();

  // Flatten input into raw float array for FAISS
  auto flat_data = parlay::map(parlay::flatten(data), [](auto x) { return static_cast<float>(x); });

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

    // index.reset();
    // index.add(k, clus.centroids.data());
    index.search(n, flat_data.data(), 1, distances.data(), assignments.data());
  } else{
    // Index used to assign points during clustering (cosine distance)
    faiss::IndexFlatIP index(d);
    clus.train(n, flat_data.data(), index);

    // index.reset();
    // index.add(k, clus.centroids.data());
    index.search(n, flat_data.data(), 1, distances.data(), assignments.data());
  }
  for (size_t i=0; i<10; i++){
    std::cout << "(" << distances[i] << ", " << assignments[i] << ") ";
  }
  std::cout << std::endl;

  // Extract centroids
  float* centroids_ptr = clus.centroids.data();
  parlay::sequence<parlay::sequence<float>> centroids(k, parlay::sequence<float>(d));
  parlay::parallel_for(0, k, [&](size_t i) {
    std::memcpy(centroids[i].begin(), centroids_ptr + i * d, d*sizeof(float)); // TODO: optimize
  });

  return std::make_pair(centroids, assignments);
}

template <typename Seq>
auto faiss_wgh_kmeans(const Seq& data, size_t d, uint32_t k, 
    parlay::sequence<float>& wghs, bool is_metric) {
  size_t n = data.size();

  // Flatten input into raw float array for FAISS
  auto flat_data = parlay::map(parlay::flatten(data), [](auto x) { return static_cast<float>(x); });

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
  parlay::sequence<parlay::sequence<float>> centroids(k, parlay::sequence<float>(d));
  parlay::parallel_for(0, k, [&](size_t i) {
    std::memcpy(centroids[i].begin(), centroids_ptr + i * d, d*sizeof(float)); // TODO: optimize
  });

  return centroids;
}

template <typename Seq>
auto faiss_kmeans_cost(const Seq& data, size_t d, uint32_t k, 
    parlay::sequence<float>& wghs={}, bool is_metric=false) {
  size_t n = data.size();

  if (wghs.size() == 0) {
    wghs = parlay::sequence<float>::from_function(n, [&](size_t i) { return 1; });
  }

  // Flatten input into raw float array for FAISS
  auto flat_data = parlay::map(parlay::flatten(data), [](auto x) { return static_cast<float>(x); });

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