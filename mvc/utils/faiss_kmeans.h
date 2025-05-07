#pragma once

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