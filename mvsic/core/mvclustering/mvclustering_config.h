#pragma once

namespace mvsic {

struct MVClusteringConfig {
  uint32_t niters = 5;  // Number of Outer Lloyd's Iterations
  // to limit size of dataset, the training set is subsampled
  uint32_t max_points_per_centroid_inner_kmeans = 20;
  uint32_t verbose = 0;   // Print debug statements: 0-nothing, 1-basic verbose, 2-computes cost,
                          //                         3-inner kmeans verbose
  char *init = "Random";  // Seeding Algorithm
  uint32_t seed = 0;      // Seed for randomized methods
  bool use_weighted_inner_kmeans = false;  // Weight the points for inner kmeans calls

  MVClusteringConfig(uint32_t niters = 5, uint32_t max_points_per_centroid_inner_kmeans = 20,
                     uint32_t verbose = 0, char *init = "Random", uint32_t seed = 0,
                     bool use_weighted_inner_kmeans = false) noexcept :
      niters(niters),
      max_points_per_centroid_inner_kmeans(max_points_per_centroid_inner_kmeans),
      verbose(verbose),
      init(init),
      seed(seed),
      use_weighted_inner_kmeans(use_weighted_inner_kmeans) {}
};

}  // namespace mvsic