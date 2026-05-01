#pragma once
#include <string>

namespace mvsic {

struct MVClusteringConfig {
  uint32_t niters = 5;  // Number of Outer Lloyd's Iterations
  uint32_t max_point_clouds_per_cluster = 0;
  // to limit size of dataset, the training set is subsampled
  uint32_t max_points_per_centroid_inner_kmeans = 20;
  uint32_t verbose = 0;  // Print debug statements: 0-nothing, 1-basic verbose, 2-computes cost,
                         //                         3-inner kmeans verbose
  std::string init = "Random";             // Seeding Algorithm
  uint32_t seed = 0;                       // Seed for randomized methods
  bool use_weighted_inner_kmeans = false;  // Weight the points for inner kmeans calls

  // Mirror of IndexParams::build_with_8btq, propagated by each MVIVF*::build()
  // before the first MVClustering call. When true, the inner Lloyd's k-means
  // (kmeans_subsample / kmeans_weighted_subsample) runs assignment through
  // the int8 TurboQuant VPDPBUSD kernel; centroid update stays float.
  bool build_with_8btq = false;

  MVClusteringConfig(uint32_t niters = 5, uint32_t max_point_clouds_per_cluster = 0,
                     uint32_t max_points_per_centroid_inner_kmeans = 20, uint32_t verbose = 0,
                     std::string init = "Random", uint32_t seed = 0,
                     bool use_weighted_inner_kmeans = false) :
      niters(niters),
      max_point_clouds_per_cluster(max_point_clouds_per_cluster),
      max_points_per_centroid_inner_kmeans(max_points_per_centroid_inner_kmeans),
      verbose(verbose),
      init(init),
      seed(seed),
      use_weighted_inner_kmeans(use_weighted_inner_kmeans) {}
};

}  // namespace mvsic