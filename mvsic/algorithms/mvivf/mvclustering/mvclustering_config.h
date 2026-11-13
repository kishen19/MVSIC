#pragma once
#include <cstdint>
#include <string>
#include "mvsic/core/types.h"

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

  // Compatibility alias for TQ8 assignment in the compiled facade, for both
  // outer cloud and inner vector Lloyd steps. Byte kernels use the selected
  // CPU backend; centroid updates stay float.
  bool build_with_8btq = false;

  // Assignment codec for outer multi-vector and inner vector Lloyd steps.
  // Float centroid updates remain unchanged. In the compiled facade the
  // old build_with_8btq flag aliases TQ8 when this field is None.
  ClusteringQuantizer assignment_quantizer = ClusteringQuantizer::None;

  // Experimental inner training choices. Inner/outer scores use IP/MaxSim;
  // defaults retain the original spherical updates, seeding and backend.
  std::string inner_init = "default";    // default, uniform, kmeans++, prefix_doubling
  std::string inner_backend = "legacy";  // legacy, adaptive (float tiles/means)

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
