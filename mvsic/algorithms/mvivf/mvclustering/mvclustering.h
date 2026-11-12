#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "mvsic/core/data.h"
#include "mvsic/core/types.h"
#include "mvsic/algorithms/mvivf/mvclustering/mvclustering_config.h"

namespace mvsic {

struct ClusteringIterationStats {
  double centroid_update_seconds = 0;
  double assignment_seconds = 0;
  double float_cost = 0;
};
struct ClusteringResult {
  PointCloudSetView centers;
  // Center positions, not source document IDs; includes every input row,
  // even if training subsampled the corpus.
  std::vector<uint32_t> assignments;
  std::vector<ClusteringIterationStats> iterations;
  ClusteringQuantizer assignment_quantizer = ClusteringQuantizer::None;
  std::string assignment_implementation;
};

// One compiled entry point. Quantization changes assignment scores only;
// float centroid means/weights, IP normalization, initialization, and training
// subsampling retain the original implementation's semantics.
ClusteringResult cluster_point_clouds(const PointCloudSetView& points, uint32_t k, uint32_t s,
                                      Metric metric, const MVClusteringConfig& config = {});

// Compiled assignment operation, also useful for testing/tuning codec choice
// independently of centroid updates. Returns positions in centers.
std::vector<uint32_t> assign_point_clouds(const PointCloudSetView& points,
                                          const PointCloudSetView& centers, Metric metric,
                                          ClusteringQuantizer quantizer, uint32_t seed = 0);

}  // namespace mvsic
