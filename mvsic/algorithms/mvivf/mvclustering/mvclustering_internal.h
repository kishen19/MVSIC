#pragma once

#include <memory>
#include <span>
#include "mvsic/algorithms/mvivf/mvclustering/mvclustering.h"

namespace mvsic {

void validate_clustering_config(const MVClusteringConfig& config);

class ClusteringKernels {
 public:
  virtual ~ClusteringKernels() = default;
  virtual ClusteringResult cluster(const PointCloudSetView& local,
                                   std::span<const uint32_t> root_rows, uint32_t k,
                                   uint32_t s) const = 0;
  virtual std::vector<uint32_t> assign(const PointCloudSetView& points,
                                       const PointCloudSetView& centers) const = 0;
};

std::unique_ptr<ClusteringKernels> make_clustering_kernels(const PointCloudSetView& root,
                                                           Metric metric,
                                                           const MVClusteringConfig& config,
                                                           bool prepare_root);

// Build-lifetime cache. A local row mapping refers to positions in root, never
// public document IDs. Prepared encodings are immutable; calls own their scratch.
class ClusteringContext {
 public:
  ClusteringContext(const PointCloudSetView& root, Metric metric, const MVClusteringConfig& config);
  ~ClusteringContext();
  ClusteringResult cluster(const PointCloudSetView& local, std::span<const uint32_t> root_rows,
                           uint32_t k, uint32_t s) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mvsic
