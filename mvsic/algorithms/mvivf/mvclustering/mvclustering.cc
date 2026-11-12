#include "mvsic/algorithms/mvivf/mvclustering/mvclustering_internal.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace mvsic {
namespace {
void ValidateValues(const PointCloudSetView& points) {
  // Scores sum up to UINT32_MAX vector distances before taking their mean.
  // Bound both operands uniformly so even a long zero query against a large
  // center remains safe without wider accumulators in scoring kernels.
  constexpr double max_norm =
      double(std::numeric_limits<float>::max()) / (16.0 * std::numeric_limits<uint32_t>::max());
  for (size_t i = 0; i < points.total_size(); ++i) {
    double norm = 0;
    for (size_t j = 0; j < points.d; ++j) {
      const double value = points.values[i * points.d + j];
      norm += value * value;
    }
    if (!(norm <= max_norm))
      throw std::invalid_argument("Clustering vector norms must be finite and fit float scoring");
  }
}
void ValidateMetric(Metric metric) {
  if (metric != Metric::IP && metric != Metric::L2) throw std::invalid_argument("Unknown metric");
}
void ValidateQuantizer(ClusteringQuantizer quantizer) {
  switch (quantizer) {
    case ClusteringQuantizer::None:
    case ClusteringQuantizer::TQ8:
    case ClusteringQuantizer::TQ4:
    case ClusteringQuantizer::TQ1: return;
  }
  throw std::invalid_argument("Unknown clustering quantizer");
}
}  // namespace

void validate_clustering_config(const MVClusteringConfig& config) {
  ValidateQuantizer(config.assignment_quantizer);
  if (config.init != "Random") throw std::invalid_argument("Supported initialization is Random");
  if (!config.max_points_per_centroid_inner_kmeans)
    throw std::invalid_argument("Clustering requires a positive inner sample limit");
  if (config.build_with_8btq && config.assignment_quantizer != ClusteringQuantizer::None &&
      config.assignment_quantizer != ClusteringQuantizer::TQ8)
    throw std::invalid_argument("build_with_8btq conflicts with assignment_quantizer");
  if (config.inner_metric != "same" && config.inner_metric != "l2" && config.inner_metric != "ip")
    throw std::invalid_argument("inner_metric must be same, l2 or ip");
  if (config.inner_init != "default" && config.inner_init != "uniform" &&
      config.inner_init != "kmeans++" && config.inner_init != "prefix_doubling")
    throw std::invalid_argument("inner_init must be default, uniform, kmeans++ or prefix_doubling");
  if (config.inner_backend != "legacy" && config.inner_backend != "adaptive")
    throw std::invalid_argument("inner_backend must be legacy or adaptive");
}

struct ClusteringContext::Impl {
  std::unique_ptr<ClusteringKernels> kernels;
};

ClusteringContext::ClusteringContext(const PointCloudSetView& root, Metric metric,
                                     const MVClusteringConfig& config) :
    impl_(std::make_unique<Impl>()) {
  // Index build has already checked root values and configuration. Selection
  // and preparation happen once; recursive calls reuse immutable encodings.
  impl_->kernels = make_clustering_kernels(root, metric, config, true);
}
ClusteringContext::~ClusteringContext() = default;
ClusteringResult ClusteringContext::cluster(const PointCloudSetView& local,
                                            std::span<const uint32_t> root_rows, uint32_t k,
                                            uint32_t s) const {
  return impl_->kernels->cluster(local, root_rows, k, s);
}

ClusteringResult cluster_point_clouds(const PointCloudSetView& points, uint32_t k, uint32_t s,
                                      Metric metric, const MVClusteringConfig& config) {
  points.validate();
  ValidateValues(points);
  ValidateMetric(metric);
  validate_clustering_config(config);
  if ((config.assignment_quantizer != ClusteringQuantizer::None || config.build_with_8btq) &&
      points.d > 65535)
    throw std::invalid_argument("Quantized clustering supports at most 65535 dimensions");
  if (k == 0 || k > points.size()) throw std::invalid_argument("Clustering requires 1 <= k <= n");
  auto kernels = make_clustering_kernels(points, metric, config, false);
  return kernels->cluster(points, {}, k, s);
}

std::vector<uint32_t> assign_point_clouds(const PointCloudSetView& points,
                                          const PointCloudSetView& centers, Metric metric,
                                          ClusteringQuantizer quantizer, uint32_t seed) {
  points.validate();
  centers.validate();
  ValidateMetric(metric);
  ValidateValues(points);
  ValidateValues(centers);
  ValidateQuantizer(quantizer);
  if (quantizer != ClusteringQuantizer::None && points.d > 65535)
    throw std::invalid_argument("Quantized clustering supports at most 65535 dimensions");
  if (points.d != centers.d) throw std::invalid_argument("Assignment dimension mismatch");
  MVClusteringConfig config;
  config.assignment_quantizer = quantizer;
  config.seed = seed;
  auto kernels = make_clustering_kernels(points, metric, config, false);
  return kernels->assign(points, centers);
}
}  // namespace mvsic
