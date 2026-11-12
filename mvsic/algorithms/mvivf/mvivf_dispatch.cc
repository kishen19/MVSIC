#include "mvsic/algorithms/mvivf/mvivf_internal.h"
#include "mvsic/algorithms/mvivf/mvclustering/mvclustering_internal.h"
#include "mvsic/core/kernels/capabilities.h"

#include <cstdlib>
#include <stdexcept>

namespace mvsic {
namespace {
kernels::CpuIsa selected_isa() {
  using kernels::CpuIsa;
  const auto caps = kernels::cpu_capabilities();
  if (!caps.x86_64_v3)
    throw std::invalid_argument(
        "This MVIVF build requires an x86-64-v3 CPU and enabled AVX OS state");
  const char* env = std::getenv("MVSIC_CPU_ISA");
  const auto requested = env ? kernels::parse_isa(env) : CpuIsa::Auto;
  constexpr CpuIsa supported[] = {CpuIsa::AVX2, CpuIsa::AVX512, CpuIsa::AVX512VNNI};
  return kernels::select_isa(requested, supported, caps);
}
}  // namespace

std::unique_ptr<ClusteringKernels> make_clustering_kernels_avx2(const PointCloudSetView&, Metric,
                                                                const MVClusteringConfig&, bool);
std::unique_ptr<ClusteringKernels> make_clustering_kernels_avx512(const PointCloudSetView&, Metric,
                                                                  const MVClusteringConfig&, bool);
std::unique_ptr<ClusteringKernels> make_clustering_kernels_vnni(const PointCloudSetView&, Metric,
                                                                const MVClusteringConfig&, bool);

std::unique_ptr<ClusteringKernels> make_clustering_kernels(const PointCloudSetView& root,
                                                           Metric metric,
                                                           const MVClusteringConfig& config,
                                                           bool prepare_root) {
  switch (selected_isa()) {
    case kernels::CpuIsa::AVX512VNNI:
      return make_clustering_kernels_vnni(root, metric, config, prepare_root);
    case kernels::CpuIsa::AVX512:
      return make_clustering_kernels_avx512(root, metric, config, prepare_root);
    default: return make_clustering_kernels_avx2(root, metric, config, prepare_root);
  }
}

namespace mvivf_internal {
std::unique_ptr<Kernels> make_kernels_avx2(const IndexParams&, const PointCloudSetView&);
std::unique_ptr<Kernels> make_kernels_avx512(const IndexParams&, const PointCloudSetView&);
std::unique_ptr<Kernels> make_kernels_vnni(const IndexParams&, const PointCloudSetView&);

std::unique_ptr<Kernels> make_kernels(const IndexParams& params,
                                      const PointCloudSetView& points) {
  switch (selected_isa()) {
    case kernels::CpuIsa::AVX512VNNI: return make_kernels_vnni(params, points);
    case kernels::CpuIsa::AVX512: return make_kernels_avx512(params, points);
    default: return make_kernels_avx2(params, points);
  }
}
}  // namespace mvivf_internal
}  // namespace mvsic
