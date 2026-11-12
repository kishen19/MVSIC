#pragma once

#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "mvsic/algorithms/mvivf/mvivf.h"

namespace mvsic::mvivf_internal {

// Scratch pairs have implicit lifetime and no owning state. Kernels assign
// both scalar fields before readers inspect them; resizing/destruction is safe.
static_assert(std::is_trivially_copy_constructible_v<Neighbor> &&
              std::is_trivially_destructible_v<Neighbor>);

// Validate once where floats enter the prepared index/query representation.
// Bound every vector for the largest uint32 cloud: scoring accumulates query
// minima in float before averaging, including a large corpus norm for zero queries.
template<bool Unrolled>
inline void validate_values_impl(const float* values, size_t vectors, uint32_t d,
                                 bool require_unit, const char* kind) {
  constexpr double max_squared_norm = double(std::numeric_limits<float>::max()) /
                                       (16.0 * std::numeric_limits<uint32_t>::max());
  const double error_factor = 4.0 * d * std::numeric_limits<double>::epsilon();
  for (size_t i = 0; i < vectors; ++i) {
    const float* row = values + i * d;
    double norm = 0;
    if constexpr (Unrolled) {
      std::array<double, 8> partial{};
      size_t j = 0;
      for (; j + partial.size() <= d; j += partial.size())
        for (size_t lane = 0; lane < partial.size(); ++lane) {
          const double value = row[j + lane];
          partial[lane] += value * value;
        }
      for (double value : partial) norm += value;
      for (; j < d; ++j) norm += double(row[j]) * row[j];
      // Independent double sums permit SIMD. Near a decision boundary, restore
      // the original order: this positive-sum error bound covers reassociation
      // for d<=65535, while float squared products are exact in double.
      const double margin = error_factor * norm + std::numeric_limits<double>::epsilon();
      if (std::abs(norm - max_squared_norm) <= margin ||
          (require_unit && std::abs(std::abs(norm - 1) - 0.002) <= margin)) {
        norm = 0;
        for (size_t k = 0; k < d; ++k) norm += double(row[k]) * row[k];
      }
    } else {
      for (size_t j = 0; j < d; ++j) norm += double(row[j]) * row[j];
    }
    if (!std::isfinite(norm)) throw std::invalid_argument(std::string(kind) + " values must be finite");
    if (norm > max_squared_norm)
      throw std::invalid_argument(std::string(kind) + " squared norms exceed the float scoring range");
    if (require_unit && std::abs(norm - 1) > 0.002)
      throw std::invalid_argument(std::string("OneBitTQ serving requires unit-norm ") + kind +
                                  " vectors; use float, TQ4 or TQ8 for general norms");
  }
}

inline void validate_values(const float* values, size_t vectors, uint32_t d,
                            bool require_unit, const char* kind) {
  if (d >= 128) validate_values_impl<true>(values, vectors, d, require_unit, kind);
  else validate_values_impl<false>(values, vectors, d, require_unit, kind);
}

/* Topology and scheduling are independent of encoded storage. */
struct Node {
  std::vector<uint32_t> children;
  std::vector<uint32_t> rows;
  PointCloudSetView centers;
  bool is_leaf() const noexcept { return children.empty(); }
};

struct EncodedNode {
  std::shared_ptr<const void> centers;
  std::shared_ptr<const void> leaf;
  PointCloudSetView raw_leaf;
  size_t center_bytes = 0;
  size_t leaf_bytes = 0;
};

struct PreparedQuery {
  virtual ~PreparedQuery() = default;
  PointCloud effective;
  PointCloud rerank;
  double compression_seconds = 0;
  double quantization_seconds = 0;
};

/* Dispatch occurs per query preparation or whole center/leaf block. Numerical
 * loops remain in private IP kernels; there is no callback per dot.
 */
class Kernels {
 public:
  virtual ~Kernels() = default;
  virtual void prepare_nodes(const std::vector<Node>& nodes,
                             std::vector<EncodedNode>& encoded) const = 0;
  virtual std::unique_ptr<PreparedQuery> prepare_query(const PointCloud& query,
                                                     const SearchParams& params) const = 0;
  virtual void center_distances(const Node& node, const EncodedNode& encoded,
                                const PreparedQuery& query, Neighbor* output) const = 0;
  virtual bool root_distances(const EncodedNode& root,
      std::span<const std::unique_ptr<PreparedQuery>> queries, Neighbor* output) const = 0;
  virtual void leaf_distances(const EncodedNode& leaf, const PreparedQuery& query,
                              Neighbor* output) const = 0;
  virtual void leaf_topk(const EncodedNode& leaf,
      std::span<const PreparedQuery* const> queries, size_t k, Neighbor* output) const = 0;
  // Called once at a request boundary, before entering parallel query work.
  virtual void prepare_rerank() const = 0;
  virtual size_t rerank(const PreparedQuery& query, std::span<Neighbor> candidates,
                        bool tq8) const = 0;
};

std::unique_ptr<Kernels> make_kernels(const IndexParams& params,
                                      const PointCloudSetView& points);

struct SearchStats {
  size_t bytes = 0;
  size_t search_cmps = 0;
  size_t probe_cmps = 0;
  std::array<double, 4> greedy{};
  double leaf_seconds = 0;
  double selection_seconds = 0;
  double rerank_seconds = 0;
  double greedy_seconds = 0;
  double compression_seconds = 0;
  double quantization_seconds = 0;
};

struct Probe {
  float distance;
  uint32_t node;
};

inline bool neighbor_less(const Neighbor& a, const Neighbor& b) {
  return a.second != b.second ? a.second < b.second : a.first < b.first;
}
template<class Range>
inline void select_neighbors(Range& candidates, size_t count) {
  if (count < candidates.size()) {
    std::nth_element(candidates.begin(), candidates.begin() + count, candidates.end(), neighbor_less);
    candidates.resize(count);
  }
  std::sort(candidates.begin(), candidates.end(), neighbor_less);
}

}  // namespace mvsic::mvivf_internal

namespace mvsic {

struct IndexMVIVF::Impl {
  uint32_t d;
  IndexParams params;
  PointCloudSetView points;
  std::vector<mvivf_internal::Node> nodes;
  std::vector<uint32_t> leaves;
  std::vector<mvivf_internal::EncodedNode> encoded;
  std::unique_ptr<mvivf_internal::Kernels> kernels;
  size_t height = 0;

  Impl(uint32_t d, const IndexParams& params);
  void validate_search_params(const SearchParams& search_params) const;
  void validate_query(const PointCloud& query) const;
  void validate_points(const PointCloudSetView& points) const;
  void validate_corpus(const PointCloudSetView& points) const;
  void prepare();
  std::vector<mvivf_internal::Probe> find_probes(const mvivf_internal::PreparedQuery& query,
      size_t nprobes, const Neighbor* root_distances, mvivf_internal::SearchStats* stats) const;
  Neighbors finish(const mvivf_internal::PreparedQuery& query, std::span<Neighbor> candidates,
                   const SearchParams& params, mvivf_internal::SearchStats* stats) const;
  Neighbors search(const PointCloud& query, const SearchParams& search_params,
                   mvivf_internal::SearchStats* stats = nullptr) const;
};

}  // namespace mvsic
