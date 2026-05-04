#pragma once

// =============================================================================
// IndexFlat — brute-force Chamfer search over a PointCloudSet (flat index).
//
// Same template axes as MVIVF flat / Vamana: <metric, LeafModel>. NoQuantizer
// uses OneToMany::TopK; leaf quantization packs the DB once via LeafModel::encode
// and scores with EncodedSet::distances_all + partial sort to cap_k.
// =============================================================================

#include <algorithm>
#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "mvsic/core/distance_measures/one_to_many.h"
#include "mvsic/core/index.h"
#include "mvsic/core/query_compression.h"
#include "mvsic/core/utils/util.h"

namespace mvsic {

namespace flat_internal {
struct Empty {};
}  // namespace flat_internal

template<bool metric, class LeafModel = NoQuantizer<metric>>
class IndexFlat : public Index<metric> {
 public:
  using ChPoint = typename Index<metric>::ChPoint;
  using QT = typename Index<metric>::QT;
  using Index<metric>::d;
  using Index<metric>::params;
  using Index<metric>::quantization_mode;

  using LeafSet = typename LeafModel::EncodedSet;
  using LeafQuery = typename LeafModel::EncodedQuery;
  using LeafParams = typename LeafModel::Params;

  static constexpr bool kHasLeafQuant = !std::is_same_v<LeafModel, NoQuantizer<metric>>;

  [[no_unique_address]]
  std::conditional_t<kHasLeafQuant, LeafModel, flat_internal::Empty> leaf_model_;
  [[no_unique_address]]
  std::conditional_t<kHasLeafQuant, LeafParams, flat_internal::Empty> leaf_params_;
  [[no_unique_address]]
  std::conditional_t<kHasLeafQuant, LeafSet, flat_internal::Empty> encoded_db_;

  size_t built_n_ = 0;

  IndexFlat(uint32_t d_) noexcept {
    d = d_;
    params = IndexParams::flat();
    set_quant_mode_();
  }

  IndexFlat(uint32_t d_, const IndexParams& p) noexcept {
    d = d_;
    params = p;
    set_quant_mode_();
  }

  template<class LP = LeafParams,
           std::enable_if_t<kHasLeafQuant && std::is_same_v<LP, LeafParams>, int> = 0>
  IndexFlat(uint32_t d_, const IndexParams& p, const LP& lp) noexcept {
    d = d_;
    params = p;
    if constexpr (kHasLeafQuant) leaf_params_ = lp;
    set_quant_mode_();
  }

  void set_quant_mode_() {
    if constexpr (!kHasLeafQuant) {
      quantization_mode = QT::None;
    } else {
      using L = LeafModel;
      if constexpr (std::is_same_v<L, pq_mv::Model<metric>>) quantization_mode = QT::PQ;
      else if constexpr (std::is_same_v<L, rabitq_mv::Model<metric>>)
        quantization_mode = QT::RaBitQ;
      else if constexpr (std::is_same_v<L, fastscan_mv::Model<metric>>)
        quantization_mode = QT::FastScan;
      else if constexpr (std::is_same_v<L, turboquant_mv::Model<metric>>)
        quantization_mode = QT::TurboQuant;
      else if constexpr (std::is_same_v<L, pqtq_mv::Model<metric>>)
        quantization_mode = QT::SPQTQ;
      else if constexpr (std::is_same_v<L, turboquant_1bit_mv::Model<metric>>)
        quantization_mode = QT::OneBitTQ;
      else if constexpr (std::is_same_v<L, turboquant_8bit_mv::Model<metric>>)
        quantization_mode = QT::EightBitTQ;
      else quantization_mode = QT::None;
    }
  }

  void build(const PointCloudSet<ChPoint>& points) override {
    built_n_ = points.size();
    if constexpr (kHasLeafQuant) {
      leaf_model_.train(points, leaf_params_);
      encoded_db_ = leaf_model_.encode(points);
    }
  }

  std::tuple<parlay::sequence<std::pair<uint32_t, float>>, size_t, std::vector<double>>
  search_with_stats(const ChPoint& query, const PointCloudSet<ChPoint>& points,
                    const SearchParams& search_params) override {
    parlay::internal::timer t;
    double t_compress = 0.0;
    CompressedPointCloud<ChPoint> compressed_storage;
    ChPoint effective_query = query;
    if (search_params.query_compression != SearchParams::QueryCompression::None) {
      uint32_t ba = 1;
      if constexpr (kHasLeafQuant) ba = LeafModel::kBatchAlignment;
      t.start();
      compressed_storage =
          compress_query<ChPoint>(query, search_params.query_compression,
                                  search_params.query_compression_threshold, ba,
                                  search_params.query_alignment,
                                  search_params.query_alignment_strict);
      effective_query = compressed_storage.view();
      t_compress = t.stop();
      t.reset();
    }
    return run_flat_search_impl_(effective_query, &query, points, search_params, t_compress);
  }

  std::tuple<parlay::sequence<std::pair<uint32_t, float>>, size_t, std::vector<double>>
  search_with_stats_compressed(const ChPoint& effective_query,
                               const PointCloudSet<ChPoint>& points,
                               const SearchParams& search_params) {
    return run_flat_search_impl_(effective_query, nullptr, points, search_params, 0.0);
  }

  std::pair<parlay::sequence<std::pair<uint32_t, float>>, size_t> search(
      const ChPoint& query, const PointCloudSet<ChPoint>& points,
      const SearchParams& search_params) override {
    auto [results, bytes, stats] = search_with_stats(query, points, search_params);
    (void)stats;
    return std::make_pair(std::move(results), bytes);
  }

  // Flat index does not persist to disk. Callers may still invoke save/load through the
  // Index API; save is a no-op. load(...) ignores `filename` and refreshes in-memory state
  // from `points`; with leaf quantization this retrains and re-encodes the database.
  void save(const std::string& filename) override { (void)filename; }

  void load(const std::string& filename, const PointCloudSet<ChPoint>& points) override {
    (void)filename;
    d = points.get_dims();
    built_n_ = points.size();
    if constexpr (kHasLeafQuant) {
      leaf_model_.train(points, leaf_params_);
      encoded_db_ = leaf_model_.encode(points);
    }
  }

 private:
  std::tuple<parlay::sequence<std::pair<uint32_t, float>>, size_t, std::vector<double>>
  run_flat_search_impl_(const ChPoint& effective_query, const ChPoint* raw_query_for_rerank,
                        const PointCloudSet<ChPoint>& points,
                        const SearchParams& search_params, double t_compress) {
    parlay::internal::timer t;
    const size_t n = points.size();
    const size_t k = search_params.k;
    size_t bytes_accessed = 0;
    size_t dist_cmps = n;

    double t_search = 0.0;
    double t_distances = 0.0;
    double t_rest = 0.0;
    double t_rerank = 0.0;

    size_t cap_k = search_params.k;
    const bool do_rerank = search_params.num_rerank > 0 && !search_params.norerank;
    if (do_rerank) {
      cap_k = std::max(cap_k, search_params.num_rerank);
    }
    cap_k = std::min(cap_k, n);

    parlay::sequence<std::pair<uint32_t, float>> visited;

    t.start();
    if constexpr (!kHasLeafQuant) {
      bytes_accessed += points.num_bytes();
      visited = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(cap_k);
      OneToMany<ChPoint, PointCloudSet<ChPoint>>::TopKIntoUninitialized(
          effective_query, points, static_cast<uint32_t>(cap_k), visited.data());
    } else {
      LeafQuery q_leaf = leaf_model_.quantize_query(effective_query);
      bytes_accessed += encoded_db_.num_bytes();
      visited = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(n);
      encoded_db_.distances_all(q_leaf, visited.data());
    }
    t_distances = t.stop();
    t.reset();

    t.start();
    if constexpr (kHasLeafQuant) {
      sort_inplace_kv(visited);
      visited.resize(std::min(cap_k, visited.size()));
    }
    t_rest = t.stop();
    t.reset();

    const ChPoint& rerank_query = [&]() -> const ChPoint& {
      if (search_params.compress_rerank) return effective_query;
      if (raw_query_for_rerank != nullptr) return *raw_query_for_rerank;
      return effective_query;
    }();

    auto final_results =
        parlay::sequence<std::pair<uint32_t, float>>::uninitialized(std::min(k, visited.size()));

    t.start();
    if (do_rerank && search_params.num_rerank > 0) {
      size_t nr = std::min(search_params.num_rerank, visited.size());
      if (search_params.tq8_rerank) {
        bytes_accessed += this->rerank_tq8_(rerank_query, points, visited, nr, final_results);
      } else {
        bytes_accessed += this->rerank(rerank_query, points, visited, nr, final_results);
      }
    } else {
      parlay::parallel_for(0, final_results.size(),
                           [&](size_t i) { final_results[i] = visited[i]; });
    }
    t_rerank = t.stop();

    std::vector<double> stats;
    stats.push_back(static_cast<double>(points.size()));
    stats.push_back(static_cast<double>(dist_cmps));
    stats.push_back(t_compress);
    stats.push_back(t_search);
    stats.push_back(t_distances);
    stats.push_back(t_rest);
    stats.push_back(t_rerank);

    return std::make_tuple(std::move(final_results), bytes_accessed, std::move(stats));
  }
};

}  // namespace mvsic
