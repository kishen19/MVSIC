#pragma once

#include <vector>
#include <tuple>
#include <variant>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <type_traits>
#include "parlay/primitives.h"

// Data Types and Kernels
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/core/distance_measures/one_to_many.h"
#include "algorithms/utils/euclidian_point.h"
#include "algorithms/utils/mips_point.h"
#include "algorithms/utils/point_range.h"
#include "algorithms/utils/beamSearch.h"

// Quantization Headers
#include "mvsic/core/quantization/pq.h"
#include "mvsic/core/quantization/pq_mv.h"
#include "mvsic/core/quantization/rabitq.h"
#include "mvsic/core/quantization/rabitq_mv.h"
#include "mvsic/core/quantization/fastscan.h"
#include "mvsic/core/quantization/fastscan_mv.h"
#include "mvsic/core/quantization/turboquant.h"
#include "mvsic/core/quantization/turboquant_mv.h"
#include "mvsic/core/quantization/pqtq.h"
#include "mvsic/core/quantization/pqtq_mv.h"
#include "mvsic/core/quantization/turboquant_1bit_mv.h"
#include "mvsic/core/quantization/turboquant_8bit_mv.h"

// Params
#include "search_params.h"
#include "index_params.h"

namespace mvsic {

// ---------------------------------------------------------------------------
// NoQuantizer — identity leaf "Model" for the templated index families.
//
// Indices such as `IndexMVIVFT<metric, CompressCenters, LeafModel>` template on
// a leaf Model. When `LeafModel = NoQuantizer<metric>`, all encode / quantize /
// distance calls are `if constexpr`-gated out and the code falls through to
// the raw PointCloudSet ChamferIP/L2 kernels. Concrete aliases such as
// `IndexMVIVFIP = IndexMVIVFT<false, false, NoQuantizer<false>>` therefore
// compile identically to a hand-written non-quantized class.
// ---------------------------------------------------------------------------
namespace no_quantizer_mv {

template<bool Metric>
struct EncodedPointCloudSet {};

template<bool Metric>
struct EncodedQueryPointCloud {};

}  // namespace no_quantizer_mv

template<bool Metric>
struct NoQuantizer {
  using ChPoint = std::conditional_t<Metric, ChamferL2_Point, ChamferIP_Point>;
  using EncodedSet = no_quantizer_mv::EncodedPointCloudSet<Metric>;
  using EncodedQuery = no_quantizer_mv::EncodedQueryPointCloud<Metric>;

  static constexpr uint32_t kClassId = 0;
  static constexpr uint32_t kBatchAlignment = 1;
  static constexpr const char* kName = "none";

  struct Params {};

  template<typename PCSet>
  void train(const PCSet& /*pcs*/, const Params& /*p*/ = {}) {}

  template<typename PCSet>
  EncodedSet encode(const PCSet& /*pcs*/) const { return {}; }

  EncodedQuery quantize_query(const ChPoint& /*q*/) const { return {}; }

  void save(std::ostream& /*out*/) const {}
  void load(std::istream& /*in*/) {}
};

// ---------------------------------------------------------------------------
// quantizer_method_of_v<M, Metric>
//
// Compile-time mapping from a concrete quantizer Model type (MV or SV) to the
// corresponding IndexParams::QuantizerType enum value.  Used by the graph-
// based index families (MUVERA, MPool, Vamana, SVH_Graph, SVH_IVF) during
// phase-4 staged refactor: the index is templated on a `LeafModel` class for
// type-level separation (so pybind11 sees distinct classes per variant, and
// users can construct `IndexFooPQIP` without touching `IndexParams`), while
// the internal hot path continues to use the battle-tested variant-based
// dispatch through `Index<metric>::train_quantizer` /
// `encode_range_quantized` / `quant_beam_search` etc.
//
// Conversion to full compile-time `if constexpr` specialization (matching the
// MVIVF family refactor) is left to a follow-up phase and does not block the
// binding-level changes this mapping enables.
// ---------------------------------------------------------------------------
namespace detail {
template<class M, bool Metric>
struct quantizer_method_of {
  static constexpr IndexParams::QuantizerType value = IndexParams::QuantizerType::None;
};
template<bool M>
struct quantizer_method_of<pq::Model<M>, M> {
  static constexpr IndexParams::QuantizerType value = IndexParams::QuantizerType::PQ;
};
template<bool M>
struct quantizer_method_of<rabitq::Model<M>, M> {
  static constexpr IndexParams::QuantizerType value = IndexParams::QuantizerType::RaBitQ;
};
template<bool M>
struct quantizer_method_of<fastscan::Model<M>, M> {
  static constexpr IndexParams::QuantizerType value = IndexParams::QuantizerType::FastScan;
};
template<bool M>
struct quantizer_method_of<turboquant::Model<M>, M> {
  static constexpr IndexParams::QuantizerType value = IndexParams::QuantizerType::TurboQuant;
};
template<bool M>
struct quantizer_method_of<pqtq::Model<M>, M> {
  static constexpr IndexParams::QuantizerType value = IndexParams::QuantizerType::SPQTQ;
};
template<bool M>
struct quantizer_method_of<pq_mv::Model<M>, M> {
  static constexpr IndexParams::QuantizerType value = IndexParams::QuantizerType::PQ;
};
template<bool M>
struct quantizer_method_of<rabitq_mv::Model<M>, M> {
  static constexpr IndexParams::QuantizerType value = IndexParams::QuantizerType::RaBitQ;
};
template<bool M>
struct quantizer_method_of<fastscan_mv::Model<M>, M> {
  static constexpr IndexParams::QuantizerType value = IndexParams::QuantizerType::FastScan;
};
template<bool M>
struct quantizer_method_of<turboquant_mv::Model<M>, M> {
  static constexpr IndexParams::QuantizerType value = IndexParams::QuantizerType::TurboQuant;
};
template<bool M>
struct quantizer_method_of<pqtq_mv::Model<M>, M> {
  static constexpr IndexParams::QuantizerType value = IndexParams::QuantizerType::SPQTQ;
};
template<bool M>
struct quantizer_method_of<turboquant_1bit_mv::Model<M>, M> {
  static constexpr IndexParams::QuantizerType value = IndexParams::QuantizerType::OneBitTQ;
};
template<bool M>
struct quantizer_method_of<turboquant_8bit_mv::Model<M>, M> {
  static constexpr IndexParams::QuantizerType value = IndexParams::QuantizerType::EightBitTQ;
};
}  // namespace detail

template<class M, bool Metric>
inline constexpr IndexParams::QuantizerType quantizer_method_of_v =
    detail::quantizer_method_of<M, Metric>::value;

// TODO(phase3-6): the legacy QuantTypes / MVQuantTypes structs below, along
// with `quantization_mode` and the `train_quantizer` / `encode_points_quantized`
// / `quantize_query_point_cloud` / `quant_distances_all` / `quant_beam_search`
// helpers in `Index<metric>`, will be removed once every index family has
// migrated to the templated <CompressCenters, LeafModel> design and no longer
// calls them.
// Quantization Traits
template<bool metric, typename Range>
struct QuantTypes {
  // PointRange Alternative
  using PQ_Range = pq::Quantized_Point_Range<Range, metric>;
  using RQ_Range = rabitq::Quantized_Point_Range<Range, metric>;
  using FS_Range = fastscan::Quantized_Point_Range<Range, metric>;
  using TQ_Range = turboquant::Quantized_Point_Range<Range, metric>;
  using PQTQ_Range = pqtq::Quantized_Point_Range<Range, metric>;
  // Query Vector
  using PQ_Query = pq::Quantized_Query<metric>;
  using RQ_Query = rabitq::Quantized_Query<metric>;
  using FS_Query = fastscan::Quantized_Query<metric>;
  using TQ_Query = turboquant::Quantized_Query<metric>;
  using PQTQ_Query = pqtq::Quantized_Query<metric>;
  // Main Model Object
  using PQ_Model = pq::Model<metric>;
  using RQ_Model = rabitq::Model<metric>;
  using FS_Model = fastscan::Model<metric>;
  using TQ_Model = turboquant::Model<metric>;
  using PQTQ_Model = pqtq::Model<metric>;
  // Unified Objects
  using QuantModel =
      std::variant<std::monostate, PQ_Model, RQ_Model, FS_Model, TQ_Model, PQTQ_Model>;
  using QuantQuery =
      std::variant<std::monostate, PQ_Query, RQ_Query, FS_Query, TQ_Query, PQTQ_Query>;
  using QuantRange =
      std::variant<std::monostate, PQ_Range, RQ_Range, FS_Range, TQ_Range, PQTQ_Range>;
};

template<bool metric, typename ChPoint>
struct MVQuantTypes {
  // PointCloudSet Alternate
  using PQ_Set = pq_mv::Quantized_Point_Cloud_Set<metric>;
  using RQ_Set = rabitq_mv::Quantized_Point_Cloud_Set<metric>;
  using FS_Set = fastscan_mv::Quantized_Point_Cloud_Set<metric>;
  using TQ_Set = turboquant_mv::Quantized_Point_Cloud_Set<metric>;
  using PQTQ_Set = pqtq_mv::Quantized_Point_Cloud_Set<metric>;
  using OBTQ_Set = turboquant_1bit_mv::Quantized_Point_Cloud_Set<metric>;
  using EBTQ_Set = turboquant_8bit_mv::Quantized_Point_Cloud_Set<metric>;
  // ChPoint (Query) Alternate
  using PQ_Query = pq_mv::Quantized_Query_Point_Cloud<metric>;
  using RQ_Query = rabitq_mv::Quantized_Query_Point_Cloud<metric>;
  using FS_Query = fastscan_mv::Quantized_Query_Point_Cloud<metric>;
  using TQ_Query = turboquant_mv::Quantized_Query_Point_Cloud<metric>;
  using PQTQ_Query = pqtq_mv::Quantized_Query_Point_Cloud<metric>;
  using OBTQ_Query = turboquant_1bit_mv::Quantized_Query_Point_Cloud<metric>;
  using EBTQ_Query = turboquant_8bit_mv::Quantized_Query_Point_Cloud<metric>;
  // Main Model Object
  using PQ_Model = pq_mv::Model<metric>;
  using RQ_Model = rabitq_mv::Model<metric>;
  using FS_Model = fastscan_mv::Model<metric>;
  using TQ_Model = turboquant_mv::Model<metric>;
  using PQTQ_Model = pqtq_mv::Model<metric>;
  using OBTQ_Model = turboquant_1bit_mv::Model<metric>;
  using EBTQ_Model = turboquant_8bit_mv::Model<metric>;
  // Unified Objects
  using QuantSet = std::variant<std::monostate, PQ_Set, FS_Set, RQ_Set, TQ_Set, PQTQ_Set, OBTQ_Set,
                                EBTQ_Set>;
  using QuantQuery = std::variant<std::monostate, PQ_Query, FS_Query, RQ_Query, TQ_Query,
                                  PQTQ_Query, OBTQ_Query, EBTQ_Query>;
  using QuantModel = std::variant<std::monostate, PQ_Model, FS_Model, RQ_Model, TQ_Model,
                                  PQTQ_Model, OBTQ_Model, EBTQ_Model>;
};

// Base Index Class
template<bool metric>
class Index {
 public:
  using ChPoint = std::conditional_t<metric, ChamferL2_Point, ChamferIP_Point>;
  using Point =
      std::conditional_t<metric, parlayANN::Euclidian_Point<float>, parlayANN::Mips_Point<float>>;
  using Range = parlayANN::PointRange<Point>;
  using MVQT = MVQuantTypes<metric, ChPoint>;
  using SVQT = QuantTypes<metric, Range>;
  using QT = IndexParams::QuantizerType;

  uint32_t d = 0;      // Embedding dimension
  IndexParams params;  // Index Building specific params
  // Modes: None, PQ, RaBitQ, FastScan, TurboQuant4Bit, TurboQuantPQ4Bit
  QT quantization_mode = QT::None;

  Index() {}
  Index(const IndexParams& params) : params(params) {}

  // Builds the index given PointCloudSet object.
  virtual void build(const PointCloudSet<ChPoint>& points) {}
  // Builds the index given raw data.
  virtual void build(uint32_t n, const float* data, const size_t* offsets, const uint32_t* ids) {
    PointCloudSet<ChPoint> points(n, d, data, offsets, ids);
    build(points);
  }

  // Returns the top-k point clouds for the query point cloud
  // Output format: < [<id, distance>, ...], # distance comparisons>
  virtual std::pair<parlay::sequence<std::pair<uint32_t, float>>, size_t> search(
      const ChPoint& query, const PointCloudSet<ChPoint>& points,
      const SearchParams& search_params) {
    auto [results, dist_cmps, timings] = search_with_stats(query, points, search_params);
    return std::make_pair(results, dist_cmps);
  }

  // Returns the top-k point clouds for each of the query point clouds
  // Default: runs search in parallel for each query
  virtual std::pair<parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>, size_t>
  search_all(const PointCloudSet<ChPoint>& query_points, const PointCloudSet<ChPoint>& points,
             const SearchParams& search_params) {
    auto cmps = parlay::sequence<size_t>::uninitialized(query_points.size());
    auto pred = parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>(query_points.size());
    parlay::parallel_for(0, query_points.size(), [&](size_t i) {
      auto [results, dist_cmps_i] = search(query_points[i], points, search_params);
      pred[i] = results;
      cmps[i] = dist_cmps_i;
    });
    return std::make_pair(pred, parlay::reduce(cmps));
  }

  // Returns some running time stats, specific to the index type
  // NOTE: Has to be defined by every index.
  virtual std::tuple<parlay::sequence<std::pair<uint32_t, float>>, size_t, std::vector<double>>
  search_with_stats(const ChPoint& query, const PointCloudSet<ChPoint>& points,
                    const SearchParams& params) {
    return std::make_tuple(parlay::sequence<std::pair<uint32_t, float>>(), 0,
                           std::vector<double>{});
  }

  // Reranks the given candidates and returns the top-k point clouds
  virtual size_t rerank(const ChPoint& query, const PointCloudSet<ChPoint>& points,
                        const parlay::sequence<std::pair<uint32_t, float>>& candidates,
                        size_t num_rerank,
                        parlay::sequence<std::pair<uint32_t, float>>& out_results) {
    auto cmp_rerank = parlay::sequence<size_t>::uninitialized(num_rerank);
    auto results_rerank =
        parlay::sequence<std::pair<uint32_t, float>>::from_function(num_rerank, [&](size_t i) {
          uint32_t id = candidates[i].first;
          auto [dist, d_c] = query.distance_w_cmps(points[id]);
          cmp_rerank[i] = d_c;
          return std::make_pair(id, dist);
        });
    size_t num_cmps = parlay::reduce(cmp_rerank);
    parlay::sort_inplace(results_rerank,
                         [](const auto& a, const auto& b) { return a.second < b.second; });
    parlay::parallel_for(0, out_results.size(),
                         [&](size_t i) { out_results[i] = results_rerank[i]; });
    return num_cmps;
  }

  virtual size_t rerank_opt(const ChPoint& query, const PointCloudSet<ChPoint>& points,
                            const parlay::sequence<std::pair<uint32_t, float>>& candidates,
                            size_t num_rerank,
                            parlay::sequence<std::pair<uint32_t, float>>& out_results) {
    auto candidate_indices =
        parlay::delayed_tabulate(num_rerank, [&](size_t i) { return candidates[i].first; });
    PointCloudSet<ChPoint> candidates_pcs(points.filter(candidate_indices), points.get_dims());
    size_t k = out_results.size();
    OneToMany<ChPoint, PointCloudSet<ChPoint>>::TopKIntoUninitialized(query, candidates_pcs, k,
                                                                      out_results.data());
    size_t num_cmps = (query.size() + candidates_pcs.total_size()) * points.get_dims();
    return num_cmps;
  }

  // ---------------------------------------------------------------------------
  // 8BTQ-encoded rerank, available to any subclass.
  //
  // Lazily trains the 8-bit TurboQuant rotator on the input points and pre-
  // encodes the full DB the first time `rerank_tq8_` is called; subsequent
  // queries reuse the cached model + DB.  Rerank scores each candidate by a
  // single `turboquant_8bit_mv_chamfer_distance(q8, db, c)` instead of an
  // exact float chamfer over the raw embeddings.
  //
  // Candidate ids in `candidates` are positions in the input PointCloudSet
  // (`points.get_id(i) == i` in the bench loaders), which is also the cloud
  // index in `tq8_rerank_db_`, so we can index it directly.
  // ---------------------------------------------------------------------------
  using BTQModel = ::mvsic::turboquant_8bit_mv::Model<metric>;
  using BTQEncSet = typename BTQModel::EncodedSet;

 protected:
  mutable std::unique_ptr<BTQModel> tq8_rerank_model_;
  mutable std::unique_ptr<BTQEncSet> tq8_rerank_db_;
  mutable std::once_flag tq8_rerank_once_;

  void ensure_tq8_rerank_db_(const PointCloudSet<ChPoint>& points) const {
    std::call_once(tq8_rerank_once_, [&] {
      auto model = std::make_unique<BTQModel>();
      model->train(points);
      auto enc = std::make_unique<BTQEncSet>(model->encode(points));
      tq8_rerank_model_ = std::move(model);
      tq8_rerank_db_ = std::move(enc);
    });
  }

 public:
  virtual size_t rerank_tq8_(const ChPoint& query, const PointCloudSet<ChPoint>& points,
                             const parlay::sequence<std::pair<uint32_t, float>>& candidates,
                             size_t num_rerank,
                             parlay::sequence<std::pair<uint32_t, float>>& out_results) const {
    ensure_tq8_rerank_db_(points);
    const auto& db = *tq8_rerank_db_;
    auto q8 = tq8_rerank_model_->quantize_query(query);

    auto scored = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(num_rerank);
    auto bytes = parlay::sequence<size_t>::uninitialized(num_rerank);
    const size_t per_vec_bytes =
        db.num_bytes_per_datapoint + sizeof(float) + (metric ? sizeof(float) : 0);
    parlay::parallel_for(0, num_rerank, [&](size_t i) {
      const uint32_t id = candidates[i].first;
      const float d = ::mvsic::turboquant_8bit_mv::turboquant_8bit_mv_chamfer_distance<metric>(
          q8, db, static_cast<size_t>(id));
      scored[i] = {id, d};
      bytes[i] = db.cloud_size(static_cast<size_t>(id)) * per_vec_bytes;
    });

    parlay::sort_inplace(scored,
                         [](const auto& a, const auto& b) { return a.second < b.second; });
    parlay::parallel_for(0, out_results.size(),
                         [&](size_t i) { out_results[i] = scored[i]; });
    return parlay::reduce(bytes);
  }

  // Write the index to a file in disk
  virtual void save(const std::string& filename) {}
  // Read the index from a file in disk
  virtual void load(const std::string& filename, const PointCloudSet<ChPoint>& points) {}

  // Quantization Helpers
  void train_quantizer(const PointCloudSet<ChPoint>& points, typename MVQT::QuantModel& Model) {
    switch (quantization_mode) {
      case QT::PQ: {
        Model.template emplace<typename MVQT::PQ_Model>();
        std::get<typename MVQT::PQ_Model>(Model).train(points, params.pq.block_size,
                                                       params.pq.num_clusters_per_block,
                                                       params.pq.num_points_per_cluster);
        break;
      }
      case QT::RaBitQ: {
        Model.template emplace<typename MVQT::RQ_Model>();
        std::get<typename MVQT::RQ_Model>(Model).train(points, params.pq.rabitq_bits);
        break;
      }
      case QT::FastScan: {
        Model.template emplace<typename MVQT::FS_Model>();
        std::get<typename MVQT::FS_Model>(Model).train(points, params.pq.block_size);
        break;
      }
      case QT::TurboQuant: {
        Model.template emplace<typename MVQT::TQ_Model>();
        std::get<typename MVQT::TQ_Model>(Model).train(points);
        break;
      }
      case QT::SPQTQ: {
        Model.template emplace<typename MVQT::PQTQ_Model>();
        std::get<typename MVQT::PQTQ_Model>(Model).train(points, params.pq.block_size);
        break;
      }
      case QT::OneBitTQ: {
        Model.template emplace<typename MVQT::OBTQ_Model>();
        std::get<typename MVQT::OBTQ_Model>(Model).train(points);
        break;
      }
      case QT::EightBitTQ: {
        Model.template emplace<typename MVQT::EBTQ_Model>();
        std::get<typename MVQT::EBTQ_Model>(Model).train(points);
        break;
      }
      default: Model = std::monostate{}; break;
    }
  }

  // SVTraits defaults to Index::SVQT (parlayANN PointRange). Pass QuantTypes<metric, YourRange>
  // explicitly when your single-vector storage uses a different range type (e.g.
  // mvsic::PointRange).
  template<typename SVTraits = SVQT, typename PR>
  void train_quantizer(const PR& points, typename SVTraits::QuantModel& Model) {
    switch (quantization_mode) {
      case QT::PQ: {
        Model.template emplace<typename SVTraits::PQ_Model>();
        std::get<typename SVTraits::PQ_Model>(Model).train(points, params.pq.block_size,
                                                           params.pq.num_clusters_per_block,
                                                           params.pq.num_points_per_cluster);
        break;
      }
      case QT::RaBitQ: {
        Model.template emplace<typename SVTraits::RQ_Model>();
        std::get<typename SVTraits::RQ_Model>(Model).train(points, params.pq.rabitq_bits);
        break;
      }
      case QT::FastScan: {
        Model.template emplace<typename SVTraits::FS_Model>();
        std::get<typename SVTraits::FS_Model>(Model).train(points, params.pq.block_size);
        break;
      }
      case QT::TurboQuant: {
        Model.template emplace<typename SVTraits::TQ_Model>();
        std::get<typename SVTraits::TQ_Model>(Model).train(points);
        break;
      }
      case QT::SPQTQ: {
        Model.template emplace<typename SVTraits::PQTQ_Model>();
        std::get<typename SVTraits::PQTQ_Model>(Model).train(points, params.pq.block_size);
        break;
      }
      default: Model = std::monostate{}; break;
    }
  }

  typename MVQT::QuantSet encode_points_quantized(const PointCloudSet<ChPoint>& points,
                                                  typename MVQT::QuantModel& Model) {
    switch (quantization_mode) {
      case QT::PQ: return std::get<typename MVQT::PQ_Model>(Model).encode(points);
      case QT::RaBitQ: return std::get<typename MVQT::RQ_Model>(Model).encode(points);
      case QT::FastScan: return std::get<typename MVQT::FS_Model>(Model).encode(points);
      case QT::TurboQuant: return std::get<typename MVQT::TQ_Model>(Model).encode(points);
      case QT::SPQTQ: return std::get<typename MVQT::PQTQ_Model>(Model).encode(points);
      case QT::OneBitTQ: return std::get<typename MVQT::OBTQ_Model>(Model).encode(points);
      case QT::EightBitTQ: return std::get<typename MVQT::EBTQ_Model>(Model).encode(points);
      case QT::None:
      default: return std::monostate{};
    }
  }

  template<typename SVTraits = SVQT, typename PR>
  typename SVTraits::QuantRange encode_range_quantized(const PR& points,
                                                       typename SVTraits::QuantModel& Model) {
    switch (quantization_mode) {
      case QT::PQ: return std::get<typename SVTraits::PQ_Model>(Model).encode(points);
      case QT::RaBitQ: return std::get<typename SVTraits::RQ_Model>(Model).encode(points);
      case QT::FastScan: return std::get<typename SVTraits::FS_Model>(Model).encode(points);
      case QT::TurboQuant: return std::get<typename SVTraits::TQ_Model>(Model).encode(points);
      case QT::SPQTQ: return std::get<typename SVTraits::PQTQ_Model>(Model).encode(points);
      case QT::None:
      default: return std::monostate{};
    }
  }

  typename MVQT::QuantQuery quantize_query_point_cloud(const ChPoint& query,
                                                       const typename MVQT::QuantModel& Model) {
    switch (quantization_mode) {
      case QT::PQ: return std::get<typename MVQT::PQ_Model>(Model).quantize_query(query);
      case QT::RaBitQ: return std::get<typename MVQT::RQ_Model>(Model).quantize_query(query);
      case QT::FastScan: return std::get<typename MVQT::FS_Model>(Model).quantize_query(query);
      case QT::TurboQuant: return std::get<typename MVQT::TQ_Model>(Model).quantize_query(query);
      case QT::SPQTQ: return std::get<typename MVQT::PQTQ_Model>(Model).quantize_query(query);
      case QT::OneBitTQ: return std::get<typename MVQT::OBTQ_Model>(Model).quantize_query(query);
      case QT::EightBitTQ:
        return std::get<typename MVQT::EBTQ_Model>(Model).quantize_query(query);
      case QT::None:
      default: return std::monostate{};
    }
  }

  template<typename SVTraits = SVQT, typename PointTy>
  typename SVTraits::QuantQuery quantize_query_point(const PointTy& query,
                                                     const typename SVTraits::QuantModel& Model) {
    switch (quantization_mode) {
      case QT::PQ: return std::get<typename SVTraits::PQ_Model>(Model).quantize_query(query);
      case QT::RaBitQ: return std::get<typename SVTraits::RQ_Model>(Model).quantize_query(query);
      case QT::FastScan: return std::get<typename SVTraits::FS_Model>(Model).quantize_query(query);
      case QT::TurboQuant:
        return std::get<typename SVTraits::TQ_Model>(Model).quantize_query(query);
      case QT::SPQTQ: return std::get<typename SVTraits::PQTQ_Model>(Model).quantize_query(query);
      case QT::None:
      default: return std::monostate{};
    }
  }

  void quant_distances_all(const typename MVQT::QuantQuery& q_var,
                           const typename MVQT::QuantSet& s_var, std::pair<uint32_t, float>* out) {
    switch (quantization_mode) {
      case QT::PQ:
        std::get<typename MVQT::PQ_Set>(s_var).distances_all(
            std::get<typename MVQT::PQ_Query>(q_var), out);
        break;
      case QT::RaBitQ:
        std::get<typename MVQT::RQ_Set>(s_var).distances_all(
            std::get<typename MVQT::RQ_Query>(q_var), out);
        break;
      case QT::FastScan:
        std::get<typename MVQT::FS_Set>(s_var).distances_all(
            std::get<typename MVQT::FS_Query>(q_var), out);
        break;
      case QT::TurboQuant:
        std::get<typename MVQT::TQ_Set>(s_var).distances_all(
            std::get<typename MVQT::TQ_Query>(q_var), out);
        break;
      case QT::SPQTQ:
        std::get<typename MVQT::PQTQ_Set>(s_var).distances_all(
            std::get<typename MVQT::PQTQ_Query>(q_var), out);
        break;
      case QT::OneBitTQ:
        std::get<typename MVQT::OBTQ_Set>(s_var).distances_all(
            std::get<typename MVQT::OBTQ_Query>(q_var), out);
        break;
      case QT::EightBitTQ:
        std::get<typename MVQT::EBTQ_Set>(s_var).distances_all(
            std::get<typename MVQT::EBTQ_Query>(q_var), out);
        break;
      case QT::None:
      default: break;
    }
  }

  template<class Graph, typename QueryParams>
  auto quant_beam_search(const Point& query, const typename SVQT::QuantQuery& q_query,
                         const Range& points, const typename SVQT::QuantRange& quantized_points,
                         const Graph& G, uint32_t start_point, const QueryParams& QP) {
    parlay::sequence<std::pair<uint32_t, float>> visited;
    size_t dist_cmps;
    size_t bytes_accessed = 0;
    switch (quantization_mode) {
      case QT::PQ: {
        auto [result, cmps] =
            parlayANN::beam_search<typename SVQT::PQ_Query, typename SVQT::PQ_Range, uint32_t>(
                std::get<typename SVQT::PQ_Query>(q_query), G,
                std::get<typename SVQT::PQ_Range>(quantized_points), start_point, QP);
        visited = result.second;
        dist_cmps = cmps;
        bytes_accessed +=
            cmps * std::get<typename SVQT::PQ_Range>(quantized_points).num_bytes_per_point();
        break;
      }
      case QT::RaBitQ: {
        auto [result, cmps] =
            parlayANN::beam_search<typename SVQT::RQ_Query, typename SVQT::RQ_Range, uint32_t>(
                std::get<typename SVQT::RQ_Query>(q_query), G,
                std::get<typename SVQT::RQ_Range>(quantized_points), start_point, QP);
        visited = result.second;
        dist_cmps = cmps;
        bytes_accessed +=
            cmps * std::get<typename SVQT::RQ_Range>(quantized_points).num_bytes_per_point();
        break;
      }
      case QT::FastScan: {
        auto [result, cmps] =
            parlayANN::beam_search<typename SVQT::FS_Query, typename SVQT::FS_Range, uint32_t>(
                std::get<typename SVQT::FS_Query>(q_query), G,
                std::get<typename SVQT::FS_Range>(quantized_points), start_point, QP);
        visited = result.second;
        dist_cmps = cmps;
        bytes_accessed +=
            cmps * std::get<typename SVQT::FS_Range>(quantized_points).num_bytes_per_point();
        break;
      }
      case QT::TurboQuant: {
        auto [result, cmps] =
            parlayANN::beam_search<typename SVQT::TQ_Query, typename SVQT::TQ_Range, uint32_t>(
                std::get<typename SVQT::TQ_Query>(q_query), G,
                std::get<typename SVQT::TQ_Range>(quantized_points), start_point, QP);
        visited = result.second;
        dist_cmps = cmps;
        bytes_accessed +=
            cmps * std::get<typename SVQT::TQ_Range>(quantized_points).num_bytes_per_point();
        break;
      }
      case QT::SPQTQ: {
        auto [result, cmps] =
            parlayANN::beam_search<typename SVQT::PQTQ_Query, typename SVQT::PQTQ_Range, uint32_t>(
                std::get<typename SVQT::PQTQ_Query>(q_query), G,
                std::get<typename SVQT::PQTQ_Range>(quantized_points), start_point, QP);
        visited = result.second;
        dist_cmps = cmps;
        bytes_accessed +=
            cmps * std::get<typename SVQT::PQTQ_Range>(quantized_points).num_bytes_per_point();
        break;
      }
      case QT::None: {
        auto [result, cmps] =
            parlayANN::beam_search<Point, Range, uint32_t>(query, G, points, start_point, QP);
        visited = result.second;
        dist_cmps = cmps;
        bytes_accessed += cmps * (points.get_dims() * sizeof(float));
        break;
      }
      default: std::cerr << "Error: Unsupported Quantization Method!" << std::endl; abort();
    }
    return std::make_tuple(visited, dist_cmps, bytes_accessed);
  }

  // Only valid for MVIVF
  virtual size_t mean_cluster_size() const noexcept {
    std::cout << "mean_cluster_size() not implemented for this index type" << std::endl;
    return 0;
  }
  virtual size_t max_cluster_size() const noexcept {
    std::cout << "max_cluster_size() not implemented for this index type" << std::endl;
    return 0;
  }
  virtual size_t get_height() const noexcept {
    std::cout << "get_height() not implemented for this index type" << std::endl;
    return 0;
  }
};

}  // namespace mvsic

#ifdef _MSC_VER
#define UNREACHABLE() __assume(0)
#else
#define UNREACHABLE() __builtin_unreachable()
#endif