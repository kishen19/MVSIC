#pragma once

#include <vector>
#include <tuple>
#include <variant>
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
#include "mvsic/core/quantization/rabitq.h"
#include "mvsic/core/quantization/fastscan.h"
#include "mvsic/core/quantization/turboquant_4bit.h"
#include "mvsic/core/quantization/turboquant_pq_4bit.h"
#include "mvsic/core/quantization/wrapper.h"

// Params
#include "search_params.h"
#include "index_params.h"

namespace mvsic {

// Quantization Traits
template<bool metric, typename Range>
struct QuantTypes {
  // PointRange Alternative
  using PQ_Range = pq::Quantized_Point_Range<Range, metric>;
  using RQ_Range = rabitq::Quantized_Point_Range<Range, metric>;
  using FS_Range = fastscan::Quantized_Point_Range<Range, metric>;
  using TQ4_Range = turboquant_4bit::Quantized_Point_Range<Range, metric>;
  using TQPQ4_Range = turboquant_pq_4bit::Quantized_Point_Range<Range, metric, 4>;
  using TQPQ8_Range = turboquant_pq_4bit::Quantized_Point_Range<Range, metric, 8>;
  // Query Vector
  using PQ_Point = pq::Quantized_Query<metric>;
  using RQ_Point = rabitq::Quantized_Query<metric>;
  using FS_Point = fastscan::Quantized_Query<metric>;
  using TQ4_Point = turboquant_4bit::Quantized_Query<metric>;
  using TQPQ4_Point = turboquant_pq_4bit::Quantized_Query<metric, 4>;
  using TQPQ8_Point = turboquant_pq_4bit::Quantized_Query<metric, 8>;
  // Main Model Object
  using PQ_Model = pq::Model<metric>;
  using RQ_Model = rabitq::Model<metric>;
  using FS_Model = fastscan::Model<metric>;
  using TQ4_Model = turboquant_4bit::Model<metric>;
  using TQPQ4_Model = turboquant_pq_4bit::Model<metric, 4>;
  using TQPQ8_Model = turboquant_pq_4bit::Model<metric, 8>;
  // Unified Objects
  using QuantModel = std::variant<std::monostate, PQ_Model, RQ_Model, FS_Model, TQ4_Model,
                                  TQPQ4_Model, TQPQ8_Model>;
  using QuantQuery = std::variant<std::monostate, PQ_Point, RQ_Point, FS_Point, TQ4_Point,
                                  TQPQ4_Point, TQPQ8_Point>;
  using QuantRange = std::variant<std::monostate, PQ_Range, RQ_Range, FS_Range, TQ4_Range,
                                  TQPQ4_Range, TQPQ8_Range>;
};

template<bool metric, typename ChPoint>
struct MVQuantTypes {
  using FlatRange = FlattenedPCRange<PointCloudSet<ChPoint>>;
  // PointCloudSet Alternate
  using PQ_Set = Quantized_Point_Cloud_Set<pq::Quantized_Point_Range<FlatRange, metric>, metric>;
  using RQ_Set =
      Quantized_Point_Cloud_Set<rabitq::Quantized_Point_Range<FlatRange, metric>, metric>;
  using FS_Set =
      Quantized_Point_Cloud_Set<fastscan::Quantized_Point_Range<FlatRange, metric>, metric>;
  using TQ4_Set =
      Quantized_Point_Cloud_Set<turboquant_4bit::Quantized_Point_Range<FlatRange, metric>, metric>;
  using TQPQ4_Set =
      Quantized_Point_Cloud_Set<turboquant_pq_4bit::Quantized_Point_Range<FlatRange, metric, 4>,
                                metric>;
  using TQPQ8_Set =
      Quantized_Point_Cloud_Set<turboquant_pq_4bit::Quantized_Point_Range<FlatRange, metric, 8>,
                                metric>;
  // ChPoint (Query) Alternate
  using PQ_Q = Quantized_Query_Point_Cloud<pq::Quantized_Query<metric>, metric>;
  using RQ_Q = Quantized_Query_Point_Cloud<rabitq::Quantized_Query<metric>, metric>;
  using FS_Q = Quantized_Query_Point_Cloud<fastscan::Quantized_Query<metric>, metric>;
  using TQ4_Q = Quantized_Query_Point_Cloud<turboquant_4bit::Quantized_Query<metric>, metric>;
  using TQPQ4_Q =
      Quantized_Query_Point_Cloud<turboquant_pq_4bit::Quantized_Query<metric, 4>, metric>;
  using TQPQ8_Q =
      Quantized_Query_Point_Cloud<turboquant_pq_4bit::Quantized_Query<metric, 8>, metric>;
  // Main Model Object
  using PQ_Model = MultiVecQuantizer<pq::Model<metric>, metric>;
  using FS_Model = MultiVecQuantizer<fastscan::Model<metric>, metric>;
  using RQ_Model = MultiVecQuantizer<rabitq::Model<metric>, metric>;
  using TQ4_Model = MultiVecQuantizer<turboquant_4bit::Model<metric>, metric>;
  using TQPQ4_Model = MultiVecQuantizer<turboquant_pq_4bit::Model<metric, 4>, metric>;
  using TQPQ8_Model = MultiVecQuantizer<turboquant_pq_4bit::Model<metric, 8>, metric>;
  // Unified Objects
  using QuantSet =
      std::variant<std::monostate, PQ_Set, FS_Set, RQ_Set, TQ4_Set, TQPQ4_Set, TQPQ8_Set>;
  using QuantQuery = std::variant<std::monostate, PQ_Q, FS_Q, RQ_Q, TQ4_Q, TQPQ4_Q, TQPQ8_Q>;
  using QuantModel = std::variant<std::monostate, PQ_Model, FS_Model, RQ_Model, TQ4_Model,
                                  TQPQ4_Model, TQPQ8_Model>;
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
      case QT::TurboQuant4Bit: {
        Model.template emplace<typename MVQT::TQ4_Model>();
        std::get<typename MVQT::TQ4_Model>(Model).train(points);
        break;
      }
      case QT::TurboQuantPQ4Bit: {
        if (params.pq.block_size == 4) {
          Model.template emplace<typename MVQT::TQPQ4_Model>();
          std::get<typename MVQT::TQPQ4_Model>(Model).train(points);
        } else if (params.pq.block_size == 8) {
          Model.template emplace<typename MVQT::TQPQ8_Model>();
          std::get<typename MVQT::TQPQ8_Model>(Model).train(points);
        } else {
          std::cerr << "TurboQuantPQ4Bit currently supports block_size 4 or 8 "
                    << "(got " << params.pq.block_size << ")." << std::endl;
          abort();
        }
        break;
      }
      default: Model = std::monostate{}; break;
    }
  }

  void train_quantizer(const Range& points, typename SVQT::QuantModel& Model) {
    switch (quantization_mode) {
      case QT::PQ: {
        Model.template emplace<typename SVQT::PQ_Model>();
        std::get<typename SVQT::PQ_Model>(Model).train(points, params.pq.block_size,
                                                       params.pq.num_clusters_per_block,
                                                       params.pq.num_points_per_cluster);
        break;
      }
      case QT::RaBitQ: {
        Model.template emplace<typename SVQT::RQ_Model>();
        std::get<typename SVQT::RQ_Model>(Model).train(points, params.pq.rabitq_bits);
        break;
      }
      case QT::FastScan: {
        Model.template emplace<typename SVQT::FS_Model>();
        std::get<typename SVQT::FS_Model>(Model).train(points, params.pq.block_size);
        break;
      }
      case QT::TurboQuant4Bit: {
        Model.template emplace<typename SVQT::TQ4_Model>();
        std::get<typename SVQT::TQ4_Model>(Model).train(points);
        break;
      }
      case QT::TurboQuantPQ4Bit: {
        if (params.pq.block_size == 4) {
          Model.template emplace<typename SVQT::TQPQ4_Model>();
          std::get<typename SVQT::TQPQ4_Model>(Model).train(points);
        } else if (params.pq.block_size == 8) {
          Model.template emplace<typename SVQT::TQPQ8_Model>();
          std::get<typename SVQT::TQPQ8_Model>(Model).train(points);
        } else {
          std::cerr << "TurboQuantPQ4Bit currently supports block_size 4 or 8 "
                    << "(got " << params.pq.block_size << ")." << std::endl;
          abort();
        }
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
      case QT::TurboQuant4Bit: return std::get<typename MVQT::TQ4_Model>(Model).encode(points);
      case QT::TurboQuantPQ4Bit:
        if (params.pq.block_size == 4)
          return std::get<typename MVQT::TQPQ4_Model>(Model).encode(points);
        return std::get<typename MVQT::TQPQ8_Model>(Model).encode(points);
      case QT::None:
      default: return std::monostate{};
    }
  }

  typename SVQT::QuantRange encode_range_quantized(const Range& points,
                                                   typename SVQT::QuantModel& Model) {
    switch (quantization_mode) {
      case QT::PQ: return std::get<typename SVQT::PQ_Model>(Model).encode(points);
      case QT::RaBitQ: return std::get<typename SVQT::RQ_Model>(Model).encode(points);
      case QT::FastScan: return std::get<typename SVQT::FS_Model>(Model).encode(points);
      case QT::TurboQuant4Bit: return std::get<typename SVQT::TQ4_Model>(Model).encode(points);
      case QT::TurboQuantPQ4Bit:
        if (params.pq.block_size == 4)
          return std::get<typename SVQT::TQPQ4_Model>(Model).encode(points);
        return std::get<typename SVQT::TQPQ8_Model>(Model).encode(points);
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
      case QT::TurboQuant4Bit:
        return std::get<typename MVQT::TQ4_Model>(Model).quantize_query(query);
      case QT::TurboQuantPQ4Bit:
        if (params.pq.block_size == 4) {
          return std::get<typename MVQT::TQPQ4_Model>(Model).quantize_query(query);
        } else {
          return std::get<typename MVQT::TQPQ8_Model>(Model).quantize_query(query);
        }
      case QT::None:
      default: return std::monostate{};
    }
  }

  typename SVQT::QuantQuery quantize_query_point(const Point& query,
                                                 const typename SVQT::QuantModel& Model) {
    switch (quantization_mode) {
      case QT::PQ: return std::get<typename SVQT::PQ_Model>(Model).quantize_query(query);
      case QT::RaBitQ: return std::get<typename SVQT::RQ_Model>(Model).quantize_query(query);
      case QT::FastScan: return std::get<typename SVQT::FS_Model>(Model).quantize_query(query);
      case QT::TurboQuant4Bit:
        return std::get<typename SVQT::TQ4_Model>(Model).quantize_query(query);
      case QT::TurboQuantPQ4Bit:
        if (params.pq.block_size == 4) {
          return std::get<typename SVQT::TQPQ4_Model>(Model).quantize_query(query);
        } else {
          return std::get<typename SVQT::TQPQ8_Model>(Model).quantize_query(query);
        }
      case QT::None:
      default: return std::monostate{};
    }
  }

  void quant_distances_all(const typename MVQT::QuantQuery& q_var,
                           const typename MVQT::QuantSet& s_var, std::pair<uint32_t, float>* out) {
    switch (quantization_mode) {
      case QT::PQ:
        std::get<typename MVQT::PQ_Set>(s_var).distances_all(std::get<typename MVQT::PQ_Q>(q_var),
                                                             out);
        break;
      case QT::RaBitQ:
        std::get<typename MVQT::RQ_Set>(s_var).distances_all(std::get<typename MVQT::RQ_Q>(q_var),
                                                             out);
        break;
      case QT::FastScan:
        std::get<typename MVQT::FS_Set>(s_var).distances_all(std::get<typename MVQT::FS_Q>(q_var),
                                                             out);
        break;
      case QT::TurboQuant4Bit:
        std::get<typename MVQT::TQ4_Set>(s_var).distances_all(std::get<typename MVQT::TQ4_Q>(q_var),
                                                              out);
        break;
      case QT::TurboQuantPQ4Bit:
        if (params.pq.block_size == 4) {
          std::get<typename MVQT::TQPQ4_Set>(s_var).distances_all(
              std::get<typename MVQT::TQPQ4_Q>(q_var), out);
        } else {
          std::get<typename MVQT::TQPQ8_Set>(s_var).distances_all(
              std::get<typename MVQT::TQPQ8_Q>(q_var), out);
        }
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
    switch (quantization_mode) {
      case QT::PQ: {
        auto [result, cmps] =
            parlayANN::beam_search<typename SVQT::PQ_Point, typename SVQT::PQ_Range, uint32_t>(
                std::get<typename SVQT::PQ_Point>(q_query), G,
                std::get<typename SVQT::PQ_Range>(quantized_points), start_point, QP);
        visited = result.second;
        dist_cmps = cmps;
        break;
      }
      case QT::RaBitQ: {
        auto [result, cmps] =
            parlayANN::beam_search<typename SVQT::RQ_Point, typename SVQT::RQ_Range, uint32_t>(
                std::get<typename SVQT::RQ_Point>(q_query), G,
                std::get<typename SVQT::RQ_Range>(quantized_points), start_point, QP);
        visited = result.second;
        dist_cmps = cmps;
        break;
      }
      case QT::FastScan: {
        auto [result, cmps] =
            parlayANN::beam_search<typename SVQT::FS_Point, typename SVQT::FS_Range, uint32_t>(
                std::get<typename SVQT::FS_Point>(q_query), G,
                std::get<typename SVQT::FS_Range>(quantized_points), start_point, QP);
        visited = result.second;
        dist_cmps = cmps;
        break;
      }
      case QT::TurboQuant4Bit: {
        auto [result, cmps] =
            parlayANN::beam_search<typename SVQT::TQ4_Point, typename SVQT::TQ4_Range, uint32_t>(
                std::get<typename SVQT::TQ4_Point>(q_query), G,
                std::get<typename SVQT::TQ4_Range>(quantized_points), start_point, QP);
        visited = result.second;
        dist_cmps = cmps;
        break;
      }
      case QT::TurboQuantPQ4Bit: {
        if (params.pq.block_size == 4) {
          auto [result, cmps] = parlayANN::beam_search<typename SVQT::TQPQ4_Point,
                                                       typename SVQT::TQPQ4_Range, uint32_t>(
              std::get<typename SVQT::TQPQ4_Point>(q_query), G,
              std::get<typename SVQT::TQPQ4_Range>(quantized_points), start_point, QP);
          visited = result.second;
          dist_cmps = cmps;
        } else {
          auto [result, cmps] = parlayANN::beam_search<typename SVQT::TQPQ8_Point,
                                                       typename SVQT::TQPQ8_Range, uint32_t>(
              std::get<typename SVQT::TQPQ8_Point>(q_query), G,
              std::get<typename SVQT::TQPQ8_Range>(quantized_points), start_point, QP);
          visited = result.second;
          dist_cmps = cmps;
        }
        break;
        case QT::None: {
          auto [result, cmps] =
              parlayANN::beam_search<Point, Range, uint32_t>(query, G, points, start_point, QP);
          visited = result.second;
          dist_cmps = cmps;
          break;
        }
        default: std::cerr << "Error: Unsupported Quantization Method!" << std::endl; abort();
      }
    }
    return std::make_pair(visited, dist_cmps);
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

#define MV_QUANT_TYPES_ALL                                                                         \
  using FlatRange = typename MVQT::FlatRange;                                                      \
  using PQ_Set = typename MVQT::PQ_Set;                                                            \
  using RQ_Set = typename MVQT::RQ_Set;                                                            \
  using FS_Set = typename MVQT::FS_Set;                                                            \
  using TQ4_Set = typename MVQT::TQ4_Set;                                                          \
  using TQPQ4_Set = typename MVQT::TQPQ4_Set;                                                      \
  using TQPQ8_Set = typename MVQT::TQPQ8_Set;                                                      \
  using PQ_Q = typename MVQT::PQ_Q;                                                                \
  using RQ_Q = typename MVQT::RQ_Q;                                                                \
  using FS_Q = typename MVQT::FS_Q;                                                                \
  using TQ4_Q = typename MVQT::TQ4_Q;                                                              \
  using TQPQ4_Q = typename MVQT::TQPQ4_Q;                                                          \
  using TQPQ8_Q = typename MVQT::TQPQ8_Q;                                                          \
  using QuantSet = typename MVQT::QuantSet;                                                        \
  using QuantQuery = typename MVQT::QuantQuery;                                                    \
  using QuantModel = typename MVQT::QuantModel;

#ifdef _MSC_VER
#define UNREACHABLE() __assume(0)
#else
#define UNREACHABLE() __builtin_unreachable()
#endif