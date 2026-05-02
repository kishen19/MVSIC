#pragma once

#include <variant>

#include "mvsic/muvera/fde/fixed_dimensional_encoding.h"
#include "mvsic/core/index.h"
#include "mvsic/core/query_compression.h"
#include "mvsic/core/types/io.h"

// ParlayANN (Vamana) includes
#include "algorithms/utils/graph.h"
#include "algorithms/utils/stats.h"
#include "algorithms/utils/types.h"
#include "algorithms/vamana/index.h"

namespace mvsic {

/* MUVERA
 - Paper: https://arxiv.org/pdf/2405.19504
 - A multi-vector retrieval system that converts point clouds to a single-vector representation
 (FDEs), reducing chamfer distance computations to Inner Product computations.
 - Uses graph-based index (Vamana here) for retrieval.
*/

// Template parameters:
//   metric     true = L2, false = IP.
//   LeafModel  concrete quantizer Model type for the FDE index; NoQuantizer
//              means raw FDEs (no quantization).  Used today purely for
//              type-level class separation (so pybind11 sees distinct bindings
//              per variant and users can construct `IndexMUVERAPQIP` without
//              manually setting IndexParams).  The index constructor auto-
//              derives `params.pq.method` from the LeafModel type.  Full
//              compile-time `if constexpr` specialization of the beam-search
//              hot path is a follow-up (see docs/roadmap).
template<bool metric, class LeafModel = NoQuantizer<metric>>
class IndexMUVERA : public Index<metric> {
 public:
  using ChPoint = typename Index<metric>::ChPoint;  // Chamfer Point Type
  using Point = typename Index<metric>::Point;
  using Range = typename Index<metric>::Range;
  using SVQT = typename Index<metric>::SVQT;
  using QuantRange = typename SVQT::QuantRange;
  using QuantQuery = typename SVQT::QuantQuery;
  using QuantModel = typename SVQT::QuantModel;
  using QT = typename Index<metric>::QT;
  using Index<metric>::d;       // Embedding dimension
  using Index<metric>::params;  // Index Params
  using Index<metric>::quantization_mode;

  static constexpr QT kLeafMethod = quantizer_method_of_v<LeafModel, metric>;

  uint32_t d_fde;                                  // FDE dimension
  Range points_fdes;                               // FDEs
  parlayANN::Graph<uint32_t> G;                    // Vamana graph
  parlayANN::BuildParams BP;                       // Vamana build parameters
  parlayANN::knn_index<Range, Range, uint32_t> I;  // Vamana index

  // Quantizer Storage (variant-based; dispatched via kLeafMethod).
  QuantModel quantizer = std::monostate{};
  QuantRange quantized_data = std::monostate{};

  IndexMUVERA(uint32_t d_) noexcept :
      Index<metric>(IndexParams::muvera()),
      BP(parlayANN::BuildParams(params.ann.R, params.ann.L, params.ann.alpha, params.ann.num_pass)),
      I(parlayANN::knn_index<Range, Range, uint32_t>(BP)) {
    d = d_;
    if constexpr (kLeafMethod != QT::None) params.pq.method = kLeafMethod;
  }
  IndexMUVERA(uint32_t d_, const IndexParams& params_in) noexcept :
      Index<metric>(params_in),
      BP(parlayANN::BuildParams(params.ann.R, params.ann.L, params.ann.alpha, params.ann.num_pass)),
      I(parlayANN::knn_index<Range, Range, uint32_t>(BP)) {
    d = d_;
    // User-provided params take precedence, but if the template was instantiated
    // for a specific quantizer we override to keep type and method aligned.
    if constexpr (kLeafMethod != QT::None) params.pq.method = kLeafMethod;
  }

  // Builds the index given PointCloudSet object.
  void build(const PointCloudSet<ChPoint>& points) override {
    // Step 1: Compute FDEs of data point clouds
    if (params.verbose >= 1) std::cout << "Computing FDEs..." << std::endl;
    auto fdes = parlay::sequence<std::vector<float>>(points.size());
    graph_mining::FixedDimensionalEncodingConfig fde_config{
        static_cast<int32_t>(d),
        params.fde.num_repetitions,
        params.fde.num_simhash_projections,
        params.fde.seed,
        (params.fde.normalize) ? graph_mining::FixedDimensionalEncodingConfig::NORMALIZE_L2
                               : graph_mining::FixedDimensionalEncodingConfig::AVERAGE,
        params.fde.projection_dimension,
        (d == params.fde.projection_dimension)
            ? graph_mining::FixedDimensionalEncodingConfig::DEFAULT_IDENTITY
            : graph_mining::FixedDimensionalEncodingConfig::AMS_SKETCH,
        params.fde.fill_empty_partitions,
        params.fde.final_projection_dimension};
    parlay::parallel_for(0, points.size(), [&](size_t i) {
      uint32_t point_size = points.get_size(i);
      float* coords = points.data(i);
      std::vector<float> point_data(coords, coords + point_size * d);
      fdes[i] = graph_mining::GenerateFixedDimensionalEncoding(point_data, fde_config);
    });
    d_fde = fdes[0].size();
    points_fdes = Range(fdes, d_fde);

    // Step 2: Build ANN index on the FDEs
    if (params.verbose >= 1) std::cout << "Building ANN Index..." << std::endl;
    G = parlayANN::Graph<uint32_t>(BP.R, points.size());
    parlayANN::stats<uint32_t> BuildStats(G.size());
    I.build_index(G, points_fdes, points_fdes, BuildStats);
    if (params.verbose >= 1) std::cout << "FDE Dimension: " << points_fdes.get_dims() << std::endl;

    // Step3: Quantization — resolved from the compile-time `LeafModel`.
    quantization_mode = kLeafMethod;
    this->train_quantizer(points_fdes, quantizer);
    quantized_data = this->encode_range_quantized(points_fdes, quantizer);
  }

  // Returns the top-k point clouds for the query point cloud
  // Output format: < [<id, distance>, ...], # distance comparisons>
  std::tuple<parlay::sequence<std::pair<uint32_t, float>>, size_t, std::vector<double>>
  search_with_stats(const ChPoint& query, const PointCloudSet<ChPoint>& points,
                    const SearchParams& search_params) override {
    parlay::internal::timer t;
    size_t bytes_accessed = 0;

    size_t dist_cmps = 0;
    double t_compress = 0.0;
    double t_fde = 0.0;
    double t_quantize = 0.0;
    double t_search = 0.0;
    double t_rerank = 0.0;

    size_t k = search_params.k;

    // Step 0: Compress query point cloud (optional)
    t.start();
    CompressedPointCloud<ChPoint> compressed_storage;
    ChPoint effective_query = query;
    if (search_params.query_compression != SearchParams::QueryCompression::None) {
      compressed_storage = compress_query<ChPoint>(
          query, search_params.query_compression,
          search_params.query_compression_threshold);
      effective_query = compressed_storage.view();
    }
    t_compress = t.stop();
    t.reset();

    // Step 1: Compute FDE of the (possibly compressed) query point cloud
    t.start();
    graph_mining::FixedDimensionalEncodingConfig fde_config{
        static_cast<int32_t>(d),
        params.fde.num_repetitions,
        params.fde.num_simhash_projections,
        params.fde.seed,
        graph_mining::FixedDimensionalEncodingConfig::DEFAULT_SUM,
        params.fde.projection_dimension,
        (d == params.fde.projection_dimension)
            ? graph_mining::FixedDimensionalEncodingConfig::DEFAULT_IDENTITY
            : graph_mining::FixedDimensionalEncodingConfig::AMS_SKETCH,
        params.fde.fill_empty_partitions,
        params.fde.final_projection_dimension};
    std::vector<float> query_vec(effective_query.data(),
                                 effective_query.data() + effective_query.size() * d);
    std::vector<float> query_fde =
        graph_mining::GenerateFixedDimensionalEncoding(query_vec, fde_config);
    assert(query_fde.size() == d_fde);
    typename Point::parameters parlayann_pr_params(d_fde);
    Point query_point(reinterpret_cast<typename Point::byte*>(query_fde.data()), -1,
                      parlayann_pr_params);
    t_fde = t.stop();
    t.reset();

    // Step 2: Run beam search
    uint32_t start_point = I.get_start();
    auto QP = parlayANN::QueryParams(search_params.num_rerank, search_params.L, search_params.cut,
                                     points.size(), params.ann.R);
    parlay::sequence<std::pair<uint32_t, float>> visited;

    // Quantize Query
    t.start();
    QuantQuery q_query_var = this->quantize_query_point(query_point, quantizer);
    t_quantize = t.stop();
    t.reset();

    // Beam Search
    t.start();
    std::tie(visited, dist_cmps, bytes_accessed) = this->quant_beam_search(
        query_point, q_query_var, points_fdes, quantized_data, G, start_point, QP);
    t_search = t.stop();
    t.reset();

    // Step 3: Re-ranking
    t.start();
    const ChPoint& rerank_query = search_params.compress_rerank ? effective_query : query;
    auto final_results =
        parlay::sequence<std::pair<uint32_t, float>>::uninitialized(std::min(k, visited.size()));
    if (!search_params.norerank) {
      size_t num_rerank = std::min(search_params.num_rerank, visited.size());
      if (search_params.tq8_rerank) {
        bytes_accessed +=
            this->rerank_tq8_(rerank_query, points, visited, num_rerank, final_results);
      } else {
        bytes_accessed += this->rerank(rerank_query, points, visited, num_rerank, final_results);
      }
    } else {
      parlay::parallel_for(0, final_results.size(),
                           [&](size_t i) { final_results[i] = visited[i]; });
    }
    t_rerank = t.stop();
    t.reset();

    std::vector<double> stats;
    stats.push_back(static_cast<double>(dist_cmps));
    stats.push_back(t_compress);
    stats.push_back(t_fde);
    stats.push_back(t_quantize);
    stats.push_back(t_search);
    stats.push_back(t_rerank);

    return std::make_tuple(final_results, bytes_accessed, stats);
  }

  // ---------------------------------------------------------------------------
  // Save / load (v3 uniform skeleton format).
  //
  // The on-disk file is variant-agnostic: any templated MUVERA variant can
  // load the same skeleton and re-train its leaf quantizer on load from the
  // persisted FDE vectors.  class_id is therefore a fixed constant.
  //
  // Layout:
  //   magic       : uint32 = 'MUVE'
  //   version     : uint32 = 3
  //   class_id    : uint32 = 0 (reserved)
  //   d_fde       : uint32
  //   graph       : parlayANN::io::save_graph payload
  //   points_fdes : raw FDE single-vector data (via save_point_range)
  //
  // save() is only valid on the raw skeleton variant
  // IndexMUVERA<metric, NoQuantizer<metric>>; runtime guard below.  Quantized
  // variants must build the raw skeleton, save it, then load() into the
  // desired templated variant which retrains on load.
  // ---------------------------------------------------------------------------
  static constexpr uint32_t kMagic = 0x4D555645u;  // 'MUVE'
  static constexpr uint32_t kVersion = 3u;         // v3: uniform skeleton format
  static constexpr uint32_t kClassId = 0u;

  void save(const std::string& filename) override {
    if constexpr (kLeafMethod != QT::None) {
      std::cerr << "[MUVERA] save() is only supported on the raw skeleton variant "
                   "(LeafModel=NoQuantizer). Build the raw skeleton, save it, then "
                   "load() into the desired templated variant."
                << std::endl;
      std::abort();
    }

    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("save: cannot open file: " + filename);
    const uint32_t magic = kMagic;
    const uint32_t ver = kVersion;
    const uint32_t cid = kClassId;
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&ver), sizeof(ver));
    out.write(reinterpret_cast<const char*>(&cid), sizeof(cid));
    out.write(reinterpret_cast<const char*>(&d_fde), sizeof(uint32_t));
    parlayANN::io::save_graph(G, out);
    parlayANN::io::save_point_range(points_fdes, out);
    out.close();
  }

  void load(const std::string& filename, const PointCloudSet<ChPoint>& /*points*/) override {
    parlay::internal::timer t_io;
    t_io.start();
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("load: cannot open file: " + filename);
    uint32_t magic = 0, ver = 0, cid = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&ver), sizeof(ver));
    in.read(reinterpret_cast<char*>(&cid), sizeof(cid));
    if (magic != kMagic) {
      throw std::runtime_error("[MUVERA] bad magic: file is not a MUVE skeleton index.");
    }
    if (ver != kVersion) {
      throw std::runtime_error(
          "[MUVERA] MUVERA index file format changed in v" + std::to_string(kVersion) +
          "; got v" + std::to_string(ver) + ". Rebuild with current code.");
    }
    if (cid != kClassId) {
      throw std::runtime_error(
          "[MUVERA] unexpected class_id " + std::to_string(cid) + " (expected " +
          std::to_string(kClassId) + " for the v" + std::to_string(kVersion) + " skeleton).");
    }
    in.read(reinterpret_cast<char*>(&d_fde), sizeof(uint32_t));
    G = parlayANN::io::load_graph<uint32_t>(in);
    I.set_start();
    auto [fdes_data, loaded_d_fde] = parlayANN::io::read_point_range<Point>(in);
    assert(loaded_d_fde == d_fde);
    points_fdes = Range(fdes_data, d_fde);
    in.close();
    double t_io_ms = t_io.stop() * 1000.0;

    parlay::internal::timer t_retrain;
    t_retrain.start();
    quantization_mode = kLeafMethod;
    this->train_quantizer(points_fdes, quantizer);
    quantized_data = this->encode_range_quantized(points_fdes, quantizer);
    double t_retrain_ms = t_retrain.stop() * 1000.0;
    std::cerr << "[MUVERA] load: tree_io=" << t_io_ms << "ms retrain=" << t_retrain_ms
              << "ms (leaf_quant=" << ((kLeafMethod != QT::None) ? 1 : 0) << ")" << std::endl;
  }
};

// Concrete aliases.  Each alias is a distinct type so pybind11 can register
// it independently; the corresponding `params.pq.method` is auto-set in the
// constructor.
using IndexMUVERAIP          = IndexMUVERA<false, NoQuantizer<false>>;
using IndexMUVERAL2          = IndexMUVERA<true,  NoQuantizer<true>>;
using IndexMUVERAPQIP        = IndexMUVERA<false, pq::Model<false>>;
using IndexMUVERAPQL2        = IndexMUVERA<true,  pq::Model<true>>;
using IndexMUVERARaBitQIP    = IndexMUVERA<false, rabitq::Model<false>>;
using IndexMUVERARaBitQL2    = IndexMUVERA<true,  rabitq::Model<true>>;
using IndexMUVERAFastScanIP  = IndexMUVERA<false, fastscan::Model<false>>;
using IndexMUVERAFastScanL2  = IndexMUVERA<true,  fastscan::Model<true>>;
using IndexMUVERATQIP        = IndexMUVERA<false, turboquant::Model<false>>;
using IndexMUVERATQL2        = IndexMUVERA<true,  turboquant::Model<true>>;
using IndexMUVERASPQTQIP     = IndexMUVERA<false, pqtq::Model<false>>;
using IndexMUVERASPQTQL2     = IndexMUVERA<true,  pqtq::Model<true>>;

using IndexMUVERAOneBitTQIP  = IndexMUVERA<false, turboquant_1bit::Model<false>>;
using IndexMUVERAOneBitTQL2  = IndexMUVERA<true,  turboquant_1bit::Model<true>>;

}  // namespace mvsic