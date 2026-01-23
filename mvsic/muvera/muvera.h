#pragma once

#include <queue>
#include <set>
#include <variant>
#include <optional>

#include "mvsic/muvera/fde/fixed_dimensional_encoding.h"
#include "mvsic/core/index.h"
#include "mvsic/core/types/io.h"

// Quantization Headers
#include "mvsic/core/quantization/pq.h"
#include "mvsic/core/quantization/rabitq.h"

// ParlayANN (Vamana) includes
#include "algorithms/utils/beamSearch.h"
#include "algorithms/utils/euclidian_point.h"
#include "algorithms/utils/graph.h"
#include "algorithms/utils/mips_point.h"
#include "algorithms/utils/point_range.h"
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

template<bool metric>
class IndexMUVERA : public Index<metric> {
 public:
  using ChPoint = typename Index<metric>::ChPoint;  // Chamfer Point Type
  using Point =
      std::conditional_t<metric, parlayANN::Euclidian_Point<float>, parlayANN::Mips_Point<float>>;
  using Range = parlayANN::PointRange<Point>;
  using Index<metric>::d;  // Embedding dimension

  // Quantizer Types
  using PQ_Range = pq::Quantized_Point_Range<Range, metric>;
  using PQ_Point = pq::Quantized_Query<metric>;
  using RQ_Range = rabitq::Quantized_Point_Range<Range, metric>;
  using RQ_Point = rabitq::Quantized_Query<metric>;
  using PQ_Model = pq::Model<metric>;
  using RQ_Model = rabitq::Model<metric>;

  using QuantModel = std::variant<std::monostate, PQ_Model, RQ_Model>;
  using QuantRange = std::variant<std::monostate, PQ_Range, RQ_Range>;
  using QT = IndexParams::QuantizerType;

  IndexParams params;
  uint32_t d_fde;                                  // FDE dimension
  Range points_fdes;                               // FDEs
  parlayANN::Graph<uint32_t> G;                    // Vamana graph
  parlayANN::BuildParams BP;                       // Vamana build parameters
  parlayANN::knn_index<Range, Range, uint32_t> I;  // Vamana index

  // Quantizer Storage
  QuantModel quantizer = std::monostate{};
  QuantRange quantized_data = std::monostate{};
  QT active_quantizer = QT::None;

  IndexMUVERA(uint32_t d_) noexcept :
      params(IndexParams::muvera()),
      BP(parlayANN::BuildParams(params.ann.R, params.ann.L, params.ann.alpha, params.ann.num_pass)),
      I(parlayANN::knn_index<Range, Range, uint32_t>(BP)) {
    d = d_;
  }
  IndexMUVERA(uint32_t d_, const IndexParams &params) noexcept :
      params(params),
      BP(parlayANN::BuildParams(params.ann.R, params.ann.L, params.ann.alpha, params.ann.num_pass)),
      I(parlayANN::knn_index<Range, Range, uint32_t>(BP)) {
    d = d_;
  }

  // Builds the index given PointCloudSet object.
  void build(const PointCloudSet<ChPoint> &points) override {
    if (params.compress_input) {
      // TODO: run Ward's HAC to compress input point clouds
    }
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
      float *coords = points.data(i);
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

    // Step3: Quantization
    active_quantizer = params.pq.method;
    switch (active_quantizer) {
      case QT::RaBitQ:
        if (params.verbose >= 1) std::cout << "Training RaBitQ..." << std::endl;
        quantizer.template emplace<RQ_Model>();
        std::get<RQ_Model>(quantizer).train(points_fdes, params.pq.rabitq_bits);
        quantized_data = std::get<RQ_Model>(quantizer).encode(points_fdes);
        break;
      case QT::PQ:
        if (params.verbose >= 1) std::cout << "Training Standard PQ..." << std::endl;
        quantizer.template emplace<PQ_Model>();
        std::get<PQ_Model>(quantizer).train(points_fdes, params.pq.block_size,
                                            params.pq.num_clusters_per_block,
                                            params.pq.num_points_per_cluster);
        quantized_data = std::get<PQ_Model>(quantizer).encode(points_fdes);
        break;
      case QT::None:
        quantizer = std::monostate{};
        quantized_data = std::monostate{};
        break;
      default: std::cerr << "Error: Unsupported Quantization Method!" << std::endl; abort();
    }
  }

  // Returns the top-k point clouds for the query point cloud
  // Output format: < [<id, distance>, ...], # distance comparisons>
  std::tuple<parlay::sequence<std::pair<uint32_t, float>>, size_t, std::vector<double>>
  search_with_stats(const ChPoint &query, const PointCloudSet<ChPoint> &points,
                    const SearchParams &search_params) override {
    parlay::internal::timer t;
    std::vector<double> timings;

    size_t k = search_params.k;
    // Step 1: Compute FDE of the query point cloud
    // FDE config
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
    // Query input
    std::vector<float> query_vec(query.data(), query.data() + query.size() * d);
    // Query FDE
    std::vector<float> query_fde =
        graph_mining::GenerateFixedDimensionalEncoding(query_vec, fde_config);
    assert(query_fde.size() == d_fde);
    typename Point::parameters parlayann_pr_params(d_fde);
    Point query_point(reinterpret_cast<typename Point::byte *>(query_fde.data()), -1,
                      parlayann_pr_params);
    timings.push_back(t.stop());
    t.reset();

    // Step 2: Run beam search
    double t_rest = 0.0;
    double t_quantize = 0.0;
    double t_search = 0.0;
    t.start();
    uint32_t start_point = I.get_start();
    auto QP = parlayANN::QueryParams(search_params.num_rerank, search_params.L, search_params.cut,
                                     points.size(), params.ann.R);
    parlay::sequence<std::pair<uint32_t, float>> visited;
    size_t dist_cmps;
    t_rest += t.stop();
    t.reset();

    switch (active_quantizer) {
      case QT::RaBitQ: {
        // Quantize Query
        t.start();
        auto &m = std::get<RQ_Model>(quantizer);
        auto q_query = m.quantize_query(query_point);
        t_quantize += t.stop();
        t.reset();
        // Search
        t.start();
        auto [result, cmps] = parlayANN::beam_search<RQ_Point, RQ_Range, uint32_t>(
            q_query, G, std::get<RQ_Range>(quantized_data), start_point, QP);
        visited = result.second;
        dist_cmps = cmps;  // RaBitQ usually counts its own ops or we estimate
        t_search += t.stop();
        t.reset();
        break;
      }
      case QT::PQ: {
        // Quantize Query
        t.start();
        auto &m = std::get<PQ_Model>(quantizer);
        auto q_query = m.quantize_query(query_point);
        t_quantize += t.stop();
        t.reset();
        // Search
        t.start();
        auto [result, cmps] = parlayANN::beam_search<PQ_Point, PQ_Range, uint32_t>(
            q_query, G, std::get<PQ_Range>(quantized_data), start_point, QP);
        visited = result.second;
        dist_cmps = cmps;
        t_search += t.stop();
        t.reset();
        break;
      }
      case QT::None: {
        t.start();
        auto [result, cmps] = parlayANN::beam_search<Point, Range, uint32_t>(
            query_point, G, points_fdes, start_point, QP);
        visited = result.second;
        dist_cmps = cmps * 2 * d_fde;
        t_search += t.stop();
        t.reset();
        break;
      }
      default: std::cerr << "Error: Unsupported Quantization Method!" << std::endl; abort();
    }
    timings.push_back(t_quantize);
    timings.push_back(t_search);
    timings.push_back(t_rest);

    // Step 3: Re-ranking
    t.start();
    auto final_results =
        parlay::sequence<std::pair<uint32_t, float>>::uninitialized(std::min(k, visited.size()));
    if (!search_params.norerank) {
      size_t num_rerank = std::min(search_params.num_rerank, visited.size());
      dist_cmps += this->rerank(query, points, visited, num_rerank, final_results);
    } else {
      parlay::parallel_for(0, final_results.size(),
                           [&](size_t i) { final_results[i] = visited[i]; });
    }
    timings.push_back(t.stop());
    t.reset();

    return std::make_tuple(final_results, dist_cmps, timings);
  }

  // Write the index to a file in disk
  void save(const std::string &filename) override {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("save: cannot open file: " + filename);

    // 1. Save graph
    parlayANN::io::save_graph(G, out);

    // 2. Save Quantizer Type Header
    out.write((char *)&d_fde, sizeof(uint32_t));
    int type_id = static_cast<int>(active_quantizer);
    out.write((char *)&type_id, sizeof(int));

    // 3. Save Quantizer Model and encodings OR Exact Vectors
    switch (active_quantizer) {
      case QT::RaBitQ:
        std::get<RQ_Model>(quantizer).save(out);
        std::get<RQ_Range>(quantized_data).save(out);
        break;
      case QT::PQ:
        std::get<PQ_Model>(quantizer).save(out);
        std::get<PQ_Range>(quantized_data).save(out);
        break;
      case QT::None: parlayANN::io::save_point_range(points_fdes, out); break;
    }
  }

  // Read the index from a file in disk
  void load(const std::string &filename, const PointCloudSet<ChPoint> &points) override {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("load: cannot open file: " + filename);

    // 1. Load Graph
    G = parlayANN::io::load_graph<uint32_t>(in);
    I.set_start();

    // 2. Load Quantizer Type
    in.read((char *)&d_fde, sizeof(uint32_t));
    int type_id;
    in.read((char *)&type_id, sizeof(int));
    active_quantizer = static_cast<QT>(type_id);

    // 3. Load Data
    switch (active_quantizer) {
      case QT::RaBitQ:
        quantizer.template emplace<RQ_Model>();
        std::get<RQ_Model>(quantizer).load(in);
        quantized_data.template emplace<RQ_Range>();
        std::get<RQ_Range>(quantized_data).load(in);
        break;
      case QT::PQ:
        quantizer.template emplace<PQ_Model>();
        std::get<PQ_Model>(quantizer).load(in);
        quantized_data.template emplace<PQ_Range>();
        std::get<PQ_Range>(quantized_data).load(in);
        break;
      case QT::None:
        auto [fdes_data, loaded_d_fde] = parlayANN::io::read_point_range<Point>(in);
        points_fdes = Range(fdes_data, d_fde);
        break;
    }
  }
};

using IndexMUVERAL2 = IndexMUVERA<true>;   // L2 metric
using IndexMUVERAIP = IndexMUVERA<false>;  // MIPS

}  // namespace mvsic