#pragma once

#include <variant>

#include "mvsic/muvera/fde/fixed_dimensional_encoding.h"
#include "mvsic/core/index.h"
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

template<bool metric>
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

  uint32_t d_fde;                                  // FDE dimension
  Range points_fdes;                               // FDEs
  parlayANN::Graph<uint32_t> G;                    // Vamana graph
  parlayANN::BuildParams BP;                       // Vamana build parameters
  parlayANN::knn_index<Range, Range, uint32_t> I;  // Vamana index

  // Quantizer Storage
  QuantModel quantizer = std::monostate{};
  QuantRange quantized_data = std::monostate{};

  IndexMUVERA(uint32_t d_) noexcept :
      Index<metric>(IndexParams::muvera()),
      BP(parlayANN::BuildParams(params.ann.R, params.ann.L, params.ann.alpha, params.ann.num_pass)),
      I(parlayANN::knn_index<Range, Range, uint32_t>(BP)) {
    d = d_;
  }
  IndexMUVERA(uint32_t d_, const IndexParams& params) noexcept :
      Index<metric>(params),
      BP(parlayANN::BuildParams(params.ann.R, params.ann.L, params.ann.alpha, params.ann.num_pass)),
      I(parlayANN::knn_index<Range, Range, uint32_t>(BP)) {
    d = d_;
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

    // Step3: Quantization
    quantization_mode = params.pq.method;
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
    double t_fde = 0.0;
    double t_quantize = 0.0;
    double t_search = 0.0;
    double t_rerank = 0.0;

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
    auto final_results =
        parlay::sequence<std::pair<uint32_t, float>>::uninitialized(std::min(k, visited.size()));
    if (!search_params.norerank) {
      size_t num_rerank = std::min(search_params.num_rerank, visited.size());
      bytes_accessed += this->rerank(query, points, visited, num_rerank, final_results);
    } else {
      parlay::parallel_for(0, final_results.size(),
                           [&](size_t i) { final_results[i] = visited[i]; });
    }
    t_rerank = t.stop();
    t.reset();

    std::vector<double> stats;
    stats.push_back(static_cast<double>(dist_cmps));
    stats.push_back(t_fde);
    stats.push_back(t_quantize);
    stats.push_back(t_search);
    stats.push_back(t_rerank);

    return std::make_tuple(final_results, bytes_accessed, stats);
  }

  // Write the index to a file in disk
  void save(const std::string& filename) override {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("save: cannot open file: " + filename);

    // 1. Save graph
    parlayANN::io::save_graph(G, out);

    // 2. Save Quantizer Type Header
    out.write((char*)&d_fde, sizeof(uint32_t));
    int type_id = static_cast<int>(quantization_mode);
    out.write((char*)&type_id, sizeof(int));

    // 3. Save Quantizer Model and encodings OR Exact Vectors
    switch (quantization_mode) {
      case QT::PQ:
        std::get<typename SVQT::PQ_Model>(quantizer).save(out);
        std::get<typename SVQT::PQ_Range>(quantized_data).save(out);
        break;
      case QT::RaBitQ:
        std::get<typename SVQT::RQ_Model>(quantizer).save(out);
        std::get<typename SVQT::RQ_Range>(quantized_data).save(out);
        break;
      case QT::FastScan:
        std::get<typename SVQT::FS_Model>(quantizer).save(out);
        std::get<typename SVQT::FS_Range>(quantized_data).save(out);
        break;
      case QT::TurboQuant:
        std::get<typename SVQT::TQ_Model>(quantizer).save(out);
        std::get<typename SVQT::TQ_Range>(quantized_data).save(out);
        break;
      case QT::SPQTQ:
        std::get<typename SVQT::PQTQ_Model>(quantizer).save(out);
        std::get<typename SVQT::PQTQ_Range>(quantized_data).save(out);
        break;
      case QT::None: parlayANN::io::save_point_range(points_fdes, out); break;
    }
  }

  // Read the index from a file in disk
  void load(const std::string& filename, const PointCloudSet<ChPoint>& points) override {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("load: cannot open file: " + filename);

    // 1. Load Graph
    G = parlayANN::io::load_graph<uint32_t>(in);
    I.set_start();

    // 2. Load Quantizer Type
    in.read((char*)&d_fde, sizeof(uint32_t));
    int type_id;
    in.read((char*)&type_id, sizeof(int));
    quantization_mode = static_cast<QT>(type_id);

    // 3. Load Data
    switch (quantization_mode) {
      case QT::PQ:
        quantizer.template emplace<typename SVQT::PQ_Model>();
        std::get<typename SVQT::PQ_Model>(quantizer).load(in);
        quantized_data.template emplace<typename SVQT::PQ_Range>();
        std::get<typename SVQT::PQ_Range>(quantized_data).load(in);
        break;
      case QT::RaBitQ:
        quantizer.template emplace<typename SVQT::RQ_Model>();
        std::get<typename SVQT::RQ_Model>(quantizer).load(in);
        quantized_data.template emplace<typename SVQT::RQ_Range>();
        std::get<typename SVQT::RQ_Range>(quantized_data).load(in);
        break;
      case QT::FastScan:
        quantizer.template emplace<typename SVQT::FS_Model>();
        std::get<typename SVQT::FS_Model>(quantizer).load(in);
        quantized_data.template emplace<typename SVQT::FS_Range>();
        std::get<typename SVQT::FS_Range>(quantized_data).load(in);
        break;
      case QT::TurboQuant:
        quantizer.template emplace<typename SVQT::TQ_Model>();
        std::get<typename SVQT::TQ_Model>(quantizer).load(in);
        quantized_data.template emplace<typename SVQT::TQ_Range>();
        std::get<typename SVQT::TQ_Range>(quantized_data).load(in);
        break;
      case QT::SPQTQ:
        quantizer.template emplace<typename SVQT::PQTQ_Model>();
        std::get<typename SVQT::PQTQ_Model>(quantizer).load(in);
        quantized_data.template emplace<typename SVQT::PQTQ_Range>();
        std::get<typename SVQT::PQTQ_Range>(quantized_data).load(in);
        break;
      case QT::None:
        auto [fdes_data, loaded_d_fde] = parlayANN::io::read_point_range<Point>(in);
        assert(loaded_d_fde == d_fde);
        points_fdes = Range(fdes_data, d_fde);
        break;
    }
  }
};

using IndexMUVERAL2 = IndexMUVERA<true>;   // L2 metric
using IndexMUVERAIP = IndexMUVERA<false>;  // MIPS

}  // namespace mvsic