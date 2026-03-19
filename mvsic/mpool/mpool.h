#pragma once

#include <variant>

#include "mvsic/core/index.h"
#include "mvsic/core/types/io.h"

// ParlayANN (Vamana) includes
#include "algorithms/utils/graph.h"
#include "algorithms/utils/stats.h"
#include "algorithms/utils/types.h"
#include "algorithms/vamana/index.h"

namespace mvsic {

/* Mean Pooling
 - A multi-vector retrieval system that converts point clouds to a single-vector representation via
 mean-pooling, i.e. each point cloud is represented by the mean of the coordinates of its points.
 - Uses graph-based index (Vamana here) for retrieval.
*/

// Helper function to compute the mean-pooled vector, given a point cloud
template<typename ChPoint>
std::vector<float> mean_pooling(const ChPoint& point, bool normalize = true);

/* =============================Mean-Pooling + Vamana Index Class============================ */
template<bool metric>
class IndexMPool : public Index<metric> {
 public:
  using ChPoint = typename Index<metric>::ChPoint;  // Chamfer Point Type
  using Point = typename Index<metric>::Point;
  using Range = typename Index<metric>::Range;
  using SVQT = typename Index<metric>::SVQT;
  using QuantRange = typename SVQT::QuantRange;
  using QuantQuery = typename SVQT::QuantQuery;
  using QuantModel = typename SVQT::QuantModel;
  using QT = typename Index<metric>::QT;
  using Index<metric>::d;  // Embedding dimension
  using Index<metric>::params;
  using Index<metric>::quantization_mode;

  Range points_mp;                                 // Mean-Pooled vectors
  parlayANN::Graph<uint32_t> G;                    // Vamana graph
  parlayANN::BuildParams BP;                       // Vamana build parameters
  parlayANN::knn_index<Range, Range, uint32_t> I;  // Vamana index

  // Quantizer Storage
  QuantModel quantizer = std::monostate{};
  QuantRange quantized_data = std::monostate{};

  IndexMPool(uint32_t d_) noexcept :
      Index<metric>(IndexParams::mpool()),
      BP(parlayANN::BuildParams(params.ann.R, params.ann.L, params.ann.alpha, params.ann.num_pass)),
      I(parlayANN::knn_index<Range, Range, uint32_t>(BP)) {
    d = d_;
  }
  IndexMPool(uint32_t d_, const IndexParams& params_) noexcept :
      Index<metric>(params_),
      BP(parlayANN::BuildParams(params.ann.R, params.ann.L, params.ann.alpha, params.ann.num_pass)),
      I(parlayANN::knn_index<Range, Range, uint32_t>(BP)) {
    d = d_;
  }

  // Builds the index given PointCloudSet object.
  void build(const PointCloudSet<ChPoint>& points) override {
    if (params.compress_input) {
      // TODO: run Ward's HAC to compress input point clouds
    }
    // Step 1: Compute Mean-Pooled vectors of the data point clouds
    if (params.verbose >= 1)
      std::cout << "Computing mean-pooled vectors of input point clouds..." << std::endl;
    auto mpvs = parlay::sequence<std::vector<float>>(points.size());
    parlay::parallel_for(0, points.size(),
                         [&](size_t i) { mpvs[i] = mean_pooling(points[i], params.normalize); });
    points_mp = Range(mpvs, d);

    // Step 2: Build ANN index on the mean-pooled points
    if (params.verbose >= 1) std::cout << "Building ANN Index..." << std::endl;
    G = parlayANN::Graph<uint32_t>(BP.R, points_mp.size());
    parlayANN::stats<uint32_t> BuildStats(G.size());
    I.build_index(G, points_mp, points_mp, BuildStats);

    // Step3: Quantization
    quantization_mode = params.pq.method;
    this->train_quantizer(points_mp, quantizer);
    quantized_data = this->encode_range_quantized(points_mp, quantizer);
  }

  // Returns the top-k point clouds for the query point cloud
  // Output format: < [<id, distance>, ...], # distance comparisons>
  std::tuple<parlay::sequence<std::pair<uint32_t, float>>, size_t, std::vector<double>>
  search_with_stats(const ChPoint& query, const PointCloudSet<ChPoint>& points,
                    const SearchParams& search_params) override {
    parlay::internal::timer t;
    std::vector<double> timings;

    size_t k = search_params.k;
    // Step 1: Compute mean-pooling of the query point cloud
    t.start();
    std::vector<float> query_mpv = mean_pooling(query, false);
    typename Point::parameters parlayann_pr_params(d);
    Point query_point(reinterpret_cast<typename Point::byte*>(query_mpv.data()), -1,
                      parlayann_pr_params);
    timings.push_back(t.stop());
    t.reset();

    // Step 2: Run beam search
    t.start();
    uint32_t start_point = I.get_start();
    auto QP = parlayANN::QueryParams(search_params.num_rerank, search_params.L, search_params.cut,
                                     points.size(), params.ann.R);
    parlay::sequence<std::pair<uint32_t, float>> visited;
    size_t dist_cmps;
    timings.push_back(t.stop());
    t.reset();

    // Quantize query
    t.start();
    QuantQuery q_query_var = this->quantize_query_point(query_point, quantizer);
    timings.push_back(t.stop());
    t.reset();
    // Run beam search
    t.start();
    std::tie(visited, dist_cmps) = this->quant_beam_search(query_point, q_query_var, points_mp,
                                                           quantized_data, G, start_point, QP);
    timings.push_back(t.stop());
    t.reset();

    // Step 3: Re-ranking
    t.start();
    auto final_results =
        parlay::sequence<std::pair<uint32_t, float>>::uninitialized(std::min(k, visited.size()));
    if (!search_params.norerank) {
      size_t num_rerank = std::min(search_params.num_rerank, visited.size());
      this->rerank(query, points, visited, num_rerank, final_results);
      dist_cmps += num_rerank;
    } else {
      parlay::parallel_for(0, final_results.size(),
                           [&](size_t i) { final_results[i] = visited[i]; });
    }
    timings.push_back(t.stop());
    t.reset();

    return std::make_tuple(final_results, dist_cmps, timings);
  }

  // Write the index to a file in disk
  void save(const std::string& filename) override {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("save: cannot open file: " + filename);

    // 1. Save graph
    parlayANN::io::save_graph(G, out);

    // 2. Save Quantizer Type Header
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
      case QT::TurboQuant4Bit:
        std::get<typename SVQT::TQ4_Model>(quantizer).save(out);
        std::get<typename SVQT::TQ4_Range>(quantized_data).save(out);
        break;
      case QT::TurboQuantPQ4Bit: {
        int block_size = params.pq.block_size;
        out.write(reinterpret_cast<const char*>(&block_size), sizeof(block_size));
        if (block_size == 4) {
          std::get<typename SVQT::TQPQ4_Model>(quantizer).save(out);
          std::get<typename SVQT::TQPQ4_Range>(quantized_data).save(out);
        } else {
          std::get<typename SVQT::TQPQ8_Model>(quantizer).save(out);
          std::get<typename SVQT::TQPQ8_Range>(quantized_data).save(out);
        }
        break;
      }
      case QT::None: parlayANN::io::save_point_range(points_mp, out); break;
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
      case QT::TurboQuant4Bit:
        quantizer.template emplace<typename SVQT::TQ4_Model>();
        std::get<typename SVQT::TQ4_Model>(quantizer).load(in);
        quantized_data.template emplace<typename SVQT::TQ4_Range>();
        std::get<typename SVQT::TQ4_Range>(quantized_data).load(in);
        break;
      case QT::TurboQuantPQ4Bit: {
        int tqpq_block_size = 0;
        in.read(reinterpret_cast<char*>(&tqpq_block_size), sizeof(int));
        params.pq.block_size = tqpq_block_size;
        if (tqpq_block_size == 4) {
          quantizer.template emplace<typename SVQT::TQPQ4_Model>();
          std::get<typename SVQT::TQPQ4_Model>(quantizer).load(in);
          quantized_data.template emplace<typename SVQT::TQPQ4_Range>();
          std::get<typename SVQT::TQPQ4_Range>(quantized_data).load(in);
        } else {
          quantizer.template emplace<typename SVQT::TQPQ8_Model>();
          std::get<typename SVQT::TQPQ8_Model>(quantizer).load(in);
          quantized_data.template emplace<typename SVQT::TQPQ8_Range>();
          std::get<typename SVQT::TQPQ8_Range>(quantized_data).load(in);
        }
        break;
      }
      case QT::None: {
        auto [mp_data, loaded_d] = parlayANN::io::read_point_range<Point>(in);
        points_mp = Range(mp_data, loaded_d);
        break;
      }
    }
  }
};

// Compute the mean-pooled vector, given a point cloud
template<typename ChPoint>
std::vector<float> mean_pooling(const ChPoint& point, bool normalize) {
  std::vector<float> mpv(point.get_dims());
  uint32_t point_size = point.size();
  float* coords = point.data();
  uint32_t d = point.get_dims();
  for (size_t j = 0; j < d; j++) {
    auto ent_j =
        parlay::delayed_seq<float>(point_size, [&](size_t k) { return coords[k * d + j]; });
    mpv[j] = parlay::reduce(ent_j) / point_size;
  }
  if (normalize) {
    auto sqrs = parlay::delayed_seq<float>(mpv.size(), [&](size_t j) { return mpv[j] * mpv[j]; });
    float norm = std::sqrt(parlay::reduce(sqrs));
    if (norm > 1e-7) {  // Avoid division by zero
      parlay::parallel_for(0, mpv.size(), [&](size_t j) { mpv[j] /= norm; });
    }
  }
  return mpv;
}

using IndexMPoolL2 = IndexMPool<true>;   // L2 metric
using IndexMPoolIP = IndexMPool<false>;  // MIPS

}  // namespace mvsic