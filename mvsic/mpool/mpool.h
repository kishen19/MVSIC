#pragma once

#include <variant>

#include "mvsic/core/index.h"
#include "mvsic/core/query_compression.h"
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
// Template parameter `LeafModel` is used for type-level variant separation
// (so pybind11 sees distinct classes per quantizer), with the constructor
// auto-setting `params.pq.method`; internal hot path retains the variant-based
// dispatch until full compile-time specialization lands.
template<bool metric, class LeafModel = NoQuantizer<metric>>
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

  static constexpr QT kLeafMethod = quantizer_method_of_v<LeafModel, metric>;

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
    if constexpr (kLeafMethod != QT::None) params.pq.method = kLeafMethod;
  }
  IndexMPool(uint32_t d_, const IndexParams& params_) noexcept :
      Index<metric>(params_),
      BP(parlayANN::BuildParams(params.ann.R, params.ann.L, params.ann.alpha, params.ann.num_pass)),
      I(parlayANN::knn_index<Range, Range, uint32_t>(BP)) {
    d = d_;
    if constexpr (kLeafMethod != QT::None) params.pq.method = kLeafMethod;
  }

  // Builds the index given PointCloudSet object.
  void build(const PointCloudSet<ChPoint>& points) override {
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

    // Step3: Quantization — resolved from the compile-time `LeafModel`.
    quantization_mode = kLeafMethod;
    this->train_quantizer(points_mp, quantizer);
    quantized_data = this->encode_range_quantized(points_mp, quantizer);
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
    double t_mean_pooling = 0.0;
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

    // Step 1: Compute mean-pooling of the (possibly compressed) query
    t.start();
    std::vector<float> query_mpv = mean_pooling(effective_query, false);
    typename Point::parameters parlayann_pr_params(d);
    Point query_point(reinterpret_cast<typename Point::byte*>(query_mpv.data()), -1,
                      parlayann_pr_params);
    t_mean_pooling = t.stop();
    t.reset();

    // Step 2: Run beam search
    uint32_t start_point = I.get_start();
    auto QP = parlayANN::QueryParams(search_params.num_rerank, search_params.L, search_params.cut,
                                     points.size(), params.ann.R);
    parlay::sequence<std::pair<uint32_t, float>> visited;

    t.start();
    QuantQuery q_query_var = this->quantize_query_point(query_point, quantizer);
    t_quantize = t.stop();
    t.reset();

    t.start();
    std::tie(visited, dist_cmps, bytes_accessed) = this->quant_beam_search(
        query_point, q_query_var, points_mp, quantized_data, G, start_point, QP);
    t_search = t.stop();
    t.reset();

    // Step 3: Re-ranking
    t.start();
    const ChPoint& rerank_query = search_params.compress_rerank ? effective_query : query;
    auto final_results =
        parlay::sequence<std::pair<uint32_t, float>>::uninitialized(std::min(k, visited.size()));
    if (!search_params.norerank) {
      size_t num_rerank = std::min(search_params.num_rerank, visited.size());
      bytes_accessed += this->rerank(rerank_query, points, visited, num_rerank, final_results);
    } else {
      parlay::parallel_for(0, final_results.size(),
                           [&](size_t i) { final_results[i] = visited[i]; });
    }
    t_rerank = t.stop();
    t.reset();

    std::vector<double> stats;
    stats.push_back(t_compress);
    stats.push_back(t_mean_pooling);
    stats.push_back(t_quantize);
    stats.push_back(t_search);
    stats.push_back(t_rerank);

    return std::make_tuple(final_results, bytes_accessed, stats);
  }

  // ---------------------------------------------------------------------------
  // Save / load (v3 uniform skeleton format).
  //
  // The on-disk file is variant-agnostic: any templated MPool variant can
  // load the same skeleton and re-train its leaf quantizer on load from the
  // persisted mean-pooled vectors.  class_id is therefore a fixed constant.
  //
  // Layout:
  //   magic       : uint32 = 'MPOO'
  //   version     : uint32 = 3
  //   class_id    : uint32 = 0 (reserved)
  //   graph       : parlayANN::io::save_graph payload
  //   points_mp   : raw mean-pooled single-vector data (via save_point_range)
  //
  // save() is only valid on the raw skeleton variant
  // IndexMPool<metric, NoQuantizer<metric>>; runtime guard below.
  // ---------------------------------------------------------------------------
  static constexpr uint32_t kMagic = 0x4D504F4Fu;  // 'MPOO'
  static constexpr uint32_t kVersion = 3u;         // v3: uniform skeleton format
  static constexpr uint32_t kClassId = 0u;

  void save(const std::string& filename) override {
    if constexpr (kLeafMethod != QT::None) {
      std::cerr << "[MPool] save() is only supported on the raw skeleton variant "
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
    parlayANN::io::save_graph(G, out);
    parlayANN::io::save_point_range(points_mp, out);
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
      throw std::runtime_error("[MPool] bad magic: file is not an MPOO skeleton index.");
    }
    if (ver != kVersion) {
      throw std::runtime_error(
          "[MPool] MPool index file format changed in v" + std::to_string(kVersion) +
          "; got v" + std::to_string(ver) + ". Rebuild with current code.");
    }
    if (cid != kClassId) {
      throw std::runtime_error(
          "[MPool] unexpected class_id " + std::to_string(cid) + " (expected " +
          std::to_string(kClassId) + " for the v" + std::to_string(kVersion) + " skeleton).");
    }
    G = parlayANN::io::load_graph<uint32_t>(in);
    I.set_start();
    auto [mp_data, loaded_d] = parlayANN::io::read_point_range<Point>(in);
    points_mp = Range(mp_data, loaded_d);
    in.close();
    double t_io_ms = t_io.stop() * 1000.0;

    parlay::internal::timer t_retrain;
    t_retrain.start();
    quantization_mode = kLeafMethod;
    this->train_quantizer(points_mp, quantizer);
    quantized_data = this->encode_range_quantized(points_mp, quantizer);
    double t_retrain_ms = t_retrain.stop() * 1000.0;
    std::cerr << "[MPool] load: tree_io=" << t_io_ms << "ms retrain=" << t_retrain_ms
              << "ms (leaf_quant=" << ((kLeafMethod != QT::None) ? 1 : 0) << ")" << std::endl;
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

using IndexMPoolIP          = IndexMPool<false, NoQuantizer<false>>;
using IndexMPoolL2          = IndexMPool<true,  NoQuantizer<true>>;
using IndexMPoolPQIP        = IndexMPool<false, pq::Model<false>>;
using IndexMPoolPQL2        = IndexMPool<true,  pq::Model<true>>;
using IndexMPoolRaBitQIP    = IndexMPool<false, rabitq::Model<false>>;
using IndexMPoolRaBitQL2    = IndexMPool<true,  rabitq::Model<true>>;
using IndexMPoolFastScanIP  = IndexMPool<false, fastscan::Model<false>>;
using IndexMPoolFastScanL2  = IndexMPool<true,  fastscan::Model<true>>;
using IndexMPoolTQIP        = IndexMPool<false, turboquant::Model<false>>;
using IndexMPoolTQL2        = IndexMPool<true,  turboquant::Model<true>>;
using IndexMPoolSPQTQIP     = IndexMPool<false, pqtq::Model<false>>;
using IndexMPoolSPQTQL2     = IndexMPool<true,  pqtq::Model<true>>;

// 1-bit TurboQuant: requires a non-_mv port of turboquant_1bit (only the
// `turboquant_1bit_mv` multi-vector variant exists today).  Uncomment the
// aliases below once a `turboquant_1bit::Model<bool>` SV-variant is ported.
//
// using IndexMPoolOneBitTQIP  = IndexMPool<false, turboquant_1bit::Model<false>>;
// using IndexMPoolOneBitTQL2  = IndexMPool<true,  turboquant_1bit::Model<true>>;

}  // namespace mvsic