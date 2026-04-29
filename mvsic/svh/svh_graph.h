#pragma once

#include <variant>
#include <unordered_map>
#include <fstream>
#include <iostream>

#include "mvsic/core/index.h"
#include "mvsic/core/query_compression.h"
#include "mvsic/core/quantization/variant_io.h"
#include "mvsic/core/types/io.h"
#include "mvsic/core/utils/util.h"

// ParlayANN (Vamana) includes
#include "algorithms/utils/graph.h"
#include "algorithms/utils/stats.h"
#include "algorithms/utils/types.h"
#include "algorithms/vamana/index.h"

namespace mvsic {

/* IndexSVHGraph: Single-Vector Heuristic with Graph-based Indexing
 *
 * This index flattens all point clouds into a single global pool of vectors.
 * A graph-based index (Vamana via parlayANN) is built over this pool.
 * * Search Strategy:
 * 1. Decompose query point cloud into individual vectors.
 * 2. Search each query vector against the global graph to find top-L neighbors.
 * 3. Map global vector IDs back to parent point cloud IDs.
 * 4. Aggregate votes across all query vectors, deduplicate, and re-rank.
 */
// Template parameter `LeafModel` selects the single-vector quantizer at the
// type level (distinct bindings per variant); the constructor auto-sets
// `params.pq.method` so the existing variant-based dispatch remains correct.
template<bool metric, class LeafModel = NoQuantizer<metric>>
class IndexSVHGraph : public Index<metric> {
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

  // Storage for flattened vectors
  Range flattened_points;

  // Mapping: global_vector_index -> {cloud_id, vector_in_cloud_offset}
  // This allows us to "attribute" a vector neighbor to a specific point cloud.
  parlay::sequence<std::pair<uint32_t, uint32_t>> vector_to_id;

  // Vamana structures from parlayANN
  parlayANN::Graph<uint32_t> G;
  parlayANN::BuildParams BP;
  parlayANN::knn_index<Range, Range, uint32_t> I;
  uint32_t start_point = 0;

  // Quantizer Storage
  QuantModel quantizer = std::monostate{};
  QuantRange quantized_data = std::monostate{};

  IndexSVHGraph(uint32_t d_) noexcept :
      Index<metric>(IndexParams::svh_graph()),
      BP(parlayANN::BuildParams(params.ann.R, params.ann.L, params.ann.alpha, params.ann.num_pass)),
      I(parlayANN::knn_index<Range, Range, uint32_t>(BP)) {
    d = d_;
    if constexpr (kLeafMethod != QT::None) params.pq.method = kLeafMethod;
  }

  IndexSVHGraph(uint32_t d_, const IndexParams& params_) noexcept :
      Index<metric>(params_),
      BP(parlayANN::BuildParams(params.ann.R, params.ann.L, params.ann.alpha, params.ann.num_pass)),
      I(parlayANN::knn_index<Range, Range, uint32_t>(BP)) {
    d = d_;
    if constexpr (kLeafMethod != QT::None) params.pq.method = kLeafMethod;
  }

  // Builds the graph index given a PointCloudSet.
  void build(const PointCloudSet<ChPoint>& points) override {
    parlay::internal::timer t;
    t.start();

    // Step 1: Flatten all point clouds into a contiguous sequence of floats
    if (params.verbose >= 1) std::cout << "[SVHGraph] Flattening point clouds..." << std::endl;
    size_t num_clouds = points.size();
    auto iota = parlay::iota(num_clouds);
    parlay::sequence<parlay::sequence<float>> flat_data = points.filter_flattened(iota);

    // Step 2: Build the mapping from global vector ID to cloud ID
    vector_to_id =
        parlay::sequence<std::pair<uint32_t, uint32_t>>::uninitialized(points.total_size());
    auto num_embs =
        parlay::delayed_tabulate(num_clouds, [&](size_t i) { return points.get_size(i); });
    auto offsets = parlay::scan(num_embs).first;

    parlay::parallel_for(0, num_clouds, [&](size_t i) {
      for (size_t j = 0; j < num_embs[i]; ++j) {
        // Store cloud index and local offset
        vector_to_id[offsets[i] + j] = {static_cast<uint32_t>(i), static_cast<uint32_t>(j)};
      }
    });

    flattened_points = Range(flat_data, d);

    // Step 3: Build Vamana graph on the flattened vectors
    if (params.verbose >= 1) {
      std::cout << "[SVHGraph] Building Vamana on " << flattened_points.size() << " vectors..."
                << std::endl;
    }
    G = parlayANN::Graph<uint32_t>(BP.R, flattened_points.size());
    parlayANN::stats<uint32_t> BuildStats(G.size());
    I.build_index(G, flattened_points, flattened_points, BuildStats);
    // Keep start-point selection consistent between build/load paths.
    I.set_start();
    start_point = I.get_start();

    // Step 4: Quantization — resolved from the compile-time `LeafModel`.
    quantization_mode = kLeafMethod;
    if (quantization_mode != QT::None) {
      if (params.verbose >= 1)
        std::cout << "[SVHGraph] Training and encoding quantizer..." << std::endl;
      this->template train_quantizer<SVQT>(flattened_points, quantizer);
      quantized_data = this->template encode_range_quantized<SVQT>(flattened_points, quantizer);
    }

    if (params.verbose >= 1) {
      std::cout << "[SVHGraph] Build complete. Total time: " << t.stop() << " sec" << std::endl;
    }
  }

  // Multi-Vector Search Logic
  std::tuple<parlay::sequence<std::pair<uint32_t, float>>, size_t, std::vector<double>>
  search_with_stats(const ChPoint& query, const PointCloudSet<ChPoint>& points,
                    const SearchParams& search_params) override {
    parlay::internal::timer t;
    std::vector<double> timings;

    size_t k = search_params.k;
    size_t num_rerank = search_params.num_rerank;

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
    timings.push_back(t.stop());  // t_compress
    t.reset();

    size_t q_size = effective_query.size();

    // Step 1: Search each query vector independently in the Graph
    t.start();

    auto all_candidates =
        parlay::sequence<std::pair<uint32_t, float>>::uninitialized(q_size * num_rerank);
    auto dist_cmps_seq = parlay::sequence<size_t>::uninitialized(q_size);
    auto bytes_accessed_seq = parlay::sequence<size_t>::uninitialized(q_size);

    auto QP = parlayANN::QueryParams(num_rerank, search_params.L, search_params.cut, G.size(),
                                     params.ann.R);

    parlay::parallel_for(0, q_size, [&](size_t i) {
      typename Point::parameters p_params(d);
      Point query_vec_p(reinterpret_cast<typename Point::byte*>(effective_query[i].data()), -1,
                        p_params);

      QuantQuery q_query_var = this->template quantize_query_point<SVQT>(query_vec_p, quantizer);

      parlay::sequence<std::pair<uint32_t, float>> visited;
      size_t comps;
      size_t bytes;
      std::tie(visited, comps, bytes) = this->quant_beam_search(
          query_vec_p, q_query_var, flattened_points, quantized_data, G, start_point, QP);

      dist_cmps_seq[i] = comps;
      bytes_accessed_seq[i] = bytes;

      size_t count = std::min(num_rerank, visited.size());
      for (size_t j = 0; j < count; ++j) {
        uint32_t cloud_id = vector_to_id[visited[j].first].first;
        all_candidates[i * num_rerank + j] = {cloud_id, visited[j].second};
      }
      for (size_t j = count; j < num_rerank; ++j) {
        all_candidates[i * num_rerank + j] = {UINT32_MAX, std::numeric_limits<float>::max()};
      }
    });

    size_t total_dist_cmps = parlay::reduce(dist_cmps_seq);
    size_t total_bytes_accessed = parlay::reduce(bytes_accessed_seq);
    timings.push_back(t.stop());  // t_graph_search
    t.reset();

    // Step 2: Aggregate across query vectors and Deduplicate Cloud IDs
    t.start();
    parlay::sort_inplace(all_candidates,
                         [](const auto& a, const auto& b) { return a.second < b.second; });

    parlay::sequence<std::pair<uint32_t, float>> unique_clouds;
    unique_clouds.reserve(num_rerank);

    int bits = std::max<int>(10, std::ceil(std::log2(num_rerank)) - 2);
    std::vector<uint32_t> hash_filter(1 << bits, -1);
    auto is_duplicate = [&](uint32_t id) -> bool {
      int loc = parlay::hash64_2(id) & ((1 << bits) - 1);
      if (hash_filter[loc] == id) return true;
      hash_filter[loc] = id;
      return false;
    };

    size_t cloud_count = 0;
    for (size_t i = 0; i < all_candidates.size() && cloud_count < num_rerank; ++i) {
      auto [cid, dist] = all_candidates[i];
      if (cid == UINT32_MAX) break;
      if (!is_duplicate(cid)) {
        unique_clouds.push_back({cid, dist});
        cloud_count++;
      }
    }
    timings.push_back(t.stop());  // t_aggregate
    t.reset();

    // Step 3: Final Re-ranking with exact Chamfer Distance
    t.start();
    const ChPoint& rerank_query = search_params.compress_rerank ? effective_query : query;
    auto final_results = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(
        std::min(k, unique_clouds.size()));
    if (!search_params.norerank && !unique_clouds.empty()) {
      size_t actual_rerank_count = std::min(num_rerank, unique_clouds.size());
      total_bytes_accessed +=
          this->rerank(rerank_query, points, unique_clouds, actual_rerank_count, final_results);
      total_dist_cmps += actual_rerank_count;
    } else {
      parlay::parallel_for(0, final_results.size(),
                           [&](size_t i) { final_results[i] = unique_clouds[i]; });
    }
    timings.push_back(t.stop());  // t_rerank

    std::vector<double> stats;
    stats.reserve(timings.size() + 1);
    stats.push_back(static_cast<double>(total_dist_cmps));
    stats.insert(stats.end(), timings.begin(), timings.end());

    return std::make_tuple(final_results, total_bytes_accessed, stats);
  }

  // ---------------------------------------------------------------------------
  // Save / load (v2 skeleton format).
  //
  // Layout:
  //   magic           : uint32 = 'SVHG'
  //   version         : uint32 = 2
  //   class_id        : uint32 = static_cast<uint32_t>(kLeafMethod)
  //   d               : unsigned
  //   vector_to_id[]  : flattened (cloud_id, vec_offset) mapping
  //   graph           : parlayANN::io::save_graph payload
  //   flattened_points: raw flattened single-vector data
  //
  // The leaf quantizer codebook + encoded data are NOT persisted; `load()`
  // re-trains and re-encodes from `flattened_points`.  Internal hot path
  // continues to use variant-based dispatch.
  // ---------------------------------------------------------------------------
  static constexpr uint32_t kMagic = 0x53564847u;  // 'SVHG'
  static constexpr uint32_t kVersion = 2u;
  static uint32_t compute_class_id_() { return static_cast<uint32_t>(kLeafMethod); }

  void save(const std::string& filename) override {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("save: cannot open file: " + filename);
    const uint32_t magic = kMagic;
    const uint32_t ver = kVersion;
    const uint32_t cid = compute_class_id_();
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&ver), sizeof(ver));
    out.write(reinterpret_cast<const char*>(&cid), sizeof(cid));
    out.write(reinterpret_cast<const char*>(&d), sizeof(unsigned));
    size_t map_sz = vector_to_id.size();
    out.write(reinterpret_cast<const char*>(&map_sz), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(vector_to_id.data()),
              map_sz * sizeof(std::pair<uint32_t, uint32_t>));
    parlayANN::io::save_graph(G, out);
    parlayANN::io::save_point_range(flattened_points, out);
    out.close();
  }

  void load(const std::string& filename, const PointCloudSet<ChPoint>& /*points*/) override {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("load: cannot open file: " + filename);
    uint32_t magic = 0, ver = 0, cid = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&ver), sizeof(ver));
    in.read(reinterpret_cast<char*>(&cid), sizeof(cid));
    if (magic != kMagic) {
      throw std::runtime_error("[SVHGraph] bad magic: file is not a SVHG v2 index.");
    }
    if (ver != kVersion) {
      throw std::runtime_error("[SVHGraph] unsupported version " + std::to_string(ver));
    }
    if (cid != compute_class_id_()) {
      throw std::runtime_error("[SVHGraph] class_id mismatch: file has " + std::to_string(cid) +
                               ", this instance is " + std::to_string(compute_class_id_()));
    }
    in.read(reinterpret_cast<char*>(&d), sizeof(unsigned));
    size_t map_sz = 0;
    in.read(reinterpret_cast<char*>(&map_sz), sizeof(size_t));
    vector_to_id.resize(map_sz);
    in.read(reinterpret_cast<char*>(vector_to_id.data()),
            map_sz * sizeof(std::pair<uint32_t, uint32_t>));
    G = parlayANN::io::load_graph<uint32_t>(in);
    I.set_start();
    start_point = I.get_start();
    if (map_sz != G.size()) {
      throw std::runtime_error("load: vector_to_id size does not match graph size");
    }
    auto [f_data, f_d] = parlayANN::io::read_point_range<Point>(in);
    flattened_points = Range(f_data, f_d);
    in.close();

    // Re-train + re-encode leaf quantizer from the loaded flattened vectors.
    quantization_mode = kLeafMethod;
    if (quantization_mode != QT::None) {
      this->template train_quantizer<SVQT>(flattened_points, quantizer);
      quantized_data = this->template encode_range_quantized<SVQT>(flattened_points, quantizer);
    }
  }

  // ---------------------------------------------------------------------------
  // Extended save / load: also persists the trained leaf quantizer + encoded
  // data.  Only loadable by the exact same concrete class.
  // ---------------------------------------------------------------------------
  void save_with_quantizer(const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("save_with_quantizer: cannot open: " + filename);
    const uint32_t magic = kMagic, ver = kVersion, cid = compute_class_id_();
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&ver), sizeof(ver));
    out.write(reinterpret_cast<const char*>(&cid), sizeof(cid));
    out.write(reinterpret_cast<const char*>(&d), sizeof(unsigned));
    size_t map_sz = vector_to_id.size();
    out.write(reinterpret_cast<const char*>(&map_sz), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(vector_to_id.data()),
              map_sz * sizeof(std::pair<uint32_t, uint32_t>));
    parlayANN::io::save_graph(G, out);
    parlayANN::io::save_point_range(flattened_points, out);
    const uint8_t has_q = (kLeafMethod != QT::None) ? 1u : 0u;
    out.write(reinterpret_cast<const char*>(&has_q), sizeof(has_q));
    if (has_q) {
      variant_io::save(quantizer, out);
      variant_io::save(quantized_data, out);
    }
    out.close();
  }

  void load_with_quantizer(const std::string& filename,
                           const PointCloudSet<ChPoint>& /*points*/) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("load_with_quantizer: cannot open: " + filename);
    uint32_t magic = 0, ver = 0, cid = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&ver), sizeof(ver));
    in.read(reinterpret_cast<char*>(&cid), sizeof(cid));
    if (magic != kMagic || ver != kVersion || cid != compute_class_id_()) {
      throw std::runtime_error("[SVHGraph] load_with_quantizer: header mismatch.");
    }
    in.read(reinterpret_cast<char*>(&d), sizeof(unsigned));
    size_t map_sz = 0;
    in.read(reinterpret_cast<char*>(&map_sz), sizeof(size_t));
    vector_to_id.resize(map_sz);
    in.read(reinterpret_cast<char*>(vector_to_id.data()),
            map_sz * sizeof(std::pair<uint32_t, uint32_t>));
    G = parlayANN::io::load_graph<uint32_t>(in);
    I.set_start();
    start_point = I.get_start();
    auto [f_data, f_d] = parlayANN::io::read_point_range<Point>(in);
    flattened_points = Range(f_data, f_d);
    uint8_t has_q = 0;
    in.read(reinterpret_cast<char*>(&has_q), sizeof(has_q));
    quantization_mode = kLeafMethod;
    if (has_q) {
      if (kLeafMethod == QT::None) {
        throw std::runtime_error(
            "[SVHGraph] load_with_quantizer: file has a quantizer payload but "
            "this concrete class is NoQuantizer.");
      }
      variant_io::load_sv<QuantModel,
                          typename SVQT::PQ_Model, typename SVQT::RQ_Model,
                          typename SVQT::FS_Model, typename SVQT::TQ_Model,
                          typename SVQT::PQTQ_Model>(quantizer, in, kLeafMethod);
      variant_io::load_sv<QuantRange,
                          typename SVQT::PQ_Range, typename SVQT::RQ_Range,
                          typename SVQT::FS_Range, typename SVQT::TQ_Range,
                          typename SVQT::PQTQ_Range>(quantized_data, in, kLeafMethod);
    } else if (kLeafMethod != QT::None) {
      this->template train_quantizer<SVQT>(flattened_points, quantizer);
      quantized_data = this->template encode_range_quantized<SVQT>(flattened_points, quantizer);
    }
    in.close();
  }
};

using IndexSVHGraphIP          = IndexSVHGraph<false, NoQuantizer<false>>;
using IndexSVHGraphL2          = IndexSVHGraph<true,  NoQuantizer<true>>;
using IndexSVHGraphPQIP        = IndexSVHGraph<false, pq::Model<false>>;
using IndexSVHGraphPQL2        = IndexSVHGraph<true,  pq::Model<true>>;
using IndexSVHGraphRaBitQIP    = IndexSVHGraph<false, rabitq::Model<false>>;
using IndexSVHGraphRaBitQL2    = IndexSVHGraph<true,  rabitq::Model<true>>;
using IndexSVHGraphFastScanIP  = IndexSVHGraph<false, fastscan::Model<false>>;
using IndexSVHGraphFastScanL2  = IndexSVHGraph<true,  fastscan::Model<true>>;
using IndexSVHGraphTQIP        = IndexSVHGraph<false, turboquant::Model<false>>;
using IndexSVHGraphTQL2        = IndexSVHGraph<true,  turboquant::Model<true>>;
using IndexSVHGraphSPQTQIP     = IndexSVHGraph<false, pqtq::Model<false>>;
using IndexSVHGraphSPQTQL2     = IndexSVHGraph<true,  pqtq::Model<true>>;

// 1-bit TurboQuant: requires a non-_mv port of turboquant_1bit (only the
// `turboquant_1bit_mv` multi-vector variant exists today).  Uncomment the
// aliases below once a `turboquant_1bit::Model<bool>` SV-variant is ported.
//
// using IndexSVHGraphOneBitTQIP  = IndexSVHGraph<false, turboquant_1bit::Model<false>>;
// using IndexSVHGraphOneBitTQL2  = IndexSVHGraph<true,  turboquant_1bit::Model<true>>;

}  // namespace mvsic