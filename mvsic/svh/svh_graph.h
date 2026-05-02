#pragma once

#include <variant>
#include <unordered_map>
#include <fstream>
#include <iostream>

#include "mvsic/core/index.h"
#include "mvsic/core/query_compression.h"
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
      compressed_storage = compress_query<ChPoint>(query, search_params.query_compression,
                                                   search_params.query_compression_threshold);
      effective_query = compressed_storage.view();
    }
    timings.push_back(t.stop());  // t_compress
    t.reset();

    size_t q_size = effective_query.size();

    // Step 1: Search each query vector independently in the Graph
    t.start();

    auto all_candidates =
        parlay::sequence<std::pair<uint32_t, float>>::uninitialized(q_size * search_params.L);
    auto dist_cmps_seq = parlay::sequence<size_t>::uninitialized(q_size);
    auto bytes_accessed_seq = parlay::sequence<size_t>::uninitialized(q_size);

    auto QP = parlayANN::QueryParams(search_params.L, 2 * search_params.L, search_params.cut,
                                     G.size(), params.ann.R);

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

      size_t count = std::min(search_params.L, visited.size());
      for (size_t j = 0; j < count; ++j) {
        uint32_t cloud_id = vector_to_id[visited[j].first].first;
        all_candidates[i * search_params.L + j] = {cloud_id, visited[j].second};
      }
      for (size_t j = count; j < search_params.L; ++j) {
        all_candidates[i * search_params.L + j] = {UINT32_MAX, std::numeric_limits<float>::max()};
      }
    });

    size_t total_dist_cmps = parlay::reduce(dist_cmps_seq);
    size_t total_bytes_accessed = parlay::reduce(bytes_accessed_seq);
    timings.push_back(t.stop());  // t_graph_search
    t.reset();

    // Step 2: Aggregate across query vectors (Chamfer-style sum-of-mins).
    //
    // Two-level reduction:
    //   1. Per query vector qv: collapse multiple flat-vector hits that belong
    //      to the same cloud c by taking the MIN single-vector distance
    //      (so cloud c contributes at most once per qv).
    //   2. Across query vectors: for each cloud c, accumulate
    //         score[c] = sum over qv that hit c of min_dist(qv, c)
    //      and rank by score[c] (lower = better). No normalization by hit
    //      count, so a cloud must accumulate contributions from multiple qv to
    //      look strong; clouds never hit are skipped.
    t.start();

    std::unordered_map<uint32_t, double> agg;
    agg.reserve(static_cast<size_t>(q_size) * search_params.L);

    for (size_t i = 0; i < q_size; ++i) {
      // Per-query-vector min over duplicate cloud hits.
      std::unordered_map<uint32_t, float> qv_min;
      qv_min.reserve(search_params.L);
      const size_t row = i * search_params.L;
      for (size_t j = 0; j < search_params.L; ++j) {
        auto [cid, dist] = all_candidates[row + j];
        if (cid == UINT32_MAX) break;  // remainder is sentinel padding
        auto it = qv_min.find(cid);
        if (it == qv_min.end()) {
          qv_min.emplace(cid, dist);
        } else if (dist < it->second) {
          it->second = dist;
        }
      }
      // Fold per-qv mins into the global sum accumulator.
      for (const auto& [cid, mn] : qv_min) {
        auto [it, inserted] = agg.try_emplace(cid, static_cast<double>(mn));
        if (!inserted) {
          it->second += static_cast<double>(mn);
        }
      }
    }

    parlay::sequence<std::pair<uint32_t, float>> unique_clouds;
    unique_clouds.reserve(agg.size());
    for (const auto& [cid, score] : agg) {
      unique_clouds.push_back({cid, static_cast<float>(score)});
    }
    parlay::sort_inplace(unique_clouds,
                         [](const auto& a, const auto& b) { return a.second < b.second; });
    if (unique_clouds.size() > num_rerank) unique_clouds.resize(num_rerank);
    timings.push_back(t.stop());  // t_aggregate
    t.reset();

    // Step 3: Final Re-ranking with exact Chamfer Distance
    t.start();
    const ChPoint& rerank_query = search_params.compress_rerank ? effective_query : query;
    auto final_results = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(
        std::min(k, unique_clouds.size()));
    if (!search_params.norerank && !unique_clouds.empty()) {
      size_t actual_rerank_count = std::min(num_rerank, unique_clouds.size());
      if (search_params.tq8_rerank) {
        total_bytes_accessed += this->rerank_tq8_(rerank_query, points, unique_clouds,
                                                  actual_rerank_count, final_results);
      } else {
        total_bytes_accessed +=
            this->rerank(rerank_query, points, unique_clouds, actual_rerank_count, final_results);
      }
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
  // Save / load (v3 uniform skeleton format).
  //
  // The on-disk file is variant-agnostic: any templated SVHGraph variant can
  // load the same skeleton and re-train its leaf quantizer on load from the
  // persisted flattened single-vectors.  class_id is therefore a fixed
  // constant.
  //
  // Layout:
  //   magic           : uint32 = 'SVHG'
  //   version         : uint32 = 3
  //   class_id        : uint32 = 0 (reserved)
  //   d               : unsigned
  //   vector_to_id[]  : flattened (cloud_id, vec_offset) mapping
  //   graph           : parlayANN::io::save_graph payload
  //   flattened_points: raw flattened single-vector data
  //
  // save() is only valid on the raw skeleton variant
  // IndexSVHGraph<metric, NoQuantizer<metric>>; runtime guard below.
  // ---------------------------------------------------------------------------
  static constexpr uint32_t kMagic = 0x53564847u;  // 'SVHG'
  static constexpr uint32_t kVersion = 3u;         // v3: uniform skeleton format
  static constexpr uint32_t kClassId = 0u;

  void save(const std::string& filename) override {
    if constexpr (kLeafMethod != QT::None) {
      std::cerr << "[SVHGraph] save() is only supported on the raw skeleton variant "
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
    parlay::internal::timer t_io;
    t_io.start();
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("load: cannot open file: " + filename);
    uint32_t magic = 0, ver = 0, cid = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&ver), sizeof(ver));
    in.read(reinterpret_cast<char*>(&cid), sizeof(cid));
    if (magic != kMagic) {
      throw std::runtime_error("[SVHGraph] bad magic: file is not a SVHG skeleton index.");
    }
    if (ver != kVersion) {
      throw std::runtime_error("[SVHGraph] SVHGraph index file format changed in v" +
                               std::to_string(kVersion) + "; got v" + std::to_string(ver) +
                               ". Rebuild with current code.");
    }
    if (cid != kClassId) {
      throw std::runtime_error("[SVHGraph] unexpected class_id " + std::to_string(cid) +
                               " (expected " + std::to_string(kClassId) + " for the v" +
                               std::to_string(kVersion) + " skeleton).");
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
    double t_io_ms = t_io.stop() * 1000.0;

    parlay::internal::timer t_retrain;
    t_retrain.start();
    quantization_mode = kLeafMethod;
    if (quantization_mode != QT::None) {
      this->template train_quantizer<SVQT>(flattened_points, quantizer);
      quantized_data = this->template encode_range_quantized<SVQT>(flattened_points, quantizer);
    }
    double t_retrain_ms = t_retrain.stop() * 1000.0;
    std::cerr << "[SVHGraph] load: tree_io=" << t_io_ms << "ms retrain=" << t_retrain_ms
              << "ms (leaf_quant=" << ((kLeafMethod != QT::None) ? 1 : 0) << ")" << std::endl;
  }
};

using IndexSVHGraphIP = IndexSVHGraph<false, NoQuantizer<false>>;
using IndexSVHGraphL2 = IndexSVHGraph<true, NoQuantizer<true>>;
using IndexSVHGraphPQIP = IndexSVHGraph<false, pq::Model<false>>;
using IndexSVHGraphPQL2 = IndexSVHGraph<true, pq::Model<true>>;
using IndexSVHGraphRaBitQIP = IndexSVHGraph<false, rabitq::Model<false>>;
using IndexSVHGraphRaBitQL2 = IndexSVHGraph<true, rabitq::Model<true>>;
using IndexSVHGraphFastScanIP = IndexSVHGraph<false, fastscan::Model<false>>;
using IndexSVHGraphFastScanL2 = IndexSVHGraph<true, fastscan::Model<true>>;
using IndexSVHGraphTQIP = IndexSVHGraph<false, turboquant::Model<false>>;
using IndexSVHGraphTQL2 = IndexSVHGraph<true, turboquant::Model<true>>;
using IndexSVHGraphSPQTQIP = IndexSVHGraph<false, pqtq::Model<false>>;
using IndexSVHGraphSPQTQL2 = IndexSVHGraph<true, pqtq::Model<true>>;

// 1-bit TurboQuant: requires a non-_mv port of turboquant_1bit (only the
// `turboquant_1bit_mv` multi-vector variant exists today).  Uncomment the
// aliases below once a `turboquant_1bit::Model<bool>` SV-variant is ported.
//
// using IndexSVHGraphOneBitTQIP  = IndexSVHGraph<false, turboquant_1bit::Model<false>>;
// using IndexSVHGraphOneBitTQL2  = IndexSVHGraph<true,  turboquant_1bit::Model<true>>;

}  // namespace mvsic