#pragma once

// =============================================================================
// Multi-Vector IVF (Flat) — templated index.
//
// Single-level IVF: cluster points into `num_clusters` groups, score the query
// against all centers, probe the top `nprobes` groups.  Templated on the same
// <metric, CompressCenters, LeafModel> axes as IndexMVIVF; see mvivf.h for the
// design rationale.
// =============================================================================

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

#include "mvsic/core/index.h"
#include "mvsic/core/mvclustering/mvclustering.h"
#include "mvsic/core/mvclustering/mvclustering_8bit.h"
#include "mvsic/core/query_compression.h"
#include "mvsic/core/utils/util.h"

namespace mvsic {

namespace mvivf_flat_internal { struct Empty {}; }

template<bool metric, bool CompressCenters = false,
         class LeafModel = NoQuantizer<metric>>
class IndexMVIVFFlat : public Index<metric> {
 public:
  using ChPoint = typename Index<metric>::ChPoint;
  using QT = typename Index<metric>::QT;
  using Index<metric>::d;
  using Index<metric>::params;
  using Index<metric>::quantization_mode;

  using CenterModel = turboquant_mv::Model<metric>;
  using CenterSet = typename CenterModel::EncodedSet;
  using CenterQuery = typename CenterModel::EncodedQuery;

  using LeafSet = typename LeafModel::EncodedSet;
  using LeafQuery = typename LeafModel::EncodedQuery;
  using LeafParams = typename LeafModel::Params;

  static constexpr bool kHasLeafQuant =
      !std::is_same_v<LeafModel, NoQuantizer<metric>>;
  static constexpr bool kHasCenterQuant = CompressCenters;

  // Cluster node: raw data (for rerank) + optionally encoded leaf.
  struct node_t {
    PointCloudSet<ChPoint> data;
    [[no_unique_address]]
    std::conditional_t<kHasLeafQuant, LeafSet, mvivf_flat_internal::Empty> encoded_leaf;
    node_t() noexcept = default;
    inline size_t get_size() const noexcept { return data.size(); }
  };

  PointCloudSet<ChPoint> centers;  // raw centers when !CompressCenters
  [[no_unique_address]]
  std::conditional_t<kHasCenterQuant, CenterSet, mvivf_flat_internal::Empty> centers_encoded;
  parlay::sequence<node_t> clusters;

  [[no_unique_address]]
  std::conditional_t<kHasCenterQuant, CenterModel, mvivf_flat_internal::Empty> center_model_;
  [[no_unique_address]]
  std::conditional_t<kHasLeafQuant, LeafModel, mvivf_flat_internal::Empty> leaf_model_;
  [[no_unique_address]]
  std::conditional_t<kHasLeafQuant, LeafParams, mvivf_flat_internal::Empty> leaf_params_;

  IndexMVIVFFlat(uint32_t d_) noexcept {
    d = d_;
    params = IndexParams::mvivf_flat();
    set_quant_mode_();
  }
  IndexMVIVFFlat(uint32_t d_, const IndexParams& p) noexcept {
    d = d_;
    params = p;
    set_quant_mode_();
  }
  template <class LP = LeafParams,
            std::enable_if_t<kHasLeafQuant && std::is_same_v<LP, LeafParams>, int> = 0>
  IndexMVIVFFlat(uint32_t d_, const IndexParams& p, const LP& lp) noexcept {
    d = d_;
    params = p;
    if constexpr (kHasLeafQuant) leaf_params_ = lp;
    set_quant_mode_();
  }

  inline uint32_t get_size(size_t i) const noexcept { return clusters[i].get_size(); }
  inline size_t num_leaves() const noexcept {
    if constexpr (kHasCenterQuant) return clusters.size();
    else return centers.size();
  }

  void set_quant_mode_() {
    if constexpr (!kHasLeafQuant) {
      quantization_mode = QT::None;
    } else {
      using L = LeafModel;
      if constexpr (std::is_same_v<L, pq_mv::Model<metric>>) quantization_mode = QT::PQ;
      else if constexpr (std::is_same_v<L, rabitq_mv::Model<metric>>) quantization_mode = QT::RaBitQ;
      else if constexpr (std::is_same_v<L, fastscan_mv::Model<metric>>) quantization_mode = QT::FastScan;
      else if constexpr (std::is_same_v<L, turboquant_mv::Model<metric>>) quantization_mode = QT::TurboQuant;
      else if constexpr (std::is_same_v<L, pqtq_mv::Model<metric>>) quantization_mode = QT::SPQTQ;
      else if constexpr (std::is_same_v<L, turboquant_1bit_mv::Model<metric>>) quantization_mode = QT::OneBitTQ;
      else if constexpr (std::is_same_v<L, turboquant_8bit_mv::Model<metric>>) quantization_mode = QT::EightBitTQ;
      else quantization_mode = QT::None;
    }
  }

  // ---------------------------------------------------------------------------
  // Build.
  // ---------------------------------------------------------------------------
  void build(const PointCloudSet<ChPoint>& points) override {
    parlay::internal::timer t;
    const size_t n = points.size();
    const size_t num_clusters =
        (params.k_per_level > 0) ? params.k_per_level
                                 : static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(n))));
    if (params.verbose >= 1) {
      std::cout << "Building index with " << n << " points, num_clusters: " << num_clusters
                << std::endl;
    }
    // Propagate the build-time 8BTQ flag into MVClusteringConfig so the inner
    // Lloyd's k-means picks the TQ8 backend in kmeans_subsample/weighted.
    params.mvclus.build_with_8btq = params.build_with_8btq;

    t.start();
    if constexpr (kHasCenterQuant) center_model_.train(points);
    if constexpr (kHasLeafQuant) leaf_model_.train(points, leaf_params_);
    if (params.verbose >= 1 && (kHasCenterQuant || kHasLeafQuant)) {
      std::cout << "[MVIVF Flat] Quantizers Trained: " << t.stop() << " sec" << std::endl;
    }
    t.reset();

    t.start();
    parlay::sequence<uint32_t> cluster_ids;
    if (params.build_with_8btq) {
      MVClustering8BTQ<metric> Clus(d, num_clusters, params.s, params.mvclus);
      Clus.train(points);
      cluster_ids = Clus.get_clustering(points);
      centers = std::move(Clus.get_centers());
    } else {
      MVClustering<metric> Clus(d, num_clusters, params.s, params.mvclus);
      Clus.train(points);
      cluster_ids = Clus.get_clustering(points);
      centers = std::move(Clus.get_centers());
    }
    auto id_pt = parlay::tabulate(n, [&](uint32_t i) {
      return std::make_pair(cluster_ids[i], i);
    });
    auto grouped = group_by_key_inplace(id_pt);
    clusters.resize(num_clusters);
    parlay::parallel_for(0, grouped.size(), [&](size_t i) {
      auto cluster_id = grouped[i][0].first;
      auto group = parlay::delayed_seq<uint32_t>(
          grouped[i].size(), [&](size_t j) { return grouped[i][j].second; });
      PointCloudSet<ChPoint> cluster_points(points.filter(group), d);
      clusters[cluster_id].data = std::move(cluster_points);
      if constexpr (kHasLeafQuant) {
        clusters[cluster_id].encoded_leaf = leaf_model_.encode(clusters[cluster_id].data);
      }
    }, 1);
    if (params.verbose >= 1) {
      std::cout << "[MVIVF Flat] Index Built: " << t.stop() << " sec" << std::endl;
      std::cout << "[MVIVF Flat] Mean cluster size: " << mean_cluster_size()
                << ", Max cluster size: " << max_cluster_size() << std::endl;
    }
    t.reset();

    if constexpr (kHasCenterQuant) {
      t.start();
      centers_encoded = center_model_.encode(centers);
      // Keep the raw `centers` populated alongside `centers_encoded` so that
      // save() can persist the exact float centers used during build.
      std::cout << "[MVIVF Flat] Encoding Centers: " << t.stop() << " sec" << std::endl;
    }
  }

  // ---------------------------------------------------------------------------
  // Probe scoring (Step 2).
  // ---------------------------------------------------------------------------
  inline std::pair<parlay::sequence<std::pair<uint32_t, float>>, size_t>
  process_probes(const ChPoint& query, const LeafQuery& q_leaf,
                 parlay::sequence<std::pair<uint32_t, float>>& probe_list) {
    const size_t nprobes = probe_list.size();
    auto sizes = parlay::delayed_tabulate(nprobes, [&](size_t i) {
      return clusters[probe_list[i].first].get_size();
    });
    auto scan_result = parlay::scan(sizes);
    auto& offsets = scan_result.first;
    size_t total = scan_result.second;
    auto visited = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(total);
    auto bytes = parlay::sequence<size_t>::uninitialized(nprobes);
    parlay::parallel_for(0, nprobes, [&](size_t i) {
      node_t& node = clusters[probe_list[i].first];
      if constexpr (kHasLeafQuant) {
        node.encoded_leaf.distances_all(q_leaf, &visited[offsets[i]]);
        bytes[i] = node.encoded_leaf.num_bytes();
      } else {
        (void)q_leaf;
        node.data.distances(query, &visited[offsets[i]]);
        bytes[i] = node.data.num_bytes();
      }
    });
    return std::make_pair(std::move(visited), parlay::reduce(bytes));
  }

  // ---------------------------------------------------------------------------
  // Single-query search with detailed stats.
  //
  // Timer labels (matches bench `is_flat` path):
  //   0  n_centers
  //   1  probe_cmps
  //   2  t_compress
  //   3  t_search
  //   4  t_leaf_dists
  //   5  t_leaf_rest
  //   6  t_rerank
  // ---------------------------------------------------------------------------
  std::tuple<parlay::sequence<std::pair<uint32_t, float>>, size_t, std::vector<double>>
  search_with_stats(const ChPoint& query, const PointCloudSet<ChPoint>& points,
                    const SearchParams& search_params) override {
    parlay::internal::timer t;
    const size_t k = search_params.k;
    size_t nprobes = search_params.nprobes;
    size_t bytes_accessed = 0;
    size_t dist_cmps = 0;
    double t_compress = 0.0, t_search = 0.0;
    double t_distances = 0.0, t_rest = 0.0, t_rerank = 0.0;

    // Step -1: query compression.
    t.start();
    CompressedPointCloud<ChPoint> compressed_storage;
    ChPoint effective_query = query;
    if (search_params.query_compression != SearchParams::QueryCompression::None) {
      uint32_t ba = 1;
      if constexpr (kHasLeafQuant) ba = LeafModel::kBatchAlignment;
      compressed_storage = compress_query<ChPoint>(
          query, search_params.query_compression,
          search_params.query_compression_threshold, ba);
      effective_query = compressed_storage.view();
    }
    t_compress = t.stop(); t.reset();

    // Step 0: quantize query.
    LeafQuery q_leaf{};
    CenterQuery q_center{};
    if constexpr (kHasLeafQuant) q_leaf = leaf_model_.quantize_query(effective_query);
    if constexpr (kHasCenterQuant) q_center = center_model_.quantize_query(effective_query);

    // Step 1: score against all centers, take top-nprobes.
    t.start();
    const size_t L = num_leaves();
    nprobes = std::min(nprobes, L);
    auto probe_list = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(L);
    if constexpr (kHasCenterQuant) {
      centers_encoded.distances_all(q_center, probe_list.data());
      bytes_accessed += centers_encoded.num_bytes();
    } else {
      centers.distances(effective_query, probe_list.data());
      bytes_accessed += centers.num_bytes();
    }
    std::nth_element(probe_list.begin(), probe_list.begin() + nprobes, probe_list.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });
    probe_list.resize(nprobes);
    t_search = t.stop();
    dist_cmps += L;
    t.reset();

    // Step 2: probe.
    t.start();
    parlay::sequence<std::pair<uint32_t, float>> visited;
    size_t bytes_pp;
    std::tie(visited, bytes_pp) = process_probes(effective_query, q_leaf, probe_list);
    bytes_accessed += bytes_pp;
    t_distances = t.stop(); t.reset();

    t.start();
    mvsic::sort_inplace_kv(visited);
    t_rest += t.stop(); t.reset();
    dist_cmps += visited.size();

    // Step 3: rerank.
    const ChPoint& rerank_query = search_params.compress_rerank ? effective_query : query;
    t.start();
    auto final_results = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(
        std::min(k, visited.size()));
    if (search_params.num_rerank > 0) {
      size_t num_rerank = std::min(search_params.num_rerank, visited.size());
      bytes_accessed += this->rerank(rerank_query, points, visited, num_rerank, final_results);
    } else {
      parlay::parallel_for(0, final_results.size(),
                           [&](size_t i) { final_results[i] = visited[i]; });
    }
    t_rerank = t.stop(); t.reset();

    std::vector<double> stats;
    stats.push_back(static_cast<double>(L));
    stats.push_back(static_cast<double>(dist_cmps));
    stats.push_back(t_compress);
    stats.push_back(t_search);
    stats.push_back(t_distances);
    stats.push_back(t_rest);
    stats.push_back(t_rerank);
    return std::make_tuple(std::move(final_results), bytes_accessed, std::move(stats));
  }

  // ---------------------------------------------------------------------------
  // Save / load (v3 skeleton format).
  //
  // Quantization-agnostic skeleton: writes magic + version + class_id +
  // IndexParams subset + cluster count + raw center floats + cluster point
  // ids.  Codebooks and encoded leaves are never persisted; any templated
  // variant can load any skeleton from this family and re-train / re-encode
  // on load using the supplied raw points.  class_id is therefore fixed.
  //
  // save() requires raw centers in `centers`, i.e. it is only valid on the
  // skeleton variant IndexMVIVFFlat<metric, /*CompressCenters=*/false,
  // NoQuantizer<metric>>.  Use the raw skeleton variant to build+save, then
  // load() into the desired templated variant.
  // ---------------------------------------------------------------------------
  static constexpr uint32_t kMagic = 0x4D464C46u;  // 'MFLF'
  static constexpr uint32_t kVersion = 3u;  // v3: uniform skeleton format (no quant on disk)
  static constexpr uint32_t kClassId = 0u;

 private:
  void write_params_(std::ostream& out) const {
    auto w = [&](auto x) { out.write(reinterpret_cast<const char*>(&x), sizeof(x)); };
    w(params.k_per_level);
    w(params.s);
    w(params.mvclus.niters);
    w(params.mvclus.max_point_clouds_per_cluster);
    w(params.mvclus.max_points_per_centroid_inner_kmeans);
  }
  void read_params_(std::istream& in) {
    auto r = [&](auto& x) { in.read(reinterpret_cast<char*>(&x), sizeof(x)); };
    r(params.k_per_level);
    r(params.s);
    r(params.mvclus.niters);
    r(params.mvclus.max_point_clouds_per_cluster);
    r(params.mvclus.max_points_per_centroid_inner_kmeans);
  }

 public:
  void save(const std::string& filename) override {
    // Skeleton save persists raw centers + cluster point ids only.  Codebooks
    // and encoded leaves are never written; any templated variant can load
    // this file and re-derive its quantization on load using the supplied raw
    // points.  Quantized variants now keep `centers` populated alongside
    // `centers_encoded` (see build()), so save() is valid for all variants.
    std::ofstream outfile(filename, std::ios::binary);
    std::cout << "Saving index to " << filename << std::endl;
    if (!outfile.is_open()) {
      std::cerr << "Error opening file for writing: " << filename << std::endl;
      return;
    }
    const uint32_t magic = kMagic, ver = kVersion, cid = kClassId;
    outfile.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    outfile.write(reinterpret_cast<const char*>(&ver), sizeof(ver));
    outfile.write(reinterpret_cast<const char*>(&cid), sizeof(cid));
    write_params_(outfile);

    size_t num = clusters.size();
    outfile.write(reinterpret_cast<const char*>(&num), sizeof(size_t));

    parlay::sequence<size_t> center_offsets = parlay::sequence<size_t>::from_function(
        centers.size(), [&](size_t i) { return centers.get_size(i) * d; });
    size_t total = parlay::scan_inplace(center_offsets);
    center_offsets.push_back(total);
    outfile.write(reinterpret_cast<const char*>(center_offsets.begin()),
                  center_offsets.size() * sizeof(size_t));
    auto coords = centers.data();
    size_t nent = centers.total_size() * centers.get_dims();
    outfile.write(reinterpret_cast<const char*>(coords), nent * sizeof(float));

    parlay::sequence<size_t> clusters_offsets = parlay::sequence<size_t>::from_function(
        clusters.size(), [&](size_t i) { return clusters[i].get_size(); });
    size_t total_cluster = parlay::scan_inplace(clusters_offsets);
    clusters_offsets.push_back(total_cluster);
    outfile.write(reinterpret_cast<const char*>(clusters_offsets.begin()),
                  clusters_offsets.size() * sizeof(size_t));
    for (size_t i = 0; i < num; ++i) {
      size_t sz = clusters[i].get_size();
      for (size_t j = 0; j < sz; ++j) {
        uint32_t pid = clusters[i].data.get_id(j);
        outfile.write(reinterpret_cast<const char*>(&pid), sizeof(uint32_t));
      }
    }
    outfile.close();
  }

  void load(const std::string& filename, const PointCloudSet<ChPoint>& points) override {
    parlay::internal::timer t_io;
    t_io.start();
    std::ifstream infile(filename, std::ios::binary);
    std::cout << "Loading index from " << filename << std::endl;
    if (!infile.is_open()) {
      std::cerr << "Error opening file for reading: " << filename << std::endl;
      return;
    }
    uint32_t magic = 0, ver = 0, cid = 0;
    infile.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    infile.read(reinterpret_cast<char*>(&ver), sizeof(ver));
    infile.read(reinterpret_cast<char*>(&cid), sizeof(cid));
    if (magic != kMagic) {
      std::cerr << "[MVIVF Flat] bad magic: file is not an MFLF skeleton index." << std::endl;
      return;
    }
    if (ver != kVersion) {
      std::cerr << "[MVIVF Flat] MVIVF Flat index file format changed in v" << kVersion
                << "; got v" << ver << ". Rebuild with current code." << std::endl;
      return;
    }
    if (cid != kClassId) {
      std::cerr << "[MVIVF Flat] unexpected class_id " << cid << " (expected " << kClassId
                << " for the v" << kVersion << " skeleton)." << std::endl;
      return;
    }
    read_params_(infile);

    size_t num = 0;
    infile.read(reinterpret_cast<char*>(&num), sizeof(size_t));
    parlay::sequence<size_t> center_offsets(num + 1);
    infile.read(reinterpret_cast<char*>(center_offsets.begin()),
                center_offsets.size() * sizeof(size_t));
    parlay::sequence<float> center_values(center_offsets[center_offsets.size() - 1]);
    infile.read(reinterpret_cast<char*>(center_values.begin()),
                center_values.size() * sizeof(float));
    PointCloudSet<ChPoint> raw_centers(num, points.get_dims(), center_values.data(),
                                       center_offsets.data(), nullptr);
    parlay::sequence<size_t> clusters_offsets(num + 1);
    infile.read(reinterpret_cast<char*>(clusters_offsets.begin()),
                clusters_offsets.size() * sizeof(size_t));
    parlay::sequence<uint32_t> clusters_values(clusters_offsets[clusters_offsets.size() - 1]);
    infile.read(reinterpret_cast<char*>(clusters_values.begin()),
                clusters_values.size() * sizeof(uint32_t));
    infile.close();
    double t_io_ms = t_io.stop() * 1000.0;

    parlay::internal::timer t_retrain;
    t_retrain.start();
    if constexpr (kHasCenterQuant) center_model_.train(points);
    if constexpr (kHasLeafQuant) leaf_model_.train(points, leaf_params_);

    if constexpr (kHasCenterQuant) {
      // Encode the persisted raw centers directly (no re-clustering): this
      // mirrors the on-build flow in mvivf.h::compress_internal_centers_.
      // Keep `centers` populated alongside `centers_encoded` so that a
      // subsequent save() round-trip preserves the exact float centers.
      centers_encoded = center_model_.encode(raw_centers);
      centers = std::move(raw_centers);
    } else {
      centers = std::move(raw_centers);
    }

    const size_t dim = points.get_dims();
    auto point_id_to_data_id = parlay::sequence<uint32_t>::uninitialized(points.size());
    parlay::parallel_for(0, points.size(),
        [&](uint32_t i) { point_id_to_data_id[points.get_id(i)] = i; });
    parlay::sequence<size_t> cluster_sizes = parlay::sequence<size_t>::from_function(
        num, [&](size_t i) { return clusters_offsets[i + 1] - clusters_offsets[i]; });
    clusters.resize(num);
    parlay::parallel_for(0, num, [&](size_t i) {
      if (cluster_sizes[i] > 0) {
        auto cluster_group = parlay::delayed_seq<uint32_t>(cluster_sizes[i], [&](size_t j) {
          uint32_t pid = clusters_values[clusters_offsets[i] + j];
          return point_id_to_data_id[pid];
        });
        clusters[i].data = PointCloudSet<ChPoint>(points.filter(cluster_group), dim);
        if constexpr (kHasLeafQuant) {
          clusters[i].encoded_leaf = leaf_model_.encode(clusters[i].data);
        }
      }
    });
    double t_retrain_ms = t_retrain.stop() * 1000.0;
    std::cerr << "[MVIVF Flat] load: tree_io=" << t_io_ms << "ms retrain=" << t_retrain_ms
              << "ms (compress_centers=" << (kHasCenterQuant ? 1 : 0)
              << ", leaf_quant=" << (kHasLeafQuant ? 1 : 0) << ")" << std::endl;
  }

  size_t mean_cluster_size() const noexcept override {
    auto cs = parlay::delayed_seq<size_t>(clusters.size(),
        [&](size_t i) { return clusters[i].get_size(); });
    return clusters.size() > 0 ? parlay::reduce(cs) / clusters.size() : 0;
  }
  size_t max_cluster_size() const noexcept override {
    auto cs = parlay::delayed_seq<size_t>(clusters.size(),
        [&](size_t i) { return clusters[i].get_size(); });
    return parlay::reduce(cs, parlay::maxm<size_t>());
  }
};

// Concrete aliases.
using IndexMVIVFFlatIP          = IndexMVIVFFlat<false, false, NoQuantizer<false>>;
using IndexMVIVFFlatL2          = IndexMVIVFFlat<true,  false, NoQuantizer<true>>;
using IndexMVIVFFlatCompressIP  = IndexMVIVFFlat<false, true,  NoQuantizer<false>>;
using IndexMVIVFFlatCompressL2  = IndexMVIVFFlat<true,  true,  NoQuantizer<true>>;

using IndexMVIVFFlatPQIP        = IndexMVIVFFlat<false, false, pq_mv::Model<false>>;
using IndexMVIVFFlatPQL2        = IndexMVIVFFlat<true,  false, pq_mv::Model<true>>;
using IndexMVIVFFlatFastScanIP  = IndexMVIVFFlat<false, false, fastscan_mv::Model<false>>;
using IndexMVIVFFlatFastScanL2  = IndexMVIVFFlat<true,  false, fastscan_mv::Model<true>>;
using IndexMVIVFFlatRaBitQIP    = IndexMVIVFFlat<false, false, rabitq_mv::Model<false>>;
using IndexMVIVFFlatRaBitQL2    = IndexMVIVFFlat<true,  false, rabitq_mv::Model<true>>;
using IndexMVIVFFlatTQIP        = IndexMVIVFFlat<false, false, turboquant_mv::Model<false>>;
using IndexMVIVFFlatTQL2        = IndexMVIVFFlat<true,  false, turboquant_mv::Model<true>>;
using IndexMVIVFFlatSPQTQIP     = IndexMVIVFFlat<false, false, pqtq_mv::Model<false>>;
using IndexMVIVFFlatSPQTQL2     = IndexMVIVFFlat<true,  false, pqtq_mv::Model<true>>;
using IndexMVIVFFlatOneBitTQIP  = IndexMVIVFFlat<false, false, turboquant_1bit_mv::Model<false>>;
using IndexMVIVFFlatOneBitTQL2  = IndexMVIVFFlat<true,  false, turboquant_1bit_mv::Model<true>>;
using IndexMVIVFFlatEightBitTQIP = IndexMVIVFFlat<false, false, turboquant_8bit_mv::Model<false>>;
using IndexMVIVFFlatEightBitTQL2 = IndexMVIVFFlat<true,  false, turboquant_8bit_mv::Model<true>>;

// CompressCenters (TQ-quantized centers) + LeafModel combinations.
using IndexMVIVFFlatCompressPQIP        = IndexMVIVFFlat<false, true, pq_mv::Model<false>>;
using IndexMVIVFFlatCompressPQL2        = IndexMVIVFFlat<true,  true, pq_mv::Model<true>>;
using IndexMVIVFFlatCompressFastScanIP  = IndexMVIVFFlat<false, true, fastscan_mv::Model<false>>;
using IndexMVIVFFlatCompressFastScanL2  = IndexMVIVFFlat<true,  true, fastscan_mv::Model<true>>;
using IndexMVIVFFlatCompressRaBitQIP    = IndexMVIVFFlat<false, true, rabitq_mv::Model<false>>;
using IndexMVIVFFlatCompressRaBitQL2    = IndexMVIVFFlat<true,  true, rabitq_mv::Model<true>>;
using IndexMVIVFFlatCompressTQIP        = IndexMVIVFFlat<false, true, turboquant_mv::Model<false>>;
using IndexMVIVFFlatCompressTQL2        = IndexMVIVFFlat<true,  true, turboquant_mv::Model<true>>;
using IndexMVIVFFlatCompressSPQTQIP     = IndexMVIVFFlat<false, true, pqtq_mv::Model<false>>;
using IndexMVIVFFlatCompressSPQTQL2     = IndexMVIVFFlat<true,  true, pqtq_mv::Model<true>>;
using IndexMVIVFFlatCompressOneBitTQIP  = IndexMVIVFFlat<false, true, turboquant_1bit_mv::Model<false>>;
using IndexMVIVFFlatCompressOneBitTQL2  = IndexMVIVFFlat<true,  true, turboquant_1bit_mv::Model<true>>;
using IndexMVIVFFlatCompressEightBitTQIP = IndexMVIVFFlat<false, true, turboquant_8bit_mv::Model<false>>;
using IndexMVIVFFlatCompressEightBitTQL2 = IndexMVIVFFlat<true,  true, turboquant_8bit_mv::Model<true>>;

}  // namespace mvsic
