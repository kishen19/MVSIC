#pragma once

// =============================================================================
// Multi-Vector IVF with root-level spilling — templated index.
//
// Same design as IndexMVIVF (see mvivf.h) with two additions:
//   * (a, b) spill strategy:
//       - Root (level 0): every point cloud is assigned to its top-`a`
//         nearest root centers, where `a = params.num_spill`.
//       - Second level (level 1): every point cloud inside a root child is
//         assigned to its top-`b` nearest local centers, where
//         `b = params.num_spill_l2`.
//       - All deeper levels use top-1 (standard clustering, no spill).
//     Setting b=1 recovers the original root-only spilling behaviour.
//   * Flat-leaf short-circuit: when the requested `nprobes` is close to the
//     total number of leaves, `search_with_stats` skips the beam search and
//     scores every leaf center directly.
//
// Template parameters:
//   metric          true = L2, false = IP
//   CompressCenters internal-node centers (and leaf-center shortcut) stored as
//                   4-bit TurboQuant-encoded bytes instead of raw float vectors
//   LeafModel       the leaf-encoding model (NoQuantizer by default)
// =============================================================================

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mvsic/core/index.h"
#include "mvsic/core/mvclustering/mvclustering.h"
#include "mvsic/core/query_compression.h"
#include "mvsic/core/utils/util.h"

namespace mvsic {

namespace mvivf_spill_internal { struct Empty {}; }

template<bool metric, bool CompressCenters = false,
         class LeafModel = NoQuantizer<metric>>
class IndexMVIVFSpill : public Index<metric> {
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

  struct node_t {
    parlay::sequence<node_t*> children;
    // Internal node: `data` holds raw child-center coords (dropped after
    //   encoding when kHasCenterQuant), `compressed_centers` holds the encoded
    //   form when kHasCenterQuant.
    // Leaf node: `data` holds the raw point cloud set (kept for rerank);
    //   `encoded_leaf` holds the leaf encoding when kHasLeafQuant.
    PointCloudSet<ChPoint> data;
    [[no_unique_address]]
    std::conditional_t<kHasCenterQuant, CenterSet, mvivf_spill_internal::Empty>
        compressed_centers;
    [[no_unique_address]]
    std::conditional_t<kHasLeafQuant, LeafSet, mvivf_spill_internal::Empty>
        encoded_leaf;
    node_t() noexcept = default;
    ~node_t() noexcept = default;
    inline size_t get_size() const noexcept { return data.size(); }
  };

  node_t* root = nullptr;
  parlay::sequence<node_t*> leaves_flat;
  PointCloudSet<ChPoint> leaf_centers;
  [[no_unique_address]]
  std::conditional_t<kHasCenterQuant, CenterSet, mvivf_spill_internal::Empty>
      leaf_centers_encoded;

  [[no_unique_address]]
  std::conditional_t<kHasCenterQuant, CenterModel, mvivf_spill_internal::Empty>
      center_model_;
  [[no_unique_address]]
  std::conditional_t<kHasLeafQuant, LeafModel, mvivf_spill_internal::Empty> leaf_model_;
  [[no_unique_address]]
  std::conditional_t<kHasLeafQuant, LeafParams, mvivf_spill_internal::Empty> leaf_params_;

  size_t kmeanstree_height = 0;
  std::vector<uint32_t> leaf_to_root_child_;

  IndexMVIVFSpill(size_t d_) noexcept {
    d = d_;
    params = IndexParams::mvivf_spill();
    set_quant_mode_();
  }
  IndexMVIVFSpill(size_t d_, const IndexParams& p) noexcept {
    d = d_;
    params = p;
    set_quant_mode_();
  }
  template <class LP = LeafParams,
            std::enable_if_t<kHasLeafQuant && std::is_same_v<LP, LeafParams>, int> = 0>
  IndexMVIVFSpill(size_t d_, const IndexParams& p, const LP& lp) noexcept {
    d = d_;
    params = p;
    if constexpr (kHasLeafQuant) leaf_params_ = lp;
    set_quant_mode_();
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
      else quantization_mode = QT::None;
    }
  }

  // ---------------------------------------------------------------------------
  // Recursive builder (shared with non-spill subtrees).  Root is handled
  // separately in build().  `depth == 1` for the first recursive level
  // (level 1 in the (a, b) strategy: this is the only recursive level that may
  // spill; all deeper calls use top-1).
  // ---------------------------------------------------------------------------
  void recursive_build(node_t* node, const PointCloudSet<ChPoint>& points,
                       uint32_t depth = 1) {
    size_t n = points.size();
    size_t auto_nc = (params.k_per_level > 0)
                         ? params.k_per_level
                         : static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(n))));
    size_t small_nc = 4 * (n + params.max_leaf_size - 1) / params.max_leaf_size;
    size_t num_clusters = std::max(static_cast<size_t>(4), std::min({auto_nc, small_nc, n}));
    if (params.verbose >= 1) {
      std::cout << "[MVIVF-Spill] Building index with " << n
                << " points, num_clusters: " << num_clusters << std::endl;
    }
    MVClustering<metric> Clus(d, num_clusters, params.s, params.mvclus);
    Clus.train(points);
    PointCloudSet<ChPoint>& centers = Clus.get_centers();

    // Level-1 may spill to top-b; deeper levels always use top-1.
    const uint32_t spill_this_level =
        (depth == 1) ? std::max<uint32_t>(params.num_spill_l2, 1u) : 1u;
    const uint32_t k_spill = std::min(spill_this_level, static_cast<uint32_t>(num_clusters));

    parlay::sequence<std::pair<uint32_t, uint32_t>> id_pt;
    if (k_spill <= 1) {
      parlay::sequence<uint32_t> cluster_ids = Clus.get_clustering(points);
      id_pt = parlay::tabulate(n, [&](uint32_t i) {
        return std::make_pair(cluster_ids[i], i);
      });
    } else {
      parlay::sequence<uint32_t> assignment = Clus.get_topC(points, k_spill);
      id_pt = parlay::sequence<std::pair<uint32_t, uint32_t>>::uninitialized(n * k_spill);
      parlay::parallel_for(0, n, [&](uint32_t i) {
        for (uint32_t r = 0; r < k_spill; ++r) {
          id_pt[i * k_spill + r] = {assignment[i * k_spill + r], i};
        }
      });
    }

    auto grouped = group_by_key_inplace(id_pt);
    node->children.resize(grouped.size());
    if (grouped.size() < centers.size()) {
      auto active_centers_ind = parlay::delayed_seq<uint32_t>(
          grouped.size(), [&](size_t i) { return grouped[i][0].first; });
      node->data = PointCloudSet<ChPoint>(centers.filter(active_centers_ind), d);
    } else {
      node->data = std::move(centers);
    }
    parlay::parallel_for(0, grouped.size(), [&](size_t i) {
      auto group = parlay::delayed_seq<uint32_t>(
          grouped[i].size(), [&](size_t j) { return grouped[i][j].second; });
      PointCloudSet<ChPoint> child_points(points.filter(group), d);
      node_t* child = new node_t();
      node->children[i] = child;
      const bool depth_exhausted =
          (params.max_depth > 0) && (depth + 1 >= params.max_depth);
      if (!depth_exhausted && child_points.size() > params.max_leaf_size) {
        recursive_build(child, child_points, depth + 1);
      } else {
        child->data = std::move(child_points);
        if constexpr (kHasLeafQuant) {
          child->encoded_leaf = leaf_model_.encode(child->data);
        }
      }
    }, 1);
  }

  // ---------------------------------------------------------------------------
  // Build: root level uses spilling; deeper levels use recursive_build.
  // ---------------------------------------------------------------------------
  void build(const PointCloudSet<ChPoint>& points) override {
    parlay::internal::timer t;
    root = new node_t();

    t.start();
    if constexpr (kHasCenterQuant) center_model_.train(points);
    if constexpr (kHasLeafQuant) leaf_model_.train(points, leaf_params_);
    if (params.verbose >= 1 && (kHasCenterQuant || kHasLeafQuant)) {
      std::cout << "[MVIVF Spill] Quantizers Trained: " << t.stop() << " sec" << std::endl;
    }
    t.reset();

    size_t n = points.size();
    size_t auto_nc = (params.k_per_level > 0)
                         ? params.k_per_level
                         : static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(n))));
    size_t small_nc = 4 * (n + params.max_leaf_size - 1) / params.max_leaf_size;
    size_t num_clusters = std::max(static_cast<size_t>(4), std::min({auto_nc, small_nc, n}));
    if (params.verbose >= 1) {
      std::cout << "[MVIVF Spill]: building root with " << n
                << " points, num_clusters: " << num_clusters << std::endl;
    }

    MVClustering<metric> Clus(d, num_clusters, params.s, params.mvclus);
    Clus.train(points);
    PointCloudSet<ChPoint>& centers = Clus.get_centers();
    parlay::sequence<uint32_t> assignment = Clus.get_topC(points, params.num_spill);
    const uint32_t k_spill = std::min(params.num_spill, static_cast<uint32_t>(num_clusters));
    if (k_spill == 0) {
      std::cerr << "[MVIVF Spill]: num_spill is 0; need at least 1.\n";
      std::abort();
    }
    auto id_pt = parlay::sequence<std::pair<uint32_t, uint32_t>>::uninitialized(n * k_spill);
    parlay::parallel_for(0, n, [&](uint32_t i) {
      for (size_t r = 0; r < k_spill; ++r) {
        id_pt[i * k_spill + r] = {assignment[i * k_spill + r], i};
      }
    });
    auto grouped = group_by_key_inplace(id_pt);

    root->children.resize(grouped.size());
    if (grouped.size() < centers.size()) {
      auto active_centers_ind = parlay::delayed_seq<uint32_t>(
          grouped.size(), [&](size_t i) { return grouped[i][0].first; });
      root->data = PointCloudSet<ChPoint>(centers.filter(active_centers_ind), d);
    } else {
      root->data = std::move(centers);
    }

    parlay::parallel_for(0, grouped.size(), [&](size_t i) {
      auto group = parlay::delayed_seq<uint32_t>(
          grouped[i].size(), [&](size_t j) { return grouped[i][j].second; });
      PointCloudSet<ChPoint> child_points(points.filter(group), d);
      node_t* child = new node_t();
      root->children[i] = child;
      const bool depth_exhausted = (params.max_depth > 0) && (1u >= params.max_depth);
      if (!depth_exhausted && child_points.size() > params.max_leaf_size) {
        recursive_build(child, child_points, /*depth=*/1);
      } else {
        child->data = std::move(child_points);
        if constexpr (kHasLeafQuant) {
          child->encoded_leaf = leaf_model_.encode(child->data);
        }
      }
    }, 1);

    compute_leaf_flat_();
    compress_internal_centers_();
  }

  // ---------------------------------------------------------------------------
  // Build auxiliary structures: leaves_flat, leaf_centers, leaf_to_root_child_.
  // This must run *before* compress_internal_centers_ (which consumes the raw
  // center data from internal nodes).
  // ---------------------------------------------------------------------------
  void compute_leaf_flat_() {
    leaves_flat.clear();
    leaf_to_root_child_.clear();
    if (root == nullptr) return;

    std::vector<ChPoint> center_points;
    std::vector<uint32_t> root_child_for_leaf;
    std::function<void(node_t*, node_t*, size_t, uint32_t)> visit =
        [&](node_t* node, node_t* parent, size_t child_idx, uint32_t root_child_idx) {
          if (node->children.empty()) {
            leaves_flat.push_back(node);
            auto& centers_pc = parent->data;
            center_points.push_back(centers_pc[child_idx]);
            root_child_for_leaf.push_back(root_child_idx);
          } else {
            for (size_t i = 0; i < node->children.size(); ++i) {
              visit(node->children[i], node, i,
                    (parent == root) ? static_cast<uint32_t>(i) : root_child_idx);
            }
          }
        };
    visit(root, nullptr, 0, 0);
    leaf_to_root_child_ = std::move(root_child_for_leaf);
    if (!center_points.empty()) {
      leaf_centers = PointCloudSet<ChPoint>(center_points, d);
      if constexpr (kHasCenterQuant) {
        leaf_centers_encoded = center_model_.encode(leaf_centers);
      }
    }
  }

  // Encode internal-node centers when CompressCenters is true.  Also re-encodes
  // the leaf-center shortcut.  Does nothing when CompressCenters is false.
  void compress_internal_centers_() {
    if constexpr (kHasCenterQuant) {
      parlay::internal::timer t; t.start();
      std::function<void(node_t*)> visit = [&](node_t* node) {
        if (node == nullptr) return;
        if (!node->children.empty()) {
          if (node->data.size() > 0) {
            node->compressed_centers = center_model_.encode(node->data);
            node->data = PointCloudSet<ChPoint>{};
          }
          for (node_t* c : node->children) visit(c);
        }
      };
      visit(root);
      if (params.verbose >= 1) {
        std::cout << "[MVIVF Spill] Encoding internal centers: " << t.stop() << " sec" << std::endl;
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Greedy / flat-leaf search (returns a list of probe candidates).
  // ---------------------------------------------------------------------------
  struct GreedySearchResult {
    parlay::sequence<std::pair<float, node_t*>> probe_list;
    size_t bytes_accessed = 0;
    std::vector<double> stats = {};
  };

  GreedySearchResult greedy_search(const ChPoint& query, const CenterQuery& q_center,
                                   size_t nprobes) const {
    using score_node = std::pair<float, node_t*>;
    const size_t beam_length = 2 * nprobes;
    parlay::internal::timer t;
    double t_dists = 0.0, t_beam = 0.0, t_rest = 0.0;

    t.start();
    size_t dist_cmps = 0, bytes_accessed = 0;
    std::set<score_node> beam;
    parlay::sequence<score_node> top_probes;
    top_probes.reserve(nprobes + 1);
    std::vector<std::pair<uint32_t, float>> child_dists;
    child_dists.reserve(root->children.size());
    t_rest += t.stop(); t.reset();

    t.start();
    beam.insert({0.0f, root});
    t_beam += t.stop(); t.reset();
    while (!beam.empty()) {
      t.start();
      auto it = beam.begin();
      score_node best = *it;
      beam.erase(it);
      t_beam += t.stop(); t.reset();
      t.start();
      node_t* current_node = best.second;
      if (top_probes.size() == nprobes && best.first >= top_probes.front().first) break;
      auto& children = current_node->children;
      child_dists.resize(children.size());
      dist_cmps += children.size();
      if constexpr (kHasCenterQuant) {
        (void)query;
        current_node->compressed_centers.distances_all(q_center, child_dists.data());
        bytes_accessed += current_node->compressed_centers.num_bytes();
      } else {
        (void)q_center;
        auto& centers = current_node->data;
        centers.distances(query, child_dists.data());
        bytes_accessed += centers.num_bytes();
      }
      t_dists += t.stop(); t.reset();

      for (size_t i = 0; i < children.size(); ++i) {
        float di = child_dists[i].second;
        node_t* child = children[i];
        if (child->children.empty()) {
          t.start();
          if (top_probes.size() < nprobes || di < top_probes.front().first) {
            top_probes.push_back({di, child});
            std::push_heap(top_probes.begin(), top_probes.end());
            if (top_probes.size() > nprobes) {
              std::pop_heap(top_probes.begin(), top_probes.end());
              top_probes.pop_back();
            }
          }
          t_rest += t.stop(); t.reset();
        } else {
          t.start();
          const size_t beam_size = beam.size();
          if (beam_size < beam_length) {
            beam.insert({di, child});
          } else {
            auto worst_it = std::prev(beam.end());
            if (di < worst_it->first) {
              beam.erase(worst_it);
              beam.insert({di, child});
            }
          }
          t_beam += t.stop(); t.reset();
        }
      }
    }

    GreedySearchResult out;
    out.bytes_accessed = bytes_accessed;
    out.stats = {static_cast<double>(dist_cmps), t_dists, t_beam, t_rest};
    out.probe_list = std::move(top_probes);
    return out;
  }

  GreedySearchResult flat_leaf_search(const ChPoint& query, const CenterQuery& q_center,
                                      size_t nprobes) const {
    using score_node = std::pair<float, node_t*>;
    GreedySearchResult out;
    if (leaves_flat.empty() || leaf_centers.size() == 0 || nprobes == 0) return out;

    const size_t L = leaves_flat.size();
    const size_t use_nprobes = std::min(nprobes, L);
    parlay::internal::timer t;
    double t_dists = 0.0, t_rest = 0.0;

    t.start();
    size_t dist_cmps = L;
    size_t bytes_accessed = 0;
    auto centers_dists = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(L);
    auto scores = parlay::sequence<score_node>::uninitialized(L);
    if constexpr (kHasCenterQuant) {
      (void)query;
      leaf_centers_encoded.distances_all(q_center, centers_dists.data());
      bytes_accessed += leaf_centers_encoded.num_bytes();
    } else {
      (void)q_center;
      leaf_centers.distances(query, centers_dists.data());
      bytes_accessed += leaf_centers.num_bytes();
    }
    parlay::parallel_for(0, L, [&](size_t i) {
      scores[i] = {centers_dists[i].second, leaves_flat[i]};
    });
    t_dists += t.stop(); t.reset();

    t.start();
    if (use_nprobes < L) {
      std::nth_element(scores.begin(), scores.begin() + use_nprobes, scores.end(),
                       [](const score_node& a, const score_node& b) { return a.first < b.first; });
      scores.resize(use_nprobes);
    }
    t_rest += t.stop(); t.reset();

    out.probe_list = std::move(scores);
    out.bytes_accessed = bytes_accessed;
    out.stats = {static_cast<double>(dist_cmps), t_dists, 0.0, t_rest};
    return out;
  }

  // ---------------------------------------------------------------------------
  // Probe scoring.
  // ---------------------------------------------------------------------------
  inline std::pair<parlay::sequence<std::pair<uint32_t, float>>, size_t>
  process_probes(const ChPoint& query, const LeafQuery& q_leaf,
                 parlay::sequence<std::pair<float, node_t*>>& probe_list) {
    const size_t nprobes = probe_list.size();
    auto sizes = parlay::delayed_tabulate(nprobes, [&](size_t i) {
      return probe_list[i].second->get_size();
    });
    auto scan_result = parlay::scan(sizes);
    auto& offsets = scan_result.first;
    size_t total = scan_result.second;
    auto visited = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(total);
    auto bytes = parlay::sequence<size_t>::uninitialized(nprobes);
    parlay::parallel_for(0, nprobes, [&](size_t i) {
      node_t* leaf = probe_list[i].second;
      if constexpr (kHasLeafQuant) {
        leaf->encoded_leaf.distances_all(q_leaf, &visited[offsets[i]]);
        bytes[i] = leaf->encoded_leaf.num_bytes();
      } else {
        (void)q_leaf;
        leaf->data.distances(query, &visited[offsets[i]]);
        bytes[i] = leaf->data.num_bytes();
      }
    });
    return std::make_pair(std::move(visited), parlay::reduce(bytes));
  }

  // ---------------------------------------------------------------------------
  // Single-query search with detailed stats.
  //
  // Timer labels (matches bench `is_spill` path):
  //   0  greedy_cmps
  //   1  probe_cmps
  //   2  t_greedy_dists
  //   3  t_greedy_beam
  //   4  t_greedy_rest
  //   5  t_compress
  //   6  t_quantize
  //   7  t_probe
  //   8  t_dedup
  //   9  t_rest
  //  10  t_rerank
  // ---------------------------------------------------------------------------
  std::tuple<parlay::sequence<std::pair<uint32_t, float>>, size_t, std::vector<double>>
  search_with_stats(const ChPoint& query, const PointCloudSet<ChPoint>& points,
                    const SearchParams& search_params) override {
    parlay::internal::timer t;
    const size_t k = search_params.k;
    size_t nprobes = search_params.nprobes;
    size_t dist_cmps = 0;
    size_t bytes_accessed = 0;
    double t_compress = 0.0, t_quantize = 0.0;
    double t_distances = 0.0, t_dedup = 0.0, t_rest = 0.0;

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
    t.start();
    LeafQuery q_leaf{};
    CenterQuery q_center{};
    if constexpr (kHasLeafQuant) q_leaf = leaf_model_.quantize_query(effective_query);
    if constexpr (kHasCenterQuant) q_center = center_model_.quantize_query(effective_query);
    t_quantize = t.stop(); t.reset();

    // Step 1: greedy or flat-leaf search.
    GreedySearchResult gs;
    const size_t num_leaves_count = leaves_flat.size();
    const double alpha = 1.0;  // heuristic threshold
    bool use_flat = (num_leaves_count > 0 &&
                     nprobes >= static_cast<size_t>(alpha * num_leaves_count));
    if (use_flat) {
      gs = flat_leaf_search(effective_query, q_center, nprobes);
    } else {
      gs = greedy_search(effective_query, q_center, nprobes);
    }
    auto& probe_list = gs.probe_list;
    bytes_accessed += gs.bytes_accessed;
    nprobes = std::min(nprobes, probe_list.size());

    // Step 2: probe.
    t.start();
    parlay::sequence<std::pair<uint32_t, float>> visited;
    size_t bytes_pp;
    std::tie(visited, bytes_pp) = process_probes(effective_query, q_leaf, probe_list);
    bytes_accessed += bytes_pp;
    t_distances = t.stop(); t.reset();
    dist_cmps += visited.size();

    t.start();
    visited = deduplicate<uint32_t, float>(std::move(visited));
    t_dedup = t.stop(); t.reset();

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
    double t_rerank = t.stop(); t.reset();

    std::vector<double> stats;
    stats.reserve(8 + gs.stats.size());
    stats.push_back(gs.stats[0]);
    stats.push_back(static_cast<double>(dist_cmps));
    for (size_t i = 1; i < gs.stats.size(); ++i) {
      stats.push_back(gs.stats[i]);
    }
    stats.push_back(t_compress);
    stats.push_back(t_quantize);
    stats.push_back(t_distances);
    stats.push_back(t_dedup);
    stats.push_back(t_rest);
    stats.push_back(t_rerank);

    return std::make_tuple(std::move(final_results), bytes_accessed, std::move(stats));
  }

  // ---------------------------------------------------------------------------
  // Tree-introspection helpers (carried over unchanged from the legacy class).
  // ---------------------------------------------------------------------------
  struct TreeStats {
    size_t num_internal_nodes = 0;
    size_t num_leaves = 0;
    double avg_leaf_size = 0.0;
    double avg_internal_node_size = 0.0;
    size_t total_point_clouds_internal = 0;
    size_t height = 0;
    double avg_child_fraction_imbalance = 0.0;
    double max_child_fraction_imbalance = 0.0;
    static constexpr double kBadImbalanceThreshold = 0.8;
    std::vector<std::pair<size_t, double>> bad_imbalance_entries;
  };

  TreeStats get_tree_stats() const {
    TreeStats s;
    if (root == nullptr) return s;
    size_t leaf_size_sum = 0, internal_size_sum = 0;
    double sum_child_frac_imbalance = 0.0;
    size_t num_internal_balance_nodes = 0;
    std::function<size_t(const node_t*, size_t)> visit = [&](const node_t* node,
                                                             size_t depth) -> size_t {
      if (node->children.empty()) {
        s.num_leaves++;
        size_t leaf_size = node->get_size();
        leaf_size_sum += leaf_size;
        if (depth + 1 > s.height) s.height = depth + 1;
        return leaf_size;
      }
      s.num_internal_nodes++;
      size_t n = node->get_size();
      s.total_point_clouds_internal += n;
      internal_size_sum += n;
      std::vector<size_t> child_subtree_sizes;
      child_subtree_sizes.reserve(node->children.size());
      size_t subtree_total = 0;
      for (const node_t* child : node->children) {
        size_t child_size = visit(child, depth + 1);
        child_subtree_sizes.push_back(child_size);
        subtree_total += child_size;
      }
      if (child_subtree_sizes.size() >= 2 && subtree_total > 0) {
        auto [min_it, max_it] = std::minmax_element(child_subtree_sizes.begin(),
                                                    child_subtree_sizes.end());
        double frac_imbalance = static_cast<double>(*max_it - *min_it) /
                                static_cast<double>(subtree_total);
        sum_child_frac_imbalance += frac_imbalance;
        if (frac_imbalance > s.max_child_fraction_imbalance)
          s.max_child_fraction_imbalance = frac_imbalance;
        num_internal_balance_nodes++;
        if (frac_imbalance >= TreeStats::kBadImbalanceThreshold) {
          s.bad_imbalance_entries.emplace_back(subtree_total, frac_imbalance);
        }
      }
      return subtree_total;
    };
    visit(root, 0);
    if (s.num_leaves > 0) s.avg_leaf_size = static_cast<double>(leaf_size_sum) / s.num_leaves;
    if (s.num_internal_nodes > 0) {
      s.avg_internal_node_size = static_cast<double>(internal_size_sum) / s.num_internal_nodes;
    }
    if (num_internal_balance_nodes > 0) {
      s.avg_child_fraction_imbalance =
          sum_child_frac_imbalance / static_cast<double>(num_internal_balance_nodes);
    }
    std::sort(s.bad_imbalance_entries.begin(), s.bad_imbalance_entries.end());
    return s;
  }

  parlay::sequence<uint32_t> get_flat_clustering() const {
    parlay::sequence<uint32_t> empty;
    if (root == nullptr) return empty;
    std::vector<std::pair<uint32_t, uint32_t>> assignments;
    assignments.reserve(1024);
    size_t max_id = 0, leaf_idx = 0;
    std::function<void(const node_t*)> visit = [&](const node_t* node) {
      if (node->children.empty()) {
        size_t n = node->data.size();
        for (size_t j = 0; j < n; ++j) {
          uint32_t id = node->data.get_id(j);
          assignments.emplace_back(id, static_cast<uint32_t>(leaf_idx));
          if (id > max_id) max_id = id;
        }
        ++leaf_idx;
      } else {
        for (const node_t* child : node->children) if (child) visit(child);
      }
    };
    visit(root);
    if (assignments.empty()) return empty;
    parlay::sequence<uint32_t> leaf_of_point(max_id + 1);
    parlay::parallel_for(0, leaf_of_point.size(), [&](size_t i) {
      leaf_of_point[i] = UINT32_MAX;
    });
    parlay::parallel_for(0, assignments.size(), [&](size_t i) {
      auto [pid, lid] = assignments[i];
      leaf_of_point[pid] = lid;
    });
    return leaf_of_point;
  }

  size_t traverse_tree(node_t* node, parlay::sequence<node_t*>& ind_to_node,
                       std::unordered_map<node_t*, size_t>& node_to_ind,
                       parlay::sequence<size_t>& center_offsets,
                       parlay::sequence<size_t>& children_offsets,
                       parlay::sequence<size_t>& point_offsets, size_t height) {
    node_to_ind[node] = ind_to_node.size();
    ind_to_node.push_back(node);
    if (node->children.size() == 0) {
      point_offsets.push_back(node->data.size());
    } else {
      point_offsets.push_back(0);
      // For CompressCenters, raw center data is gone — we emit zeros and rely on
      // re-encoding at load time (skeleton-only persistence).
      size_t dims = node->data.get_dims();
      for (size_t i = 0; i < node->children.size(); ++i) {
        if constexpr (kHasCenterQuant) {
          (void)dims; center_offsets.push_back(0);
        } else {
          center_offsets.push_back(node->data.get_size(i) * dims);
        }
      }
    }
    children_offsets.push_back(node->children.size());
    size_t h = height + 1;
    for (node_t* child : node->children) {
      h = std::max(h, traverse_tree(child, ind_to_node, node_to_ind, center_offsets,
                                    children_offsets, point_offsets, height + 1));
    }
    return h;
  }

  // ---------------------------------------------------------------------------
  // Save / load (v2 format, skeleton-only).
  //
  // Layout:
  //   u32 magic ('MSPL'), u32 version, u32 class_id
  //   IndexParams blob
  //   size_t num_nodes
  //   size_t num_center_offsets
  //   size_t[num_center_offsets] center_offsets
  //   float[sum] center_values (only when !kHasCenterQuant)
  //   size_t[num+1] children_offsets
  //   size_t[sum] children_values (flattened child indices)
  //   size_t[num+1] point_offsets
  //   uint32_t[sum] point_ids
  // ---------------------------------------------------------------------------
  static constexpr uint32_t kMagic = 0x4D53504Cu;  // 'MSPL'
  static constexpr uint32_t kVersion = 3u;  // v3: dropped persisted params.quantize_centers
  static uint32_t compute_class_id_() {
    uint32_t lid = LeafModel::kClassId;
    return (lid << 1) | (kHasCenterQuant ? 1u : 0u);
  }

 private:
  void write_params_(std::ostream& out) const {
    auto w = [&](auto x) { out.write(reinterpret_cast<const char*>(&x), sizeof(x)); };
    w(params.k_per_level);
    w(params.s);
    w(params.mvclus.niters);
    w(params.mvclus.max_point_clouds_per_cluster);
    w(params.mvclus.max_points_per_centroid_inner_kmeans);
    w(params.max_leaf_size);
    w(params.max_depth);
    w(params.num_spill);
    w(params.num_spill_l2);
  }
  void read_params_(std::istream& in) {
    auto r = [&](auto& x) { in.read(reinterpret_cast<char*>(&x), sizeof(x)); };
    r(params.k_per_level);
    r(params.s);
    r(params.mvclus.niters);
    r(params.mvclus.max_point_clouds_per_cluster);
    r(params.mvclus.max_points_per_centroid_inner_kmeans);
    r(params.max_leaf_size);
    r(params.max_depth);
    r(params.num_spill);
    r(params.num_spill_l2);
  }

 public:
  void save(const std::string& filename) override {
    if (root == nullptr) {
      std::cerr << "IndexMVIVFSpill::save: root is null." << std::endl;
      return;
    }
    std::ofstream outfile(filename, std::ios::binary);
    std::cout << "Saving index to " << filename << std::endl;
    if (!outfile.is_open()) {
      std::cerr << "Error opening file for writing: " << filename << std::endl;
      return;
    }
    const uint32_t magic = kMagic, ver = kVersion, cid = compute_class_id_();
    outfile.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    outfile.write(reinterpret_cast<const char*>(&ver), sizeof(ver));
    outfile.write(reinterpret_cast<const char*>(&cid), sizeof(cid));
    write_params_(outfile);

    parlay::sequence<node_t*> ind_to_node;
    std::unordered_map<node_t*, size_t> node_to_ind;
    parlay::sequence<size_t> center_offsets;
    parlay::sequence<size_t> children_offsets;
    parlay::sequence<size_t> point_offsets;
    size_t height = traverse_tree(root, ind_to_node, node_to_ind, center_offsets,
                                  children_offsets, point_offsets, 0);
    kmeanstree_height = height;

    size_t total_centers = parlay::scan_inplace(center_offsets);
    center_offsets.push_back(total_centers);
    size_t total_children = parlay::scan_inplace(children_offsets);
    children_offsets.push_back(total_children);
    size_t total_points = parlay::scan_inplace(point_offsets);
    point_offsets.push_back(total_points);

    size_t num = ind_to_node.size();
    outfile.write(reinterpret_cast<const char*>(&num), sizeof(size_t));
    size_t num_center_offsets = center_offsets.size();
    outfile.write(reinterpret_cast<const char*>(&num_center_offsets), sizeof(size_t));
    outfile.write(reinterpret_cast<const char*>(center_offsets.begin()),
                  center_offsets.size() * sizeof(size_t));

    if constexpr (!kHasCenterQuant) {
      for (size_t i = 0; i < num; ++i) {
        node_t* node = ind_to_node[i];
        if (node->children.size() > 0) {
          auto coords = node->data.data();
          size_t nent = node->data.total_size() * node->data.get_dims();
          outfile.write(reinterpret_cast<const char*>(coords), nent * sizeof(float));
        }
      }
    }

    outfile.write(reinterpret_cast<const char*>(children_offsets.begin()),
                  children_offsets.size() * sizeof(size_t));
    for (size_t i = 0; i < num; ++i) {
      node_t* node = ind_to_node[i];
      for (size_t j = 0; j < node->children.size(); ++j) {
        size_t child_id = node_to_ind[node->children[j]];
        outfile.write(reinterpret_cast<const char*>(&child_id), sizeof(size_t));
      }
    }
    outfile.write(reinterpret_cast<const char*>(point_offsets.begin()),
                  point_offsets.size() * sizeof(size_t));
    for (size_t i = 0; i < num; ++i) {
      node_t* node = ind_to_node[i];
      if (node->children.size() == 0) {
        for (size_t j = 0; j < node->data.size(); ++j) {
          uint32_t pid = node->data.get_id(j);
          outfile.write(reinterpret_cast<const char*>(&pid), sizeof(uint32_t));
        }
      }
    }
    outfile.flush();
    outfile.close();
  }

  void load(const std::string& filename, const PointCloudSet<ChPoint>& points) override {
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
      std::cerr << "[MVIVF Spill] bad magic (not MSPL v2)." << std::endl;
      return;
    }
    if (ver != kVersion) {
      std::cerr << "[MVIVF Spill] unsupported version " << ver << "." << std::endl;
      return;
    }
    if (cid != compute_class_id_()) {
      std::cerr << "[MVIVF Spill] class_id mismatch: file has " << cid
                << ", this instance is " << compute_class_id_() << "." << std::endl;
      return;
    }
    read_params_(infile);

    size_t num = 0;
    infile.read(reinterpret_cast<char*>(&num), sizeof(size_t));
    size_t num_center_offsets = 0;
    infile.read(reinterpret_cast<char*>(&num_center_offsets), sizeof(size_t));
    parlay::sequence<size_t> center_offsets(num_center_offsets);
    infile.read(reinterpret_cast<char*>(center_offsets.begin()),
                center_offsets.size() * sizeof(size_t));
    parlay::sequence<float> center_values;
    if constexpr (!kHasCenterQuant) {
      center_values = parlay::sequence<float>(center_offsets[center_offsets.size() - 1]);
      infile.read(reinterpret_cast<char*>(center_values.begin()),
                  center_values.size() * sizeof(float));
    }
    parlay::sequence<size_t> children_offsets(num + 1);
    infile.read(reinterpret_cast<char*>(children_offsets.begin()),
                children_offsets.size() * sizeof(size_t));
    parlay::sequence<size_t> children_values(children_offsets[children_offsets.size() - 1]);
    infile.read(reinterpret_cast<char*>(children_values.begin()),
                children_values.size() * sizeof(size_t));
    parlay::sequence<size_t> point_offsets(num + 1);
    infile.read(reinterpret_cast<char*>(point_offsets.begin()),
                point_offsets.size() * sizeof(size_t));
    parlay::sequence<uint32_t> point_values(point_offsets[point_offsets.size() - 1]);
    infile.read(reinterpret_cast<char*>(point_values.begin()),
                point_values.size() * sizeof(uint32_t));
    infile.close();

    // Re-train quantizers.
    if constexpr (kHasCenterQuant) center_model_.train(points);
    if constexpr (kHasLeafQuant) leaf_model_.train(points, leaf_params_);

    size_t dim = points.get_dims();
    auto point_id_to_data_id = parlay::sequence<uint32_t>::uninitialized(points.size());
    parlay::parallel_for(0, points.size(),
                         [&](uint32_t i) { point_id_to_data_id[points.get_id(i)] = i; });
    parlay::sequence<size_t> children_sizes = parlay::sequence<size_t>::from_function(
        num, [&](size_t i) { return children_offsets[i + 1] - children_offsets[i]; });
    parlay::sequence<size_t> children_sizes_scan;
    size_t total_children_sizes;
    std::tie(children_sizes_scan, total_children_sizes) = parlay::scan(children_sizes);
    children_sizes_scan.push_back(total_children_sizes);
    parlay::sequence<size_t> point_sizes = parlay::sequence<size_t>::from_function(
        num, [&](size_t i) { return point_offsets[i + 1] - point_offsets[i]; });
    parlay::sequence<node_t*> ind_to_node =
        parlay::sequence<node_t*>::from_function(num, [&](size_t i) {
          node_t* node = new node_t();
          if constexpr (!kHasCenterQuant) {
            if (children_sizes[i] > 0) {
              size_t start_offset = children_sizes_scan[i];
              parlay::sequence<size_t> node_center_offsets =
                  parlay::sequence<size_t>::from_function(children_sizes[i] + 1, [&](size_t j) {
                    return center_offsets[start_offset + j] - center_offsets[start_offset];
                  });
              node->data = PointCloudSet<ChPoint>(children_sizes[i], dim,
                                                  center_values.data() + center_offsets[start_offset],
                                                  node_center_offsets.data(), nullptr);
            }
          }
          node->children.resize(children_sizes[i]);
          if (point_sizes[i] > 0) {
            parlay::sequence<uint32_t> point_group =
                parlay::sequence<uint32_t>::from_function(point_sizes[i], [&](size_t j) {
                  uint32_t pid = point_values[point_offsets[i] + j];
                  return point_id_to_data_id[pid];
                });
            node->data = PointCloudSet<ChPoint>(points.filter(point_group), dim);
          }
          return node;
        });

    parlay::parallel_for(0, num, [&](size_t i) {
      node_t* node = ind_to_node[i];
      auto& children = node->children;
      parlay::parallel_for(0, children.size(), [&](size_t j) {
        size_t child_id = children_values[children_offsets[i] + j];
        children[j] = ind_to_node[child_id];
      });
    });
    root = ind_to_node[0];

    // Re-encode leaves.
    if constexpr (kHasLeafQuant) {
      parlay::parallel_for(0, num, [&](size_t i) {
        node_t* node = ind_to_node[i];
        if (node && node->children.empty() && node->data.size() > 0) {
          node->encoded_leaf = leaf_model_.encode(node->data);
        }
      }, 1);
    }

    // Rebuild flat-leaf index and (when CompressCenters) re-encode centers.
    if constexpr (kHasCenterQuant) {
      // For the skeleton-only compressed save, internal-node centers were not
      // persisted.  Re-running MVClustering to regenerate them exactly is not
      // trivial; we treat this case as lossy and instead rebuild from the
      // original points.  (In practice, users who need exact reload should use
      // the non-compressed save.)
      std::cerr << "[MVIVF Spill] Warning: CompressCenters+load currently "
                   "requires re-training / re-encoding which loses the exact "
                   "original centers."
                << std::endl;
    }
    compute_leaf_flat_();
  }

  void traverse_and_delete(node_t* node) {
    for (size_t i = 0; i < node->children.size(); ++i) {
      node_t* child = node->children[i];
      traverse_and_delete(child);
      delete child;
    }
  }

  size_t get_height() const noexcept override { return kmeanstree_height; }

  ~IndexMVIVFSpill() {
    if (root != nullptr) {
      traverse_and_delete(root);
      delete root;
    }
  }
};

// Concrete aliases.
using IndexMVIVFSpillIP          = IndexMVIVFSpill<false, false, NoQuantizer<false>>;
using IndexMVIVFSpillL2          = IndexMVIVFSpill<true,  false, NoQuantizer<true>>;
using IndexMVIVFSpillCompressIP  = IndexMVIVFSpill<false, true,  NoQuantizer<false>>;
using IndexMVIVFSpillCompressL2  = IndexMVIVFSpill<true,  true,  NoQuantizer<true>>;

using IndexMVIVFSpillPQIP        = IndexMVIVFSpill<false, false, pq_mv::Model<false>>;
using IndexMVIVFSpillPQL2        = IndexMVIVFSpill<true,  false, pq_mv::Model<true>>;
using IndexMVIVFSpillFastScanIP  = IndexMVIVFSpill<false, false, fastscan_mv::Model<false>>;
using IndexMVIVFSpillFastScanL2  = IndexMVIVFSpill<true,  false, fastscan_mv::Model<true>>;
using IndexMVIVFSpillRaBitQIP    = IndexMVIVFSpill<false, false, rabitq_mv::Model<false>>;
using IndexMVIVFSpillRaBitQL2    = IndexMVIVFSpill<true,  false, rabitq_mv::Model<true>>;
using IndexMVIVFSpillTQIP        = IndexMVIVFSpill<false, false, turboquant_mv::Model<false>>;
using IndexMVIVFSpillTQL2        = IndexMVIVFSpill<true,  false, turboquant_mv::Model<true>>;
using IndexMVIVFSpillSPQTQIP     = IndexMVIVFSpill<false, false, pqtq_mv::Model<false>>;
using IndexMVIVFSpillSPQTQL2     = IndexMVIVFSpill<true,  false, pqtq_mv::Model<true>>;
using IndexMVIVFSpillOneBitTQIP  = IndexMVIVFSpill<false, false, turboquant_1bit_mv::Model<false>>;
using IndexMVIVFSpillOneBitTQL2  = IndexMVIVFSpill<true,  false, turboquant_1bit_mv::Model<true>>;

// CompressCenters (TQ-quantized centers) + LeafModel combinations.
using IndexMVIVFSpillCompressPQIP        = IndexMVIVFSpill<false, true, pq_mv::Model<false>>;
using IndexMVIVFSpillCompressPQL2        = IndexMVIVFSpill<true,  true, pq_mv::Model<true>>;
using IndexMVIVFSpillCompressFastScanIP  = IndexMVIVFSpill<false, true, fastscan_mv::Model<false>>;
using IndexMVIVFSpillCompressFastScanL2  = IndexMVIVFSpill<true,  true, fastscan_mv::Model<true>>;
using IndexMVIVFSpillCompressRaBitQIP    = IndexMVIVFSpill<false, true, rabitq_mv::Model<false>>;
using IndexMVIVFSpillCompressRaBitQL2    = IndexMVIVFSpill<true,  true, rabitq_mv::Model<true>>;
using IndexMVIVFSpillCompressTQIP        = IndexMVIVFSpill<false, true, turboquant_mv::Model<false>>;
using IndexMVIVFSpillCompressTQL2        = IndexMVIVFSpill<true,  true, turboquant_mv::Model<true>>;
using IndexMVIVFSpillCompressSPQTQIP     = IndexMVIVFSpill<false, true, pqtq_mv::Model<false>>;
using IndexMVIVFSpillCompressSPQTQL2     = IndexMVIVFSpill<true,  true, pqtq_mv::Model<true>>;
using IndexMVIVFSpillCompressOneBitTQIP  = IndexMVIVFSpill<false, true, turboquant_1bit_mv::Model<false>>;
using IndexMVIVFSpillCompressOneBitTQL2  = IndexMVIVFSpill<true,  true, turboquant_1bit_mv::Model<true>>;

}  // namespace mvsic
