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
#include <atomic>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mvsic/core/index.h"
#include "mvsic/core/mvclustering/mvclustering.h"
#include "mvsic/core/mvclustering/mvclustering_8bit.h"
#include "mvsic/core/query_compression.h"
#include "mvsic/core/utils/interval_heap.h"
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
      else if constexpr (std::is_same_v<L, turboquant_1bit_asym_mv::Model<metric>>) quantization_mode = QT::OneBitTQAsym;
      else if constexpr (std::is_same_v<L, turboquant_8bit_mv::Model<metric>>) quantization_mode = QT::EightBitTQ;
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
    // Level-1 may spill to top-b; deeper levels always use top-1.
    const uint32_t spill_this_level =
        (depth == 1) ? std::max<uint32_t>(params.num_spill_l2, 1u) : 1u;
    const uint32_t k_spill = std::min(spill_this_level, static_cast<uint32_t>(num_clusters));

    PointCloudSet<ChPoint> centers;
    parlay::sequence<std::pair<uint32_t, uint32_t>> id_pt;
    auto run_clus = [&](auto& Clus) {
      Clus.train(points);
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
      centers = std::move(Clus.get_centers());
    };
    if (params.build_with_8btq) {
      MVClustering8BTQ<metric> Clus(d, num_clusters, params.s, params.mvclus);
      run_clus(Clus);
    } else {
      MVClustering<metric> Clus(d, num_clusters, params.s, params.mvclus);
      run_clus(Clus);
    }

    auto grouped = group_by_key_inplace(id_pt);
    // Safety guard: if clustering collapses to a single non-empty group (e.g.
    // a duplicate-heavy subcluster where every input assigns to the same
    // center), no useful split is possible.  Recursing again would just
    // collapse again and blow the stack, so accept an oversized leaf here.
    if (grouped.size() <= 1) {
      if (params.verbose >= 1) {
        std::cout << "[MVIVF-Spill] Cluster collapse at depth " << depth << " ("
                  << n << " points > max_leaf_size=" << params.max_leaf_size
                  << "); making this node a leaf." << std::endl;
      }
      auto all_idx =
          parlay::tabulate(n, [](size_t i) { return static_cast<uint32_t>(i); });
      node->data = PointCloudSet<ChPoint>(points.filter(all_idx), d);
      if constexpr (kHasLeafQuant) {
        node->encoded_leaf = leaf_model_.encode(node->data);
      }
      return;
    }
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
    // Propagate the build-time 8BTQ flag into MVClusteringConfig so the inner
    // Lloyd's k-means picks the TQ8 backend in kmeans_subsample/weighted.
    params.mvclus.build_with_8btq = params.build_with_8btq;

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

    const uint32_t k_spill = std::min(params.num_spill, static_cast<uint32_t>(num_clusters));
    if (k_spill == 0) {
      std::cerr << "[MVIVF Spill]: num_spill is 0; need at least 1.\n";
      std::abort();
    }
    PointCloudSet<ChPoint> centers;
    parlay::sequence<uint32_t> assignment;
    auto run_clus = [&](auto& Clus) {
      Clus.train(points);
      assignment = Clus.get_topC(points, params.num_spill);
      centers = std::move(Clus.get_centers());
    };
    if (params.build_with_8btq) {
      MVClustering8BTQ<metric> Clus(d, num_clusters, params.s, params.mvclus);
      run_clus(Clus);
    } else {
      MVClustering<metric> Clus(d, num_clusters, params.s, params.mvclus);
      run_clus(Clus);
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

  // Encode internal-node centers when CompressCenters is true.  Keeps the raw
  // `node->data` floats alongside the encoded form so that save() has the
  // exact float centers used during build.  Does nothing when CompressCenters
  // is false.
  void compress_internal_centers_() {
    if constexpr (kHasCenterQuant) {
      parlay::internal::timer t; t.start();
      std::function<void(node_t*)> visit = [&](node_t* node) {
        if (node == nullptr) return;
        if (!node->children.empty()) {
          if (node->data.size() > 0) {
            node->compressed_centers = center_model_.encode(node->data);
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

  GreedySearchResult greedy_search(
      const ChPoint& query, const CenterQuery& q_center, size_t nprobes,
      const std::pair<uint32_t, float>* precomputed_root_dists = nullptr) const {
    using score_node = std::pair<float, node_t*>;
    const size_t beam_length = 2 * nprobes;

    // The per-child timer.start/stop pair the previous version sprinkled
    // through the inner loop was ~90ns per iteration.  At nprobes=2048 with
    // root fan-out ~1.6k and beam_length 4k, that is ~10k+ child iterations
    // per query × 90ns = ~1ms/q of timer overhead — same order as the actual
    // beam/heap work.  We drop the breakdown stats here; callers only ever
    // read stats[0] (dist_cmps) on this hot path.
    size_t dist_cmps = 0, bytes_accessed = 0;
    IntervalHeap<score_node> beam(beam_length);
    parlay::sequence<score_node> top_probes;
    top_probes.reserve(nprobes + 1);
    std::vector<std::pair<uint32_t, float>> child_dists;
    child_dists.reserve(root->children.size());

    beam.push({0.0f, root});
    while (!beam.empty()) {
      score_node best = beam.pop_min();
      node_t* current_node = best.second;
      if (top_probes.size() == nprobes && best.first >= top_probes.front().first) break;
      auto& children = current_node->children;
      const size_t nc = children.size();
      child_dists.resize(nc);
      dist_cmps += nc;
      if (precomputed_root_dists != nullptr && current_node == root) {
        std::memcpy(child_dists.data(), precomputed_root_dists,
                    nc * sizeof(std::pair<uint32_t, float>));
      } else if constexpr (kHasCenterQuant) {
        (void)query;
        current_node->compressed_centers.distances_all(q_center, child_dists.data());
        bytes_accessed += current_node->compressed_centers.num_bytes();
      } else {
        (void)q_center;
        auto& centers = current_node->data;
        centers.distances(query, child_dists.data());
        bytes_accessed += centers.num_bytes();
      }

      // Hoist the read of beam.top_max() and top_probes.front().first out of
      // the per-child loop so they aren't reloaded each iteration.  Track
      // them as locals and resync only when we mutate the corresponding
      // structure.
      float beam_max = beam.size() == beam_length ? beam.top_max().first
                                                  : std::numeric_limits<float>::infinity();
      float top_worst = top_probes.size() == nprobes ? top_probes.front().first
                                                     : std::numeric_limits<float>::infinity();
      for (size_t i = 0; i < nc; ++i) {
        const float di = child_dists[i].second;
        node_t* child = children[i];
        if (child->children.empty()) {
          if (di < top_worst || top_probes.size() < nprobes) {
            top_probes.push_back({di, child});
            std::push_heap(top_probes.begin(), top_probes.end());
            if (top_probes.size() > nprobes) {
              std::pop_heap(top_probes.begin(), top_probes.end());
              top_probes.pop_back();
            }
            top_worst = top_probes.size() == nprobes ? top_probes.front().first
                                                     : std::numeric_limits<float>::infinity();
          }
        } else {
          if (beam.size() < beam_length) {
            beam.push({di, child});
            if (beam.size() == beam_length) beam_max = beam.top_max().first;
          } else if (di < beam_max) {
            beam.replace_max({di, child});
            beam_max = beam.top_max().first;
          }
        }
      }
    }

    GreedySearchResult out;
    out.bytes_accessed = bytes_accessed;
    out.stats = {static_cast<double>(dist_cmps), 0.0, 0.0, 0.0};
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
  // Timer labels (matches benchmarks/methods.yaml `mvivf_spill.labels`):
  //   0  search_cmps   (greedy dist cmps)
  //   1  probe_cmps
  //   2  t_search_dists
  //   3  t_search_beam
  //   4  t_search_rest
  //   5  t_search_top_level
  //   6  t_compress
  //   7  t_quant
  //   8  t_leaf_dists
  //   9  t_leaf_dedup
  //  10  t_leaf_rest
  //  11  t_rerank
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
          search_params.query_compression_threshold, ba,
          search_params.query_alignment,
          search_params.query_alignment_strict);
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

    // Step 1: greedy search. Flat-leaf fast path (use when nprobes >=
    // num_leaves) is intentionally disabled; on small datasets it produces
    // a misleading QPS plateau in the tail of the sweep because the entire
    // greedy tree-walk is bypassed.  Force greedy search throughout.
    GreedySearchResult gs;
    [[maybe_unused]] const size_t num_leaves_count = leaves_flat.size();
    constexpr bool use_flat = false;
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
  // Many-query cache-efficient search (search_all_new).
  //
  // Mirrors IndexMVIVF::search_all_new (mvivf.h): for LeafModel types that
  // expose `ManyToMany::TopKIntoUninitialized` (FastScan / SPQTQ / 1-bit TQ),
  // we group probed leaves and run the batched M2M kernel per leaf.  Other
  // quantizers fall through to parallel 1-to-N scoring.
  //
  // Spill differs from the regular MVIVF only in step 4: a point id may
  // appear in multiple probed leaves (root spill = a, level-1 spill = b),
  // so per-query candidate lists must be deduplicated before rerank.  The
  // dedup uses `deduplicate_and_topC` which sorts by (value, key) and drops
  // duplicates in one pass, then trims to `num_rerank`.
  // ---------------------------------------------------------------------------
 private:
  template<class M>
  struct has_many_to_many_ : std::false_type {};
  template<>
  struct has_many_to_many_<fastscan_mv::Model<metric>> : std::true_type {
    using type = fastscan_mv::ManyToMany<LeafSet>;
  };
  template<>
  struct has_many_to_many_<pqtq_mv::Model<metric>> : std::true_type {
    using type = pqtq_mv::ManyToMany<LeafSet>;
  };
  template<>
  struct has_many_to_many_<turboquant_1bit_mv::Model<metric>> : std::true_type {
    using type = turboquant_1bit_mv::ManyToMany<LeafSet>;
  };
  static constexpr bool kUseM2M = has_many_to_many_<LeafModel>::value;

 public:
  std::pair<parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>, size_t> search_all_new(
      const PointCloudSet<ChPoint>& query_points, const PointCloudSet<ChPoint>& points,
      const SearchParams& search_params) {
    if (search_params.nprobes <= 2) {
      return Index<metric>::search_all(query_points, points, search_params);
    }
    parlay::internal::timer t;
    const size_t num_q = query_points.size();
    const size_t num_leaves = leaves_flat.size();
    const size_t k = search_params.k;
    size_t nprobes = std::min(num_leaves, search_params.nprobes);
    size_t bytes_accessed = 0;
    size_t dist_cmps = 0;

    // Step -1: optional query compression.
    PointCloudSet<ChPoint> compressed_storage;
    const PointCloudSet<ChPoint>* eff_ptr = &query_points;
    if (search_params.query_compression != SearchParams::QueryCompression::None) {
      t.start();
      uint32_t ba = 1;
      if constexpr (kHasLeafQuant) ba = LeafModel::kBatchAlignment;
      compressed_storage =
          compress_point_cloud_set<ChPoint>(query_points, search_params.query_compression,
                                            search_params.query_compression_threshold, ba,
                                            search_params.query_alignment,
                                            search_params.query_alignment_strict);
      eff_ptr = &compressed_storage;
      t.stop();
      std::cout << "[MVIVF-Spill] Query Compression: " << t.total_time() << " sec" << std::endl;
      t.reset();
    }
    const auto& eff_queries = *eff_ptr;
    const auto& rerank_queries = search_params.compress_rerank ? eff_queries : query_points;

    // Step 0: pre-quantize queries.
    t.start();
    parlay::sequence<LeafQuery> q_leaves(num_q);
    if constexpr (kHasLeafQuant) {
      parlay::parallel_for(
          0, num_q, [&](size_t i) { q_leaves[i] = leaf_model_.quantize_query(eff_queries[i]); });
    }
    t.stop();
    std::cout << "[MVIVF-Spill] Query Quantization: " << t.total_time() << " sec" << std::endl;
    t.reset();

    // Step 0b (optional): root-level many-to-many.
    //
    // When enabled, share root-child distance work across all queries: pre-
    // quantize every q_center, fuse them, score against
    // root->compressed_centers once. Each per-query greedy_search seeds its
    // first iteration from the corresponding row. Only fires when
    // CompressCenters=true (so root->compressed_centers exists) and the root
    // has internal children (i.e. flat_leaf_search isn't taken).
    parlay::sequence<CenterQuery> q_centers;
    parlay::sequence<std::pair<uint32_t, float>> root_dists;
    size_t num_root_children = 0;
    bool use_root_m2m = false;
    if constexpr (kHasCenterQuant) {
      if (search_params.root_m2m && root != nullptr) {
        num_root_children = root->compressed_centers.num_clouds();
        if (num_root_children > 0) {
          use_root_m2m = true;
          t.start();
          q_centers = parlay::sequence<CenterQuery>(num_q);
          parlay::parallel_for(0, num_q, [&](size_t i) {
            q_centers[i] = center_model_.quantize_query(eff_queries[i]);
          });
          t.stop();
          std::cout << "[MVIVF-Spill] Root M2M Quantize: " << t.total_time() << " sec" << std::endl;
          t.reset();

          t.start();
          turboquant_mv::FusedQueryBatch<metric> fq;
          std::vector<const CenterQuery*> q_ptrs(num_q);
          for (size_t i = 0; i < num_q; ++i) q_ptrs[i] = &q_centers[i];
          fq.Build(q_ptrs);
          root_dists = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(
              num_q * num_root_children);
          turboquant_mv::chamfer_score_all_fused(fq, root->compressed_centers, root_dists.data());
          t.stop();
          std::cout << "[MVIVF-Spill] Root M2M Score (" << num_q << "Q x " << num_root_children
                    << "C): " << t.total_time() << " sec" << std::endl;
          t.reset();
          bytes_accessed += root->compressed_centers.num_bytes();
        }
      }
    }

    // Step 1: parallel greedy search.  When root_m2m is on, hand each
    // greedy_search the precomputed row of root-child distances so it can
    // skip the per-query top-level scoring (which dominates greedy CPU when
    // the root has a large fan-out).
    t.start();
    auto leaf_query_pairs =
        parlay::sequence<std::pair<node_t*, std::pair<uint32_t, uint32_t>>>::uninitialized(
            nprobes * num_q);
    auto dist_cmps_gs = parlay::sequence<size_t>::uninitialized(num_q);
    auto bytes_gs = parlay::sequence<size_t>::uninitialized(num_q);
    parlay::parallel_for(0, num_q, [&](uint32_t i) {
      CenterQuery q_center{};
      if constexpr (kHasCenterQuant) {
        if (use_root_m2m) {
          q_center = std::move(q_centers[i]);
        } else {
          q_center = center_model_.quantize_query(eff_queries[i]);
        }
      }
      // Flat-leaf fast path is disabled (see search_with_stats); always
      // route through greedy_search.
      constexpr bool use_flat = false;
      [[maybe_unused]] const size_t _num_leaves_for_flat = num_leaves;
      const std::pair<uint32_t, float>* root_row =
          (use_root_m2m && !use_flat) ? root_dists.data() + i * num_root_children : nullptr;
      GreedySearchResult gs = use_flat
                                  ? flat_leaf_search(eff_queries[i], q_center, nprobes)
                                  : greedy_search(eff_queries[i], q_center, nprobes, root_row);
      dist_cmps_gs[i] = static_cast<size_t>(gs.stats.empty() ? 0.0 : gs.stats[0]);
      bytes_gs[i] = gs.bytes_accessed;
      const uint32_t np = static_cast<uint32_t>(std::min(nprobes, gs.probe_list.size()));
      // Pad the unused tail with nullptr so group_by_key_inplace can drop them.
      parlay::parallel_for(0, nprobes, [&](uint32_t j) {
        leaf_query_pairs[i * nprobes + j] =
            (j < np) ? std::make_pair(gs.probe_list[j].second, std::make_pair(i, j))
                     : std::make_pair(static_cast<node_t*>(nullptr), std::make_pair(i, j));
      });
    });
    t.stop();
    std::cout << "[MVIVF-Spill] Greedy Search: " << t.total_time() << " sec" << std::endl;
    t.reset();
    bytes_accessed += parlay::reduce(bytes_gs);
    dist_cmps += parlay::reduce(dist_cmps_gs);

    // ---------------------------------------------------------------------
    // Sub-batching path (OBTQ-only). Mirrors the OBTQ sub-batch path in
    // mvivf.h: reorganizes Step 2 + Step 3 around query-major sub-batches so
    // each core only touches queries from a single sub-batch of size B at any
    // moment, keeping the broadcasted query working set per core bounded by
    // B * qbuf_size and amortizing each leaf load over more queries.  The
    // spill-specific point-id dedup just layers on top of the same per-query
    // aggregation logic used below.
    // ---------------------------------------------------------------------
    if constexpr (std::is_same_v<LeafModel, turboquant_1bit_mv::Model<metric>>) {
      const char* sb_env = std::getenv("MVIVF_SUBBATCH");
      const size_t kSubbatchB = sb_env ? static_cast<size_t>(std::atoi(sb_env)) : 0;
      if (kSubbatchB > 0 && kSubbatchB <= 64) {
        using SetType = LeafSet;
        using QueryType = LeafQuery;
        using M2MType = typename has_many_to_many_<LeafModel>::type;

        // ----- Step 2': build per-sub-batch leaf-want lists -----
        t.start();
        const size_t B = kSubbatchB;
        const size_t num_subbatches = (num_q + B - 1) / B;
        const size_t num_rerank_sb = std::max(search_params.num_rerank, k);
        const size_t max_cands_per_query_sb = nprobes * num_rerank_sb;

        // Build leaf_ptr -> leaf_idx map once.
        std::unordered_map<node_t*, uint32_t> leaf_to_idx;
        leaf_to_idx.reserve(num_leaves * 2);
        for (uint32_t l = 0; l < num_leaves; ++l) leaf_to_idx[leaves_flat[l]] = l;

        struct LeafWant {
          uint32_t leaf_idx;
          uint64_t mask;
        };
        auto subbatch_lists = parlay::sequence<parlay::sequence<LeafWant>>(num_subbatches);

        parlay::parallel_for(0, num_subbatches, [&](size_t sb) {
          const size_t q_start = sb * B;
          const size_t q_end = std::min(q_start + B, num_q);
          const size_t b_size = q_end - q_start;
          // Pack (leaf_idx << 8) | q_off; b_size <= 64 fits in the low byte.
          std::vector<uint64_t> entries;
          entries.reserve(b_size * nprobes);
          for (size_t q = q_start; q < q_end; ++q) {
            const uint8_t q_off = static_cast<uint8_t>(q - q_start);
            for (size_t j = 0; j < nprobes; ++j) {
              auto* leaf_ptr = leaf_query_pairs[q * nprobes + j].first;
              if (!leaf_ptr) continue;  // padded slot for queries with fewer probes
              auto it = leaf_to_idx.find(leaf_ptr);
              if (it == leaf_to_idx.end()) continue;
              entries.push_back((static_cast<uint64_t>(it->second) << 8) | q_off);
            }
          }
          std::sort(entries.begin(), entries.end());
          std::vector<LeafWant> result;
          result.reserve(entries.size());
          for (size_t i = 0; i < entries.size();) {
            const uint32_t lidx = static_cast<uint32_t>(entries[i] >> 8);
            uint64_t mask = 0;
            while (i < entries.size() && static_cast<uint32_t>(entries[i] >> 8) == lidx) {
              mask |= (1ULL << (entries[i] & 0xff));
              ++i;
            }
            result.push_back({lidx, mask});
          }
          subbatch_lists[sb] = parlay::sequence<LeafWant>(result.begin(), result.end());
        });
        t.stop();
        std::cout << "[MVIVF-Spill] SB Group: " << t.total_time() << " sec" << std::endl;
        t.reset();

        // ----- Step 3': process leaves in (sub-batch, leaf) parallel pairs -----
        t.start();
        auto all_candidates_sb = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(
            num_q * max_cands_per_query_sb);
        auto cand_counts_sb =
            std::unique_ptr<std::atomic<size_t>[]>(new std::atomic<size_t>[num_q] {});

        auto safe_scatter_sb = [&](uint32_t q_id, const std::pair<uint32_t, float>* src,
                                   size_t count) {
          size_t slot = cand_counts_sb[q_id].fetch_add(count, std::memory_order_relaxed);
          if (slot + count > max_cands_per_query_sb) return;
          size_t base_idx = q_id * max_cands_per_query_sb + slot;
          for (size_t c = 0; c < count; ++c) all_candidates_sb[base_idx + c] = src[c];
        };

        auto leaf_bytes_sb = parlay::sequence<size_t>(num_subbatches, 0);
        auto leaf_cmps_sb = parlay::sequence<size_t>(num_subbatches, 0);

        parlay::parallel_for(0, num_subbatches, [&](size_t sb) {
          const size_t q_start = sb * B;
          const size_t q_end = std::min(q_start + B, num_q);
          const size_t b_size = q_end - q_start;
          std::vector<const QueryType*> sb_queries(b_size);
          for (size_t qi = 0; qi < b_size; ++qi) {
            sb_queries[qi] = &q_leaves[q_start + qi];
          }
          auto& leaves_for_sb = subbatch_lists[sb];

          auto per_leaf_cmps = parlay::sequence<size_t>(leaves_for_sb.size(), 0);
          parlay::parallel_for(0, leaves_for_sb.size(), [&](size_t li) {
            const uint32_t leaf_idx = leaves_for_sb[li].leaf_idx;
            const uint64_t mask = leaves_for_sb[li].mask;
            node_t* leaf = leaves_flat[leaf_idx];
            const SetType& leaf_data = leaf->encoded_leaf;
            const size_t C = std::min<size_t>(num_rerank_sb, leaf->get_size());

            // Gather wanted queries from sub-batch.
            std::vector<const QueryType*> wanted;
            wanted.reserve(b_size);
            std::vector<uint32_t> wanted_global;
            wanted_global.reserve(b_size);
            for (size_t qi = 0; qi < b_size; ++qi) {
              if (mask & (1ULL << qi)) {
                wanted.push_back(sb_queries[qi]);
                wanted_global.push_back(static_cast<uint32_t>(q_start + qi));
              }
            }
            if (wanted.empty()) return;

            std::vector<std::pair<uint32_t, float>> results(wanted.size() * num_rerank_sb);
            M2MType::TopKIntoUninitialized(wanted, leaf_data, num_rerank_sb, results.data(),
                                           /*q_block=*/4,
                                           /*parallel_query_blocks=*/false);
            per_leaf_cmps[li] = leaf->get_size() * wanted.size();
            for (size_t wi = 0; wi < wanted.size(); ++wi) {
              safe_scatter_sb(wanted_global[wi], results.data() + wi * num_rerank_sb, C);
            }
          });
          leaf_cmps_sb[sb] = parlay::reduce(per_leaf_cmps);
        });
        t.stop();
        std::cout << "[MVIVF-Spill] SB Probing & Scattering: " << t.total_time() << " sec"
                  << std::endl;
        t.reset();
        bytes_accessed += parlay::reduce(leaf_bytes_sb);
        dist_cmps += parlay::reduce(leaf_cmps_sb);

        // ----- Aggregation + Rerank (with spill dedup) -----
        // Identical to the non-SB aggregation below: compact sentinels, partial
        // sort to top-K window, sort-by-id + collapse adjacent dups (min value),
        // partial sort to top-num_rerank by value, then rerank.  See the
        // detailed comment in the non-SB aggregation block for why the K-window
        // bound `num_rerank * max_dups` is sufficient.
        if (search_params.tq8_rerank && search_params.num_rerank > 0) {
          parlay::internal::timer t_db;
          t_db.start();
          this->ensure_tq8_rerank_db_(points);
          t_db.stop();
          std::cout << "[MVIVF-Spill] SB TQ8 DB Encode (one-time): " << t_db.total_time() << " sec"
                    << std::endl;
        }
        t.start();
        auto final_results = parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>(num_q);
        auto bytes_accessed_rerank = parlay::sequence<size_t>::uninitialized(num_q);
        const size_t max_dups_sb =
            static_cast<size_t>(std::max<uint32_t>(params.num_spill, 1u)) *
            static_cast<size_t>(std::max<uint32_t>(params.num_spill_l2, 1u));
        parlay::parallel_for(0, num_q, [&](size_t q_id) {
          size_t base_idx = q_id * max_cands_per_query_sb;
          size_t num_scattered = std::min(cand_counts_sb[q_id].load(std::memory_order_relaxed),
                                          max_cands_per_query_sb);
          auto* cands = all_candidates_sb.begin() + base_idx;

          size_t num_valid = 0;
          for (size_t c = 0; c < num_scattered; ++c) {
            if (cands[c].first != UINT32_MAX) {
              if (num_valid != c) cands[num_valid] = cands[c];
              ++num_valid;
            }
          }

          const size_t K_part = std::min(num_valid, num_rerank_sb * max_dups_sb);
          if (K_part < num_valid) {
            std::nth_element(cands, cands + K_part, cands + num_valid,
                             [](const auto& a, const auto& b) { return a.second < b.second; });
          }

          std::sort(cands, cands + K_part,
                    [](const auto& a, const auto& b) { return a.first < b.first; });
          size_t deduped = 0;
          for (size_t c = 0; c < K_part;) {
            const uint32_t id = cands[c].first;
            float min_v = cands[c].second;
            size_t j = c + 1;
            while (j < K_part && cands[j].first == id) {
              if (cands[j].second < min_v) min_v = cands[j].second;
              ++j;
            }
            cands[deduped++] = {id, min_v};
            c = j;
          }

          size_t take = std::min(num_rerank_sb, deduped);
          if (take > 0 && take < deduped) {
            std::nth_element(cands, cands + take, cands + deduped,
                             [](const auto& a, const auto& b) { return a.second < b.second; });
          }

          parlay::sequence<std::pair<uint32_t, float>> top_cands;
          top_cands.reserve(take);
          for (size_t c = 0; c < take; ++c) top_cands.push_back(cands[c]);

          auto q_final = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(
              std::min(k, top_cands.size()));
          bytes_accessed_rerank[q_id] = 0;

          if (search_params.num_rerank > 0) {
            size_t actual_rerank = std::min(num_rerank_sb, top_cands.size());
            if (search_params.tq8_rerank) {
              bytes_accessed_rerank[q_id] = this->rerank_tq8_(rerank_queries[q_id], points,
                                                              top_cands, actual_rerank, q_final);
            } else {
              bytes_accessed_rerank[q_id] = this->rerank(rerank_queries[q_id], points, top_cands,
                                                         actual_rerank, q_final);
            }
          } else {
            for (size_t c = 0; c < q_final.size(); ++c) q_final[c] = top_cands[c];
          }

          final_results[q_id] = std::move(q_final);
        });
        t.stop();
        std::cout << "[MVIVF-Spill] SB Aggregation and Re-ranking: " << t.total_time() << " sec"
                  << std::endl;
        t.reset();
        bytes_accessed += parlay::reduce(bytes_accessed_rerank);
        std::cout << "[MVIVF-Spill] Bytes Accessed: " << bytes_accessed << std::endl;
        std::cout << "[MVIVF-Spill] Dist Cmps: " << dist_cmps << std::endl;
        return std::make_pair(std::move(final_results), bytes_accessed);
      }
    }

    // Step 2: group by leaf.
    //
    // group_by_key_inplace sorts by leaf pointer; nullptr pads sort to a
    // single trailing group which we ignore in step 3.
    t.start();
    auto grouped = mvsic::group_by_key_inplace(leaf_query_pairs);
    t.stop();
    std::cout << "[MVIVF-Spill] Grouping: " << t.total_time() << " sec" << std::endl;
    t.reset();

    // Step 3: probe each leaf in parallel.
    t.start();
    const size_t num_rerank = std::max(search_params.num_rerank, k);
    const size_t max_cands = nprobes * num_rerank;
    auto all_candidates =
        parlay::sequence<std::pair<uint32_t, float>>::uninitialized(num_q * max_cands);
    auto cand_counts = std::unique_ptr<std::atomic<size_t>[]>(new std::atomic<size_t>[num_q] {});
    auto leaf_bytes = parlay::sequence<size_t>::uninitialized(grouped.size());
    auto leaf_cmps = parlay::sequence<size_t>::uninitialized(grouped.size());

    auto safe_scatter = [&](uint32_t q_id, const std::pair<uint32_t, float>* src, size_t count) {
      size_t slot = cand_counts[q_id].fetch_add(count, std::memory_order_relaxed);
      if (slot + count > max_cands) return;
      size_t base_idx = q_id * max_cands + slot;
      for (size_t c = 0; c < count; ++c) all_candidates[base_idx + c] = src[c];
    };

    parlay::parallel_for(0, grouped.size(), [&](size_t i) {
      auto& group = grouped[i];
      node_t* leaf = group[0].first;
      leaf_bytes[i] = 0;
      leaf_cmps[i] = 0;
      if (leaf == nullptr) return;  // padded entries from queries with fewer probes
      size_t C = std::min<size_t>(num_rerank, leaf->get_size());
      size_t nq_grp = group.size();
      leaf_cmps[i] = leaf->get_size() * nq_grp;
      if constexpr (!kHasLeafQuant) {
        auto query_ids = parlay::delayed_tabulate(
            nq_grp, [&](size_t j) { return group[j].second.first; });
        PointCloudSet<ChPoint> batched(eff_queries.filter(query_ids), d);
        auto leaf_results = leaf->data.distances(batched, num_rerank);
        parlay::parallel_for(0, nq_grp, [&](size_t j) {
          uint32_t q_id = group[j].second.first;
          safe_scatter(q_id, leaf_results.begin() + j * num_rerank, C);
        });
      } else if constexpr (kUseM2M) {
        using M2M = typename has_many_to_many_<LeafModel>::type;
        std::vector<const LeafQuery*> typed(nq_grp);
        for (size_t j = 0; j < nq_grp; ++j) typed[j] = &q_leaves[group[j].second.first];
        std::vector<std::pair<uint32_t, float>> batch_results(nq_grp * num_rerank);
        M2M::TopKIntoUninitialized(typed, leaf->encoded_leaf, num_rerank, batch_results.data(),
                                   /*q_block=*/8, /*parallel_query_blocks=*/true);
        for (size_t j = 0; j < nq_grp; ++j) {
          uint32_t q_id = group[j].second.first;
          safe_scatter(q_id, batch_results.data() + j * num_rerank, C);
        }
      } else {
        parlay::parallel_for(0, nq_grp, [&](size_t j) {
          uint32_t q_id = group[j].second.first;
          auto all_dists =
              parlay::sequence<std::pair<uint32_t, float>>::uninitialized(leaf->get_size());
          leaf->encoded_leaf.distances_all(q_leaves[q_id], all_dists.data());
          if (C < all_dists.size()) {
            std::nth_element(all_dists.begin(), all_dists.begin() + C, all_dists.end(),
                             [](const auto& a, const auto& b) { return a.second < b.second; });
          }
          safe_scatter(q_id, all_dists.data(), C);
        });
      }
    });
    t.stop();
    std::cout << "[MVIVF-Spill] Probing & Scattering: " << t.total_time() << " sec" << std::endl;
    t.reset();
    bytes_accessed += parlay::reduce(leaf_bytes);
    dist_cmps += parlay::reduce(leaf_cmps);

    // Step 4: per-query dedup + aggregate + rerank.
    //
    // Dedup is required because spill replicates points across leaves (root
    // spill = a, level-1 spill = b), so a single point id can show up in
    // multiple per-leaf scatter buffers.  We do this sequentially per query
    // to avoid nesting parallel_for inside the outer parlay::parallel_for.
    //
    // The naive approach (full sort by id, collapse adjacent, nth_element)
    // pays O(N log N) where N = nprobes * num_rerank.  For large num_rerank
    // this dominates aggregation cost.  Instead we exploit a tight bound:
    // any single point id appears in the candidate pool at most `max_dups`
    // times, where max_dups <= num_spill * num_spill_l2 (root replication
    // factor times level-1 replication factor; deeper levels use top-1).
    // Therefore the top `num_rerank` distinct ids by value contribute at
    // most num_rerank * max_dups entries to the pool, all with values <=
    // the num_rerank-th distinct id's value.  So:
    //   (1) compact sentinels in place                           // O(N)
    //   (2) nth_element by value to bring top K = num_rerank *   // O(N)
    //       max_dups entries to the front
    //   (3) std::sort that K-window by id, collapse duplicates   // O(K log K)
    //       keeping min value
    //   (4) nth_element by value on the deduped window to get    // O(K')
    //       the top num_rerank distinct
    //   (5) rerank as in the regular MVIVF path
    if (search_params.tq8_rerank && search_params.num_rerank > 0) {
      parlay::internal::timer t_db;
      t_db.start();
      this->ensure_tq8_rerank_db_(points);
      t_db.stop();
      std::cout << "[MVIVF-Spill] TQ8 DB Encode (one-time): " << t_db.total_time() << " sec"
                << std::endl;
    }
    t.start();
    auto final_results = parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>(num_q);
    auto bytes_accessed_rerank = parlay::sequence<size_t>::uninitialized(num_q);
    parlay::parallel_for(0, num_q, [&](size_t q_id) {
      size_t base_idx = q_id * max_cands;
      size_t num_scattered = std::min(cand_counts[q_id].load(std::memory_order_relaxed), max_cands);
      auto* cands = all_candidates.begin() + base_idx;

      // (1) Compact: drop UINT32_MAX sentinels written by TopKIntoUninitialized
      // padding when a leaf had fewer than num_rerank points.
      size_t num_valid = 0;
      for (size_t c = 0; c < num_scattered; ++c) {
        if (cands[c].first != UINT32_MAX) {
          if (num_valid != c) cands[num_valid] = cands[c];
          ++num_valid;
        }
      }

      // (2) nth_element: bring the top K = num_rerank * max_dups entries by
      // distance to [0..K).  Because every distinct id has at most max_dups
      // occurrences in the pool, the top num_rerank distinct are guaranteed
      // to lie within these K entries.  Skipping the rest saves an
      // O(N log N) full sort when N >> K (typical: N = nprobes *
      // num_rerank, K = O(num_rerank)).
      const size_t max_dups =
          static_cast<size_t>(std::max<uint32_t>(params.num_spill, 1u)) *
          static_cast<size_t>(std::max<uint32_t>(params.num_spill_l2, 1u));
      const size_t K_part = std::min(num_valid, num_rerank * max_dups);
      if (K_part < num_valid) {
        std::nth_element(cands, cands + K_part, cands + num_valid,
                         [](const auto& a, const auto& b) { return a.second < b.second; });
      }

      // (3) Sequential dedup on the K-window: sort by id, collapse adjacent
      // duplicates keeping the smallest value.  In place to avoid alloc.
      std::sort(cands, cands + K_part,
                [](const auto& a, const auto& b) { return a.first < b.first; });
      size_t deduped = 0;
      for (size_t c = 0; c < K_part;) {
        const uint32_t id = cands[c].first;
        float min_v = cands[c].second;
        size_t j = c + 1;
        while (j < K_part && cands[j].first == id) {
          if (cands[j].second < min_v) min_v = cands[j].second;
          ++j;
        }
        cands[deduped++] = {id, min_v};
        c = j;
      }

      // (4) Top num_rerank by value (deduped is already <= 2 * num_rerank
      // after step 2, so this is cheap).
      size_t take = std::min(num_rerank, deduped);
      if (take > 0 && take < deduped) {
        std::nth_element(cands, cands + take, cands + deduped,
                         [](const auto& a, const auto& b) { return a.second < b.second; });
      }

      parlay::sequence<std::pair<uint32_t, float>> top_cands;
      top_cands.reserve(take);
      for (size_t c = 0; c < take; ++c) top_cands.push_back(cands[c]);

      auto q_final = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(
          std::min(k, top_cands.size()));
      bytes_accessed_rerank[q_id] = 0;

      // (4) Rerank.
      if (search_params.num_rerank > 0) {
        size_t actual_rerank = std::min(num_rerank, top_cands.size());
        if (search_params.tq8_rerank) {
          bytes_accessed_rerank[q_id] =
              this->rerank_tq8_(rerank_queries[q_id], points, top_cands, actual_rerank, q_final);
        } else {
          bytes_accessed_rerank[q_id] =
              this->rerank(rerank_queries[q_id], points, top_cands, actual_rerank, q_final);
        }
      } else {
        for (size_t c = 0; c < q_final.size(); ++c) q_final[c] = top_cands[c];
      }

      final_results[q_id] = std::move(q_final);
    });
    t.stop();
    std::cout << "[MVIVF-Spill] Aggregation and Re-ranking: " << t.total_time() << " sec"
              << std::endl;
    t.reset();
    bytes_accessed += parlay::reduce(bytes_accessed_rerank);

    std::cout << "[MVIVF-Spill] Bytes Accessed: " << bytes_accessed << std::endl;
    std::cout << "[MVIVF-Spill] Dist Cmps: " << dist_cmps << std::endl;

    return std::make_pair(std::move(final_results), bytes_accessed);
  }

  // Route the virtual `search_all` to the batched, leaf-grouped path so
  // every caller (including run_search_all_sweep in core/bench_utils.h, and
  // anything else going through the base Index<metric> interface) gets the
  // optimized path instead of falling through to the per-query parallel
  // loop in Index<metric>::search_all.  Mirrors IndexMVIVF::search_all.
  std::pair<parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>, size_t> search_all(
      const PointCloudSet<ChPoint>& query_points, const PointCloudSet<ChPoint>& points,
      const SearchParams& search_params) override {
    return search_all_new(query_points, points, search_params);
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
      size_t dims = node->data.get_dims();
      for (size_t i = 0; i < node->children.size(); ++i) {
        center_offsets.push_back(node->data.get_size(i) * dims);
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
  // Save / load (v4 skeleton format).
  //
  // Quantization-agnostic skeleton: writes magic + version + class_id +
  // IndexParams subset + tree topology + raw float centers + leaf point ids.
  // Codebooks and encoded leaves are never persisted; any templated variant
  // can load any skeleton from this family and re-train / re-encode on load
  // using the supplied raw points.  class_id is therefore fixed.
  //
  // Layout:
  //   u32 magic ('MSPL'), u32 version, u32 class_id
  //   IndexParams blob
  //   size_t num_nodes
  //   size_t num_center_offsets
  //   size_t[num_center_offsets] center_offsets
  //   float[sum] center_values
  //   size_t[num+1] children_offsets
  //   size_t[sum] children_values (flattened child indices)
  //   size_t[num+1] point_offsets
  //   uint32_t[sum] point_ids
  //
  // save() requires raw centers to be present, i.e. it is only valid on the
  // skeleton variant IndexMVIVFSpill<metric, /*CompressCenters=*/false,
  // NoQuantizer<metric>>.  Use the raw skeleton variant to build+save, then
  // load() into the desired templated variant.
  // ---------------------------------------------------------------------------
  static constexpr uint32_t kMagic = 0x4D53504Cu;  // 'MSPL'
  static constexpr uint32_t kVersion = 4u;  // v4: uniform skeleton format (no quant on disk)
  static constexpr uint32_t kClassId = 0u;

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
    // Skeleton save persists raw internal-node centers + leaf point ids only.
    // Codebooks and encoded leaves are never written; any templated variant
    // can load this file and re-derive its quantization on load using the
    // supplied raw points.  Quantized variants now keep `node->data`
    // populated alongside `compressed_centers` (see compress_internal_centers_),
    // so save() is valid for all variants.
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
    const uint32_t magic = kMagic, ver = kVersion, cid = kClassId;
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

    for (size_t i = 0; i < num; ++i) {
      node_t* node = ind_to_node[i];
      if (node->children.size() > 0) {
        auto coords = node->data.data();
        size_t nent = node->data.total_size() * node->data.get_dims();
        outfile.write(reinterpret_cast<const char*>(coords), nent * sizeof(float));
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
      std::cerr << "[MVIVF Spill] bad magic: file is not an MSPL skeleton index." << std::endl;
      return;
    }
    if (ver != kVersion) {
      std::cerr << "[MVIVF Spill] MVIVF Spill index file format changed in v" << kVersion
                << "; got v" << ver << ". Rebuild with current code." << std::endl;
      return;
    }
    if (cid != kClassId) {
      std::cerr << "[MVIVF Spill] unexpected class_id " << cid << " (expected " << kClassId
                << " for the v" << kVersion << " skeleton)." << std::endl;
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
    parlay::sequence<float> center_values(center_offsets[center_offsets.size() - 1]);
    infile.read(reinterpret_cast<char*>(center_values.begin()),
                center_values.size() * sizeof(float));
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
    double t_io_ms = t_io.stop() * 1000.0;

    parlay::internal::timer t_retrain;
    t_retrain.start();
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
    if (root != nullptr) {
      // Defensive cleanup.  In practice load() is called on a fresh instance.
    }
    root = ind_to_node[0];

    // compute_leaf_flat_() needs raw centers in internal nodes; run it before
    // compress_internal_centers_() clears them.
    compute_leaf_flat_();
    compress_internal_centers_();

    if constexpr (kHasLeafQuant) {
      parlay::parallel_for(0, num, [&](size_t i) {
        node_t* node = ind_to_node[i];
        if (node && node->children.empty() && node->data.size() > 0) {
          node->encoded_leaf = leaf_model_.encode(node->data);
        }
      }, 1);
    }
    double t_retrain_ms = t_retrain.stop() * 1000.0;
    std::cerr << "[MVIVF Spill] load: tree_io=" << t_io_ms << "ms retrain=" << t_retrain_ms
              << "ms (compress_centers=" << (kHasCenterQuant ? 1 : 0)
              << ", leaf_quant=" << (kHasLeafQuant ? 1 : 0) << ")" << std::endl;
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
using IndexMVIVFSpillOneBitTQAsymIP = IndexMVIVFSpill<false, false, turboquant_1bit_asym_mv::Model<false>>;
using IndexMVIVFSpillOneBitTQAsymL2 = IndexMVIVFSpill<true,  false, turboquant_1bit_asym_mv::Model<true>>;
using IndexMVIVFSpillEightBitTQIP = IndexMVIVFSpill<false, false, turboquant_8bit_mv::Model<false>>;
using IndexMVIVFSpillEightBitTQL2 = IndexMVIVFSpill<true,  false, turboquant_8bit_mv::Model<true>>;

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
using IndexMVIVFSpillCompressOneBitTQAsymIP = IndexMVIVFSpill<false, true, turboquant_1bit_asym_mv::Model<false>>;
using IndexMVIVFSpillCompressOneBitTQAsymL2 = IndexMVIVFSpill<true,  true, turboquant_1bit_asym_mv::Model<true>>;
using IndexMVIVFSpillCompressEightBitTQIP = IndexMVIVFSpill<false, true, turboquant_8bit_mv::Model<false>>;
using IndexMVIVFSpillCompressEightBitTQL2 = IndexMVIVFSpill<true,  true, turboquant_8bit_mv::Model<true>>;

}  // namespace mvsic
