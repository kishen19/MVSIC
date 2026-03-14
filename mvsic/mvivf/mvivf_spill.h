#pragma once

#include <algorithm>
#include <functional>
#include <optional>
#include <queue>
#include <set>
#include <type_traits>
#include <variant>
#include <vector>

// #include "absl/container/btree_set.h"

#include "mvsic/core/index.h"
#include "mvsic/core/mvclustering/mvclustering.h"
#include "mvsic/core/utils/util.h"
#include "mvsic/core/distance_measures/many_to_many.h"

// Quantization Headers
#include "mvsic/core/quantization/pq.h"
#include "mvsic/core/quantization/rabitq.h"
#include "mvsic/core/quantization/fastscan.h"
#include "mvsic/core/quantization/turboquant_4bit.h"
#include "mvsic/core/quantization/turboquant_pq_4bit.h"
#include "mvsic/core/quantization/wrapper.h"

namespace mvsic {

/* Multi-Vector IVF Index
  Indexing:
  - Builds a kmeans Tree by recursively running MV-Lloyd's algorithm on the input point clouds.
  - At each level, if the cluster size is larger than `max_leaf_size`, it runs the MV-Lloyd's
  algorithm on that cluster with `num_clusters` clusters. Each cluster is represented by a "center"
  point cloud. Search:
  - For a given query point cloud, it runs a greedy search and determines `nprobes` top candidate
  leaf-level clusters to probe.
  - For each candidate probe cluster, it computes the top-k point clouds from that cluster.
  - Finally, it re-ranks the results based on distances to return the top-k point clouds.
*/

template<bool metric>
class IndexMVIVFSpill : public Index<metric> {
 public:
  using ChPoint = typename Index<metric>::ChPoint;  // Chamfer Point Type
  using Index<metric>::d;                           // Embedding dimension

  // Multi-Vector Quantizer Types using the Wrapper
  using FlatRange = FlattenedPCRange<PointCloudSet<ChPoint>>;
  using PQ_Enc = pq::Quantized_Point_Range<FlatRange, metric>;
  using FS_Enc = fastscan::Quantized_Point_Range<FlatRange, metric>;
  using RQ_Enc = rabitq::Quantized_Point_Range<FlatRange, metric>;
  using TQ4_Enc = turboquant_4bit::Quantized_Point_Range<FlatRange, metric>;
  using TQPQ4_Enc = turboquant_pq_4bit::Quantized_Point_Range<FlatRange, metric, 4>;
  using TQPQ8_Enc = turboquant_pq_4bit::Quantized_Point_Range<FlatRange, metric, 8>;

  using PQ_Set = Quantized_Point_Cloud_Set<PQ_Enc, metric>;
  using FS_Set = Quantized_Point_Cloud_Set<FS_Enc, metric>;
  using RQ_Set = Quantized_Point_Cloud_Set<RQ_Enc, metric>;
  using TQ4_Set = Quantized_Point_Cloud_Set<TQ4_Enc, metric>;
  using TQPQ4_Set = Quantized_Point_Cloud_Set<TQPQ4_Enc, metric>;
  using TQPQ8_Set = Quantized_Point_Cloud_Set<TQPQ8_Enc, metric>;
  using QuantSet =
      std::variant<std::monostate, PQ_Set, FS_Set, RQ_Set, TQ4_Set, TQPQ4_Set, TQPQ8_Set>;

  using PQ_Model = MultiVecQuantizer<pq::Model<metric>, metric>;
  using FS_Model = MultiVecQuantizer<fastscan::Model<metric>, metric>;
  using RQ_Model = MultiVecQuantizer<rabitq::Model<metric>, metric>;
  using TQ4_Model = MultiVecQuantizer<turboquant_4bit::Model<metric>, metric>;
  using TQPQ4_Model = MultiVecQuantizer<turboquant_pq_4bit::Model<metric, 4>, metric>;
  using TQPQ8_Model = MultiVecQuantizer<turboquant_pq_4bit::Model<metric, 8>, metric>;
  using QuantModel = std::variant<std::monostate, PQ_Model, FS_Model, RQ_Model, TQ4_Model,
                                  TQPQ4_Model, TQPQ8_Model>;

  // helper for decltype
  template<class M, class Q>
  using QQueryT = decltype(std::declval<M &>().quantize_query(std::declval<Q const &>()));
  using PQ_Q = QQueryT<PQ_Model, ChPoint>;
  using FS_Q = QQueryT<FS_Model, ChPoint>;
  using RQ_Q = QQueryT<RQ_Model, ChPoint>;
  using TQ4_Q = QQueryT<TQ4_Model, ChPoint>;
  using TQPQ4_Q = QQueryT<TQPQ4_Model, ChPoint>;
  using TQPQ8_Q = QQueryT<TQPQ8_Model, ChPoint>;
  using QuantQuery = std::variant<std::monostate, PQ_Q, FS_Q, RQ_Q, TQ4_Q, TQPQ4_Q, TQPQ8_Q>;

  using QT = IndexParams::QuantizerType;

  // kmeans tree nodes
  struct node_t {
    parlay::sequence<node_t *> children;
    // For internal nodes: data = centers of children
    // For leaves:         data = points in the cluster
    PointCloudSet<ChPoint> data;
    QuantSet quantized_data;  // std::monostate for inner nodes and unquantized leaves.

    node_t() noexcept : children(), data(), quantized_data(std::monostate{}) {}
    ~node_t() noexcept {}

    inline size_t get_size() const noexcept { return data.size(); }
  };

  IndexParams params;
  node_t *root = nullptr;  // Root of the k-means tree

  // Leaf Data: Accessed directly during search when nprobes is large
  std::vector<node_t *> leaves_flat;    // Flat list of leaves for large-nprobes search
  PointCloudSet<ChPoint> leaf_centers;  // One representative center per leaf
  QuantSet leaf_centers_quant;          // Quantized centers for flat search (when enabled)

  // Quantizer Storage
  QuantModel quantizer = std::monostate{};
  QT active_quantizer = QT::None;

  // Center quantization: when params.quantize_centers is enabled, we *always* use TurboQuant4Bit
  // (TQ4) for internal-node / leaf-center scoring, regardless of the leaf quantizer.
  //
  // If the leaf quantizer is already TQ4, we reuse the same model instance (no duplicate training).
  std::optional<TQ4_Model> center_quantizer_owned;
  TQ4_Model *center_quantizer = nullptr;  // points to either owned model or leaf model

  // Stats
  size_t kmeanstree_height = 0;

  IndexMVIVFSpill(size_t d_) noexcept : params(IndexParams::mvivf()) { d = d_; }
  IndexMVIVFSpill(size_t d_, const IndexParams &params) noexcept : params(params) { d = d_; }

  void init_center_quantizer(const PointCloudSet<ChPoint> &points) {
    center_quantizer = nullptr;
    center_quantizer_owned.reset();
    if (!params.quantize_centers) return;

    if (active_quantizer == QT::TurboQuant4Bit) {
      center_quantizer = &std::get<TQ4_Model>(quantizer);
      return;
    }

    // Train a dedicated TQ4 model for center scoring.
    center_quantizer_owned.emplace();
    center_quantizer_owned->train(points);
    center_quantizer = &(*center_quantizer_owned);
  }

  QuantSet encode_points_quantized(const PointCloudSet<ChPoint> &points) {
    switch (active_quantizer) {
      case QT::PQ: return std::get<PQ_Model>(quantizer).encode(points);
      case QT::FastScan: return std::get<FS_Model>(quantizer).encode(points);
      case QT::RaBitQ: return std::get<RQ_Model>(quantizer).encode(points);
      case QT::TurboQuant4Bit: return std::get<TQ4_Model>(quantizer).encode(points);
      case QT::TurboQuantPQ4Bit:
        if (params.pq.block_size == 4) return std::get<TQPQ4_Model>(quantizer).encode(points);
        return std::get<TQPQ8_Model>(quantizer).encode(points);
      case QT::None:
      default: return std::monostate{};
    }
  }

  QuantSet encode_centers_tq(const PointCloudSet<ChPoint> &points) {
    if (!center_quantizer) return std::monostate{};
    return center_quantizer->encode(points);  // returns TQ4_Set (wrapped in QuantSet)
  }

  void quant_distances_all(const QuantQuery &q_var, const QuantSet &s_var,
                           std::pair<uint32_t, float> *out) const {
    switch (active_quantizer) {
      case QT::PQ: std::get<PQ_Set>(s_var).distances_all(std::get<PQ_Q>(q_var), out); break;
      case QT::FastScan: std::get<FS_Set>(s_var).distances_all(std::get<FS_Q>(q_var), out); break;
      case QT::RaBitQ: std::get<RQ_Set>(s_var).distances_all(std::get<RQ_Q>(q_var), out); break;
      case QT::TurboQuant4Bit:
        std::get<TQ4_Set>(s_var).distances_all(std::get<TQ4_Q>(q_var), out);
        break;
      case QT::TurboQuantPQ4Bit:
        if (params.pq.block_size == 4) {
          std::get<TQPQ4_Set>(s_var).distances_all(std::get<TQPQ4_Q>(q_var), out);
        } else {
          std::get<TQPQ8_Set>(s_var).distances_all(std::get<TQPQ8_Q>(q_var), out);
        }
        break;
      case QT::None:
      default: break;
    }
  }

  // Recursive kmeans tree builder (used for all levels below the root).
  void recursive_build(node_t *node, const PointCloudSet<ChPoint> &points) {
    size_t n = points.size();
    // Number of centers: Dynamic
    // main_val: either k_per_level or sqrt(n)
    // small_nc: If n is small, choosing ~ n/max_leaf_size is better
    size_t auto_nc = (params.k_per_level > 0) ? params.k_per_level
                                              : static_cast<size_t>(std::ceil(std::sqrt(n)));
    size_t small_nc = 4 * (n + params.max_leaf_size - 1) / params.max_leaf_size;
    size_t num_clusters = std::max(static_cast<size_t>(4), std::min({auto_nc, small_nc, n}));
    if (params.verbose >= 1) {
      std::cout << "Building index with " << n << " points, num_clusters: " << num_clusters
                << std::endl;
    }
    // Step 1: Run MV-Lloyds on points
    MVClustering<metric> Clus(d, num_clusters, params.s, params.mvclus);
    Clus.train(points);
    PointCloudSet<ChPoint> &centers = Clus.centers;
    parlay::sequence<uint32_t> &cluster_ids = Clus.cluster_ids;
    // Step 2: Collect Clusters
    auto id_pt = parlay::tabulate(n, [&](uint32_t i) { return std::make_pair(cluster_ids[i], i); });
    auto grouped = group_by_key_inplace(id_pt);
    // Step 3: Update children nodes and recurse for large nodes
    node->children.resize(grouped.size());
    if (grouped.size() < centers.size()) {
      auto active_centers_ind = parlay::delayed_seq<uint32_t>(
          grouped.size(), [&](size_t i) { return grouped[i][0].first; });
      node->data = PointCloudSet<ChPoint>(centers.filter(active_centers_ind), d);
    } else {
      node->data = std::move(centers);
    }
    // Quantize internal-node centers if requested (always using TQ4).
    if (params.quantize_centers) {
      node->quantized_data = encode_centers_tq(node->data);
    }
    parlay::parallel_for(
        0, grouped.size(),
        [&](size_t i) {
          auto cluster_id = grouped[i][0].first;
          auto group = parlay::delayed_seq<uint32_t>(
              grouped[i].size(), [&](size_t j) { return grouped[i][j].second; });
          PointCloudSet<ChPoint> child_points = PointCloudSet<ChPoint>(points.filter(group), d);
          node_t *child = new node_t();
          node->children[i] = child;
          if (child_points.size() > params.max_leaf_size) {  // Recurse
            recursive_build(child, child_points);
          } else {  // Leaf Node
            child->data = std::move(child_points);
            // Always quantize leaf nodes when a quantizer is active; Step 2 of search assumes
            // that leaf->quantized_data holds encoded points whenever active_quantizer != None.
            if (active_quantizer != QT::None) {
              child->quantized_data = encode_points_quantized(child->data);
            }
          }
        },
        1);
  }

  // Builds the index given PointCloudSet object.
  // For IndexMVIVFSpill, we modify only the **first level** of the tree:
  // every point cloud is assigned to its closest and second-closest root center,
  // effectively duplicating it across two top-level subtrees. All deeper levels
  // are built using the standard recursive_build().
  void build(const PointCloudSet<ChPoint> &points) override {
    root = new node_t();
    if (params.compress_input) {
      // TODO: run Ward's HAC to compress input point clouds
    }

    // Quantization
    active_quantizer = params.pq.method;
    switch (active_quantizer) {
      case QT::PQ: {
        quantizer.template emplace<PQ_Model>();
        std::get<PQ_Model>(quantizer).train(points, params.pq.block_size,
                                            params.pq.num_clusters_per_block,
                                            params.pq.num_points_per_cluster);
        break;
      }
      case QT::FastScan: {
        quantizer.template emplace<FS_Model>();
        std::get<FS_Model>(quantizer).train(points, params.pq.block_size);
        break;
      }
      case QT::RaBitQ: {
        quantizer.template emplace<RQ_Model>();
        std::get<RQ_Model>(quantizer).train(points, params.pq.rabitq_bits);
        break;
      }
      case QT::TurboQuant4Bit: {
        quantizer.template emplace<TQ4_Model>();
        std::get<TQ4_Model>(quantizer).train(points);
        break;
      }
      case QT::TurboQuantPQ4Bit: {
        if (params.pq.block_size == 4) {
          quantizer.template emplace<TQPQ4_Model>();
          std::get<TQPQ4_Model>(quantizer).train(points);
        } else if (params.pq.block_size == 8) {
          quantizer.template emplace<TQPQ8_Model>();
          std::get<TQPQ8_Model>(quantizer).train(points);
        } else {
          std::cerr << "IndexMVIVFSpill: TurboQuantPQ4Bit currently supports block_size 4 or 8 "
                    << "(got " << params.pq.block_size << ")." << std::endl;
          abort();
        }
        break;
      }
      default: quantizer = std::monostate{}; break;
    }

    // Center quantization (internal-node / leaf-center scoring): always TQ4.
    init_center_quantizer(points);

    // ---------- First level (root) with spill ----------
    size_t n = points.size();
    size_t auto_nc = (params.k_per_level > 0)
                         ? params.k_per_level
                         : static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(n))));
    size_t small_nc = 4 * (n + params.max_leaf_size - 1) / params.max_leaf_size;
    size_t num_clusters = std::max(static_cast<size_t>(4), std::min({auto_nc, small_nc, n}));
    if (params.verbose >= 1) {
      std::cout << "IndexMVIVFSpill: building root with " << n
                << " points, num_clusters: " << num_clusters << std::endl;
    }

    // Run MV-Lloyds once on the full dataset to get root centers.
    MVClustering<metric> Clus(d, num_clusters, params.s, params.mvclus);
    Clus.train(points);
    PointCloudSet<ChPoint> &centers = Clus.centers;
    size_t C = centers.size();

    // For each point, compute distances to all centers (as in compute_cluster_ids_naive)
    // and record both the best and second-best center indices.
    parlay::sequence<uint32_t> best_idx(n);
    parlay::sequence<uint32_t> second_idx(n);
    parlay::parallel_for(0, n, [&](uint32_t i) {
      auto dist =
          parlay::delayed_tabulate(C, [&](uint32_t j) { return points[i].distance(centers[j]); });
      uint32_t best = 0, second = 0;
      float best_d = std::numeric_limits<float>::infinity();
      float second_d = std::numeric_limits<float>::infinity();
      for (uint32_t j = 0; j < C; ++j) {
        float d_ij = dist[j];
        if (d_ij < best_d) {
          second_d = best_d;
          second = best;
          best_d = d_ij;
          best = j;
        } else if (d_ij < second_d) {
          second_d = d_ij;
          second = j;
        }
      }
      best_idx[i] = best;
      // Spill to second-best only when distances are very close.
      const float ratio = params.spill_ratio;
      bool spill = (second != best) && (ratio > 0.0f) &&
                   (best_d <= 1e-9f || second_d <= best_d * (1.0f + ratio));
      second_idx[i] = spill ? second : UINT32_MAX;
    });

    // Build (center_id, point_id) pairs for both best and second-best assignments.
    auto id_pt =
        parlay::sequence<std::pair<uint32_t, uint32_t>>::from_function(2 * n, [&](size_t t) {
          uint32_t i = static_cast<uint32_t>(t / 2);
          bool is_second = (t % 2 == 1);
          if (!is_second) {
            return std::make_pair(best_idx[i], i);
          }
          uint32_t s = second_idx[i];
          if (s == UINT32_MAX) {
            return std::make_pair(UINT32_MAX, i);
          }
          return std::make_pair(s, i);
        });
    auto id_pt_valid = parlay::filter(
        id_pt, [&](const std::pair<uint32_t, uint32_t> &p) { return p.first != UINT32_MAX; });
    auto grouped = group_by_key_inplace(id_pt_valid);

    // Root stores centers corresponding to active (non-empty) groups.
    root->children.resize(grouped.size());
    if (grouped.size() < centers.size()) {
      auto active_centers_ind = parlay::delayed_seq<uint32_t>(
          grouped.size(), [&](size_t i) { return grouped[i][0].first; });
      root->data = PointCloudSet<ChPoint>(centers.filter(active_centers_ind), d);
    } else {
      root->data = std::move(centers);
    }
    if (params.quantize_centers) {
      root->quantized_data = encode_centers_tq(root->data);
    }

    // Build subtrees below each root child using standard recursive_build().
    parlay::parallel_for(
        0, grouped.size(),
        [&](size_t i) {
          auto group = parlay::delayed_seq<uint32_t>(
              grouped[i].size(), [&](size_t j) { return grouped[i][j].second; });
          PointCloudSet<ChPoint> child_points(points.filter(group), d);
          node_t *child = new node_t();
          root->children[i] = child;
          if (child_points.size() > params.max_leaf_size) {
            recursive_build(child, child_points);
          } else {
            child->data = std::move(child_points);
            if (active_quantizer != QT::None) {
              child->quantized_data = encode_points_quantized(child->data);
            }
          }
        },
        1);

    // Build flat leaf index and leaf centers for optional flat-leaf search
    leaves_flat.clear();
    leaf_centers_quant = std::monostate{};
    if (root) {
      std::vector<ChPoint> center_points;
      std::function<void(node_t *, node_t *, size_t)> visit = [&](node_t *node, node_t *parent,
                                                                  size_t child_idx) {
        if (node->children.empty()) {
          leaves_flat.push_back(node);
          auto &centers_pc = parent->data;
          center_points.push_back(centers_pc[child_idx]);
        } else {
          for (size_t i = 0; i < node->children.size(); ++i) {
            visit(node->children[i], node, i);
          }
        }
      };
      visit(root, nullptr, 0);

      if (!center_points.empty()) {
        leaf_centers = PointCloudSet<ChPoint>(center_points, d);

        if (params.quantize_centers) {
          leaf_centers_quant = encode_centers_tq(leaf_centers);
        }
      }
    }
  }

  // Output type of Greedy Search
  struct GreedySearchResult {
    parlay::sequence<std::pair<float, node_t *>> probe_list;
    size_t dist_cmps = 0;
    std::vector<double> timings = {};
  };

  // Simple Beam Search using std::set
  GreedySearchResult greedy_search(const ChPoint &query, const QuantQuery &q_query_var,
                                   const TQ4_Q *q_center_query, size_t nprobes) const {
    using score_node = std::pair<float, node_t *>;
    auto less = [](const score_node &a, const score_node &b) {
      return a.first < b.first || (a.first == b.first && a.second < b.second);
    };
    // Modifying this doesn't affect performance significantly, as beam management is not the
    // bottleneck.
    const size_t beam_length = 2 * nprobes;
    parlay::internal::timer t;
    double t_dists = 0.0;
    double t_beam = 0.0;
    double t_rest = 0.0;

    t.start();
    size_t dist_cmps = 0;
    std::set<score_node> beam;
    parlay::sequence<score_node> top_probes;
    top_probes.reserve(nprobes + 1);
    std::vector<std::pair<uint32_t, float>> child_dists;  // Scratch memory
    child_dists.reserve(root->children.size());           // Reserve reasonable initial capacity
    t_rest += t.stop();
    t.reset();

    // Initial seed
    t.start();
    beam.insert({0.0f, root});
    t_beam += t.stop();
    t.reset();

    while (!beam.empty()) {
      // Pop the best node (smallest distance)
      t.start();
      auto it = beam.begin();
      score_node best = *it;
      beam.erase(it);
      t_beam += t.stop();
      t.reset();
      t.start();
      node_t *current_node = best.second;
      // Check if current score is worse than the worst in top_probes
      if (top_probes.size() == nprobes && best.first >= top_probes.front().first) {
        break;  // Beam has converged
      }
      auto &children = current_node->children;
      if (children.empty()) continue;
      // Compute distances to children
      child_dists.resize(children.size());
      dist_cmps += children.size();
      if (!params.quantize_centers) {
        auto &centers = current_node->data;
        centers.distances(query, child_dists.data());
      } else {
        // Center scoring always uses TQ4 when enabled.
        if (q_center_query) {
          std::get<TQ4_Set>(current_node->quantized_data)
              .distances_all(*q_center_query, child_dists.data());
        } else {
          // Should not happen in practice; fall back to exact.
          auto &centers = current_node->data;
          centers.distances(query, child_dists.data());
        }
      }
      t_dists += t.stop();
      t.reset();

      for (size_t i = 0; i < children.size(); ++i) {
        float d = child_dists[i].second;
        node_t *child = children[i];
        if (child->children.empty()) {
          t.start();
          // It's a leaf node: add to probe candidates if better than current worst
          if (top_probes.size() < nprobes || d < top_probes.front().first) {
            top_probes.push_back({d, child});
            std::push_heap(top_probes.begin(), top_probes.end());
            if (top_probes.size() > nprobes) {
              std::pop_heap(top_probes.begin(), top_probes.end());
              top_probes.pop_back();  // Prune the farthest node
            }
          }
          t_rest += t.stop();
          t.reset();
        } else {
          t.start();
          // Internal node: add to beam if it's better than the current worst
          const size_t beam_size = beam.size();
          if (beam_size < beam_length) {
            beam.insert({d, child});
          } else {
            // Only check worst if beam is full
            auto worst_it = std::prev(beam.end());
            if (d < worst_it->first) {
              beam.erase(worst_it);
              beam.insert({d, child});
            }
          }
          t_beam += t.stop();
          t.reset();
        }
      }
    }

    GreedySearchResult out;
    out.dist_cmps = dist_cmps;
    out.timings = {t_dists, t_beam, t_rest};
    out.probe_list = std::move(top_probes);
    return out;
  }

  // Flat leaf scoring: score all leaves by distance to a precomputed representative center and
  // pick the best nprobes leaves.
  GreedySearchResult flat_leaf_search(const ChPoint &query, const QuantQuery &q_query_var,
                                      const TQ4_Q *q_center_query, size_t nprobes) const {
    using score_node = std::pair<float, node_t *>;
    GreedySearchResult out;
    if (leaves_flat.empty() || leaf_centers.size() == 0 || nprobes == 0) {
      return out;
    }

    const size_t L = leaf_centers.size();
    const size_t use_nprobes = std::min(nprobes, L);

    parlay::internal::timer t;
    double t_dists = 0.0;
    double t_rest = 0.0;

    t.start();
    size_t dist_cmps = L;
    auto centers_dists = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(L);
    auto scores = parlay::sequence<score_node>::uninitialized(L);

    if (!params.quantize_centers) {
      // Exact Chamfer distance to the representative centers.
      leaf_centers.distances(query, centers_dists.data());
    } else {
      if (q_center_query) {
        std::get<TQ4_Set>(leaf_centers_quant).distances_all(*q_center_query, centers_dists.data());
      } else {
        leaf_centers.distances(query, centers_dists.data());
      }
    }
    parlay::parallel_for(0, L, [&](size_t i) {
      scores[i] = {centers_dists[i].second, leaves_flat[i]};
    });
    t_dists += t.stop();
    t.reset();

    // Select top-nprobes leaves.
    t.start();
    if (use_nprobes < L) {
      std::nth_element(scores.begin(), scores.begin() + use_nprobes, scores.end(),
                       [](const score_node &a, const score_node &b) { return a.first < b.first; });
      scores.resize(use_nprobes);
    }
    t_rest += t.stop();
    t.reset();

    out.probe_list = std::move(scores);
    out.dist_cmps = dist_cmps;
    out.timings = {t_dists, /*t_beam=*/0.0, t_rest};
    return out;
  }

  std::tuple<parlay::sequence<std::pair<uint32_t, float>>, size_t, std::vector<double>>
  search_with_stats(const ChPoint &query, const PointCloudSet<ChPoint> &points,
                    const SearchParams &search_params) override {
    parlay::internal::timer t;
    std::vector<double> timings;

    size_t k = search_params.k;
    size_t nprobes = search_params.nprobes;
    size_t dist_cmps = 0;

    double t_quantize = 0.0;
    double t_distances = 0.0;
    double t_rest = 0.0;

    // -------------------------
    // Step 0: Quantize Query
    // -------------------------
    QuantQuery q_query_var;
    TQ4_Q q_center_query;
    const TQ4_Q *q_center_query_ptr = nullptr;
    t.start();
    switch (active_quantizer) {
      case QT::PQ: q_query_var = std::get<PQ_Model>(quantizer).quantize_query(query); break;
      case QT::FastScan: q_query_var = std::get<FS_Model>(quantizer).quantize_query(query); break;
      case QT::RaBitQ: q_query_var = std::get<RQ_Model>(quantizer).quantize_query(query); break;
      case QT::TurboQuant4Bit:
        q_query_var = std::get<TQ4_Model>(quantizer).quantize_query(query);
        break;
      case QT::TurboQuantPQ4Bit:
        if (params.pq.block_size == 4) {
          q_query_var = std::get<TQPQ4_Model>(quantizer).quantize_query(query);
        } else {
          q_query_var = std::get<TQPQ8_Model>(quantizer).quantize_query(query);
        }
        break;
      case QT::None:
      default: q_query_var = std::monostate{}; break;
    }
    if (params.quantize_centers && center_quantizer) {
      q_center_query = center_quantizer->quantize_query(query);
      q_center_query_ptr = &q_center_query;
    }
    t_quantize = t.stop();
    t.reset();

    // -------------------------
    // Step 1: Greedy search
    // -------------------------
    GreedySearchResult gs;
    const size_t num_leaves = leaves_flat.size();
    const double alpha = 1.0;  // heuristic threshold: TODO: set this
    bool use_flat = (num_leaves > 0 && nprobes >= static_cast<size_t>(alpha * num_leaves));
    if (use_flat) {
      gs = flat_leaf_search(query, q_query_var, q_center_query_ptr, nprobes);
    } else {
      gs = greedy_search(query, q_query_var, q_center_query_ptr, nprobes);
    }
    auto probe_list = std::move(gs.probe_list);
    dist_cmps += gs.dist_cmps;
    nprobes = std::min(nprobes, probe_list.size());

    // -------------------------
    // Step 2: Probe clusters in probe_list
    // -------------------------
    t.start();
    auto sizes = parlay::delayed_tabulate(
        nprobes, [&](size_t i) { return probe_list[i].second->get_size(); });
    auto scan_result = parlay::scan(sizes);
    auto &offsets = scan_result.first;
    size_t total_size = scan_result.second;
    auto visited = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(total_size);
    t_rest += t.stop();
    t.reset();

    if (active_quantizer == QT::None) {
      t_quantize = 0.0;
      t.start();
      parlay::parallel_for(0, nprobes, [&](size_t i) {
        node_t *leaf_node = probe_list[i].second;
        leaf_node->data.distances(query, &visited[offsets[i]]);
      });
      t_distances = t.stop();
      t.reset();
    } else {
      t.start();
      switch (active_quantizer) {
        case QT::PQ:
          parlay::parallel_for(0, nprobes, [&](size_t i) {
            node_t *leaf = probe_list[i].second;
            std::get<PQ_Set>(leaf->quantized_data)
                .distances_all(std::get<PQ_Q>(q_query_var), &visited[offsets[i]]);
            const size_t leaf_k = leaf->get_size();
            parlay::parallel_for(
                0, leaf_k, [&](size_t j) { visited[offsets[i] + j].first = leaf->data.get_id(j); });
          });
          break;
        case QT::FastScan:
          parlay::parallel_for(0, nprobes, [&](size_t i) {
            node_t *leaf = probe_list[i].second;
            std::get<FS_Set>(leaf->quantized_data)
                .distances_all(std::get<FS_Q>(q_query_var), &visited[offsets[i]]);
            const size_t leaf_k = leaf->get_size();
            parlay::parallel_for(
                0, leaf_k, [&](size_t j) { visited[offsets[i] + j].first = leaf->data.get_id(j); });
          });
          break;
        case QT::RaBitQ:
          parlay::parallel_for(0, nprobes, [&](size_t i) {
            node_t *leaf = probe_list[i].second;
            std::get<RQ_Set>(leaf->quantized_data)
                .distances_all(std::get<RQ_Q>(q_query_var), &visited[offsets[i]]);
            const size_t leaf_k = leaf->get_size();
            parlay::parallel_for(
                0, leaf_k, [&](size_t j) { visited[offsets[i] + j].first = leaf->data.get_id(j); });
          });
          break;
        case QT::TurboQuant4Bit:
          parlay::parallel_for(0, nprobes, [&](size_t i) {
            node_t *leaf = probe_list[i].second;
            std::get<TQ4_Set>(leaf->quantized_data)
                .distances_all(std::get<TQ4_Q>(q_query_var), &visited[offsets[i]]);
            const size_t leaf_k = leaf->get_size();
            parlay::parallel_for(
                0, leaf_k, [&](size_t j) { visited[offsets[i] + j].first = leaf->data.get_id(j); });
          });
          break;
        case QT::TurboQuantPQ4Bit:
          if (params.pq.block_size == 4) {
            parlay::parallel_for(0, nprobes, [&](size_t i) {
              node_t *leaf = probe_list[i].second;
              std::get<TQPQ4_Set>(leaf->quantized_data)
                  .distances_all(std::get<TQPQ4_Q>(q_query_var), &visited[offsets[i]]);
              const size_t leaf_k = leaf->get_size();
              parlay::parallel_for(0, leaf_k, [&](size_t j) {
                visited[offsets[i] + j].first = leaf->data.get_id(j);
              });
            });
          } else {
            parlay::parallel_for(0, nprobes, [&](size_t i) {
              node_t *leaf = probe_list[i].second;
              std::get<TQPQ8_Set>(leaf->quantized_data)
                  .distances_all(std::get<TQPQ8_Q>(q_query_var), &visited[offsets[i]]);
              const size_t leaf_k = leaf->get_size();
              parlay::parallel_for(0, leaf_k, [&](size_t j) {
                visited[offsets[i] + j].first = leaf->data.get_id(j);
              });
            });
          }
          break;
        case QT::None:
        default: break;
      }
      t_distances = t.stop();
      t.reset();
    }

    t.start();
    // First, sort by (id, distance) so that duplicates (caused by spill at the root)
    // are adjacent and the smallest-distance copy comes first.
    parlay::sort_inplace(visited, [](const auto &a, const auto &b) {
      if (a.first != b.first) return a.first < b.first;
      return a.second < b.second;
    });
    // Deduplicate in-place by id, keeping the closest occurrence.
    size_t write = 0;
    for (size_t read = 0; read < visited.size(); ++read) {
      if (read == 0 || visited[read].first != visited[read - 1].first) {
        visited[write++] = visited[read];
      }
    }
    visited.resize(write);
    // Now sort by distance for downstream ranking.
    parlay::sort_inplace(visited, [](const auto &a, const auto &b) { return a.second < b.second; });
    t_rest += t.stop();
    t.reset();

    timings.push_back(dist_cmps);
    dist_cmps += total_size;
    for (double time : gs.timings) {
      timings.push_back(time);
    }
    timings.push_back(t_quantize);
    timings.push_back(t_distances);
    timings.push_back(t_rest);

    // -------------------------
    // Step 3: Re-ranking
    // -------------------------
    t.start();
    auto final_results =
        parlay::sequence<std::pair<uint32_t, float>>::uninitialized(std::min(k, visited.size()));
    if (search_params.num_rerank > 0) {
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

  // Stats from a single tree traversal (number of nodes, leaves, sizes, height).
  struct TreeStats {
    size_t num_internal_nodes = 0;
    size_t num_leaves = 0;
    double avg_leaf_size = 0.0;           // average number of point clouds per leaf
    double avg_internal_node_size = 0.0;  // average number of children (centers) per internal node
    size_t total_point_clouds_internal = 0;  // sum over internal nodes of node->data.size()
    size_t height = 0;
    // Balance metrics:
    // For each internal node with at least two children, we look at the subtree
    // sizes (# of leaf point clouds under each child). If the child subtree sizes
    // are perfectly balanced, the imbalance is 0; if one child contains all points
    // and the others are empty, the imbalance is 1.
    //
    //   imbalance(node) = (max_child_subtree_size - min_child_subtree_size)
    //                     / sum_child_subtree_sizes
    //
    // We aggregate this across internal nodes.
    double avg_child_fraction_imbalance = 0.0;  // average imbalance over internal nodes
    double max_child_fraction_imbalance = 0.0;  // worst-case imbalance over internal nodes
    // Nodes with imbalance >= kBadImbalanceThreshold: (subtree_size, imbalance_ratio), sorted by
    // subtree_size.
    static constexpr double kBadImbalanceThreshold = 0.8;
    std::vector<std::pair<size_t, double>> bad_imbalance_entries;
  };

  // Fills TreeStats by traversing the k-means tree. Call after build() or load().
  TreeStats get_tree_stats() const {
    TreeStats s;
    if (root == nullptr) return s;

    size_t leaf_size_sum = 0;
    size_t internal_size_sum = 0;
    double sum_child_frac_imbalance = 0.0;
    size_t num_internal_balance_nodes = 0;

    // Returns the total number of leaf point clouds in the subtree rooted at `node`.
    std::function<size_t(const node_t *, size_t)> visit = [&](const node_t *node,
                                                              size_t depth) -> size_t {
      if (node->children.empty()) {
        s.num_leaves++;
        size_t leaf_size = node->get_size();
        leaf_size_sum += leaf_size;
        if (depth + 1 > s.height) s.height = depth + 1;  // number of levels (matches get_height())
        return leaf_size;
      }

      s.num_internal_nodes++;
      size_t n = node->get_size();
      s.total_point_clouds_internal += n;
      internal_size_sum += n;

      std::vector<size_t> child_subtree_sizes;
      child_subtree_sizes.reserve(node->children.size());
      size_t subtree_total = 0;
      for (const node_t *child : node->children) {
        size_t child_size = visit(child, depth + 1);
        child_subtree_sizes.push_back(child_size);
        subtree_total += child_size;
      }

      // Compute per-node imbalance based on child subtree sizes.
      if (child_subtree_sizes.size() >= 2 && subtree_total > 0) {
        auto [min_it, max_it] =
            std::minmax_element(child_subtree_sizes.begin(), child_subtree_sizes.end());
        size_t min_sz = *min_it;
        size_t max_sz = *max_it;
        double frac_imbalance =
            static_cast<double>(max_sz - min_sz) / static_cast<double>(subtree_total);
        sum_child_frac_imbalance += frac_imbalance;
        if (frac_imbalance > s.max_child_fraction_imbalance) {
          s.max_child_fraction_imbalance = frac_imbalance;
        }
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

  // Returns a flat clustering of the database points based on the current tree leaves.
  //
  // The output is a sequence `leaf_of_point` such that:
  //   - `leaf_of_point[id]` is the (0-based) leaf index containing point with global id `id`.
  //   - Leaf indices are assigned in depth-first order over the tree.
  //
  // Assumes that point ids are 0..N-1 and match the ids stored in the leaf PointCloudSets.
  parlay::sequence<uint32_t> get_flat_clustering() const {
    parlay::sequence<uint32_t> empty;
    if (root == nullptr) return empty;

    std::vector<std::pair<uint32_t, uint32_t>> assignments;
    assignments.reserve(1024);
    size_t max_id = 0;
    size_t leaf_idx = 0;

    std::function<void(const node_t *)> visit = [&](const node_t *node) {
      if (node->children.empty()) {
        size_t n = node->data.size();
        for (size_t j = 0; j < n; ++j) {
          uint32_t id = node->data.get_id(j);
          assignments.emplace_back(id, static_cast<uint32_t>(leaf_idx));
          if (id > max_id) max_id = id;
        }
        ++leaf_idx;
      } else {
        for (const node_t *child : node->children) {
          if (child != nullptr) visit(child);
        }
      }
    };

    visit(root);
    if (assignments.empty()) return empty;

    parlay::sequence<uint32_t> leaf_of_point(max_id + 1);
    parlay::parallel_for(0, leaf_of_point.size(), [&](size_t i) { leaf_of_point[i] = UINT32_MAX; });

    parlay::parallel_for(0, assignments.size(), [&](size_t i) {
      auto [pid, lid] = assignments[i];
      leaf_of_point[pid] = lid;
    });

    return leaf_of_point;
  }

  // Traversing the k-means tree: returns the height of the tree
  size_t traverse_tree(node_t *node, parlay::sequence<node_t *> &ind_to_node,
                       std::unordered_map<node_t *, size_t> &node_to_ind,
                       parlay::sequence<size_t> &center_offsets,
                       parlay::sequence<size_t> &children_offsets,
                       parlay::sequence<size_t> &point_offsets, size_t height) {
    node_to_ind[node] = ind_to_node.size();
    ind_to_node.push_back(node);
    if (node->children.size() == 0) {              // leaves
      point_offsets.push_back(node->data.size());  // Always add point IDs
    } else {                                       // Internal nodes
      point_offsets.push_back(0);
      size_t dims = node->data.get_dims();
      for (size_t i = 0; i < node->children.size(); i++) {
        center_offsets.push_back(node->data.get_size(i) * dims);  // # embeddings in center[i]
      }
    }
    children_offsets.push_back(node->children.size());
    size_t h = height + 1;
    for (node_t *child : node->children) {
      h = std::max(h, traverse_tree(child, ind_to_node, node_to_ind, center_offsets,
                                    children_offsets, point_offsets, height + 1));
    }
    return h;
  }

  // Write the index to a file in disk
  void save(const std::string &filename) override {
    std::ofstream outfile(filename, std::ios::binary);
    std::cout << "Saving index to " << filename << std::endl;
    if (!outfile.is_open()) {
      std::cerr << "Error opening file for writing: " << filename << std::endl;
      return;
    }

    // Collect data
    parlay::sequence<node_t *> ind_to_node;
    std::unordered_map<node_t *, size_t> node_to_ind;
    parlay::sequence<size_t> center_offsets;
    parlay::sequence<size_t> children_offsets;
    parlay::sequence<size_t> point_offsets;

    size_t height = traverse_tree(root, ind_to_node, node_to_ind, center_offsets, children_offsets,
                                  point_offsets, 0);
    kmeanstree_height = height;
    std::cout << "Height of tree: " << height << std::endl;

    size_t total_center_sizes = parlay::scan_inplace(center_offsets);
    center_offsets.push_back(total_center_sizes);
    size_t total_children_sizes = parlay::scan_inplace(children_offsets);
    children_offsets.push_back(total_children_sizes);
    size_t total_point_sizes = parlay::scan_inplace(point_offsets);
    point_offsets.push_back(total_point_sizes);

    // Write num
    size_t num = ind_to_node.size();
    outfile.write(reinterpret_cast<const char *>(&num), sizeof(size_t));
    // Write center offsets
    size_t num_center_offsets = center_offsets.size();
    outfile.write(reinterpret_cast<const char *>(&num_center_offsets), sizeof(size_t));
    outfile.write(reinterpret_cast<const char *>(center_offsets.begin()),
                  center_offsets.size() * sizeof(size_t));
    // Write centers values
    for (size_t i = 0; i < num; ++i) {  // TODO: make parallel
      node_t *node = ind_to_node[i];
      if (node->children.size() > 0) {  // Internal Nodes only
        auto coords = node->data.data();
        size_t num_entries = (node->data.total_size()) * (node->data.get_dims());
        outfile.write(reinterpret_cast<const char *>(coords), num_entries * sizeof(float));
      }
    }
    // Write children offsets
    outfile.write(reinterpret_cast<const char *>(children_offsets.begin()),
                  children_offsets.size() * sizeof(size_t));
    // Write children values
    for (size_t i = 0; i < num; ++i) {
      node_t *node = ind_to_node[i];
      parlay::sequence<node_t *> children = node->children;
      for (size_t j = 0; j < children.size(); ++j) {  // TODO: make parallel
        node_t *child = children[j];
        size_t child_id = node_to_ind[child];
        outfile.write(reinterpret_cast<const char *>(&child_id), sizeof(size_t));
      }
    }
    // Write point offsets
    outfile.write(reinterpret_cast<const char *>(point_offsets.begin()),
                  point_offsets.size() * sizeof(size_t));
    // Write point values
    for (size_t i = 0; i < num; ++i) {
      node_t *node = ind_to_node[i];
      if (node->children.size() == 0) {  // <-- UPDATED: leaves only
        PointCloudSet<ChPoint> points = node->data;
        for (size_t j = 0; j < points.size(); ++j) {  // TODO: make parallel
          uint32_t point_id = points.get_id(j);
          outfile.write(reinterpret_cast<const char *>(&point_id), sizeof(uint32_t));
        }
      }
    }

    // Quantization: save only the MODEL (leaf encodings are reconstructed on load)
    int type_id = static_cast<int>(active_quantizer);
    outfile.write(reinterpret_cast<const char *>(&type_id), sizeof(int));

    switch (active_quantizer) {
      case QT::PQ: std::get<PQ_Model>(quantizer).save(outfile); break;
      case QT::FastScan: std::get<FS_Model>(quantizer).save(outfile); break;
      case QT::RaBitQ: std::get<RQ_Model>(quantizer).save(outfile); break;
      case QT::TurboQuant4Bit: std::get<TQ4_Model>(quantizer).save(outfile); break;
      case QT::TurboQuantPQ4Bit: {
        // Persist the actual TQPQ block size used by the model (4 or 8).
        int tqpq_block_size = 0;
        if (std::holds_alternative<TQPQ4_Model>(quantizer)) {
          tqpq_block_size = 4;
        } else if (std::holds_alternative<TQPQ8_Model>(quantizer)) {
          tqpq_block_size = 8;
        } else {
          std::cerr << "IndexMVIVF::save: TurboQuantPQ4Bit active, but quantizer variant is "
                       "neither TQPQ4_Model nor TQPQ8_Model.\n";
          abort();
        }
        outfile.write(reinterpret_cast<const char *>(&tqpq_block_size), sizeof(int));
        if (tqpq_block_size == 4) {
          std::get<TQPQ4_Model>(quantizer).save(outfile);
        } else {
          std::get<TQPQ8_Model>(quantizer).save(outfile);
        }
        break;
      }
      case QT::None:
      default:
        // nothing
        break;
    }

    outfile.close();
  }

  // Read the index from a file in disk
  void load(const std::string &filename, const PointCloudSet<ChPoint> &points) override {
    std::ifstream infile(filename, std::ios::binary);
    std::cout << "Loading index from " << filename << std::endl;
    if (!infile.is_open()) {
      std::cerr << "Error opening file for reading: " << filename << std::endl;
      return;
    }

    // Read number of nodes
    size_t num = 0;
    infile.read(reinterpret_cast<char *>(&num), sizeof(size_t));
    // Read center offsets
    size_t num_center_offsets = 0;
    infile.read(reinterpret_cast<char *>(&num_center_offsets), sizeof(size_t));
    parlay::sequence<size_t> center_offsets(num_center_offsets);
    infile.read(reinterpret_cast<char *>(center_offsets.begin()),
                center_offsets.size() * sizeof(size_t));
    // Read centers values
    parlay::sequence<float> center_values(center_offsets[center_offsets.size() - 1]);
    infile.read(reinterpret_cast<char *>(center_values.begin()),
                center_values.size() * sizeof(float));
    // Read children offsets
    parlay::sequence<size_t> children_offsets(num + 1);
    infile.read(reinterpret_cast<char *>(children_offsets.begin()),
                children_offsets.size() * sizeof(size_t));
    // Read children values
    parlay::sequence<size_t> children_values(children_offsets[children_offsets.size() - 1]);
    infile.read(reinterpret_cast<char *>(children_values.begin()),
                children_values.size() * sizeof(size_t));
    // Read point offsets
    parlay::sequence<size_t> point_offsets(num + 1);
    infile.read(reinterpret_cast<char *>(point_offsets.begin()),
                point_offsets.size() * sizeof(size_t));
    // Read point values
    parlay::sequence<uint32_t> point_values(point_offsets[point_offsets.size() - 1]);
    infile.read(reinterpret_cast<char *>(point_values.begin()),
                point_values.size() * sizeof(uint32_t));

    // Quantization
    int type_id = 0;
    infile.read(reinterpret_cast<char *>(&type_id), sizeof(int));
    active_quantizer = static_cast<QT>(type_id);

    switch (active_quantizer) {
      case QT::PQ:
        quantizer.template emplace<PQ_Model>();
        std::get<PQ_Model>(quantizer).load(infile);
        break;
      case QT::FastScan:
        quantizer.template emplace<FS_Model>();
        std::get<FS_Model>(quantizer).load(infile);
        break;
      case QT::RaBitQ:
        quantizer.template emplace<RQ_Model>();
        std::get<RQ_Model>(quantizer).load(infile);
        break;
      case QT::TurboQuant4Bit:
        quantizer.template emplace<TQ4_Model>();
        std::get<TQ4_Model>(quantizer).load(infile);
        break;
      case QT::TurboQuantPQ4Bit: {
        // Read and restore the TQPQ block size used when the model was trained.
        int tqpq_block_size = 0;
        infile.read(reinterpret_cast<char *>(&tqpq_block_size), sizeof(int));
        params.pq.block_size = tqpq_block_size;
        if (tqpq_block_size == 4) {
          quantizer.template emplace<TQPQ4_Model>();
          std::get<TQPQ4_Model>(quantizer).load(infile);
        } else if (tqpq_block_size == 8) {
          quantizer.template emplace<TQPQ8_Model>();
          std::get<TQPQ8_Model>(quantizer).load(infile);
        } else {
          std::cerr << "IndexMVIVF::load: TurboQuantPQ4Bit model with unsupported block_size="
                    << tqpq_block_size << " (expected 4 or 8)." << std::endl;
          abort();
        }
        break;
      }
      case QT::None:
      default: quantizer = std::monostate{}; break;
    }
    infile.close();

    // Center quantization (internal-node / leaf-center scoring): always TQ4.
    init_center_quantizer(points);

    // Build the index
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
    parlay::sequence<node_t *> ind_to_node =
        parlay::sequence<node_t *>::from_function(num, [&](size_t i) {
          node_t *node = new node_t();
          if (children_sizes[i] > 0) {  // Internal Nodes
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
                  uint32_t point_id = point_values[point_offsets[i] + j];
                  return point_id_to_data_id[point_id];
                });
            node->data = PointCloudSet<ChPoint>(points.filter(point_group), dim);
          }
          return node;
        });

    // Set children pointers
    parlay::parallel_for(0, num, [&](size_t i) {
      node_t *node = ind_to_node[i];
      parlay::sequence<node_t *> &children = node->children;
      parlay::parallel_for(0, children.size(), [&](size_t j) {
        size_t child_id = children_values[children_offsets[i] + j];
        children[j] = ind_to_node[child_id];
      });
    });
    root = ind_to_node[0];

    // Rebuild flat leaf index and leaf centers
    leaves_flat.clear();
    leaf_centers_quant = std::monostate{};
    if (root) {
      std::vector<ChPoint> center_points;
      std::function<void(node_t *, node_t *, size_t)> visit = [&](node_t *node, node_t *parent,
                                                                  size_t child_idx) {
        if (node->children.empty()) {
          leaves_flat.push_back(node);
          auto &centers_pc = parent->data;
          center_points.push_back(centers_pc[child_idx]);
        } else {
          for (size_t i = 0; i < node->children.size(); ++i) {
            visit(node->children[i], node, i);
          }
        }
      };
      visit(root, nullptr, 0);

      if (!center_points.empty()) {
        leaf_centers = PointCloudSet<ChPoint>(center_points, d);

        if (params.quantize_centers) {
          leaf_centers_quant = encode_centers_tq(leaf_centers);
        }
      }
    }

    // Re-encode nodes (since we only saved the model)
    parlay::parallel_for(
        0, num,
        [&](size_t i) {
          node_t *node = ind_to_node[i];
          if (!node) return;
          if (node->data.size() == 0) return;

          if (!node->children.empty()) {  // internal node
            if (params.quantize_centers) {
              node->quantized_data = encode_centers_tq(node->data);
            }
            return;
          }

          // leaf node
          if (active_quantizer != QT::None) {
            node->quantized_data = encode_points_quantized(node->data);
          }
        },
        /*granularity=*/1);
  }

  // Traversing the tree and deleting nodes
  void traverse_and_delete(node_t *node) {
    for (size_t i = 0; i < node->children.size(); i++) {
      node_t *child = node->children[i];
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

using IndexMVIVFSpillL2 = IndexMVIVFSpill<true>;   // Instantiates for L2 metric (metric = true)
using IndexMVIVFSpillIP = IndexMVIVFSpill<false>;  // Instantiates for MIPS      (metric = false)

}  // namespace mvsic