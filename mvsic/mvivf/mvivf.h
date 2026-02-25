#pragma once

#include <algorithm>
#include <functional>
#include <optional>
#include <queue>
#include <set>
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
class IndexMVIVF : public Index<metric> {
 public:
  using ChPoint = typename Index<metric>::ChPoint;  // Chamfer Point Type
  using Index<metric>::d;                           // Embedding dimension

  // Multi-Vector Quantizer Types using the Wrapper
  using FlatRange = FlattenedPCRange<PointCloudSet<ChPoint>>;
  using PQ_Enc = pq::Quantized_Point_Range<FlatRange, metric>;
  using FS_Enc = fastscan::Quantized_Point_Range<FlatRange, metric>;
  using RQ_Enc = rabitq::Quantized_Point_Range<FlatRange, metric>;
  using TQ4_Enc = turboquant_4bit::Quantized_Point_Range<FlatRange, metric>;

  using PQ_Set = Quantized_Point_Cloud_Set<PQ_Enc, metric>;
  using FS_Set = Quantized_Point_Cloud_Set<FS_Enc, metric>;
  using RQ_Set = Quantized_Point_Cloud_Set<RQ_Enc, metric>;
  using TQ4_Set = Quantized_Point_Cloud_Set<TQ4_Enc, metric>;
  using QuantSet = std::variant<std::monostate, PQ_Set, FS_Set, RQ_Set, TQ4_Set>;

  using PQ_Model = MultiVecQuantizer<pq::Model<metric>, metric>;
  using FS_Model = MultiVecQuantizer<fastscan::Model<metric>, metric>;
  using RQ_Model = MultiVecQuantizer<rabitq::Model<metric>, metric>;
  using TQ4_Model = MultiVecQuantizer<turboquant_4bit::Model<metric>, metric>;
  using QuantModel = std::variant<std::monostate, PQ_Model, FS_Model, RQ_Model, TQ4_Model>;

  // helper for decltype
  template<class M, class Q>
  using QQueryT = decltype(std::declval<M &>().quantize_query(std::declval<Q const &>()));
  using PQ_Q = QQueryT<PQ_Model, ChPoint>;
  using FS_Q = QQueryT<FS_Model, ChPoint>;
  using RQ_Q = QQueryT<RQ_Model, ChPoint>;
  using TQ4_Q = QQueryT<TQ4_Model, ChPoint>;
  using QuantQuery = std::variant<std::monostate, PQ_Q, FS_Q, RQ_Q, TQ4_Q>;

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

  std::vector<node_t *> leaves_flat;    // Flat list of leaves for large-nprobes search
  PointCloudSet<ChPoint> leaf_centers;  // One representative center per leaf
  QuantSet leaf_centers_quant;          // Quantized centers for flat search (when enabled)

  // Quantizer Storage
  QuantModel quantizer = std::monostate{};
  QT active_quantizer = QT::None;

  // Stats
  size_t kmeanstree_height = 0;

  IndexMVIVF(size_t d_) noexcept : params(IndexParams::mvivf()) { d = d_; }
  IndexMVIVF(size_t d_, const IndexParams &params) noexcept : params(params) { d = d_; }

  // Recursive kmeans tree builder
  void recursive_build(node_t *node, const PointCloudSet<ChPoint> &points) {
    size_t n = points.size();
    size_t num_clusters = (params.k_per_level > 0) ? params.k_per_level
                                                   : static_cast<size_t>(std::ceil(std::sqrt(n)));
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
    // Quantize centers
    if (params.quantize_centers) {
      switch (active_quantizer) {
        case QT::PQ: {
          auto &m = std::get<PQ_Model>(quantizer);
          node->quantized_data = m.encode(node->data);
          break;
        }
        case QT::FastScan: {
          auto &m = std::get<FS_Model>(quantizer);
          node->quantized_data = m.encode(node->data);
          break;
        }
        case QT::RaBitQ: {
          auto &m = std::get<RQ_Model>(quantizer);
          node->quantized_data = m.encode(node->data);
          break;
        }
        case QT::TurboQuant4Bit: {
          auto &m = std::get<TQ4_Model>(quantizer);
          node->quantized_data = m.encode(node->data);
          break;
        }
        case QT::None:
        default: node->quantized_data = std::monostate{}; break;
      }
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
            switch (active_quantizer) {
              case QT::PQ: {
                auto &m = std::get<PQ_Model>(quantizer);
                child->quantized_data = m.encode(child->data);
                break;
              }
              case QT::FastScan: {
                auto &m = std::get<FS_Model>(quantizer);
                child->quantized_data = m.encode(child->data);
                break;
              }
              case QT::RaBitQ: {
                auto &m = std::get<RQ_Model>(quantizer);
                child->quantized_data = m.encode(child->data);
                break;
              }
              case QT::TurboQuant4Bit: {
                auto &m = std::get<TQ4_Model>(quantizer);
                child->quantized_data = m.encode(child->data);
                break;
              }
              case QT::None:
              default: child->quantized_data = std::monostate{}; break;
            }
          }
        },
        1);
  }

  // Builds the index given PointCloudSet object.
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
      default: quantizer = std::monostate{}; break;
    }

    // Recursively build k-means tree
    recursive_build(root, points);

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

        if (params.quantize_centers && active_quantizer != QT::None) {
          switch (active_quantizer) {
            case QT::PQ: {
              auto &m = std::get<PQ_Model>(quantizer);
              leaf_centers_quant = m.encode(leaf_centers);
              break;
            }
            case QT::FastScan: {
              auto &m = std::get<FS_Model>(quantizer);
              leaf_centers_quant = m.encode(leaf_centers);
              break;
            }
            case QT::RaBitQ: {
              auto &m = std::get<RQ_Model>(quantizer);
              leaf_centers_quant = m.encode(leaf_centers);
              break;
            }
            case QT::TurboQuant4Bit: {
              auto &m = std::get<TQ4_Model>(quantizer);
              leaf_centers_quant = m.encode(leaf_centers);
              break;
            }
            case QT::None:
            default: leaf_centers_quant = std::monostate{}; break;
          }
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
                                   size_t nprobes) const {
    using score_node = std::pair<float, node_t *>;
    auto less = [](const score_node &a, const score_node &b) {
      return a.first < b.first || (a.first == b.first && a.second < b.second);
    };
    // Modifying this doesn't affect performance significantly, as beam management is not the
    // bottleneck.
    const size_t beam_length = 2 * nprobes;
    parlay::internal::timer t;
    double num_chamfer_cmps = 0.0;
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
      num_chamfer_cmps += children.size();
      if (!params.quantize_centers) {
        auto &centers = current_node->data;
        dist_cmps += centers.distances(query, child_dists.data());
      } else {
        switch (active_quantizer) {
          case QT::RaBitQ: {
            auto &q_query = std::get<RQ_Q>(q_query_var);
            auto &qleaf = std::get<RQ_Set>(current_node->quantized_data);
            qleaf.distances_all(q_query, child_dists.data());
            break;
          }
          case QT::TurboQuant4Bit: {
            auto &q_query = std::get<TQ4_Q>(q_query_var);
            auto &qleaf = std::get<TQ4_Set>(current_node->quantized_data);
            qleaf.distances_all(q_query, child_dists.data());
            break;
          }
          case QT::PQ: {
            auto &q_query = std::get<PQ_Q>(q_query_var);
            auto &qleaf = std::get<PQ_Set>(current_node->quantized_data);
            qleaf.distances_all(q_query, child_dists.data());
            break;
          }
          case QT::FastScan: {
            auto &q_query = std::get<FS_Q>(q_query_var);
            auto &qleaf = std::get<FS_Set>(current_node->quantized_data);
            qleaf.distances_all(q_query, child_dists.data());
            break;
          }
          case QT::None:
          default:
            std::cerr << "Error: Invalid quantizer type in greedy search." << std::endl;
            abort();
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
    out.timings = {num_chamfer_cmps, t_dists, t_beam, t_rest};
    out.probe_list = std::move(top_probes);
    return out;
  }

  // Flat leaf scoring: score all leaves by distance to a precomputed representative center and
  // pick the best nprobes leaves.
  GreedySearchResult flat_leaf_search(const ChPoint &query, const QuantQuery &q_query_var,
                                      size_t nprobes) const {
    using score_node = std::pair<float, node_t *>;
    GreedySearchResult out;
    if (leaves_flat.empty() || leaf_centers.size() == 0 || nprobes == 0) {
      return out;
    }

    const size_t L = leaf_centers.size();
    const size_t use_nprobes = std::min(nprobes, L);

    parlay::internal::timer t;
    double num_chamfer_cmps = 0.0;
    double t_dists = 0.0;
    double t_rest = 0.0;

    t.start();
    auto centers_dists = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(L);
    auto scores = parlay::sequence<score_node>::uninitialized(L);

    if (!params.quantize_centers || active_quantizer == QT::None) {
      // Exact Chamfer distance to the representative centers.
      leaf_centers.distances(query, centers_dists.data());
      num_chamfer_cmps = static_cast<double>(L);
    } else {
      switch (active_quantizer) {
        case QT::RaBitQ: {
          auto &q_query = std::get<RQ_Q>(q_query_var);
          auto &qcent = std::get<RQ_Set>(leaf_centers_quant);
          qcent.distances_all(q_query, centers_dists.data());
          break;
        }
        case QT::TurboQuant4Bit: {
          auto &q_query = std::get<TQ4_Q>(q_query_var);
          auto &qcent = std::get<TQ4_Set>(leaf_centers_quant);
          qcent.distances_all(q_query, centers_dists.data());
          break;
        }
        case QT::PQ: {
          auto &q_query = std::get<PQ_Q>(q_query_var);
          auto &qcent = std::get<PQ_Set>(leaf_centers_quant);
          qcent.distances_all(q_query, centers_dists.data());
          break;
        }
        case QT::FastScan: {
          auto &q_query = std::get<FS_Q>(q_query_var);
          auto &qcent = std::get<FS_Set>(leaf_centers_quant);
          qcent.distances_all(q_query, centers_dists.data());
          break;
        }
        case QT::None:
        default:
          std::cerr << "Error: Invalid quantizer type in flat leaf search." << std::endl;
          abort();
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
    out.dist_cmps = 0;  // we only counted center distances as num_chamfer_cmps
    out.timings = {num_chamfer_cmps, t_dists, /*t_beam=*/0.0, t_rest};
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
    t.start();
    switch (active_quantizer) {
      case QT::RaBitQ: {
        auto &m = std::get<RQ_Model>(quantizer);
        q_query_var = m.quantize_query(query);
        break;
      }
      case QT::TurboQuant4Bit: {
        auto &m = std::get<TQ4_Model>(quantizer);
        q_query_var = m.quantize_query(query);
        break;
      }
      case QT::PQ: {
        auto &m = std::get<PQ_Model>(quantizer);
        q_query_var = m.quantize_query(query);
        break;
      }
      case QT::FastScan: {
        auto &m = std::get<FS_Model>(quantizer);
        q_query_var = m.quantize_query(query);
        break;
      }
      case QT::None: {
        q_query_var = std::monostate{};
        break;
      }
      default: abort();
    }
    t_quantize = t.stop();
    t.reset();

    // -------------------------
    // Step 1: Greedy search
    // -------------------------
    GreedySearchResult gs;
    const size_t num_leaves = leaves_flat.size();
    const double alpha = 0.5;  // heuristic threshold
    bool use_flat = (num_leaves > 0 && nprobes >= static_cast<size_t>(alpha * num_leaves));
    if (use_flat) {
      gs = flat_leaf_search(query, q_query_var, nprobes);
    } else {
      gs = greedy_search(query, q_query_var, nprobes);
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

    switch (active_quantizer) {
      case QT::RaBitQ: {
        t.start();
        auto &q_query = std::get<RQ_Q>(q_query_var);
        parlay::parallel_for(0, nprobes, [&](size_t i) {
          node_t *leaf = probe_list[i].second;
          auto &qleaf = std::get<RQ_Set>(leaf->quantized_data);
          qleaf.distances_all(q_query, &visited[offsets[i]]);
          parlay::parallel_for(0, leaf->data.size(), [&](size_t j) {
            visited[offsets[i] + j].first = leaf->data.get_id(j);
          });
        });
        t_distances = t.stop();
        t.reset();
        break;
      }
      case QT::TurboQuant4Bit: {
        t.start();
        auto &q_query = std::get<TQ4_Q>(q_query_var);
        parlay::parallel_for(0, nprobes, [&](size_t i) {
          node_t *leaf = probe_list[i].second;
          auto &qleaf = std::get<TQ4_Set>(leaf->quantized_data);
          qleaf.distances_all(q_query, &visited[offsets[i]]);
          parlay::parallel_for(0, leaf->data.size(), [&](size_t j) {
            visited[offsets[i] + j].first = leaf->data.get_id(j);
          });
        });
        t_distances = t.stop();
        t.reset();
        break;
      }
      case QT::PQ: {
        t.start();
        auto &q_query = std::get<PQ_Q>(q_query_var);
        parlay::parallel_for(0, nprobes, [&](size_t i) {
          node_t *leaf = probe_list[i].second;
          auto &qleaf = std::get<PQ_Set>(leaf->quantized_data);
          qleaf.distances_all(q_query, &visited[offsets[i]]);
          parlay::parallel_for(0, leaf->data.size(), [&](size_t j) {
            visited[offsets[i] + j].first = leaf->data.get_id(j);
          });
        });
        t_distances = t.stop();
        t.reset();
        break;
      }
      case QT::FastScan: {
        t.start();
        auto &q_query = std::get<FS_Q>(q_query_var);
        parlay::parallel_for(0, nprobes, [&](size_t i) {
          node_t *leaf = probe_list[i].second;
          auto &qleaf = std::get<FS_Set>(leaf->quantized_data);
          qleaf.distances_all(q_query, &visited[offsets[i]]);
          parlay::parallel_for(0, leaf->data.size(), [&](size_t j) {
            visited[offsets[i] + j].first = leaf->data.get_id(j);
          });
        });
        t_distances = t.stop();
        t.reset();
        break;
      }
      case QT::None: {
        t_quantize = 0.0;
        t.start();
        auto leaf_dist_cmps = parlay::sequence<size_t>::uninitialized(nprobes);
        parlay::parallel_for(0, nprobes, [&](size_t i) {
          node_t *leaf_node = probe_list[i].second;
          leaf_dist_cmps[i] = leaf_node->data.distances(query, &visited[offsets[i]]);
        });
        dist_cmps += parlay::reduce(leaf_dist_cmps);
        t_distances = t.stop();
        t.reset();
        break;
      }
    }

    t.start();
    parlay::sort_inplace(visited, [](const auto &a, const auto &b) { return a.second < b.second; });
    t_rest += t.stop();
    t.reset();

    timings.push_back(total_size);
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
      dist_cmps += this->rerank(query, points, visited, num_rerank, final_results);
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
  };

  // Fills TreeStats by traversing the k-means tree. Call after build() or load().
  TreeStats get_tree_stats() const {
    TreeStats s;
    if (root == nullptr) return s;

    size_t leaf_size_sum = 0;
    size_t internal_size_sum = 0;

    std::function<void(const node_t *, size_t)> visit = [&](const node_t *node, size_t depth) {
      if (node->children.empty()) {
        s.num_leaves++;
        leaf_size_sum += node->get_size();
        if (depth + 1 > s.height) s.height = depth + 1;  // number of levels (matches get_height())
      } else {
        s.num_internal_nodes++;
        size_t n = node->get_size();
        s.total_point_clouds_internal += n;
        internal_size_sum += n;
        for (const node_t *child : node->children)
          visit(child, depth + 1);
      }
    };

    visit(root, 0);
    if (s.num_leaves > 0) s.avg_leaf_size = static_cast<double>(leaf_size_sum) / s.num_leaves;
    if (s.num_internal_nodes > 0) {
      s.avg_internal_node_size = static_cast<double>(internal_size_sum) / s.num_internal_nodes;
    }
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
      case QT::None:
      default: quantizer = std::monostate{}; break;
    }
    infile.close();

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

        if (params.quantize_centers && active_quantizer != QT::None) {
          switch (active_quantizer) {
            case QT::PQ: {
              auto &m = std::get<PQ_Model>(quantizer);
              leaf_centers_quant = m.encode(leaf_centers);
              break;
            }
            case QT::FastScan: {
              auto &m = std::get<FS_Model>(quantizer);
              leaf_centers_quant = m.encode(leaf_centers);
              break;
            }
            case QT::RaBitQ: {
              auto &m = std::get<RQ_Model>(quantizer);
              leaf_centers_quant = m.encode(leaf_centers);
              break;
            }
            case QT::TurboQuant4Bit: {
              auto &m = std::get<TQ4_Model>(quantizer);
              leaf_centers_quant = m.encode(leaf_centers);
              break;
            }
            case QT::None:
            default: leaf_centers_quant = std::monostate{}; break;
          }
        }
      }
    }

    // Re-encode nodes (since we only saved the model)
    if (active_quantizer != QT::None) {
      parlay::parallel_for(
          0, num,
          [&](size_t i) {
            node_t *node = ind_to_node[i];
            if (!node) return;
            if (!params.quantize_centers && !node->children.empty()) return;  // internal node
            if (node->data.size() == 0) return;

            switch (active_quantizer) {
              case QT::PQ: {
                auto &m = std::get<PQ_Model>(quantizer);
                node->quantized_data = m.encode(node->data);
                break;
              }
              case QT::FastScan: {
                auto &m = std::get<FS_Model>(quantizer);
                node->quantized_data = m.encode(node->data);
                break;
              }
              case QT::RaBitQ: {
                auto &m = std::get<RQ_Model>(quantizer);
                node->quantized_data = m.encode(node->data);
                break;
              }
              case QT::TurboQuant4Bit: {
                auto &m = std::get<TQ4_Model>(quantizer);
                node->quantized_data = m.encode(node->data);
                break;
              }
              case QT::None:
              default: node->quantized_data = std::monostate{}; break;
            }
          },
          /*granularity=*/1);
    }
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

  ~IndexMVIVF() {
    if (root != nullptr) {
      traverse_and_delete(root);
      delete root;
    }
  }
};

using IndexMVIVFL2 = IndexMVIVF<true>;   // Instantiates for L2 metric (metric = true)
using IndexMVIVFIP = IndexMVIVF<false>;  // Instantiates for MIPS      (metric = false)

}  // namespace mvsic