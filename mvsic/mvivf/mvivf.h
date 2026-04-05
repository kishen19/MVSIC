#pragma once

#include <algorithm>
#include <functional>
#include <set>
#include <type_traits>
#include <unordered_set>
#include <variant>
#include <vector>

#include "mvsic/core/index.h"
#include "mvsic/core/mvclustering/mvclustering.h"
#include "mvsic/core/utils/util.h"

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
  using MVQT = typename Index<metric>::MVQT;
  using QuantSet = typename MVQT::QuantSet;
  using QuantQuery = typename MVQT::QuantQuery;
  using QuantModel = typename MVQT::QuantModel;
  using TQ_Set = typename MVQT::TQ_Set;
  using TQ_Query = typename MVQT::TQ_Query;
  using TQ_Model = typename MVQT::TQ_Model;
  using QT = typename Index<metric>::QT;
  using Index<metric>::d;       // Embedding dimension
  using Index<metric>::params;  // Index Params
  using Index<metric>::quantization_mode;

  // kmeans tree nodes
  struct node_t {
    parlay::sequence<node_t*> children;
    // For internal nodes: data = centers of children
    // For leaves:         data = points in the cluster
    PointCloudSet<ChPoint> data;
    QuantSet quantized_data;  // std::monostate for inner nodes and unquantized leaves.
    node_t() noexcept : children(), data(), quantized_data(std::monostate{}) {}
    ~node_t() noexcept {}
    inline size_t get_size() const noexcept { return data.size(); }
  };

  node_t* root = nullptr;  // Root of the k-means tree

  // Leaf Data: Accessed directly during search when nprobes is large
  parlay::sequence<node_t*> leaves_flat;  // Flat list of leaves for large-nprobes search
  PointCloudSet<ChPoint> leaf_centers;    // One representative center per leaf
  QuantSet leaf_centers_quant;            // Quantized centers for flat search (when enabled)
  // Center quantization: when params.quantize_centers is enabled, we *always* use TurboQuant
  // (TQ) for internal-node / leaf-center scoring, regardless of the leaf quantizer.
  // This is because greedy search/flat search requires high-quality scores
  TQ_Model center_quantizer;
  QuantModel quantizer = std::monostate{};

  // Stats
  size_t kmeanstree_height = 0;
  // For each leaf index, the root-level child index (for diversity stats).
  std::vector<uint32_t> leaf_to_root_child_;

  IndexMVIVF(size_t d_) noexcept {
    d = d_;
    params = IndexParams::mvivf();
  }
  IndexMVIVF(size_t d_, const IndexParams& params_) noexcept {
    d = d_;
    params = params_;
  }

  // Quantization Helpers
  void init_tq_quantizer(const PointCloudSet<ChPoint>& points) {
    if (!params.quantize_centers) return;
    center_quantizer.train(points);
  }

  TQ_Set encode_tq(const PointCloudSet<ChPoint>& points) {
    return center_quantizer.encode(points);  // returns TQ_Set
  }

  // Recursive kmeans tree builder
  void recursive_build(node_t* node, const PointCloudSet<ChPoint>& points) {
    size_t n = points.size();
    // Number of centers: Dynamic
    // main_val: either k_per_level or sqrt(n)
    // small_nc: If n is small, choosing ~n/max_leaf_size is better
    size_t auto_nc = (params.k_per_level > 0) ? params.k_per_level
                                              : static_cast<size_t>(std::ceil(std::sqrt(n)));
    size_t small_nc = 4 * (n + params.max_leaf_size - 1) / params.max_leaf_size;
    size_t num_clusters = std::max(static_cast<size_t>(4), std::min({auto_nc, small_nc, n}));
    if (params.verbose >= 1) {
      std::cout << "[MVIVF] Building index with " << n << " points, num_clusters: " << num_clusters
                << std::endl;
    }
    // Step 1: Run MV-Lloyds on points
    MVClustering<metric> Clus(d, num_clusters, params.s, params.mvclus);
    Clus.train(points);
    PointCloudSet<ChPoint>& centers = Clus.get_centers();
    parlay::sequence<uint32_t> cluster_ids = Clus.get_clustering(points);
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
    // Quantize internal-node centers if enabled.
    if (params.quantize_centers) {
      node->quantized_data = encode_tq(node->data);
    }
    parlay::parallel_for(
        0, grouped.size(),
        [&](size_t i) {
          auto cluster_id = grouped[i][0].first;
          auto group = parlay::delayed_seq<uint32_t>(
              grouped[i].size(), [&](size_t j) { return grouped[i][j].second; });
          PointCloudSet<ChPoint> child_points = PointCloudSet<ChPoint>(points.filter(group), d);
          node_t* child = new node_t();
          node->children[i] = child;
          if (child_points.size() > params.max_leaf_size) {  // Recurse
            recursive_build(child, child_points);
          } else {  // Leaf Node
            child->data = std::move(child_points);
            child->quantized_data = this->encode_points_quantized(child->data, quantizer);
          }
        },
        1);
  }

  // Builds the index given PointCloudSet object.
  void build(const PointCloudSet<ChPoint>& points) override {
    parlay::internal::timer t;
    root = new node_t();
    // Quantization
    t.start();
    quantization_mode = params.pq.method;
    this->train_quantizer(points, quantizer);
    // Center quantization (internal-node / leaf-center scoring): always TQ.
    init_tq_quantizer(points);
    if (params.verbose >= 1 && quantization_mode != QT::None) {
      std::cout << "[MVIVF] Quantizers Trained: " << t.stop() << " sec" << std::endl;
    }
    t.reset();
    // Recursively build k-means tree
    t.start();
    recursive_build(root, points);
    if (params.verbose >= 1) {
      std::cout << "[MVIVF] MV-Kmeans-Tree Built: " << t.stop() << " sec" << std::endl;
    }
    t.reset();

    // Build flat leaf index and leaf centers for optional flat-leaf search
    t.start();
    leaves_flat.clear();
    leaf_centers_quant = std::monostate{};
    leaf_to_root_child_.clear();
    if (root) {
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

        if (params.quantize_centers) {
          leaf_centers_quant = encode_tq(leaf_centers);
        }
      }
    }
    if (params.verbose >= 1) {
      std::cout << "[MVIVF] Leaf-data computed: " << t.stop() << " sec" << std::endl;
    }
  }

  // Output type of Greedy Search
  struct GreedySearchResult {
    parlay::sequence<std::pair<float, node_t*>> probe_list;
    size_t bytes_accessed = 0;
    std::vector<double> stats = {};
  };

  // Simple Beam Search with early exit
  GreedySearchResult greedy_search(const ChPoint& query, const TQ_Query& q_query,
                                   size_t nprobes) const {
    using score_node = std::pair<float, node_t*>;
    auto less = [](const score_node& a, const score_node& b) {
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
    size_t bytes_accessed = 0;
    std::set<score_node> beam;
    parlay::sequence<score_node> top_probes;
    top_probes.reserve(nprobes + 1);
    std::vector<std::pair<uint32_t, float>> child_dists;  // Scratch memory
    child_dists.reserve(root->children.size());           // Reserve reasonable initial capacity
    t_rest += t.stop();
    t.reset();

    // Initial
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
      node_t* current_node = best.second;
      // Check if current score is worse than the worst in top_probes
      if (top_probes.size() == nprobes && best.first >= top_probes.front().first) {
        break;  // Beam has converged
      }
      auto& children = current_node->children;
      // Compute distances to children
      child_dists.resize(children.size());
      if (params.quantize_centers) {  // Center scoring always uses TQ when enabled.
        std::get<TQ_Set>(current_node->quantized_data).distances_all(q_query, child_dists.data());
        bytes_accessed += std::get<TQ_Set>(current_node->quantized_data).num_bytes();
      } else {
        auto& centers = current_node->data;
        centers.distances(query, child_dists.data());
        bytes_accessed += centers.num_bytes();
      }
      dist_cmps += children.size();
      t_dists += t.stop();
      t.reset();

      for (size_t i = 0; i < children.size(); ++i) {
        float d = child_dists[i].second;
        node_t* child = children[i];
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
    out.bytes_accessed = bytes_accessed;
    out.stats = {static_cast<double>(dist_cmps), t_dists, t_beam, t_rest};
    out.probe_list = std::move(top_probes);
    return out;
  }

  // Flat leaf scoring: score all leaves and pick the best nprobes leaves.
  GreedySearchResult flat_leaf_search(const ChPoint& query, const TQ_Query& q_query,
                                      size_t nprobes) const {
    using score_node = std::pair<float, node_t*>;
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
    size_t bytes_accessed = 0;
    auto centers_dists = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(L);
    auto scores = parlay::sequence<score_node>::uninitialized(L);

    if (params.quantize_centers) {
      std::get<TQ_Set>(leaf_centers_quant).distances_all(q_query, centers_dists.data());
      bytes_accessed += std::get<TQ_Set>(leaf_centers_quant).num_bytes();
    } else {
      leaf_centers.distances(query, centers_dists.data());
      bytes_accessed += leaf_centers.num_bytes();
    }
    parlay::parallel_for(0, L,
                         [&](size_t i) { scores[i] = {centers_dists[i].second, leaves_flat[i]}; });
    t_dists += t.stop();
    t.reset();

    // Select top-nprobes leaves.
    t.start();
    if (use_nprobes < L) {
      std::nth_element(scores.begin(), scores.begin() + use_nprobes, scores.end(),
                       [](const score_node& a, const score_node& b) { return a.first < b.first; });
      scores.resize(use_nprobes);
    }
    t_rest += t.stop();
    t.reset();

    out.probe_list = std::move(scores);
    out.bytes_accessed = bytes_accessed;
    out.stats = {static_cast<double>(dist_cmps), t_dists, /*t_beam=*/0.0, t_rest};
    return out;
  }

  inline auto process_probes(const ChPoint& query, const QuantQuery& q_query_var,
                             parlay::sequence<std::pair<float, node_t*>>& probe_list) {
    const size_t nprobes = probe_list.size();
    auto sizes = parlay::delayed_tabulate(
        nprobes, [&](size_t i) { return probe_list[i].second->get_size(); });
    auto scan_result = parlay::scan(sizes);
    auto& offsets = scan_result.first;
    size_t& total_size = scan_result.second;
    auto visited = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(total_size);
    auto bytes_accessed = parlay::sequence<size_t>::uninitialized(nprobes);

    auto process_probes_quant = [&]<typename SetType>(const auto& q_query) {
      parlay::parallel_for(0, nprobes, [&](size_t i) {
        node_t* leaf = probe_list[i].second;
        auto* leaf_data = std::get_if<SetType>(&leaf->quantized_data);
        if (!leaf_data) UNREACHABLE();
        leaf_data->distances_all(q_query, &visited[offsets[i]]);
        bytes_accessed[i] = leaf_data->num_bytes();
      });
    };

    switch (quantization_mode) {
      case QT::PQ:
        process_probes_quant.template operator()<typename MVQT::PQ_Set>(
            std::get<typename MVQT::PQ_Query>(q_query_var));
        break;
      case QT::RaBitQ:
        process_probes_quant.template operator()<typename MVQT::RQ_Set>(
            std::get<typename MVQT::RQ_Query>(q_query_var));
        break;
      case QT::FastScan:
        process_probes_quant.template operator()<typename MVQT::FS_Set>(
            std::get<typename MVQT::FS_Query>(q_query_var));
        break;
      case QT::TurboQuant:
        process_probes_quant.template operator()<typename MVQT::TQ_Set>(
            std::get<typename MVQT::TQ_Query>(q_query_var));
        break;
      case QT::None:
        parlay::parallel_for(0, nprobes, [&](size_t i) {
          node_t* leaf_node = probe_list[i].second;
          leaf_node->data.distances(query, &visited[offsets[i]]);
          bytes_accessed[i] = leaf_node->data.num_bytes();
        });
        break;
    }
    return std::make_pair(visited, parlay::reduce(bytes_accessed));
  }

  std::tuple<parlay::sequence<std::pair<uint32_t, float>>, size_t, std::vector<double>>
  search_with_stats(const ChPoint& query, const PointCloudSet<ChPoint>& points,
                    const SearchParams& search_params) override {
    parlay::internal::timer t;

    size_t k = search_params.k;
    size_t nprobes = search_params.nprobes;
    size_t bytes_accessed = 0;

    std::vector<double> stats;
    size_t dist_cmps = 0;
    double t_quantize = 0.0;
    double t_distances = 0.0;
    double t_rest = 0.0;
    double t_rerank = 0.0;

    // -------------------------
    // Step 0: Quantize Query
    // -------------------------
    QuantQuery q_query_var = this->quantize_query_point_cloud(query, quantizer);
    TQ_Query q_center_query;
    t.start();
    if (params.quantize_centers) {
      q_center_query = center_quantizer.quantize_query(query);
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
      gs = flat_leaf_search(query, q_center_query, nprobes);
    } else {
      gs = greedy_search(query, q_center_query, nprobes);
    }
    auto& probe_list = gs.probe_list;
    bytes_accessed += gs.bytes_accessed;
    nprobes = std::min(nprobes, probe_list.size());
    stats = {gs.stats[0]};

    // -------------------------
    // Step 2: Probe clusters in probe_list
    // -------------------------

    t.start();
    parlay::sequence<std::pair<uint32_t, float>> visited;
    size_t bytes_accessed_pp;
    std::tie(visited, bytes_accessed_pp) = process_probes(query, q_query_var, probe_list);
    bytes_accessed += bytes_accessed_pp;
    dist_cmps += visited.size();
    t_distances = t.stop();
    t.reset();

    t.start();
    mvsic::sort_inplace_kv(visited);
    t_rest += t.stop();
    t.reset();
    // -------------------------
    // Step 3: Re-ranking
    // -------------------------
    t.start();
    auto final_results =
        parlay::sequence<std::pair<uint32_t, float>>::uninitialized(std::min(k, visited.size()));
    if (search_params.num_rerank > 0) {
      size_t num_rerank = std::min(search_params.num_rerank, visited.size());
      bytes_accessed += this->rerank(query, points, visited, num_rerank, final_results);
    } else {
      parlay::parallel_for(0, final_results.size(),
                           [&](size_t i) { final_results[i] = visited[i]; });
    }
    t_rerank = t.stop();
    t.reset();

    stats = {gs.stats[0]};                            // #cmps in greedy/flat search
    stats.push_back(static_cast<size_t>(dist_cmps));  // #cmps during probing
    // Stats from greedy/flat search
    for (int i = 1; i < gs.stats.size(); i++) {
      stats.push_back(gs.stats[i]);
    }
    // Stats from probing and reranking
    stats.push_back(t_quantize);
    stats.push_back(t_distances);
    stats.push_back(t_rest);
    stats.push_back(t_rerank);

    return std::make_tuple(std::move(final_results), bytes_accessed, stats);
  }

  // Traversing the k-means tree: returns the height of the tree
  size_t traverse_tree(node_t* node, parlay::sequence<node_t*>& ind_to_node,
                       std::unordered_map<node_t*, size_t>& node_to_ind,
                       parlay::sequence<size_t>& center_offsets,
                       parlay::sequence<size_t>& children_offsets,
                       parlay::sequence<size_t>& point_offsets, size_t height) {
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
    for (node_t* child : node->children) {
      h = std::max(h, traverse_tree(child, ind_to_node, node_to_ind, center_offsets,
                                    children_offsets, point_offsets, height + 1));
    }
    return h;
  }

  // Write the index to a file in disk
  void save(const std::string& filename) override {
    std::ofstream outfile(filename, std::ios::binary);
    std::cout << "Saving index to " << filename << std::endl;
    if (!outfile.is_open()) {
      std::cerr << "Error opening file for writing: " << filename << std::endl;
      return;
    }

    // Collect data
    parlay::sequence<node_t*> ind_to_node;
    std::unordered_map<node_t*, size_t> node_to_ind;
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
    outfile.write(reinterpret_cast<const char*>(&num), sizeof(size_t));
    // Write center offsets
    size_t num_center_offsets = center_offsets.size();
    outfile.write(reinterpret_cast<const char*>(&num_center_offsets), sizeof(size_t));
    outfile.write(reinterpret_cast<const char*>(center_offsets.begin()),
                  center_offsets.size() * sizeof(size_t));
    // Write centers values
    for (size_t i = 0; i < num; ++i) {
      node_t* node = ind_to_node[i];
      if (node->children.size() > 0) {  // Internal Nodes only
        auto coords = node->data.data();
        size_t num_entries = (node->data.total_size()) * (node->data.get_dims());
        outfile.write(reinterpret_cast<const char*>(coords), num_entries * sizeof(float));
      }
    }
    // Write children offsets
    outfile.write(reinterpret_cast<const char*>(children_offsets.begin()),
                  children_offsets.size() * sizeof(size_t));
    // Write children values
    for (size_t i = 0; i < num; ++i) {
      node_t* node = ind_to_node[i];
      parlay::sequence<node_t*> children = node->children;
      for (size_t j = 0; j < children.size(); ++j) {  // TODO: make parallel
        node_t* child = children[j];
        size_t child_id = node_to_ind[child];
        outfile.write(reinterpret_cast<const char*>(&child_id), sizeof(size_t));
      }
    }
    // Write point offsets
    outfile.write(reinterpret_cast<const char*>(point_offsets.begin()),
                  point_offsets.size() * sizeof(size_t));
    // Write point values
    for (size_t i = 0; i < num; ++i) {
      node_t* node = ind_to_node[i];
      if (node->children.size() == 0) {  // <-- UPDATED: leaves only
        PointCloudSet<ChPoint> points = node->data;
        for (size_t j = 0; j < points.size(); ++j) {  // TODO: make parallel
          uint32_t point_id = points.get_id(j);
          outfile.write(reinterpret_cast<const char*>(&point_id), sizeof(uint32_t));
        }
      }
    }

    // Quantization: save only the MODEL (leaf encodings are reconstructed on load)
    int type_id = static_cast<int>(quantization_mode);
    outfile.write(reinterpret_cast<const char*>(&type_id), sizeof(int));

    std::visit(
        [&](auto& model) {
          // Get the concrete type of the model the variant is currently holding
          using ModelType = std::decay_t<decltype(model)>;

          // If it's a dummy type or uninitialized, do nothing (handles QT::None)
          if constexpr (std::is_same_v<ModelType, std::monostate>) {
            return;
          } else {
            model.save(outfile);
          }
        },
        quantizer);

    outfile.close();
  }

  // Read the index from a file in disk
  // TODO: Change this to a constructor, and also save the index_params.
  void load(const std::string& filename, const PointCloudSet<ChPoint>& points) override {
    std::ifstream infile(filename, std::ios::binary);
    std::cout << "Loading index from " << filename << std::endl;
    if (!infile.is_open()) {
      std::cerr << "Error opening file for reading: " << filename << std::endl;
      return;
    }

    // Read number of nodes
    size_t num = 0;
    infile.read(reinterpret_cast<char*>(&num), sizeof(size_t));
    // Read center offsets
    size_t num_center_offsets = 0;
    infile.read(reinterpret_cast<char*>(&num_center_offsets), sizeof(size_t));
    parlay::sequence<size_t> center_offsets(num_center_offsets);
    infile.read(reinterpret_cast<char*>(center_offsets.begin()),
                center_offsets.size() * sizeof(size_t));
    // Read centers values
    parlay::sequence<float> center_values(center_offsets[center_offsets.size() - 1]);
    infile.read(reinterpret_cast<char*>(center_values.begin()),
                center_values.size() * sizeof(float));
    // Read children offsets
    parlay::sequence<size_t> children_offsets(num + 1);
    infile.read(reinterpret_cast<char*>(children_offsets.begin()),
                children_offsets.size() * sizeof(size_t));
    // Read children values
    parlay::sequence<size_t> children_values(children_offsets[children_offsets.size() - 1]);
    infile.read(reinterpret_cast<char*>(children_values.begin()),
                children_values.size() * sizeof(size_t));
    // Read point offsets
    parlay::sequence<size_t> point_offsets(num + 1);
    infile.read(reinterpret_cast<char*>(point_offsets.begin()),
                point_offsets.size() * sizeof(size_t));
    // Read point values
    parlay::sequence<uint32_t> point_values(point_offsets[point_offsets.size() - 1]);
    infile.read(reinterpret_cast<char*>(point_values.begin()),
                point_values.size() * sizeof(uint32_t));

    // Quantization
    int type_id = 0;
    infile.read(reinterpret_cast<char*>(&type_id), sizeof(int));
    quantization_mode = static_cast<QT>(type_id);
    std::cout << "Quantization mode: " << type_id << std::endl;

    switch (quantization_mode) {
      case QT::PQ: quantizer.template emplace<typename MVQT::PQ_Model>().load(infile); break;
      case QT::RaBitQ: quantizer.template emplace<typename MVQT::RQ_Model>().load(infile); break;
      case QT::FastScan: quantizer.template emplace<typename MVQT::FS_Model>().load(infile); break;
      case QT::TurboQuant:
        quantizer.template emplace<typename MVQT::TQ_Model>().load(infile);
        break;
      case QT::None:
      default: quantizer = std::monostate{}; break;
    }
    infile.close();

    // Center quantization (internal-node / leaf-center scoring): always TQ.
    init_tq_quantizer(points);

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
    parlay::sequence<node_t*> ind_to_node =
        parlay::sequence<node_t*>::from_function(num, [&](size_t i) {
          node_t* node = new node_t();
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
      node_t* node = ind_to_node[i];
      parlay::sequence<node_t*>& children = node->children;
      parlay::parallel_for(0, children.size(), [&](size_t j) {
        size_t child_id = children_values[children_offsets[i] + j];
        children[j] = ind_to_node[child_id];
      });
    });
    root = ind_to_node[0];

    // Rebuild flat leaf index and leaf centers
    leaves_flat.clear();
    leaf_centers_quant = std::monostate{};
    leaf_to_root_child_.clear();
    if (root) {
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
        if (params.quantize_centers) {
          leaf_centers_quant = encode_tq(leaf_centers);
        }
      }
    }

    // Re-encode nodes (since we only saved the model)
    parlay::parallel_for(
        0, num,
        [&](size_t i) {
          node_t* node = ind_to_node[i];
          if (!node) return;
          if (node->data.size() == 0) return;
          if (!node->children.empty()) {  // internal node
            if (params.quantize_centers) {
              node->quantized_data = encode_tq(node->data);
            }
            return;
          }
          // leaf node
          if (quantization_mode != QT::None) {
            node->quantized_data = this->encode_points_quantized(node->data, quantizer);
          }
        },
        /*granularity=*/1);
  }

  // Traversing the tree and deleting nodes
  void traverse_and_delete(node_t* node) {
    for (size_t i = 0; i < node->children.size(); i++) {
      node_t* child = node->children[i];
      traverse_and_delete(child);
      delete child;
    }
  }

  ~IndexMVIVF() {
    if (root != nullptr) {
      traverse_and_delete(root);
      delete root;
    }
  }

  // Computing Tree Stats
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
    std::function<size_t(const node_t*, size_t)> visit = [&](const node_t* node,
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
      for (const node_t* child : node->children) {
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
        for (const node_t* child : node->children) {
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

  // Returns, for each point id, the index of the root-level child subtree that contains it.
  //
  // The output is a sequence `root_child_of_point` such that:
  //   - `root_child_of_point[id]` is the (0-based) index of the direct child of the root whose
  //     subtree contains point with global id `id`.
  //   - If a point id does not appear in the tree (should not happen for a well-formed index),
  //     its entry will be UINT32_MAX.
  //
  // Assumes that point ids are 0..N-1 and match the ids stored in the leaf PointCloudSets.
  parlay::sequence<uint32_t> get_root_child_clustering() const {
    parlay::sequence<uint32_t> empty;
    if (root == nullptr) return empty;
    if (root->children.empty()) return empty;

    std::vector<std::pair<uint32_t, uint32_t>> assignments;
    assignments.reserve(1024);
    size_t max_id = 0;

    std::function<void(const node_t*, uint32_t)> visit = [&](const node_t* node,
                                                             uint32_t root_child_idx) {
      if (node->children.empty()) {
        size_t n = node->data.size();
        for (size_t j = 0; j < n; ++j) {
          uint32_t id = node->data.get_id(j);
          assignments.emplace_back(id, root_child_idx);
          if (id > max_id) max_id = id;
        }
      } else {
        for (size_t i = 0; i < node->children.size(); ++i) {
          uint32_t next_root_child_idx = root_child_idx;
          if (node == root) {
            next_root_child_idx = static_cast<uint32_t>(i);
          }
          visit(node->children[i], next_root_child_idx);
        }
      }
    };

    for (size_t i = 0; i < root->children.size(); ++i) {
      const node_t* child = root->children[i];
      if (child != nullptr) visit(child, static_cast<uint32_t>(i));
    }

    if (assignments.empty()) return empty;

    parlay::sequence<uint32_t> root_child_of_point(max_id + 1);
    parlay::parallel_for(0, root_child_of_point.size(),
                         [&](size_t i) { root_child_of_point[i] = UINT32_MAX; });

    parlay::parallel_for(0, assignments.size(), [&](size_t i) {
      auto [pid, rid] = assignments[i];
      root_child_of_point[pid] = rid;
    });

    return root_child_of_point;
  }

  // Returns, for each point id, the list of leaf indices (0-based) that contain it.
  // Non-spill: 1 leaf per point. Spill: 1 or 2 leaves per point.
  // Used for greedy set-cover diversity and best/second-best leaf metrics.
  std::vector<std::vector<uint32_t>> get_leaves_of_point() const {
    std::vector<std::vector<uint32_t>> out;
    if (root == nullptr) return out;

    std::vector<std::pair<uint32_t, uint32_t>> assignments;
    assignments.reserve(1024);
    size_t max_id = 0;
    size_t leaf_idx = 0;

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
        for (const node_t* child : node->children) {
          if (child != nullptr) visit(child);
        }
      }
    };
    visit(root);
    if (assignments.empty()) return out;

    out.resize(max_id + 1);
    for (const auto& [pid, lid] : assignments) {
      out[pid].push_back(lid);
    }
    return out;
  }

  // Returns, for each point id, the list of root-child indices that contain it (1 or 2 with spill).
  // Derived from get_leaves_of_point() and leaf_to_root_child_. Call after build/load.
  std::vector<std::vector<uint32_t>> get_root_children_of_point() const {
    auto leaves_of_point = get_leaves_of_point();
    if (leaves_of_point.empty() || leaf_to_root_child_.empty()) return {};
    std::vector<std::vector<uint32_t>> out(leaves_of_point.size());
    for (size_t pid = 0; pid < leaves_of_point.size(); ++pid) {
      std::unordered_set<uint32_t> rids;
      for (uint32_t lid : leaves_of_point[pid]) {
        if (lid < leaf_to_root_child_.size()) rids.insert(leaf_to_root_child_[lid]);
      }
      out[pid].assign(rids.begin(), rids.end());
    }
    return out;
  }

  size_t get_height() const noexcept override { return kmeanstree_height; }

  // [Working]: Specialized search_all function that is more cache-efficient.
  // Idea:
  // 1. Run the greedy/flat search part independently in parallel, collecting the output in a single
  // pass. This emits per query nprobes many leaves that it would like to probe.
  // 2. Group queries by the leaves they would like to probe.
  // 3. For each leaf, we score all queries agains all points in the leaf to return top num_rerank/k
  // results. This is done via a cache-efficient many-to-many kernel. Currently implemented for
  // Exact and FastScan. It is hard to obtain speedups on exact, as leaves and queries can be huge,
  // and independent processing is already good. Finally, scatter the results into a global buffer.
  // 4. Re-rank if necessary.

  std::pair<parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>, size_t> search_all_new(
      const PointCloudSet<ChPoint>& query_points, const PointCloudSet<ChPoint>& points,
      const SearchParams& search_params) {
    if (search_params.nprobes <= 16) {
      return Index<metric>::search_all(query_points, points, search_params);
    }
    parlay::internal::timer t;

    const size_t num_q = query_points.size();
    const size_t num_leaves = leaves_flat.size();
    const size_t k = search_params.k;
    size_t nprobes = std::min(num_leaves, search_params.nprobes);
    size_t bytes_accessed = 0;
    size_t dist_cmps = 0;

    // ---------------------------------------------------------------------
    // Step 0: Pre-Quantize Queries
    // ---------------------------------------------------------------------
    t.start();
    auto q_queries = parlay::sequence<QuantQuery>::uninitialized(num_q);
    if (quantization_mode != QT::None) {
      parlay::parallel_for(0, num_q, [&](size_t i) {
        q_queries[i] = this->quantize_query_point_cloud(query_points[i], quantizer);
      });
    }
    t.stop();
    std::cout << "[MVIVF] Query Quantization: " << t.total_time() << " sec" << std::endl;
    t.reset();

    // ---------------------------------------------------------------------
    // Step 1: Greedy Search
    // ---------------------------------------------------------------------
    t.start();
    auto leaf_query_pairs =
        parlay::sequence<std::pair<node_t*, std::pair<uint32_t, uint32_t>>>::uninitialized(nprobes *
                                                                                           num_q);
    auto dist_cmps_gs = parlay::sequence<size_t>::uninitialized(num_q);
    auto bytes_accessed_gs = parlay::sequence<size_t>::uninitialized(num_q);

    parlay::parallel_for(0, num_q, [&](uint32_t i) {
      TQ_Query q_center_query;
      if (params.quantize_centers) {
        q_center_query = center_quantizer.quantize_query(query_points[i]);
      }
      const double alpha = 1.0;  // heuristic threshold: TODO: set this
      bool use_flat = (num_leaves > 0 && nprobes >= static_cast<size_t>(alpha * num_leaves));
      GreedySearchResult gs;
      if (use_flat) {
        gs = flat_leaf_search(query_points[i], q_center_query, nprobes);
      } else {
        gs = greedy_search(query_points[i], q_center_query, nprobes);
      }
      auto& probe_list = gs.probe_list;
      dist_cmps_gs[i] = static_cast<size_t>(gs.stats[0]);
      bytes_accessed_gs[i] = gs.bytes_accessed;
      parlay::parallel_for(0, probe_list.size(), [&](uint32_t j) {
        leaf_query_pairs[i * nprobes + j] = {probe_list[j].second, std::make_pair(i, j)};
      });
    });
    t.stop();
    std::cout << "[MVIVF] Greedy Search: " << t.total_time() << " sec" << std::endl;
    t.reset();
    bytes_accessed += parlay::reduce(bytes_accessed_gs);
    dist_cmps += parlay::reduce(dist_cmps_gs);

    // ---------------------------------------------------------------------
    // Step 2: Group
    // ---------------------------------------------------------------------
    t.start();
    auto grouped = mvsic::group_by_key_inplace(leaf_query_pairs);
    t.stop();
    std::cout << "[MVIVF] Grouping: " << t.total_time() << " sec" << std::endl;
    t.reset();

    // ---------------------------------------------------------------------
    // Step 3: Leaf Probing
    // ---------------------------------------------------------------------
    t.start();
    size_t num_rerank = std::max(search_params.num_rerank, k);
    size_t max_cands_per_query = nprobes * num_rerank;

    auto all_candidates = parlay::sequence<std::pair<uint32_t, float>>(
        num_q * max_cands_per_query, {UINT32_MAX, std::numeric_limits<float>::max()});

    // HELPER 1: Default Parallel 1-to-N Loop (for PQ, RaBitQ, TQ)
    auto process_leaf_quant = [&]<typename SetType, typename QueryType>(
                                  node_t* leaf, const auto& group, size_t C) {
      auto* leaf_data = std::get_if<SetType>(&leaf->quantized_data);
      if (!leaf_data) UNREACHABLE();
      parlay::parallel_for(0, group.size(), [&](size_t j) {
        uint32_t q_id = group[j].second.first;
        uint32_t probe_idx = group[j].second.second;
        auto* q_query_ptr = std::get_if<QueryType>(&q_queries[q_id]);
        if (!q_query_ptr) UNREACHABLE();
        // Compute scores to all points in the leaf
        auto all_dists =
            parlay::sequence<std::pair<uint32_t, float>>::uninitialized(leaf->get_size());
        leaf_data->distances_all(*q_query_ptr, all_dists.data());
        // Collect top C
        if (C < all_dists.size()) {
          std::nth_element(all_dists.begin(), all_dists.begin() + C, all_dists.end(),
                           [](const auto& a, const auto& b) { return a.second < b.second; });
        }
        // Scatter to global buffer
        size_t base_idx = (q_id * max_cands_per_query) + (probe_idx * num_rerank);
        for (size_t c = 0; c < C; ++c) {
          all_candidates[base_idx + c] = all_dists[c];
        }
      });
    };

    // HELPER 2: Batched Many-to-Many M2M Kernel (for FastScan)
    auto process_leaf_quant_m2m = [&]<typename SetType, typename QueryType, typename M2MType>(
                                      node_t* leaf, const auto& group, size_t C) {
      auto* leaf_data = std::get_if<SetType>(&leaf->quantized_data);
      if (!leaf_data) UNREACHABLE();
      size_t num_queries_in_group = group.size();
      // Gather raw pointers to the strongly-typed queries
      std::vector<const QueryType*> typed_queries(num_queries_in_group);
      parlay::parallel_for(0, num_queries_in_group, [&](size_t j) {
        uint32_t q_id = group[j].second.first;
        auto* q_query_ptr = std::get_if<QueryType>(&q_queries[q_id]);
        if (!q_query_ptr) UNREACHABLE();
        typed_queries[j] = q_query_ptr;
      });

      std::vector<std::pair<uint32_t, float>> batch_results(num_queries_in_group * num_rerank);
      M2MType::TopKIntoUninitialized(typed_queries, *leaf_data, num_rerank, batch_results.data(),
                                     num_queries_in_group, /*parallel_query_blocks=*/true);
      // Scatter to global buffer
      parlay::parallel_for(0, num_queries_in_group, [&](size_t j) {
        uint32_t q_id = group[j].second.first;
        uint32_t probe_idx = group[j].second.second;
        size_t base_idx = (q_id * max_cands_per_query) + (probe_idx * num_rerank);
        for (size_t c = 0; c < C; ++c) {
          all_candidates[base_idx + c] = batch_results[j * num_rerank + c];
        }
      });
    };

    // For each leaf group in parallel:
    parlay::parallel_for(0, grouped.size(), [&](size_t i) {
      auto& group = grouped[i];
      node_t* leaf = group[0].first;
      size_t C = std::min<size_t>(num_rerank, leaf->get_size());
      size_t num_queries_in_group = group.size();
      switch (quantization_mode) {
        case QT::PQ:
          process_leaf_quant.template operator()<typename MVQT::PQ_Set, typename MVQT::PQ_Query>(
              leaf, group, C);
          break;
        case QT::RaBitQ:
          process_leaf_quant.template operator()<typename MVQT::RQ_Set, typename MVQT::RQ_Query>(
              leaf, group, C);
          break;
        case QT::FastScan:
          // Route to FastScan's M2M Kernel
          process_leaf_quant_m2m
              .template operator()<typename MVQT::FS_Set, typename MVQT::FS_Query,
                                   fastscan_mv::ManyToMany<typename MVQT::FS_Set>>(leaf, group, C);
          break;
        case QT::TurboQuant:
          process_leaf_quant.template operator()<typename MVQT::TQ_Set, typename MVQT::TQ_Query>(
              leaf, group, C);
          break;
        case QT::None: {
          auto query_ids = parlay::delayed_tabulate(
              num_queries_in_group, [&](size_t j) { return group[j].second.first; });
          PointCloudSet<ChPoint> batched_queries(query_points.filter(query_ids), d);
          auto leaf_results = leaf->data.distances(batched_queries, num_rerank);
          parlay::parallel_for(0, num_queries_in_group, [&](size_t j) {
            uint32_t q_id = group[j].second.first;
            uint32_t probe_idx = group[j].second.second;
            size_t base_idx = (q_id * max_cands_per_query) + (probe_idx * num_rerank);
            for (size_t c = 0; c < C; ++c) {
              all_candidates[base_idx + c] = leaf_results[j * num_rerank + c];
            }
          });
          break;
        }
        default: UNREACHABLE();
      }
    });
    t.stop();
    std::cout << "[MVIVF] Probing & Scattering: " << t.total_time() << " sec" << std::endl;
    t.reset();
    // TODO: update bytes_accessed and dist_cmps

    // ---------------------------------------------------------------------
    // Step 4: Aggregation and Re-ranking
    // ---------------------------------------------------------------------
    t.start();
    auto final_results = parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>(num_q);
    auto bytes_accessed_rerank = parlay::sequence<size_t>::uninitialized(num_q);

    parlay::parallel_for(0, num_q, [&](size_t q_id) {
      size_t base_idx = q_id * max_cands_per_query;
      auto q_cands = parlay::sequence<std::pair<uint32_t, float>>(
          all_candidates.begin() + base_idx,
          all_candidates.begin() + base_idx + max_cands_per_query);
      parlay::sort_inplace(q_cands,
                           [](const auto& a, const auto& b) { return a.second < b.second; });
      parlay::sequence<std::pair<uint32_t, float>> top_cands;
      top_cands.reserve(num_rerank);

      for (size_t c = 0; c < q_cands.size() && top_cands.size() < num_rerank; ++c) {
        if (q_cands[c].first == UINT32_MAX) break;
        top_cands.push_back(q_cands[c]);
      }
      auto q_final = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(
          std::min(k, top_cands.size()));
      bytes_accessed_rerank[q_id] = 0;

      if (search_params.num_rerank > 0) {
        size_t actual_rerank = std::min(num_rerank, top_cands.size());
        bytes_accessed_rerank[q_id] =
            this->rerank(query_points[q_id], points, top_cands, actual_rerank, q_final);
      } else {
        for (size_t c = 0; c < q_final.size(); ++c) {
          q_final[c] = top_cands[c];
        }
      }

      final_results[q_id] = std::move(q_final);
    });
    t.stop();
    std::cout << "[MVIVF] Aggregation and Re-ranking: " << t.total_time() << " sec" << std::endl;
    t.reset();
    bytes_accessed += parlay::reduce(bytes_accessed_rerank);

    std::cout << "[MVIVF] Bytes Accessed: " << bytes_accessed << std::endl;
    std::cout << "[MVIVF] Dist Cmps: " << dist_cmps << std::endl;

    return std::make_pair(std::move(final_results), bytes_accessed);
  }
};

using IndexMVIVFL2 = IndexMVIVF<true>;   // Instantiates for L2 metric (metric = true)
using IndexMVIVFIP = IndexMVIVF<false>;  // Instantiates for MIPS      (metric = false)

}  // namespace mvsic