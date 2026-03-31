#pragma once

#include <algorithm>
#include <queue>
#include <set>
#include <tuple>
#include <vector>
#include <optional>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "mvsic/core/index.h"
#include "mvsic/core/mvclustering/mvclustering.h"
#include "mvsic/core/utils/util.h"

namespace mvsic {

/* Multi-Vector IVF Index
  Indexing:
  - Phase 1: Builds a temporary pointer-based kmeans tree.
  - Phase 2: Flattens the tree into 3 optimized contiguous buffers:
      1. centroids_pcs: Internal nodes' centers (optimized for greedy_search)
      2. leaf_centroids_pcs: Leaf representatives (optimized for flat_search)
      3. clusters_pcs: The original point clouds, shuffled so each leaf cluster is contiguous.

  Search:
  - flat_search: Scores query against all leaf_centroids_pcs and selects top `nprobes`.
  - greedy_search: Traverses centroids_pcs to find top `nprobes` leaves.
  - process_probes: Slices the inverted list (clusters_pcs) and computes exact distances.
*/

template<bool metric>
class IndexMVIVF : public Index<metric> {
 public:
  using ChPoint = typename Index<metric>::ChPoint;
  using Index<metric>::d;
  using Index<metric>::params;

  // Output type of Coarse Search that returns candidate leaves to probe
  struct GreedySearchResult {
    parlay::sequence<std::pair<float, uint32_t>> probe_list;  // <distance, node_idx>
    size_t dist_cmps = 0;
    std::vector<double> timings = {};
  };

  IndexMVIVF(size_t d_) noexcept;
  IndexMVIVF(size_t d_, const IndexParams& params_) noexcept;
  ~IndexMVIVF() noexcept;

  // Core API
  void build(const PointCloudSet<ChPoint>& points) override;
  void save(const std::string& filename) override;
  void load(const std::string& filename, const PointCloudSet<ChPoint>& points) override;

  std::tuple<parlay::sequence<std::pair<uint32_t, float>>, size_t, std::vector<double>>
  search_with_stats(const ChPoint& query, const PointCloudSet<ChPoint>& points,
                    const SearchParams& search_params) override;

  std::pair<parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>, size_t> search_all_new(
      const PointCloudSet<ChPoint>& query_points, const PointCloudSet<ChPoint>& points,
      const SearchParams& search_params);

  size_t get_height() const noexcept override { return kmeanstree_height; }

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

  TreeStats get_tree_stats() const;
  parlay::sequence<uint32_t> get_flat_clustering() const;
  parlay::sequence<uint32_t> get_root_child_clustering() const;
  std::vector<std::vector<uint32_t>> get_leaves_of_point() const;
  std::vector<std::vector<uint32_t>> get_root_children_of_point() const;

 private:
  // ---------------------------------------------------------
  // 1. Temporary Construction Node
  // ---------------------------------------------------------
  struct build_node_t {
    parlay::sequence<build_node_t*> children;
    PointCloudSet<ChPoint> centers;
    PointCloudSet<ChPoint> leaf_points;

    build_node_t() noexcept : children(), centers(), leaf_points() {}
    ~build_node_t() noexcept {
      for (auto child : children)
        delete child;
    }
  };

  // ---------------------------------------------------------
  // 2. Production Index Node (Optimized for 16-bytes per node)
  // ---------------------------------------------------------
  struct index_node_t {
    union {
      uint32_t children_start;  // Offset into `index_nodes` (Internal)
      uint32_t data_offset;     // Offset into `clusters_pcs` (Leaf)
    };
    union {
      uint32_t num_children;  // Number of children / centers (Internal)
      uint32_t num_data;      // Number of point clouds in this leaf (Leaf)
    };
    uint32_t center_idx_start;  // Offset into `centroids_pcs` (Internal)
    bool is_leaf;               // 1 byte (padded to 4 by compiler)
  };

  // ---------------------------------------------------------
  // Global Index State (The 3-Buffer Architecture)
  // ---------------------------------------------------------
  parlay::sequence<index_node_t> index_nodes;
  parlay::sequence<uint32_t> leaf_to_node;      // Maps leaf_centroids_pcs idx to index_nodes idx
  parlay::sequence<uint32_t> shuffled_indices;  // The only mapping saved to disk!

  PointCloudSet<ChPoint> centroids_pcs;       // For greedy_search
  PointCloudSet<ChPoint> leaf_centroids_pcs;  // For flat_search
  PointCloudSet<ChPoint> clusters_pcs;        // The Shuffled Inverted List

  // Stats
  size_t kmeanstree_height = 0;
  std::vector<uint32_t> leaf_to_root_child_;

  // Internal Helpers
  void recursive_build(build_node_t* node, const PointCloudSet<ChPoint>& points);
  void flatten_tree(build_node_t* temp_root, const PointCloudSet<ChPoint>& original_points);

  GreedySearchResult greedy_search(const ChPoint& query, size_t nprobes) const;
  GreedySearchResult flat_search(const ChPoint& query, size_t nprobes) const;

  auto process_probes(const ChPoint& query,
                      parlay::sequence<std::pair<float, uint32_t>>& probe_list) const;

  // Serialization Helpers
  void save_pcs(std::ofstream& out, const PointCloudSet<ChPoint>& pcs);
  PointCloudSet<ChPoint> load_pcs(std::ifstream& in);
};

using IndexMVIVFL2 = IndexMVIVF<true>;
using IndexMVIVFIP = IndexMVIVF<false>;

}  // namespace mvsic

// Include the heavily templated implementation
#include "mvivf-inl.h"