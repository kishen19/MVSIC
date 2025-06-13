#pragma once

#include <queue>
#include <set>
#include "src/common/index.h"
#include "src/mvc/mvkmeans.h"
#include "src/utils/top_neighbors.h"

namespace mvivf {

/* Params Type */
struct IndexMVIVFParams {
  size_t minsize = 100;  // (Expected) Minsize of leaf clusters (not enforced)
  size_t maxsize = 500;  // Maxsize of leaf clusters (enforced)
  size_t s = 0;          // Number of vectors in a center. Default 0 - average # vectors/doc
  size_t iters = 5;      // Number of Outer Lloyd's Iterations
  std::string seeding = "Random";  // Seeding Algorithm
  size_t os_rate = 20;             // Oversampling rate for Inner Kmeans
  bool verbose = false;            // Print debug statements
};

/* Multi-Vector IVF Internal Node Type */
template<typename ChPoint>
struct IndexMVIVFNode {
  parlay::sequence<IndexMVIVFNode *> children;  // Children
  ChPoint center;                               // Except root, every node has a center-set
  PointCloudSet<ChPoint> points;                // Only leaf nodes have points

  IndexMVIVFNode() :
      children(parlay::sequence<IndexMVIVFNode *>(0)),
      center(ChPoint()),
      points(PointCloudSet<ChPoint>()) {}

  inline void set_center(const ChPoint &center_) { center = center_; }
  inline void set_points(const PointCloudSet<ChPoint> &points_) { points = points_; }
};

/* Main Multi-Vector IVF Class */
template<bool metric>
class IndexMVIVF : Index<metric>, IndexMVIVFParams {
 public:
  using ChPoint = Index<metric>::ChPoint;
  using node_t = IndexMVIVFNode<ChPoint>;
  // using node_allocator = parlay::type_allocator<node_t>;
  using Index<metric>::d;

  node_t *root = nullptr;

  IndexMVIVF(size_t d_) noexcept { d = d_; }
  IndexMVIVF(size_t d_, const IndexMVIVFParams &params) noexcept : IndexMVIVFParams(params) {
    d = d_;
  }
  ~IndexMVIVF();

  // Builds the index given a point cloud set.
  void build(const PointCloudSet<ChPoint> &points) override {
    root = new node_t();
    // root = node_allocator::create();
    build_helper(root, points);
  }
  // Recursively builds the kmeans tree
  void build_helper(node_t *node, const PointCloudSet<ChPoint> &points);
  // Returns the top-k point clouds for the query point cloud
  // Output format: < [<id, distance>, ...], # distance comparisons>
  std::pair<parlay::sequence<std::pair<size_t, float>>, size_t> search(
      const ChPoint &query, const PointCloudSet<ChPoint> &points,
      const SearchParams &params) override;
  // Traversing the k-means tree: returns the height of the tree
  // TODO: get more stats about the tree
  size_t traverse_tree(node_t *node, parlay::sequence<node_t *> &ind_to_node,
                       std::unordered_map<node_t *, size_t> &node_to_ind,
                       parlay::sequence<size_t> &center_offsets,
                       parlay::sequence<size_t> &children_offsets,
                       parlay::sequence<size_t> &point_offsets, size_t height);
  // Write the index to a file in disk
  void save(const std::string &filename) override;
  // Read the index from a file in disk
  void load(const std::string &filename, const PointCloudSet<ChPoint> &points) override;
  // Traversing the tree and deleting nodes
  void traverse_and_delete(node_t *node);
};

/* -----------------------------------------Implementation-----------------------------------------*/

// Recursively builds the kmeans tree
template<bool metric>
void IndexMVIVF<metric>::build_helper(node_t *node, const PointCloudSet<ChPoint> &points) {
  size_t n = points.size();
  size_t mp = std::max((size_t)4, maxsize / minsize);
  // Num Clusters = min(sqrt(n), mp*n/maxsize)
  size_t num_clusters = std::min(mp * std::ceil(n / maxsize), std::ceil(std::sqrt(n)));
  if (verbose) {
    std::cout << "Building index with " << n << " points, maxsize: " << maxsize
              << ", num_clusters: " << num_clusters << std::endl;
  }
  // Step 1: Run MV-Lloyds on points and collect clusters
  MVClusteringParams params(iters, seeding, os_rate, false, verbose);
  MVClustering<metric> clus(d, num_clusters, s, params);
  clus.train(points);
  PointCloudSet<ChPoint> centers = clus.centers;
  parlay::sequence<size_t> cluster_ids = clus.cluster_ids;
  auto id_pt = parlay::delayed_seq<std::pair<size_t, size_t>>(
      n, [&](size_t i) { return std::make_pair(cluster_ids[i], i); });
  auto grouped = parlay::group_by_index(id_pt, num_clusters);
  // Step 2: Update children nodes and recurse for large nodes
  parlay::sequence<node_t *> children = parlay::sequence<node_t *>(num_clusters);
  // for (size_t i = 0; i < num_clusters; i++) {
  parlay::parallel_for(0, num_clusters, [&](size_t i) {
    node_t *child = new node_t();
    // node_t *child = node_allocator::create();
    child->set_center(centers[i]);
    PointCloudSet<ChPoint> child_points = PointCloudSet<ChPoint>(points.filter(grouped[i]), d);
    if (child_points.size() > maxsize) {
      build_helper(child, child_points);
    } else {
      child->set_points(child_points);
    }
    children[i] = child;
  });
  node->children = children;
}

// Returns the top-k point clouds for the query point cloud
// Output format: < [<id, distance>, ...], # distance comparisons>
template<bool metric>
std::pair<parlay::sequence<std::pair<size_t, float>>, size_t> IndexMVIVF<metric>::search(
    const ChPoint &query, const PointCloudSet<ChPoint> &points, const SearchParams &params) {
  size_t k = params.k;
  size_t nprobes = params.nprobes;
  size_t beam_length = params.beam_length;
  // probe_list contains the final candidate leaf nodes to probe
  parlay::sequence<std::pair<float, node_t *>> probe_list;
  std::set<std::pair<float, node_t *>> beam;
  // absl::btree_set<std::pair<float, node_t*>> probe_list;
  // absl::btree_set<std::pair<float, node_t*>> beam;
  size_t dist_cmps = 0;
  auto add_to_probe_list = [&](std::pair<float, node_t *> p) { probe_list.push_back(p); };
  auto add_to_beam = [&](std::pair<float, node_t *> p) -> bool {
    if (beam.size() < beam_length || p.first < beam.rbegin()->first) {
      beam.insert(p);
      if (beam.size() > beam_length) {
        beam.erase(std::prev(beam.end()));  // Remove the farthest node
      }
      return true;
    }
    return false;
  };
  // Step 1: Greedy search to find candidate probe clusters
  // Add root to beam
  add_to_beam({std::numeric_limits<float>::max(), root});
  while (beam.size() > 0) {
    // Pop the best node from the beam
    std::pair<float, node_t *> best = *beam.begin();
    beam.erase(beam.begin());
    node_t *current_node = best.second;
    // Compute distances from query to children
    parlay::sequence<size_t> cmps(current_node->children.size());
    auto children = current_node->children;
    auto res =
        parlay::sequence<std::pair<float, node_t *>>::from_function(children.size(), [&](size_t i) {
          node_t *child = children[i];
          float dist = query.distance(child->center);
          cmps[i] = (query.size() * child->center.size());
          return std::make_pair(dist, child);
        });
    dist_cmps += parlay::reduce(cmps);
    // Collect leaf and non-leaf nodes
    auto new_nodes_to_beam = parlay::filter(res, [](const auto &p) {
      return p.second->children.size() != 0;  // Only keep nodes that are not leaves
    });
    auto new_nodes_to_probe = parlay::filter(res, [](const auto &p) {
      return p.second->children.size() == 0;  // Only keep leaf nodes
    });
    parlay::sort_inplace(new_nodes_to_beam, [](const auto &a, const auto &b) {
      return a.first < b.first;  // Sort by distance
    });
    // Add new nodes to beam and probe list
    for (size_t i = 0; i < std::min(beam_length, new_nodes_to_beam.size()); i++) {
      if (!add_to_beam(new_nodes_to_beam[i])) break;
    }
    for (size_t i = 0; i < new_nodes_to_probe.size(); i++) {
      add_to_probe_list(new_nodes_to_probe[i]);
    }
  }
  // Step 2: Probe clusters in probe_list
  parlay::sort_inplace(probe_list);
  size_t nprobes_minimal = 0, cur = 0;
  while (nprobes_minimal < probe_list.size() && cur <= k) {
    cur += probe_list[nprobes_minimal].second->points.size();
    nprobes_minimal++;
  }
  nprobes = std::min(probe_list.size(), std::max(nprobes, nprobes_minimal));
  auto probe_dist_cmps = parlay::sequence<size_t>::uninitialized(nprobes);
  auto results = parlay::sequence<parlay::sequence<std::pair<size_t, float>>>::from_function(
      nprobes, [&](size_t i) {
        std::pair<float, node_t *> p = *std::next(probe_list.begin(), i);
        node_t *node = p.second;
        PointCloudSet<ChPoint> cluster_points = node->points;
        auto [res, d_c] = get_knn(query, cluster_points, k);
        probe_dist_cmps[i] = d_c;
        return res;
      });
  dist_cmps += parlay::reduce(probe_dist_cmps);
  // Step 3: Re-ranking (lightweight; no new distance cmps)
  // Flatten, sort and return top k // TODO: coarse and fine distances for
  // better performance
  // TODO: make a separate function called re-ranking, to test other strategies
  auto flattened_results = parlay::flatten(results);
  parlay::sort_inplace(flattened_results, [](const auto &a, const auto &b) {
    return a.second < b.second;  // Sort by distance
  });
  auto final_results = parlay::sequence<std::pair<size_t, float>>::from_function(
      std::min(k, flattened_results.size()), [&](size_t i) { return flattened_results[i]; });
  return std::make_pair(final_results, dist_cmps);
}

template<bool metric>
size_t IndexMVIVF<metric>::traverse_tree(node_t *node, parlay::sequence<node_t *> &ind_to_node,
                                         std::unordered_map<node_t *, size_t> &node_to_ind,
                                         parlay::sequence<size_t> &center_offsets,
                                         parlay::sequence<size_t> &children_offsets,
                                         parlay::sequence<size_t> &point_offsets, size_t height) {
  node_to_ind[node] = ind_to_node.size();
  ind_to_node.push_back(node);
  center_offsets.push_back((node->center.size()) * (node->center.get_dims()));
  children_offsets.push_back(node->children.size());
  point_offsets.push_back(node->points.size());
  size_t h = height + 1;
  for (node_t *child : node->children) {
    h = std::max(h, traverse_tree(child, ind_to_node, node_to_ind, center_offsets, children_offsets,
                                  point_offsets, height + 1));
  }
  return h;
}

template<bool metric>
void IndexMVIVF<metric>::save(const std::string &filename) {
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
  outfile.write(reinterpret_cast<const char *>(center_offsets.begin()),
                center_offsets.size() * sizeof(size_t));
  // Write center values
  for (size_t i = 0; i < num; ++i) {  // TODO: make parallel
    node_t *node = ind_to_node[i];
    if (node->center.size() > 0) {
      auto coords = node->center.get_slice();
      outfile.write(reinterpret_cast<const char *>(coords.begin()), coords.size() * sizeof(float));
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
    PointCloudSet<ChPoint> points = node->points;
    for (size_t j = 0; j < points.size(); ++j) {
      size_t point_id = points.get_id(j);
      outfile.write(reinterpret_cast<const char *>(&point_id), sizeof(size_t));
    }
  }
  outfile.close();
}

template<bool metric>
void IndexMVIVF<metric>::load(const std::string &filename, const PointCloudSet<ChPoint> &points) {
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
  parlay::sequence<size_t> center_offsets(num + 1);
  infile.read(reinterpret_cast<char *>(center_offsets.begin()),
              center_offsets.size() * sizeof(size_t));
  // Read center values
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
  parlay::sequence<size_t> point_values(point_offsets[point_offsets.size() - 1]);
  infile.read(reinterpret_cast<char *>(point_values.begin()), point_values.size() * sizeof(size_t));

  // Build the index
  size_t dim = points.get_dims();
  auto point_id_to_data_id = parlay::sequence<size_t>::uninitialized(points.size());
  parlay::parallel_for(0, points.size(),
                       [&](size_t i) { point_id_to_data_id[points.get_id(i)] = i; });
  parlay::sequence<size_t> center_sizes = parlay::sequence<size_t>::from_function(
      num, [&](size_t i) { return center_offsets[i + 1] - center_offsets[i]; });
  parlay::sequence<size_t> children_sizes = parlay::sequence<size_t>::from_function(
      num, [&](size_t i) { return children_offsets[i + 1] - children_offsets[i]; });
  parlay::sequence<size_t> point_sizes = parlay::sequence<size_t>::from_function(
      num, [&](size_t i) { return point_offsets[i + 1] - point_offsets[i]; });
  parlay::sequence<node_t *> ind_to_node =
      parlay::sequence<node_t *>::from_function(num, [&](size_t i) {
        node_t *node = new node_t();
        // node_t *node = node_allocator::create();
        if (center_sizes[i] > 0) {
          node->set_center(
              ChPoint(center_sizes[i] / dim, dim, center_values.begin() + center_offsets[i]));
        }
        node->children.resize(children_sizes[i]);
        if (point_sizes[i] > 0) {
          parlay::sequence<size_t> point_group =
              parlay::sequence<size_t>::from_function(point_sizes[i], [&](size_t j) {
                size_t point_id = point_values[point_offsets[i] + j];
                return point_id_to_data_id[point_id];
              });
          node->set_points(PointCloudSet<ChPoint>(points.filter(point_group), dim));
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
  infile.close();
}

template<bool metric>
void IndexMVIVF<metric>::traverse_and_delete(node_t *node) {
  for (size_t i = 0; i < node->children.size(); i++) {
    node_t *child = node->children[i];
    traverse_and_delete(child);
    delete child;
    // node_allocator::destroy(node);
  }
}

template<bool metric>
IndexMVIVF<metric>::~IndexMVIVF() {
  // traverse_and_delete(root);
  // delete root;
  // node_allocator::destroy(root);
  root = nullptr;
}

template struct IndexMVIVF<true>;   // Instantiates for L2 metric (metric = true)
template struct IndexMVIVF<false>;  // Instantiates for MIPS      (metric =
                                    // false)

}  // namespace mvivf