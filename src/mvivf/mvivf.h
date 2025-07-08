#pragma once

#include <queue>
#include <set>
#include "src/common/index.h"
#include "src/mvc/mvkmeans.h"
#include "src/utils/top_neighbors.h"

namespace mvivf {

/* =========================================Params Type======================================== */
struct IndexMVIVFParams {
  size_t minsize = 100;  // (Expected) Minsize of leaf clusters (not enforced)
  size_t maxsize = 500;  // Maxsize of leaf clusters (enforced)
  size_t s = 0;          // Number of vectors in a center. Default 0 - average # vectors/doc
  size_t iters = 5;      // Number of Outer Lloyd's Iterations
  std::string seeding = "Random";  // Seeding Algorithm
  size_t os_rate = 20;             // Oversampling rate for Inner Kmeans
  bool verbose = false;            // Print debug statements
};
/* =============================Multi-Vector IVF Internal Node Type============================ */
template<typename ChPoint>
struct IndexMVIVFNode {
  parlay::sequence<IndexMVIVFNode *> children;
  // For internal nodes: data = centers of children
  // For leaves:         data = points in the cluster
  PointCloudSet<ChPoint> data;
  IndexMVIVFNode() noexcept :
      children(parlay::sequence<IndexMVIVFNode *>(0)), data(PointCloudSet<ChPoint>()) {}
  inline void set_children(const parlay::sequence<IndexMVIVFNode *> &children_) noexcept {
    children = std::move(children_);
  }
  inline void set_data(const PointCloudSet<ChPoint> &data_) noexcept { data = std::move(data_); }
};
/* ===================================Multi-Vector IVF Class=================================== */
template<bool metric>
class IndexMVIVF : Index<metric>, IndexMVIVFParams {
 public:
  using ChPoint = Index<metric>::ChPoint;
  using node_t = IndexMVIVFNode<ChPoint>;
  using Index<metric>::d;  // Embedding dimension
  node_t *root = nullptr;  // Root of the k-means tree

  IndexMVIVF(size_t d_) noexcept { d = d_; }
  IndexMVIVF(size_t d_, const IndexMVIVFParams &params) noexcept : IndexMVIVFParams(params) {
    d = d_;
  }
  ~IndexMVIVF();
  /* ----------------------------Overridden Functions---------------------------- */
  void build(const PointCloudSet<ChPoint> &points) override;
  std::pair<parlay::sequence<std::pair<size_t, float>>, size_t> search(
      const ChPoint &query, const PointCloudSet<ChPoint> &points,
      const SearchParams &params) override;
  void save(const std::string &filename) override;
  void load(const std::string &filename, const PointCloudSet<ChPoint> &points) override;

  /* ------------------------------Helper Functions------------------------------ */
  // Recursively builds the kmeans tree
  void build_helper(node_t *node, const PointCloudSet<ChPoint> &points);
  // Traversing the k-means tree: returns the height of the tree
  // TODO: get more stats about the tree
  size_t traverse_tree(node_t *node, parlay::sequence<node_t *> &ind_to_node,
                       std::unordered_map<node_t *, size_t> &node_to_ind,
                       parlay::sequence<size_t> &center_offsets,
                       parlay::sequence<size_t> &children_offsets,
                       parlay::sequence<size_t> &point_offsets, size_t height);
  // Traversing the tree and deleting nodes
  void traverse_and_delete(node_t *node);
};

/* =======================================Implementation======================================= */
// Builds the index given a point cloud set.
template<bool metric>
void IndexMVIVF<metric>::build(const PointCloudSet<ChPoint> &points) {
  root = new node_t();
  build_helper(root, points);
}

// Recursively builds the kmeans tree
template<bool metric>
void IndexMVIVF<metric>::build_helper(node_t *node, const PointCloudSet<ChPoint> &points) {
  size_t n = points.size();
  size_t mp = std::max((size_t)4, maxsize / minsize);
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
  parlay::parallel_for(0, num_clusters, [&](size_t i) {
    node_t *child = new node_t();
    PointCloudSet<ChPoint> child_points = PointCloudSet<ChPoint>(points.filter(grouped[i]), d);
    if (child_points.size() > maxsize) {  // Recurse
      build_helper(child, child_points);
    } else {  // Leaf Node
      child->set_data(child_points);
    }
    children[i] = child;
  });
  node->set_children(children);
  node->set_data(centers);
}

// Returns the top-k point clouds for the query point cloud
// Output format: < [<id, distance>, ...], No. of distance computations >
template<bool metric>
std::pair<parlay::sequence<std::pair<size_t, float>>, size_t> IndexMVIVF<metric>::search(
    const ChPoint &query, const PointCloudSet<ChPoint> &points, const SearchParams &params) {
  size_t k = params.k;
  size_t nprobes = params.nprobes;
  size_t beam_length = params.beam_length;
  // probe_list: contains the final candidate leaf nodes to probe
  parlay::sequence<std::pair<float, node_t *>> probe_list;
  std::set<std::pair<float, node_t *>> beam;
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
  size_t dist_cmps = 0;
  while (beam.size() > 0) {
    // Pop the best node from the beam
    std::pair<float, node_t *> best = *beam.begin();
    beam.erase(beam.begin());
    node_t *current_node = best.second;
    // Compute distances from query to children
    auto children = current_node->children;
    auto centers = current_node->data;
    // Note: children.size() == centers.size()
    auto [all_dists, dist_cmps_node] = centers.distances(query);
    dist_cmps += dist_cmps_node;
    auto res = parlay::sequence<std::pair<float, node_t *>>::from_function(
        children.size(), [&](size_t i) { return std::make_pair(all_dists[i], children[i]); });
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
  // Find the minimum number of probes needed to obtain k neighbors
  size_t nprobes_minimal = 0, cur = 0;
  while (nprobes_minimal < probe_list.size() && cur <= k) {
    cur += probe_list[nprobes_minimal].second->data.size();
    nprobes_minimal++;
  }
  nprobes = std::min(probe_list.size(), std::max(nprobes, nprobes_minimal));
  auto probe_dist_cmps = parlay::sequence<size_t>::uninitialized(nprobes);
  auto results = parlay::sequence<parlay::sequence<std::pair<size_t, float>>>::from_function(
      nprobes, [&](size_t i) {
        std::pair<float, node_t *> p = *std::next(probe_list.begin(), i);
        node_t *node = p.second;
        PointCloudSet<ChPoint> cluster_points = node->data;
        auto [res, d_c] = get_knn(query, cluster_points, k);
        probe_dist_cmps[i] = d_c;
        return res;
      });
  dist_cmps += parlay::reduce(probe_dist_cmps);
  // Step 3: Re-ranking (lightweight; no new distance cmps)
  // Flatten, sort and return top k
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
  if (node->children.size() == 0) {  // leaves
    point_offsets.push_back(node->data.size());
  } else {  // Internal nodes
    point_offsets.push_back(0);
    size_t dims = node->data.get_dims();
    for (size_t i = 0; i < node->children.size(); i++) {
      center_offsets.push_back(node->data.get_size(i) * dims);  // # embeddings in center[i]
    }
  }
  children_offsets.push_back(node->children.size());
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
    if (node->children.size() == 0) {  // leaves only
      PointCloudSet<ChPoint> points = node->data;
      for (size_t j = 0; j < points.size(); ++j) {  // TODO: make parallel
        size_t point_id = points.get_id(j);
        outfile.write(reinterpret_cast<const char *>(&point_id), sizeof(size_t));
      }
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
  parlay::sequence<size_t> point_values(point_offsets[point_offsets.size() - 1]);
  infile.read(reinterpret_cast<char *>(point_values.begin()), point_values.size() * sizeof(size_t));

  // Build the index
  size_t dim = points.get_dims();
  auto point_id_to_data_id = parlay::sequence<size_t>::uninitialized(points.size());
  parlay::parallel_for(0, points.size(),
                       [&](size_t i) { point_id_to_data_id[points.get_id(i)] = i; });
  parlay::sequence<size_t> children_sizes = parlay::sequence<size_t>::from_function(
      num, [&](size_t i) { return children_offsets[i + 1] - children_offsets[i]; });
  auto [children_sizes_scan, total_children_sizes] = parlay::scan(children_sizes);
  children_sizes_scan.push_back(total_children_sizes);
  assert(total_children_sizes == center_offsets.);
  parlay::sequence<size_t> point_sizes = parlay::sequence<size_t>::from_function(
      num, [&](size_t i) { return point_offsets[i + 1] - point_offsets[i]; });
  parlay::sequence<node_t *> ind_to_node =
      parlay::sequence<node_t *>::from_function(num, [&](size_t i) {
        node_t *node = new node_t();
        if (children_sizes[i] > 0) {  // Internal Nodes
          size_t start_offset = children_sizes_scan[i];
          size_t end_offset = children_sizes_scan[i + 1];
          parlay::sequence<size_t> node_center_offsets =
              parlay::sequence<size_t>::from_function(children_sizes[i] + 1, [&](size_t j) {
                return center_offsets[start_offset + j] - center_offsets[start_offset];
              });

          node->set_data(PointCloudSet<ChPoint>(children_sizes[i], dim,
                                                center_values.data() + center_offsets[start_offset],
                                                node_center_offsets.data(), nullptr));
        }
        node->children.resize(children_sizes[i]);
        if (point_sizes[i] > 0) {
          parlay::sequence<size_t> point_group =
              parlay::sequence<size_t>::from_function(point_sizes[i], [&](size_t j) {
                size_t point_id = point_values[point_offsets[i] + j];
                return point_id_to_data_id[point_id];
              });
          node->set_data(PointCloudSet<ChPoint>(points.filter(point_group), dim));
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
  }
}

template<bool metric>
IndexMVIVF<metric>::~IndexMVIVF() {
  // traverse_and_delete(root);
  // delete root;
  // root = nullptr;
}

template struct IndexMVIVF<true>;   // Instantiates for L2 metric (metric = true)
template struct IndexMVIVF<false>;  // Instantiates for MIPS      (metric = false)

}  // namespace mvivf