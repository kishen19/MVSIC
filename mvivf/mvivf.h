#pragma once

#include <set>
#include "mvc/mvkmeans.h"
#include "utils/top_neighbors.h"
#include "index.h"

namespace mvivf {

struct IndexMVIVFParams{
  size_t minsize = 100;
  size_t maxsize = 500;
  size_t s = 0;
  size_t iters = 5;
  std::string seeding = "Random";
  bool verbose = false;
};

template <typename ChPoint>
struct IndexMVIVFNode {
  parlay::sequence<IndexMVIVFNode*> children; // Children
  ChPoint center; // Except root, every node has a center-set
  PointCloudSet<ChPoint> points; // Only leaf nodes have points

  IndexMVIVFNode() : children(parlay::sequence<IndexMVIVFNode*>(0)), center(ChPoint()), 
    points(PointCloudSet<ChPoint>()) {}

  inline void set_center(const ChPoint& center_) { center = center_; } // TODO: check this
  inline void set_points(const PointCloudSet<ChPoint>& points_) { points = points_; } // TODO: check this
};

template <bool metric>
struct IndexMVIVF : Index<metric>, IndexMVIVFParams{
  using ChPoint = Index<metric>::ChPoint;
  using node_t = IndexMVIVFNode<ChPoint>;
  using Index<metric>::d;

  node_t* root = nullptr;

  IndexMVIVF(size_t d_) noexcept { d = d_; }
  IndexMVIVF(size_t d_, const IndexMVIVFParams& params) noexcept 
    : IndexMVIVFParams(params) { d = d_; }

  // Builds the index given a point cloud set.
  void build(const PointCloudSet<ChPoint>& points) override {
    root = new node_t(); // Parlay::allocator
    build_helper(root, points);
  }
  // Recursively builds the kmeans tree
  void build_helper(node_t* node, const PointCloudSet<ChPoint>& points);
  // Returns the top-k point clouds for the query point cloud
  // Output format: < [<id, distance>, ...], # distance comparisons>
  std::pair<parlay::sequence<std::pair<size_t, float>>, size_t> search(
    const ChPoint& query, const PointCloudSet<ChPoint>& points, size_t k, 
    const SearchParams& params) override;
  // Traversing the k-means tree: returns the height of the tree
  // TODO: get more stats about the tree
  size_t traverse_tree(node_t* node, parlay::sequence<node_t*>& ind_to_node,
    std::unordered_map<node_t*, size_t>& node_to_ind, parlay::sequence<size_t>& center_offsets,
    parlay::sequence<size_t>& children_offsets, parlay::sequence<size_t>& point_offsets, 
    size_t height);
  // Write the index to a file in disk
  void save(const std::string& filename) override;
  // Read the index from a file in disk
  void load(const std::string& filename, const PointCloudSet<ChPoint>& points) override;
};

// Recursively builds the kmeans tree
template <bool metric>
void IndexMVIVF<metric>::build_helper(node_t* node, const PointCloudSet<ChPoint>& points) {
  size_t n = points.size();
  size_t mp = std::max((size_t)4, maxsize/minsize);
  // Num Clusters = min(sqrt(n), mp*n/maxsize)
  size_t num_clusters = std::min(mp * std::ceil(n/maxsize), 
    std::ceil(std::sqrt(n)));
  if (verbose){
    std::cout << "Building index with " << n << " points, maxsize: " 
              << maxsize << ", num_clusters: " << num_clusters << std::endl;
  }
  // Step 1: Run MV-Lloyds on points and collect clusters
  MVClusteringParams params(iters, seeding, false, verbose);
  MVClustering<metric> clus(d, num_clusters, s, params);
  clus.train(points);
  PointCloudSet<ChPoint> centers = clus.centers;
  parlay::sequence<size_t> cluster_ids = clus.cluster_ids;
  auto id_pt = parlay::delayed_seq<std::pair<size_t, size_t>>(n, [&](size_t i) { 
    return std::make_pair(cluster_ids[i], i); });
  auto grouped = parlay::group_by_index(id_pt, num_clusters);
  // Step 2: Update children nodes and recurse for large nodes
  parlay::sequence<node_t*> children = parlay::sequence<node_t*>::from_function(num_clusters,
    [&](size_t i) {
      node_t* child = new node_t(); // TODO: use parlay allocator
      child->set_center(centers[i]);
      PointCloudSet<ChPoint> child_points = PointCloudSet<ChPoint>(
        points.filter(grouped[i]), d);
      std::cout << "Child size: " << child_points.size() << std::endl;
      if (child_points.size() > maxsize) {
        build_helper(child, child_points);
      } else {
        child->set_points(child_points);
      }
      return child;
    }, 10000);
  node->children = children;
}

// Returns the top-k point clouds for the query point cloud
// Output format: < [<id, distance>, ...], # distance comparisons>
template <bool metric>
std::pair<parlay::sequence<std::pair<size_t, float>>, size_t> IndexMVIVF<metric>::search(
    const ChPoint& query, const PointCloudSet<ChPoint>& points, size_t k, 
    const SearchParams& params) {
  size_t nprobes = params.nprobes;
  size_t beam_length = params.beam_length;
  if (beam_length == 0) { beam_length = 2 * nprobes; } // Default
  // probe_list contains the final candidate leaf nodes to probe
  std::set<std::pair<float, node_t*>> probe_list; // Change to absl
  std::set<std::pair<float, node_t*>> beam; // Change to absl
  size_t dist_cmps=0;
  auto add_to_probe_list = [&](node_t* node, float dist) {
    if (probe_list.size() < nprobes || dist < probe_list.rbegin()->first) {
      probe_list.insert({ dist, node });
      if (probe_list.size() > k) {
        probe_list.erase(--probe_list.end()); // Remove the farthest probe
      }
    }
  };
  auto add_to_beam = [&](node_t* node, float dist) {
    if (beam.size() < beam_length || dist < beam.rbegin()->first) {
      beam.insert({ dist, node });
      if (beam.size() > beam_length) {
        beam.erase(--beam.end()); // Remove the farthest node
      }
    }
  };
  // Step 1: Greedy search to find candidate probe clusters
  // Add root to beam
  add_to_beam(root, std::numeric_limits<float>::max());
  while (beam.size() > 0) {
    // Pop the best node from the beam
    std::pair<float, node_t*> best = *beam.begin();
    beam.erase(beam.begin());
    node_t* current_node = best.second;

    // Compute distances from query to children
    auto res = parlay::sequence<std::pair<float, node_t*>>::from_function(
      current_node->children.size(), [&](size_t i) {
        node_t* child = current_node->children[i];
        float dist = query.distance(child->center);
        return std::make_pair(dist, child);
      });
    dist_cmps += res.size();

    // Collect leaf and non-leaf nodes
    auto new_nodes_to_beam = parlay::filter(res, [](const auto& p) {
      return p.second->children.size() != 0; // Only keep nodes that are not leaves
    });
    auto new_nodes_to_probe = parlay::filter(res, [](const auto& p) {
      return p.second->children.size() == 0; // Only keep leaf nodes
    });
    parlay::sort_inplace(new_nodes_to_beam, [](const auto& a, const auto& b) {
      return a.first < b.first; // Sort by distance
    });
    parlay::sort_inplace(new_nodes_to_probe, [](const auto& a, const auto& b) {
      return a.first < b.first; // Sort by distance
    });

    // Add new nodes to beam and probe list
    for (size_t i = 0; i < std::min((size_t)beam_length, new_nodes_to_beam.size()); i++) {
      add_to_beam(new_nodes_to_beam[i].second, new_nodes_to_beam[i].first);
    }
    for (size_t i = 0; i < std::min((size_t)nprobes, new_nodes_to_probe.size()); i++) {
      add_to_probe_list(new_nodes_to_probe[i].second, new_nodes_to_probe[i].first);
    }
  }
  // Step 2: Probe clusters in probe_list
  auto probe_dist_cmps = parlay::sequence<size_t>::uninitialized(probe_list.size());
  auto results = parlay::sequence<parlay::sequence<std::pair<size_t, float>>>::from_function(
      probe_list.size(), [&](size_t i) {
    std::pair<float, node_t*> p = *std::next(probe_list.begin(), i);
    node_t* node = p.second;
    PointCloudSet<ChPoint> cluster_points = node->points;
    auto [res, d_c] = get_knn(query, cluster_points, k);
    probe_dist_cmps[i] = d_c;
    return res;
  });
  dist_cmps += parlay::reduce(probe_dist_cmps);
  // Step 3: Re-ranking (lightweight; no new distance cmps)
  // Flatten, sort and return top k // TODO: coarse and fine distances for better performance
  auto flattened_results = parlay::flatten(results);
  parlay::sort_inplace(flattened_results, [](const auto& a, const auto& b) {
    return a.second < b.second; // Sort by distance
  });
  auto final_results = parlay::sequence<std::pair<size_t, float>>::from_function(
      std::min((size_t)k, flattened_results.size()), [&](size_t i) { 
    return flattened_results[i]; });
  return std::make_pair(final_results, dist_cmps);
}

template <bool metric>
size_t IndexMVIVF<metric>::traverse_tree(node_t* node, 
    parlay::sequence<node_t*>& ind_to_node,
    std::unordered_map<node_t*, size_t>& node_to_ind, 
    parlay::sequence<size_t>& center_offsets,
    parlay::sequence<size_t>& children_offsets, 
    parlay::sequence<size_t>& point_offsets, size_t height) {
  node_to_ind[node] = ind_to_node.size();
  ind_to_node.push_back(node);
  center_offsets.push_back((node->center.size())*(node->center.get_dims()));
  children_offsets.push_back(node->children.size());
  point_offsets.push_back(node->points.size());
  size_t h = height+1;
  for (node_t* child : node->children) {
    h = std::max(h, traverse_tree(child, ind_to_node, node_to_ind, 
      center_offsets, children_offsets, point_offsets, height+1));
  }
  return h;
}

template <bool metric>
void IndexMVIVF<metric>::save(const std::string& filename) {
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

  size_t height = traverse_tree(root, ind_to_node, node_to_ind, center_offsets, 
    children_offsets, point_offsets, 0);
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
  outfile.write(reinterpret_cast<const char*>(center_offsets.begin()),
                center_offsets.size() * sizeof(size_t));
  // Write center values
  for (size_t i = 0; i < num; ++i) {// TODO: make parallel
    node_t* node = ind_to_node[i];
    if (node->center.size() > 0) {
      auto coords = node->center.get_slice();
      outfile.write(reinterpret_cast<const char*>(coords.begin()),
                    coords.size() * sizeof(float));
    }
  }
  // Write children offsets
  outfile.write(reinterpret_cast<const char*>(children_offsets.begin()),
                children_offsets.size() * sizeof(size_t));
  // Write children values
  for (size_t i = 0; i < num; ++i) {
    node_t* node = ind_to_node[i];
    parlay::sequence<node_t*> children = node->children;
    for (size_t j = 0; j < children.size(); ++j) { // TODO: make parallel
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
    PointCloudSet<ChPoint> points = node->points;
    for (size_t j = 0; j < points.size(); ++j) {
      size_t point_id = points.get_id(j);
      outfile.write(reinterpret_cast<const char*>(&point_id), sizeof(size_t));
    }
  }
  outfile.close();
}

template <bool metric>
void IndexMVIVF<metric>::load(const std::string& filename, 
    const PointCloudSet<ChPoint>& points) {
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
  parlay::sequence<size_t> center_offsets(num + 1);
  infile.read(reinterpret_cast<char*>(center_offsets.begin()),
              center_offsets.size() * sizeof(size_t));
  // Read center values
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
  parlay::sequence<size_t> point_values(point_offsets[point_offsets.size() - 1]);
  infile.read(reinterpret_cast<char*>(point_values.begin()),
              point_values.size() * sizeof(size_t));

  // Build the index 
  size_t dim = points.get_dims();
  auto point_id_to_data_id = parlay::sequence<size_t>::uninitialized(points.size());
  parlay::parallel_for(0, points.size(), [&](size_t i) {
    point_id_to_data_id[points.get_id(i)] = i;
  });
  parlay::sequence<size_t> center_sizes = parlay::sequence<size_t>::from_function(num, 
    [&](size_t i) { return center_offsets[i + 1] - center_offsets[i]; });
  parlay::sequence<size_t> children_sizes = parlay::sequence<size_t>::from_function(num, 
    [&](size_t i) { return children_offsets[i + 1] - children_offsets[i]; });
  parlay::sequence<size_t> point_sizes = parlay::sequence<size_t>::from_function(num, 
    [&](size_t i) { return point_offsets[i + 1] - point_offsets[i]; });
  parlay::sequence<node_t*> ind_to_node = parlay::sequence<node_t*>::from_function(num, 
    [&](size_t i) { 
    node_t* node = new node_t(); // TODO: use paralay allocator
    if (center_sizes[i] > 0) {
      node->set_center(ChPoint(center_sizes[i]/dim, dim, 
        center_values.begin() + center_offsets[i])); 
    }
    node->children.resize(children_sizes[i]);
    if (point_sizes[i] > 0){
      parlay::sequence<size_t> point_group = parlay::sequence<size_t>::from_function(
        point_sizes[i], [&](size_t j) {
          size_t point_id = point_values[point_offsets[i] + j];
          return point_id_to_data_id[point_id]; 
        });
      node->set_points(PointCloudSet<ChPoint>(points.filter(point_group), dim));
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
  infile.close();
}

template struct IndexMVIVF<true>;  // Instantiates for L2 metric (metric = true)
template struct IndexMVIVF<false>; // Instantiates for MIPS      (metric = false)

} // namespace mvivf