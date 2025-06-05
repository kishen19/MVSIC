#pragma once

#include <set>
// #include "absl/container/btree_set.h"
#include "parlay/primitives.h"
#include "mvc/utils/euclidian_point.h"
#include "mvc/utils/faiss_kmeans.h"
#include "mvc/utils/mips_point.h"
#include "mvc/utils/point_range.h"
#include "mvivf/index.h"
#include "mvivf/utils/top_neighbors.h"


namespace mvivf {
using namespace mvivf;

struct IndexSVHParams{
  size_t minsize = 100;
  size_t maxsize = 500;
  bool verbose = false;
  size_t os_rate = 20;
};

template <typename Point, typename Range>
struct IndexSVHNode {
  parlay::sequence<IndexSVHNode*> children; // Children
  Point center; // Except root, every node has a center-set
  Range points; // Only leaf nodes have points
  parlay::sequence<std::pair<size_t,size_t>> ids; // Only leaf nodes have ids
  IndexSVHNode() : children(parlay::sequence<IndexSVHNode*>(0)), center(Point()), 
    points(Range()) {}
  inline void set_points(const Range& points_, 
    const parlay::sequence<std::pair<size_t,size_t>>& ids_) { 
      points = points_; 
      ids = ids_;
    }
  inline void set_center(const Point& center_) { center = center_; }
};

template <bool metric>
struct IndexSVH :Index<metric>, IndexSVHParams {
  using ChPoint = Index<metric>::ChPoint;
  using Point = std::conditional_t<metric, Euclidian_Point<float>, Mips_Point<float>>;
  using Range = PointRange<float, Point>;
  using node_t = IndexSVHNode<Point, Range>;
  using Index<metric>::d;

  node_t* root = nullptr;

  IndexSVH(size_t d_) noexcept { d = d_; }
  IndexSVH(size_t d_, const IndexSVHParams& params) noexcept 
    : IndexSVHParams(params) { d = d_; }

  // Builds the index given a point cloud set.
  void build(const PointCloudSet<ChPoint>& points) override;
  // Recursively builds the kmeans tree
  void build_helper(node_t* node, 
    const parlay::sequence<parlay::sequence<float>>& points,
    const parlay::sequence<std::pair<size_t,size_t>>& ids);
  // 
  std::pair<parlay::sequence<std::pair<size_t, float>>, size_t> search_each(
    const Point& query, const PointCloudSet<ChPoint>& points,
    const SearchParams& params);
  // Returns the top-k point clouds for the query point cloud
  // Output format: < [<id, distance>, ...], # distance comparisons>
  std::pair<parlay::sequence<std::pair<size_t, float>>, size_t> search(
    const ChPoint& query, const PointCloudSet<ChPoint>& points,
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

template <bool metric>
void IndexSVH<metric>::build(const PointCloudSet<ChPoint>& points) {
  size_t n = points.size();
  auto iota = parlay::iota(n);
  parlay::sequence<parlay::sequence<float>> data = points.filter_flattened(iota);
  auto ids = parlay::sequence<std::pair<size_t,size_t>>::uninitialized(points.total_size());
  auto num_embs = parlay::delayed_tabulate(n, [&](size_t i){
    return points.get_size(i);
  });
  auto [offsets, _] = parlay::scan(num_embs);
  parlay::parallel_for(0, n, [&](size_t i){
    for (size_t j = 0; j < num_embs[i]; ++j) {
      ids[offsets[i] + j] = {i,j};
    }
  });
  root = new node_t();
  build_helper(root, data, ids);
}

// Recursively builds the kmeans tree
template <bool metric>
void IndexSVH<metric>::build_helper(node_t* node, 
    const parlay::sequence<parlay::sequence<float>>& points, 
    const parlay::sequence<std::pair<size_t,size_t>>& ids) {
  size_t n = points.size();
  size_t mp = std::max((size_t)4, maxsize/minsize);
  // Num Clusters = min(sqrt(n), mp*n/maxsize)
  size_t num_clusters = std::min(mp * std::ceil(n/maxsize), 
    std::ceil(std::sqrt(n)));
  if (verbose){
    std::cout << "Building index with " << n << " points, maxsize: " 
              << maxsize << ", num_clusters: " << num_clusters << std::endl;
  }
  // Step 1: Run k-means on data and collect clusters
  auto [centers_, cluster_ids, active_indices] = faiss_kmeans_assign(points, d, 
    num_clusters, metric, maxsize, os_rate);
  Range centers = Range(centers_, d);
  auto id_pt = parlay::delayed_seq<std::pair<size_t, size_t>>(n, [&](size_t i) { 
    return std::make_pair(cluster_ids[i], i); });
  auto grouped = parlay::group_by_index(id_pt, num_clusters);
  // Step 2: Update children nodes and recurse for large nodes
  parlay::sequence<node_t*> children = parlay::sequence<node_t*>::from_function(active_indices.size(),
    [&](size_t id) {
      size_t i = active_indices[id];
      node_t* child = new node_t(); // TODO: use parlay allocator
      child->set_center(centers[i]);
      auto child_points = parlay::tabulate(grouped[i].size(), [&](size_t j) {
        return points[grouped[i][j]];});
      auto child_ids = parlay::tabulate(grouped[i].size(),
        [&](size_t j) { return ids[grouped[i][j]]; });
      if (child_points.size() > maxsize) {
        build_helper(child, child_points, child_ids);
      } else {
        auto child_range = Range(child_points, d);
        child->set_points(child_range, child_ids);
      }
      return child;
    });
  node->children = children;
}

template <bool metric>
std::pair<parlay::sequence<std::pair<size_t, float>>, size_t> IndexSVH<metric>::search(
    const ChPoint& query, const PointCloudSet<ChPoint>& points, 
    const SearchParams& params) {
  size_t q = query.size();
  size_t k = params.k;
  size_t cands = params.cands;
  auto results = parlay::sequence<parlay::sequence<std::pair<size_t, float>>>(q);
  auto dist_cmps = parlay::sequence<size_t>::uninitialized(q);
  parlay::parallel_for(0, q, [&](size_t i) {
    std::tie(results[i], dist_cmps[i]) = search_each(Point(query.get_coords(i), 
      query.get_dims(), query.get_dims()), points, params);
  });
  parlay::sequence<size_t> candidates;
  std::unordered_set<size_t> seen;
  size_t done = 0, i=0;
  while (done < q){
    for (size_t j=0; j<q; j++){
      if (i < results[j].size()){
        if (seen.find(results[j][i].first) == seen.end()){
          candidates.push_back(results[j][i].first);
          seen.insert(results[j][i].first);
        }
      }
      if (i == results[j].size()-1 || (i==0 && results[j].size()==0)){
        done++;
      }
    }
    i++;
  }
  // Final re-ranking
  auto new_cands = parlay::sequence<std::pair<float, size_t>>::from_function( 
      std::min(cands, candidates.size()), [&](size_t i) {
    size_t id = candidates[i];
    float new_dist = query.distance(points[id]);
    return std::make_pair(new_dist, id);
  });
  parlay::sort_inplace(new_cands);
  parlay::sequence<std::pair<size_t, float>> final_results;
  final_results.push_back({new_cands[0].second, new_cands[0].first});
  auto [prev_id, prev_dist] = final_results[0];
  for (size_t i = 1; i < new_cands.size(); i++) {
    if (final_results.size() < k) {
      auto [dist, id] = new_cands[i];
      if (id != prev_id) {
        final_results.push_back({id, dist});
        prev_dist = dist;
        prev_id = id;
      }
    }
    if (final_results.size() == k) {
      break;
    }
  }
  return std::make_pair(final_results, parlay::reduce(dist_cmps) + new_cands.size());
}

template <bool metric>
std::pair<parlay::sequence<std::pair<size_t, float>>, size_t> IndexSVH<metric>::search_each(
    const Point& query, const PointCloudSet<ChPoint>& points,
    const SearchParams& params) {
  // size_t k = params.k;
  size_t cands = params.cands;
  size_t nprobes = params.nprobes;
  size_t beam_length = params.beam_length;
  if (beam_length == 0) { beam_length = nprobes; } // Default
  // TODO: have separete nprobes for indiv points and reranking
  // probe_list will contain the candidate nodes to probe
  parlay::sequence<std::pair<float, node_t*>> probe_list;
  std::set<std::pair<float, node_t*>> beam;
  // absl::btree_set<std::pair<float, node_t*>> probe_list;
  // absl::btree_set<std::pair<float, node_t*>> beam;
  size_t dist_cmps=0;
  auto add_to_probe_list = [&](std::pair<float, node_t*> p) {
    probe_list.push_back(p);
  };
  auto add_to_beam = [&](std::pair<float, node_t*> p) {
    if (beam.size() < beam_length || p.first < beam.rbegin()->first) {
      beam.insert(p);
      if (beam.size() > beam_length) {
        beam.erase(--beam.end()); // Remove the farthest node
      }
    }
  };
  // Step 1: Greedy search to find candidate probe clusters
  // Add root to beam
  add_to_beam({std::numeric_limits<float>::max(), root});
  while (beam.size() > 0) {
    // Pop the best node from the beam
    auto best = *beam.begin();
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
    // Add new nodes to beam and probe list
    for (size_t i = 0; i < std::min(beam_length, new_nodes_to_beam.size()); i++) {
      add_to_beam(new_nodes_to_beam[i]);
    }
    for (size_t i = 0; i < std::min(nprobes, new_nodes_to_probe.size()); i++) {
      add_to_probe_list(new_nodes_to_probe[i]);
    }
  }
  // Step 2: Probe clusters in probe_list
  parlay::sort_inplace(probe_list);
  nprobes = std::min(nprobes, probe_list.size());
  auto probe_dist_cmps = parlay::sequence<size_t>::uninitialized(nprobes);
  auto results = parlay::sequence<parlay::sequence<std::pair<size_t, 
    float>>>::from_function(nprobes, [&](size_t i) {
      std::pair<float, node_t*> p = *std::next(probe_list.begin(), i);
      node_t* node = p.second;
      Range cluster_points = node->points;
      auto [res, d_c] = mvivf::get_knn_ids(query, cluster_points, node->ids, cands);
      probe_dist_cmps[i] = d_c;
      return res;
    });
  dist_cmps += parlay::reduce(probe_dist_cmps);
  // Step 3: Re-ranking
  auto flattened_results = parlay::flatten(results);
  parlay::sort_inplace(flattened_results); // Sort by ids and then distance
  // Keep only the first copy of same id elements
  auto cutoff_indices = parlay::delayed_seq<size_t>(flattened_results.size(), 
      [&](size_t i) {
    return i == 0 || flattened_results[i].first != flattened_results[i - 1].first;
  });
  auto indices = parlay::pack_index(cutoff_indices);
  auto filtered_results = parlay::tabulate(indices.size(), [&](size_t i){
    return flattened_results[indices[i]];
  });
  parlay::sort_inplace(filtered_results, [](const auto& a, const auto& b) {
    return a.second < b.second; // Sort by distance
  });
  auto final_results = parlay::sequence<std::pair<size_t, float>>::from_function(
    std::min(cands, filtered_results.size()), [&](size_t i) { 
      return filtered_results[i]; });
  return std::make_pair(final_results, dist_cmps);
}

template <bool metric>
size_t IndexSVH<metric>::traverse_tree(node_t* node, 
  parlay::sequence<node_t*>& ind_to_node,
  std::unordered_map<node_t*, size_t>& node_to_ind, 
  parlay::sequence<size_t>& center_offsets,
  parlay::sequence<size_t>& children_offsets, 
  parlay::sequence<size_t>& point_offsets, size_t height) {
  node_to_ind[node] = ind_to_node.size();
  ind_to_node.push_back(node);
  center_offsets.push_back(node->center.get_dims()); // 0 if empty center
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
void IndexSVH<metric>::save(const std::string& filename) {
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
    if (node->center.get_dims() > 0) {
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
  // Write point values (pairs)
  for (size_t i = 0; i < num; ++i) {
    node_t* node = ind_to_node[i];
    auto point_ids = node->ids;
    outfile.write(reinterpret_cast<const char*>(point_ids.begin()),
                  point_ids.size() * sizeof(std::pair<size_t,size_t>));
  }
  outfile.close();
}

template <bool metric>
void IndexSVH<metric>::load(const std::string& filename, 
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
  parlay::sequence<std::pair<size_t,size_t>> point_values(point_offsets[point_offsets.size() - 1]);
  infile.read(reinterpret_cast<char*>(point_values.begin()),
              point_values.size() * sizeof(std::pair<size_t, size_t>));

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
      node->set_center(Point(center_values.begin() + center_offsets[i], 
        dim, dim));
    }
    node->children.resize(children_sizes[i]);
    if (point_sizes[i] > 0){
      auto point_ids = parlay::tabulate(point_sizes[i], [&](size_t j) {
        return point_values[point_offsets[i] + j];
      });
      auto point_group = parlay::tabulate(point_sizes[i], [&](size_t j) {
        auto [point_id, emb_id] = point_ids[j];
        auto act_point_id = point_id_to_data_id[point_id];
        return points[act_point_id][emb_id];
      });
      node->set_points(Range(point_group, dim), point_ids);
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

template struct IndexSVH<true>;  // Instantiates for L2 metric (metric = true)
template struct IndexSVH<false>; // Instantiates for MIPS      (metric = false)

} // namespace mvivf