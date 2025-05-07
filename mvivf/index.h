#pragma once

#include <set>
#include "parlay/primitives.h"
#include "utils/utils.h"
#include "mvc/mvkmeans.h"

namespace mvivf {

template <typename ChPoint, typename PointCloud>
struct index_node {
  parlay::sequence<index_node*> children; // Children
  ChPoint center; // Except root, every node has a center-set
  PointCloud points; // Only leaf nodes have points

  index_node() : children(parlay::sequence<index_node*>(0)), center(ChPoint()), 
    points(PointCloud()) {}

  inline void set_points(const PointCloud& points_) { points = points_; }
  inline void set_center(const ChPoint& center_) { center = center_; }
};

template <typename T, typename PointCloud>
struct Index {
  using ChPoint = typename PointCloud::point_type;
  using Range = typename PointCloud::range_type;
  using ind_node = index_node<ChPoint, PointCloud>;
  ind_node* root;

  Index() : root(nullptr) {}

  void Build(PointCloud& points, uint32_t maxsize=0, long s=0, long iters=5,
      std::string seeding = "Random") {
    // , std::string kmeans_dist_algo = "ANNS", 
      // std::string kmeans_seeding = "PrefixDoubling", long kmeans_iters = 10
    root = new ind_node();
    BuildHelper(root, points, maxsize, s, iters, seeding);
      // , kmeans_dist_algo, kmeans_seeding, kmeans_iters
  }

  // Recursively builds the kmeans tree
  void BuildHelper(ind_node* node, PointCloud& data, uint32_t maxsize, long s,
      long iters, std::string seeding) {
    //  , std::string kmeans_dist_algo, std::string kmeans_seeding, long kmeans_iters = 10
    if (maxsize == 0) { 
      maxsize = std::sqrt(data.size()); 
    }
    // Num Clusters = min(sqrt(n), 2n/maxsize)
    uint32_t num_clusters = std::min(2 * std::ceil(data.size() / maxsize), 
      std::ceil(std::sqrt(data.size()))); // TODO: add max num of children 
    std::cout << "Building index with " << data.size() << " points, maxsize: " 
              << maxsize << ", num_clusters: " << num_clusters << std::endl;
    PointCloud centers;
    parlay::sequence<uint32_t> cluster_ids;
    // Step 1: Run MV-Lloyd on data and collect clusters
    std::tie(centers, cluster_ids) = mvkmeans<Range>(data, num_clusters, s, 
      iters, seeding);//, kmeans_dist_algo, kmeans_seeding, kmeans_iters);
    auto id_pt = parlay::delayed_seq<std::pair<uint32_t, uint32_t>>(data.size
      (), [&](size_t i) { return std::make_pair(cluster_ids[i], i); });
    auto grouped = parlay::group_by_index(id_pt, num_clusters);
    // Step 2: Update children nodes
    auto children = parlay::sequence<ind_node*>::from_function(num_clusters,
      [&](size_t i) {
        ind_node* child = new ind_node();
        child->set_center(centers[i]);
        return child;
      });
    node->children = children;
    // Step 3: Recurse on children with large clusters // TODO: merge with step 2 again
    parlay::parallel_for(0, num_clusters, [&](size_t i) {
      auto child_data = PointCloud(data.filter(grouped[i]), data.get_dims());
      if (child_data.size() > maxsize) {
        BuildHelper(children[i], child_data, maxsize, s, iters, seeding);
        // ,kmeans_dist_algo, kmeans_seeding, kmeans_iters);
      }
      else {
        children[i]->set_points(child_data);
      }
      });
  }

  // Returns the k-NN via a beam-search-like routine
  parlay::sequence<std::pair<uint32_t, T>> Search(const ChPoint& query, uint32_t k,
    uint32_t nprobes = 1, uint32_t beam_length = 0) {
    if (beam_length == 0) { beam_length = 2 * nprobes; }
    // probe_list will contain the candidate nodes to probe
    std::set<std::pair<T, ind_node*>> probe_list;
    std::set<std::pair<T, ind_node*>> beam;

    auto add_to_probe_list = [&](ind_node* node, T dist) {
      if (probe_list.size() < nprobes || dist < probe_list.rbegin()->first) {
        probe_list.insert({ dist, node });
        if (probe_list.size() > k) {
          probe_list.erase(--probe_list.end()); // Remove the farthest probe
        }
      }
      };

    auto add_to_beam = [&](ind_node* node, T dist) {
      if (beam.size() < beam_length || dist < beam.rbegin()->first) {
        beam.insert({ dist, node });
        if (beam.size() > beam_length) {
          beam.erase(--beam.end()); // Remove the farthest node
        }
      }
      };

    // Step 1: Greedy search to find candidate probe clusters
    // Add root to beam
    add_to_beam(root, std::numeric_limits<T>::max());
    while (beam.size() > 0) {
      // Pop the best node from the beam
      auto best = *beam.begin();
      beam.erase(beam.begin());
      ind_node* current_node = best.second;

      // Compute distances from query to children
      auto res = parlay::sequence<std::pair<T, ind_node*>>::from_function(
        current_node->children.size(), [&](size_t i) {
          ind_node* child = current_node->children[i];
          T dist = query.distance(child->center);
          return std::make_pair(dist, child);
        });

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
    auto results = parlay::sequence<parlay::sequence<std::pair<uint32_t, 
      T>>>::from_function(probe_list.size(), [&](size_t i) {
        auto p = *std::next(probe_list.begin(), i);
        ind_node* node = p.second;
        auto cluster_points = node->points;
        return get_knn(query, cluster_points, k);
      });
    // Flatten, sort and return top k
    // TODO: coarse and fine distances for better performance
    auto flattened_results = parlay::flatten(results);
    parlay::sort_inplace(flattened_results, [](const auto& a, const auto& b) {
      return a.second < b.second; // Sort by distance
      });
    auto final_results = parlay::sequence<std::pair<uint32_t, T>>::from_function(std::min((size_t)k, flattened_results.size()), [&](size_t i) { 
      return flattened_results[i]; });
    return final_results;
  }

  size_t traverse_tree(ind_node* node, parlay::sequence<ind_node*>& ind_to_node,
    std::unordered_map<ind_node*, uint32_t>& node_to_ind, parlay::sequence<size_t>& center_offsets,
    parlay::sequence<size_t>& children_offsets, parlay::sequence<size_t>& point_offsets, size_t height) {
    node_to_ind[node] = ind_to_node.size();
    ind_to_node.push_back(node);
    center_offsets.push_back((node->center.size())*(node->center.get_dims()));
    children_offsets.push_back(node->children.size());
    point_offsets.push_back(node->points.size());
    size_t h = height+1;
    for (auto child : node->children) {
      h = std::max(h, traverse_tree(child, ind_to_node, node_to_ind, 
        center_offsets, children_offsets, point_offsets, height+1));
    }
    return h;
  }

  void Save(const std::string& filename) {
    std::ofstream outfile(filename, std::ios::binary);
    std::cout << "Saving index to " << filename << std::endl;
    if (!outfile.is_open()) {
      std::cerr << "Error opening file for writing: " << filename << std::endl;
      return;
    }
    // Collect data
    parlay::sequence<ind_node*> ind_to_node;
    std::unordered_map<ind_node*, uint32_t> node_to_ind;
    parlay::sequence<size_t> center_offsets;
    parlay::sequence<size_t> children_offsets;
    parlay::sequence<size_t> point_offsets;

    auto height = traverse_tree(root, ind_to_node, node_to_ind, center_offsets, children_offsets, point_offsets, 0);
    std::cout << "Height of tree: " << height << std::endl;
    
    auto total_center_sizes = parlay::scan_inplace(center_offsets);
    center_offsets.push_back(total_center_sizes);
    auto total_children_sizes = parlay::scan_inplace(children_offsets);
    children_offsets.push_back(total_children_sizes);
    auto total_point_sizes = parlay::scan_inplace(point_offsets);
    point_offsets.push_back(total_point_sizes);
    
    // Write num
    uint32_t num = ind_to_node.size();
    outfile.write(reinterpret_cast<const char*>(&num), sizeof(uint32_t));
    // Write center offsets
    outfile.write(reinterpret_cast<const char*>(center_offsets.begin()),
                  center_offsets.size() * sizeof(size_t));
    // Write center values
    for (size_t i = 0; i < num; ++i) {// TODO: make parallel
      auto node = ind_to_node[i];
      if (node->center.size() > 0) {
        auto coords = node->center.coords();
        outfile.write(reinterpret_cast<const char*>(coords.begin()),
                      coords.size() * sizeof(T));
      }
    }
    // Write children offsets
    outfile.write(reinterpret_cast<const char*>(children_offsets.begin()),
                  children_offsets.size() * sizeof(size_t));
    // Write children values
    for (size_t i = 0; i < num; ++i) {
      auto node = ind_to_node[i];
      auto children = node->children;
      for (size_t j = 0; j < children.size(); ++j) { // TODO: make parallel
        auto child = children[j];
        uint32_t child_id = node_to_ind[child];
        outfile.write(reinterpret_cast<const char*>(&child_id), sizeof(uint32_t));
      }
    }
    // Write point offsets
    outfile.write(reinterpret_cast<const char*>(point_offsets.begin()),
                  point_offsets.size() * sizeof(size_t));
    // Write point values
    for (size_t i = 0; i < num; ++i) {
      auto node = ind_to_node[i];
      auto points = node->points;
      for (size_t j = 0; j < points.size(); ++j) {
        auto point_id = points.get_id(j);
        outfile.write(reinterpret_cast<const char*>(&point_id), sizeof(uint32_t));
      }
    }
    outfile.close();
  }

  void Load(const std::string& filename, PointCloud& data) {
    std::ifstream infile(filename, std::ios::binary);
    std::cout << "Loading index from " << filename << std::endl;
    if (!infile.is_open()) {
      std::cerr << "Error opening file for reading: " << filename << std::endl;
      return;
    }
    // Read number of nodes
    uint32_t num = 0;
    infile.read(reinterpret_cast<char*>(&num), sizeof(uint32_t));
    // Read center offsets
    parlay::sequence<size_t> center_offsets(num + 1);
    infile.read(reinterpret_cast<char*>(center_offsets.begin()),
                center_offsets.size() * sizeof(size_t));
    // Read center values
    parlay::sequence<T> center_values(center_offsets[center_offsets.size() - 1]);
    infile.read(reinterpret_cast<char*>(center_values.begin()),
                center_values.size() * sizeof(T));
    // Read children offsets
    parlay::sequence<size_t> children_offsets(num + 1);
    infile.read(reinterpret_cast<char*>(children_offsets.begin()),
                children_offsets.size() * sizeof(size_t));
    // Read children values
    parlay::sequence<uint32_t> children_values(children_offsets[children_offsets.size() - 1]);
    infile.read(reinterpret_cast<char*>(children_values.begin()),
                children_values.size() * sizeof(uint32_t));
    // Read point offsets
    parlay::sequence<size_t> point_offsets(num + 1);
    infile.read(reinterpret_cast<char*>(point_offsets.begin()),
                point_offsets.size() * sizeof(size_t));
    // Read point values
    parlay::sequence<uint32_t> point_values(point_offsets[point_offsets.size() - 1]);
    infile.read(reinterpret_cast<char*>(point_values.begin()),
                point_values.size() * sizeof(uint32_t));

    // Build the index 
    size_t dim = data.get_dims();
    auto point_id_to_data_id = parlay::sequence<uint32_t>::uninitialized(data.size());
    parlay::parallel_for(0, data.size(), [&](size_t i) {
      point_id_to_data_id[data.get_id(i)] = i;
    });
    auto center_sizes = parlay::sequence<size_t>::from_function(num, 
      [&](size_t i) { return center_offsets[i + 1] - center_offsets[i]; });
    auto children_sizes = parlay::sequence<size_t>::from_function(num, 
      [&](size_t i) { return children_offsets[i + 1] - children_offsets[i]; });
    auto point_sizes = parlay::sequence<size_t>::from_function(num, 
      [&](size_t i) { return point_offsets[i + 1] - point_offsets[i]; });
    auto ind_to_node = parlay::sequence<ind_node*>::from_function(num, 
      [&](size_t i) { 
      auto node = new ind_node();
      if (center_sizes[i] > 0) {
        node->set_center(ChPoint(center_values.begin() + center_offsets[i], 
          center_sizes[i]/dim, dim)); 
      }
      node->children.resize(children_sizes[i]);
      if (point_sizes[i] > 0){
        auto point_group = parlay::sequence<uint32_t>::from_function(
          point_sizes[i], [&](size_t j) {
            auto point_id = point_values[point_offsets[i] + j];
            return point_id_to_data_id[point_id]; 
          });
        node->set_points(PointCloud(data.filter(point_group), dim));
      }
      return node;
    });
    
    // Set children pointers
    parlay::parallel_for(0, num, [&](size_t i) {
      auto node = ind_to_node[i];
      auto& children = node->children;
      parlay::parallel_for(0, children.size(), [&](size_t j) {
        auto child_id = children_values[children_offsets[i] + j];
        children[j] = ind_to_node[child_id];
      });
    });
    root = ind_to_node[0];
    infile.close();
  }
};

} // namespace mvivf