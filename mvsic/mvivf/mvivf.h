#pragma once

#include <queue>
#include <set>

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
  using Index<metric>::d;                           // Embedding dimension

  struct node_t {
    parlay::sequence<node_t *> children;
    // For internal nodes: data = centers of children
    // For leaves:         data = points in the cluster
    PointCloudSet<ChPoint> data;
    QuantizedPointCloudSet<ChPoint> pq_data;  // Only for leaves if PQ is enabled
    node_t() noexcept :
        children(parlay::sequence<node_t *>(0)),
        data(PointCloudSet<ChPoint>()),
        pq_data(QuantizedPointCloudSet<ChPoint>()) {}
    ~node_t() noexcept {}

    inline size_t get_size() const noexcept {
      if (pq_data.size() > 0)
        return pq_data.size();
      else
        return data.size();
    }
  };

  IndexParams params;
  node_t *root = nullptr;  // Root of the k-means tree

  IndexMVIVF(size_t d_) noexcept : params(IndexParams::mvivf()) { d = d_; }
  IndexMVIVF(size_t d_, const IndexParams &params) noexcept : params(params) { d = d_; }

  // Recursive kmeans tree builder
  void recursive_build(node_t *node, const PointCloudSet<ChPoint> &points) {
    size_t n = points.size();
    size_t num_clusters = (params.k_per_level > 0) ? params.k_per_level : std::ceil(std::sqrt(n));
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
            if (params.pq.enabled) {
              child->pq_data = QuantizedPointCloudSet<ChPoint>(
                  child_points, params.pq.num_blocks,
                  std::min(static_cast<size_t>(params.pq.num_clusters_per_block),
                           child_points.total_size()),
                  params.pq.sample_size);
            } else {
              child->data = std::move(child_points);
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
    recursive_build(root, points);
  }

  std::tuple<parlay::sequence<std::pair<uint32_t, float>>, size_t, std::vector<double>>
  search_with_stats(const ChPoint &query, const PointCloudSet<ChPoint> &points,
                    const SearchParams &search_params) override {
    parlay::internal::timer t;
    std::vector<double> timings;

    size_t k = search_params.k;
    size_t nprobes = search_params.nprobes;
    size_t beam_length = 2 * search_params.nprobes;
    size_t dist_cmps = 0;
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
    t.start();
    add_to_beam({std::numeric_limits<float>::max(), root});
    while (beam.size() > 0) {
      // Pop the best node from the beam
      std::pair<float, node_t *> best = *beam.begin();
      beam.erase(beam.begin());
      node_t *current_node = best.second;
      // Compute distances from query to children
      auto &children = current_node->children;
      auto &centers = current_node->data;
      // Note: children.size() == centers.size()
      parlay::sequence<std::pair<uint32_t, float>> id_dist;
      size_t dist_cmps_node;
      std::tie(id_dist, dist_cmps_node) = centers.distances(query);
      dist_cmps += dist_cmps_node;
      auto res = parlay::sequence<std::pair<float, node_t *>>::from_function(
          id_dist.size(), [&](size_t i) { return std::make_pair(id_dist[i].second, children[i]); });
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
    parlay::sort_inplace(probe_list);
    timings.push_back(t.stop());
    t.reset();

    // Step 2: Probe clusters in probe_list
    t.start();
    // Find the minimum number of probes needed to obtain k neighbors
    size_t nprobes_minimal = 0, cur = 0;
    while (nprobes_minimal < probe_list.size() && cur <= k) {
      cur += probe_list[nprobes_minimal].second->get_size();
      nprobes_minimal++;
    }
    nprobes = std::min(probe_list.size(), std::max(nprobes, nprobes_minimal));
    auto sizes = parlay::delayed_tabulate(
        nprobes, [&](size_t i) { return probe_list[i].second->get_size(); });
    auto scan_result = parlay::scan(sizes);
    auto &offsets = scan_result.first;
    size_t total_size = scan_result.second;
    auto visited = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(total_size);
    auto leaf_dist_cmps = parlay::sequence<size_t>::uninitialized(nprobes);
    parlay::parallel_for(0, nprobes, [&](size_t i) {
      node_t *leaf_node = probe_list[i].second;
      if (params.pq.enabled) {
        leaf_dist_cmps[i] = leaf_node->pq_data.distances(query, &visited[offsets[i]]);
      } else {
        leaf_dist_cmps[i] = leaf_node->data.distances(query, &visited[offsets[i]]);
      }
    });
    dist_cmps += parlay::reduce(leaf_dist_cmps);
    parlay::sort_inplace(visited, [](const auto &a, const auto &b) {
      return a.second < b.second;  // Sort by distance
    });
    timings.push_back(t.stop());
    t.reset();

    // Step 3: Re-ranking
    t.start();
    auto final_results =
        parlay::sequence<std::pair<uint32_t, float>>::uninitialized(std::min(k, visited.size()));
    if (search_params.num_rerank > 0) {
      size_t num_rerank = std::min(search_params.num_rerank, visited.size());
      auto cmp_rerank = parlay::sequence<size_t>::uninitialized(num_rerank);
      auto results_rerank =
          parlay::sequence<std::pair<uint32_t, float>>::from_function(num_rerank, [&](size_t i) {
            uint32_t id = visited[i].first;
            auto [dist, d_c] = query.distance_w_cmps(points[id]);
            cmp_rerank[i] = d_c;
            return std::make_pair(id, dist);
          });
      dist_cmps += parlay::reduce(cmp_rerank);
      parlay::sort_inplace(results_rerank, [](const auto &a, const auto &b) {
        return a.second < b.second;  // Sort by distance
      });
      parlay::parallel_for(0, final_results.size(),
                           [&](size_t i) { final_results[i] = results_rerank[i]; });
    } else {
      parlay::parallel_for(0, final_results.size(),
                           [&](size_t i) { final_results[i] = visited[i]; });
    }
    timings.push_back(t.stop());
    t.reset();

    return std::make_tuple(final_results, dist_cmps, timings);
  }

  // Traversing the k-means tree: returns the height of the tree
  // TODO: get more stats about the tree
  size_t traverse_tree(node_t *node, parlay::sequence<node_t *> &ind_to_node,
                       std::unordered_map<node_t *, size_t> &node_to_ind,
                       parlay::sequence<size_t> &center_offsets,
                       parlay::sequence<size_t> &children_offsets,
                       parlay::sequence<size_t> &point_offsets,
                       parlay::sequence<size_t> &pq_offsets, size_t height) {
    node_to_ind[node] = ind_to_node.size();
    ind_to_node.push_back(node);
    if (node->children.size() == 0) {  // leaves
      if (params.pq.enabled) {
        point_offsets.push_back(0);
        pq_offsets.push_back(node->pq_data.size());
      } else {
        point_offsets.push_back(node->data.size());
        pq_offsets.push_back(0);
      }
    } else {  // Internal nodes
      point_offsets.push_back(0);
      pq_offsets.push_back(0);
      size_t dims = node->data.get_dims();
      for (size_t i = 0; i < node->children.size(); i++) {
        center_offsets.push_back(node->data.get_size(i) * dims);  // # embeddings in center[i]
      }
    }
    children_offsets.push_back(node->children.size());
    size_t h = height + 1;
    for (node_t *child : node->children) {
      h = std::max(h, traverse_tree(child, ind_to_node, node_to_ind, center_offsets,
                                    children_offsets, point_offsets, pq_offsets, height + 1));
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

    outfile.write(reinterpret_cast<const char *>(&params.pq.enabled), sizeof(params.pq.enabled));

    // Collect data
    parlay::sequence<node_t *> ind_to_node;
    std::unordered_map<node_t *, size_t> node_to_ind;
    parlay::sequence<size_t> center_offsets;
    parlay::sequence<size_t> children_offsets;
    parlay::sequence<size_t> point_offsets;
    parlay::sequence<size_t> pq_offsets;

    size_t height = traverse_tree(root, ind_to_node, node_to_ind, center_offsets, children_offsets,
                                  point_offsets, pq_offsets, 0);
    std::cout << "Height of tree: " << height << std::endl;

    size_t total_center_sizes = parlay::scan_inplace(center_offsets);
    center_offsets.push_back(total_center_sizes);
    size_t total_children_sizes = parlay::scan_inplace(children_offsets);
    children_offsets.push_back(total_children_sizes);
    size_t total_point_sizes = parlay::scan_inplace(point_offsets);
    point_offsets.push_back(total_point_sizes);
    size_t total_pq_sizes = parlay::scan_inplace(pq_offsets);
    pq_offsets.push_back(total_pq_sizes);

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
      if (node->children.size() == 0 && !params.pq.enabled) {  // leaves only
        PointCloudSet<ChPoint> points = node->data;
        for (size_t j = 0; j < points.size(); ++j) {  // TODO: make parallel
          uint32_t point_id = points.get_id(j);
          outfile.write(reinterpret_cast<const char *>(&point_id), sizeof(uint32_t));
        }
      }
    }

    if (params.pq.enabled) {
      outfile.write(reinterpret_cast<const char *>(pq_offsets.begin()),
                    pq_offsets.size() * sizeof(size_t));
      for (size_t i = 0; i < num; ++i) {
        node_t *node = ind_to_node[i];
        if (node->children.size() == 0) {  // leaves only
          node->pq_data.save(outfile);
        }
      }
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

    infile.read(reinterpret_cast<char *>(&params.pq.enabled), sizeof(params.pq.enabled));

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

    parlay::sequence<size_t> pq_offsets;
    if (params.pq.enabled) {
      pq_offsets.resize(num + 1);
      infile.read(reinterpret_cast<char *>(pq_offsets.begin()), (num + 1) * sizeof(size_t));
    }

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

    if (params.pq.enabled) {
      for (size_t i = 0; i < num; ++i) {
        if (children_sizes[i] == 0) {  // Leaf node
          ind_to_node[i]->pq_data = QuantizedPointCloudSet<ChPoint>(infile);
        }
      }
    }

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

  // Traversing the tree and deleting nodes
  void traverse_and_delete(node_t *node) {
    for (size_t i = 0; i < node->children.size(); i++) {
      node_t *child = node->children[i];
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

  // size_t mean_cluster_size() const noexcept override {
  //   auto cluster_sizes =
  //       parlay::delayed_seq<size_t>(clusters.size(), [&](size_t i) { return clusters[i].size();
  //       });
  //   return parlay::reduce(cluster_sizes) / clusters.size();
  // }

  // size_t max_cluster_size() const noexcept override {
  //   auto cluster_sizes =
  //       parlay::delayed_seq<size_t>(clusters.size(), [&](size_t i) { return clusters[i].size();
  //       });
  //   return parlay::reduce(cluster_sizes, parlay::maxm<size_t>());
  // }
};

using IndexMVIVFL2 = IndexMVIVF<true>;   // Instantiates for L2 metric (metric = true)
using IndexMVIVFIP = IndexMVIVF<false>;  // Instantiates for MIPS      (metric = false)

}  // namespace mvsic