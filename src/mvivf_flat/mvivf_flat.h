#pragma once

#include <queue>
#include <set>
#include "src/common/index.h"
#include "src/mvc/mvkmeans.h"
#include "src/utils/top_neighbors.h"
#include "src/utils/vqsort_utils.h"

namespace mvivf {

/* ===================================Multi-Vector IVF Class=================================== */
template<bool metric>
class IndexMVIVFFlat : Index<metric> {
 public:
  using ChPoint = Index<metric>::ChPoint;  // Chamfer Point Type
  using Index<metric>::d;                  // Embedding dimension

  size_t maxsize = 500;                          // Maxsize of leaf clusters (enforced)
  double s = 1.0;                                // s * (average # vectors)/num_docs
  size_t iters = 5;                              // Number of Outer Lloyd's Iterations
  size_t os_rate = 20;                           // Oversampling factor for Inner Kmeans
  bool verbose = false;                          // Print debug statements
  PointCloudSet<ChPoint> centers;                // Centers of clusters
  std::vector<PointCloudSet<ChPoint>> clusters;  // Clusters of points

  IndexMVIVFFlat(size_t d_) noexcept { d = d_; }
  IndexMVIVFFlat(size_t d_, const IndexParams &params) noexcept {
    d = d_;
    maxsize = params.maxsize;
    s = params.s;
    iters = params.iters;
    os_rate = params.os_rate;
    verbose = params.verbose;
  }
  /* ----------------------------Overridden Functions---------------------------- */
  void build(const PointCloudSet<ChPoint> &points) override;
  std::pair<parlay::sequence<std::pair<size_t, float>>, size_t> search(
      const ChPoint &query, const PointCloudSet<ChPoint> &points,
      const SearchParams &params) override;
  void save(const std::string &filename) override {}
  void load(const std::string &filename, const PointCloudSet<ChPoint> &points) override {}
};

/* =======================================Implementation======================================= */
// Builds the index given a point cloud set.
template<bool metric>
void IndexMVIVFFlat<metric>::build(const PointCloudSet<ChPoint> &points) {
  size_t n = points.size();
  size_t mp = std::max((size_t)4, maxsize / 100);
  size_t num_clusters = mp * std::ceil(n / maxsize);
  if (verbose) {
    std::cout << "Building index with " << n << " points, maxsize: " << maxsize
              << ", num_clusters: " << num_clusters << std::endl;
  }
  // Step 1: Run MV-Lloyds on points
  MVClusteringParams params(iters, "Random", os_rate, verbose);
  MVClustering<metric> clus(d, num_clusters, s, params);
  clus.train(points);
  centers = std::move(clus.centers);
  parlay::sequence<size_t> &cluster_ids = clus.cluster_ids;
  // Step 2: Collect Clusters
  auto id_pt_del = parlay::delayed_seq<std::pair<size_t, size_t>>(
      n, [&](size_t i) { return std::make_pair(cluster_ids[i], i); });
  auto id_pt =
      parlay::sort(id_pt_del, [](const auto &a, const auto &b) { return a.first < b.first; });
  auto cutoff_indices = parlay::delayed_seq<size_t>(
      n + 1, [&](size_t i) { return i == 0 || i == n || id_pt[i].first != id_pt[i - 1].first; });
  auto cluster_offsets = parlay::pack_index(cutoff_indices);
  // Step 3: Collect point clouds by clusters
  clusters.resize(num_clusters);
  parlay::parallel_for(0, cluster_offsets.size() - 1, [&](size_t i) {
    size_t start_index = cluster_offsets[i];
    size_t end_index = cluster_offsets[i + 1];
    size_t cluster_id = id_pt[start_index].first;
    auto group = parlay::delayed_seq<size_t>(
        end_index - start_index, [&](size_t j) { return id_pt[start_index + j].second; });
    PointCloudSet<ChPoint> cluster_points = PointCloudSet<ChPoint>(points.filter(group), d);
    clusters[i] = std::move(cluster_points);
  });
}

// Returns the top-k point clouds for the query point cloud
// Output format: < [<id, distance>, ...], No. of distance computations >
template<bool metric>
std::pair<parlay::sequence<std::pair<size_t, float>>, size_t> IndexMVIVFFlat<metric>::search(
    const ChPoint &query, const PointCloudSet<ChPoint> &points, const SearchParams &params) {
  size_t k = params.k;
  size_t nprobes = params.nprobes;
  size_t dist_cmps = 0;
  // Step 1: Compute distances to centers
  auto [all_dists, dist_cmps_node] = centers.distances(query);
  dist_cmps += dist_cmps_node;
  // TODO: Add support for vqsort and vqpartialsort
  // auto dist_id = parlay::sequence<std::pair<size_t, float>>::from_function(
  //     all_dists.size(), [&](size_t i) { return std::make_pair(i, all_dists[i]); });
  // parlay::sort_inplace(dist_id, [](const auto &a, const auto &b) { return a.second < b.second;
  // });
  auto dist_id_d = parlay::sequence<double>::from_function(
      all_dists.size(), [&](size_t i) { return packFloatAndInt(all_dists[i], i); });
  VQSort(dist_id_d.begin(), dist_id_d.end());
  auto dist_id = parlay::sequence<std::pair<size_t, float>>::uninitialized(all_dists.size());
  parlay::parallel_for(0, dist_id_d.size(), [&](size_t i) {
    auto [ext_float, ind] = unpackDouble2(dist_id_d[i]);
    dist_id[i] = std::make_pair(ind, ext_float);
  });

  // Step 2: Probe top nprobe clusters
  // Find the minimum number of probes needed to obtain k neighbors
  size_t nprobes_minimal = 0, cur = 0;
  while (nprobes_minimal < dist_id.size() && cur <= k) {
    cur += clusters[dist_id[nprobes_minimal].first].size();
    nprobes_minimal++;
  }
  nprobes = std::min(dist_id.size(), std::max(nprobes, nprobes_minimal));
  auto probe_dist_cmps = parlay::sequence<size_t>::uninitialized(nprobes);
  auto sizes = parlay::delayed_seq<size_t>(
      nprobes, [&](size_t i) { return std::min(clusters[dist_id[i].first].size(), k); });
  auto [offsets, total_size] = parlay::scan(sizes);
  auto results = parlay::sequence<std::pair<size_t, float>>::uninitialized(total_size);
  parlay::parallel_for(0, nprobes, [&](size_t i) {
    size_t cluster_id = dist_id[i].first;
    probe_dist_cmps[i] =
        get_knn_into_uninitialized(query, clusters[cluster_id], k, &results[offsets[i]]);
  });
  dist_cmps += parlay::reduce(probe_dist_cmps);
  // Step 3: Re-ranking
  // TODO: replace with vqsort and vqpartialsort
  parlay::sort_inplace(results, [](const auto &a, const auto &b) {
    return a.second < b.second;  // Sort by distance
  });
  auto final_results = parlay::sequence<std::pair<size_t, float>>::from_function(
      std::min(k, results.size()), [&](size_t i) { return results[i]; });
  return std::make_pair(final_results, dist_cmps);
}

// template<bool metric>
// void IndexMVIVFFlat<metric>::save(const std::string &filename) {
//   std::ofstream outfile(filename, std::ios::binary);
//   std::cout << "Saving index to " << filename << std::endl;
//   if (!outfile.is_open()) {
//     std::cerr << "Error opening file for writing: " << filename << std::endl;
//     return;
//   }
//   // Collect data
//   parlay::sequence<node_t *> ind_to_node;
//   std::unordered_map<node_t *, size_t> node_to_ind;
//   parlay::sequence<size_t> center_offsets;
//   parlay::sequence<size_t> children_offsets;
//   parlay::sequence<size_t> point_offsets;

//   size_t height = traverse_tree(root, ind_to_node, node_to_ind, center_offsets, children_offsets,
//                                 point_offsets, 0);
//   std::cout << "Height of tree: " << height << std::endl;

//   size_t total_center_sizes = parlay::scan_inplace(center_offsets);
//   center_offsets.push_back(total_center_sizes);
//   size_t total_children_sizes = parlay::scan_inplace(children_offsets);
//   children_offsets.push_back(total_children_sizes);
//   size_t total_point_sizes = parlay::scan_inplace(point_offsets);
//   point_offsets.push_back(total_point_sizes);

//   // Write num
//   size_t num = ind_to_node.size();
//   outfile.write(reinterpret_cast<const char *>(&num), sizeof(size_t));
//   // Write center offsets
//   size_t num_center_offsets = center_offsets.size();
//   outfile.write(reinterpret_cast<const char *>(&num_center_offsets), sizeof(size_t));
//   outfile.write(reinterpret_cast<const char *>(center_offsets.begin()),
//                 center_offsets.size() * sizeof(size_t));
//   // Write centers values
//   for (size_t i = 0; i < num; ++i) {  // TODO: make parallel
//     node_t *node = ind_to_node[i];
//     if (node->children.size() > 0) {  // Internal Nodes only
//       auto coords = node->data.data();
//       size_t num_entries = (node->data.total_size()) * (node->data.get_dims());
//       outfile.write(reinterpret_cast<const char *>(coords), num_entries * sizeof(float));
//     }
//   }
//   // Write children offsets
//   outfile.write(reinterpret_cast<const char *>(children_offsets.begin()),
//                 children_offsets.size() * sizeof(size_t));
//   // Write children values
//   for (size_t i = 0; i < num; ++i) {
//     node_t *node = ind_to_node[i];
//     parlay::sequence<node_t *> children = node->children;
//     for (size_t j = 0; j < children.size(); ++j) {  // TODO: make parallel
//       node_t *child = children[j];
//       size_t child_id = node_to_ind[child];
//       outfile.write(reinterpret_cast<const char *>(&child_id), sizeof(size_t));
//     }
//   }
//   // Write point offsets
//   outfile.write(reinterpret_cast<const char *>(point_offsets.begin()),
//                 point_offsets.size() * sizeof(size_t));
//   // Write point values
//   for (size_t i = 0; i < num; ++i) {
//     node_t *node = ind_to_node[i];
//     if (node->children.size() == 0) {  // leaves only
//       PointCloudSet<ChPoint> points = node->data;
//       for (size_t j = 0; j < points.size(); ++j) {  // TODO: make parallel
//         size_t point_id = points.get_id(j);
//         outfile.write(reinterpret_cast<const char *>(&point_id), sizeof(size_t));
//       }
//     }
//   }
//   outfile.close();
// }

// template<bool metric>
// void IndexMVIVFFlat<metric>::load(const std::string &filename,
//                                   const PointCloudSet<ChPoint> &points) {
//   std::ifstream infile(filename, std::ios::binary);
//   std::cout << "Loading index from " << filename << std::endl;
//   if (!infile.is_open()) {
//     std::cerr << "Error opening file for reading: " << filename << std::endl;
//     return;
//   }
//   // Read number of nodes
//   size_t num = 0;
//   infile.read(reinterpret_cast<char *>(&num), sizeof(size_t));
//   // Read center offsets
//   size_t num_center_offsets = 0;
//   infile.read(reinterpret_cast<char *>(&num_center_offsets), sizeof(size_t));
//   parlay::sequence<size_t> center_offsets(num_center_offsets);
//   infile.read(reinterpret_cast<char *>(center_offsets.begin()),
//               center_offsets.size() * sizeof(size_t));
//   // Read centers values
//   parlay::sequence<float> center_values(center_offsets[center_offsets.size() - 1]);
//   infile.read(reinterpret_cast<char *>(center_values.begin()),
//               center_values.size() * sizeof(float));
//   // Read children offsets
//   parlay::sequence<size_t> children_offsets(num + 1);
//   infile.read(reinterpret_cast<char *>(children_offsets.begin()),
//               children_offsets.size() * sizeof(size_t));
//   // Read children values
//   parlay::sequence<size_t> children_values(children_offsets[children_offsets.size() - 1]);
//   infile.read(reinterpret_cast<char *>(children_values.begin()),
//               children_values.size() * sizeof(size_t));
//   // Read point offsets
//   parlay::sequence<size_t> point_offsets(num + 1);
//   infile.read(reinterpret_cast<char *>(point_offsets.begin()),
//               point_offsets.size() * sizeof(size_t));
//   // Read point values
//   parlay::sequence<size_t> point_values(point_offsets[point_offsets.size() - 1]);
//   infile.read(reinterpret_cast<char *>(point_values.begin()), point_values.size() *
//   sizeof(size_t));

//   // Build the index
//   size_t dim = points.get_dims();
//   auto point_id_to_data_id = parlay::sequence<size_t>::uninitialized(points.size());
//   parlay::parallel_for(0, points.size(),
//                        [&](size_t i) { point_id_to_data_id[points.get_id(i)] = i; });
//   parlay::sequence<size_t> children_sizes = parlay::sequence<size_t>::from_function(
//       num, [&](size_t i) { return children_offsets[i + 1] - children_offsets[i]; });
//   auto [children_sizes_scan, total_children_sizes] = parlay::scan(children_sizes);
//   children_sizes_scan.push_back(total_children_sizes);
//   assert(total_children_sizes == center_offsets.);
//   parlay::sequence<size_t> point_sizes = parlay::sequence<size_t>::from_function(
//       num, [&](size_t i) { return point_offsets[i + 1] - point_offsets[i]; });
//   parlay::sequence<node_t *> ind_to_node =
//       parlay::sequence<node_t *>::from_function(num, [&](size_t i) {
//         node_t *node = new node_t();
//         if (children_sizes[i] > 0) {  // Internal Nodes
//           size_t start_offset = children_sizes_scan[i];
//           size_t end_offset = children_sizes_scan[i + 1];
//           parlay::sequence<size_t> node_center_offsets =
//               parlay::sequence<size_t>::from_function(children_sizes[i] + 1, [&](size_t j) {
//                 return center_offsets[start_offset + j] - center_offsets[start_offset];
//               });

//           node->set_data(PointCloudSet<ChPoint>(children_sizes[i], dim,
//                                                 center_values.data() +
//                                                 center_offsets[start_offset],
//                                                 node_center_offsets.data(), nullptr));
//         }
//         node->children.resize(children_sizes[i]);
//         if (point_sizes[i] > 0) {
//           parlay::sequence<size_t> point_group =
//               parlay::sequence<size_t>::from_function(point_sizes[i], [&](size_t j) {
//                 size_t point_id = point_values[point_offsets[i] + j];
//                 return point_id_to_data_id[point_id];
//               });
//           node->set_data(PointCloudSet<ChPoint>(points.filter(point_group), dim));
//         }
//         return node;
//       });

//   // Set children pointers
//   parlay::parallel_for(0, num, [&](size_t i) {
//     node_t *node = ind_to_node[i];
//     parlay::sequence<node_t *> &children = node->children;
//     parlay::parallel_for(0, children.size(), [&](size_t j) {
//       size_t child_id = children_values[children_offsets[i] + j];
//       children[j] = ind_to_node[child_id];
//     });
//   });
//   root = ind_to_node[0];
//   infile.close();
// }

using IndexMVIVFFlatL2 = IndexMVIVFFlat<true>;   // L2 metric
using IndexMVIVFFlatIP = IndexMVIVFFlat<false>;  // MIPS

}  // namespace mvivf