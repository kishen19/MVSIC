#pragma once

#include <queue>
#include <set>
#include "src/common/index.h"
#include "src/mvc/mvkmeans.h"
#include "src/utils/top_neighbors.h"
#include "src/utils/sort_utils.h"

namespace mvivf {

/* Multi-Vector IVF Index: Flat version
  Indexing:
  - Runs the MV-Lloyd's algorithm to cluster the input point clouds into `num_clusters` clusters.
    Each cluster is now represented by a "center" point cloud.
  Search:
  - For a given query point cloud, it computes distances to all centers point clouds,
   and then probes the top `nprobes` clusters.
*/

/* =================================Multi-Vector IVF Flat Class=============================== */
template<bool metric>
class IndexMVIVFFlat : public Index<metric> {
 public:
  using ChPoint = Index<metric>::ChPoint;  // Chamfer Point Type
  using Index<metric>::d;                  // Embedding dimension

  size_t num_clusters = 300;  // Number of clusters
  bool verbose = false;       // Print debug statements

  // MV-Lloyd's parameters
  double s = 1.0;       // s * (average # vectors)/num_docs
  size_t iters = 5;     // Number of Outer Lloyd's Iterations
  size_t os_rate = 20;  // Oversampling factor for Inner Kmeans

  PointCloudSet<ChPoint> centers;                     // Centers of clusters
  std::vector<PointCloudSet<ChPoint>> clusters = {};  // Clusters of points

  IndexMVIVFFlat(size_t d_) noexcept { d = d_; }
  IndexMVIVFFlat(size_t d_, const IndexParams &params) noexcept {
    d = d_;
    num_clusters = params.num_clusters;
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
  void save(const std::string &filename) override;
  void load(const std::string &filename, const PointCloudSet<ChPoint> &points) override;
  /* ------------------------------Helper Functions------------------------------ */
  std::pair<parlay::sequence<std::pair<size_t, float>>, size_t> search_with_stats(
      const ChPoint &query, const PointCloudSet<ChPoint> &points, const SearchParams &params) {}
  size_t mean_cluster_size() const noexcept;
  size_t max_cluster_size() const noexcept;
};

/* =======================================Implementation======================================= */
// Builds the index given a point cloud set.
template<bool metric>
void IndexMVIVFFlat<metric>::build(const PointCloudSet<ChPoint> &points) {
  size_t n = points.size();
  if (verbose) {
    std::cout << "Building index with " << n << " points, num_clusters: " << num_clusters
              << std::endl;
  }
  // Step 1: Run MV-Lloyds on points
  MVClusteringParams clus_params(iters, "Random", os_rate, verbose);
  MVClustering<metric> Clus(d, num_clusters, s, clus_params);
  Clus.train(points);
  centers = std::move(Clus.centers);
  parlay::sequence<size_t> &cluster_ids = Clus.cluster_ids;
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
  parlay::parallel_for(
      0, cluster_offsets.size() - 1,
      [&](size_t i) {
        size_t start_index = cluster_offsets[i];
        size_t end_index = cluster_offsets[i + 1];
        size_t cluster_id = id_pt[start_index].first;
        auto group = parlay::delayed_seq<size_t>(
            end_index - start_index, [&](size_t j) { return id_pt[start_index + j].second; });
        PointCloudSet<ChPoint> cluster_points = PointCloudSet<ChPoint>(points.filter(group), d);
        clusters[cluster_id] = std::move(cluster_points);
      },
      1);
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
  auto dist_id = parlay::sequence<std::pair<size_t, float>>::from_function(
      all_dists.size(), [&](size_t i) { return std::make_pair(i, all_dists[i]); });
  parlay::sort_inplace(dist_id, [](const auto &a, const auto &b) { return a.second < b.second; });

  // Step 2: Probe top nprobe clusters
  // Find the minimum number of probes needed to obtain k neighbors
  size_t nprobes_minimal = 0, cur = 0;
  while (nprobes_minimal < dist_id.size() && cur < k) {
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
#ifdef USE_HWY
  auto results_d = parlay::sequence<double>::from_function(results.size(), [&](size_t i) {
    return packFloatAndInt(results[i].second, results[i].first);
  });
  // VQSort(dist_id_d.begin(), dist_id_d.end());
  VQPartialSort(results_d.begin(), results_d.end(), std::min(k, results_d.size()));
  auto final_results = parlay::sequence<std::pair<size_t, float>>::from_function(
      std::min(k, results.size()), [&](size_t i) {
        auto [ext_float, ind] = unpackDouble2(results_d[i]);
        return std::make_pair(ind, ext_float);
      });
#else
  parlay::sort_inplace(results, [](const auto &a, const auto &b) {
    return a.second < b.second;  // Sort by distance
  });
  auto final_results = parlay::sequence<std::pair<size_t, float>>::from_function(
      std::min(k, results.size()), [&](size_t i) { return results[i]; });
#endif
  return std::make_pair(final_results, dist_cmps);
}

template<bool metric>
void IndexMVIVFFlat<metric>::save(const std::string &filename) {
  std::ofstream outfile(filename, std::ios::binary);
  std::cout << "Saving index to " << filename << std::endl;
  if (!outfile.is_open()) {
    std::cerr << "Error opening file for writing: " << filename << std::endl;
    return;
  }
  // Collect data
  parlay::sequence<size_t> center_offsets = parlay::sequence<size_t>::from_function(
      centers.size(), [&](size_t i) { return centers.get_size(i) * d; });
  size_t total_center_sizes = parlay::scan_inplace(center_offsets);
  center_offsets.push_back(total_center_sizes);
  parlay::sequence<size_t> clusters_offsets = parlay::sequence<size_t>::from_function(
      clusters.size(), [&](size_t i) { return clusters[i].size(); });
  size_t total_clusters_size = parlay::scan_inplace(clusters_offsets);
  clusters_offsets.push_back(total_clusters_size);

  // Write num
  size_t num = centers.size();
  outfile.write(reinterpret_cast<const char *>(&num), sizeof(size_t));
  // Write center offsets
  outfile.write(reinterpret_cast<const char *>(center_offsets.begin()),
                center_offsets.size() * sizeof(size_t));
  // Write centers values
  auto coords = centers.data();
  size_t num_entries = (centers.total_size()) * (centers.get_dims());
  outfile.write(reinterpret_cast<const char *>(coords), num_entries * sizeof(float));
  // Write clusters offsets
  outfile.write(reinterpret_cast<const char *>(clusters_offsets.begin()),
                clusters_offsets.size() * sizeof(size_t));
  // Write clusters values
  for (size_t i = 0; i < num; ++i) {
    if (clusters[i].size() > 0) {
      for (size_t j = 0; j < clusters[i].size(); ++j) {
        size_t point_id = clusters[i].get_id(j);
        outfile.write(reinterpret_cast<const char *>(&point_id), sizeof(size_t));
      }
    }
  }
  outfile.close();
}

template<bool metric>
void IndexMVIVFFlat<metric>::load(const std::string &filename,
                                  const PointCloudSet<ChPoint> &points) {
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
  size_t num_center_offsets = num + 1;
  parlay::sequence<size_t> center_offsets(num_center_offsets);
  infile.read(reinterpret_cast<char *>(center_offsets.begin()),
              center_offsets.size() * sizeof(size_t));
  // Read centers values
  parlay::sequence<float> center_values(center_offsets[center_offsets.size() - 1]);
  infile.read(reinterpret_cast<char *>(center_values.begin()),
              center_values.size() * sizeof(float));
  // Read clusters offsets
  parlay::sequence<size_t> clusters_offsets(num + 1);
  infile.read(reinterpret_cast<char *>(clusters_offsets.begin()),
              clusters_offsets.size() * sizeof(size_t));
  // Read clusters values
  parlay::sequence<size_t> clusters_values(clusters_offsets[clusters_offsets.size() - 1]);
  infile.read(reinterpret_cast<char *>(clusters_values.begin()),
              clusters_values.size() * sizeof(size_t));

  // Build the index
  size_t dim = points.get_dims();
  auto point_id_to_data_id = parlay::sequence<size_t>::uninitialized(points.size());
  parlay::parallel_for(0, points.size(),
                       [&](size_t i) { point_id_to_data_id[points.get_id(i)] = i; });
  parlay::sequence<size_t> clusters_sizes = parlay::sequence<size_t>::from_function(
      num, [&](size_t i) { return clusters_offsets[i + 1] - clusters_offsets[i]; });
  centers = PointCloudSet<ChPoint>(num, dim, center_values.data(), center_offsets.data(), nullptr);
  clusters.resize(num);
  parlay::parallel_for(0, num, [&](size_t i) {
    if (clusters_sizes[i] > 0) {
      auto cluster_group = parlay::delayed_seq<size_t>(clusters_sizes[i], [&](size_t j) {
        size_t point_id = clusters_values[clusters_offsets[i] + j];
        return point_id_to_data_id[point_id];
      });
      clusters[i] = PointCloudSet<ChPoint>(points.filter(cluster_group), dim);
    }
  });
  infile.close();
}

// Returns the mean cluster size
template<bool metric>
size_t IndexMVIVFFlat<metric>::mean_cluster_size() const noexcept {
  auto cluster_sizes =
      parlay::delayed_seq<size_t>(clusters.size(), [&](size_t i) { return clusters[i].size(); });
  return parlay::reduce(cluster_sizes) / clusters.size();
}

// Returns the mean cluster size
template<bool metric>
size_t IndexMVIVFFlat<metric>::max_cluster_size() const noexcept {
  auto cluster_sizes =
      parlay::delayed_seq<size_t>(clusters.size(), [&](size_t i) { return clusters[i].size(); });
  return parlay::reduce(cluster_sizes, parlay::maxm<size_t>());
}

using IndexMVIVFFlatL2 = IndexMVIVFFlat<true>;   // L2 metric
using IndexMVIVFFlatIP = IndexMVIVFFlat<false>;  // MIPS

}  // namespace mvivf