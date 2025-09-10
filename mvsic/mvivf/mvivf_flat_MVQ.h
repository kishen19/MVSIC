#pragma once

#include <queue>
#include <set>
#include "src/common/index.h"
#include "src/mvc/mvkmeans.h"
#include "src/utils/ip_point.h"
#include "src/utils/kmeans_util.h"
#include "src/utils/l2_point.h"
#include "src/utils/point_range.h"
#include "src/utils/sort_utils.h"
#include "src/utils/top_neighbors.h"

namespace mvivf {

/* Multi-Vector IVF Index: Flat version with MV Quantization
  Indexing:
  - Runs the MV-Lloyd's algorithm to cluster the input point clouds into `num_clusters` clusters.
    Each cluster is now represented by a "center" point cloud.
  Search:
  - For a given query point cloud, it computes distances to all centers point clouds,
   and then probes the top `nprobes` clusters.
*/

/* =================================Multi-Vector IVF Flat Class=============================== */
template<bool metric>
class IndexMVIVFFlatMVQ : public Index<metric> {
 public:
  using ChPoint = Index<metric>::ChPoint;  // Chamfer Point Type
  using Point = std::conditional_t<metric, L2_Point<float>, IP_Point<float>>;
  using Range = PointRange<float, Point>;
  using Index<metric>::d;  // Embedding dimension

  struct LeafCluster {
    parlay::sequence<size_t> point_ids;
    Range centroids;
    parlay::sequence<parlay::sequence<uint8_t>> LUT;

    size_t size() const noexcept { return point_ids.size(); }
  };

  size_t num_clusters = 300;       // Number of clusters
  size_t num_leaf_centroids = 16;  //
  bool verbose = false;            // Print debug statements

  // MV-Lloyd's parameters
  double s = 1.0;       // s * (average # vectors)/num_docs
  size_t iters = 5;     // Number of Outer Lloyd's Iterations
  size_t os_rate = 20;  // Oversampling factor for Inner Kmeans

  PointCloudSet<ChPoint> centers;          // Centers of clusters
  std::vector<LeafCluster> clusters = {};  //

  IndexMVIVFFlatMVQ(size_t d_) noexcept { d = d_; }
  IndexMVIVFFlatMVQ(size_t d_, const IndexParams &params) noexcept {
    d = d_;
    num_clusters = params.num_clusters;
    assert(params.num_leaf_centroids < 256);
    num_leaf_centroids = params.num_leaf_centroids;
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
void IndexMVIVFFlatMVQ<metric>::build(const PointCloudSet<ChPoint> &points) {
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
        auto group = parlay::sequence<size_t>::from_function(
            end_index - start_index, [&](size_t j) { return id_pt[start_index + j].second; });
        auto group_data = points.filter_flattened(group);
#ifdef USE_TOP_N_ASSIGN
        const size_t TOP_N_ASSIGNMENTS = 4;
        auto [centroids, top_n_assignments] = kmeans_subsample_top_n_assign<metric>(
            group_data, num_leaf_centroids, TOP_N_ASSIGNMENTS, os_rate, verbose);
        // Build LUT
        auto num_embs = parlay::delayed_seq<size_t>(
            group.size(), [&](size_t j) { return points.get_size(group[j]); });
        auto [offsets, _total_group_size] = parlay::scan(num_embs);
        offsets.push_back(_total_group_size);
        auto LUT = parlay::sequence<parlay::sequence<uint8_t>>(group.size());
        parlay::parallel_for(0, group.size(), [&](size_t j) {
          size_t start_offset_index = offsets[j];
          size_t end_offset_index = offsets[j + 1];

          auto point_cloud_top_n_assignments =
              top_n_assignments.cut(start_offset_index, end_offset_index);
          auto flattened_assignments = parlay::flatten(point_cloud_top_n_assignments);

          auto unique_assignments = parlay::unique(flattened_assignments);
          LUT[j] = parlay::map(unique_assignments,
                               [](size_t asgn) { return static_cast<uint8_t>(asgn); });
        });
        clusters[cluster_id].point_ids = std::move(group);
        clusters[cluster_id].centroids = Range(centroids, d);
        clusters[cluster_id].LUT = std::move(LUT);
#else
        auto [centroids, assignment] =
            kmeans_subsample_assign_only<metric>(group_data, num_leaf_centroids, os_rate, verbose);
        // Build LUT
        auto num_embs = parlay::delayed_seq<size_t>(
            group.size(), [&](size_t j) { return points.get_size(group[j]); });
        auto [offsets, _total_group_size] = parlay::scan(num_embs);
        offsets.push_back(_total_group_size);
        auto LUT = parlay::sequence<parlay::sequence<uint8_t>>(group.size());
        parlay::parallel_for(0, group.size(), [&](size_t j) {
          size_t start_offset_index = offsets[j];
          size_t end_offset_index = offsets[j + 1];
          size_t sz = end_offset_index - start_offset_index;
          auto pc_assignments = parlay::delayed_seq<uint8_t>(sz, [&](size_t k) {
            return static_cast<uint8_t>(assignment[start_offset_index + k]);
          });
          LUT[j] = parlay::unique(pc_assignments);
        });
        clusters[cluster_id].point_ids = std::move(group);
        clusters[cluster_id].centroids = Range(centroids, d);
        clusters[cluster_id].LUT = std::move(LUT);
#endif
      },
      1);
}

// Returns the top-k point clouds for the query point cloud
// Output format: < [<id, distance>, ...], No. of distance computations >
template<bool metric>
std::pair<parlay::sequence<std::pair<size_t, float>>, size_t> IndexMVIVFFlatMVQ<metric>::search(
    const ChPoint &query, const PointCloudSet<ChPoint> &points, const SearchParams &params) {
  size_t k = params.k;
  size_t nprobes = params.nprobes;
  size_t cands = params.cands;
  size_t dist_cmps = 0;
  // Step 1: Compute distances to centers
  auto [all_dists, dist_cmps_node] = centers.distances(query);
  dist_cmps += dist_cmps_node;
  auto dist_id = parlay::sequence<std::pair<size_t, float>>::from_function(
      all_dists.size(), [&](size_t i) { return std::make_pair(i, all_dists[i]); });
  parlay::sort_inplace(dist_id, [](const auto &a, const auto &b) { return a.second < b.second; });

  // Step 2: Probe top nprobe clusters
  // Find the minimum number of probes needed to obtain k neighbors
  // TODO: compute prefix sum, and start from nprobes and walk upwards
  size_t nprobes_minimal = 0, cur = 0;
  while (nprobes_minimal < dist_id.size() && cur < k) {
    cur += clusters[dist_id[nprobes_minimal].first].size();
    nprobes_minimal++;
  }
  nprobes = std::min(dist_id.size(), std::max(nprobes, nprobes_minimal));
  auto sizes = parlay::delayed_seq<size_t>(
      nprobes, [&](size_t i) { return std::min(clusters[dist_id[i].first].size(), cands); });
  auto [offsets, total_size] = parlay::scan(sizes);
  auto probe_dist_cmps = parlay::sequence<size_t>::uninitialized(nprobes);
  auto results = parlay::sequence<std::pair<size_t, float>>::uninitialized(total_size);
  parlay::parallel_for(0, nprobes, [&](size_t i) {
    size_t cluster_id = dist_id[i].first;
    probe_dist_cmps[i] = get_knn_via_centroids_into_uninitialized(query, clusters[cluster_id],
                                                                  cands, &results[offsets[i]]);
  });
  dist_cmps += parlay::reduce(probe_dist_cmps);
  // Step 3: Re-ranking
  auto rerank_dist_cmps = parlay::sequence<size_t>::uninitialized(total_size);
  parlay::parallel_for(0, total_size, [&](size_t i) {
    size_t id = results[i].first;
    auto [dist, d_c] = query.distance_w_cmps(points[id]);
    results[i].second = dist;
    rerank_dist_cmps[i] = d_c;
  });
  dist_cmps += parlay::reduce(rerank_dist_cmps);
  parlay::sort_inplace(results, [](const auto &a, const auto &b) {
    return a.second < b.second;  // Sort by distance
  });
  auto final_results = parlay::sequence<std::pair<size_t, float>>::from_function(
      std::min(k, results.size()), [&](size_t i) { return results[i]; });
  return std::make_pair(final_results, dist_cmps);
}

template<bool metric>
void IndexMVIVFFlatMVQ<metric>::save(const std::string &filename) {
  std::ofstream outfile(filename, std::ios::binary);
  std::cout << "Saving index to " << filename << std::endl;
  if (!outfile.is_open()) {
    std::cerr << "Error opening file for writing: " << filename << std::endl;
    return;
  }

  // Write num_clusters
  size_t num = centers.size();
  outfile.write(reinterpret_cast<const char *>(&num), sizeof(size_t));

  // Write top-level centers (PointCloudSet)
  // You already have this logic from the non-quantized version.
  parlay::sequence<size_t> center_offsets = parlay::sequence<size_t>::from_function(
      centers.size(), [&](size_t i) { return centers.get_size(i) * d; });
  size_t total_center_sizes = parlay::scan_inplace(center_offsets);
  center_offsets.push_back(total_center_sizes);
  outfile.write(reinterpret_cast<const char *>(center_offsets.begin()),
                center_offsets.size() * sizeof(size_t));
  auto coords = centers.data();
  size_t num_entries = centers.total_size() * centers.get_dims();
  outfile.write(reinterpret_cast<const char *>(coords), num_entries * sizeof(float));

  // Collect and write point IDs
  parlay::sequence<size_t> point_id_offsets = parlay::sequence<size_t>::from_function(
      clusters.size(), [&](size_t i) { return clusters[i].point_ids.size(); });
  size_t total_points = parlay::scan_inplace(point_id_offsets);
  point_id_offsets.push_back(total_points);
  outfile.write(reinterpret_cast<const char *>(point_id_offsets.begin()),
                point_id_offsets.size() * sizeof(size_t));
  parlay::sequence<size_t> all_point_ids(total_points);
  parlay::parallel_for(0, clusters.size(), [&](size_t i) {
    parlay::copy(clusters[i].point_ids,
                 all_point_ids.cut(point_id_offsets[i], point_id_offsets[i + 1]));
  });
  outfile.write(reinterpret_cast<const char *>(all_point_ids.begin()),
                total_points * sizeof(size_t));

  // Collect and write cluster-level centroids (Ranges)
  parlay::sequence<size_t> leaf_centroid_offsets = parlay::sequence<size_t>::from_function(
      clusters.size(), [&](size_t i) { return clusters[i].centroids.size(); });
  size_t total_leaf_centroids = parlay::scan_inplace(leaf_centroid_offsets);
  leaf_centroid_offsets.push_back(total_leaf_centroids);
  outfile.write(reinterpret_cast<const char *>(leaf_centroid_offsets.begin()),
                leaf_centroid_offsets.size() * sizeof(size_t));
  parlay::sequence<float> all_leaf_centroids(total_leaf_centroids * d);
  parlay::parallel_for(0, clusters.size(), [&](size_t i) {
    if (clusters[i].centroids.size() > 0) {
      parlay::copy(
          parlay::delayed_seq<float>(clusters[i].centroids.size() * d,
                                     [&](size_t j) { return clusters[i].centroids.data()[j]; }),
          all_leaf_centroids.cut(leaf_centroid_offsets[i] * d, leaf_centroid_offsets[i + 1] * d));
    }
  });
  outfile.write(reinterpret_cast<const char *>(all_leaf_centroids.begin()),
                total_leaf_centroids * d * sizeof(float));

  // Collect and write LUTs
  parlay::sequence<size_t> lut_offsets =
      parlay::sequence<size_t>::from_function(clusters.size(), [&](size_t i) {
        size_t cluster_lut_size = 0;
        for (const auto &lut_entry : clusters[i].LUT) {
          cluster_lut_size += lut_entry.size();
        }
        return cluster_lut_size;
      });
  size_t total_lut_size = parlay::scan_inplace(lut_offsets);
  lut_offsets.push_back(total_lut_size);
  outfile.write(reinterpret_cast<const char *>(lut_offsets.begin()),
                lut_offsets.size() * sizeof(size_t));
  parlay::sequence<uint8_t> all_lut_data(total_lut_size);
  parlay::parallel_for(0, clusters.size(), [&](size_t i) {
    size_t current_offset = lut_offsets[i];
    for (const auto &lut_entry : clusters[i].LUT) {
      parlay::copy(lut_entry, all_lut_data.cut(current_offset, current_offset + lut_entry.size()));
      current_offset += lut_entry.size();
    }
  });
  outfile.write(reinterpret_cast<const char *>(all_lut_data.begin()),
                total_lut_size * sizeof(uint8_t));

  // Write LUT entry counts per cluster
  parlay::sequence<size_t> lut_entry_counts(clusters.size());
  for (size_t i = 0; i < clusters.size(); ++i) {
    lut_entry_counts[i] = clusters[i].LUT.size();
  }
  outfile.write(reinterpret_cast<const char *>(lut_entry_counts.begin()),
                lut_entry_counts.size() * sizeof(size_t));

  // Write the size of each LUT entry
  auto all_lut_entries =
      parlay::flatten(parlay::map(clusters, [](const auto &c) { return c.LUT; }));
  parlay::sequence<size_t> all_lut_entry_sizes =
      parlay::map(all_lut_entries, [](const auto &entry) { return entry.size(); });
  outfile.write(reinterpret_cast<const char *>(all_lut_entry_sizes.begin()),
                all_lut_entry_sizes.size() * sizeof(size_t));

  outfile.close();
}

template<bool metric>
void IndexMVIVFFlatMVQ<metric>::load(const std::string &filename,
                                     const PointCloudSet<ChPoint> &points) {
  std::ifstream infile(filename, std::ios::binary);
  std::cout << "Loading index from " << filename << std::endl;
  if (!infile.is_open()) {
    std::cerr << "Error opening file for reading: " << filename << std::endl;
    return;
  }

  // Read num_clusters
  size_t num = 0;
  infile.read(reinterpret_cast<char *>(&num), sizeof(size_t));
  num_clusters = num;

  // Read top-level centers
  size_t num_center_offsets = num + 1;
  parlay::sequence<size_t> center_offsets(num_center_offsets);
  infile.read(reinterpret_cast<char *>(center_offsets.begin()),
              center_offsets.size() * sizeof(size_t));
  parlay::sequence<float> center_values(center_offsets.back());
  infile.read(reinterpret_cast<char *>(center_values.begin()),
              center_values.size() * sizeof(float));
  centers = PointCloudSet<ChPoint>(num, d, center_values.data(), center_offsets.data(), nullptr);

  // Read point IDs
  size_t num_point_id_offsets = num + 1;
  parlay::sequence<size_t> point_id_offsets(num_point_id_offsets);
  infile.read(reinterpret_cast<char *>(point_id_offsets.begin()),
              point_id_offsets.size() * sizeof(size_t));
  parlay::sequence<size_t> all_point_ids(point_id_offsets.back());
  infile.read(reinterpret_cast<char *>(all_point_ids.begin()),
              all_point_ids.size() * sizeof(size_t));

  // Read cluster-level centroids
  size_t num_leaf_centroid_offsets = num + 1;
  parlay::sequence<size_t> leaf_centroid_offsets(num_leaf_centroid_offsets);
  infile.read(reinterpret_cast<char *>(leaf_centroid_offsets.begin()),
              leaf_centroid_offsets.size() * sizeof(size_t));
  parlay::sequence<float> all_leaf_centroids(leaf_centroid_offsets.back() * d);
  infile.read(reinterpret_cast<char *>(all_leaf_centroids.begin()),
              all_leaf_centroids.size() * sizeof(float));

  // Read LUT data
  size_t num_lut_offsets = num + 1;
  parlay::sequence<size_t> lut_offsets(num_lut_offsets);
  infile.read(reinterpret_cast<char *>(lut_offsets.begin()), lut_offsets.size() * sizeof(size_t));
  parlay::sequence<uint8_t> all_lut_data(lut_offsets.back());
  infile.read(reinterpret_cast<char *>(all_lut_data.begin()),
              all_lut_data.size() * sizeof(uint8_t));

  // Read LUT entry counts per cluster
  parlay::sequence<size_t> lut_entry_counts(num);
  infile.read(reinterpret_cast<char *>(lut_entry_counts.begin()), num * sizeof(size_t));
  size_t total_lut_entries = parlay::reduce(lut_entry_counts);

  // Read all LUT entry sizes
  parlay::sequence<size_t> all_lut_entry_sizes(total_lut_entries);
  infile.read(reinterpret_cast<char *>(all_lut_entry_sizes.begin()),
              total_lut_entries * sizeof(size_t));

  // Reconstruct the clusters
  clusters.resize(num);
  auto [lut_entry_counts_scan, total_entries_check] = parlay::scan(lut_entry_counts);

  parlay::parallel_for(0, num, [&](size_t i) {
    auto point_id_slice = all_point_ids.cut(point_id_offsets[i], point_id_offsets[i + 1]);
    clusters[i].point_ids = parlay::sequence<size_t>(point_id_slice.begin(), point_id_slice.end());

    // Reconstruct leaf centroids
    auto leaf_centroids_data_slice =
        all_leaf_centroids.cut(leaf_centroid_offsets[i] * d, leaf_centroid_offsets[i + 1] * d);
    parlay::sequence<parlay::sequence<float>> temp_centroids_seq(leaf_centroids_data_slice.size() /
                                                                 d);
    parlay::parallel_for(0, temp_centroids_seq.size(), [&](size_t j) {
      auto centroid_slice = leaf_centroids_data_slice.cut(j * d, (j + 1) * d);
      temp_centroids_seq[j] = parlay::sequence<float>(centroid_slice.begin(), centroid_slice.end());
    });
    clusters[i].centroids = Range(temp_centroids_seq, d);

    // Reconstruct LUTs
    size_t num_entries_in_cluster = lut_entry_counts[i];
    clusters[i].LUT.resize(num_entries_in_cluster);

    size_t start_entry_idx = lut_entry_counts_scan[i];
    size_t current_data_offset = lut_offsets[i];

    for (size_t j = 0; j < num_entries_in_cluster; ++j) {
      size_t entry_size = all_lut_entry_sizes[start_entry_idx + j];
      auto entry_data_slice =
          all_lut_data.cut(current_data_offset, current_data_offset + entry_size);
      clusters[i].LUT[j] =
          parlay::sequence<uint8_t>(entry_data_slice.begin(), entry_data_slice.end());
      current_data_offset += entry_size;
    }
  });

  infile.close();
}

// Returns the mean cluster size
template<bool metric>
size_t IndexMVIVFFlatMVQ<metric>::mean_cluster_size() const noexcept {
  auto cluster_sizes = parlay::delayed_seq<size_t>(
      clusters.size(), [&](size_t i) { return clusters[i].point_ids.size(); });
  return parlay::reduce(cluster_sizes) / clusters.size();
}

// Returns the mean cluster size
template<bool metric>
size_t IndexMVIVFFlatMVQ<metric>::max_cluster_size() const noexcept {
  auto cluster_sizes = parlay::delayed_seq<size_t>(
      clusters.size(), [&](size_t i) { return clusters[i].point_ids.size(); });
  return parlay::reduce(cluster_sizes, parlay::maxm<size_t>());
}

using IndexMVIVFFlatMVQL2 = IndexMVIVFFlatMVQ<true>;   // L2 metric
using IndexMVIVFFlatMVQIP = IndexMVIVFFlatMVQ<false>;  // MIPS

}  // namespace mvivf