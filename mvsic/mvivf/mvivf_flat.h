#pragma once

#include <queue>
#include <set>

#include "mvsic/core/index.h"
#include "mvsic/core/mvclustering/mvclustering.h"
#include "mvsic/core/utils/top_neighbors.h"
#include "mvsic/core/utils/util.h"

namespace mvsic {

/* Multi-Vector IVF Index: Flat version (MVIVF_Flat)
  Indexing:
  - Runs the MV-Lloyd's algorithm to cluster the input point clouds into `num_clusters` clusters.
    Each cluster is now represented by a "center" point cloud.
  Search:
  - For a given query point cloud, it computes distances to all centers point clouds,
   and then probes the top `nprobes` clusters.
  Params:
  - k_per_level: Number of clusters (num_clusters)
*/

template<bool metric>
class IndexMVIVFFlat : public Index<metric> {
 public:
  using ChPoint = typename Index<metric>::ChPoint;  // Chamfer Point Type
  using Index<metric>::d;                           // Embedding dimension

  IndexParams params;
  PointCloudSet<ChPoint> centers;                          // Centers of clusters
  parlay::sequence<PointCloudSet<ChPoint>> clusters = {};  // Clusters of points

  IndexMVIVFFlat(uint32_t d_) noexcept : params(IndexParams::mvivf_flat()) { d = d_; }
  IndexMVIVFFlat(uint32_t d_, const IndexParams &params) noexcept : params(params) { d = d_; }

  // Builds the index given PointCloudSet object.
  void build(const PointCloudSet<ChPoint> &points) override {
    size_t n = points.size();
    size_t num_clusters = (params.k_per_level > 0) ? params.k_per_level : std::ceil(std::sqrt(n));
    if (params.verbose >= 1) {
      std::cout << "Building index with " << n << " points, num_clusters: " << num_clusters
                << std::endl;
    }
    if (params.compress_input) {
      // TODO: run Ward's HAC to compress input point clouds
    }
    // Step 1: Run MV-Lloyds on points
    MVClustering<metric> Clus(d, num_clusters, params.s, params.mvclus);
    Clus.train(points);
    centers = std::move(Clus.centers);
    parlay::sequence<uint32_t> &cluster_ids = Clus.cluster_ids;
    // Step 2: Collect Clusters
    auto id_pt = parlay::tabulate(n, [&](uint32_t i) { return std::make_pair(cluster_ids[i], i); });
    auto grouped = group_by_key_inplace(id_pt);
    // Step 3: Collect point clouds by clusters
    clusters.resize(num_clusters);
    parlay::parallel_for(
        0, grouped.size(),
        [&](size_t i) {
          auto cluster_id = grouped[i][0].first;
          auto group = parlay::delayed_seq<uint32_t>(
              grouped[i].size(), [&](size_t j) { return grouped[i][j].second; });
          PointCloudSet<ChPoint> cluster_points = PointCloudSet<ChPoint>(points.filter(group), d);
          clusters[cluster_id] = std::move(cluster_points);
        },
        1);
    if (params.use_PQ) {
      // TODO: PQ
    }
    if (params.verbose >= 1) {
      std::cout << "Index built. Mean cluster size: " << mean_cluster_size()
                << ", Max cluster size: " << max_cluster_size() << std::endl;
    }
  }

  // Returns the top-k point clouds for the query point cloud
  // Output format: < [<id, distance>, ...], # distance comparisons>
  std::pair<parlay::sequence<std::pair<uint32_t, float>>, size_t> search(
      const ChPoint &query, const PointCloudSet<ChPoint> &points,
      const SearchParams &search_params) override {
    size_t k = search_params.k;
    size_t nprobes = search_params.nprobes;
    size_t dist_cmps = 0;
    // Step 1: Compute distances to centers
    parlay::sequence<std::pair<uint32_t, float>> id_dist;
    std::tie(id_dist, dist_cmps) = centers.distances(query);
    parlay::sort_inplace(id_dist, [](const auto &a, const auto &b) { return a.second < b.second; });

    // Step 2: Probe top nprobe clusters
    // Find the minimum number of probes needed to obtain k neighbors
    size_t nprobes_minimal = 0, cur = 0;
    while (nprobes_minimal < id_dist.size() && cur < k) {
      cur += clusters[id_dist[nprobes_minimal].first].size();
      nprobes_minimal++;
    }
    nprobes = std::min(id_dist.size(), std::max(nprobes, nprobes_minimal));
    auto sizes = parlay::delayed_tabulate(
        nprobes, [&](size_t i) { return clusters[id_dist[i].first].size(); });
    auto scan_result = parlay::scan(sizes);
    auto &offsets = scan_result.first;
    size_t total_size = scan_result.second;
    auto visited = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(total_size);
    auto leaf_dist_cmps = parlay::sequence<size_t>::uninitialized(nprobes);
    parlay::parallel_for(0, nprobes, [&](size_t i) {
      uint32_t cluster_id = id_dist[i].first;
      leaf_dist_cmps[i] =
          get_knn_into_uninitialized<ChPoint>(query, clusters[cluster_id], &visited[offsets[i]]);
    });
    dist_cmps += parlay::reduce(leaf_dist_cmps);
    parlay::sort_inplace(visited, [](const auto &a, const auto &b) {
      return a.second < b.second;  // Sort by distance
    });
    // Step 3: Re-ranking
    auto final_results =
        parlay::sequence<std::pair<uint32_t, float>>::uninitialized(std::min(k, visited.size()));
    // auto results_d = parlay::sequence<double>::from_function(results.size(), [&](size_t i) {
    //   return packFloatAndInt(results[i].second, results[i].first);
    // });
    // // VQSort(id_dist_d.begin(), id_dist_d.end());
    // VQPartialSort(results_d.begin(), results_d.end(), std::min(k, results_d.size()));
    // auto final_results = parlay::sequence<std::pair<size_t, float>>::from_function(
    //     std::min(k, results.size()), [&](size_t i) {
    //       auto [ext_float, ind] = unpackDouble2(results_d[i]);
    //       return std::make_pair(ind, ext_float);
    //     });
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
    return std::make_pair(final_results, dist_cmps);
  }

  // Write the index to a file in disk
  void save(const std::string &filename) override {
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
          uint32_t point_id = clusters[i].get_id(j);
          outfile.write(reinterpret_cast<const char *>(&point_id), sizeof(uint32_t));
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
    parlay::sequence<uint32_t> clusters_values(clusters_offsets[clusters_offsets.size() - 1]);
    infile.read(reinterpret_cast<char *>(clusters_values.begin()),
                clusters_values.size() * sizeof(uint32_t));

    // Build the index
    size_t dim = points.get_dims();
    auto point_id_to_data_id = parlay::sequence<uint32_t>::uninitialized(points.size());
    parlay::parallel_for(0, points.size(),
                         [&](uint32_t i) { point_id_to_data_id[points.get_id(i)] = i; });
    parlay::sequence<size_t> clusters_sizes = parlay::sequence<size_t>::from_function(
        num, [&](size_t i) { return clusters_offsets[i + 1] - clusters_offsets[i]; });
    centers =
        PointCloudSet<ChPoint>(num, dim, center_values.data(), center_offsets.data(), nullptr);
    clusters.resize(num);
    parlay::parallel_for(0, num, [&](size_t i) {
      if (clusters_sizes[i] > 0) {
        auto cluster_group = parlay::delayed_seq<uint32_t>(clusters_sizes[i], [&](size_t j) {
          uint32_t point_id = clusters_values[clusters_offsets[i] + j];
          return point_id_to_data_id[point_id];
        });
        clusters[i] = PointCloudSet<ChPoint>(points.filter(cluster_group), dim);
      }
    });
    infile.close();
  }

  size_t mean_cluster_size() const noexcept override {
    auto cluster_sizes =
        parlay::delayed_seq<size_t>(clusters.size(), [&](size_t i) { return clusters[i].size(); });
    return parlay::reduce(cluster_sizes) / clusters.size();
  }

  size_t max_cluster_size() const noexcept override {
    auto cluster_sizes =
        parlay::delayed_seq<size_t>(clusters.size(), [&](size_t i) { return clusters[i].size(); });
    return parlay::reduce(cluster_sizes, parlay::maxm<size_t>());
  }
};

using IndexMVIVFFlatL2 = IndexMVIVFFlat<true>;   // Instantiates for L2 metric (metric = true)
using IndexMVIVFFlatIP = IndexMVIVFFlat<false>;  // Instantiates for MIPS      (metric = false)

}  // namespace mvsic