#pragma once

#include <queue>
#include <set>

#include "mvsic/core/index.h"
#include "mvsic/core/mvclustering/mvclustering.h"
#include "mvsic/core/utils/util.h"

// New Quantization & Wrapper Headers
#include "mvsic/core/quantization/pq.h"
#include "mvsic/core/quantization/rabitq.h"
#include "mvsic/core/quantization/scann.h"
#include "mvsic/core/quantization/fastscan.h"
#include "mvsic/core/quantization/wrapper.h"

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

  // Multi-Vector Quantizer Types
  using FlatRange = FlattenedPCRange<PointCloudSet<ChPoint>>;
  using PQ_Set = Quantized_Point_Cloud_Set<pq::Quantized_Point_Range<FlatRange, metric>, metric>;
  using RaBitQ_Set =
      Quantized_Point_Cloud_Set<rabitq::Quantized_Point_Range<FlatRange, metric>, metric>;
  using ScaNN_Set = Quantized_Point_Cloud_Set<pq::ScaNN_Point_Range<FlatRange, metric>, metric>;
  using FastScan_Set =
      Quantized_Point_Cloud_Set<fastscan::Quantized_Point_Range<FlatRange, metric>, metric>;
  using QT = IndexParams::QuantizerType;

  IndexParams params;
  PointCloudSet<ChPoint> centers;  // Centers of clusters
  parlay::sequence<PointCloudSet<ChPoint>> clusters = {};

  // Quantizer Storage (Active based on params)
  std::optional<PQ_Set> quantizer_pq;
  std::optional<RaBitQ_Set> quantizer_rabitq;
  std::optional<ScaNN_Set> quantizer_scann;
  std::optional<FastScan_Set> quantizer_fastscan;
  QT active_quantizer = QT::None;

  IndexMVIVFFlat(uint32_t d_) noexcept : params(IndexParams::mvivf_flat()) { d = d_; }
  IndexMVIVFFlat(uint32_t d_, const IndexParams &params) noexcept : params(params) { d = d_; }

  inline uint32_t get_size(size_t i) const noexcept { return clusters[i].size(); }

  inline size_t num_leaves() const noexcept { return centers.size(); }

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

    // Quantization
    active_quantizer = params.pq.method;
    switch (active_quantizer) {
      case QT::RaBitQ: quantizer_rabitq.emplace(points, params.pq.rabitq_bits); break;
      case QT::ScaNN:
        quantizer_scann.emplace(points, params.pq.block_size, params.pq.num_clusters_per_block,
                                params.pq.num_points_per_cluster, params.pq.scann_threshold);
        break;
      case QT::PQ:
        quantizer_pq.emplace(points, params.pq.block_size, params.pq.num_clusters_per_block,
                             params.pq.num_points_per_cluster);
        break;
      case QT::FastScan: quantizer_fastscan.emplace(points, params.pq.block_size); break;
      case QT::None: break;
      default: std::cout << "Error: Unknown quantization method." << std::endl; abort();
    }

    if (params.verbose >= 1) {
      std::cout << "Index built. Mean cluster size: " << mean_cluster_size()
                << ", Max cluster size: " << max_cluster_size() << std::endl;
    }
  }

  // Returns the top-k point clouds for the query point cloud
  // Output format: < [<id, distance>, ...], # distance comparisons>
  std::tuple<parlay::sequence<std::pair<uint32_t, float>>, size_t, std::vector<double>>
  search_with_stats(const ChPoint &query, const PointCloudSet<ChPoint> &points,
                    const SearchParams &search_params) override {
    parlay::internal::timer t;
    std::vector<double> timings;

    size_t k = search_params.k;
    size_t nprobes = search_params.nprobes;
    size_t dist_cmps = 0;
    // Step 1: Compute distances to centers
    t.start();
    parlay::sequence<std::pair<uint32_t, float>> id_dist;
    std::tie(id_dist, dist_cmps) = centers.distances(query);
    parlay::sort_inplace(id_dist, [](const auto &a, const auto &b) { return a.second < b.second; });
    timings.push_back(t.stop());
    t.reset();

    // Step 2: Probe top nprobe clusters
    t.start();
    // Find the minimum number of probes needed to obtain k neighbors
    size_t nprobes_minimal = 0, cur = 0;
    while (nprobes_minimal < id_dist.size() && cur < k) {
      cur += get_size(id_dist[nprobes_minimal].first);  // size of cluster
      nprobes_minimal++;
    }
    nprobes = std::min(id_dist.size(), std::max(nprobes, nprobes_minimal));
    // Allocate space for storing results of each probe cluster
    auto sizes =
        parlay::delayed_tabulate(nprobes, [&](size_t i) { return get_size(id_dist[i].first); });
    auto scan_result = parlay::scan(sizes);
    auto &offsets = scan_result.first;
    size_t total_size = scan_result.second;
    auto visited = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(total_size);

    switch (active_quantizer) {
      case QT::RaBitQ: {
        auto q_query = quantizer_rabitq->quantize_query(query);
        parlay::parallel_for(0, nprobes, [&](size_t i) {
          uint32_t cluster_id = id_dist[i].first;
          auto &leaf_points = clusters[cluster_id];
          parlay::sequence<uint32_t> leaf_indices =
              parlay::tabulate(leaf_points.size(), [&](size_t j) { return leaf_points.get_id(j); });
          quantizer_rabitq->distances(q_query, leaf_indices.data(), leaf_indices.size(),
                                      &visited[offsets[i]]);
        });
        // TODO: fix this
        // dist_cmps += total_size;
        break;
      }
      case QT::ScaNN: {
        auto q_query = quantizer_scann->quantize_query(query);
        parlay::parallel_for(0, nprobes, [&](size_t i) {
          uint32_t cluster_id = id_dist[i].first;
          auto &leaf_points = clusters[cluster_id];
          parlay::sequence<uint32_t> leaf_indices =
              parlay::tabulate(leaf_points.size(), [&](size_t j) { return leaf_points.get_id(j); });
          quantizer_scann->distances(q_query, leaf_indices.data(), leaf_indices.size(),
                                     &visited[offsets[i]]);
        });
        // TODO: fix this
        // dist_cmps += total_size;
        break;
      }
      case QT::PQ: {
        auto q_query = quantizer_pq->quantize_query(query);
        parlay::parallel_for(0, nprobes, [&](size_t i) {
          uint32_t cluster_id = id_dist[i].first;
          auto &leaf_points = clusters[cluster_id];
          parlay::sequence<uint32_t> leaf_indices =
              parlay::tabulate(leaf_points.size(), [&](size_t j) { return leaf_points.get_id(j); });
          quantizer_pq->distances(q_query, leaf_indices.data(), leaf_indices.size(),
                                  &visited[offsets[i]]);
        });
        // TODO: fix this
        // dist_cmps += total_size;
        break;
      }
      case QT::FastScan: {
        auto q_query = quantizer_fastscan->quantize_query(query);
        parlay::parallel_for(0, nprobes, [&](size_t i) {
          uint32_t cluster_id = id_dist[i].first;
          auto &leaf_points = clusters[cluster_id];
          parlay::sequence<uint32_t> leaf_indices =
              parlay::tabulate(leaf_points.size(), [&](size_t j) { return leaf_points.get_id(j); });
          quantizer_fastscan->distances(q_query, leaf_indices.data(), leaf_indices.size(),
                                        &visited[offsets[i]]);
        });
        // TODO: fix this
        // dist_cmps += total_size;
        break;
      }
      case QT::None: {
        auto leaf_dist_cmps = parlay::sequence<size_t>::uninitialized(nprobes);
        parlay::parallel_for(0, nprobes, [&](size_t i) {
          uint32_t cluster_id = id_dist[i].first;
          leaf_dist_cmps[i] = clusters[cluster_id].distances(query, &visited[offsets[i]]);
        });
        dist_cmps += parlay::reduce(leaf_dist_cmps);  // Add total non-PQ distance comparisons
        break;
      }
      default: {
        std::cout << "Error: Unknown quantization method." << std::endl;
        abort();
      }
    }
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
      dist_cmps += this->rerank(query, points, visited, num_rerank, final_results);
    } else {
      parlay::parallel_for(0, final_results.size(),
                           [&](size_t i) { final_results[i] = visited[i]; });
    }
    timings.push_back(t.stop());
    t.reset();

    return std::make_tuple(final_results, dist_cmps, timings);
  }

  // Write the index to a file in disk
  void save(const std::string &filename) override {
    std::ofstream outfile(filename, std::ios::binary);
    std::cout << "Saving index to " << filename << std::endl;
    if (!outfile.is_open()) {
      std::cerr << "Error opening file for writing: " << filename << std::endl;
      return;
    }

    // Collect data for centers
    parlay::sequence<size_t> center_offsets = parlay::sequence<size_t>::from_function(
        centers.size(), [&](size_t i) { return centers.get_size(i) * d; });
    size_t total_center_sizes = parlay::scan_inplace(center_offsets);
    center_offsets.push_back(total_center_sizes);

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

    parlay::sequence<size_t> clusters_offsets = parlay::sequence<size_t>::from_function(
        clusters.size(), [&](size_t i) { return clusters[i].size(); });
    size_t total_clusters_size = parlay::scan_inplace(clusters_offsets);
    clusters_offsets.push_back(total_clusters_size);
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

    // Quantization
    int type_id = static_cast<int>(active_quantizer);
    outfile.write((char *)&type_id, sizeof(int));
    switch (active_quantizer) {
      case QT::RaBitQ: quantizer_rabitq->save(outfile); break;
      case QT::ScaNN: quantizer_scann->save(outfile); break;
      case QT::PQ: quantizer_pq->save(outfile); break;
      case QT::FastScan: quantizer_fastscan->save(outfile); break;
      default: break;
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

    size_t dim = points.get_dims();
    centers =
        PointCloudSet<ChPoint>(num, dim, center_values.data(), center_offsets.data(), nullptr);

    // Read clusters offsets
    parlay::sequence<size_t> clusters_offsets(num + 1);
    infile.read(reinterpret_cast<char *>(clusters_offsets.begin()),
                clusters_offsets.size() * sizeof(size_t));
    // Read clusters values
    parlay::sequence<uint32_t> clusters_values(clusters_offsets[clusters_offsets.size() - 1]);
    infile.read(reinterpret_cast<char *>(clusters_values.begin()),
                clusters_values.size() * sizeof(uint32_t));

    // Quantization
    int type_id;
    infile.read((char *)&type_id, sizeof(int));
    active_quantizer = static_cast<QT>(type_id);
    switch (active_quantizer) {
      case QT::RaBitQ:
        quantizer_rabitq.emplace();
        quantizer_rabitq->load(infile);
        break;
      case QT::ScaNN:
        quantizer_scann.emplace();
        quantizer_scann->load(infile);
        break;
      case QT::PQ:
        quantizer_pq.emplace();
        quantizer_pq->load(infile);
        break;
      case QT::FastScan:
        quantizer_fastscan.emplace();
        quantizer_fastscan->load(infile);
        break;
      default: break;
    }

    // Build the index
    auto point_id_to_data_id = parlay::sequence<uint32_t>::uninitialized(points.size());
    parlay::parallel_for(0, points.size(),
                         [&](uint32_t i) { point_id_to_data_id[points.get_id(i)] = i; });
    parlay::sequence<size_t> clusters_sizes = parlay::sequence<size_t>::from_function(
        num, [&](size_t i) { return clusters_offsets[i + 1] - clusters_offsets[i]; });
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
        parlay::delayed_seq<size_t>(num_leaves(), [&](size_t i) { return get_size(i); });
    return parlay::reduce(cluster_sizes) / num_leaves();
  }

  size_t max_cluster_size() const noexcept override {
    auto cluster_sizes =
        parlay::delayed_seq<size_t>(num_leaves(), [&](size_t i) { return get_size(i); });
    return parlay::reduce(cluster_sizes, parlay::maxm<size_t>());
  }
};

using IndexMVIVFFlatL2 = IndexMVIVFFlat<true>;   // Instantiates for L2 metric (metric = true)
using IndexMVIVFFlatIP = IndexMVIVFFlat<false>;  // Instantiates for MIPS      (metric = false)

}  // namespace mvsic