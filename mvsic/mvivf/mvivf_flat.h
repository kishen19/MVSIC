#pragma once

#include <fstream>
#include <type_traits>
#include <variant>

#include "mvsic/core/index.h"
#include "mvsic/core/mvclustering/mvclustering.h"
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
  using MVQT = typename Index<metric>::MVQT;
  using QuantSet = typename MVQT::QuantSet;
  using QuantQuery = typename MVQT::QuantQuery;
  using QuantModel = typename MVQT::QuantModel;
  using TQ_Set = typename MVQT::TQ_Set;
  using TQ_Query = typename MVQT::TQ_Query;
  using TQ_Model = typename MVQT::TQ_Model;
  using QT = typename Index<metric>::QT;
  using Index<metric>::d;       // Embedding dimension
  using Index<metric>::params;  // Index Params
  using Index<metric>::quantization_mode;

  struct node_t {
    PointCloudSet<ChPoint> data;
    QuantSet quantized_data;

    node_t() noexcept : data(), quantized_data(std::monostate{}) {}
    inline size_t get_size() const noexcept { return data.size(); }
  };

  PointCloudSet<ChPoint> centers;  // Centers of clusters
  TQ_Set centers_quant;
  parlay::sequence<node_t> clusters = {};
  // Quantizer Storage
  QuantModel quantizer = std::monostate{};
  TQ_Model center_quantizer;

  IndexMVIVFFlat(uint32_t d_) noexcept {
    d = d_;
    params = IndexParams::mvivf_flat();
  }
  IndexMVIVFFlat(uint32_t d_, const IndexParams& params_) noexcept {
    d = d_;
    params = params_;
  }

  inline uint32_t get_size(size_t i) const noexcept { return clusters[i].get_size(); }

  inline size_t num_leaves() const noexcept { return centers.size(); }

  // Quantization Helpers
  void init_tq_quantizer(const PointCloudSet<ChPoint>& points) {
    if (!params.quantize_centers) return;
    center_quantizer.train(points);
  }

  TQ_Set encode_tq(const PointCloudSet<ChPoint>& points) {
    return center_quantizer.encode(points);  // returns TQ_Set
  }

  // Builds the index given PointCloudSet object.
  void build(const PointCloudSet<ChPoint>& points) override {
    parlay::internal::timer t;
    size_t n = points.size();
    size_t num_clusters = (params.k_per_level > 0) ? params.k_per_level : std::ceil(std::sqrt(n));
    if (params.verbose >= 1) {
      std::cout << "Building index with " << n << " points, num_clusters: " << num_clusters
                << std::endl;
    }

    // Quantization
    t.start();
    quantization_mode = params.pq.method;
    this->train_quantizer(points, quantizer);
    init_tq_quantizer(points);
    if (params.verbose >= 1 && quantization_mode != QT::None) {
      std::cout << "[MVIVF Flat] Quantizers Trained: " << t.stop() << " sec" << std::endl;
    }
    t.reset();

    // Run MV-Lloyds on points
    t.start();
    MVClustering<metric> Clus(d, num_clusters, params.s, params.mvclus);
    Clus.train(points);
    parlay::sequence<uint32_t> cluster_ids = Clus.get_clustering(points);
    centers = std::move(Clus.get_centers());  // Stealing from Clus, as we don't use it anymore
    // Collect Clusters
    auto id_pt = parlay::tabulate(n, [&](uint32_t i) { return std::make_pair(cluster_ids[i], i); });
    auto grouped = group_by_key_inplace(id_pt);
    // Collect point clouds by clusters
    clusters.resize(num_clusters);
    parlay::parallel_for(
        0, grouped.size(),
        [&](size_t i) {
          auto cluster_id = grouped[i][0].first;
          auto group = parlay::delayed_seq<uint32_t>(
              grouped[i].size(), [&](size_t j) { return grouped[i][j].second; });
          PointCloudSet<ChPoint> cluster_points = PointCloudSet<ChPoint>(points.filter(group), d);
          clusters[cluster_id].data = std::move(cluster_points);
          clusters[cluster_id].quantized_data =
              this->encode_points_quantized(clusters[cluster_id].data, quantizer);
        },
        1);
    if (params.verbose >= 1) {
      std::cout << "[MVIVF Flat] Index Built: " << t.stop() << " sec" << std::endl;
      std::cout << "[MVIVF Flat] Mean cluster size: " << mean_cluster_size()
                << ", Max cluster size: " << max_cluster_size() << std::endl;
    }
    t.reset();

    if (params.quantize_centers) {
      t.start();
      centers_quant = encode_tq(centers);
      std::cout << "[MVIVF Flat] Encoding Centers: " << t.stop() << " sec" << std::endl;
    }
  }

  inline auto process_probes(const ChPoint& query, const QuantQuery& q_query_var,
                             parlay::sequence<std::pair<uint32_t, float>>& probe_list) {
    const size_t nprobes = probe_list.size();
    auto sizes = parlay::delayed_tabulate(
        nprobes, [&](size_t i) { return clusters[probe_list[i].first].get_size(); });
    auto scan_result = parlay::scan(sizes);
    auto& offsets = scan_result.first;
    size_t total_size = scan_result.second;
    auto visited = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(total_size);
    auto bytes_accessed = parlay::sequence<size_t>::uninitialized(nprobes);

    auto process_probes_quant = [&]<typename SetType>(const auto& q_query) {
      parlay::parallel_for(0, nprobes, [&](size_t i) {
        node_t& node = clusters[probe_list[i].first];
        auto* node_data = std::get_if<SetType>(&node.quantized_data);
        if (!node_data) UNREACHABLE();
        node_data->distances_all(q_query, &visited[offsets[i]]);
        bytes_accessed[i] = node_data->num_bytes();
      });
    };

    switch (quantization_mode) {
      case QT::PQ:
        process_probes_quant.template operator()<typename MVQT::PQ_Set>(
            std::get<typename MVQT::PQ_Query>(q_query_var));
        break;
      case QT::RaBitQ:
        process_probes_quant.template operator()<typename MVQT::RQ_Set>(
            std::get<typename MVQT::RQ_Query>(q_query_var));
        break;
      case QT::FastScan:
        process_probes_quant.template operator()<typename MVQT::FS_Set>(
            std::get<typename MVQT::FS_Query>(q_query_var));
        break;
      case QT::TurboQuant:
        process_probes_quant.template operator()<typename MVQT::TQ_Set>(
            std::get<typename MVQT::TQ_Query>(q_query_var));
        break;
      case QT::SPQTQ:
        process_probes_quant.template operator()<typename MVQT::PQTQ_Set>(
            std::get<typename MVQT::PQTQ_Query>(q_query_var));
        break;
      case QT::None:
        parlay::parallel_for(0, nprobes, [&](size_t i) {
          node_t& node = clusters[probe_list[i].first];
          node.data.distances(query, &visited[offsets[i]]);
          bytes_accessed[i] = node.data.num_bytes();
        });
        break;
    }
    return std::make_pair(visited, parlay::reduce(bytes_accessed));
  }

  // Returns the top-k point clouds for the query point cloud
  // Output format: < [<id, distance>, ...], # distance comparisons>
  std::tuple<parlay::sequence<std::pair<uint32_t, float>>, size_t, std::vector<double>>
  search_with_stats(const ChPoint& query, const PointCloudSet<ChPoint>& points,
                    const SearchParams& search_params) override {
    parlay::internal::timer t;

    size_t k = search_params.k;
    size_t nprobes = search_params.nprobes;
    size_t bytes_accessed = 0;

    std::vector<double> stats;
    size_t dist_cmps = 0;
    double t_search = 0.0;
    double t_quantize = 0.0;
    double t_distances = 0.0;
    double t_rest = 0.0;
    double t_rerank = 0.0;

    // -------------------------
    // Step 0: Quantize Query
    // -------------------------
    QuantQuery q_query_var = this->quantize_query_point_cloud(query, quantizer);
    TQ_Query q_center_query;
    t.start();
    if (params.quantize_centers) {
      q_center_query = center_quantizer.quantize_query(query);
    }
    t_quantize = t.stop();
    t.reset();

    // -------------------------
    // Step 1: Compute distances to centers
    // -------------------------
    t.start();
    const size_t L = centers.size();
    nprobes = std::min(nprobes, L);
    auto probe_list = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(L);

    if (params.quantize_centers) {
      centers_quant.distances_all(q_center_query, probe_list.data());
      bytes_accessed += centers_quant.num_bytes();
    } else {
      centers.distances(query, probe_list.data());
      bytes_accessed += centers.num_bytes();
    }

    std::nth_element(probe_list.begin(), probe_list.begin() + nprobes, probe_list.end(),
                     [](const auto& a, const auto& b) { return a.second < b.second; });
    probe_list.resize(nprobes);
    t_search = t.stop();
    dist_cmps += L;
    t.reset();

    // -------------------------
    // Step 2: Probe top nprobe clusters
    // -------------------------
    t.start();
    parlay::sequence<std::pair<uint32_t, float>> visited;
    size_t bytes_accessed_pp;
    std::tie(visited, bytes_accessed_pp) = process_probes(query, q_query_var, probe_list);
    bytes_accessed += bytes_accessed_pp;
    t_distances = t.stop();

    t.start();
    mvsic::sort_inplace_kv(visited);
    t_rest += t.stop();
    t.reset();

    dist_cmps += visited.size();

    // -------------------------
    // Step 3: Re-ranking
    // -------------------------
    t.start();
    auto final_results =
        parlay::sequence<std::pair<uint32_t, float>>::uninitialized(std::min(k, visited.size()));
    if (search_params.num_rerank > 0) {
      size_t num_rerank = std::min(search_params.num_rerank, visited.size());
      bytes_accessed += this->rerank(query, points, visited, num_rerank, final_results);
    } else {
      parlay::parallel_for(0, final_results.size(),
                           [&](size_t i) { final_results[i] = visited[i]; });
    }
    t_rerank = t.stop();
    t.reset();

    stats.push_back(static_cast<size_t>(L));
    stats.push_back(static_cast<double>(dist_cmps));
    stats.push_back(t_search);
    stats.push_back(t_distances);
    stats.push_back(t_rest);
    stats.push_back(t_rerank);

    return std::make_tuple(final_results, bytes_accessed, stats);
  }

  // Write the index to a file in disk
  void save(const std::string& filename) override {
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
    outfile.write(reinterpret_cast<const char*>(&num), sizeof(size_t));
    // Write center offsets
    outfile.write(reinterpret_cast<const char*>(center_offsets.begin()),
                  center_offsets.size() * sizeof(size_t));
    // Write centers values
    auto coords = centers.data();
    size_t num_entries = (centers.total_size()) * (centers.get_dims());
    outfile.write(reinterpret_cast<const char*>(coords), num_entries * sizeof(float));

    parlay::sequence<size_t> clusters_offsets = parlay::sequence<size_t>::from_function(
        clusters.size(), [&](size_t i) { return clusters[i].get_size(); });
    size_t total_clusters_size = parlay::scan_inplace(clusters_offsets);
    clusters_offsets.push_back(total_clusters_size);
    // Write clusters offsets
    outfile.write(reinterpret_cast<const char*>(clusters_offsets.begin()),
                  clusters_offsets.size() * sizeof(size_t));
    // Write clusters values
    for (size_t i = 0; i < num; ++i) {
      if (clusters[i].get_size() > 0) {
        for (size_t j = 0; j < clusters[i].get_size(); ++j) {
          uint32_t point_id = clusters[i].data.get_id(j);
          outfile.write(reinterpret_cast<const char*>(&point_id), sizeof(uint32_t));
        }
      }
    }

    // quantizer MODEL only
    const int type_id = static_cast<int>(quantization_mode);
    outfile.write(reinterpret_cast<const char*>(&type_id), sizeof(int));

    std::visit(
        [&](auto& model) {
          using ModelType = std::decay_t<decltype(model)>;
          if constexpr (std::is_same_v<ModelType, std::monostate>) {
            return;
          } else {
            model.save(outfile);
          }
        },
        quantizer);

    outfile.close();
  }

  // Read the index from a file in disk
  void load(const std::string& filename, const PointCloudSet<ChPoint>& points) override {
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
    size_t num_center_offsets = num + 1;
    parlay::sequence<size_t> center_offsets(num_center_offsets);
    infile.read(reinterpret_cast<char*>(center_offsets.begin()),
                center_offsets.size() * sizeof(size_t));
    // Read centers values
    parlay::sequence<float> center_values(center_offsets[center_offsets.size() - 1]);
    infile.read(reinterpret_cast<char*>(center_values.begin()),
                center_values.size() * sizeof(float));

    size_t dim = points.get_dims();
    centers =
        PointCloudSet<ChPoint>(num, dim, center_values.data(), center_offsets.data(), nullptr);

    // Read clusters offsets
    parlay::sequence<size_t> clusters_offsets(num + 1);
    infile.read(reinterpret_cast<char*>(clusters_offsets.begin()),
                clusters_offsets.size() * sizeof(size_t));
    // Read clusters values
    parlay::sequence<uint32_t> clusters_values(clusters_offsets[clusters_offsets.size() - 1]);
    infile.read(reinterpret_cast<char*>(clusters_values.begin()),
                clusters_values.size() * sizeof(uint32_t));

    // quantizer model
    int type_id = 0;
    infile.read(reinterpret_cast<char*>(&type_id), sizeof(int));
    quantization_mode = static_cast<QT>(type_id);

    switch (quantization_mode) {
      case QT::PQ: quantizer.template emplace<typename MVQT::PQ_Model>().load(infile); break;
      case QT::RaBitQ: quantizer.template emplace<typename MVQT::RQ_Model>().load(infile); break;
      case QT::FastScan: quantizer.template emplace<typename MVQT::FS_Model>().load(infile); break;
      case QT::TurboQuant:
        quantizer.template emplace<typename MVQT::TQ_Model>().load(infile);
        break;
      case QT::SPQTQ:
        quantizer.template emplace<typename MVQT::PQTQ_Model>().load(infile);
        break;
      case QT::None:
      default: quantizer = std::monostate{}; break;
    }
    infile.close();

    // Center quantization (internal-node / leaf-center scoring): always TQ.
    init_tq_quantizer(points);
    if (params.quantize_centers) {
      centers_quant = encode_tq(centers);
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
        clusters[i].data = PointCloudSet<ChPoint>(points.filter(cluster_group), dim);
        clusters[i].quantized_data = this->encode_points_quantized(clusters[i].data, quantizer);
      }
    });
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