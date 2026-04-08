#pragma once

#include <fstream>
#include <optional>
#include <queue>
#include <set>
#include <unordered_map>

#include "mvsic/core/index.h"
#include "mvsic/core/utils/util.h"
#include "mvsic/core/utils/kmeans_util.h"
#include "mvsic/core/types/point_range.h"
#include "mvsic/core/types/ip_point.h"
#include "mvsic/core/types/l2_point.h"

namespace mvsic {

/* Single-Vector Heuristic with IVF
  Indexing:
  - Builds a kmeans Tree by recursively running Lloyd's algorithm on the input point clouds.
  - At each level, if the cluster size is larger than `maxsize`, it runs Lloyd's algorithm on that
  cluster with `num_clusters` clusters. Each cluster is represented by its center point.
  Search:
*/
template<bool metric>
class IndexSVHIVF : public Index<metric> {
 public:
  using ChPoint = typename Index<metric>::ChPoint;  // Chamfer Point Type
  using Point = std::conditional_t<metric, mvsic::L2_Point<float>, mvsic::IP_Point<float>>;
  using Range = mvsic::PointRange<float, Point>;
  using FlatRange = FlattenedPCRange<PointCloudSet<ChPoint>>;
  using SVQT = QuantTypes<metric, Range>;
  using QuantRange = typename SVQT::QuantRange;
  using QuantQuery = typename SVQT::QuantQuery;
  using QuantModel = typename SVQT::QuantModel;
  using TQ_Range = typename SVQT::TQ_Range;
  using TQ_Query = typename SVQT::TQ_Query;
  using TQ_Model = typename SVQT::TQ_Model;
  using QT = typename Index<metric>::QT;
  using Index<metric>::d;       // Embedding dimension
  using Index<metric>::params;  // Index Params
  using Index<metric>::quantization_mode;

  struct node_t {
    parlay::sequence<node_t*> children;
    // For internal nodes: data = centers of children
    // For leaves:         data = points in the cluster
    Range data;
    QuantRange quantized_data;
    parlay::sequence<std::pair<size_t, size_t>> ids;  // Only leaf nodes have ids
    node_t() noexcept : children(), data(), quantized_data(std::monostate()), ids() {}
    ~node_t() noexcept {}
    inline size_t get_size() const noexcept { return data.size(); }
    inline size_t get_id(size_t i) const noexcept { return ids[i].first; }
  };

  node_t* root = nullptr;  // Root of the k-means tree

  // Quantizer Storage
  QuantModel quantizer = std::monostate{};
  TQ_Model center_quantizer;

  IndexSVHIVF(size_t d_) noexcept {
    d = d_;
    params = IndexParams::svh_ivf();
  }
  IndexSVHIVF(size_t d_, const IndexParams& params_) noexcept {
    d = d_;
    params = params_;
  }

  // Quantization Helpers
  template<typename PR>
  void init_tq_quantizer(const PR& points) {
    if (!params.quantize_centers) return;
    center_quantizer.train(points);
  }

  template<typename PR>
  TQ_Range encode_tq(const PR& points) {
    return center_quantizer.encode(points);  // returns TQ_Set
  }

  // Recursively builds the kmeans tree
  void recursive_build(node_t* node, const parlay::sequence<parlay::sequence<float>>& points,
                       const parlay::sequence<std::pair<size_t, size_t>>& ids) {
    size_t n = points.size();
    size_t auto_nc = (params.k_per_level > 0) ? params.k_per_level
                                              : static_cast<size_t>(std::ceil(std::sqrt(n)));
    size_t small_nc = 4 * (n + params.max_leaf_size - 1) / params.max_leaf_size;
    size_t num_clusters = std::max(static_cast<size_t>(4), std::min({auto_nc, small_nc, n}));
    if (params.verbose >= 1) {
      std::cout << "[SVHIVF] Building index with " << n << " points, num_clusters: " << num_clusters
                << std::endl;
    }
    // Step 1: Run k-means on data and collect clusters
    auto kmeans_out =
        kmeans_subsample_assign<metric>(points, num_clusters, params.max_points_per_centroid,
                                        params.max_leaf_size, params.verbose >= 2);
    auto& centers = std::get<0>(kmeans_out);
    auto& cluster_ids = std::get<1>(kmeans_out);
    auto& active_indices = std::get<2>(kmeans_out);
    // Step 2: Collect Clusters
    auto id_pt = parlay::delayed_seq<std::pair<size_t, size_t>>(
        n, [&](size_t i) { return std::make_pair(cluster_ids[i], i); });
    auto grouped = parlay::group_by_index(id_pt, num_clusters);
    // Step 3: Update children nodes and recurse for large nodes
    node->children.resize(active_indices.size());
    if (active_indices.size() < centers.size()) {
      auto active_centers = parlay::delayed_tabulate(
          active_indices.size(), [&](size_t i) { return centers[active_indices[i]]; });
      node->data = Range(active_centers, d);
    } else {
      node->data = Range(centers, d);
    }
    // Quantize centers: Encoding
    if (params.quantize_centers) {
      node->quantized_data = encode_tq(node->data);
    }
    parlay::parallel_for(
        0, active_indices.size(),
        [&](size_t id) {
          size_t i = active_indices[id];
          node_t* child = new node_t();
          auto child_points =
              parlay::tabulate(grouped[i].size(), [&](size_t j) { return points[grouped[i][j]]; });
          auto child_ids =
              parlay::tabulate(grouped[i].size(), [&](size_t j) { return ids[grouped[i][j]]; });
          if (child_points.size() > params.max_leaf_size) {  // Recurse
            recursive_build(child, child_points, child_ids);
          } else {  // Leaf Node
            child->data = Range(child_points, d);
            child->ids = child_ids;
            child->quantized_data =
                this->template encode_range_quantized<SVQT>(child->data, quantizer);
          }
          node->children[id] = child;
        },
        1);
  }

  void build(const PointCloudSet<ChPoint>& points) override {
    parlay::internal::timer t;
    t.start();
    root = new node_t();
    // Prepare data for building
    size_t n = points.size();
    auto iota = parlay::iota(n);
    parlay::sequence<parlay::sequence<float>> data = points.filter_flattened(iota);
    auto ids = parlay::sequence<std::pair<size_t, size_t>>::uninitialized(points.total_size());
    auto num_embs = parlay::delayed_tabulate(n, [&](size_t i) { return points.get_size(i); });
    auto offsets = parlay::scan(num_embs).first;
    parlay::parallel_for(0, n, [&](size_t i) {
      for (size_t j = 0; j < num_embs[i]; ++j) {
        ids[offsets[i] + j] = {i, j};
      }
    });

    // Quantization: Training
    quantization_mode = params.pq.method;
    FlatRange flat_points(points);
    this->template train_quantizer<SVQT>(flat_points, quantizer);
    init_tq_quantizer(flat_points);

    recursive_build(root, data, ids);
    if (params.verbose >= 1) {
      std::cout << "[SVHIVF] Leaf-data computed: " << t.stop() << " sec" << std::endl;
    }
  }

  // Output type of Greedy Search
  struct GreedySearchResult {
    parlay::sequence<std::pair<float, node_t*>> probe_list;
    size_t dist_cmps = 0;
    size_t bytes_accessed = 0;
    std::vector<double> timings = {};
  };

  // Simple Beam Search using std::set
  GreedySearchResult greedy_search(const Point& query_point, const TQ_Query& q_query,
                                   size_t nprobes) const {
    using score_node = std::pair<float, node_t*>;
    const size_t beam_length = 2 * nprobes;
    parlay::internal::timer t;
    double t_dists = 0.0;
    double t_beam = 0.0;
    double t_rest = 0.0;

    t.start();
    size_t dist_cmps = 0;
    size_t bytes_accessed = 0;
    std::set<score_node> beam;
    parlay::sequence<score_node> top_probes;
    top_probes.reserve(nprobes + 1);
    std::vector<float> child_dists;
    child_dists.reserve(root->children.size());
    t_rest += t.stop();
    t.reset();

    // Initial
    t.start();
    beam.insert({0.0f, root});
    t_beam += t.stop();
    t.reset();
    while (!beam.empty()) {
      // Pop the best node (smallest distance)
      t.start();
      auto it = beam.begin();
      score_node best = *it;
      beam.erase(it);
      t_beam += t.stop();
      t.reset();
      t.start();
      node_t* current_node = best.second;
      // Check if current score is worse than the worst in top_probes
      if (top_probes.size() == nprobes && best.first >= top_probes.front().first) {
        break;  // Beam has converged
      }
      auto& children = current_node->children;
      // Compute distances to children
      child_dists.resize(children.size());
      dist_cmps += children.size();
      if (params.quantize_centers) {  // Center scoring always uses TQ when enabled.
        q_query.distances_all(std::get<TQ_Range>(current_node->quantized_data), child_dists.data());
        bytes_accessed += std::get<TQ_Range>(current_node->quantized_data).num_bytes_per_point() * current_node->data.size();
      } else {
        auto& centers = current_node->data;
        const size_t nc = centers.size();
        for (size_t j = 0; j < nc; ++j) {
          child_dists[j] = query_point.distance(centers[j]);
        }
        bytes_accessed += nc * (centers.get_dims() * sizeof(float));
      }
      t_dists += t.stop();
      t.reset();

      for (size_t i = 0; i < children.size(); ++i) {
        float d = child_dists[i];
        node_t* child = children[i];
        if (child->children.empty()) {
          t.start();
          // It's a leaf node: add to probe candidates if better than current worst
          if (top_probes.size() < nprobes || d < top_probes.front().first) {
            top_probes.push_back({d, child});
            std::push_heap(top_probes.begin(), top_probes.end());
            if (top_probes.size() > nprobes) {
              std::pop_heap(top_probes.begin(), top_probes.end());
              top_probes.pop_back();  // Prune the farthest node
            }
          }
          t_rest += t.stop();
          t.reset();
        } else {
          t.start();
          // Internal node: add to beam if it's better than the current worst
          const size_t beam_size = beam.size();
          if (beam_size < beam_length) {
            beam.insert({d, child});
          } else {
            // Only check worst if beam is full
            auto worst_it = std::prev(beam.end());
            if (d < worst_it->first) {
              beam.erase(worst_it);
              beam.insert({d, child});
            }
          }
          t_beam += t.stop();
          t.reset();
        }
      }
    }

    GreedySearchResult out;
    out.dist_cmps = dist_cmps;
    out.bytes_accessed = bytes_accessed;
    out.timings = {t_dists, t_beam, t_rest};
    out.probe_list = std::move(top_probes);
    return out;
  }

  inline auto process_probes(const Point& query_point, const QuantQuery& q_query_var,
                             parlay::sequence<std::pair<float, node_t*>>& probe_list) {
    const size_t nprobes = probe_list.size();
    auto sizes = parlay::delayed_tabulate(
        nprobes, [&](size_t i) { return probe_list[i].second->get_size(); });
    auto scan_result = parlay::scan(sizes);
    auto& offsets = scan_result.first;
    size_t total_size = scan_result.second;
    auto visited = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(total_size);
    auto tmp_dists = parlay::sequence<float>::uninitialized(total_size);
    auto bytes_accessed = parlay::sequence<size_t>::uninitialized(nprobes);

    auto process_probes_quant = [&]<typename RangeType>(const auto& q_query) {
      parlay::parallel_for(0, nprobes, [&](size_t i) {
        node_t* leaf = probe_list[i].second;
        auto* leaf_data = std::get_if<RangeType>(&leaf->quantized_data);
        if (!leaf_data) UNREACHABLE();
        q_query.distances_all(*leaf_data, tmp_dists.begin() + offsets[i]);
        bytes_accessed[i] = leaf_data->num_bytes_per_point() * leaf->get_size();
      });
    };

    switch (quantization_mode) {
      case QT::PQ:
        process_probes_quant.template operator()<typename SVQT::PQ_Range>(
            std::get<typename SVQT::PQ_Query>(q_query_var));
        break;
      case QT::RaBitQ:
        process_probes_quant.template operator()<typename SVQT::RQ_Range>(
            std::get<typename SVQT::RQ_Query>(q_query_var));
        break;
      case QT::FastScan:
        process_probes_quant.template operator()<typename SVQT::FS_Range>(
            std::get<typename SVQT::FS_Query>(q_query_var));
        break;
      case QT::TurboQuant:
        process_probes_quant.template operator()<typename SVQT::TQ_Range>(
            std::get<typename SVQT::TQ_Query>(q_query_var));
        break;
      case QT::SPQTQ:
        process_probes_quant.template operator()<typename SVQT::PQTQ_Range>(
            std::get<typename SVQT::PQTQ_Query>(q_query_var));
        break;
      case QT::None:
        parlay::parallel_for(0, nprobes, [&](size_t i) {
          node_t* leaf_node = probe_list[i].second;
          const size_t off = offsets[i];
          const size_t sz = leaf_node->get_size();
          for (size_t j = 0; j < sz; ++j) {
            tmp_dists[off + j] = query_point.distance(leaf_node->data[j]);
          }
          bytes_accessed[i] = sz * (leaf_node->data.get_dims() * sizeof(float));
        });
        break;
    }

    parlay::parallel_for(0, nprobes, [&](size_t i) {
      node_t* leaf = probe_list[i].second;
      const size_t off = offsets[i];
      const size_t sz = leaf->get_size();
      for (size_t j = 0; j < sz; ++j) {
        visited[off + j] = {static_cast<uint32_t>(leaf->get_id(j)), tmp_dists[off + j]};
      }
    });
    return std::make_pair(visited, parlay::reduce(bytes_accessed));
  }

  // Search for nearest neighbors of a single query point
  auto search_each(const Point& query_point, size_t nprobes, size_t num_rerank,
                   std::pair<uint32_t, float>* output) {
    parlay::internal::timer t;
    std::vector<double> timings;
    size_t dist_cmps = 0;
    size_t bytes_accessed = 0;

    double t_quantize = 0.0;
    double t_distances = 0.0;
    double t_rest = 0.0;

    // Step 0: Quantize Query
    QuantQuery q_query_point_var =
        this->template quantize_query_point<SVQT>(query_point, quantizer);
    TQ_Query q_center_query;
    t.start();
    if (params.quantize_centers) {
      q_center_query = center_quantizer.quantize_query(query_point);
    }
    t_quantize = t.stop();
    t.reset();

    // Step 1: Greedy search to find candidate probe clusters
    auto gs = greedy_search(query_point, q_center_query, nprobes);
    auto& probe_list = gs.probe_list;
    dist_cmps += gs.dist_cmps;
    bytes_accessed += gs.bytes_accessed;
    nprobes = std::min(nprobes, probe_list.size());
    timings.push_back(static_cast<double>(dist_cmps));
    for (double time : gs.timings) {
      timings.push_back(time);
    }

    // Step 2: Probe clusters in probe_list
    t.start();
    parlay::sequence<std::pair<uint32_t, float>> visited;
    size_t bytes_pp;
    std::tie(visited, bytes_pp) = process_probes(query_point, q_query_point_var, probe_list);
    bytes_accessed += bytes_pp;
    t_distances = t.stop();
    t.reset();

    dist_cmps += visited.size();

    t.start();
    parlay::sort_inplace(visited, [](const auto& a, const auto& b) { return a.second < b.second; });
    t_rest += t.stop();
    t.reset();

    // Push the 3 Step-2 sub-timings (instead of 1)
    timings.push_back(t_quantize);
    timings.push_back(t_distances);
    timings.push_back(t_rest);

    // Step 3: De-dup
    // used as a hash filter (can give false negative -- i.e. can say
    // not in table when it is)
    double t_dedup = 0.0;
    t.start();
    int bits = std::max<int>(10, std::ceil(std::log2(num_rerank)) - 2);
    std::vector<uint32_t> hash_filter(1 << bits, -1);
    auto has_been_seen = [&](uint32_t a) -> bool {
      int loc = parlay::hash64_2(a) & ((1 << bits) - 1);
      if (hash_filter[loc] == a) return true;
      hash_filter[loc] = a;
      return false;
    };
    size_t count = 0;
    for (size_t i = 0; i < visited.size() && count < num_rerank; i++) {
      auto [id, dist] = visited[i];
      if (!has_been_seen(id)) {
        output[count++] = {id, dist};
      }
    }
    for (size_t i = count; i < num_rerank; i++) {
      output[i] = {UINT32_MAX, std::numeric_limits<float>::max()};
    }
    t_dedup = t.stop();
    timings.push_back(t_dedup);
    return std::make_tuple(dist_cmps, bytes_accessed, timings);
  }

  std::tuple<parlay::sequence<std::pair<uint32_t, float>>, size_t, std::vector<double>>
  search_with_stats(const ChPoint& query, const PointCloudSet<ChPoint>& points,
                    const SearchParams& search_params) override {
    parlay::internal::timer t;
    std::vector<double> timings;

    size_t k = search_params.k;
    size_t nprobes = search_params.nprobes;
    size_t num_rerank = search_params.num_rerank;
    size_t dist_cmps = 0;
    size_t bytes_accessed = 0;
    size_t q = query.size();
    // Step 1: Search each query independently to obtain candidates
    t.start();
    auto results = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(q * num_rerank);
    auto in_dist_cmps = parlay::sequence<size_t>::uninitialized(q);
    auto in_bytes_accessed = parlay::sequence<size_t>::uninitialized(q);
    auto timings_each = parlay::sequence<std::vector<double>>(q);
    parlay::parallel_for(0, q, [&](size_t i) {
      std::tie(in_dist_cmps[i], in_bytes_accessed[i], timings_each[i]) =
          search_each(query[i], nprobes, num_rerank, &results[i * num_rerank]);
    });
    dist_cmps += parlay::reduce(in_dist_cmps);
    bytes_accessed += parlay::reduce(in_bytes_accessed);
    timings.push_back(t.stop());
    t.reset();

    // Step 2: Dedup
    t.start();
    parlay::sort_inplace(results, [](const auto& a, const auto& b) { return a.second < b.second; });
    parlay::sequence<std::pair<uint32_t, float>> visited;
    visited.reserve(num_rerank);
    int bits = std::max<int>(10, std::ceil(std::log2(num_rerank)) - 2);
    std::vector<uint32_t> hash_filter(1 << bits, -1);
    auto has_been_seen = [&](uint32_t a) -> bool {
      int loc = parlay::hash64_2(a) & ((1 << bits) - 1);
      if (hash_filter[loc] == a) return true;
      hash_filter[loc] = a;
      return false;
    };
    size_t count = 0;
    for (size_t i = 0; i < q * num_rerank && count < num_rerank; i++) {
      auto [id, dist] = results[i];
      if (id == UINT32_MAX) break;
      if (!has_been_seen(id)) {
        visited.push_back({id, dist});
        count++;
      }
    }
    timings.push_back(t.stop());
    t.reset();

    // Step 3: Re-ranking
    t.start();
    auto final_results =
        parlay::sequence<std::pair<uint32_t, float>>::uninitialized(std::min(k, visited.size()));
    if (search_params.num_rerank > 0) {
      size_t num_rerank = std::min(search_params.num_rerank, visited.size());
      bytes_accessed += this->rerank(query, points, visited, num_rerank, final_results);
      dist_cmps += num_rerank;
    } else {
      parlay::parallel_for(0, final_results.size(),
                           [&](size_t i) { final_results[i] = visited[i]; });
    }
    timings.push_back(t.stop());
    t.reset();

    std::vector<double> stats;
    stats.reserve(timings.size() + 1);
    stats.push_back(static_cast<double>(dist_cmps));
    stats.insert(stats.end(), timings.begin(), timings.end());

    size_t cur = stats.size();
    for (size_t i = 0; i < timings_each[0].size(); i++) {
      stats.push_back(0.0);
    }
    for (auto& v : timings_each) {
      for (size_t i = 0; i < v.size(); i++) {
        stats[cur + i] += v[i];
      }
    }
    return std::make_tuple(final_results, bytes_accessed, stats);
  }

  void save(const std::string& filename) override {
    if (root == nullptr) {
      std::cerr << "IndexSVHIVF::save: root is null (index not built).\n";
      return;
    }
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile.is_open()) {
      std::cerr << "Error opening file for writing: " << filename << std::endl;
      return;
    }

    // Collect data by traversing the tree
    parlay::sequence<node_t*> ind_to_node;
    std::unordered_map<node_t*, size_t> node_to_ind;
    parlay::sequence<size_t> center_offsets;
    parlay::sequence<size_t> children_offsets;
    parlay::sequence<size_t> point_id_offsets;

    traverse_tree(root, ind_to_node, node_to_ind, center_offsets, children_offsets,
                  point_id_offsets);

    size_t num_nodes = ind_to_node.size();
    outfile.write(reinterpret_cast<const char*>(&d), sizeof(unsigned));
    outfile.write(reinterpret_cast<const char*>(&num_nodes), sizeof(size_t));

    // Write center data for internal nodes
    auto center_scan = parlay::scan(center_offsets).first;
    auto center_data = parlay::sequence<float>::uninitialized(parlay::reduce(center_offsets) * d);
    parlay::parallel_for(0, num_nodes, [&](size_t i) {
      if (!ind_to_node[i]->children.empty()) {
        auto node_centers = ind_to_node[i]->data;
        size_t offset = center_scan[i];
        parlay::parallel_for(0, node_centers.size(), [&](size_t j) {
          std::copy(node_centers[j].data(), node_centers[j].data() + d,
                    center_data.begin() + (offset + j) * d);
        });
      }
    });
    outfile.write(reinterpret_cast<const char*>(center_offsets.data()), num_nodes * sizeof(size_t));
    outfile.write(reinterpret_cast<const char*>(center_data.data()),
                  center_data.size() * sizeof(float));

    // Write children data
    auto children_scan = parlay::scan(children_offsets).first;
    auto children_indices =
        parlay::sequence<size_t>::uninitialized(parlay::reduce(children_offsets));
    parlay::parallel_for(0, num_nodes, [&](size_t i) {
      auto node_children = ind_to_node[i]->children;
      size_t offset = children_scan[i];
      for (size_t j = 0; j < node_children.size(); ++j) {
        children_indices[offset + j] = node_to_ind[node_children[j]];
      }
    });
    outfile.write(reinterpret_cast<const char*>(children_offsets.data()),
                  num_nodes * sizeof(size_t));
    outfile.write(reinterpret_cast<const char*>(children_indices.data()),
                  children_indices.size() * sizeof(size_t));

    // Write point ID data for leaf nodes
    auto point_id_scan = parlay::scan(point_id_offsets).first;
    auto point_ids = parlay::sequence<std::pair<size_t, size_t>>::uninitialized(
        parlay::reduce(point_id_offsets));
    parlay::parallel_for(0, num_nodes, [&](size_t i) {
      if (ind_to_node[i]->children.empty()) {
        auto node_ids = ind_to_node[i]->ids;
        size_t offset = point_id_scan[i];
        for (size_t j = 0; j < node_ids.size(); ++j) {
          point_ids[offset + j] = node_ids[j];
        }
      }
    });
    outfile.write(reinterpret_cast<const char*>(point_id_offsets.data()),
                  num_nodes * sizeof(size_t));
    outfile.write(reinterpret_cast<const char*>(point_ids.data()),
                  point_ids.size() * sizeof(std::pair<size_t, size_t>));

    // Leaf / flat-vector quantization model (SVQT, same as build).
    int type_id = static_cast<int>(quantization_mode);
    outfile.write(reinterpret_cast<const char*>(&type_id), sizeof(int));
    switch (quantization_mode) {
      case QT::PQ: std::get<typename SVQT::PQ_Model>(quantizer).save(outfile); break;
      case QT::FastScan: std::get<typename SVQT::FS_Model>(quantizer).save(outfile); break;
      case QT::RaBitQ: std::get<typename SVQT::RQ_Model>(quantizer).save(outfile); break;
      case QT::TurboQuant: std::get<typename SVQT::TQ_Model>(quantizer).save(outfile); break;
      case QT::SPQTQ: std::get<typename SVQT::PQTQ_Model>(quantizer).save(outfile); break;
      case QT::None:
      default: break;
    }
    outfile.close();
  }

  void load(const std::string& filename, const PointCloudSet<ChPoint>& points) override {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile.is_open()) {
      std::cerr << "Error opening file for reading: " << filename << std::endl;
      return;
    }

    size_t num_nodes;
    infile.read(reinterpret_cast<char*>(&d), sizeof(unsigned));
    infile.read(reinterpret_cast<char*>(&num_nodes), sizeof(size_t));

    parlay::sequence<size_t> center_offsets(num_nodes);
    infile.read(reinterpret_cast<char*>(center_offsets.data()), num_nodes * sizeof(size_t));
    size_t total_centers = parlay::reduce(center_offsets);
    parlay::sequence<float> center_data(total_centers * d);
    infile.read(reinterpret_cast<char*>(center_data.data()), center_data.size() * sizeof(float));

    parlay::sequence<size_t> children_offsets(num_nodes);
    infile.read(reinterpret_cast<char*>(children_offsets.data()), num_nodes * sizeof(size_t));
    size_t total_children = parlay::reduce(children_offsets);
    parlay::sequence<size_t> children_indices(total_children);
    infile.read(reinterpret_cast<char*>(children_indices.data()), total_children * sizeof(size_t));

    parlay::sequence<size_t> point_id_offsets(num_nodes);
    infile.read(reinterpret_cast<char*>(point_id_offsets.data()), num_nodes * sizeof(size_t));
    size_t total_ids = parlay::reduce(point_id_offsets);
    parlay::sequence<std::pair<size_t, size_t>> point_ids(total_ids);
    infile.read(reinterpret_cast<char*>(point_ids.data()),
                total_ids * sizeof(std::pair<size_t, size_t>));

    int type_id;
    infile.read(reinterpret_cast<char*>(&type_id), sizeof(int));
    quantization_mode = static_cast<QT>(type_id);
    switch (quantization_mode) {
      case QT::PQ:
        quantizer.template emplace<typename SVQT::PQ_Model>();
        std::get<typename SVQT::PQ_Model>(quantizer).load(infile);
        break;
      case QT::FastScan:
        quantizer.template emplace<typename SVQT::FS_Model>();
        std::get<typename SVQT::FS_Model>(quantizer).load(infile);
        break;
      case QT::RaBitQ:
        quantizer.template emplace<typename SVQT::RQ_Model>();
        std::get<typename SVQT::RQ_Model>(quantizer).load(infile);
        break;
      case QT::TurboQuant:
        quantizer.template emplace<typename SVQT::TQ_Model>();
        std::get<typename SVQT::TQ_Model>(quantizer).load(infile);
        break;
      case QT::SPQTQ:
        quantizer.template emplace<typename SVQT::PQTQ_Model>();
        std::get<typename SVQT::PQTQ_Model>(quantizer).load(infile);
        break;
      case QT::None:
      default: quantizer = std::monostate{}; break;
    }
    infile.close();

    auto ind_to_node =
        parlay::sequence<node_t*>::from_function(num_nodes, [&](size_t i) { return new node_t(); });

    auto center_scan = parlay::scan(center_offsets).first;
    auto children_scan = parlay::scan(children_offsets).first;
    auto point_id_scan = parlay::scan(point_id_offsets).first;

    // Build a map from point cloud ID to its index in the `points` object
    auto point_id_to_data_id = parlay::sequence<uint32_t>(points.size());
    parlay::parallel_for(0, points.size(),
                         [&](size_t i) { point_id_to_data_id[points.get_id(i)] = i; });

    // Reconstruct nodes
    parlay::parallel_for(0, num_nodes, [&](size_t i) {
      node_t* node = ind_to_node[i];
      if (children_offsets[i] > 0) {  // Internal node
        auto centers_seq = parlay::sequence<parlay::sequence<float>>::from_function(
            center_offsets[i], [&](size_t j) {
              auto s = parlay::sequence<float>::uninitialized(d);
              std::copy(center_data.begin() + (center_scan[i] + j) * d,
                        center_data.begin() + (center_scan[i] + j + 1) * d, s.begin());
              return s;
            });
        node->data = Range(centers_seq, d);
        node->children = parlay::sequence<node_t*>::from_function(
            children_offsets[i],
            [&](size_t j) { return ind_to_node[children_indices[children_scan[i] + j]]; });
      } else {  // Leaf node
        node->ids = parlay::sequence<std::pair<size_t, size_t>>::from_function(
            point_id_offsets[i], [&](size_t j) { return point_ids[point_id_scan[i] + j]; });

        auto point_group = parlay::delayed_tabulate(point_id_offsets[i], [&](size_t j) {
          auto [pc_id, vec_id] = point_ids[point_id_scan[i] + j];
          uint32_t data_id = point_id_to_data_id[pc_id];
          return points[data_id][vec_id];
        });
        node->data = Range(point_group, d);
      }
    });

    root = ind_to_node[0];

    // Center TurboQuant is not serialized; retrain on the full dataset (same idea as MVIVF spill).
    if (params.quantize_centers) {
      FlatRange flat_points(points);
      init_tq_quantizer(flat_points);
    }

    // Re-encode: internal nodes use center_quantizer (encode_tq); leaves use leaf quantizer.
    parlay::parallel_for(
        0, num_nodes,
        [&](size_t i) {
          node_t* node = ind_to_node[i];
          if (!node->children.empty()) {
            if (params.quantize_centers) {
              node->quantized_data = encode_tq(node->data);
            } else {
              node->quantized_data = std::monostate{};
            }
          } else if (!node->ids.empty()) {
            node->quantized_data =
                this->template encode_range_quantized<SVQT>(node->data, quantizer);
          }
        },
        1);
  }

  void traverse_tree(node_t* node, parlay::sequence<node_t*>& ind_to_node,
                     std::unordered_map<node_t*, size_t>& node_to_ind,
                     parlay::sequence<size_t>& center_offsets,
                     parlay::sequence<size_t>& children_offsets,
                     parlay::sequence<size_t>& point_id_offsets) {
    if (node == nullptr) return;
    node_to_ind[node] = ind_to_node.size();
    ind_to_node.push_back(node);
    center_offsets.push_back(node->children.empty() ? 0 : node->data.size());
    children_offsets.push_back(node->children.size());
    point_id_offsets.push_back(node->children.empty() ? node->ids.size() : 0);

    for (node_t* child : node->children) {
      traverse_tree(child, ind_to_node, node_to_ind, center_offsets, children_offsets,
                    point_id_offsets);
    }
  }

  void traverse_and_delete(node_t* node) {
    if (node == nullptr) return;
    for (node_t* child : node->children) {
      traverse_and_delete(child);
    }
    delete node;
  }

  ~IndexSVHIVF() {
    if (root != nullptr) {
      traverse_and_delete(root);
    }
  }
};

using IndexSVHIVFL2 = IndexSVHIVF<true>;   // Instantiates for L2 metric (metric = true)
using IndexSVHIVFIP = IndexSVHIVF<false>;  // Instantiates for MIPS      (metric = false)

}  // namespace mvsic