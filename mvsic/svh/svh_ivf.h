#pragma once

#include <set>
#include <queue>
#include <optional>

#include "mvsic/core/index.h"
#include "mvsic/core/utils/util.h"
#include "mvsic/core/utils/kmeans_util.h"
#include "mvsic/core/types/point_range.h"
#include "mvsic/core/types/ip_point.h"
#include "mvsic/core/types/l2_point.h"

// Quantization Headers
#include "mvsic/core/quantization/pq.h"
#include "mvsic/core/quantization/rabitq.h"
#include "mvsic/core/quantization/fastscan.h"

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
  using Index<metric>::d;  // Embedding dimension

  // Quantizer Types
  using PQ_Range = pq::Quantized_Point_Range<Range, metric>;
  using PQ_Point = pq::Quantized_Query<metric>;
  using RQ_Range = rabitq::Quantized_Point_Range<Range, metric>;
  using RQ_Point = rabitq::Quantized_Query<metric>;
  using FS_Range = fastscan::Quantized_Point_Range<Range, metric>;
  using FS_Point = fastscan::Quantized_Query<metric>;
  using PQ_Model = pq::Model<metric>;
  using RQ_Model = rabitq::Model<metric>;
  using FS_Model = fastscan::Model<metric>;

  using QuantModel = std::variant<std::monostate, PQ_Model, RQ_Model, FS_Model>;
  using QuantRange = std::variant<std::monostate, PQ_Range, RQ_Range, FS_Range>;
  using QuantQuery = std::variant<std::monostate, PQ_Point, RQ_Point, FS_Point>;
  using QT = IndexParams::QuantizerType;

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

  IndexParams params;
  node_t* root = nullptr;  // Root of the k-means tree

  // Quantizer Storage
  QuantModel quantizer = std::monostate{};
  QT active_quantizer = QT::None;

  IndexSVHIVF(size_t d_) noexcept : params(IndexParams::svh_ivf()) { d = d_; }
  IndexSVHIVF(size_t d_, const IndexParams& params) noexcept : params(params) { d = d_; }

  // Recursively builds the kmeans tree
  void recursive_build(node_t* node, const parlay::sequence<parlay::sequence<float>>& points,
                       const parlay::sequence<std::pair<size_t, size_t>>& ids) {
    size_t n = points.size();
    size_t num_clusters = (params.k_per_level > 0) ? params.k_per_level : std::ceil(std::sqrt(n));
    if (params.verbose >= 1) {
      std::cout << "Building index with " << n << " points, num_clusters: " << num_clusters
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
      switch (active_quantizer) {
        case QT::PQ: {
          auto& m = std::get<PQ_Model>(quantizer);
          node->quantized_data = m.encode(node->data);
          break;
        }
        case QT::FastScan: {
          auto& m = std::get<FS_Model>(quantizer);
          node->quantized_data = m.encode(node->data);
          break;
        }
        case QT::RaBitQ: {
          auto& m = std::get<RQ_Model>(quantizer);
          node->quantized_data = m.encode(node->data);
          break;
        }
        case QT::None:
        default: node->quantized_data = std::monostate{}; break;
      }
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
            // Quantization: Encoding
            switch (active_quantizer) {
              case QT::PQ: {
                auto& m = std::get<PQ_Model>(quantizer);
                child->quantized_data = m.encode(child->data);
                break;
              }
              case QT::RaBitQ: {
                auto& m = std::get<RQ_Model>(quantizer);
                child->quantized_data = m.encode(child->data);
                break;
              }
              case QT::FastScan: {
                auto& m = std::get<FS_Model>(quantizer);
                child->quantized_data = m.encode(child->data);
                break;
              }
              case QT::None:
              default: child->quantized_data = std::monostate{}; break;
            }
          }
          node->children[id] = child;
        },
        1);
  }

  void build(const PointCloudSet<ChPoint>& points) override {
    root = new node_t();
    if (params.compress_input) {
      // TODO: run Ward's HAC to compress input point clouds
    }

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
    active_quantizer = params.pq.method;
    switch (active_quantizer) {
      case QT::RaBitQ:
        if (params.verbose >= 1) std::cout << "Training RaBitQ..." << std::endl;
        quantizer.template emplace<RQ_Model>();
        std::get<RQ_Model>(quantizer).train(Range(data, d), params.pq.rabitq_bits);
        break;
      case QT::PQ:
        if (params.verbose >= 1) std::cout << "Training Standard PQ..." << std::endl;
        quantizer.template emplace<PQ_Model>();
        std::get<PQ_Model>(quantizer).train(Range(data, d), params.pq.block_size,
                                            params.pq.num_clusters_per_block,
                                            params.pq.num_points_per_cluster);
        break;
      case QT::FastScan:
        if (params.verbose >= 1) std::cout << "Training FastScan PQ..." << std::endl;
        quantizer.template emplace<FS_Model>();
        std::get<FS_Model>(quantizer).train(Range(data, d), params.pq.block_size);
        break;
      case QT::None: quantizer = std::monostate{}; break;
      default: std::cerr << "Error: Unsupported Quantization Method!" << std::endl; abort();
    }

    recursive_build(root, data, ids);
  }

  // Output type of Greedy Search
  struct GreedySearchResult {
    parlay::sequence<std::pair<float, node_t*>> probe_list;
    size_t dist_cmps = 0;
    double time = 0.0;
  };

  // Simple Beam Search using std::set
  GreedySearchResult greedy_search(const Point& query_point, const QuantQuery& q_query_point_var,
                                   size_t nprobes) const {
    using score_node = std::pair<float, node_t*>;
    auto less = [](const score_node& a, const score_node& b) {
      return a.first < b.first || (a.first == b.first && a.second < b.second);
    };
    const size_t beam_length = 2 * nprobes;
    parlay::internal::timer t;
    t.start();

    size_t dist_cmps = 0;
    std::set<score_node> beam;
    std::vector<score_node> probe_vec;  // To collect leaf nodes
    std::vector<float> child_dists;     // Scratch memory

    // Initial seed
    beam.insert({0.0f, root});

    while (!beam.empty()) {
      // Pop the best node (smallest distance)
      auto it = beam.begin();
      score_node best = *it;
      beam.erase(it);
      node_t* current_node = best.second;
      auto& children = current_node->children;
      if (children.empty()) continue;
      // Compute distances to children
      child_dists.resize(children.size());
      if (!params.quantize_centers) {
        auto& centers = current_node->data;
        parlay::parallel_for(0, children.size(), [&](uint32_t i) {
          auto [dist, d_c] = query_point.distance_w_cmps(centers[i]);
          child_dists[i] = dist;
        });
      } else {
        switch (active_quantizer) {
          case QT::RaBitQ: {
            auto& q_query = std::get<RQ_Point>(q_query_point_var);
            auto& qleaf = std::get<RQ_Range>(current_node->quantized_data);
            q_query.distances_all(qleaf, child_dists.data());
            break;
          }
          case QT::PQ: {
            auto& q_query = std::get<PQ_Point>(q_query_point_var);
            auto& qleaf = std::get<PQ_Range>(current_node->quantized_data);
            q_query.distances_all(qleaf, child_dists.data());
            break;
          }
          case QT::FastScan: {
            auto& q_query = std::get<FS_Point>(q_query_point_var);
            auto& qleaf = std::get<FS_Range>(current_node->quantized_data);
            q_query.distances_all(qleaf, child_dists.data());
            break;
          }
          case QT::None:
          default:
            std::cerr << "Error: Invalid quantizer type in greedy search." << std::endl;
            abort();
        }
      }

      for (size_t i = 0; i < children.size(); ++i) {
        float d = child_dists[i];
        node_t* child = children[i];
        if (child->children.empty()) {
          // It's a leaf node: add to probe candidates
          probe_vec.push_back({d, child});
        } else {
          // Internal node: add to beam if it's better than the current worst
          if (beam.size() < beam_length || d < beam.rbegin()->first) {
            beam.insert({d, child});
            if (beam.size() > beam_length) {
              beam.erase(std::prev(beam.end()));  // Prune the farthest node
            }
          }
        }
      }
    }
    // Finalize probes: take best nprobes leaves
    if (probe_vec.size() > nprobes) {
      std::nth_element(probe_vec.begin(), probe_vec.begin() + nprobes, probe_vec.end(), less);
      probe_vec.resize(nprobes);
    }
    parlay::sort_inplace(probe_vec, less);

    GreedySearchResult out;
    out.dist_cmps = dist_cmps;
    out.time = t.stop();
    out.probe_list = parlay::sequence<score_node>::from_function(
        probe_vec.size(), [&](size_t i) { return probe_vec[i]; });
    return out;
  }

  // Search for nearest neighbors of a single query point
  std::tuple<size_t, std::vector<double>> search_each(const Point& query_point, size_t nprobes,
                                                      size_t num_rerank,
                                                      std::pair<uint32_t, float>* output) {
    parlay::internal::timer t;
    std::vector<double> timings;
    size_t dist_cmps = 0;

    // Step 0: Quantize Query
    double t_quantize = 0.0;
    t.start();
    QuantQuery q_query_point_var;
    switch (active_quantizer) {
      case QT::RaBitQ: {
        auto& m = std::get<RQ_Model>(quantizer);
        q_query_point_var = m.quantize_query(query_point);
        break;
      }
      case QT::PQ: {
        auto& m = std::get<PQ_Model>(quantizer);
        q_query_point_var = m.quantize_query(query_point);
        break;
      }
      case QT::FastScan: {
        auto& m = std::get<FS_Model>(quantizer);
        q_query_point_var = m.quantize_query(query_point);
        break;
      }
      case QT::None: {
        q_query_point_var = std::monostate{};
        break;
      }
      default: abort();
    }
    t_quantize = t.stop();
    t.reset();

    // Step 1: Greedy search to find candidate probe clusters
    auto gs = greedy_search(query_point, q_query_point_var, nprobes);
    auto probe_list = std::move(gs.probe_list);
    dist_cmps += gs.dist_cmps;
    timings.push_back(gs.time);
    nprobes = std::min(nprobes, probe_list.size());

    // Step 2: Probe clusters in probe_list
    double t_rest = 0.0;
    t.start();
    auto sizes = parlay::delayed_tabulate(
        nprobes, [&](size_t i) { return probe_list[i].second->get_size(); });
    auto scan_result = parlay::scan(sizes);
    auto& offsets = scan_result.first;
    size_t total_size = scan_result.second;
    auto visited = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(total_size);
    auto calc_dists = parlay::sequence<float>::uninitialized(total_size);
    t_rest += t.stop();
    t.reset();

    // --- quantize + distances ---
    double t_distances = 0.0;
    switch (active_quantizer) {
      case QT::RaBitQ: {
        t.start();
        auto& q_query = std::get<RQ_Point>(q_query_point_var);
        parlay::parallel_for(0, nprobes, [&](size_t i) {
          node_t* leaf = probe_list[i].second;
          auto& qleaf = std::get<RQ_Range>(leaf->quantized_data);
          q_query.distances_all(qleaf, &calc_dists[offsets[i]]);
          parlay::parallel_for(0, leaf->get_size(), [&](size_t j) {
            visited[offsets[i] + j] = {leaf->get_id(j), calc_dists[offsets[i] + j]};
          });
        });
        t_distances = t.stop();
        t.reset();
        break;
      }
      case QT::PQ: {
        t.start();
        auto& q_query = std::get<PQ_Point>(q_query_point_var);
        parlay::parallel_for(0, nprobes, [&](size_t i) {
          node_t* leaf = probe_list[i].second;
          auto& qleaf = std::get<PQ_Range>(leaf->quantized_data);
          q_query.distances_all(qleaf, &calc_dists[offsets[i]]);
          parlay::parallel_for(0, leaf->get_size(), [&](size_t j) {
            visited[offsets[i] + j] = {leaf->get_id(j), calc_dists[offsets[i] + j]};
          });
        });
        t_distances = t.stop();
        t.reset();
        break;
      }
      case QT::FastScan: {
        t.start();
        auto& q_query = std::get<FS_Point>(q_query_point_var);
        parlay::parallel_for(0, nprobes, [&](size_t i) {
          node_t* leaf = probe_list[i].second;
          auto& qleaf = std::get<FS_Range>(leaf->quantized_data);
          q_query.distances_all(qleaf, &calc_dists[offsets[i]]);
          parlay::parallel_for(0, leaf->get_size(), [&](size_t j) {
            visited[offsets[i] + j] = {leaf->get_id(j), calc_dists[offsets[i] + j]};
          });
        });
        t_distances = t.stop();
        t.reset();
        break;
      }
      case QT::None: {
        t.start();
        parlay::parallel_for(0, nprobes, [&](size_t i) {
          node_t* leaf_node = probe_list[i].second;
          auto& leaf_points = leaf_node->data;
          parlay::parallel_for(0, leaf_node->get_size(), [&](size_t j) {
            visited[offsets[i] + j] = {leaf_node->get_id(j), query_point.distance(leaf_points[j])};
          });
        });
        t_distances = t.stop();
        t.reset();
        break;
      }
    }

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
    for (size_t i = 0; i < total_size && count < num_rerank; i++) {
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
    return std::make_tuple(dist_cmps, timings);
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
    size_t q = query.size();

    // Step 1: Search each query independently to obtain candidates
    auto results = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(q * num_rerank);
    auto in_dist_cmps = parlay::sequence<size_t>::uninitialized(q);
    parlay::parallel_for(0, q, [&](size_t i) {
      std::vector<double> dummy_timings;
      std::tie(in_dist_cmps[i], dummy_timings) =
          search_each(query[i], nprobes, num_rerank, &results[i * num_rerank]);
    });
    // Step 2: Dedup
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

    // Step 3: Re-ranking
    t.start();
    auto final_results =
        parlay::sequence<std::pair<uint32_t, float>>::uninitialized(std::min(k, visited.size()));
    if (!search_params.norerank) {
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

  void save(const std::string& filename) override {
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
    auto center_data =
        parlay::sequence<float>::uninitialized(parlay::reduce(center_offsets) * d);
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
    outfile.write(reinterpret_cast<const char*>(center_offsets.data()),
                  num_nodes * sizeof(size_t));
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

    // Write quantization model
    int type_id = static_cast<int>(active_quantizer);
    outfile.write(reinterpret_cast<const char*>(&type_id), sizeof(int));
    switch (active_quantizer) {
      case QT::PQ: std::get<PQ_Model>(quantizer).save(outfile); break;
      case QT::FastScan: std::get<FS_Model>(quantizer).save(outfile); break;
      case QT::RaBitQ: std::get<RQ_Model>(quantizer).save(outfile); break;
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
    active_quantizer = static_cast<QT>(type_id);
    switch (active_quantizer) {
      case QT::PQ:
        quantizer.template emplace<PQ_Model>();
        std::get<PQ_Model>(quantizer).load(infile);
        break;
      case QT::FastScan:
        quantizer.template emplace<FS_Model>();
        std::get<FS_Model>(quantizer).load(infile);
        break;
      case QT::RaBitQ:
        quantizer.template emplace<RQ_Model>();
        std::get<RQ_Model>(quantizer).load(infile);
        break;
      default: break;
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

    // Re-encode data
    parlay::parallel_for(
        0, num_nodes,
        [&](size_t i) {
          node_t* node = ind_to_node[i];
          if ((!node->children.empty() && params.quantize_centers) ||
              (node->children.empty() && !node->ids.empty())) {
            switch (active_quantizer) {
              case QT::PQ:
                node->quantized_data = std::get<PQ_Model>(quantizer).encode(node->data);
                break;
              case QT::FastScan:
                node->quantized_data = std::get<FS_Model>(quantizer).encode(node->data);
                break;
              case QT::RaBitQ:
                node->quantized_data = std::get<RQ_Model>(quantizer).encode(node->data);
                break;
              default: break;
            }
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