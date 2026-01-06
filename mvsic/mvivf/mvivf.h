#pragma once

#include <queue>
#include <set>
#include <optional>

#include "absl/container/btree_set.h"

#include "mvsic/core/index.h"
#include "mvsic/core/mvclustering/mvclustering.h"
#include "mvsic/core/utils/util.h"
#include "mvsic/core/distance_measures/many_to_many.h"

// Quantization Headers
#include "mvsic/core/quantization/pq.h"
#include "mvsic/core/quantization/rabitq.h"
#include "mvsic/core/quantization/scann.h"
#include "mvsic/core/quantization/wrapper.h"

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

  // Multi-Vector Quantizer Types using the Wrapper
  using FlatRange = FlattenedPCRange<PointCloudSet<ChPoint>>;
  using PQ_Set = Quantized_Point_Cloud_Set<pq::Quantized_Point_Range<FlatRange, metric>, metric>;
  using RaBitQ_Set =
      Quantized_Point_Cloud_Set<rabitq::Quantized_Point_Range<FlatRange, metric>, metric>;
  using ScaNN_Set = Quantized_Point_Cloud_Set<pq::ScaNN_Point_Range<FlatRange, metric>, metric>;
  using QT = IndexParams::QuantizerType;

  // kmeans tree nodes
  struct node_t {
    parlay::sequence<node_t *> children;
    // For internal nodes: data = centers of children
    // For leaves:         data = points in the cluster
    PointCloudSet<ChPoint> data;
    node_t() noexcept : children(parlay::sequence<node_t *>(0)), data(PointCloudSet<ChPoint>()) {}
    ~node_t() noexcept {}

    inline size_t get_size() const noexcept { return data.size(); }
  };

  IndexParams params;
  node_t *root = nullptr;  // Root of the k-means tree

  // Quantizer Storage
  // TODO: optimize this by using std::variant
  std::optional<PQ_Set> quantizer_pq;
  std::optional<RaBitQ_Set> quantizer_rabitq;
  std::optional<ScaNN_Set> quantizer_scann;
  QT active_quantizer = QT::None;

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
            child->data = std::move(child_points);
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

    // Quantization
    active_quantizer = params.pq.method;
    switch (active_quantizer) {
      case QT::RaBitQ:
        if (params.verbose >= 1) std::cout << "Training RaBitQ..." << std::endl;
        quantizer_rabitq.emplace(points, params.pq.rabitq_bits);
        break;
      case QT::ScaNN:
        if (params.verbose >= 1) std::cout << "Training ScaNN..." << std::endl;
        quantizer_scann.emplace(points, params.pq.num_blocks, params.pq.num_clusters_per_block,
                                params.pq.num_points_per_cluster, params.pq.scann_threshold);
        break;
      case QT::PQ:
        if (params.verbose >= 1) std::cout << "Training PQ..." << std::endl;
        quantizer_pq.emplace(points, params.pq.num_blocks, params.pq.num_clusters_per_block,
                             params.pq.num_points_per_cluster);
        break;
      default: break;
    }
  }

  // Output type of Greedy Search
  struct GreedySearchResult {
    parlay::sequence<std::pair<float, node_t *>> probe_list;
    size_t dist_cmps_step1 = 0;
    double time_step1 = 0.0;
  };

  // Simple Beam Search using std::set
  GreedySearchResult greedy_search(const ChPoint &query, size_t nprobes) const {
    using score_node = std::pair<float, node_t *>;
    auto less = [](const score_node &a, const score_node &b) {
      return a.first < b.first || (a.first == b.first && a.second < b.second);
    };
    const size_t beam_length = 2 * nprobes;
    parlay::internal::timer t;
    t.start();

    size_t dist_cmps = 0;
    std::set<score_node> beam;
    std::vector<score_node> probe_vec;                    // To collect leaf nodes
    std::vector<std::pair<uint32_t, float>> child_dists;  // Scratch memory

    // Initial seed
    beam.insert({0.0f, root});

    while (!beam.empty()) {
      // Pop the best node (smallest distance)
      auto it = beam.begin();
      score_node best = *it;
      beam.erase(it);
      node_t *current_node = best.second;
      auto &children = current_node->children;
      auto &centers = current_node->data;
      if (children.empty()) continue;
      // Compute distances to children
      child_dists.resize(centers.size());
      dist_cmps += centers.distances_naive(query, child_dists.data());
      for (size_t i = 0; i < children.size(); ++i) {
        float d = child_dists[i].second;
        node_t *child = children[i];
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
    out.dist_cmps_step1 = dist_cmps;
    out.time_step1 = t.stop();
    out.probe_list = parlay::sequence<score_node>::from_function(
        probe_vec.size(), [&](size_t i) { return probe_vec[i]; });
    return out;
  }

  std::tuple<parlay::sequence<std::pair<uint32_t, float>>, size_t, std::vector<double>>
  search_with_stats(const ChPoint &query, const PointCloudSet<ChPoint> &points,
                    const SearchParams &search_params) override {
    parlay::internal::timer t;
    std::vector<double> timings;

    size_t k = search_params.k;
    size_t nprobes = search_params.nprobes;
    size_t dist_cmps = 0;

    // -------------------------
    // Step 1: Greedy search (heap-based)
    // -------------------------
    auto gs = greedy_search(query, nprobes);
    auto probe_list = std::move(gs.probe_list);
    dist_cmps += gs.dist_cmps_step1;
    timings.push_back(gs.time_step1);

    // -------------------------
    // Step 2: Probe clusters in probe_list
    // -------------------------
    double t_quantize = 0.0;
    double t_distances = 0.0;
    double t_rest = 0.0;

    // --- "rest" part 1: sizes/scan/allocation ---
    t.start();
    auto sizes = parlay::delayed_tabulate(
        nprobes, [&](size_t i) { return probe_list[i].second->get_size(); });
    auto scan_result = parlay::scan(sizes);
    auto &offsets = scan_result.first;
    size_t total_size = scan_result.second;
    auto visited = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(total_size);
    t_rest += t.stop();
    t.reset();

    // --- quantize + distances ---
    switch (active_quantizer) {
      case QT::RaBitQ: {
        t.start();
        auto q_query = quantizer_rabitq->quantize_query(query);
        t_quantize = t.stop();
        t.reset();

        t.start();
        parlay::parallel_for(0, nprobes, [&](size_t i) {
          node_t *leaf = probe_list[i].second;
          auto leaf_indices = parlay::delayed_tabulate(
              leaf->data.size(), [&](size_t j) { return leaf->data.get_id(j); });
          quantizer_rabitq->distances(q_query, leaf_indices, leaf_indices.size(),
                                      &visited[offsets[i]]);
        });
        t_distances = t.stop();
        t.reset();
        break;
      }
      case QT::ScaNN: {
        t.start();
        auto q_query = quantizer_scann->quantize_query(query);
        t_quantize = t.stop();
        t.reset();

        t.start();
        parlay::parallel_for(0, nprobes, [&](size_t i) {
          node_t *leaf = probe_list[i].second;
          auto leaf_indices = parlay::delayed_tabulate(
              leaf->data.size(), [&](size_t j) { return leaf->data.get_id(j); });
          quantizer_scann->distances(q_query, leaf_indices, leaf_indices.size(),
                                     &visited[offsets[i]]);
        });
        t_distances = t.stop();
        t.reset();
        break;
      }
      case QT::PQ: {
        t.start();
        auto q_query = quantizer_pq->quantize_query(query);
        t_quantize = t.stop();
        t.reset();

        t.start();
        parlay::parallel_for(0, nprobes, [&](size_t i) {
          node_t *leaf = probe_list[i].second;
          auto leaf_indices = parlay::delayed_tabulate(
              leaf->data.size(), [&](size_t j) { return leaf->data.get_id(j); });
          quantizer_pq->distances(q_query, leaf_indices, leaf_indices.size(), &visited[offsets[i]]);
        });
        t_distances = t.stop();
        t.reset();
        break;
      }
      case QT::None: {
        t_quantize = 0.0;

        t.start();
        auto leaf_dist_cmps = parlay::sequence<size_t>::uninitialized(nprobes);
        parlay::parallel_for(0, nprobes, [&](size_t i) {
          node_t *leaf_node = probe_list[i].second;
          leaf_dist_cmps[i] = leaf_node->data.distances_naive(query, &visited[offsets[i]]);
        });
        dist_cmps += parlay::reduce(leaf_dist_cmps);
        t_distances = t.stop();
        t.reset();
        break;
      }
    }

    // --- "rest" part 2: sorting visited ---
    t.start();
    parlay::sort_inplace(visited, [](const auto &a, const auto &b) { return a.second < b.second; });
    t_rest += t.stop();
    t.reset();

    // Push the 3 Step-2 sub-timings (instead of 1)
    timings.push_back(t_quantize);
    timings.push_back(t_distances);
    timings.push_back(t_rest);

    // -------------------------
    // Step 3: Re-ranking
    // -------------------------
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

  // Traversing the k-means tree: returns the height of the tree
  // TODO: get more stats about the tree
  size_t traverse_tree(node_t *node, parlay::sequence<node_t *> &ind_to_node,
                       std::unordered_map<node_t *, size_t> &node_to_ind,
                       parlay::sequence<size_t> &center_offsets,
                       parlay::sequence<size_t> &children_offsets,
                       parlay::sequence<size_t> &point_offsets, size_t height) {
    node_to_ind[node] = ind_to_node.size();
    ind_to_node.push_back(node);
    if (node->children.size() == 0) {              // leaves
      point_offsets.push_back(node->data.size());  // Always add point IDs
    } else {                                       // Internal nodes
      point_offsets.push_back(0);
      size_t dims = node->data.get_dims();
      for (size_t i = 0; i < node->children.size(); i++) {
        center_offsets.push_back(node->data.get_size(i) * dims);  // # embeddings in center[i]
      }
    }
    children_offsets.push_back(node->children.size());
    size_t h = height + 1;
    for (node_t *child : node->children) {
      h = std::max(h, traverse_tree(child, ind_to_node, node_to_ind, center_offsets,
                                    children_offsets, point_offsets, height + 1));
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

    // Collect data
    parlay::sequence<node_t *> ind_to_node;
    std::unordered_map<node_t *, size_t> node_to_ind;
    parlay::sequence<size_t> center_offsets;
    parlay::sequence<size_t> children_offsets;
    parlay::sequence<size_t> point_offsets;

    size_t height = traverse_tree(root, ind_to_node, node_to_ind, center_offsets, children_offsets,
                                  point_offsets, 0);  // <-- UPDATED
    std::cout << "Height of tree: " << height << std::endl;

    size_t total_center_sizes = parlay::scan_inplace(center_offsets);
    center_offsets.push_back(total_center_sizes);
    size_t total_children_sizes = parlay::scan_inplace(children_offsets);
    children_offsets.push_back(total_children_sizes);
    size_t total_point_sizes = parlay::scan_inplace(point_offsets);
    point_offsets.push_back(total_point_sizes);

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
      if (node->children.size() == 0) {  // <-- UPDATED: leaves only
        PointCloudSet<ChPoint> points = node->data;
        for (size_t j = 0; j < points.size(); ++j) {  // TODO: make parallel
          uint32_t point_id = points.get_id(j);
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
      default: break;
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
  // // Greedy Search to find top nprobes leaf nodes
  // GreedySearchResult greedy_search_heaps(const ChPoint &query, size_t nprobes) const {
  //   using score_node = std::pair<float, node_t *>;
  //   auto less = [](const score_node &a, const score_node &b) {
  //     return a.first < b.first || (a.first == b.first && a.second < b.second);
  //   };
  //   const size_t beam_length = 2 * nprobes;
  //   parlay::internal::timer t;
  //   t.start();

  //   // expand_heap is a MIN-heap on distance, implemented as a MAX-heap on (-dist)
  //   // store (neg_dist, node*)
  //   std::vector<score_node> expand_heap;
  //   expand_heap.reserve(beam_length * 2 + 1024);
  //   // beam_heap is a MAX-heap on distance (worst dist at top)
  //   std::vector<score_node> beam_heap;
  //   beam_heap.reserve(beam_length + 8);
  //   // Output
  //   std::vector<score_node> probe_vec;
  //   probe_vec.reserve(std::max<size_t>(nprobes * 4, 1024));
  //   std::vector<std::pair<uint32_t, float>> child_dists;  // scratch
  //   auto expand_cmp = [](const score_node &a, const score_node &b) {
  //     // heap largest first; we want smallest dist => largest (-dist)
  //     return a.first < b.first;
  //   };
  //   auto beam_cmp = [](const score_node &a, const score_node &b) {
  //     // heap largest dist first
  //     return a.first < b.first;
  //   };
  //   auto beam_threshold = [&]() -> float {
  //     if (beam_heap.size() < beam_length) return std::numeric_limits<float>::infinity();
  //     return beam_heap.front().first;  // worst (largest) dist
  //   };
  //   auto try_push_internal = [&](float dist, node_t *node) {
  //     float thr = beam_threshold();
  //     if (beam_heap.size() < beam_length || dist < thr) {
  //       // maintain bounded max-heap (beam)
  //       beam_heap.emplace_back(dist, node);
  //       std::push_heap(beam_heap.begin(), beam_heap.end(), beam_cmp);
  //       if (beam_heap.size() > beam_length) {
  //         std::pop_heap(beam_heap.begin(), beam_heap.end(), beam_cmp);
  //         beam_heap.pop_back();
  //       }
  //       // push to expansion heap as (-dist, node)
  //       expand_heap.emplace_back(-dist, node);
  //       std::push_heap(expand_heap.begin(), expand_heap.end(), expand_cmp);
  //     }
  //   };
  //   size_t dist_cmps = 0;
  //   // Seed root at distance 0
  //   try_push_internal(0.0f, root);

  //   while (!expand_heap.empty()) {
  //     // pop best (min dist) => pop largest (-dist)
  //     std::pop_heap(expand_heap.begin(), expand_heap.end(), expand_cmp);
  //     score_node cur = expand_heap.back();
  //     expand_heap.pop_back();

  //     float cur_dist = -cur.first;
  //     node_t *cur_node = cur.second;

  //     // Skip stale entries
  //     float thr = beam_threshold();
  //     if (beam_heap.size() >= beam_length && cur_dist > thr) continue;

  //     auto &centers = cur_node->data;
  //     auto &children = cur_node->children;
  //     if (children.empty()) continue;

  //     // distances to children (in centers order)
  //     child_dists.resize(centers.size());
  //     size_t d_c = centers.distances_naive(query, child_dists.data());
  //     dist_cmps += d_c;

  //     const size_t deg = children.size();
  //     for (size_t i = 0; i < deg; ++i) {
  //       node_t *child = children[i];
  //       float d = child_dists[i].second;
  //       if (child->children.empty()) {
  //         probe_vec.emplace_back(d, child);
  //         // Keep leaf candidates bounded to avoid huge sorts
  //         if (probe_vec.size() > nprobes * 8) {
  //           size_t keep = std::min(probe_vec.size(), nprobes * 4);
  //           if (keep < probe_vec.size()) {  // critical: nth must be < end()
  //             std::nth_element(probe_vec.begin(), probe_vec.begin() + keep, probe_vec.end(),
  //             less); probe_vec.resize(keep);
  //           }
  //         }
  //       } else {
  //         try_push_internal(d, child);
  //       }
  //     }
  //   }
  //   // Finalize probes: take best nprobes leaves
  //   if (probe_vec.size() > nprobes) {
  //     std::nth_element(probe_vec.begin(), probe_vec.begin() + nprobes, probe_vec.end(), less);
  //     probe_vec.resize(nprobes);
  //   }
  //   parlay::sort_inplace(probe_vec, less);

  //   GreedySearchResult out;
  //   out.dist_cmps_step1 = dist_cmps;
  //   out.time_step1 = t.stop();

  //   out.probe_list = parlay::sequence<score_node>::from_function(
  //       probe_vec.size(), [&](size_t i) { return probe_vec[i]; });
  //   return out;
  // }

  // // Greedy Search using Abseil B-Tree Set
  // GreedySearchResult greedy_search_absl(const ChPoint &query, size_t nprobes) const {
  //   using score_node = std::pair<float, node_t *>;
  //   auto less = [](const score_node &a, const score_node &b) {
  //     return a.first < b.first || (a.first == b.first && a.second < b.second);
  //   };

  //   const size_t beam_length = 2 * nprobes;
  //   parlay::internal::timer t;
  //   t.start();

  //   size_t dist_cmps = 0;
  //   // B-tree set: The best of both worlds for N=2048
  //   absl::btree_set<score_node, decltype(less)> beam(less);
  //   std::vector<score_node> probe_vec;
  //   probe_vec.reserve(nprobes * 4);
  //   std::vector<std::pair<uint32_t, float>> child_dists;

  //   beam.insert({0.0f, root});

  //   while (!beam.empty()) {
  //     // Pop the best node
  //     auto it = beam.begin();
  //     score_node best = *it;
  //     beam.erase(it);

  //     node_t *current_node = best.second;
  //     if (current_node->children.empty()) continue;

  //     auto &centers = current_node->data;
  //     auto &children = current_node->children;

  //     child_dists.resize(centers.size());
  //     dist_cmps += centers.distances_naive(query, child_dists.data());

  //     for (size_t i = 0; i < children.size(); ++i) {
  //       float d = child_dists[i].second;
  //       node_t *child = children[i];

  //       if (child->children.empty()) {
  //         probe_vec.emplace_back(d, child);
  //       } else {
  //         // Pruning logic
  //         if (beam.size() < beam_length || d < beam.rbegin()->first) {
  //           beam.insert({d, child});
  //           if (beam.size() > beam_length) {
  //             // Constant time or very fast leaf removal in B-tree
  //             beam.erase(std::prev(beam.end()));
  //           }
  //         }
  //       }
  //     }
  //   }

  //   // Finalize probes
  //   if (probe_vec.size() > nprobes) {
  //     std::nth_element(probe_vec.begin(), probe_vec.begin() + nprobes, probe_vec.end(), less);
  //     probe_vec.resize(nprobes);
  //   }
  //   parlay::sort_inplace(probe_vec, less);

  //   GreedySearchResult out;
  //   out.dist_cmps_step1 = dist_cmps;
  //   out.time_step1 = t.stop();
  //   out.probe_list = parlay::sequence<score_node>::from_function(
  //       probe_vec.size(), [&](size_t i) { return probe_vec[i]; });

  //   return out;
  // }
};

using IndexMVIVFL2 = IndexMVIVF<true>;   // Instantiates for L2 metric (metric = true)
using IndexMVIVFIP = IndexMVIVF<false>;  // Instantiates for MIPS      (metric = false)

}  // namespace mvsic