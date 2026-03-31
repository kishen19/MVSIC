#pragma once

#include "mvivf_base.h"

namespace mvsic {

template<bool metric>
IndexMVIVF<metric>::IndexMVIVF(size_t d_) noexcept {
  d = d_;
  params = IndexParams::mvivf();
}

template<bool metric>
IndexMVIVF<metric>::IndexMVIVF(size_t d_, const IndexParams& params_) noexcept {
  d = d_;
  params = params_;
}

template<bool metric>
IndexMVIVF<metric>::~IndexMVIVF() noexcept {
  // Relying on RAII for memory cleanup of contiguous buffers
}

// -----------------------------------------------------------------------------
// Phase 1: Dynamic Tree Construction (Untouched Baseline)
// -----------------------------------------------------------------------------
template<bool metric>
void IndexMVIVF<metric>::recursive_build(build_node_t* node, const PointCloudSet<ChPoint>& points) {
  size_t n = points.size();

  size_t auto_nc =
      (params.k_per_level > 0) ? params.k_per_level : static_cast<size_t>(std::ceil(std::sqrt(n)));
  size_t small_nc = 4 * (n + params.max_leaf_size - 1) / params.max_leaf_size;
  size_t num_clusters = std::max(static_cast<size_t>(4), std::min({auto_nc, small_nc, n}));

  if (params.verbose >= 1) {
    std::cout << "[MVIVF] Building node with " << n << " points, num_clusters: " << num_clusters
              << std::endl;
  }

  MVClustering<metric> Clus(d, num_clusters, params.s, params.mvclus);
  Clus.train(points);
  PointCloudSet<ChPoint>& centers = Clus.get_centers();
  parlay::sequence<uint32_t> cluster_ids = Clus.get_clustering(points);

  auto id_pt = parlay::tabulate(n, [&](uint32_t i) { return std::make_pair(cluster_ids[i], i); });
  auto grouped = group_by_key_inplace(id_pt);

  node->children.resize(grouped.size());

  if (grouped.size() < centers.size()) {
    auto active_centers_ind = parlay::delayed_seq<uint32_t>(
        grouped.size(), [&](size_t i) { return grouped[i][0].first; });
    node->centers = PointCloudSet<ChPoint>(centers.filter(active_centers_ind), d);
  } else {
    node->centers = std::move(centers);
  }

  parlay::parallel_for(
      0, grouped.size(),
      [&](size_t i) {
        auto group = parlay::delayed_seq<uint32_t>(grouped[i].size(),
                                                   [&](size_t j) { return grouped[i][j].second; });
        PointCloudSet<ChPoint> child_points = PointCloudSet<ChPoint>(points.filter(group), d);
        build_node_t* child = new build_node_t();
        node->children[i] = child;

        if (child_points.size() > params.max_leaf_size) {
          recursive_build(child, child_points);
        } else {
          child->leaf_points = std::move(child_points);
        }
      },
      1);
}

// -----------------------------------------------------------------------------
// Phase 2: Tree Flattening
// -----------------------------------------------------------------------------
template<bool metric>
void IndexMVIVF<metric>::flatten_tree(build_node_t* temp_root,
                                      const PointCloudSet<ChPoint>& original_points) {
  size_t total_nodes = 0;
  size_t total_leaves = 0;

  std::function<void(build_node_t*)> compute_stats = [&](build_node_t* node) {
    total_nodes++;
    if (node->children.empty())
      total_leaves++;
    else
      for (auto child : node->children)
        compute_stats(child);
  };
  compute_stats(temp_root);

  index_nodes = parlay::sequence<index_node_t>::uninitialized(total_nodes);
  leaf_to_node.reserve(total_leaves);
  leaf_to_root_child_.clear();

  std::vector<ChPoint> flat_centers;
  std::vector<ChPoint> flat_leaf_centers;

  shuffled_indices.reserve(original_points.size());

  // Queue stores <node, parent's centroid representative, depth, root_child_idx>
  using QueueItem = std::tuple<build_node_t*, std::optional<ChPoint>, size_t, uint32_t>;
  std::queue<QueueItem> q;
  q.push({temp_root, std::nullopt, 1, 0});

  uint32_t node_idx = 0;
  uint32_t next_child_idx = 1;
  size_t max_depth = 0;

  while (!q.empty()) {
    auto [curr, rep, depth, root_child_idx] = q.front();
    q.pop();

    if (depth > max_depth) max_depth = depth;

    index_node_t& flat_curr = index_nodes[node_idx];
    flat_curr.is_leaf = curr->children.empty();

    if (flat_curr.is_leaf) {
      flat_curr.data_offset = shuffled_indices.size();
      flat_curr.num_data = curr->leaf_points.size();

      if (rep.has_value()) {
        leaf_to_node.push_back(node_idx);
        flat_leaf_centers.push_back(rep.value());
        leaf_to_root_child_.push_back(root_child_idx);
      }

      for (size_t i = 0; i < curr->leaf_points.size(); ++i) {
        // get_id natively retrieves the global root index safely due to how 'filter' works
        shuffled_indices.push_back(curr->leaf_points[i].get_id());
      }
    } else {
      flat_curr.children_start = next_child_idx;
      flat_curr.num_children = curr->children.size();
      flat_curr.center_idx_start = flat_centers.size();

      next_child_idx += curr->children.size();

      for (size_t i = 0; i < curr->centers.size(); ++i) {
        flat_centers.push_back(curr->centers[i]);
        uint32_t next_root_child = (node_idx == 0) ? static_cast<uint32_t>(i) : root_child_idx;
        q.push({curr->children[i], curr->centers[i], depth + 1, next_root_child});
      }
    }
    node_idx++;
  }

  kmeanstree_height = max_depth;

  centroids_pcs = PointCloudSet<ChPoint>(flat_centers, d);
  leaf_centroids_pcs = PointCloudSet<ChPoint>(flat_leaf_centers, d);
  clusters_pcs = original_points.shuffle(shuffled_indices);
}

template<bool metric>
void IndexMVIVF<metric>::build(const PointCloudSet<ChPoint>& points) {
  parlay::internal::timer t;

  t.start();
  build_node_t* temp_root = new build_node_t();
  recursive_build(temp_root, points);
  if (params.verbose >= 1) std::cout << "[MVIVF] Phase 1 (MVClustering): " << t.stop() << " sec\n";

  t.reset();
  t.start();
  flatten_tree(temp_root, points);
  if (params.verbose >= 1)
    std::cout << "[MVIVF] Phase 2 (Flatten & Shuffle): " << t.stop() << " sec\n";

  delete temp_root;
}

// -----------------------------------------------------------------------------
// Search Engines
// -----------------------------------------------------------------------------
template<bool metric>
auto IndexMVIVF<metric>::flat_search(const ChPoint& query, size_t nprobes) const
    -> GreedySearchResult {
  GreedySearchResult out;
  if (leaf_centroids_pcs.size() == 0 || nprobes == 0) return out;

  parlay::internal::timer t;
  const size_t L = leaf_centroids_pcs.size();
  const size_t use_nprobes = std::min(nprobes, L);

  t.start();
  auto centers_dists = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(L);

  leaf_centroids_pcs.distances(query, centers_dists.data());
  double t_dists = t.stop();
  t.reset();

  t.start();
  if (use_nprobes < L) {
    std::nth_element(centers_dists.begin(), centers_dists.begin() + use_nprobes,
                     centers_dists.end(),
                     [](const auto& a, const auto& b) { return a.second < b.second; });
    centers_dists.resize(use_nprobes);
  }

  out.probe_list = parlay::sequence<std::pair<float, uint32_t>>::uninitialized(use_nprobes);
  parlay::parallel_for(0, use_nprobes, [&](size_t i) {
    uint32_t leaf_idx = centers_dists[i].first;
    out.probe_list[i] = {centers_dists[i].second, leaf_to_node[leaf_idx]};
  });

  out.dist_cmps = L;
  out.timings = {t_dists, 0.0, t.stop()};
  return out;
}

template<bool metric>
auto IndexMVIVF<metric>::greedy_search(const ChPoint& query, size_t nprobes) const
    -> GreedySearchResult {
  using score_node = std::pair<float, uint32_t>;
  const size_t beam_length = 2 * nprobes;
  parlay::internal::timer t;
  double t_dists = 0.0, t_beam = 0.0, t_rest = 0.0;

  t.start();
  size_t dist_cmps = 0;
  std::set<score_node> beam;
  parlay::sequence<score_node> top_probes;
  top_probes.reserve(nprobes + 1);
  t_rest += t.stop();
  t.reset();

  beam.insert({0.0f, 0});

  while (!beam.empty()) {
    t.start();
    auto it = beam.begin();
    score_node best = *it;
    beam.erase(it);
    t_beam += t.stop();
    t.reset();

    t.start();
    uint32_t current_idx = best.second;
    const index_node_t& current_node = index_nodes[current_idx];

    if (top_probes.size() == nprobes && best.first >= top_probes.front().first) break;

    auto centers_slice = centroids_pcs.slice(
        current_node.center_idx_start, current_node.center_idx_start + current_node.num_children);
    auto child_dists =
        parlay::sequence<std::pair<uint32_t, float>>::uninitialized(current_node.num_children);

    centers_slice.distances(query, child_dists.data());

    dist_cmps += current_node.num_children;
    t_dists += t.stop();
    t.reset();

    for (size_t i = 0; i < current_node.num_children; ++i) {
      float d = child_dists[i].second;
      uint32_t child_idx = current_node.children_start + i;
      const index_node_t& child = index_nodes[child_idx];

      if (child.is_leaf) {
        t.start();
        if (top_probes.size() < nprobes || d < top_probes.front().first) {
          top_probes.push_back({d, child_idx});
          std::push_heap(top_probes.begin(), top_probes.end());
          if (top_probes.size() > nprobes) {
            std::pop_heap(top_probes.begin(), top_probes.end());
            top_probes.pop_back();
          }
        }
        t_rest += t.stop();
        t.reset();
      } else {
        t.start();
        if (beam.size() < beam_length) {
          beam.insert({d, child_idx});
        } else {
          auto worst_it = std::prev(beam.end());
          if (d < worst_it->first) {
            beam.erase(worst_it);
            beam.insert({d, child_idx});
          }
        }
        t_beam += t.stop();
        t.reset();
      }
    }
  }

  GreedySearchResult out;
  out.dist_cmps = dist_cmps;
  out.timings = {t_dists, t_beam, t_rest};
  out.probe_list = std::move(top_probes);
  return out;
}

template<bool metric>
auto IndexMVIVF<metric>::process_probes(
    const ChPoint& query, parlay::sequence<std::pair<float, uint32_t>>& probe_list) const {

  const size_t nprobes = probe_list.size();
  auto sizes = parlay::delayed_tabulate(nprobes, [&](size_t i) -> size_t {
    return static_cast<size_t>(index_nodes[probe_list[i].second].num_data);
  });

  auto scan_result = parlay::scan(sizes);
  auto& offsets = scan_result.first;
  size_t total_size = scan_result.second;

  auto visited = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(total_size);

  parlay::parallel_for(0, nprobes, [&](size_t i) {
    const index_node_t& leaf = index_nodes[probe_list[i].second];
    auto leaf_slice = clusters_pcs.slice(leaf.data_offset, leaf.data_offset + leaf.num_data);
    leaf_slice.distances(query, &visited[offsets[i]]);
  });

  return visited;
}

template<bool metric>
std::tuple<parlay::sequence<std::pair<uint32_t, float>>, size_t, std::vector<double>>
IndexMVIVF<metric>::search_with_stats(const ChPoint& query, const PointCloudSet<ChPoint>& points,
                                      const SearchParams& search_params) {
  parlay::internal::timer t;
  std::vector<double> timings;
  size_t dist_cmps = 0;

  const size_t num_leaves = leaf_centroids_pcs.size();
  const double alpha = 1.0;
  bool use_flat =
      (num_leaves > 0 && search_params.nprobes >= static_cast<size_t>(alpha * num_leaves));

  GreedySearchResult gs;
  if (use_flat) {
    gs = flat_search(query, search_params.nprobes);
  } else {
    gs = greedy_search(query, search_params.nprobes);
  }

  auto& probe_list = gs.probe_list;
  dist_cmps += gs.dist_cmps;

  timings.push_back(dist_cmps);
  for (double time : gs.timings)
    timings.push_back(time);

  t.start();
  auto visited = process_probes(query, probe_list);
  double t_distances = t.stop();
  t.reset();

  dist_cmps += visited.size();

  t.start();
  mvsic::sort_inplace_kv(visited);
  double t_rest = t.stop();
  t.reset();

  timings.push_back(0.0);
  timings.push_back(t_distances);
  timings.push_back(t_rest);

  t.start();
  size_t k = std::min(search_params.k, visited.size());
  auto final_results = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(k);

  if (search_params.num_rerank > 0 && search_params.num_rerank < visited.size()) {
    size_t actual_rerank = std::min(search_params.num_rerank, visited.size());
    dist_cmps += this->rerank(query, points, visited, actual_rerank, final_results);
  } else {
    parlay::parallel_for(0, k, [&](size_t i) { final_results[i] = visited[i]; });
  }

  timings.push_back(t.stop());
  t.reset();

  return std::make_tuple(std::move(final_results), dist_cmps, timings);
}

// -----------------------------------------------------------------------------
// Search All New (Batched Search)
// -----------------------------------------------------------------------------
template<bool metric>
std::pair<parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>, size_t>
IndexMVIVF<metric>::search_all_new(const PointCloudSet<ChPoint>& query_points,
                                   const PointCloudSet<ChPoint>& points,
                                   const SearchParams& search_params) {

  if (search_params.nprobes <= 16) {
    return Index<metric>::search_all(query_points, points, search_params);
  }
  parlay::internal::timer t;

  const size_t num_q = query_points.size();
  const size_t num_leaves = leaf_centroids_pcs.size();
  const size_t k = search_params.k;
  size_t nprobes = std::min(num_leaves, search_params.nprobes);
  size_t dist_cmps = 0;

  // Greedy Search
  auto leaf_query_pairs =
      parlay::sequence<std::pair<uint32_t, std::pair<uint32_t, uint32_t>>>::uninitialized(nprobes *
                                                                                          num_q);
  auto dist_cmps_gs = parlay::sequence<size_t>::uninitialized(num_q);

  parlay::parallel_for(0, num_q, [&](uint32_t i) {
    const double alpha = 1.0;  // heuristic threshold: TODO: set this
    bool use_flat = (num_leaves > 0 && nprobes >= static_cast<size_t>(alpha * num_leaves));
    GreedySearchResult gs;
    if (use_flat) {
      gs = flat_search(query_points[i], nprobes);
    } else {
      gs = greedy_search(query_points[i], nprobes);
    }
    auto& probe_list = gs.probe_list;
    dist_cmps_gs[i] = gs.dist_cmps;
    for (uint32_t j = 0; j < probe_list.size(); j++) {
      leaf_query_pairs[i * nprobes + j] = {probe_list[j].second, std::make_pair(i, j)};
    }
  });

  // Group
  auto grouped = mvsic::group_by_key_inplace(leaf_query_pairs);

  // ---------------------------------------------------------------------
  // Step 3: Leaf Probing
  // ---------------------------------------------------------------------
  size_t num_rerank = std::max(search_params.num_rerank, k);
  size_t max_cands_per_query = nprobes * num_rerank;

  // Allocate flat array for all possible candidates across all queries.
  auto all_candidates = parlay::sequence<std::pair<uint32_t, float>>(
      num_q * max_cands_per_query, {UINT32_MAX, std::numeric_limits<float>::max()});

  parlay::parallel_for(0, grouped.size(), [&](size_t i) {
    auto& group = grouped[i];
    uint32_t leaf_idx = group[0].first;
    const index_node_t& leaf = index_nodes[leaf_idx];
    size_t C = std::min<size_t>(num_rerank, leaf.num_data);
    size_t num_queries_in_group = group.size();

    // Extract the global query IDs that need to probe this specific leaf
    auto query_ids = parlay::delayed_tabulate(num_queries_in_group,
                                              [&](size_t j) { return group[j].second.first; });
    // Synthesize a PointCloudSet for just this batch of queries
    PointCloudSet<ChPoint> batched_queries(query_points.filter(query_ids), d);

    auto leaf_slice = clusters_pcs.slice(leaf.data_offset, leaf.data_offset + leaf.num_data);
    auto leaf_results = leaf_slice.distances(batched_queries, num_rerank);

    // Scatter the batched results back into the global lock-free array
    parlay::parallel_for(0, num_queries_in_group, [&](size_t j) {
      uint32_t q_id = group[j].second.first;
      uint32_t probe_idx = group[j].second.second;
      // Calculate absolute base index in the 1D flat array
      size_t base_idx = (q_id * max_cands_per_query) + (probe_idx * num_rerank);
      for (size_t c = 0; c < C; ++c) {
        all_candidates[base_idx + c] = leaf_results[j * num_rerank + c];
      }
    });
  });

  // ---------------------------------------------------------------------
  // Step 4: Aggregation and Re-ranking (Dedup Skipped)
  // ---------------------------------------------------------------------
  auto final_results = parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>(num_q);
  auto dist_cmps_rerank = parlay::sequence<size_t>::uninitialized(num_q);

  parlay::parallel_for(0, num_q, [&](size_t q_id) {
    size_t base_idx = q_id * max_cands_per_query;
    auto q_cands = parlay::sequence<std::pair<uint32_t, float>>(
        all_candidates.begin() + base_idx, all_candidates.begin() + base_idx + max_cands_per_query);

    // Sort candidates (valid hits bubble to the front, UINT32_MAX sinks to the back)
    parlay::sort_inplace(q_cands, [](const auto& a, const auto& b) { return a.second < b.second; });

    // Collect the top valid candidates (up to num_rerank)
    parlay::sequence<std::pair<uint32_t, float>> top_cands;
    top_cands.reserve(num_rerank);

    for (size_t c = 0; c < q_cands.size() && top_cands.size() < num_rerank; ++c) {
      if (q_cands[c].first == UINT32_MAX) break;  // Reached the empty/padded slots, stop looking
      top_cands.push_back(q_cands[c]);
    }

    // Re-ranking phase
    auto q_final =
        parlay::sequence<std::pair<uint32_t, float>>::uninitialized(std::min(k, top_cands.size()));
    dist_cmps_rerank[q_id] = 0;

    if (search_params.num_rerank > 0) {
      size_t actual_rerank = std::min(num_rerank, top_cands.size());
      dist_cmps_rerank[q_id] =
          this->rerank(query_points[q_id], points, top_cands, actual_rerank, q_final);
    } else {
      for (size_t c = 0; c < q_final.size(); ++c) {
        q_final[c] = top_cands[c];
      }
    }

    final_results[q_id] = std::move(q_final);
  });

  // Aggregate distance comparisons from both the greedy search and the reranking phase
  dist_cmps += parlay::reduce(dist_cmps_gs);
  dist_cmps += parlay::reduce(dist_cmps_rerank);

  return std::make_pair(std::move(final_results), dist_cmps);
}

// -----------------------------------------------------------------------------
// Tree Statistics and Flattened Clustering Information
// -----------------------------------------------------------------------------
template<bool metric>
auto IndexMVIVF<metric>::get_tree_stats() const -> TreeStats {
  TreeStats s;
  if (index_nodes.empty()) return s;

  size_t leaf_size_sum = 0;
  size_t internal_size_sum = 0;
  double sum_child_frac_imbalance = 0.0;
  size_t num_internal_balance_nodes = 0;

  // Traverses the flattened array recursively. Root is always index 0.
  std::function<size_t(uint32_t, size_t)> visit = [&](uint32_t node_idx, size_t depth) -> size_t {
    const index_node_t& node = index_nodes[node_idx];
    if (node.is_leaf) {
      s.num_leaves++;
      size_t leaf_size = node.num_data;
      leaf_size_sum += leaf_size;
      if (depth + 1 > s.height) s.height = depth + 1;
      return leaf_size;
    }

    s.num_internal_nodes++;
    size_t n = node.num_children;
    s.total_point_clouds_internal += n;
    internal_size_sum += n;

    std::vector<size_t> child_subtree_sizes;
    child_subtree_sizes.reserve(node.num_children);
    size_t subtree_total = 0;

    for (size_t i = 0; i < node.num_children; ++i) {
      size_t child_size = visit(node.children_start + i, depth + 1);
      child_subtree_sizes.push_back(child_size);
      subtree_total += child_size;
    }

    // Compute per-node imbalance based on child subtree sizes.
    if (child_subtree_sizes.size() >= 2 && subtree_total > 0) {
      auto [min_it, max_it] =
          std::minmax_element(child_subtree_sizes.begin(), child_subtree_sizes.end());
      size_t min_sz = *min_it;
      size_t max_sz = *max_it;
      double frac_imbalance =
          static_cast<double>(max_sz - min_sz) / static_cast<double>(subtree_total);
      sum_child_frac_imbalance += frac_imbalance;
      if (frac_imbalance > s.max_child_fraction_imbalance) {
        s.max_child_fraction_imbalance = frac_imbalance;
      }
      num_internal_balance_nodes++;
      if (frac_imbalance >= TreeStats::kBadImbalanceThreshold) {
        s.bad_imbalance_entries.emplace_back(subtree_total, frac_imbalance);
      }
    }

    return subtree_total;
  };

  visit(0, 0);

  if (s.num_leaves > 0) s.avg_leaf_size = static_cast<double>(leaf_size_sum) / s.num_leaves;
  if (s.num_internal_nodes > 0) {
    s.avg_internal_node_size = static_cast<double>(internal_size_sum) / s.num_internal_nodes;
  }
  if (num_internal_balance_nodes > 0) {
    s.avg_child_fraction_imbalance =
        sum_child_frac_imbalance / static_cast<double>(num_internal_balance_nodes);
  }
  std::sort(s.bad_imbalance_entries.begin(), s.bad_imbalance_entries.end());
  return s;
}

template<bool metric>
parlay::sequence<uint32_t> IndexMVIVF<metric>::get_flat_clustering() const {
  parlay::sequence<uint32_t> empty;
  if (index_nodes.empty()) return empty;

  std::vector<std::pair<uint32_t, uint32_t>> assignments;
  assignments.reserve(1024);
  size_t max_id = 0;
  size_t leaf_idx = 0;

  std::function<void(uint32_t)> visit = [&](uint32_t node_idx) {
    const index_node_t& node = index_nodes[node_idx];
    if (node.is_leaf) {
      auto leaf_slice = clusters_pcs.slice(node.data_offset, node.data_offset + node.num_data);
      size_t n = leaf_slice.size();
      for (size_t j = 0; j < n; ++j) {
        uint32_t id = leaf_slice.get_id(j);
        assignments.emplace_back(id, static_cast<uint32_t>(leaf_idx));
        if (id > max_id) max_id = id;
      }
      ++leaf_idx;
    } else {
      for (size_t i = 0; i < node.num_children; ++i) {
        visit(node.children_start + i);
      }
    }
  };

  visit(0);
  if (assignments.empty()) return empty;

  parlay::sequence<uint32_t> leaf_of_point(max_id + 1);
  parlay::parallel_for(0, leaf_of_point.size(), [&](size_t i) { leaf_of_point[i] = UINT32_MAX; });

  parlay::parallel_for(0, assignments.size(), [&](size_t i) {
    auto [pid, lid] = assignments[i];
    leaf_of_point[pid] = lid;
  });

  return leaf_of_point;
}

template<bool metric>
parlay::sequence<uint32_t> IndexMVIVF<metric>::get_root_child_clustering() const {
  parlay::sequence<uint32_t> empty;
  if (index_nodes.empty() || index_nodes[0].is_leaf || index_nodes[0].num_children == 0)
    return empty;

  std::vector<std::pair<uint32_t, uint32_t>> assignments;
  assignments.reserve(1024);
  size_t max_id = 0;

  std::function<void(uint32_t, uint32_t)> visit = [&](uint32_t node_idx, uint32_t root_child_idx) {
    const index_node_t& node = index_nodes[node_idx];
    if (node.is_leaf) {
      auto leaf_slice = clusters_pcs.slice(node.data_offset, node.data_offset + node.num_data);
      size_t n = leaf_slice.size();
      for (size_t j = 0; j < n; ++j) {
        uint32_t id = leaf_slice.get_id(j);
        assignments.emplace_back(id, root_child_idx);
        if (id > max_id) max_id = id;
      }
    } else {
      for (size_t i = 0; i < node.num_children; ++i) {
        uint32_t next_root_child_idx = root_child_idx;
        if (node_idx == 0) {
          next_root_child_idx = static_cast<uint32_t>(i);
        }
        visit(node.children_start + i, next_root_child_idx);
      }
    }
  };

  const index_node_t& root = index_nodes[0];
  for (size_t i = 0; i < root.num_children; ++i) {
    visit(root.children_start + i, static_cast<uint32_t>(i));
  }

  if (assignments.empty()) return empty;

  parlay::sequence<uint32_t> root_child_of_point(max_id + 1);
  parlay::parallel_for(0, root_child_of_point.size(),
                       [&](size_t i) { root_child_of_point[i] = UINT32_MAX; });

  parlay::parallel_for(0, assignments.size(), [&](size_t i) {
    auto [pid, rid] = assignments[i];
    root_child_of_point[pid] = rid;
  });

  return root_child_of_point;
}

template<bool metric>
std::vector<std::vector<uint32_t>> IndexMVIVF<metric>::get_leaves_of_point() const {
  std::vector<std::vector<uint32_t>> out;
  if (index_nodes.empty()) return out;

  std::vector<std::pair<uint32_t, uint32_t>> assignments;
  assignments.reserve(1024);
  size_t max_id = 0;
  size_t leaf_idx = 0;

  std::function<void(uint32_t)> visit = [&](uint32_t node_idx) {
    const index_node_t& node = index_nodes[node_idx];
    if (node.is_leaf) {
      auto leaf_slice = clusters_pcs.slice(node.data_offset, node.data_offset + node.num_data);
      size_t n = leaf_slice.size();
      for (size_t j = 0; j < n; ++j) {
        uint32_t id = leaf_slice.get_id(j);
        assignments.emplace_back(id, static_cast<uint32_t>(leaf_idx));
        if (id > max_id) max_id = id;
      }
      ++leaf_idx;
    } else {
      for (size_t i = 0; i < node.num_children; ++i) {
        visit(node.children_start + i);
      }
    }
  };

  visit(0);
  if (assignments.empty()) return out;

  out.resize(max_id + 1);
  for (const auto& [pid, lid] : assignments) {
    out[pid].push_back(lid);
  }
  return out;
}

template<bool metric>
std::vector<std::vector<uint32_t>> IndexMVIVF<metric>::get_root_children_of_point() const {
  auto leaves_of_point = get_leaves_of_point();
  if (leaves_of_point.empty() || leaf_to_root_child_.empty()) return {};
  std::vector<std::vector<uint32_t>> out(leaves_of_point.size());

  for (size_t pid = 0; pid < leaves_of_point.size(); ++pid) {
    std::unordered_set<uint32_t> rids;
    for (uint32_t lid : leaves_of_point[pid]) {
      if (lid < leaf_to_root_child_.size()) rids.insert(leaf_to_root_child_[lid]);
    }
    out[pid].assign(rids.begin(), rids.end());
  }
  return out;
}

// -----------------------------------------------------------------------------
// Serialization
// -----------------------------------------------------------------------------
template<bool metric>
void IndexMVIVF<metric>::save_pcs(std::ofstream& out, const PointCloudSet<ChPoint>& pcs) {
  uint32_t n = pcs.size();
  uint32_t dims = pcs.get_dims();
  size_t total_size = pcs.total_size();

  out.write(reinterpret_cast<const char*>(&n), sizeof(uint32_t));
  out.write(reinterpret_cast<const char*>(&dims), sizeof(uint32_t));
  out.write(reinterpret_cast<const char*>(&total_size), sizeof(size_t));

  if (n == 0) return;

  auto offsets_slice = pcs.get_offsets();
  out.write(reinterpret_cast<const char*>(offsets_slice.begin()), (n + 1) * sizeof(size_t));
  out.write(reinterpret_cast<const char*>(pcs.data()), total_size * dims * sizeof(float));

  auto ids_slice = pcs.get_ids();
  if (ids_slice.size() > 0) {
    bool has_ids = true;
    out.write(reinterpret_cast<const char*>(&has_ids), sizeof(bool));
    out.write(reinterpret_cast<const char*>(ids_slice.begin()), n * sizeof(uint32_t));
  } else {
    bool has_ids = false;
    out.write(reinterpret_cast<const char*>(&has_ids), sizeof(bool));
  }
}

template<bool metric>
PointCloudSet<typename IndexMVIVF<metric>::ChPoint> IndexMVIVF<metric>::load_pcs(
    std::ifstream& in) {
  uint32_t n, dims;
  size_t total_size;

  in.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
  in.read(reinterpret_cast<char*>(&dims), sizeof(uint32_t));
  in.read(reinterpret_cast<char*>(&total_size), sizeof(size_t));

  if (n == 0) return PointCloudSet<ChPoint>();

  parlay::sequence<size_t> offsets(n + 1);
  in.read(reinterpret_cast<char*>(offsets.begin()), (n + 1) * sizeof(size_t));

  parlay::sequence<float> values(total_size * dims);
  in.read(reinterpret_cast<char*>(values.begin()), total_size * dims * sizeof(float));

  bool has_ids;
  in.read(reinterpret_cast<char*>(&has_ids), sizeof(bool));

  if (has_ids) {
    parlay::sequence<uint32_t> ids(n);
    in.read(reinterpret_cast<char*>(ids.begin()), n * sizeof(uint32_t));
    return PointCloudSet<ChPoint>(n, dims, values.data(), offsets.data(), ids.data());
  } else {
    return PointCloudSet<ChPoint>(n, dims, values.data(), offsets.data(), nullptr);
  }
}

template<bool metric>
void IndexMVIVF<metric>::save(const std::string& filename) {
  std::ofstream out(filename, std::ios::binary);
  if (!out.is_open()) {
    std::cerr << "Error opening file for writing: " << filename << std::endl;
    return;
  }

  out.write(reinterpret_cast<const char*>(&kmeanstree_height), sizeof(size_t));

  size_t num_nodes = index_nodes.size();
  out.write(reinterpret_cast<const char*>(&num_nodes), sizeof(size_t));
  out.write(reinterpret_cast<const char*>(index_nodes.begin()), num_nodes * sizeof(index_node_t));

  size_t num_leaves = leaf_to_node.size();
  out.write(reinterpret_cast<const char*>(&num_leaves), sizeof(size_t));
  out.write(reinterpret_cast<const char*>(leaf_to_node.begin()), num_leaves * sizeof(uint32_t));

  size_t num_root_children_map = leaf_to_root_child_.size();
  out.write(reinterpret_cast<const char*>(&num_root_children_map), sizeof(size_t));
  out.write(reinterpret_cast<const char*>(leaf_to_root_child_.data()),
            num_root_children_map * sizeof(uint32_t));

  // Only dump necessary buffers. clusters_pcs is recreated on load via shuffle.
  save_pcs(out, centroids_pcs);
  save_pcs(out, leaf_centroids_pcs);

  size_t num_shuffled = shuffled_indices.size();
  out.write(reinterpret_cast<const char*>(&num_shuffled), sizeof(size_t));
  out.write(reinterpret_cast<const char*>(shuffled_indices.begin()),
            num_shuffled * sizeof(uint32_t));

  out.close();
  if (params.verbose >= 1)
    std::cout << "Successfully saved flat index to " << filename << std::endl;
}

template<bool metric>
void IndexMVIVF<metric>::load(const std::string& filename, const PointCloudSet<ChPoint>& points) {
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cerr << "Error opening file for reading: " << filename << std::endl;
    return;
  }

  in.read(reinterpret_cast<char*>(&kmeanstree_height), sizeof(size_t));

  size_t num_nodes;
  in.read(reinterpret_cast<char*>(&num_nodes), sizeof(size_t));
  index_nodes = parlay::sequence<index_node_t>::uninitialized(num_nodes);
  in.read(reinterpret_cast<char*>(index_nodes.begin()), num_nodes * sizeof(index_node_t));

  size_t num_leaves;
  in.read(reinterpret_cast<char*>(&num_leaves), sizeof(size_t));
  leaf_to_node = parlay::sequence<uint32_t>::uninitialized(num_leaves);
  in.read(reinterpret_cast<char*>(leaf_to_node.begin()), num_leaves * sizeof(uint32_t));

  size_t num_root_children_map;
  in.read(reinterpret_cast<char*>(&num_root_children_map), sizeof(size_t));
  leaf_to_root_child_.resize(num_root_children_map);
  in.read(reinterpret_cast<char*>(leaf_to_root_child_.data()),
          num_root_children_map * sizeof(uint32_t));

  centroids_pcs = load_pcs(in);
  leaf_centroids_pcs = load_pcs(in);

  size_t num_shuffled;
  in.read(reinterpret_cast<char*>(&num_shuffled), sizeof(size_t));
  shuffled_indices = parlay::sequence<uint32_t>::uninitialized(num_shuffled);
  in.read(reinterpret_cast<char*>(shuffled_indices.begin()), num_shuffled * sizeof(uint32_t));

  // Instantly recreate the massive clusters_pcs without reading raw floats from disk
  clusters_pcs = points.shuffle(shuffled_indices);

  in.close();
  if (params.verbose >= 1)
    std::cout << "Successfully loaded flat index from " << filename << std::endl;
}

}  // namespace mvsic