// bench_mvivf_leaf_diversity.cpp
//
// For a pre-built MVIVF index, measures how dispersed the top-k
// ground-truth neighbors of each query are across leaf clusters.
//
// Usage (file mode only):
//   bazel run //microbenchmark/mvivf:bench_mvivf_leaf_diversity -- \
//     -i <db.pcs> -q <queries.pcs> -gt <gt.bin> -x <index.bin> \
//     [-dist_func <L2|IP>] [-k <u32>] [-mm]
//
// where:
//   - <db.pcs> / <queries.pcs> are PointCloudSet binaries.
//   - <gt.bin> is a ground-truth file in the format used by ReadGT
//              (int k_gt; then k_gt pairs (float dist, uint32_t id) per query).
//   - <index.bin> is an MVIVF index file saved via IndexMVIVF{L2,IP}::save().
//
// For each query, we look at its top-k ground-truth neighbors (by id),
// map each neighbor to its leaf id via the index tree, and count how
// many distinct leaves appear. We then report average, median, min,
// and max leaf counts over all queries.

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "parlay/primitives.h"

#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/core/utils/parse_command_line.h"
#include "mvsic/mvivf/mvivf.h"

using namespace mvsic;

template<typename ChPoint, typename IndexT>
static int run_leaf_diversity(commandLine& P) {
  using PC = PointCloudSet<ChPoint>;

  char* dbFile = P.getOptionValue("-i");
  char* qFile = P.getOptionValue("-q");
  char* gtFile = P.getOptionValue("-gt");
  char* indexFile = P.getOptionValue("-x");

  if (dbFile == nullptr || qFile == nullptr || gtFile == nullptr || indexFile == nullptr) {
    std::cerr << "ERROR: require -i <dbFile>, -q <qFile>, -gt <gtFile>, -x <indexFile>\n";
    return 1;
  }

  uint32_t k = static_cast<uint32_t>(P.getOptionIntValue("-k", 10));
  bool mm = P.getOption("-mm");

  // Load data.
  PC db(dbFile, mm);
  PC queries(qFile, /*is_mmap=*/false);

  if (queries.size() == 0 || db.size() == 0) {
    std::cerr << "ERROR: empty DB or query set.\n";
    return 1;
  }
  if (db.get_dims() != queries.get_dims()) {
    std::cerr << "ERROR: DB dims (" << db.get_dims() << ") != Query dims (" << queries.get_dims()
              << ")\n";
    return 1;
  }

  std::cout << "DB: clouds=" << db.size() << "  dims=" << db.get_dims()
            << "  total_vecs=" << db.total_size() << "  avg_k=" << std::fixed
            << std::setprecision(2) << db.average_size() << "\n";
  std::cout << "Q : clouds=" << queries.size() << "  dims=" << queries.get_dims()
            << "  total_vecs=" << queries.total_size() << "  avg_k=" << std::fixed
            << std::setprecision(2) << queries.average_size() << "\n";
  std::cout << "k=" << k << "\n";

  // Load ground truth.
  std::string gt_path(gtFile);
  auto gt = ReadGT(gt_path, queries.size());
  if (gt.size() != queries.size()) {
    std::cerr << "ERROR: GT size (" << gt.size() << ") != number of queries (" << queries.size()
              << ")\n";
    return 1;
  }
  if (gt[0].size() == 0) {
    std::cerr << "ERROR: GT has zero neighbors per query.\n";
    return 1;
  }

  if (k > gt[0].size()) {
    std::cout << "WARNING: requested k=" << k << " but GT only has " << gt[0].size()
              << " neighbors per query; clamping k.\n";
    k = static_cast<uint32_t>(gt[0].size());
  }

  // Load index.
  IndexParams params = IndexParams::mvivf();
  IndexT index(db.get_dims(), params);
  std::string idx_path(indexFile);
  index.load(idx_path, db);

  // Basic tree stats and balance metrics.
  auto tree_stats = index.get_tree_stats();
  std::cout << "\n=== MVIVF tree structure stats ===\n";
  std::cout << "num_internal_nodes            : " << tree_stats.num_internal_nodes << "\n";
  std::cout << "num_leaves                    : " << tree_stats.num_leaves << "\n";
  std::cout << "avg_leaf_size                 : " << tree_stats.avg_leaf_size << "\n";
  std::cout << "avg_internal_node_size        : " << tree_stats.avg_internal_node_size << "\n";
  std::cout << "total_point_clouds_internal   : " << tree_stats.total_point_clouds_internal << "\n";
  std::cout << "height                        : " << tree_stats.height << "\n";
  std::cout << "avg_child_fraction_imbalance  : " << tree_stats.avg_child_fraction_imbalance
            << "\n";
  std::cout << "max_child_fraction_imbalance  : " << tree_stats.max_child_fraction_imbalance
            << "\n";
  std::cout << "bad_imbalance (>= " << tree_stats.kBadImbalanceThreshold
            << ") (subtree_size, imbalance_ratio), sorted by subtree_size: ";
  for (size_t i = 0; i < tree_stats.bad_imbalance_entries.size(); ++i) {
    if (i > 0) std::cout << ", ";
    const auto& p = tree_stats.bad_imbalance_entries[i];
    std::cout << "(" << p.first << ", " << std::fixed << std::setprecision(4) << p.second << ")";
  }
  std::cout << "\n";

  // Multi-leaf/root-child clustering (supports spill: 1 or 2 leaves/root-children per point).
  auto leaves_of_point = index.get_leaves_of_point();
  auto root_children_of_point = index.get_root_children_of_point();
  if (leaves_of_point.empty()) {
    std::cerr << "ERROR: leaves_of_point is empty; index may not be built correctly.\n";
    return 1;
  }
  if (root_children_of_point.empty()) {
    std::cerr << "ERROR: root_children_of_point is empty; index may not be built correctly.\n";
    return 1;
  }

  const size_t num_queries = queries.size();
  // Distinct counts (original metric).
  std::vector<size_t> leaf_counts(num_queries, 0);
  std::vector<size_t> root_child_counts(num_queries, 0);
  // Greedy set-cover size and best/second-best counts (in greedy order).
  std::vector<size_t> greedy_leaf_cover_size(num_queries, 0);
  std::vector<size_t> greedy_root_cover_size(num_queries, 0);
  std::vector<size_t> best_leaf_count(num_queries, 0);
  std::vector<size_t> second_best_leaf_count(num_queries, 0);
  std::vector<size_t> best_root_child_count(num_queries, 0);
  std::vector<size_t> second_best_root_child_count(num_queries, 0);

  size_t missing_ids = 0;
  size_t max_seen_leaf = 0;
  size_t max_seen_root_child = 0;

  for (size_t qi = 0; qi < num_queries; ++qi) {
    const auto& row = gt[qi];
    const size_t kk = std::min<size_t>(k, row.size());

    // Point IDs for this query's top-k.
    std::vector<uint32_t> pids;
    pids.reserve(kk);
    std::unordered_set<uint32_t> leaves_distinct;
    std::unordered_set<uint32_t> root_children_distinct;
    for (size_t j = 0; j < kk; ++j) {
      uint32_t pid = row[j].first;
      if (pid >= leaves_of_point.size()) {
        ++missing_ids;
        continue;
      }
      const auto& lids = leaves_of_point[pid];
      const auto& rids = root_children_of_point[pid];
      if (lids.empty()) continue;
      pids.push_back(pid);
      for (uint32_t lid : lids) {
        leaves_distinct.insert(lid);
        if (lid > max_seen_leaf) max_seen_leaf = lid;
      }
      for (uint32_t rid : rids) {
        root_children_distinct.insert(rid);
        if (rid != UINT32_MAX && rid > max_seen_root_child) max_seen_root_child = rid;
      }
    }
    leaf_counts[qi] = leaves_distinct.size();
    root_child_counts[qi] = root_children_distinct.size();

    // Greedy set cover for leaves: repeatedly pick the leaf covering the most uncovered points.
    std::vector<size_t> picked_leaf_counts;
    std::unordered_set<size_t> uncovered_leaf;
    for (size_t j = 0; j < pids.size(); ++j) uncovered_leaf.insert(j);
    while (!uncovered_leaf.empty()) {
      size_t best_cover = 0;
      uint32_t best_leaf = UINT32_MAX;
      for (uint32_t lid : leaves_distinct) {
        size_t count = 0;
        for (size_t j : uncovered_leaf) {
          const auto& lids_j = leaves_of_point[pids[j]];
          if (std::find(lids_j.begin(), lids_j.end(), lid) != lids_j.end()) ++count;
        }
        if (count > best_cover) {
          best_cover = count;
          best_leaf = lid;
        }
      }
      if (best_leaf == UINT32_MAX || best_cover == 0) break;
      picked_leaf_counts.push_back(best_cover);
      for (size_t j = 0; j < pids.size(); ++j) {
        if (uncovered_leaf.count(j)) {
          const auto& lids_j = leaves_of_point[pids[j]];
          if (std::find(lids_j.begin(), lids_j.end(), best_leaf) != lids_j.end())
            uncovered_leaf.erase(j);
        }
      }
    }
    greedy_leaf_cover_size[qi] = picked_leaf_counts.size();
    best_leaf_count[qi] = picked_leaf_counts.empty() ? 0 : picked_leaf_counts[0];
    second_best_leaf_count[qi] = picked_leaf_counts.size() >= 2 ? picked_leaf_counts[1] : 0;

    // Greedy set cover for root children.
    std::vector<size_t> picked_root_counts;
    std::unordered_set<size_t> uncovered_root;
    for (size_t j = 0; j < pids.size(); ++j) uncovered_root.insert(j);
    while (!uncovered_root.empty()) {
      size_t best_cover = 0;
      uint32_t best_rid = UINT32_MAX;
      for (uint32_t rid : root_children_distinct) {
        size_t count = 0;
        for (size_t j : uncovered_root) {
          const auto& rids_j = root_children_of_point[pids[j]];
          if (std::find(rids_j.begin(), rids_j.end(), rid) != rids_j.end()) ++count;
        }
        if (count > best_cover) {
          best_cover = count;
          best_rid = rid;
        }
      }
      if (best_rid == UINT32_MAX || best_cover == 0) break;
      picked_root_counts.push_back(best_cover);
      for (size_t j = 0; j < pids.size(); ++j) {
        if (uncovered_root.count(j)) {
          const auto& rids_j = root_children_of_point[pids[j]];
          if (std::find(rids_j.begin(), rids_j.end(), best_rid) != rids_j.end())
            uncovered_root.erase(j);
        }
      }
    }
    greedy_root_cover_size[qi] = picked_root_counts.size();
    best_root_child_count[qi] = picked_root_counts.empty() ? 0 : picked_root_counts[0];
    second_best_root_child_count[qi] =
        picked_root_counts.size() >= 2 ? picked_root_counts[1] : 0;
  }

  if (missing_ids > 0) {
    std::cout << "WARNING: encountered " << missing_ids << " GT ids outside [0, "
              << (leaves_of_point.size() - 1) << "]; those neighbors were ignored.\n";
  }

  // Aggregate statistics.
  if (num_queries == 0) {
    std::cerr << "ERROR: no queries.\n";
    return 1;
  }

  // Leaf-level diversity stats.
  double sum_leaf = std::accumulate(leaf_counts.begin(), leaf_counts.end(), 0.0);
  double avg_leaf = sum_leaf / static_cast<double>(num_queries);

  std::vector<size_t> sorted_leaf_counts = leaf_counts;
  std::sort(sorted_leaf_counts.begin(), sorted_leaf_counts.end());
  size_t min_leaf = sorted_leaf_counts.front();
  size_t max_leaf = sorted_leaf_counts.back();
  double median_leaf;
  if (num_queries % 2 == 1) {
    median_leaf = static_cast<double>(sorted_leaf_counts[num_queries / 2]);
  } else {
    size_t a = sorted_leaf_counts[num_queries / 2 - 1];
    size_t b = sorted_leaf_counts[num_queries / 2];
    median_leaf = 0.5 * (static_cast<double>(a) + static_cast<double>(b));
  }

  std::cout << "\n=== MVIVF leaf diversity for top-" << k << " GT neighbors ===\n";
  std::cout << "Distinct leaves per query (over " << num_queries << " queries):\n";
  std::cout << "  avg   : " << avg_leaf << "\n";
  std::cout << "  median: " << median_leaf << "\n";
  std::cout << "  min   : " << min_leaf << "\n";
  std::cout << "  max   : " << max_leaf << "\n";
  std::cout << "  total leaves seen: " << (max_seen_leaf + 1) << "\n";

  // Greedy set-cover size and best/second-best leaf counts (in greedy order).
  auto agg = [num_queries](const std::vector<size_t>& v, double& avg, double& median,
                           size_t& min_v, size_t& max_v) {
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    avg = sum / static_cast<double>(num_queries);
    std::vector<size_t> s = v;
    std::sort(s.begin(), s.end());
    min_v = s.front();
    max_v = s.back();
    if (num_queries % 2 == 1) {
      median = static_cast<double>(s[num_queries / 2]);
    } else {
      median = 0.5 * (static_cast<double>(s[num_queries / 2 - 1]) + s[num_queries / 2]);
    }
  };
  double avg_greedy_leaf, median_greedy_leaf, avg_best_leaf, median_best_leaf;
  double avg_second_leaf, median_second_leaf;
  size_t min_greedy_leaf, max_greedy_leaf, min_best_leaf, max_best_leaf;
  size_t min_second_leaf, max_second_leaf;
  agg(greedy_leaf_cover_size, avg_greedy_leaf, median_greedy_leaf, min_greedy_leaf, max_greedy_leaf);
  agg(best_leaf_count, avg_best_leaf, median_best_leaf, min_best_leaf, max_best_leaf);
  agg(second_best_leaf_count, avg_second_leaf, median_second_leaf, min_second_leaf, max_second_leaf);
  std::cout << "Greedy min-cover size (leaves, estimate):\n";
  std::cout << "  avg   : " << avg_greedy_leaf << "  median: " << median_greedy_leaf
            << "  min: " << min_greedy_leaf << "  max: " << max_greedy_leaf << "\n";
  std::cout << "Best leaf (1st in greedy order) – count of top-" << k << " in that leaf:\n";
  std::cout << "  avg   : " << avg_best_leaf << "  median: " << median_best_leaf
            << "  min: " << min_best_leaf << "  max: " << max_best_leaf << "\n";
  std::cout << "Second-best leaf (2nd in greedy order) – count of top-" << k << " in that leaf:\n";
  std::cout << "  avg   : " << avg_second_leaf << "  median: " << median_second_leaf
            << "  min: " << min_second_leaf << "  max: " << max_second_leaf << "\n";

  // Root-child (subtree) diversity stats.
  double sum_root = std::accumulate(root_child_counts.begin(), root_child_counts.end(), 0.0);
  double avg_root = sum_root / static_cast<double>(num_queries);

  std::vector<size_t> sorted_root_counts = root_child_counts;
  std::sort(sorted_root_counts.begin(), sorted_root_counts.end());
  size_t min_root = sorted_root_counts.front();
  size_t max_root = sorted_root_counts.back();
  double median_root;
  if (num_queries % 2 == 1) {
    median_root = static_cast<double>(sorted_root_counts[num_queries / 2]);
  } else {
    size_t a = sorted_root_counts[num_queries / 2 - 1];
    size_t b = sorted_root_counts[num_queries / 2];
    median_root = 0.5 * (static_cast<double>(a) + static_cast<double>(b));
  }

  std::cout << "\n=== MVIVF root-child (subtree) diversity for top-" << k << " GT neighbors ===\n";
  std::cout << "Distinct root children per query (over " << num_queries << " queries):\n";
  std::cout << "  avg   : " << avg_root << "\n";
  std::cout << "  median: " << median_root << "\n";
  std::cout << "  min   : " << min_root << "\n";
  std::cout << "  max   : " << max_root << "\n";
  std::cout << "  total root children seen: " << (max_seen_root_child + 1) << "\n";

  double avg_greedy_root, median_greedy_root, avg_best_root, median_best_root;
  double avg_second_root, median_second_root;
  size_t min_greedy_root, max_greedy_root, min_best_root, max_best_root;
  size_t min_second_root, max_second_root;
  agg(greedy_root_cover_size, avg_greedy_root, median_greedy_root, min_greedy_root, max_greedy_root);
  agg(best_root_child_count, avg_best_root, median_best_root, min_best_root, max_best_root);
  agg(second_best_root_child_count, avg_second_root, median_second_root, min_second_root,
      max_second_root);
  std::cout << "Greedy min-cover size (root children, estimate):\n";
  std::cout << "  avg   : " << avg_greedy_root << "  median: " << median_greedy_root
            << "  min: " << min_greedy_root << "  max: " << max_greedy_root << "\n";
  std::cout << "Best root child (1st in greedy order) – count of top-" << k << " in that subtree:\n";
  std::cout << "  avg   : " << avg_best_root << "  median: " << median_best_root
            << "  min: " << min_best_root << "  max: " << max_best_root << "\n";
  std::cout << "Second-best root child (2nd in greedy order) – count of top-" << k << ":\n";
  std::cout << "  avg   : " << avg_second_root << "  median: " << median_second_root
            << "  min: " << min_second_root << "  max: " << max_second_root << "\n";

  return 0;
}

int main(int argc, char** argv) {
  commandLine P(argc, argv,
                "[-dist_func <L2|IP>] "
                "[-k <k>] [-mm] "
                "-i <dbFile> -q <qFile> -gt <gtFile> -x <indexFile>");

  std::string df = P.getOptionValue("-dist_func", "IP");

  if (df == "L2" || df == "l2") {
    return run_leaf_diversity<ChamferL2_Point, IndexMVIVFL2>(P);
  }
  return run_leaf_diversity<ChamferIP_Point, IndexMVIVFIP>(P);
}
