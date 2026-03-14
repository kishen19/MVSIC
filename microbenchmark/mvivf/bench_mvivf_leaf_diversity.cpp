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

  // Flat clustering: map point id -> leaf id.
  auto leaf_of_point = index.get_flat_clustering();
  if (leaf_of_point.size() == 0) {
    std::cerr << "ERROR: flat clustering is empty; index may not be built correctly.\n";
    return 1;
  }

  // Root-level clustering: map point id -> root child (subtree) id.
  auto root_child_of_point = index.get_root_child_clustering();
  if (root_child_of_point.size() == 0) {
    std::cerr << "ERROR: root-child clustering is empty; index may not be built correctly.\n";
    return 1;
  }

  // For each query, count distinct leaves among its top-k GT neighbors.
  const size_t num_queries = queries.size();
  std::vector<size_t> leaf_counts(num_queries, 0);
  std::vector<size_t> root_child_counts(num_queries, 0);

  size_t missing_ids = 0;
  size_t max_seen_leaf = 0;
  size_t max_seen_root_child = 0;

  for (size_t qi = 0; qi < num_queries; ++qi) {
    const auto& row = gt[qi];
    const size_t kk = std::min<size_t>(k, row.size());
    std::unordered_set<uint32_t> leaves;
    leaves.reserve(kk);
    std::unordered_set<uint32_t> root_children;
    root_children.reserve(kk);

    for (size_t j = 0; j < kk; ++j) {
      uint32_t pid = row[j].first;
      if (pid >= leaf_of_point.size()) {
        ++missing_ids;
        continue;
      }
      uint32_t lid = leaf_of_point[pid];
      leaves.insert(lid);
      if (lid != UINT32_MAX && lid > max_seen_leaf) max_seen_leaf = lid;

       if (pid < root_child_of_point.size()) {
        uint32_t rid = root_child_of_point[pid];
        root_children.insert(rid);
        if (rid != UINT32_MAX && rid > max_seen_root_child) max_seen_root_child = rid;
      }
    }
    leaf_counts[qi] = leaves.size();
    root_child_counts[qi] = root_children.size();
  }

  if (missing_ids > 0) {
    std::cout << "WARNING: encountered " << missing_ids << " GT ids outside [0, "
              << (leaf_of_point.size() - 1) << "]; those neighbors were ignored.\n";
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
