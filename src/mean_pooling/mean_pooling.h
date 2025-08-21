#pragma once

#include <queue>
#include <set>
#include "algorithms/utils/graph.h"
#include "algorithms/utils/stats.h"
#include "algorithms/utils/types.h"
#include "algorithms/vamana/index.h"
#include "algorithms/utils/euclidian_point.h"
#include "algorithms/utils/mips_point.h"
#include "src/common/index.h"
#include "src/utils/point_range.h"

namespace mvivf {

/* =============================Mean-Pooling + Vamana Index Class============================ */
template<bool metric>
class IndexMPV : Index<metric> {
 public:
  using ChPoint = Index<metric>::ChPoint;
  using Point = std::conditional_t<metric, Euclidian_Point<float>, Mips_Point<float>>;
  using Range = PointRange<float, Point>;
  using pid = std::pair<size_t, float>;
  using Index<metric>::d;  // Embedding dimension

  size_t R = 200;         // Max Outdegree of the routing graph
  size_t L = 600;         // Beam length
  double alpha = 1.2;     // Robust pruning parameter
  bool two_pass = false;  // Two-pass graph construction
  bool verbose = false;   // Print debug statements

  Range points_mp;  // Mean-Pooled points
  Graph<size_t> G;  // Vamana graph
  BuildParams BP;
  knn_index<Point, Range, size_t> I;  // Vamana index

  IndexMPV(size_t d_) noexcept :
      BP(BuildParams(R, L, alpha, two_pass)), I(knn_index<Point, Range, size_t>(BP)) {
    d = d_;
  }
  IndexMPV(size_t d_, const IndexParams &params) noexcept :
      R(params.R),
      L(params.L),
      alpha(params.alpha),
      two_pass(params.two_pass),
      verbose(params.verbose),
      BP(BuildParams(R, L, alpha, two_pass)),
      I(knn_index<Point, Range, size_t>(BP)) {
    d = d_;
  }
  /* ----------------------------Overridden Functions---------------------------- */
  void build(const PointCloudSet<ChPoint> &points) override;
  std::pair<parlay::sequence<std::pair<size_t, float>>, size_t> search(
      const ChPoint &query, const PointCloudSet<ChPoint> &points,
      const SearchParams &params) override;
  void save(const std::string &filename) override;
  void load(const std::string &filename, const PointCloudSet<ChPoint> &points) override;
};

/* =======================================Implementation======================================= */
// Builds the index given a point cloud set.
template<bool metric>
void IndexMPV<metric>::build(const PointCloudSet<ChPoint> &points) {
  std::cout << "Building index..." << std::endl;
  // Step 1: Compute Mean-Pooled points of the data point clouds
  auto mpvs = parlay::sequence<std::vector<float>>::uninitialized(points.size());
  parlay::parallel_for(0, points.size(), [&](size_t i) {
    size_t point_size = points.get_size(i);
    float *coords = points.get_coords(i);
    mpvs[i].resize(d);
    for (size_t j = 0; j < d; j++) {
      auto ent_i =
          parlay::delayed_seq<float>(point_size, [&](size_t k) { return coords[k * d + j]; });
      mpvs[i][j] = parlay::reduce(ent_i) / point_size;  // Mean pooling
    }
  });
  points_mp = Range(mpvs, d);
  // Step 2: Build Vamana index on the mean-pooled points
  G = Graph<size_t>(BP.R, points_mp.size());
  stats<size_t> BuildStats(G.size());
  I.build_index(G, points_mp, BuildStats);
}

// Returns the top-k point clouds for the query point cloud
// Output format: < [<id, distance>, ...], No. of distance computations >
template<bool metric>
std::pair<parlay::sequence<std::pair<size_t, float>>, size_t> IndexMPV<metric>::search(
    const ChPoint &query, const PointCloudSet<ChPoint> &points, const SearchParams &params) {
  size_t k = params.k;
  // Step 1: Compute mean-pooling of the query point cloud
  std::vector<float> query_mpv(d);
  float *q_coords = query.data();
  for (size_t j = 0; j < d; j++) {
    auto ent_i =
        parlay::delayed_seq<float>(query.size(), [&](size_t k) { return q_coords[k * d + j]; });
    query_mpv[j] = parlay::reduce(ent_i) / query.size();  // Mean pooling
  }
  auto query_point = Point(query_mpv.data(), d, d, -1);

  // Step 2: Run beam search and collect top cand neighbors
  size_t start_point = I.get_start();
  auto QP = QueryParams(k, params.beamSize, params.cut, params.limit, params.degree_limit);
  auto [result, dist_cmps] =
      beam_search<Point, Range, size_t>(query_point, G, points_mp, start_point, QP);
  parlay::sequence<pid> visited = result.second;
  dist_cmps = (dist_cmps * 2 * d) / d;  // TODO: fix this
  // Step 3: Re-rank the candidates and return top k
  auto cmp_rerank = parlay::sequence<size_t>::uninitialized(visited.size());
  auto results_rerank =
      parlay::sequence<std::pair<float, size_t>>::from_function(visited.size(), [&](size_t i) {
        size_t id = visited[i].first;
        auto [dist, d_c] = query.distance_w_cmps(points[id]);
        cmp_rerank[i] = d_c;
        return std::make_pair(dist, id);
      });
  dist_cmps += parlay::reduce(cmp_rerank);
  parlay::sort_inplace(results_rerank);
  auto final_results = parlay::sequence<std::pair<size_t, float>>::from_function(
      std::min(k, results_rerank.size()),
      [&](size_t i) { return std::make_pair(results_rerank[i].second, results_rerank[i].first); });
  return std::make_pair(final_results, dist_cmps);
}

template<bool metric>
void IndexMPV<metric>::save(const std::string &filename) {
  std::string graph_filename = filename;
  size_t pos = graph_filename.rfind(".");
  if (pos != std::string::npos) {
    graph_filename.insert(pos, "_graph");
  } else {
    // If . not found, append _graph.bin
    graph_filename += "_graph.bin";
  }

  char *graph_filename_c = (char *)graph_filename.c_str();
  G.save(graph_filename_c);

  std::string mpvs_filename = filename;
  pos = mpvs_filename.rfind(".");
  if (pos != std::string::npos) {
    mpvs_filename.insert(pos, "_mpvs");
  } else {
    // If . not found, append _fdes.bin
    mpvs_filename += "_mpvs.bin";
  }

  char *mpvs_filename_c = (char *)mpvs_filename.c_str();
  points_mp.save(mpvs_filename_c);
}

template<bool metric>
void IndexMPV<metric>::load(const std::string &filename, const PointCloudSet<ChPoint> &points) {
  // Construct graph filename
  std::string graph_filename = filename;
  size_t pos = graph_filename.rfind(".");
  if (pos != std::string::npos) {
    graph_filename.insert(pos, "_graph");
  } else {
    graph_filename += "_graph.bin";
  }
  char *graph_filename_c = (char *)graph_filename.c_str();
  G = Graph<size_t>(graph_filename_c);
  I.set_start();

  // Construct mpvs filename
  std::string mpvs_filename = filename;
  pos = mpvs_filename.rfind(".");
  if (pos != std::string::npos) {
    mpvs_filename.insert(pos, "_mpvs");
  } else {
    mpvs_filename += "_mpvs.bin";
  }
  char *mpvs_filename_c = (char *)mpvs_filename.c_str();
  points_mp = Range(mpvs_filename_c);
}

template struct IndexMPV<true>;   // Instantiates for L2 metric (metric = true)
template struct IndexMPV<false>;  // Instantiates for MIPS      (metric = false)

}  // namespace mvivf