#pragma once

#include <queue>
#include <set>
#include "algorithms/utils/euclidian_point.h"
#include "algorithms/utils/graph.h"
#include "algorithms/utils/mips_point.h"
#include "algorithms/utils/stats.h"
#include "algorithms/utils/types.h"
#include "algorithms/vamana/index.h"
#include "src/common/index.h"
#include "src/utils/point_range.h"

namespace mvivf {

/* =============================Mean-Pooling + Vamana Index Class============================ */
template<bool metric>
class IndexMPV : public Index<metric> {
 public:
  using ChPoint = Index<metric>::ChPoint;
  using Point = std::conditional_t<metric, Euclidian_Point<float>, Mips_Point<float>>;
  using Range = PointRange<float, Point>;
  using pid = std::pair<size_t, float>;
  using Index<metric>::d;  // Embedding dimension

  bool normalize = true;  // Are the mean-pooled vectors normalized?
  bool verbose = false;   // Print debug statements
  // Vamana parameters
  size_t R = 200;         // Max Outdegree of the routing graph
  size_t L = 600;         // Beam length
  double alpha = 1.2;     // Robust pruning parameter
  bool two_pass = false;  // Two-pass graph construction

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
      BP(BuildParams(R, L, alpha, two_pass)),
      I(knn_index<Point, Range, size_t>(BP)) {
    d = d_;
    normalize = params.normalize;
    verbose = params.verbose;
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
// Compute the mean-pooled vector, given a point cloud
template<typename ChPoint>
std::vector<float> mean_pooling(const ChPoint &point, bool normalize_ = true) {
  std::vector<float> mpv(point.get_dims());
  size_t point_size = point.size();
  float *coords = point.data();
  size_t d = point.get_dims();
  for (size_t j = 0; j < d; j++) {
    auto ent_j =
        parlay::delayed_seq<float>(point_size, [&](size_t k) { return coords[k * d + j]; });
    mpv[j] = parlay::reduce(ent_j) / point_size;
  }
  if (normalize_) {
    auto sqrs = parlay::delayed_seq<float>(mpv.size(), [&](size_t j) { return mpv[j] * mpv[j]; });
    float norm = std::sqrt(parlay::reduce(sqrs));
    if (norm > 1e-7) {  // Avoid division by zero
      parlay::parallel_for(0, mpv.size(), [&](size_t j) { mpv[j] /= norm; });
    }
  }
  return mpv;
}

// Builds the index given a point cloud set.
template<bool metric>
void IndexMPV<metric>::build(const PointCloudSet<ChPoint> &points) {
  std::cout << "Building index..." << std::endl;
  // Step 1: Compute Mean-Pooled vectors of the data point clouds
  auto mpvs = parlay::sequence<std::vector<float>>::uninitialized(points.size());
  parlay::parallel_for(0, points.size(), [&](size_t i) {
    mpvs[i].resize(d);
    mpvs[i] = mean_pooling(points[i], normalize);
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
  std::vector<float> query_mpv = mean_pooling(query, params.normalize);
  auto query_point = Point(query_mpv.data(), d, d, -1);

  // Step 2: Run beam search
  size_t start_point = I.get_start();
  auto QP = QueryParams(k, params.beamSize, params.cut, params.limit, params.degree_limit);
  auto [result, dist_cmps] =
      beam_search<Point, Range, size_t>(query_point, G, points_mp, start_point, QP);
  parlay::sequence<pid> visited = result.second;
  dist_cmps = (dist_cmps * 2 * d) / d;  // TODO: fix this
  // Step 3: Re-rank the candidates and return top k
  // Step 3: Re-rank the candidates and return top k
  auto final_results =
      parlay::sequence<std::pair<size_t, float>>::uninitialized(std::min(k, visited.size()));
  if (params.rerank) {
    auto cmp_rerank = parlay::sequence<size_t>::uninitialized(visited.size());
    auto results_rerank =
        parlay::sequence<std::pair<size_t, float>>::from_function(visited.size(), [&](size_t i) {
          size_t id = visited[i].first;
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
    parlay::parallel_for(0, final_results.size(), [&](size_t i) { final_results[i] = visited[i]; });
  }
  return std::make_pair(final_results, dist_cmps);
}

template<bool metric>
void IndexMPV<metric>::save(const std::string &filename) {
  std::string graph_filename = filename;
  size_t pos = graph_filename.rfind(".");
  if (pos != std::string::npos) {
    graph_filename.insert(pos, "_graph.mpv");
  } else {
    // If . not found, append _graph.mpv.bin
    graph_filename += "_graph.mpv.bin";
  }

  char *graph_filename_c = (char *)graph_filename.c_str();
  G.save(graph_filename_c);

  std::string mpvs_filename = filename;
  pos = mpvs_filename.rfind(".");
  if (pos != std::string::npos) {
    mpvs_filename.insert(pos, "_mpvs.mpv");
  } else {
    // If . not found, append _mpvs.mpv.bin
    mpvs_filename += "_mpvs.mpv.bin";
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
    graph_filename.insert(pos, "_graph.mpv");
  } else {
    graph_filename += "_graph.mpv.bin";
  }
  char *graph_filename_c = (char *)graph_filename.c_str();
  G = Graph<size_t>(graph_filename_c);
  I.set_start();

  // Construct mpvs filename
  std::string mpvs_filename = filename;
  pos = mpvs_filename.rfind(".");
  if (pos != std::string::npos) {
    mpvs_filename.insert(pos, "_mpvs.mpv");
  } else {
    mpvs_filename += "_mpvs.mpv.bin";
  }
  char *mpvs_filename_c = (char *)mpvs_filename.c_str();
  points_mp = Range(mpvs_filename_c);
}

template struct IndexMPV<true>;   // Instantiates for L2 metric (metric = true)
template struct IndexMPV<false>;  // Instantiates for MIPS      (metric = false)

}  // namespace mvivf