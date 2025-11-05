#pragma once

#include <queue>
#include <set>
#include "algorithms/utils/euclidian_point.h"
#include "algorithms/utils/graph.h"
#include "algorithms/utils/mips_point.h"
#include "algorithms/utils/stats.h"
#include "algorithms/utils/types.h"
#include "algorithms/vamana/index.h"
#include "mvsic/core/index.h"
#include "mvsic/core/types/point_range.h"

namespace mvsic {

/* Mean Pooling
 - A multi-vector retrieval system that converts point clouds to a single-vector representation via
 mean-pooling, i.e. each point cloud is represented by the mean of the coordinates of its points.
 - Uses graph-based index (Vamana here) for retrieval.
*/

// Helper function to compute the mean-pooled vector, given a point cloud
template<typename ChPoint>
std::vector<float> mean_pooling(const ChPoint &point, bool normalize = true);

/* =============================Mean-Pooling + Vamana Index Class============================ */
template<bool metric>
class IndexMPool : public Index<metric> {
 public:
  using ChPoint = typename Index<metric>::ChPoint;  // Chamfer Point Type
  using Point = std::conditional_t<metric, Euclidian_Point<float>, Mips_Point<float>>;
  using Range = PointRange<float, Point>;
  using Index<metric>::d;  // Embedding dimension

  IndexParams params;
  Range points_mp;                      // Mean-Pooled vectors
  Graph<uint32_t> G;                    // Vamana graph
  BuildParams BP;                       // Vamana build parameters
  knn_index<Point, Range, uint32_t> I;  // Vamana index

  IndexMPool(uint32_t d_) noexcept :
      params(IndexParams::mpool()),
      BP(BuildParams(params.vamana.R, params.vamana.L, params.vamana.alpha,
                     params.vamana.two_pass)),
      I(knn_index<Point, Range, uint32_t>(BP)) {
    d = d_;
  }
  IndexMPool(uint32_t d_, const IndexParams &params) noexcept :
      params(params),
      BP(BuildParams(params.vamana.R, params.vamana.L, params.vamana.alpha,
                     params.vamana.two_pass)),
      I(knn_index<Point, Range, uint32_t>(BP)) {
    d = d_;
  }

  // Builds the index given PointCloudSet object.
  void build(const PointCloudSet<ChPoint> &points) override {
    // Step 1: Compute Mean-Pooled vectors of the data point clouds
    if (params.verbose >= 1)
      std::cout << "Computing mean-pooled vectors of input point clouds..." << std::endl;
    auto mpvs = parlay::sequence<std::vector<float>>(points.size());
    parlay::parallel_for(0, points.size(),
                         [&](size_t i) { mpvs[i] = mean_pooling(points[i], params.normalize); });
    points_mp = Range(mpvs, d);
    // Step 2: Build Vamana index on the mean-pooled points
    G = Graph<uint32_t>(BP.R, points_mp.size());
    stats<uint32_t> BuildStats(G.size());
    I.build_index(G, points_mp, BuildStats);
  }

  // Returns the top-k point clouds for the query point cloud
  // Output format: < [<id, distance>, ...], # distance comparisons>
  std::pair<parlay::sequence<std::pair<uint32_t, float>>, size_t> search(
      const ChPoint &query, const PointCloudSet<ChPoint> &points,
      const SearchParams &search_params) override {
    size_t k = search_params.k;
    // Step 1: Compute mean-pooling of the query point cloud
    std::vector<float> query_mpv = mean_pooling(query, false);
    auto query_point = Point(query_mpv.data(), d, d, -1);

    // Step 2: Run beam search
    uint32_t start_point = I.get_start();
    auto QP = QueryParams(search_params.num_rerank, search_params.L, search_params.cut,
                          search_params.limit, search_params.degree_limit);
    auto [result, dist_cmps] =
        beam_search<Point, Range, uint32_t>(query_point, G, points_mp, start_point, QP);
    parlay::sequence<std::pair<uint32_t, float>> visited = result.second;
    dist_cmps = (dist_cmps * 2 * d);

    // Step 3: Re-ranking
    auto final_results =
        parlay::sequence<std::pair<uint32_t, float>>::uninitialized(std::min(k, visited.size()));
    if (!search_params.norerank) {
      size_t num_rerank = std::min(search_params.num_rerank, visited.size());
      auto cmp_rerank = parlay::sequence<size_t>::uninitialized(num_rerank);
      auto results_rerank =
          parlay::sequence<std::pair<uint32_t, float>>::from_function(num_rerank, [&](size_t i) {
            uint32_t id = visited[i].first;
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
      parlay::parallel_for(0, final_results.size(),
                           [&](size_t i) { final_results[i] = visited[i]; });
    }
    return std::make_pair(final_results, dist_cmps);
  }

  // Write the index to a file in disk
  void save(const std::string &filename) override {
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

  // Read the index from a file in disk
  void load(const std::string &filename,
            const PointCloudSet<ChPoint> &points) override {  // Construct graph filename
    std::string graph_filename = filename;
    size_t pos = graph_filename.rfind(".");
    if (pos != std::string::npos) {
      graph_filename.insert(pos, "_graph.mpv");
    } else {
      graph_filename += "_graph.mpv.bin";
    }
    char *graph_filename_c = (char *)graph_filename.c_str();
    G = Graph<uint32_t>(graph_filename_c);
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
};

// Compute the mean-pooled vector, given a point cloud
template<typename ChPoint>
std::vector<float> mean_pooling(const ChPoint &point, bool normalize) {
  std::vector<float> mpv(point.get_dims());
  uint32_t point_size = point.size();
  float *coords = point.data();
  uint32_t d = point.get_dims();
  for (size_t j = 0; j < d; j++) {
    auto ent_j =
        parlay::delayed_seq<float>(point_size, [&](size_t k) { return coords[k * d + j]; });
    mpv[j] = parlay::reduce(ent_j) / point_size;
  }
  if (normalize) {
    auto sqrs = parlay::delayed_seq<float>(mpv.size(), [&](size_t j) { return mpv[j] * mpv[j]; });
    float norm = std::sqrt(parlay::reduce(sqrs));
    if (norm > 1e-7) {  // Avoid division by zero
      parlay::parallel_for(0, mpv.size(), [&](size_t j) { mpv[j] /= norm; });
    }
  }
  return mpv;
}

using IndexMPoolL2 = IndexMPool<true>;   // L2 metric
using IndexMPoolIP = IndexMPool<false>;  // MIPS

}  // namespace mvsic