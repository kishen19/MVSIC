#pragma once

#include <queue>
#include <set>

#include "mvsic/core/index.h"
#include "mvsic/core/types/io.h"
#include "mvsic/core/types/quantized_point.h"

// ParlayANN (Vamana) includes
#include "algorithms/utils/beamSearch.h"
#include "algorithms/utils/euclidian_point.h"
#include "algorithms/utils/graph.h"
#include "algorithms/utils/mips_point.h"
#include "algorithms/utils/point_range.h"
#include "algorithms/utils/stats.h"
#include "algorithms/utils/types.h"
#include "algorithms/vamana/index.h"

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
  using Point =
      std::conditional_t<metric, parlayANN::Euclidian_Point<float>, parlayANN::Mips_Point<float>>;
  using QPoint = QuantizedPoint<Point>;
  using Range = parlayANN::PointRange<Point>;
  using QRange = QuantizedPointRange<Point>;
  using Index<metric>::d;  // Embedding dimension

  IndexParams params;
  Range points_mp;                                 // Mean-Pooled vectors
  QRange pq_points_mp;                             // PQ Mean-Pooled vectors
  parlayANN::Graph<uint32_t> G;                    // Vamana graph
  parlayANN::BuildParams BP;                       // Vamana build parameters
  parlayANN::knn_index<Range, Range, uint32_t> I;  // Vamana index

  IndexMPool(uint32_t d_) noexcept :
      params(IndexParams::mpool()),
      BP(parlayANN::BuildParams(params.ann.R, params.ann.L, params.ann.alpha, params.ann.num_pass)),
      I(parlayANN::knn_index<Range, Range, uint32_t>(BP)) {
    d = d_;
  }
  IndexMPool(uint32_t d_, const IndexParams &params) noexcept :
      params(params),
      BP(parlayANN::BuildParams(params.ann.R, params.ann.L, params.ann.alpha, params.ann.num_pass)),
      I(parlayANN::knn_index<Range, Range, uint32_t>(BP)) {
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
    if (params.pq.enabled) {
      pq_points_mp = QRange(points_mp, params.pq.num_blocks, params.pq.num_clusters_per_block,
                              params.pq.sample_size);
    }
    // Step 2: Build Vamana index on the mean-pooled points
    if (params.verbose >= 1) std::cout << "Building ANN Index..." << std::endl;
    G = parlayANN::Graph<uint32_t>(BP.R, points_mp.size());
    parlayANN::stats<uint32_t> BuildStats(G.size());
    I.build_index(G, points_mp, points_mp, BuildStats);
  }

  // Returns the top-k point clouds for the query point cloud
  // Output format: < [<id, distance>, ...], # distance comparisons>
  std::tuple<parlay::sequence<std::pair<uint32_t, float>>, size_t, std::vector<double>>
  search_with_stats(const ChPoint &query, const PointCloudSet<ChPoint> &points,
                    const SearchParams &search_params) override {
    parlay::internal::timer t;
    std::vector<double> timings;

    size_t k = search_params.k;
    // Step 1: Compute mean-pooling of the query point cloud
    t.start();
    std::vector<float> query_mpv = mean_pooling(query, false);
    typename Point::parameters parlayann_pr_params(d);
    Point query_point(reinterpret_cast<typename Point::byte *>(query_mpv.data()), -1,
                      parlayann_pr_params);
    timings.push_back(t.stop());
    t.reset();

    // Step 2: Run beam search
    t.start();
    uint32_t start_point = I.get_start();
    auto QP = parlayANN::QueryParams(search_params.num_rerank, search_params.L, search_params.cut,
                                     points.size(), params.ann.R);
    parlay::sequence<std::pair<uint32_t, float>> visited;
    size_t dist_cmps;

    if (params.pq.enabled) {
      QPoint pq_query_point(query_point, pq_points_mp);
      auto [result, cmps] = parlayANN::beam_search<QPoint, QRange, uint32_t>(
          pq_query_point, G, pq_points_mp, start_point, QP);
      visited = result.second;
      dist_cmps = cmps * pq_query_point.get_dist_cmps();
    } else {
      auto [result, cmps] =
          beam_search<Point, Range, uint32_t>(query_point, G, points_mp, start_point, QP);
      visited = result.second;
      dist_cmps = cmps * 2 * d;
    }
    timings.push_back(t.stop());
    t.reset();

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

  // Write the index to a file in disk
  void save(const std::string &filename) override {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("save: cannot open file: " + filename);

    // Save graph
    parlayANN::io::save_graph(G, out);

    // Save Mean-Pooled vectors (point_range)
    if (params.pq.enabled) {
      pq_points_mp.save(out);
    } else {
      parlayANN::io::save_point_range(points_mp, out);
    }
  }

  // Read the index from a file in disk
  void load(const std::string &filename, const PointCloudSet<ChPoint> &points) override {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("load: cannot open file: " + filename);

    // Load graph
    G = parlayANN::io::load_graph<uint32_t>(in);
    I.set_start();

    // Load Mean-Pooled vectors (point_range)
    if (params.pq.enabled) {
      pq_points_mp = QRange(in);
    } else {
      auto [mpvs_data, loaded_d] = parlayANN::io::read_point_range<Point>(in);
      points_mp = Range(mpvs_data, d);
    }
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