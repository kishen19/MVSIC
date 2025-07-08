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
#include "fde/fixed_dimensional_encoding.h"

namespace mvivf {

/* =========================================Params Type======================================== */
struct IndexMUVERAParams {
  int num_repetitions = 40;         // Number of independent repetitions for FDE generation
  int num_simhash_projections = 6;  // Number of SimHash projections used to partition space
  int seed = 1;                     // Seed for the FDE generation process
  int projection_dimension = 128;   // Dimension to which points are reduced via random projections
  bool fill_empty_partitions = false;  // Fill empty partitions with nearest point coordinates
  int final_projection_dimension = 0;  // Dimension to which the final FDE is projected
  size_t R = 200;                      // Max Outdegree of the routing graph
  size_t L = 600;                      // Beam length
  double alpha = 1.2;                  // Robust pruning parameter
  bool two_pass = false;               // Two-pass graph construction
  bool verbose = false;                // Print debug statements
};
/* ===================================MUVERA Index Class=================================== */
template<bool metric>
class IndexMUVERA : Index<metric>, IndexMUVERAParams {
 public:
  using ChPoint = Index<metric>::ChPoint;
  using Point = std::conditional_t<metric, Euclidian_Point<float>, Mips_Point<float>>;
  using Range = PointRange<float, Point>;
  using pid = std::pair<size_t, float>;
  using Index<metric>::d;  // Embedding dimension
  size_t d_fde;            // FDE dimension

  Range points_fdes;  // FDEs
  Graph<size_t> G;    // Vamana graph
  BuildParams BP;
  knn_index<Point, Range, size_t> I;  // Vamana index

  IndexMUVERA(size_t d_) noexcept :
      BP(BuildParams(R, L, alpha, two_pass)), I(knn_index<Point, Range, size_t>(BP)) {
    d = d_;
  }
  IndexMUVERA(size_t d_, const IndexMUVERAParams &params) noexcept :
      IndexMUVERAParams(params),
      BP(BuildParams(R, L, alpha, two_pass)),
      I(knn_index<Point, Range, size_t>(BP)) {
    d = d_;
  }
  ~IndexMUVERA();
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
void IndexMUVERA<metric>::build(const PointCloudSet<ChPoint> &points) {
  std::cout << "Building index..." << std::endl;
  // Step 1: Compute FDEs of data point clouds
  auto fdes = parlay::sequence<std::vector<float>>::uninitialized(points.size());
  graph_mining::FixedDimensionalEncodingConfig fde_config(
      d, num_repetitions, num_simhash_projections, seed,
      graph_mining::FixedDimensionalEncodingConfig::AVERAGE, projection_dimension,
      (d == projection_dimension) ? graph_mining::FixedDimensionalEncodingConfig::DEFAULT_IDENTITY
                                  : graph_mining::FixedDimensionalEncodingConfig::AMS_SKETCH,
      fill_empty_partitions, final_projection_dimension);
  parlay::parallel_for(0, points.size(), [&](size_t i) {
    size_t point_size = points.get_size(i);
    float *coords = points.get_coords(i);
    std::vector<float> point_data(coords, coords + point_size * d);
    fdes[i] = graph_mining::GenerateDocumentFixedDimensionalEncoding(point_data, fde_config);
  });
  d_fde = fdes[0].size();
  points_fdes = Range(fdes, d_fde);
  // Step 2: Build Vamana index on the FDEs
  G = Graph<size_t>(BP.R, points.size());
  stats<size_t> BuildStats(G.size());
  I.build_index(G, points_fdes, BuildStats);
  if (verbose) std::cout << "FDE Dimension: " << points_fdes.get_dims() << std::endl;
}

// Returns the top-k point clouds for the query point cloud
// Output format: < [<id, distance>, ...], No. of distance computations >
template<bool metric>
std::pair<parlay::sequence<std::pair<size_t, float>>, size_t> IndexMUVERA<metric>::search(
    const ChPoint &query, const PointCloudSet<ChPoint> &points, const SearchParams &params) {
  size_t k = params.k;
  // Step 1: Compute FDE of the query point cloud
  graph_mining::FixedDimensionalEncodingConfig fde_config(
      d, num_repetitions, num_simhash_projections, seed,
      graph_mining::FixedDimensionalEncodingConfig::DEFAULT_SUM, projection_dimension,
      (d == projection_dimension) ? graph_mining::FixedDimensionalEncodingConfig::DEFAULT_IDENTITY
                                  : graph_mining::FixedDimensionalEncodingConfig::AMS_SKETCH,
      fill_empty_partitions, final_projection_dimension);
  std::vector<float> query_vec(query.data(), query.data() + query.size() * d);
  std::vector<float> query_fde =
      graph_mining::GenerateQueryFixedDimensionalEncoding(query_vec, fde_config);
  assert(query_fde.size() == d_fde);
  // Step 2: Run beam search and collect top cand neighbors
  size_t start_point = I.get_start();
  auto QP = QueryParams(1, params.beamSize, params.cut, params.limit, params.degree_limit);
  auto query_point = Point(query_fde.data(), d_fde, d_fde, -1);
  auto [result, dist_cmps] =
      beam_search<Point, Range, size_t>(query_point, G, points_fdes, start_point, QP);
  parlay::sequence<pid> visited = result.second;
  dist_cmps = (dist_cmps * 2 * d_fde) / d;  // TODO: fix this
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
void IndexMUVERA<metric>::save(const std::string &filename) {
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

  std::string fdes_filename = filename;
  pos = fdes_filename.rfind(".");
  if (pos != std::string::npos) {
    fdes_filename.insert(pos, "_fdes");
  } else {
    // If . not found, append _fdes.bin
    fdes_filename += "_fdes.bin";
  }

  char *fdes_filename_c = (char *)fdes_filename.c_str();
  points_fdes.save(fdes_filename_c);
}

template<bool metric>
void IndexMUVERA<metric>::load(const std::string &filename, const PointCloudSet<ChPoint> &points) {
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

  // Construct fdes filename
  std::string fdes_filename = filename;
  pos = fdes_filename.rfind(".");
  if (pos != std::string::npos) {
    fdes_filename.insert(pos, "_fdes");
  } else {
    fdes_filename += "_fdes.bin";
  }
  char *fdes_filename_c = (char *)fdes_filename.c_str();
  points_fdes = Range(fdes_filename_c);
  d_fde = points_fdes.get_dims();
}

template<bool metric>
IndexMUVERA<metric>::~IndexMUVERA() {
  // traverse_and_delete(root);
  // delete root;
  // root = nullptr;
}

template struct IndexMUVERA<true>;   // Instantiates for L2 metric (metric = true)
template struct IndexMUVERA<false>;  // Instantiates for MIPS      (metric = false)

}  // namespace mvivf