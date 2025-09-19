#pragma once

#include <queue>
#include <set>

#include "mvsic/muvera/fde/fixed_dimensional_encoding.h"
#include "mvsic/core/index.h"
#include "mvsic/core/utils/point_range.h"

// ParlayANN (Vamana) includes
#include "algorithms/utils/euclidian_point.h"
#include "algorithms/utils/graph.h"
#include "algorithms/utils/mips_point.h"
#include "algorithms/utils/stats.h"
#include "algorithms/utils/types.h"
#include "algorithms/vamana/index.h"

namespace mvsic {

/* MUVERA
 - Paper: https://arxiv.org/pdf/2405.19504
 - A multi-vector retrieval system that converts point clouds to a single-vector representation
 (FDEs), reducing chamfer distance computations to Inner Product computations.
 - Uses graph-based index (Vamana here) for retrieval.
*/

template<bool metric>
class IndexMUVERA : public Index<metric> {
 public:
  using ChPoint = typename Index<metric>::ChPoint;  // Chamfer Point Type
  using Point = std::conditional_t<metric, Euclidian_Point<float>, Mips_Point<float>>;
  using Range = PointRange<float, Point>;
  using Index<metric>::d;  // Embedding dimension

  IndexParams params;
  uint32_t d_fde;                       // FDE dimension
  Range points_fdes;                    // FDEs
  Graph<uint32_t> G;                    // Vamana graph
  BuildParams BP;                       // Vamana build parameters
  knn_index<Point, Range, uint32_t> I;  // Vamana index

  IndexMUVERA(uint32_t d_) noexcept :
      params(IndexParams::muvera()),
      BP(BuildParams(params.vamana.R, params.vamana.L, params.vamana.alpha,
                     params.vamana.two_pass)),
      I(knn_index<Point, Range, uint32_t>(BP)) {
    d = d_;
  }
  IndexMUVERA(uint32_t d_, const IndexParams &params) noexcept :
      params(params),
      BP(BuildParams(params.vamana.R, params.vamana.L, params.vamana.alpha,
                     params.vamana.two_pass)),
      I(knn_index<Point, Range, uint32_t>(BP)) {
    d = d_;
  }

  // Builds the index given PointCloudSet object.
  void build(const PointCloudSet<ChPoint> &points) override {
    if (params.compress_input) {
      // TODO: run Ward's HAC to compress input point clouds
    }
    // Step 1: Compute FDEs of data point clouds
    if (params.verbose >= 1) std::cout << "Computing FDEs..." << std::endl;
    auto fdes = parlay::sequence<std::vector<float>>(points.size());
    graph_mining::FixedDimensionalEncodingConfig fde_config{
        static_cast<int32_t>(d),
        params.fde.num_repetitions,
        params.fde.num_simhash_projections,
        params.fde.seed,
        (params.fde.normalize) ? graph_mining::FixedDimensionalEncodingConfig::NORMALIZE_L2
                               : graph_mining::FixedDimensionalEncodingConfig::AVERAGE,
        params.fde.projection_dimension,
        (d == params.fde.projection_dimension)
            ? graph_mining::FixedDimensionalEncodingConfig::DEFAULT_IDENTITY
            : graph_mining::FixedDimensionalEncodingConfig::AMS_SKETCH,
        params.fde.fill_empty_partitions,
        params.fde.final_projection_dimension};
    parlay::parallel_for(0, points.size(), [&](size_t i) {
      uint32_t point_size = points.get_size(i);
      float *coords = points.data(i);
      std::vector<float> point_data(coords, coords + point_size * d);
      fdes[i] = graph_mining::GenerateFixedDimensionalEncoding(point_data, fde_config);
    });
    d_fde = fdes[0].size();
    points_fdes = Range(fdes, d_fde);
    if (params.use_PQ) {
      // TODO: PQ
    }
    // Step 2: Build Vamana index on the FDEs
    if (params.verbose >= 1) std::cout << "Building Vamana Index..." << std::endl;
    G = Graph<uint32_t>(BP.R, points.size());
    stats<uint32_t> BuildStats(G.size());
    I.build_index(G, points_fdes, BuildStats);
    if (params.verbose >= 1) std::cout << "FDE Dimension: " << points_fdes.get_dims() << std::endl;
  }

  // Returns the top-k point clouds for the query point cloud
  // Output format: < [<id, distance>, ...], # distance comparisons>
  std::pair<parlay::sequence<std::pair<uint32_t, float>>, size_t> search(
      const ChPoint &query, const PointCloudSet<ChPoint> &points,
      const SearchParams &search_params) override {
    size_t k = search_params.k;
    // Step 1: Compute FDE of the query point cloud
    // FDE config
    graph_mining::FixedDimensionalEncodingConfig fde_config{
        static_cast<int32_t>(d),
        params.fde.num_repetitions,
        params.fde.num_simhash_projections,
        params.fde.seed,
        graph_mining::FixedDimensionalEncodingConfig::DEFAULT_SUM,
        params.fde.projection_dimension,
        (d == params.fde.projection_dimension)
            ? graph_mining::FixedDimensionalEncodingConfig::DEFAULT_IDENTITY
            : graph_mining::FixedDimensionalEncodingConfig::AMS_SKETCH,
        params.fde.fill_empty_partitions,
        params.fde.final_projection_dimension};
    // Query input
    std::vector<float> query_vec(query.data(), query.data() + query.size() * d);
    // Query FDE
    std::vector<float> query_fde =
        graph_mining::GenerateFixedDimensionalEncoding(query_vec, fde_config);
    assert(query_fde.size() == d_fde);

    // Step 2: Run beam search
    uint32_t start_point = I.get_start();
    auto QP = QueryParams(search_params.num_rerank, search_params.L, search_params.cut,
                          search_params.limit, search_params.degree_limit);
    auto query_point = Point(query_fde.data(), d_fde, d_fde, -1);
    auto [result, dist_cmps] =
        beam_search<Point, Range, uint32_t>(query_point, G, points_fdes, start_point, QP);
    parlay::sequence<std::pair<uint32_t, float>> visited = result.second;
    dist_cmps = dist_cmps * 2 * d_fde;

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
      graph_filename.insert(pos, "_graph.muvera");
    } else {
      // If . not found, append _graph.muvera.bin
      graph_filename += "_graph.muvera.bin";
    }

    char *graph_filename_c = (char *)graph_filename.c_str();
    G.save(graph_filename_c);

    std::string fdes_filename = filename;
    pos = fdes_filename.rfind(".");
    if (pos != std::string::npos) {
      fdes_filename.insert(pos, "_fdes.muvera");
    } else {
      // If . not found, append _fdes.muvera.bin
      fdes_filename += "_fdes.muvera.bin";
    }

    char *fdes_filename_c = (char *)fdes_filename.c_str();
    points_fdes.save(fdes_filename_c);
  }

  // Read the index from a file in disk
  void load(const std::string &filename, const PointCloudSet<ChPoint> &points) override {
    // Construct graph filename
    std::string graph_filename = filename;
    size_t pos = graph_filename.rfind(".");
    if (pos != std::string::npos) {
      graph_filename.insert(pos, "_graph.muvera");
    } else {
      graph_filename += "_graph.muvera.bin";
    }
    char *graph_filename_c = (char *)graph_filename.c_str();
    G = Graph<uint32_t>(graph_filename_c);
    I.set_start();

    // Construct fdes filename
    std::string fdes_filename = filename;
    pos = fdes_filename.rfind(".");
    if (pos != std::string::npos) {
      fdes_filename.insert(pos, "_fdes.muvera");
    } else {
      fdes_filename += "_fdes.muvera.bin";
    }
    char *fdes_filename_c = (char *)fdes_filename.c_str();
    points_fdes = Range(fdes_filename_c);
    d_fde = points_fdes.get_dims();
  }
};

using IndexMUVERAL2 = IndexMUVERA<true>;   // L2 metric
using IndexMUVERAIP = IndexMUVERA<false>;  // MIPS

}  // namespace mvsic