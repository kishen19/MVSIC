#pragma once

#include <vector>
#include <tuple>
#include <variant>

#include "parlay/primitives.h"

#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/core/distance_measures/one_to_many.h"
#include "search_params.h"
#include "index_params.h"

namespace mvsic {

template<bool metric>
class Index {
 public:
  using ChPoint = std::conditional_t<metric, ChamferL2_Point, ChamferIP_Point>;
  // Embedding dimension
  uint32_t d = 0;
  /* ------------------------------Build Functions------------------------------ */
  // Builds the index given PointCloudSet object.
  virtual void build(const PointCloudSet<ChPoint>& points) {}
  // Builds the index given raw data.
  virtual void build(uint32_t n, const float* data, const size_t* offsets, const uint32_t* ids) {
    PointCloudSet<ChPoint> points(n, d, data, offsets, ids);
    build(points);
  }
  /* ------------------------------Search Functions----------------------------- */
  // Returns the top-k point clouds for the query point cloud
  // Output format: < [<id, distance>, ...], # distance comparisons>
  virtual std::pair<parlay::sequence<std::pair<uint32_t, float>>, size_t> search(
      const ChPoint& query, const PointCloudSet<ChPoint>& points,
      const SearchParams& search_params) {
    auto [results, dist_cmps, timings] = search_with_stats(query, points, search_params);
    return std::make_pair(results, dist_cmps);
  }
  // Returns the top-k point clouds for each of the query point clouds
  // Default: runs search in parallel for each query
  virtual std::pair<parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>, size_t>
  search_all(const PointCloudSet<ChPoint>& query_points, const PointCloudSet<ChPoint>& points,
             const SearchParams& search_params) {
    auto cmps = parlay::sequence<size_t>::uninitialized(query_points.size());
    auto pred = parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>(query_points.size());
    parlay::parallel_for(0, query_points.size(), [&](size_t i) {
      auto [results, dist_cmps_i] = search(query_points[i], points, search_params);
      pred[i] = results;
      cmps[i] = dist_cmps_i;
    });
    return std::make_pair(pred, parlay::reduce(cmps));
  }
  // Returns some running time stats, specific to the index type
  // NOTE: Has to be defined by every index.
  virtual std::tuple<parlay::sequence<std::pair<uint32_t, float>>, size_t, std::vector<double>>
  search_with_stats(const ChPoint& query, const PointCloudSet<ChPoint>& points,
                    const SearchParams& params) {
    return std::make_tuple(parlay::sequence<std::pair<uint32_t, float>>(), 0,
                           std::vector<double>{});
  }

  // Reranks the given candidates and returns the top-k point clouds
  virtual size_t rerank(const ChPoint& query, const PointCloudSet<ChPoint>& points,
                        const parlay::sequence<std::pair<uint32_t, float>>& candidates,
                        size_t num_rerank,
                        parlay::sequence<std::pair<uint32_t, float>>& out_results) {
    auto cmp_rerank = parlay::sequence<size_t>::uninitialized(num_rerank);
    auto results_rerank =
        parlay::sequence<std::pair<uint32_t, float>>::from_function(num_rerank, [&](size_t i) {
          uint32_t id = candidates[i].first;
          auto [dist, d_c] = query.distance_w_cmps(points[id]);
          cmp_rerank[i] = d_c;
          return std::make_pair(id, dist);
        });
    size_t num_cmps = parlay::reduce(cmp_rerank);
    parlay::sort_inplace(results_rerank,
                         [](const auto& a, const auto& b) { return a.second < b.second; });
    parlay::parallel_for(0, out_results.size(),
                         [&](size_t i) { out_results[i] = results_rerank[i]; });
    return num_cmps;
  }

  virtual size_t rerank_opt(const ChPoint& query, const PointCloudSet<ChPoint>& points,
                            const parlay::sequence<std::pair<uint32_t, float>>& candidates,
                            size_t num_rerank,
                            parlay::sequence<std::pair<uint32_t, float>>& out_results) {
    // 1. Get the indices of the candidates to rerank.
    auto candidate_indices =
        parlay::delayed_tabulate(num_rerank, [&](size_t i) { return candidates[i].first; });

    // 2. Create a new PointCloudSet from the candidates using filter and the constructor.
    PointCloudSet<ChPoint> candidates_pcs(points.filter(candidate_indices), points.get_dims());

    // 3. Use OneToMany::TopK to get the top k results from the candidates.
    size_t k = out_results.size();
    OneToMany<ChPoint, PointCloudSet<ChPoint>>::TopKIntoUninitialized(query, candidates_pcs, k,
                                                                      out_results.data());

    // 4. Estimate the number of comparisons.
    size_t num_cmps = (query.size() + candidates_pcs.total_size()) * points.get_dims();
    return num_cmps;
  }

  /* -----------------------------Load/Save Functions-------------------------- */
  // Write the index to a file in disk
  virtual void save(const std::string& filename) {}
  // Read the index from a file in disk
  virtual void load(const std::string& filename, const PointCloudSet<ChPoint>& points) {}
  /* ------------------------------Helper Functions------------------------------ */
  // Only valid for MVIVF
  virtual size_t mean_cluster_size() const noexcept {
    std::cout << "mean_cluster_size() not implemented for this index type" << std::endl;
    return 0;
  }
  virtual size_t max_cluster_size() const noexcept {
    std::cout << "max_cluster_size() not implemented for this index type" << std::endl;
    return 0;
  }
  virtual size_t get_height() const noexcept {
    std::cout << "get_height() not implemented for this index type" << std::endl;
    return 0;
  }
};

// template struct Index<true>;   // Instantiates for L2 metric (metric = true)
// template struct Index<false>;  // Instantiates for MIPS      (metric = false)

}  // namespace mvsic