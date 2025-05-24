#include "parlay/sequence.h"
#include "utils/faiss_kmeans.h"

template <typename Range>
float brute_force_cost(const Range& points, const Range& centers, 
    const parlay::sequence<float>& weights) {
  auto min_distances = parlay::delayed_seq<float>(points.size(), [&](size_t i) {
    auto new_distances = parlay::delayed_seq<float>(centers.size(), [&](size_t j) {
      return points[i].distance(centers[j]);
    });
    float smallest_new_distance = reduce(new_distances, parlay::minm<float>());
    return smallest_new_distance*weights[i];
  });
  return parlay::reduce(min_distances);
}

template <typename Range, typename ChPoint>
auto lowerbound(const PointCloudSet<ChPoint>& points, size_t k, size_t s = 0) {
  uint32_t n = points.size();
  uint32_t d = points.get_dims();
  bool is_metric = points[0].is_metric();

  auto num_embeddings = parlay::delayed_seq<size_t>(points.size(),
    [&](size_t i) { return points.get_size(i); });
  if (s == 0) {
    s = parlay::reduce(num_embeddings) / n;
    std::cout << "Average number of embeddings per point: " << s << std::endl;
  }
  auto [offsets, total_emb] = parlay::scan(num_embeddings);

  auto group = parlay::iota(n);
  auto weights = parlay::sequence<float>::uninitialized(total_emb);
  parlay::parallel_for(0, n, [&](size_t i) {
    auto offset = offsets[i];
    float wgh = 1.0/(float)points.get_size(i);
    parlay::parallel_for(offset, offset + num_embeddings[i], [&](size_t j){
      weights[j] = wgh;
    });
  });
  auto data = points.filter_flattened(group);
  auto centers = Range(faiss_wgh_kmeans(data, d, k*s, weights, is_metric), d);
  auto data_range = Range(data, d);
  auto cost = brute_force_cost<float>(data_range, centers, weights);
  return cost;
}