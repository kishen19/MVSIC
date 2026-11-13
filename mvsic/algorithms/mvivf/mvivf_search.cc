#include "mvsic/algorithms/mvivf/mvivf_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>

#include "mvsic/core/utils/interval_heap.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

namespace mvsic {
namespace {

using namespace mvivf_internal;
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
struct ProbeLess {
  bool operator()(const Probe& a, const Probe& b) const noexcept {
    return a.distance != b.distance ? a.distance < b.distance : a.node < b.node;
  }
};
size_t checked_product(size_t a, size_t b) {
  if (b && a > std::numeric_limits<size_t>::max() / b) throw std::length_error("MVIVF batch is too large");
  return a * b;
}

}  // namespace

void IndexMVIVF::Impl::validate_search_params(const SearchParams& p) const {
  if (!kernels) throw std::logic_error("MVIVF must be built or loaded before search");
  if (!p.method.empty() && p.method != "mvivf") throw std::invalid_argument("Expected MVIVF search parameters");
  if (!p.k || !p.nprobes || (p.num_rerank && p.num_rerank < p.k))
    throw std::invalid_argument("Require k > 0, nprobes > 0 and num_rerank == 0 or >= k");
  if (!std::isfinite(p.query_compression_threshold) ||
      (p.query_compression == SearchParams::QueryCompression::Wards &&
       p.query_compression_threshold < 0))
    throw std::invalid_argument("Invalid query compression threshold");
  switch (p.query_compression) {
    case SearchParams::QueryCompression::None:
    case SearchParams::QueryCompression::Carve:
    case SearchParams::QueryCompression::Wards: break;
    default: throw std::invalid_argument("Unknown query compression method");
  }
}

void IndexMVIVF::Impl::validate_query(const PointCloud& query) const {
  if (!query.n || query.d != d || !query.values) throw std::invalid_argument("Invalid MVIVF query shape");
  const bool unit = params.pq.method == IndexParams::QuantizerType::OneBitTQ;
  validate_values(query.values, query.n, d, unit, "query");
}

std::vector<Probe> IndexMVIVF::Impl::find_probes(const PreparedQuery& query, size_t nprobes,
    const Neighbor* root_distances, SearchStats* stats) const {
  if (nodes[0].is_leaf()) return {{0, 0}};
  nprobes = std::min(nprobes, leaves.size());
  const size_t beam_length = std::min(nodes.size(), 2 * nprobes);
  IntervalHeap<Probe, ProbeLess> beam(beam_length);
  std::vector<Probe> top_probes;
  top_probes.reserve(nprobes + 1);
  parlay::sequence<Neighbor> distances;
  beam.push({0, 0});
  const auto total_start = stats ? Clock::now() : Clock::time_point{};
  while (!beam.empty()) {
    auto start = stats ? Clock::now() : Clock::time_point{};
    const Probe best = beam.pop_min();
    if (stats) stats->greedy[1] += seconds(start);
    if (top_probes.size() == nprobes && best.distance >= top_probes.front().distance) break;
    const Node& node = nodes[best.node];
    start = stats ? Clock::now() : Clock::time_point{};
    const Neighbor* scores = root_distances;
    if (best.node != 0 || !root_distances) {
      if (distances.size() < node.children.size())
        distances = parlay::sequence<Neighbor>::uninitialized(node.children.size());
      kernels->center_distances(node, encoded[best.node], query, distances.data());
      scores = distances.data();
      if (stats) stats->bytes += encoded[best.node].center_bytes;
    }
    if (stats) {
      stats->search_cmps += node.children.size();
      stats->greedy[0] += seconds(start);
    }
    for (size_t i = 0; i < node.children.size(); ++i) {
      const Probe child{scores[i].second, node.children[i]};
      start = stats ? Clock::now() : Clock::time_point{};
      if (nodes[child.node].is_leaf()) {
        if (top_probes.size() < nprobes || ProbeLess{}(child, top_probes.front())) {
          top_probes.push_back(child);
          std::push_heap(top_probes.begin(), top_probes.end(), ProbeLess{});
          if (top_probes.size() > nprobes) {
            std::pop_heap(top_probes.begin(), top_probes.end(), ProbeLess{});
            top_probes.pop_back();
          }
        }
        if (stats) stats->greedy[2] += seconds(start);
      } else {
        if (beam.size() < beam_length) beam.push(child);
        else if (ProbeLess{}(child, beam.top_max())) beam.replace_max(child);
        if (stats) stats->greedy[1] += seconds(start);
      }
    }
    if (stats && best.node == 0) stats->greedy[3] = seconds(total_start);
  }
  return top_probes;
}

Neighbors IndexMVIVF::Impl::finish(const PreparedQuery& query, std::span<Neighbor> candidates,
                                  const SearchParams& p, SearchStats* stats) const {
  auto start = stats ? Clock::now() : Clock::time_point{};
  auto select = [&](size_t count, bool ordered) {
    if (count < candidates.size()) {
      std::nth_element(candidates.begin(), candidates.begin() + count, candidates.end(), neighbor_less);
      candidates = candidates.first(count);
    }
    if (ordered) std::sort(candidates.begin(), candidates.end(), neighbor_less);
  };
  select(p.num_rerank ? p.num_rerank : p.k, !p.num_rerank);
  if (stats) stats->selection_seconds += seconds(start);
  if (p.num_rerank) {
    start = stats ? Clock::now() : Clock::time_point{};
    const size_t bytes = kernels->rerank(query, candidates, p.tq8_rerank);
    select(p.k, true);
    if (stats) { stats->bytes += bytes; stats->rerank_seconds += seconds(start); }
  }
  Neighbors result(candidates.begin(), candidates.end());
  return result;
}

Neighbors IndexMVIVF::Impl::search(const PointCloud& query, const SearchParams& p,
                                  SearchStats* stats) const {
  auto prepared = kernels->prepare_query(query, p);
  auto start = stats ? Clock::now() : Clock::time_point{};
  auto probes = find_probes(*prepared, p.nprobes, nullptr, stats);
  if (stats) {
    stats->greedy_seconds = seconds(start);
    stats->compression_seconds = prepared->compression_seconds;
    stats->quantization_seconds = prepared->quantization_seconds;
  }
  start = stats ? Clock::now() : Clock::time_point{};
  auto offsets = parlay::sequence<size_t>::uninitialized(probes.size() + 1);
  offsets[0] = 0;
  for (size_t i = 0; i < probes.size(); ++i) {
    offsets[i + 1] = offsets[i] + nodes[probes[i].node].rows.size();
    if (stats) stats->bytes += encoded[probes[i].node].leaf_bytes;
  }
  auto candidates = parlay::sequence<Neighbor>::uninitialized(offsets.back());
  parlay::parallel_for(0, probes.size(), [&](size_t i) {
    kernels->leaf_distances(encoded[probes[i].node], *prepared, candidates.data() + offsets[i]);
  });
  if (stats) { stats->probe_cmps = candidates.size(); stats->leaf_seconds = seconds(start); }
  return finish(*prepared, {candidates.data(), candidates.size()}, p, stats);
}

Neighbors IndexMVIVF::search(const PointCloud& query, const PointCloudSetView& points,
                             const SearchParams& params) const {
  impl_->validate_points(points);
  return search(query, params);
}
Neighbors IndexMVIVF::search(const PointCloud& query, const SearchParams& params) const {
  impl_->validate_search_params(params);
  impl_->validate_query(query);
  if (params.num_rerank && params.tq8_rerank) impl_->kernels->prepare_rerank();
  return impl_->search(query, params);
}
SearchResultWithStats IndexMVIVF::search_with_stats(const PointCloud& query,
    const PointCloudSetView& points, const SearchParams& params) const {
  impl_->validate_points(points);
  impl_->validate_search_params(params);
  impl_->validate_query(query);
  SearchStats stats;
  if (params.num_rerank && params.tq8_rerank) {
    const auto start = Clock::now();
    impl_->kernels->prepare_rerank();
    stats.rerank_seconds = seconds(start);
  }
  auto result = impl_->search(query, params, &stats);
  // Preserve the established twelve timer/counter positions.
  return {std::move(result), stats.bytes,
      {double(stats.search_cmps), double(stats.probe_cmps), stats.greedy[0], stats.greedy[1],
       stats.greedy[2], stats.greedy[3], stats.compression_seconds, stats.quantization_seconds,
       stats.leaf_seconds, stats.selection_seconds, stats.rerank_seconds, stats.greedy_seconds}};
}
BatchNeighbors IndexMVIVF::search_all(const PointCloudSetView& queries,
    const PointCloudSetView& points, const SearchParams& params) const {
  impl_->validate_points(points);
  return search_all(queries, params);
}

BatchNeighbors IndexMVIVF::search_all(const PointCloudSetView& queries, const SearchParams& params) const {
  queries.validate(false);
  impl_->validate_search_params(params);
  if (queries.d != impl_->d) throw std::invalid_argument("MVIVF query dimension mismatch");
  // Keep small requests serial. Large batches validate independent clouds on
  // workers, but publish the first input error on the caller in row order.
  if (queries.n < 32) {
    for (size_t i = 0; i < queries.n; ++i) impl_->validate_query(queries[i]);
  } else {
    std::mutex error_mutex;
    size_t first_error = queries.n;
    std::exception_ptr error;
    parlay::parallel_for(0, queries.n, [&](size_t i) {
      try {
        impl_->validate_query(queries[i]);
      } catch (...) {
        std::lock_guard lock(error_mutex);
        if (i < first_error) {
          first_error = i;
          error = std::current_exception();
        }
      }
    });
    if (error) std::rethrow_exception(error);
  }
  if (queries.n && params.num_rerank && params.tq8_rerank) impl_->kernels->prepare_rerank();
  BatchNeighbors output(queries.n);
  // Grouping cannot amortize its bookkeeping at one or two probes.
  if (params.nprobes <= 2) {
    parlay::parallel_for(0, queries.n, [&](size_t i) { output[i] = impl_->search(queries[i], params); });
    return output;
  }
  const size_t wanted = std::min<size_t>(params.num_rerank ? params.num_rerank : params.k, impl_->points.n);
  const size_t probes_per_query = std::min(params.nprobes, impl_->leaves.size());
  const bool root_m2m = params.root_m2m && impl_->params.quantize_centers && !impl_->nodes[0].is_leaf();
  // Bound candidate/root scratch while keeping ordinary batches fused. The leaf
  // scatter below owns fixed disjoint ranges, so it needs neither atomics nor
  // uninitialized unused candidate slots.
  const size_t candidate_bytes = checked_product(checked_product(probes_per_query, wanted), sizeof(Neighbor));
  const size_t root_bytes = root_m2m ? checked_product(impl_->nodes[0].children.size(), sizeof(Neighbor)) : 0;
  if (candidate_bytes > std::numeric_limits<size_t>::max() - root_bytes)
    throw std::length_error("MVIVF batch is too large");
  const size_t bytes_per_query = candidate_bytes + root_bytes;
  const size_t batch_size = std::max<size_t>(1, std::min<size_t>(queries.n,
      (size_t(512) << 20) / std::max<size_t>(bytes_per_query, 1)));
  struct Request { uint32_t node, query; size_t offset; };
  for (size_t first = 0; first < queries.n; first += batch_size) {
    const size_t count = std::min(batch_size, size_t(queries.n) - first);
    // Construct owning elements normally; Parlay also destroys them in parallel.
    parlay::sequence<std::unique_ptr<PreparedQuery>> prepared(count);
    parlay::parallel_for(0, count, [&](size_t i) {
      prepared[i] = impl_->kernels->prepare_query(queries[first + i], params);
    });
    parlay::sequence<Neighbor> root_distances;
    if (root_m2m) {
      root_distances = parlay::sequence<Neighbor>::uninitialized(checked_product(count, impl_->nodes[0].children.size()));
      if (!impl_->kernels->root_distances(impl_->encoded[0], {prepared.data(), prepared.size()}, root_distances.data()))
        root_distances.clear();
    }
    parlay::sequence<std::vector<Probe>> probes(count);
    parlay::parallel_for(0, count, [&](size_t i) {
      const Neighbor* root = root_distances.empty() ? nullptr : root_distances.data() + i * impl_->nodes[0].children.size();
      probes[i] = impl_->find_probes(*prepared[i], params.nprobes, root, nullptr);
    });
    // Query-local ranges preserve probe order without serial per-leaf growth.
    // Check the flat allocation bound before scans or uninitialized storage.
    const size_t max_requests = checked_product(count, probes_per_query);
    (void)checked_product(max_requests, sizeof(Request));
    auto request_offsets = parlay::sequence<size_t>::uninitialized(count + 1);
    auto candidate_offsets = parlay::sequence<size_t>::uninitialized(count + 1);
    parlay::parallel_for(0, count, [&](size_t i) {
      request_offsets[i] = probes[i].size();
      size_t size = 0;
      for (const auto& probe : probes[i])
        size += std::min(wanted, impl_->nodes[probe.node].rows.size());
      candidate_offsets[i] = size;
    });
    request_offsets[count] = candidate_offsets[count] = 0;
    parlay::scan_inplace(request_offsets);
    parlay::scan_inplace(candidate_offsets);
    auto requests = parlay::sequence<Request>::uninitialized(request_offsets.back());
    parlay::parallel_for(0, count, [&](size_t i) {
      size_t request = request_offsets[i], offset = 0;
      for (const auto& probe : probes[i]) {
        parlay::assign_uninitialized(requests[request++], Request{probe.node, uint32_t(i), offset});
        offset += std::min(wanted, impl_->nodes[probe.node].rows.size());
      }
    });
    // Preserve ascending query order within each leaf, including group8
    // blocks. Offset retains probe order if a leaf occurs more than once.
    parlay::sort_inplace(requests, [](const Request& a, const Request& b) {
      if (a.node != b.node) return a.node < b.node;
      return a.query != b.query ? a.query < b.query : a.offset < b.offset;
    });
    auto group_offsets = parlay::pack_index(parlay::delayed_tabulate(requests.size(), [&](size_t i) {
      return i == 0 || requests[i].node != requests[i - 1].node;
    }));
    auto candidates = parlay::sequence<Neighbor>::uninitialized(candidate_offsets.back());
    parlay::parallel_for(0, group_offsets.size(), [&](size_t group) {
      const size_t begin = group_offsets[group];
      const size_t end = group + 1 == group_offsets.size() ? requests.size() : group_offsets[group + 1];
      const auto jobs = parlay::make_slice(requests.begin() + begin, requests.begin() + end);
      const uint32_t node = jobs[0].node;
      const size_t k = std::min(wanted, impl_->nodes[node].rows.size());
      auto pointers = parlay::sequence<const PreparedQuery*>::uninitialized(jobs.size());
      for (size_t i = 0; i < jobs.size(); ++i) pointers[i] = prepared[jobs[i].query].get();
      auto scores = parlay::sequence<Neighbor>::uninitialized(checked_product(jobs.size(), k));
      impl_->kernels->leaf_topk(impl_->encoded[node], {pointers.data(), pointers.size()}, k, scores.data());
      for (size_t i = 0; i < jobs.size(); ++i)
        std::copy_n(scores.data() + i * k, k,
                    candidates.data() + candidate_offsets[jobs[i].query] + jobs[i].offset);
    }, 1);
    parlay::parallel_for(0, count, [&](size_t i) {
      output[first + i] = impl_->finish(*prepared[i],
          {candidates.data() + candidate_offsets[i], candidate_offsets[i + 1] - candidate_offsets[i]}, params, nullptr);
    });
  }
  return output;
}

}  // namespace mvsic
