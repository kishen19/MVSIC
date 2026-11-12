#include "mvsic/algorithms/benchmark.h"

#include <chrono>
#include <limits>
#include <stdexcept>
#include "parlay/parallel.h"

namespace mvsic {

BenchmarkResult benchmark(const Index& index, const PointCloudSetView& queries,
                          const SearchParams& params, const std::string& mode, size_t warmup,
                          size_t repeats) {
  queries.validate();
  if (!repeats || repeats > 100000 || warmup > 100000)
    throw std::invalid_argument("invalid benchmark repetition or warmup count");
  const bool batch = mode == "throughput_batch";
  if (!batch && mode != "latency_single" && mode != "latency_parallel")
    throw std::invalid_argument("unknown benchmark mode");
  BenchmarkResult result;
  auto run = [&] {
    using Clock = std::chrono::steady_clock;
    const auto warmup_start = Clock::now();
    for (size_t i = 0; i < warmup; ++i) {
      const auto start = Clock::now();
      if (batch) {
        const auto ignored = index.search_all(queries, params);
        if (i == 0)
          result.first_call_seconds = std::chrono::duration<double>(Clock::now() - start).count();
      } else {
        const auto ignored = index.search(queries[i % queries.size()], params);
        if (i == 0)
          result.first_call_seconds = std::chrono::duration<double>(Clock::now() - start).count();
      }
    }
    if (warmup)
      result.warmup_seconds = std::chrono::duration<double>(Clock::now() - warmup_start).count();
    double best = std::numeric_limits<double>::infinity();
    result.seconds.reserve(repeats);
    if (!batch) result.query_seconds.reserve(repeats);
    for (size_t repeat = 0; repeat < repeats; ++repeat) {
      BatchNeighbors neighbors(batch ? 0 : queries.size());
      double seconds = 0;
      if (batch) {
        const auto start = Clock::now();
        neighbors = index.search_all(queries, params);
        seconds = std::chrono::duration<double>(Clock::now() - start).count();
        if (!warmup && repeat == 0) result.first_call_seconds = seconds;
      } else {
        std::vector<double> timings(queries.size());
        for (size_t i = 0; i < queries.size(); ++i) {
          const auto start = Clock::now();
          neighbors[i] = index.search(queries[i], params);
          timings[i] = std::chrono::duration<double>(Clock::now() - start).count();
          if (!warmup && repeat == 0 && i == 0) result.first_call_seconds = timings[i];
          seconds += timings[i];
        }
        result.query_seconds.push_back(std::move(timings));
      }
      result.seconds.push_back(seconds);
      if (seconds < best) {
        best = seconds;
        result.selected_repeat = repeat;
        result.neighbors = std::move(neighbors);
      }
    }
  };
  if (mode == "latency_single")
    parlay::execute_with_scheduler(1, run);
  else
    run();
  return result;
}

}  // namespace mvsic
