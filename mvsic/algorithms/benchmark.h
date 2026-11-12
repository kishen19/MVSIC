#pragma once

#include "mvsic/core/index.h"

namespace mvsic {

struct BenchmarkResult {
  BatchNeighbors neighbors;
  std::vector<double> seconds;
  std::vector<std::vector<double>> query_seconds;
  size_t selected_repeat = 0;
  double warmup_seconds = 0;
  double first_call_seconds = 0;
};

// Times native calls without Python conversion. Every repetition is retained;
// predictions correspond to the fastest repetition, as in the original runner.
BenchmarkResult benchmark(const Index& index, const PointCloudSetView& queries,
                          const SearchParams& params, const std::string& mode, size_t warmup,
                          size_t repeats);

}  // namespace mvsic
