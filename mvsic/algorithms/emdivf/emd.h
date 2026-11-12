#pragma once

#include <cstdint>
#include <vector>

namespace emdivf {

// Uniform-mass 1-Wasserstein distance with Euclidean vector ground cost.
// Equal-size clouds use exact minimum-cost assignment. Different sizes use
// exact balanced transport; no vectors are padded or dropped.
double Emd(const float* a, uint32_t na, const float* b, uint32_t nb, uint32_t dim);

// Returns min(EMD, cutoff). Row/column transport lower bounds may avoid the
// assignment solve, without changing nearest-center updates. No quantization.
double EmdCapped(const float* a, uint32_t na, const float* b, uint32_t nb,
                 uint32_t dim, double cutoff);

struct TopKResult {
  std::vector<uint32_t> indices;
  std::vector<double> distances;
  uint32_t assignments = 0;
};

// Exact top-k EMD for equal-cardinality clouds. All ground-cost matrices are
// formed, then a rigorous bidirectional nearest-neighbor lower bound is used
// to avoid assignment solves that cannot enter the result.
TopKResult EmdTopKEqual(const float* query, uint32_t n,
                        const std::vector<const float*>& clouds, uint32_t dim, uint32_t k);

}  // namespace emdivf
