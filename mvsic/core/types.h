#pragma once

#include <cstdint>

namespace mvsic {

// The metric selects vector arithmetic. Cloud scores remain directed means.
// L2 is a clustering objective; compiled MVIVF retrieval uses IP/MaxSim only.
enum class Metric : uint8_t { IP = 0, L2 = 1, InnerProduct = IP, SquaredL2 = L2 };

// Clustering assignment is independent of search-time leaf quantization.
enum class ClusteringQuantizer : uint8_t { None = 0, TQ8 = 1, TQ4 = 2, TQ1 = 3 };

}  // namespace mvsic
