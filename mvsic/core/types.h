#pragma once

#include <cstdint>

namespace mvsic {

// Clustering assignment is independent of search-time leaf quantization.
enum class ClusteringQuantizer : uint8_t { None = 0, TQ8 = 1, TQ4 = 2, TQ1 = 3 };

}  // namespace mvsic
