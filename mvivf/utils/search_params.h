#pragma once

namespace mvivf{

struct SearchParams{
  size_t k = 0; // Number of nearest neighbors;
  size_t s = 0; // Default: Average number of embeddings in the input point clouds
  size_t nprobes = 1; // Number of leaf-level nodes probed
  size_t beam_length = 0; // Max beam size, default: 2*nprobes
};

} // namespace mvivf