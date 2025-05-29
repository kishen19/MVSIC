#pragma once

namespace mvivf{

struct SearchParams{
  size_t k = 0; // Number of nearest neighbors;
  size_t s = 0; // Default: Average number of embeddings in the input point clouds
  size_t nprobes = 1; // Number of leaf-level nodes probed
  size_t beam_length = 0; // Max beam size, default: 2*nprobes

  size_t k_in = 0;
  size_t k_out = 0;

  // mvivf search params
  SearchParams(size_t k, size_t s, size_t nprobes, size_t beam_length)
    : k(k), s(s), nprobes(nprobes), beam_length(beam_length) {}

  // single vector heuristic
  SearchParams(size_t k, size_t s, size_t nprobes, size_t beam_length, size_t k_in, 
    size_t k_out) : k(k), s(s), nprobes(nprobes), beam_length(beam_length), k_in(k_in),
    k_out(k_out) {}
};

} // namespace mvivf