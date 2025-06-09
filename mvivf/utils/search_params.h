#pragma once

namespace mvivf {

struct SearchParams {
  // Common Params
  size_t k;            // Number of nearest neighbors;
  size_t nprobes;      // Number of leaf-level nodes probed
  size_t beam_length;  // Max beam size, default: 2*nprobes

  // MVIVF Specific Params

  // Single Vector Heuristic Specific Params
  size_t cands = 0;

  // mvivf search params
  SearchParams(size_t k, size_t nprobes, size_t beam_length_) : k(k), nprobes(nprobes) {
    if (beam_length_ == 0) {
      beam_length = nprobes;
    } else {
      beam_length = beam_length_;
    }
  }

  // single vector heuristic
  SearchParams(size_t k, size_t nprobes, size_t beam_length_, size_t cands) :
      k(k), nprobes(nprobes), cands(cands) {
    if (beam_length_ == 0) {
      beam_length = nprobes;
    } else {
      beam_length = beam_length_;
    }
  }
};

}  // namespace mvivf