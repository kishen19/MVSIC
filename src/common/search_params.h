#pragma once

namespace mvivf {

struct SearchParams {
 public:
  // Common Params
  std::string method;  // Index name
  size_t k;            // Number of nearest neighbors;

  // MVIVF Params
  size_t nprobes;      // Number of leaf-level nodes probed
  size_t beam_length;  // Max beam size, default: 2*nprobes

  // Single Vector Heuristic Specific Params
  size_t cands;

  // Vamana Params
  size_t beamSize;
  double cut;
  size_t limit;
  size_t degree_limit;

  // Muvera Params

  // mvivf search params
  static SearchParams mvivf(size_t k, size_t nprobes, size_t beam_length = 0) {
    SearchParams params;
    params.method = "mvivf";
    params.k = k;
    params.nprobes = nprobes;
    if (beam_length == 0) {
      params.beam_length = nprobes;
    } else {
      params.beam_length = beam_length;
    }
    return params;
  }

  // mvivf_flat search params
  static SearchParams mvivf_flat(size_t k, size_t nprobes) {
    SearchParams params;
    params.method = "mvivf_flat";
    params.k = k;
    params.nprobes = nprobes;
    return params;
  }

  // single vector heuristic
  static SearchParams svh(size_t k, size_t nprobes, size_t cands, size_t beam_length = 0) {
    SearchParams params;
    params.method = "svh";
    params.k = k;
    params.nprobes = nprobes;
    params.cands = cands;
    if (beam_length == 0) {
      params.beam_length = nprobes;
    } else {
      params.beam_length = beam_length;
    }
    return params;
  }

  // vamana
  static SearchParams vamana(size_t k, size_t L, double cut, size_t limit, size_t degree_limit) {
    SearchParams params;
    params.method = "vamana";
    params.k = k;
    params.beamSize = L;
    params.cut = cut;
    params.limit = limit;
    params.degree_limit = degree_limit;
    return params;
  }

  // muvera
  static SearchParams muvera(size_t k, size_t L, double cut, size_t limit, size_t degree_limit) {
    SearchParams params;
    params.method = "muvera";
    params.k = k;
    params.beamSize = L;
    params.cut = cut;
    params.limit = limit;
    params.degree_limit = degree_limit;
    return params;
  }

 private:
  SearchParams() = default;
};

}  // namespace mvivf