#pragma once

namespace mvsic {

struct SearchParams {
 public:
  // Common Params
  std::string method;  // Index name
  size_t k;            // Number of nearest neighbors;

  // mvivf Params
  size_t nprobes;      // Number of leaf-level nodes probed
  size_t beam_length;  // Max beam size, default: 2*nprobes

  // Single Vector Heuristic Specific Params
  size_t cands;

  // Vamana (muvera, mpool, mvvamana) Params
  size_t L;
  double cut;
  size_t limit;
  size_t degree_limit;

  bool rerank = true;
  size_t num_candidates_to_rerank;

  // mvivf search params
  static SearchParams mvivf(size_t k, size_t nprobes, size_t beam_length = 0) {
    SearchParams params;
    params.method = "mvivf";
    params.k = k;
    params.nprobes = nprobes;
    if (beam_length == 0) {
      params.beam_length = 2 * nprobes;
    } else {
      params.beam_length = beam_length;
    }
    return params;
  }

  // mvivf with MVQ search params
  static SearchParams mvivf_mvq(size_t k, size_t nprobes, size_t cands, size_t beam_length = 0) {
    SearchParams params;
    params.method = "mvivf_mvq";
    params.k = k;
    params.nprobes = nprobes;
    params.cands = cands;
    if (beam_length == 0) {
      params.beam_length = 2 * nprobes;
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

  // mvivf_flat with MVQ search params
  static SearchParams mvivf_flat_mvq(size_t k, size_t nprobes, size_t cands) {
    SearchParams params;
    params.method = "mvivf_flat_mvq";
    params.k = k;
    params.nprobes = nprobes;
    params.cands = cands;
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
  static SearchParams mvvamana(size_t k, size_t L, double cut, size_t limit, size_t degree_limit) {
    SearchParams params;
    params.method = "mvvamana";
    params.k = k;
    params.L = L;
    params.cut = cut;
    params.limit = limit;
    params.degree_limit = degree_limit;
    return params;
  }

  // muvera
  static SearchParams muvera(size_t k, size_t L, double cut, size_t limit, size_t degree_limit,
                             bool rerank, size_t num_candidates_to_rerank) {
    SearchParams params;
    params.method = "muvera";
    params.k = k;
    params.L = L;
    params.cut = cut;
    params.limit = limit;
    params.degree_limit = degree_limit;
    params.rerank = rerank;
    params.num_candidates_to_rerank = num_candidates_to_rerank;
    return params;
  }

  // mean_pooling
  static SearchParams mpool(size_t k, size_t L, double cut, size_t limit, size_t degree_limit,
                            bool rerank, size_t num_candidates_to_rerank) {
    SearchParams params;
    params.method = "mpool";
    params.k = k;
    params.L = L;
    params.cut = cut;
    params.limit = limit;
    params.degree_limit = degree_limit;
    params.rerank = rerank;
    params.num_candidates_to_rerank = num_candidates_to_rerank;
    return params;
  }

  //  private:
  SearchParams() = default;
};

}  // namespace mvsic