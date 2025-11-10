#pragma once

namespace mvsic {

struct SearchParams {
 public:
  // Common Params
  std::string method;  // Index name
  size_t k;            // Number of nearest neighbors
  size_t num_rerank;   // Number of candidates to rerank: 0 if no reranking

  // mvivf
  size_t nprobes;  // Number of leaf-level nodes probed

  // ANN (muvera, mpool, vamana) Params
  size_t L;
  double cut;

  // muvera, mpool, vamana
  bool norerank = false;

  // mvivf search params
  static SearchParams mvivf(size_t k, size_t nprobes, size_t num_rerank = 0) {
    SearchParams params;
    params.method = "mvivf";
    params.k = k;
    assert(k > 0);
    params.nprobes = nprobes;
    assert(nprobes > 0);
    params.num_rerank = num_rerank;
    assert(num_rerank == 0 || num_rerank >= k);
    return params;
  }

  // mvivf_flat search params
  static SearchParams mvivf_flat(size_t k, size_t nprobes, size_t num_rerank = 0) {
    SearchParams params;
    params.method = "mvivf_flat";
    params.k = k;
    assert(k > 0);
    params.nprobes = nprobes;
    assert(nprobes > 0);
    params.num_rerank = num_rerank;
    assert(num_rerank == 0 || num_rerank >= k);
    return params;
  }

  // single vector heuristic
  // static SearchParams svh(size_t k, size_t nprobes, size_t cands, size_t beam_length = 0) {
  //   SearchParams params;
  //   params.method = "svh";
  //   params.k = k;
  //   params.nprobes = nprobes;
  //   params.cands = cands;
  //   if (beam_length == 0) {
  //     params.beam_length = nprobes;
  //   } else {
  //     params.beam_length = beam_length;
  //   }
  //   return params;
  // }

  // vamana
  static SearchParams vamana(size_t k, size_t L, double cut = 1.35, size_t num_rerank = 0) {
    SearchParams params;
    params.method = "vamana";
    params.k = k;
    params.L = L;
    params.cut = cut;
    params.num_rerank = num_rerank;
    return params;
  }

  // muvera
  static SearchParams muvera(size_t k, size_t L, size_t num_rerank, double cut = 1.35,
                             bool norerank = false) {
    SearchParams params;
    params.method = "muvera";
    params.k = k;
    params.L = L;
    params.cut = cut;
    params.norerank = norerank;
    params.num_rerank = num_rerank;
    assert(num_rerank >= k);
    return params;
  }

  // mean_pooling
  static SearchParams mpool(size_t k, size_t L, size_t num_rerank, double cut = 1.35,
                            bool norerank = false) {
    SearchParams params;
    params.method = "mpool";
    params.k = k;
    params.L = L;
    params.cut = cut;
    params.norerank = norerank;
    params.num_rerank = num_rerank;
    assert(num_rerank >= k);
    return params;
  }

  SearchParams() = default;
};

}  // namespace mvsic