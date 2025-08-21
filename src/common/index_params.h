#pragma once

namespace mvivf {

struct IndexParams {
 public:
  // Common Params
  std::string method;    // Index name
  bool verbose = false;  // Print debug statements

  // mvivf_flat and mvivf params
  size_t num_clusters = 300;  // Number of clusters
  size_t maxsize = 500;       // Maxsize of leaf clusters (enforced)
  double s = 1.0;             // s * (average # vectors)/num_docs
  size_t iters = 5;           // Number of Outer Lloyd's Iterations
  size_t os_rate = 20;        // Oversampling factor for Inner Kmeans

  // MPV and MUVERA params
  size_t R = 32;
  size_t L = 64;
  double alpha = 1.0;
  bool two_pass = false;  // Two-pass graph construction

  // mvivf index params
  static IndexParams mvivf(size_t num_clusters, size_t maxsize, double s = 1.0, size_t iters = 5,
                           size_t os_rate = 20, bool verbose = false) {
    IndexParams params;
    params.method = "mvivf";
    params.num_clusters = num_clusters;
    params.maxsize = maxsize;
    params.s = s;
    params.iters = iters;
    params.os_rate = os_rate;
    params.verbose = verbose;
    return params;
  }

  // mvivf_flat index params
  static IndexParams mvivf_flat(size_t num_clusters, double s = 1.0, size_t iters = 5,
                                size_t os_rate = 20, bool verbose = false) {
    IndexParams params;
    params.method = "mvivf_flat";
    params.num_clusters = num_clusters;
    params.s = s;
    params.iters = iters;
    params.os_rate = os_rate;
    params.verbose = verbose;
    return params;
  }

  // mpv index params
  static IndexParams mpv(size_t R, size_t L, double alpha = 1.2, bool two_pass = false,
                         bool verbose = false) {
    IndexParams params;
    params.method = "mpv";
    params.R = R;
    params.L = L;
    params.alpha = alpha;
    params.two_pass = two_pass;
    params.verbose = verbose;
    return params;
  }

  // single vector heuristic
  // static IndexParams svh(size_t k, size_t nprobes, size_t cands, size_t beam_length = 0) {
  //   IndexParams params;
  //   params.method = "svh";
  //   params.k = k;
  //   params.nprobes = nprobes;
  //   params.cands = cands;
  //   if (beam_length == 0) {
  //     params.beam_length = nprobes;
  //   } else {
  //     params.beam_length = beam_length_;
  //   }
  //   return params;
  // }

  // // vamana
  // static IndexParams vamana(size_t k, size_t L, double cut, size_t limit, size_t degree_limit) {
  //   IndexParams params;
  //   params.method = "vamana";
  //   params.k = k;
  //   params.beamSize = L;
  //   params.cut = cut;
  //   params.limit = limit;
  //   params.degree_limit = degree_limit;
  //   return params;
  // }

  // // muvera
  // static IndexParams muvera(size_t k, size_t L, double cut, size_t limit, size_t degree_limit) {
  //   IndexParams params;
  //   params.method = "muvera";
  //   params.k = k;
  //   params.beamSize = L;
  //   params.cut = cut;
  //   params.limit = limit;
  //   params.degree_limit = degree_limit;
  //   return params;
  // }

 private:
  IndexParams() = default;
};

}  // namespace mvivf