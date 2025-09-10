#pragma once

namespace mvsic {

struct IndexParams {
 public:
  // Common Params
  std::string method;    // Index name
  uint32_t verbose = 0;  // Print debug statements

  // mvivf params
  uint32_t num_clusters = 300;  // Number of clusters
  uint32_t maxsize = 500;       // Maxsize of leaf clusters (enforced)
  uint32_t s = 0;               // Number of points in centroid point cloud
  uint32_t iters = 5;           // Number of Outer Lloyd's Iterations
  uint32_t os_rate = 20;        // Oversampling factor for Inner Kmeans

  // MVQ params
  uint32_t num_leaf_centroids = 16;  // Number of centroids per leaf cluster

  // MUVERA: FDE params
  struct fde_config {
    uint32_t num_repetitions = 20;         // Number of independent repetitions for FDE generation
    uint32_t num_simhash_projections = 4;  // Number of SimHash projections used
    uint32_t seed = 1;                     // Seed for the FDE generation process
    uint32_t projection_dimension = 8;     // Projected Dimension (via random projections)
    bool fill_empty_partitions = false;    // Fill empty partitions with nearest point coordinates
    uint32_t final_projection_dimension = 0;  // Dimension to which the final FDE is projected
    bool normalize = true;
  };
  fde_config fde;

  // Vamana params
  struct vamana_config {
    uint32_t R = 200;
    uint32_t L = 600;
    double alpha = 1.2;
    bool two_pass = false;
  };
  vamana_config vamana;

  // Used by mpool
  bool normalize = true;

  // mvivf index params
  static IndexParams mvivf(size_t num_clusters, size_t maxsize, double s = 1.0, size_t iters = 5,
                           size_t os_rate = 20, uint32_t verbose = 0) {
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

  // mvivf index params
  static IndexParams mvivf_mvq(size_t num_clusters, size_t maxsize, size_t num_leaf_centroids = 16,
                               double s = 1.0, size_t iters = 5, size_t os_rate = 20,
                               uint32_t verbose = 0) {
    IndexParams params;
    params.method = "mvivf_mvq";
    params.num_clusters = num_clusters;
    params.maxsize = maxsize;
    params.num_leaf_centroids = num_leaf_centroids;
    params.s = s;
    params.iters = iters;
    params.os_rate = os_rate;
    params.verbose = verbose;
    return params;
  }

  // mvivf_flat index params
  static IndexParams mvivf_flat(size_t num_clusters, double s = 1.0, size_t iters = 5,
                                size_t os_rate = 20, uint32_t verbose = 0) {
    IndexParams params;
    params.method = "mvivf_flat";
    params.num_clusters = num_clusters;
    params.s = s;
    params.iters = iters;
    params.os_rate = os_rate;
    params.verbose = verbose;
    return params;
  }

  // mvivf_flat index params
  static IndexParams mvivf_flat_mvq(size_t num_clusters, size_t num_leaf_centroids = 16,
                                    double s = 1.0, size_t iters = 5, size_t os_rate = 20,
                                    uint32_t verbose = 0) {
    IndexParams params;
    params.method = "mvivf_flat";
    params.num_clusters = num_clusters;
    params.num_leaf_centroids = num_leaf_centroids;
    params.s = s;
    params.iters = iters;
    params.os_rate = os_rate;
    params.verbose = verbose;
    return params;
  }

  // muvera index params
  static IndexParams muvera(uint32_t num_repetitions = 20, uint32_t num_simhash_projections = 4,
                            uint32_t seed = 1, uint32_t projection_dimension = 8,
                            bool fill_empty_partitions = false,
                            uint32_t final_projection_dimension = 0, bool normalize = false,
                            uint32_t R = 200, uint32_t L = 600, double alpha = 1.2,
                            bool two_pass = false, uint32_t verbose = 0) {
    IndexParams params;
    params.method = "muvera";
    params.fde = fde_config(num_repetitions, num_simhash_projections, seed, projection_dimension,
                            fill_empty_partitions, final_projection_dimension, normalize);
    params.vamana = vamana_config(R, L, alpha, two_pass);
    params.verbose = verbose;
    return params;
  }

  // mpool
  static IndexParams mpool(uint32_t R = 200, uint32_t L = 600, double alpha = 1.2,
                           bool two_pass = false, bool normalize = true, uint32_t verbose = 0) {
    IndexParams params;
    params.method = "mpool";
    params.vamana = vamana_config(R, L, alpha, two_pass);
    params.normalize = normalize;
    params.verbose = verbose;
    return params;
  }

  // vamana
  static IndexParams mvvamana(uint32_t R = 200, uint32_t L = 600, double alpha = 1.2,
                              bool two_pass = false, uint32_t verbose = 0) {
    IndexParams params;
    params.method = "mvvamana";
    params.vamana = vamana_config(R, L, alpha, two_pass);
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

  //  private:
  IndexParams() = default;
};

}  // namespace mvsic