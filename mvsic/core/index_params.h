#pragma once

#include "mvsic/core/mvclustering/mvclustering_config.h"

namespace mvsic {

struct IndexParams {
 public:
  // Common Params
  std::string method;           // Index name
  uint32_t verbose = 0;         // Print debug statements
  bool compress_input = false;  // Compress input points clouds using Ward's method
  bool use_PQ = false;          // Use Product Quantization

  // mvivf
  uint32_t k_per_level = 0;      // Number of clusters at each node. [0 = 4*sqrt(n)]
  uint32_t max_leaf_size = 200;  // Maximum size of leaves (enforced)

  // MVClustering params
  MVClusteringConfig mvclus;
  uint32_t s = 0;  // Size of each centroid point cloud (0 = avg input point cloud size)

  // muvera: FDE params
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

  // mpool and muvera
  bool normalize = true;

  static IndexParams mvivf(uint32_t k_per_level = 0, uint32_t max_leaf_size = 200,
                           bool compress_input = false, bool use_PQ = false, uint32_t verbose = 0,
                           uint32_t niters = 5, uint32_t max_points_per_centroid_inner_kmeans = 20,
                           char *init = "Random", uint32_t seed = 0,
                           bool use_weighted_inner_kmeans = false, uint32_t s = 0) {
    IndexParams params;
    params.method = "mvivf";
    params.k_per_level = k_per_level;
    params.max_leaf_size = max_leaf_size;
    params.compress_input = compress_input;
    params.use_PQ = use_PQ;
    params.verbose = verbose;
    params.mvclus = MVClusteringConfig(niters, max_points_per_centroid_inner_kmeans,
                                       (params.verbose > 0) ? params.verbose - 1 : 0, init, seed,
                                       use_weighted_inner_kmeans);
    params.s = s;
    return params;
  }

  static IndexParams mvivf_flat(uint32_t k_per_level = 0, bool compress_input = false,
                                bool use_PQ = false, uint32_t verbose = 0, uint32_t niters = 5,
                                uint32_t max_points_per_centroid_inner_kmeans = 20,
                                char *init = "Random", uint32_t seed = 0,
                                bool use_weighted_inner_kmeans = false, uint32_t s = 0) {
    IndexParams params;
    params.method = "mvivf_flat";
    params.k_per_level = k_per_level;
    params.compress_input = compress_input;
    params.use_PQ = use_PQ;
    params.verbose = verbose;
    params.mvclus = MVClusteringConfig(niters, max_points_per_centroid_inner_kmeans,
                                       (params.verbose > 0) ? params.verbose - 1 : 0, init, seed,
                                       use_weighted_inner_kmeans);
    params.s = s;
    return params;
  }

  static IndexParams muvera(uint32_t num_repetitions = 20, uint32_t num_simhash_projections = 4,
                            uint32_t seed = 1, uint32_t projection_dimension = 8,
                            bool fill_empty_partitions = false,
                            uint32_t final_projection_dimension = 0, bool normalize = false,
                            uint32_t R = 200, uint32_t L = 600, double alpha = 1.2,
                            bool two_pass = false, uint32_t verbose = 0) {
    IndexParams params;
    params.method = "muvera";
    params.fde = fde_config{
        num_repetitions,       num_simhash_projections,    seed,     projection_dimension,
        fill_empty_partitions, final_projection_dimension, normalize};
    params.vamana = vamana_config{R, L, alpha, two_pass};
    params.verbose = verbose;
    return params;
  }

  static IndexParams mpool(uint32_t R = 200, uint32_t L = 600, double alpha = 1.2,
                           bool two_pass = false, bool normalize = true, uint32_t verbose = 0) {
    IndexParams params;
    params.method = "mpool";
    params.vamana = vamana_config{R, L, alpha, two_pass};
    params.normalize = normalize;
    params.verbose = verbose;
    return params;
  }

  static IndexParams mvvamana(uint32_t R = 200, uint32_t L = 600, double alpha = 1.2,
                              bool two_pass = false, uint32_t verbose = 0) {
    IndexParams params;
    params.method = "mvvamana";
    params.vamana = vamana_config{R, L, alpha, two_pass};
    params.verbose = verbose;
    return params;
  }

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