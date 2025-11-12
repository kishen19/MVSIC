#pragma once

#include <string>
#include <iostream>
#include "mvsic/core/mvclustering/mvclustering_config.h"

namespace mvsic {

struct IndexParams {
 public:
  // Common Params
  std::string method;           // Index name
  uint32_t verbose = 0;         // Print debug statements
  bool compress_input = false;  // Compress input points clouds using Ward's method

  // mvivf
  uint32_t k_per_level = 0;      // Number of clusters at each node. [0 = 4*sqrt(n)]
  uint32_t max_leaf_size = 200;  // Maximum size of leaves (enforced)

  // MVClustering params
  MVClusteringConfig mvclus = MVClusteringConfig();
  uint32_t s = 0;  // Size of each centroid point cloud (0 = avg input point cloud size)

  // muvera: FDE params
  struct fde_config {
    int32_t num_repetitions = 20;            // Number of independent repetitions for FDE generation
    int32_t num_simhash_projections = 4;     // Number of SimHash projections used
    int32_t seed = 1;                        // Seed for the FDE generation process
    int32_t projection_dimension = 8;        // Projected Dimension (via random projections)
    bool fill_empty_partitions = false;      // Fill empty partitions with nearest point coordinates
    int32_t final_projection_dimension = 0;  // Dimension to which the final FDE is projected
    bool normalize = true;
  };
  fde_config fde = fde_config();

  // ANN params (Using ParlayANN's Vamana)
  struct ann_config {
    uint32_t R = 200;
    uint32_t L = 600;
    double alpha = 1.2;
    int num_pass = 1;
  };
  ann_config ann = ann_config();

  // mpool and muvera
  bool normalize = true;

  // vamana params
  uint32_t R = 200;
  uint32_t L = 600;
  double alpha = 1.2;
  bool two_pass = false;

  // PQ params
  struct pq_config {
    bool enabled = false;                   // Whether to use PQ
    uint32_t num_blocks = 8;                // Number of blocks
    uint32_t num_clusters_per_block = 256;  // Number of clusters per block
    uint32_t sample_size = 100000;          // Sample size for training
  };
  pq_config pq = pq_config();

  static IndexParams mvivf(uint32_t k_per_level = 0, uint32_t max_leaf_size = 200,
                           bool compress_input = false, uint32_t verbose = 0, uint32_t niters = 5,
                           uint32_t max_points_per_centroid_inner_kmeans = 20,
                           std::string init = "Random", uint32_t seed = 0,
                           bool use_weighted_inner_kmeans = false, uint32_t s = 0,
                           bool pq_enabled = false, uint32_t num_blocks = 8,
                           uint32_t num_clusters_per_block = 256, uint32_t sample_size = 100000) {
    IndexParams params;
    params.method = "mvivf";
    params.k_per_level = k_per_level;
    params.max_leaf_size = max_leaf_size;
    params.compress_input = compress_input;
    params.pq = {pq_enabled, num_blocks, num_clusters_per_block, sample_size};
    params.verbose = verbose;
    params.mvclus = MVClusteringConfig(niters, max_points_per_centroid_inner_kmeans,
                                       (params.verbose > 0) ? params.verbose - 1 : 0, init, seed,
                                       use_weighted_inner_kmeans);
    params.s = s;
    return params;
  }

  static IndexParams mvivf_flat(uint32_t k_per_level = 0, bool compress_input = false,
                                uint32_t verbose = 0, uint32_t niters = 5,
                                uint32_t max_points_per_centroid_inner_kmeans = 20,
                                std::string init = "Random", uint32_t seed = 0,
                                bool use_weighted_inner_kmeans = false, uint32_t s = 0,
                                bool pq_enabled = false, uint32_t num_blocks = 8,
                                uint32_t num_clusters_per_block = 256,
                                uint32_t sample_size = 100000) {
    IndexParams params;
    params.method = "mvivf_flat";
    params.k_per_level = k_per_level;
    params.compress_input = compress_input;
    params.pq = {pq_enabled, num_blocks, num_clusters_per_block, sample_size};
    params.verbose = verbose;
    params.mvclus = MVClusteringConfig(niters, max_points_per_centroid_inner_kmeans,
                                       (params.verbose > 0) ? params.verbose - 1 : 0, init, seed,
                                       use_weighted_inner_kmeans);
    params.s = s;
    return params;
  }

  static IndexParams muvera_custom(
      int32_t num_repetitions = 20, int32_t num_simhash_projections = 4, int32_t seed = 1,
      int32_t projection_dimension = 8, bool fill_empty_partitions = false,
      int32_t final_projection_dimension = 0, bool normalize = false, uint32_t R = 200,
      uint32_t L = 600, double alpha = 1.1, int num_pass = 1, bool compress_input = false,
      uint32_t verbose = 0, bool pq_enabled = false, uint32_t num_blocks = 8,
      uint32_t num_clusters_per_block = 256, uint32_t sample_size = 100000) {
    IndexParams params;
    params.method = "muvera";
    params.compress_input = compress_input;
    params.pq = {pq_enabled, num_blocks, num_clusters_per_block, sample_size};
    params.fde = fde_config{
        num_repetitions,       num_simhash_projections,    seed,     projection_dimension,
        fill_empty_partitions, final_projection_dimension, normalize};
    params.ann = ann_config{R, L, alpha, num_pass};
    params.verbose = verbose;
    return params;
  }

  static IndexParams muvera(int32_t d_fde = 2560, int32_t seed = 1,
                            bool fill_empty_partitions = false,
                            int32_t final_projection_dimension = 0, bool normalize = false,
                            uint32_t R = 200, uint32_t L = 600, double alpha = 1.1,
                            int num_pass = 1, bool compress_input = false, uint32_t verbose = 0,
                            bool pq_enabled = false, uint32_t num_blocks = 8,
                            uint32_t num_clusters_per_block = 256, uint32_t sample_size = 100000) {
    IndexParams params;
    params.method = "muvera";
    params.compress_input = compress_input;
    params.pq = {pq_enabled, num_blocks, num_clusters_per_block, sample_size};
    int32_t num_repetitions, num_simhash_projections, projection_dimension;
    if (d_fde == 640) {
      num_repetitions = 10;
      num_simhash_projections = 3;
      projection_dimension = 8;
    } else if (d_fde == 1280) {
      num_repetitions = 20;
      num_simhash_projections = 3;
      projection_dimension = 8;
    } else if (d_fde == 1920) {
      num_repetitions = 15;
      num_simhash_projections = 4;
      projection_dimension = 8;
    } else if (d_fde == 2560) {
      num_repetitions = 20;
      num_simhash_projections = 4;
      projection_dimension = 8;
    } else if (d_fde == 5120) {
      num_repetitions = 20;
      num_simhash_projections = 5;
      projection_dimension = 8;
    } else if (d_fde == 10240) {
      num_repetitions = 20;
      num_simhash_projections = 5;
      projection_dimension = 16;
    } else {
      std::cout
          << "Invalid value for d_fde: allowed values are {640, 1280, 1920, 2560, 5120, 10240}"
          << std::endl;
      abort();
    }
    params.fde = fde_config{
        num_repetitions,       num_simhash_projections,    seed,     projection_dimension,
        fill_empty_partitions, final_projection_dimension, normalize};
    params.ann = ann_config{R, L, alpha, num_pass};
    params.verbose = verbose;
    return params;
  }

  static IndexParams mpool(uint32_t R = 200, uint32_t L = 600, double alpha = 1.2, int num_pass = 1,
                           bool normalize = true, bool compress_input = false, uint32_t verbose = 0,
                           bool pq_enabled = false, uint32_t num_blocks = 8,
                           uint32_t num_clusters_per_block = 256, uint32_t sample_size = 100000) {
    IndexParams params;
    params.method = "mpool";
    params.compress_input = compress_input;
    params.pq = {pq_enabled, num_blocks, num_clusters_per_block, sample_size};
    params.ann = ann_config{R, L, alpha, num_pass};
    params.normalize = normalize;
    params.verbose = verbose;
    return params;
  }

  static IndexParams vamana(uint32_t R = 200, uint32_t L = 600, double alpha = 1.2,
                            bool two_pass = false, bool compress_input = false,
                            uint32_t verbose = 0, bool pq_enabled = false, uint32_t num_blocks = 8,
                            uint32_t num_clusters_per_block = 256, uint32_t sample_size = 100000) {
    IndexParams params;
    params.method = "vamana";
    params.R = R;
    params.L = L;
    params.alpha = alpha;
    params.two_pass = two_pass;
    params.compress_input = compress_input;
    params.pq = {pq_enabled, num_blocks, num_clusters_per_block, sample_size};
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