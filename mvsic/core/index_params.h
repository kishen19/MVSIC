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
  uint32_t k_per_level = 0;       // Number of clusters at each node. [0 = 4*sqrt(n)]
  uint32_t max_leaf_size = 500;   // Maximum size of leaves (enforced)
  bool quantize_centers = false;  // Quantize center point clouds

  // MVIVFSpill: assign each point to its top num_spill nearest root centers.
  uint32_t num_spill = 2;

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
  enum class QuantizerType { None, PQ, RaBitQ, FastScan, TurboQuant };

  struct pq_config {
    QuantizerType method = QuantizerType::None;  // PQ method
    uint32_t block_size = 8;                     // Number of blocks
    uint32_t num_clusters_per_block = 16;        // Number of clusters per block
    uint32_t num_points_per_cluster = 100;       // Subsample size*k
    uint32_t rabitq_bits = 4;                    // Total bits for RaBitQ
  };
  pq_config pq = pq_config();

  uint32_t max_points_per_centroid = 100;

  static IndexParams mvivf(uint32_t k_per_level = 0, uint32_t max_leaf_size = 500,
                           bool compress_input = false, uint32_t verbose = 0, uint32_t niters = 5,
                           uint32_t max_point_clouds_per_cluster = 100,
                           uint32_t max_points_per_centroid_inner_kmeans = 20,
                           std::string init = "Random", uint32_t seed = 0,
                           bool use_weighted_inner_kmeans = true, uint32_t s = 0,
                           uint32_t pq_method = 0, uint32_t block_size = 8,
                           uint32_t num_clusters_per_block = 16,
                           uint32_t num_points_per_cluster = 100, uint32_t rabitq_bits = 4,
                           bool quantize_centers = false) {
    IndexParams params;
    params.method = "mvivf";
    params.k_per_level = k_per_level;
    params.max_leaf_size = max_leaf_size;
    params.quantize_centers = quantize_centers;
    params.compress_input = compress_input;
    params.pq = {static_cast<QuantizerType>(pq_method), block_size, num_clusters_per_block,
                 num_points_per_cluster, rabitq_bits};
    params.verbose = verbose;
    params.mvclus = MVClusteringConfig(
        niters, max_point_clouds_per_cluster, max_points_per_centroid_inner_kmeans,
        (params.verbose > 0) ? params.verbose - 1 : 0, init, seed, use_weighted_inner_kmeans);
    params.s = s;
    return params;
  }

  static IndexParams mvivf_flat(uint32_t k_per_level = 0, bool compress_input = false,
                                uint32_t verbose = 0, uint32_t niters = 5,
                                uint32_t max_point_clouds_per_cluster = 100,
                                uint32_t max_points_per_centroid_inner_kmeans = 20,
                                std::string init = "Random", uint32_t seed = 0,
                                bool use_weighted_inner_kmeans = true, uint32_t s = 0,
                                uint32_t pq_method = 0, uint32_t block_size = 8,
                                uint32_t num_clusters_per_block = 16,
                                uint32_t num_points_per_cluster = 100, uint32_t rabitq_bits = 4,
                                bool quantize_centers = false) {
    IndexParams params;
    params.method = "mvivf_flat";
    params.k_per_level = k_per_level;
    params.quantize_centers = quantize_centers;
    params.compress_input = compress_input;
    params.pq = {static_cast<QuantizerType>(pq_method), block_size, num_clusters_per_block,
                 num_points_per_cluster, rabitq_bits};
    params.verbose = verbose;
    params.mvclus = MVClusteringConfig(
        niters, max_point_clouds_per_cluster, max_points_per_centroid_inner_kmeans,
        (params.verbose > 0) ? params.verbose - 1 : 0, init, seed, use_weighted_inner_kmeans);
    params.s = s;
    return params;
  }

  static IndexParams mvivf_spill(uint32_t k_per_level = 0, uint32_t max_leaf_size = 500,
                                 uint32_t num_spill = 2, bool compress_input = false,
                                 uint32_t verbose = 0, uint32_t niters = 5,
                                 uint32_t max_point_clouds_per_cluster = 100,
                                 uint32_t max_points_per_centroid_inner_kmeans = 20,
                                 std::string init = "Random", uint32_t seed = 0,
                                 bool use_weighted_inner_kmeans = true, uint32_t s = 0,
                                 uint32_t pq_method = 0, uint32_t block_size = 8,
                                 uint32_t num_clusters_per_block = 16,
                                 uint32_t num_points_per_cluster = 100, uint32_t rabitq_bits = 4,
                                 bool quantize_centers = false) {
    IndexParams params;
    params.method = "mvivf_spill";
    params.k_per_level = k_per_level;
    params.max_leaf_size = max_leaf_size;
    params.num_spill = num_spill;
    params.quantize_centers = quantize_centers;
    params.compress_input = compress_input;
    params.pq = {static_cast<QuantizerType>(pq_method), block_size, num_clusters_per_block,
                 num_points_per_cluster, rabitq_bits};
    params.verbose = verbose;
    params.mvclus = MVClusteringConfig(
        niters, max_point_clouds_per_cluster, max_points_per_centroid_inner_kmeans,
        (params.verbose > 0) ? params.verbose - 1 : 0, init, seed, use_weighted_inner_kmeans);
    params.s = s;
    return params;
  }

  static IndexParams muvera_custom(
      int32_t num_repetitions = 20, int32_t num_simhash_projections = 4, int32_t seed = 1,
      int32_t projection_dimension = 8, bool fill_empty_partitions = false,
      int32_t final_projection_dimension = 0, bool normalize = true, uint32_t R = 200,
      uint32_t L = 600, double alpha = 1.1, int num_pass = 1, bool compress_input = false,
      uint32_t verbose = 0, uint32_t pq_method = 0, uint32_t block_size = 8,
      uint32_t num_clusters_per_block = 16, uint32_t num_points_per_cluster = 100,
      uint32_t rabitq_bits = 4) {
    IndexParams params;
    params.method = "muvera";
    params.compress_input = compress_input;
    params.pq = {static_cast<QuantizerType>(pq_method), block_size, num_clusters_per_block,
                 num_points_per_cluster, rabitq_bits};
    params.fde = fde_config{
        num_repetitions,       num_simhash_projections,    seed,     projection_dimension,
        fill_empty_partitions, final_projection_dimension, normalize};
    params.ann = ann_config{R, L, alpha, num_pass};
    params.verbose = verbose;
    return params;
  }

  static IndexParams muvera(int32_t d_fde = 2560, int32_t seed = 1,
                            bool fill_empty_partitions = false,
                            int32_t final_projection_dimension = 0, bool normalize = true,
                            uint32_t R = 200, uint32_t L = 600, double alpha = 1.1,
                            int num_pass = 1, bool compress_input = false, uint32_t verbose = 0,
                            uint32_t pq_method = 0, uint32_t block_size = 8,
                            uint32_t num_clusters_per_block = 16,
                            uint32_t num_points_per_cluster = 100, uint32_t rabitq_bits = 4) {
    IndexParams params;
    params.method = "muvera";
    params.compress_input = compress_input;
    params.pq = {static_cast<QuantizerType>(pq_method), block_size, num_clusters_per_block,
                 num_points_per_cluster, rabitq_bits};
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
                           uint32_t pq_method = 0, uint32_t block_size = 8,
                           uint32_t num_clusters_per_block = 16,
                           uint32_t num_points_per_cluster = 100, uint32_t rabitq_bits = 4) {
    IndexParams params;
    params.method = "mpool";
    params.compress_input = compress_input;
    params.pq = {static_cast<QuantizerType>(pq_method), block_size, num_clusters_per_block,
                 num_points_per_cluster, rabitq_bits};
    params.ann = ann_config{R, L, alpha, num_pass};
    params.normalize = normalize;
    params.verbose = verbose;
    return params;
  }

  static IndexParams vamana(uint32_t R = 200, uint32_t L = 600, double alpha = 1.2,
                            bool two_pass = false, bool compress_input = false,
                            uint32_t verbose = 0, uint32_t pq_method = 0, uint32_t block_size = 8,
                            uint32_t num_clusters_per_block = 16,
                            uint32_t num_points_per_cluster = 100, uint32_t rabitq_bits = 4) {
    IndexParams params;
    params.method = "vamana";
    params.R = R;
    params.L = L;
    params.alpha = alpha;
    params.two_pass = two_pass;
    params.compress_input = compress_input;
    params.pq = {static_cast<QuantizerType>(pq_method), block_size, num_clusters_per_block,
                 num_points_per_cluster, rabitq_bits};
    params.verbose = verbose;
    return params;
  }

  static IndexParams svh_ivf(uint32_t k_per_level = 0, uint32_t max_leaf_size = 500,
                             bool compress_input = false, uint32_t verbose = 0,
                             uint32_t max_points_per_centroid = 100, uint32_t pq_method = 0,
                             uint32_t block_size = 8, uint32_t num_clusters_per_block = 16,
                             uint32_t num_points_per_cluster = 100, uint32_t rabitq_bits = 4,
                             bool quantize_centers = false) {
    IndexParams params;
    params.method = "svh_ivf";
    params.k_per_level = k_per_level;
    params.max_leaf_size = max_leaf_size;
    params.max_points_per_centroid = max_points_per_centroid;
    params.quantize_centers = quantize_centers;
    params.compress_input = compress_input;
    params.pq = {static_cast<QuantizerType>(pq_method), block_size, num_clusters_per_block,
                 num_points_per_cluster, rabitq_bits};
    params.verbose = verbose;
    return params;
  }

  static IndexParams svh_graph(uint32_t R = 200, uint32_t L = 600, double alpha = 1.2,
                               int num_pass = 1, bool compress_input = false, uint32_t verbose = 0,
                               uint32_t pq_method = 0, uint32_t block_size = 8,
                               uint32_t num_clusters_per_block = 16,
                               uint32_t num_points_per_cluster = 100, uint32_t rabitq_bits = 4) {
    IndexParams params;
    params.method = "svh_graph";
    params.ann = ann_config{R, L, alpha, num_pass};
    params.compress_input = compress_input;
    params.pq = {static_cast<QuantizerType>(pq_method), block_size, num_clusters_per_block,
                 num_points_per_cluster, rabitq_bits};
    params.verbose = verbose;
    return params;
  }

  //  private:
  IndexParams() = default;
};

}  // namespace mvsic