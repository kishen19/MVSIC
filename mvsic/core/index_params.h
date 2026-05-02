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

  // mvivf (and other IVF-style) structural params.
  uint32_t k_per_level = 0;      // Number of clusters at each node. [0 = 4*sqrt(n)]
  uint32_t max_leaf_size = 500;  // Maximum size of leaves (enforced)
  uint32_t max_depth = 0;  // Maximum tree depth. 0 = exhaustive (stop when leaves <= max_leaf_size)

  // Legacy: used only by svh_ivf's runtime quantize_centers check; MVIVF selects
  // center compression at compile time via the CompressCenters template param.
  // TODO: get rid of once obsolete
  bool quantize_centers = true;

  // MVIVFSpill: (a, b) spill strategy.
  //   `num_spill`    = a  -> top-a nearest centers at the root (first level).
  //   `num_spill_l2` = b  -> top-b nearest centers at the second level.
  //   All deeper levels use top-1 (i.e. regular clustering, no spill).
  //
  // Setting b=1 recovers the original root-only spilling.  Setting a=b=1 turns
  // the index into a plain MVIVF (no spilling anywhere).
  uint32_t num_spill = 2;
  uint32_t num_spill_l2 = 1;

  // MVClustering params
  MVClusteringConfig mvclus = MVClusteringConfig();
  uint32_t s = 0;  // Size of each centroid point cloud (0 = avg input point cloud size)

  // When true, MVIVF*'s `recursive_build_` runs k-means assignment through
  // the 8-bit TurboQuant VPDPBUSD panel kernel (see MVClustering8BTQ).
  // Centroid update still runs on float points, so the resulting index has
  // float centers + (optionally float-trained) leaf models. Default off.
  bool build_with_8btq = false;

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

  // Quantizer enum. MVIVF family indices pick their leaf quantizer at compile time
  // via the LeafModel template parameter; this enum is only tagged onto `pq.method`
  // by the single-vector graph families (MUVERA, MPool, SVH_Graph, SVH_IVF) so the
  // legacy variant-based helpers on `Index<metric>` keep dispatching correctly.
  enum class QuantizerType {
    None,
    PQ,
    RaBitQ,
    FastScan,
    TurboQuant,
    SPQTQ,
    OneBitTQ,
    EightBitTQ,
    OneBitTQAsym
  };

  // Legacy quantizer hyperparameters. Read by `Index<metric>::train_quantizer`
  // (variant-based dispatch) for the graph-family single-vector indices only.
  // MVIVF routes these through the quantizer-typed `Model::Params` ctor arg instead.
  struct pq_config {
    QuantizerType method = QuantizerType::None;
    uint32_t block_size = 8;
    uint32_t num_clusters_per_block = 16;
    uint32_t num_points_per_cluster = 100;
    uint32_t rabitq_bits = 4;
  };
  pq_config pq = pq_config();

  uint32_t max_points_per_centroid = 100;

  // MVIVF
  static IndexParams mvivf(uint32_t k_per_level = 0, uint32_t max_leaf_size = 500,
                           bool compress_input = false, uint32_t verbose = 0, uint32_t niters = 5,
                           uint32_t max_point_clouds_per_cluster = 100,
                           uint32_t max_points_per_centroid_inner_kmeans = 20,
                           std::string init = "Random", uint32_t seed = 0,
                           bool use_weighted_inner_kmeans = true, uint32_t s = 0,
                           uint32_t max_depth = 0, bool build_with_8btq = false) {
    IndexParams params;
    params.method = "mvivf";
    params.k_per_level = k_per_level;
    params.max_leaf_size = max_leaf_size;
    params.max_depth = max_depth;
    params.compress_input = compress_input;
    params.verbose = verbose;
    params.mvclus = MVClusteringConfig(
        niters, max_point_clouds_per_cluster, max_points_per_centroid_inner_kmeans,
        (params.verbose > 0) ? params.verbose - 1 : 0, init, seed, use_weighted_inner_kmeans);
    params.s = s;
    params.build_with_8btq = build_with_8btq;
    return params;
  }

  static IndexParams mvivf_flat(uint32_t k_per_level = 0, bool compress_input = false,
                                uint32_t verbose = 0, uint32_t niters = 5,
                                uint32_t max_point_clouds_per_cluster = 100,
                                uint32_t max_points_per_centroid_inner_kmeans = 20,
                                std::string init = "Random", uint32_t seed = 0,
                                bool use_weighted_inner_kmeans = true, uint32_t s = 0,
                                bool build_with_8btq = false) {
    IndexParams params;
    params.method = "mvivf_flat";
    params.k_per_level = k_per_level;
    params.compress_input = compress_input;
    params.verbose = verbose;
    params.mvclus = MVClusteringConfig(
        niters, max_point_clouds_per_cluster, max_points_per_centroid_inner_kmeans,
        (params.verbose > 0) ? params.verbose - 1 : 0, init, seed, use_weighted_inner_kmeans);
    params.s = s;
    params.build_with_8btq = build_with_8btq;
    return params;
  }

  static IndexParams mvivf_spill(uint32_t k_per_level = 0, uint32_t max_leaf_size = 500,
                                 uint32_t num_spill = 2, bool compress_input = false,
                                 uint32_t verbose = 0, uint32_t niters = 5,
                                 uint32_t max_point_clouds_per_cluster = 100,
                                 uint32_t max_points_per_centroid_inner_kmeans = 20,
                                 std::string init = "Random", uint32_t seed = 0,
                                 bool use_weighted_inner_kmeans = true, uint32_t s = 0,
                                 uint32_t max_depth = 0, uint32_t num_spill_l2 = 1,
                                 bool build_with_8btq = false) {
    IndexParams params;
    params.method = "mvivf_spill";
    params.k_per_level = k_per_level;
    params.max_leaf_size = max_leaf_size;
    params.max_depth = max_depth;
    params.num_spill = num_spill;
    params.num_spill_l2 = num_spill_l2;
    params.compress_input = compress_input;
    params.verbose = verbose;
    params.mvclus = MVClusteringConfig(
        niters, max_point_clouds_per_cluster, max_points_per_centroid_inner_kmeans,
        (params.verbose > 0) ? params.verbose - 1 : 0, init, seed, use_weighted_inner_kmeans);
    params.s = s;
    params.build_with_8btq = build_with_8btq;
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
                             bool quantize_centers = true) {
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

  static IndexParams flat(bool compress_input = false, uint32_t verbose = 0, uint32_t pq_method = 0,
                          uint32_t block_size = 8, uint32_t num_clusters_per_block = 16,
                          uint32_t num_points_per_cluster = 100, uint32_t rabitq_bits = 4) {
    IndexParams params;
    params.method = "flat";
    params.compress_input = compress_input;
    params.verbose = verbose;
    params.pq = {static_cast<QuantizerType>(pq_method), block_size, num_clusters_per_block,
                 num_points_per_cluster, rabitq_bits};
    return params;
  }

  //  private:
  IndexParams() = default;
};

}  // namespace mvsic