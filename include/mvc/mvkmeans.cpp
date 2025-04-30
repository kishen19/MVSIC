#include "parlay/sequence.h"
#include "lloyds/kmeans.h"
#include "algorithms/bench/parse_command_line.h"
#include "algorithms/utils/euclidian_point.h"
#include "algorithms/utils/mips_point.h"
#include "algorithms/utils/point_range.h"
#include "algorithms/utils/types.h"

#include "seeding/uniformlyrandom.h"
#include "utils/pointcloud.h"

// #include "algorithms/bench/parse_command_line.h"
// #include "algorithms/utils/euclidian_point.h"
// #include "algorithms/utils/mips_point.h"
// #include "algorithms/utils/point_range.h"
// #include "algorithms/utils/types.h"
// #include "parlay/io.h"

template <typename DistTy, typename PointCloud>
DistTy sum_of_squared_cost(const PointCloud& points, const PointCloud& centers,
                   const parlay::sequence<uint32_t>& clusters) {
  auto distances = parlay::delayed_seq<DistTy>(points.size(), [&](size_t i) {
    return points[i].distance(centers[clusters[i]]);
  });
  return parlay::reduce(distances);
}

template <typename PointCloud>
parlay::sequence<uint32_t> compute_cluster_ids(const PointCloud &points, 
    const PointCloud &centers) {
  size_t n = points.size();
  size_t k = centers.size();
  parlay::sequence<uint32_t> updated_cluster_ids(n);
  parlay::parallel_for(0, n, [&](size_t i) {
    auto dist = parlay::tabulate(
        k, [&](size_t j) { return points[i].distance(centers[j]); });
    updated_cluster_ids[i] = parlay::min_element(dist) - begin(dist);
  });
  return updated_cluster_ids;
}

template <typename DistTy, typename PointTy, typename PointCloud>
auto mvkmeans(const PointCloud& points, size_t k, size_t s = 0,
    long iters = 5, std::string seeding="Random",
    std::string kmeans_dist_algo = "ANNS", std::string kmeans_seeding = "PrefixDoubling",
    long kmeans_iters = 20) {
  uint32_t n = points.size();
  uint32_t d = points.dimension();
  
  if (s == 0){
    auto num_emb = parlay::delayed_seq<size_t>(points.size(), [&](size_t i) {
      return points.num_embeddings(i);
    });
    s = parlay::reduce(num_emb)/n;
  }

  // Initialization 
  PointCloud centers;
  parlay::sequence<uint32_t> cluster_ids;

  if (seeding == "Random"){
    centers = UniformlyRandomMV<DistTy>(points, k);
  } else {
    std::cout << "Error: seeding algorithm not specified correctly"
              << std::endl;
    abort();
  }
  
  cluster_ids = compute_cluster_ids(points, centers);
  DistTy seed_cost = sum_of_squared_cost<DistTy>(points, centers, cluster_ids);
  std::cout << "Seeding cost: " << seed_cost << std::endl;

  // Lloyd's Step
  DistTy cost;
  for (long it=0; it<iters; it++){
    // Compute new centers
    auto id_pt = parlay::delayed_seq<std::pair<uint32_t, uint32_t>>(n, [&](size_t i) {
      return std::make_pair(cluster_ids[i], i);
    });
    auto grouped = parlay::group_by_index(id_pt, k);
    auto new_centers = parlay::sequence<PointRange<DistTy, PointTy>>(k);
    parlay::parallel_for(0, k, [&](size_t i) {
      if(grouped[i].size() > 0){
        auto data = points.GetCluster(grouped[i]);
        new_centers[i] = kmeans(data, s, kmeans_seeding, kmeans_dist_algo, kmeans_iters);
      } else {
        static uint32_t seed = 42;
        uint32_t id = parlay::hash32(seed++) % n;
        new_centers[i] = PointRange<DistTy, PointTy>(points[id], d);
      }
    });
    centers = PointCloud(new_centers, d);
    // Reassign points
    cluster_ids = compute_cluster_ids(points, centers);
    cost = sum_of_squared_cost<DistTy>(points, centers, cluster_ids);
    std::cout << "Lloyd's iteration " << it << ": cost = " << cost
              << std::endl;
  }
}

int main(int argc, char* argv[]) {
  commandLine P(argc, argv,
                "[-i <inFile>] [-k <num_centers>] [-s <num_embeddings>]"
                "[-data_type <tp>] [-dist_func <dist_func>]" 
                "[-seed <algorithm>] [-iters <num_iters>]" 
                "[-kmeans_seed <algorithm>] [-kmeans_dist <algorithm>]");

  std::string tp = P.getOptionValue("-data_type", "float");
  std::string df = P.getOptionValue("-dist_func", "Euclidian");

  if ((tp != "uint8") && (tp != "int8") && (tp != "float")) {
    std::cout << "Error: vector type not specified correctly, specify int8, "
                 "uint8, or float"
              << std::endl;
    abort();
  }

  auto k = P.getOptionLongValue("-k", 10);
  auto s = P.getOptionLongValue("-s", 0);
  auto seeding = P.getOptionValue("-seed", "Random");
  auto iters = P.getOptionLongValue("-iters", 5);
  auto kmeans_seeding = P.getOptionValue("-kmeans_seed", "PrefixDoubling");
  auto kmeans_dist_algo = P.getOptionValue("-kmeans_dist", "ANNS");
  auto kmeans_iters = P.getOptionLongValue("-kmeans_iters", 20);


  if (tp == "float") {
    if (df == "Euclidian"){
      auto points = PointCloud<float, PointRange<float, Euclidian_Point<float>>>(P.getOptionValue("-i"));
      mvkmeans<float, Euclidian_Point<float>>(points, k, s, iters, seeding, 
          kmeans_dist_algo, kmeans_seeding, kmeans_iters);
    } else if (df == "MIPS") {
      auto points = PointCloud<float, PointRange<float, Mips_Point<float>>>(P.getOptionValue("-i"));
      mvkmeans<float, Mips_Point<float>>(points, k, s, iters, seeding, 
          kmeans_dist_algo, kmeans_seeding, kmeans_iters);
    }
  } else if (tp == "uint8") {
    if (df == "Euclidian"){
      auto points = PointCloud<uint8_t, PointRange<uint8_t, Euclidian_Point<uint8_t>>>(P.getOptionValue("-i"));
      mvkmeans<uint8_t, Euclidian_Point<uint8_t>>(points, k, s, iters, seeding, 
          kmeans_dist_algo, kmeans_seeding, kmeans_iters);
    } else if (df == "MIPS") {
      auto points = PointCloud<uint8_t, PointRange<uint8_t, Mips_Point<uint8_t>>>(P.getOptionValue("-i"));
      mvkmeans<uint8_t, Mips_Point<uint8_t>>(points, k, s, iters, seeding, 
          kmeans_dist_algo, kmeans_seeding, kmeans_iters);
    }
  } else if (tp == "int8") {
    if (df == "Euclidian"){
      auto points = PointCloud<int8_t, PointRange<int8_t, Euclidian_Point<int8_t>>>(P.getOptionValue("-i"));
      mvkmeans<int8_t, Euclidian_Point<int8_t>>(points, k, s, iters, seeding, 
          kmeans_dist_algo, kmeans_seeding, kmeans_iters);
    } else if (df == "MIPS") {
      auto points = PointCloud<int8_t, PointRange<int8_t, Mips_Point<int8_t>>>(P.getOptionValue("-i"));
      mvkmeans<int8_t, Mips_Point<int8_t>>(points, k, s, iters, seeding, 
          kmeans_dist_algo, kmeans_seeding, kmeans_iters);
    }
  }
  return 0;
}
