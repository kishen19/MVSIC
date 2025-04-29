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
DistTy sum_of_squared_cost(PointCloud& points, PointCloud& centers,
                   parlay::sequence<uint32_t>& clusters) {
  auto distances = parlay::delayed_seq<DistTy>(points.size(), [&](size_t i) {
    return points[i].distance(centers[clusters[i]]);
  });
  return parlay::reduce(distances);
}

template <typename PointCloud>
parlay::sequence<uint32_t> compute_cluster_ids(PointCloud &points, 
    PointCloud &centers) {
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
auto mvkmeans(PointCloud& points, size_t k, size_t s = 0,
  long iters = 5, string seeding="Random"){
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
  for (long t=0; t<iters; t++){
    // Compute new centers
    auto id_pt = parlay::delayed_seq<std::pair<uint32_t, uint32_t>>(k, [&](size_t i) {
      return std::make_pair(cluster_ids[i], i);
    });
    auto grouped = parlay::group_by_index(id_pt, k);
    auto new_centers = parlay::sequence<PointRange<DistTy, PointTy>>::uninitialized(k);
    parlay::parallel_for(0, k, [&](size_t i) {
      assert(grouped[i].size() > 0);
      auto data = points.GetCluster(grouped[i]);
      new_centers[i] = kmeans(data, s);
    });
    centers = PointCloud(new_centers, d);
    // Reassign points
    cluster_ids = compute_cluster_ids(points, centers);
    cost = sum_of_squared_cost<DistTy>(points, centers, cluster_ids);
    std::cout << "Lloyd's iteration " << t << ": cost = " << cost
              << std::endl;
  }
}

int main(int argc, char* argv[]) {
  commandLine P(argc, argv,
                "[-i <inFile>] [-k <num_centers>] [-data_type <tp>]"
                "[-seed <algorithm>]" "[-iters <num_iters>]"
                "[-dist_func <dist_func>]");

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
  auto iters = P.getOptionLongValue("-iters", 10);

  if (tp == "float") {
    if (df == "Euclidian"){
      auto points = PointCloud<float, PointRange<float, Euclidian_Point<float>>>(P.getOptionValue("-i"));
      mvkmeans<float, Euclidian_Point<float>>(points, k, s, iters, seeding);
    } else if (df == "MIPS") {
      auto points = PointCloud<float, PointRange<float, Mips_Point<float>>>(P.getOptionValue("-i"));
      mvkmeans<float, Mips_Point<float>>(points, k, s, iters, seeding);
    }
  } else if (tp == "uint8") {
    if (df == "Euclidian"){
      auto points = PointCloud<uint8_t, PointRange<uint8_t, Euclidian_Point<uint8_t>>>(P.getOptionValue("-i"));
      mvkmeans<uint8_t, Euclidian_Point<uint8_t>>(points, k, s, iters, seeding);
    } else if (df == "MIPS") {
      auto points = PointCloud<uint8_t, PointRange<uint8_t, Mips_Point<uint8_t>>>(P.getOptionValue("-i"));
      mvkmeans<uint8_t, Mips_Point<uint8_t>>(points, k, s, iters, seeding);
    }
  } else if (tp == "int8") {
    if (df == "Euclidian"){
      auto points = PointCloud<int8_t, PointRange<int8_t, Euclidian_Point<int8_t>>>(P.getOptionValue("-i"));
      mvkmeans<int8_t, Euclidian_Point<int8_t>>(points, k, s, iters, seeding);
    } else if (df == "MIPS") {
      auto points = PointCloud<int8_t, PointRange<int8_t, Mips_Point<int8_t>>>(P.getOptionValue("-i"));
      mvkmeans<int8_t, Mips_Point<int8_t>>(points, k, s, iters, seeding);
    }
  }
  return 0;
}
