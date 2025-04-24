#include "parlay/sequence.h"
#include "/home/kishen/MVC/external/kmeans/lloyds/kmeans.cpp"

#include "seeding/uniformlyrandom.h"
#include "utils/pointcloud.h"

// #include "algorithms/bench/parse_command_line.h"
// #include "algorithms/utils/euclidian_point.h"
// #include "algorithms/utils/mips_point.h"
// #include "algorithms/utils/point_range.h"
// #include "algorithms/utils/types.h"
// #include "parlay/io.h"

template <typename T, typename PointCloud>
T sum_of_squared_cost(const PointCloud& points, const PointCloud& centers,
                   const parlay::sequence<uint32_t>& clusters) {
  auto distances = parlay::delayed_seq<T>(points.size(), [&](size_t i) {
    return points[i].distance(centers[clusters[i]]);
  });
  return parlay::reduce(distances);
}

template <typename T, typename PointCloud>
parlay::sequence<uint32_t> compute_cluster_ids(const PointCloud &points, const PointCloud &centers) {
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

template <typename T, typename PointCloud>
auto mvkmeans(const PointCloud& points, size_t k, size_t s=0,
  string seeding="Random"){
  uint32_t n = points.size();
  uint32_t d = points.dims();
  
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
    centers = UniformlyRandom<T>(points, k);
  } else {
    std::cout << "Error: seeding algorithm not specified correctly"
              << std::endl;
    abort();
  }
  
  auto cluster_ids = compute_cluster_ids(points, centers);
  T seed_cost = sum_of_squared_cost(points, centers, cluster_ids);
  std::cout << "Seeding cost: " << seed_cost << std::endl;

  // Lloyd's Step
  T cost;
  for (int t=0; t<iters; t++){
    // Compute new centers
    //    First group individual vectors by cluster
    auto id_pt = parlay::delayed_seq<std::pair<uint32_t, uint32_t>>(k, [&](size_t i) {
      return std::make_pair(cluster_ids[i], i);
    });
    auto grouped = parlay::group_by_index(id_pt, k);
    auto new_centers = parlay::sequence<parlay::sequence<T>>(k);
    parlay::parallel_for(0, k, [&](size_t i) {
      auto data = points.GetCluster(grouped[i]);
      new_centers[i] = kmeans(PointRange<T>(data, d), s);
    });
    centers = PointCloud(new_centers);
    // Reassign points
    cluster_ids = compute_cluster_ids(points, centers);
    cost = SumOfSquaredCost(points, centers, cluster_ids);
    std::cout << "Lloyd's iteration " << t << ": cost = " << cost
              << std::endl;
  }
}

int main(int argc, char* argv[]) {
  commandLine P(argc, argv,
                "[-i <inFile>] [-k <num_centers>] [-data_type <tp>]"
                "[-seed <algorithm>]");

  std::string tp = P.getOptionValue("-data_type", "float");

  if ((tp != "uint8") && (tp != "int8") && (tp != "float")) {
    std::cout << "Error: vector type not specified correctly, specify int8, "
                 "uint8, or float"
              << std::endl;
    abort();
  }

  if (tp == "float") {
    auto points = PointCloud<float>(P.getOptionValue("-i"));
    auto k = P.getOptionLongValue("-k", 10);
    auto s = P.getOptionLongValue("-s", 0);
    auto seeding = P.getOptionValue("-seed", "Random");
    mvkmeans<float, PointCloud<float>>(points, k, s, seeding);
  } else if (tp == "uint8") {
    auto points = PointCloud<uint8_t>(P.getOptionValue("-i"));
    auto k = P.getOptionLongValue("-k", 10);
    auto s = P.getOptionLongValue("-s", 0);
    auto seeding = P.getOptionValue("-seed", "Random");
    mvkmeans<uint8_t, PointCloud<uint8_t>>(points, k, s, seeding);
  } else if (tp == "int8") {
    auto points = PointCloud<int8_t>(P.getOptionValue("-i"));
    auto k = P.getOptionLongValue("-k", 10);
    auto s = P.getOptionLongValue("-s", 0);
    auto seeding = P.getOptionValue("-seed", "Random");
    mvkmeans<int8_t, PointCloud<int8_t>>(points, k, s, seeding);
  }
  return 0;
}
