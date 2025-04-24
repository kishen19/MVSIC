#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "algorithms/bench/parse_command_line.h"
#include "algorithms/utils/euclidian_point.h"
#include "algorithms/utils/mips_point.h"
#include "algorithms/utils/point_range.h"
#include "algorithms/utils/types.h"
#include "anns.h"
#include "multi-swap.h"
#include "pairwise.h"
#include "parlay/io.h"
#include "parlay/sequence.h"
#include "seeding/kmeansparallel.h"
#include "seeding/kmeansplusplus.h"
#include "seeding/ksetcover.h"
#include "seeding/prefixdoubling.h"
#include "seeding/uniformlyrandom.h"
#include "seeding/wards.h"
#include "utils/evals.h"

// #define PERROUND

template <typename T>
std::tuple<size_t, size_t, parlay::sequence<T>> ReadBigANN(
    const char* filename) {
  uint32_t n;
  uint32_t num_dimensions;
  std::ifstream ifs(filename);
  ifs.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
  ifs.read(reinterpret_cast<char*>(&num_dimensions), sizeof(uint32_t));
  printf("n: %u, num_dimensions: %u\n", n, num_dimensions);
  parlay::sequence<T> points(n * num_dimensions);
  ifs.read(reinterpret_cast<char*>(points.begin()),
           n * num_dimensions * sizeof(T));
  ifs.close();
  return std::make_tuple(n, num_dimensions, points);
}

template <typename T>
parlay::sequence<parlay::sequence<T>> ReadPBBS(const char* filename) {
  auto chars = parlay::chars_from_file(std::string(filename));
  auto tokens_seq = tokens(chars);
  std::ostringstream oss;
  oss << tokens_seq[0];
  auto header = oss.str();

  std::string default_header = "pbbs_sequencePoint";
  assert(header.substr(0, default_header.size()) == default_header);
  size_t dim = stoull(header.substr(default_header.size(),
                                    header.size() - default_header.size() - 1));
  size_t n = (tokens_seq.size() - 1) / dim;

  parlay::sequence<parlay::sequence<T>> points(n, parlay::sequence<T>(dim));
  parlay::parallel_for(0, n, [&](size_t i) {
    for (size_t j = 0; j < dim; ++j) {
      points[i][j] = chars_to_double(tokens_seq[dim * i + j + 1]);
    }
  });
  return points;
}

parlay::sequence<uint32_t> ReadGT(const char* filename) {
  std::ifstream ifs(filename);

  parlay::sequence<uint32_t> values;
  std::string line;
  while (std::getline(ifs, line)) {
    std::istringstream iss(line);
    uint32_t value;
    if (!(iss >> value)) {
      throw std::runtime_error("Error reading value from file");
    }
    values.push_back(value);
  }
  ifs.close();
  return values;
}

template <typename T, typename Range>
T SumOfSquaredCost(const Range& points,
                   const parlay::sequence<uint32_t>& center_ids) {
  auto min_distances = parlay::delayed_seq<T>(points.size(), [&](size_t i) {
    auto new_distances = parlay::delayed_seq<T>(
        center_ids.size(),
        [&](size_t j) { return points[i].distance(points[center_ids[j]]); });
    T smallest_new_distance = reduce(new_distances, parlay::minm<T>());
    return smallest_new_distance;
  });
  return parlay::reduce(min_distances);
}

template <typename T, typename Range>
T SumOfSquaredCost(const Range& points, const Range& centers) {
  auto min_distances = parlay::delayed_seq<T>(points.size(), [&](size_t i) {
    auto new_distances = parlay::delayed_seq<T>(centers.size(), [&](size_t j) {
      return points[i].distance(centers[j]);
    });
    T smallest_new_distance = reduce(new_distances, parlay::minm<T>());
    return smallest_new_distance;
  });
  return parlay::reduce(min_distances);
}

template <typename T, typename Range>
T SumOfSquaredCost(const Range& points, const Range& centers,
                   const parlay::sequence<uint32_t>& clusters) {
  auto distances = parlay::delayed_seq<T>(points.size(), [&](size_t i) {
    return points[i].distance(centers[clusters[i]]);
  });
  return parlay::reduce(distances);
}

template <class DistTy, class PointTy>
void run(commandLine& P) {
  char* const iFile = P.getOptionValue("-i");
  size_t k = P.getOptionIntValue("-k", 10);
#ifdef PERROUND
  int num_rounds = P.getOptionIntValue("-rounds", 1);
#else
  int num_rounds = P.getOptionIntValue("-rounds", 5);
#endif
  std::string filename(iFile);
  size_t idx = filename.find_last_of('.');
  if (idx == std::string::npos) {
    std::cerr << "Error: No file extension provided" << std::endl;
    abort();
  }
  std::string subfix = filename.substr(idx + 1);
  PointRange<DistTy, PointTy> points;
  if (subfix == "fbin") {
    points = PointRange<DistTy, PointTy>(iFile);
  } else if (subfix == "pbbs") {
    auto data = ReadPBBS<DistTy>(iFile);
    int dim = data[0].size();
    points = PointRange<DistTy, PointTy>(data, dim);
  } else {
    std::cerr << "Error: Invalid file extension" << std::endl;
    abort();
  }
  std::string seed_algo = P.getOptionValue("-seed", "PrefixDoubling");
  std::string dist_algo = P.getOptionValue("-dist", "Pairwise");
#ifdef PERROUND
  size_t lloyds_iterations = P.getOptionIntValue("-iters", 50);
#else
  size_t lloyds_iterations = P.getOptionIntValue("-iters", 10);
#endif
  size_t multi_swap_iterations = P.getOptionIntValue("-ms-iters", 2);
  size_t max_swaps = P.getOptionIntValue("-ms-iters", 5);
  double epsw = P.getOptionDoubleValue("-epsw", 0.8);
  double deltaw = P.getOptionDoubleValue("-deltaw", 1.0);
  double samw = P.getOptionDoubleValue("-samw", 20);
  bool wghw = P.getOption("-wghw");

  BuildParams BP;
  if (dist_algo == "ANNS" || seed_algo == "KSetCover" || seed_algo == "Wards") {
    printf("Building params\n");
    long R = P.getOptionLongValue("-R", 16);
    long L = P.getOptionLongValue("-L", 32);
    double alpha = P.getOptionDoubleValue("-a", 1.2);
    bool two_pass = P.getOption("-tp");
    BP = BuildParams(R, L, alpha, two_pass);
  }

  std::cout << "### Application: K-Means" << std::endl;
  std::cout << "### Points: " << iFile << std::endl;
  std::cout << "### Seeding Algorithm: " << seed_algo << std::endl;
  std::cout << "### Distance Oracle: " << dist_algo << std::endl;
  std::cout << "### Threads: " << parlay::num_workers() << std::endl;
  std::cout << "### n: " << points.size() << std::endl;
  std::cout << "### d: " << points.get_dims() << std::endl;
  std::cout << "### k: " << k << std::endl;
  std::cout << "### ------------------------------------" << std::endl;
  std::cout << "### ------------------------------------" << std::endl;

  double total_seeding_time = 0;
  double total_multiswap_time = 0;
  std::vector<double> per_round_lloyds_time(lloyds_iterations);
  std::vector<double> per_round_cost(lloyds_iterations);
  parlay::sequence<uint32_t> center_ids;
  parlay::sequence<uint32_t> cluster_ids;
  PointRange<DistTy, PointTy> centers;
  std::ofstream ofs("kmeans.tsv", std::ios::app);
  ofs << std::fixed << std::setprecision(6);
  ofs << iFile << '\t' << seed_algo << '\t' << dist_algo << '\t';
  ofs << points.size() << '\t' << points.get_dims() << '\t' << k << '\t';
#ifdef PERROUND
  std::ofstream ofs_round("per_round.tsv", std::ios::app);
  ofs_round << iFile << '\t' << seed_algo << '\t' << dist_algo << '\n';
#endif

  double cost1 = 0, cost2 = 0, cost3 = 0;
  for (int i = 0; i <= num_rounds; i++) {
    if (i == 0) {
      printf("\n### Warmup round ---------------------\n");
    } else {
      printf("\n### Round %d ----------------------\n", i);
    }
    parlay::internal::timer st;
    st.start();
    if (seed_algo == "SequentialPlusPlus") {
      center_ids = SequentialPlusPlus<DistTy>(points, k);
      centers = copyPoints<PointTy>(points, center_ids);
    } else if (seed_algo == "PrefixDoubling") {
      center_ids = PrefixDoubling<DistTy>(points, k);
      centers = copyPoints<PointTy>(points, center_ids);
    } else if (seed_algo == "UniformlyRandom") {
      center_ids = UniformlyRandom<DistTy>(points, k);
      centers = copyPoints<PointTy>(points, center_ids);
    } else if (seed_algo == "ParallelPlusPlus") {
      center_ids = ParallelPlusPlus<DistTy>(points, k);
      centers = copyPoints<PointTy>(points, center_ids);
    } else if (seed_algo == "Wards") {
      centers = Wards<DistTy, PointTy>(points, k, BP, epsw, deltaw, samw, wghw);
    } else if (seed_algo == "KSetCover") {
      center_ids = KSetCover<DistTy, PointTy>(points, k, BP);
      centers = copyPoints<PointTy>(points, center_ids);
    } else {
      std::cout << "Error: seeding algorithm not specified correctly"
                << std::endl;
      abort();
    }
    st.stop();

    printf("Seeding time: %f\n", st.total_time());
    if (i > 0) {
      total_seeding_time += st.total_time();
    }

    DistTy cost = SumOfSquaredCost<DistTy>(points, centers);
    std::cout << "sum_squared_distances before multi-swap: " << cost
              << std::endl;
    cost1 = cost;

    // parlay::internal::timer mt;
    // for (size_t j = 0; j < multi_swap_iterations; j++) {
    //   bool swapped = multi_swap_greedy<DistTy>(points, center_ids,
    //   max_swaps); if (!swapped) {
    //     break;
    //   }
    // }
    // if (i > 0) {
    //   total_multiswap_time += mt.total_time();
    // }

    if (dist_algo == "Pairwise") {
      cluster_ids = compute_cluster_ids_pairwise<PointTy>(points, centers);
    } else if (dist_algo == "ANNS") {
      cluster_ids = compute_cluster_ids_anns<PointTy>(points, centers, BP);
    } else {
      std::cout << "Error: distance oracle not specified correctly"
                << std::endl;
      abort();
    }

    cost = SumOfSquaredCost<DistTy>(points, centers);
    std::cout << "sum_squared_distances before Lloyd's (after multi-swap): "
              << cost << std::endl;
    cost2 = cost;

    // centers are not necessarily from the point set after this step

#ifdef PERROUND
    if (i > 0) {
      ofs_round << st.total_time() << '\t' << cost << '\n';
    }
#endif

    parlay::internal::timer lt;
    for (size_t j = 0; j < lloyds_iterations; j++) {
      lt.start();
      if (dist_algo == "Pairwise") {
        std::tie(centers, cluster_ids) =
            lloyds_pairwise<PointTy>(points, centers, cluster_ids);
      } else if (dist_algo == "ANNS") {
        std::tie(centers, cluster_ids) =
            lloyds_anns<PointTy>(points, centers, cluster_ids, BP);
      } else {
        std::cout << "Error: distance oracle not specified correctly"
                  << std::endl;
        abort();
      }
      lt.stop();
      assert(centers.size() == k);
      cost = SumOfSquaredCost<DistTy>(points, centers);
      std::cout << j + 1 << " iterations: "
                << "cost = " << cost << " time = " << lt.total_time()
                << std::endl;

      if (i > 0) {
        per_round_lloyds_time[j] += lt.total_time();
        per_round_cost[j] = cost;
#ifdef PERROUND
        ofs_round << lt.total_time() << '\t' << cost << '\n';
#endif
      }
      cost3 = cost;
      lt.reset();
    }
  }
  printf("\n### Results -------------------------------\n");
  printf("Average Seeding time: %f\n", total_seeding_time / num_rounds);
  for (size_t i = 0; i < lloyds_iterations; i++) {
    printf("Average Lloyd's time of %zu iteration: %f\n", i + 1,
           per_round_lloyds_time[i] / num_rounds);
    printf("Cost after %zu iteration\n", i + 1, per_round_cost[i]);
  }
  double total_lloyd_time =
      accumulate(begin(per_round_lloyds_time), end(per_round_lloyds_time), 0.0);
  printf("Average total Lloyd's time: %f\n\n\n\n",
         total_lloyd_time / num_rounds);
  ofs << total_seeding_time / num_rounds << '\t' << cost1 << '\t';
  // ofs << total_multiswap_time / num_rounds << '\t' << cost2 << '\t';
  ofs << total_lloyd_time / num_rounds << '\t' << cost3 << '\n';

  if (P.getOption("-stats")) {
    char* const gtFile = P.getOptionValue("-gt");
    auto gt = ReadGT(gtFile);
    auto [ari, nmi] = computeMetrics(gt, cluster_ids);
    ofs << ari << '\t' << nmi << '\n';
  }
  ofs.close();
#ifdef PERROUND
  ofs_round.close();
#endif

  std::ofstream cluster_file("/home/kishen/Parallel-KMeans/cluster_ids.txt");
  for (const auto& id : cluster_ids) {
    cluster_file << id << '\n';
  }
  cluster_file.close();
}

int main(int argc, char* argv[]) {
  commandLine P(argc, argv,
                "[-a <alpha>] [-k <num_centers>] [-data_type <tp>] [-dist_func "
                "<df>] [-seed <algorithm>] [-dist <algorithm>] [-i <inFile>]");

  std::string tp = P.getOptionValue("-data_type", "float");
  std::string df = P.getOptionValue("-dist_func", "Euclidian");

  if ((tp != "uint8") && (tp != "int8") && (tp != "float")) {
    std::cout << "Error: vector type not specified correctly, specify int8, "
                 "uint8, or float"
              << std::endl;
    abort();
  }

  if (df != "Euclidian" && df != "mips") {
    std::cout << "Error: specify distance type Euclidian or mips" << std::endl;
    abort();
  }

  if (tp == "float") {
    if (df == "Euclidian") {
      run<float, Euclidian_Point<float>>(P);
    } else if (df == "mips") {
      run<float, Mips_Point<float>>(P);
    }
  } else if (tp == "uint8") {
    if (df == "Euclidian") {
      run<uint8_t, Euclidian_Point<uint8_t>>(P);
    } else if (df == "mips") {
      run<uint8_t, Mips_Point<uint8_t>>(P);
    }
  } else if (tp == "int8") {
    if (df == "Euclidian") {
      run<int8_t, Euclidian_Point<int8_t>>(P);
    } else if (df == "mips") {
      run<int8_t, Mips_Point<int8_t>>(P);
    }
  }
  return 0;
}
