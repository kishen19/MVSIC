// FastScan ManyToMany microbenchmark.
//
// - Sweep -q_blocks: passed as TopKIntoUninitialized(..., q_block, parallel_query_blocks).
// - Default matches MVIVF leaf probing: sequential query blocking (parallel_query_blocks=false)
//   so outer parallelism can come from -num_leaves independent (500-doc) corpora.
// - Use -par_inner to use parlay::blocked_for over query chunks inside one TopK call (old style).
//
// -num_leaves N: train one model, build N independent encoded corpora and N query sets. Each timed
//   iteration uses parlay::parallel_for(0, N, ...) so all N leaves are processed in parallel (same
//   idea as MVIVF parallel_for over grouped leaves). Inner TopK uses sequential query blocks unless
//   -par_inner (avoids nested parallelism over query chunks inside each leaf).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "parlay/parallel.h"

#include "mvsic/core/quantization/fastscan_mv.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/core/utils/parse_command_line.h"

using ChPoint = mvsic::ChamferIP_Point;
using PCS = mvsic::PointCloudSet<ChPoint>;
namespace fsmv = mvsic::fastscan_mv;
using QuantSet = fsmv::Quantized_Point_Cloud_Set<false>;
using QuantQuery = fsmv::Quantized_Query_Point_Cloud<false>;
using M2M = fsmv::ManyToMany<QuantSet>;

struct Timer {
  using clock = std::chrono::steady_clock;
  clock::time_point t0;
  void start() { t0 = clock::now(); }
  double sec() const { return std::chrono::duration<double>(clock::now() - t0).count(); }
};

static double median(std::vector<double>& v) {
  if (v.empty()) return 0.0;
  const size_t mid = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + mid, v.end());
  return v[mid];
}

template<typename F>
static double bench(F&& fn, int reps) {
  std::vector<double> times;
  times.reserve(reps);
  for (int r = 0; r < reps; ++r) {
    Timer t;
    t.start();
    fn();
    times.push_back(t.sec());
  }
  return median(times);
}

static void fill_random(float* dst, size_t n, uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> unif(-1.0f, 1.0f);
  for (size_t i = 0; i < n; ++i) dst[i] = unif(gen);
}

static PCS make_random_pcs_fixed(size_t n, uint32_t d, size_t cloud_size, uint32_t seed) {
  std::vector<float> values(n * cloud_size * d);
  fill_random(values.data(), values.size(), seed);

  std::vector<size_t> offsets(n + 1);
  for (size_t i = 0; i <= n; ++i) offsets[i] = i * cloud_size * d;

  std::vector<uint32_t> ids(n);
  for (size_t i = 0; i < n; ++i) ids[i] = static_cast<uint32_t>(i);

  return PCS(static_cast<uint32_t>(n), d, values.data(), offsets.data(), ids.data());
}

static std::vector<size_t> parse_sizes(const std::string& s) {
  std::vector<size_t> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (tok.empty()) continue;
    out.push_back(static_cast<size_t>(std::stoull(tok)));
  }
  return out;
}

int main(int argc, char** argv) {
  mvsic::commandLine P(argc, argv,
                       "[-num_queries <n>] [-num_points <n>] [-num_leaves <n>] [-dims <d>] "
                       "[-q_size <n>] [-p_size <n>] [-k <k>] [-reps <r>] [-pq_block_size <n>] "
                       "[-q_blocks <comma_sizes>] [-par_inner]");

  const size_t num_queries = static_cast<size_t>(P.getOptionLongValue("-num_queries", 2048));
  const size_t num_points = static_cast<size_t>(P.getOptionLongValue("-num_points", 500));
  const size_t num_leaves = static_cast<size_t>(P.getOptionLongValue("-num_leaves", 1));
  const uint32_t dims = static_cast<uint32_t>(P.getOptionIntValue("-dims", 128));
  const size_t q_size = static_cast<size_t>(P.getOptionLongValue("-q_size", 32));
  const size_t p_size = static_cast<size_t>(P.getOptionLongValue("-p_size", 128));
  const uint32_t k = static_cast<uint32_t>(P.getOptionIntValue("-k", 10));
  const int reps = P.getOptionIntValue("-reps", 5);
  const uint32_t pq_block_size = static_cast<uint32_t>(P.getOptionIntValue("-pq_block_size", 8));
  const std::string q_blocks_str = P.getOptionValue("-q_blocks", "4,8,16,32,64,128,256,512,1024");
  const bool parallel_query_blocks = P.getOption("-par_inner");

  auto q_blocks = parse_sizes(q_blocks_str);
  if (q_blocks.empty()) {
    std::cerr << "Empty -q_blocks list.\n";
    return 1;
  }
  if (num_leaves == 0) {
    std::cerr << "-num_leaves must be >= 1.\n";
    return 1;
  }

  const size_t total_query_clouds = num_leaves * num_queries;

  std::cout << "FastScan M2M microbenchmark (IP)\n"
            << "  leaves: parlay::parallel_for(0, " << num_leaves << ") — all leaves run in parallel\n"
            << "  parallel_query_blocks (inside each TopK)=" << (parallel_query_blocks ? "true (-par_inner)\n" : "false (default)\n")
            << "  per leaf: num_queries=" << num_queries << ", num_points=" << num_points
            << " => " << total_query_clouds << " query clouds total\n"
            << "  dims=" << dims << ", q_size=" << q_size << ", p_size=" << p_size << ", k=" << k
            << ", reps=" << reps << ", pq_block_size=" << pq_block_size << "\n";

  // Train once on a fixed corpus (same codebooks for all leaves).
  PCS train_corpus = make_random_pcs_fixed(num_points, dims, p_size, 42);
  fsmv::Model<false> model;
  model.train(train_corpus, pq_block_size);

  std::vector<PCS> leaf_points(num_leaves);
  std::vector<QuantSet> leaf_encoded(num_leaves);
  std::vector<PCS> leaf_queries(num_leaves);
  std::vector<std::vector<QuantQuery>> leaf_q_quant(num_leaves);
  std::vector<std::vector<const QuantQuery*>> all_queries(num_leaves);
  std::vector<std::vector<std::pair<uint32_t, float>>> results(num_leaves);

  for (size_t L = 0; L < num_leaves; ++L) {
    leaf_points[L] = make_random_pcs_fixed(num_points, dims, p_size, static_cast<uint32_t>(1000 + L));
    leaf_encoded[L] = model.encode(leaf_points[L]);
    leaf_queries[L] =
        make_random_pcs_fixed(num_queries, dims, q_size, static_cast<uint32_t>(20000 + L));
    leaf_q_quant[L].resize(num_queries);
    parlay::parallel_for(0, num_queries, [&](size_t i) {
      leaf_q_quant[L][i] = model.quantize_query(leaf_queries[L][i]);
    });
    all_queries[L].resize(num_queries);
    for (size_t i = 0; i < num_queries; ++i) all_queries[L][i] = &leaf_q_quant[L][i];
    results[L].resize(num_queries * k);
  }

  std::vector<std::pair<size_t, double>> timings;
  timings.reserve(q_blocks.size());

  uint64_t checksum = 0;
  for (size_t qb : q_blocks) {
    const size_t q_block = std::max<size_t>(1, std::min(qb, num_queries));

    // Outer parallelism: one TopK per leaf, all leaves at once (MVIVF-style).
    auto run_once = [&]() {
      parlay::parallel_for(0, num_leaves, [&](size_t L) {
        M2M::TopKIntoUninitialized(all_queries[L], leaf_encoded[L], k, results[L].data(), q_block,
                                   parallel_query_blocks);
      });
    };

    const double t = bench(run_once, reps);
    timings.push_back({q_block, t});
    checksum += results[num_leaves - 1][(num_queries - 1) * k].first;
  }

  double best = std::numeric_limits<double>::max();
  for (const auto& [_, t] : timings) best = std::min(best, t);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "\nQueryBlock  MedianTime(s)  QPS(all clouds)  vs_best\n";
  std::cout << "--------------------------------------------------------\n";
  for (const auto& [qb, t] : timings) {
    const double qps = static_cast<double>(total_query_clouds) / t;
    std::cout << std::setw(10) << qb << "  " << std::setw(12) << t << "  " << std::setw(15) << qps
              << "  " << std::setw(7) << (t / best) << "x\n";
  }
  std::cout << "checksum: " << checksum << "\n";

  return 0;
}
