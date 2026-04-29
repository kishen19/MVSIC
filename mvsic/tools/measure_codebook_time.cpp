// Measure codebook training (and encode) time for each quantizer on a given .pcs file.
//
// Usage:
//   bazel run //mvsic/tools:measure_codebook_time -- \
//       -d /path/to/points.pcs \
//       [-metric ip|l2] \
//       [-quantizer {all|pq|fastscan|rabitq|turboquant|spqtq|onebittq}] \
//       [-block_size 32] [-rabitq_bits 7]
//
// Outputs one JSON line per quantizer:
//   {"dataset":"...","quantizer":"fastscan","n":N,"d":D,"train_sec":...,"encode_sec":...}
//
// This is intended to be consumed by data-tools/measure_codebook_time.py.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "mvsic/core/quantization/fastscan_mv.h"
#include "mvsic/core/quantization/pq_mv.h"
#include "mvsic/core/quantization/pqtq_mv.h"
#include "mvsic/core/quantization/rabitq_mv.h"
#include "mvsic/core/quantization/turboquant_1bit_mv.h"
#include "mvsic/core/quantization/turboquant_mv.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/core/utils/parse_command_line.h"

using namespace mvsic;

namespace {

inline double now_sec() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

template <bool Metric, typename Model, typename... TrainArgs>
void measure(const std::string& dataset, const std::string& name,
             const PointCloudSet<
                 std::conditional_t<Metric, ChamferL2_Point, ChamferIP_Point>>& pcs,
             TrainArgs... train_args) {
  Model m;
  double t0 = now_sec();
  m.train(pcs, train_args...);
  double t1 = now_sec();
  double enc0 = now_sec();
  (void)m.encode(pcs);
  double enc1 = now_sec();
  std::cout << "{\"dataset\":\"" << dataset << "\",\"quantizer\":\"" << name
            << "\",\"metric\":\"" << (Metric ? "l2" : "ip") << "\",\"n\":"
            << pcs.size() << ",\"d\":" << pcs.get_dims()
            << ",\"train_sec\":" << (t1 - t0)
            << ",\"encode_sec\":" << (enc1 - enc0) << "}\n";
}

template <bool Metric>
void run(commandLine& P) {
  using PCPoint = std::conditional_t<Metric, ChamferL2_Point, ChamferIP_Point>;
  using PCS = PointCloudSet<PCPoint>;

  const char* data_path = P.getOptionValue("-d");
  if (!data_path) {
    std::cerr << "Error: specify -d <points.pcs>\n";
    std::exit(1);
  }
  const std::string dataset(data_path);
  const std::string which = P.getOptionValue("-quantizer", "all");
  const uint32_t block_size = P.getOptionIntValue("-block_size", 32);
  const uint32_t rabitq_bits = P.getOptionIntValue("-rabitq_bits", 7);

  bool mmap = P.getOption("-mmap");
  auto pcs = PCS(data_path, mmap);
  std::cerr << "Loaded " << pcs.size() << " point clouds, dim=" << pcs.get_dims()
            << " (metric=" << (Metric ? "l2" : "ip") << ")\n";

  auto match = [&](const std::string& name) {
    return which == "all" || which == name;
  };

  if (match("pq")) {
    measure<Metric, pq_mv::Model<Metric>>(dataset, "pq", pcs, block_size);
  }
  if (match("fastscan")) {
    measure<Metric, fastscan_mv::Model<Metric>>(dataset, "fastscan", pcs, block_size);
  }
  if (match("rabitq")) {
    measure<Metric, rabitq_mv::Model<Metric>>(dataset, "rabitq", pcs, rabitq_bits);
  }
  if (match("turboquant")) {
    measure<Metric, turboquant_mv::Model<Metric>>(dataset, "turboquant", pcs);
  }
  if (match("spqtq")) {
    measure<Metric, pqtq_mv::Model<Metric>>(dataset, "spqtq", pcs, block_size);
  }
  if (match("onebittq")) {
    measure<Metric, turboquant_1bit_mv::Model<Metric>>(dataset, "onebittq", pcs);
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  commandLine P(argc, argv,
                "-d <pcs> [-metric ip|l2] [-quantizer {all|pq|fastscan|rabitq|"
                "turboquant|spqtq|onebittq}] [-block_size N] [-rabitq_bits N] "
                "[-mmap]");
  std::string metric = P.getOptionValue("-metric", "ip");
  if (metric == "l2") {
    run<true>(P);
  } else {
    run<false>(P);
  }
  return 0;
}
