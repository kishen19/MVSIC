// Compile against the original tree with MVSIC_BASELINE, or link the new library.
// Both executables use the same command line and on-disk result format.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/resource.h>
#include <unistd.h>

#ifdef MVSIC_BASELINE
#include "mvsic/mvivf/mvivf.h"
#else
#include "mvsic/algorithms/mvivf/mvivf.h"
#endif

namespace {
using Clock = std::chrono::steady_clock;
using Row = std::vector<std::pair<uint32_t, float>>;
using Rows = std::vector<Row>;

double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

size_t peak_rss() {
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  return size_t(usage.ru_maxrss) * 1024;  // Linux measurement host.
}

size_t rss() {
  std::ifstream file("/proc/self/statm");
  size_t pages = 0, resident = 0;
  file >> pages >> resident;
  return resident * size_t(sysconf(_SC_PAGESIZE));
}

void snapshot_prepared_memory(const std::string& output) {
  // Both comparison executables record placement after lazy cache preparation,
  // outside the timing loop. Keep raw NUMA pages and Mems_allowed_list evidence.
  for (const auto& item : std::vector<std::pair<std::string, std::string>>{
           {"/proc/self/numa_maps", ".numa_maps"}, {"/proc/self/status", ".proc_status"}}) {
    std::ifstream source(item.first);
    std::ofstream target(output + item.second);
    if (!source || !target) throw std::runtime_error("Cannot open prepared-memory snapshot");
    target << source.rdbuf();
    target.close();
    if (source.bad() || !target) throw std::runtime_error("Cannot save prepared-memory snapshot");
  }
}

struct Args {
  std::map<std::string, std::string> values;
  Args(int argc, char** argv) {
    const std::vector<std::string> allowed = {"points", "queries", "index", "output", "mode", "metric",
        "k", "nprobes", "rerank", "tq8-rerank", "compression", "threshold", "root-m2m",
        "repeats", "warmup", "leaf-size", "branching", "depth", "niters", "s", "seed", "assignment"};
    for (int i = 1; i < argc; i += 2) {
      std::string flag = argv[i];
      if (i + 1 == argc || flag.rfind("--", 0) ||
          std::find(allowed.begin(), allowed.end(), flag.substr(2)) == allowed.end())
        throw std::invalid_argument("unknown option or missing value: " + flag);
      if (!values.emplace(flag.substr(2), argv[i + 1]).second)
        throw std::invalid_argument("duplicate option: " + flag);
    }
  }
  std::string str(const std::string& key, const std::string& fallback = "") const {
    auto item = values.find(key);
    return item == values.end() ? fallback : item->second;
  }
  uint32_t number(const std::string& key, uint32_t fallback) const {
    const auto value = str(key, std::to_string(fallback));
    size_t used = 0;
    const auto parsed = std::stoull(value, &used);
    if (used != value.size() || parsed > UINT32_MAX || value.front() == '-')
      throw std::invalid_argument("invalid integer: " + key);
    return uint32_t(parsed);
  }
  bool boolean(const std::string& key, bool fallback) const {
    const auto value = number(key, fallback);
    if (value > 1) throw std::invalid_argument("expected 0 or 1: " + key);
    return value != 0;
  }
};

void write_predictions(const std::string& filename, const Rows& rows, uint32_t k) {
  std::ofstream output(filename, std::ios::binary);
  output.exceptions(std::ios::badbit | std::ios::failbit);
  const uint32_t count = rows.size();
  output.write(reinterpret_cast<const char*>(&count), 4);
  output.write(reinterpret_cast<const char*>(&k), 4);
  for (const auto& row : rows) for (uint32_t j = 0; j < k; ++j) {
    const uint32_t id = j < row.size() ? row[j].first : UINT32_MAX;
    const float distance = j < row.size() ? row[j].second : std::numeric_limits<float>::infinity();
    output.write(reinterpret_cast<const char*>(&id), 4);
    output.write(reinterpret_cast<const char*>(&distance), 4);
  }
}

void array(std::ostream& output, const std::vector<double>& values) {
  output << '[';
  for (size_t i = 0; i < values.size(); ++i) output << (i ? "," : "") << values[i];
  output << ']';
}

void run(const Args& args) {
  const auto mode = args.str("mode", "latency");
  if (mode != "build" && mode != "latency" && mode != "batch")
    throw std::invalid_argument("mode must be build, latency or batch");
  const auto points_path = args.str("points"), skeleton = args.str("index"), output_path = args.str("output");
  if (points_path.empty() || skeleton.empty() || output_path.empty())
    throw std::invalid_argument("--points, --index and --output are required");
  const auto load_start = Clock::now();
#ifdef MVSIC_BASELINE
  using Cloud = mvsic::ChamferIP_Point;
  using Set = mvsic::PointCloudSet<Cloud>;
  using Index = mvsic::IndexMVIVF<false, true, mvsic::turboquant_1bit_mv::Model<false>>;
  Set points(points_path.c_str(), true);
#else
  using Set = mvsic::PointCloudSetView;
  using Index = mvsic::IndexMVIVF;
  Set points = mvsic::load_point_clouds(points_path, true);
#endif
  const double load_seconds = seconds(load_start);
  const auto assignment = args.str("assignment", "float");
  if (assignment != "float" && assignment != "tq8")
    throw std::invalid_argument("comparison supports assignment float or tq8 shared by both implementations");
  auto ip = mvsic::IndexParams::mvivf(args.number("branching", 0), args.number("leaf-size", 100),
      false, 0, args.number("niters", 5), 100, 20, "Random", args.number("seed", 0), true,
      args.number("s", 0), args.number("depth", 0), assignment == "tq8");
#ifdef MVSIC_BASELINE
  Index index(points.get_dims(), ip);
#else
  ip.pq.method = mvsic::IndexParams::QuantizerType::OneBitTQ;
  ip.quantize_centers = true;
  Index index(points.get_dims(), ip);
#endif
  const size_t rss_before = rss();
  const auto prepare_start = Clock::now();
  if (mode == "build") index.build(points);
  else index.load(skeleton, points);
  const double prepare_seconds = seconds(prepare_start);
  const size_t rss_prepared = rss();
  std::ofstream output(output_path);
  output.exceptions(std::ios::badbit | std::ios::failbit);
  output << std::setprecision(17) << "{\"mode\":\"" << mode << "\",\"load_seconds\":" << load_seconds
         << ",\"prepare_seconds\":" << prepare_seconds << ",\"rss_before_bytes\":" << rss_before
         << ",\"rss_prepared_bytes\":" << rss_prepared;
  if (mode == "build") {
    const auto start = Clock::now();
    index.save(skeleton);
    output << ",\"save_seconds\":" << seconds(start) << ",\"peak_rss_bytes\":" << peak_rss() << "}\n";
    return;
  }
  if (args.str("queries").empty()) throw std::invalid_argument("--queries is required for search");
#ifdef MVSIC_BASELINE
  Set queries(args.str("queries").c_str(), false);
#else
  Set queries = mvsic::load_point_clouds(args.str("queries"), false);
#endif
  if (queries.size() == 0) throw std::invalid_argument("empty queries");
  const auto k = args.number("k", 10), nprobes = args.number("nprobes", 16), rerank = args.number("rerank", 50);
  const auto repeats = args.number("repeats", 5), warmup = args.number("warmup", 10);
  if (!k || !nprobes || !repeats || (rerank && rerank < k)) throw std::invalid_argument("invalid search counts");
  using Compression = mvsic::SearchParams::QueryCompression;
  const auto compression = args.str("compression", "None");
  if (compression != "None" && compression != "Wards" && compression != "Carve")
    throw std::invalid_argument("unsupported compression");
  auto sp = mvsic::SearchParams::mvivf(k, nprobes, rerank, args.boolean("tq8-rerank", true),
      compression == "None" ? Compression::None : compression == "Wards" ? Compression::Wards : Compression::Carve,
      std::stof(args.str("threshold", "0.5")));
  sp.root_m2m = args.boolean("root-m2m", true);
  auto search_one = [&](size_t q) {
#ifdef MVSIC_BASELINE
    return index.search(queries[q], points, sp).first;
#else
    return index.search(queries[q], sp);
#endif
  };
  auto search_batch = [&]() {
#ifdef MVSIC_BASELINE
    return index.search_all(queries, points, sp).first;
#else
    return index.search_all(queries, sp);
#endif
  };
  // TQ8 caches initialize on first use. Record that cost separately from warm
  // search so a preparation-policy change cannot masquerade as free work.
  const auto first_start = Clock::now();
  search_one(0);
  const double first_query_seconds = seconds(first_start);
  snapshot_prepared_memory(output_path);
  const auto warm_start = Clock::now();
  if (mode == "batch") {
    if (warmup) search_batch();
  } else for (size_t q = 0; q < std::min<size_t>(warmup, queries.size()); ++q) search_one(q);
  const double warmup_seconds = seconds(warm_start);
  std::vector<double> times(repeats);
  std::vector<std::vector<double>> query_times(repeats);
  Rows best;
  size_t selected = 0;
  for (size_t r = 0; r < repeats; ++r) {
    Rows rows(queries.size());
    if (mode == "batch") {
      auto start = Clock::now();
      auto result = search_batch();
      times[r] = seconds(start);
      for (size_t q = 0; q < queries.size(); ++q) rows[q].assign(result[q].begin(), result[q].end());
    } else {
      query_times[r].resize(queries.size());
      for (size_t q = 0; q < queries.size(); ++q) {
        auto start = Clock::now();
        auto result = search_one(q);
        query_times[r][q] = seconds(start);
        times[r] += query_times[r][q];
        rows[q].assign(result.begin(), result.end());
      }
    }
    if (r == 0 || times[r] < times[selected]) { selected = r; best = std::move(rows); }
  }
  write_predictions(output_path + ".predictions", best, k);
  output << ",\"first_query_seconds\":" << first_query_seconds
         << ",\"memory_placement_policy\":\"inherited Linux policy; no placement enforced\""
         << ",\"prepared_memory_snapshot\":\"after first query and before warmup; .numa_maps and .proc_status sidecars\""
         << ",\"warmup_seconds\":" << warmup_seconds << ",\"queries\":" << queries.size()
         << ",\"k\":" << k << ",\"selected_repeat\":" << selected
         << ",\"qps\":" << (queries.size() / times[selected]) << ",\"seconds\":";
  array(output, times);
  output << ",\"query_seconds\":";
  if (mode == "batch") output << "null";
  else {
    output << '[';
    for (size_t r = 0; r < repeats; ++r) { if (r) output << ','; array(output, query_times[r]); }
    output << ']';
  }
  output << ",\"rss_final_bytes\":" << rss() << ",\"peak_rss_bytes\":" << peak_rss() << "}\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Args args(argc, argv);
    if (args.str("metric", "ip") != "ip")
      throw std::invalid_argument("MVIVF retrieval supports only metric ip; L2 remains a clustering objective");
    run(args);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
