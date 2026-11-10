#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include <sys/resource.h>

#include "mvsic/vamana/vamana.h"
#include "mvsic/core/stats.h"

// Benchmark Vamana leaf quantization and reranking on a shared saved graph.
// Each invocation measures one leaf quantizer with optional TQ8 reranking.
namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using PC = mvsic::PointCloudSet<mvsic::ChamferIP_Point>;

struct Options {
  std::string method = "mv_vamana", quant = "none";
  std::string points, queries, gt, index, csv, build_stats, query_ids, quantizer_cache_prefix;
  std::vector<size_t> widths{10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512};
  uint32_t degree = 64, build_width = 128;
  double alpha = 1.0;
  size_t k = 10, sample = 1000, repeats = 1, num_rerank = 0, max_points = 0;
  uint32_t seed = 42;
  bool two_pass = false, rebuild = false, mmap = false, build_only = false, tq8_rerank = true,
       early_stop = false;
};

void make_parent(const std::string& path) {
  if (!fs::path(path).parent_path().empty()) fs::create_directories(fs::path(path).parent_path());
}

template<typename Model>
bool load_shared_quantizer(Model& model, const std::string& path, size_t dimension) {
  if (!fs::exists(path)) return false;
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot read shared quantizer: " + path);
  // Both model formats begin with BaseEncoder::dim. Check it before load()
  // constructs a rotator, so an unrelated dataset cache fails explicitly.
  size_t saved_dimension = 0;
  input.read(reinterpret_cast<char*>(&saved_dimension), sizeof(saved_dimension));
  if (!input || saved_dimension != dimension)
    throw std::invalid_argument("shared quantizer dimension mismatch: " + path);
  input.seekg(0);
  model.load(input);
  if (!input || model.encoder.dim != dimension || model.encoder.padded_dim < dimension || !model.encoder.rotator)
    throw std::runtime_error("invalid shared quantizer: " + path);
  std::cout << "Loaded shared quantizer " << path << '\n' << std::flush;
  return true;
}

template<typename Model>
void save_shared_quantizer(const Model& model, const std::string& path) {
  make_parent(path);
  const std::string temporary = path + ".tmp";
  std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
  if (!output) throw std::runtime_error("cannot write shared quantizer: " + temporary);
  model.save(output);
  output.flush();
  if (!output) throw std::runtime_error("failed writing shared quantizer: " + temporary);
  output.close();
  fs::rename(temporary, path);
  std::cout << "Saved shared quantizer " << path << '\n' << std::flush;
}

// Benchmark-only access to the existing rerank cache. Sharing the serialized
// default rotators removes random-device variation between quantization
// variants; search and reranking use the standard Vamana implementation.
template<typename IndexT>
class BenchIndex : public IndexT {
 public:
  using IndexT::IndexT;

  void prepare_shared_rerank(const PC& points, const std::string& prefix) {
    std::call_once(this->tq8_rerank_once_, [&] {
      auto model = std::make_unique<typename IndexT::BTQModel>();
      const std::string path = prefix + ".tq8";
      if (!load_shared_quantizer(*model, path, points.get_dims())) {
        model->train(points);
        save_shared_quantizer(*model, path);
      }
      auto encoded = std::make_unique<typename IndexT::BTQEncSet>(model->encode(points));
      this->tq8_rerank_model_ = std::move(model);
      this->tq8_rerank_db_ = std::move(encoded);
    });
  }
};

std::string json_string(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c == '\n') out << "\\n";
    else if (c == '\r') out << "\\r";
    else if (c == '\t') out << "\\t";
    else out << c;
  }
  out << '"';
  return out.str();
}

size_t parse_size(const std::string& value, const std::string& key) {
  if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument(key + " requires an unsigned integer");
  const auto parsed = std::stoull(value);
  if (parsed > std::numeric_limits<size_t>::max())
    throw std::out_of_range(key + " exceeds size_t range");
  return static_cast<size_t>(parsed);
}

uint32_t parse_u32(const std::string& value, const std::string& key) {
  const auto parsed = parse_size(value, key);
  if (parsed > std::numeric_limits<uint32_t>::max())
    throw std::out_of_range(key + " exceeds uint32 range");
  return static_cast<uint32_t>(parsed);
}

Options parse(int argc, char** argv) {
  Options result;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help") {
      std::cout << "MV-Vamana leaf quantization latency benchmark\n"
          "  --method mv_vamana  (optional; retained for existing experiment commands)\n"
          "  --points PATH --queries PATH --gt PATH --index PATH --csv PATH\n"
          "  --build-stats PATH  (default: INDEX.build.json)\n"
          "  --R 64 --build-L 128 --alpha 1.0 [--two-pass]\n"
          "  --k 10 --L 10,16,24,32,48,64,96,128,192,256,384,512\n"
          "  --sample 1000 --seed 42 --repeats 1\n"
          "  --early-stop  (suite defaults: recall 1, plateau/drop, or QPS < 10)\n"
          "  --query-ids PATH  (one query row index per line; overrides sampling)\n"
          "  --quantizer-cache-prefix PATH  (persist .tq4/.tq1 leaves and shared .tq8 rerank)\n"
          "  --quant none|tq|tq4|tq1  (tq4 aliases tq)\n"
          "  --num-rerank 0 --tq8-rerank true|false\n"
          "  --max-points N  (build-only pilot on first N documents)\n"
          "  --rebuild --mmap --build-only\n"
          "Existing graphs are loaded unless --rebuild is given. A sample of\n"
          "zero uses all queries. Search uses Chamfer without query compression,\n"
          "and one thread per query. Set PARLAY_NUM_THREADS for build.\n";
      std::exit(0);
    }
    if (key == "--two-pass") { result.two_pass = true; continue; }
    if (key == "--rebuild") { result.rebuild = true; continue; }
    if (key == "--mmap") { result.mmap = true; continue; }
    if (key == "--build-only") { result.build_only = true; continue; }
    if (key == "--early-stop") { result.early_stop = true; continue; }
    if (++i == argc) throw std::invalid_argument("missing value for " + key);
    const std::string value = argv[i];
    if (key == "--method") result.method = value;
    else if (key == "--quant") result.quant = value;
    else if (key == "--num-rerank") result.num_rerank = parse_size(value, key);
    else if (key == "--max-points") result.max_points = parse_size(value, key);
    else if (key == "--tq8-rerank") {
      if (value != "true" && value != "false") throw std::invalid_argument("--tq8-rerank must be true or false");
      result.tq8_rerank = value == "true";
    }
    else if (key == "--points") result.points = value;
    else if (key == "--queries") result.queries = value;
    else if (key == "--query-ids") result.query_ids = value;
    else if (key == "--quantizer-cache-prefix") result.quantizer_cache_prefix = value;
    else if (key == "--gt") result.gt = value;
    else if (key == "--index") result.index = value;
    else if (key == "--csv") result.csv = value;
    else if (key == "--build-stats") result.build_stats = value;
    else if (key == "--R") result.degree = parse_u32(value, key);
    else if (key == "--build-L") result.build_width = parse_u32(value, key);
    else if (key == "--alpha") result.alpha = std::stod(value);
    else if (key == "--k") result.k = parse_size(value, key);
    else if (key == "--sample") result.sample = parse_size(value, key);
    else if (key == "--seed") result.seed = parse_u32(value, key);
    else if (key == "--repeats") result.repeats = parse_size(value, key);
    else if (key == "--L") {
      result.widths.clear();
      std::istringstream stream(value);
      std::string field;
      while (std::getline(stream, field, ',')) result.widths.push_back(parse_size(field, key));
    } else throw std::invalid_argument("unknown option: " + key);
  }
  if (result.points.empty() || result.index.empty())
    throw std::invalid_argument("--points and --index are required");
  if (!result.build_only && (result.queries.empty() || result.gt.empty() || result.csv.empty()))
    throw std::invalid_argument("--queries, --gt and --csv are required unless --build-only");
  if (!result.degree || !result.build_width || !std::isfinite(result.alpha) || result.alpha <= 0 || !result.k ||
      !result.repeats || result.widths.empty())
    throw std::invalid_argument("R, build-L, alpha, k, repeats and L must be positive");
  for (size_t width : result.widths)
    if (width < result.k) throw std::invalid_argument("each search L must be at least k");
  if (result.quant == "tq4") result.quant = "tq";
  if (result.quant != "none" && result.quant != "tq" && result.quant != "tq1")
    throw std::invalid_argument("--quant must be none, tq, tq4 or tq1");
  if (result.method != "mv_vamana")
    throw std::invalid_argument("--method must be mv_vamana");
  if (result.num_rerank && result.num_rerank < result.k)
    throw std::invalid_argument("--num-rerank must be zero or at least k");
  if (result.max_points && !result.build_only)
    throw std::invalid_argument("--max-points requires --build-only so full-corpus GT is never misused");
  if (result.build_stats.empty()) result.build_stats = result.index + ".build.json";
  return result;
}

template<typename Index>
void write_build_stats(const Options& o, const Index& index, const PC& points,
                       bool loaded, double seconds) {
  // Preserve original build timing when reloading our own graph.
  if (loaded && fs::exists(o.build_stats)) return;
  size_t edges = 0, max_degree = 0;
  for (size_t i = 0; i < index.G.size(); ++i) {
    edges += index.G[i].size();
    max_degree = std::max(max_degree, index.G[i].size());
  }
  std::vector<bool> seen(points.size(), false);
  std::vector<uint32_t> queue{index.start_point};
  seen[index.start_point] = true;
  for (size_t head = 0; head < queue.size(); ++head) {
    const auto neighbors = index.G[queue[head]];
    for (size_t j = 0; j < neighbors.size(); ++j) {
      const auto next = neighbors[j];
      if (!seen[next]) { seen[next] = true; queue.push_back(next); }
    }
  }
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  make_parent(o.build_stats);
  std::ofstream out(o.build_stats);
  if (!out) throw std::runtime_error("cannot write " + o.build_stats);
  out << std::setprecision(12)
      << "{\n  \"method\": " << json_string(o.method)
      << ",\n  \"points_path\": " << json_string(o.points)
      << ",\n  \"num_points\": " << points.size()
      << ",\n  \"num_vectors\": " << points.total_size()
      << ",\n  \"dimension\": " << points.get_dims()
      << ",\n  \"R\": " << o.degree << ",\n  \"build_L\": " << o.build_width
      << ",\n  \"alpha\": " << o.alpha
      << ",\n  \"two_pass\": " << (o.two_pass ? "true" : "false")
      << ",\n  \"loaded_existing_graph\": " << (loaded ? "true" : "false")
      << ",\n  \"build_seconds\": ";
  if (loaded) out << "null"; else out << seconds;
  out << ",\n  \"index_bytes\": " << fs::file_size(o.index)
      << ",\n  \"peak_rss_kib\": " << usage.ru_maxrss
      << ",\n  \"mean_degree\": " << double(edges) / points.size()
      << ",\n  \"max_degree\": " << max_degree
      << ",\n  \"reachable_nodes\": " << queue.size()
      << ",\n  \"prune_ground_cost\": "
      << json_string("Chamfer IP")
      << ",\n  \"query_distance\": \"Chamfer IP\"\n}\n";
}

template<typename Index>
void run_search(const Options& o, Index& index, const PC& points) {
  const PC all_queries(o.queries.c_str());
  if (!all_queries.size() || o.k > points.size()) throw std::invalid_argument("invalid query count or k");
  auto gt_path = o.gt;
  const auto all_gt = ReadGT(gt_path, all_queries.size());
  if (all_queries.get_dims() != points.get_dims()) throw std::invalid_argument("query and corpus dimensions differ");
  for (const auto& row : all_gt)
    if (row.size() < o.k) throw std::invalid_argument("ground truth has fewer than k neighbors");
  std::vector<size_t> selected;
  if (!o.query_ids.empty()) {
    std::ifstream input(o.query_ids);
    if (!input) throw std::runtime_error("cannot read query IDs: " + o.query_ids);
    std::unordered_set<size_t> seen;
    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line)) {
      ++line_number;
      const auto begin = line.find_first_not_of(" \t\r");
      const auto end = line.find_last_not_of(" \t\r");
      const auto field = begin == std::string::npos ? std::string{} : line.substr(begin, end - begin + 1);
      const auto id = parse_size(field, o.query_ids + ":" + std::to_string(line_number));
      if (id >= all_queries.size()) throw std::invalid_argument("query ID out of range: " + std::to_string(id));
      if (!seen.insert(id).second) throw std::invalid_argument("duplicate query ID: " + std::to_string(id));
      selected.push_back(id);
    }
    if (!input.eof()) throw std::runtime_error("failed reading query IDs: " + o.query_ids);
    if (selected.empty()) throw std::invalid_argument("query ID file is empty: " + o.query_ids);
  } else {
    selected.resize(all_queries.size());
    std::iota(selected.begin(), selected.end(), size_t{0});
    std::mt19937 rng(o.seed);
    std::shuffle(selected.begin(), selected.end(), rng);
    if (o.sample && o.sample < selected.size()) selected.resize(o.sample);
    std::sort(selected.begin(), selected.end());
  }
  std::vector<mvsic::ChamferIP_Point> selected_views;
  selected_views.reserve(selected.size());
  for (size_t id : selected) selected_views.push_back(all_queries[id]);
  const PC queries(selected_views, all_queries.get_dims());
  const auto gt = parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>::from_function(
      selected.size(), [&](size_t i) { return all_gt[selected[i]]; });

  make_parent(o.csv);
  std::ofstream sample_ids(o.csv + ".query_ids.txt");
  if (!sample_ids) throw std::runtime_error("cannot write query sample IDs");
  for (size_t id : selected) sample_ids << id << '\n';
  std::ofstream out(o.csv);
  if (!out) throw std::runtime_error("cannot write " + o.csv);
  out << "method,quant,num_rerank,tq8_rerank,repeat,k,L,num_queries,query_seed,R,build_L,alpha,two_pass,"
         "recall_1_k,recall_k_k,QPS_seq,mean_latency_ms,p50_latency_ms,p95_latency_ms,"
         "p99_latency_ms,avg_bytes_accessed,avg_search_cmps\n";
  out << std::setprecision(12);
  for (size_t repeat = 0; repeat < o.repeats; ++repeat) {
    // The standard suite evaluates ascending L values and uses the observed
    // recall trend for early stopping. Optional full sweeps alternate order.
    auto widths = o.widths;
    if (!o.early_stop && repeat % 2) std::reverse(widths.begin(), widths.end());
    std::vector<double> recalls;
    for (size_t width : widths) {
      const auto sp = mvsic::SearchParams::vamana(o.k, width, 1.35, o.num_rerank, o.tq8_rerank);
      const auto stats = compute_stats_latency(index, points, queries, gt, sp);
      const double avg_search_cmps = stats.avg_timings.empty() ? 0 : stats.avg_timings[0] / queries.size();
      out << o.method << ',' << o.quant << ',' << o.num_rerank << ',' << int(o.tq8_rerank) << ',' << repeat << ',' << o.k << ',' << width << ',' << queries.size()
          << ',' << o.seed << ',' << o.degree << ',' << o.build_width << ',' << o.alpha
          << ',' << int(o.two_pass) << ',' << stats.recall_1_k << ',' << stats.recall_k_k
          << ',' << stats.QPS_seq << ',' << 1000.0 / stats.QPS_seq
          << ',' << 1000 * stats.latency_p50 << ',' << 1000 * stats.latency_p95
          << ',' << 1000 * stats.latency_p99 << ',' << stats.avg_cmps
          << ',' << avg_search_cmps << '\n';
      out.flush();
      std::cout << o.method << " quant=" << o.quant << " nr=" << o.num_rerank << " repeat=" << repeat << " L=" << width
                << " recall@" << o.k << '=' << stats.recall_k_k
                << " mean_ms=" << 1000.0 / stats.QPS_seq
                << " p95_ms=" << 1000 * stats.latency_p95 << '\n' << std::flush;
      if (o.early_stop) {
        recalls.push_back(stats.recall_k_k);
        const size_t count = recalls.size();
        const char* stop_reason = nullptr;
        if (stats.recall_k_k >= 1.0) stop_reason = "recall@k reached 1.0";
        else if (count > 3 && recalls[count - 1] == recalls[count - 2] &&
                 recalls[count - 1] == recalls[count - 3]) stop_reason = "recall plateau";
        else if (count > 1 && recalls[count - 1] < recalls[count - 2]) stop_reason = "recall dropped";
        else if (stats.QPS_seq > 0 && stats.QPS_seq < 10.0) stop_reason = "QPS below 10";
        if (stop_reason) {
          std::cout << "Stopping sweep: " << stop_reason << '\n' << std::flush;
          break;
        }
      }
    }
  }
}
template<typename LeafIndex>
void run(const Options& o) {
  Eigen::setNbThreads(1);
  PC points(o.points.c_str(), o.mmap);
  if (o.max_points && o.max_points < points.size()) {
    std::vector<mvsic::ChamferIP_Point> subset;
    subset.reserve(o.max_points);
    for (size_t i = 0; i < o.max_points; ++i) subset.push_back(points[i]);
    points = PC(subset, points.get_dims());
  }
  if (!points.size()) throw std::invalid_argument("empty dataset");
  if (o.degree >= points.size()) throw std::invalid_argument("R must be smaller than dataset size");
  const auto params = mvsic::IndexParams::vamana(o.degree, o.build_width, o.alpha, o.two_pass,
                                               false, 1);
  {
    mvsic::IndexVamanaIP index(points.get_dims(), params);
    const bool loaded = fs::exists(o.index) && !o.rebuild;
    const auto started = Clock::now();
    if (loaded) index.load(o.index, points);
    else {
      index.build(points);
      make_parent(o.index);
      index.save(o.index);
    }
    const double build_seconds = std::chrono::duration<double>(Clock::now() - started).count();
    if (index.G.size() != points.size() || index.start_point >= points.size())
      throw std::invalid_argument("loaded graph does not match corpus size or has an invalid start point");
    if (index.G.max_degree() != o.degree)
      throw std::invalid_argument("loaded graph degree differs from --R; use a separate index path or --rebuild");
    write_build_stats(o, index, points, loaded, build_seconds);
    std::cout << o.method << (loaded ? " loaded in " : " built/saved in ")
              << build_seconds << " seconds\n" << std::flush;
    if (o.build_only) return;
    if (o.quant == "none") {
      run_search(o, index, points);
      return;
    }
  }
  // Quantized variants share the graph skeleton; training and encoding on
  // load stay outside the timed query loop. The raw graph is released first.
  BenchIndex<LeafIndex> quantized_index(points.get_dims(), params);
  quantized_index.load(o.index, points);
  if (!o.quantizer_cache_prefix.empty()) {
    const bool one_bit = o.quant == "tq1";
    const auto path = o.quantizer_cache_prefix + (one_bit ? ".tq1" : ".tq4");
    const bool primary_loaded = load_shared_quantizer(quantized_index.leaf_model_, path, points.get_dims());
    bool model_loaded = primary_loaded;
    if (!model_loaded) {
      // Both MV leaf models serialize exactly turboquant::BaseEncoder. TQ1
      // computes its sign-code byte width from padded_dim, independently of
      // the 4-bit encoder's byte-width field, so the same rotator is valid.
      // Seed a missing leaf cache from the other bit depth when available.
      const auto alternate = o.quantizer_cache_prefix + (one_bit ? ".tq4" : ".tq1");
      model_loaded = load_shared_quantizer(quantized_index.leaf_model_, alternate, points.get_dims());
    }
    if (model_loaded)
      quantized_index.leaf_encoded_ = quantized_index.leaf_model_.encode(points);
    if (!primary_loaded)
      save_shared_quantizer(quantized_index.leaf_model_, path);
    if (o.num_rerank > 0 && o.tq8_rerank)
      quantized_index.prepare_shared_rerank(points, o.quantizer_cache_prefix);
  }
  run_search(o, quantized_index, points);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse(argc, argv);
    if (options.quant == "tq1") run<mvsic::IndexVamanaOneBitTQIP>(options);
    else run<mvsic::IndexVamanaTQIP>(options);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Vamana quantization benchmark: " << error.what() << '\n';
    return 1;
  }
}
