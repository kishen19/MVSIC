#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "mvsic/algorithms/emdivf/emdivf.h"
#include "parlay/primitives.h"

namespace {
using Args = std::map<std::string, std::string>;

std::string Get(const Args& args, const std::string& name, const std::string& fallback = "") {
  const auto found = args.find(name);
  if (found != args.end()) return found->second;
  if (!fallback.empty()) return fallback;
  throw std::invalid_argument("missing --" + name);
}

uint32_t Unsigned(const std::string& value) {
  size_t used = 0;
  const auto parsed = std::stoull(value, &used);
  if (value.empty() || value.front() == '-' || used != value.size() || parsed > UINT32_MAX)
    throw std::invalid_argument("expected an unsigned 32-bit integer: " + value);
  return static_cast<uint32_t>(parsed);
}

std::vector<uint32_t> Numbers(const std::string& text) {
  std::vector<uint32_t> result;
  size_t begin = 0;
  while (begin < text.size()) {
    const auto end = text.find(',', begin);
    result.push_back(Unsigned(text.substr(begin, end - begin)));
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  if (result.empty()) throw std::invalid_argument("empty parameter sweep");
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

// The embed/ exact-Chamfer GT layout: uint32 k, then query-major records
// (float32 distance, uint32 document row). Qrels/gold relevance are not read.
std::vector<std::vector<uint32_t>> ReadExactGT(const std::filesystem::path& path,
                                             size_t queries, uint32_t k,
                                             size_t documents) {
  std::ifstream in(path, std::ios::binary);
  uint32_t stored = 0;
  in.read(reinterpret_cast<char*>(&stored), sizeof(stored));
  const uint64_t expected = 4 + uint64_t(queries) * stored * 8;
  if (!in || stored < k || std::filesystem::file_size(path) != expected)
    throw std::runtime_error("exact GT row count or neighbor count does not match queries");
  std::vector<std::vector<uint32_t>> gt(queries, std::vector<uint32_t>(k));
  for (size_t i = 0; i < queries; ++i) for (uint32_t j = 0; j < stored; ++j) {
    float distance;
    uint32_t id;
    in.read(reinterpret_cast<char*>(&distance), sizeof(distance));
    in.read(reinterpret_cast<char*>(&id), sizeof(id));
    if (!in || !std::isfinite(distance) || id >= documents)
      throw std::runtime_error("invalid exact GT record");
    if (j < k) gt[i][j] = id;
  }
  return gt;
}

double Mean(const std::vector<double>& values) {
  return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double Quantile(std::vector<double> values, double p) {
  std::sort(values.begin(), values.end());
  const double position = p * (values.size() - 1);
  const size_t lo = position, hi = std::min(lo + 1, values.size() - 1);
  return values[lo] + (position - lo) * (values[hi] - values[lo]);
}

void Search(const Args& args) {
  auto buckets = emdivf::Load(Get(args, "index"));
  const auto m = buckets.centers.size();
  const auto stored_b = buckets.max_bucket_size;
  emdivf::CloudSet train(Get(args, "train").c_str(), true);
  emdivf::CloudSet docs(Get(args, "docs").c_str(), true);
  emdivf::CloudSet queries(Get(args, "queries").c_str(), true);
  if (!queries.size()) throw std::invalid_argument("no query point clouds");
  emdivf::SearchParams params;
  params.k = Unsigned(Get(args, "k", "10"));
  params.shortlist = Unsigned(Get(args, "shortlist", "128"));
  params.bucket_size = Unsigned(Get(args, "b", "0"));
  const auto compression = Get(args, "query-compress", "wards");
  if (compression != "wards" && compression != "none")
    throw std::invalid_argument("--query-compress must be wards or none");
  params.compress_query = compression == "wards";
  params.compression_threshold = std::stof(Get(args, "compression-threshold", "0.5"));
  if (!params.k) throw std::invalid_argument("--k must be positive");
  const auto probes = Numbers(Get(args, "nprobes", "8,16,32,64,128"));
  const auto reranks = Numbers(Get(args, "num-rerank", "100,200,500"));
  const auto requested_timing = Unsigned(Get(args, "timing-queries", "0"));
  const size_t timed = requested_timing ? std::min<size_t>(requested_timing, queries.size())
                                      : queries.size();
  const auto gt = ReadExactGT(Get(args, "gt"), queries.size(), params.k, docs.size());
  emdivf::SearchIndex index(train, docs, std::move(buckets), Get(args, "graph"));
  const auto output = std::filesystem::path(Get(args, "output"));
  if (!output.parent_path().empty()) std::filesystem::create_directories(output.parent_path());
  std::ofstream out(output);
  if (!out) throw std::runtime_error("cannot write search results");
  out << std::setprecision(10)
      << "method,m,b,k,nprobes,shortlist,num_rerank,queries,timed_queries,recall_k,recall_10,"
         "candidates_mean,prototype_cmps_mean,route_tq4_ms,route_rerank_tq8_ms,"
         "score_tq1_ms,final_rerank_tq8_ms,total_ms,total_ms_p50,total_ms_p95,total_ms_p99,qps\n";
  for (auto nprobes : probes) for (auto rerank : reranks) {
    params.nprobes = nprobes;
    params.num_rerank = rerank;
    std::vector<double> recall(queries.size()), candidates(queries.size()), comparisons(queries.size());
    parlay::parallel_for(0, queries.size(), [&](size_t qi) {
      const auto result = index.Search(queries[qi], params);
      std::unordered_set<uint32_t> ids;
      for (auto neighbor : result.neighbors) ids.insert(neighbor.document);
      for (auto id : gt[qi]) recall[qi] += ids.count(id) / double(params.k);
      candidates[qi] = result.candidates_scored;
      comparisons[qi] = result.prototype_comparisons;
    }, 1);
    std::vector<double> route(timed), center(timed), score(timed), final(timed), total(timed);
    parlay::execute_with_scheduler(1, [&] {
      for (size_t qi = 0; qi < timed; ++qi) {
        const auto result = index.Search(queries[qi], params);
        route[qi] = result.routing_ms;
        center[qi] = result.prototype_rerank_ms;
        score[qi] = result.scoring_ms;
        final[qi] = result.final_rerank_ms;
        total[qi] = result.total_ms;
      }
    });
    out << "EMDIVF," << m << ',' << (params.bucket_size ? params.bucket_size : stored_b)
        << ',' << params.k << ',' << nprobes << ',' << params.shortlist << ',' << rerank
        << ',' << queries.size() << ',' << timed << ',' << Mean(recall) << ',';
    if (params.k == 10) out << Mean(recall);
    out << ',' << Mean(candidates) << ','
        << Mean(comparisons) << ',' << Mean(route) << ',' << Mean(center) << ','
        << Mean(score) << ',' << Mean(final) << ',' << Mean(total) << ','
        << Quantile(total, .5) << ',' << Quantile(total, .95) << ',' << Quantile(total, .99)
        << ',' << 1000.0 / Mean(total) << '\n';
    out.flush();
    if (!out) throw std::runtime_error("failed to write search results");
    std::cerr << "probes=" << nprobes << " rerank=" << rerank
              << " recall=" << Mean(recall) << " ms=" << Mean(total) << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) throw std::invalid_argument("commands: graph, search; options are --name value");
    Args args;
    for (int i = 2; i < argc; i += 2) {
      if (i + 1 >= argc || std::string(argv[i]).rfind("--", 0) != 0)
        throw std::invalid_argument("expected --name value pairs");
      args[std::string(argv[i]).substr(2)] = argv[i + 1];
    }
    const std::string command = argv[1];
    if (command == "graph") {
      emdivf::CloudSet train(Get(args, "train").c_str(), true);
      const auto index = emdivf::Load(Get(args, "index"));
      emdivf::RoutingBuildParams params;
      params.degree = Unsigned(Get(args, "R", "32"));
      params.beam = Unsigned(Get(args, "L-build", "200"));
      params.alpha = std::stod(Get(args, "alpha", "1.2"));
      emdivf::BuildRoutingGraph(train, index, Get(args, "graph"), params);
    } else if (command == "search") Search(args);
    else throw std::invalid_argument("unknown command: " + command);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
