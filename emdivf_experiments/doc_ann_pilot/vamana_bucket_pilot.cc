// Isolated diagnostic: use an existing document Vamana graph to propose
// EMDIVF prototype buckets. Compare its evaluated/visited document pool
// against saved exact Chamfer top-B buckets, without changing either index.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "mvsic/algorithms/emdivf/index.h"
#include "mvsic/core/search_params.h"
#include "mvsic/algorithms/vamana/beam_search.h"
#include "mvsic/algorithms/vamana/graph.h"

namespace {

using Clock = std::chrono::steady_clock;
using Graph = mvsic::vamana::Graph<uint32_t>;

struct Args {
  std::string docs;
  std::string train;
  std::string emd_index;
  std::string graph;
  std::string output;
  std::vector<uint32_t> beams;
  uint32_t centers = 16;
  uint32_t b = 512;
};

std::vector<uint32_t> ParseBeams(const std::string& text) {
  std::vector<uint32_t> result;
  size_t pos = 0;
  while (pos < text.size()) {
    auto end = text.find(',', pos);
    auto token = text.substr(pos, end == std::string::npos ? end : end - pos);
    result.push_back(static_cast<uint32_t>(std::stoul(token)));
    if (end == std::string::npos) break;
    pos = end + 1;
  }
  if (result.empty()) throw std::invalid_argument("empty beam list");
  return result;
}

Args Parse(int argc, char** argv) {
  Args a;
  a.beams = {512, 1024, 2048, 4096};
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 >= argc) throw std::invalid_argument("argument needs value");
    std::string k = argv[i], v = argv[i + 1];
    if (k == "--docs") a.docs = v;
    else if (k == "--train") a.train = v;
    else if (k == "--emd-index") a.emd_index = v;
    else if (k == "--graph") a.graph = v;
    else if (k == "--output") a.output = v;
    else if (k == "--beams") a.beams = ParseBeams(v);
    else if (k == "--centers") a.centers = static_cast<uint32_t>(std::stoul(v));
    else if (k == "--b") a.b = static_cast<uint32_t>(std::stoul(v));
    else throw std::invalid_argument("unknown argument: " + k);
  }
  if (a.docs.empty() || a.train.empty() || a.emd_index.empty() || a.graph.empty() ||
      a.output.empty() || !a.centers || !a.b)
    throw std::invalid_argument("need --docs --train --emd-index --graph --output --centers --b");
  return a;
}

Graph LoadGraph(const std::string& path, size_t expected) {
  std::ifstream in(path, std::ios::binary);
  uint32_t magic = 0, version = 0, class_id = 0, start = 0;
  in.read(reinterpret_cast<char*>(&magic), 4);
  in.read(reinterpret_cast<char*>(&version), 4);
  in.read(reinterpret_cast<char*>(&class_id), 4);
  in.read(reinterpret_cast<char*>(&start), 4);
  if (!in || magic != 0x56414D41u || version != 3 || class_id != 0 || start != 0)
    throw std::runtime_error("invalid MV-Vamana v3 skeleton");
  Graph graph(in);
  if (graph.size() != expected) throw std::runtime_error("graph/document count mismatch");
  return graph;
}

double Ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    auto args = Parse(argc, argv);
    emdivf::CloudSet docs(args.docs.c_str(), true);
    emdivf::CloudSet train(args.train.c_str(), true);
    auto index = emdivf::Load(args.emd_index);
    if (docs.size() != index.corpus_size || train.size() != index.train_size ||
        docs.get_dims() != train.get_dims() || args.b > index.max_bucket_size ||
        args.centers > index.centers.size())
      throw std::runtime_error("source/index dimensions mismatch");
    auto graph = LoadGraph(args.graph, docs.size());
    std::ofstream out(args.output);
    if (!out) throw std::runtime_error("cannot open output CSV");
    out << "center_rank,center_train_row,beam,b,visited,exact_chamfer_cmps,"
           "search_ms,topb_found_in_visited,topb_found_fraction,"
           "topb_exact_rerank_overlap,topb_exact_rerank_fraction\n";
    out << std::setprecision(10);
    for (uint32_t ci = 0; ci < args.centers; ++ci) {
      // Cover early and later k-center stages; same centers at every beam.
      uint32_t rank = args.centers == 1 ? 0 :
          ci * static_cast<uint32_t>(index.centers.size() - 1) / (args.centers - 1);
      uint32_t train_row = index.centers[rank];
      auto query = train[train_row];
      std::vector<uint8_t> exact(docs.size(), 0);
      for (uint32_t j = 0; j < args.b; ++j) exact[index.buckets[rank][j].document] = 1;
      for (uint32_t beam : args.beams) {
        auto params = mvsic::SearchParams::vamana(0, beam, 0.0);
        auto start = Clock::now();
        auto [found, cmps, bytes] = mvsic::vamana::beam_search(
            query, graph, docs, uint32_t{0}, params);
        auto ms = Ms(start);
        auto& visited = found.second;
        size_t hits = 0, rerank_hits = 0;
        std::vector<std::pair<uint32_t, float>> reranked(visited.begin(), visited.end());
        std::sort(reranked.begin(), reranked.end(), [](auto a, auto b) {
          return std::tie(a.second, a.first) < std::tie(b.second, b.first);
        });
        for (const auto& item : visited) hits += exact[item.first] != 0;
        for (size_t j = 0; j < std::min<size_t>(args.b, reranked.size()); ++j)
          rerank_hits += exact[reranked[j].first] != 0;
        out << rank << ',' << train_row << ',' << beam << ',' << args.b << ','
            << visited.size() << ',' << cmps << ',' << ms << ',' << hits << ','
            << double(hits) / args.b << ',' << rerank_hits << ','
            << double(rerank_hits) / args.b << '\n';
        std::cerr << "center " << ci + 1 << '/' << args.centers << " beam " << beam
                  << " cmps=" << cmps << " topB=" << rerank_hits << '/' << args.b
                  << " ms=" << ms << '\n';
      }
      out.flush();
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
