// Isolated diagnostic: use a saved MVIVF document tree as a cheap candidate
// generator for EMDIVF prototype top-B buckets. No index is modified.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "mvsic/emdivf/index.h"
#include "mvsic/mvivf/mvivf.h"

namespace {

using Clock = std::chrono::steady_clock;
using MVIVF = mvsic::IndexMVIVF<false, true, mvsic::NoQuantizer<false>>;

struct Args {
  std::string docs, train, emd_index, mvivf_index, output;
  uint32_t centers = 32, b = 512;
  std::vector<uint32_t> probes = {8, 16, 32, 64, 128, 256, 512};
};

std::vector<uint32_t> ParseList(const std::string& text) {
  std::vector<uint32_t> out;
  size_t pos = 0;
  while (pos < text.size()) {
    auto end = text.find(',', pos);
    out.push_back(static_cast<uint32_t>(std::stoul(text.substr(pos, end - pos))));
    if (end == std::string::npos) break;
    pos = end + 1;
  }
  if (out.empty()) throw std::invalid_argument("empty probe list");
  return out;
}

Args Parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 >= argc) throw std::invalid_argument("argument needs value");
    std::string k = argv[i], v = argv[i + 1];
    if (k == "--docs") a.docs = v;
    else if (k == "--train") a.train = v;
    else if (k == "--emd-index") a.emd_index = v;
    else if (k == "--mvivf-index") a.mvivf_index = v;
    else if (k == "--output") a.output = v;
    else if (k == "--centers") a.centers = static_cast<uint32_t>(std::stoul(v));
    else if (k == "--b") a.b = static_cast<uint32_t>(std::stoul(v));
    else if (k == "--probes") a.probes = ParseList(v);
    else throw std::invalid_argument("unknown argument: " + k);
  }
  if (a.docs.empty() || a.train.empty() || a.emd_index.empty() ||
      a.mvivf_index.empty() || a.output.empty() || !a.centers || !a.b)
    throw std::invalid_argument("missing required path or invalid count");
  return a;
}

double Ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    auto a = Parse(argc, argv);
    emdivf::CloudSet docs(a.docs.c_str(), true);
    emdivf::CloudSet train(a.train.c_str(), true);
    auto reference = emdivf::Load(a.emd_index);
    if (docs.size() != reference.corpus_size || train.size() != reference.train_size ||
        docs.get_dims() != train.get_dims() || a.b > reference.max_bucket_size ||
        a.centers > reference.centers.size())
      throw std::runtime_error("source/index dimensions mismatch");
    MVIVF index(docs.get_dims());
    auto load_start = Clock::now();
    index.load(a.mvivf_index, docs);
    std::cerr << "MVIVF load and TQ4 center encode ms=" << Ms(load_start)
              << " leaves=" << index.leaves_flat.size() << '\n';
    std::ofstream out(a.output);
    if (!out) throw std::runtime_error("cannot open output CSV");
    out << "center_rank,center_train_row,nprobes,b,leaves,unique_candidates,"
           "center_cmps,route_ms,exact_candidate_score_ms,topb_found_in_candidates,"
           "topb_found_in_selected,topb_candidate_fraction,topb_selected_fraction\n";
    out << std::setprecision(10);
    std::vector<uint32_t> marks(docs.size(), 0);
    uint32_t generation = 0;
    for (uint32_t ci = 0; ci < a.centers; ++ci) {
      uint32_t rank = a.centers == 1 ? 0 :
          ci * static_cast<uint32_t>(reference.centers.size() - 1) / (a.centers - 1);
      auto train_row = reference.centers[rank];
      auto query = train[train_row];
      auto q_center = index.center_model_.quantize_query(query);
      std::vector<uint8_t> gold(docs.size(), 0);
      for (uint32_t j = 0; j < a.b; ++j) gold[reference.buckets[rank][j].document] = 1;
      for (uint32_t nprobes : a.probes) {
        auto route_start = Clock::now();
        auto selected = index.greedy_search(query, q_center, nprobes);
        auto route_ms = Ms(route_start);
        std::vector<uint32_t> ids;
        ++generation;
        for (const auto& [score, leaf] : selected.probe_list) {
          for (size_t j = 0; j < leaf->data.size(); ++j) {
            uint32_t doc = leaf->data.get_id(j);
            if (doc >= docs.size()) throw std::runtime_error("bad leaf document ID");
            if (marks[doc] != generation) {
              marks[doc] = generation;
              ids.push_back(doc);
            }
          }
        }
        auto score_start = Clock::now();
        uint32_t candidate_hits = 0;
        std::vector<std::pair<float, uint32_t>> scores;
        scores.reserve(ids.size());
        for (auto id : ids) {
          scores.push_back({query.distance(docs[id]), id});
          candidate_hits += gold[id] != 0;
        }
        const auto keep = std::min<size_t>(a.b, scores.size());
        if (scores.size() > keep) {
          std::nth_element(scores.begin(), scores.begin() + keep, scores.end());
          scores.resize(keep);
        }
        uint32_t selected_hits = 0;
        for (const auto& [score, id] : scores) selected_hits += gold[id] != 0;
        auto score_ms = Ms(score_start);
        out << rank << ',' << train_row << ',' << nprobes << ',' << a.b << ','
            << selected.probe_list.size() << ',' << ids.size() << ','
            << (selected.stats.empty() ? 0 : selected.stats[0]) << ','
            << route_ms << ',' << score_ms << ',' << candidate_hits << ','
            << selected_hits << ',' << double(candidate_hits) / a.b << ','
            << double(selected_hits) / a.b << '\n';
        std::cerr << "center " << ci + 1 << '/' << a.centers << " probes " << nprobes
                  << " candidates=" << ids.size() << " topB=" << selected_hits << '/' << a.b
                  << " ms=" << route_ms + score_ms << '\n';
      }
      out.flush();
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
