// Upstream hnswlib multi-vector baseline runner.
//
// Build: flatten document tokens into an HNSW graph (one node per token, doc_id
//        embedded in each point). Search: per query-token graph probes, union
//        candidate documents, exact asymmetric chamfer-IP rerank.
//
// Data layout under --hnswlib-data (from benchmarks/hnswlib/hnswlib_preprocess.py):
//   base_vecs.npy, base_docids.npy, doc_offsets.npy,
//   q_vecs.npy, q_lens.npy, .READY
//
// CSV columns (search):
//   k, recall_1_k, recall_k_k, QPS_seq, QPS_par, avg_cmps,
//   t_graph_search, t_rerank, ef_search, rerank_k

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <omp.h>

#include "cnpy.h"
#include "hnswlib/hnswlib.h"
#include "hnswlib/stop_condition.h"

namespace {

using docidtype = uint32_t;

struct Params {
  std::string mode;
  std::string hnswlib_data;
  std::string index_dir;
  std::string gt_path;
  std::string output_csv = "-";
  std::string metric = "ip";

  int dim = 0;
  int m_index = 16;
  int ef_construction = 200;
  int k = 10;
  std::vector<int> ef_list;
  std::vector<int> rerank_list;
  int threads = 1;
  double qps_stop_below = 0.0;
  int warmup = 10;
  std::vector<int> query_indices;
};

struct QueryView {
  const float* data = nullptr;
  int vecnum = 0;
};

struct DataBundle {
  int dim = 0;
  int n_doc = 0;
  std::vector<float> base_vecs;
  std::vector<docidtype> base_docids;
  std::vector<uint64_t> doc_offsets;
  std::vector<float> q_vecs;
  std::vector<int64_t> q_lens;
  std::vector<QueryView> queries;
};

struct GTRow {
  std::vector<uint32_t> ids;
};

void die(const std::string& msg) {
  std::cerr << "[hnswlib_runner] error: " << msg << std::endl;
  std::exit(2);
}

std::vector<int> parse_int_list(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) continue;
    out.push_back(std::stoi(item));
  }
  return out;
}

void print_help() {
  std::cerr <<
      "Usage: hnswlib_runner --mode <build|search> --hnswlib-data <DIR> [opts]\n"
      "\n"
      "Common:\n"
      "  --hnswlib-data DIR  preprocessed tree (benchmarks/hnswlib/hnswlib_preprocess.py)\n"
      "  --dim N             vector dimension\n"
      "  --metric ip|l2      distance for graph edges (default ip)\n"
      "  --m-index 16        HNSW M\n"
      "  --ef-construction 200\n"
      "  --threads N         OpenMP threads (default 1)\n"
      "\n"
      "build:\n"
      "  --mode build --out <INDEX_DIR>\n"
      "\n"
      "search:\n"
      "  --mode search --index <INDEX_DIR>\n"
      "  --gt <GT_FILE>\n"
      "  --k N\n"
      "  --ef-list 40,80\n"
      "  --rerank-list 32,64\n"
      "  --output-csv -|FILE\n"
      "  --qps-stop-below FLOAT\n"
      "  --warmup N\n"
      "  --query-indices 0,3,7   evaluate only these query row indices\n";
}

Params parse_args(int argc, char** argv) {
  Params p;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto need_arg = [&](int j) {
      if (j >= argc) die("missing value for " + a);
      return std::string(argv[j]);
    };
    if (a == "--help" || a == "-h") {
      print_help();
      std::exit(0);
    } else if (a == "--mode") {
      p.mode = need_arg(++i);
    } else if (a == "--hnswlib-data") {
      p.hnswlib_data = need_arg(++i);
    } else if (a == "--out" || a == "--index") {
      p.index_dir = need_arg(++i);
    } else if (a == "--gt") {
      p.gt_path = need_arg(++i);
    } else if (a == "--output-csv") {
      p.output_csv = need_arg(++i);
    } else if (a == "--dim") {
      p.dim = std::stoi(need_arg(++i));
    } else if (a == "--metric") {
      p.metric = need_arg(++i);
    } else if (a == "--m-index") {
      p.m_index = std::stoi(need_arg(++i));
    } else if (a == "--ef-construction") {
      p.ef_construction = std::stoi(need_arg(++i));
    } else if (a == "--threads") {
      p.threads = std::stoi(need_arg(++i));
    } else if (a == "--k") {
      p.k = std::stoi(need_arg(++i));
    } else if (a == "--ef-list") {
      p.ef_list = parse_int_list(need_arg(++i));
    } else if (a == "--rerank-list") {
      p.rerank_list = parse_int_list(need_arg(++i));
    } else if (a == "--qps-stop-below") {
      p.qps_stop_below = std::stod(need_arg(++i));
    } else if (a == "--warmup") {
      p.warmup = std::stoi(need_arg(++i));
    } else if (a == "--query-indices") {
      p.query_indices = parse_int_list(need_arg(++i));
    } else {
      die("unknown flag: " + a);
    }
  }
  if (p.mode != "build" && p.mode != "search") die("--mode must be build or search");
  if (p.hnswlib_data.empty()) die("--hnswlib-data is required");
  if (p.index_dir.empty()) die("--out / --index is required");
  if (p.mode == "search") {
    if (p.gt_path.empty()) die("--gt is required in search mode");
    if (p.ef_list.empty()) die("--ef-list is required in search mode");
    if (p.rerank_list.empty()) die("--rerank-list is required in search mode");
  }
  return p;
}

template <typename T>
static std::vector<T> load_npy_1d(const std::string& path) {
  cnpy::NpyArray a = cnpy::npy_load(path);
  if (a.shape.size() != 1) die(path + " must be 1D");
  std::vector<T> out(a.shape[0]);
  if (a.word_size == sizeof(T)) {
    std::memcpy(out.data(), a.data<T>(), out.size() * sizeof(T));
  } else {
    die(path + ": unexpected dtype size");
  }
  return out;
}

template <typename T>
static void load_npy_2d(const std::string& path, std::vector<T>& out, int& rows, int& cols) {
  cnpy::NpyArray a = cnpy::npy_load(path);
  if (a.shape.size() != 2) die(path + " must be 2D");
  rows = (int)a.shape[0];
  cols = (int)a.shape[1];
  out.resize((size_t)rows * cols);
  if (a.word_size == sizeof(T)) {
    std::memcpy(out.data(), a.data<T>(), out.size() * sizeof(T));
  } else {
    die(path + ": unexpected dtype size");
  }
}

void load_hnswlib_data(const Params& p, DataBundle& d) {
  const std::string root = p.hnswlib_data;
  int rows = 0, cols = 0;
  load_npy_2d<float>(root + "/base_vecs.npy", d.base_vecs, rows, cols);
  if (p.dim > 0 && cols != p.dim) die("base_vecs.npy dim mismatch");
  d.dim = cols;
  d.base_docids = load_npy_1d<docidtype>(root + "/base_docids.npy");
  if (d.base_docids.size() != (size_t)rows) die("base_docids length mismatch");

  {
    cnpy::NpyArray off = cnpy::npy_load(root + "/doc_offsets.npy");
    if (off.shape.size() != 1) die("doc_offsets.npy must be 1D");
    d.n_doc = (int)off.shape[0] - 1;
    d.doc_offsets.resize(off.shape[0]);
    if (off.word_size == 8) {
      uint64_t* raw = off.data<uint64_t>();
      for (size_t i = 0; i < off.shape[0]; i++) d.doc_offsets[i] = raw[i];
    } else {
      die("doc_offsets.npy must be uint64");
    }
  }

  int q_tok_rows = 0;
  int q_cols = 0;
  load_npy_2d<float>(root + "/q_vecs.npy", d.q_vecs, q_tok_rows, q_cols);
  if (q_cols != d.dim) die("q_vecs.npy dim mismatch");
  d.q_lens = load_npy_1d<int64_t>(root + "/q_lens.npy");
  (void)q_tok_rows;

  size_t off = 0;
  d.queries.reserve(d.q_lens.size());
  for (size_t i = 0; i < d.q_lens.size(); i++) {
    int vn = (int)d.q_lens[i];
    d.queries.push_back({d.q_vecs.data() + off, vn});
    off += (size_t)vn * d.dim;
  }
}

void apply_query_indices(DataBundle& d, const std::vector<int>& indices) {
  if (indices.empty()) return;
  std::vector<QueryView> filtered;
  filtered.reserve(indices.size());
  for (int idx : indices) {
    if (idx < 0 || idx >= (int)d.queries.size()) {
      die("query index out of range: " + std::to_string(idx));
    }
    filtered.push_back(d.queries[idx]);
  }
  d.queries = std::move(filtered);
}

std::vector<GTRow> load_gt_subset(const std::string& path,
                                  const std::vector<int>& indices) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("cannot open gt file: " + path);
  int32_t nn = 0;
  f.read(reinterpret_cast<char*>(&nn), sizeof(int32_t));
  const size_t rec_size =
      (size_t)nn * (sizeof(float) + sizeof(uint32_t));
  std::vector<GTRow> rows;
  rows.reserve(indices.size());
  for (int idx : indices) {
    if (idx < 0) die("negative gt query index");
    f.seekg(4 + (size_t)idx * rec_size, std::ios::beg);
    GTRow row;
    row.ids.resize(nn);
    for (int j = 0; j < nn; j++) {
      float dist;
      uint32_t id;
      f.read(reinterpret_cast<char*>(&dist), sizeof(float));
      f.read(reinterpret_cast<char*>(&id), sizeof(uint32_t));
      row.ids[j] = id;
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

std::vector<GTRow> load_gt(const std::string& path, size_t n_query) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("cannot open gt file: " + path);
  int32_t nn = 0;
  f.read(reinterpret_cast<char*>(&nn), sizeof(int32_t));
  std::vector<GTRow> rows(n_query);
  for (size_t i = 0; i < n_query; i++) {
    rows[i].ids.resize(nn);
    for (int j = 0; j < nn; j++) {
      float dist;
      uint32_t id;
      f.read(reinterpret_cast<char*>(&dist), sizeof(float));
      f.read(reinterpret_cast<char*>(&id), sizeof(uint32_t));
      rows[i].ids[j] = id;
    }
  }
  return rows;
}

std::pair<double, double> compute_recall_for_query(
    const std::vector<std::pair<int, float>>& pred,
    const std::vector<uint32_t>& gt_ids,
    int k) {
  if (gt_ids.empty()) return {0.0, 0.0};
  int kk = std::min((int)gt_ids.size(), k);
  std::unordered_set<uint32_t> gt_topk(gt_ids.begin(), gt_ids.begin() + kk);
  double r1 = 0.0;
  if (!pred.empty() && gt_topk.count((uint32_t)pred[0].first)) r1 = 1.0;
  int hits = 0;
  int pk = std::min((int)pred.size(), k);
  for (int i = 0; i < pk; i++) {
    if (gt_topk.count((uint32_t)pred[i].first)) hits++;
  }
  double rk = (double)hits / (double)kk;
  return {r1, rk};
}

float chamfer_ip_score(const float* q_data, int q_n, const float* d_data, int d_n, int dim) {
  if (q_n == 0 || d_n == 0) return 1e9f;
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A(
      q_data, q_n, dim);
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B(
      d_data, d_n, dim);
  Eigen::MatrixXf C = A * B.transpose();
  float sim = C.rowwise().maxCoeff().mean();
  return 1.0f - sim;
}

struct IndexHandle {
  std::unique_ptr<hnswlib::SpaceInterface<float>> space;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> alg;
  size_t point_size = 0;
  std::vector<docidtype> label_to_docid;
};

IndexHandle build_index(const Params& p, const DataBundle& d) {
  IndexHandle h;
  const size_t n_tokens = d.base_vecs.size() / (size_t)d.dim;
  if (n_tokens == 0) die("empty base index");

  if (p.metric == "l2") {
    h.space = std::make_unique<hnswlib::MultiVectorL2Space<docidtype>>(d.dim);
  } else {
    h.space = std::make_unique<hnswlib::MultiVectorInnerProductSpace<docidtype>>(d.dim);
  }
  h.point_size = h.space->get_data_size();
  h.alg = std::make_unique<hnswlib::HierarchicalNSW<float>>(
      h.space.get(), n_tokens, p.m_index, p.ef_construction);

  std::vector<char> point(h.point_size);
  h.label_to_docid.resize(n_tokens);
  for (size_t i = 0; i < n_tokens; i++) {
    std::memcpy(point.data(), d.base_vecs.data() + i * d.dim, d.dim * sizeof(float));
    auto* mv = dynamic_cast<hnswlib::BaseMultiVectorSpace<docidtype>*>(h.space.get());
    if (!mv) die("unexpected space type");
    mv->set_doc_id(point.data(), d.base_docids[i]);
    h.label_to_docid[i] = d.base_docids[i];
    h.alg->addPoint(point.data(), (hnswlib::labeltype)i);
  }

  return h;
}

void save_index(IndexHandle& h, const std::string& path) {
  h.alg->saveIndex(path);
}

void load_index(IndexHandle& h, const Params& p, const DataBundle& d, const std::string& path) {
  const size_t n_tokens = d.base_vecs.size() / (size_t)d.dim;
  if (p.metric == "l2") {
    h.space = std::make_unique<hnswlib::MultiVectorL2Space<docidtype>>(d.dim);
  } else {
    h.space = std::make_unique<hnswlib::MultiVectorInnerProductSpace<docidtype>>(d.dim);
  }
  h.point_size = h.space->get_data_size();
  h.alg = std::make_unique<hnswlib::HierarchicalNSW<float>>(
      h.space.get(), n_tokens, p.m_index, p.ef_construction);
  h.alg->loadIndex(path, h.space.get(), n_tokens);
  h.label_to_docid = d.base_docids;
}

struct Timings {
  double total = 0;
  double graph = 0;
  double rerank = 0;
};

Timings search_one(const Params& p,
                   IndexHandle& h,
                   const DataBundle& d,
                   const QueryView& query,
                   int k,
                   int ef_search,
                   int rerank_k,
                   std::vector<std::pair<int, float>>& res) {
  res.clear();
  Timings t;
  auto* mv = dynamic_cast<hnswlib::BaseMultiVectorSpace<docidtype>*>(h.space.get());
  if (!mv) die("unexpected space type");

  const double t0 = omp_get_wtime();
  std::unordered_set<docidtype> candidates;
  std::vector<char> qbuf(h.point_size);
  const size_t ef_collect = std::max((size_t)ef_search, (size_t)k);

  for (int ti = 0; ti < query.vecnum; ti++) {
    std::memcpy(qbuf.data(), query.data + (size_t)ti * d.dim, d.dim * sizeof(float));
    hnswlib::MultiVectorSearchStopCondition<docidtype, float> stop(
        *mv, (size_t)std::min(rerank_k, (int)ef_collect), ef_collect);
    auto hits = h.alg->searchStopConditionClosest(qbuf.data(), stop);
    for (const auto& pr : hits) {
      hnswlib::labeltype label = pr.second;
      if (label >= h.label_to_docid.size()) continue;
      candidates.insert(h.label_to_docid[label]);
      if ((int)candidates.size() >= rerank_k) break;
    }
    if ((int)candidates.size() >= rerank_k) break;
  }
  const double t_graph_end = omp_get_wtime();
  t.graph = t_graph_end - t0;

  std::vector<std::pair<float, docidtype>> scored;
  scored.reserve(candidates.size());
  for (docidtype doc_id : candidates) {
    if (doc_id >= (docidtype)d.n_doc) continue;
    const uint64_t b = d.doc_offsets[doc_id];
    const uint64_t e = d.doc_offsets[doc_id + 1];
    const int d_n = (int)(e - b);
    const float* d_ptr = d.base_vecs.data() + b * d.dim;
    float score = chamfer_ip_score(query.data, query.vecnum, d_ptr, d_n, d.dim);
    scored.emplace_back(score, doc_id);
  }
  std::partial_sort(
      scored.begin(),
      scored.begin() + std::min((int)scored.size(), k),
      scored.end(),
      [](const auto& a, const auto& b) { return a.first < b.first; });
  int out_n = std::min((int)scored.size(), k);
  res.resize(out_n);
  for (int i = 0; i < out_n; i++) {
    res[i] = {(int)scored[i].second, scored[i].first};
  }
  const double t_end = omp_get_wtime();
  t.rerank = t_end - t_graph_end;
  t.total = t_end - t0;
  return t;
}

void write_meta_json(const std::string& dir, const Params& p, const DataBundle& d,
                     double build_time_sec) {
  std::ofstream f(dir + "/hnswlib_meta.json");
  f << "{\n"
    << "  \"dim\": " << d.dim << ",\n"
    << "  \"metric\": \"" << p.metric << "\",\n"
    << "  \"m_index\": " << p.m_index << ",\n"
    << "  \"ef_construction\": " << p.ef_construction << ",\n"
    << "  \"n_doc\": " << d.n_doc << ",\n"
    << "  \"n_tokens\": " << (d.base_vecs.size() / (size_t)d.dim) << ",\n"
    << "  \"build_time_sec\": " << build_time_sec << "\n"
    << "}\n";
}

std::ostream* open_csv(const Params& p, std::ofstream& file_holder) {
  if (p.output_csv == "-") return &std::cout;
  file_holder.open(p.output_csv);
  if (!file_holder) die("cannot open " + p.output_csv + " for writing");
  return &file_holder;
}

void emit_csv_header(std::ostream& os) {
  os << "k,recall_1_k,recall_k_k,QPS_seq,QPS_par,avg_cmps,"
        "t_graph_search,t_rerank,ef_search,rerank_k"
     << std::endl;
}

void emit_csv_row(std::ostream& os, int k, double r1, double rk, double qps_seq, double qps_par,
                  double t_graph, double t_rerank, int ef, int rrk) {
  os << k << "," << r1 << "," << rk << "," << qps_seq << "," << qps_par << ",0,"
     << t_graph << "," << t_rerank << "," << ef << "," << rrk << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  Params p = parse_args(argc, argv);
  omp_set_num_threads(std::max(1, p.threads));

  DataBundle d;
  load_hnswlib_data(p, d);
  if (p.dim <= 0) p.dim = d.dim;

  if (p.mode == "build") {
    auto t0 = omp_get_wtime();
    IndexHandle idx = build_index(p, d);
    const std::string index_path = p.index_dir + "/index.bin";
    save_index(idx, index_path);
    const double build_time = omp_get_wtime() - t0;
    write_meta_json(p.index_dir, p, d, build_time);
    std::cerr << "[hnswlib_runner] build done in " << build_time << "s -> " << index_path
              << std::endl;
    return 0;
  }

  apply_query_indices(d, p.query_indices);

  IndexHandle idx;
  load_index(idx, p, d, p.index_dir + "/index.bin");
  std::vector<GTRow> gt =
      p.query_indices.empty()
          ? load_gt(p.gt_path, d.queries.size())
          : load_gt_subset(p.gt_path, p.query_indices);

  std::ofstream file_holder;
  std::ostream* os = open_csv(p, file_holder);
  emit_csv_header(*os);

  if (p.warmup > 0 && !d.queries.empty()) {
    int warm_ef = *std::min_element(p.ef_list.begin(), p.ef_list.end());
    int warm_rr = *std::max_element(p.rerank_list.begin(), p.rerank_list.end());
    std::vector<std::pair<int, float>> res;
    int n = std::min(p.warmup, (int)d.queries.size());
    for (int i = 0; i < n; i++) {
      search_one(p, idx, d, d.queries[i], p.k, warm_ef, warm_rr, res);
    }
  }

  for (int rrk : p.rerank_list) {
    bool stop_chain = false;
    for (int ef : p.ef_list) {
      if (stop_chain) break;
      double sum_total = 0, sum_graph = 0, sum_rerank = 0;
      double sum_r1 = 0, sum_rk = 0;
      auto wall_t0 = std::chrono::steady_clock::now();
      std::vector<std::pair<int, float>> res;
      for (size_t qi = 0; qi < d.queries.size(); qi++) {
        auto t = search_one(p, idx, d, d.queries[qi], p.k, ef, rrk, res);
        sum_total += t.total;
        sum_graph += t.graph;
        sum_rerank += t.rerank;
        auto rec = compute_recall_for_query(res, gt[qi].ids, p.k);
        sum_r1 += rec.first;
        sum_rk += rec.second;
      }
      double wall = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - wall_t0)
                        .count();
      int nq = (int)d.queries.size();
      double r1 = sum_r1 / nq;
      double rk = sum_rk / nq;
      double qps_seq = sum_total > 0 ? (double)nq / sum_total : 0.0;
      double qps_par = wall > 0 ? (double)nq / wall : 0.0;
      double qps_report_seq = (p.threads == 1) ? qps_seq : 0.0;
      double qps_report_par = (p.threads == 1) ? qps_seq : qps_par;

      emit_csv_row(*os, p.k, r1, rk, qps_report_seq, qps_report_par,
                   sum_graph / nq, sum_rerank / nq, ef, rrk);
      os->flush();

      std::cerr << "[hnswlib_runner] ef=" << ef << " rerank=" << rrk << " R@" << p.k << "=" << rk
                << " QPS=" << qps_report_par << std::endl;

      if (p.qps_stop_below > 0 && qps_report_par > 0 && qps_report_par < p.qps_stop_below) {
        std::cerr << "[hnswlib_runner] qps " << qps_report_par << " < --qps-stop-below "
                  << p.qps_stop_below << "; halting rerank_k=" << rrk << " chain" << std::endl;
        stop_chain = true;
      }
    }
  }

  return 0;
}
