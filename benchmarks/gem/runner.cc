// GEM (sigmod26gem) CLI runner.
//
// Bridges the gap between MVSIC's benchmark harness (which speaks .pcs +
// .gt files and dispatches via Python subprocess) and upstream GEM's
// hnswlib-with-extensions C++ library.
//
// Two modes:
//
//   --mode build  : read the gem_data/<ds>/ tree (preproduced by
//                   benchmarks/gem/gem_preprocess.py), construct the GEM fine /
//                   coarse / cluster_set indices, and write the resulting
//                   hnswlib graph to <index_dir>/0.bin plus a tiny
//                   gem_meta.json next to it.
//
//   --mode search : reload the index, optionally repair it (the upstream
//                   load path needs `repair_fine_graph_structure` to
//                   rebuild cluster_entries before the first search), and
//                   sweep the cartesian product of --ef-list x
//                   --rerank-list, emitting one CSV row per combo to
//                   stdout (or --output-csv FILE).  Per-combo qps_thresh
//                   early termination supported via --qps-stop-below.
//
// Upstream-equivalence notes:
//
//   * `Solution` here is a near-verbatim lift of upstream's
//     `example_vecset_search_gem.cpp::Solution` (build/search/load/save/
//     repair).  The only changes are: (a) hardcoded constants
//     (NUM_GRAPH_CLUSTER, NUM_CLUSTER, NPROB, M_index, EF_index,
//     VECTOR_DIM, rerankK, CPU_num) become instance members populated
//     from Params, and (b) we replaced upstream's verbose per-cluster
//     progress prints with milestone prints to avoid flooding stdout
//     when the runner is plumbed through a Python subprocess that
//     line-parses stdout.
//
//   * The data layout under --gem-data MUST match upstream's evqa /
//     msmarco loader expectations:
//
//       cdata/centroids.npy           : float16 [k1, dim]
//       cdata/coarse_centroids.npy    : float32 [k2, dim]
//       cdata/coarse_cluster_info.txt : one space-separated list of
//                                       doc_ids per line, one line per
//                                       coarse cluster
//       docdata/encoding<i>_float16.npy : float16 [shard_vec_count, dim]
//       docdata/doc_codes_<i>.npy       : int32  [shard_vec_count]
//       docdata/doclens<i>.npy          : int64  [shard_doc_count]
//       qdata/qembs.npy                 : float32[n_q, q_max, dim]
//                                         OR a flat float32[total_q_vec, dim]
//                                         combined with qdata/qlens.npy
//                                         (int64) when query lengths vary.
//
// CSV columns emitted in search mode:
//
//   k, recall_1_k, recall_k_k, QPS_seq, QPS_par,
//   avg_cmps, t_filter, t_graph_search, t_rerank,
//   ef_search, rerank_k, nprobe

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cblas.h>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <omp.h>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "cnpy.h"
#include "hnswlib/hnswlib.h"
#include "hnswlib/space_l2.h"
#include "hnswlib/vectorset.h"

// ---------------------------------------------------------------------------
// CLI

struct Params {
  std::string mode;             // "build" | "search"
  std::string gem_data;         // gem_data/<ds>/ root
  std::string index_dir;        // where to write (build) / read (search) 0.bin
  std::string gt_path;          // MVSIC <ds>_chamfer_neighbors.gt (search)
  std::string output_csv = "-"; // "-" for stdout, otherwise file path

  int dim = 128;
  int m_index = 24;
  int ef_construction = 80;
  int num_fine_clusters = 0;    // upstream NUM_CLUSTER (k1); 0 = read from cdata
  int num_coarse_clusters = 0;  // upstream NUM_GRAPH_CLUSTER (k2); 0 = read
  int num_doc_shards = 0;       // number of encoding<i>_float16.npy shards
  int nprobe_t = 4;             // upstream NPROB (top-t cluster filter)

  int k = 10;
  std::vector<int> ef_list;
  std::vector<int> rerank_list;
  int threads = 1;
  double qps_stop_below = 0.0;  // 0 = disabled
  bool repair_after_load = true;
  int warmup = 10;
  std::vector<int> query_indices;  // empty = evaluate all queries
};

static std::vector<int> parse_int_list(const std::string& s) {
  std::vector<int> out;
  std::string item;
  std::stringstream ss(s);
  while (std::getline(ss, item, ',')) {
    if (item.empty()) continue;
    out.push_back(std::stoi(item));
  }
  return out;
}

static void die(const std::string& msg) {
  std::cerr << "[gem_runner] error: " << msg << std::endl;
  std::exit(2);
}

static void print_help() {
  std::cerr <<
    "Usage: gem_runner --mode <build|search> --gem-data <DIR> [opts]\n"
    "\n"
    "Common options:\n"
    "  --gem-data DIR              gem_data/<ds>/ tree (cdata/, docdata/, qdata/)\n"
    "  --num-doc-shards N          number of docdata/encoding<i>_float16.npy shards\n"
    "  --dim N                     vector dim (default 128)\n"
    "  --num-fine-clusters K1      override k1 (default: shape of centroids.npy)\n"
    "  --num-coarse-clusters K2    override k2 (default: shape of coarse_centroids.npy)\n"
    "  --m-index 24                hnswlib M (graph degree)\n"
    "  --ef-construction 80        hnswlib ef_construction\n"
    "  --nprobe-t 4                per-token top-t cluster filter\n"
    "  --threads N                 OpenMP / BLAS threads (default 1)\n"
    "\n"
    "build mode:\n"
    "  --mode build --out <INDEX_DIR>\n"
    "\n"
    "search mode:\n"
    "  --mode search --index <INDEX_DIR>\n"
    "  --gt <GT_FILE>              MVSIC ground-truth (.gt) file\n"
    "  --k N                       top-k (default 10)\n"
    "  --ef-list 100,200,400       sweep these ef_search values\n"
    "  --rerank-list 128,256,512   sweep these rerank_k values\n"
    "  --output-csv -|FILE         where to emit CSV rows (default -, stdout)\n"
    "  --qps-stop-below FLOAT      bail after a combo's QPS falls below this\n"
    "  --no-repair                 skip repair_fine_graph_structure on load\n"
    "  --warmup N                  warm-up queries (default 10)\n"
    "  --query-indices 0,3,7       evaluate only these query row indices\n";
}

static Params parse_args(int argc, char** argv) {
  Params p;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto need_arg = [&](int j) {
      if (j >= argc) die("missing value for " + a);
      return std::string(argv[j]);
    };
    if (a == "--help" || a == "-h") { print_help(); std::exit(0); }
    else if (a == "--mode") p.mode = need_arg(++i);
    else if (a == "--gem-data") p.gem_data = need_arg(++i);
    else if (a == "--out" || a == "--index") p.index_dir = need_arg(++i);
    else if (a == "--gt") p.gt_path = need_arg(++i);
    else if (a == "--output-csv") p.output_csv = need_arg(++i);
    else if (a == "--dim") p.dim = std::stoi(need_arg(++i));
    else if (a == "--num-fine-clusters") p.num_fine_clusters = std::stoi(need_arg(++i));
    else if (a == "--num-coarse-clusters") p.num_coarse_clusters = std::stoi(need_arg(++i));
    else if (a == "--num-doc-shards") p.num_doc_shards = std::stoi(need_arg(++i));
    else if (a == "--m-index") p.m_index = std::stoi(need_arg(++i));
    else if (a == "--ef-construction") p.ef_construction = std::stoi(need_arg(++i));
    else if (a == "--nprobe-t") p.nprobe_t = std::stoi(need_arg(++i));
    else if (a == "--threads") p.threads = std::stoi(need_arg(++i));
    else if (a == "--k") p.k = std::stoi(need_arg(++i));
    else if (a == "--ef-list") p.ef_list = parse_int_list(need_arg(++i));
    else if (a == "--rerank-list") p.rerank_list = parse_int_list(need_arg(++i));
    else if (a == "--qps-stop-below") p.qps_stop_below = std::stod(need_arg(++i));
    else if (a == "--no-repair") p.repair_after_load = false;
    else if (a == "--warmup") p.warmup = std::stoi(need_arg(++i));
    else if (a == "--query-indices") p.query_indices = parse_int_list(need_arg(++i));
    else die("unknown flag: " + a);
  }
  if (p.mode != "build" && p.mode != "search") die("--mode must be build or search");
  if (p.gem_data.empty()) die("--gem-data is required");
  if (p.index_dir.empty()) die("--out / --index is required");
  if (p.mode == "search") {
    if (p.gt_path.empty()) die("--gt is required in search mode");
    if (p.ef_list.empty()) die("--ef-list is required in search mode");
    if (p.rerank_list.empty()) die("--rerank-list is required in search mode");
  }
  return p;
}

// ---------------------------------------------------------------------------
// IEEE 754 half-precision -> float32. (Lifted from upstream, used to widen
// FP16 fine-centroid + FP16 base-token data into float32 working buffers.)
static inline float half_to_float(uint16_t h) {
  int s = (h >> 15) & 0x1;
  int e = (h >> 10) & 0x1F;
  int f = h & 0x3FF;
  if (e == 0) return (s ? -1 : 1) * std::ldexp(f, -24);
  if (e == 31) return (s ? -1 : 1) * (f ? NAN : INFINITY);
  return (s ? -1 : 1) * std::ldexp(f + 1024, e - 15 - 10);
}

// ---------------------------------------------------------------------------
// gem_data loader. Mirrors upstream's load_from_evqa / load_from_msmarco
// closely; the runner accepts both fixed-length queries (qembs.npy with
// 3 dims) and variable-length queries (flat qembs.npy + qlens.npy).
struct DataBundle {
  int dim = 0;
  // Base.
  std::vector<float> base_data;        // [total_base_vec * dim], float32
  std::vector<int> base_data_codes;    // [total_base_vec], int (centroid id)
  std::vector<vectorset> base;         // n_doc vectorsets pointing into base_data
  // Centroids.
  int k1 = 0, k2 = 0;
  std::vector<float> center_data;      // [k1 * dim] float32 (from centroids.npy fp16)
  std::vector<float> graph_center_data;// [k2 * dim] float32 (from coarse_centroids.npy fp32)
  // Coarse cluster -> doc ids.
  std::vector<std::vector<hnswlib::labeltype>> cluster_set; // size k2
  // Queries.
  std::vector<float> query_data;       // flat float32
  std::vector<vectorset> query;        // n_query vectorsets
};

static void load_centroids(const Params& p, DataBundle& d) {
  // Fine centroids: centroids.npy is float16, shape (k1, dim).
  {
    cnpy::NpyArray a = cnpy::npy_load(p.gem_data + "/cdata/centroids.npy");
    if (a.shape.size() != 2) die("centroids.npy must be 2D");
    d.k1 = (int)a.shape[0];
    if ((int)a.shape[1] != p.dim) die("centroids.npy dim mismatch");
    if (p.num_fine_clusters > 0 && p.num_fine_clusters != d.k1)
      die("--num-fine-clusters disagrees with centroids.npy");
    d.center_data.resize((size_t)d.k1 * p.dim);
    if (a.word_size == 2) {
      uint16_t* raw = a.data<uint16_t>();
      for (size_t i = 0; i < d.center_data.size(); i++)
        d.center_data[i] = half_to_float(raw[i]);
    } else if (a.word_size == 4) {
      float* raw = a.data<float>();
      std::memcpy(d.center_data.data(), raw, d.center_data.size() * sizeof(float));
    } else {
      die("centroids.npy must be float16 or float32");
    }
  }
  // Coarse centroids: coarse_centroids.npy is float32, shape (k2, dim).
  {
    cnpy::NpyArray a = cnpy::npy_load(p.gem_data + "/cdata/coarse_centroids.npy");
    if (a.shape.size() != 2) die("coarse_centroids.npy must be 2D");
    d.k2 = (int)a.shape[0];
    if ((int)a.shape[1] != p.dim) die("coarse_centroids.npy dim mismatch");
    if (p.num_coarse_clusters > 0 && p.num_coarse_clusters != d.k2)
      die("--num-coarse-clusters disagrees with coarse_centroids.npy");
    d.graph_center_data.resize((size_t)d.k2 * p.dim);
    if (a.word_size == 4) {
      float* raw = a.data<float>();
      std::memcpy(d.graph_center_data.data(), raw, d.graph_center_data.size() * sizeof(float));
    } else if (a.word_size == 2) {
      uint16_t* raw = a.data<uint16_t>();
      for (size_t i = 0; i < d.graph_center_data.size(); i++)
        d.graph_center_data[i] = half_to_float(raw[i]);
    } else {
      die("coarse_centroids.npy must be float32 or float16");
    }
  }
}

static void load_cluster_info(const Params& p, DataBundle& d) {
  std::string path = p.gem_data + "/cdata/coarse_cluster_info.txt";
  std::ifstream f(path);
  if (!f) die("cannot open " + path);
  d.cluster_set.assign(d.k2, {});
  std::string line;
  int line_id = 0;
  while (std::getline(f, line)) {
    if (line_id >= d.k2) {
      std::cerr << "[gem_runner] warn: coarse_cluster_info.txt has more lines than k2=" << d.k2 << std::endl;
      break;
    }
    std::istringstream iss(line);
    hnswlib::labeltype id;
    while (iss >> id) d.cluster_set[line_id].push_back(id);
    line_id++;
  }
}

static void load_base_shards(const Params& p, DataBundle& d) {
  // Two pass: first pass counts total vectors and total docs so we can
  // resize() once; second pass fills.
  size_t total_vec = 0, total_doc = 0;
  for (int i = 0; i < p.num_doc_shards; i++) {
    std::string emb_path = p.gem_data + "/docdata/encoding" + std::to_string(i) + "_float16.npy";
    cnpy::NpyArray a = cnpy::npy_load(emb_path);
    if (a.shape.size() != 2) die(emb_path + ": expected 2D");
    if ((int)a.shape[1] != p.dim) die(emb_path + ": dim mismatch");
    total_vec += a.shape[0];
    std::string len_path = p.gem_data + "/docdata/doclens" + std::to_string(i) + ".npy";
    cnpy::NpyArray l = cnpy::npy_load(len_path);
    total_doc += l.shape[0];
  }

  d.base_data.resize(total_vec * (size_t)p.dim);
  d.base_data_codes.resize(total_vec);
  d.base.reserve(total_doc);

  size_t vec_off = 0;  // offset into base_data in floats
  size_t code_off = 0; // offset into base_data_codes in ints
  for (int i = 0; i < p.num_doc_shards; i++) {
    std::string emb_path = p.gem_data + "/docdata/encoding" + std::to_string(i) + "_float16.npy";
    std::string code_path = p.gem_data + "/docdata/doc_codes_" + std::to_string(i) + ".npy";
    std::string len_path = p.gem_data + "/docdata/doclens" + std::to_string(i) + ".npy";

    cnpy::NpyArray emb = cnpy::npy_load(emb_path);
    cnpy::NpyArray cod = cnpy::npy_load(code_path);
    cnpy::NpyArray len = cnpy::npy_load(len_path);

    size_t shard_vec = emb.shape[0];
    size_t shard_doc = len.shape[0];

    if (emb.word_size == 2) {
      uint16_t* raw = emb.data<uint16_t>();
      for (size_t j = 0; j < shard_vec * (size_t)p.dim; j++)
        d.base_data[vec_off + j] = half_to_float(raw[j]);
    } else if (emb.word_size == 4) {
      std::memcpy(d.base_data.data() + vec_off,
                  emb.data<float>(),
                  shard_vec * p.dim * sizeof(float));
    } else die(emb_path + ": expected float16 or float32");

    if (cod.word_size == 4) {
      int32_t* raw = cod.data<int32_t>();
      for (size_t j = 0; j < shard_vec; j++)
        d.base_data_codes[code_off + j] = (int)raw[j];
    } else die(code_path + ": expected int32");

    // doclens: upstream's loader reads as complex<int> and takes .real(),
    // i.e. low 32 bits of an int64. We accept either int32 or int64.
    std::vector<int> shard_lens(shard_doc);
    if (len.word_size == 4) {
      int32_t* raw = len.data<int32_t>();
      for (size_t j = 0; j < shard_doc; j++) shard_lens[j] = (int)raw[j];
    } else if (len.word_size == 8) {
      int64_t* raw = len.data<int64_t>();
      for (size_t j = 0; j < shard_doc; j++) shard_lens[j] = (int)raw[j];
    } else die(len_path + ": expected int32 or int64");

    size_t doc_vec_off = vec_off;
    size_t doc_code_off = code_off;
    for (size_t j = 0; j < shard_doc; j++) {
      int dl = shard_lens[j];
      d.base.emplace_back(d.base_data.data() + doc_vec_off,
                          d.base_data_codes.data() + doc_code_off,
                          p.dim, dl);
      doc_vec_off += (size_t)dl * p.dim;
      doc_code_off += (size_t)dl;
    }

    vec_off += shard_vec * (size_t)p.dim;
    code_off += shard_vec;
  }
}

static void load_queries(const Params& p, DataBundle& d) {
  std::string emb_path = p.gem_data + "/qdata/qembs.npy";
  std::string len_path = p.gem_data + "/qdata/qlens.npy";

  cnpy::NpyArray emb = cnpy::npy_load(emb_path);
  bool have_lens = std::filesystem::exists(len_path);

  if (emb.shape.size() == 3) {
    // [n_q, q_max, dim] -- fixed-length queries (most common).
    int n_q = (int)emb.shape[0];
    int q_max = (int)emb.shape[1];
    if ((int)emb.shape[2] != p.dim) die("qembs.npy dim mismatch");
    size_t total = (size_t)n_q * q_max * p.dim;
    d.query_data.resize(total);
    if (emb.word_size == 4) {
      std::memcpy(d.query_data.data(), emb.data<float>(), total * sizeof(float));
    } else if (emb.word_size == 2) {
      uint16_t* raw = emb.data<uint16_t>();
      for (size_t j = 0; j < total; j++) d.query_data[j] = half_to_float(raw[j]);
    } else die("qembs.npy must be float32 or float16");

    std::vector<int> qlens;
    if (have_lens) {
      cnpy::NpyArray len = cnpy::npy_load(len_path);
      qlens.resize(len.shape[0]);
      if (len.word_size == 4) {
        int32_t* raw = len.data<int32_t>();
        for (size_t j = 0; j < qlens.size(); j++) qlens[j] = (int)raw[j];
      } else if (len.word_size == 8) {
        int64_t* raw = len.data<int64_t>();
        for (size_t j = 0; j < qlens.size(); j++) qlens[j] = (int)raw[j];
      } else die(len_path + ": expected int32 or int64");
      if ((int)qlens.size() != n_q) die("qlens.npy size != n_queries");
    }

    size_t off = 0;
    for (int j = 0; j < n_q; j++) {
      int vn = have_lens ? qlens[j] : q_max;
      d.query.emplace_back(d.query_data.data() + off, nullptr, p.dim, vn);
      off += (size_t)q_max * p.dim;
    }
  } else if (emb.shape.size() == 2) {
    // Flat: [total_q_vec, dim] -- variable-length, requires qlens.npy.
    if (!have_lens) die("qembs.npy is 2D but qlens.npy is missing");
    if ((int)emb.shape[1] != p.dim) die("qembs.npy dim mismatch");
    size_t total = (size_t)emb.shape[0] * p.dim;
    d.query_data.resize(total);
    if (emb.word_size == 4) {
      std::memcpy(d.query_data.data(), emb.data<float>(), total * sizeof(float));
    } else if (emb.word_size == 2) {
      uint16_t* raw = emb.data<uint16_t>();
      for (size_t j = 0; j < total; j++) d.query_data[j] = half_to_float(raw[j]);
    } else die("qembs.npy must be float32 or float16");

    cnpy::NpyArray len = cnpy::npy_load(len_path);
    std::vector<int> qlens(len.shape[0]);
    if (len.word_size == 4) {
      int32_t* raw = len.data<int32_t>();
      for (size_t j = 0; j < qlens.size(); j++) qlens[j] = (int)raw[j];
    } else if (len.word_size == 8) {
      int64_t* raw = len.data<int64_t>();
      for (size_t j = 0; j < qlens.size(); j++) qlens[j] = (int)raw[j];
    } else die(len_path + ": expected int32 or int64");

    size_t off = 0;
    for (size_t j = 0; j < qlens.size(); j++) {
      d.query.emplace_back(d.query_data.data() + off, nullptr, p.dim, qlens[j]);
      off += (size_t)qlens[j] * p.dim;
    }
  } else {
    die("qembs.npy must be 2D or 3D");
  }
}

static void apply_query_indices(DataBundle& d, const std::vector<int>& indices) {
  if (indices.empty()) return;
  std::vector<vectorset> filtered;
  filtered.reserve(indices.size());
  for (int idx : indices) {
    if (idx < 0 || idx >= (int)d.query.size()) {
      die("query index out of range: " + std::to_string(idx));
    }
    filtered.push_back(d.query[idx]);
  }
  d.query = std::move(filtered);
}

// ---------------------------------------------------------------------------
// MVSIC .gt file: int32 num_neighbors, then per-query num_neighbors
// repetitions of (float32 dist, uint32 doc_id). We only need doc_ids.
struct GTRow { std::vector<uint32_t> ids; };
static std::vector<GTRow> load_gt_subset(const std::string& path,
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
static std::vector<GTRow> load_gt(const std::string& path, size_t n_query) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("cannot open gt file: " + path);
  int32_t nn = 0;
  f.read(reinterpret_cast<char*>(&nn), sizeof(int32_t));
  std::vector<GTRow> rows(n_query);
  for (size_t i = 0; i < n_query; i++) {
    rows[i].ids.resize(nn);
    for (int j = 0; j < nn; j++) {
      float dist; uint32_t id;
      f.read(reinterpret_cast<char*>(&dist), sizeof(float));
      f.read(reinterpret_cast<char*>(&id), sizeof(uint32_t));
      rows[i].ids[j] = id;
    }
  }
  return rows;
}

// recall_1_k: |{top-1 pred} intersect top-k gt|  -- always 0 or 1 per query.
// recall_k_k: |{top-k pred} intersect top-k gt| / k
// (mirrors mvsic::compute_scores semantics.)
static std::pair<double, double> compute_recall_for_query(
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

// ---------------------------------------------------------------------------
// Helper carried verbatim from upstream (top-NPROB cluster IDs per query).
static void get_unique_top_k_indices_col(const std::vector<float>& matrix,
                                         int rows, int cols, int topk,
                                         std::unordered_set<int>& unique_indices) {
  std::vector<int> all_scores((size_t)rows * topk);
  #pragma omp parallel for
  for (int row = 0; row < rows; ++row) {
    std::vector<std::pair<float, int>> scores(cols);
    for (int col = 0; col < cols; ++col) {
      scores[col] = {matrix[(size_t)row * cols + col], col};
    }
    std::partial_sort(scores.begin(), scores.begin() + topk, scores.end(),
                      [](const std::pair<float, int>& a,
                         const std::pair<float, int>& b) {
                        return a.first > b.first;
                      });
    for (int i = 0; i < topk; ++i) {
      all_scores[(size_t)row * topk + i] = scores[i].second;
    }
  }
  for (int row = 0; row < rows; ++row) {
    for (int i = 0; i < topk; ++i) {
      unique_indices.insert(all_scores[(size_t)row * topk + i]);
    }
  }
}

// ---------------------------------------------------------------------------
// Solution: GEM index + sweep driver. Lift of upstream's
// example_vecset_search_gem.cpp::Solution with global constants made
// instance fields and the BLAS-heavy 1-CPU rerank path kept (it matches
// the upstream taskset -c 0 mode the README recommends).
class Solution {
 public:
  Solution(const Params& p, DataBundle& d)
      : params_(p), data_(d), rerank_k_(p.rerank_list.empty() ? 0 : p.rerank_list.back()) {}

  void build() {
    double t0 = omp_get_wtime();
    space_ptr_ = new hnswlib::L2VSSpace(params_.dim);
    cluster_entries_.assign(data_.k2, -1);
    alg_ = new hnswlib::HierarchicalNSW<float>(space_ptr_, data_.base.size() + 1,
                                               params_.m_index, params_.ef_construction);

    // cluster_distance[i,j] = inner product between fine centroid i and j,
    // but with the upstream-hardcoded row stride NUM_CLUSTER_CALC (262144),
    // because `L2SqrVecClusterEMD` (used during build) addresses it as
    // `cluster_dis[q->codes[i] * NUM_CLUSTER_CALC + p->codes[j]]`. We only
    // populate the first k1 columns of each row; the rest stays zero and
    // is never read because doc-token codes are in [0, k1).
    //
    // Memory: k1 * NUM_CLUSTER_CALC * 4 bytes (e.g. k1=1024 -> 1 GB,
    // k1=4096 -> 4 GB). NUM_CLUSTER_CALC=262144 is upstream's hard cap on
    // the fine cluster count; bail loudly if the preprocessor chose a
    // larger k1 since the build would otherwise silently read garbage.
    if ((long long)data_.k1 > (long long)NUM_CLUSTER_CALC) {
      std::cerr << "[gem_runner] fatal: k1=" << data_.k1
                << " exceeds upstream NUM_CLUSTER_CALC=" << NUM_CLUSTER_CALC
                << "; reduce --num-fine-clusters (preprocess) or recompile "
                   "sigmod26gem with a larger NUM_CLUSTER_CALC." << std::endl;
      std::exit(3);
    }
    const long long stride = NUM_CLUSTER_CALC;
    std::vector<float> cluster_distance((long long)data_.k1 * stride, 0.0f);
    // ldc = NUM_CLUSTER_CALC so the BLAS output lands at the strided rows
    // upstream's distance kernel expects, without needing a post-copy.
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                data_.k1, data_.k1, params_.dim,
                1.0f,
                data_.center_data.data(), params_.dim,
                data_.center_data.data(), params_.dim,
                0.0f,
                cluster_distance.data(), (int)stride);

    for (int i = 0; i < (int)data_.cluster_set.size(); i++) {
      const auto& doc_ids = data_.cluster_set[i];
      if (doc_ids.empty()) continue;
      double ct = omp_get_wtime();
      alg_->entry_map.clear();
      cluster_entries_[i] = doc_ids[0];
      alg_->addClusterPointEntry(&data_.base[doc_ids[0]],
                                 cluster_distance.data(),
                                 doc_ids[0], doc_ids[0]);
      if (doc_ids.size() > 1) {
        #pragma omp parallel for schedule(dynamic)
        for (int j = 1; j < (int)doc_ids.size(); j++) {
          alg_->addClusterPointEntry(&data_.base[doc_ids[j]],
                                     cluster_distance.data(),
                                     doc_ids[j], doc_ids[0]);
        }
      }
      // Print at most ~50 progress lines per build regardless of cluster count.
      int log_every = std::max(1, (int)data_.cluster_set.size() / 50);
      if (i % log_every == 0 || i == (int)data_.cluster_set.size() - 1) {
        std::cerr << "[gem_runner] cluster " << i << "/" << data_.cluster_set.size()
                  << " (size=" << doc_ids.size() << ", "
                  << (omp_get_wtime() - ct) << "s)" << std::endl;
      }
    }
    std::cerr << "[gem_runner] build time: " << (omp_get_wtime() - t0) << "s" << std::endl;
  }

  void save(const std::string& dir) {
    std::filesystem::create_directories(dir);
    std::string path = dir + "/0.bin";
    alg_->saveIndex(path);
    std::cerr << "[gem_runner] saved " << path << std::endl;
  }

  void load(const std::string& dir) {
    double t0 = omp_get_wtime();
    self_cluster_set_ = data_.cluster_set;
    space_ptr_ = new hnswlib::L2VSSpace(params_.dim);
    cluster_entries_.assign(data_.k2, -1);
    alg_ = new hnswlib::HierarchicalNSW<float>(space_ptr_, data_.base.size() + 1,
                                               params_.m_index, params_.ef_construction);
    std::string path = dir + "/0.bin";
    alg_->loadIndex(path, space_ptr_);
    for (int i = 0; i < (int)data_.cluster_set.size(); i++) {
      const auto& doc_ids = data_.cluster_set[i];
      if (doc_ids.empty()) continue;
      cluster_entries_[i] = doc_ids[0];
      for (auto id : doc_ids) {
        alg_->loadDataAddress(&data_.base[id], id);
      }
    }
    alg_->search_set.resize(alg_->max_elements_);
    std::cerr << "[gem_runner] loaded " << path << " in "
              << (omp_get_wtime() - t0) << "s" << std::endl;
  }

  void repair() {
    if (!params_.repair_after_load) return;
    double t0 = omp_get_wtime();
    for (int i = 0; i < (int)data_.cluster_set.size(); i++) {
      const auto& cs = data_.cluster_set[i];
      if (cs.empty()) continue;
      alg_->entry_map.clear();
      for (auto d : cs) alg_->entry_map[alg_->label_lookup_[d]] = i;
      std::vector<std::pair<hnswlib::tableint, int>> search_result =
          alg_->searchNodesForFix(cs[0], i);
      int l = 0;
      for (int j = 1; j < (int)cs.size(); j++) {
        int dn = alg_->label_lookup_[cs[j]];
        if (alg_->entry_map[dn] != i + 1) {
          while (l < (int)search_result.size()) {
            if (alg_->canAddEdgeinter(search_result[l].first)) {
              alg_->mutuallyConnectTwoInterElement(search_result[l].first, dn);
              if (alg_->canAddEdgeinter(dn)) {
                alg_->mutuallyConnectTwoInterElement(dn, search_result[l].first);
              }
              break;
            } else { l++; }
          }
          auto more = alg_->searchNodesForFix(cs[j], i);
          for (auto& x : more) search_result.push_back(x);
        }
      }
      if (i % 10000 == 0) std::cerr << "[gem_runner] repaired " << i << std::endl;
    }
    std::cerr << "[gem_runner] repair time: " << (omp_get_wtime() - t0) << "s" << std::endl;
  }

  // search a single query under (ef, rerank_k); writes top-k results into res,
  // returns per-query (total_time, filter_time, graph_time, rerank_time).
  struct Timings { double total, filter, graph, rerank; };
  Timings search_one(const vectorset& query,
                     std::vector<float>& q_cluster_scores,
                     std::vector<float>& col_q_scores,
                     int k, int ef,
                     std::vector<std::pair<int, float>>& res) {
    res.clear();
    res.resize(rerank_k_);
    alg_->search_set.assign(alg_->search_set.size(), 0);

    double t0 = omp_get_wtime();
    hnswlib::fast_dot_product_blas(query.vecnum, params_.dim, data_.k2,
                                   query.data, data_.graph_center_data.data(),
                                   q_cluster_scores.data());
    std::unordered_set<int> unique_indices;
    get_unique_top_k_indices_col(q_cluster_scores, query.vecnum, data_.k2,
                                 params_.nprobe_t, unique_indices);
    hnswlib::fast_dot_product_blas(data_.k1, params_.dim, query.vecnum,
                                   data_.center_data.data(), query.data,
                                   col_q_scores.data());
    vectorset query_cluster(col_q_scores.data(), nullptr, data_.k1, query.vecnum);

    std::vector<hnswlib::labeltype> entry_points;
    for (int idx : unique_indices) {
      if (cluster_entries_[idx] != -1) {
        entry_points.push_back(cluster_entries_[idx]);
        for (int j : self_cluster_set_[idx]) {
          alg_->search_set[alg_->label_lookup_[j]] = true;
        }
      }
    }
    double t_filter_end = omp_get_wtime();

    alg_->setEf(ef);
    auto pq = alg_->searchKnnClusterEntries(&query_cluster, ef, entry_points);
    std::vector<std::pair<float, hnswlib::labeltype>> merge_result;
    while (!pq.empty()) { merge_result.push_back(pq.top()); pq.pop(); }
    double t_graph_end = omp_get_wtime();

    int numrerank = std::min((int)merge_result.size(), rerank_k_);
    std::partial_sort(
        merge_result.begin(), merge_result.begin() + numrerank, merge_result.end(),
        [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
          return a.first < b.first;
        });
    res.resize(numrerank);

    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        A_mat(query.data, query.vecnum, params_.dim);
    for (int i = 0; i < numrerank; i++) {
      hnswlib::labeltype ind = merge_result[i].second;
      Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
          B_mat(data_.base[ind].data, data_.base[ind].vecnum, params_.dim);
      Eigen::MatrixXf C = A_mat * B_mat.transpose();
      res[i] = std::make_pair((int)ind,
                              1.0f - C.rowwise().maxCoeff().sum() / query.vecnum);
    }
    std::sort(res.begin(), res.end(),
              [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                return a.second < b.second;
              });
    if ((int)res.size() > k) res.resize(k);
    double t_end = omp_get_wtime();
    return {t_end - t0, t_filter_end - t0, t_graph_end - t_filter_end, t_end - t_graph_end};
  }

  int& rerank_k() { return rerank_k_; }

 private:
  const Params& params_;
  DataBundle& data_;
  hnswlib::L2VSSpace* space_ptr_ = nullptr;
  hnswlib::HierarchicalNSW<float>* alg_ = nullptr;
  std::vector<int> cluster_entries_;
  std::vector<std::vector<hnswlib::labeltype>> self_cluster_set_;
  int rerank_k_;
};

// ---------------------------------------------------------------------------
// CSV emit helpers.
static std::ostream* open_csv(const Params& p, std::ofstream& file_holder) {
  if (p.output_csv == "-") return &std::cout;
  file_holder.open(p.output_csv);
  if (!file_holder) die("cannot open " + p.output_csv + " for writing");
  return &file_holder;
}

static void emit_csv_header(std::ostream& os) {
  os << "k,recall_1_k,recall_k_k,QPS_seq,QPS_par,avg_cmps,"
        "t_filter,t_graph_search,t_rerank,"
        "ef_search,rerank_k,nprobe"
     << std::endl;
}

static void emit_csv_row(std::ostream& os,
                         int k, double r1, double rk,
                         double qps_seq, double qps_par, double avg_cmps,
                         double t_filter, double t_graph, double t_rerank,
                         int ef, int rrk, int nprobe) {
  os << k << ","
     << r1 << ","
     << rk << ","
     << qps_seq << ","
     << qps_par << ","
     << avg_cmps << ","
     << t_filter << ","
     << t_graph << ","
     << t_rerank << ","
     << ef << ","
     << rrk << ","
     << nprobe
     << std::endl;
}

// ---------------------------------------------------------------------------
// Driver.

static void write_meta_json(const std::string& dir, const Params& p,
                            int k1, int k2, double build_time_sec) {
  std::ofstream f(dir + "/gem_meta.json");
  f << "{\n"
    << "  \"dim\": " << p.dim << ",\n"
    << "  \"m_index\": " << p.m_index << ",\n"
    << "  \"ef_construction\": " << p.ef_construction << ",\n"
    << "  \"num_fine_clusters\": " << k1 << ",\n"
    << "  \"num_coarse_clusters\": " << k2 << ",\n"
    << "  \"num_doc_shards\": " << p.num_doc_shards << ",\n"
    << "  \"nprobe_t_default\": " << p.nprobe_t << ",\n"
    << "  \"build_time_sec\": " << build_time_sec << "\n"
    << "}\n";
}

int main(int argc, char** argv) {
  Params p = parse_args(argc, argv);

  omp_set_num_threads(p.threads);
  openblas_set_num_threads(p.threads);
  Eigen::setNbThreads(p.threads);
  omp_set_nested(1);

  DataBundle d;
  d.dim = p.dim;
  load_centroids(p, d);
  load_cluster_info(p, d);
  load_base_shards(p, d);

  if (p.mode == "search") {
    load_queries(p, d);
    apply_query_indices(d, p.query_indices);
  }

  if (p.mode == "build") {
    Solution solution(p, d);
    double t0 = omp_get_wtime();
    solution.build();
    double build_time = omp_get_wtime() - t0;
    solution.save(p.index_dir);
    write_meta_json(p.index_dir, p, d.k1, d.k2, build_time);
    std::cerr << "[gem_runner] build done in " << build_time << "s" << std::endl;
    return 0;
  }

  // search mode.
  Solution solution(p, d);
  solution.load(p.index_dir);
  solution.repair();

  std::vector<GTRow> gt =
      p.query_indices.empty()
          ? load_gt(p.gt_path, d.query.size())
          : load_gt_subset(p.gt_path, p.query_indices);

  std::ofstream file_holder;
  std::ostream* os = open_csv(p, file_holder);
  emit_csv_header(*os);

  // Per-query scratch buffers.
  std::vector<float> q_cluster_scores;
  std::vector<float> col_q_scores;
  size_t q_max = 0;
  for (auto& q : d.query) q_max = std::max(q_max, q.vecnum);
  q_cluster_scores.resize(q_max * (size_t)d.k2);
  col_q_scores.resize((size_t)d.k1 * q_max);

  // Warm-up: smallest ef, largest rerank, a handful of queries. Touches the
  // BLAS / Eigen path so subsequent timings are steady-state.
  if (p.warmup > 0 && !d.query.empty()) {
    int warm_ef = *std::min_element(p.ef_list.begin(), p.ef_list.end());
    int warm_rr = *std::max_element(p.rerank_list.begin(), p.rerank_list.end());
    solution.rerank_k() = warm_rr;
    std::vector<std::pair<int, float>> res;
    int n = std::min(p.warmup, (int)d.query.size());
    for (int i = 0; i < n; i++) {
      solution.search_one(d.query[i], q_cluster_scores, col_q_scores,
                          p.k, warm_ef, res);
    }
  }

  // Outer sweep: rerank_k first (cheaper to vary), ef_search inner.
  // Early termination via --qps-stop-below applies per rerank_k chain.
  for (int rrk : p.rerank_list) {
    solution.rerank_k() = rrk;
    bool stop_chain = false;
    for (int ef : p.ef_list) {
      if (stop_chain) break;
      double sum_total = 0, sum_filter = 0, sum_graph = 0, sum_rerank = 0;
      double sum_r1 = 0, sum_rk = 0;
      auto wall_t0 = std::chrono::steady_clock::now();
      std::vector<std::pair<int, float>> res;
      for (size_t qi = 0; qi < d.query.size(); qi++) {
        auto t = solution.search_one(d.query[qi], q_cluster_scores, col_q_scores,
                                     p.k, ef, res);
        sum_total += t.total;
        sum_filter += t.filter;
        sum_graph += t.graph;
        sum_rerank += t.rerank;
        auto rec = compute_recall_for_query(res, gt[qi].ids, p.k);
        sum_r1 += rec.first;
        sum_rk += rec.second;
      }
      double wall = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - wall_t0).count();
      int nq = (int)d.query.size();
      double r1 = sum_r1 / nq;
      double rk = sum_rk / nq;
      double qps_seq = sum_total > 0 ? (double)nq / sum_total : 0.0;
      double qps_par = wall > 0 ? (double)nq / wall : 0.0;
      double t_filter = sum_filter / nq;
      double t_graph = sum_graph / nq;
      double t_rerank = sum_rerank / nq;

      // For latency runs (threads=1) QPS_seq IS the metric; the runner
      // mirrors the convention from FastPlaidWrapper / IGPWrapper where
      // QPS_par is reported equal to QPS_seq in the single-thread path.
      double qps_report_seq = (p.threads == 1) ? qps_seq : 0.0;
      double qps_report_par = (p.threads == 1) ? qps_seq : qps_par;

      emit_csv_row(*os, p.k, r1, rk, qps_report_seq, qps_report_par,
                   0.0,  // avg_cmps not tracked by GEM
                   t_filter, t_graph, t_rerank,
                   ef, rrk, p.nprobe_t);
      os->flush();

      std::cerr << "[gem_runner] ef=" << ef << " rerank=" << rrk
                << " R@" << p.k << "=" << rk << " QPS=" << qps_report_par
                << std::endl;

      if (p.qps_stop_below > 0 && qps_report_par > 0 &&
          qps_report_par < p.qps_stop_below) {
        std::cerr << "[gem_runner] qps " << qps_report_par
                  << " < --qps-stop-below " << p.qps_stop_below
                  << "; halting rerank_k=" << rrk << " chain" << std::endl;
        stop_chain = true;
      }
    }
  }

  return 0;
}
