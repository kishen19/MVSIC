// Direct-MKL port of data-tools/compute_ground_truth.py.
//
// Algorithm (1:1 with the python tool):
//   * Database is mmap'd; scanned in chunks of `--chunk_clouds` clouds.
//   * For each chunk, the contiguous float32 block of vectors is copied into
//     D_chunk (M x d).
//   * Queries are loaded fully into RAM (`Q_stack`, shape (n_queries*Q, d))
//     and required to be uniform-length per cloud (Q vectors per cloud).
//   * For each query batch [qb_start, qb_end), R = B*Q rows:
//       IP (R x M) = cblas_sgemm(Q_batch, D_chunk^T)         <- MKL, all cores
//       per_cloud_max[r, c] = max IP[r, d_starts[c]..d_ends[c])  (parlay)
//       chunk_dists[b, c] = -mean_q per_cloud_max[b*Q + q, c]    (parlay)
//   * One top-k merge per chunk into the running (n_queries x k) tables.
//
// Output format (drop-in replacement for utils.ReadGT / stats::ReadGT):
//   int32 k
//   for each query i in [0, n_queries):
//     for each neighbor j in [0, k):
//       float32 distance
//       uint32  id
//
// Why a C++ tool when there's already a python one:  the python tool's GEMM
// is bottlenecked on the bundled scipy-openblas wheel, which is hard-capped
// at MAX_THREADS=64 — so on a 288-core box the CPU sits at ~22% peak.  Linking
// MKL directly fixes that; the segment-reduce/merge work moves from numpy's
// single-threaded reduceat to a parlay::parallel_for over rows.

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <mkl.h>

#include "parlay/parallel.h"
#include "parlay/primitives.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>


// ---------------------------------------------------------------------------
// .pcs format helpers
// ---------------------------------------------------------------------------

// .pcs layout (mvsic/core/types/point_cloud_set.h):
//   uint64 dims
//   uint64 n
//   uint64 num_vectors
//   float32[num_vectors * dims] values
//   uint64 num_offsets             (>= n + 1)
//   uint64[num_offsets] offsets    (cumulative *float* counts, i.e. vec_idx*dims)

struct PCSMmap {
  std::string path;
  int fd = -1;
  size_t file_size = 0;
  const std::byte* base = nullptr;

  uint64_t dims = 0;
  uint64_t n = 0;
  uint64_t num_vectors = 0;
  uint64_t num_offsets = 0;

  const float* values = nullptr;       // (num_vectors * dims) floats
  const uint64_t* raw_offsets = nullptr;  // num_offsets entries, in float-counts
  std::vector<int64_t> offs_vec;       // n+1 entries, in vec-index units

  ~PCSMmap() {
    if (base != nullptr) munmap(const_cast<std::byte*>(base), file_size);
    if (fd >= 0) close(fd);
  }
};

static void pcs_open_mmap(const std::string& path, PCSMmap& out) {
  out.path = path;
  out.fd = open(path.c_str(), O_RDONLY);
  if (out.fd < 0) {
    throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));
  }
  struct stat st{};
  if (fstat(out.fd, &st) != 0) {
    throw std::runtime_error("fstat failed on " + path);
  }
  out.file_size = static_cast<size_t>(st.st_size);
  if (out.file_size < 24) {
    throw std::runtime_error(path + ": file truncated before pcs header end");
  }
  void* m = mmap(nullptr, out.file_size, PROT_READ, MAP_SHARED, out.fd, 0);
  if (m == MAP_FAILED) {
    throw std::runtime_error("mmap failed on " + path + ": " + std::strerror(errno));
  }
  out.base = static_cast<const std::byte*>(m);

  // Pages we won't read for a long time (the offsets array, after we copy it
  // out) shouldn't fight the page cache with the main values stream.
  madvise(m, out.file_size, MADV_RANDOM);

  std::memcpy(&out.dims, out.base + 0, 8);
  std::memcpy(&out.n, out.base + 8, 8);
  std::memcpy(&out.num_vectors, out.base + 16, 8);

  const size_t values_off = 24;
  const size_t values_bytes = out.num_vectors * out.dims * sizeof(float);
  const size_t num_offsets_off = values_off + values_bytes;
  if (out.file_size < num_offsets_off + 8) {
    throw std::runtime_error(path + ": file truncated before num_offsets");
  }
  out.values = reinterpret_cast<const float*>(out.base + values_off);

  std::memcpy(&out.num_offsets, out.base + num_offsets_off, 8);
  if (out.num_offsets < out.n + 1) {
    throw std::runtime_error(path + ": num_offsets < n + 1");
  }
  const size_t offsets_off = num_offsets_off + 8;
  if (out.file_size < offsets_off + out.num_offsets * 8) {
    throw std::runtime_error(path + ": file truncated before offsets array");
  }
  out.raw_offsets = reinterpret_cast<const uint64_t*>(out.base + offsets_off);

  // Validate the first n+1 offsets and convert to vec-index units (matches
  // _read_offsets_vec in compute_ground_truth.py).
  out.offs_vec.assign(out.n + 1, 0);
  if (out.raw_offsets[0] != 0) {
    throw std::runtime_error(path + ": offsets[0] != 0");
  }
  const uint64_t expected_last = out.num_vectors * out.dims;
  if (out.raw_offsets[out.n] != expected_last) {
    throw std::runtime_error(path + ": offsets[n] != num_vectors*dims");
  }
  for (uint64_t i = 0; i <= out.n; ++i) {
    const uint64_t v = out.raw_offsets[i];
    if (v % out.dims != 0) {
      throw std::runtime_error(path + ": offset not a multiple of dims");
    }
    if (i > 0 && v < out.raw_offsets[i - 1]) {
      throw std::runtime_error(path + ": offsets not monotonic");
    }
    out.offs_vec[i] = static_cast<int64_t>(v / out.dims);
  }
}


// ---------------------------------------------------------------------------
// Small CLI parser (no external deps).
// ---------------------------------------------------------------------------

struct Args {
  std::string database;
  std::string queries;
  std::string output;
  int k = 2000;
  std::string dist_func = "IP";
  int chunk_clouds = 20000;
  int batch_queries = 64;
  int threads = 0;          // 0 = leave default (use whatever MKL/parlay decide)
  bool quiet = false;
  double progress_interval = 60.0;  // seconds; <= 0 disables
};

[[noreturn]] static void usage_and_exit(const char* prog) {
  std::cerr <<
    "Usage: " << prog << " -i <db.pcs> -q <queries.pcs> -o <gt.bin>\n"
    "       [-k <K=2000>] [-dist_func IP|L2]\n"
    "       [--chunk_clouds N=20000] [--batch_queries N=64]\n"
    "       [--threads N=0]  [--progress_interval SEC=60] [--quiet]\n";
  std::exit(2);
}

static Args parse_args(int argc, char** argv) {
  Args a;
  auto need_value = [&](int& i) -> const char* {
    if (i + 1 >= argc) usage_and_exit(argv[0]);
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    if (s == "-i" || s == "--database") a.database = need_value(i);
    else if (s == "-q" || s == "--queries") a.queries = need_value(i);
    else if (s == "-o" || s == "--output") a.output = need_value(i);
    else if (s == "-k") a.k = std::atoi(need_value(i));
    else if (s == "-dist_func" || s == "--dist_func") a.dist_func = need_value(i);
    else if (s == "--chunk_clouds") a.chunk_clouds = std::atoi(need_value(i));
    else if (s == "--batch_queries") a.batch_queries = std::atoi(need_value(i));
    else if (s == "--threads") a.threads = std::atoi(need_value(i));
    else if (s == "--quiet") a.quiet = true;
    else if (s == "--progress_interval") a.progress_interval = std::atof(need_value(i));
    else if (s == "-h" || s == "--help") usage_and_exit(argv[0]);
    else { std::cerr << "unknown arg: " << s << "\n"; usage_and_exit(argv[0]); }
  }
  if (a.database.empty() || a.queries.empty() || a.output.empty()) {
    usage_and_exit(argv[0]);
  }
  if (a.dist_func != "IP" && a.dist_func != "L2") {
    std::cerr << "dist_func must be IP or L2\n";
    std::exit(2);
  }
  if (a.k <= 0 || a.chunk_clouds <= 0 || a.batch_queries <= 0) {
    std::cerr << "k, chunk_clouds, batch_queries must be positive\n";
    std::exit(2);
  }
  return a;
}


// ---------------------------------------------------------------------------
// Progress reporting (matches the python tool's [progress ...] format).
// ---------------------------------------------------------------------------

static std::string fmt_dur(double secs) {
  if (secs < 0) secs = 0;
  long total = static_cast<long>(secs);
  long d = total / 86400;
  long h = (total / 3600) % 24;
  long m = (total / 60) % 60;
  long s = total % 60;
  char buf[64];
  if (d > 0) std::snprintf(buf, sizeof(buf), "%ldd%02ld:%02ld:%02ld", d, h, m, s);
  else       std::snprintf(buf, sizeof(buf), "%ld:%02ld:%02ld", h, m, s);
  return buf;
}

static std::string now_str() {
  std::time_t t = std::time(nullptr);
  std::tm tm{};
  localtime_r(&t, &tm);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
  return buf;
}

struct Timings {
  double gemm = 0, reduce = 0, merge = 0, io = 0;
};

static void emit_progress(int chunks_done, int n_chunks,
                          int64_t clouds_done, int64_t n_db,
                          double t_loop_start_sec,
                          const Timings& t,
                          const char* prefix = "progress") {
  using clk = std::chrono::steady_clock;
  double elapsed = std::chrono::duration<double>(
      clk::now().time_since_epoch()).count() - t_loop_start_sec;
  double pct = n_chunks ? 100.0 * chunks_done / n_chunks : 100.0;
  double rate_chunks = elapsed > 0 ? chunks_done / elapsed : 0.0;
  double rate_clouds = elapsed > 0 ? clouds_done / elapsed : 0.0;
  std::string eta = (chunks_done > 0 && chunks_done < n_chunks)
      ? fmt_dur(elapsed * (n_chunks - chunks_done) / chunks_done)
      : "0:00:00";
  std::cout << "[" << prefix << " " << now_str() << "] "
            << chunks_done << "/" << n_chunks << " chunks ("
            << std::fixed << std::setprecision(1) << pct << "%)  "
            << "clouds=" << clouds_done << "/" << n_db << "  "
            << "elapsed=" << fmt_dur(elapsed) << "  "
            << "eta=" << eta << "  "
            << "rate=" << std::setprecision(2) << rate_chunks << " chunks/s ("
            << std::setprecision(0) << rate_clouds << " clouds/s)\n";
  if (elapsed > 0) {
    auto pct_of = [&](double x) { return 100.0 * x / elapsed; };
    std::cout << "    timing: "
              << "gemm=" << std::setprecision(1) << t.gemm << "s ("
              << std::setprecision(0) << pct_of(t.gemm) << "%)  "
              << "reduce=" << std::setprecision(1) << t.reduce << "s ("
              << std::setprecision(0) << pct_of(t.reduce) << "%)  "
              << "merge=" << std::setprecision(1) << t.merge << "s ("
              << std::setprecision(0) << pct_of(t.merge) << "%)  "
              << "chunk_copy=" << std::setprecision(1) << t.io << "s ("
              << std::setprecision(0) << pct_of(t.io) << "%)\n";
  }
  std::cout.flush();
}

static double now_sec() {
  using clk = std::chrono::steady_clock;
  return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}


// ---------------------------------------------------------------------------
// Top-k merge — combines a (n_queries, C) chunk distance matrix with the
// running (n_queries, k) top-k tables.  Per-row independent → row-parallel.
//
// Each row's combined candidate set is k + C = the running top-k plus the
// fresh chunk; we use std::nth_element to find the k smallest, then do a
// single pass to gather the new (dist, id) pairs.  Same shape as the python
// `_merge_topk_chunk`'s argpartition path.
// ---------------------------------------------------------------------------

static void merge_topk_chunk(int n_queries, int k, int C, int64_t db_start,
                             const float* chunk_dists,
                             float* topk_dists, uint32_t* topk_ids) {
  parlay::parallel_for(0, n_queries, [&](size_t i) {
    const int total = k + C;
    // candidate index space: [0, k) -> existing topk slot, [k, k+C) -> fresh
    // chunk column (id = db_start + (j - k)).
    std::vector<int> idx(total);
    std::iota(idx.begin(), idx.end(), 0);
    auto dist_at = [&](int j) -> float {
      return j < k ? topk_dists[i * k + j] : chunk_dists[i * C + (j - k)];
    };
    if (C > 0) {
      std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                       [&](int a, int b) { return dist_at(a) < dist_at(b); });
    }
    // Gather the surviving k entries into local buffers, then copy back.
    std::vector<float> new_d(k);
    std::vector<uint32_t> new_id(k);
    for (int j = 0; j < k; ++j) {
      const int src = idx[j];
      new_d[j] = dist_at(src);
      new_id[j] = src < k
          ? topk_ids[i * k + src]
          : static_cast<uint32_t>(db_start + (src - k));
    }
    std::memcpy(&topk_dists[i * k], new_d.data(), k * sizeof(float));
    std::memcpy(&topk_ids[i * k], new_id.data(), k * sizeof(uint32_t));
  });
}


// ---------------------------------------------------------------------------
// Core compute.
// ---------------------------------------------------------------------------

static void compute_ground_truth(const Args& A) {
  const bool verbose = !A.quiet;
  const bool is_ip = (A.dist_func == "IP");

  // ---- Threads ----
  if (A.threads > 0) {
    mkl_set_num_threads(A.threads);
    if (verbose) std::cout << "[threads] mkl=" << A.threads
                           << "  parlay=" << parlay::num_workers() << "\n";
  } else if (verbose) {
    std::cout << "[threads] mkl=" << mkl_get_max_threads()
              << "  parlay=" << parlay::num_workers() << "\n";
  }

  // ---- Queries (small, fully resident) ----
  PCSMmap Q;
  pcs_open_mmap(A.queries, Q);
  const int64_t n_queries = static_cast<int64_t>(Q.n);
  const int64_t d = static_cast<int64_t>(Q.dims);

  // Detect uniform-vectors-per-cloud (msmarco/scifact/etc. ColBERT all hit
  // this path).  We only support uniform queries; the python tool falls back
  // to a slower per-query reduceat for the non-uniform case.
  const int64_t Q0 = Q.offs_vec[1] - Q.offs_vec[0];
  bool uniform = (Q0 > 0);
  for (int64_t i = 1; i <= n_queries && uniform; ++i) {
    if (Q.offs_vec[i] - Q.offs_vec[i - 1] != Q0) uniform = false;
  }
  if (!uniform) {
    throw std::runtime_error(
        "queries are not uniform per cloud — this C++ tool only supports the "
        "fast path.  Use compute_ground_truth.py for non-uniform queries.");
  }
  const int64_t Qper = Q0;
  const float invQ = 1.0f / static_cast<float>(Qper);

  // Copy the query values into a writable, page-locked-friendly buffer (the
  // mmap is read-only; we never mutate, but having our own owning storage
  // simplifies lifetime).
  std::vector<float> Qbuf(Q.num_vectors * d);
  std::memcpy(Qbuf.data(), Q.values, Qbuf.size() * sizeof(float));

  // L2: precompute per-query-vector squared norms (flat, size = n_queries*Qper).
  std::vector<float> q_norms_sq;
  if (!is_ip) {
    q_norms_sq.assign(static_cast<size_t>(n_queries) * Qper, 0.0f);
    parlay::parallel_for(0, q_norms_sq.size(), [&](size_t r) {
      const float* row = &Qbuf[r * d];
      float s = 0;
      for (int64_t t = 0; t < d; ++t) s += row[t] * row[t];
      q_norms_sq[r] = s;
    });
  }

  if (verbose) {
    std::cout << "[queries] " << n_queries << " clouds x " << Qper
              << " vectors (uniform), dim=" << d << "\n";
  }

  // ---- Database (mmap'd, walked in chunks) ----
  PCSMmap D;
  pcs_open_mmap(A.database, D);
  if (static_cast<int64_t>(D.dims) != d) {
    throw std::runtime_error("dim mismatch between queries and database");
  }
  const int64_t n_db = static_cast<int64_t>(D.n);
  const int chunk_clouds = A.chunk_clouds;
  const int batch_queries = A.batch_queries;
  const int n_db_chunks = static_cast<int>((n_db + chunk_clouds - 1) / chunk_clouds);

  if (verbose) {
    std::cout << "[database] " << n_db << " clouds, " << D.num_vectors
              << " vectors, dim=" << d << " (mmap)\n";
    std::cout << "[config] k=" << A.k << " dist=" << A.dist_func
              << " chunk_clouds=" << chunk_clouds
              << " batch_queries=" << batch_queries << "\n";
    std::cout.flush();
  }

  // We want page-cache to evict the chunk after we're done with it (the file
  // is way bigger than RAM and we won't revisit any chunk).  MADV_SEQUENTIAL
  // hints the kernel to do read-ahead and drop pages behind us.
  madvise(const_cast<std::byte*>(D.base), D.file_size, MADV_SEQUENTIAL);

  // ---- Top-k tables ----
  const int k = A.k;
  std::vector<float> topk_dists(static_cast<size_t>(n_queries) * k,
                                std::numeric_limits<float>::infinity());
  std::vector<uint32_t> topk_ids(static_cast<size_t>(n_queries) * k, 0u);

  // ---- Pre-allocate per-chunk scratch buffers ----
  // Walk the offsets once to find the largest chunk (in vectors).
  int64_t max_chunk_M = 0;
  for (int64_t s = 0; s < n_db; s += chunk_clouds) {
    int64_t e = std::min<int64_t>(s + chunk_clouds, n_db);
    int64_t M = D.offs_vec[e] - D.offs_vec[s];
    if (M > max_chunk_M) max_chunk_M = M;
  }
  const int64_t R_max = static_cast<int64_t>(batch_queries) * Qper;

  std::vector<float> D_chunk(static_cast<size_t>(max_chunk_M) * d);
  std::vector<float> IP_buf(static_cast<size_t>(R_max) * max_chunk_M);
  std::vector<float> per_cloud_red(static_cast<size_t>(R_max) * chunk_clouds);
  std::vector<float> chunk_dists(static_cast<size_t>(n_queries) * chunk_clouds);
  std::vector<float> d_norms_sq;
  if (!is_ip) d_norms_sq.assign(max_chunk_M, 0.0f);

  if (verbose) {
    auto gb = [](size_t bytes) { return bytes / (1024.0 * 1024.0 * 1024.0); };
    std::cout << "[memory] D_chunk=" << std::fixed << std::setprecision(2)
              << gb(D_chunk.size() * 4) << "G  IP=" << gb(IP_buf.size() * 4)
              << "G  per_cloud_red=" << gb(per_cloud_red.size() * 4)
              << "G  chunk_dists=" << gb(chunk_dists.size() * 4) << "G\n";
    std::cout.flush();
  }

  // ---- Main loop ----
  Timings T;
  const double t_loop_start = now_sec();
  double last_log = t_loop_start;
  int chunks_done = 0;
  const bool log_progress = verbose && A.progress_interval > 0;
  if (log_progress) {
    std::cout << "[progress " << now_str() << "] starting: 0/" << n_db_chunks
              << " chunks (0/" << n_db << " clouds)\n";
    std::cout.flush();
  }

  for (int64_t db_start = 0; db_start < n_db; db_start += chunk_clouds) {
    const int64_t db_end = std::min<int64_t>(db_start + chunk_clouds, n_db);
    const int64_t chunk_vec_start = D.offs_vec[db_start];
    const int64_t chunk_vec_end = D.offs_vec[db_end];
    const int64_t M = chunk_vec_end - chunk_vec_start;
    const int64_t C = db_end - db_start;

    if (M == 0) {
      ++chunks_done;
      if (log_progress && now_sec() - last_log >= A.progress_interval) {
        emit_progress(chunks_done, n_db_chunks, db_end, n_db, t_loop_start, T);
        last_log = now_sec();
      }
      continue;
    }

    // Materialize a contiguous f32 chunk of DB vectors.
    {
      const double t0 = now_sec();
      std::memcpy(D_chunk.data(), D.values + chunk_vec_start * d,
                  static_cast<size_t>(M) * d * sizeof(float));
      T.io += now_sec() - t0;
    }

    // Local per-cloud offsets within D_chunk (vector-index units, length C+1).
    std::vector<int64_t> d_offs_local(C + 1);
    for (int64_t c = 0; c <= C; ++c) {
      d_offs_local[c] = D.offs_vec[db_start + c] - chunk_vec_start;
    }

    if (!is_ip) {
      // d_norms_sq[v] = |D_chunk[v]|^2  (parallel over rows).
      parlay::parallel_for(0, M, [&](size_t v) {
        const float* row = &D_chunk[v * d];
        float s = 0;
        for (int64_t t = 0; t < d; ++t) s += row[t] * row[t];
        d_norms_sq[v] = s;
      });
    }

    // Iterate query batches over this DB chunk.  Each batch writes its
    // contribution into chunk_dists[qb_start:qb_end, :] directly.
    for (int64_t qb_start = 0; qb_start < n_queries; qb_start += batch_queries) {
      const int64_t qb_end = std::min<int64_t>(qb_start + batch_queries, n_queries);
      const int64_t B = qb_end - qb_start;
      const int64_t R = B * Qper;
      const float* Q_batch = &Qbuf[qb_start * Qper * d];

      // ---- The big GEMM: IP (R x M) = Q_batch (R x d) @ D_chunk (M x d)^T ----
      {
        const double t0 = now_sec();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    static_cast<MKL_INT>(R), static_cast<MKL_INT>(M),
                    static_cast<MKL_INT>(d),
                    1.0f,
                    Q_batch, static_cast<MKL_INT>(d),
                    D_chunk.data(), static_cast<MKL_INT>(d),
                    0.0f,
                    IP_buf.data(), static_cast<MKL_INT>(M));
        T.gemm += now_sec() - t0;
      }

      // ---- Per-cloud reduce + per-query sum ----
      const double t0 = now_sec();
      // For each row r in [0, R), and each cloud c in [0, C), reduce
      // IP[r, d_offs_local[c]..d_offs_local[c+1]) to a single scalar
      // (max for IP, min for L2 after the |q-d|^2 transform).
      parlay::parallel_for(0, R, [&](size_t r) {
        const float* ip_row = &IP_buf[r * M];
        float* out_row = &per_cloud_red[r * C];
        if (is_ip) {
          for (int64_t c = 0; c < C; ++c) {
            const int64_t s = d_offs_local[c];
            const int64_t e = d_offs_local[c + 1];
            if (s == e) {
              out_row[c] = -std::numeric_limits<float>::infinity();
              continue;
            }
            float m = ip_row[s];
            for (int64_t v = s + 1; v < e; ++v) {
              if (ip_row[v] > m) m = ip_row[v];
            }
            out_row[c] = m;
          }
        } else {
          // sq_dist[r, v] = |q|^2 + |x|^2 - 2 <q, x>, clipped at 0.
          // reduce min over v in [s, e).
          const float qn = q_norms_sq[qb_start * Qper + r];
          for (int64_t c = 0; c < C; ++c) {
            const int64_t s = d_offs_local[c];
            const int64_t e = d_offs_local[c + 1];
            if (s == e) {
              out_row[c] = std::numeric_limits<float>::infinity();
              continue;
            }
            float m = std::numeric_limits<float>::infinity();
            for (int64_t v = s; v < e; ++v) {
              float sq = qn + d_norms_sq[v] - 2.0f * ip_row[v];
              if (sq < 0) sq = 0;
              if (sq < m) m = sq;
            }
            out_row[c] = m;
          }
        }
      });

      // For each query b in the batch, sum per_cloud_red across its Qper
      // query-rows and write the (negated, for IP) mean into chunk_dists.
      parlay::parallel_for(0, B, [&](size_t b) {
        for (int64_t c = 0; c < C; ++c) {
          float s = 0;
          for (int64_t q = 0; q < Qper; ++q) {
            s += per_cloud_red[(b * Qper + q) * C + c];
          }
          float val = s * invQ;
          if (is_ip) val = -val;
          chunk_dists[(qb_start + b) * C + c] = val;
        }
      });
      T.reduce += now_sec() - t0;
    }

    // ---- One merge per chunk ----
    {
      const double t0 = now_sec();
      merge_topk_chunk(n_queries, k, static_cast<int>(C), db_start,
                       chunk_dists.data(), topk_dists.data(), topk_ids.data());
      T.merge += now_sec() - t0;
    }

    ++chunks_done;
    if (log_progress && now_sec() - last_log >= A.progress_interval) {
      emit_progress(chunks_done, n_db_chunks, db_end, n_db, t_loop_start, T);
      last_log = now_sec();
    }
  }

  if (log_progress) {
    emit_progress(chunks_done, n_db_chunks, n_db, n_db, t_loop_start, T,
                  "finished");
  }

  // ---- Final per-row sort: lexicographic (distance, id) ----
  // Ties on distance are broken by id ascending so the output is bit-identical
  // across implementations / platforms (the python tool does the same).
  parlay::parallel_for(0, n_queries, [&](size_t i) {
    std::vector<int> idx(k);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
      const float da = topk_dists[i * k + a];
      const float db = topk_dists[i * k + b];
      if (da != db) return da < db;
      return topk_ids[i * k + a] < topk_ids[i * k + b];
    });
    std::vector<float> sd(k);
    std::vector<uint32_t> si(k);
    for (int j = 0; j < k; ++j) {
      sd[j] = topk_dists[i * k + idx[j]];
      si[j] = topk_ids[i * k + idx[j]];
    }
    std::memcpy(&topk_dists[i * k], sd.data(), k * sizeof(float));
    std::memcpy(&topk_ids[i * k], si.data(), k * sizeof(uint32_t));
  });

  // ---- Write output: int32 k, then for each query k * (f32, u32) pairs ----
  if (verbose) std::cout << "[write] " << A.output << "\n";
  std::ofstream out(A.output, std::ios::binary);
  if (!out) throw std::runtime_error("cannot open output: " + A.output);
  const int32_t k32 = static_cast<int32_t>(k);
  out.write(reinterpret_cast<const char*>(&k32), sizeof(int32_t));
  // The on-disk pair layout is { float32 dist; uint32 id; } — emit it
  // contiguously per query.
  std::vector<char> rowbuf(static_cast<size_t>(k) * 8);
  for (int64_t i = 0; i < n_queries; ++i) {
    char* p = rowbuf.data();
    for (int j = 0; j < k; ++j) {
      std::memcpy(p + 0, &topk_dists[i * k + j], 4);
      std::memcpy(p + 4, &topk_ids[i * k + j], 4);
      p += 8;
    }
    out.write(rowbuf.data(), rowbuf.size());
  }
  out.close();
  if (verbose) std::cout << "[done] ground truth written to " << A.output << "\n";
}


int main(int argc, char** argv) {
  try {
    Args a = parse_args(argc, argv);
    compute_ground_truth(a);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
