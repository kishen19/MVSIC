// Microbenchmark: 1BTQ inner kernel — direct popcnt_d32 vs Harley-Seal CSA.
//
// Tests Idea B from 1BTQ-Optimization-Ideas.md: replace per-(panel,query)
// VPOPCNTD ops with a Harley-Seal carry-save tree that reduces 8 input
// vectors to 4 popcount inputs (ones/twos/fours/eights) at the cost of
// 7 CSA operations.
//
// Both kernels produce per-(panel,query,lane) 32-bit Hamming totals across
// num_tiles tiles. The benchmark verifies output bitwise equality and times
// each kernel on a sweep of (num_panels, num_queries_per_batch, num_tiles)
// shapes representative of fiqa (128D → 4 tiles), 256D (8 tiles), 512D (16
// tiles), and 1024D (32 tiles).
//
// Run:
//   bazel build -c opt //microbenchmark/mvivf:bench_1btq_harley_seal
//   ./bazel-bin/microbenchmark/mvivf/bench_1btq_harley_seal

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr size_t kPanelPoints = 16;
constexpr size_t kPanelLaneBytes = 4;
constexpr size_t kTileBytes = kPanelPoints * kPanelLaneBytes;  // 64

// =========================================================================
// Per-32-bit-lane popcount (matches popcnt_d32 in turboquant_1bit_mv.h).
// =========================================================================
inline __m512i popcnt_d32(__m512i x) {
#if defined(__AVX512VPOPCNTDQ__)
  return _mm512_popcnt_epi32(x);
#else
#error "Requires VPOPCNTDQ"
#endif
}

// =========================================================================
// Variant A: current direct popcnt_d32 kernel (mirrors
// hamming_micro_kernel_4panel<4> in the production header).
// =========================================================================
template<size_t Mq>
__attribute__((noinline))
void kernel_direct(const uint8_t* qbuf, size_t qbuf_tile_stride,
                   const uint8_t* panel0, const uint8_t* panel1,
                   const uint8_t* panel2, const uint8_t* panel3,
                   size_t num_tiles, __m512i* acc0, __m512i* acc1,
                   __m512i* acc2, __m512i* acc3) {
  for (size_t q = 0; q < Mq; ++q) {
    acc0[q] = _mm512_setzero_si512();
    acc1[q] = _mm512_setzero_si512();
    acc2[q] = _mm512_setzero_si512();
    acc3[q] = _mm512_setzero_si512();
  }
  for (size_t t = 0; t < num_tiles; ++t) {
    const __m512i pa = _mm512_loadu_si512(
        reinterpret_cast<const __m512i*>(panel0 + t * kTileBytes));
    const __m512i pb = _mm512_loadu_si512(
        reinterpret_cast<const __m512i*>(panel1 + t * kTileBytes));
    const __m512i pc = _mm512_loadu_si512(
        reinterpret_cast<const __m512i*>(panel2 + t * kTileBytes));
    const __m512i pd = _mm512_loadu_si512(
        reinterpret_cast<const __m512i*>(panel3 + t * kTileBytes));
    for (size_t q = 0; q < Mq; ++q) {
      const __m512i qb = _mm512_loadu_si512(
          reinterpret_cast<const __m512i*>(qbuf + q * qbuf_tile_stride + t * kTileBytes));
      acc0[q] = _mm512_add_epi32(acc0[q], popcnt_d32(_mm512_xor_si512(pa, qb)));
      acc1[q] = _mm512_add_epi32(acc1[q], popcnt_d32(_mm512_xor_si512(pb, qb)));
      acc2[q] = _mm512_add_epi32(acc2[q], popcnt_d32(_mm512_xor_si512(pc, qb)));
      acc3[q] = _mm512_add_epi32(acc3[q], popcnt_d32(_mm512_xor_si512(pd, qb)));
    }
  }
}

// =========================================================================
// Variant B: Harley-Seal CSA popcount.
//
// CSA(a, b, c) returns (sum, carry) such that, per bit position p,
//   a_p + b_p + c_p = sum_p + 2 * carry_p
// We feed 8 input XOR vectors x[0..7] for one (panel,query) pair through a
// Mula-style 8-input HS tree that produces (ones, twos, fours, eights),
// per bit p, with x[0]_p + ... + x[7]_p = ones_p + 2 twos_p + 4 fours_p +
// 8 eights_p. The tile-bound popcount per (panel,query) per 8 tiles becomes:
//   acc += popcnt_d32(ones) + 2*popcnt_d32(twos)
//        + 4*popcnt_d32(fours) + 8*popcnt_d32(eights)
// — 4 popcounts instead of 8.
// =========================================================================
static inline void csa(__m512i& h, __m512i& l, __m512i a, __m512i b, __m512i c) {
  const __m512i u = _mm512_xor_si512(a, b);
  l = _mm512_xor_si512(u, c);
  h = _mm512_or_si512(_mm512_and_si512(a, b), _mm512_and_si512(u, c));
}

// Reduce 8 inputs into (ones, twos, fours, eights). 7 CSAs.
static inline void hs8(const __m512i x[8], __m512i& ones, __m512i& twos,
                       __m512i& fours, __m512i& eights) {
  __m512i twosA, twosB, foursA, foursB;
  csa(twosA, ones, x[0], x[1], x[2]);   // ones holds x0^x1^x2; twosA holds maj
  csa(twosB, ones, ones, x[3], x[4]);   // fold x3,x4 into ones
  csa(foursA, twos, twosA, twosB, _mm512_setzero_si512());
  // Now: x[0]+..+x[4] (per bit) = ones + 2*twos + 4*foursA  (twos = twosA^twosB,
  // foursA = twosA & twosB).
  __m512i twosC, twosD;
  csa(twosC, ones, ones, x[5], x[6]);   // fold x5,x6 into ones; twosC = maj
  csa(twosD, ones, ones, x[7], _mm512_setzero_si512());  // fold x7
  // ones now holds parity of all 8 inputs.
  csa(foursB, twos, twos, twosC, twosD);
  // twos now holds (twosA^twosB) ^ twosC ^ twosD.
  // foursB = MAJ(prev_twos, twosC, twosD).
  csa(eights, fours, foursA, foursB, _mm512_setzero_si512());
}

template<size_t Mq>
__attribute__((noinline))
void kernel_harley_seal(const uint8_t* qbuf, size_t qbuf_tile_stride,
                        const uint8_t* panel0, const uint8_t* panel1,
                        const uint8_t* panel2, const uint8_t* panel3,
                        size_t num_tiles, __m512i* acc0, __m512i* acc1,
                        __m512i* acc2, __m512i* acc3) {
  for (size_t q = 0; q < Mq; ++q) {
    acc0[q] = _mm512_setzero_si512();
    acc1[q] = _mm512_setzero_si512();
    acc2[q] = _mm512_setzero_si512();
    acc3[q] = _mm512_setzero_si512();
  }
  // Process 8 tiles at a time per (panel,query) pair via HS.
  size_t t = 0;
  for (; t + 8 <= num_tiles; t += 8) {
    // Load 8 panel tiles for each of the 4 panels and broadcast queries.
    __m512i pa[8], pb[8], pc[8], pd[8];
    for (size_t i = 0; i < 8; ++i) {
      pa[i] = _mm512_loadu_si512(
          reinterpret_cast<const __m512i*>(panel0 + (t + i) * kTileBytes));
      pb[i] = _mm512_loadu_si512(
          reinterpret_cast<const __m512i*>(panel1 + (t + i) * kTileBytes));
      pc[i] = _mm512_loadu_si512(
          reinterpret_cast<const __m512i*>(panel2 + (t + i) * kTileBytes));
      pd[i] = _mm512_loadu_si512(
          reinterpret_cast<const __m512i*>(panel3 + (t + i) * kTileBytes));
    }
    for (size_t q = 0; q < Mq; ++q) {
      __m512i qb[8];
      for (size_t i = 0; i < 8; ++i) {
        qb[i] = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(
            qbuf + q * qbuf_tile_stride + (t + i) * kTileBytes));
      }
      // Process each panel.
      for (int panel_idx = 0; panel_idx < 4; ++panel_idx) {
        const __m512i* p = (panel_idx == 0) ? pa : (panel_idx == 1) ? pb
                              : (panel_idx == 2)               ? pc
                                                               : pd;
        __m512i x[8];
        for (size_t i = 0; i < 8; ++i) x[i] = _mm512_xor_si512(p[i], qb[i]);
        __m512i ones, twos, fours, eights;
        hs8(x, ones, twos, fours, eights);
        const __m512i pc_ones = popcnt_d32(ones);
        const __m512i pc_twos = popcnt_d32(twos);
        const __m512i pc_fours = popcnt_d32(fours);
        const __m512i pc_eights = popcnt_d32(eights);
        const __m512i sum = _mm512_add_epi32(
            _mm512_add_epi32(pc_ones, _mm512_slli_epi32(pc_twos, 1)),
            _mm512_add_epi32(_mm512_slli_epi32(pc_fours, 2),
                             _mm512_slli_epi32(pc_eights, 3)));
        switch (panel_idx) {
          case 0: acc0[q] = _mm512_add_epi32(acc0[q], sum); break;
          case 1: acc1[q] = _mm512_add_epi32(acc1[q], sum); break;
          case 2: acc2[q] = _mm512_add_epi32(acc2[q], sum); break;
          case 3: acc3[q] = _mm512_add_epi32(acc3[q], sum); break;
        }
      }
    }
  }
  // Tail: scalar-style (matches direct kernel) for remaining tiles.
  for (; t < num_tiles; ++t) {
    const __m512i pa = _mm512_loadu_si512(
        reinterpret_cast<const __m512i*>(panel0 + t * kTileBytes));
    const __m512i pb = _mm512_loadu_si512(
        reinterpret_cast<const __m512i*>(panel1 + t * kTileBytes));
    const __m512i pc = _mm512_loadu_si512(
        reinterpret_cast<const __m512i*>(panel2 + t * kTileBytes));
    const __m512i pd = _mm512_loadu_si512(
        reinterpret_cast<const __m512i*>(panel3 + t * kTileBytes));
    for (size_t q = 0; q < Mq; ++q) {
      const __m512i qb = _mm512_loadu_si512(
          reinterpret_cast<const __m512i*>(qbuf + q * qbuf_tile_stride + t * kTileBytes));
      acc0[q] = _mm512_add_epi32(acc0[q], popcnt_d32(_mm512_xor_si512(pa, qb)));
      acc1[q] = _mm512_add_epi32(acc1[q], popcnt_d32(_mm512_xor_si512(pb, qb)));
      acc2[q] = _mm512_add_epi32(acc2[q], popcnt_d32(_mm512_xor_si512(pc, qb)));
      acc3[q] = _mm512_add_epi32(acc3[q], popcnt_d32(_mm512_xor_si512(pd, qb)));
    }
  }
}

// =========================================================================
// Test harness.
// =========================================================================
struct Shape {
  size_t num_tiles;
  size_t num_panels;     // total panels in cloud (must be a multiple of 4)
  size_t num_queries;    // total queries (must be a multiple of 4)
  const char* label;
};

constexpr size_t kMq = 4;

void verify_equal(const std::vector<int32_t>& a, const std::vector<int32_t>& b,
                  const char* label) {
  if (a.size() != b.size()) {
    std::printf("[%s] SIZE MISMATCH: %zu vs %zu\n", label, a.size(), b.size());
    std::abort();
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      std::printf("[%s] MISMATCH at idx %zu: direct=%d hs=%d\n", label, i,
                  a[i], b[i]);
      std::abort();
    }
  }
}

double now_sec() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void run_shape(const Shape& s, int reps) {
  // Allocate panel and query data.
  std::mt19937_64 rng(0xC0FFEEULL ^ s.num_tiles);
  const size_t panel_bytes = s.num_tiles * kTileBytes;
  std::vector<uint8_t> panels(s.num_panels * panel_bytes);
  for (auto& b : panels) b = static_cast<uint8_t>(rng());
  // For each query, broadcast its tile-words across 16 lanes (matches
  // pre_broadcast_query layout in the production header).
  const size_t qbuf_tile_stride = s.num_tiles * kTileBytes;
  std::vector<uint8_t> qbuf(s.num_queries * qbuf_tile_stride);
  for (size_t q = 0; q < s.num_queries; ++q) {
    for (size_t t = 0; t < s.num_tiles; ++t) {
      uint32_t word = static_cast<uint32_t>(rng());
      uint8_t* dst = qbuf.data() + q * qbuf_tile_stride + t * kTileBytes;
      for (size_t lane = 0; lane < kPanelPoints; ++lane) {
        std::memcpy(dst + lane * 4, &word, 4);
      }
    }
  }

  // Output buffers: per-(panel-group, query, lane) i32. We collect direct
  // and HS outputs into flat int32 arrays for verification.
  const size_t num_panel_groups = s.num_panels / 4;
  const size_t out_per_call = 4 /*panels per group*/ * kMq * 16 /*lanes*/;
  std::vector<int32_t> out_direct(num_panel_groups * (s.num_queries / kMq) * out_per_call);
  std::vector<int32_t> out_hs = out_direct;

  auto run_one = [&](auto&& kfn, std::vector<int32_t>& out) {
    for (size_t qi = 0; qi < s.num_queries; qi += kMq) {
      const uint8_t* q_base = qbuf.data() + qi * qbuf_tile_stride;
      for (size_t pg = 0; pg < num_panel_groups; ++pg) {
        const size_t p0 = pg * 4;
        __m512i a0[kMq], a1[kMq], a2[kMq], a3[kMq];
        kfn(q_base, qbuf_tile_stride, panels.data() + (p0 + 0) * panel_bytes,
            panels.data() + (p0 + 1) * panel_bytes,
            panels.data() + (p0 + 2) * panel_bytes,
            panels.data() + (p0 + 3) * panel_bytes, s.num_tiles, a0, a1, a2, a3);
        const size_t out_off =
            ((qi / kMq) * num_panel_groups + pg) * out_per_call;
        for (size_t q = 0; q < kMq; ++q) {
          alignas(64) int32_t buf[16];
          _mm512_store_si512(reinterpret_cast<__m512i*>(buf), a0[q]);
          for (int j = 0; j < 16; ++j) out[out_off + (q * 4 + 0) * 16 + j] = buf[j];
          _mm512_store_si512(reinterpret_cast<__m512i*>(buf), a1[q]);
          for (int j = 0; j < 16; ++j) out[out_off + (q * 4 + 1) * 16 + j] = buf[j];
          _mm512_store_si512(reinterpret_cast<__m512i*>(buf), a2[q]);
          for (int j = 0; j < 16; ++j) out[out_off + (q * 4 + 2) * 16 + j] = buf[j];
          _mm512_store_si512(reinterpret_cast<__m512i*>(buf), a3[q]);
          for (int j = 0; j < 16; ++j) out[out_off + (q * 4 + 3) * 16 + j] = buf[j];
        }
      }
    }
  };

  // Verify (single rep).
  run_one(kernel_direct<kMq>, out_direct);
  run_one(kernel_harley_seal<kMq>, out_hs);
  verify_equal(out_direct, out_hs, s.label);

  // Time both.
  auto bench = [&](auto&& kfn) {
    std::vector<int32_t> sink(out_direct.size());
    // warmup
    run_one(kfn, sink);
    std::vector<double> times;
    times.reserve(reps);
    for (int r = 0; r < reps; ++r) {
      double t0 = now_sec();
      for (int inner = 0; inner < 50; ++inner) run_one(kfn, sink);
      double t1 = now_sec();
      times.push_back((t1 - t0) / 50);
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
  };

  const double t_direct = bench(kernel_direct<kMq>);
  const double t_hs = bench(kernel_harley_seal<kMq>);

  // Report ns/(panel*query*tile) = total work / time.
  const size_t pq_tiles = s.num_panels * s.num_queries * s.num_tiles;
  std::printf(
      "[%-10s] tiles=%2zu panels=%4zu queries=%4zu  direct=%.2f ns "
      "hs=%.2f ns  speedup=%.2fx\n",
      s.label, s.num_tiles, s.num_panels, s.num_queries,
      t_direct * 1e9 / pq_tiles, t_hs * 1e9 / pq_tiles, t_direct / t_hs);
}

}  // namespace

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  std::printf("Harley-Seal vs direct popcnt_d32 microbench (Mq=%zu, 4-panel)\n",
              kMq);
  // Realistic shapes for typical cloud sizes:
  // - fiqa: 128D → 4 tiles per panel; cloud ~16 points → 1 panel per cloud
  //   but a typical leaf has ~30 clouds × ~16 points = ~30 panels.
  // - With 32 query embeddings per cloud and 4 mq batches → 8 query batches.
  std::vector<Shape> shapes = {
      {4, 32, 32, "fiqa-128"},     // 128D
      {8, 32, 32, "256D"},
      {16, 32, 32, "512D"},
      {32, 32, 32, "1024D"},
      {4, 128, 32, "fiqa-large"},  // bigger leaves
      {8, 128, 32, "256D-large"},
  };
  for (const auto& s : shapes) run_shape(s, 9);
  return 0;
}
