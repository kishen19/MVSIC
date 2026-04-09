// Microbenchmark: 1BTQ inner kernel — direct popcnt_d32 vs VBMI vpermb+dpbusd.
//
// Tests Idea A from 1BTQ-Optimization-Ideas.md: replace per-(panel,query)
// VPOPCNTD ops with a per-query precomputed nibble Hamming LUT and
// vpermb + vpdpbusd lookups.
//
// IMPORTANT: all kernels are inlined via template + force_inline so that the
// compiler can keep the 16 per-(panel,query) accumulators in registers across
// the full tile sweep. The production kernel in turboquant_1bit_mv.h is also
// inlined by its caller (chamfer_panels / score_one_db_cloud), so this
// matches real-world code gen.
//
// Run:
//   bazel build -c opt //microbenchmark/mvivf:bench_1btq_vbmi
//   ./bazel-bin/microbenchmark/mvivf/bench_1btq_vbmi

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
constexpr size_t kMq = 4;

// Force-inline attribute: the production 1BTQ kernel is inlined into its
// caller and keeps all 16 accumulators in zmm registers. Without this the
// compiler spills to memory on every popcount, which bakes in ~50% overhead
// and invalidates the microbench.
#define ALWAYS_INLINE __attribute__((always_inline)) inline

// =========================================================================
// Per-32-bit-lane popcount (matches popcnt_d32 in turboquant_1bit_mv.h).
// =========================================================================
ALWAYS_INLINE __m512i popcnt_d32(__m512i x) {
#if defined(__AVX512VPOPCNTDQ__)
  return _mm512_popcnt_epi32(x);
#else
#error "Requires VPOPCNTDQ"
#endif
}

// =========================================================================
// Variant A: current direct popcnt_d32 kernel (baseline).
// =========================================================================
template<size_t Mq>
ALWAYS_INLINE
void kernel_direct(const uint8_t* qbuf, size_t qbuf_tile_stride,
                   const uint8_t* panel0, const uint8_t* panel1,
                   const uint8_t* panel2, const uint8_t* panel3,
                   size_t num_tiles,
                   const uint8_t* /*lut_lo_unused*/,
                   const uint8_t* /*lut_hi_unused*/,
                   size_t /*lut_stride_unused*/,
                   __m512i* acc0, __m512i* acc1,
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
// Variant B: VBMI vpermb nibble-LUT + vpdpbusd fold.
// =========================================================================
template<size_t Mq>
ALWAYS_INLINE
void kernel_vbmi_v1(const uint8_t* qbuf, size_t qbuf_tile_stride,
                    const uint8_t* panel0, const uint8_t* panel1,
                    const uint8_t* panel2, const uint8_t* panel3,
                    size_t num_tiles,
                    const uint8_t* lut_lo,
                    const uint8_t* lut_hi,
                    size_t lut_tile_stride,
                    __m512i* acc0, __m512i* acc1,
                    __m512i* acc2, __m512i* acc3) {
  (void)qbuf;
  (void)qbuf_tile_stride;
  const __m512i low_mask = _mm512_set1_epi8(0x0F);
  const __m512i byte_offset = _mm512_set_epi8(
      48, 32, 16, 0, 48, 32, 16, 0, 48, 32, 16, 0, 48, 32, 16, 0,
      48, 32, 16, 0, 48, 32, 16, 0, 48, 32, 16, 0, 48, 32, 16, 0,
      48, 32, 16, 0, 48, 32, 16, 0, 48, 32, 16, 0, 48, 32, 16, 0,
      48, 32, 16, 0, 48, 32, 16, 0, 48, 32, 16, 0, 48, 32, 16, 0);
  const __m512i ones = _mm512_set1_epi8(1);

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
    const __m512i pc_ = _mm512_loadu_si512(
        reinterpret_cast<const __m512i*>(panel2 + t * kTileBytes));
    const __m512i pd = _mm512_loadu_si512(
        reinterpret_cast<const __m512i*>(panel3 + t * kTileBytes));

    const __m512i pa_lo = _mm512_or_si512(_mm512_and_si512(pa, low_mask), byte_offset);
    const __m512i pa_hi = _mm512_or_si512(
        _mm512_and_si512(_mm512_srli_epi16(pa, 4), low_mask), byte_offset);
    const __m512i pb_lo = _mm512_or_si512(_mm512_and_si512(pb, low_mask), byte_offset);
    const __m512i pb_hi = _mm512_or_si512(
        _mm512_and_si512(_mm512_srli_epi16(pb, 4), low_mask), byte_offset);
    const __m512i pc_lo = _mm512_or_si512(_mm512_and_si512(pc_, low_mask), byte_offset);
    const __m512i pc_hi = _mm512_or_si512(
        _mm512_and_si512(_mm512_srli_epi16(pc_, 4), low_mask), byte_offset);
    const __m512i pd_lo = _mm512_or_si512(_mm512_and_si512(pd, low_mask), byte_offset);
    const __m512i pd_hi = _mm512_or_si512(
        _mm512_and_si512(_mm512_srli_epi16(pd, 4), low_mask), byte_offset);

    for (size_t q = 0; q < Mq; ++q) {
      const __m512i lut_l = _mm512_loadu_si512(
          reinterpret_cast<const __m512i*>(lut_lo + q * lut_tile_stride + t * kTileBytes));
      const __m512i lut_h = _mm512_loadu_si512(
          reinterpret_cast<const __m512i*>(lut_hi + q * lut_tile_stride + t * kTileBytes));

      {
        const __m512i sa_lo = _mm512_permutexvar_epi8(pa_lo, lut_l);
        const __m512i sa_hi = _mm512_permutexvar_epi8(pa_hi, lut_h);
        acc0[q] = _mm512_dpbusd_epi32(acc0[q], _mm512_add_epi8(sa_lo, sa_hi), ones);
      }
      {
        const __m512i sb_lo = _mm512_permutexvar_epi8(pb_lo, lut_l);
        const __m512i sb_hi = _mm512_permutexvar_epi8(pb_hi, lut_h);
        acc1[q] = _mm512_dpbusd_epi32(acc1[q], _mm512_add_epi8(sb_lo, sb_hi), ones);
      }
      {
        const __m512i sc_lo = _mm512_permutexvar_epi8(pc_lo, lut_l);
        const __m512i sc_hi = _mm512_permutexvar_epi8(pc_hi, lut_h);
        acc2[q] = _mm512_dpbusd_epi32(acc2[q], _mm512_add_epi8(sc_lo, sc_hi), ones);
      }
      {
        const __m512i sd_lo = _mm512_permutexvar_epi8(pd_lo, lut_l);
        const __m512i sd_hi = _mm512_permutexvar_epi8(pd_hi, lut_h);
        acc3[q] = _mm512_dpbusd_epi32(acc3[q], _mm512_add_epi8(sd_lo, sd_hi), ones);
      }
    }
  }
}

// =========================================================================
// LUT construction for VBMI variants.
// =========================================================================
void build_luts_for_query(const uint8_t* q_packed, size_t num_bytes,
                          size_t num_tiles, uint8_t* lut_lo_out,
                          uint8_t* lut_hi_out) {
  std::memset(lut_lo_out, 0, num_tiles * 64);
  std::memset(lut_hi_out, 0, num_tiles * 64);
  for (size_t t = 0; t < num_tiles; ++t) {
    for (size_t b = 0; b < 4; ++b) {
      const size_t byte_idx = t * 4 + b;
      const uint8_t qb = (byte_idx < num_bytes) ? q_packed[byte_idx] : 0u;
      const uint8_t q_lo = qb & 0x0F;
      const uint8_t q_hi = (qb >> 4) & 0x0F;
      for (size_t i = 0; i < 16; ++i) {
        lut_lo_out[t * 64 + b * 16 + i] =
            static_cast<uint8_t>(__builtin_popcount(static_cast<unsigned>(i ^ q_lo)));
        lut_hi_out[t * 64 + b * 16 + i] =
            static_cast<uint8_t>(__builtin_popcount(static_cast<unsigned>(i ^ q_hi)));
      }
    }
  }
}

// =========================================================================
// Benchmark drivers. One per variant, templated on the kernel. Each driver
// is compiled as a separate __attribute__((noinline)) function so that the
// timing loop doesn't accidentally optimize the kernel away across variants,
// but the kernel itself is force-inlined into the driver so accumulators
// stay in registers.
// =========================================================================
struct KernelInputs {
  const uint8_t* qbuf;
  size_t qbuf_tile_stride;
  const uint8_t* panels;
  size_t panel_bytes;
  size_t num_tiles;
  size_t num_queries;
  size_t num_panels;
  const uint8_t* lut_lo_all;
  const uint8_t* lut_hi_all;
  size_t lut_tile_stride;   // per-query stride
  size_t lut_batch_bytes;   // per-kMq-batch stride
  int32_t* out;             // flat output: (num_batches * num_panel_groups * Mq * 4panels * 16 lanes)
};

template<typename Kernel>
__attribute__((noinline))
void run_one(const KernelInputs& in, Kernel k) {
  const size_t num_panel_groups = in.num_panels / 4;
  const size_t out_per_call = 4 * kMq * 16;
  for (size_t qi_batch = 0; qi_batch < in.num_queries; qi_batch += kMq) {
    const size_t bi = qi_batch / kMq;
    const uint8_t* q_base = in.qbuf + qi_batch * in.qbuf_tile_stride;
    const uint8_t* lut_lo_base = in.lut_lo_all + bi * in.lut_batch_bytes;
    const uint8_t* lut_hi_base = in.lut_hi_all + bi * in.lut_batch_bytes;
    for (size_t pg = 0; pg < num_panel_groups; ++pg) {
      const size_t p0 = pg * 4;
      __m512i a0[kMq], a1[kMq], a2[kMq], a3[kMq];
      k(q_base, in.qbuf_tile_stride,
        in.panels + (p0 + 0) * in.panel_bytes,
        in.panels + (p0 + 1) * in.panel_bytes,
        in.panels + (p0 + 2) * in.panel_bytes,
        in.panels + (p0 + 3) * in.panel_bytes,
        in.num_tiles, lut_lo_base, lut_hi_base, in.lut_tile_stride,
        a0, a1, a2, a3);
      const size_t out_off = ((qi_batch / kMq) * num_panel_groups + pg) * out_per_call;
      // Prevent the compiler from discarding the accs by writing them out.
      for (size_t q = 0; q < kMq; ++q) {
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(in.out + out_off + (q * 4 + 0) * 16), a0[q]);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(in.out + out_off + (q * 4 + 1) * 16), a1[q]);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(in.out + out_off + (q * 4 + 2) * 16), a2[q]);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(in.out + out_off + (q * 4 + 3) * 16), a3[q]);
      }
    }
  }
}

// Concrete drivers: lambda wraps the kernel call so each variant becomes a
// distinct instantiation of run_one (separate noinline function body per
// variant, each with the kernel force-inlined).
auto k_direct = [](const uint8_t* qbuf, size_t qs, const uint8_t* p0, const uint8_t* p1,
                    const uint8_t* p2, const uint8_t* p3, size_t nt,
                    const uint8_t* ll, const uint8_t* lh, size_t lts,
                    __m512i* a0, __m512i* a1, __m512i* a2, __m512i* a3) {
  kernel_direct<kMq>(qbuf, qs, p0, p1, p2, p3, nt, ll, lh, lts, a0, a1, a2, a3);
};

auto k_vbmi_v1 = [](const uint8_t* qbuf, size_t qs, const uint8_t* p0, const uint8_t* p1,
                     const uint8_t* p2, const uint8_t* p3, size_t nt,
                     const uint8_t* ll, const uint8_t* lh, size_t lts,
                     __m512i* a0, __m512i* a1, __m512i* a2, __m512i* a3) {
  kernel_vbmi_v1<kMq>(qbuf, qs, p0, p1, p2, p3, nt, ll, lh, lts, a0, a1, a2, a3);
};

__attribute__((noinline))
void run_direct(const KernelInputs& in) { run_one(in, k_direct); }
__attribute__((noinline))
void run_vbmi_v1(const KernelInputs& in) { run_one(in, k_vbmi_v1); }

// =========================================================================
// Test harness.
// =========================================================================
struct Shape {
  size_t num_tiles;
  size_t num_panels;
  size_t num_queries;
  const char* label;
};

void verify_equal(const std::vector<int32_t>& a, const std::vector<int32_t>& b,
                  const char* label, const char* variant) {
  if (a.size() != b.size()) {
    std::printf("[%s/%s] SIZE MISMATCH: %zu vs %zu\n", label, variant, a.size(), b.size());
    std::abort();
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      std::printf("[%s/%s] MISMATCH at idx %zu: direct=%d vbmi=%d\n", label,
                  variant, i, a[i], b[i]);
      std::abort();
    }
  }
}

double now_sec() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void run_shape(const Shape& s, int reps) {
  std::mt19937_64 rng(0xC0FFEEULL ^ s.num_tiles);
  const size_t panel_bytes = s.num_tiles * kTileBytes;
  std::vector<uint8_t> panels(s.num_panels * panel_bytes);
  for (auto& b : panels) b = static_cast<uint8_t>(rng());

  const size_t qbuf_tile_stride = s.num_tiles * kTileBytes;
  std::vector<uint8_t> qbuf(s.num_queries * qbuf_tile_stride);

  const size_t num_bytes_per_q = s.num_tiles * 4;
  std::vector<uint8_t> q_packed(s.num_queries * num_bytes_per_q);

  for (size_t q = 0; q < s.num_queries; ++q) {
    for (size_t t = 0; t < s.num_tiles; ++t) {
      uint32_t word = static_cast<uint32_t>(rng());
      std::memcpy(q_packed.data() + q * num_bytes_per_q + t * 4, &word, 4);
      uint8_t* dst = qbuf.data() + q * qbuf_tile_stride + t * kTileBytes;
      for (size_t lane = 0; lane < kPanelPoints; ++lane) {
        std::memcpy(dst + lane * 4, &word, 4);
      }
    }
  }

  const size_t lut_tile_stride = s.num_tiles * kTileBytes;
  const size_t lut_batch_bytes = kMq * lut_tile_stride;
  const size_t n_batches = s.num_queries / kMq;
  std::vector<uint8_t> lut_lo_all(n_batches * lut_batch_bytes);
  std::vector<uint8_t> lut_hi_all(n_batches * lut_batch_bytes);
  for (size_t bi = 0; bi < n_batches; ++bi) {
    for (size_t q = 0; q < kMq; ++q) {
      const size_t qi = bi * kMq + q;
      build_luts_for_query(q_packed.data() + qi * num_bytes_per_q, num_bytes_per_q,
                           s.num_tiles,
                           lut_lo_all.data() + bi * lut_batch_bytes + q * lut_tile_stride,
                           lut_hi_all.data() + bi * lut_batch_bytes + q * lut_tile_stride);
    }
  }

  const size_t num_panel_groups = s.num_panels / 4;
  const size_t out_per_call = 4 * kMq * 16;
  const size_t out_total = num_panel_groups * (s.num_queries / kMq) * out_per_call;
  std::vector<int32_t> out_direct(out_total);
  std::vector<int32_t> out_vbmi(out_total);

  KernelInputs in{
      qbuf.data(),     qbuf_tile_stride, panels.data(),   panel_bytes,
      s.num_tiles,     s.num_queries,    s.num_panels,    lut_lo_all.data(),
      lut_hi_all.data(), lut_tile_stride, lut_batch_bytes, out_direct.data()};

  // Verify.
  run_direct(in);
  in.out = out_vbmi.data();
  run_vbmi_v1(in);
  verify_equal(out_direct, out_vbmi, s.label, "v1");

  // Time.
  auto bench = [&](void (*fn)(const KernelInputs&), int32_t* out_buf) {
    KernelInputs local = in;
    local.out = out_buf;
    fn(local);  // warmup
    std::vector<double> times;
    times.reserve(reps);
    for (int r = 0; r < reps; ++r) {
      double t0 = now_sec();
      for (int inner = 0; inner < 50; ++inner) fn(local);
      double t1 = now_sec();
      times.push_back((t1 - t0) / 50);
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
  };

  const double t_direct = bench(&run_direct, out_direct.data());
  const double t_vbmi_v1 = bench(&run_vbmi_v1, out_vbmi.data());

  const size_t pq_tiles = s.num_panels * s.num_queries * s.num_tiles;
  std::printf(
      "[%-10s] tiles=%2zu panels=%4zu queries=%4zu  "
      "direct=%.2f  v1=%.2f (%.2fx)  ns/(P*Q*T)\n",
      s.label, s.num_tiles, s.num_panels, s.num_queries,
      t_direct * 1e9 / pq_tiles,
      t_vbmi_v1 * 1e9 / pq_tiles, t_direct / t_vbmi_v1);
}

}  // namespace

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  std::printf("VBMI vs direct popcnt_d32 microbench (Mq=%zu, 4-panel, inline)\n", kMq);
  std::vector<Shape> shapes = {
      {4, 32, 32, "fiqa-128"},
      {8, 32, 32, "256D"},
      {16, 32, 32, "512D"},
      {32, 32, 32, "1024D"},
      {4, 128, 32, "fiqa-large"},
      {8, 128, 32, "256D-large"},
  };
  for (const auto& s : shapes) run_shape(s, 9);
  return 0;
}
