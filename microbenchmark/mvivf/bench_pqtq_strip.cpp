// Standalone microbenchmark for the SPQTQ vnni_scan_strip kernels.
//
// Goal: isolate the per-strip-call overhead ("plumbing" — acc init, prefetch,
// decode, mask blends) from the inner vpdpbusd/vpermb loop. The loop's port
// analysis says ~15 cyc/iter; with nb4=4 (typical fiqa m=8) the loop body is
// ~60 cyc, but the measured per-strip cost in the chamfer hot path is ~74 cyc.
// We want to know whether that 14 cyc gap is actual function plumbing or just
// the cost of the kernel's decode+min epilogue.
//
// Measures cyc/strip-call by summing rdtsc deltas across a tight loop that
// calls the kernel many times against the same in-L1 strip with the same in-L1
// scan LUTs and norms. Reports for nb4 in {4, 8, 12, 16} (i.e., 1, 2, 3, 4
// inner-loop iterations) so the plumbing overhead can be backed out from
// extrapolation to nb4=0.
//
// Run:
//   bazel build -c opt //microbenchmark/mvivf:bench_pqtq_strip
//   taskset -c 0 ./bazel-bin/microbenchmark/mvivf/bench_pqtq_strip [-iters N]

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#include <x86intrin.h>
#endif

#include "mvsic/core/quantization/pqtq.h"

using ::mvsic::pqtq::internal::vnni_scan_strip_fused3_minacc;
using ::mvsic::pqtq::internal::vnni_scan_strip_fused1_minacc;

// rdtscp serializes after, rdtsc + lfence pair before; gives reasonable
// per-call timing on Granite Rapids.
static inline uint64_t rdtsc_start() {
  unsigned int aux;
  _mm_lfence();
  uint64_t t = __rdtscp(&aux);
  _mm_lfence();
  return t;
}
static inline uint64_t rdtsc_end() {
  unsigned int aux;
  _mm_lfence();
  uint64_t t = __rdtscp(&aux);
  _mm_lfence();
  return t;
}

// Build a single fake strip + scan-LUT layout matching what pqtq.h emits.
// Strip is 128 bytes per 4-block group; nb4/4 groups → strip_stride bytes.
// Scan LUT is also 64 bytes per 4-block-group iter (one __m512 load per iter).
struct Fixture {
  std::vector<uint8_t> strip;        // strip_stride bytes
  std::vector<uint8_t> scan_lut0;    // (nb4/4) * 64 bytes
  std::vector<uint8_t> scan_lut1;
  std::vector<uint8_t> scan_lut2;
  std::vector<float> norms;          // 64 floats
  std::vector<float> sqns;           // 64 floats
  uint32_t nb4;
  int32_t bias_int;
};

static Fixture make_fixture(uint32_t nb4, std::mt19937& rng) {
  Fixture f;
  f.nb4 = nb4;
  // num_blocks (nb) ≈ nb4 (for the bench it's fine to set them equal).
  f.bias_int = static_cast<int32_t>(nb4) * 128;
  const size_t strip_bytes = (nb4 / 4) * 128;
  f.strip.resize(strip_bytes);
  f.scan_lut0.resize((nb4 / 4) * 64);
  f.scan_lut1.resize((nb4 / 4) * 64);
  f.scan_lut2.resize((nb4 / 4) * 64);
  f.norms.resize(64);
  f.sqns.resize(64);
  std::uniform_int_distribution<int> u8(0, 255);
  for (auto& b : f.strip) b = static_cast<uint8_t>(u8(rng));
  for (auto& b : f.scan_lut0) b = static_cast<uint8_t>(u8(rng));
  for (auto& b : f.scan_lut1) b = static_cast<uint8_t>(u8(rng));
  for (auto& b : f.scan_lut2) b = static_cast<uint8_t>(u8(rng));
  std::uniform_real_distribution<float> uf(0.5f, 2.0f);
  for (auto& v : f.norms) v = uf(rng);
  for (auto& v : f.sqns) v = uf(rng);
  return f;
}

// Time many calls to the 3-query fused kernel.
template<bool Metric>
static double time_fused3(const Fixture& f, size_t iters) {
  alignas(16) const __mmask16 full_masks[4] = {0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF};
  __m512 min0 = _mm512_set1_ps(std::numeric_limits<float>::max());
  __m512 min1 = _mm512_set1_ps(std::numeric_limits<float>::max());
  __m512 min2 = _mm512_set1_ps(std::numeric_limits<float>::max());

  // Touch everything once to ensure L1 residency.
  for (volatile uint8_t b : f.strip) (void)b;
  for (volatile uint8_t b : f.scan_lut0) (void)b;
  for (volatile uint8_t b : f.scan_lut1) (void)b;
  for (volatile uint8_t b : f.scan_lut2) (void)b;

  uint64_t t0 = rdtsc_start();
  for (size_t i = 0; i < iters; ++i) {
    vnni_scan_strip_fused3_minacc<Metric>(
        f.strip.data(), f.nb4,
        f.scan_lut0.data(), f.scan_lut1.data(), f.scan_lut2.data(),
        f.norms.data(), f.sqns.data(),
        1.0f, 1.0f, 1.0f, 0.5f, 0.5f, 0.5f,
        f.bias_int, full_masks, /*prefetch=*/nullptr, min0, min1, min2);
  }
  uint64_t t1 = rdtsc_end();

  // Sink the result so the loop can't be DCE'd.
  alignas(64) float buf0[16];
  alignas(64) float buf1[16];
  alignas(64) float buf2[16];
  _mm512_store_ps(buf0, min0);
  _mm512_store_ps(buf1, min1);
  _mm512_store_ps(buf2, min2);
  volatile float sink = buf0[0] + buf1[0] + buf2[0];
  (void)sink;

  return static_cast<double>(t1 - t0) / static_cast<double>(iters);
}

template<bool Metric>
static double time_fused1(const Fixture& f, size_t iters) {
  alignas(16) const __mmask16 full_masks[4] = {0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF};
  __m512 mv = _mm512_set1_ps(std::numeric_limits<float>::max());

  for (volatile uint8_t b : f.strip) (void)b;
  for (volatile uint8_t b : f.scan_lut0) (void)b;

  uint64_t t0 = rdtsc_start();
  for (size_t i = 0; i < iters; ++i) {
    vnni_scan_strip_fused1_minacc<Metric>(
        f.strip.data(), f.nb4, f.scan_lut0.data(),
        f.norms.data(), f.sqns.data(),
        1.0f, 0.5f, f.bias_int, full_masks, /*prefetch=*/nullptr, mv);
  }
  uint64_t t1 = rdtsc_end();

  alignas(64) float buf[16];
  _mm512_store_ps(buf, mv);
  volatile float sink = buf[0];
  (void)sink;

  return static_cast<double>(t1 - t0) / static_cast<double>(iters);
}

int main(int argc, char** argv) {
  size_t iters = 200000;
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "-iters") == 0) iters = std::strtoull(argv[i + 1], nullptr, 10);
  }

  std::printf("# bench_pqtq_strip: iters=%zu\n", iters);
  std::printf("# All runs use full_masks (no tail blend), prefetch=nullptr.\n");
  std::printf("# Reported value: cycles per kernel call (rdtscp/rdtscp).\n\n");
  std::printf("%-12s %-10s %-12s %-12s %-12s %-12s\n",
              "kernel", "Metric", "nb4=4", "nb4=8", "nb4=12", "nb4=16");

  std::mt19937 rng(0xABCDEF01);

  // Warm up branch predictors / icache.
  {
    auto f = make_fixture(4, rng);
    (void)time_fused3<false>(f, std::min<size_t>(iters, 1000));
    (void)time_fused3<true>(f, std::min<size_t>(iters, 1000));
    (void)time_fused1<false>(f, std::min<size_t>(iters, 1000));
    (void)time_fused1<true>(f, std::min<size_t>(iters, 1000));
  }

  auto run_row = [&](const char* label, auto fn, bool metric) {
    auto f4 = make_fixture(4, rng);
    auto f8 = make_fixture(8, rng);
    auto f12 = make_fixture(12, rng);
    auto f16 = make_fixture(16, rng);
    double c4 = fn(f4, iters);
    double c8 = fn(f8, iters);
    double c12 = fn(f12, iters);
    double c16 = fn(f16, iters);
    std::printf("%-12s %-10s %-12.2f %-12.2f %-12.2f %-12.2f\n", label,
                metric ? "L2" : "IP", c4, c8, c12, c16);
  };

  run_row("fused3", [](const Fixture& f, size_t it) { return time_fused3<false>(f, it); }, false);
  run_row("fused3", [](const Fixture& f, size_t it) { return time_fused3<true>(f, it); }, true);
  run_row("fused1", [](const Fixture& f, size_t it) { return time_fused1<false>(f, it); }, false);
  run_row("fused1", [](const Fixture& f, size_t it) { return time_fused1<true>(f, it); }, true);

  // Best-fit linear extrapolation: cyc(nb4) = a + b * (nb4/4).
  // a is the per-call plumbing; b is the inner-loop body.
  auto fit = [](double y4, double y8, double y12, double y16) {
    // Iter counts: 1, 2, 3, 4. y_i = a + b*i.
    // Least-squares for x={1,2,3,4}, y={y4,y8,y12,y16}:
    //   sum_x=10, sum_x2=30, n=4, sum_y, sum_xy.
    double sx = 10, sx2 = 30;
    double sy = y4 + y8 + y12 + y16;
    double sxy = 1 * y4 + 2 * y8 + 3 * y12 + 4 * y16;
    double n = 4;
    double b = (n * sxy - sx * sy) / (n * sx2 - sx * sx);
    double a = (sy - b * sx) / n;
    return std::make_pair(a, b);
  };

  std::printf("\n# Linear fit: cyc/call = plumbing + body * (nb4/4)\n");
  {
    auto f4 = make_fixture(4, rng);
    auto f8 = make_fixture(8, rng);
    auto f12 = make_fixture(12, rng);
    auto f16 = make_fixture(16, rng);
    double c4 = time_fused3<false>(f4, iters);
    double c8 = time_fused3<false>(f8, iters);
    double c12 = time_fused3<false>(f12, iters);
    double c16 = time_fused3<false>(f16, iters);
    auto [a, b] = fit(c4, c8, c12, c16);
    std::printf("fused3 IP: plumbing=%.2f cyc, body=%.2f cyc/iter\n", a, b);
  }
  {
    auto f4 = make_fixture(4, rng);
    auto f8 = make_fixture(8, rng);
    auto f12 = make_fixture(12, rng);
    auto f16 = make_fixture(16, rng);
    double c4 = time_fused3<true>(f4, iters);
    double c8 = time_fused3<true>(f8, iters);
    double c12 = time_fused3<true>(f12, iters);
    double c16 = time_fused3<true>(f16, iters);
    auto [a, b] = fit(c4, c8, c12, c16);
    std::printf("fused3 L2: plumbing=%.2f cyc, body=%.2f cyc/iter\n", a, b);
  }
  {
    auto f4 = make_fixture(4, rng);
    auto f8 = make_fixture(8, rng);
    auto f12 = make_fixture(12, rng);
    auto f16 = make_fixture(16, rng);
    double c4 = time_fused1<false>(f4, iters);
    double c8 = time_fused1<false>(f8, iters);
    double c12 = time_fused1<false>(f12, iters);
    double c16 = time_fused1<false>(f16, iters);
    auto [a, b] = fit(c4, c8, c12, c16);
    std::printf("fused1 IP: plumbing=%.2f cyc, body=%.2f cyc/iter\n", a, b);
  }
  {
    auto f4 = make_fixture(4, rng);
    auto f8 = make_fixture(8, rng);
    auto f12 = make_fixture(12, rng);
    auto f16 = make_fixture(16, rng);
    double c4 = time_fused1<true>(f4, iters);
    double c8 = time_fused1<true>(f8, iters);
    double c12 = time_fused1<true>(f12, iters);
    double c16 = time_fused1<true>(f16, iters);
    auto [a, b] = fit(c4, c8, c12, c16);
    std::printf("fused1 L2: plumbing=%.2f cyc, body=%.2f cyc/iter\n", a, b);
  }

  return 0;
}
