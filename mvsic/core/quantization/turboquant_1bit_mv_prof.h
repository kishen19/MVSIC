#pragma once

// Lightweight rdtsc-based cycle counters for the 1BTQ leaf kernel.
// Compiled out unless MVSIC_1BTQ_PROF is defined. When enabled, accumulates
// global counts of cycles in distances_all, qbuf build, per-cloud body,
// 4-panel kernel calls, 1-panel kernel calls, and reduce.
//
// Single global Counters (not thread-local). The bench_seq path is
// single-threaded for the timed loop so contention is irrelevant; we add
// std::atomic<uint64_t> with relaxed ops to remain correct if any caller
// ever runs us under multiple threads.
//
// rdtsc ticks at the invariant TSC frequency (constant_tsc on this CPU,
// 2.3 GHz nominal) regardless of core boost frequency, so "cycles" are
// really TSC ticks; convert via TSC_GHZ env var (default 2.3) for time.
// For comparing fractions (qbuf vs kernel vs per-cloud), the constant-TSC
// scaling is the same across all counters and cancels out.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#ifdef MVSIC_1BTQ_PROF
#include <x86intrin.h>
#endif

namespace mvsic {
namespace turboquant_1bit_mv {
namespace prof {

struct Counters {
  std::atomic<uint64_t> distances_all_cy{0};
  std::atomic<uint64_t> qbuf_build_cy{0};
  std::atomic<uint64_t> per_cloud_cy{0};
  std::atomic<uint64_t> kernel_4p_cy{0};
  std::atomic<uint64_t> kernel_1p_cy{0};
  std::atomic<uint64_t> reduce_cy{0};

  std::atomic<uint64_t> n_distances_all{0};
  std::atomic<uint64_t> n_clouds{0};
  std::atomic<uint64_t> n_4panel_calls{0};
  std::atomic<uint64_t> n_1panel_calls{0};
  std::atomic<uint64_t> n_reduce_calls{0};

  std::atomic<uint64_t> total_panel_bytes{0};
  std::atomic<uint64_t> total_hammings{0};
  std::atomic<uint64_t> total_vpopcntd{0};
};

inline Counters& global() {
  static Counters c;
  return c;
}

inline void reset() {
  Counters& c = global();
  c.distances_all_cy.store(0, std::memory_order_relaxed);
  c.qbuf_build_cy.store(0, std::memory_order_relaxed);
  c.per_cloud_cy.store(0, std::memory_order_relaxed);
  c.kernel_4p_cy.store(0, std::memory_order_relaxed);
  c.kernel_1p_cy.store(0, std::memory_order_relaxed);
  c.reduce_cy.store(0, std::memory_order_relaxed);
  c.n_distances_all.store(0, std::memory_order_relaxed);
  c.n_clouds.store(0, std::memory_order_relaxed);
  c.n_4panel_calls.store(0, std::memory_order_relaxed);
  c.n_1panel_calls.store(0, std::memory_order_relaxed);
  c.n_reduce_calls.store(0, std::memory_order_relaxed);
  c.total_panel_bytes.store(0, std::memory_order_relaxed);
  c.total_hammings.store(0, std::memory_order_relaxed);
  c.total_vpopcntd.store(0, std::memory_order_relaxed);
}

inline double tsc_ghz() {
  const char* s = std::getenv("TSC_GHZ");
  return s ? std::atof(s) : 2.3;
}

inline void print_summary(const char* tag = "1BTQ-prof") {
  Counters& c = global();
  const uint64_t n_da = c.n_distances_all.load(std::memory_order_relaxed);
  const uint64_t n_clouds = c.n_clouds.load(std::memory_order_relaxed);
  if (n_da == 0 && n_clouds == 0) {
    std::cerr << "[" << tag << "] no samples (profiler not active or no distances_all calls)\n";
    return;
  }
  const uint64_t da_cy = c.distances_all_cy.load(std::memory_order_relaxed);
  const uint64_t qb_cy = c.qbuf_build_cy.load(std::memory_order_relaxed);
  const uint64_t pc_cy = c.per_cloud_cy.load(std::memory_order_relaxed);
  const uint64_t k4_cy = c.kernel_4p_cy.load(std::memory_order_relaxed);
  const uint64_t k1_cy = c.kernel_1p_cy.load(std::memory_order_relaxed);
  const uint64_t rd_cy = c.reduce_cy.load(std::memory_order_relaxed);
  const uint64_t n_4p = c.n_4panel_calls.load(std::memory_order_relaxed);
  const uint64_t n_1p = c.n_1panel_calls.load(std::memory_order_relaxed);
  const uint64_t n_rd = c.n_reduce_calls.load(std::memory_order_relaxed);
  const uint64_t pb = c.total_panel_bytes.load(std::memory_order_relaxed);
  const uint64_t hm = c.total_hammings.load(std::memory_order_relaxed);
  const uint64_t vp = c.total_vpopcntd.load(std::memory_order_relaxed);
  const double ghz = tsc_ghz();
  auto cy_to_s = [&](uint64_t cy) { return cy / (ghz * 1e9); };
  auto pct = [&](uint64_t cy) { return da_cy > 0 ? 100.0 * cy / da_cy : 0.0; };
  const double t_da = cy_to_s(da_cy);
  std::cerr << "\n[" << tag << "] (TSC_GHZ=" << ghz << ")\n";
  std::cerr << "  n_distances_all  = " << n_da << "\n";
  std::cerr << "  n_clouds         = " << n_clouds << "\n";
  std::cerr << "  n_4panel_calls   = " << n_4p << "\n";
  std::cerr << "  n_1panel_calls   = " << n_1p << "\n";
  std::cerr << "  n_reduce_calls   = " << n_rd << "\n";
  std::cerr << "  total_panel_B    = " << pb << "  (" << pb / 1e9 << " GB)\n";
  std::cerr << "  total_hammings   = " << hm << "\n";
  std::cerr << "  total_vpopcntd   = " << vp << "\n";
  std::cerr << "  distances_all    = " << t_da << " s\n";
  std::cerr << "  qbuf_build       = " << cy_to_s(qb_cy) << " s (" << pct(qb_cy) << "%)\n";
  std::cerr << "  per_cloud_body   = " << cy_to_s(pc_cy) << " s (" << pct(pc_cy) << "%)\n";
  std::cerr << "  kernel_4p_total  = " << cy_to_s(k4_cy) << " s (" << pct(k4_cy) << "%)\n";
  std::cerr << "  kernel_1p_total  = " << cy_to_s(k1_cy) << " s (" << pct(k1_cy) << "%)\n";
  std::cerr << "  reduce_total     = " << cy_to_s(rd_cy) << " s (" << pct(rd_cy) << "%)\n";
  if (t_da > 0) {
    std::cerr << "  panel GB/s       = " << (pb / 1e9) / t_da << "\n";
    std::cerr << "  hamming/sec      = " << hm / t_da << "\n";
    std::cerr << "  vpopcntd/sec     = " << vp / t_da << "\n";
  }
  if (n_clouds > 0) {
    std::cerr << "  cy/cloud (body)  = " << double(pc_cy) / n_clouds << "\n";
    std::cerr << "  cy/cloud (all)   = " << double(da_cy) / n_clouds << "\n";
  }
  if (n_4p > 0) std::cerr << "  cy/4p_call       = " << double(k4_cy) / n_4p << "\n";
  if (n_1p > 0) std::cerr << "  cy/1p_call       = " << double(k1_cy) / n_1p << "\n";
  if (n_rd > 0) std::cerr << "  cy/reduce_call   = " << double(rd_cy) / n_rd << "\n";
  std::cerr.flush();
}

#ifdef MVSIC_1BTQ_PROF
struct AtExitRegistrar {
  AtExitRegistrar() { std::atexit([]() { print_summary(); }); }
};
inline AtExitRegistrar& at_exit_registrar() {
  static AtExitRegistrar r;
  return r;
}

struct ScopedTimer {
  uint64_t start;
  std::atomic<uint64_t>* dst;
  std::atomic<uint64_t>* count;
  ScopedTimer(std::atomic<uint64_t>* d, std::atomic<uint64_t>* n) : start(__rdtsc()), dst(d), count(n) {
    (void)at_exit_registrar();
  }
  ~ScopedTimer() {
    dst->fetch_add(__rdtsc() - start, std::memory_order_relaxed);
    if (count) count->fetch_add(1, std::memory_order_relaxed);
  }
};
#define MVSIC_1BTQ_PROF_TIMER(field, count_field)                                       \
  ::mvsic::turboquant_1bit_mv::prof::ScopedTimer __mvsic_prof_##field##_##__LINE__(     \
      &::mvsic::turboquant_1bit_mv::prof::global().field,                               \
      &::mvsic::turboquant_1bit_mv::prof::global().count_field)
#define MVSIC_1BTQ_PROF_TIMER_NO_COUNT(field)                                           \
  ::mvsic::turboquant_1bit_mv::prof::ScopedTimer __mvsic_prof_##field##_##__LINE__(     \
      &::mvsic::turboquant_1bit_mv::prof::global().field, nullptr)
#define MVSIC_1BTQ_PROF_ADD(field, value)                                               \
  do {                                                                                  \
    ::mvsic::turboquant_1bit_mv::prof::global().field.fetch_add(                        \
        (value), std::memory_order_relaxed);                                            \
  } while (0)
#define MVSIC_1BTQ_PROF_INC(field)                                                      \
  do {                                                                                  \
    ::mvsic::turboquant_1bit_mv::prof::global().field.fetch_add(                        \
        1, std::memory_order_relaxed);                                                  \
  } while (0)
#else
#define MVSIC_1BTQ_PROF_TIMER(field, count_field) ((void)0)
#define MVSIC_1BTQ_PROF_TIMER_NO_COUNT(field) ((void)0)
#define MVSIC_1BTQ_PROF_ADD(field, value) ((void)0)
#define MVSIC_1BTQ_PROF_INC(field) ((void)0)
#endif

}  // namespace prof
}  // namespace turboquant_1bit_mv
}  // namespace mvsic
