#include <benchmark/benchmark.h>
#include <immintrin.h>
#include <vector>
#include <cstring>

// YOUR EXACT SETTINGS
constexpr uint32_t NUM_BYTES = 64; // 128D / 2
constexpr size_t STRIP_STRIDE = 64 * NUM_BYTES;
constexpr size_t BASE_LANE = 56;   // 8 bytes in S0, 8 bytes in S1
constexpr size_t K_VNNI_POINTS = 16;

struct RealGatherData {
    alignas(64) uint8_t db_strips[2 * STRIP_STRIDE];
    alignas(64) uint8_t panel[NUM_BYTES * K_VNNI_POINTS * 4];

    RealGatherData() {
        for (int i = 0; i < 2 * STRIP_STRIDE; ++i) db_strips[i] = i % 256;
    }
};

RealGatherData g_real;

// ==============================================================================
// VERSION 1: Current Memcpy + Scalar Loop
// ==============================================================================
static void BM_Turbo_Gather_Memcpy(benchmark::State& state) {
    const uint8_t* s0 = g_real.db_strips;
    const uint8_t* s1 = g_real.db_strips + STRIP_STRIDE;
    uint8_t* panel = g_real.panel;
    uint8_t gather_buf[NUM_BYTES * 64]; 

    for (auto _ : state) {
        // The "Scalar Cliff"
        for (size_t j = 0; j < NUM_BYTES; ++j) {
            std::memcpy(gather_buf + j * 64, s0 + j * 64 + BASE_LANE, 8);
            std::memcpy(gather_buf + j * 64 + 8, s1 + j * 64, 8);
        }
        // Simplified version of your decode logic to ensure work is done
        for (size_t j = 0; j < NUM_BYTES; ++j) {
            __m128i v = _mm_loadu_si128((__m128i*)(gather_buf + j * 64));
            _mm_storeu_si128((__m128i*)(panel + j * 64), v);
        }
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Turbo_Gather_Memcpy)->UseRealTime();

// ==============================================================================
// VERSION 2: Proposed AVX-512 Masked Load (No intermediate buffer)
// ==============================================================================
#ifdef __AVX512F__
static void BM_Turbo_Gather_AVX512(benchmark::State& state) {
    const uint8_t* s0 = g_real.db_strips;
    const uint8_t* s1 = g_real.db_strips + STRIP_STRIDE;
    uint8_t* panel = g_real.panel;

    const __mmask16 m_lo = 0x00FF; 
    const __mmask16 m_hi = 0xFF00; 

    for (auto _ : state) {
        for (size_t j = 0; j < NUM_BYTES; ++j) {
            // Load and merge directly in one sequence
            __m128i v = _mm_maskz_loadu_epi8(m_lo, s0 + j * 64 + BASE_LANE);
            // Anchor logic: s1 - 8 aligns s1[0] to register[8]
            v = _mm_mask_loadu_epi8(v, m_hi, s1 + j * 64 - 8);
            
            _mm_storeu_si128((__m128i*)(panel + j * 64), v);
        }
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Turbo_Gather_AVX512)->UseRealTime();
#endif

BENCHMARK_MAIN();