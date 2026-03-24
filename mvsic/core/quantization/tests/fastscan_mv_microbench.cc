#include <benchmark/benchmark.h>
#include "mvsic/core/quantization/fastscan_mv.h"  // The exact header you provided

using namespace mvsic::fastscan_mv;

// EXACT PARAMETERS
constexpr uint32_t NUM_BLOCKS = 16;    // 128 dims / 8 block_size
constexpr size_t Q_CLOUD_SIZE = 32;    // 32 query vectors
constexpr size_t DB_CLOUD_SIZE = 128;  // 128 DB vectors (2 strips)

struct ActualBenchSetup {
  Quantized_Query_Point_Cloud<true> q;
  Quantized_Point_Cloud_Set<true> db;

  ActualBenchSetup() {
    // Setup Query Cloud
    q.num_queries = Q_CLOUD_SIZE;
    q.num_blocks = NUM_BLOCKS;
    q.flat_int_luts.resize(Q_CLOUD_SIZE * NUM_BLOCKS * 16, 5);
    q.min_dists.assign(Q_CLOUD_SIZE, 0.0f);
    q.scales.assign(Q_CLOUD_SIZE, 1.0f);

    // Setup DB Cloud Set
    db.num_blocks = NUM_BLOCKS;
    size_t strip_stride = NUM_BLOCKS * 32;
    // 128 vectors = 2 full strips
    db.packed_codes = parlay::sequence<uint8_t>(2 * strip_stride, 0xAB);
    db.offsets = {0, 128};
    db.sizes_unpadded = {128};
    db.ids = {0};
  }
};

ActualBenchSetup g_env;

static void BM_Chamfer_Actual_Library(benchmark::State& state) {
  for (auto _ : state) {
    // Calling the ACTUAL function from your header
    float d = fastscan_mv_chamfer_distance(g_env.q, g_env.db, 0, 128);
    benchmark::DoNotOptimize(d);
  }
}
BENCHMARK(BM_Chamfer_Actual_Library)->UseRealTime();

BENCHMARK_MAIN();
// #include <benchmark/benchmark.h>
// #include <immintrin.h>
// #include <vector>
// #include <memory>

// // User-specified parameters
// constexpr uint32_t NUM_BLOCKS = 16;  // 128D / 8
// constexpr size_t STRIP_STRIDE = NUM_BLOCKS * 32;
// constexpr size_t Q_CLOUD_SIZE = 32;
// constexpr size_t DB_CLOUD_SIZE = 168;
// constexpr size_t NUM_STRIPS = DB_CLOUD_SIZE / 64;  // Exactly 2 strips

// struct L1TestData {
//   alignas(64) uint8_t db[NUM_STRIPS * STRIP_STRIDE];
//   alignas(64) uint8_t q_luts[Q_CLOUD_SIZE * NUM_BLOCKS * 16];

//   L1TestData() {
//     for (auto& b : db)
//       b = 0xAF;
//     for (auto& b : q_luts)
//       b = 0x07;
//   }
// };

// L1TestData g_l1;

// // ==============================================================================
// // TEST 1: Current (Query-Outer) - Touches DB 32 times
// // ==============================================================================
// static void BM_L1_Current(benchmark::State& state) {
//   float* results = (float*)aligned_alloc(64, Q_CLOUD_SIZE * sizeof(float));

//   for (auto _ : state) {
//     for (size_t qi = 0; qi < Q_CLOUD_SIZE; ++qi) {
//       const uint8_t* q_lut = g_l1.q_luts + qi * NUM_BLOCKS * 16;
//       __m512i running_min = _mm512_set1_epi16(0xFFFF);

//       for (size_t s = 0; s < NUM_STRIPS; ++s) {
//         const uint8_t* codes = g_l1.db + s * STRIP_STRIDE;
//         __m512i acc_even = _mm512_setzero_si512();
//         __m512i acc_odd = _mm512_setzero_si512();

//         for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
//           __m256i packed = _mm256_loadu_si256((__m256i*)(codes + b * 32));
//           __m128i lut128 = _mm_loadu_si128((__m128i*)(q_lut + b * 16));
//           __m256i lut256 = _mm256_broadcastsi128_si256(lut128);
//           acc_even = _mm512_add_epi16(
//               acc_even, _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(
//                             lut256, _mm256_and_si256(packed, _mm256_set1_epi8(0x0F)))));
//           acc_odd = _mm512_add_epi16(
//               acc_odd,
//               _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(
//                   lut256, _mm256_and_si256(_mm256_srli_epi16(packed, 4),
//                   _mm256_set1_epi8(0x0F)))));
//         }
//         running_min = _mm512_min_epu16(running_min, acc_even);
//         running_min = _mm512_min_epu16(running_min, acc_odd);
//       }
//       benchmark::DoNotOptimize(running_min);
//     }
//   }
//   free(results);
// }
// BENCHMARK(BM_L1_Current)->UseRealTime();

// // ==============================================================================
// // TEST 2: Inverted (Database-Outer) - Touches DB 1 time
// // ==============================================================================
// static void BM_L1_Inverted(benchmark::State& state) {
//   for (auto _ : state) {
//     alignas(64) __m512i all_mins[Q_CLOUD_SIZE];
//     for (int i = 0; i < Q_CLOUD_SIZE; ++i)
//       all_mins[i] = _mm512_set1_epi16(0xFFFF);

//     for (size_t s = 0; s < NUM_STRIPS; ++s) {
//       const uint8_t* codes_base = g_l1.db + s * STRIP_STRIDE;

//       for (size_t qi = 0; qi < Q_CLOUD_SIZE; ++qi) {
//         const uint8_t* q_lut = g_l1.q_luts + qi * NUM_BLOCKS * 16;
//         __m512i acc_even = _mm512_setzero_si512();
//         __m512i acc_odd = _mm512_setzero_si512();
//         const uint8_t* b_codes = codes_base;

//         for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
//           __m256i packed = _mm256_loadu_si256((__m256i*)b_codes);
//           b_codes += 32;
//           __m128i lut128 = _mm_loadu_si128((__m128i*)(q_lut + b * 16));
//           __m256i lut256 = _mm256_broadcastsi128_si256(lut128);
//           acc_even = _mm512_add_epi16(
//               acc_even, _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(
//                             lut256, _mm256_and_si256(packed, _mm256_set1_epi8(0x0F)))));
//           acc_odd = _mm512_add_epi16(
//               acc_odd,
//               _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(
//                   lut256, _mm256_and_si256(_mm256_srli_epi16(packed, 4),
//                   _mm256_set1_epi8(0x0F)))));
//         }
//         all_mins[qi] = _mm512_min_epu16(all_mins[qi], acc_even);
//         all_mins[qi] = _mm512_min_epu16(all_mins[qi], acc_odd);
//       }
//     }
//     benchmark::DoNotOptimize(all_mins);
//   }
// }
// BENCHMARK(BM_L1_Inverted)->UseRealTime();

// BENCHMARK_MAIN();