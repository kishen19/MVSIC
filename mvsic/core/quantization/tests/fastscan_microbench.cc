#include <benchmark/benchmark.h>
#include <immintrin.h>
#include <vector>
#include <memory>
#include <random>
#include <cstdint>

#include "parlay/parallel.h"
#include "parlay/sequence.h"
#include "mvsic/core/quantization/fastscan.h"

// --- Global Mock Data ---
constexpr uint32_t NUM_BLOCKS_A = 32;

struct MockStrip {
  // 64 vectors * 32 blocks / 2 (since 4-bit codes) = 1024 bytes per strip.
  // We align to 64 bytes to mimic ParlayLib's sequence alignment.
  alignas(64) uint8_t packed_codes[NUM_BLOCKS_A * 32];
  alignas(64) uint8_t int_lut[NUM_BLOCKS_A * 16];

  MockStrip() {
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint16_t> dist(0, 15);
    for (size_t i = 0; i < sizeof(packed_codes); ++i)
      packed_codes[i] = dist(rng);
    for (size_t i = 0; i < sizeof(int_lut); ++i)
      int_lut[i] = dist(rng);
  }
};

MockStrip global_data;

// ==============================================================================
// EXPERIMENT 1: The Accumulation Bottleneck (Extract vs. Unpack)
// ==============================================================================

// Method A: The current implementation using cross-lane extraction
static void BM_AVX2_Accumulate_Extract(benchmark::State& state) {
  const uint8_t* codes_ptr = global_data.packed_codes;
  const uint8_t* lut_ptr = global_data.int_lut;
  const __m256i low_mask = _mm256_set1_epi8(0x0F);

  for (auto _ : state) {
    __m256i acc_even_lo = _mm256_setzero_si256();
    __m256i acc_even_hi = _mm256_setzero_si256();
    const uint8_t* c_ptr = codes_ptr;

    for (uint32_t b = 0; b < NUM_BLOCKS_A; ++b) {
      const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c_ptr));
      c_ptr += 32;

      const __m256i codes_even = _mm256_and_si256(packed, low_mask);
      const __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lut_ptr + b * 16));
      const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);

      const __m256i scores_even_u8 = _mm256_shuffle_epi8(lut256, codes_even);

      // The instruction under test
      acc_even_lo = _mm256_add_epi16(acc_even_lo,
                                     _mm256_cvtepu8_epi16(_mm256_castsi256_si128(scores_even_u8)));
      acc_even_hi = _mm256_add_epi16(
          acc_even_hi, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(scores_even_u8, 1)));
    }
    benchmark::DoNotOptimize(acc_even_lo);
    benchmark::DoNotOptimize(acc_even_hi);
    benchmark::ClobberMemory();
  }
}

// Method B: The proposed implementation using in-lane unpack
static void BM_AVX2_Accumulate_Unpack(benchmark::State& state) {
  const uint8_t* codes_ptr = global_data.packed_codes;
  const uint8_t* lut_ptr = global_data.int_lut;
  const __m256i low_mask = _mm256_set1_epi8(0x0F);
  const __m256i zero = _mm256_setzero_si256();  // hoisted out of loop

  for (auto _ : state) {
    __m256i acc_even_lo = _mm256_setzero_si256();
    __m256i acc_even_hi = _mm256_setzero_si256();
    const uint8_t* c_ptr = codes_ptr;

    for (uint32_t b = 0; b < NUM_BLOCKS_A; ++b) {
      const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c_ptr));
      c_ptr += 32;

      const __m256i codes_even = _mm256_and_si256(packed, low_mask);
      const __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lut_ptr + b * 16));
      const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);

      const __m256i scores_even_u8 = _mm256_shuffle_epi8(lut256, codes_even);

      // The instruction under test
      acc_even_lo = _mm256_add_epi16(acc_even_lo, _mm256_unpacklo_epi8(scores_even_u8, zero));
      acc_even_hi = _mm256_add_epi16(acc_even_hi, _mm256_unpackhi_epi8(scores_even_u8, zero));
    }
    benchmark::DoNotOptimize(acc_even_lo);
    benchmark::DoNotOptimize(acc_even_hi);
    benchmark::ClobberMemory();
  }
}

// ==============================================================================
// EXPERIMENT 2: Memory Alignment Constraints (LoadU vs. Load + Assume Aligned)
// ==============================================================================

// Method A: Standard unaligned load (compiler plays it safe)
static void BM_AVX2_Load_Unaligned(benchmark::State& state) {
  const uint8_t* codes_ptr = global_data.packed_codes;

  for (auto _ : state) {
    __m256i sum = _mm256_setzero_si256();
    const uint8_t* c_ptr = codes_ptr;
    for (uint32_t b = 0; b < NUM_BLOCKS_A; ++b) {
      // Unaligned load
      const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c_ptr));
      sum = _mm256_add_epi8(sum, packed);
      c_ptr += 32;
    }
    benchmark::DoNotOptimize(sum);
    benchmark::ClobberMemory();
  }
}

// Method B: C++20 aligned load guarantee
static void BM_AVX2_Load_AssumeAligned(benchmark::State& state) {
  const uint8_t* codes_ptr = global_data.packed_codes;

  for (auto _ : state) {
    __m256i sum = _mm256_setzero_si256();
    // C++20 feature: explicit alignment guarantee
    const uint8_t* aligned_ptr = std::assume_aligned<32>(codes_ptr);

    for (uint32_t b = 0; b < NUM_BLOCKS_A; ++b) {
      // Aligned load
      const __m256i packed = _mm256_load_si256(reinterpret_cast<const __m256i*>(aligned_ptr));
      sum = _mm256_add_epi8(sum, packed);
      aligned_ptr += 32;
    }
    benchmark::DoNotOptimize(sum);
    benchmark::ClobberMemory();
  }
}

// ==============================================================================
// EXPERIMENT 3: Full Function Macro-Benchmark
// ==============================================================================
// --- Mock Context for Full Function Tests ---
struct MockQuery {
  alignas(64) uint8_t int_lut[NUM_BLOCKS_A * 16];
  float min_dist = 1.2f;
  float scale = 0.05f;
  uint32_t num_blocks = NUM_BLOCKS_A;

  MockQuery() {
    std::mt19937 rng(1337);
    std::uniform_int_distribution<uint16_t> dist(0, 255);
    for (size_t i = 0; i < sizeof(int_lut); ++i)
      int_lut[i] = dist(rng);
  }

  inline float decode(uint16_t int_dist) const {
    return (min_dist * static_cast<float>(num_blocks)) + (static_cast<float>(int_dist) * scale);
  }
};

MockQuery global_query;
alignas(64) float global_results[64];

// Method A: Original Full Function (Cross-Lane Extracts + Unaligned Loads)
static void BM_Full_Scan_Original(benchmark::State& state) {
  const uint8_t* codes_ptr = global_data.packed_codes;
  const MockQuery& q = global_query;
  float* results = global_results;

  const __m256i low_mask = _mm256_set1_epi8(0x0F);

  for (auto _ : state) {
    __m256i acc_even_lo = _mm256_setzero_si256();
    __m256i acc_even_hi = _mm256_setzero_si256();
    __m256i acc_odd_lo = _mm256_setzero_si256();
    __m256i acc_odd_hi = _mm256_setzero_si256();
    const uint8_t* c_ptr = codes_ptr;

    for (uint32_t b = 0; b < q.num_blocks; ++b) {
      const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c_ptr));
      c_ptr += 32;
      const __m256i codes_even = _mm256_and_si256(packed, low_mask);
      const __m256i codes_odd = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);

      const __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&q.int_lut[b * 16]));
      const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);
      const __m256i scores_even_u8 = _mm256_shuffle_epi8(lut256, codes_even);
      const __m256i scores_odd_u8 = _mm256_shuffle_epi8(lut256, codes_odd);

      acc_even_lo = _mm256_add_epi16(acc_even_lo,
                                     _mm256_cvtepu8_epi16(_mm256_castsi256_si128(scores_even_u8)));
      acc_even_hi = _mm256_add_epi16(
          acc_even_hi, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(scores_even_u8, 1)));
      acc_odd_lo =
          _mm256_add_epi16(acc_odd_lo, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(scores_odd_u8)));
      acc_odd_hi = _mm256_add_epi16(
          acc_odd_hi, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(scores_odd_u8, 1)));
    }

    alignas(32) uint16_t raw_even[32];
    alignas(32) uint16_t raw_odd[32];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(raw_even), acc_even_lo);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(raw_even + 16), acc_even_hi);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(raw_odd), acc_odd_lo);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(raw_odd + 16), acc_odd_hi);

    for (int i = 0; i < 32; ++i) {
      results[i * 2] = q.decode(raw_even[i]);
      results[i * 2 + 1] = q.decode(raw_odd[i]);
    }

    benchmark::DoNotOptimize(results);
    benchmark::ClobberMemory();
  }
}

// Method B: Optimized Full Function (In-Lane Unpack + Aligned Loads)
static void BM_Full_Scan_Optimized(benchmark::State& state) {
  const uint8_t* codes_ptr = global_data.packed_codes;
  const MockQuery& q = global_query;
  float* results = global_results;

  const __m256i low_mask = _mm256_set1_epi8(0x0F);
  const __m256i zero = _mm256_setzero_si256();

  for (auto _ : state) {
    __m256i acc_even_lo = _mm256_setzero_si256();
    __m256i acc_even_hi = _mm256_setzero_si256();
    __m256i acc_odd_lo = _mm256_setzero_si256();
    __m256i acc_odd_hi = _mm256_setzero_si256();

    const uint8_t* aligned_codes = std::assume_aligned<32>(codes_ptr);

    for (uint32_t b = 0; b < q.num_blocks; ++b) {
      const __m256i packed = _mm256_load_si256(reinterpret_cast<const __m256i*>(aligned_codes));
      aligned_codes += 32;
      const __m256i codes_even = _mm256_and_si256(packed, low_mask);
      const __m256i codes_odd = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);

      const __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&q.int_lut[b * 16]));
      const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);
      const __m256i scores_even_u8 = _mm256_shuffle_epi8(lut256, codes_even);
      const __m256i scores_odd_u8 = _mm256_shuffle_epi8(lut256, codes_odd);

      acc_even_lo = _mm256_add_epi16(acc_even_lo, _mm256_unpacklo_epi8(scores_even_u8, zero));
      acc_even_hi = _mm256_add_epi16(acc_even_hi, _mm256_unpackhi_epi8(scores_even_u8, zero));
      acc_odd_lo = _mm256_add_epi16(acc_odd_lo, _mm256_unpacklo_epi8(scores_odd_u8, zero));
      acc_odd_hi = _mm256_add_epi16(acc_odd_hi, _mm256_unpackhi_epi8(scores_odd_u8, zero));
    }

    alignas(32) uint16_t raw_even_lo[16], raw_even_hi[16];
    alignas(32) uint16_t raw_odd_lo[16], raw_odd_hi[16];

    _mm256_storeu_si256(reinterpret_cast<__m256i*>(raw_even_lo), acc_even_lo);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(raw_even_hi), acc_even_hi);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(raw_odd_lo), acc_odd_lo);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(raw_odd_hi), acc_odd_hi);

    for (int i = 0; i < 8; ++i) {
      results[i * 2] = q.decode(raw_even_lo[i]);
      results[(i + 8) * 2] = q.decode(raw_even_hi[i]);
      results[(i + 16) * 2] = q.decode(raw_even_lo[i + 8]);
      results[(i + 24) * 2] = q.decode(raw_even_hi[i + 8]);

      results[i * 2 + 1] = q.decode(raw_odd_lo[i]);
      results[(i + 8) * 2 + 1] = q.decode(raw_odd_hi[i]);
      results[(i + 16) * 2 + 1] = q.decode(raw_odd_lo[i + 8]);
      results[(i + 24) * 2 + 1] = q.decode(raw_odd_hi[i + 8]);
    }

    benchmark::DoNotOptimize(results);
    benchmark::ClobberMemory();
  }
}

// ==============================================================================
// EXPERIMENT 4: The Aligned Load Full Function Test
// ==============================================================================

// Method A: Full Function with Unaligned Loads (The baseline)
static void BM_Full_Scan_Extract_Unaligned(benchmark::State& state) {
  const uint8_t* codes_ptr = global_data.packed_codes;
  const MockQuery& q = global_query;
  float* results = global_results;
  const __m256i low_mask = _mm256_set1_epi8(0x0F);

  for (auto _ : state) {
    __m256i acc_even_lo = _mm256_setzero_si256();
    __m256i acc_even_hi = _mm256_setzero_si256();
    __m256i acc_odd_lo = _mm256_setzero_si256();
    __m256i acc_odd_hi = _mm256_setzero_si256();
    const uint8_t* c_ptr = codes_ptr;

    for (uint32_t b = 0; b < q.num_blocks; ++b) {
      // UNALIGNED LOAD INSTRUCTION
      const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c_ptr));
      c_ptr += 32;

      const __m256i codes_even = _mm256_and_si256(packed, low_mask);
      const __m256i codes_odd = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);

      const __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&q.int_lut[b * 16]));
      const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);
      const __m256i scores_even_u8 = _mm256_shuffle_epi8(lut256, codes_even);
      const __m256i scores_odd_u8 = _mm256_shuffle_epi8(lut256, codes_odd);

      acc_even_lo = _mm256_add_epi16(acc_even_lo,
                                     _mm256_cvtepu8_epi16(_mm256_castsi256_si128(scores_even_u8)));
      acc_even_hi = _mm256_add_epi16(
          acc_even_hi, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(scores_even_u8, 1)));
      acc_odd_lo =
          _mm256_add_epi16(acc_odd_lo, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(scores_odd_u8)));
      acc_odd_hi = _mm256_add_epi16(
          acc_odd_hi, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(scores_odd_u8, 1)));
    }

    alignas(32) uint16_t raw_even[32];
    alignas(32) uint16_t raw_odd[32];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(raw_even), acc_even_lo);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(raw_even + 16), acc_even_hi);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(raw_odd), acc_odd_lo);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(raw_odd + 16), acc_odd_hi);

    for (int i = 0; i < 32; ++i) {
      results[i * 2] = q.decode(raw_even[i]);
      results[i * 2 + 1] = q.decode(raw_odd[i]);
    }

    benchmark::DoNotOptimize(results);
    benchmark::ClobberMemory();
  }
}

// Method B: Full Function with Aligned Loads (The optimized alignment)
static void BM_Full_Scan_Extract_Aligned(benchmark::State& state) {
  const uint8_t* codes_ptr = global_data.packed_codes;
  const MockQuery& q = global_query;
  float* results = global_results;
  const __m256i low_mask = _mm256_set1_epi8(0x0F);

  for (auto _ : state) {
    __m256i acc_even_lo = _mm256_setzero_si256();
    __m256i acc_even_hi = _mm256_setzero_si256();
    __m256i acc_odd_lo = _mm256_setzero_si256();
    __m256i acc_odd_hi = _mm256_setzero_si256();

    // C++20 ALIGNMENT GUARANTEE
    const uint8_t* aligned_codes = std::assume_aligned<32>(codes_ptr);

    for (uint32_t b = 0; b < q.num_blocks; ++b) {
      // ALIGNED LOAD INSTRUCTION
      const __m256i packed = _mm256_load_si256(reinterpret_cast<const __m256i*>(aligned_codes));
      aligned_codes += 32;

      const __m256i codes_even = _mm256_and_si256(packed, low_mask);
      const __m256i codes_odd = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);

      const __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&q.int_lut[b * 16]));
      const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);
      const __m256i scores_even_u8 = _mm256_shuffle_epi8(lut256, codes_even);
      const __m256i scores_odd_u8 = _mm256_shuffle_epi8(lut256, codes_odd);

      acc_even_lo = _mm256_add_epi16(acc_even_lo,
                                     _mm256_cvtepu8_epi16(_mm256_castsi256_si128(scores_even_u8)));
      acc_even_hi = _mm256_add_epi16(
          acc_even_hi, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(scores_even_u8, 1)));
      acc_odd_lo =
          _mm256_add_epi16(acc_odd_lo, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(scores_odd_u8)));
      acc_odd_hi = _mm256_add_epi16(
          acc_odd_hi, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(scores_odd_u8, 1)));
    }

    alignas(32) uint16_t raw_even[32];
    alignas(32) uint16_t raw_odd[32];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(raw_even), acc_even_lo);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(raw_even + 16), acc_even_hi);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(raw_odd), acc_odd_lo);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(raw_odd + 16), acc_odd_hi);

    for (int i = 0; i < 32; ++i) {
      results[i * 2] = q.decode(raw_even[i]);
      results[i * 2 + 1] = q.decode(raw_odd[i]);
    }

    benchmark::DoNotOptimize(results);
    benchmark::ClobberMemory();
  }
}

// ==============================================================================
// EXPERIMENT 5: The AVX-512 Heavyweight (Shuffle Path)
// ==============================================================================

#if defined(__AVX512F__) && defined(__AVX512BW__)
static void BM_Full_Scan_AVX512(benchmark::State& state) {
  const uint8_t* codes_ptr = global_data.packed_codes;
  const MockQuery& q = global_query;
  float* results = global_results;

  const __m256i low_mask = _mm256_set1_epi8(0x0F);

  for (auto _ : state) {
    __m512i acc_even = _mm512_setzero_si512();
    __m512i acc_odd = _mm512_setzero_si512();

    // C++20 Alignment Hint
    const uint8_t* aligned_codes = std::assume_aligned<32>(codes_ptr);

    for (uint32_t b = 0; b < q.num_blocks; ++b) {
      const __m256i packed = _mm256_load_si256(reinterpret_cast<const __m256i*>(aligned_codes));
      aligned_codes += 32;

      const __m256i codes_even = _mm256_and_si256(packed, low_mask);
      const __m256i codes_odd = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);

      const __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&q.int_lut[b * 16]));
      const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);

      const __m256i scores_even_u8 = _mm256_shuffle_epi8(lut256, codes_even);
      const __m256i scores_odd_u8 = _mm256_shuffle_epi8(lut256, codes_odd);

      // AVX-512 Magic: One instruction handles the zero-extension natively
      // with zero cross-lane extraction penalties.
      acc_even = _mm512_add_epi16(acc_even, _mm512_cvtepu8_epi16(scores_even_u8));
      acc_odd = _mm512_add_epi16(acc_odd, _mm512_cvtepu8_epi16(scores_odd_u8));
    }

    alignas(64) uint16_t raw_even[32];
    alignas(64) uint16_t raw_odd[32];

    // Two massive 64-byte aligned stores instead of four 32-byte stores
    _mm512_store_si512(reinterpret_cast<__m512i*>(raw_even), acc_even);
    _mm512_store_si512(reinterpret_cast<__m512i*>(raw_odd), acc_odd);

    // Sequential memory access is completely preserved!
    for (int i = 0; i < 32; ++i) {
      results[i * 2] = q.decode(raw_even[i]);
      results[i * 2 + 1] = q.decode(raw_odd[i]);
    }

    benchmark::DoNotOptimize(results);
    benchmark::ClobberMemory();
  }
}
#endif

// BENCHMARK(BM_AVX2_Accumulate_Extract);
// BENCHMARK(BM_AVX2_Accumulate_Unpack);
// BENCHMARK(BM_AVX2_Load_Unaligned);
// BENCHMARK(BM_AVX2_Load_AssumeAligned);
// BENCHMARK(BM_Full_Scan_Original);
// BENCHMARK(BM_Full_Scan_Optimized);
// BENCHMARK(BM_Full_Scan_Extract_Unaligned);
// BENCHMARK(BM_Full_Scan_Extract_Aligned);
// BENCHMARK(BM_Full_Scan_AVX512);

// ==============================================================================
// EXPERIMENT 6 (CORRECTED): ParlayLib Scheduler Granularity Sweep
// ==============================================================================

constexpr size_t MACRO_DB_SIZE = 1000;
constexpr uint32_t DIM = 128;
constexpr uint32_t BLOCK_SIZE = 8;
constexpr uint32_t NUM_BLOCKS = DIM / BLOCK_SIZE;

// Instantiate the ACTUAL classes from your header
mvsic::fastscan::Quantized_Point_Range<std::vector<float>, false> global_real_db;
mvsic::fastscan::Quantized_Query<false> global_real_query(NUM_BLOCKS);
parlay::sequence<float> global_real_out;

// Setup struct to initialize the dummy data once before benchmarks run
struct RealDBSetup {
  RealDBSetup() {
    global_real_db.num_blocks = NUM_BLOCKS;
    global_real_db.n_points = MACRO_DB_SIZE;
    global_real_db.dim = DIM;
    global_real_db.dim_per_block = BLOCK_SIZE;

    size_t n_padded = ((MACRO_DB_SIZE + 63) / 64) * 64;
    size_t num_strips = n_padded / 64;
    size_t strip_stride = NUM_BLOCKS * 32;

    // Allocate and fill with dummy bytes
    global_real_db.packed_codes = parlay::sequence<uint8_t>(num_strips * strip_stride, 0xAF);
    global_real_out.resize(n_padded);

    // Fill query LUT with dummy scales
    for (size_t i = 0; i < NUM_BLOCKS * 16; ++i) {
      global_real_query.int_lut[i] = static_cast<uint8_t>(i % 256);
    }
    global_real_query.scale = 0.5f;
    global_real_query.min_dist = 1.0f;
  }
} setup_instance;

static void BM_Parlay_Granularity_Real(benchmark::State& state) {
  const size_t granularity = state.range(0);

  const auto& db = global_real_db;
  const auto& q = global_real_query;
  float* out = global_real_out.data();

  const size_t n_full_strips = db.n_points / 64;
  const size_t strip_stride = db.num_blocks * 32;

  // --- MANUAL WARMUP: The AVX-512 / AVX2 State Transition ---
  // Hitting the ACTUAL scan_64_chunk kernel to force the CPU into its
  // heavy vector-math power tier and load the L1i instruction cache.
  parlay::parallel_for(
      0, n_full_strips,
      [&](size_t s) {
        const uint8_t* codes_ptr = db.packed_codes.data() + s * strip_stride;
        db.scan_64_chunk(q, codes_ptr, out + s * 64);
      },
      64);

  for (auto _ : state) {
    parlay::parallel_for(
        0, n_full_strips,
        [&](size_t s) {
          const uint8_t* codes_ptr = db.packed_codes.data() + s * strip_stride;
          // Calls your meticulously optimized SIMD kernel
          db.scan_64_chunk(q, codes_ptr, out + s * 64);
        },
        granularity);
    benchmark::ClobberMemory();
  }
}

BENCHMARK(BM_Parlay_Granularity_Real)
    ->RangeMultiplier(2)
    ->Range(1, 4096)
    ->UseRealTime()
    ->MinWarmUpTime(0.5)
    ->MinTime(1.0);

BENCHMARK_MAIN();