#include <benchmark/benchmark.h>
#include <immintrin.h>
#include <vector>
#include <random>
#include <cstdint>

// --- Benchmark Parameters ---
constexpr uint32_t NUM_POINTS = 65536;
constexpr uint32_t NUM_BLOCKS = 64;
constexpr uint32_t K = 16;

// ============================================================================
// Kernel 1: Original AVX-512 Striped Kernel (1-Block per Iteration)
// ============================================================================
#ifdef __AVX512F__
void scan_original_avx512(const uint8_t* codes, const uint8_t* lut, uint16_t* results) {
  const size_t strip_stride = NUM_BLOCKS * 32;
  const size_t n_full_strips = NUM_POINTS / 64;
  const __m256i low_mask = _mm256_set1_epi8(0x0F);

  for (size_t s = 0; s < n_full_strips; ++s) {
    const uint8_t* codes_ptr = codes + s * strip_stride;

    __m512i acc_even = _mm512_setzero_si512();
    __m512i acc_odd = _mm512_setzero_si512();

    for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
      const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_ptr));
      codes_ptr += 32;

      const __m256i codes_even = _mm256_and_si256(packed, low_mask);
      const __m256i codes_odd = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);

      const __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&lut[b * K]));
      const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);

      const __m256i scores_even_u8 = _mm256_shuffle_epi8(lut256, codes_even);
      const __m256i scores_odd_u8 = _mm256_shuffle_epi8(lut256, codes_odd);

      acc_even = _mm512_add_epi16(acc_even, _mm512_cvtepu8_epi16(scores_even_u8));
      acc_odd = _mm512_add_epi16(acc_odd, _mm512_cvtepu8_epi16(scores_odd_u8));
    }

    _mm512_storeu_si512(reinterpret_cast<__m512i*>(&results[s * 64]), acc_even);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(&results[s * 64 + 32]), acc_odd);
  }
}
#endif

// ============================================================================
// Kernel 2: VBMI 2-Block (Unrolled 256-bit registers)
// ============================================================================
#ifdef __AVX512VBMI__
void scan_vbmi_2block(const uint8_t* codes, const uint8_t* lut, uint16_t* results) {
  const size_t strip_stride = (NUM_BLOCKS / 2) * 64;
  const size_t n_full_strips = NUM_POINTS / 64;

  const __m256i low_mask = _mm256_set1_epi8(0x0F);
  const __m256i ones = _mm256_set1_epi8(1);
  const __m256i offset_mask = _mm256_set1_epi16(0x1000);

  for (size_t s = 0; s < n_full_strips; ++s) {
    const uint8_t* codes_ptr = codes + s * strip_stride;

    __m256i acc_even1 = _mm256_setzero_si256();
    __m256i acc_odd1 = _mm256_setzero_si256();
    __m256i acc_even2 = _mm256_setzero_si256();
    __m256i acc_odd2 = _mm256_setzero_si256();

    for (uint32_t b = 0; b < NUM_BLOCKS; b += 2) {
      const __m256i packed1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_ptr));
      const __m256i packed2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_ptr + 32));
      codes_ptr += 64;

      __m256i c_even1 = _mm256_or_si256(_mm256_and_si256(packed1, low_mask), offset_mask);
      __m256i c_odd1 =
          _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(packed1, 4), low_mask), offset_mask);
      __m256i c_even2 = _mm256_or_si256(_mm256_and_si256(packed2, low_mask), offset_mask);
      __m256i c_odd2 =
          _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(packed2, 4), low_mask), offset_mask);

      const __m256i lut_combined =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&lut[b * K]));

      __m256i s_even1 = _mm256_permutexvar_epi8(c_even1, lut_combined);
      __m256i s_odd1 = _mm256_permutexvar_epi8(c_odd1, lut_combined);
      __m256i s_even2 = _mm256_permutexvar_epi8(c_even2, lut_combined);
      __m256i s_odd2 = _mm256_permutexvar_epi8(c_odd2, lut_combined);

      acc_even1 = _mm256_add_epi16(acc_even1, _mm256_maddubs_epi16(s_even1, ones));
      acc_odd1 = _mm256_add_epi16(acc_odd1, _mm256_maddubs_epi16(s_odd1, ones));
      acc_even2 = _mm256_add_epi16(acc_even2, _mm256_maddubs_epi16(s_even2, ones));
      acc_odd2 = _mm256_add_epi16(acc_odd2, _mm256_maddubs_epi16(s_odd2, ones));
    }

    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&results[s * 64]), acc_even1);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&results[s * 64 + 16]), acc_odd1);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&results[s * 64 + 32]), acc_even2);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&results[s * 64 + 48]), acc_odd2);
  }
}
#endif
// ============================================================================
// Kernel 3: VBMI 512-bit (Native 512-bit, 2-Block)
// ============================================================================
#ifdef __AVX512VBMI__
void scan_vbmi_512(const uint8_t* codes, const uint8_t* lut, uint16_t* results) {
  const size_t strip_stride = (NUM_BLOCKS / 2) * 64;
  const size_t n_full_strips = NUM_POINTS / 64;

  const __m512i low_mask = _mm512_set1_epi8(0x0F);
  const __m512i offset_mask = _mm512_set1_epi16(0x1000);
  const __m512i ones = _mm512_set1_epi8(1);

  for (size_t s = 0; s < n_full_strips; ++s) {
    const uint8_t* codes_ptr = codes + s * strip_stride;

    __m512i acc_even = _mm512_setzero_si512();
    __m512i acc_odd = _mm512_setzero_si512();

    for (uint32_t b = 0; b < NUM_BLOCKS; b += 2) {
      const __m512i packed = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes_ptr));
      codes_ptr += 64;

      __m512i codes_even = _mm512_or_si512(_mm512_and_si512(packed, low_mask), offset_mask);
      __m512i codes_odd =
          _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(packed, 4), low_mask), offset_mask);

      const __m256i lut256 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&lut[b * K]));
      const __m512i lut512 = _mm512_castsi256_si512(lut256);

      __m512i scores_even_u8 = _mm512_permutexvar_epi8(codes_even, lut512);
      __m512i scores_odd_u8 = _mm512_permutexvar_epi8(codes_odd, lut512);

      acc_even = _mm512_add_epi16(acc_even, _mm512_maddubs_epi16(scores_even_u8, ones));
      acc_odd = _mm512_add_epi16(acc_odd, _mm512_maddubs_epi16(scores_odd_u8, ones));
    }

    _mm512_storeu_si512(reinterpret_cast<__m512i*>(&results[s * 64]), acc_even);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(&results[s * 64 + 32]), acc_odd);
  }
}
#endif
// ============================================================================
// Kernel 4: Zero LUT Load (Fully Unrolled 512-bit VBMI)
// ============================================================================
#ifdef __AVX512VBMI__
void scan_vbmi_512_unrolled(const uint8_t* codes, const uint8_t* lut, uint16_t* results) {
  const size_t strip_stride = (NUM_BLOCKS / 2) * 64;
  const size_t n_full_strips = NUM_POINTS / 64;

  const __m512i low_mask = _mm512_set1_epi8(0x0F);
  const __m512i offset_mask = _mm512_set1_epi16(0x1000);
  const __m512i ones = _mm512_set1_epi8(1);

  // PRE-LOAD THE ENTIRE LUT INTO REGISTERS
  __m512i preloaded_luts[NUM_BLOCKS / 2];
  for (uint32_t i = 0; i < NUM_BLOCKS / 2; ++i) {
    preloaded_luts[i] = _mm512_castsi256_si512(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&lut[i * 2 * K])));
  }

  for (size_t s = 0; s < n_full_strips; ++s) {
    const uint8_t* codes_ptr = codes + s * strip_stride;
    __m512i acc_even = _mm512_setzero_si512();
    __m512i acc_odd = _mm512_setzero_si512();

#pragma GCC unroll 8
    for (uint32_t b = 0; b < NUM_BLOCKS; b += 2) {
      const __m512i packed = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes_ptr));
      codes_ptr += 64;

      __m512i codes_even = _mm512_or_si512(_mm512_and_si512(packed, low_mask), offset_mask);
      __m512i codes_odd =
          _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(packed, 4), low_mask), offset_mask);

      __m512i scores_even_u8 = _mm512_permutexvar_epi8(codes_even, preloaded_luts[b / 2]);
      __m512i scores_odd_u8 = _mm512_permutexvar_epi8(codes_odd, preloaded_luts[b / 2]);

      acc_even = _mm512_add_epi16(acc_even, _mm512_maddubs_epi16(scores_even_u8, ones));
      acc_odd = _mm512_add_epi16(acc_odd, _mm512_maddubs_epi16(scores_odd_u8, ones));
    }
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(&results[s * 64]), acc_even);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(&results[s * 64 + 32]), acc_odd);
  }
}
#endif

// ============================================================================
// Kernel 5: AVX512-VNNI (4-Block, 32-bit Native Accumulation)
// ============================================================================
#ifdef __AVX512VNNI__
void scan_vnni_32bit(const uint8_t* codes, const uint8_t* lut, uint32_t* results_32) {
  const size_t strip_stride = (NUM_BLOCKS / 4) * 64;
  const size_t n_full_strips = NUM_POINTS / 32;

  const __m512i low_mask = _mm512_set1_epi8(0x0F);
  const __m512i ones = _mm512_set1_epi8(1);
  const __m512i offset_mask = _mm512_set1_epi32(0x30201000);

  for (size_t s = 0; s < n_full_strips; ++s) {
    const uint8_t* codes_ptr = codes + s * strip_stride;

    __m512i acc_even = _mm512_setzero_si512();
    __m512i acc_odd = _mm512_setzero_si512();

#pragma GCC unroll 8
    for (uint32_t b = 0; b < NUM_BLOCKS; b += 4) {
      const __m512i packed = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes_ptr));
      codes_ptr += 64;

      __m512i codes_even = _mm512_or_si512(_mm512_and_si512(packed, low_mask), offset_mask);
      __m512i codes_odd =
          _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(packed, 4), low_mask), offset_mask);

      const __m512i lut64 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&lut[b * K]));

      __m512i scores_even_u8 = _mm512_permutexvar_epi8(codes_even, lut64);
      __m512i scores_odd_u8 = _mm512_permutexvar_epi8(codes_odd, lut64);

      acc_even = _mm512_dpbusd_epi32(acc_even, scores_even_u8, ones);
      acc_odd = _mm512_dpbusd_epi32(acc_odd, scores_odd_u8, ones);
    }

    _mm512_storeu_si512(reinterpret_cast<__m512i*>(&results_32[s * 32]), acc_even);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(&results_32[s * 32 + 16]), acc_odd);
  }
}
#endif

// ============================================================================
// Kernel 6: Zero LUT Load (Fully Unrolled VNNI 32-bit)
// ============================================================================
#ifdef __AVX512VNNI__
void scan_vnni_32bit_unrolled(const uint8_t* codes, const uint8_t* lut, uint32_t* results_32) {
  const size_t strip_stride = (NUM_BLOCKS / 4) * 64;
  const size_t n_full_strips = NUM_POINTS / 32;

  const __m512i low_mask = _mm512_set1_epi8(0x0F);
  const __m512i ones = _mm512_set1_epi8(1);
  const __m512i offset_mask = _mm512_set1_epi32(0x30201000);

  // PRE-LOAD THE ENTIRE LUT INTO REGISTERS (Only needs 4 registers for 16 blocks!)
  __m512i preloaded_luts[NUM_BLOCKS / 4];
  for (uint32_t i = 0; i < NUM_BLOCKS / 4; ++i) {
    preloaded_luts[i] = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&lut[i * 4 * K]));
  }

  for (size_t s = 0; s < n_full_strips; ++s) {
    const uint8_t* codes_ptr = codes + s * strip_stride;

    __m512i acc_even = _mm512_setzero_si512();
    __m512i acc_odd = _mm512_setzero_si512();

#pragma GCC unroll 8
    for (uint32_t b = 0; b < NUM_BLOCKS; b += 4) {
      const __m512i packed = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes_ptr));
      codes_ptr += 64;

      __m512i codes_even = _mm512_or_si512(_mm512_and_si512(packed, low_mask), offset_mask);
      __m512i codes_odd =
          _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(packed, 4), low_mask), offset_mask);

      // ZERO L1 CACHE FETCHES
      __m512i scores_even_u8 = _mm512_permutexvar_epi8(codes_even, preloaded_luts[b / 4]);
      __m512i scores_odd_u8 = _mm512_permutexvar_epi8(codes_odd, preloaded_luts[b / 4]);

      // 1 Instruction: Multiplies by 1, adds 4 blocks, accumulates to 32-bit
      acc_even = _mm512_dpbusd_epi32(acc_even, scores_even_u8, ones);
      acc_odd = _mm512_dpbusd_epi32(acc_odd, scores_odd_u8, ones);
    }

    _mm512_storeu_si512(reinterpret_cast<__m512i*>(&results_32[s * 32]), acc_even);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(&results_32[s * 32 + 16]), acc_odd);
  }
}
#endif
// ============================================================================
// Kernel 7: The Final Boss (4x Accumulator ILP Unrolled VNNI)
// ============================================================================
#ifdef __AVX512VNNI__
void scan_vnni_32bit_unrolled_4x(const uint8_t* codes, const uint8_t* lut, uint32_t* results_32) {
  // We process 64 points per loop -> 128 bytes of codes per block-chunk
  const size_t strip_stride = (NUM_BLOCKS / 4) * 128;
  const size_t n_full_strips = NUM_POINTS / 64;

  const __m512i low_mask = _mm512_set1_epi8(0x0F);
  const __m512i ones = _mm512_set1_epi8(1);
  const __m512i offset_mask = _mm512_set1_epi32(0x30201000);

  // PRE-LOAD THE ENTIRE LUT INTO REGISTERS
  __m512i preloaded_luts[NUM_BLOCKS / 4];
  for (uint32_t i = 0; i < NUM_BLOCKS / 4; ++i) {
    preloaded_luts[i] = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&lut[i * 4 * K]));
  }

  for (size_t s = 0; s < n_full_strips; ++s) {
    const uint8_t* codes_ptr = codes + s * strip_stride;

    // 4 Independent Dependency Chains to hide 5-cycle VNNI latency
    __m512i acc_even1 = _mm512_setzero_si512();
    __m512i acc_odd1 = _mm512_setzero_si512();
    __m512i acc_even2 = _mm512_setzero_si512();
    __m512i acc_odd2 = _mm512_setzero_si512();

#pragma GCC unroll 8
    for (uint32_t b = 0; b < NUM_BLOCKS; b += 4) {
      // Load 128 bytes (64 points * 4 blocks)
      const __m512i packed1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes_ptr));
      const __m512i packed2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes_ptr + 64));
      codes_ptr += 128;

      __m512i c_even1 = _mm512_or_si512(_mm512_and_si512(packed1, low_mask), offset_mask);
      __m512i c_odd1 =
          _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(packed1, 4), low_mask), offset_mask);
      __m512i c_even2 = _mm512_or_si512(_mm512_and_si512(packed2, low_mask), offset_mask);
      __m512i c_odd2 =
          _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(packed2, 4), low_mask), offset_mask);

      __m512i s_even1 = _mm512_permutexvar_epi8(c_even1, preloaded_luts[b / 4]);
      __m512i s_odd1 = _mm512_permutexvar_epi8(c_odd1, preloaded_luts[b / 4]);
      __m512i s_even2 = _mm512_permutexvar_epi8(c_even2, preloaded_luts[b / 4]);
      __m512i s_odd2 = _mm512_permutexvar_epi8(c_odd2, preloaded_luts[b / 4]);

      // 4 parallel instructions perfectly saturate the execution ports
      acc_even1 = _mm512_dpbusd_epi32(acc_even1, s_even1, ones);
      acc_odd1 = _mm512_dpbusd_epi32(acc_odd1, s_odd1, ones);
      acc_even2 = _mm512_dpbusd_epi32(acc_even2, s_even2, ones);
      acc_odd2 = _mm512_dpbusd_epi32(acc_odd2, s_odd2, ones);
    }

    _mm512_storeu_si512(reinterpret_cast<__m512i*>(&results_32[s * 64]), acc_even1);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(&results_32[s * 64 + 16]), acc_odd1);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(&results_32[s * 64 + 32]), acc_even2);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(&results_32[s * 64 + 48]), acc_odd2);
  }
}
#endif
// ============================================================================
// Google Benchmark Harness
// ============================================================================

class FastScanFixture : public benchmark::Fixture {
 public:
  std::vector<uint8_t> codes;
  std::vector<uint8_t> lut;
  std::vector<uint16_t> results_16;
  std::vector<uint32_t> results_32;

  void SetUp(const ::benchmark::State& state) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist_code(0, 255);

    size_t total_code_bytes = (NUM_POINTS / 2) * NUM_BLOCKS;
    codes.resize(total_code_bytes);
    lut.resize(NUM_BLOCKS * K);
    results_16.resize(NUM_POINTS);
    results_32.resize(NUM_POINTS);

    for (auto& val : codes)
      val = dist_code(rng);
    for (auto& val : lut)
      val = dist_code(rng);
  }
};

#ifdef __AVX512F__
BENCHMARK_F(FastScanFixture, BM_Original_AVX512)(benchmark::State& state) {
  for (auto _ : state) {
    scan_original_avx512(codes.data(), lut.data(), results_16.data());
    benchmark::DoNotOptimize(results_16);
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * codes.size());
  state.SetItemsProcessed(state.iterations() * NUM_POINTS);
}
#endif
#ifdef __AVX512VBMI__
BENCHMARK_F(FastScanFixture, BM_VBMI_2Block)(benchmark::State& state) {
  for (auto _ : state) {
    scan_vbmi_2block(codes.data(), lut.data(), results_16.data());
    benchmark::DoNotOptimize(results_16);
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * codes.size());
  state.SetItemsProcessed(state.iterations() * NUM_POINTS);
}

BENCHMARK_F(FastScanFixture, BM_VBMI_512)(benchmark::State& state) {
  for (auto _ : state) {
    scan_vbmi_512(codes.data(), lut.data(), results_16.data());
    benchmark::DoNotOptimize(results_16);
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * codes.size());
  state.SetItemsProcessed(state.iterations() * NUM_POINTS);
}

BENCHMARK_F(FastScanFixture, BM_VBMI_512_Unrolled)(benchmark::State& state) {
  for (auto _ : state) {
    scan_vbmi_512_unrolled(codes.data(), lut.data(), results_16.data());
    benchmark::DoNotOptimize(results_16);
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * codes.size());
  state.SetItemsProcessed(state.iterations() * NUM_POINTS);
}
#endif
#ifdef __AVX512VNNI__
BENCHMARK_F(FastScanFixture, BM_VNNI_32Bit)(benchmark::State& state) {
  for (auto _ : state) {
    scan_vnni_32bit(codes.data(), lut.data(), results_32.data());
    benchmark::DoNotOptimize(results_32);
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * codes.size());
  state.SetItemsProcessed(state.iterations() * NUM_POINTS);
}

BENCHMARK_F(FastScanFixture, BM_VNNI_32Bit_Unrolled)(benchmark::State& state) {
  for (auto _ : state) {
    scan_vnni_32bit_unrolled(codes.data(), lut.data(), results_32.data());
    benchmark::DoNotOptimize(results_32);
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * codes.size());
  state.SetItemsProcessed(state.iterations() * NUM_POINTS);
}

BENCHMARK_F(FastScanFixture, BM_VNNI_32Bit_Unrolled_4x)(benchmark::State& state) {
  for (auto _ : state) {
    scan_vnni_32bit_unrolled_4x(codes.data(), lut.data(), results_32.data());
    benchmark::DoNotOptimize(results_32);
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * codes.size());
  state.SetItemsProcessed(state.iterations() * NUM_POINTS);
}
#endif
BENCHMARK_MAIN();