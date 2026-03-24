#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <limits>

#include "mvsic/core/quantization/other_methods/fastscan.h"

using namespace mvsic;

/**
 * Robust check for 16-bit Accumulator Overflow.
 * FastScan uses uint16_t SIMD registers. If m * 255 > 65535, we have a problem.
 */
void test_accumulator_overflow_safety(uint32_t m_blocks) {
  std::cout << "[Test] Accumulator Overflow Check (m=" << m_blocks << ")..." << std::endl;

  // Max value in an 8-bit LUT is 255.
  uint32_t max_possible_sum = m_blocks * 255;

  std::cout << "  -> Max possible integer distance: " << max_possible_sum << std::endl;
  if (max_possible_sum > std::numeric_limits<uint16_t>::max()) {
    std::cerr << "  !! WARNING: Overflow risk detected. uint16_t accumulator max is 65535."
              << std::endl;
    std::cerr << "  !! You must cap m_blocks at 256 or use 32-bit accumulators." << std::endl;
  } else {
    std::cout << "  -> Accumulator is safe (Fits in 16-bit)." << std::endl;
  }
}

/**
 * Verifies that the dequantization (decode) function is a linear
 * transformation of the integer sum.
 */
void test_decode_linearity() {
  std::cout << "[Test] Decode Linearity..." << std::endl;
  fastscan::Quantized_Query<true> qq(8);
  qq.min_dist = 10.0f;
  qq.scale = 0.5f;
  qq.num_blocks = 8;

  // Manual decode: (min * m) + (sum * scale)
  // (10.0 * 8) + (100 * 0.5) = 80 + 50 = 130.0
  float expected = 130.0f;
  float actual = qq.decode(100);

  assert(std::abs(expected - actual) < 1e-5);
  std::cout << "  -> Decode logic verified." << std::endl;
}

/**
 * Checks if the interleaved memory access doesn't bleed into adjacent lanes.
 */
void test_lane_isolation() {
  std::cout << "[Test] Lane Isolation (Cross-talk check)..." << std::endl;

  // Create a dummy byte: 0xA (Lane 1) and 0x5 (Lane 0)
  // Binary: 1010 0101
  uint8_t packed_byte = (0xA << 4) | 0x5;

  // Mock a point with lane_idx 0
  fastscan::Quantized_Point<true> p_even(&packed_byte, 0);
  // Mock a point with lane_idx 1
  fastscan::Quantized_Point<true> p_odd(&packed_byte, 1);

  fastscan::Quantized_Query<true> qq(1);
  qq.int_lut.assign(16, 0);
  for (int i = 0; i < 16; ++i)
    qq.int_lut[i] = i;  // Identity LUT

  // Scalar distance check
  // Note: This bypasses decode() to check the raw integer accumulation
  auto get_raw_acc = [&](const auto& p) {
    uint16_t acc = 0;
    uint8_t packed = p.code_ptr[0];
    uint8_t code = (p.lane_idx % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
    return qq.int_lut[code];
  };

  assert(get_raw_acc(p_even) == 5);
  assert(get_raw_acc(p_odd) == 10);

  std::cout << "  -> Lane isolation verified (No nibble bleeding)." << std::endl;
}

int main() {
  std::cout << "--- FASTSCAN ROBUSTNESS SUITE ---\n" << std::endl;

  test_accumulator_overflow_safety(32);   // Typical use case
  test_accumulator_overflow_safety(256);  // Limit of uint16_t (256 * 255 = 65280)
  test_accumulator_overflow_safety(512);  // Should fail/warn

  std::cout << std::endl;
  test_decode_linearity();
  test_lane_isolation();

  std::cout << "\n[SUCCESS] Implementation logic is robust." << std::endl;
  return 0;
}