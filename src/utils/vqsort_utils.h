#pragma once

// #include "hwy/contrib/sort/vqsort-inl.h"
#include "hwy/contrib/sort/vqsort.h"
// #include "hwy/highway.h"

namespace mvivf {

template<class Iter>
void VQSort(Iter b, Iter e) {
  //  std::sort(b, e);
  hwy::VQSort(b, e - b, hwy::SortAscending());
}

// A union allows us to access the same piece of memory in different ways.
// Here, we can write a 'double' and read its raw 64 bits as a 'uint64_t'.
union DoubleConverter {
  double d;
  uint64_t i;
};

/**
 * @brief Packs a float and an 28-bit integer into a double.
 *
 * The function converts the float to a double, then overwrites the
 * lowest 28 bits of the double's mantissa with the provided integer value.
 *
 * @param float_val The floating-point value to pack.
 * @param int_val The 28-bit integer value to pack (must be in range [0, 268435455]).
 * @return A double containing the packed data.
 */
double packFloatAndInt(float float_val, uint32_t int_val) {
  DoubleConverter converter;

  // 1. Start by converting the float to a double. This sets the sign,
  //    exponent, and the most significant bits of the mantissa correctly.
  converter.d = static_cast<double>(float_val);

  // 2. Define masks for bitwise operations.
  // Mask to clear the lowest 28 bits (all 1s except for the last 28 bits).
  // ~0ULL is all 1s. (1ULL << 28) is 268435456. 268435456 - 1 is 268435455, which is 28 set
  // bits. The inverse gives us a mask to clear those bits.
  const uint64_t clear_mask = ~((1ULL << 28) - 1);

  // Mask to ensure the input integer is only 28 bits.  is 0...011111111111.
  const uint64_t int_mask = (1ULL << 28) - 1;

  // 3. Clear the lowest 28 bits of the double's integer representation.
  converter.i &= clear_mask;

  // 4. Combine (OR) the cleared integer with the 28-bit value.
  // We mask int_val to be safe and ensure no higher bits are set.
  converter.i |= (static_cast<uint64_t>(int_val) & int_mask);

  // 5. Return the result as a double.
  return converter.d;
}

double packFloatAndInt(float float_val, size_t int_val) {
  uint32_t int_val_32 = static_cast<uint32_t>(int_val);
  return packFloatAndInt(float_val, int_val_32);
}

/**
 * @brief Unpacks a double back into a float and an 28-bit integer.
 *
 * @param packed_val The double containing the packed data.
 * @return A std::pair containing the extracted float and the 28-bit integer.
 */
std::pair<float, uint32_t> unpackDouble(double packed_val) {
  DoubleConverter converter;
  converter.d = packed_val;

  // 1. Define the mask to extract the lowest 28 bits.
  const uint64_t extract_mask = (1ULL << 28) - 1;

  // 2. Extract the 28-bit integer by ANDing with the mask.
  uint32_t extracted_int = static_cast<uint32_t>(converter.i & extract_mask);

  // 3. To get the original float value, we must clear the packed integer bits
  //    from the double's representation.
  converter.i &= ~extract_mask;

  // 4. Convert the modified double back to a float.
  float extracted_float = static_cast<float>(converter.d);

  return {extracted_float, extracted_int};
}

std::pair<float, size_t> unpackDouble2(double packed_val) {
  std::pair<float, uint32_t> unpacked = unpackDouble(packed_val);
  return {unpacked.first, static_cast<size_t>(unpacked.second)};
}

}  // namespace mvivf