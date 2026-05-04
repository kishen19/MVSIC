#pragma once

#include "parlay/primitives.h"
#include "hwy/contrib/sort/vqsort.h"

namespace mvsic {

template<typename K, typename V>
auto group_by_key_delayed(parlay::sequence<std::pair<K, V>>& seq) {
  seq = parlay::sort(seq);
  auto starts = parlay::delayed_tabulate(seq.size(), [&](size_t i) {
    if (i == 0 || seq[i].first != seq[i - 1].first) return true;
    return false;
  });
  auto offsets = parlay::pack_index(starts);
  return parlay::tabulate(offsets.size(), [&](size_t i) {
    size_t start = offsets[i];
    size_t end = i == offsets.size() - 1 ? seq.size() : offsets[i + 1];
    return parlay::delayed_tabulate(end - start, [&](size_t j) { return seq[start + j].second; });
  });
}

template<typename K, typename V>
auto group_by_key_inplace(parlay::sequence<std::pair<K, V>>& seq) {
  seq = parlay::sort(seq);
  auto starts = parlay::delayed_tabulate(seq.size(), [&](size_t i) {
    if (i == 0 || seq[i].first != seq[i - 1].first) return true;
    return false;
  });
  auto offsets = parlay::pack_index(starts);
  return parlay::tabulate(offsets.size(), [&](size_t i) {
    size_t start = offsets[i];
    size_t end = i == offsets.size() - 1 ? seq.size() : offsets[i + 1];
    return parlay::make_slice(seq.begin() + start, seq.begin() + end);
  });
}

/*=====================VQSort Helpers=====================*/
template<class Iter>
void VQSort(Iter b, Iter e) {
  hwy::VQSort(b, e - b, hwy::SortAscending());
}

template<class Iter>
void VQPartialSort(Iter b, Iter e, size_t k) {
  hwy::VQPartialSort(b, e - b, k, hwy::SortAscending());
}

// A union allows us to access the same piece of memory in different ways.
// Here, we can write a 'double' and read its raw 64 bits as a 'uint64_t'.
union DoubleConverter {
  double d;
  uint64_t i;
};

/*
 - Packs a float and a 28-bit integer into a double.
 - The function converts the float to a double, then overwrites the lowest 28 bits of the double's
   mantissa with the provided integer value.
*/
inline double packFloatAndInt(float float_val, uint32_t int_val) {
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

inline double packFloatAndInt(float float_val, size_t int_val) {
  uint32_t int_val_32 = static_cast<uint32_t>(int_val);
  return packFloatAndInt(float_val, int_val_32);
}

/*
  - Unpacks a double back into a float and a 28-bit integer.
  - returns an std::pair containing the extracted float and the 28-bit integer as a uint32_t.
 */
inline std::pair<float, uint32_t> unpackDouble(double packed_val) {
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

inline std::pair<float, size_t> unpackDoubletoSizeT(double packed_val) {
  std::pair<float, uint32_t> unpacked = unpackDouble(packed_val);
  return {unpacked.first, static_cast<size_t>(unpacked.second)};
}

// Deduplicate by key, keeping the smallest value per key, and return results
// sorted by (value, key). Specialized for (uint32_t, float) pairs.
//
// Strategy:
//  - For small n, use VQSort on packed (value,key) keys, then deduplicate via
//    pack_index on ids.
//  - For larger n, fall back to Parlay sort + pack_index by key.
//
// This matches the best-performing hybrid variant from the microbenchmark
// microbenchmark/mvivf/bench_dedup.cpp.
inline parlay::sequence<std::pair<uint32_t, float>> deduplicate(
    parlay::sequence<std::pair<uint32_t, float>>& seq) {
  const size_t n = seq.size();
  if (n == 0) return seq;
  constexpr size_t kHybridThresholdN = 60000;
  if (n <= kHybridThresholdN) {
    parlay::sequence<double> packed = parlay::sequence<double>::uninitialized(n);
    parlay::parallel_for(
        0, n, [&](size_t i) { packed[i] = packFloatAndInt(seq[i].second, seq[i].first); });
    VQSort(packed.begin(), packed.end());

    parlay::sequence<uint32_t> ids = parlay::sequence<uint32_t>::uninitialized(n);
    parlay::parallel_for(0, n, [&](size_t i) {
      auto unpacked = unpackDoubletoSizeT(packed[i]);
      ids[i] = static_cast<uint32_t>(unpacked.second);
    });

    auto starts = parlay::delayed_tabulate(n, [&](size_t i) {
      if (i == 0) return true;
      return ids[i] != ids[i - 1];
    });
    auto offsets = parlay::pack_index(starts);

    parlay::sequence<std::pair<uint32_t, float>> out =
        parlay::sequence<std::pair<uint32_t, float>>::uninitialized(offsets.size());
    parlay::parallel_for(0, offsets.size(), [&](size_t j) {
      auto [dist, id_sz] = unpackDoubletoSizeT(packed[offsets[j]]);
      out[j] = {static_cast<uint32_t>(id_sz), dist};
    });
    return out;
  }
  // Fallback / large-n path: Parlay sort + pack_index by key.
  parlay::sort_inplace(seq, [](const auto& a, const auto& b) {
    if (a.second != b.second) return a.second < b.second;
    return a.first < b.first;
  });
  auto starts = parlay::delayed_tabulate(n, [&](size_t i) {
    if (i == 0) return true;
    return seq[i].first != seq[i - 1].first;
  });
  auto offsets = parlay::pack_index(starts);
  parlay::sequence<std::pair<uint32_t, float>> out =
      parlay::sequence<std::pair<uint32_t, float>>::uninitialized(offsets.size());
  parlay::parallel_for(0, offsets.size(), [&](size_t i) { out[i] = seq[offsets[i]]; });
  return out;
}

inline parlay::sequence<std::pair<uint32_t, float>> deduplicate_and_topC(
    parlay::sequence<std::pair<uint32_t, float>>& seq, size_t C) {
  const size_t n = seq.size();
  if (n == 0) return seq;
  constexpr size_t kHybridThresholdN = 60000;
  if (n <= kHybridThresholdN) {
    parlay::sequence<double> packed = parlay::sequence<double>::uninitialized(n);
    parlay::parallel_for(
        0, n, [&](size_t i) { packed[i] = packFloatAndInt(seq[i].second, seq[i].first); });
    VQSort(packed.begin(), packed.end());

    parlay::sequence<uint32_t> ids = parlay::sequence<uint32_t>::uninitialized(n);
    parlay::parallel_for(0, n, [&](size_t i) {
      auto unpacked = unpackDoubletoSizeT(packed[i]);
      ids[i] = static_cast<uint32_t>(unpacked.second);
    });

    auto starts = parlay::delayed_tabulate(n, [&](size_t i) {
      if (i == 0) return true;
      return ids[i] != ids[i - 1];
    });
    auto offsets = parlay::pack_index(starts);
    C = std::min(C, offsets.size());
    parlay::sequence<std::pair<uint32_t, float>> out =
        parlay::sequence<std::pair<uint32_t, float>>::uninitialized(C);
    parlay::parallel_for(0, C, [&](size_t j) {
      auto [dist, id_sz] = unpackDoubletoSizeT(packed[offsets[j]]);
      out[j] = {static_cast<uint32_t>(id_sz), dist};
    });
    return out;
  }
  // Fallback / large-n path: Parlay sort + pack_index by key.
  parlay::sort_inplace(seq, [](const auto& a, const auto& b) {
    if (a.second != b.second) return a.second < b.second;
    return a.first < b.first;
  });
  auto starts = parlay::delayed_tabulate(n, [&](size_t i) {
    if (i == 0) return true;
    return seq[i].first != seq[i - 1].first;
  });
  auto offsets = parlay::pack_index(starts);
  C = std::min(C, offsets.size());
  parlay::sequence<std::pair<uint32_t, float>> out =
      parlay::sequence<std::pair<uint32_t, float>>::uninitialized(C);
  parlay::parallel_for(0, C, [&](size_t i) { out[i] = seq[offsets[i]]; });
  return out;
}

// Deduplicate by key, keeping the smallest value per key, and return results
// sorted by (value, key). This assumes V is totally ordered and K is equality-
// comparable and ordered.
//
// Strategy:
//  - For small n (when K = uint32_t and V = float), use VQSort on packed
//    (value,key) keys, then deduplicate via pack_index.
//  - Otherwise (or for large n / other types), fall back to Parlay sort +
//    pack_index by key.
//
// This matches the best-performing hybrid variant from the microbenchmark
// microbenchmark/mvivf/bench_dedup.cpp.
template<typename K, typename V>
parlay::sequence<std::pair<K, V>> deduplicate(parlay::sequence<std::pair<K, V>> seq) {
  const size_t n = seq.size();
  if (n == 0) return seq;
  constexpr size_t kHybridThresholdN = 60000;
  if constexpr (std::is_same_v<K, uint32_t> && std::is_same_v<V, float>) {
    if (n <= kHybridThresholdN) {
      parlay::sequence<double> packed = parlay::sequence<double>::uninitialized(n);
      parlay::parallel_for(
          0, n, [&](size_t i) { packed[i] = packFloatAndInt(seq[i].second, seq[i].first); });
      VQSort(packed.begin(), packed.end());
      parlay::sequence<uint32_t> ids = parlay::sequence<uint32_t>::uninitialized(n);
      parlay::parallel_for(0, n, [&](size_t i) {
        auto unpacked = unpackDoubletoSizeT(packed[i]);
        ids[i] = static_cast<uint32_t>(unpacked.second);
      });
      auto starts = parlay::delayed_tabulate(n, [&](size_t i) {
        if (i == 0) return true;
        return ids[i] != ids[i - 1];
      });
      auto offsets = parlay::pack_index(starts);
      parlay::sequence<std::pair<K, V>> out =
          parlay::sequence<std::pair<K, V>>::uninitialized(offsets.size());
      parlay::parallel_for(0, offsets.size(), [&](size_t j) {
        auto [dist, id_sz] = unpackDoubletoSizeT(packed[offsets[j]]);
        out[j] = {static_cast<uint32_t>(id_sz), dist};
      });
      return out;
    }
  }
  // Fallback / large-n path: Parlay sort + pack_index by key.
  parlay::sort_inplace(seq, [](const auto& a, const auto& b) {
    if (a.second != b.second) return a.second < b.second;
    return a.first < b.first;
  });
  auto starts = parlay::delayed_tabulate(n, [&](size_t i) {
    if (i == 0) return true;
    return seq[i].first != seq[i - 1].first;
  });
  auto offsets = parlay::pack_index(starts);
  parlay::sequence<std::pair<K, V>> out =
      parlay::sequence<std::pair<K, V>>::uninitialized(offsets.size());
  parlay::parallel_for(0, offsets.size(), [&](size_t i) { out[i] = seq[offsets[i]]; });
  return out;
}

// Sort a sequence of (uint32_t, float) pairs by (value, key). Picks the fastest
// path based on size *and* the active scheduler:
//   - small n: VQSort on packed (value,key) doubles.
//   - large n with >1 parlay worker: parlay's parallel sample sort.
//   - large n with 1 parlay worker (e.g. seq benches pinned via
//     execute_with_scheduler(1, ...)): VQSort. Parlay sort under a single
//     worker is dominated by its own scheduling overhead and loses to a
//     well-vectorized serial sort.
inline void sort_inplace_kv(parlay::sequence<std::pair<uint32_t, float>>& seq) {
  const size_t n = seq.size();
  if (n == 0) return;
  constexpr size_t kSortHybridThresholdN = 32000;
  const bool use_vqsort = n <= kSortHybridThresholdN || parlay::num_workers() <= 1;
  if (use_vqsort) {
    parlay::sequence<double> packed = parlay::sequence<double>::uninitialized(n);
    parlay::parallel_for(
        0, n, [&](size_t i) { packed[i] = packFloatAndInt(seq[i].second, seq[i].first); });
    VQSort(packed.begin(), packed.end());
    parlay::parallel_for(0, n, [&](size_t i) {
      auto [dist, id_sz] = unpackDoubletoSizeT(packed[i]);
      seq[i] = {static_cast<uint32_t>(id_sz), dist};
    });
    return;
  }
  parlay::sort_inplace(seq, [](const auto& a, const auto& b) {
    return (a.second != b.second) ? a.second < b.second : a.first < b.first;
  });
}

}  // namespace mvsic