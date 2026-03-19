#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "parlay/primitives.h"
#include "parlay/sequence.h"

#include "mvsic/core/utils/parse_command_line.h"
#include "mvsic/core/utils/util.h"

using Pair = std::pair<uint32_t, float>;

namespace {

inline bool by_id_then_dist(const Pair& a, const Pair& b) {
  if (a.first != b.first) return a.first < b.first;
  return a.second < b.second;
}

inline bool by_dist_then_id(const Pair& a, const Pair& b) {
  if (a.second != b.second) return a.second < b.second;
  return a.first < b.first;
}

inline void canonicalize(parlay::sequence<Pair>& v) {
  parlay::sort_inplace(v, by_id_then_dist);
}

inline bool equal_answers(parlay::sequence<Pair> a, parlay::sequence<Pair> b) {
  if (a.size() != b.size()) return false;
  canonicalize(a);
  canonicalize(b);
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].first != b[i].first || a[i].second != b[i].second) return false;
  }
  return true;
}

parlay::sequence<Pair> dedup_baseline_two_sorts(parlay::sequence<Pair> visited) {
  parlay::sort_inplace(visited, by_id_then_dist);
  size_t write = 0;
  for (size_t read = 0; read < visited.size(); ++read) {
    if (read == 0 || visited[read].first != visited[read - 1].first) {
      visited[write++] = visited[read];
    }
  }
  visited.resize(write);
  parlay::sort_inplace(visited, by_dist_then_id);
  return visited;
}

parlay::sequence<Pair> dedup_hash_min_then_sort(parlay::sequence<Pair> visited) {
  std::unordered_map<uint32_t, float> best;
  best.reserve(visited.size());
  for (const auto& p : visited) {
    auto it = best.find(p.first);
    if (it == best.end()) {
      best.emplace(p.first, p.second);
    } else if (p.second < it->second) {
      it->second = p.second;
    }
  }

  parlay::sequence<Pair> out = parlay::sequence<Pair>::uninitialized(best.size());
  size_t idx = 0;
  for (const auto& kv : best) {
    out[idx++] = {kv.first, kv.second};
  }
  parlay::sort_inplace(out, by_dist_then_id);
  return out;
}

parlay::sequence<Pair> dedup_single_sort_dist_then_id_hashset(parlay::sequence<Pair> visited) {
  parlay::sort_inplace(visited, by_dist_then_id);
  std::unordered_set<uint32_t> seen;
  seen.reserve(visited.size());

  size_t write = 0;
  for (size_t i = 0; i < visited.size(); ++i) {
    auto [it, inserted] = seen.insert(visited[i].first);
    (void)it;
    if (inserted) visited[write++] = visited[i];
  }
  visited.resize(write);
  return visited;
}

// Method requested by user:
// - Sort by (distance, id)
// - Use delayed_tabulate + pack_index over pair-value starts
// - Take first occurrence of each distinct pair
//
// IMPORTANT: This exactly dedups by id only if all duplicates of an id have the same distance.
parlay::sequence<Pair> dedup_single_sort_dist_then_id_pack_index(parlay::sequence<Pair> visited) {
  parlay::sort_inplace(visited, by_dist_then_id);
  auto starts = parlay::delayed_tabulate(visited.size(), [&](size_t i) {
    if (i == 0) return true;
    return visited[i] != visited[i - 1];
  });
  auto offsets = parlay::pack_index(starts);
  auto out = parlay::tabulate(offsets.size(), [&](size_t i) { return visited[offsets[i]]; });
  return out;
}

// VQSort-based variant: pack (dist, id) into a double key, use Highway's
// VQSort on the packed array, then linear dedup while unpacking.
parlay::sequence<Pair> dedup_vqsort_packed(parlay::sequence<Pair> visited) {
  const size_t n = visited.size();
  if (n == 0) return visited;

  parlay::sequence<double> packed = parlay::sequence<double>::uninitialized(n);
  parlay::parallel_for(0, n, [&](size_t i) {
    packed[i] = mvsic::packFloatAndInt(visited[i].second, visited[i].first);
  });

  mvsic::VQSort(packed.begin(), packed.end());

  // Unpack ids to compute starts for each new id.
  parlay::sequence<uint32_t> ids = parlay::sequence<uint32_t>::uninitialized(n);
  parlay::parallel_for(0, n, [&](size_t i) {
    auto unpacked = mvsic::unpackDoubletoSizeT(packed[i]);
    ids[i] = static_cast<uint32_t>(unpacked.second);
  });

  auto starts = parlay::delayed_tabulate(n, [&](size_t i) {
    if (i == 0) return true;
    return ids[i] != ids[i - 1];
  });
  auto offsets = parlay::pack_index(starts);

  parlay::sequence<Pair> out = parlay::sequence<Pair>::uninitialized(offsets.size());
  parlay::parallel_for(0, offsets.size(), [&](size_t j) {
    auto [dist, id_sz] = mvsic::unpackDoubletoSizeT(packed[offsets[j]]);
    out[j] = {static_cast<uint32_t>(id_sz), dist};
  });
  return out;
}

// Single sort by (dist,id) + in-place dedup by id (no pack_index).
parlay::sequence<Pair> dedup_single_sort_inplace(parlay::sequence<Pair> visited) {
  parlay::sort_inplace(visited, by_dist_then_id);
  if (visited.empty()) return visited;
  size_t write = 1;
  for (size_t i = 1; i < visited.size(); ++i) {
    if (visited[i].first != visited[write - 1].first) {
      visited[write++] = visited[i];
    }
  }
  visited.resize(write);
  return visited;
}

parlay::sequence<Pair> dedup_hybrid_hash_or_pack_index(parlay::sequence<Pair> visited,
                                                       size_t hybrid_threshold_n) {
  if (visited.size() <= hybrid_threshold_n) {
    return dedup_hash_min_then_sort(std::move(visited));
  }
  return dedup_single_sort_dist_then_id_pack_index(std::move(visited));
}

// Hybrid2: VQSort for small n, pack-index for large n.
parlay::sequence<Pair> dedup_hybrid_vqsort_or_pack_index(parlay::sequence<Pair> visited,
                                                         size_t hybrid2_threshold_n) {
  if (visited.size() <= hybrid2_threshold_n) {
    return dedup_vqsort_packed(std::move(visited));
  }
  return dedup_single_sort_dist_then_id_pack_index(std::move(visited));
}

parlay::sequence<Pair> make_visited(size_t n, uint32_t max_dups, uint64_t seed) {
  if (max_dups == 0) max_dups = 1;
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<uint32_t> dup_dist(1, max_dups);
  std::uniform_real_distribution<float> val_dist(0.0f, 1.0f);

  // Create id multiplicities: each id appears in [1, max_dups], total exactly n elements.
  std::vector<uint32_t> multiplicities;
  multiplicities.reserve(n);
  size_t total = 0;
  while (total < n) {
    uint32_t d = dup_dist(rng);
    if (total + d > n) d = static_cast<uint32_t>(n - total);
    multiplicities.push_back(d);
    total += d;
  }

  const size_t num_ids = multiplicities.size();
  std::vector<float> id_dist(num_ids);
  for (size_t id = 0; id < num_ids; ++id) {
    id_dist[id] = val_dist(rng);
  }

  parlay::sequence<Pair> out = parlay::sequence<Pair>::uninitialized(n);
  size_t pos = 0;
  for (size_t id = 0; id < num_ids; ++id) {
    for (uint32_t r = 0; r < multiplicities[id]; ++r) {
      out[pos++] = {static_cast<uint32_t>(id), id_dist[id]};
    }
  }

  // Shuffle to avoid favorable order.
  for (size_t i = n; i > 1; --i) {
    std::uniform_int_distribution<size_t> pick(0, i - 1);
    size_t j = pick(rng);
    std::swap(out[i - 1], out[j]);
  }
  return out;
}

template<typename Fn>
double time_method_ms(const parlay::sequence<Pair>& in, Fn&& fn, uint32_t warmup, uint32_t iters) {
  for (uint32_t i = 0; i < warmup; ++i) {
    auto tmp = fn(in);
    (void)tmp;
  }
  auto t0 = std::chrono::steady_clock::now();
  for (uint32_t i = 0; i < iters; ++i) {
    auto tmp = fn(in);
    (void)tmp;
  }
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / static_cast<double>(iters);
}

}  // namespace

int main(int argc, char** argv) {
  mvsic::commandLine P(
      argc, argv,
      "[-min_mult 1] [-max_mult 2048] [-base 500] "
      "[-max_dups 10] [-seed 12345] [-iters 5] [-warmup 2] [-hybrid_threshold_n 32000] [-check]");

  const size_t base = static_cast<size_t>(P.getOptionLongValue("-base", 500));
  const uint32_t min_mult = static_cast<uint32_t>(P.getOptionIntValue("-min_mult", 1));
  const uint32_t max_mult = static_cast<uint32_t>(P.getOptionIntValue("-max_mult", 2048));
  const uint32_t max_dups = static_cast<uint32_t>(P.getOptionIntValue("-max_dups", 10));
  const uint64_t seed = static_cast<uint64_t>(P.getOptionLongValue("-seed", 12345));
  const uint32_t iters = static_cast<uint32_t>(P.getOptionIntValue("-iters", 5));
  const uint32_t warmup = static_cast<uint32_t>(P.getOptionIntValue("-warmup", 2));
  const size_t hybrid_threshold_n =
      static_cast<size_t>(P.getOptionLongValue("-hybrid_threshold_n", 16000));
  const size_t hybrid2_threshold_n =
      static_cast<size_t>(P.getOptionLongValue("-hybrid2_threshold_n", 60000));
  const bool check = P.getOption("-check");

  if (min_mult == 0 || max_mult == 0 || min_mult > max_mult) {
    std::cerr << "Invalid multiplier range.\n";
    return 1;
  }
  if ((min_mult & (min_mult - 1)) != 0 || (max_mult & (max_mult - 1)) != 0) {
    std::cerr << "Expected power-of-two multipliers for min/max.\n";
    return 1;
  }
  std::cout << "Dedup benchmark\n";
  std::cout << "base=" << base << " multipliers=[" << min_mult << ".." << max_mult
            << "] (powers of 2), max_dups=" << max_dups << " seed=" << seed << " warmup=" << warmup
            << " iters=" << iters << " hybrid_threshold_n=" << hybrid_threshold_n
            << " hybrid2_threshold_n=" << hybrid2_threshold_n
            << " check=" << (check ? "true" : "false") << "\n\n";

  // First table: ns/element per method.
  std::cout << std::left << std::setw(12) << "n" << std::setw(14) << "baseline_ns/e"
            << std::setw(14) << "hash_ns/e" << std::setw(16) << "one_sort_ns/e" << std::setw(16)
            << "pack_idx_ns/e" << std::setw(16) << "inplace_ns/e" << std::setw(16) << "vqsort_ns/e"
            << std::setw(16) << "hybrid_ns/e" << std::setw(16) << "hybrid2_ns/e"
            << "\n";

  for (uint32_t mult = min_mult; mult <= max_mult; mult <<= 1) {
    const size_t n = base * static_cast<size_t>(mult);
    auto input = make_visited(n, max_dups, seed + mult);

    if (check) {
      auto out1 = dedup_baseline_two_sorts(input);
      auto out2 = dedup_hash_min_then_sort(input);
      auto out3 = dedup_single_sort_dist_then_id_hashset(input);
      auto out4 = dedup_single_sort_dist_then_id_pack_index(input);
      auto out5 = dedup_single_sort_inplace(input);
      auto out6 = dedup_vqsort_packed(input);
      auto out7 = dedup_hybrid_hash_or_pack_index(input, hybrid_threshold_n);
      auto out8 = dedup_hybrid_vqsort_or_pack_index(input, hybrid2_threshold_n);

      const bool ok12 = equal_answers(out1, out2);
      const bool ok13 = equal_answers(out1, out3);
      const bool ok14 = equal_answers(out1, out4);
      const bool ok15 = equal_answers(out1, out5);
      const bool ok16 = equal_answers(out1, out6);
      const bool ok17 = equal_answers(out1, out7);
      const bool ok18 = equal_answers(out1, out8);
      if (!(ok12 && ok13 && ok14 && ok15 && ok16 && ok17 && ok18)) {
        std::cerr << "Mismatch at n=" << n << " (ok12=" << ok12 << ", ok13=" << ok13
                  << ", ok14=" << ok14 << ", ok15=" << ok15 << ", ok16=" << ok16
                  << ", ok17=" << ok17 << ", ok18=" << ok18 << ")\n";
        return 2;
      }
    }

    const double t1 = time_method_ms(input, dedup_baseline_two_sorts, warmup, iters);
    const double t2 = time_method_ms(input, dedup_hash_min_then_sort, warmup, iters);
    const double t3 = time_method_ms(input, dedup_single_sort_dist_then_id_hashset, warmup, iters);
    const double t4 =
        time_method_ms(input, dedup_single_sort_dist_then_id_pack_index, warmup, iters);
    const double t5 = time_method_ms(input, dedup_single_sort_inplace, warmup, iters);
    const double t6 = time_method_ms(input, dedup_vqsort_packed, warmup, iters);
    const double t7 = time_method_ms(
        input,
        [&](const parlay::sequence<Pair>& in) {
          return dedup_hybrid_hash_or_pack_index(in, hybrid_threshold_n);
        },
        warmup, iters);
    const double t8 = time_method_ms(
        input,
        [&](const parlay::sequence<Pair>& in) {
          return dedup_hybrid_vqsort_or_pack_index(in, hybrid2_threshold_n);
        },
        warmup, iters);

    const double ns1 = (t1 * 1e6) / static_cast<double>(n);
    const double ns2 = (t2 * 1e6) / static_cast<double>(n);
    const double ns3 = (t3 * 1e6) / static_cast<double>(n);
    const double ns4 = (t4 * 1e6) / static_cast<double>(n);
    const double ns5 = (t5 * 1e6) / static_cast<double>(n);
    const double ns6 = (t6 * 1e6) / static_cast<double>(n);
    const double ns7 = (t7 * 1e6) / static_cast<double>(n);
    const double ns8 = (t8 * 1e6) / static_cast<double>(n);

    std::cout << std::left << std::setw(12) << n << std::setw(14) << std::fixed
              << std::setprecision(2) << ns1 << std::setw(14) << ns2 << std::setw(16) << ns3
              << std::setw(16) << ns4 << std::setw(16) << ns5 << std::setw(16) << ns6
              << std::setw(16) << ns7 << std::setw(16) << ns8 << "\n";
  }

  // Second table: total time in milliseconds per method.
  std::cout << "\n";
  std::cout << std::left << std::setw(12) << "n" << std::setw(14) << "baseline_ms" << std::setw(14)
            << "hash_ms" << std::setw(16) << "one_sort_ms" << std::setw(16) << "pack_idx_ms"
            << std::setw(16) << "inplace_ms" << std::setw(16) << "vqsort_ms" << std::setw(16)
            << "hybrid_ms" << std::setw(16) << "hybrid2_ms"
            << "\n";

  for (uint32_t mult = min_mult; mult <= max_mult; mult <<= 1) {
    const size_t n = base * static_cast<size_t>(mult);
    auto input = make_visited(n, max_dups, seed + mult);

    const double t1 = time_method_ms(input, dedup_baseline_two_sorts, warmup, iters);
    const double t2 = time_method_ms(input, dedup_hash_min_then_sort, warmup, iters);
    const double t3 = time_method_ms(input, dedup_single_sort_dist_then_id_hashset, warmup, iters);
    const double t4 =
        time_method_ms(input, dedup_single_sort_dist_then_id_pack_index, warmup, iters);
    const double t5 = time_method_ms(input, dedup_single_sort_inplace, warmup, iters);
    const double t6 = time_method_ms(input, dedup_vqsort_packed, warmup, iters);
    const double t7 = time_method_ms(
        input,
        [&](const parlay::sequence<Pair>& in) {
          return dedup_hybrid_hash_or_pack_index(in, hybrid_threshold_n);
        },
        warmup, iters);
    const double t8 = time_method_ms(
        input,
        [&](const parlay::sequence<Pair>& in) {
          return dedup_hybrid_vqsort_or_pack_index(in, hybrid2_threshold_n);
        },
        warmup, iters);

    std::cout << std::left << std::setw(12) << n << std::setw(14) << std::fixed
              << std::setprecision(3) << t1 << std::setw(14) << t2 << std::setw(16) << t3
              << std::setw(16) << t4 << std::setw(16) << t5 << std::setw(16) << t6 << std::setw(16)
              << t7 << std::setw(16) << t8 << "\n";
  }

  return 0;
}
