#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "parlay/primitives.h"
#include "parlay/sequence.h"

#include "mvsic/core/utils/parse_command_line.h"
#include "mvsic/core/utils/util.h"

using Pair = std::pair<uint32_t, float>;
struct ScoreNode {
  float score;
  void* ptr;
};

namespace {

parlay::sequence<Pair> make_data(size_t n, bool allow_repeats, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist_u(0.0f, 1.0f);

  parlay::sequence<Pair> v = parlay::sequence<Pair>::uninitialized(n);

  if (!allow_repeats) {
    // Distinct ids: 0..n-1 with random floats, shuffled.
    for (size_t i = 0; i < n; ++i) {
      v[i] = {static_cast<uint32_t>(i), dist_u(rng)};
    }
    // Shuffle to remove any pre-sorted bias.
    for (size_t i = n; i > 1; --i) {
      std::uniform_int_distribution<size_t> pick(0, i - 1);
      size_t j = pick(rng);
      std::swap(v[i - 1], v[j]);
    }
  } else {
    // Allow repetitions: ids drawn from a smaller range, values random.
    const uint32_t id_range = static_cast<uint32_t>(std::max<size_t>(1, n / 10));
    std::uniform_int_distribution<uint32_t> id_dist(0, id_range - 1);
    for (size_t i = 0; i < n; ++i) {
      v[i] = {id_dist(rng), dist_u(rng)};
    }
  }

  return v;
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

parlay::sequence<Pair> sort_std(const parlay::sequence<Pair>& in) {
  std::vector<Pair> v(in.begin(), in.end());
  std::sort(v.begin(), v.end(), [](const Pair& a, const Pair& b) {
    if (a.second != b.second) return a.second < b.second;
    return a.first < b.first;
  });
  parlay::sequence<Pair> out = parlay::sequence<Pair>::uninitialized(v.size());
  parlay::parallel_for(0, v.size(), [&](size_t i) { out[i] = v[i]; });
  return out;
}

parlay::sequence<Pair> sort_parlay(const parlay::sequence<Pair>& in) {
  parlay::sequence<Pair> v = in;  // copy
  parlay::sort_inplace(v, [](const Pair& a, const Pair& b) {
    if (a.second != b.second) return a.second < b.second;
    return a.first < b.first;
  });
  return v;
}

parlay::sequence<Pair> sort_vqsort_packed(const parlay::sequence<Pair>& in) {
  const size_t n = in.size();
  if (n == 0) return parlay::sequence<Pair>();

  parlay::sequence<double> packed = parlay::sequence<double>::uninitialized(n);
  parlay::parallel_for(
      0, n, [&](size_t i) { packed[i] = mvsic::packFloatAndInt(in[i].second, in[i].first); });

  mvsic::VQSort(packed.begin(), packed.end());

  parlay::sequence<Pair> out = parlay::sequence<Pair>::uninitialized(n);
  parlay::parallel_for(0, n, [&](size_t i) {
    auto [dist, id_sz] = mvsic::unpackDoubletoSizeT(packed[i]);
    out[i] = {static_cast<uint32_t>(id_sz), dist};
  });
  return out;
}

parlay::sequence<Pair> sort_mvsic(const parlay::sequence<Pair>& in) {
  parlay::sequence<Pair> v = in;  // copy
  mvsic::sort_inplace_kv(v);
  return v;
}

//==================== Top-k / nth_element-style benchmark ====================//

parlay::sequence<ScoreNode> make_scores(size_t n, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist_u(0.0f, 1.0f);
  parlay::sequence<ScoreNode> v = parlay::sequence<ScoreNode>::uninitialized(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = ScoreNode{dist_u(rng), reinterpret_cast<void*>(i)};
  }
  return v;
}

template<typename Fn>
double time_scores_ms(const parlay::sequence<ScoreNode>& in, Fn&& fn, uint32_t warmup,
                      uint32_t iters) {
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

parlay::sequence<ScoreNode> topk_std_nth(const parlay::sequence<ScoreNode>& in, size_t k) {
  std::vector<ScoreNode> v(in.begin(), in.end());
  if (k < v.size()) {
    std::nth_element(v.begin(), v.begin() + k, v.end(),
                     [](const ScoreNode& a, const ScoreNode& b) { return a.score < b.score; });
    v.resize(k);
  }
  parlay::sequence<ScoreNode> out = parlay::sequence<ScoreNode>::uninitialized(v.size());
  parlay::parallel_for(0, v.size(), [&](size_t i) { out[i] = v[i]; });
  return out;
}

parlay::sequence<ScoreNode> topk_std_sort(const parlay::sequence<ScoreNode>& in, size_t k) {
  std::vector<ScoreNode> v(in.begin(), in.end());
  std::sort(v.begin(), v.end(),
            [](const ScoreNode& a, const ScoreNode& b) { return a.score < b.score; });
  if (k < v.size()) v.resize(k);
  parlay::sequence<ScoreNode> out = parlay::sequence<ScoreNode>::uninitialized(v.size());
  parlay::parallel_for(0, v.size(), [&](size_t i) { out[i] = v[i]; });
  return out;
}

parlay::sequence<ScoreNode> topk_parlay_sort(const parlay::sequence<ScoreNode>& in, size_t k) {
  parlay::sequence<ScoreNode> v = in;
  parlay::sort_inplace(v, [](const ScoreNode& a, const ScoreNode& b) { return a.score < b.score; });
  if (k < v.size()) v.resize(k);
  return v;
}

parlay::sequence<ScoreNode> topk_vq_partial(const parlay::sequence<ScoreNode>& in, size_t k) {
  const size_t n = in.size();
  if (n == 0 || k == 0) return parlay::sequence<ScoreNode>();
  k = std::min(k, n);

  parlay::sequence<double> packed = parlay::sequence<double>::uninitialized(n);
  parlay::parallel_for(0, n, [&](size_t i) {
    // Pack (score, index)
    packed[i] = mvsic::packFloatAndInt(in[i].score, static_cast<uint32_t>(i));
  });

  mvsic::VQPartialSort(packed.begin(), packed.end(), k);

  parlay::sequence<ScoreNode> out = parlay::sequence<ScoreNode>::uninitialized(k);
  parlay::parallel_for(0, k, [&](size_t i) {
    auto [score, idx_sz] = mvsic::unpackDoubletoSizeT(packed[i]);
    size_t idx = static_cast<size_t>(idx_sz);
    out[i] = ScoreNode{score, in[idx].ptr};
  });
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  mvsic::commandLine P(argc, argv,
                       "[-min_mult 1] [-max_mult 2048] [-base 500] "
                       "[-seed 12345] [-iters 5] [-warmup 2] [-allow_repeats] [-k 1000]");

  const size_t base = static_cast<size_t>(P.getOptionLongValue("-base", 500));
  const uint32_t min_mult = static_cast<uint32_t>(P.getOptionIntValue("-min_mult", 1));
  const uint32_t max_mult = static_cast<uint32_t>(P.getOptionIntValue("-max_mult", 2048));
  const uint64_t seed = static_cast<uint64_t>(P.getOptionLongValue("-seed", 12345));
  const uint32_t iters = static_cast<uint32_t>(P.getOptionIntValue("-iters", 5));
  const uint32_t warmup = static_cast<uint32_t>(P.getOptionIntValue("-warmup", 2));
  const bool allow_repeats = P.getOption("-allow_repeats");
  size_t k_param = static_cast<size_t>(P.getOptionLongValue("-k", 1000));

  if (min_mult == 0 || max_mult == 0 || min_mult > max_mult) {
    std::cerr << "Invalid multiplier range.\n";
    return 1;
  }
  if ((min_mult & (min_mult - 1)) != 0 || (max_mult & (max_mult - 1)) != 0) {
    std::cerr << "Expected power-of-two multipliers for min/max.\n";
    return 1;
  }

  std::cout << "Sort benchmark (pairs of {uint32_t, float})\n";
  std::cout << "base=" << base << " multipliers=[" << min_mult << ".." << max_mult
            << "] (powers of 2), seed=" << seed << " warmup=" << warmup << " iters=" << iters
            << " allow_repeats=" << (allow_repeats ? "true" : "false") << "\n\n";

  // Table 1: ns/element per method.
  std::cout << std::left << std::setw(12) << "n" << std::setw(16) << "std_ns/e" << std::setw(16)
            << "parlay_ns/e" << std::setw(16) << "vqsort_ns/e" << std::setw(16) << "mvsic_ns/e"
            << "\n";

  for (uint32_t mult = min_mult; mult <= max_mult; mult <<= 1) {
    const size_t n = base * static_cast<size_t>(mult);
    auto input = make_data(n, allow_repeats, seed + mult);

    const double t_std = time_method_ms(input, sort_std, warmup, iters);
    const double t_par = time_method_ms(input, sort_parlay, warmup, iters);
    const double t_vq = time_method_ms(input, sort_vqsort_packed, warmup, iters);
    const double t_mvsic = time_method_ms(input, sort_mvsic, warmup, iters);

    const double std_ns = (t_std * 1e6) / static_cast<double>(n);
    const double par_ns = (t_par * 1e6) / static_cast<double>(n);
    const double vq_ns = (t_vq * 1e6) / static_cast<double>(n);
    const double mvsic_ns = (t_mvsic * 1e6) / static_cast<double>(n);

    std::cout << std::left << std::setw(12) << n << std::setw(16) << std::fixed
              << std::setprecision(2) << std_ns << std::setw(16) << par_ns << std::setw(16) << vq_ns
              << std::setw(16) << mvsic_ns << "\n";
  }

  // Table 2: total time in ms per method.
  std::cout << "\n";
  std::cout << std::left << std::setw(12) << "n" << std::setw(16) << "std_ms" << std::setw(16)
            << "parlay_ms" << std::setw(16) << "vqsort_ms" << std::setw(16) << "mvsic_ms" << "\n";

  for (uint32_t mult = min_mult; mult <= max_mult; mult <<= 1) {
    const size_t n = base * static_cast<size_t>(mult);
    auto input = make_data(n, allow_repeats, seed + mult);

    const double t_std = time_method_ms(input, sort_std, warmup, iters);
    const double t_par = time_method_ms(input, sort_parlay, warmup, iters);
    const double t_vq = time_method_ms(input, sort_vqsort_packed, warmup, iters);
    const double t_mvsic = time_method_ms(input, sort_mvsic, warmup, iters);

    std::cout << std::left << std::setw(12) << n << std::setw(16) << std::fixed
              << std::setprecision(3) << t_std << std::setw(16) << t_par << std::setw(16) << t_vq
              << std::setw(16) << t_mvsic << "\n";
  }

  //==================== Top-k benchmark (nth_element-style) ==================//

  std::cout << "\nTop-k benchmark (ScoreNode: {float score, void* ptr})\n";
  std::cout << "k_param=" << k_param << "\n\n";

  std::cout << std::left << std::setw(12) << "n" << std::setw(16) << "nth_ns/e" << std::setw(16)
            << "full_std_ns/e" << std::setw(16) << "parlay_ns/e" << std::setw(16)
            << "vqpartial_ns/e" << "\n";

  for (uint32_t mult = min_mult; mult <= max_mult; mult <<= 1) {
    const size_t n = base * static_cast<size_t>(mult);
    auto input = make_scores(n, seed + 1234 + mult);
    const size_t k = std::min(k_param, n);

    const double t_nth = time_scores_ms(
        input, [&](const parlay::sequence<ScoreNode>& s) { return topk_std_nth(s, k); }, warmup,
        iters);
    const double t_full_std = time_scores_ms(
        input, [&](const parlay::sequence<ScoreNode>& s) { return topk_std_sort(s, k); }, warmup,
        iters);
    const double t_parlay = time_scores_ms(
        input, [&](const parlay::sequence<ScoreNode>& s) { return topk_parlay_sort(s, k); }, warmup,
        iters);
    const double t_vqpartial = time_scores_ms(
        input, [&](const parlay::sequence<ScoreNode>& s) { return topk_vq_partial(s, k); }, warmup,
        iters);

    const double nth_ns = (t_nth * 1e6) / static_cast<double>(n);
    const double full_std_ns = (t_full_std * 1e6) / static_cast<double>(n);
    const double parlay_ns = (t_parlay * 1e6) / static_cast<double>(n);
    const double vq_ns = (t_vqpartial * 1e6) / static_cast<double>(n);

    std::cout << std::left << std::setw(12) << n << std::setw(16) << std::fixed
              << std::setprecision(2) << nth_ns << std::setw(16) << full_std_ns << std::setw(16)
              << parlay_ns << std::setw(16) << vq_ns << "\n";
  }

  std::cout << "\n";
  std::cout << std::left << std::setw(12) << "n" << std::setw(16) << "nth_ms" << std::setw(16)
            << "full_std_ms" << std::setw(16) << "parlay_ms" << std::setw(16) << "vqpartial_ms"
            << "\n";

  for (uint32_t mult = min_mult; mult <= max_mult; mult <<= 1) {
    const size_t n = base * static_cast<size_t>(mult);
    auto input = make_scores(n, seed + 1234 + mult);
    const size_t k = std::min(k_param, n);

    const double t_nth = time_scores_ms(
        input, [&](const parlay::sequence<ScoreNode>& s) { return topk_std_nth(s, k); }, warmup,
        iters);
    const double t_full_std = time_scores_ms(
        input, [&](const parlay::sequence<ScoreNode>& s) { return topk_std_sort(s, k); }, warmup,
        iters);
    const double t_parlay = time_scores_ms(
        input, [&](const parlay::sequence<ScoreNode>& s) { return topk_parlay_sort(s, k); }, warmup,
        iters);
    const double t_vqpartial = time_scores_ms(
        input, [&](const parlay::sequence<ScoreNode>& s) { return topk_vq_partial(s, k); }, warmup,
        iters);

    std::cout << std::left << std::setw(12) << n << std::setw(16) << std::fixed
              << std::setprecision(3) << t_nth << std::setw(16) << t_full_std << std::setw(16)
              << t_parlay << std::setw(16) << t_vqpartial << "\n";
  }

  return 0;
}
