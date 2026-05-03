#include "mvsic/core/utils/interval_heap.h"

#include <algorithm>
#include <random>
#include <set>
#include <vector>

#include "gtest/gtest.h"

namespace mvsic {
namespace {

using ScoreNode = std::pair<float, int>;

// Validate Atkinson min-max heap invariants on the underlying buffer by
// scanning every parent/descendant pair.  The IntervalHeap class doesn't
// expose its buffer; we re-derive correctness by comparing against the
// std::multiset oracle in CrossCheck below.

// Mirror push / pop_min / replace_max against std::multiset and assert that
// top_min / top_max / pop ordering match.  std::multiset is a strict-weak
// ordering oracle so iterators give us min (begin) and max (prev(end)).
TEST(IntervalHeapTest, CrossCheckRandomOps) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-100.f, 100.f);
  std::uniform_int_distribution<int> id_dist(0, 1 << 20);

  for (int seed = 0; seed < 16; ++seed) {
    rng.seed(static_cast<uint32_t>(seed));
    IntervalHeap<ScoreNode> heap;
    std::multiset<ScoreNode> oracle;

    // Phase 1: random pushes (1024 entries).
    for (int i = 0; i < 1024; ++i) {
      ScoreNode v{dist(rng), id_dist(rng)};
      heap.push(v);
      oracle.insert(v);
      ASSERT_EQ(heap.size(), oracle.size());
      ASSERT_EQ(heap.top_min(), *oracle.begin());
      ASSERT_EQ(heap.top_max(), *std::prev(oracle.end()));
    }

    // Phase 2: interleaved pops + pushes + replace_max.
    for (int i = 0; i < 4096; ++i) {
      int op = std::uniform_int_distribution<int>(0, 3)(rng);
      if (op == 0 && !oracle.empty()) {
        ScoreNode got = heap.pop_min();
        ScoreNode expected = *oracle.begin();
        oracle.erase(oracle.begin());
        ASSERT_EQ(got, expected);
      } else if (op == 1 && !oracle.empty()) {
        ScoreNode got = heap.pop_max();
        ScoreNode expected = *std::prev(oracle.end());
        oracle.erase(std::prev(oracle.end()));
        ASSERT_EQ(got, expected);
      } else if (op == 2 && !oracle.empty()) {
        ScoreNode v{dist(rng), id_dist(rng)};
        heap.replace_max(v);
        oracle.erase(std::prev(oracle.end()));
        oracle.insert(v);
      } else {
        ScoreNode v{dist(rng), id_dist(rng)};
        heap.push(v);
        oracle.insert(v);
      }
      ASSERT_EQ(heap.size(), oracle.size());
      if (!oracle.empty()) {
        ASSERT_EQ(heap.top_min(), *oracle.begin());
        ASSERT_EQ(heap.top_max(), *std::prev(oracle.end()));
      }
    }

    // Phase 3: drain and confirm sorted-ascending pop_min order.
    std::vector<ScoreNode> drained;
    while (!heap.empty()) drained.push_back(heap.pop_min());
    ASSERT_EQ(drained.size(), oracle.size());
    auto it = oracle.begin();
    for (size_t k = 0; k < drained.size(); ++k, ++it) {
      ASSERT_EQ(drained[k], *it);
    }
  }
}

// Mimic the greedy_search beam usage pattern: bounded-capacity DEPQ where we
// pop the min and possibly evict the max with a smaller candidate.
TEST(IntervalHeapTest, BeamUsagePattern) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(0.f, 1000.f);

  for (size_t cap : {1ul, 2ul, 3ul, 8ul, 64ul, 1024ul, 16384ul}) {
    IntervalHeap<ScoreNode> beam(cap);
    std::multiset<ScoreNode> oracle;

    // Stream 4*cap candidates; keep only the best `cap`-many.  After this,
    // beam should hold exactly the same set as the oracle's best cap entries.
    const size_t N = 4 * cap;
    for (size_t i = 0; i < N; ++i) {
      ScoreNode v{dist(rng), static_cast<int>(i)};
      if (beam.size() < cap) {
        beam.push(v);
        oracle.insert(v);
      } else if (v < beam.top_max()) {
        beam.replace_max(v);
        oracle.erase(std::prev(oracle.end()));
        oracle.insert(v);
      }
      ASSERT_EQ(beam.size(), oracle.size());
      ASSERT_LE(beam.size(), cap);
    }

    // Pop everything and confirm the same multiset comes out in ascending
    // order.
    std::vector<ScoreNode> drained;
    while (!beam.empty()) drained.push_back(beam.pop_min());
    ASSERT_EQ(drained.size(), oracle.size());
    auto it = oracle.begin();
    for (size_t k = 0; k < drained.size(); ++k, ++it) {
      ASSERT_EQ(drained[k], *it);
    }
    ASSERT_TRUE(std::is_sorted(drained.begin(), drained.end()));
  }
}

TEST(IntervalHeapTest, EmptyAndSingleton) {
  IntervalHeap<int> h;
  ASSERT_TRUE(h.empty());
  ASSERT_EQ(h.size(), 0u);

  h.push(42);
  ASSERT_EQ(h.size(), 1u);
  ASSERT_EQ(h.top_min(), 42);
  ASSERT_EQ(h.top_max(), 42);
  ASSERT_EQ(h.pop_min(), 42);
  ASSERT_TRUE(h.empty());

  h.push(7);
  ASSERT_EQ(h.pop_max(), 7);
  ASSERT_TRUE(h.empty());
}

TEST(IntervalHeapTest, AscendingAndDescendingInputs) {
  // Worst-case input shapes that often catch sift bugs.
  for (bool ascending : {true, false}) {
    IntervalHeap<int> h;
    for (int i = 0; i < 1000; ++i) h.push(ascending ? i : (999 - i));
    for (int expected = 0; expected < 1000; ++expected) {
      ASSERT_EQ(h.pop_min(), expected);
    }
    ASSERT_TRUE(h.empty());
  }
}

}  // namespace
}  // namespace mvsic
