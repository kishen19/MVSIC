#pragma once

// =============================================================================
// IntervalHeap — bounded-capacity double-ended priority queue on a flat vector.
//
// Designed as a drop-in replacement for the `std::set<score_node>` beam used
// inside IndexMVIVF{,Spill}::greedy_search.  That beam needs exactly four
// operations: pop_min (extract the next frontier to expand), top_max + push
// (try to add a candidate when not full), and replace_max (when full and the
// new candidate is strictly better than the current worst).  None of std::set's
// other capabilities (ordered iteration, find by key, stable iterators) are
// used — but every one of them is paid for via per-node heap allocations and
// scattered memory layout.  This class keeps the same operations on a single
// contiguous std::vector with no per-element allocations.
//
// Layout: Atkinson's min-max heap [Atkinson, Sack, Santoro, Strothotte 1986].
// Levels alternate min/max: nodes on even levels (0, 2, 4, ...) are <= every
// descendant; nodes on odd levels are >= every descendant.  Therefore buf_[0]
// is the global min and max(buf_[1], buf_[2]) is the global max (with the
// obvious specializations for size <= 2).  All operations are O(log N) and
// touch only the path between the affected node and the root, in a contiguous
// array — cache-friendly even at the large beam sizes we run with (nprobes
// up to 16K → beam up to 32K entries ≈ 512 KB, fits in L2).
//
// Comparator semantics mirror std::set: `Less` is a strict weak ordering;
// "min" is the unique element x such that `Less(y, x)` is false for all y in
// the heap.  Default is std::less<T>.
// =============================================================================

#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

namespace mvsic {

template <class T, class Less = std::less<T>>
class IntervalHeap {
 public:
  IntervalHeap() = default;
  explicit IntervalHeap(size_t capacity_hint) { buf_.reserve(capacity_hint); }

  bool empty() const noexcept { return buf_.empty(); }
  size_t size() const noexcept { return buf_.size(); }
  void clear() noexcept { buf_.clear(); }
  void reserve(size_t n) { buf_.reserve(n); }

  // Peek without removing.  Undefined if empty (caller checks).
  const T& top_min() const noexcept { return buf_[0]; }
  const T& top_max() const noexcept {
    const size_t n = buf_.size();
    if (n == 1) return buf_[0];
    if (n == 2) return buf_[1];
    return less_(buf_[1], buf_[2]) ? buf_[2] : buf_[1];
  }

  void push(const T& v) {
    buf_.push_back(v);
    sift_up_(buf_.size() - 1);
  }
  void push(T&& v) {
    buf_.push_back(std::move(v));
    sift_up_(buf_.size() - 1);
  }

  T pop_min() {
    T r = std::move(buf_[0]);
    if (buf_.size() == 1) {
      buf_.pop_back();
      return r;
    }
    buf_[0] = std::move(buf_.back());
    buf_.pop_back();
    if (!buf_.empty()) sift_down_min_(0);
    return r;
  }

  T pop_max() {
    const size_t n = buf_.size();
    if (n <= 2) {
      T r = std::move(buf_[n - 1]);
      buf_.pop_back();
      return r;
    }
    const size_t mi = max_index_();
    T r = std::move(buf_[mi]);
    if (mi == n - 1) {
      buf_.pop_back();
      return r;
    }
    buf_[mi] = std::move(buf_.back());
    buf_.pop_back();
    sift_down_max_(mi);
    return r;
  }

  // Replace the current max with v, preserving heap invariants.  Implemented
  // as pop_max followed by push: both are O(log N), and the combined version's
  // bookkeeping is bug-prone.  Caller should ensure v compares less than
  // top_max() — otherwise the resulting heap is still valid but the eviction
  // is a no-op in spirit.
  void replace_max(const T& v) {
    pop_max();
    push(v);
  }
  void replace_max(T&& v) {
    pop_max();
    push(std::move(v));
  }

 private:
  std::vector<T> buf_;
  Less less_{};

  // Returns the index of the maximum element when size() >= 3.  Caller
  // handles size <= 2 directly.
  size_t max_index_() const noexcept {
    return less_(buf_[1], buf_[2]) ? 2 : 1;
  }

  // True iff node `i` is on a min level (even level number).  Level of node i
  // (0-indexed) is floor(log2(i+1)); we extract that with clz on `i+1`.  For
  // i+1 in [2^k, 2^(k+1)), level == k, and clzll(i+1) == 63 - k, so the parity
  // of `k` is the parity of `(63 - clzll(i+1))`.
  static bool is_min_level_(size_t i) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    const int lz = __builtin_clzll(static_cast<unsigned long long>(i + 1));
    return ((63 - lz) & 1) == 0;
#else
    size_t v = i + 1;
    int level = 0;
    while (v > 1) { v >>= 1; ++level; }
    return (level & 1) == 0;
#endif
  }

  void sift_up_(size_t i) {
    if (i == 0) return;
    const size_t p = (i - 1) / 2;
    if (is_min_level_(i)) {
      // i is on min level (parent p on max level).  If a[i] > a[p], a[i] is
      // too big for any min position — push it up the max chain instead.
      if (less_(buf_[p], buf_[i])) {
        std::swap(buf_[i], buf_[p]);
        sift_up_max_(p);
      } else {
        sift_up_min_(i);
      }
    } else {
      if (less_(buf_[i], buf_[p])) {
        std::swap(buf_[i], buf_[p]);
        sift_up_min_(p);
      } else {
        sift_up_max_(i);
      }
    }
  }

  void sift_up_min_(size_t i) {
    // Walk the same-parity ancestor chain (grandparents on min levels).
    while (i > 2) {
      const size_t gp = ((i - 1) / 2 - 1) / 2;
      if (less_(buf_[i], buf_[gp])) {
        std::swap(buf_[i], buf_[gp]);
        i = gp;
      } else {
        break;
      }
    }
  }

  void sift_up_max_(size_t i) {
    while (i > 2) {
      const size_t gp = ((i - 1) / 2 - 1) / 2;
      if (less_(buf_[gp], buf_[i])) {
        std::swap(buf_[i], buf_[gp]);
        i = gp;
      } else {
        break;
      }
    }
  }

  // Find index of smallest among {i, children of i, grandchildren of i} that
  // exist.  Used by sift_down_min_; returns i if i is already the smallest.
  size_t smallest_descendant_(size_t i) const {
    const size_t n = buf_.size();
    size_t best = i;
    const size_t c0 = 2 * i + 1;
    const size_t c1 = 2 * i + 2;
    if (c0 < n && less_(buf_[c0], buf_[best])) best = c0;
    if (c1 < n && less_(buf_[c1], buf_[best])) best = c1;
    const size_t g0 = 2 * c0 + 1;  // 4i + 3
    const size_t g1 = 2 * c0 + 2;  // 4i + 4
    const size_t g2 = 2 * c1 + 1;  // 4i + 5
    const size_t g3 = 2 * c1 + 2;  // 4i + 6
    if (g0 < n && less_(buf_[g0], buf_[best])) best = g0;
    if (g1 < n && less_(buf_[g1], buf_[best])) best = g1;
    if (g2 < n && less_(buf_[g2], buf_[best])) best = g2;
    if (g3 < n && less_(buf_[g3], buf_[best])) best = g3;
    return best;
  }

  size_t largest_descendant_(size_t i) const {
    const size_t n = buf_.size();
    size_t best = i;
    const size_t c0 = 2 * i + 1;
    const size_t c1 = 2 * i + 2;
    if (c0 < n && less_(buf_[best], buf_[c0])) best = c0;
    if (c1 < n && less_(buf_[best], buf_[c1])) best = c1;
    const size_t g0 = 2 * c0 + 1;
    const size_t g1 = 2 * c0 + 2;
    const size_t g2 = 2 * c1 + 1;
    const size_t g3 = 2 * c1 + 2;
    if (g0 < n && less_(buf_[best], buf_[g0])) best = g0;
    if (g1 < n && less_(buf_[best], buf_[g1])) best = g1;
    if (g2 < n && less_(buf_[best], buf_[g2])) best = g2;
    if (g3 < n && less_(buf_[best], buf_[g3])) best = g3;
    return best;
  }

  void sift_down_min_(size_t i) {
    while (true) {
      const size_t m = smallest_descendant_(i);
      if (m == i) return;
      const size_t c0 = 2 * i + 1;
      const size_t c1 = 2 * i + 2;
      const bool m_is_child = (m == c0 || m == c1);
      std::swap(buf_[i], buf_[m]);
      if (m_is_child) return;
      // m is a grandchild on a min level; its parent (a child of the original
      // i) is on a max level.  After the swap above, the value now at m may be
      // larger than that parent — fix the max-level invariant if so.
      const size_t pm = (m - 1) / 2;
      if (less_(buf_[pm], buf_[m])) std::swap(buf_[pm], buf_[m]);
      i = m;
    }
  }

  void sift_down_max_(size_t i) {
    while (true) {
      const size_t m = largest_descendant_(i);
      if (m == i) return;
      const size_t c0 = 2 * i + 1;
      const size_t c1 = 2 * i + 2;
      const bool m_is_child = (m == c0 || m == c1);
      std::swap(buf_[i], buf_[m]);
      if (m_is_child) return;
      const size_t pm = (m - 1) / 2;
      if (less_(buf_[m], buf_[pm])) std::swap(buf_[pm], buf_[m]);
      i = m;
    }
  }
};

}  // namespace mvsic
