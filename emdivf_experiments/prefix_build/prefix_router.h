#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "parlay/primitives.h"

namespace emdivf_prefix {

// Small Vamana-style graph over prototype rows. A build round searches a
// frozen prefix, computes outgoing lists independently, and only then publishes
// those lists and the merged reverse edges. Scores are directed query->center
// Chamfer distances, rather than a symmetric distance matrix.
class PrefixRouter {
 public:
  using Score = std::function<float(uint32_t, uint32_t)>;

  PrefixRouter(uint32_t max_m, uint32_t degree, float alpha, Score score)
      : graph_(max_m), degree_(degree), alpha_(alpha), score_(std::move(score)) {
    if (max_m == 0 || degree == 0 || !std::isfinite(alpha) || alpha < 1 || !score_)
      throw std::invalid_argument("invalid prefix router parameters");
  }

  void BuildSeed(uint32_t n) {
    if (n == 0 || n > graph_.size() || active_ != 0)
      throw std::invalid_argument("invalid prefix router seed");
    std::vector<std::vector<uint32_t>> outgoing(n);
    parlay::parallel_for(0, n, [&](size_t row) {
      std::vector<uint32_t> candidates;
      candidates.reserve(n - 1);
      for (uint32_t other = 0; other < n; ++other)
        if (other != row) candidates.push_back(other);
      outgoing[row] = Prune(static_cast<uint32_t>(row), std::move(candidates));
    }, 1);
    // Add reverse edges after all forward neighborhoods have been computed.
    std::vector<std::vector<uint32_t>> reverse(n);
    for (uint32_t row = 0; row < n; ++row)
      for (uint32_t other : outgoing[row]) reverse[other].push_back(row);
    parlay::parallel_for(0, n, [&](size_t row) {
      auto candidates = std::move(outgoing[row]);
      candidates.insert(candidates.end(), reverse[row].begin(), reverse[row].end());
      graph_[row] = Prune(static_cast<uint32_t>(row), std::move(candidates));
    }, 1);
    AddBackbone(n);
    active_ = n;
  }

  // Results have increasing directed Chamfer distance; ties use prototype ID.
  // Every candidate is scored at most once. keep<=beam is required so callers
  // cannot accidentally request more output prototypes than the search beam.
  std::vector<uint32_t> Search(uint32_t query, uint32_t active,
                               uint32_t beam, uint32_t keep) const {
    if (active == 0 || active > active_ || query >= graph_.size() || beam == 0 ||
        keep == 0 || keep > beam)
      throw std::invalid_argument("invalid prefix router search");
    beam = std::min(beam, active);
    keep = std::min(keep, beam);
    struct Entry {
      float distance;
      uint32_t id;
      bool expanded;
    };
    const auto better = [](const Entry& left, const Entry& right) {
      return left.distance < right.distance ||
             (left.distance == right.distance && left.id < right.id);
    };
    std::vector<uint8_t> seen(active);
    std::vector<Entry> frontier;
    frontier.reserve(beam + 1);
    auto consider = [&](uint32_t id) {
      if (id >= active || seen[id]) return;
      seen[id] = 1;
      if (id == query) return;
      const float distance = score_(query, id);
      if (!std::isfinite(distance))
        throw std::runtime_error("nonfinite prefix routing score");
      Entry entry{distance, id, false};
      if (frontier.size() == beam && !better(entry, frontier.back())) return;
      frontier.insert(std::lower_bound(frontier.begin(), frontier.end(), entry, better), entry);
      if (frontier.size() > beam) frontier.pop_back();
    };
    // Prototype zero is the deterministic entry point. The fallback is needed
    // only when a caller searches a prototype already in the active prefix.
    consider(query == 0 && active > 1 ? 1 : 0);
    while (true) {
      const auto next = std::find_if(frontier.begin(), frontier.end(),
                                     [](const Entry& entry) { return !entry.expanded; });
      if (next == frontier.end()) break;
      const uint32_t id = next->id;
      next->expanded = true;
      for (uint32_t neighbor : graph_[id]) consider(neighbor);
    }
    std::vector<uint32_t> result;
    result.reserve(std::min<size_t>(keep, frontier.size()));
    for (size_t i = 0; i < std::min<size_t>(keep, frontier.size()); ++i)
      result.push_back(frontier[i].id);
    return result;
  }

  void InsertBatch(uint32_t begin, uint32_t end, uint32_t beam) {
    if (begin != active_ || begin == 0 || end <= begin || end > graph_.size() || beam == 0)
      throw std::invalid_argument("invalid prefix router insertion batch");
    std::vector<std::vector<uint32_t>> outgoing(end - begin);
    parlay::parallel_for(begin, end, [&](size_t row) {
      auto candidates = Search(static_cast<uint32_t>(row), begin, beam, beam);
      outgoing[row - begin] = Prune(static_cast<uint32_t>(row), std::move(candidates));
    }, 1);

    // This deterministic merge is small (batch_size*degree entries); expensive
    // searches and pruning run in parallel. No adjacency mutates during search.
    std::vector<std::vector<uint32_t>> reverse(begin);
    for (uint32_t row = begin; row < end; ++row)
      for (uint32_t other : outgoing[row - begin]) reverse[other].push_back(row);
    for (uint32_t row = begin; row < end; ++row)
      graph_[row] = std::move(outgoing[row - begin]);
    std::vector<uint32_t> affected;
    for (uint32_t row = 0; row < begin; ++row)
      if (!reverse[row].empty()) affected.push_back(row);
    parlay::parallel_for(0, affected.size(), [&](size_t index) {
      const uint32_t row = affected[index];
      auto candidates = graph_[row];
      candidates.insert(candidates.end(), reverse[row].begin(), reverse[row].end());
      graph_[row] = Prune(row, std::move(candidates));
    }, 1);
    AddBackbone(end);
    active_ = end;
  }

  uint32_t active() const { return active_; }
  const std::vector<std::vector<uint32_t>>& adjacency() const { return graph_; }

 private:
  std::vector<std::vector<uint32_t>> graph_;
  uint32_t degree_;
  float alpha_;
  Score score_;
  uint32_t active_ = 0;

  void AddBackbone(uint32_t n) {
    // Large doubling rounds can lose every reverse edge into a new point after
    // pruning. Reserve one slot for a deterministic ring so all published
    // prototypes remain reachable. These links are published only after the
    // round's searches; a new point cannot influence a peer's bucket search.
    if (n <= 1) return;
    parlay::parallel_for(0, n, [&](size_t row) {
      const uint32_t next = (static_cast<uint32_t>(row) + 1) % n;
      auto& neighbors = graph_[row];
      if (std::find(neighbors.begin(), neighbors.end(), next) != neighbors.end()) return;
      if (neighbors.size() >= degree_) neighbors.resize(degree_ - 1);
      neighbors.push_back(next);
    });
  }

  std::vector<uint32_t> Prune(uint32_t row, std::vector<uint32_t> candidates) const {
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    std::vector<std::pair<float, uint32_t>> scored;
    scored.reserve(candidates.size());
    for (uint32_t other : candidates) {
      if (other == row) continue;
      const float score = score_(row, other);
      if (!std::isfinite(score)) throw std::runtime_error("nonfinite prefix prune score");
      scored.emplace_back(score, other);
    }
    std::sort(scored.begin(), scored.end());
    std::vector<uint32_t> selected;
    selected.reserve(std::min<size_t>(degree_, scored.size()));
    std::vector<uint8_t> removed(scored.size());
    for (size_t i = 0; i < scored.size() && selected.size() < degree_; ++i) {
      if (removed[i]) continue;
      const uint32_t best = scored[i].second;
      selected.push_back(best);
      for (size_t j = i + 1; j < scored.size(); ++j) {
        if (removed[j]) continue;
        // Normalized-vector Chamfer is negative mean max-IP. Add one only for
        // pruning so alpha operates on nonnegative distances; routing retains
        // the original raw score and therefore its original order.
        const float between = std::max(0.0f, 1.0f + score_(best, scored[j].second));
        const float radial = std::max(0.0f, 1.0f + scored[j].first);
        if (alpha_ * between <= radial) removed[j] = 1;
      }
    }
    return selected;
  }
};

}  // namespace emdivf_prefix
