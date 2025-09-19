#pragma once

#include <set>
#include "parlay/primitives.h"
#include "mvsic/core/index.h"

#include "graph.h"
#include "beam_search.h"

namespace mvsic {

/* Multi-vector Vamana
 - Computes Vamana Index with Chamfer Distances
*/
template<bool metric>
class IndexVamana : public Index<metric> {
 public:
  using ChPoint = typename Index<metric>::ChPoint;  // Chamfer Point Type
  using pid = std::pair<uint32_t, float>;
  using Index<metric>::d;  // Embedding dimension

  IndexParams params;
  Graph<uint32_t> G;     // Vamana Graph
  uint32_t start_point;  // Starting Point of the graph

  IndexVamana(uint32_t d_) noexcept : params(IndexParams::mvvamana()) { d = d_; }
  IndexVamana(uint32_t d_, const IndexParams& params) noexcept : params(params) { d = d_; }

  inline void set_start() noexcept { start_point = 0; }

  inline uint32_t get_start() noexcept { return start_point; }

  // robustPrune routine as found in DiskANN paper, with the exception
  // that the new candidate set is added to the field new_nbhs instead
  // of directly replacing the out_nbh of p
  std::pair<parlay::sequence<uint32_t>, long> robustPrune(uint32_t p, parlay::sequence<pid>& cand,
                                                          const PointCloudSet<ChPoint>& points,
                                                          double alpha, bool add = true) {
    // add out neighbors of p to the candidate set.
    size_t out_size = G[p].size();
    std::vector<pid> candidates;
    long distance_comps = 0;
    for (auto x : cand)
      candidates.push_back(x);

    if (add) {
      for (size_t i = 0; i < out_size; i++) {
        distance_comps++;
        candidates.push_back(std::make_pair(G[p][i], points[p].distance(points[G[p][i]])));
      }
    }

    // Sort the candidate set according to distance from p
    auto less = [&](std::pair<uint32_t, float> a, std::pair<uint32_t, float> b) {
      return a.second < b.second || (a.second == b.second && a.first < b.first);
    };
    std::sort(candidates.begin(), candidates.end(), less);

    // remove any duplicates
    auto new_end = std::unique(candidates.begin(), candidates.end(),
                               [&](auto x, auto y) { return x.first == y.first; });
    candidates = std::vector(candidates.begin(), new_end);

    std::vector<uint32_t> new_nbhs;
    new_nbhs.reserve(params.vamana.R);

    size_t candidate_idx = 0;

    while (new_nbhs.size() < params.vamana.R && candidate_idx < candidates.size()) {
      // Don't need to do modifications.
      int p_star = candidates[candidate_idx].first;
      candidate_idx++;
      if (p_star == p || p_star == -1) {
        continue;
      }

      new_nbhs.push_back(p_star);

      for (size_t i = candidate_idx; i < candidates.size(); i++) {
        int p_prime = candidates[i].first;
        if (p_prime != -1) {
          distance_comps++;
          float dist_starprime = points[p_star].distance(points[p_prime]);
          float dist_pprime = candidates[i].second;
          if (alpha * dist_starprime <= dist_pprime) {
            candidates[i].first = -1;
          }
        }
      }
    }

    auto new_neighbors_seq = parlay::to_sequence(new_nbhs);
    return std::pair(new_neighbors_seq, distance_comps);
  }

  // wrapper to allow calling robustPrune on a sequence of candidates
  // that do not come with precomputed distances
  std::pair<parlay::sequence<uint32_t>, long> robustPrune(uint32_t p,
                                                          parlay::sequence<uint32_t> candidates,
                                                          const PointCloudSet<ChPoint>& points,
                                                          double alpha, bool add = true) {

    parlay::sequence<pid> cc;
    long distance_comps = 0;
    cc.reserve(candidates.size());  // + size_of(p->out_nbh));
    for (size_t i = 0; i < candidates.size(); ++i) {
      distance_comps++;
      cc.push_back(std::make_pair(candidates[i], points[p].distance(points[candidates[i]])));
    }
    auto [ngh_seq, dc] = robustPrune(p, cc, points, alpha, add);
    return std::pair(ngh_seq, dc + distance_comps);
  }

  // add ngh to candidates without adding any repeats
  template<typename rangeType1, typename rangeType2>
  void add_neighbors_without_repeats(const rangeType1& ngh, rangeType2& candidates) {
    std::unordered_set<uint32_t> a;
    for (auto c : candidates)
      a.insert(c);
    for (int i = 0; i < ngh.size(); i++)
      if (a.count(ngh[i]) == 0) candidates.push_back(ngh[i]);
  }

  void batch_insert(parlay::sequence<uint32_t>& inserts, const PointCloudSet<ChPoint>& points,
                    double alpha, bool random_order = false, double base = 2,
                    double max_fraction = .02, bool print = false) {
    for (int p : inserts) {
      if (p < 0 || p > (int)G.size()) {
        std::cout << "ERROR: invalid point " << p << " given to batch_insert" << std::endl;
        abort();
      }
    }
    size_t n = G.size();
    size_t m = inserts.size();
    size_t inc = 0;
    size_t count = 0;
    float frac = 0.0;
    float progress_inc = .1;
    size_t max_batch_size =
        std::min(static_cast<size_t>(max_fraction * static_cast<float>(n)), 1000000ul);
    // fix bug where max batch size could be set to zero
    if (max_batch_size == 0) max_batch_size = n;
    parlay::sequence<int> rperm;
    if (random_order)
      rperm = parlay::random_permutation<int>(static_cast<int>(m));
    else
      rperm = parlay::tabulate(m, [&](int i) { return i; });
    auto shuffled_inserts = parlay::tabulate(m, [&](size_t i) { return inserts[rperm[i]]; });
    parlay::internal::timer t_beam("beam search time");
    parlay::internal::timer t_bidirect("bidirect time");
    parlay::internal::timer t_prune("prune time");
    t_beam.stop();
    t_bidirect.stop();
    t_prune.stop();
    while (count < m) {
      size_t floor;
      size_t ceiling;
      if (pow(base, inc) <= max_batch_size) {
        floor = static_cast<size_t>(pow(base, inc)) - 1;
        ceiling = std::min(static_cast<size_t>(pow(base, inc + 1)) - 1, m);
        count = std::min(static_cast<size_t>(pow(base, inc + 1)) - 1, m);
      } else {
        floor = count;
        ceiling = std::min(count + static_cast<size_t>(max_batch_size), m);
        count += static_cast<size_t>(max_batch_size);
      }
      parlay::sequence<parlay::sequence<uint32_t>> new_out_(ceiling - floor);
      // search for each node starting from the start_point, then call
      // robustPrune with the visited list as its candidate set
      t_beam.start();
      parlay::parallel_for(floor, ceiling, [&](size_t i) {
        size_t index = shuffled_inserts[i];
        SearchParams search_params = SearchParams::mvvamana(
            (long)0, params.vamana.L, (double)0.0, (long)points.size(), (long)G.max_degree());
        parlay::sequence<pid> visited =
            (beam_search<uint32_t>(points[index], G, points, start_point, search_params))
                .first.second;
        new_out_[i - floor] = robustPrune(index, visited, points, alpha).first;
      });
      t_beam.stop();
      // make each edge bidirectional by first adding each new edge
      //(i,j) to a sequence, then semisorting the sequence by key values
      t_bidirect.start();
      auto to_flatten = parlay::tabulate(ceiling - floor, [&](size_t i) {
        uint32_t index = shuffled_inserts[i + floor];
        auto edges = parlay::tabulate(
            new_out_[i].size(), [&](size_t j) { return std::make_pair(new_out_[i][j], index); });
        return edges;
      });
      parlay::parallel_for(floor, ceiling, [&](size_t i) {
        G[shuffled_inserts[i]].update_neighbors(new_out_[i - floor]);
      });
      auto grouped_by = parlay::group_by_key(parlay::flatten(to_flatten));
      t_bidirect.stop();
      t_prune.start();
      // finally, add the bidirectional edges; if they do not make
      // the vertex exceed the degree bound, just add them to out_nbhs;
      // otherwise, use robustPrune on the vertex with user-specified alpha
      parlay::parallel_for(0, grouped_by.size(), [&](size_t j) {
        auto& [index, candidates] = grouped_by[j];
        size_t newsize = candidates.size() + G[index].size();
        if (newsize <= params.vamana.R) {
          add_neighbors_without_repeats(G[index], candidates);
          G[index].update_neighbors(candidates);
        } else {
          auto new_out_2_ = robustPrune(index, std::move(candidates), points, alpha).first;
          G[index].update_neighbors(new_out_2_);
        }
      });
      t_prune.stop();
      if (print) {
        auto ind = frac * n;
        if (floor <= ind && ceiling > ind) {
          frac += progress_inc;
          std::cout << "Pass " << 100 * frac << "% complete" << std::endl;
        }
      }
      inc += 1;
    }
    t_beam.total();
    t_bidirect.total();
    t_prune.total();
  }

  // Builds the index given PointCloudSet object.
  void build(const PointCloudSet<ChPoint>& points) override {
    if (params.verbose >= 1) std::cout << "Building graph..." << std::endl;
    set_start();
    G = Graph<uint32_t>(params.vamana.R, points.size());
    auto inserts = parlay::tabulate(points.size(), [&](uint32_t i) { return i; });
    if (params.vamana.two_pass) batch_insert(inserts, points, 1.0, true, 2, .02);
    batch_insert(inserts, points, params.vamana.alpha, true, 2, .02);
    parlay::parallel_for(0, G.size(), [&](long i) {
      auto less = [&](uint32_t j, uint32_t k) {
        return points[i].distance(points[j]) < points[i].distance(points[k]);
      };
      G[i].sort(less);
    });
    if (params.verbose >= 1) std::cout << "Graph built." << std::endl;
  }

  // Returns the top-k point clouds for the query point cloud
  // Output format: < [<id, distance>, ...], # distance comparisons>
  std::pair<parlay::sequence<std::pair<uint32_t, float>>, size_t> search(
      const ChPoint& query, const PointCloudSet<ChPoint>& points,
      const SearchParams& search_params) override {
    size_t k = search_params.k;
    auto [result, dist_cmps] = beam_search<uint32_t>(query, G, points, start_point, search_params);
    parlay::sequence<pid> visited = result.second;
    auto final_results = parlay::sequence<std::pair<uint32_t, float>>::from_function(
        std::min(k, visited.size()), [&](size_t i) { return visited[i]; });
    return std::make_pair(final_results, dist_cmps);
  }

  // Write the index to a file in disk
  void save(const std::string& filename) override {
    char* filename_c = (char*)filename.c_str();
    G.save(filename_c);
  }

  // Read the index from a file in disk
  void load(const std::string& filename, const PointCloudSet<ChPoint>& points) override {
    char* filename_c = (char*)filename.c_str();
    G = Graph<uint32_t>(filename_c);
    set_start();
  }
};

using IndexVamanaL2 = IndexVamana<true>;   // L2 metric
using IndexVamanaIP = IndexVamana<false>;  // MIPS

}  // namespace mvsic