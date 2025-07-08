#pragma once

#include <set>
#include "parlay/primitives.h"
#include "src/common/index.h"
#include "src/utils/top_neighbors.h"

#include "graph.h"
#include "beam_search.h"

namespace mvivf {

/* =========================================Params Type======================================== */
struct IndexVamanaParams {
  size_t R = 64;          // Max Outdegree of the routing graph
  size_t L = 64;          // Beam length
  double alpha = 1.2;     // Robust pruning parameter
  bool two_pass = false;  // Two-pass graph construction
  bool verbose = false;   // Print debug statements
};
/* ==================================Multi-Vector Vamana Class================================= */
template<bool metric>
class IndexVamana : Index<metric>, IndexVamanaParams {
 public:
  using ChPoint = Index<metric>::ChPoint;
  using indexType = size_t;
  using distanceType = float;
  using pid = std::pair<indexType, distanceType>;
  using Index<metric>::d;  // Embedding dimension
  Graph<size_t> G;         // Vamana Graph
  indexType start_point;   // Starting Point of the graph

  IndexVamana(size_t d_) noexcept { d = d_; }
  IndexVamana(size_t d_, const IndexVamanaParams& params) noexcept : IndexVamanaParams(params) {
    d = d_;
  }
  /* ----------------------------Overridden Functions---------------------------- */
  void build(const PointCloudSet<ChPoint>& points) override;
  std::pair<parlay::sequence<std::pair<size_t, float>>, size_t> search(
      const ChPoint& query, const PointCloudSet<ChPoint>& points,
      const SearchParams& params) override;
  void save(const std::string& filename) override;
  void load(const std::string& filename, const PointCloudSet<ChPoint>& points) override;
  /* ------------------------------Helper Functions------------------------------ */
  inline void set_start() noexcept;
  inline indexType get_start() noexcept;
  // Pruning routine
  std::pair<parlay::sequence<indexType>, long> robustPrune(indexType p, parlay::sequence<pid>& cand,
                                                           const PointCloudSet<ChPoint>& Points,
                                                           double alpha, bool add = true);
  // Wrapper to allow calling robustPrune on a sequence of candidates
  std::pair<parlay::sequence<indexType>, long> robustPrune(indexType p,
                                                           parlay::sequence<indexType> candidates,
                                                           const PointCloudSet<ChPoint>& Points,
                                                           double alpha, bool add = true);
  // Adds neighbors to candidates without adding any repeats
  template<typename rangeType1, typename rangeType2>
  void add_neighbors_without_repeats(const rangeType1& ngh, rangeType2& candidates);
  // Batch insert routine
  void batch_insert(parlay::sequence<indexType>& inserts, const PointCloudSet<ChPoint>& Points,
                    double alpha, bool random_order = false, double base = 2,
                    double max_fraction = .02, bool print = true);
};

/* =======================================Implementation======================================= */
template<bool metric>
inline void IndexVamana<metric>::set_start() noexcept {
  start_point = 0;
}

template<bool metric>
inline typename IndexVamana<metric>::indexType IndexVamana<metric>::get_start() noexcept {
  return start_point;
}

// robustPrune routine as found in DiskANN paper, with the exception
// that the new candidate set is added to the field new_nbhs instead
// of directly replacing the out_nbh of p
template<bool metric>
std::pair<parlay::sequence<typename IndexVamana<metric>::indexType>, long>
IndexVamana<metric>::robustPrune(indexType p, parlay::sequence<pid>& cand,
                                 const PointCloudSet<ChPoint>& Points, double alpha, bool add) {
  // add out neighbors of p to the candidate set.
  size_t out_size = G[p].size();
  std::vector<pid> candidates;
  long distance_comps = 0;
  for (auto x : cand)
    candidates.push_back(x);

  if (add) {
    for (size_t i = 0; i < out_size; i++) {
      distance_comps++;
      candidates.push_back(std::make_pair(G[p][i], Points[p].distance(Points[G[p][i]])));
    }
  }

  // Sort the candidate set according to distance from p
  auto less = [&](std::pair<indexType, distanceType> a, std::pair<indexType, distanceType> b) {
    return a.second < b.second || (a.second == b.second && a.first < b.first);
  };
  std::sort(candidates.begin(), candidates.end(), less);

  // remove any duplicates
  auto new_end = std::unique(candidates.begin(), candidates.end(),
                             [&](auto x, auto y) { return x.first == y.first; });
  candidates = std::vector(candidates.begin(), new_end);

  std::vector<indexType> new_nbhs;
  new_nbhs.reserve(R);

  size_t candidate_idx = 0;

  while (new_nbhs.size() < R && candidate_idx < candidates.size()) {
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
        distanceType dist_starprime = Points[p_star].distance(Points[p_prime]);
        distanceType dist_pprime = candidates[i].second;
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
template<bool metric>
std::pair<parlay::sequence<typename IndexVamana<metric>::indexType>, long>
IndexVamana<metric>::robustPrune(indexType p, parlay::sequence<indexType> candidates,
                                 const PointCloudSet<ChPoint>& Points, double alpha, bool add) {

  parlay::sequence<pid> cc;
  long distance_comps = 0;
  cc.reserve(candidates.size());  // + size_of(p->out_nbh));
  for (size_t i = 0; i < candidates.size(); ++i) {
    distance_comps++;
    cc.push_back(std::make_pair(candidates[i], Points[p].distance(Points[candidates[i]])));
  }
  auto [ngh_seq, dc] = robustPrune(p, cc, Points, alpha, add);
  return std::pair(ngh_seq, dc + distance_comps);
}

// add ngh to candidates without adding any repeats
template<bool metric>
template<typename rangeType1, typename rangeType2>
void IndexVamana<metric>::add_neighbors_without_repeats(const rangeType1& ngh,
                                                        rangeType2& candidates) {
  std::unordered_set<indexType> a;
  for (auto c : candidates)
    a.insert(c);
  for (int i = 0; i < ngh.size(); i++)
    if (a.count(ngh[i]) == 0) candidates.push_back(ngh[i]);
}

template<bool metric>
void IndexVamana<metric>::build(const PointCloudSet<ChPoint>& Points) {
  std::cout << "Building graph..." << std::endl;
  set_start();
  G = Graph<indexType>(R, Points.size());
  parlay::sequence<indexType> inserts =
      parlay::tabulate(Points.size(), [&](size_t i) { return static_cast<indexType>(i); });
  if (two_pass) batch_insert(inserts, Points, 1.0, true, 2, .02);
  batch_insert(inserts, Points, alpha, true, 2, .02);
  parlay::parallel_for(0, G.size(), [&](long i) {
    auto less = [&](indexType j, indexType k) {
      return Points[i].distance(Points[j]) < Points[i].distance(Points[k]);
    };
    G[i].sort(less);
  });
}

template<bool metric>
void IndexVamana<metric>::batch_insert(parlay::sequence<indexType>& inserts,
                                       const PointCloudSet<ChPoint>& Points, double alpha,
                                       bool random_order, double base, double max_fraction,
                                       bool print) {
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
    parlay::sequence<parlay::sequence<indexType>> new_out_(ceiling - floor);
    // search for each node starting from the start_point, then call
    // robustPrune with the visited list as its candidate set
    t_beam.start();
    parlay::parallel_for(floor, ceiling, [&](size_t i) {
      size_t index = shuffled_inserts[i];
      SearchParams params((long)0, L, (double)0.0, (long)Points.size(), (long)G.max_degree());
      parlay::sequence<pid> visited =
          (beam_search<indexType>(Points[index], G, Points, start_point, params)).first.second;
      new_out_[i - floor] = robustPrune(index, visited, Points, alpha).first;
    });
    t_beam.stop();
    // make each edge bidirectional by first adding each new edge
    //(i,j) to a sequence, then semisorting the sequence by key values
    t_bidirect.start();
    auto to_flatten = parlay::tabulate(ceiling - floor, [&](size_t i) {
      indexType index = shuffled_inserts[i + floor];
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
      if (newsize <= R) {
        add_neighbors_without_repeats(G[index], candidates);
        G[index].update_neighbors(candidates);
      } else {
        auto new_out_2_ = robustPrune(index, std::move(candidates), Points, alpha).first;
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

template<bool metric>
std::pair<parlay::sequence<std::pair<size_t, float>>, size_t> IndexVamana<metric>::search(
    const ChPoint& query, const PointCloudSet<ChPoint>& Points, const SearchParams& params) {
  indexType k = params.k;
  auto [result, dist_cmps] = beam_search<indexType>(query, G, Points, start_point, params);
  parlay::sequence<pid> visited = result.second;
  auto final_results = parlay::sequence<std::pair<indexType, distanceType>>::from_function(
      std::min(k, visited.size()), [&](size_t i) {
        return std::make_pair(static_cast<size_t>(visited[i].first),
                              static_cast<float>(visited[i].second));
      });
  return std::make_pair(final_results, dist_cmps);
}

template<bool metric>
void IndexVamana<metric>::save(const std::string& filename) {
  char* filename_c = (char*)filename.c_str();
  G.save(filename_c);
}

template<bool metric>
void IndexVamana<metric>::load(const std::string& filename, const PointCloudSet<ChPoint>& points) {
  char* filename_c = (char*)filename.c_str();
  G = Graph<indexType>(filename_c);
  set_start();
}

template struct IndexVamana<true>;   // Instantiates for L2 metric (metric = true)
template struct IndexVamana<false>;  // Instantiates for MIPS      (metric = false)

}  // namespace mvivf