#pragma once

#include "parlay/internal/get_time.h"
#include "parlay/primitives.h"

// #include "algorithms/utils/beamSearch.h"
#include "algorithms/utils/graph.h"
#include "algorithms/utils/stats.h"
#include "algorithms/utils/types.h"

#include "dynann.h"
#include "union_find.h"

/*
  Exact Ward's HAC Implementation
  Running time: O(n^2 * polylog(n))
*/
template<typename IdType, typename Range>
struct OracleExact {
  using Point = typename Range::pT;
  using DistTy = typename Point::distanceType;
  using map = aug_map<aug_entry<IdType, DistTy>>;

  parlay::sequence<map> neighbors;  // Nearest neighbors of clusters
  union_find<IdType> uf;            // Union Find: for maintaining clusters
  Range *Points;
  IdType n;

  OracleExact() : n(0) {}

  template<typename Seq>
  inline void init(Range &Points_, Seq &weights) {
    n = Points_.size();
    Points = &Points_;
    uf = union_find<IdType>(n, weights);

    neighbors = parlay::sequence<map>::from_function(n, [&](IdType i) {
      DistTy max_dist = std::numeric_limits<DistTy>::max();
      auto dists_to =
          parlay::sequence<std::pair<IdType, DistTy>>::from_function(n - 1, [&](IdType j) {
            std::pair<IdType, DistTy> output;
            if (j < i) {
              IdType w1 = weights[i], w2 = weights[j];
              output = std::make_pair(j, Points_[i].distance(Points_[j]) * (w1 * w2) / (w1 + w2));
            } else if (j >= i) {
              IdType w1 = weights[i], w2 = weights[j + 1];
              output = std::make_pair(j + 1,
                                      Points_[i].distance(Points_[j + 1]) * (w1 * w2) / (w1 + w2));
            }
            return output;
          });
      return map(dists_to);
    });
  }

  inline IdType find_cluster(IdType i) { return uf.find_compress(i); }

  inline DistTy linkage(IdType u, IdType v) {
    // u and v have to be representatives of clusters
    auto val = neighbors[u].find(v, std::numeric_limits<DistTy>::max());
    return val;
  }

  inline std::pair<IdType, DistTy> nearest_neighbor(IdType u) {
    auto [val, key] = neighbors[u].aug_val();
    return std::make_pair(key, val);
  }

  inline IdType merge_clusters(IdType u, IdType v) {
    size_t szu = uf.get_size(u);
    size_t szv = uf.get_size(v);
    IdType w = uf.unite(u, v);
    if (w != u) {
      std::swap(u, v);
      std::swap(szu, szv);
    }
    // u is the larger cluster
    IdType z = w ^ u ^ v;  // z=v
    assert(szu >= szv);
    assert(w == u);
    Points->centroid(u, v, szu, szv);  // Update u to the centroid
    size_t szw = szu + szv;

    DistTy max_dist = std::numeric_limits<DistTy>::max();
    // Step 1: Remove z from the neighbors of w
    neighbors[w] = map::remove(std::move(neighbors[w]), z);
    // Step 2: Update distances to/from w
    auto new_dist = parlay::sequence<std::pair<IdType, DistTy>>::uninitialized(neighbors[w].size());
    auto map_f = [&](const auto &et, size_t ind) {
      auto [key, val] = et;
      neighbors[key] = map::remove(std::move(neighbors[key]), z);  // remove z
      neighbors[key] = map::remove(std::move(neighbors[key]), w);  // remove w
      size_t szkey = uf.get_size(key);
      auto tmp = (*Points)[w].distance((*Points)[key]) * szkey * szw / (szkey + szw);
      neighbors[key].insert(std::make_pair(w, tmp));  // add updated w
      new_dist[ind] = std::make_pair(key, tmp);
    };
    map::map_index(neighbors[w], map_f);
    neighbors[w].clear();  // Empty w
    neighbors[w] = map::multi_insert(std::move(neighbors[w]), new_dist);
    return w;
  }

  inline auto get_center_ids() {
    auto reps = parlay::filter(parlay::iota(n), [&](IdType i) { return uf.find_compress(i) == i; });
    return reps;
  }
};

template<typename IdType, typename DistTy, typename Range, class Seq>
auto WardsHAC(Range &Points, uint32_t k, Seq &weights, int verbose = 1) {
  using kv = std::tuple<DistTy, IdType, IdType>;  // Heap element type
  parlay::internal::timer t("Exact Ward's HAC");

  // Initialize Oracle
  auto Oracle = OracleExact<IdType, Range>();
  Oracle.init(Points, weights);
  size_t n = Oracle.n;

  // Setting verbose variables
  verbose = std::max(0, std::min(2, verbose));
  float verb_per = (verbose == 0) ? n : (verbose == 1) ? n / 10 : n / 100;
  float verb_inc = verb_per;

  // Maintain set of active clusters
  // TODO: use PAM instead and make parallel
  auto tmp = parlay::iota(n);
  auto active_clusters = std::unordered_set<IdType>(tmp.begin(), tmp.end());
  auto get_best_pair = [&]() {
    auto best_pair = std::make_tuple(std::numeric_limits<DistTy>::max(), n, n);
    for (auto i : active_clusters) {
      auto [ngh, dist] = Oracle.nearest_neighbor(i);
      auto cur_pair = std::make_tuple(dist, i, ngh);
      if (cur_pair < best_pair) {
        best_pair = cur_pair;
      }
    }
    return best_pair;
  };
  t.next("Init Done");

  IdType u, v, w, u_orig, v_orig, cur = 0;
  DistTy dis, total_cost = 0;
  while (cur < n - k) {
    if (cur + 1 >= verb_per) {
      std::cout << "### " << verb_per * 100 / n << "\% done, Cost: " << total_cost << std::endl;
      verb_per += verb_inc;
    }
    std::tie(dis, u_orig, v_orig) = get_best_pair();
    u = Oracle.find_cluster(u_orig);
    v = Oracle.find_cluster(v_orig);
    // merge step
    w = Oracle.merge_clusters(u, v);
    total_cost += dis;
    cur++;
    active_clusters.erase(w ^ u ^ v);
  }
  t.next("HAC done");
  std::cout << "Total Cost: " << total_cost << std::endl;
  auto center_ids = Oracle.get_center_ids();
  return center_ids;
}

/*
  Approximate Ward's HAC Implementation
  Running time: O(n * polylog(n))
*/
template<typename IdType, typename Range>
struct OracleApx {
  using Point = typename Range::pT;
  using DistTy = typename Point::distanceType;
  using GraphI = Graph<IdType>;
  using findex = dyn_knn_index<Point, Range, IdType>;

  union_find<IdType> uf;
  Range *Points;
  IdType n;
  parlay::sequence<GraphI> G_list;
  parlay::sequence<findex *> I_list;
  BuildParams BP;
  QueryParams QP;
  double delta;
  size_t num_levels;

  OracleApx(BuildParams BP_, QueryParams QP_, double delta_) :
      n(0), BP(BP_), QP(QP_), delta(delta_) {}

  inline size_t get_level(size_t s) { return std::floor(std::log(s) / std::log(1 + delta)); }

  template<typename Seq>
  inline void init(Range &Points_, Seq &weights) {
    n = Points_.size();
    Points = &Points_;
    Points->AddZero();
    uf = union_find<IdType>(n, weights);

    // Building the index
    auto total_weight = parlay::reduce(weights);
    num_levels = (size_t)std::ceil(std::log(total_weight) / std::log(1 + delta)) + 1;
    std::cout << "Building index with " << num_levels << " levels" << std::endl;
    G_list = parlay::sequence<GraphI>::from_function(num_levels,
                                                     [&](size_t i) { return GraphI(BP.R, n + 1); });
    I_list = parlay::sequence<findex *>::from_function(num_levels, [&](size_t i) {
      auto I = new findex(n + 1, BP, std::floor(std::pow(1 + delta, i)));
      I->set_start(n);
      return I;
    });
    // Bucket points by level
    auto levels = parlay::sequence<std::pair<size_t, IdType>>::from_function(
        n, [&](size_t i) { return std::make_pair(get_level(uf.get_size(i)), i); });
    parlay::sort_inplace(levels);
    auto cutoff_indices = parlay::delayed_seq<size_t>(n + 1, [&](size_t i) {
      return i == 0 || i == n || levels[i].first != levels[i - 1].first;
    });
    auto offsets = parlay::pack_index(cutoff_indices);
    // Build index for each level
    for (size_t i = 0; i < offsets.size() - 1; i++) {
      auto start_index = offsets[i];
      auto end_index = offsets[i + 1];
      if (end_index - start_index > 0) {
        auto level = levels[start_index].first;
        auto sub_points = parlay::sequence<IdType>::from_function(
            end_index - start_index, [&](size_t j) { return levels[j + start_index].second; });
        stats<unsigned int> BuildStats(n + 1);
        I_list[level]->batch_insert(sub_points, G_list[level], *Points, BuildStats);
      }
    }
  }

  inline IdType find_cluster(IdType i) { return uf.find_compress(i); }

  inline DistTy linkage(IdType u, IdType v) {
    // u and v have to be representatives of clusters
    auto su = uf.get_size(u);
    auto sv = uf.get_size(v);
    return (*Points)[u].distance((*Points)[v]) * su * sv / (su + sv);
  }

  inline std::pair<IdType, DistTy> nearest_neighbor(IdType u) {
    auto cands =
        parlay::sequence<std::pair<DistTy, IdType>>::from_function(num_levels, [&](size_t i) {
          auto out = I_list[i]->search(u, G_list[i], *Points);
          if (out == u) {
            return std::make_pair(std::numeric_limits<DistTy>::max(), out);
          } else {
            return std::make_pair(linkage(u, out), out);
          }
        });
    auto min_elem = parlay::min_element(cands);
    return std::make_pair(min_elem->second, min_elem->first);
  }

  inline IdType merge_clusters(IdType u, IdType v) {
    size_t szu = uf.get_size(u);
    size_t szv = uf.get_size(v);
    IdType w = uf.unite(u, v);
    if (w != u) {
      std::swap(u, v);
      std::swap(szu, szv);
    }
    assert(szu >= szv);
    assert(w == u);
    Points->centroid(u, v, szu, szv);  // Update u to the centroid
    // Update the index
    size_t level_u = get_level(szu);
    size_t level_v = get_level(szv);
    auto szw = szu + szv;
    size_t level_w = get_level(szw);
    I_list[level_v]->inplace_delete(v, G_list[level_v], *Points);
    if (level_w != level_u) {
      I_list[level_u]->inplace_delete(u, G_list[level_u], *Points);
      I_list[level_w]->insert(w, G_list[level_w], *Points);
    }  // Else, w didn't move by much, so don't change
    return w;
  }

  inline auto get_center_ids() {
    auto reps = parlay::filter(parlay::iota(n), [&](IdType i) { return uf.find_compress(i) == i; });
    return reps;
  }
};

template<typename IdType, typename DistTy, typename Range, typename Seq>
auto ApxWardsHAC(Range &Points, uint32_t k, double eps, double delta, Seq &weights, BuildParams BP,
                 int verbose = 1) {
  using kv = std::tuple<DistTy, IdType, IdType>;  // Heap element type
  parlay::internal::timer t("HAC (HeapMin Method)");

  // Initialize Oracle
  QueryParams QP(2, BP.L, 10.0, Points.size(), BP.R);
  auto Oracle = OracleApx<IdType, Range>(BP, QP, delta);
  Oracle.init(Points, weights);
  size_t n = Oracle.n;

  // Setting verbose variables
  verbose = std::max(0, std::min(2, verbose));
  float verb_per = (verbose == 0) ? n : (verbose == 1) ? n / 10 : n / 100;
  float verb_inc = verb_per;

  // Initialize Heap
  auto h = parlay::sequence<kv>::from_function(n, [&](IdType i) {
    auto [ngh, dis] = Oracle.nearest_neighbor(i);
    return std::make_tuple(dis, i, ngh);
  });
  // TODO: Use own heap implementation
  auto H = std::priority_queue<kv, std::vector<kv>, std::greater<kv>>(h.begin(), h.end());
  t.next("Init Done");
  IdType u, v, w, u_orig, v_orig, cur = 0;
  DistTy dis, total_cost = 0;
  double one_plus_eps = 1 + eps;
  while (cur < n - k) {
    if (cur + 1 >= verb_per) {
      std::cout << "### " << verb_per * 100 / n << "\% done, Cost: " << total_cost << std::endl;
      verb_per += verb_inc;
    }
    std::tie(dis, u_orig, v_orig) = H.top();
    H.pop();
    u = Oracle.find_cluster(u_orig);
    v = Oracle.find_cluster(v_orig);
    if (u == v) {  // stale pair
      continue;
    } else {  // merge step
      auto actual_dist = Oracle.linkage(u, v);
      if (actual_dist <=
          one_plus_eps * dis) {  // If d(u,v) is at most (1+eps)*d(u_orig,v_orig)<=(1+eps)*opt
        w = Oracle.merge_clusters(u, v);
        total_cost += actual_dist;
        // dendrogram[rep[u]] = {cur + n, actual_dist};
        // dendrogram[rep[v]] = {cur + n, actual_dist};
        // rep[w] = cur + n;
        cur++;
      } else {  // Search nearest point to u, say v', and check if d(u,v')<=(1+eps)*dist <=
                // (1+eps)opt
        auto [best_ngh, best_ngh_dis] = Oracle.nearest_neighbor(u);
        if (best_ngh_dis <= one_plus_eps * dis) {
          w = Oracle.merge_clusters(u, best_ngh);
          total_cost += best_ngh_dis;
          cur++;
        } else {
          // Otherwise, go to next pair. Add the nearest neighbor of u to
          // the heap
          w = u;
        }
      }
      // Add the nearest neighbor of w to the heap
      auto [ngh, ngh_dist] = Oracle.nearest_neighbor(w);
      if (ngh != w) {
        H.push(std::make_tuple(ngh_dist, w, ngh));
      }
    }
  }
  t.next("HAC done");
  std::cout << "Total Cost: " << total_cost << std::endl;
  auto center_ids = Oracle.get_center_ids();
  return center_ids;
}