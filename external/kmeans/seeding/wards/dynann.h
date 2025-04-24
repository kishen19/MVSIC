// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <math.h>
#include <algorithm>
#include <random>
#include <set>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"

#include "algorithms/utils/beamSearch.h"
#include "algorithms/utils/graph.h"
#include "algorithms/utils/NSGDist.h"
#include "algorithms/utils/types.h"

#include "aug_map.h"
#include "dyn_point_range.h"

template<typename Point, typename PointRange, typename indexType>
struct dyn_knn_index {
  using distanceType = typename Point::distanceType;
  using pid = std::pair<indexType, distanceType>;
  using PR = PointRange;
  using GraphI = Graph<indexType>;
  using map = pam_map<basic_entry<indexType>>;
  
  BuildParams BP;
  // std::set<indexType> delete_set;
  map delete_set;
  map point_set;
  indexType start_point;
  indexType level;

  dyn_knn_index(BuildParams &BP_, indexType l) : BP(BP_), level(l) {}

  indexType get_start() { return start_point; }

  //robustPrune routine as found in DiskANN paper, with the exception
  //that the new candidate set is added to the field new_nbhs instead
  //of directly replacing the out_nbh of p
  parlay::sequence<indexType> robustPrune(indexType p, parlay::sequence<pid>& cand,
                    GraphI &G, PR &Points, double alpha, bool add = true) {
    // add out neighbors of p to the candidate set.
    size_t out_size = G[p].size();
    std::vector<pid> candidates;
    for (auto x : cand) candidates.push_back(x);

    if(add){
      for (size_t i=0; i<out_size; i++) {
        candidates.push_back(std::make_pair(G[p][i], Points[G[p][i]].distance(Points[p])));
      }
    }

    // Sort the candidate set according to distance from p
    auto less = [&](pid a, pid b) { return a.second < b.second; };
    std::sort(candidates.begin(), candidates.end(), less);

    // remove any duplicates
    auto new_end =std::unique(candidates.begin(), candidates.end(),
			      [&] (auto x, auto y) {return x.first == y.first;});
    candidates = std::vector(candidates.begin(), new_end);
    
    std::vector<indexType> new_nbhs;
    new_nbhs.reserve(BP.R);

    size_t candidate_idx = 0;

    while (new_nbhs.size() < BP.R && candidate_idx < candidates.size()) {
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
          distanceType dist_starprime = Points[p_star].distance(Points[p_prime]);
          distanceType dist_pprime = candidates[i].second;
          if (alpha * dist_starprime <= dist_pprime) {
            candidates[i].first = -1;
          }
        }
      }
    }

    auto new_neighbors_seq = parlay::to_sequence(new_nbhs);
    return new_neighbors_seq;
  }

  //wrapper to allow calling robustPrune on a sequence of candidates 
  //that do not come with precomputed distances
  parlay::sequence<indexType> robustPrune(indexType p, parlay::sequence<indexType> candidates,
                    GraphI &G, PR &Points, double alpha, bool add = true){

    parlay::sequence<pid> cc;
    cc.reserve(candidates.size()); // + size_of(p->out_nbh));
    for (size_t i=0; i<candidates.size(); ++i) {
      cc.push_back(std::make_pair(candidates[i], Points[candidates[i]].distance(Points[p])));
    }
    return robustPrune(p, cc, G, Points, alpha, add);
  }

  // add ngh to candidates without adding any repeats
  template<typename rangeType1, typename rangeType2>
  void add_neighbors_without_repeats(const rangeType1 &ngh, rangeType2& candidates) {
    std::unordered_set<indexType> a;
    for (auto c : candidates) a.insert(c);
    for (int i=0; i < ngh.size(); i++) 
      if (a.count(ngh[i]) == 0) candidates.push_back(ngh[i]);
  }

  void set_start(indexType s){
    start_point = s;
    point_set = map::insert(std::move(point_set),std::make_pair(s, 0));
  }

  void build_index(GraphI &G, PR &Points, stats<indexType> &BuildStats, indexType s=0) {
    set_start(s);
    parlay::sequence<indexType> inserts = parlay::tabulate(Points.size(), [&] (size_t i){
					    return static_cast<indexType>(i);});
              if(BP.two_pass) batch_insert(inserts, G, Points, BuildStats, 1.0, true, 2, .02, false);
              batch_insert(inserts, G, Points, BuildStats, BP.alpha, true, 2, .02, true);
    parlay::parallel_for (0, G.size(), [&] (long i) {
      auto less = [&] (indexType j, indexType k) {
		    return Points[i].distance(Points[j]) < Points[i].distance(Points[k]);};
      G[i].sort(less);});
    auto entries = parlay::sequence<std::pair<indexType, indexType>>::from_function(Points.size(), [&] (size_t i){
      return std::make_pair(i, 0);});
    point_set = map::multi_insert(std::move(point_set), entries);
  }

  // Inserts the given point into the graph.
  // First, computes candidate neighbors via beam search
  // Second, prunes out excess candidates and assigns neighbors
  // Finally, makes these edges bidirectional
  void insert(indexType p, GraphI &G, PR &Points) {
    auto alpha = BP.alpha;
    QueryParams QP((long) 0, BP.L, (double) 0.0, (long) Points.size(), (long) G.max_degree());
    parlay::sequence<pid> visited = 
      (beam_search<Point, PointRange, indexType>(Points[p], G, Points, start_point, QP)).first.second;
    auto nghs = robustPrune(p, visited, G, Points, alpha);
    G[p].update_neighbors(nghs);
    parlay::parallel_for(0, nghs.size(), [&](size_t i) {
      auto ngh = nghs[i];
      auto candidates = parlay::sequence<indexType>(1,p);
      if (G[ngh].size() < BP.R){
        add_neighbors_without_repeats(G[ngh], candidates);
	      G[ngh].update_neighbors(candidates);
      } else{
        auto new_out_2_ = robustPrune(ngh, std::move(candidates), G, Points, alpha);
        G[ngh].update_neighbors(new_out_2_);
      }
    });
    point_set = map::insert(std::move(point_set),std::make_pair(p, 0));
  }

  void batch_insert(parlay::sequence<indexType> &inserts,
                     GraphI &G, PR &Points, stats<indexType> &BuildStats, 
                     double alpha = 1.2, bool random_order = false, double base = 2,
                     double max_fraction = .02, bool update=true) {
    size_t n = G.size();
    size_t m = inserts.size();
    size_t inc = 0;
    size_t count = 0;
    float frac = 0.0;
    float progress_inc = .1;
    size_t max_batch_size = std::min(
        static_cast<size_t>(max_fraction * static_cast<float>(n)), 1000000ul);
    // fix bug where max batch size could be set to zero 
    if(max_batch_size == 0) max_batch_size = n;
    parlay::sequence<int> rperm;
    if (random_order)
      rperm = parlay::random_permutation<int>(static_cast<int>(m));
    else
      rperm = parlay::tabulate(m, [&](int i) { return i; });
    auto shuffled_inserts =
        parlay::tabulate(m, [&](size_t i) { return inserts[rperm[i]]; });
    parlay::internal::timer t_beam("beam search time");
    parlay::internal::timer t_bidirect("bidirect time");
    parlay::internal::timer t_prune("prune time");
    t_beam.stop();
    t_bidirect.stop();
    t_prune.stop();
    auto uf = new union_find<indexType>(G.size());
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
      parlay::sequence<parlay::sequence<indexType>> new_out_(ceiling-floor);
      // search for each node starting from the start_point, then call
      // robustPrune with the visited list as its candidate set
      t_beam.start();
      parlay::parallel_for(floor, ceiling, [&](size_t i) {
        size_t index = shuffled_inserts[i];
        QueryParams QP((long) 0, BP.L, (double) 0.0, (long) Points.size(), (long) G.max_degree());
        parlay::sequence<pid> visited = 
          (beam_search<Point, PointRange, indexType>(Points[index], G, Points, start_point, QP)).first.second;
        BuildStats.increment_visited(index, visited.size());
        new_out_[i-floor] = robustPrune(index, visited, G, Points, alpha); 
      });
      t_beam.stop();
      // make each edge bidirectional by first adding each new edge
      //(i,j) to a sequence, then semisorting the sequence by key values
      t_bidirect.start();
      auto to_flatten = parlay::tabulate(ceiling - floor, [&](size_t i) {
        indexType index = shuffled_inserts[i + floor];
        auto edges =
            parlay::tabulate(new_out_[i].size(), [&](size_t j) {
              return std::make_pair(new_out_[i][j], index);
            });
        return edges;
      });

      parlay::parallel_for(floor, ceiling, [&](size_t i) {
        G[shuffled_inserts[i]].update_neighbors(new_out_[i-floor]);
      });
      auto grouped_by = parlay::group_by_key(parlay::flatten(to_flatten));
      t_bidirect.stop();
      t_prune.start();
      // finally, add the bidirectional edges; if they do not make
      // the vertex exceed the degree bound, just add them to out_nbhs;
      // otherwise, use robustPrune on the vertex with user-specified alpha
      parlay::parallel_for(0, grouped_by.size(), [&](size_t j) {
        auto &[index, candidates] = grouped_by[j];
	      size_t newsize = candidates.size() + G[index].size();
        if (newsize <= BP.R) {
          add_neighbors_without_repeats(G[index], candidates);
          G[index].update_neighbors(candidates);
        } else {
          auto new_out_2_ = robustPrune(index, std::move(candidates), G, Points, alpha);
	        G[index].update_neighbors(new_out_2_);    
        }
      });
      t_prune.stop();
      if (true) {
        auto ind = frac * n;
        if (floor <= ind && ceiling > ind) {
          frac += progress_inc;
          std::cout << "Pass " << 100 * frac << "% complete"
                    << std::endl;
        }
      }
      inc += 1;
    }
    t_beam.total();
    t_bidirect.total();
    t_prune.total();
    if (update){
      auto entries = parlay::sequence<std::pair<indexType, indexType>>::from_function(inserts.size(), [&] (size_t i){
        return std::make_pair(inserts[i], 0);});
      point_set = map::multi_insert(std::move(point_set), entries);
    }
  }

  void lazy_delete(indexType p, GraphI &G, PR &Points) {
    if (p > (int)G.size()) {
      std::cout << "ERROR: invalid point " << p << " given to lazy_delete"
                << std::endl;
      abort();
    }
    if (p == start_point) {
      std::cout << "Deleting start_point not permitted; continuing" << std::endl;
      return;
    }
    delete_set = map::insert(std::move(delete_set),std::make_pair(p, 0));
    point_set = map::remove(std::move(point_set), p);
    // Consolidate only when the number of deleted points exceeds 10% of
    // the max number of points at that level.
    if (delete_set.size() > std::max((size_t)1000, G.size()/level/10)) {
      std::cout << "Consolidating deletes" << std::endl;
      consolidate_deletes(G, Points);
    }
  }

  void consolidate_deletes(GraphI &G, PR &Points) {
    // Update the neighborhood of deleted points
    // by removing deleted points from the neighborhood
    auto map_deletes = [&] (const auto &et) {
			auto [i, v_] = et;
      parlay::sequence<indexType> new_edges;
      for (int j=0; j<G[i].size(); j++){
        if (!delete_set.find(G[i][j])){
          new_edges.push_back(G[i][j]);
        }
      }
      if(new_edges.size() < G[i].size()){ // Nghs changed
        G[i].update_neighbors(new_edges);
      } 
    };
    map::map_void(delete_set, map_deletes);
    // Update the neighborhood of non-deleted points
    auto map_points = [&] (const auto &et) {
      auto [i, v_] = et;
      parlay::sequence<indexType> new_out;
      bool modify = false;
      for(int j=0; j<G[i].size(); j++){
        if(!delete_set.find(G[i][j])){
          new_out.push_back(G[i][j]);
        } else {
          modify = true;
          int index = G[i][j];
          for (int k=0; k<G[index].size(); k++){
            if (G[index][k] != i){
              new_out.push_back(G[index][k]);
            }
          }
        }
        //TODO only prune if overflow happens
        if(modify){
          auto new_nghs = robustPrune(i, new_out, G, Points, BP.alpha, false);
          G[i].update_neighbors(new_nghs);
        }
      }  
    };
    map::map_void(point_set, map_points);
    
    auto map_clear = [&] (const auto &et) {
      auto [i, v_] = et;
      G[i].clear_neighbors();
    };
    map::map_void(delete_set, map_clear);
    delete_set.clear();
  }

  void inplace_delete(indexType p, GraphI &G, PR &Points) {
    if (p > (int)G.size()) {
      std::cout << "ERROR: invalid point " << p << " given to lazy_delete"
                << std::endl;
      abort();
    }
    if (p == start_point) {
      std::cout << "Deleting start_point not permitted; continuing" << std::endl;
      return;
    }
    auto QP = QueryParams(10, 128, 1.35, (long)G.size(), (long)G.max_degree());
    auto [pairElts, dist_cmps] = beam_search(Points[p], G, Points, start_point, QP);
    auto [beamElts, visited] = pairElts;
    size_t ind = 0;
    auto candidates = parlay::sequence<indexType>();
    while (ind < visited.size() && candidates.size() < 10){
      if (visited[ind].first != p && visited[ind].first != start_point && !delete_set.find(visited[ind].first)){ 
        candidates.push_back(visited[ind].first);
      }
      ind++;
    }
    if (candidates.size() > 0){
      size_t c = 3;
      // Fix In-neighbors of p
      parlay::parallel_for(0, visited.size(), [&](size_t i) {
        auto z = visited[i].first;
        auto z_nghs = parlay::sequence<indexType>();
        bool modify = false;
        for (int j=0; j<G[z].size(); j++){
          if (G[z][j] == p){
            modify = true;
          } else {
            z_nghs.push_back(G[z][j]);
          }
        }
        if (modify){
          auto Cz = parlay::sequence<indexType>();
          auto dist_z = parlay::sequence<std::pair<distanceType, indexType>>::from_function(candidates.size(), [&] (size_t k){
            return std::make_pair(Points[z].distance(Points[candidates[k]]), candidates[k]);
          });
          std::sort(dist_z.begin(), dist_z.end());
          for (size_t k = 0; k < std::min(c, dist_z.size()); k++) {
            Cz.push_back(dist_z[k].second);
          }
          add_neighbors_without_repeats(z_nghs, Cz);
          if (Cz.size() > BP.R){
            auto new_nghs = robustPrune(z, Cz, G, Points, 1.2, false);
            G[z].update_neighbors(new_nghs);
          } else {
            G[z].update_neighbors(Cz);
          }
        }
      });
      // Fix Out-neighbors of p
      for (int i=0; i<G[p].size(); i++){
        auto w = G[p][i];
        if (delete_set.find(w)){
          continue;
        }
        auto w_nghs = parlay::sequence<indexType>();
        auto dist_w = parlay::sequence<std::pair<distanceType, indexType>>::from_function(candidates.size(), [&] (size_t k){
          return std::make_pair(Points[w].distance(Points[candidates[k]]), candidates[k]);
        });
        std::sort(dist_w.begin(), dist_w.end());
        for (size_t k = 0; k < std::min(c, dist_w.size()); k++) {
          auto y = dist_w[k].second;
          auto Cw = parlay::sequence<indexType>(1,w);
          add_neighbors_without_repeats(G[y], Cw);
          if (Cw.size() > BP.R){
            auto new_nghs = robustPrune(y, Cw, G, Points, 1.2, false);
            G[y].update_neighbors(new_nghs);
          } else {
            G[y].update_neighbors(Cw);
          }
        }
      }
    }
    // Delete p
    delete_set = map::insert(std::move(delete_set),std::make_pair(p, 0));
    point_set = map::remove(std::move(point_set), p);
    G[p].clear_neighbors();

    // Consolidate only when the number of deleted points exceeds 10% of
    // the max number of points at that level.
    if (delete_set.size() > std::max((size_t)500, point_set.size()/4)) {
      std::cout << "Consolidating deletes" << std::endl;
      simple_consolidate_deletes(G, Points);
    }
  }

  void simple_consolidate_deletes(GraphI &G, PR &Points) {
    // Remove deleted points from the neighborhood of non-deleted points
    auto map_points = [&] (const auto &et) {
      auto [i, v_] = et;
      parlay::sequence<indexType> new_out;
      for(int j=0; j<G[i].size(); j++){
        if(!delete_set.find(G[i][j])){
          new_out.push_back(G[i][j]);
        }
      }
      G[i].update_neighbors(new_out);
    };
    map::map_void(point_set, map_points);
    delete_set.clear();
  }

  auto search(indexType p, GraphI &G, PR &Points) {
    if (point_set.size() == 0){
      std::cerr << "Index not built yet" << std::endl;
      abort();
    } else if (point_set.size() == 1){
      return p;
    } else if (point_set.size() <= 500){
      auto dists = parlay::sequence<std::pair<distanceType, indexType>>(point_set.size());
      auto map_dist = [&] (const auto &et, const auto &ind) {
        auto [i, v_] = et;
        if (i!=p && i!=start_point) {
          dists[ind] = std::make_pair(Points[p].distance(Points[i]), i);
        } else {
          dists[ind] = std::make_pair(std::numeric_limits<distanceType>::max(), p);
        }
      };
      map::map_index(point_set, map_dist);
      auto min_elem = parlay::min_element(dists);
      return min_elem->second;
    } else{
      auto QP = QueryParams(1, BP.L, 1.35, (long)G.size(), (long)G.max_degree());
      auto [pairElts, dist_cmps] = beam_search(Points[p], G, Points, start_point, QP);
      auto [beamElts, out] = pairElts;
      size_t ind = 0;
      while (ind < out.size()){
        if (out[ind].first != p && out[ind].first != start_point && !delete_set.find(out[ind].first)){ 
          return out[ind].first;
        }
        ind++;
      }
      return p;
    }
  }
};