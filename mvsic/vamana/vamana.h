#pragma once

#include <variant>
#include <fstream>
#include <cstdint>

#include "mvsic/core/index.h"
#include "mvsic/core/query_compression.h"

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
  using MVQT = typename Index<metric>::MVQT;
  using QuantSet = typename MVQT::QuantSet;
  using QuantQuery = typename MVQT::QuantQuery;
  using QuantModel = typename MVQT::QuantModel;
  using QT = typename Index<metric>::QT;
  using Index<metric>::d;       // Embedding dimension
  using Index<metric>::params;  // Index Params
  using Index<metric>::quantization_mode;

  vamana::Graph<uint32_t> G;  // Vamana Graph
  uint32_t start_point;       // Starting Point of the graph

  // Quantizer Storage
  QuantModel quantizer = std::monostate{};
  QuantSet quantized_points = std::monostate{};

  IndexVamana(uint32_t d_) noexcept {
    d = d_;
    params = IndexParams::vamana();
  }
  IndexVamana(uint32_t d_, const IndexParams& params_) noexcept {
    d = d_;
    params = params_;
  }

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
    new_nbhs.reserve(params.R);

    size_t candidate_idx = 0;

    while (new_nbhs.size() < params.R && candidate_idx < candidates.size()) {
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
        SearchParams search_params = SearchParams::vamana((long)0, params.L, (double)0.0);
        parlay::sequence<pid> visited =
            std::get<0>(
                mvsic::vamana::beam_search(points[index], G, points, start_point, search_params))
                .second;
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
        if (newsize <= params.R) {
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
    // Step 1: Train Quantizer
    quantization_mode = params.pq.method;
    this->train_quantizer(points, quantizer);

    // Step 2: Encode Data
    quantized_points = this->encode_points_quantized(points, quantizer);

    // Step 3: Build Graph (uses exact distances)
    if (params.verbose >= 1) std::cout << "Building graph..." << std::endl;
    set_start();
    G = vamana::Graph<uint32_t>(params.R, points.size());
    auto inserts = parlay::tabulate(points.size(), [&](uint32_t i) { return i; });
    if (params.two_pass) batch_insert(inserts, points, 1.0, true, 2, .02);
    batch_insert(inserts, points, params.alpha, true, 2, .02);
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
  std::tuple<parlay::sequence<std::pair<uint32_t, float>>, size_t, std::vector<double>>
  search_with_stats(const ChPoint& query, const PointCloudSet<ChPoint>& points,
                    const SearchParams& search_params) override {
    parlay::internal::timer t;
    size_t k = search_params.k;
    size_t bytes_accessed = 0;

    size_t dist_cmps = 0;
    double t_compress = 0.0;
    double t_quantize = 0.0;
    double t_search = 0.0;
    double t_rerank = 0.0;

    // Step 0: Compress query point cloud (optional)
    t.start();
    CompressedPointCloud<ChPoint> compressed_storage;
    ChPoint effective_query = query;
    if (search_params.query_compression != SearchParams::QueryCompression::None) {
      uint32_t ba = qc_internal::batch_alignment(quantization_mode);
      compressed_storage = compress_query<ChPoint>(
          query, search_params.query_compression,
          search_params.query_compression_threshold, ba);
      effective_query = compressed_storage.view();
    }
    t_compress = t.stop();
    t.reset();

    // Step 1: Quantize Query
    t.start();
    QuantQuery q_query_var = this->quantize_query_point_cloud(effective_query, quantizer);
    t_quantize = t.stop();
    t.reset();

    // Step 2: Run beam search
    t.start();
    parlay::sequence<pid> visited;

    switch (quantization_mode) {
      case QT::PQ: {
        auto& q = std::get<typename MVQT::PQ_Query>(q_query_var);
        auto& d = std::get<typename MVQT::PQ_Set>(quantized_points);
        auto [result, cmps, bytes_acc] =
            mvsic::vamana::beam_search(q, G, d, start_point, search_params);
        visited = result.second;
        dist_cmps = cmps;
        bytes_accessed = bytes_acc;
        break;
      }
      case QT::RaBitQ: {
        auto& q = std::get<typename MVQT::RQ_Query>(q_query_var);
        auto& d = std::get<typename MVQT::RQ_Set>(quantized_points);
        auto [result, cmps, bytes_acc] =
            mvsic::vamana::beam_search(q, G, d, start_point, search_params);
        visited = result.second;
        dist_cmps = cmps;
        bytes_accessed = bytes_acc;
        break;
      }
      case QT::FastScan: {
        auto& q = std::get<typename MVQT::FS_Query>(q_query_var);
        auto& d = std::get<typename MVQT::FS_Set>(quantized_points);
        auto [result, cmps, bytes_acc] =
            mvsic::vamana::beam_search(q, G, d, start_point, search_params);
        visited = result.second;
        dist_cmps = cmps;
        bytes_accessed = bytes_acc;
        break;
      }
      case QT::TurboQuant: {
        auto& q = std::get<typename MVQT::TQ_Query>(q_query_var);
        auto& d = std::get<typename MVQT::TQ_Set>(quantized_points);
        auto [result, cmps, bytes_acc] =
            mvsic::vamana::beam_search(q, G, d, start_point, search_params);
        visited = result.second;
        dist_cmps = cmps;
        bytes_accessed = bytes_acc;
        break;
      }
      case QT::None: {
        auto [result, cmps, bytes_acc] =
            mvsic::vamana::beam_search(effective_query, G, points, start_point, search_params);
        visited = result.second;
        dist_cmps = cmps;
        bytes_accessed = bytes_acc;
        break;
      }
      default: break;
    }
    t_search = t.stop();
    t.reset();

    // Step 3: Re-ranking
    t.start();
    const ChPoint& rerank_query = search_params.compress_rerank ? effective_query : query;
    auto final_results =
        parlay::sequence<std::pair<uint32_t, float>>::uninitialized(std::min(k, visited.size()));
    if (search_params.num_rerank > 0) {
      size_t num_rerank = std::min(search_params.num_rerank, visited.size());
      bytes_accessed += this->rerank(rerank_query, points, visited, num_rerank, final_results);
    } else {
      parlay::parallel_for(0, final_results.size(),
                           [&](size_t i) { final_results[i] = visited[i]; });
    }
    t_rerank = t.stop();
    t.reset();

    std::vector<double> stats;
    stats.push_back(static_cast<double>(dist_cmps));
    stats.push_back(t_compress);
    stats.push_back(t_quantize);
    stats.push_back(t_search);
    stats.push_back(t_rerank);

    return std::make_tuple(final_results, bytes_accessed, stats);
  }

  // Write the index to a file in disk
  void save(const std::string& filename) override {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
      throw std::runtime_error("save: cannot open file: " + filename);
    }
    // 1. Save graph
    G.save(out);
    // 2. Save Quantizer Model
    int type_id = static_cast<int>(quantization_mode);
    out.write(reinterpret_cast<const char*>(&type_id), sizeof(int));
    switch (quantization_mode) {
      case QT::PQ: std::get<typename MVQT::PQ_Model>(quantizer).save(out); break;
      case QT::FastScan: std::get<typename MVQT::FS_Model>(quantizer).save(out); break;
      case QT::RaBitQ: std::get<typename MVQT::RQ_Model>(quantizer).save(out); break;
      case QT::TurboQuant: std::get<typename MVQT::TQ_Model>(quantizer).save(out); break;
      case QT::None: break;
      default: break;
    }
    out.close();
  }

  // Read the index from a file in disk
  void load(const std::string& filename, const PointCloudSet<ChPoint>& points) override {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
      throw std::runtime_error("load: cannot open file: " + filename);
    }
    // 1. Load Graph
    G = vamana::Graph<uint32_t>(in);
    set_start();

    // 2. Load Quantizer Model
    int type_id;
    in.read(reinterpret_cast<char*>(&type_id), sizeof(int));
    quantization_mode = static_cast<QT>(type_id);
    switch (quantization_mode) {
      case QT::PQ: quantizer.template emplace<typename MVQT::PQ_Model>().load(in); break;
      case QT::FastScan: quantizer.template emplace<typename MVQT::FS_Model>().load(in); break;
      case QT::RaBitQ: quantizer.template emplace<typename MVQT::RQ_Model>().load(in); break;
      case QT::TurboQuant: quantizer.template emplace<typename MVQT::TQ_Model>().load(in); break;
      case QT::None: break;
      default: break;
    }
    in.close();

    // 3. Re-encode data
    quantized_points = this->encode_points_quantized(points, quantizer);
  }
};

using IndexVamanaL2 = IndexVamana<true>;   // L2 metric
using IndexVamanaIP = IndexVamana<false>;  // MIPS

}  // namespace mvsic
