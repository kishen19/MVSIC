#pragma once

// Surgical refactor of the inner-Lloyd's k-means used in MVClustering's
// centroid-update step.  The algorithm (seed -> repeat {assign, update}) is
// expressed in terms of a small `Backend` concept so the same iteration loop
// can drive different point representations (float today, an 8-bit
// TurboQuant encoding next).  Only the encoding changes; the algorithm is
// fixed.
//
// The driver and the float backend below preserve the exact behavior of the
// previous kmeans_subsample / kmeans_weighted_subsample call path (parlayANN
// PrefixDoubling / UniformlyRandom seed, blocked Eigen pairwise assignment,
// per-cluster mean / weighted-mean centroid update with IP-side
// re-normalization, parlayANN's static-counter empty-cluster sampling).

#include <Eigen/Core>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string_view>
#include <utility>
#include <vector>

#include "algorithms/utils/euclidian_point.h"
#include "algorithms/utils/mips_point.h"
#include "algorithms/utils/point_range.h"
#include "mvsic/core/quantization/turboquant_8bit_mv.h"
#include "parlay/delayed_sequence.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
// `lloyds/kmeans.h` has #pragma once and transitively brings in
// `seeding/prefixdoubling.h` and `seeding/uniformlyrandom.h` (which lack
// include guards) plus `lloyds/pairwise.h`'s
// `compute_cluster_ids_pairwise_blocked`. Including those headers directly
// would cause redefinition errors when this file and the `kmeans_util.h`
// path both pull them in.
#include "lloyds/kmeans.h"

namespace mvsic::lloyds {

// ---------------------------------------------------------------------------
// CenterSet: flat row-major float buffer of k * d floats.
//
// The Lloyd's driver and every backend produce / consume centers in this
// shape.  This keeps the algorithm encoding-agnostic while giving downstream
// callers a single concrete type to set_point_cloud against.
// ---------------------------------------------------------------------------
class CenterSet {
 public:
  CenterSet() = default;
  CenterSet(uint32_t k, uint32_t d)
      : data_(static_cast<size_t>(k) * static_cast<size_t>(d), 0.0f), k_(k), d_(d) {}

  size_t size() const { return k_; }
  uint32_t get_dims() const { return d_; }

  float* operator[](size_t i) { return data_.data() + static_cast<size_t>(i) * d_; }
  const float* operator[](size_t i) const {
    return data_.data() + static_cast<size_t>(i) * d_;
  }

  float* data() { return data_.data(); }
  const float* data() const { return data_.data(); }

 private:
  parlay::sequence<float> data_;
  uint32_t k_ = 0;
  uint32_t d_ = 0;
};

// ---------------------------------------------------------------------------
// Backend concept (compile-time duck-typed).
//
// Required member functions:
//   size_t   size() const;
//   uint32_t dims() const;
//   bool     is_metric() const;     // true for L2, false for IP/Mips
//
//   parlay::sequence<uint32_t> seed_uniform_random(uint32_t k);
//   parlay::sequence<uint32_t> seed_prefix_doubling(uint32_t k);
//
//   CenterSet copy_centers_by_ids(const parlay::sequence<uint32_t>& ids) const;
//
//   void assign(const CenterSet& centers,
//               parlay::sequence<uint32_t>& cluster_ids);
//
//   CenterSet compute_means(parlay::sequence<uint32_t>& cluster_ids, uint32_t k);
//   CenterSet compute_means_weighted(parlay::sequence<uint32_t>& cluster_ids,
//                                    const parlay::sequence<float>& weights,
//                                    uint32_t k);
//
//   float cost(const CenterSet& centers,
//              const parlay::sequence<uint32_t>& cluster_ids) const;
// ---------------------------------------------------------------------------

template <class Backend>
CenterSet lloyds_kmeans(Backend& B, uint32_t k, std::string_view seed_algo, size_t niters,
                        bool verbose,
                        const parlay::sequence<float>* opt_weights = nullptr) {
  parlay::sequence<uint32_t> center_ids;
  if (seed_algo == "PrefixDoubling") {
    center_ids = B.seed_prefix_doubling(k);
  } else if (seed_algo == "UniformlyRandom") {
    center_ids = B.seed_uniform_random(k);
  } else {
    std::cerr << "[lloyds_kmeans] Error: unsupported seed_algo '" << seed_algo << "'."
              << std::endl;
    std::abort();
  }

  CenterSet centers = B.copy_centers_by_ids(center_ids);

  parlay::sequence<uint32_t> cluster_ids;
  B.assign(centers, cluster_ids);
  if (verbose) {
    std::cout << "[lloyds_kmeans] cost (Seeding): " << B.cost(centers, cluster_ids)
              << std::endl;
  }

  for (size_t it = 0; it < niters; ++it) {
    if (opt_weights != nullptr) {
      centers = B.compute_means_weighted(cluster_ids, *opt_weights, k);
    } else {
      centers = B.compute_means(cluster_ids, k);
    }
    B.assign(centers, cluster_ids);
    if (verbose) {
      std::cout << "[lloyds_kmeans] cost (" << it + 1 << "): "
                << B.cost(centers, cluster_ids) << std::endl;
    }
  }
  return centers;
}

// ---------------------------------------------------------------------------
// FloatLloydsBackend: stores points as a parlayANN PointRange<float>.  All
// operations preserve the exact semantics of the previous parlayANN-based
// path:
//   * seeding            -> parlayANN PrefixDoubling / UniformlyRandom
//   * assignment         -> Eigen-blocked pairwise GEMM (BLOCK_SIZE = 256)
//   * mean / weighted    -> per-cluster float reduction; IP-side L2 normalize
//   * empty cluster      -> sample from input via the same hash32-based PRNG
//                           with a per-call seed counter (matches parlayANN's
//                           static `seed` semantics)
// ---------------------------------------------------------------------------
template <bool metric>
class FloatLloydsBackend {
 public:
  using PointTy =
      std::conditional_t<metric, parlayANN::Euclidian_Point<float>, parlayANN::Mips_Point<float>>;
  using Range = parlayANN::PointRange<PointTy>;

  // Build from an existing parlay::sequence<parlay::sequence<float>>.  The
  // PointRange constructor performs a parallel deep copy into its aligned
  // buffer, matching the behavior of the previous kmeans_subsample wrapper.
  FloatLloydsBackend(const parlay::sequence<parlay::sequence<float>>& data, uint32_t d)
      : range_(data, d) {}

  // For callers that already hold a sampled / delayed sequence: same as above,
  // any Seq with operator[] -> something with operator[] returning float and a
  // .size() will work (parlayANN's translate_point handles the per-row copy).
  template <typename Seq>
  FloatLloydsBackend(const Seq& data, uint32_t d) : range_(data, d) {}

  size_t size() const { return range_.size(); }
  uint32_t dims() const { return range_.get_dims(); }
  static constexpr bool is_metric() { return metric; }

  parlay::sequence<uint32_t> seed_uniform_random(uint32_t k) {
    return UniformlyRandom<float>(range_, k);
  }
  parlay::sequence<uint32_t> seed_prefix_doubling(uint32_t k) {
    return PrefixDoubling<float>(range_, k);
  }

  CenterSet copy_centers_by_ids(const parlay::sequence<uint32_t>& ids) const {
    const uint32_t d = range_.get_dims();
    CenterSet out(static_cast<uint32_t>(ids.size()), d);
    parlay::parallel_for(0, ids.size(), [&](size_t i) {
      auto src = range_[ids[i]];
      float* dst = out[i];
      for (uint32_t t = 0; t < d; ++t) dst[t] = src[t];
    });
    return out;
  }

  // Assignment: build a temporary Range over the float CenterSet (deep-copy
  // of k * d floats, cheap) and delegate to parlayANN's blocked Eigen kernel
  // so we use the exact same GEMM path as before.
  void assign(const CenterSet& centers, parlay::sequence<uint32_t>& cluster_ids) {
    const uint32_t d = range_.get_dims();
    const size_t k = centers.size();
    auto centers_seq = parlay::tabulate(k, [&](size_t i) {
      parlay::sequence<float> row(d);
      const float* src = centers[i];
      for (uint32_t t = 0; t < d; ++t) row[t] = src[t];
      return row;
    });
    Range centers_range(centers_seq, d);
    cluster_ids = compute_cluster_ids_pairwise_blocked<PointTy>(range_, centers_range);
  }

  CenterSet compute_means(parlay::sequence<uint32_t>& cluster_ids, uint32_t k) {
    return compute_means_impl_(cluster_ids, k, /*weights=*/nullptr);
  }

  CenterSet compute_means_weighted(parlay::sequence<uint32_t>& cluster_ids,
                                   const parlay::sequence<float>& weights, uint32_t k) {
    return compute_means_impl_(cluster_ids, k, &weights);
  }

  float cost(const CenterSet& centers,
             const parlay::sequence<uint32_t>& cluster_ids) const {
    const uint32_t d = range_.get_dims();
    const size_t k = centers.size();
    auto centers_seq = parlay::tabulate(k, [&](size_t i) {
      parlay::sequence<float> row(d);
      const float* src = centers[i];
      for (uint32_t t = 0; t < d; ++t) row[t] = src[t];
      return row;
    });
    Range centers_range(centers_seq, d);
    auto distances = parlay::delayed_seq<float>(range_.size(), [&](size_t i) {
      return range_[i].distance(centers_range[cluster_ids[i]]);
    });
    return parlay::reduce(distances);
  }

 private:
  CenterSet compute_means_impl_(parlay::sequence<uint32_t>& cluster_ids, uint32_t k,
                                const parlay::sequence<float>* weights) {
    const size_t n = range_.size();
    const uint32_t d = range_.get_dims();

    auto pairs = parlay::sequence<std::pair<uint32_t, uint32_t>>::from_function(
        n, [&](size_t i) { return std::make_pair(cluster_ids[i], static_cast<uint32_t>(i)); });
    auto grouped = parlay::group_by_index(pairs, k);

    CenterSet out(k, d);
    // Mirror parlayANN's `static uint32_t seed = 0` inside
    // compute_centroids_from_clusters / compute_centroids_from_clusters_weighted:
    // a per-template-instance counter shared across all calls in the program,
    // advanced by `k` after each empty-cluster sampling step.  We keep two
    // separate statics so the unweighted and weighted paths get distinct
    // sequences, matching the two distinct parlayANN function templates.
    static uint32_t empty_seed_unweighted = 0;
    static uint32_t empty_seed_weighted = 0;
    uint32_t empty_seed_base =
        (weights == nullptr) ? empty_seed_unweighted : empty_seed_weighted;

    parlay::parallel_for(0, k, [&](size_t cid) {
      const auto& g = grouped[cid];
      if (g.size() > 0) {
        if (weights != nullptr) {
          // Weighted mean: sum_j w_j * x_j / sum_j w_j.
          auto subset_w = parlay::delayed_seq<float>(
              g.size(), [&](size_t j) { return (*weights)[g[j]]; });
          float total_w = parlay::reduce(subset_w);
          if (total_w == 0.0f) total_w = 1.0f;
          for (uint32_t dim = 0; dim < d; ++dim) {
            auto coord_w = parlay::delayed_seq<float>(g.size(), [&](size_t j) {
              return static_cast<float>(range_[g[j]][dim]) * (*weights)[g[j]];
            });
            out[cid][dim] = parlay::reduce(coord_w) / total_w;
          }
        } else {
          // Plain mean: sum_j x_j (final divide / normalize below).
          for (uint32_t dim = 0; dim < d; ++dim) {
            auto coord = parlay::delayed_seq<float>(g.size(), [&](size_t j) {
              return static_cast<float>(range_[g[j]][dim]);
            });
            out[cid][dim] = parlay::reduce(coord);
          }
          if constexpr (metric) {
            float num_points = static_cast<float>(g.size());
            for (uint32_t dim = 0; dim < d; ++dim) {
              out[cid][dim] /= num_points;
            }
          }
        }
        // IP / Mips: re-normalize the (weighted) sum to unit norm.
        if constexpr (!metric) {
          double sum_sqrs = 0.0;
          for (uint32_t dim = 0; dim < d; ++dim) {
            double v = out[cid][dim];
            sum_sqrs += v * v;
          }
          if (sum_sqrs != 0.0) {
            float inv = 1.0f / static_cast<float>(std::sqrt(sum_sqrs));
            for (uint32_t dim = 0; dim < d; ++dim) {
              out[cid][dim] *= inv;
            }
          }
        }
      } else {
        // Empty cluster: sample a point from the data, mirroring parlayANN's
        // hash32(seed + cid) % n with a per-call seed offset.
        uint32_t id = parlay::hash32(empty_seed_base + static_cast<uint32_t>(cid)) % n;
        auto p = range_[id];
        for (uint32_t dim = 0; dim < d; ++dim) {
          out[cid][dim] = p[dim];
        }
        cluster_ids[id] = static_cast<uint32_t>(cid);
      }
    });
    if (weights == nullptr) {
      empty_seed_unweighted += k;
    } else {
      empty_seed_weighted += k;
    }
    return out;
  }

  Range range_;
};

// ---------------------------------------------------------------------------
// TQ8VectorCache: read-only view into a build-top-level per-vector encoded
// query buffer. When TQ8LloydsBackend is constructed with one of these (plus
// a per-row index into the buffer), the per-row rotate+quantize step is
// skipped entirely — the encoded query state is gathered out of the shared
// buffer and the kernel runs against centers encoded under `model`'s rotator.
//
// Only consumed by the cache-aware constructors below; the existing data-only
// constructors keep using GetSharedModel_ and re-encoding from scratch.
// ---------------------------------------------------------------------------
template <bool metric>
struct TQ8VectorCache {
  using QModel = ::mvsic::turboquant_8bit_mv::Model<metric>;
  const QModel* model = nullptr;
  size_t q_stride = 0;
  const int8_t* q_data = nullptr;
  const float* q_nsf = nullptr;
  const float* q_sqn = nullptr;
  const int32_t* q_bsum = nullptr;
};

// ---------------------------------------------------------------------------
// TQ8LloydsBackend: same surface as FloatLloydsBackend, but the inner
// assignment runs through the int8 TurboQuant VPDPBUSD panel kernel from
// `turboquant_8bit_mv.h`.  Centers stay float; only the n×k pairwise
// distance step uses int8.
//
// At construction we:
//   * keep a `parlayANN::PointRange<float>` for seeding, copy_centers_by_ids,
//     compute_means, and cost (all of which want float random access);
//   * train the 8BTQ Hadamard rotator (`Model<metric>::train`) on the
//     point dimension;
//   * pre-encode every input row as a 1-vector EncodedQuery
//     (Quantized_Query_Point_Cloud<metric>) — this happens once and is
//     reused across every Lloyd iteration.
//
// At each `assign` we encode the float CenterSet as a
// Quantized_Point_Cloud_Set (k 1-vector clouds) and call
// `ManyToMany::TopKIntoUninitialized` with k=1, exactly mirroring
// `MVClustering8BTQ::train`'s outer assignment step.
//
// Cache-aware overload: when given a TQ8VectorCache + per-row index list,
// the encoded-query state is gathered from the cache instead of re-rotated/
// quantized.  Centers are encoded with the *cache's* model (so the rotator
// matches the one that produced the cached encodings).
// ---------------------------------------------------------------------------
template <bool metric>
class TQ8LloydsBackend {
 public:
  using PointTy =
      std::conditional_t<metric, parlayANN::Euclidian_Point<float>, parlayANN::Mips_Point<float>>;
  using Range = parlayANN::PointRange<PointTy>;
  using QModel = ::mvsic::turboquant_8bit_mv::Model<metric>;
  using EncSet = typename QModel::EncodedSet;
  using EncQuery = typename QModel::EncodedQuery;

  TQ8LloydsBackend(const parlay::sequence<parlay::sequence<float>>& data, uint32_t d)
      : d_(d), model_(&GetSharedModel_(d)) {
    n_ = data.size();
    data_.resize(static_cast<size_t>(n_) * d_);
    parlay::parallel_for(0, n_, [&](size_t i) {
      float* dst = data_.data() + i * d_;
      for (uint32_t t = 0; t < d_; ++t) dst[t] = data[i][t];
    });
    build_encoded_();
  }

  template <typename Seq>
  TQ8LloydsBackend(const Seq& data, uint32_t d) : d_(d), model_(&GetSharedModel_(d)) {
    n_ = data.size();
    data_.resize(static_cast<size_t>(n_) * d_);
    parlay::parallel_for(0, n_, [&](size_t i) {
      auto src = data[i];
      float* dst = data_.data() + i * d_;
      for (uint32_t t = 0; t < d_; ++t) dst[t] = src[t];
    });
    build_encoded_();
  }

  // Cache-aware constructors: float points are still copied (compute_means /
  // cost / seed_prefix_doubling need float random access), but the encoded
  // query state is gathered from the supplied `cache` using `vec_root_indices`
  // -- skipping the rotate+quantize work that build_encoded_ otherwise does.
  TQ8LloydsBackend(const parlay::sequence<parlay::sequence<float>>& data, uint32_t d,
                   const TQ8VectorCache<metric>& cache,
                   const parlay::sequence<uint32_t>& vec_root_indices)
      : d_(d), model_(cache.model) {
    n_ = data.size();
    data_.resize(static_cast<size_t>(n_) * d_);
    parlay::parallel_for(0, n_, [&](size_t i) {
      float* dst = data_.data() + i * d_;
      for (uint32_t t = 0; t < d_; ++t) dst[t] = data[i][t];
    });
    gather_encoded_(cache, vec_root_indices);
  }

  template <typename Seq>
  TQ8LloydsBackend(const Seq& data, uint32_t d, const TQ8VectorCache<metric>& cache,
                   const parlay::sequence<uint32_t>& vec_root_indices)
      : d_(d), model_(cache.model) {
    n_ = data.size();
    data_.resize(static_cast<size_t>(n_) * d_);
    parlay::parallel_for(0, n_, [&](size_t i) {
      auto src = data[i];
      float* dst = data_.data() + i * d_;
      for (uint32_t t = 0; t < d_; ++t) dst[t] = src[t];
    });
    gather_encoded_(cache, vec_root_indices);
  }

  // Cache-aware ctor that takes ownership of a pre-filled flat float buffer
  // of size n*d (row-major).  Skips the per-row deep copy the seq<seq> ctors
  // do.  Caller must guarantee data_flat.size() == n * d.
  TQ8LloydsBackend(parlay::sequence<float>&& data_flat, size_t n, uint32_t d,
                   const TQ8VectorCache<metric>& cache,
                   const parlay::sequence<uint32_t>& vec_root_indices)
      : d_(d), model_(cache.model) {
    n_ = n;
    data_ = std::move(data_flat);
    gather_encoded_(cache, vec_root_indices);
  }

  size_t size() const { return n_; }
  uint32_t dims() const { return d_; }
  static constexpr bool is_metric() { return metric; }

  parlay::sequence<uint32_t> seed_uniform_random(uint32_t k) {
    // Inline UniformlyRandom: the parlayANN function only reads points.size().
    parlay::sequence<uint32_t> centers(k);
    parlay::parallel_for(0, k, [&](size_t i) {
      centers[i] = parlay::hash32(static_cast<uint32_t>(i)) % n_;
    });
    return centers;
  }
  parlay::sequence<uint32_t> seed_prefix_doubling(uint32_t k) {
    // PrefixDoubling computes distances between points, so we need a
    // PointRange.  Lazily build one only when L2 builds need it.
    auto pr_view = MakeRangeView_();
    Range range(pr_view, d_);
    return PrefixDoubling<float>(range, k);
  }

  CenterSet copy_centers_by_ids(const parlay::sequence<uint32_t>& ids) const {
    CenterSet out(static_cast<uint32_t>(ids.size()), d_);
    parlay::parallel_for(0, ids.size(), [&](size_t i) {
      const float* src = data_.data() + static_cast<size_t>(ids[i]) * d_;
      float* dst = out[i];
      std::memcpy(dst, src, d_ * sizeof(float));
    });
    return out;
  }

  // Encode `centers` as a single multi-vector DB cloud of size k and run the
  // SingleCloudArgmin kernel: one VPDPBUSD panel processes 16 real centers,
  // so SIMD throughput is fully utilized (vs the 1/16-fill we'd get by making
  // each center its own 1-vec cloud).  The kernel returns argmin lane in
  // [0, k) per query, which we write directly into cluster_ids.
  void assign(const CenterSet& centers, parlay::sequence<uint32_t>& cluster_ids) {
    cluster_ids.resize(n_);
    if (n_ == 0 || centers.size() == 0) return;

    SingleCloudCenterSetView center_view(centers);
    EncSet center_db = model_->encode(center_view);

    ::mvsic::turboquant_8bit_mv::SingleCloudArgmin<metric>::RunFlat(
        enc_flat_data_.data(), enc_flat_nsf_.data(), enc_flat_sqn_.data(),
        enc_flat_bsum_.data(), enc_q_stride_, n_, center_db, cluster_ids.data());
  }

  CenterSet compute_means(parlay::sequence<uint32_t>& cluster_ids, uint32_t k) {
    return compute_means_impl_(cluster_ids, k, /*weights=*/nullptr);
  }

  CenterSet compute_means_weighted(parlay::sequence<uint32_t>& cluster_ids,
                                   const parlay::sequence<float>& weights, uint32_t k) {
    return compute_means_impl_(cluster_ids, k, &weights);
  }

  // Cost in float against the float view -- we want exact distances here so
  // the printed cost is comparable to FloatLloydsBackend's.
  float cost(const CenterSet& centers,
             const parlay::sequence<uint32_t>& cluster_ids) const {
    auto distances = parlay::delayed_seq<float>(n_, [&](size_t i) {
      const float* p = data_.data() + i * d_;
      const float* c = centers[cluster_ids[i]];
      float s = 0.0f;
      if constexpr (metric) {
        for (uint32_t t = 0; t < d_; ++t) {
          const float v = p[t] - c[t];
          s += v * v;
        }
      } else {
        for (uint32_t t = 0; t < d_; ++t) s += p[t] * c[t];
        s = -s;
      }
      return s;
    });
    return parlay::reduce(distances);
  }

 private:
  // PCSet-shaped view of a CenterSet as exactly ONE multi-vector cloud of
  // size k (not k 1-vec clouds).  This packing is what lets SingleCloudArgmin
  // fill every panel with 16 real centers; the previous "k 1-vec clouds"
  // layout left 15/16 of each panel as neutral padding and starved the
  // VPDPBUSD throughput.
  class SingleCloudCenterSetView {
   public:
    explicit SingleCloudCenterSetView(const CenterSet& c)
        : values_(c.data()),
          k_(static_cast<uint32_t>(c.size())),
          d_(c.get_dims()) {
      // get_offsets() advertises a single cloud spanning all k*d floats, so
      // Model::encode walks it as one cloud of n_vecs = k.
      offsets_ = parlay::sequence<size_t>{0, static_cast<size_t>(k_) * static_cast<size_t>(d_)};
    }

    uint32_t get_dims() const noexcept { return d_; }
    auto get_offsets() const noexcept {
      return parlay::make_slice(offsets_.begin(), offsets_.end());
    }
    auto get_ids() const noexcept {
      return parlay::make_slice(empty_ids_.begin(), empty_ids_.end());
    }
    const float* data() const noexcept { return values_; }

   private:
    const float* values_;
    uint32_t k_;
    uint32_t d_;
    parlay::sequence<size_t> offsets_;
    parlay::sequence<uint32_t> empty_ids_;
  };

  // Trivial shim used to feed `Model::train` -- it only consumes get_dims().
  struct TrainShim {
    uint32_t d;
    uint32_t get_dims() const noexcept { return d; }
  };

  // Process-wide cached model per (metric, dim).  The Hadamard rotator's
  // weights only need to be self-consistent within a single TQ8LloydsBackend
  // instance (so the cached per-row queries match the per-iteration center
  // encoding); they don't need to differ across calls.  Sharing replaces tens
  // of thousands of `choose_rotator` calls (each doing a `random_device`
  // syscall + mt19937 seed + `4*padded_dim/8` random bytes) with one.
  static const QModel& GetSharedModel_(uint32_t d) {
    static std::mutex mu;
    static std::map<uint32_t, std::unique_ptr<QModel>> cache;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(d);
    if (it == cache.end()) {
      auto m = std::make_unique<QModel>();
      TrainShim shim{d};
      m->train(shim);
      it = cache.emplace(d, std::move(m)).first;
    }
    return *it->second;
  }

  // Cache path: copy per-row encoded state out of a shared per-vector buffer.
  // The cache must have been produced with `cache.model`'s rotator, and the
  // n entries of `vec_root_indices` must each index into the cache's per-row
  // arrays.  q_stride is taken from the cache (matches the model's padded
  // dim, exactly as build_encoded_ would compute it).
  void gather_encoded_(const TQ8VectorCache<metric>& cache,
                       const parlay::sequence<uint32_t>& vec_root_indices) {
    enc_q_stride_ = cache.q_stride;
    enc_flat_data_.assign(static_cast<size_t>(n_) * enc_q_stride_, 0);
    enc_flat_nsf_.assign(n_, 0.0f);
    enc_flat_sqn_.assign(n_, 0.0f);
    enc_flat_bsum_.assign(n_, 0);
    if (n_ == 0) return;
    parlay::parallel_for(0, n_, [&](size_t i) {
      const size_t src = static_cast<size_t>(vec_root_indices[i]);
      std::memcpy(enc_flat_data_.data() + i * enc_q_stride_,
                  cache.q_data + src * enc_q_stride_,
                  enc_q_stride_ * sizeof(int8_t));
      enc_flat_nsf_[i] = cache.q_nsf[src];
      enc_flat_sqn_[i] = cache.q_sqn[src];
      enc_flat_bsum_[i] = cache.q_bsum[src];
    });
  }

  // Encode every input row inline into flat parallel arrays (one int8 stripe
  // per row + one float/int32 per row), with a per-worker scratch q_rot
  // buffer.  Avoids the n separate Quantized_Query_Point_Cloud structs that
  // `Model::quantize_query` would allocate (each with 4 internal vectors)
  // and the per-row q_rot heap alloc inside that function.
  void build_encoded_() {
    const size_t n = n_;
    const size_t padded_dim = model_->encoder.padded_dim;
    enc_q_stride_ = (padded_dim + 3) & ~3;

    enc_flat_data_.assign(n * enc_q_stride_, 0);
    enc_flat_nsf_.assign(n, 0.0f);
    enc_flat_sqn_.assign(n, 0.0f);
    enc_flat_bsum_.assign(n, 0);

    if (n == 0) return;

    const auto* rotator = model_->encoder.rotator.get();
    const size_t num_workers = parlay::num_workers();
    // Per-worker scratch row for the rotated values.  parlay::parallel_for
    // may schedule any task on any worker, but only one task at a time per
    // worker, so workspaces[parlay::worker_id()] is always exclusively held.
    std::vector<std::vector<float>> workspaces(num_workers,
                                               std::vector<float>(padded_dim));

    parlay::parallel_for(0, n, [&](size_t i) {
      const float* p = data_.data() + i * d_;
      auto& ws = workspaces[parlay::worker_id()];
      rotator->rotate(p, ws.data());

      float sqr_norm = 0.0f;
      float max_value = 0.0f;
      for (size_t t = 0; t < padded_dim; ++t) {
        const float v = ws[t];
        sqr_norm += v * v;
        const float a = std::abs(v);
        if (a > max_value) max_value = a;
      }
      if (sqr_norm == 0.0f || !std::isfinite(sqr_norm) || max_value == 0.0f) {
        // enc_flat_* already zero-initialized; nsf/sqn/bsum stay at 0.
        return;
      }

      const float norm = std::sqrt(sqr_norm);
      const float sf = 127.0f / max_value;
      int64_t quant_norm = 0;
      int32_t byte_sum = 0;
      int8_t* q_out = enc_flat_data_.data() + i * enc_q_stride_;
      for (size_t t = 0; t < padded_dim; ++t) {
        const int snapped = static_cast<int>(std::lround(ws[t] * sf));
        const int8_t iv =
            static_cast<int8_t>(snapped < -127 ? -127 : (snapped > 127 ? 127 : snapped));
        q_out[t] = iv;
        quant_norm += static_cast<int64_t>(iv) * static_cast<int64_t>(iv);
        byte_sum += iv;
      }
      enc_flat_nsf_[i] =
          quant_norm > 0 ? norm / std::sqrt(static_cast<float>(quant_norm)) : 0.0f;
      enc_flat_sqn_[i] = sqr_norm;
      enc_flat_bsum_[i] = byte_sum;
    });
  }

  // Same per-cluster reduction as FloatLloydsBackend::compute_means_impl_.
  // We keep two separate statics for the empty-cluster PRNG so the TQ8
  // backend's sequence is independent of the float backend's (per
  // next_steps.md).
  CenterSet compute_means_impl_(parlay::sequence<uint32_t>& cluster_ids, uint32_t k,
                                const parlay::sequence<float>* weights) {
    const size_t n = n_;

    auto pairs = parlay::sequence<std::pair<uint32_t, uint32_t>>::from_function(
        n, [&](size_t i) { return std::make_pair(cluster_ids[i], static_cast<uint32_t>(i)); });
    auto grouped = parlay::group_by_index(pairs, k);

    CenterSet out(k, d_);
    static uint32_t empty_seed_unweighted = 0;
    static uint32_t empty_seed_weighted = 0;
    uint32_t empty_seed_base =
        (weights == nullptr) ? empty_seed_unweighted : empty_seed_weighted;

    parlay::parallel_for(0, k, [&](size_t cid) {
      const auto& g = grouped[cid];
      float* out_row = out[cid];
      if (g.size() > 0) {
        // Cache-friendly accumulation: one streaming pass through each
        // assigned point's d floats, instead of d separate strided reductions.
        std::memset(out_row, 0, d_ * sizeof(float));
        if (weights != nullptr) {
          float total_w = 0.0f;
          for (size_t j = 0; j < g.size(); ++j) {
            const uint32_t pi = g[j];
            const float w = (*weights)[pi];
            total_w += w;
            const float* p = data_.data() + static_cast<size_t>(pi) * d_;
            for (uint32_t t = 0; t < d_; ++t) {
              out_row[t] += w * p[t];
            }
          }
          if (total_w == 0.0f) total_w = 1.0f;
          const float inv_w = 1.0f / total_w;
          for (uint32_t t = 0; t < d_; ++t) out_row[t] *= inv_w;
        } else {
          for (size_t j = 0; j < g.size(); ++j) {
            const float* p = data_.data() + static_cast<size_t>(g[j]) * d_;
            for (uint32_t t = 0; t < d_; ++t) {
              out_row[t] += p[t];
            }
          }
          if constexpr (metric) {
            const float inv_n = 1.0f / static_cast<float>(g.size());
            for (uint32_t t = 0; t < d_; ++t) out_row[t] *= inv_n;
          }
        }
        if constexpr (!metric) {
          double sum_sqrs = 0.0;
          for (uint32_t t = 0; t < d_; ++t) {
            double v = out_row[t];
            sum_sqrs += v * v;
          }
          if (sum_sqrs != 0.0) {
            float inv = 1.0f / static_cast<float>(std::sqrt(sum_sqrs));
            for (uint32_t t = 0; t < d_; ++t) out_row[t] *= inv;
          }
        }
      } else {
        uint32_t id = parlay::hash32(empty_seed_base + static_cast<uint32_t>(cid)) % n;
        const float* p = data_.data() + static_cast<size_t>(id) * d_;
        std::memcpy(out_row, p, d_ * sizeof(float));
        cluster_ids[id] = static_cast<uint32_t>(cid);
      }
    });
    if (weights == nullptr) {
      empty_seed_unweighted += k;
    } else {
      empty_seed_weighted += k;
    }
    return out;
  }

  // Adapter that lets parlayANN::PointRange<...> consume our flat float buffer
  // when seed_prefix_doubling needs it (L2 only).  Avoids paying the
  // PointRange aligned_alloc + madvise on the unweighted IP path.
  struct RangeView {
    const float* base;
    size_t n;
    uint32_t d;
    size_t size() const { return n; }
    parlay::slice<const float*, const float*> operator[](size_t i) const {
      return parlay::make_slice(base + i * d, base + (i + 1) * d);
    }
  };
  RangeView MakeRangeView_() const { return RangeView{data_.data(), n_, d_}; }

  // Flat row-major float buffer (n * d).  Replaces parlayANN::PointRange<float>
  // (which aligned_alloc'd 2MB and madvise'd huge pages per backend
  // construction — costly when we make tens of thousands of inner-kmeans
  // backends per build).
  parlay::sequence<float> data_;
  size_t n_ = 0;
  uint32_t d_ = 0;
  // Non-owning pointer into the static (metric, dim) -> Model cache.
  const QModel* model_ = nullptr;
  // Flat per-row encoded query state.  Replaces n separate
  // Quantized_Query_Point_Cloud structs.
  size_t enc_q_stride_ = 0;
  parlay::sequence<int8_t> enc_flat_data_;
  parlay::sequence<float> enc_flat_nsf_;
  parlay::sequence<float> enc_flat_sqn_;
  parlay::sequence<int32_t> enc_flat_bsum_;
};

}  // namespace mvsic::lloyds
