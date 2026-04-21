#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "parlay/primitives.h"
#include "mvsic/core/index_params.h"
#include "mvsic/core/search_params.h"
#include "mvsic/core/types/point_cloud_set.h"

namespace mvsic {

// Owns the float buffer for a compressed query point cloud.
// Produces a non-owning ChPoint view via view().
template<typename ChPoint>
struct CompressedPointCloud {
  std::shared_ptr<float[]> data;
  uint32_t n = 0;
  uint32_t dims = 0;
  uint32_t id = 0;

  ChPoint view() const { return ChPoint(n, dims, data.get(), id); }
  bool empty() const { return n == 0; }
};

namespace qc_internal {

using QT = IndexParams::QuantizerType;
using QC = SearchParams::QueryCompression;

// SIMD batch widths used by the quantized Chamfer kernels.
// Keeping the compressed query vector count a multiple of B avoids
// the scalar epilogue in the inner distance loops.
inline uint32_t batch_alignment(QT qt) {
  switch (qt) {
    case QT::TurboQuant: return 6;  // kVnniMq
    case QT::OneBitTQ: return 4;    // kMq1bit
    case QT::SPQTQ: return 3;       // kMvBatch
    case QT::FastScan: return 6;    // SCAN_Q_BATCH
    default: return 1;
  }
}

inline uint32_t align_down(uint32_t n, uint32_t B) {
  if (B <= 1 || n <= B) return n;
  return (n / B) * B;
}

inline std::shared_ptr<float[]> alloc_floats(size_t count) {
  return std::shared_ptr<float[]>(static_cast<float*>(parlay::p_malloc(count * sizeof(float))),
                                  parlay::p_free);
}

}  // namespace qc_internal

// ============================================================================
// Ball Carving  (MUVERA §C.3)
// ============================================================================
// Greedy ball carving with a pairwise similarity threshold.
//   IP  (is_metric=false): merge when <q_i, q_j> >= threshold.  Centroids = SUM.
//   L2  (is_metric=true):  merge when ||q_i-q_j||^2 <= threshold. Centroids = MEAN.
// After clustering, output count is aligned down to a multiple of batch_align
// by merging the closest remaining centroids.
template<typename ChPoint>
CompressedPointCloud<ChPoint> ball_carving(const ChPoint& query, float threshold,
                                           uint32_t batch_align = 1) {
  const uint32_t n = query.size();
  const uint32_t d = query.get_dims();

  if (n == 0) return {};
  if (n <= batch_align) {
    auto buf = qc_internal::alloc_floats(size_t(n) * d);
    std::memcpy(buf.get(), query.data(), size_t(n) * d * sizeof(float));
    return {std::move(buf), n, d, query.get_id()};
  }

  // --- 1. Pairwise similarity via Eigen GEMM ---
  using RowMap =
      Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  RowMap Q(query.data(), n, d);

  static constexpr uint32_t kMaxStackN = 64;
  alignas(64) float sim_stack[kMaxStackN * kMaxStackN];
  std::vector<float> sim_heap_storage;
  float* sim_data;

  if (n <= kMaxStackN) {
    sim_data = sim_stack;
  } else {
    sim_heap_storage.resize(size_t(n) * n);
    sim_data = sim_heap_storage.data();
  }

  using RowMatMap =
      Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  RowMatMap sim(sim_data, n, n);

  float effective_threshold;
  if constexpr (ChPoint::is_metric()) {
    // L2: sim(i,j) = -||q_i - q_j||^2.  Merge when sim >= -threshold.
    Eigen::VectorXf norms = Q.rowwise().squaredNorm();
    sim.noalias() = 2.0f * (Q * Q.transpose());
    sim.colwise() -= norms;
    sim.rowwise() -= norms.transpose();
    effective_threshold = -threshold;
  } else {
    // IP: sim(i,j) = <q_i, q_j>.  Merge when sim >= threshold.
    sim.noalias() = Q * Q.transpose();
    effective_threshold = threshold;
  }

  // --- 2. Greedy ball carving ---
  // Bitmask path for n <= 64 (covers ColBERT-style queries).
  std::vector<std::vector<uint32_t>> clusters;
  clusters.reserve(n);

  if (n <= 64) {
    uint64_t unclustered = (n == 64) ? ~0ULL : (1ULL << n) - 1;
    while (unclustered) {
      uint32_t center = __builtin_ctzll(unclustered);
      unclustered &= ~(1ULL << center);

      std::vector<uint32_t> cluster = {center};
      uint64_t remaining = unclustered;
      while (remaining) {
        uint32_t j = __builtin_ctzll(remaining);
        remaining &= remaining - 1;  // clear lowest set bit
        if (sim_data[size_t(center) * n + j] >= effective_threshold) {
          cluster.push_back(j);
          unclustered &= ~(1ULL << j);
        }
      }
      clusters.push_back(std::move(cluster));
    }
  } else {
    std::vector<bool> assigned(n, false);
    for (uint32_t i = 0; i < n; ++i) {
      if (assigned[i]) continue;
      assigned[i] = true;
      std::vector<uint32_t> cluster = {i};
      for (uint32_t j = i + 1; j < n; ++j) {
        if (!assigned[j] && sim_data[size_t(i) * n + j] >= effective_threshold) {
          cluster.push_back(j);
          assigned[j] = true;
        }
      }
      clusters.push_back(std::move(cluster));
    }
  }

  // --- 3. Compute centroids ---
  uint32_t k = static_cast<uint32_t>(clusters.size());
  auto buf = qc_internal::alloc_floats(size_t(k) * d);

  for (uint32_t c = 0; c < k; ++c) {
    float* centroid = buf.get() + size_t(c) * d;
    std::memset(centroid, 0, d * sizeof(float));
    for (uint32_t idx : clusters[c]) {
      const float* vec = query.data(idx);
      for (uint32_t t = 0; t < d; ++t)
        centroid[t] += vec[t];
    }
    if constexpr (ChPoint::is_metric()) {
      float inv = 1.0f / static_cast<float>(clusters[c].size());
      for (uint32_t t = 0; t < d; ++t)
        centroid[t] *= inv;
    }
  }

  // --- 4. Align cluster count to batch size ---
  uint32_t target = qc_internal::align_down(k, batch_align);
  while (k > target && k > 1) {
    // Find closest pair of centroids (L2).
    float best_dist = std::numeric_limits<float>::max();
    uint32_t best_i = 0, best_j = 1;
    for (uint32_t i = 0; i < k; ++i) {
      const float* ci = buf.get() + size_t(i) * d;
      for (uint32_t j = i + 1; j < k; ++j) {
        const float* cj = buf.get() + size_t(j) * d;
        float dist = 0;
        for (uint32_t t = 0; t < d; ++t) {
          float diff = ci[t] - cj[t];
          dist += diff * diff;
        }
        if (dist < best_dist) {
          best_dist = dist;
          best_i = i;
          best_j = j;
        }
      }
    }

    // Merge best_j into best_i.
    float* ci = buf.get() + size_t(best_i) * d;
    const float* cj = buf.get() + size_t(best_j) * d;
    if constexpr (ChPoint::is_metric()) {
      size_t ni = clusters[best_i].size();
      size_t nj = clusters[best_j].size();
      float wi = static_cast<float>(ni) / static_cast<float>(ni + nj);
      float wj = static_cast<float>(nj) / static_cast<float>(ni + nj);
      for (uint32_t t = 0; t < d; ++t)
        ci[t] = wi * ci[t] + wj * cj[t];
    } else {
      for (uint32_t t = 0; t < d; ++t)
        ci[t] += cj[t];
    }
    for (uint32_t idx : clusters[best_j])
      clusters[best_i].push_back(idx);

    // Swap-erase cluster best_j.
    if (best_j < k - 1) {
      std::memcpy(buf.get() + size_t(best_j) * d, buf.get() + size_t(k - 1) * d, d * sizeof(float));
      clusters[best_j] = std::move(clusters[k - 1]);
    }
    clusters.pop_back();
    --k;
  }

  return {std::move(buf), k, d, query.get_id()};
}

// ============================================================================
// Ward's Method
// ============================================================================
// Agglomerative clustering using Ward linkage.
//   Ward distance: d_W(i,j) = n_i*n_j/(n_i+n_j) * ||c_i - c_j||^2
//   Merge the closest pair while d_W <= threshold, then align to batch_align.
//   Centroids are always the weighted mean.
// Uses the Lance-Williams recurrence for O(n^2) total update cost.
template<typename ChPoint>
CompressedPointCloud<ChPoint> wards_compress(const ChPoint& query, float threshold,
                                             uint32_t batch_align = 1) {
  const uint32_t n = query.size();
  const uint32_t d = query.get_dims();

  if (n == 0) return {};
  if (n <= batch_align) {
    auto buf = qc_internal::alloc_floats(size_t(n) * d);
    std::memcpy(buf.get(), query.data(), size_t(n) * d * sizeof(float));
    return {std::move(buf), n, d, query.get_id()};
  }

  // Centroid storage (mutable copy of original vectors).
  std::vector<float> centroids(size_t(n) * d);
  std::memcpy(centroids.data(), query.data(), size_t(n) * d * sizeof(float));

  std::vector<uint32_t> sizes(n, 1);
  std::vector<bool> active(n, true);
  uint32_t num_active = n;

  // --- Initial pairwise Ward distance matrix ---
  using RowMap =
      Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  RowMap Q(query.data(), n, d);

  static constexpr uint32_t kMaxStackN = 64;
  alignas(64) float ward_stack[kMaxStackN * kMaxStackN];
  std::vector<float> ward_heap_storage;
  float* ward_data;

  if (n <= kMaxStackN) {
    ward_data = ward_stack;
  } else {
    ward_heap_storage.resize(size_t(n) * n);
    ward_data = ward_heap_storage.data();
  }

  {
    using RowMatMap =
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
    RowMatMap ward(ward_data, n, n);

    Eigen::VectorXf norms = Q.rowwise().squaredNorm();
    ward.noalias() = -2.0f * (Q * Q.transpose());
    ward.colwise() += norms;
    ward.rowwise() += norms.transpose();
    ward = ward.cwiseMax(0.0f);
    // For singletons: ward(i,j) = 0.5 * ||c_i - c_j||^2
    ward *= 0.5f;
  }

  auto get_ward = [&](uint32_t i, uint32_t j) -> float& { return ward_data[size_t(i) * n + j]; };

  // Find minimum active Ward distance pair.
  auto find_min = [&]() -> std::tuple<float, uint32_t, uint32_t> {
    float best = std::numeric_limits<float>::max();
    uint32_t bi = 0, bj = 0;
    for (uint32_t i = 0; i < n; ++i) {
      if (!active[i]) continue;
      for (uint32_t j = i + 1; j < n; ++j) {
        if (!active[j]) continue;
        if (get_ward(i, j) < best) {
          best = get_ward(i, j);
          bi = i;
          bj = j;
        }
      }
    }
    return {best, bi, bj};
  };

  // Merge cluster bj into bi, updating Ward matrix via Lance-Williams.
  auto do_merge = [&](uint32_t bi, uint32_t bj) {
    uint32_t ni = sizes[bi];
    uint32_t nj = sizes[bj];
    uint32_t nij = ni + nj;
    float wi = static_cast<float>(ni) / static_cast<float>(nij);
    float wj = static_cast<float>(nj) / static_cast<float>(nij);

    float* ci = centroids.data() + size_t(bi) * d;
    const float* cj = centroids.data() + size_t(bj) * d;
    for (uint32_t t = 0; t < d; ++t)
      ci[t] = wi * ci[t] + wj * cj[t];

    sizes[bi] = nij;
    active[bj] = false;
    --num_active;

    float d_ij = get_ward(bi, bj);
    for (uint32_t kk = 0; kk < n; ++kk) {
      if (!active[kk] || kk == bi) continue;
      uint32_t nk = sizes[kk];
      float d_ik = get_ward(std::min(bi, kk), std::max(bi, kk));
      float d_jk = get_ward(std::min(bj, kk), std::max(bj, kk));
      float new_d = (static_cast<float>(ni + nk) * d_ik + static_cast<float>(nj + nk) * d_jk -
                     static_cast<float>(nk) * d_ij) /
                    static_cast<float>(nij + nk);
      get_ward(bi, kk) = new_d;
      get_ward(kk, bi) = new_d;
    }
  };

  // Phase 1: threshold-based merging.
  while (num_active > 1) {
    auto [best, bi, bj] = find_min();
    if (best > threshold) break;
    do_merge(bi, bj);
  }

  // Phase 2: align to batch size.
  if (batch_align > 1) {
    uint32_t aligned_target = qc_internal::align_down(num_active, batch_align);
    while (num_active > aligned_target && num_active > 1) {
      auto [best, bi, bj] = find_min();
      do_merge(bi, bj);
    }
  }

  // --- Collect active centroids ---
  uint32_t k = num_active;
  auto buf = qc_internal::alloc_floats(size_t(k) * d);
  uint32_t out = 0;
  for (uint32_t i = 0; i < n; ++i) {
    if (!active[i]) continue;
    std::memcpy(buf.get() + size_t(out) * d, centroids.data() + size_t(i) * d, d * sizeof(float));
    ++out;
  }

  return {std::move(buf), k, d, query.get_id()};
}

// ============================================================================
// Dispatcher
// ============================================================================
template<typename ChPoint>
CompressedPointCloud<ChPoint> compress_query(const ChPoint& query,
                                             SearchParams::QueryCompression method, float threshold,
                                             uint32_t batch_align = 1) {
  switch (method) {
    case SearchParams::QueryCompression::Carve:
      return ball_carving(query, threshold, batch_align);
    case SearchParams::QueryCompression::Wards:
      return wards_compress(query, threshold, batch_align);
    default: return {};
  }
}

// ============================================================================
// Batch: compress every query in a PointCloudSet
// ============================================================================
template<typename ChPoint>
PointCloudSet<ChPoint> compress_point_cloud_set(const PointCloudSet<ChPoint>& queries,
                                                SearchParams::QueryCompression method,
                                                float threshold, uint32_t batch_align = 1) {
  const size_t nq = queries.size();
  const uint32_t d = queries.get_dims();

  auto compressed = parlay::sequence<CompressedPointCloud<ChPoint>>::from_function(
      nq, [](size_t) { return CompressedPointCloud<ChPoint>{}; });

  parlay::parallel_for(0, nq, [&](size_t i) {
    compressed[i] = compress_query(queries[i], method, threshold, batch_align);
  });

  // Pack into a contiguous buffer and build PointCloudSet.
  auto offsets = parlay::sequence<size_t>::from_function(
      nq + 1, [&](size_t i) { return (i == 0) ? size_t(0) : size_t(compressed[i - 1].n) * d; });
  parlay::scan_inclusive_inplace(offsets);
  size_t total_floats = offsets[nq];

  auto values = qc_internal::alloc_floats(total_floats);
  auto ids = parlay::tabulate<uint32_t>(nq, [&](size_t i) { return compressed[i].id; });

  parlay::parallel_for(0, nq, [&](size_t i) {
    if (compressed[i].n > 0) {
      std::memcpy(values.get() + offsets[i], compressed[i].data.get(),
                  size_t(compressed[i].n) * d * sizeof(float));
    }
  });

  return PointCloudSet<ChPoint>(static_cast<uint32_t>(nq), d, values.get(), offsets.data(),
                                ids.data());
}

}  // namespace mvsic
