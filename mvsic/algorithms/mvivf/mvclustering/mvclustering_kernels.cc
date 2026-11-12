#include "mvsic/algorithms/mvivf/mvclustering/mvclustering_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include "mvsic/algorithms/mvivf/mvclustering/mvclustering_impl.h"
#include "mvsic/core/quantization/turboquant_mv.h"
#include "mvsic/core/quantization/turboquant_8bit_mv.h"
#include "mvsic/core/quantization/turboquant_1bit.h"
#include "mvsic/core/quantization/turboquant_1bit_mv.h"
#include "mvsic/core/kernels/panels_avx2.h"

namespace mvsic {
namespace {
template<bool metric>
using CloudSet = PointCloudSet<std::conditional_t<metric, ChamferL2_Point, ChamferIP_Point>>;

template<bool metric>
CloudSet<metric> Borrow(const PointCloudSetView& x) {
  return CloudSet<metric>::borrow(x.n, x.d, x.values, x.offsets, x.ids);
}

// Rotation depends only on dimensions and seed, not the training values.
// Share immutable models across the many inner/recursive clustering calls.
template<class Model>
const Model& SharedModel(uint32_t d, uint32_t seed) {
  static std::mutex mutex;
  static std::map<std::pair<uint32_t, uint32_t>, std::unique_ptr<Model>> models;
  std::lock_guard<std::mutex> lock(mutex);
  const auto key = std::make_pair(d, seed);
  auto it = models.find(key);
  if (it == models.end()) {
    auto model = std::make_unique<Model>();
    model->encoder.seed_ = seed;
    model->encoder.train(d);
    it = models.emplace(key, std::move(model)).first;
  }
  return *it->second;
}

// Flat per-vector input encoding has identical bytes/norms for L2 and IP.
// Keeping this view independent of the outer metric avoids duplicating every
// inner implementation when outer and inner objectives differ.
struct PreparedVectors {
  explicit PreparedVectors(const PointCloudSetView& input) : root(input) {}
  PointCloudSetView root;
  size_t vector_stride = 0;
  std::vector<int8_t> vector_data;
  std::vector<float> vector_norms, vector_squared_norms;
  std::vector<int32_t> vector_byte_sums;
};

template<bool metric>
struct PreparedAssignment : PreparedVectors {
  std::vector<typename turboquant_8bit_mv::Model<metric>::EncodedQuery> tq8;
  std::vector<typename turboquant_mv::Model<metric>::EncodedQuery> tq4;
  std::vector<turboquant_1bit::Quantized_Query<metric>> tq1;

  PreparedAssignment(const PointCloudSetView& input, ClusteringQuantizer codec, uint32_t seed) :
      PreparedVectors(input) {
    auto points = Borrow<metric>(root);
    if (codec == ClusteringQuantizer::TQ8) {
      const auto& model = SharedModel<turboquant_8bit_mv::Model<metric>>(root.d, seed);
      tq8.resize(root.n);
      parlay::parallel_for(0, root.n, [&](size_t i) { tq8[i] = model.quantize_query(points[i]); });
      const size_t stride = (model.encoder.padded_dim + 3) & ~size_t{3};
      const size_t vectors = root.total_size();
      if (vectors > std::numeric_limits<size_t>::max() / stride)
        throw std::length_error("Clustering encoding size overflows");
      vector_data.resize(vectors * stride);
      vector_norms.resize(vectors);
      vector_squared_norms.resize(vectors);
      vector_byte_sums.resize(vectors);
      parlay::parallel_for(0, root.n, [&](size_t i) {
        const size_t begin = root.offsets[i] / root.d;
        const auto& q = tq8[i];
        std::copy(q.flat_query_data.begin(), q.flat_query_data.end(),
                  vector_data.begin() + begin * stride);
        std::copy(q.norm_scaling_factors.begin(), q.norm_scaling_factors.end(),
                  vector_norms.begin() + begin);
        std::copy(q.unquantized_squared_norms.begin(), q.unquantized_squared_norms.end(),
                  vector_squared_norms.begin() + begin);
        std::copy(q.byte_sums.begin(), q.byte_sums.end(), vector_byte_sums.begin() + begin);
      });
      vector_stride = stride;
    } else if (codec == ClusteringQuantizer::TQ4) {
      const auto& model = SharedModel<turboquant_mv::Model<metric>>(root.d, seed);
      tq4.resize(root.n);
      parlay::parallel_for(0, root.n, [&](size_t i) { tq4[i] = model.quantize_query(points[i]); });
    } else if (codec == ClusteringQuantizer::TQ1) {
      const auto& model = SharedModel<turboquant_1bit::Model<metric>>(root.d, seed);
      tq1.resize(root.total_size());
      parlay::parallel_for(0, tq1.size(), [&](size_t i) {
        tq1[i] = model.quantize_query(root.values + i * root.d);
      });
    }
  }
};

template<bool metric>
class Assignment {
 public:
  virtual ~Assignment() = default;
  virtual void run(const CloudSet<metric>& points, const CloudSet<metric>& centers,
                   parlay::sequence<uint32_t>& out) = 0;
};

template<bool metric, class Model, template<class> class Batch>
class PanelAssignment final : public Assignment<metric> {
 public:
  explicit PanelAssignment(uint32_t seed,
                           const std::vector<typename Model::EncodedQuery>* prepared) :
      seed_(seed), prepared_(prepared) {}
  void run(const CloudSet<metric>& points, const CloudSet<metric>& centers,
           parlay::sequence<uint32_t>& out) override {
    if (!model_) model_ = &SharedModel<Model>(points.get_dims(), seed_);
    // Cache immutable query encodings over all Lloyd iterations. A distinct
    // full-corpus assignment after subsampled training is encoded once too.
    if (source_ != points.data() || source_rows_ != points.size() ||
        source_offsets_ != points.get_offsets().begin()) {
      pointers_.resize(points.size());
      if (prepared_) {
        for (size_t i = 0; i < points.size(); ++i)
          pointers_[i] = &(*prepared_)[points.get_id(i)];
      } else {
        query_.resize(points.size());
        parlay::parallel_for(0, points.size(),
                             [&](size_t i) { query_[i] = model_->quantize_query(points[i]); });
        for (size_t i = 0; i < points.size(); ++i)
          pointers_[i] = &query_[i];
      }
      source_ = points.data();
      source_rows_ = points.size();
      source_offsets_ = points.get_offsets().begin();
    }
    // Seeded centers inherit source IDs and may duplicate them. Stripping IDs
    // makes the batch kernel report center POSITIONS deterministically.
    const auto offsets = centers.get_offsets();
    auto positional = CloudSet<metric>::borrow(centers.size(), centers.get_dims(), centers.data(),
                                               offsets.begin());
    auto db = model_->encode(positional);
    parlay::sequence<std::pair<uint32_t, float>> top(points.size());
    Batch<typename Model::EncodedSet>::TopKIntoUninitialized(pointers_, db, 1, top.data());
    out.resize(points.size());
    parlay::parallel_for(0, points.size(), [&](size_t i) { out[i] = top[i].first; });
  }

 private:
  const Model* model_ = nullptr;
  uint32_t seed_;
  const float* source_ = nullptr;
  const size_t* source_offsets_ = nullptr;
  size_t source_rows_ = 0;
  const std::vector<typename Model::EncodedQuery>* prepared_;
  std::vector<typename Model::EncodedQuery> query_;
  std::vector<const typename Model::EncodedQuery*> pointers_;
};

#if defined(__AVX2__) || defined(__AVX512F__)
// General-norm TQ1 centers use the serving Hamming panel layout, with norms
// retained per vector. The float score must be computed BEFORE taking minima.
struct OneBitPanels {
  size_t tiles, bytes, dimension;
  std::vector<size_t> first, count;
  std::vector<uint32_t> words;
  std::vector<float> norms, squared;

  template<class Database, class Clouds>
  OneBitPanels(const Database& db, const Clouds& centers, bool singleton) :
      tiles((db.num_bytes_per_datapoint + 3) / 4),
      bytes(db.num_bytes_per_datapoint),
      dimension(db.padded_dim) {
    const size_t n = singleton ? 1 : centers.size();
    first.resize(n + 1);
    count.resize(n);
    for (size_t c = 0; c < n; ++c) {
      count[c] = singleton ? centers.size() : centers.get_size(c);
      first[c + 1] = first[c] + (count[c] + 15) / 16;
    }
    // Padding is read by SIMD before masking, so its bytes/norms must be zero.
    words.resize(first.back() * tiles * 16);
    norms.resize(first.back() * 16);
    squared.resize(norms.size());
    const auto offsets = centers.get_offsets();
    parlay::parallel_for(0, n, [&](size_t c) {
      const size_t begin = singleton ? 0 : offsets[c] / centers.get_dims();
      for (size_t v = 0; v < count[c]; ++v) {
        const size_t from = begin + v, panel = first[c] + v / 16, lane = v % 16;
        norms[panel * 16 + lane] = db.norm_scaling_factors[from];
        squared[panel * 16 + lane] = db.unquantized_squared_norms[from];
        for (size_t tile = 0; tile < tiles; ++tile)
          std::memcpy(&words[(panel * tiles + tile) * 16 + lane],
                      db.packed_codes.data() + from * bytes + tile * 4,
                      std::min<size_t>(4, bytes - tile * 4));
      }
    });
  }
};

struct OneBitKernel {
#if defined(__AVX512F__)
  static constexpr size_t Width = 16;
  using Float = __m512;
  using Integer = __m512i;
  static Float infinity() {
    return _mm512_set1_ps(std::numeric_limits<float>::infinity());
  }
  static Float minimum(Float a, Float b) {
    return _mm512_min_ps(a, b);
  }
  static void store(float* out, Float value) {
    _mm512_storeu_ps(out, value);
  }
  static void store(uint32_t* out, Integer value) {
    _mm512_storeu_si512(out, value);
  }
  template<bool Metric>
  static Float scores(const turboquant_1bit::Quantized_Query<Metric>& q, const OneBitPanels& db,
                      size_t panel, size_t half, size_t valid) {
    Integer counts = _mm512_setzero_si512();
    for (size_t tile = 0; tile < db.tiles; ++tile) {
      uint32_t word = 0;
      std::memcpy(&word, q.query_data.data() + tile * 4, std::min<size_t>(4, db.bytes - tile * 4));
      const auto data = _mm512_loadu_si512(db.words.data() + (panel * db.tiles + tile) * 16 + half);
      counts = _mm512_add_epi32(counts, turboquant_1bit_mv::internal::popcnt_d32(
                                            _mm512_xor_si512(data, _mm512_set1_epi32(word))));
    }
    auto dot =
        _mm512_mul_ps(_mm512_set1_ps(turboquant_1bit::internal::kCentroidSqr),
                      _mm512_sub_ps(_mm512_set1_ps(float(db.dimension)),
                                    _mm512_mul_ps(_mm512_set1_ps(2), _mm512_cvtepi32_ps(counts))));
    auto distance =
        _mm512_mul_ps(_mm512_mul_ps(_mm512_xor_ps(dot, _mm512_set1_ps(-0.0f)),
                                    _mm512_loadu_ps(db.norms.data() + panel * 16 + half)),
                      _mm512_set1_ps(q.norm_scaling_factor));
    if constexpr (Metric)
      distance = _mm512_add_ps(_mm512_add_ps(_mm512_loadu_ps(db.squared.data() + panel * 16 + half),
                                             _mm512_mul_ps(_mm512_set1_ps(2), distance)),
                               _mm512_set1_ps(q.unquantized_squared_norm));
    return _mm512_mask_mov_ps(infinity(), __mmask16((uint32_t(1) << valid) - 1), distance);
  }
  static Integer indices(size_t base) {
    return _mm512_add_epi32(
        _mm512_set1_epi32(int(base)),
        _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
  }
  static void improve(Float candidate, Integer candidate_id, Float& best, Integer& id) {
    const auto mask = _mm512_cmp_ps_mask(candidate, best, _CMP_LT_OQ);
    best = _mm512_mask_mov_ps(best, mask, candidate);
    id = _mm512_mask_mov_epi32(id, mask, candidate_id);
  }
#elif defined(__AVX2__)
  static constexpr size_t Width = 8;
  using Float = __m256;
  using Integer = __m256i;
  static Float infinity() {
    return _mm256_set1_ps(std::numeric_limits<float>::infinity());
  }
  static Float minimum(Float a, Float b) {
    return _mm256_min_ps(a, b);
  }
  static void store(float* out, Float value) {
    _mm256_storeu_ps(out, value);
  }
  static void store(uint32_t* out, Integer value) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), value);
  }
  template<bool Metric>
  static Float scores(const turboquant_1bit::Quantized_Query<Metric>& q, const OneBitPanels& db,
                      size_t panel, size_t half, size_t valid) {
    Integer counts = _mm256_setzero_si256();
    for (size_t tile = 0; tile < db.tiles; ++tile) {
      uint32_t word = 0;
      std::memcpy(&word, q.query_data.data() + tile * 4, std::min<size_t>(4, db.bytes - tile * 4));
      const auto data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
          db.words.data() + (panel * db.tiles + tile) * 16 + half));
      counts = _mm256_add_epi32(
          counts, cpu::popcount32_avx2(_mm256_xor_si256(data, _mm256_set1_epi32(word))));
    }
    auto dot =
        _mm256_mul_ps(_mm256_set1_ps(turboquant_1bit::internal::kCentroidSqr),
                      _mm256_sub_ps(_mm256_set1_ps(float(db.dimension)),
                                    _mm256_mul_ps(_mm256_set1_ps(2), _mm256_cvtepi32_ps(counts))));
    auto distance =
        _mm256_mul_ps(_mm256_mul_ps(_mm256_xor_ps(dot, _mm256_set1_ps(-0.0f)),
                                    _mm256_loadu_ps(db.norms.data() + panel * 16 + half)),
                      _mm256_set1_ps(q.norm_scaling_factor));
    if constexpr (Metric)
      distance = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(db.squared.data() + panel * 16 + half),
                                             _mm256_mul_ps(_mm256_set1_ps(2), distance)),
                               _mm256_set1_ps(q.unquantized_squared_norm));
    const auto mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(int(valid)),
                                         _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    return _mm256_blendv_ps(infinity(), distance, _mm256_castsi256_ps(mask));
  }
  static Integer indices(size_t base) {
    return _mm256_add_epi32(_mm256_set1_epi32(int(base)),
                            _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
  }
  static void improve(Float candidate, Integer candidate_id, Float& best, Integer& id) {
    const auto mask = _mm256_cmp_ps(candidate, best, _CMP_LT_OQ);
    best = _mm256_blendv_ps(best, candidate, mask);
    id = _mm256_blendv_epi8(id, candidate_id, _mm256_castps_si256(mask));
  }
#endif
};
#endif

// Assignment keeps the single-vector codec's general-norm semantics. Reuse
// across-center SIMD popcount, but never the unit-norm minimum-Hamming shortcut.
// Float means, weighting and normalization remain in the existing Lloyd backend.
template<bool metric>
class OneBitAssignment final : public Assignment<metric> {
  using Model = turboquant_1bit::Model<metric>;
  struct VectorView {
    const float* values;
    size_t n;
    uint32_t d;
    size_t size() const { return n; }
    uint32_t get_dims() const { return d; }
    const float* location(size_t i) const { return values + i * d; }
  };

 public:
  explicit OneBitAssignment(uint32_t seed, const PreparedAssignment<metric>* prepared,
                            bool singleton) :
      seed_(seed), prepared_(prepared), singleton_(singleton) {}
  void run(const CloudSet<metric>& points, const CloudSet<metric>& centers,
           parlay::sequence<uint32_t>& out) override {
    if (!model_) model_ = &SharedModel<Model>(points.get_dims(), seed_);
    if (source_ != points.data() || source_rows_ != points.size() ||
        source_offsets_ != points.get_offsets().begin()) {
      pointers_.resize(points.total_size());
      if (prepared_) {
        const size_t d = points.get_dims();
        const auto offsets = points.get_offsets();
        parlay::parallel_for(0, points.size(), [&](size_t i) {
          const size_t from = prepared_->root.offsets[points.get_id(i)] / d;
          const size_t to = offsets[i] / d;
          for (size_t j = 0; j < points.get_size(i); ++j)
            pointers_[to + j] = &prepared_->tq1[from + j];
        });
      } else {
        query_.resize(points.total_size());
        parlay::parallel_for(0, query_.size(), [&](size_t i) {
          query_[i] = model_->quantize_query(points.data() + i * points.get_dims());
          pointers_[i] = &query_[i];
        });
      }
      source_ = points.data();
      source_rows_ = points.size();
      source_offsets_ = points.get_offsets().begin();
    }
    VectorView vectors{centers.data(), centers.total_size(), centers.get_dims()};
    const auto db = model_->encode(vectors);
    out.resize(points.size());
#if defined(__AVX2__) || defined(__AVX512F__)
    const OneBitPanels panels(db, centers, singleton_);
    if (singleton_) {
      // Inner Lloyd points/centers are single vectors. Select this layout once
      // when constructing the assignment, not by checking shapes in hot loops.
      parlay::parallel_for(0, out.size(), [&](size_t row) {
        auto best = OneBitKernel::infinity();
        auto ids = OneBitKernel::indices(0);
        for (size_t v = 0; v < panels.count[0]; v += OneBitKernel::Width) {
          const auto distance =
              OneBitKernel::scores(*pointers_[row], panels, v / 16, v % 16,
                                   std::min(OneBitKernel::Width, panels.count[0] - v));
          OneBitKernel::improve(distance, OneBitKernel::indices(v), best, ids);
        }
        float values[OneBitKernel::Width];
        uint32_t positions[OneBitKernel::Width];
        OneBitKernel::store(values, best);
        OneBitKernel::store(positions, ids);
        float score = std::numeric_limits<float>::infinity();
        uint32_t winner = 0;
        for (size_t lane = 0; lane < OneBitKernel::Width; ++lane)
          if (values[lane] < score || (values[lane] == score && positions[lane] < winner)) {
            score = values[lane];
            winner = positions[lane];
          }
        out[row] = winner;
      });
    } else {
      const auto offsets = points.get_offsets();
      const size_t d = points.get_dims();
      parlay::parallel_for(0, out.size(), [&](size_t row) {
        float score = std::numeric_limits<float>::infinity();
        uint32_t winner = 0;
        const size_t begin = offsets[row] / d, end = offsets[row + 1] / d;
        for (size_t c = 0; c < panels.count.size(); ++c) {
          float sum = 0;
          for (size_t q = begin; q < end; ++q) {
            auto best = OneBitKernel::infinity();
            for (size_t v = 0; v < panels.count[c]; v += OneBitKernel::Width)
              best = OneBitKernel::minimum(
                  best,
                  OneBitKernel::scores(*pointers_[q], panels, panels.first[c] + v / 16, v % 16,
                                       std::min(OneBitKernel::Width, panels.count[c] - v)));
            float values[OneBitKernel::Width];
            OneBitKernel::store(values, best);
            sum += *std::min_element(values, values + OneBitKernel::Width);
          }
          if (sum < score) {
            score = sum;
            winner = static_cast<uint32_t>(c);
          }
        }
        out[row] = winner;
      });
    }
#else
    const auto po = points.get_offsets();
    const auto co = centers.get_offsets();
    const size_t d = points.get_dims();
    parlay::parallel_for(0, points.size(), [&](size_t i) {
      float best = std::numeric_limits<float>::infinity();
      uint32_t winner = 0;
      for (size_t c = 0; c < centers.size(); ++c) {
        float score = 0;
        for (size_t q = po[i] / d; q < po[i + 1] / d; ++q) {
          float nearest = std::numeric_limits<float>::infinity();
          for (size_t v = co[c] / d; v < co[c + 1] / d; ++v)
            nearest = std::min(nearest, pointers_[q]->distance(db[v]));
          score += nearest;
        }
        // All public inputs are nonempty. Compare sums rather than averages:
        // the positive divisor is constant across centers for this input row.
        if (score < best) {
          best = score;
          winner = static_cast<uint32_t>(c);
        }
      }
      out[i] = winner;
    });
#endif
  }

 private:
  const Model* model_ = nullptr;
  uint32_t seed_;
  const float* source_ = nullptr;
  const size_t* source_offsets_ = nullptr;
  size_t source_rows_ = 0;
  const PreparedAssignment<metric>* prepared_;
  bool singleton_;
  std::vector<const turboquant_1bit::Quantized_Query<metric>*> pointers_;
  std::vector<turboquant_1bit::Quantized_Query<metric>> query_;
};

template<bool metric>
std::unique_ptr<Assignment<metric>> MakeAssignment(
    ClusteringQuantizer quantizer, uint32_t seed,
    const PreparedAssignment<metric>* prepared = nullptr, bool singleton = false) {
  switch (quantizer) {
    case ClusteringQuantizer::None: return {};
    case ClusteringQuantizer::TQ8:
      return std::make_unique<PanelAssignment<metric, turboquant_8bit_mv::Model<metric>,
                                              turboquant_8bit_mv::ManyToMany>>(
          seed, prepared ? &prepared->tq8 : nullptr);
    case ClusteringQuantizer::TQ4:
      return std::make_unique<
          PanelAssignment<metric, turboquant_mv::Model<metric>, turboquant_mv::ManyToMany>>(
          seed, prepared ? &prepared->tq4 : nullptr);
    case ClusteringQuantizer::TQ1:
      return std::make_unique<OneBitAssignment<metric>>(seed, prepared, singleton);
  }
  throw std::invalid_argument("Unknown clustering quantizer");
}

// Reuse the existing float seeding/mean/weighted-mean backend, overriding
// only its assignment operation. This keeps the update arithmetic float.
template<bool metric>
class QuantizedInnerBackend : public lloyds::FloatLloydsBackend<metric> {
  using Base = lloyds::FloatLloydsBackend<metric>;
  using TQ4Model = turboquant_mv::Model<metric>;

 public:
  QuantizedInnerBackend(const parlay::sequence<parlay::sequence<float>>& data, uint32_t d,
                        ClusteringQuantizer codec, uint32_t seed, bool adaptive) :
      Base(data, d, adaptive),
      d_(d),
      values_(data.size() * d),
      offsets_(data.size() + 1),
      assignment_(codec == ClusteringQuantizer::TQ4
                      ? nullptr
                      : MakeAssignment<metric>(codec, seed, nullptr, true)) {
    parlay::parallel_for(0, data.size(), [&](size_t i) {
      std::copy(data[i].begin(), data[i].end(), values_.begin() + i * d_);
      offsets_[i] = i * d_;
    });
    offsets_.back() = values_.size();
    points_ = CloudSet<metric>::borrow(data.size(), d_, values_.data(), offsets_.data());
    if (codec == ClusteringQuantizer::TQ4) {
      tq4_model_ = &SharedModel<TQ4Model>(d, seed);
      const size_t offsets[] = {0, values_.size()};
      const auto packed = CloudSet<metric>::borrow(1, d, values_.data(), offsets);
      tq4_queries_ = tq4_model_->quantize_query(packed[0]);
    }
  }
  void assign(const lloyds::CenterSet& centers, parlay::sequence<uint32_t>& out) {
    if (tq4_model_) {
      // Pack centers into full panels, as in TQ8 inner assignment. Treating
      // every center as its own cloud leaves 15 of 16 panel lanes unused.
      const size_t offsets[] = {0, centers.size() * d_};
      const auto packed = CloudSet<metric>::borrow(1, d_, centers.data(), offsets);
      const auto db = tq4_model_->encode(packed);
      out.resize(points_.size());
      turboquant_8bit_mv::SingleCloudArgmin<metric>::RunFlat(
          tq4_queries_.flat_query_data.data(), tq4_queries_.norm_scaling_factors.data(),
          tq4_queries_.unquantized_squared_norms.data(), tq4_queries_.byte_sums.data(),
          tq4_queries_.q_stride, points_.size(), db, out.data());
      return;
    }
    std::vector<size_t> offsets(centers.size() + 1);
    for (size_t i = 0; i <= centers.size(); ++i)
      offsets[i] = i * d_;
    auto cs = CloudSet<metric>::borrow(centers.size(), d_, centers.data(), offsets.data());
    assignment_->run(points_, cs, out);
  }

 private:
  uint32_t d_;
  std::vector<float> values_;
  std::vector<size_t> offsets_;
  CloudSet<metric> points_;
  std::unique_ptr<Assignment<metric>> assignment_;
  const TQ4Model* tq4_model_ = nullptr;
  typename TQ4Model::EncodedQuery tq4_queries_;
};

template<bool metric>
lloyds::CenterSet Inner(const parlay::sequence<parlay::sequence<float>>& data,
                        const parlay::sequence<float>* weights, uint32_t k, uint32_t limit,
                        bool verbose, ClusteringQuantizer codec, const MVClusteringConfig& config,
                        const PreparedVectors* prepared,
                        const parlay::sequence<uint32_t>& document_ids) {
  // Keep the original hash32 sample aligned with document weights and cached
  // root encodings. Full samples borrow the already materialized input rows.
  const size_t n = std::min(data.size(), static_cast<size_t>(limit) * k);
  parlay::sequence<parlay::sequence<float>> sampled;
  parlay::sequence<float> sampled_weights;
  if (n != data.size()) {
    sampled.resize(n);
    if (weights) sampled_weights = parlay::sequence<float>::uninitialized(n);
    parlay::parallel_for(0, n, [&](size_t i) {
      const size_t row = parlay::hash32(static_cast<uint32_t>(i)) % data.size();
      sampled[i] = data[row];
      if (weights) sampled_weights[i] = (*weights)[row];
    });
  }
  const auto& points = n == data.size() ? data : sampled;
  const auto* point_weights = weights && n != data.size() ? &sampled_weights : weights;
  const uint32_t d = data[0].size();
  const bool adaptive = config.inner_backend == "adaptive";
  std::optional<lloyds::LloydsInitialization> initialization;
  if (config.inner_init != "default")
    initialization =
        lloyds::initialize_inner<metric>(points, k, config.inner_init, config.seed, point_weights,
                                         codec == ClusteringQuantizer::None);
  auto run = [&](auto& backend) {
    return lloyds::lloyds_kmeans(backend, k, metric ? "PrefixDoubling" : "UniformlyRandom", 10,
                                 verbose, point_weights,
                                 initialization ? &*initialization : nullptr);
  };
  if (codec == ClusteringQuantizer::None) {
    lloyds::FloatLloydsBackend<metric> backend(points, d, adaptive);
    return run(backend);
  }
  if (codec == ClusteringQuantizer::TQ8) {
    // Pack all centers in one cloud so SingleCloudArgmin fills every SIMD lane.
    std::unique_ptr<lloyds::TQ8LloydsBackend<metric>> backend;
    if (prepared && prepared->root.total_size() <= std::numeric_limits<uint32_t>::max()) {
      // document_ids follow filter_flattened's order and cover each data row.
      auto root_vectors = parlay::sequence<uint32_t>::uninitialized(data.size());
      size_t cursor = 0;
      for (uint32_t row : document_ids) {
        const size_t begin = prepared->root.offsets[row] / prepared->root.d;
        const size_t end = prepared->root.offsets[row + 1] / prepared->root.d;
        for (size_t i = begin; i < end; ++i)
          root_vectors[cursor++] = i;
      }
      auto sampled_vectors = parlay::sequence<uint32_t>::uninitialized(n);
      for (size_t i = 0; i < n; ++i) {
        const size_t row =
            n == data.size() ? i : parlay::hash32(static_cast<uint32_t>(i)) % data.size();
        sampled_vectors[i] = root_vectors[row];
      }
      // Query encoding is metric-independent and always retains squared norms.
      // The inner objective gets its own typed model with the same seeded
      // rotation, sharing immutable bytes without a cast or second encoding.
      const auto& model = SharedModel<turboquant_8bit_mv::Model<metric>>(d, config.seed);
      const lloyds::TQ8VectorCache<metric> cache{&model,
                                                 prepared->vector_stride,
                                                 prepared->vector_data.data(),
                                                 prepared->vector_norms.data(),
                                                 prepared->vector_squared_norms.data(),
                                                 prepared->vector_byte_sums.data()};
      backend =
          std::make_unique<lloyds::TQ8LloydsBackend<metric>>(points, d, cache, sampled_vectors);
    } else {
      backend = std::make_unique<lloyds::TQ8LloydsBackend<metric>>(points, d, config.seed);
    }
    return run(*backend);
  }
  QuantizedInnerBackend<metric> backend(points, d, codec, config.seed, adaptive);
  return run(backend);
}

template<bool metric>
ClusteringResult Cluster(const PointCloudSetView& input, uint32_t k, uint32_t s,
                         const MVClusteringConfig& original,
                         const PreparedAssignment<metric>* prepared = nullptr,
                         std::span<const uint32_t> root_rows = {}) {
  if (static_cast<size_t>(k) * s > std::numeric_limits<size_t>::max() / input.d / sizeof(float))
    throw std::invalid_argument("Clustering center storage size overflows");
  auto config = original;
  auto codec = config.assignment_quantizer;
  if (codec == ClusteringQuantizer::None && config.build_with_8btq)
    codec = ClusteringQuantizer::TQ8;
  auto assignment = MakeAssignment<metric>(codec, config.seed, prepared);
  config.build_with_8btq = codec == ClusteringQuantizer::TQ8;
  MVClustering<metric> trainer(input.d, k, s, config);
  if (assignment) {
    trainer.assignment_function = [&](const auto& points, const auto& centers, auto& ids) {
      assignment->run(points, centers, ids);
    };
  }
  if (assignment || config.inner_metric != "same" || config.inner_init != "default" ||
      config.inner_backend != "legacy") {
    const bool inner_l2 = config.inner_metric == "l2" || (config.inner_metric == "same" && metric);
    trainer.inner_function = [codec, config, prepared, inner_l2](
                                 const auto& data, const auto* weights, auto count, auto limit,
                                 bool verbose, const auto& document_ids) {
      if (inner_l2)
        return Inner<true>(data, weights, count, limit, verbose, codec, config, prepared,
                           document_ids);
      return Inner<false>(data, weights, count, limit, verbose, codec, config, prepared,
                          document_ids);
    };
  }
  auto points = prepared ? CloudSet<metric>::borrow(input.n, input.d, input.values, input.offsets,
                                                    root_rows.data())
                         : Borrow<metric>(input);
  trainer.train(points);
  // Validate the published float centers once before the final assignment.
  // Input norm bounds and float means establish the invariant inside Lloyd.
  for (size_t i = 0; i < trainer.centers.total_size(); ++i) {
    double norm = 0;
    for (size_t j = 0; j < input.d; ++j) {
      const double value = trainer.centers.data()[i * input.d + j];
      norm += value * value;
    }
    constexpr double max_norm =
        double(std::numeric_limits<float>::max()) / (16.0 * std::numeric_limits<uint32_t>::max());
    if (!(norm <= max_norm))
      throw std::runtime_error("Clustering produced a center outside float scoring range");
  }
  auto assignments = trainer.get_clustering(points);
  auto centers = std::make_shared<CloudSet<metric>>(std::move(trainer.centers));
  const auto offsets = centers->get_offsets();
  ClusteringResult result;
  result.centers = {static_cast<uint32_t>(centers->size()),
                    input.d,
                    centers->data(),
                    offsets.begin(),
                    nullptr,
                    centers};
  result.assignments.assign(assignments.begin(), assignments.end());
  result.assignment_quantizer = codec;
  result.assignment_implementation = codec == ClusteringQuantizer::None ? "float_eigen"
                                     : codec == ClusteringQuantizer::TQ1
                                         ? "tq1_norm_aware_hamming"
                                         : "prepared_multivector_panels";
  for (const auto& it : trainer._iteration_stats)
    result.iterations.push_back({it.centroid_update_time, it.assignment_time, it.cost});
  return result;
}

template<bool metric>
std::vector<uint32_t> Assign(const PointCloudSetView& input, const PointCloudSetView& centers,
                             ClusteringQuantizer quantizer, uint32_t seed) {
  auto points = Borrow<metric>(input);
  auto cs = Borrow<metric>(centers);
  parlay::sequence<uint32_t> ids;
  // Inputs are nonempty, so total vector count equal to cloud count means
  // every cloud is a singleton. This is an assignment-boundary decision.
  const bool singleton = points.total_size() == points.size() && cs.total_size() == cs.size();
  auto assignment = MakeAssignment<metric>(quantizer, seed, nullptr, singleton);
  if (assignment)
    assignment->run(points, cs, ids);
  else {
    auto top = ManyToMany<CloudSet<metric>>::Top(points, cs);
    ids.resize(points.size());
    for (size_t i = 0; i < points.size(); ++i)
      ids[i] = top[i].first;
  }
  return {ids.begin(), ids.end()};
}
template<bool metric>
class MetricClusteringKernels final : public ClusteringKernels {
 public:
  MetricClusteringKernels(const PointCloudSetView& root, const MVClusteringConfig& config,
                          bool prepare_root) :
      config_(config) {
    if (config_.assignment_quantizer == ClusteringQuantizer::None && config_.build_with_8btq)
      config_.assignment_quantizer = ClusteringQuantizer::TQ8;
    if (prepare_root && config_.assignment_quantizer != ClusteringQuantizer::None)
      prepared_ = std::make_unique<PreparedAssignment<metric>>(root, config_.assignment_quantizer,
                                                               config_.seed);
  }
  ClusteringResult cluster(const PointCloudSetView& local, std::span<const uint32_t> root_rows,
                           uint32_t k, uint32_t s) const override {
    return Cluster<metric>(local, k, s, config_, prepared_.get(), root_rows);
  }
  std::vector<uint32_t> assign(const PointCloudSetView& points,
                               const PointCloudSetView& centers) const override {
    return Assign<metric>(points, centers, config_.assignment_quantizer, config_.seed);
  }

 private:
  MVClusteringConfig config_;
  std::unique_ptr<PreparedAssignment<metric>> prepared_;
};
}  // namespace

#ifndef MVSIC_CLUSTERING_FACTORY
#define MVSIC_CLUSTERING_FACTORY make_clustering_kernels
#endif
__attribute__((visibility("default"))) std::unique_ptr<ClusteringKernels> MVSIC_CLUSTERING_FACTORY(
    const PointCloudSetView& root, Metric metric, const MVClusteringConfig& config,
    bool prepare_root) {
  switch (metric) {
    case Metric::IP:
      return std::make_unique<MetricClusteringKernels<false>>(root, config, prepare_root);
    case Metric::L2:
      return std::make_unique<MetricClusteringKernels<true>>(root, config, prepare_root);
  }
  throw std::invalid_argument("Unknown metric");
}
}  // namespace mvsic
