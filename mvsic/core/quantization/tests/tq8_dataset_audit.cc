// Paired diagnostic: same encoded vectors, old saturating arithmetic versus
// corrected production kernel. Exact top-100 input is a fixed rerank pool.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>
#include "parlay/primitives.h"
#include "mvsic/core/stats.h"
#include "mvsic/core/quantization/turboquant_8bit_mv.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/point_cloud_set.h"

namespace tq = mvsic::turboquant_8bit_mv;

float Legacy(const tq::Model<false>::EncodedQuery& q,
             const tq::Model<false>::EncodedSet& db, size_t c) {
#ifdef __AVX512F__
  float total = 0;
  for (size_t qi = 0; qi < q.num_queries; ++qi) {
    __m512 best = _mm512_set1_ps(std::numeric_limits<float>::max());
    const auto* qp = q.flat_query_data.data() + qi * q.q_stride;
    const size_t np = (db.panel_offsets[c+1]-db.panel_offsets[c])/db.panel_bytes;
    for (size_t p = 0; p < np; ++p) {
      __m512i acc = _mm512_setzero_si512();
      const auto* panel = db.panel_data.data() + db.panel_offsets[c] + p*db.panel_bytes;
      for (size_t t = 0; t < db.total_tiles; ++t) {
        auto a = _mm512_loadu_si512(panel + t*64);
        int32_t packed;
        std::memcpy(&packed, qp+t*4, 4);
        auto b = _mm512_set1_epi32(packed);
        acc = _mm512_add_epi32(acc, _mm512_madd_epi16(
            _mm512_maddubs_epi16(a,b), _mm512_set1_epi16(1)));
      }
      const size_t po = db.point_offsets[c] + p*16;
      tq::internal::vnni_chamfer_epilogue<false>(acc, q.byte_sums[qi],
          q.norm_scaling_factors[qi], q.unquantized_squared_norms[qi],
          db.norm_scaling_factors.data()+po, db.unquantized_squared_norms.data()+po, best);
    }
    total += _mm512_reduce_min_ps(best);
  }
  return total / q.num_queries;
#else
  throw std::runtime_error("This diagnostic needs AVX512");
#endif
}

int main(int argc, char** argv) {
  if (argc != 4) { std::cerr << "usage: tq8_dataset_audit docs.pcs queries.pcs exact.gt\n"; return 1; }
  using Set = mvsic::PointCloudSet<mvsic::ChamferIP_Point>;
  Set docs(argv[1]), queries(argv[2]);
  std::string gt_path(argv[3]);
  auto gt = ReadGT(gt_path, queries.size());
  tq::Model<false> model;
  model.train(docs);
  auto db = model.encode(docs);
  struct Result { double old_recall=0, fixed_recall=0, old_error=0, fixed_error=0; size_t changed=0; };
  std::vector<Result> results(queries.size());
  parlay::parallel_for(0, queries.size(), [&](size_t qi) {
    auto raw = queries[qi];
    auto q = model.quantize_query(raw);
    std::vector<std::pair<float,uint32_t>> old, fixed;
    std::unordered_set<uint32_t> truth;
    for (size_t i=0;i<10;++i) truth.insert(gt[qi][i].first);
    for (auto [id, unused] : gt[qi]) {
      float exact = raw.distance(docs[id]);
      float a = Legacy(q,db,id);
      float b = tq::turboquant_8bit_mv_chamfer_distance<false>(q,db,id);
      old.push_back({a,id}); fixed.push_back({b,id});
      results[qi].old_error += std::abs(a-exact)/gt[qi].size();
      results[qi].fixed_error += std::abs(b-exact)/gt[qi].size();
      results[qi].changed += std::abs(a-b)>1e-6;
    }
    std::sort(old.begin(),old.end()); std::sort(fixed.begin(),fixed.end());
    for (size_t i=0;i<10;++i) {
      results[qi].old_recall += truth.count(old[i].second)/10.0;
      results[qi].fixed_recall += truth.count(fixed[i].second)/10.0;
    }
  },1);
  Result sum;
  for (auto r:results) { sum.old_recall+=r.old_recall;sum.fixed_recall+=r.fixed_recall;
    sum.old_error+=r.old_error;sum.fixed_error+=r.fixed_error;sum.changed+=r.changed; }
  std::cout << "queries,old_recall,fixed_recall,old_mean_abs_error,fixed_mean_abs_error,changed_scores\n"
    << queries.size() << ',' << sum.old_recall/queries.size() << ',' << sum.fixed_recall/queries.size()
    << ',' << sum.old_error/queries.size() << ',' << sum.fixed_error/queries.size() << ',' << sum.changed << '\n';
}
