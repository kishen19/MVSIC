#include "bind_macros.h"

#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/mvivf/mvivf_spill.h"

void register_mvivf_spill_l2_b(py::module_& m) {
  BIND_INDEX(IndexMVIVFSpillTQL2, ChamferL2_Point, "IndexMVIVFSpillTQL2")
  BIND_INDEX(IndexMVIVFSpillSPQTQL2, ChamferL2_Point, "IndexMVIVFSpillSPQTQL2")
  BIND_INDEX(IndexMVIVFSpillOneBitTQL2, ChamferL2_Point, "IndexMVIVFSpillOneBitTQL2")
  BIND_INDEX(IndexMVIVFSpillEightBitTQL2, ChamferL2_Point, "IndexMVIVFSpillEightBitTQL2")
  BIND_INDEX(IndexMVIVFSpillCompressTQL2, ChamferL2_Point, "IndexMVIVFSpillCompressTQL2")
  BIND_INDEX(IndexMVIVFSpillCompressSPQTQL2, ChamferL2_Point, "IndexMVIVFSpillCompressSPQTQL2")
  BIND_INDEX(IndexMVIVFSpillCompressOneBitTQL2, ChamferL2_Point,
             "IndexMVIVFSpillCompressOneBitTQL2")
  BIND_INDEX(IndexMVIVFSpillCompressEightBitTQL2, ChamferL2_Point,
             "IndexMVIVFSpillCompressEightBitTQL2")

  BIND_COMPUTE_STATS(IndexMVIVFSpillTQL2, ChamferL2_Point, "MVIVFSpillTQL2")
  BIND_COMPUTE_STATS(IndexMVIVFSpillSPQTQL2, ChamferL2_Point, "MVIVFSpillSPQTQL2")
  BIND_COMPUTE_STATS(IndexMVIVFSpillOneBitTQL2, ChamferL2_Point, "MVIVFSpillOneBitTQL2")
  BIND_COMPUTE_STATS(IndexMVIVFSpillEightBitTQL2, ChamferL2_Point, "MVIVFSpillEightBitTQL2")
  BIND_COMPUTE_STATS(IndexMVIVFSpillCompressTQL2, ChamferL2_Point, "MVIVFSpillCompressTQL2")
  BIND_COMPUTE_STATS(IndexMVIVFSpillCompressSPQTQL2, ChamferL2_Point, "MVIVFSpillCompressSPQTQL2")
  BIND_COMPUTE_STATS(IndexMVIVFSpillCompressOneBitTQL2, ChamferL2_Point,
                     "MVIVFSpillCompressOneBitTQL2")
  BIND_COMPUTE_STATS(IndexMVIVFSpillCompressEightBitTQL2, ChamferL2_Point,
                     "MVIVFSpillCompressEightBitTQL2")
}
