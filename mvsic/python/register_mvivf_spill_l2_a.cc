#include "bind_macros.h"

#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/mvivf/mvivf_spill.h"

void register_mvivf_spill_l2_a(py::module_& m) {
  BIND_INDEX(IndexMVIVFSpillL2, ChamferL2_Point, "IndexMVIVFSpillL2")
  BIND_INDEX(IndexMVIVFSpillCompressL2, ChamferL2_Point, "IndexMVIVFSpillCompressL2")
  BIND_INDEX(IndexMVIVFSpillPQL2, ChamferL2_Point, "IndexMVIVFSpillPQL2")
  BIND_INDEX(IndexMVIVFSpillFastScanL2, ChamferL2_Point, "IndexMVIVFSpillFastScanL2")
  BIND_INDEX(IndexMVIVFSpillRaBitQL2, ChamferL2_Point, "IndexMVIVFSpillRaBitQL2")
  BIND_INDEX(IndexMVIVFSpillCompressPQL2, ChamferL2_Point, "IndexMVIVFSpillCompressPQL2")
  BIND_INDEX(IndexMVIVFSpillCompressFastScanL2, ChamferL2_Point,
             "IndexMVIVFSpillCompressFastScanL2")
  BIND_INDEX(IndexMVIVFSpillCompressRaBitQL2, ChamferL2_Point, "IndexMVIVFSpillCompressRaBitQL2")

  BIND_COMPUTE_STATS(IndexMVIVFSpillL2, ChamferL2_Point, "MVIVFSpillL2")
  BIND_COMPUTE_STATS(IndexMVIVFSpillCompressL2, ChamferL2_Point, "MVIVFSpillCompressL2")
  BIND_COMPUTE_STATS(IndexMVIVFSpillPQL2, ChamferL2_Point, "MVIVFSpillPQL2")
  BIND_COMPUTE_STATS(IndexMVIVFSpillFastScanL2, ChamferL2_Point, "MVIVFSpillFastScanL2")
  BIND_COMPUTE_STATS(IndexMVIVFSpillRaBitQL2, ChamferL2_Point, "MVIVFSpillRaBitQL2")
  BIND_COMPUTE_STATS(IndexMVIVFSpillCompressPQL2, ChamferL2_Point, "MVIVFSpillCompressPQL2")
  BIND_COMPUTE_STATS(IndexMVIVFSpillCompressFastScanL2, ChamferL2_Point,
                     "MVIVFSpillCompressFastScanL2")
  BIND_COMPUTE_STATS(IndexMVIVFSpillCompressRaBitQL2, ChamferL2_Point, "MVIVFSpillCompressRaBitQL2")
}
