#include "bind_macros.h"

#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/mvivf/mvivf_flat.h"

void register_mvivf_flat_l2_a(py::module_& m) {
  BIND_INDEX(IndexMVIVFFlatL2, ChamferL2_Point, "IndexMVIVFFlatL2")
  BIND_INDEX(IndexMVIVFFlatCompressL2, ChamferL2_Point, "IndexMVIVFFlatCompressL2")
  BIND_INDEX(IndexMVIVFFlatPQL2, ChamferL2_Point, "IndexMVIVFFlatPQL2")
  BIND_INDEX(IndexMVIVFFlatFastScanL2, ChamferL2_Point, "IndexMVIVFFlatFastScanL2")
  BIND_INDEX(IndexMVIVFFlatRaBitQL2, ChamferL2_Point, "IndexMVIVFFlatRaBitQL2")
  BIND_INDEX(IndexMVIVFFlatCompressPQL2, ChamferL2_Point, "IndexMVIVFFlatCompressPQL2")
  BIND_INDEX(IndexMVIVFFlatCompressFastScanL2, ChamferL2_Point, "IndexMVIVFFlatCompressFastScanL2")
  BIND_INDEX(IndexMVIVFFlatCompressRaBitQL2, ChamferL2_Point, "IndexMVIVFFlatCompressRaBitQL2")

  BIND_COMPUTE_STATS(IndexMVIVFFlatL2, ChamferL2_Point, "MVIVFFlatL2")
  BIND_COMPUTE_STATS(IndexMVIVFFlatCompressL2, ChamferL2_Point, "MVIVFFlatCompressL2")
  BIND_COMPUTE_STATS(IndexMVIVFFlatPQL2, ChamferL2_Point, "MVIVFFlatPQL2")
  BIND_COMPUTE_STATS(IndexMVIVFFlatFastScanL2, ChamferL2_Point, "MVIVFFlatFastScanL2")
  BIND_COMPUTE_STATS(IndexMVIVFFlatRaBitQL2, ChamferL2_Point, "MVIVFFlatRaBitQL2")
  BIND_COMPUTE_STATS(IndexMVIVFFlatCompressPQL2, ChamferL2_Point, "MVIVFFlatCompressPQL2")
  BIND_COMPUTE_STATS(IndexMVIVFFlatCompressFastScanL2, ChamferL2_Point,
                     "MVIVFFlatCompressFastScanL2")
  BIND_COMPUTE_STATS(IndexMVIVFFlatCompressRaBitQL2, ChamferL2_Point, "MVIVFFlatCompressRaBitQL2")
}
