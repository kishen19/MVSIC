#include "bind_macros.h"

#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/svh/svh_ivf.h"

void register_svh_ivf_l2_b(py::module_& m) {
  BIND_INDEX(IndexSVHIVFCompressL2, ChamferL2_Point, "IndexSVHIVFCompressL2")
  BIND_INDEX(IndexSVHIVFCompressPQL2, ChamferL2_Point, "IndexSVHIVFCompressPQL2")
  BIND_INDEX(IndexSVHIVFCompressFastScanL2, ChamferL2_Point, "IndexSVHIVFCompressFastScanL2")
  BIND_INDEX(IndexSVHIVFCompressRaBitQL2, ChamferL2_Point, "IndexSVHIVFCompressRaBitQL2")
  BIND_INDEX(IndexSVHIVFCompressTQL2, ChamferL2_Point, "IndexSVHIVFCompressTQL2")
  BIND_INDEX(IndexSVHIVFCompressSPQTQL2, ChamferL2_Point, "IndexSVHIVFCompressSPQTQL2")

  BIND_COMPUTE_STATS(IndexSVHIVFCompressL2, ChamferL2_Point, "SVHIVFCompressL2")
  BIND_COMPUTE_STATS(IndexSVHIVFCompressPQL2, ChamferL2_Point, "SVHIVFCompressPQL2")
  BIND_COMPUTE_STATS(IndexSVHIVFCompressFastScanL2, ChamferL2_Point, "SVHIVFCompressFastScanL2")
  BIND_COMPUTE_STATS(IndexSVHIVFCompressRaBitQL2, ChamferL2_Point, "SVHIVFCompressRaBitQL2")
  BIND_COMPUTE_STATS(IndexSVHIVFCompressTQL2, ChamferL2_Point, "SVHIVFCompressTQL2")
  BIND_COMPUTE_STATS(IndexSVHIVFCompressSPQTQL2, ChamferL2_Point, "SVHIVFCompressSPQTQL2")
}
