#include "bind_macros.h"

#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/mvivf/mvivf_flat.h"

void register_mvivf_flat_l2_b(py::module_& m) {
  BIND_INDEX(IndexMVIVFFlatTQL2, ChamferL2_Point, "IndexMVIVFFlatTQL2")
  BIND_INDEX(IndexMVIVFFlatSPQTQL2, ChamferL2_Point, "IndexMVIVFFlatSPQTQL2")
  BIND_INDEX(IndexMVIVFFlatOneBitTQL2, ChamferL2_Point, "IndexMVIVFFlatOneBitTQL2")
  BIND_INDEX(IndexMVIVFFlatEightBitTQL2, ChamferL2_Point, "IndexMVIVFFlatEightBitTQL2")
  BIND_INDEX(IndexMVIVFFlatCompressTQL2, ChamferL2_Point, "IndexMVIVFFlatCompressTQL2")
  BIND_INDEX(IndexMVIVFFlatCompressSPQTQL2, ChamferL2_Point, "IndexMVIVFFlatCompressSPQTQL2")
  BIND_INDEX(IndexMVIVFFlatCompressOneBitTQL2, ChamferL2_Point, "IndexMVIVFFlatCompressOneBitTQL2")
  BIND_INDEX(IndexMVIVFFlatCompressEightBitTQL2, ChamferL2_Point,
             "IndexMVIVFFlatCompressEightBitTQL2")

  BIND_COMPUTE_STATS(IndexMVIVFFlatTQL2, ChamferL2_Point, "MVIVFFlatTQL2")
  BIND_COMPUTE_STATS(IndexMVIVFFlatSPQTQL2, ChamferL2_Point, "MVIVFFlatSPQTQL2")
  BIND_COMPUTE_STATS(IndexMVIVFFlatOneBitTQL2, ChamferL2_Point, "MVIVFFlatOneBitTQL2")
  BIND_COMPUTE_STATS(IndexMVIVFFlatEightBitTQL2, ChamferL2_Point, "MVIVFFlatEightBitTQL2")
  BIND_COMPUTE_STATS(IndexMVIVFFlatCompressTQL2, ChamferL2_Point, "MVIVFFlatCompressTQL2")
  BIND_COMPUTE_STATS(IndexMVIVFFlatCompressSPQTQL2, ChamferL2_Point, "MVIVFFlatCompressSPQTQL2")
  BIND_COMPUTE_STATS(IndexMVIVFFlatCompressOneBitTQL2, ChamferL2_Point,
                     "MVIVFFlatCompressOneBitTQL2")
  BIND_COMPUTE_STATS(IndexMVIVFFlatCompressEightBitTQL2, ChamferL2_Point,
                     "MVIVFFlatCompressEightBitTQL2")
}
