#include "bind_macros.h"

#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/mvivf/mvivf.h"

void register_mvivf_l2_b(py::module_& m) {
  BIND_INDEX(IndexMVIVFTQL2, ChamferL2_Point, "IndexMVIVFTQL2")
  BIND_INDEX(IndexMVIVFSPQTQL2, ChamferL2_Point, "IndexMVIVFSPQTQL2")
  BIND_INDEX(IndexMVIVFOneBitTQL2, ChamferL2_Point, "IndexMVIVFOneBitTQL2")
  BIND_INDEX(IndexMVIVFEightBitTQL2, ChamferL2_Point, "IndexMVIVFEightBitTQL2")
  BIND_INDEX(IndexMVIVFCompressTQL2, ChamferL2_Point, "IndexMVIVFCompressTQL2")
  BIND_INDEX(IndexMVIVFCompressSPQTQL2, ChamferL2_Point, "IndexMVIVFCompressSPQTQL2")
  BIND_INDEX(IndexMVIVFCompressOneBitTQL2, ChamferL2_Point, "IndexMVIVFCompressOneBitTQL2")
  BIND_INDEX(IndexMVIVFCompressEightBitTQL2, ChamferL2_Point, "IndexMVIVFCompressEightBitTQL2")

  BIND_COMPUTE_STATS(IndexMVIVFTQL2, ChamferL2_Point, "MVIVFTQL2")
  BIND_COMPUTE_STATS(IndexMVIVFSPQTQL2, ChamferL2_Point, "MVIVFSPQTQL2")
  BIND_COMPUTE_STATS(IndexMVIVFOneBitTQL2, ChamferL2_Point, "MVIVFOneBitTQL2")
  BIND_COMPUTE_STATS(IndexMVIVFEightBitTQL2, ChamferL2_Point, "MVIVFEightBitTQL2")
  BIND_COMPUTE_STATS(IndexMVIVFCompressTQL2, ChamferL2_Point, "MVIVFCompressTQL2")
  BIND_COMPUTE_STATS(IndexMVIVFCompressSPQTQL2, ChamferL2_Point, "MVIVFCompressSPQTQL2")
  BIND_COMPUTE_STATS(IndexMVIVFCompressOneBitTQL2, ChamferL2_Point, "MVIVFCompressOneBitTQL2")
  BIND_COMPUTE_STATS(IndexMVIVFCompressEightBitTQL2, ChamferL2_Point, "MVIVFCompressEightBitTQL2")
}
