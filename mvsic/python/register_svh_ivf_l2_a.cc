#include "bind_macros.h"

#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/svh/svh_ivf.h"

void register_svh_ivf_l2_a(py::module_& m) {
  BIND_INDEX(IndexSVHIVFL2, ChamferL2_Point, "IndexSVHIVFL2")
  BIND_INDEX(IndexSVHIVFPQL2, ChamferL2_Point, "IndexSVHIVFPQL2")
  BIND_INDEX(IndexSVHIVFFastScanL2, ChamferL2_Point, "IndexSVHIVFFastScanL2")
  BIND_INDEX(IndexSVHIVFRaBitQL2, ChamferL2_Point, "IndexSVHIVFRaBitQL2")
  BIND_INDEX(IndexSVHIVFTQL2, ChamferL2_Point, "IndexSVHIVFTQL2")
  BIND_INDEX(IndexSVHIVFSPQTQL2, ChamferL2_Point, "IndexSVHIVFSPQTQL2")

  BIND_COMPUTE_STATS(IndexSVHIVFL2, ChamferL2_Point, "SVHIVFL2")
  BIND_COMPUTE_STATS(IndexSVHIVFPQL2, ChamferL2_Point, "SVHIVFPQL2")
  BIND_COMPUTE_STATS(IndexSVHIVFFastScanL2, ChamferL2_Point, "SVHIVFFastScanL2")
  BIND_COMPUTE_STATS(IndexSVHIVFRaBitQL2, ChamferL2_Point, "SVHIVFRaBitQL2")
  BIND_COMPUTE_STATS(IndexSVHIVFTQL2, ChamferL2_Point, "SVHIVFTQL2")
  BIND_COMPUTE_STATS(IndexSVHIVFSPQTQL2, ChamferL2_Point, "SVHIVFSPQTQL2")
}
