#include "bind_macros.h"

#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/svh/svh_ivf.h"

void register_svh_ivf_ip_a(py::module_& m) {
  BIND_INDEX(IndexSVHIVFIP, ChamferIP_Point, "IndexSVHIVFIP")
  BIND_INDEX(IndexSVHIVFPQIP, ChamferIP_Point, "IndexSVHIVFPQIP")
  BIND_INDEX(IndexSVHIVFFastScanIP, ChamferIP_Point, "IndexSVHIVFFastScanIP")
  BIND_INDEX(IndexSVHIVFRaBitQIP, ChamferIP_Point, "IndexSVHIVFRaBitQIP")
  BIND_INDEX(IndexSVHIVFTQIP, ChamferIP_Point, "IndexSVHIVFTQIP")
  BIND_INDEX(IndexSVHIVFSPQTQIP, ChamferIP_Point, "IndexSVHIVFSPQTQIP")

  BIND_COMPUTE_STATS(IndexSVHIVFIP, ChamferIP_Point, "SVHIVFIP")
  BIND_COMPUTE_STATS(IndexSVHIVFPQIP, ChamferIP_Point, "SVHIVFPQIP")
  BIND_COMPUTE_STATS(IndexSVHIVFFastScanIP, ChamferIP_Point, "SVHIVFFastScanIP")
  BIND_COMPUTE_STATS(IndexSVHIVFRaBitQIP, ChamferIP_Point, "SVHIVFRaBitQIP")
  BIND_COMPUTE_STATS(IndexSVHIVFTQIP, ChamferIP_Point, "SVHIVFTQIP")
  BIND_COMPUTE_STATS(IndexSVHIVFSPQTQIP, ChamferIP_Point, "SVHIVFSPQTQIP")
}
