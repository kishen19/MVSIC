#include "bind_macros.h"

#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/svh/svh_ivf.h"

void register_svh_ivf_ip_b(py::module_& m) {
  BIND_INDEX(IndexSVHIVFCompressIP, ChamferIP_Point, "IndexSVHIVFCompressIP")
  BIND_INDEX(IndexSVHIVFCompressPQIP, ChamferIP_Point, "IndexSVHIVFCompressPQIP")
  BIND_INDEX(IndexSVHIVFCompressFastScanIP, ChamferIP_Point, "IndexSVHIVFCompressFastScanIP")
  BIND_INDEX(IndexSVHIVFCompressRaBitQIP, ChamferIP_Point, "IndexSVHIVFCompressRaBitQIP")
  BIND_INDEX(IndexSVHIVFCompressTQIP, ChamferIP_Point, "IndexSVHIVFCompressTQIP")
  BIND_INDEX(IndexSVHIVFCompressSPQTQIP, ChamferIP_Point, "IndexSVHIVFCompressSPQTQIP")

  BIND_COMPUTE_STATS(IndexSVHIVFCompressIP, ChamferIP_Point, "SVHIVFCompressIP")
  BIND_COMPUTE_STATS(IndexSVHIVFCompressPQIP, ChamferIP_Point, "SVHIVFCompressPQIP")
  BIND_COMPUTE_STATS(IndexSVHIVFCompressFastScanIP, ChamferIP_Point, "SVHIVFCompressFastScanIP")
  BIND_COMPUTE_STATS(IndexSVHIVFCompressRaBitQIP, ChamferIP_Point, "SVHIVFCompressRaBitQIP")
  BIND_COMPUTE_STATS(IndexSVHIVFCompressTQIP, ChamferIP_Point, "SVHIVFCompressTQIP")
  BIND_COMPUTE_STATS(IndexSVHIVFCompressSPQTQIP, ChamferIP_Point, "SVHIVFCompressSPQTQIP")
}
