#include "bind_macros.h"

#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/mvivf/mvivf_flat.h"

void register_mvivf_flat_ip_a(py::module_& m) {
  BIND_INDEX(IndexMVIVFFlatIP, ChamferIP_Point, "IndexMVIVFFlatIP")
  BIND_INDEX(IndexMVIVFFlatCompressIP, ChamferIP_Point, "IndexMVIVFFlatCompressIP")
  BIND_INDEX(IndexMVIVFFlatPQIP, ChamferIP_Point, "IndexMVIVFFlatPQIP")
  BIND_INDEX(IndexMVIVFFlatFastScanIP, ChamferIP_Point, "IndexMVIVFFlatFastScanIP")
  BIND_INDEX(IndexMVIVFFlatRaBitQIP, ChamferIP_Point, "IndexMVIVFFlatRaBitQIP")
  BIND_INDEX(IndexMVIVFFlatCompressPQIP, ChamferIP_Point, "IndexMVIVFFlatCompressPQIP")
  BIND_INDEX(IndexMVIVFFlatCompressFastScanIP, ChamferIP_Point, "IndexMVIVFFlatCompressFastScanIP")
  BIND_INDEX(IndexMVIVFFlatCompressRaBitQIP, ChamferIP_Point, "IndexMVIVFFlatCompressRaBitQIP")

  BIND_COMPUTE_STATS(IndexMVIVFFlatIP, ChamferIP_Point, "MVIVFFlatIP")
  BIND_COMPUTE_STATS(IndexMVIVFFlatCompressIP, ChamferIP_Point, "MVIVFFlatCompressIP")
  BIND_COMPUTE_STATS(IndexMVIVFFlatPQIP, ChamferIP_Point, "MVIVFFlatPQIP")
  BIND_COMPUTE_STATS(IndexMVIVFFlatFastScanIP, ChamferIP_Point, "MVIVFFlatFastScanIP")
  BIND_COMPUTE_STATS(IndexMVIVFFlatRaBitQIP, ChamferIP_Point, "MVIVFFlatRaBitQIP")
  BIND_COMPUTE_STATS(IndexMVIVFFlatCompressPQIP, ChamferIP_Point, "MVIVFFlatCompressPQIP")
  BIND_COMPUTE_STATS(IndexMVIVFFlatCompressFastScanIP, ChamferIP_Point,
                     "MVIVFFlatCompressFastScanIP")
  BIND_COMPUTE_STATS(IndexMVIVFFlatCompressRaBitQIP, ChamferIP_Point, "MVIVFFlatCompressRaBitQIP")
}
