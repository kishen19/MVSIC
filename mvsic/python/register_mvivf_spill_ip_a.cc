#include "bind_macros.h"

#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/mvivf/mvivf_spill.h"

void register_mvivf_spill_ip_a(py::module_& m) {
  BIND_INDEX(IndexMVIVFSpillIP, ChamferIP_Point, "IndexMVIVFSpillIP")
  BIND_INDEX(IndexMVIVFSpillCompressIP, ChamferIP_Point, "IndexMVIVFSpillCompressIP")
  BIND_INDEX(IndexMVIVFSpillPQIP, ChamferIP_Point, "IndexMVIVFSpillPQIP")
  BIND_INDEX(IndexMVIVFSpillFastScanIP, ChamferIP_Point, "IndexMVIVFSpillFastScanIP")
  BIND_INDEX(IndexMVIVFSpillRaBitQIP, ChamferIP_Point, "IndexMVIVFSpillRaBitQIP")
  BIND_INDEX(IndexMVIVFSpillCompressPQIP, ChamferIP_Point, "IndexMVIVFSpillCompressPQIP")
  BIND_INDEX(IndexMVIVFSpillCompressFastScanIP, ChamferIP_Point,
             "IndexMVIVFSpillCompressFastScanIP")
  BIND_INDEX(IndexMVIVFSpillCompressRaBitQIP, ChamferIP_Point, "IndexMVIVFSpillCompressRaBitQIP")

  BIND_COMPUTE_STATS(IndexMVIVFSpillIP, ChamferIP_Point, "MVIVFSpillIP")
  BIND_COMPUTE_STATS(IndexMVIVFSpillCompressIP, ChamferIP_Point, "MVIVFSpillCompressIP")
  BIND_COMPUTE_STATS(IndexMVIVFSpillPQIP, ChamferIP_Point, "MVIVFSpillPQIP")
  BIND_COMPUTE_STATS(IndexMVIVFSpillFastScanIP, ChamferIP_Point, "MVIVFSpillFastScanIP")
  BIND_COMPUTE_STATS(IndexMVIVFSpillRaBitQIP, ChamferIP_Point, "MVIVFSpillRaBitQIP")
  BIND_COMPUTE_STATS(IndexMVIVFSpillCompressPQIP, ChamferIP_Point, "MVIVFSpillCompressPQIP")
  BIND_COMPUTE_STATS(IndexMVIVFSpillCompressFastScanIP, ChamferIP_Point,
                     "MVIVFSpillCompressFastScanIP")
  BIND_COMPUTE_STATS(IndexMVIVFSpillCompressRaBitQIP, ChamferIP_Point, "MVIVFSpillCompressRaBitQIP")
}
