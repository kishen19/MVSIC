#include "bind_macros.h"

#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/mvivf/mvivf_flat.h"

void register_mvivf_flat_ip_b(py::module_& m) {
  BIND_INDEX(IndexMVIVFFlatTQIP, ChamferIP_Point, "IndexMVIVFFlatTQIP")
  BIND_INDEX(IndexMVIVFFlatSPQTQIP, ChamferIP_Point, "IndexMVIVFFlatSPQTQIP")
  BIND_INDEX(IndexMVIVFFlatOneBitTQIP, ChamferIP_Point, "IndexMVIVFFlatOneBitTQIP")
  BIND_INDEX(IndexMVIVFFlatEightBitTQIP, ChamferIP_Point, "IndexMVIVFFlatEightBitTQIP")
  BIND_INDEX(IndexMVIVFFlatCompressTQIP, ChamferIP_Point, "IndexMVIVFFlatCompressTQIP")
  BIND_INDEX(IndexMVIVFFlatCompressSPQTQIP, ChamferIP_Point, "IndexMVIVFFlatCompressSPQTQIP")
  BIND_INDEX(IndexMVIVFFlatCompressOneBitTQIP, ChamferIP_Point, "IndexMVIVFFlatCompressOneBitTQIP")
  BIND_INDEX(IndexMVIVFFlatCompressEightBitTQIP, ChamferIP_Point,
             "IndexMVIVFFlatCompressEightBitTQIP")

  BIND_COMPUTE_STATS(IndexMVIVFFlatTQIP, ChamferIP_Point, "MVIVFFlatTQIP")
  BIND_COMPUTE_STATS(IndexMVIVFFlatSPQTQIP, ChamferIP_Point, "MVIVFFlatSPQTQIP")
  BIND_COMPUTE_STATS(IndexMVIVFFlatOneBitTQIP, ChamferIP_Point, "MVIVFFlatOneBitTQIP")
  BIND_COMPUTE_STATS(IndexMVIVFFlatEightBitTQIP, ChamferIP_Point, "MVIVFFlatEightBitTQIP")
  BIND_COMPUTE_STATS(IndexMVIVFFlatCompressTQIP, ChamferIP_Point, "MVIVFFlatCompressTQIP")
  BIND_COMPUTE_STATS(IndexMVIVFFlatCompressSPQTQIP, ChamferIP_Point, "MVIVFFlatCompressSPQTQIP")
  BIND_COMPUTE_STATS(IndexMVIVFFlatCompressOneBitTQIP, ChamferIP_Point,
                     "MVIVFFlatCompressOneBitTQIP")
  BIND_COMPUTE_STATS(IndexMVIVFFlatCompressEightBitTQIP, ChamferIP_Point,
                     "MVIVFFlatCompressEightBitTQIP")
}
