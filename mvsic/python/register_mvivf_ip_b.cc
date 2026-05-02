#include "bind_macros.h"

#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/mvivf/mvivf.h"

void register_mvivf_ip_b(py::module_& m) {
  BIND_INDEX(IndexMVIVFTQIP, ChamferIP_Point, "IndexMVIVFTQIP")
  BIND_INDEX(IndexMVIVFSPQTQIP, ChamferIP_Point, "IndexMVIVFSPQTQIP")
  BIND_INDEX(IndexMVIVFOneBitTQIP, ChamferIP_Point, "IndexMVIVFOneBitTQIP")
  BIND_INDEX(IndexMVIVFEightBitTQIP, ChamferIP_Point, "IndexMVIVFEightBitTQIP")
  BIND_INDEX(IndexMVIVFCompressTQIP, ChamferIP_Point, "IndexMVIVFCompressTQIP")
  BIND_INDEX(IndexMVIVFCompressSPQTQIP, ChamferIP_Point, "IndexMVIVFCompressSPQTQIP")
  BIND_INDEX(IndexMVIVFCompressOneBitTQIP, ChamferIP_Point, "IndexMVIVFCompressOneBitTQIP")
  BIND_INDEX(IndexMVIVFCompressEightBitTQIP, ChamferIP_Point, "IndexMVIVFCompressEightBitTQIP")

  BIND_COMPUTE_STATS(IndexMVIVFTQIP, ChamferIP_Point, "MVIVFTQIP")
  BIND_COMPUTE_STATS(IndexMVIVFSPQTQIP, ChamferIP_Point, "MVIVFSPQTQIP")
  BIND_COMPUTE_STATS(IndexMVIVFOneBitTQIP, ChamferIP_Point, "MVIVFOneBitTQIP")
  BIND_COMPUTE_STATS(IndexMVIVFEightBitTQIP, ChamferIP_Point, "MVIVFEightBitTQIP")
  BIND_COMPUTE_STATS(IndexMVIVFCompressTQIP, ChamferIP_Point, "MVIVFCompressTQIP")
  BIND_COMPUTE_STATS(IndexMVIVFCompressSPQTQIP, ChamferIP_Point, "MVIVFCompressSPQTQIP")
  BIND_COMPUTE_STATS(IndexMVIVFCompressOneBitTQIP, ChamferIP_Point, "MVIVFCompressOneBitTQIP")
  BIND_COMPUTE_STATS(IndexMVIVFCompressEightBitTQIP, ChamferIP_Point, "MVIVFCompressEightBitTQIP")
}
