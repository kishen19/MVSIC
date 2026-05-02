#include "bind_macros.h"

#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/mvivf/mvivf_spill.h"

void register_mvivf_spill_ip_b(py::module_& m) {
  BIND_INDEX(IndexMVIVFSpillTQIP, ChamferIP_Point, "IndexMVIVFSpillTQIP")
  BIND_INDEX(IndexMVIVFSpillSPQTQIP, ChamferIP_Point, "IndexMVIVFSpillSPQTQIP")
  BIND_INDEX(IndexMVIVFSpillOneBitTQIP, ChamferIP_Point, "IndexMVIVFSpillOneBitTQIP")
  BIND_INDEX(IndexMVIVFSpillEightBitTQIP, ChamferIP_Point, "IndexMVIVFSpillEightBitTQIP")
  BIND_INDEX(IndexMVIVFSpillCompressTQIP, ChamferIP_Point, "IndexMVIVFSpillCompressTQIP")
  BIND_INDEX(IndexMVIVFSpillCompressSPQTQIP, ChamferIP_Point, "IndexMVIVFSpillCompressSPQTQIP")
  BIND_INDEX(IndexMVIVFSpillCompressOneBitTQIP, ChamferIP_Point,
             "IndexMVIVFSpillCompressOneBitTQIP")
  BIND_INDEX(IndexMVIVFSpillCompressEightBitTQIP, ChamferIP_Point,
             "IndexMVIVFSpillCompressEightBitTQIP")

  BIND_COMPUTE_STATS(IndexMVIVFSpillTQIP, ChamferIP_Point, "MVIVFSpillTQIP")
  BIND_COMPUTE_STATS(IndexMVIVFSpillSPQTQIP, ChamferIP_Point, "MVIVFSpillSPQTQIP")
  BIND_COMPUTE_STATS(IndexMVIVFSpillOneBitTQIP, ChamferIP_Point, "MVIVFSpillOneBitTQIP")
  BIND_COMPUTE_STATS(IndexMVIVFSpillEightBitTQIP, ChamferIP_Point, "MVIVFSpillEightBitTQIP")
  BIND_COMPUTE_STATS(IndexMVIVFSpillCompressTQIP, ChamferIP_Point, "MVIVFSpillCompressTQIP")
  BIND_COMPUTE_STATS(IndexMVIVFSpillCompressSPQTQIP, ChamferIP_Point, "MVIVFSpillCompressSPQTQIP")
  BIND_COMPUTE_STATS(IndexMVIVFSpillCompressOneBitTQIP, ChamferIP_Point,
                     "MVIVFSpillCompressOneBitTQIP")
  BIND_COMPUTE_STATS(IndexMVIVFSpillCompressEightBitTQIP, ChamferIP_Point,
                     "MVIVFSpillCompressEightBitTQIP")
}
