#include "bind_macros.h"

#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/muvera/muvera.h"

void register_muvera_ip(py::module_& m) {
  BIND_INDEX(IndexMUVERAIP, ChamferIP_Point, "IndexMUVERAIP")
  BIND_INDEX(IndexMUVERAPQIP, ChamferIP_Point, "IndexMUVERAPQIP")
  BIND_INDEX(IndexMUVERAFastScanIP, ChamferIP_Point, "IndexMUVERAFastScanIP")
  BIND_INDEX(IndexMUVERARaBitQIP, ChamferIP_Point, "IndexMUVERARaBitQIP")
  BIND_INDEX(IndexMUVERATQIP, ChamferIP_Point, "IndexMUVERATQIP")
  BIND_INDEX(IndexMUVERASPQTQIP, ChamferIP_Point, "IndexMUVERASPQTQIP")
  BIND_INDEX(IndexMUVERAOneBitTQIP, ChamferIP_Point, "IndexMUVERAOneBitTQIP")

  BIND_COMPUTE_STATS(IndexMUVERAIP, ChamferIP_Point, "MUVERAIP")
  BIND_COMPUTE_STATS(IndexMUVERAPQIP, ChamferIP_Point, "MUVERAPQIP")
  BIND_COMPUTE_STATS(IndexMUVERAFastScanIP, ChamferIP_Point, "MUVERAFastScanIP")
  BIND_COMPUTE_STATS(IndexMUVERARaBitQIP, ChamferIP_Point, "MUVERARaBitQIP")
  BIND_COMPUTE_STATS(IndexMUVERATQIP, ChamferIP_Point, "MUVERATQIP")
  BIND_COMPUTE_STATS(IndexMUVERASPQTQIP, ChamferIP_Point, "MUVERASPQTQIP")
  BIND_COMPUTE_STATS(IndexMUVERAOneBitTQIP, ChamferIP_Point, "MUVERAOneBitTQIP")
}
