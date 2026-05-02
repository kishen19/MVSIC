#include "bind_macros.h"

#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/mpool/mpool.h"

void register_mpool_ip(py::module_& m) {
  BIND_INDEX(IndexMPoolIP, ChamferIP_Point, "IndexMPoolIP")
  BIND_INDEX(IndexMPoolPQIP, ChamferIP_Point, "IndexMPoolPQIP")
  BIND_INDEX(IndexMPoolFastScanIP, ChamferIP_Point, "IndexMPoolFastScanIP")
  BIND_INDEX(IndexMPoolRaBitQIP, ChamferIP_Point, "IndexMPoolRaBitQIP")
  BIND_INDEX(IndexMPoolTQIP, ChamferIP_Point, "IndexMPoolTQIP")
  BIND_INDEX(IndexMPoolSPQTQIP, ChamferIP_Point, "IndexMPoolSPQTQIP")

  BIND_COMPUTE_STATS(IndexMPoolIP, ChamferIP_Point, "MPoolIP")
  BIND_COMPUTE_STATS(IndexMPoolPQIP, ChamferIP_Point, "MPoolPQIP")
  BIND_COMPUTE_STATS(IndexMPoolFastScanIP, ChamferIP_Point, "MPoolFastScanIP")
  BIND_COMPUTE_STATS(IndexMPoolRaBitQIP, ChamferIP_Point, "MPoolRaBitQIP")
  BIND_COMPUTE_STATS(IndexMPoolTQIP, ChamferIP_Point, "MPoolTQIP")
  BIND_COMPUTE_STATS(IndexMPoolSPQTQIP, ChamferIP_Point, "MPoolSPQTQIP")
}
