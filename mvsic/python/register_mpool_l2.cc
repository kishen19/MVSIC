#include "bind_macros.h"

#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/mpool/mpool.h"

void register_mpool_l2(py::module_& m) {
  BIND_INDEX(IndexMPoolL2, ChamferL2_Point, "IndexMPoolL2")
  BIND_INDEX(IndexMPoolPQL2, ChamferL2_Point, "IndexMPoolPQL2")
  BIND_INDEX(IndexMPoolFastScanL2, ChamferL2_Point, "IndexMPoolFastScanL2")
  BIND_INDEX(IndexMPoolRaBitQL2, ChamferL2_Point, "IndexMPoolRaBitQL2")
  BIND_INDEX(IndexMPoolTQL2, ChamferL2_Point, "IndexMPoolTQL2")
  BIND_INDEX(IndexMPoolSPQTQL2, ChamferL2_Point, "IndexMPoolSPQTQL2")

  BIND_COMPUTE_STATS(IndexMPoolL2, ChamferL2_Point, "MPoolL2")
  BIND_COMPUTE_STATS(IndexMPoolPQL2, ChamferL2_Point, "MPoolPQL2")
  BIND_COMPUTE_STATS(IndexMPoolFastScanL2, ChamferL2_Point, "MPoolFastScanL2")
  BIND_COMPUTE_STATS(IndexMPoolRaBitQL2, ChamferL2_Point, "MPoolRaBitQL2")
  BIND_COMPUTE_STATS(IndexMPoolTQL2, ChamferL2_Point, "MPoolTQL2")
  BIND_COMPUTE_STATS(IndexMPoolSPQTQL2, ChamferL2_Point, "MPoolSPQTQL2")
}
