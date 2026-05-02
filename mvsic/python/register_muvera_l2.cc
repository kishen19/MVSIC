#include "bind_macros.h"

#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/muvera/muvera.h"

void register_muvera_l2(py::module_& m) {
  BIND_INDEX(IndexMUVERAL2, ChamferL2_Point, "IndexMUVERAL2")
  BIND_INDEX(IndexMUVERAPQL2, ChamferL2_Point, "IndexMUVERAPQL2")
  BIND_INDEX(IndexMUVERAFastScanL2, ChamferL2_Point, "IndexMUVERAFastScanL2")
  BIND_INDEX(IndexMUVERARaBitQL2, ChamferL2_Point, "IndexMUVERARaBitQL2")
  BIND_INDEX(IndexMUVERATQL2, ChamferL2_Point, "IndexMUVERATQL2")
  BIND_INDEX(IndexMUVERASPQTQL2, ChamferL2_Point, "IndexMUVERASPQTQL2")
  BIND_INDEX(IndexMUVERAOneBitTQL2, ChamferL2_Point, "IndexMUVERAOneBitTQL2")

  BIND_COMPUTE_STATS(IndexMUVERAL2, ChamferL2_Point, "MUVERAL2")
  BIND_COMPUTE_STATS(IndexMUVERAPQL2, ChamferL2_Point, "MUVERAPQL2")
  BIND_COMPUTE_STATS(IndexMUVERAFastScanL2, ChamferL2_Point, "MUVERAFastScanL2")
  BIND_COMPUTE_STATS(IndexMUVERARaBitQL2, ChamferL2_Point, "MUVERARaBitQL2")
  BIND_COMPUTE_STATS(IndexMUVERATQL2, ChamferL2_Point, "MUVERATQL2")
  BIND_COMPUTE_STATS(IndexMUVERASPQTQL2, ChamferL2_Point, "MUVERASPQTQL2")
  BIND_COMPUTE_STATS(IndexMUVERAOneBitTQL2, ChamferL2_Point, "MUVERAOneBitTQL2")
}
