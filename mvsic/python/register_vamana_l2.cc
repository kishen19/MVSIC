#include "bind_macros.h"

#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/vamana/vamana.h"

void register_vamana_l2(py::module_& m) {
  BIND_INDEX(IndexVamanaL2, ChamferL2_Point, "IndexVamanaL2")
  BIND_INDEX(IndexVamanaPQL2, ChamferL2_Point, "IndexVamanaPQL2")
  BIND_INDEX(IndexVamanaFastScanL2, ChamferL2_Point, "IndexVamanaFastScanL2")
  BIND_INDEX(IndexVamanaRaBitQL2, ChamferL2_Point, "IndexVamanaRaBitQL2")
  BIND_INDEX(IndexVamanaTQL2, ChamferL2_Point, "IndexVamanaTQL2")
  BIND_INDEX(IndexVamanaSPQTQL2, ChamferL2_Point, "IndexVamanaSPQTQL2")
  BIND_INDEX(IndexVamanaOneBitTQL2, ChamferL2_Point, "IndexVamanaOneBitTQL2")
  BIND_INDEX(IndexVamanaEightBitTQL2, ChamferL2_Point, "IndexVamanaEightBitTQL2")

  BIND_COMPUTE_STATS(IndexVamanaL2, ChamferL2_Point, "VamanaL2")
  BIND_COMPUTE_STATS(IndexVamanaPQL2, ChamferL2_Point, "VamanaPQL2")
  BIND_COMPUTE_STATS(IndexVamanaFastScanL2, ChamferL2_Point, "VamanaFastScanL2")
  BIND_COMPUTE_STATS(IndexVamanaRaBitQL2, ChamferL2_Point, "VamanaRaBitQL2")
  BIND_COMPUTE_STATS(IndexVamanaTQL2, ChamferL2_Point, "VamanaTQL2")
  BIND_COMPUTE_STATS(IndexVamanaSPQTQL2, ChamferL2_Point, "VamanaSPQTQL2")
  BIND_COMPUTE_STATS(IndexVamanaOneBitTQL2, ChamferL2_Point, "VamanaOneBitTQL2")
  BIND_COMPUTE_STATS(IndexVamanaEightBitTQL2, ChamferL2_Point, "VamanaEightBitTQL2")
}
