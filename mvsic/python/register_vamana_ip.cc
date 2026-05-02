#include "bind_macros.h"

#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/vamana/vamana.h"

void register_vamana_ip(py::module_& m) {
  BIND_INDEX(IndexVamanaIP, ChamferIP_Point, "IndexVamanaIP")
  BIND_INDEX(IndexVamanaPQIP, ChamferIP_Point, "IndexVamanaPQIP")
  BIND_INDEX(IndexVamanaFastScanIP, ChamferIP_Point, "IndexVamanaFastScanIP")
  BIND_INDEX(IndexVamanaRaBitQIP, ChamferIP_Point, "IndexVamanaRaBitQIP")
  BIND_INDEX(IndexVamanaTQIP, ChamferIP_Point, "IndexVamanaTQIP")
  BIND_INDEX(IndexVamanaSPQTQIP, ChamferIP_Point, "IndexVamanaSPQTQIP")
  BIND_INDEX(IndexVamanaOneBitTQIP, ChamferIP_Point, "IndexVamanaOneBitTQIP")
  BIND_INDEX(IndexVamanaEightBitTQIP, ChamferIP_Point, "IndexVamanaEightBitTQIP")

  BIND_COMPUTE_STATS(IndexVamanaIP, ChamferIP_Point, "VamanaIP")
  BIND_COMPUTE_STATS(IndexVamanaPQIP, ChamferIP_Point, "VamanaPQIP")
  BIND_COMPUTE_STATS(IndexVamanaFastScanIP, ChamferIP_Point, "VamanaFastScanIP")
  BIND_COMPUTE_STATS(IndexVamanaRaBitQIP, ChamferIP_Point, "VamanaRaBitQIP")
  BIND_COMPUTE_STATS(IndexVamanaTQIP, ChamferIP_Point, "VamanaTQIP")
  BIND_COMPUTE_STATS(IndexVamanaSPQTQIP, ChamferIP_Point, "VamanaSPQTQIP")
  BIND_COMPUTE_STATS(IndexVamanaOneBitTQIP, ChamferIP_Point, "VamanaOneBitTQIP")
  BIND_COMPUTE_STATS(IndexVamanaEightBitTQIP, ChamferIP_Point, "VamanaEightBitTQIP")
}
