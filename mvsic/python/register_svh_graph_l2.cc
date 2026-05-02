#include "bind_macros.h"

#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/svh/svh_graph.h"

void register_svh_graph_l2(py::module_& m) {
  BIND_INDEX(IndexSVHGraphL2, ChamferL2_Point, "IndexSVHGraphL2")
  BIND_INDEX(IndexSVHGraphPQL2, ChamferL2_Point, "IndexSVHGraphPQL2")
  BIND_INDEX(IndexSVHGraphFastScanL2, ChamferL2_Point, "IndexSVHGraphFastScanL2")
  BIND_INDEX(IndexSVHGraphRaBitQL2, ChamferL2_Point, "IndexSVHGraphRaBitQL2")
  BIND_INDEX(IndexSVHGraphTQL2, ChamferL2_Point, "IndexSVHGraphTQL2")
  BIND_INDEX(IndexSVHGraphSPQTQL2, ChamferL2_Point, "IndexSVHGraphSPQTQL2")

  BIND_COMPUTE_STATS(IndexSVHGraphL2, ChamferL2_Point, "SVHGraphL2")
  BIND_COMPUTE_STATS(IndexSVHGraphPQL2, ChamferL2_Point, "SVHGraphPQL2")
  BIND_COMPUTE_STATS(IndexSVHGraphFastScanL2, ChamferL2_Point, "SVHGraphFastScanL2")
  BIND_COMPUTE_STATS(IndexSVHGraphRaBitQL2, ChamferL2_Point, "SVHGraphRaBitQL2")
  BIND_COMPUTE_STATS(IndexSVHGraphTQL2, ChamferL2_Point, "SVHGraphTQL2")
  BIND_COMPUTE_STATS(IndexSVHGraphSPQTQL2, ChamferL2_Point, "SVHGraphSPQTQL2")
}
