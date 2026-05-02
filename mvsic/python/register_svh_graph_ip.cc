#include "bind_macros.h"

#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/svh/svh_graph.h"

void register_svh_graph_ip(py::module_& m) {
  BIND_INDEX(IndexSVHGraphIP, ChamferIP_Point, "IndexSVHGraphIP")
  BIND_INDEX(IndexSVHGraphPQIP, ChamferIP_Point, "IndexSVHGraphPQIP")
  BIND_INDEX(IndexSVHGraphFastScanIP, ChamferIP_Point, "IndexSVHGraphFastScanIP")
  BIND_INDEX(IndexSVHGraphRaBitQIP, ChamferIP_Point, "IndexSVHGraphRaBitQIP")
  BIND_INDEX(IndexSVHGraphTQIP, ChamferIP_Point, "IndexSVHGraphTQIP")
  BIND_INDEX(IndexSVHGraphSPQTQIP, ChamferIP_Point, "IndexSVHGraphSPQTQIP")

  BIND_COMPUTE_STATS(IndexSVHGraphIP, ChamferIP_Point, "SVHGraphIP")
  BIND_COMPUTE_STATS(IndexSVHGraphPQIP, ChamferIP_Point, "SVHGraphPQIP")
  BIND_COMPUTE_STATS(IndexSVHGraphFastScanIP, ChamferIP_Point, "SVHGraphFastScanIP")
  BIND_COMPUTE_STATS(IndexSVHGraphRaBitQIP, ChamferIP_Point, "SVHGraphRaBitQIP")
  BIND_COMPUTE_STATS(IndexSVHGraphTQIP, ChamferIP_Point, "SVHGraphTQIP")
  BIND_COMPUTE_STATS(IndexSVHGraphSPQTQIP, ChamferIP_Point, "SVHGraphSPQTQIP")
}
