#include "bind_macros.h"

#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/mvivf/mvivf.h"

void register_mvivf_ip_a(py::module_& m) {
  BIND_INDEX(IndexMVIVFIP, ChamferIP_Point, "IndexMVIVFIP")
  BIND_INDEX(IndexMVIVFCompressIP, ChamferIP_Point, "IndexMVIVFCompressIP")
  BIND_INDEX(IndexMVIVFPQIP, ChamferIP_Point, "IndexMVIVFPQIP")
  BIND_INDEX(IndexMVIVFFastScanIP, ChamferIP_Point, "IndexMVIVFFastScanIP")
  BIND_INDEX(IndexMVIVFRaBitQIP, ChamferIP_Point, "IndexMVIVFRaBitQIP")
  BIND_INDEX(IndexMVIVFCompressPQIP, ChamferIP_Point, "IndexMVIVFCompressPQIP")
  BIND_INDEX(IndexMVIVFCompressFastScanIP, ChamferIP_Point, "IndexMVIVFCompressFastScanIP")
  BIND_INDEX(IndexMVIVFCompressRaBitQIP, ChamferIP_Point, "IndexMVIVFCompressRaBitQIP")

  BIND_COMPUTE_STATS(IndexMVIVFIP, ChamferIP_Point, "MVIVFIP")
  BIND_COMPUTE_STATS(IndexMVIVFCompressIP, ChamferIP_Point, "MVIVFCompressIP")
  BIND_COMPUTE_STATS(IndexMVIVFPQIP, ChamferIP_Point, "MVIVFPQIP")
  BIND_COMPUTE_STATS(IndexMVIVFFastScanIP, ChamferIP_Point, "MVIVFFastScanIP")
  BIND_COMPUTE_STATS(IndexMVIVFRaBitQIP, ChamferIP_Point, "MVIVFRaBitQIP")
  BIND_COMPUTE_STATS(IndexMVIVFCompressPQIP, ChamferIP_Point, "MVIVFCompressPQIP")
  BIND_COMPUTE_STATS(IndexMVIVFCompressFastScanIP, ChamferIP_Point, "MVIVFCompressFastScanIP")
  BIND_COMPUTE_STATS(IndexMVIVFCompressRaBitQIP, ChamferIP_Point, "MVIVFCompressRaBitQIP")

  m.def(
      "get_mvivf_tree_stats",
      [](mvsic::IndexMVIVFIP& index) {
        auto s = index.get_tree_stats();
        py::dict d;
        d["num_internal_nodes"] = s.num_internal_nodes;
        d["num_leaves"] = s.num_leaves;
        d["avg_leaf_size"] = s.avg_leaf_size;
        d["avg_internal_node_size"] = s.avg_internal_node_size;
        d["total_point_clouds_internal"] = s.total_point_clouds_internal;
        d["height"] = s.height;
        return d;
      },
      py::arg("index"), "Returns tree stats for an MVIVF IP index.");
}
