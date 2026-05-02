#include "bind_macros.h"

#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/mvivf/mvivf.h"

void register_mvivf_l2_a(py::module_& m) {
  BIND_INDEX(IndexMVIVFL2, ChamferL2_Point, "IndexMVIVFL2")
  BIND_INDEX(IndexMVIVFCompressL2, ChamferL2_Point, "IndexMVIVFCompressL2")
  BIND_INDEX(IndexMVIVFPQL2, ChamferL2_Point, "IndexMVIVFPQL2")
  BIND_INDEX(IndexMVIVFFastScanL2, ChamferL2_Point, "IndexMVIVFFastScanL2")
  BIND_INDEX(IndexMVIVFRaBitQL2, ChamferL2_Point, "IndexMVIVFRaBitQL2")
  BIND_INDEX(IndexMVIVFCompressPQL2, ChamferL2_Point, "IndexMVIVFCompressPQL2")
  BIND_INDEX(IndexMVIVFCompressFastScanL2, ChamferL2_Point, "IndexMVIVFCompressFastScanL2")
  BIND_INDEX(IndexMVIVFCompressRaBitQL2, ChamferL2_Point, "IndexMVIVFCompressRaBitQL2")

  BIND_COMPUTE_STATS(IndexMVIVFL2, ChamferL2_Point, "MVIVFL2")
  BIND_COMPUTE_STATS(IndexMVIVFCompressL2, ChamferL2_Point, "MVIVFCompressL2")
  BIND_COMPUTE_STATS(IndexMVIVFPQL2, ChamferL2_Point, "MVIVFPQL2")
  BIND_COMPUTE_STATS(IndexMVIVFFastScanL2, ChamferL2_Point, "MVIVFFastScanL2")
  BIND_COMPUTE_STATS(IndexMVIVFRaBitQL2, ChamferL2_Point, "MVIVFRaBitQL2")
  BIND_COMPUTE_STATS(IndexMVIVFCompressPQL2, ChamferL2_Point, "MVIVFCompressPQL2")
  BIND_COMPUTE_STATS(IndexMVIVFCompressFastScanL2, ChamferL2_Point, "MVIVFCompressFastScanL2")
  BIND_COMPUTE_STATS(IndexMVIVFCompressRaBitQL2, ChamferL2_Point, "MVIVFCompressRaBitQL2")

  // MVIVF-only: tree stats. L2 overload lives here; IP overload in register_mvivf_ip_a.
  m.def(
      "get_mvivf_tree_stats",
      [](mvsic::IndexMVIVFL2& index) {
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
      py::arg("index"), "Returns tree stats for an MVIVF L2 index.");
}
