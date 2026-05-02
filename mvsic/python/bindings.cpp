#include "bind_macros.h"

PYBIND11_MODULE(mvsic, m) {
  m.doc() = "Python bindings for mvsic";

  // Common types must register first: enums, IndexParams, SearchParams,
  // ChamferL2/IP_Point, PointCloudSet{L2,IP}, Stats/StatsExtended, recall fns.
  // The per-family registers reference these classes by C++ type and rely on
  // them already being known to pybind11.
  register_common(m);

  // MVIVF (tree variant). The two `get_mvivf_tree_stats` overloads live in
  // the L2/IP "_a" TUs respectively and form one Python overload set.
  register_mvivf_l2_a(m);
  register_mvivf_l2_b(m);
  register_mvivf_ip_a(m);
  register_mvivf_ip_b(m);

  register_mvivf_flat_l2_a(m);
  register_mvivf_flat_l2_b(m);
  register_mvivf_flat_ip_a(m);
  register_mvivf_flat_ip_b(m);

  register_mvivf_spill_l2_a(m);
  register_mvivf_spill_l2_b(m);
  register_mvivf_spill_ip_a(m);
  register_mvivf_spill_ip_b(m);

  register_muvera_l2(m);
  register_muvera_ip(m);

  register_vamana_l2(m);
  register_vamana_ip(m);

  register_mpool_l2(m);
  register_mpool_ip(m);

  register_svh_ivf_l2_a(m);
  register_svh_ivf_l2_b(m);
  register_svh_ivf_ip_a(m);
  register_svh_ivf_ip_b(m);

  register_svh_graph_l2(m);
  register_svh_graph_ip(m);
}
