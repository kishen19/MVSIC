package(default_visibility = ["//visibility:public"])


cc_library(
  name = "index",
  hdrs = ["index.h"],
  deps = [
    "@parlaylib//parlay:primitives",
    "//mvc/utils:chamfer_ip_point",
    "//mvc/utils:chamfer_l2_point",
    "//mvc/utils:point_cloud_set",
    "//mvivf/utils:search_params",
  ],
)

cc_library(
  name = "mvivf",
  hdrs = ["mvivf.h"],
  deps = [
    "@absl//absl/container:btree",
    "//mvc:mvkmeans",
    "//mvivf/utils:top_neighbors",
    ":index",
  ],
)

cc_binary(
  name = "bench",
  srcs = ["bench.cpp"],
  deps = [
    "//mvc/utils:chamfer_ip_point",
    "//mvc/utils:chamfer_l2_point",
    "//mvc/utils:parse_command_line",
    "//mvc/utils:point_cloud_set",
    "//mvivf/utils:stats",
    ":mvivf",
  ],
)