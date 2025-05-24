package(default_visibility = ["//visibility:public"])


cc_library(
  name = "lower_bounds",
  hdrs = ["lower_bounds.h"],
  deps = [
    "@parlaylib//parlay:sequence",
    "//mvc/utils:faiss_kmeans",
  ],
)

cc_library(
  name = "mvkmeans",
  hdrs = ["mvkmeans.h"],
  deps = [
    "@parlaylib//parlay:primitives",
    "//mvc/seeding:uniformlyrandom",
    "//mvc/utils:faiss_kmeans",
    "//mvc/utils:pointcloudset",
    "//mvc/utils:chamfer_l2_point",
    "//mvc/utils:chamfer_ip_point",
    ":lower_bounds",
  ],
)

cc_binary(
  name = "bench",
  srcs = ["bench.cpp"],
  deps = [
    "//mvc/utils:point_range",
    "//mvc/utils:euclidian_point",
    "//mvc/utils:mips_point",
    "//mvc/utils:parse_command_line",
    "//mvc/utils:pointcloudset",
    "//mvc/utils:chamfer_l2_point",
    "//mvc/utils:chamfer_ip_point",
    ":mvkmeans",
  ],
)