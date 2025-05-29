package(default_visibility = ["//visibility:public"])


cc_library(
    name = "svheuristic",
    hdrs = ["svheuristic.h"],
    deps = [
      "@parlaylib//parlay:primitives",
      "//mvc/utils:euclidian_point",
      "//mvc/utils:faiss_kmeans",
      "//mvc/utils:mips_point",
      "//mvc/utils:point_range",
      "//mvivf:index",
      "//mvivf/utils:top_neighbors",
    ],
)

cc_binary(
    name = "svbench",
    srcs = ["svbench.cpp"],
    deps = [
      "//mvc/utils:chamfer_ip_point",
      "//mvc/utils:chamfer_l2_point",
      "//mvc/utils:euclidian_point",
      "//mvc/utils:mips_point",
      "//mvc/utils:parse_command_line",
      "//mvc/utils:point_cloud_set",
      "//mvc/utils:point_range",
      "//mvivf/utils:stats",
      ":svheuristic",
    ],
)