package(default_visibility = ["//visibility:public"])


cc_library(
  name = "csvfile",
  hdrs = ["csvfile.h"],
)

cc_library(
  name = "search_params",
  hdrs = ["search_params.h"],
)

cc_library(
    name = "stats",
    hdrs = ["stats.h"],
    deps = [
      "@parlaylib//parlay:primitives",
      "//mvc/utils:point_cloud_set",
      ":csvfile",
      ":search_params",
    ],
)

cc_library(
    name = "top_neighbors",
    hdrs = ["top_neighbors.h"],
    deps = [
      "@parlaylib//parlay:primitives",
      "//mvc/utils:point_cloud_set",
    ],
)