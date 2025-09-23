package(default_visibility = ["//visibility:public"])

cc_library(
    name = "index",
    hdrs = ["index.h"],
    deps = [
        ":index_params",
        ":search_params",
        "//mvsic/core/utils:chamfer_ip_point",
        "//mvsic/core/utils:chamfer_l2_point",
        "//mvsic/core/utils:point_cloud_set",
        "//mvsic/core/utils:quantized_point_cloud_set",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "index_params",
    hdrs = ["index_params.h"],
    deps = [
        "//mvsic/core/mvclustering:mvclustering_config",
    ],
)

cc_library(
    name = "search_params",
    hdrs = ["search_params.h"],
)

cc_library(
    name = "stats",
    hdrs = ["stats.h"],
    deps = [
        ":search_params",
        "//mvsic/core/utils:csvfile",
        "@parlaylib//parlay:primitives",
    ],
)
