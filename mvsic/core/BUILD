package(default_visibility = ["//visibility:public"])

cc_library(
    name = "index",
    hdrs = ["index.h"],
    deps = [
        ":index_params",
        ":search_params",
        "//mvsic/core/distance_measures:one_to_many",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/types:quantized_chamfer_point",
        "//mvsic/core/types:quantized_point_cloud_set",
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
