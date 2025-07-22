package(default_visibility = ["//visibility:public"])

cc_library(
    name = "index",
    hdrs = ["index.h"],
    deps = [
        ":index_params",
        ":search_params",
        "//src/utils:chamfer_ip_point",
        "//src/utils:chamfer_l2_point",
        "//src/utils:point_cloud_set",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "search_params",
    hdrs = ["search_params.h"],
)

cc_library(
    name = "index_params",
    hdrs = ["index_params.h"],
)
