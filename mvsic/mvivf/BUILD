package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mvivf",
    hdrs = ["mvivf.h"],
    deps = [
        "//mvsic/core:index",
        "//mvsic/core/mvclustering",
        "//mvsic/core/utils:top_neighbors",
        "//mvsic/core/utils:util",
    ],
)

cc_library(
    name = "mvivf_flat",
    hdrs = ["mvivf_flat.h"],
    deps = [
        "//mvsic/core:index",
        "//mvsic/core/mvclustering",
        "//mvsic/core/utils:top_neighbors",
        "//mvsic/core/utils:util",
    ],
)

cc_binary(
    name = "bench",
    srcs = ["bench.cpp"],
    deps = [
        ":mvivf",
        ":mvivf_flat",
        "//mvsic/core:stats",
        "//mvsic/core/utils:chamfer_ip_point",
        "//mvsic/core/utils:chamfer_l2_point",
        "//mvsic/core/utils:parse_command_line",
        "//mvsic/core/utils:point_cloud_set",
    ],
)
