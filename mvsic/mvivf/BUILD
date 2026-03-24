load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mvivf",
    hdrs = ["mvivf.h"],
    deps = [
        "//mvsic/core:index",
        "//mvsic/core/mvclustering",
        "//mvsic/core/utils:util",
    ],
)

cc_library(
    name = "mvivf_spill",
    hdrs = ["mvivf_spill.h"],
    deps = [
        "//mvsic/core:index",
        "//mvsic/core/mvclustering",
        "//mvsic/core/utils:util",
    ],
)

cc_library(
    name = "mvivf_flat",
    hdrs = ["mvivf_flat.h"],
    deps = [
        "//mvsic/core:index",
        "//mvsic/core/mvclustering",
        "//mvsic/core/utils:util",
    ],
)

cc_binary(
    name = "bench",
    srcs = ["bench.cpp"],
    deps = [
        ":mvivf",
        ":mvivf_flat",
        ":mvivf_spill",
        "//mvsic/core:stats",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/utils:parse_command_line",
    ],
)
