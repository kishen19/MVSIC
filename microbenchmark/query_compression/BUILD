load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "sweep_ball_carving",
    srcs = ["sweep_ball_carving.cpp"],
    deps = [
        ":sweep_impl",
        "//mvsic/core:bench_utils",
    ],
)

cc_binary(
    name = "sweep_wards",
    srcs = ["sweep_wards.cpp"],
    deps = [
        ":sweep_impl",
        "//mvsic/core:bench_utils",
    ],
)

cc_library(
    name = "sweep_impl",
    hdrs = ["sweep_impl.h"],
    deps = [
        "//mvsic/flat:flat",
        "//mvsic/core:query_compression",
        "//mvsic/core:stats",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/utils:parse_command_line",
    ],
)
