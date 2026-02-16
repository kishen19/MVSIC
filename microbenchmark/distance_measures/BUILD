load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "one_to_many_bench",
    srcs = ["one_to_many_bench.cpp"],
    deps = [
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:point_cloud_set",
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "chamfer_ip_bench",
    srcs = ["chamfer_ip_bench.cpp"],
    copts = [
        "-mavx2",
        "-mfma",
    ],
    deps = [
        "//mvsic/core/distance_measures:one_to_one",
        "@parlaylib//parlay:primitives",
    ],
)
