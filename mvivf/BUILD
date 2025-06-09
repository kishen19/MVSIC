load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "index",
    hdrs = ["index.h"],
    deps = [
        "//mvc/utils:chamfer_ip_point",
        "//mvc/utils:chamfer_l2_point",
        "//mvc/utils:point_cloud_set",
        "//mvivf/utils:search_params",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "mvivf",
    hdrs = ["mvivf.h"],
    deps = [
        ":index",
        "//mvc:mvkmeans",
        "//mvivf/utils:top_neighbors",
        "@absl//absl/container:btree",
    ],
)

cc_binary(
    name = "bench",
    srcs = ["bench.cpp"],
    deps = [
        ":mvivf",
        "//mvc/utils:chamfer_ip_point",
        "//mvc/utils:chamfer_l2_point",
        "//mvc/utils:parse_command_line",
        "//mvc/utils:point_cloud_set",
        "//mvivf/utils:stats",
    ],
)
