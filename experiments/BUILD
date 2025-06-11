# load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "mvivf_benchmark",
    srcs = ["mvivf_benchmark.cpp"],
    deps = [
        "//mvc/utils:chamfer_ip_point",
        "//mvc/utils:chamfer_l2_point",
        "//mvc/utils:parse_command_line",
        "//mvc/utils:point_cloud_set",
        "//mvivf",
        "//mvivf/utils:stats",
    ],
)

cc_binary(
    name = "svh_benchmark",
    srcs = ["svh_benchmark.cpp"],
    deps = [
        "//baselines:svheuristic",
        "//mvc/utils:chamfer_ip_point",
        "//mvc/utils:chamfer_l2_point",
        "//mvc/utils:euclidian_point",
        "//mvc/utils:mips_point",
        "//mvc/utils:parse_command_line",
        "//mvc/utils:point_cloud_set",
        "//mvc/utils:point_range",
        "//mvivf/utils:stats",
    ],
)
