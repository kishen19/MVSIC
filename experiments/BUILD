package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "mvivf_benchmark",
    srcs = ["mvivf_benchmark.cpp"],
    deps = [
        "//src/mvivf",
        "//src/utils:chamfer_ip_point",
        "//src/utils:chamfer_l2_point",
        "//src/utils:parse_command_line",
        "//src/utils:point_cloud_set",
        "//src/utils:stats",
    ],
)

cc_binary(
    name = "svh_benchmark",
    srcs = ["svh_benchmark.cpp"],
    deps = [
        "//src/svh:svheuristic",
        "//src/utils:chamfer_ip_point",
        "//src/utils:chamfer_l2_point",
        "//src/utils:ip_point",
        "//src/utils:l2_point",
        "//src/utils:parse_command_line",
        "//src/utils:point_cloud_set",
        "//src/utils:point_range",
        "//src/utils:stats",
    ],
)

cc_binary(
    name = "vamana_benchmark",
    srcs = ["vamana_benchmark.cpp"],
    deps = [
        "//src/utils:chamfer_ip_point",
        "//src/utils:chamfer_l2_point",
        "//src/utils:parse_command_line",
        "//src/utils:point_cloud_set",
        "//src/utils:stats",
        "//src/vamana",
    ],
)
