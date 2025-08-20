package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mvivf_flat",
    hdrs = ["mvivf_flat.h"],
    deps = [
        "//src/common:index",
        "//src/mvc:mvkmeans",
        "//src/utils:sort_utils",
        "//src/utils:top_neighbors",
    ],
)

cc_binary(
    name = "bench",
    srcs = ["bench.cpp"],
    deps = [
        ":mvivf_flat",
        "//src/utils:chamfer_ip_point",
        "//src/utils:chamfer_l2_point",
        "//src/utils:parse_command_line",
        "//src/utils:point_cloud_set",
        "//src/utils:stats",
    ],
)
