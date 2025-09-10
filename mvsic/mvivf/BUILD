package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mvivf",
    hdrs = ["mvivf.h"],
    deps = [
        "//src/common:index",
        "//src/mvc:mvkmeans",
        "//src/utils:top_neighbors",
    ],
)

cc_library(
    name = "mvivf_MVQ",
    hdrs = ["mvivf_MVQ.h"],
    deps = [
        "//src/common:index",
        "//src/mvc:mvkmeans",
        "//src/utils:ip_point",
        "//src/utils:kmeans_util",
        "//src/utils:l2_point",
        "//src/utils:point_range",
        "//src/utils:sort_utils",
        "//src/utils:top_neighbors",
    ],
)

cc_binary(
    name = "bench",
    srcs = ["bench.cpp"],
    deps = [
        ":mvivf",
        ":mvivf_MVQ",
        "//src/utils:chamfer_ip_point",
        "//src/utils:chamfer_l2_point",
        "//src/utils:parse_command_line",
        "//src/utils:point_cloud_set",
        "//src/utils:stats",
    ],
)
