package(default_visibility = ["//visibility:public"])

# cc_library(
#     name = "lower_bounds",
#     hdrs = ["lower_bounds.h"],
#     deps = [
#         "//mvc/utils:faiss_kmeans",
#         "@parlaylib//parlay:sequence",
#     ],
# )

cc_library(
    name = "mvclustering_config",
    hdrs = ["mvclustering_config.h"],
    deps = [],
)

cc_library(
    name = "mvclustering",
    hdrs = ["mvclustering.h"],
    deps = [
        ":mvclustering_config",
        "//mvsic/core/mvclustering/seeding:uniformlyrandom",
        "//mvsic/core/utils:chamfer_ip_point",
        "//mvsic/core/utils:chamfer_l2_point",
        "//mvsic/core/utils:kmeans_util",
        "//mvsic/core/utils:point_cloud_set",
        "//mvsic/core/utils:util",
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "bench",
    srcs = ["bench.cpp"],
    deps = [
        "//mvsic/core/mvclustering",
        "//mvsic/core/utils:chamfer_ip_point",
        "//mvsic/core/utils:chamfer_l2_point",
        "//mvsic/core/utils:parse_command_line",
        "//mvsic/core/utils:point_cloud_set",
    ],
)
