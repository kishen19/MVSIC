load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

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
    name = "evals",
    hdrs = ["evals.h"],
    deps = [],
)

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
        "//mvsic/core/distance_measures:many_to_many",
        "//mvsic/core/mvclustering/seeding:uniformlyrandom",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/utils:kmeans_util",
        "//mvsic/core/utils:util",
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "bench",
    srcs = ["bench.cpp"],
    deps = [
        ":evals",
        "//mvsic/core/mvclustering",
        "//mvsic/core/utils:parse_command_line",
    ],
)
