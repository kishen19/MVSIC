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
    name = "mvkmeans",
    hdrs = ["mvkmeans.h"],
    deps = [
        # ":lower_bounds",
        "//src/mvc/seeding:uniformlyrandom",
        "//src/utils:chamfer_ip_point",
        "//src/utils:chamfer_l2_point",
        "//src/utils:kmeans_util",
        "//src/utils:point_cloud_set",
        "@parlayann//algorithms/utils:euclidian_point",
        "@parlayann//algorithms/utils:mips_point",
        "@parlayann//algorithms/utils:point_range",
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "bench",
    srcs = ["bench.cpp"],
    deps = [
        ":mvkmeans",
        "//src/utils:chamfer_ip_point",
        "//src/utils:chamfer_l2_point",
        "//src/utils:parse_command_line",
        "//src/utils:point_cloud_set",
    ],
)
