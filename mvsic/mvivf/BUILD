load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mvivf",
    hdrs = ["mvivf.h"],
    deps = [
        "//mvsic/core:index",
        "//mvsic/core:query_compression",
        "//mvsic/core/mvclustering",
        "//mvsic/core/utils:util",
    ],
)

cc_library(
    name = "mvivf_spill",
    hdrs = ["mvivf_spill.h"],
    deps = [
        "//mvsic/core:index",
        "//mvsic/core:query_compression",
        "//mvsic/core/mvclustering",
        "//mvsic/core/utils:util",
    ],
)

cc_library(
    name = "mvivf_flat",
    hdrs = ["mvivf_flat.h"],
    deps = [
        "//mvsic/core:index",
        "//mvsic/core:query_compression",
        "//mvsic/core/mvclustering",
        "//mvsic/core/utils:util",
    ],
)

cc_binary(
    name = "bench",
    srcs = ["bench.cpp"],
    deps = [
        ":mvivf",
        ":mvivf_flat",
        ":mvivf_spill",
        "//mvsic/core:stats",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/utils:parse_command_line",
    ],
)

# cc_binary(
#     name = "build_index",
#     srcs = ["build_index.cpp"],
#     deps = [
#         ":mvivf",
#         "//mvsic/core/types:chamfer_ip_point",
#         "//mvsic/core/types:point_cloud_set",
#         "//mvsic/core/utils:parse_command_line",
#     ],
# )

cc_binary(
    name = "bench_seq",
    srcs = ["bench_seq.cpp"],
    defines = ["PARLAY_SEQUENTIAL"],
    deps = [
        ":mvivf",
        ":mvivf_flat",
        ":mvivf_spill",
        "//mvsic/core:stats",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/utils:parse_command_line",
    ],
)

cc_binary(
    name = "bench_build",
    srcs = ["bench_build.cpp"],
    deps = [
        ":mvivf",
        ":mvivf_flat",
        ":mvivf_spill",
        "//mvsic/core:bench_utils",
    ],
)

cc_binary(
    name = "bench_search",
    srcs = ["bench_search.cpp"],
    deps = [
        ":mvivf",
        ":mvivf_flat",
        ":mvivf_spill",
        "//mvsic/core:bench_utils",
    ],
)

# bench_search_all is split across 48 per-cell translation units under
# bench_search_all_inst/ (one .cc per metric × variant × quantizer; each TU
# instantiates compress=true and compress=false). Bazel compiles them in
# parallel; the dispatch TU only sees `extern template` declarations.
cc_library(
    name = "bench_search_all_inst",
    srcs = glob(["bench_search_all_inst/*.cc"]),
    hdrs = [
        "bench_search_all_impl.h",
        "bench_search_all_inst.h",
    ],
    deps = [
        ":mvivf",
        ":mvivf_flat",
        ":mvivf_spill",
        "//mvsic/core:bench_utils",
    ],
)

cc_binary(
    name = "bench_search_all",
    srcs = ["bench_search_all.cpp"],
    deps = [
        ":bench_search_all_inst",
        ":mvivf",
        ":mvivf_flat",
        ":mvivf_spill",
        "//mvsic/core:bench_utils",
    ],
)

cc_binary(
    name = "bench_search_seq",
    srcs = ["bench_search_seq.cpp"],
    defines = ["PARLAY_SEQUENTIAL"],
    deps = [
        ":mvivf",
        ":mvivf_flat",
        ":mvivf_spill",
        "//mvsic/core:bench_utils",
    ],
)
