load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "bench_mvivf_leaf_diversity",
    srcs = ["bench_mvivf_leaf_diversity.cpp"],
    deps = [
        "//mvsic/mvivf:mvivf",
        "//mvsic/core:stats",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/utils:parse_command_line",
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "bench_dedup",
    srcs = ["bench_dedup.cpp"],
    deps = [
        "//mvsic/core/utils:parse_command_line",
        "//mvsic/core/utils:util",
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "bench_sort",
    srcs = ["bench_sort.cpp"],
    deps = [
        "//mvsic/core/utils:parse_command_line",
        "//mvsic/core/utils:util",
        "@parlaylib//parlay:primitives",
    ],
)

# bench_search_all is split across 14 per-cell translation units under
# bench_search_all_inst/ (one .cc per metric × quantizer; each TU instantiates
# compress=true and compress=false). Bazel compiles them in parallel; the
# dispatch TU only sees `extern template` declarations.
cc_library(
    name = "bench_search_all_inst",
    srcs = glob(["bench_search_all_inst/*.cc"]),
    hdrs = [
        "bench_search_all_impl.h",
        "bench_search_all_inst.h",
    ],
    deps = [
        "//mvsic/core:stats",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/mvivf",
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "bench_search_all",
    srcs = ["bench_search_all.cpp"],
    deps = [
        ":bench_search_all_inst",
        "//mvsic/core:bench_utils",
        "//mvsic/core:stats",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/utils:parse_command_line",
        "//mvsic/mvivf",
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "bench_fastscan_m2m",
    srcs = ["bench_fastscan_m2m.cpp"],
    deps = [
        "//mvsic/core/quantization:fastscan_mv",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/utils:parse_command_line",
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "bench_chamfer_scoring",
    srcs = [
        "bench_chamfer_scoring.cpp",
        "legacy_tq_mv.h",
    ],
    deps = [
        "//mvsic/core/quantization:turboquant_mv",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/utils:parse_command_line",
        "//mvsic/mvivf:mvivf",
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "bench_chamfer_scoring_st",
    srcs = [
        "bench_chamfer_scoring_st.cpp",
        "legacy_tq_mv.h",
    ],
    deps = [
        "//mvsic/core/quantization:turboquant_mv",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/utils:parse_command_line",
        "//mvsic/mvivf:mvivf",
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "bench_1btq_harley_seal",
    srcs = ["bench_1btq_harley_seal.cpp"],
    deps = [],
)

cc_binary(
    name = "bench_1btq_vbmi",
    srcs = ["bench_1btq_vbmi.cpp"],
    deps = [],
)
