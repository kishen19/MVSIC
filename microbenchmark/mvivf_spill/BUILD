load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

package(default_visibility = ["//visibility:public"])

# bench_search_all is split across 16 per-cell translation units under
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
        "//mvsic/mvivf:mvivf_spill",
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
        "//mvsic/mvivf:mvivf_spill",
        "@parlaylib//parlay:primitives",
    ],
)
