load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mvivf_spill",
    hdrs = ["mvivf_spill.h"],
    deps = [
        "//mvsic/core:legacy_index",
        "//mvsic/core:query_compression",
        "//mvsic/algorithms/mvivf/mvclustering:mvclustering_impl",
        "//mvsic/core/utils:util",
    ],
)

cc_library(
    name = "mvivf_flat",
    hdrs = ["mvivf_flat.h"],
    deps = [
        "//mvsic/core:legacy_index",
        "//mvsic/core:query_compression",
        "//mvsic/algorithms/mvivf/mvclustering:mvclustering_impl",
        "//mvsic/core/utils:util",
    ],
)

# Optional Flat/Spill benchmarks retain their template instantiations.
# Regular MVIVF uses the installed config-based benchmark and compiled API.
# bench_inst/ contains one TU per variant × quantizer × compression.
cc_library(
    name = "bench_inst",
    srcs = glob(["bench_inst/*.cc"]),
    hdrs = [
        "bench_impl.h",
        "bench_inst.h",
        "bench_run_one_decl.h",
    ],
    deps = [
        ":mvivf_flat",
        ":mvivf_spill",
        "//mvsic/core:query_compression",
        "//mvsic/algorithms:stats",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:point_cloud_set",
    ],
)

cc_binary(
    name = "bench",
    srcs = ["bench.cpp"],
    deps = [
        ":bench_inst",
        ":mvivf_flat",
        ":mvivf_spill",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/utils:parse_command_line",
    ],
)

cc_binary(
    name = "bench_build",
    srcs = ["bench_build.cpp"],
    deps = [
        ":mvivf_flat",
        ":mvivf_spill",
        "//mvsic/algorithms:bench_utils",
    ],
)

cc_binary(
    name = "bench_search",
    srcs = ["bench_search.cpp"],
    deps = [
        ":mvivf_flat",
        ":mvivf_spill",
        "//mvsic/algorithms:bench_utils",
    ],
)

# bench_search_all is split across 18 per-cell translation units under
# bench_search_all_inst/ (one .cc per variant × quantizer; each TU
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
        ":mvivf_flat",
        ":mvivf_spill",
        "//mvsic/algorithms:bench_utils",
    ],
)

cc_binary(
    name = "bench_search_all",
    srcs = ["bench_search_all.cpp"],
    deps = [
        ":bench_search_all_inst",
        ":mvivf_flat",
        ":mvivf_spill",
        "//mvsic/algorithms:bench_utils",
    ],
)

# bench_search_seq is split across per-cell translation units under
# bench_search_seq_inst/ (one .cc per variant × quantizer; each TU
# instantiates compress=true and compress=false). Bazel compiles them in
# parallel; the dispatch TU only sees `extern template` declarations.
cc_library(
    name = "bench_search_seq_inst",
    srcs = glob(["bench_search_seq_inst/*.cc"]),
    hdrs = [
        "bench_search_seq_impl.h",
        "bench_search_seq_inst.h",
    ],
    deps = [
        ":mvivf_flat",
        ":mvivf_spill",
        "//mvsic/algorithms:bench_utils",
    ],
)

cc_binary(
    name = "bench_search_seq",
    srcs = ["bench_search_seq.cpp"],
    # Loading runs with full parlay parallelism; the per-query timed loop is
    # pinned to 1 worker by compute_stats_extended_p_threaded's
    # parlay::execute_with_scheduler(1, ...) wrapper, so QPS_seq still measures
    # single-threaded search.
    deps = [
        ":bench_search_seq_inst",
        ":mvivf_flat",
        ":mvivf_spill",
        "//mvsic/algorithms:bench_utils",
    ],
)
