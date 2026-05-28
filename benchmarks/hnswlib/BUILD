load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

# Standalone CLI for upstream hnswlib multi-vector token indexing.
# Invoked as a subprocess by HnswlibWrapper in benchmarks/framework_utils.py.
cc_binary(
    name = "hnswlib_runner",
    srcs = ["runner.cc"],
    copts = ["-O3", "-march=native", "-fopenmp", "-w"],
    linkopts = ["-fopenmp"],
    deps = [
        "@eigen",
        "@hnswlib_upstream//:hnswlib",
        "@sigmod26gem//:cnpy",
    ],
)
