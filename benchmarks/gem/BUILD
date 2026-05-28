load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

# Standalone CLI runner that lifts upstream sigmod26gem's `Solution` class
# (build_fine_cluster / search_with_fine_cluster / save_fine_cluster /
# load_fine_cluster / repair_fine_graph_structure) into a clean driver
# that can be invoked as a subprocess by GEMWrapper in
# benchmarks/framework_utils.py.
#
# Upstream's single-shot `example_vecset_search_gem.cpp` is intentionally
# *not* in our build graph: it hardcodes a dataset selector and allocates
# ~13 GB at startup for an MSMARCO train query file that doesn't exist on
# any system but the authors'. By depending only on the upstream library
# bits (@sigmod26gem//:hnswlib, :cnpy, :otlib) we keep upstream itself
# unmodified -- the runner is the only place that knows about GEM's
# data-on-disk conventions and our .pcs/.gt conversions.
cc_binary(
    name = "gem_runner",
    srcs = ["runner.cc"],
    copts = ["-Ofast", "-march=native", "-fopenmp", "-w"],
    linkopts = ["-fopenmp", "-lopenblas"],
    deps = [
        "@sigmod26gem//:cnpy",
        "@sigmod26gem//:hnswlib",
        "@sigmod26gem//:otlib",
    ],
)
