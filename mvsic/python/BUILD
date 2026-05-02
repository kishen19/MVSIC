load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")
load("@rules_python//python:defs.bzl", "py_library")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

# Shared header: macros (BIND_INDEX / BIND_COMPUTE_STATS), parlay::sequence
# pybind11 type caster, and forward declarations of every register_*() entry
# point. Header-only so it isn't its own compile job.
cc_library(
    name = "bind_macros",
    hdrs = ["bind_macros.h"],
    deps = [
        "@parlaylib//parlay:primitives",
    ],
)

# ---------- Common types (must register before any per-family TU) ----------
pybind_library(
    name = "register_common",
    srcs = ["register_common.cc"],
    alwayslink = True,
    deps = [
        ":bind_macros",
        "//mvsic/core:index_params",
        "//mvsic/core:search_params",
        "//mvsic/core:stats",
        "//mvsic/core/mvclustering:mvclustering_config",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
    ],
)

# ---------- MVIVF (tree variant) ----------
[
    pybind_library(
        name = "register_mvivf_" + slab,
        srcs = ["register_mvivf_" + slab + ".cc"],
        alwayslink = True,
        deps = [
            ":bind_macros",
            "//mvsic/core:stats",
            "//mvsic/core/types:" + ("chamfer_l2_point" if slab.startswith("l2") else "chamfer_ip_point"),
            "//mvsic/core/types:point_cloud_set",
            "//mvsic/mvivf",
        ],
    )
    for slab in ["l2_a", "l2_b", "ip_a", "ip_b"]
]

# ---------- MVIVF Flat ----------
[
    pybind_library(
        name = "register_mvivf_flat_" + slab,
        srcs = ["register_mvivf_flat_" + slab + ".cc"],
        alwayslink = True,
        deps = [
            ":bind_macros",
            "//mvsic/core:stats",
            "//mvsic/core/types:" + ("chamfer_l2_point" if slab.startswith("l2") else "chamfer_ip_point"),
            "//mvsic/core/types:point_cloud_set",
            "//mvsic/mvivf:mvivf_flat",
        ],
    )
    for slab in ["l2_a", "l2_b", "ip_a", "ip_b"]
]

# ---------- MVIVF Spill ----------
[
    pybind_library(
        name = "register_mvivf_spill_" + slab,
        srcs = ["register_mvivf_spill_" + slab + ".cc"],
        alwayslink = True,
        deps = [
            ":bind_macros",
            "//mvsic/core:stats",
            "//mvsic/core/types:" + ("chamfer_l2_point" if slab.startswith("l2") else "chamfer_ip_point"),
            "//mvsic/core/types:point_cloud_set",
            "//mvsic/mvivf:mvivf_spill",
        ],
    )
    for slab in ["l2_a", "l2_b", "ip_a", "ip_b"]
]

# ---------- MUVERA ----------
[
    pybind_library(
        name = "register_muvera_" + pt,
        srcs = ["register_muvera_" + pt + ".cc"],
        alwayslink = True,
        deps = [
            ":bind_macros",
            "//mvsic/core:stats",
            "//mvsic/core/types:chamfer_" + pt + "_point",
            "//mvsic/core/types:point_cloud_set",
            "//mvsic/muvera",
        ],
    )
    for pt in ["l2", "ip"]
]

# ---------- Vamana ----------
[
    pybind_library(
        name = "register_vamana_" + pt,
        srcs = ["register_vamana_" + pt + ".cc"],
        alwayslink = True,
        deps = [
            ":bind_macros",
            "//mvsic/core:stats",
            "//mvsic/core/types:chamfer_" + pt + "_point",
            "//mvsic/core/types:point_cloud_set",
            "//mvsic/vamana",
        ],
    )
    for pt in ["l2", "ip"]
]

# ---------- MPool ----------
[
    pybind_library(
        name = "register_mpool_" + pt,
        srcs = ["register_mpool_" + pt + ".cc"],
        alwayslink = True,
        deps = [
            ":bind_macros",
            "//mvsic/core:stats",
            "//mvsic/core/types:chamfer_" + pt + "_point",
            "//mvsic/core/types:point_cloud_set",
            "//mvsic/mpool",
        ],
    )
    for pt in ["l2", "ip"]
]

# ---------- SVH IVF ----------
[
    pybind_library(
        name = "register_svh_ivf_" + slab,
        srcs = ["register_svh_ivf_" + slab + ".cc"],
        alwayslink = True,
        deps = [
            ":bind_macros",
            "//mvsic/core:stats",
            "//mvsic/core/types:" + ("chamfer_l2_point" if slab.startswith("l2") else "chamfer_ip_point"),
            "//mvsic/core/types:point_cloud_set",
            "//mvsic/svh:svh_ivf",
        ],
    )
    for slab in ["l2_a", "l2_b", "ip_a", "ip_b"]
]

# ---------- SVH Graph ----------
[
    pybind_library(
        name = "register_svh_graph_" + pt,
        srcs = ["register_svh_graph_" + pt + ".cc"],
        alwayslink = True,
        deps = [
            ":bind_macros",
            "//mvsic/core:stats",
            "//mvsic/core/types:chamfer_" + pt + "_point",
            "//mvsic/core/types:point_cloud_set",
            "//mvsic/svh:svh_graph",
        ],
    )
    for pt in ["l2", "ip"]
]

# ---------- The Python extension itself ----------
# bindings.cpp is a tiny dispatch TU that just calls each register_*() in order.
# All actual binding code lives in the per-family pybind_library targets above,
# which Bazel compiles in parallel.
pybind_extension(
    name = "mvsic",
    srcs = ["bindings.cpp"],
    deps = [
        ":bind_macros",
        ":register_common",
        ":register_mvivf_l2_a",
        ":register_mvivf_l2_b",
        ":register_mvivf_ip_a",
        ":register_mvivf_ip_b",
        ":register_mvivf_flat_l2_a",
        ":register_mvivf_flat_l2_b",
        ":register_mvivf_flat_ip_a",
        ":register_mvivf_flat_ip_b",
        ":register_mvivf_spill_l2_a",
        ":register_mvivf_spill_l2_b",
        ":register_mvivf_spill_ip_a",
        ":register_mvivf_spill_ip_b",
        ":register_muvera_l2",
        ":register_muvera_ip",
        ":register_vamana_l2",
        ":register_vamana_ip",
        ":register_mpool_l2",
        ":register_mpool_ip",
        ":register_svh_ivf_l2_a",
        ":register_svh_ivf_l2_b",
        ":register_svh_ivf_ip_a",
        ":register_svh_ivf_ip_b",
        ":register_svh_graph_l2",
        ":register_svh_graph_ip",
    ],
)

py_library(
    name = "python_pkg",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
)
