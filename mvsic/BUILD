load("@rules_python//python:packaging.bzl", "py_wheel")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "mvsic_py",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//mvsic/python:mvsic",
        "//mvsic/python:python_pkg",
    ],
)

py_wheel(
    name = "mvsic_wheel",
    distribution = "mvsic",
    python_tag = "py3",
    version = "0.0.1",
    deps = [
        ":mvsic_py",
        "//mvsic/python:mvsic",
        "//mvsic/python:python_pkg",
    ],
)
