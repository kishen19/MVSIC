load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "python",
    srcs = glob(["*.py"]),
    data = glob(["configs/*.json", "configs/*.yaml", "configs/*.yml"]),
)
