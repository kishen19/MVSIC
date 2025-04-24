# package(default_visibility = ["//:__subpackages__"])
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "uniformlyrandom",
    hdrs = ["uniformlyrandom.h"],
    deps = [
      "@parlaylib//parlay:parallel",
      "@parlaylib//parlay:primitives",
      "@parlaylib//parlay:sequence",
      "//mvc/utils:pointcloud",
    ],
)