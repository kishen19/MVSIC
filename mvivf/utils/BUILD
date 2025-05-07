package(default_visibility = ["//visibility:public"])


cc_library(
  name = "csvfile",
  hdrs = ["csvfile.h"],
)

cc_library(
    name = "stats",
    hdrs = ["stats.h"],
    deps = [
      "@parlaylib//parlay:parallel",
      ":csvfile",
    ],
)

cc_library(
    name = "utils",
    hdrs = ["utils.h"],
    deps = [
      "@parlaylib//parlay:parallel",
    ],
)