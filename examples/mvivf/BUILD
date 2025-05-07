
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "stats",
    hdrs = ["stats.h"],
    deps = [
      "@parlaylib//parlay:parallel",
    ],
)

cc_library(
    name = "utils",
    hdrs = ["utils.h"],
    deps = [
      "@parlaylib//parlay:parallel",
    ],
)

cc_library(
    name = "index",
    hdrs = ["index.h"],
    deps = [
      "@parlaylib//parlay:primitives",
      "@mvc//mvc:mvkmeans",
      ":utils",
    ],
)

cc_binary(
    name = "bench",
    srcs = ["bench.cpp"],
    deps = [
      ":index",
      ":stats",
      "@parlayann//algorithms/utils:point_range",
      "@parlayann//algorithms/utils:euclidian_point",
      "@parlayann//algorithms/utils:mips_point",
      "@parlayann//algorithms/bench:parse_command_line",
      "@mvc//mvc/utils:pointcloud",
      "@mvc//mvc/utils:chamferpoint",
    ],
)