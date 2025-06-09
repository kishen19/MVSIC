# package(default_visibility = ["//:__subpackages__"])
package(default_visibility = ["//visibility:public"])

cc_library(
  name = "anns",
  hdrs = ["anns.h"],
  deps = [
    "@parlaylib//parlay:parallel",
    "@parlaylib//parlay:primitives",
    "@parlaylib//parlay:sequence",
    "@parlaylib//parlay:delayed_sequence",
    "@parlayann//algorithms/vamana:index",
    "@parlayann//algorithms/utils:types",
    "@parlayann//algorithms/utils:stats",
  ],
)

cc_library(
  name = "kmeans",
  hdrs = ["kmeans.h"],
  deps = [
    ":pairwise",
    ":anns",
    ":multi-swap",
    "//seeding:kmeansplusplus",
    "//seeding:prefixdoubling",
    "//seeding:uniformlyrandom",
    "//seeding:kmeansparallel",
    "//seeding:ksetcover",
    # "//seeding:wards",
    #"//utils:evals",
    "@parlaylib//parlay:io",
    "@parlaylib//parlay:sequence",
    "@parlayann//algorithms/utils:point_range",
    "@parlayann//algorithms/utils:types",
  ],
)

cc_binary(
  name = "kmeans_runner",
  srcs = ["kmeans.cpp"],
  deps = [
    ":kmeans",
    "@parlayann//algorithms/utils:euclidian_point",
    "@parlayann//algorithms/utils:mips_point",
    "@parlayann//algorithms/bench:parse_command_line",
  ],
)

cc_library(
  name = "multi-swap",
  hdrs = ["multi-swap.h"],
  deps = [
    "//seeding:kmeans_utils",
    "@parlaylib//parlay:parallel",
    "@parlaylib//parlay:primitives",
    "@parlaylib//parlay:sequence",
    "@parlaylib//parlay:monoid",
    "//seeding:utils",
  ],
)

cc_library(
  name = "pairwise",
  hdrs = ["pairwise.h"],
  deps = [
    "@parlaylib//parlay:parallel",
    "@parlaylib//parlay:primitives",
    "@parlaylib//parlay:sequence",
  ],
)