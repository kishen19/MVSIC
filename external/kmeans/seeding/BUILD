# package(default_visibility = ["//:__subpackages__"])
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "utils",
    hdrs = ["utils.h"],
)

cc_library(
    name = "kmeans_utils",
    hdrs = ["kmeans_utils.h"],
    deps = [
      "@parlaylib//parlay:parallel",
      "@parlaylib//parlay:primitives",
      "@parlaylib//parlay:sequence",
      "@parlaylib//parlay:delayed_sequence",
    ],
)

cc_library(
    name = "kmeansplusplus",
    hdrs = ["kmeansplusplus.h"],
    deps = [
      "@parlaylib//parlay:parallel",
      "@parlaylib//parlay:primitives",
      "@parlaylib//parlay:sequence",
      "//seeding:kmeans_utils",
    ],
)

cc_library(
    name = "prefixdoubling",
    hdrs = ["prefixdoubling.h"],
    deps = [
      "@parlaylib//parlay:parallel",
      "@parlaylib//parlay:primitives",
      "@parlaylib//parlay:sequence",
      "@parlaylib//parlay:delayed_sequence",
      "//seeding:kmeans_utils",
    ],
)

cc_library(
    name = "uniformlyrandom",
    hdrs = ["uniformlyrandom.h"],
    deps = [
      "@parlaylib//parlay:parallel",
      "@parlaylib//parlay:primitives",
      "@parlaylib//parlay:sequence",
      "@parlaylib//parlay:delayed_sequence",
    ],
)

cc_library(
    name = "kmeansparallel",
    hdrs = ["kmeansparallel.h"],
    deps = [
      "@parlaylib//parlay:parallel",
      "@parlaylib//parlay:primitives",
      "@parlaylib//parlay:sequence",
      "@parlaylib//parlay:delayed_sequence",
      "//seeding:utils",
      "//seeding:kmeans_utils",
      "kmeansplusplus",
    ],
)

cc_library(
    name = "ksetcover",
    hdrs = ["ksetcover.h"],
    deps = [
      "@parlaylib//parlay:parallel",
      "@parlaylib//parlay:primitives",
      "@parlaylib//parlay:sequence",
      "@parlaylib//parlay:delayed_sequence",
      "//seeding:utils",
      "//seeding:kmeans_utils",
      "kmeansplusplus",
    ],
)

cc_library(
    name = "wards",
    hdrs = ["wards.h"],
    deps = [
      "@parlaylib//parlay:parallel",
      "@parlaylib//parlay:primitives",
      "@parlaylib//parlay:sequence",
      "@parlaylib//parlay:delayed_sequence",
      "//seeding/wards:wards_hac",
      "//seeding/wards:dyn_point_range",
    ],
)