# package(default_visibility = ["//:__subpackages__"])
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "chamferpoint",
    hdrs = ["chamferpoint.h"],
    deps = [
      "@parlaylib//parlay:parallel",
      "@parlaylib//parlay:primitives",
      "@parlaylib//parlay:sequence",
      "@eigen//:eigen",
    ],
)

cc_library(
  name = "faiss_kmeans",
  hdrs = ["faiss_kmeans.h"],
  deps = [
    "@parlaylib//parlay:primitives",
    "@faiss//:faiss_core",
  ],
)

cc_library(
    name = "pointcloud",
    hdrs = ["pointcloud.h"],
    deps = [
      "@parlaylib//parlay:parallel",
      "@parlaylib//parlay:primitives",
      "@parlaylib//parlay:sequence",
      ":chamferpoint",
    ],
)

