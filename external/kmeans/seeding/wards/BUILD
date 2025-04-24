package(
    default_visibility = ["//visibility:public"],
)
licenses(["notice"])

cc_library(
    name = "aug_map",
    hdrs = ["aug_map.h"],
    deps = [
        "@cpam//cpam:cpam",
        "@cpam//pam:pam",
    ],
)

cc_library(
    name = "dyn_point_range",
    hdrs = ["dyn_point_range.h"],
    deps = [
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay/internal:file_map",
    ],
)

cc_library(
    name = "dynann",
    hdrs = ["dynann.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:random",
        "@parlayann//algorithms/utils:beamSearch",
        "@parlayann//algorithms/utils:graph",
        "@parlayann//algorithms/utils:NSGDist",
        "@parlayann//algorithms/utils:types",
        ":dyn_point_range",
        ":aug_map",
    ],
)

cc_library(
    name = "union_find",
    hdrs = ["union_find.h"],
    deps = [
        "@parlaylib//parlay:primitives"
    ],
)

cc_library(
    name = "wards_hac",
    hdrs = ["wards_hac.h"],
    deps = [
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay/internal:get_time",
        "@parlayann//algorithms/utils:beamSearch",
        "@parlayann//algorithms/utils:graph",
        "@parlayann//algorithms/utils:stats",
        "@parlayann//algorithms/utils:types",
        ":dynann",
        ":union_find",
    ],
)