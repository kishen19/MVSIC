include_guard(GLOBAL)
include(FetchContent)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Overrides are read-only inputs. All patched headers/sources are copied into
# this build's private overlay; shared checkouts and Bazel caches are untouched.
function(mvsic_dependency name required_file url checksum)
  string(TOUPPER "${name}" upper)
  set(MVSIC_${upper}_SOURCE_DIR "" CACHE PATH "Read-only ${name} source checkout")
  if(MVSIC_${upper}_SOURCE_DIR)
    get_filename_component(source "${MVSIC_${upper}_SOURCE_DIR}" ABSOLUTE)
  else()
    set(hash_args)
    if(checksum)
      list(APPEND hash_args URL_HASH "SHA256=${checksum}")
    endif()
    # Populate only: upstream projects may otherwise build benchmarks, tests,
    # or native-only libraries with options inappropriate for the wheel.
    FetchContent_Declare(mvsic_${name}
      URL "${url}" ${hash_args}
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      SOURCE_SUBDIR _mvsic_no_upstream_project)
    FetchContent_MakeAvailable(mvsic_${name})
    set(source "${mvsic_${name}_SOURCE_DIR}")
  endif()
  if(required_file AND NOT EXISTS "${source}/${required_file}")
    message(FATAL_ERROR "${name} source lacks ${required_file}: ${source}")
  endif()
  set(MVSIC_${upper}_ROOT "${source}" PARENT_SCOPE)
endfunction()

# Same Eigen revision as MODULE.bazel's 4.0.0-20241125.bcr.2.
# Source metadata: bazel-central-registry/modules/eigen/4.0.0-20241125.bcr.2/source.json
mvsic_dependency(eigen Eigen/Core
  "https://github.com/eigen-mirror/eigen/archive/8ad4344ca79f2f248bc5ed70eec72e4b9c4d5e88.zip"
  "be45858c0e3fbb574e3b9442fd9270deeddeb9d8950561ebc8cde317898dedc7")
mvsic_dependency(parlay ""
  "https://github.com/ParAlg/parlaylib/archive/refs/tags/bzlmod-2025.tar.gz"
  "57c177c860c5daaf963fbca9dff9d5172ec73fcdad16a713cc1af6b151c84640")
mvsic_dependency(rabitq include/rabitqlib/utils/rotator.hpp
  "https://github.com/VectorDB-NTU/RaBitQ-Library/archive/abd9ddb61c9267b547e7486a3f526336fd1177e2.zip"
  "19e58dad0ad48f7f64c4ad5abbdb96045a231b56e91289f705ad62e05a5f2115")
mvsic_dependency(parlayann algorithms/utils/point_range.h
  "https://github.com/kishen19/ParlayANN/archive/91805dbb5aa2f26992d804ebfa1991f6383ca86d.tar.gz"
  "9c8abf6837f55a046b447e6e7c75568472dd1dec3c06960c50367b4212b7e201")
mvsic_dependency(highway hwy/contrib/sort/vqsort.h
  "https://github.com/google/highway/releases/download/1.3.0/highway-1.3.0.tar.gz"
  "e8d696900b45f4123be8a9d6866f4e7b6831bf599f4b9c178964d968e6a58a69")

# Bazel's Parlay source cache strips include/. Accept either the repository
# root or that header root as an override.
if(EXISTS "${MVSIC_PARLAY_ROOT}/include/parlay/scheduler.h")
  set(MVSIC_PARLAY_INCLUDE "${MVSIC_PARLAY_ROOT}/include")
elseif(EXISTS "${MVSIC_PARLAY_ROOT}/parlay/scheduler.h")
  set(MVSIC_PARLAY_INCLUDE "${MVSIC_PARLAY_ROOT}")
else()
  message(FATAL_ERROR "Parlay source lacks parlay/scheduler.h")
endif()

set(MVSIC_DEPENDENCY_OVERLAY "${CMAKE_CURRENT_BINARY_DIR}/_deps/mvsic_overlay")
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/prepare_dependencies.py"
    --parlay "${MVSIC_PARLAY_INCLUDE}"
    --rabitq "${MVSIC_RABITQ_ROOT}"
    --parlayann "${MVSIC_PARLAYANN_ROOT}"
    --output "${MVSIC_DEPENDENCY_OVERLAY}"
    --patches "${CMAKE_CURRENT_LIST_DIR}/../third_party"
  RESULT_VARIABLE overlay_status
  OUTPUT_VARIABLE overlay_output
  ERROR_VARIABLE overlay_error)
if(NOT overlay_status EQUAL 0)
  message(FATAL_ERROR "Dependency overlay failed:\n${overlay_output}\n${overlay_error}")
endif()

# RaBitQ's runtime dispatcher references all these SIMD families, even though
# MVIVF primarily needs the rotation entry points. Compile ISA sources apart;
# generic dispatch stays portable and no global -march=native is applied.
set(rabitq_root "${MVSIC_DEPENDENCY_OVERLAY}/rabitq")
set(rabitq_common
  "${rabitq_root}/src/utils/cpu_features.cpp"
  "${rabitq_root}/src/simd/dispatch.cpp")
set(rabitq_avx2)
set(rabitq_avx512)
foreach(family pack_excode space_excode space fastscan rotator)
  list(APPEND rabitq_avx2 "${rabitq_root}/src/simd/${family}_avx2.cpp")
  list(APPEND rabitq_avx512 "${rabitq_root}/src/simd/${family}_avx512.cpp")
endforeach()
list(APPEND rabitq_avx2 "${rabitq_root}/src/simd/warmup_avx2.cpp")
set(rabitq_avx512_popcnt "${rabitq_root}/src/simd/warmup_avx512.cpp")
add_library(mvsic_rabitq STATIC
  ${rabitq_common} ${rabitq_avx2} ${rabitq_avx512} ${rabitq_avx512_popcnt})
set_target_properties(mvsic_rabitq PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
target_compile_features(mvsic_rabitq PUBLIC cxx_std_20)
target_include_directories(mvsic_rabitq SYSTEM PUBLIC
  "${rabitq_root}/include" "${MVSIC_EIGEN_ROOT}")
target_link_libraries(mvsic_rabitq PUBLIC OpenMP::OpenMP_CXX)
set_source_files_properties(${rabitq_avx2} PROPERTIES
  COMPILE_OPTIONS "-mavx2;-mfma")
set_source_files_properties(${rabitq_avx512} PROPERTIES
  COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mfma")
set_source_files_properties(${rabitq_avx512_popcnt} PROPERTIES
  COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mavx512vpopcntdq;-mfma")

add_library(mvsic_dependencies INTERFACE)
target_include_directories(mvsic_dependencies SYSTEM BEFORE INTERFACE
  "${MVSIC_DEPENDENCY_OVERLAY}/parlay"
  "${MVSIC_DEPENDENCY_OVERLAY}/parlayann"
  "${rabitq_root}/include"
  "${MVSIC_EIGEN_ROOT}"
  "${PROJECT_SOURCE_DIR}/mvsic/core/kmeans"
  "${MVSIC_HIGHWAY_ROOT}")
target_compile_definitions(mvsic_dependencies INTERFACE PARLAY_SHARED_SCHEDULER=1)
target_link_libraries(mvsic_dependencies INTERFACE
  mvsic_rabitq Threads::Threads OpenMP::OpenMP_CXX)

# util.h exposes Highway sort declarations to legacy callers. New MVIVF does
# not instantiate those wrappers, so production needs only the headers.
# A parity target using the old implementation can request the actual library.
function(mvsic_enable_legacy_highway)
  if(NOT TARGET hwy_contrib)
    set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(HWY_ENABLE_CONTRIB ON CACHE BOOL "" FORCE)
    set(HWY_FORCE_STATIC_LIBS ON CACHE BOOL "" FORCE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    add_subdirectory("${MVSIC_HIGHWAY_ROOT}"
                     "${CMAKE_CURRENT_BINARY_DIR}/_deps/highway-build" EXCLUDE_FROM_ALL)
  endif()
endfunction()

# Comparison programs deliberately retain the original header-only scheduler
# and never link mvsic_core or the runtime facade.
add_library(mvsic_baseline_dependencies INTERFACE)
target_include_directories(mvsic_baseline_dependencies SYSTEM BEFORE INTERFACE
  "${MVSIC_PARLAY_INCLUDE}"
  "${MVSIC_DEPENDENCY_OVERLAY}/parlayann"
  "${rabitq_root}/include"
  "${MVSIC_EIGEN_ROOT}"
  "${PROJECT_SOURCE_DIR}/mvsic/core/kmeans"
  "${MVSIC_HIGHWAY_ROOT}")
target_link_libraries(mvsic_baseline_dependencies INTERFACE
  mvsic_rabitq Threads::Threads OpenMP::OpenMP_CXX)
if(MVSIC_BUILD_COMPARISON)
  mvsic_enable_legacy_highway()
  target_link_libraries(mvsic_baseline_dependencies INTERFACE hwy_contrib)
endif()
