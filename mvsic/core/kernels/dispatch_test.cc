#include "mvsic/core/kernels/dispatch.h"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>
#include "gtest/gtest.h"

namespace mvsic::kernels {
namespace {
TEST(CpuDispatch, SharedSelectionHonorsMethodSupportAndOsState) {
  CpuCapabilities caps;
  caps.os_avx = caps.os_avx512 = true;
  caps.avx2 = caps.avx512f = caps.avx512bw = caps.avx512dq = caps.avx512vl = caps.avx512vnni = true;
  caps.amx_tile = caps.amx_int8 = caps.os_amx_state = caps.amx_permission = true;
  constexpr CpuIsa limited[] = {CpuIsa::AVX2, CpuIsa::AVX512};
  EXPECT_EQ(select_isa(CpuIsa::Auto, limited, caps), CpuIsa::AVX512);
  EXPECT_THROW(select_isa(CpuIsa::AVX512VNNI, limited, caps), std::invalid_argument);
  EXPECT_THROW(select_isa(CpuIsa::AMX, limited, caps), std::invalid_argument);
  caps.os_avx512 = false;
  EXPECT_EQ(select_isa(CpuIsa::Auto, limited, caps), CpuIsa::AVX2);
  caps.os_avx = false;
  EXPECT_THROW(select_isa(CpuIsa::Auto, limited, caps), std::invalid_argument);
  EXPECT_THROW(select_isa(CpuIsa::Auto, {}, caps), std::invalid_argument);
}

TEST(CpuDispatch, RejectsUnsupportedAndUnimplementedAmx) {
  CpuCapabilities none;
  EXPECT_EQ(select_dot_kernel(CpuIsa::Auto, none).isa, CpuIsa::Scalar);
  EXPECT_THROW(select_dot_kernel(CpuIsa::AVX2, none), std::invalid_argument);
  CpuCapabilities all;
  all.os_avx = all.os_avx512 = true;
  all.avx2 = all.avx512f = all.avx512bw = all.avx512dq = all.avx512vl = all.avx512vnni = true;
  all.amx_tile = all.amx_int8 = all.os_amx_state = all.amx_permission = true;
  EXPECT_FALSE(compiled_isa(CpuIsa::AMX));
  EXPECT_THROW(select_dot_kernel(CpuIsa::AMX, all), std::invalid_argument);
  if (compiled_isa(CpuIsa::AVX512VNNI))
    EXPECT_EQ(select_dot_kernel(CpuIsa::Auto, all).isa, CpuIsa::AVX512VNNI);
  all.os_avx512 = false;
  if (compiled_isa(CpuIsa::AVX2)) EXPECT_EQ(select_dot_kernel(CpuIsa::Auto, all).isa, CpuIsa::AVX2);
}

TEST(CpuDispatch, EveryUsableKernelMatchesScalarForTailsAndExtremeBytes) {
  const auto caps = cpu_capabilities();
  std::mt19937 rng(42);
  for (auto isa : {CpuIsa::Scalar, CpuIsa::AVX2, CpuIsa::AVX512, CpuIsa::AVX512VNNI}) {
    DotDispatch kernel;
    try { kernel = select_dot_kernel(isa, caps); } catch (const std::invalid_argument&) { continue; }
    for (size_t n : {size_t{0},1ul,7ul,31ul,32ul,63ul,64ul,129ul,4096ul,65535ul}) {
      std::vector<uint8_t> a(n);
      std::vector<int8_t> b(n);
      for (size_t i = 0; i < n; ++i) { a[i] = rng() & 255; b[i] = static_cast<int8_t>(rng() & 255); }
      EXPECT_EQ(kernel.dot(a.data(), b.data(), n), detail::dot_scalar(a.data(), b.data(), n));
      std::fill(a.begin(), a.end(), 255);
      for (int extreme : {-128, 127}) {
        std::fill(b.begin(), b.end(), static_cast<int8_t>(extreme));
        EXPECT_EQ(kernel.dot(a.data(), b.data(), n), detail::dot_scalar(a.data(), b.data(), n));
      }
    }
  }
}

TEST(CpuDispatch, PreparedPanelsCorrectUnsignedBiasAndStrides) {
  const auto kernel = select_dot_kernel();
  // Three documents, two queries, nontrivial row strides and signed values.
  const uint8_t docs[] = {128,128,128,0, 255,1,128,0, 129,130,131,0};
  const int8_t queries[] = {127,-128,3,0,0, -2,4,7,0,0};
  const int32_t sums[] = {2,9};
  int32_t out[6];
  batch_dot(kernel, {docs,3,3,4}, {queries,sums,2,3,5}, out);
  for (size_t q = 0; q < 2; ++q) for (size_t p = 0; p < 3; ++p) {
    int32_t expected = 0;
    for (size_t j = 0; j < 3; ++j)
      expected += (static_cast<int32_t>(docs[p*4+j])-128)*queries[q*5+j];
    EXPECT_EQ(out[q*3+p], expected);
  }
  EXPECT_EQ(out[0], 0);
  EXPECT_EQ(out[3], 0);
  EXPECT_THROW(batch_dot(kernel, {docs,1,kMaxDotDimensions+1,kMaxDotDimensions+1},
                        {queries,sums,1,kMaxDotDimensions+1,kMaxDotDimensions+1}, out),
               std::invalid_argument);
}
}
}  // namespace mvsic::kernels
