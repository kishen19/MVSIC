#pragma once

#include <cstddef>
#include <cstdint>
#include "mvsic/core/kernels/capabilities.h"

namespace mvsic::kernels {

using DotU8I8 = int32_t (*)(const uint8_t*, const int8_t*, size_t);
struct DotDispatch {
  CpuIsa isa = CpuIsa::Scalar;
  DotU8I8 dot = nullptr;
};

// Safe int32 accumulation for every u8/s8 input, including -128. Selection
// happens once per operation, outside the document/query hot loops.
inline constexpr size_t kMaxDotDimensions = 65535;
bool compiled_isa(CpuIsa isa) noexcept;
DotDispatch select_dot_kernel(CpuIsa requested = CpuIsa::Auto);
DotDispatch select_dot_kernel(CpuIsa requested, const CpuCapabilities& caps);

// Encoders own these prepared panels; execution only borrows them. DB bytes
// are unsigned (signed code +128), query bytes signed. Strides are bytes.
// Per-query byte sums must match the encoded query when bias is subtracted.
struct DocumentPanelView {
  const uint8_t* data = nullptr;
  size_t rows = 0, dimensions = 0, stride = 0;
};
struct QueryPanelView {
  const int8_t* data = nullptr;
  const int32_t* byte_sums = nullptr;
  size_t rows = 0, dimensions = 0, stride = 0;
};

// Row-major [query,document] output. The caller provides rows_q*rows_d slots.
// This generic prepared-panel adapter is a correctness baseline; existing
// packed multi-vector kernels remain separate optimized operations.
void batch_dot(const DotDispatch& kernel, DocumentPanelView docs, QueryPanelView queries,
               int32_t* output, bool subtract_unsigned_bias = true);

namespace detail {
int32_t dot_scalar(const uint8_t*, const int8_t*, size_t);
int32_t dot_avx2(const uint8_t*, const int8_t*, size_t);
int32_t dot_avx512(const uint8_t*, const int8_t*, size_t);
int32_t dot_vnni(const uint8_t*, const int8_t*, size_t);
}  // namespace detail
}  // namespace mvsic::kernels
