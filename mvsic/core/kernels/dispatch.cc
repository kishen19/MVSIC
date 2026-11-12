#include "mvsic/core/kernels/dispatch.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace mvsic::kernels {

namespace {
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
constexpr CpuIsa dot_isas[] = {CpuIsa::Scalar, CpuIsa::AVX2, CpuIsa::AVX512, CpuIsa::AVX512VNNI};
#else
constexpr CpuIsa dot_isas[] = {CpuIsa::Scalar};
#endif
}  // namespace

bool compiled_isa(CpuIsa isa) noexcept {
  for (auto compiled : dot_isas) if (isa == compiled) return true;
  return false;
}

DotDispatch select_dot_kernel(CpuIsa requested) {
  return select_dot_kernel(requested, cpu_capabilities());
}
DotDispatch select_dot_kernel(CpuIsa requested, const CpuCapabilities& caps) {
  requested = select_isa(requested, dot_isas, caps);
  switch (requested) {
    case CpuIsa::AVX2: return {requested, detail::dot_avx2};
    case CpuIsa::AVX512: return {requested, detail::dot_avx512};
    case CpuIsa::AVX512VNNI: return {requested, detail::dot_vnni};
    default: return {CpuIsa::Scalar, detail::dot_scalar};
  }
}

namespace {
template<DotU8I8 dot>
void batch_dot_impl(DocumentPanelView docs, QueryPanelView queries,
                    int32_t* output, bool subtract_unsigned_bias) {
  for (size_t q = 0; q < queries.rows; ++q) {
    const int32_t bias = subtract_unsigned_bias ? 128 * queries.byte_sums[q] : 0;
    for (size_t p = 0; p < docs.rows; ++p) {
      output[q * docs.rows + p] =
          dot(docs.data + p * docs.stride, queries.data + q * queries.stride,
              docs.dimensions) - bias;
    }
  }
}
}  // namespace

void batch_dot(const DotDispatch& kernel, DocumentPanelView docs, QueryPanelView queries,
               int32_t* output, bool subtract_unsigned_bias) {
  if (!kernel.dot || docs.dimensions != queries.dimensions ||
      docs.dimensions > kMaxDotDimensions || docs.stride < docs.dimensions ||
      queries.stride < queries.dimensions) {
    throw std::invalid_argument("Invalid prepared dot panels or kernel");
  }
  if (docs.rows == 0 || queries.rows == 0) return;
  if (!output || !docs.data || !queries.data ||
      (subtract_unsigned_bias && !queries.byte_sums) ||
      queries.rows > std::numeric_limits<size_t>::max() / docs.rows ||
      (docs.stride && docs.rows > std::numeric_limits<size_t>::max() / docs.stride) ||
      (queries.stride && queries.rows > std::numeric_limits<size_t>::max() / queries.stride)) {
    throw std::invalid_argument("Missing dot panel storage or overflowing output size");
  }
  // Select once per prepared panel operation. Each specialization makes direct
  // calls to its concrete ISA kernel throughout the document/query loops.
  switch (kernel.isa) {
    case CpuIsa::Scalar:
      return batch_dot_impl<detail::dot_scalar>(docs, queries, output, subtract_unsigned_bias);
    case CpuIsa::AVX2:
      return batch_dot_impl<detail::dot_avx2>(docs, queries, output, subtract_unsigned_bias);
    case CpuIsa::AVX512:
      return batch_dot_impl<detail::dot_avx512>(docs, queries, output, subtract_unsigned_bias);
    case CpuIsa::AVX512VNNI:
      return batch_dot_impl<detail::dot_vnni>(docs, queries, output, subtract_unsigned_bias);
    default: throw std::invalid_argument("Unsupported prepared panel ISA");
  }
}

namespace detail {
int32_t dot_scalar(const uint8_t* a, const int8_t* b, size_t n) {
  int32_t out = 0;
  for (size_t i = 0; i < n; ++i) out += static_cast<int32_t>(a[i]) * b[i];
  return out;
}
}  // namespace detail
}  // namespace mvsic::kernels
