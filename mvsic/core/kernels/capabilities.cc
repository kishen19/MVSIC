#include "mvsic/core/kernels/capabilities.h"

#include <algorithm>
#include <stdexcept>
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif
#endif

namespace mvsic::kernels {

CpuCapabilities cpu_capabilities() noexcept {
  CpuCapabilities c;
#if defined(__x86_64__) || defined(__i386__)
  c.x86 = true;
  unsigned a = 0, b = 0, d = 0, e = 0;
  if (!__get_cpuid(1, &a, &b, &e, &d)) return c;
  if ((e & (1u << 27)) != 0) {
    unsigned lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    c.xcr0 = (static_cast<uint64_t>(hi) << 32) | lo;
  }
  c.os_avx = ((e & (1u << 28)) != 0) && (c.xcr0 & 0x6) == 0x6;
  c.os_avx512 = c.os_avx && (c.xcr0 & 0xe6) == 0xe6;
  // x86-64-v3 is the portable package baseline, including the v2 features.
  constexpr unsigned base_ecx = (1u << 0) | (1u << 9) | (1u << 12) | (1u << 13) | (1u << 19) |
                                (1u << 20) | (1u << 22) | (1u << 23) | (1u << 29);
  const bool base = (e & base_ecx) == base_ecx;
  bool bmi = false;
  if (__get_cpuid_max(0, nullptr) >= 7) {
    __cpuid_count(7, 0, a, b, e, d);
    c.avx2 = (b & (1u << 5)) != 0;
    bmi = (b & ((1u << 3) | (1u << 8))) == ((1u << 3) | (1u << 8));
    c.avx512f = (b & (1u << 16)) != 0;
    c.avx512dq = (b & (1u << 17)) != 0;
    c.avx512bw = (b & (1u << 30)) != 0;
    c.avx512vl = (b & (1u << 31)) != 0;
    c.avx512vbmi = (e & (1u << 1)) != 0;
    c.avx512vnni = (e & (1u << 11)) != 0;
    c.avx512vpopcntdq = (e & (1u << 14)) != 0;
    c.amx_bf16 = (d & (1u << 22)) != 0;
    c.amx_tile = (d & (1u << 24)) != 0;
    c.amx_int8 = (d & (1u << 25)) != 0;
  }
  bool extended = false;
  if (__get_cpuid_max(0x80000000, nullptr) >= 0x80000001) {
    __cpuid(0x80000001, a, b, e, d);
    extended = (e & ((1u << 0) | (1u << 5))) == ((1u << 0) | (1u << 5));
  }
  c.x86_64_v3 = c.usable_avx2() && base && bmi && extended;
  constexpr uint64_t tile_state = (uint64_t{1} << 17) | (uint64_t{1} << 18);
  c.os_amx_state = (c.xcr0 & tile_state) == tile_state;
#if defined(__linux__) && defined(SYS_arch_prctl)
  // ARCH_GET_XCOMP_PERM is read-only. ARCH_REQ_XCOMP_PERM is intentionally
  // absent: enabling AMX is an explicit future backend responsibility.
  constexpr unsigned long get_xcomp_perm = 0x1022;
  unsigned long permissions = 0;
  if (syscall(SYS_arch_prctl, get_xcomp_perm, &permissions) == 0) {
    c.amx_permission = (permissions & (1ul << 18)) != 0;
  }
#endif
#endif
  return c;
}

const char* isa_name(CpuIsa isa) noexcept {
  switch (isa) {
    case CpuIsa::Auto: return "auto";
    case CpuIsa::Scalar: return "scalar";
    case CpuIsa::AVX2: return "avx2";
    case CpuIsa::AVX512: return "avx512";
    case CpuIsa::AVX512VNNI: return "avx512_vnni";
    case CpuIsa::AMX: return "amx";
  }
  return "unknown";
}

CpuIsa parse_isa(const std::string& value) {
  if (value == "auto") return CpuIsa::Auto;
  if (value == "scalar") return CpuIsa::Scalar;
  if (value == "avx2") return CpuIsa::AVX2;
  if (value == "avx512") return CpuIsa::AVX512;
  if (value == "avx512_vnni" || value == "vnni") return CpuIsa::AVX512VNNI;
  if (value == "amx") return CpuIsa::AMX;
  throw std::invalid_argument("Unknown CPU ISA: " + value);
}

CpuIsa select_isa(CpuIsa requested, std::span<const CpuIsa> supported,
                  const CpuCapabilities& caps) {
  const auto usable = [&](CpuIsa isa) {
    if (std::find(supported.begin(), supported.end(), isa) == supported.end()) return false;
    switch (isa) {
      case CpuIsa::Scalar: return true;
      case CpuIsa::AVX2: return caps.usable_avx2();
      case CpuIsa::AVX512: return caps.usable_avx512();
      case CpuIsa::AVX512VNNI: return caps.usable_vnni();
      case CpuIsa::AMX: return caps.usable_amx_int8();
      default: return false;
    }
  };
  if (requested == CpuIsa::Auto) {
    for (auto isa : {CpuIsa::AMX, CpuIsa::AVX512VNNI, CpuIsa::AVX512, CpuIsa::AVX2, CpuIsa::Scalar})
      if (usable(isa)) return isa;
  } else if (usable(requested)) {
    return requested;
  }
  throw std::invalid_argument(std::string("CPU backend is not implemented or usable: ") +
                              isa_name(requested));
}

}  // namespace mvsic::kernels
