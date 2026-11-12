#pragma once

#include <cstdint>
#include <span>
#include <string>

namespace mvsic::kernels {

enum class CpuIsa { Auto, Scalar, AVX2, AVX512, AVX512VNNI, AMX };

// Hardware flags and OS state are deliberately separate: CPUID alone does
// not make AVX-512 or AMX safe. Detection never requests AMX permission.
struct CpuCapabilities {
  bool x86 = false;
  bool os_avx = false;
  bool os_avx512 = false;
  bool x86_64_v3 = false;
  bool avx2 = false;
  bool avx512f = false;
  bool avx512bw = false;
  bool avx512dq = false;
  bool avx512vl = false;
  bool avx512vnni = false;
  bool avx512vbmi = false;
  bool avx512vpopcntdq = false;
  bool amx_tile = false;
  bool amx_int8 = false;
  bool amx_bf16 = false;
  bool os_amx_state = false;
  bool amx_permission = false;
  uint64_t xcr0 = 0;

  bool usable_avx2() const noexcept { return os_avx && avx2; }
  bool usable_avx512() const noexcept {
    return os_avx512 && avx512f && avx512bw && avx512dq && avx512vl;
  }
  bool usable_vnni() const noexcept { return usable_avx512() && avx512vnni; }
  bool usable_amx_int8() const noexcept {
    return amx_tile && amx_int8 && os_amx_state && amx_permission;
  }
};

CpuCapabilities cpu_capabilities() noexcept;
const char* isa_name(CpuIsa isa) noexcept;
CpuIsa parse_isa(const std::string& value);

// Methods supply the backends they actually compile. Hardware/OS selection is
// shared; the factory and any minimum package ISA remain method responsibilities.
CpuIsa select_isa(CpuIsa requested, std::span<const CpuIsa> supported,
                  const CpuCapabilities& caps = cpu_capabilities());

}  // namespace mvsic::kernels
