// A narrow test module using the original registration code and conversion path.
// Link with ORIGINAL_ROOT/mvsic/python/register_common.cc. It is not a replacement
// baseline wheel and its compile time is not the legacy full-extension build cost.
// Only the supported IP retrieval specialization is instantiated here.
#include "mvsic/python/bind_macros.h"
#include "mvsic/mvivf/mvivf.h"

PYBIND11_MODULE(_mvsic_baseline, m) {
  register_common(m);
  BIND_INDEX(IndexMVIVFCompressOneBitTQIP, ChamferIP_Point, "IndexMVIVFCompressOneBitTQIP");
}
