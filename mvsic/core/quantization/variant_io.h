#pragma once

// =============================================================================
// Helpers to save / load `std::variant<...>`-based quantizer models and
// encoded single-vector ranges used by MUVERA / MPool / SVH_Graph / SVH_IVF.
//
// These families still carry a runtime variant because their quantizer Models
// (non-`_mv` variants) do not expose the uniform `kClassId` / `EncodedSet` /
// `Params` surface that MVIVF and Vamana use.  The helpers below centralise
// the `switch(QuantizerType)` dispatch used by each family's
// `save_with_quantizer` / `load_with_quantizer`.
//
// The QT enum layout is assumed to match the one in `IndexParams::QuantizerType`:
//   None = 0, PQ = 1, RaBitQ = 2, FastScan = 3, TurboQuant = 4, SPQTQ = 5,
//   OneBitTQ = 6, EightBitTQ = 7.
//
// Neither helper writes the `QT` tag itself — the concrete family already
// encodes that information in its `class_id` header (via `kLeafMethod`), and
// on load the caller inspects `kLeafMethod` to decide which alternative to
// emplace.  This keeps the on-disk layout tight.
// =============================================================================

#include <iostream>
#include <type_traits>
#include <variant>

#include "mvsic/core/index_params.h"

namespace mvsic::variant_io {

// Saves the currently-held alternative of a variant by calling `x.save(out)`.
// Does nothing when the variant holds `std::monostate`.
template <class Variant>
inline void save(const Variant& v, std::ostream& out) {
  std::visit([&](auto const& x) {
    using T = std::decay_t<decltype(x)>;
    if constexpr (!std::is_same_v<T, std::monostate>) {
      x.save(out);
    }
  }, v);
}

// Emplaces the variant alternative matching `method` and loads it from `in`.
// `Types` is the pack of ordered alternatives that map to
// {PQ, RaBitQ, FastScan, TurboQuant, SPQTQ} (in that order).  Any other QT
// (None / OneBitTQ) resets the variant to monostate.
template <class Variant, class PQ, class RQ, class FS, class TQ, class PQTQ>
inline void load_sv(Variant& v, std::istream& in,
                    IndexParams::QuantizerType method) {
  using QT = IndexParams::QuantizerType;
  switch (method) {
    case QT::PQ:         { v.template emplace<PQ>();   std::get<PQ>(v).load(in);   break; }
    case QT::RaBitQ:     { v.template emplace<RQ>();   std::get<RQ>(v).load(in);   break; }
    case QT::FastScan:   { v.template emplace<FS>();   std::get<FS>(v).load(in);   break; }
    case QT::TurboQuant: { v.template emplace<TQ>();   std::get<TQ>(v).load(in);   break; }
    case QT::SPQTQ:      { v.template emplace<PQTQ>(); std::get<PQTQ>(v).load(in); break; }
    case QT::None:
    case QT::OneBitTQ:
    case QT::EightBitTQ:
    default: v = std::monostate{}; break;
  }
}

}  // namespace mvsic::variant_io
