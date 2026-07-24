#pragma once

#include <cstdint>

#include "stlq/coeff/span.h"

namespace stlq {

// Optional refinement hook for the "version 2" lossy coefficient quantization.
// Not used in the current integration; kept for future extension.
struct IcmIlsRefiner {
  virtual ~IcmIlsRefiner() = default;

  // `q` is int8 with length m*nc (layer-major).
  // scales_root/scales_linkage are length m.
  virtual void RefineCluster(int cluster_id,
                             int m,
                             int nc,
                             Span<const std::uint8_t> is_linkage,
                             Span<std::int8_t> q,
                             Span<float> scales_root,
                             Span<float> scales_linkage) = 0;
};

struct NoopIcmIlsRefiner final : IcmIlsRefiner {
  void RefineCluster(int, int, int, Span<const std::uint8_t>, Span<std::int8_t>, Span<float>, Span<float>) override {}
};

}  // namespace stlq

