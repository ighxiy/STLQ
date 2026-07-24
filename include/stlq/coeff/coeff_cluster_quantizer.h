#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "stlq/coeff/span.h"

namespace stlq {

// Quantization config mirroring key knobs in scale.jl (but adapted to streaming per-cluster).
struct CoeffQuantConfig {
  // bits per layer (signed), size m. Each bits>=2.
  std::vector<int> bits_per_layer;

  // Percentile candidates for layer0 vs the rest.
  std::vector<double> p_first_candidates{99.5, 99.8, 100.0};
  std::vector<double> p_rest_candidates{99.5, 99.8, 100.0};

  // Optionally override candidates by generating a dense range.
  bool auto_search_p = false;
  // (min, max, step)
  std::tuple<double, double, double> p_first_range{99.0, 100.0, 0.1};
  std::tuple<double, double, double> p_rest_range{99.0, 100.0, 0.1};

  // Use weighted quantile / weighted SSE. Weight is typically ||c||^2 for the selected code.
  bool use_weighted_quantile = false;

  // If false, disallow p that would clip values whose |round(a)| exceeds chosen Amax.
  bool allow_clip = true;

  // If true, fit an additional per-layer scale s via least squares a ≈ s*q.
  // This is the representation you asked for: store q (int8) + s (float).
  bool fit_scale = true;
};

// Per-layer quantization outcome.
struct CoeffQuantLayerStats {
  double p_used = 100.0;
  double clip_ratio = 0.0;
  float step_delta = 1.0f;  // Δ = Amax/Q
  float scale = 1.0f;       // s (if fit_scale), else Δ
};

// Quantization result for one cluster and one group (root or linkage).
struct CoeffQuantGroupResult {
  // scales.size() == m
  std::vector<float> scales;
  // stats.size() == m
  std::vector<CoeffQuantLayerStats> stats;
};

// Quantize coefficients for a single cluster, optionally split into root/linkage groups.
//
// Inputs are CLUSTER-LOCAL and layer-major:
// - a[layer*nc + i] : float coefficient for layer and in-cluster index i
// - B[layer*nc + i] : 1-based code index (optional, only used if weighted)
// - is_linkage[i]     : 0=root, 1=linkage
//
// Optional weights:
// - norm2_root[layer][code] and norm2_linkage[layer][code] are 1-based norm^2 tables.
//   Pass empty spans to disable weighting.
//
// Outputs:
// - q_out[layer*nc + i] is quantized signed integer (stored in int8) with range given by bits_per_layer[layer].
// - result_root / result_linkage contain per-layer scales to store with the cluster.
void QuantizeClusterCoeffsSplitRootLinkage(
    int m,
    int nc,
    Span<const float> a,
    Span<const std::uint32_t> B,
    Span<const std::uint8_t> is_linkage,
    const std::vector<Span<const float>>& norm2_root_layers,
    const std::vector<Span<const float>>& norm2_linkage_layers,
    const CoeffQuantConfig& cfg,
    Span<std::int8_t> q_out,
    CoeffQuantGroupResult* result_root,
    CoeffQuantGroupResult* result_linkage);

// Helper: make bits_per_layer = [bits_first; fill(bits_rest, m-1)]
std::vector<int> MakeBitsPerLayer(int m, int bits_first, int bits_rest);

}  // namespace stlq

