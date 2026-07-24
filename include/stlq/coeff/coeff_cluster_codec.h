#pragma once

#include <cstdint>
#include <vector>

#include "stlq/coeff/coeff_cluster_quantizer.h"
#include "stlq/coeff/huffman_canonical.h"
#include "stlq/coeff/icmils_hook.h"
#include "stlq/coeff/span.h"

namespace stlq {

enum class HuffmanGranularity {
  // Build one Huffman model per (cluster, group) and encode all layers into a single stream.
  kPerClusterAllLayers = 0,
  // Build one Huffman model per (cluster, group, layer).
  kPerLayer = 1,
};

struct HuffmanConfig {
  HuffmanGranularity granularity = HuffmanGranularity::kPerClusterAllLayers;

  // Fixed alphabet range for int8 symbols.
  // If set to <=0, we use alphabet [-Qmax, Qmax] where Qmax is max over bits_per_layer.
  // If set to 256, we use full int8 range [-128,127].
  int alphabet_size = 0;
};

// Compressed payload for one group (root or linkage) in a single cluster.
struct ClusterGroupCompressed {
  // Huffman models. Size is 1 (all-layers) or m (per-layer).
  std::vector<HuffmanCanonicalModel> models;
  // Serialized models. Same count as models; included so you can store without rebuilding.
  std::vector<std::vector<std::uint8_t>> models_serialized;
  // Payload bitstreams. Same count as models.
  std::vector<std::vector<std::uint8_t>> payloads;

  // Number of symbols encoded per stream, for decoding.
  // For all-layers: N = n_group * m
  // For per-layer:  N[l] = n_group
  std::vector<std::uint32_t> num_symbols;
};

struct ClusterCoeffCompressed {
  std::uint32_t cluster_id = 0;
  std::uint32_t n_in_cluster = 0;
  std::uint32_t n_root = 0;
  std::uint32_t n_linkage = 0;

  // Per-layer scales to store alongside the compressed coeffs.
  std::vector<float> scale_root;   // size m
  std::vector<float> scale_linkage;  // size m

  ClusterGroupCompressed root;
  ClusterGroupCompressed linkage;
};

// End-to-end: quantize (lossy) + Huffman compress (lossless) for one cluster.
//
// Inputs are CLUSTER-LOCAL and layer-major.
//
// `q_tmp` is a scratch buffer (int8) of size m*nc to avoid reallocation.
ClusterCoeffCompressed QuantizeAndCompressClusterCoeffs(
    std::uint32_t cluster_id,
    int m,
    int nc,
    Span<const float> a,
    Span<const std::uint32_t> B,
    Span<const std::uint8_t> is_linkage,
    const std::vector<Span<const float>>& norm2_root_layers,
    const std::vector<Span<const float>>& norm2_linkage_layers,
    const CoeffQuantConfig& qcfg,
    const HuffmanConfig& hcfg,
    Span<std::int8_t> q_tmp,
    IcmIlsRefiner* refiner = nullptr);

// Huffman compress already-quantized coefficients `q` (int8, layer-major) using the same
// root/linkage split as `QuantizeAndCompressClusterCoeffs`, and attach caller-provided scales.
//
// This is used when q/scales are produced by an external refinement step (e.g., ICM).
ClusterCoeffCompressed CompressClusterCoeffsFromQ(
    std::uint32_t cluster_id,
    int m,
    int nc,
    Span<const std::int8_t> q,
    Span<const std::uint8_t> is_linkage,
    const std::vector<float>& scale_root,
    const std::vector<float>& scale_linkage,
    const CoeffQuantConfig& qcfg,
    const HuffmanConfig& hcfg);

// Decode helper. Reconstruct q into `q_out` (int8, size m*nc).
// Caller must supply same `is_linkage` ordering and configs used for encoding.
void DecodeClusterCoeffs(
    const ClusterCoeffCompressed& comp,
    int m,
    int nc,
    Span<const std::uint8_t> is_linkage,
    Span<std::int8_t> q_out);

}  // namespace stlq
