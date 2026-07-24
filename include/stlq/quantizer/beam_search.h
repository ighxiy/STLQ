#pragma once

#include "stlq/core/kernel_provider.h"
#include "stlq/common/types.h"

namespace stlq {

struct Precomp;
struct PrecompLargeRoot;

struct BeamSearchLargeRootTiming {
    double norm_x2 = 0.0;
    double root_gemm = 0.0;
    double root_update = 0.0;
    double expand = 0.0;
};

void BeamSearchPrefixLS(const ColMajorMatrix<float>& X,
                        const Precomp& precomp,
                        const ColMajorMatrix<float>& xC,
                        int H_beam,
                        ColMajorMatrix<FullCode>* B);

// Large-root variant: root layer (layer0) is returned as `cluster_id` (uint32),
// while layers 1..m-1 are returned as `B_small` ((m-1)×n, Code=uint8).
void BeamSearchPrefixLSLargeRoot(const ColMajorMatrix<float>& X,
                                 const PrecompLargeRoot& pre,
                                 const ColMajorMatrix<float>& xC_small,
                                 int H_beam,
                                 std::vector<std::uint32_t>* cluster_id,
                                 ColMajorMatrix<Code>* B_small,
                                 BeamSearchLargeRootTiming* timing = nullptr);

// Internal helper: expand layers 1..m-1 using precomputed root candidates (per-sample top-H for layer0).
// This is shared by the CPU and CUDA large-root implementations to keep identical beam semantics.
void BeamSearchPrefixLSLargeRootExpandFromRootCandidates(
    const PrecompLargeRoot& pre,
    const ColMajorMatrix<float>& xC_small,
    int q0,
    int qlen,
    int H_beam,
    const float* root_errs,   // [qlen, H_beam]
    const int* root_codes,    // [qlen, H_beam]
    const float* root_a0,     // [qlen, H_beam]
    const float* root_r00,    // [qlen, H_beam]
    std::vector<std::uint32_t>* cluster_id,
    ColMajorMatrix<Code>* B_small);

#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
// Fast-path: use CUDA for the root-layer beam init (C0'X + top-H candidates), then finish the
// remaining layers on CPU to preserve exact beam semantics.
//
// Notes:
// - Intended for large-root mode only.
// - Currently optimized for small H_beam (1 or 2). For larger H_beam, this falls back to CPU.
void BeamSearchPrefixLSLargeRootCuda(const ColMajorMatrix<float>& X,
                                     const PrecompLargeRoot& pre,
                                     const ColMajorMatrix<float>& xC_small,
                                     int H_beam,
                                     bool allow_tf32,
                                     std::vector<std::uint32_t>* cluster_id,
                                     ColMajorMatrix<Code>* B_small,
                                     BeamSearchLargeRootTiming* timing = nullptr);

// Fast-path: fully CUDA for H_beam=2 in large-root mode (root init + expand layers 1..m-1).
// Falls back to the existing CPU expand for other H_beam values.
void BeamSearchPrefixLSLargeRootCudaH2(const ColMajorMatrix<float>& X,
                                       const PrecompLargeRoot& pre,
                                       const ColMajorMatrix<float>& xC_small,
                                       bool allow_tf32,
                                       std::vector<std::uint32_t>* cluster_id,
                                       ColMajorMatrix<Code>* B_small,
                                       BeamSearchLargeRootTiming* timing = nullptr);

// Fast-path: fully CUDA for H_beam=4 in large-root mode (root init + expand layers 1..m-1).
// Falls back to the existing CPU expand for other H_beam values.
void BeamSearchPrefixLSLargeRootCudaH4(const ColMajorMatrix<float>& X,
                                       const PrecompLargeRoot& pre,
                                       const ColMajorMatrix<float>& xC_small,
                                       bool allow_tf32,
                                       std::vector<std::uint32_t>* cluster_id,
                                       ColMajorMatrix<Code>* B_small,
                                       BeamSearchLargeRootTiming* timing = nullptr);

// Device-xC variants: avoid re-uploading xC_small when it was produced by a GPU GEMM.
void BeamSearchPrefixLSLargeRootCudaH2DeviceXc(const ColMajorMatrix<float>& X,
                                               const PrecompLargeRoot& pre,
                                               const DeviceMatF32View& xC_small_dev,
                                               bool allow_tf32,
                                               std::vector<std::uint32_t>* cluster_id,
                                               ColMajorMatrix<Code>* B_small,
                                               BeamSearchLargeRootTiming* timing = nullptr);

void BeamSearchPrefixLSLargeRootCudaH4DeviceXc(const ColMajorMatrix<float>& X,
                                               const PrecompLargeRoot& pre,
                                               const DeviceMatF32View& xC_small_dev,
                                               bool allow_tf32,
                                               std::vector<std::uint32_t>* cluster_id,
                                               ColMajorMatrix<Code>* B_small,
                                               BeamSearchLargeRootTiming* timing = nullptr);

// Variants that also return the best beam error per sample (same semantics as the internal
// beam error used for selection, after prefix-LS). This enables computing beam MSE without
// downloading xC_small to host.
void BeamSearchPrefixLSLargeRootCudaH2DeviceXcWithBestErr(const ColMajorMatrix<float>& X,
                                                          const PrecompLargeRoot& pre,
                                                          const DeviceMatF32View& xC_small_dev,
                                                          bool allow_tf32,
                                                          std::vector<std::uint32_t>* cluster_id,
                                                          ColMajorMatrix<Code>* B_small,
                                                          std::vector<float>* best_err,
                                                          BeamSearchLargeRootTiming* timing = nullptr);

void BeamSearchPrefixLSLargeRootCudaH4DeviceXcWithBestErr(const ColMajorMatrix<float>& X,
                                                          const PrecompLargeRoot& pre,
                                                          const DeviceMatF32View& xC_small_dev,
                                                          bool allow_tf32,
                                                          std::vector<std::uint32_t>* cluster_id,
                                                          ColMajorMatrix<Code>* B_small,
                                                          std::vector<float>* best_err,
                                                          BeamSearchLargeRootTiming* timing = nullptr);

// Device-X variants: avoid uploading X again when it is already cached on device (e.g. from a prior GPU GEMM).
void BeamSearchPrefixLSLargeRootCudaH2DeviceXcDeviceX(const DeviceMatF32View& X_dev,
                                                      const PrecompLargeRoot& pre,
                                                      const DeviceMatF32View& xC_small_dev,
                                                      bool allow_tf32,
                                                      std::vector<std::uint32_t>* cluster_id,
                                                      ColMajorMatrix<Code>* B_small,
                                                      BeamSearchLargeRootTiming* timing = nullptr);

void BeamSearchPrefixLSLargeRootCudaH2DeviceXcDeviceXWithBestErr(const DeviceMatF32View& X_dev,
                                                                 const PrecompLargeRoot& pre,
                                                                 const DeviceMatF32View& xC_small_dev,
                                                                 bool allow_tf32,
                                                                 std::vector<std::uint32_t>* cluster_id,
                                                                 ColMajorMatrix<Code>* B_small,
                                                                 std::vector<float>* best_err,
                                                                 BeamSearchLargeRootTiming* timing = nullptr);

void BeamSearchPrefixLSLargeRootCudaH4DeviceXcDeviceX(const DeviceMatF32View& X_dev,
                                                      const PrecompLargeRoot& pre,
                                                      const DeviceMatF32View& xC_small_dev,
                                                      bool allow_tf32,
                                                      std::vector<std::uint32_t>* cluster_id,
                                                      ColMajorMatrix<Code>* B_small,
                                                      BeamSearchLargeRootTiming* timing = nullptr);

void BeamSearchPrefixLSLargeRootCudaH4DeviceXcDeviceXWithBestErr(const DeviceMatF32View& X_dev,
                                                                 const PrecompLargeRoot& pre,
                                                                 const DeviceMatF32View& xC_small_dev,
                                                                 bool allow_tf32,
                                                                 std::vector<std::uint32_t>* cluster_id,
                                                                 ColMajorMatrix<Code>* B_small,
                                                                 std::vector<float>* best_err,
                                                                 BeamSearchLargeRootTiming* timing = nullptr);
#endif

}  // namespace stlq
