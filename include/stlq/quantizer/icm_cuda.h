#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "stlq/core/kernel_provider.h"
#include "stlq/common/types.h"

namespace stlq {

struct PrecompLargeRoot;

struct CudaIcmTiming {
    // Host-visible device timings (seconds). Only filled when non-null timing is provided.
    double upload_fixed = 0.0;      // H2D uploads (xC/norms/cluster_id) for this batch
    double upload_B = 0.0;          // H2D upload of initial codes for this batch
    double icm_init = 0.0;          // initial RunIcmDevice (before ILS loop)
    double ils_copy_perturb = 0.0;  // copy+perturb (fused kernel)
    double ils_icm = 0.0;           // RunIcmDevice calls inside ILS loop
    double ils_accept = 0.0;        // AcceptIfBetter kernels
    double download = 0.0;          // D2H downloads (B/a/cost)
};

// CUDA-accelerated ILS+ICM for the basic encoding stage.
// Enabled automatically when runtime.use_cuda=true (and built with -DSTLQ_ENABLE_CUDA=ON).
//
// Notes:
// - Semantics follow icm.cpp, but numerics may differ slightly (GPU vs CPU).
// - Only used in the "basic" stage (training/basic streaming and baseset/basic streaming).
void DynamicIcmWithIlsNoAbsNoNormalCuda(const ColMajorMatrix<float>& X,
                                       const Precomp& precomp,
                                       const ColMajorMatrix<float>& xC,
                                       int icm_iters,
                                       int ils_iters,
                                       int perturb_k,
                                       std::uint32_t seed,
                                       ColMajorMatrix<FullCode>* B,
                                       ColMajorMatrix<float>* a,
                                       std::vector<float>* X_norm2,
                                       std::vector<float>* cost,
                                       bool print_progress = true,
                                       std::uint64_t sample_id_offset = 0,
                                       CudaIcmTiming* timing = nullptr);

void DynamicIcmWithIlsAbsNoNormalCuda(const ColMajorMatrix<float>& X,
                                     const Precomp& precomp,
                                     const ColMajorMatrix<float>& xC,
                                     int icm_iters,
                                     int ils_iters,
                                     int perturb_k,
                                     std::uint32_t seed,
                                     ColMajorMatrix<FullCode>* B,
                                     ColMajorMatrix<float>* a,
                                     std::vector<float>* X_norm2,
                                     std::vector<float>* cost,
                                     bool print_progress = true,
                                     std::uint64_t sample_id_offset = 0,
                                     CudaIcmTiming* timing = nullptr);

// Same as DynamicIcmWithIls*Cuda, but consumes a device view of xC to avoid re-uploading host xC.
// Intended for the base streaming full-precomp path where xC is already computed on GPU.
void DynamicIcmWithIlsNoAbsNoNormalCudaDeviceXc(const ColMajorMatrix<float>& X,
                                               const Precomp& precomp,
                                               const DeviceMatF32View& xC_dev,
                                               int icm_iters,
                                               int ils_iters,
                                               int perturb_k,
                                               std::uint32_t seed,
                                               ColMajorMatrix<FullCode>* B,
                                               ColMajorMatrix<float>* a,
                                               std::vector<float>* X_norm2,
                                               std::vector<float>* cost,
                                               bool print_progress = true,
                                               std::uint64_t sample_id_offset = 0,
                                               CudaIcmTiming* timing = nullptr);

void DynamicIcmWithIlsAbsNoNormalCudaDeviceXc(const ColMajorMatrix<float>& X,
                                             const Precomp& precomp,
                                             const DeviceMatF32View& xC_dev,
                                             int icm_iters,
                                             int ils_iters,
                                             int perturb_k,
                                             std::uint32_t seed,
                                             ColMajorMatrix<FullCode>* B,
                                             ColMajorMatrix<float>* a,
                                             std::vector<float>* X_norm2,
                                             std::vector<float>* cost,
                                             bool print_progress = true,
                                             std::uint64_t sample_id_offset = 0,
                                             CudaIcmTiming* timing = nullptr);

void DynamicIcmWithIlsNoAbsNoNormalLargeRootCuda(const ColMajorMatrix<float>& X,
                                                const PrecompLargeRoot& pre,
                                                const ColMajorMatrix<float>& xC_small,
                                                int icm_iters,
                                                int ils_iters,
                                                int perturb_k,
                                                std::uint32_t seed,
                                                const std::vector<std::uint32_t>& cluster_id,
                                                ColMajorMatrix<Code>* B_small,
                                                ColMajorMatrix<float>* a,
                                                std::vector<float>* X_norm2,
                                                std::vector<float>* cost,
                                                bool print_progress = true,
                                                std::uint64_t sample_id_offset = 0,
                                                CudaIcmTiming* timing = nullptr);

void DynamicIcmWithIlsNoAbsNoNormalLargeRootCudaDeviceXc(const ColMajorMatrix<float>& X,
                                                        const PrecompLargeRoot& pre,
                                                        const DeviceMatF32View& xC_small_dev,
                                                        int icm_iters,
                                                        int ils_iters,
                                                        int perturb_k,
                                                        std::uint32_t seed,
                                                        const std::vector<std::uint32_t>& cluster_id,
                                                        ColMajorMatrix<Code>* B_small,
                                                        ColMajorMatrix<float>* a,
                                                        std::vector<float>* X_norm2,
                                                        std::vector<float>* cost,
                                                        bool print_progress = true,
                                                        std::uint64_t sample_id_offset = 0,
                                                        CudaIcmTiming* timing = nullptr);

void DynamicIcmWithIlsAbsNoNormalLargeRootCuda(const ColMajorMatrix<float>& X,
                                              const PrecompLargeRoot& pre,
                                              const ColMajorMatrix<float>& xC_small,
                                              int icm_iters,
                                              int ils_iters,
                                              int perturb_k,
                                              std::uint32_t seed,
                                              const std::vector<std::uint32_t>& cluster_id,
                                              ColMajorMatrix<Code>* B_small,
                                              ColMajorMatrix<float>* a,
                                              std::vector<float>* X_norm2,
                                              std::vector<float>* cost,
                                              bool print_progress = true,
                                              std::uint64_t sample_id_offset = 0,
                                              CudaIcmTiming* timing = nullptr);

void DynamicIcmWithIlsAbsNoNormalLargeRootCudaDeviceXc(const ColMajorMatrix<float>& X,
                                                      const PrecompLargeRoot& pre,
                                                      const DeviceMatF32View& xC_small_dev,
                                                      int icm_iters,
                                                      int ils_iters,
                                                      int perturb_k,
                                                      std::uint32_t seed,
                                                      const std::vector<std::uint32_t>& cluster_id,
                                                      ColMajorMatrix<Code>* B_small,
                                                      ColMajorMatrix<float>* a,
                                                      std::vector<float>* X_norm2,
                                                      std::vector<float>* cost,
                                                      bool print_progress = true,
                                                      std::uint64_t sample_id_offset = 0,
                                                      CudaIcmTiming* timing = nullptr);

// Same as DynamicIcmWithIls*LargeRootCudaDeviceXc, but computes xC0 (dot(C0[:,cid], X[:,i]))
// on the GPU from a device view of X (typically the cached rotated tile).
void DynamicIcmWithIlsNoAbsNoNormalLargeRootCudaDeviceXcDeviceX(const DeviceMatF32View& X_dev,
                                                               const PrecompLargeRoot& pre,
                                                               const DeviceMatF32View& xC_small_dev,
                                                               int icm_iters,
                                                               int ils_iters,
                                                               int perturb_k,
                                                               std::uint32_t seed,
                                                               const std::vector<std::uint32_t>& cluster_id,
                                                               ColMajorMatrix<Code>* B_small,
                                                               ColMajorMatrix<float>* a,
                                                               std::vector<float>* X_norm2,
                                                               std::vector<float>* cost,
                                                               bool print_progress = true,
                                                               std::uint64_t sample_id_offset = 0,
                                                               CudaIcmTiming* timing = nullptr);

void DynamicIcmWithIlsAbsNoNormalLargeRootCudaDeviceXcDeviceX(const DeviceMatF32View& X_dev,
                                                             const PrecompLargeRoot& pre,
                                                             const DeviceMatF32View& xC_small_dev,
                                                             int icm_iters,
                                                             int ils_iters,
                                                             int perturb_k,
                                                             std::uint32_t seed,
                                                             const std::vector<std::uint32_t>& cluster_id,
                                                             ColMajorMatrix<Code>* B_small,
                                                             ColMajorMatrix<float>* a,
                                                             std::vector<float>* X_norm2,
                                                             std::vector<float>* cost,
                                                             bool print_progress = true,
                                                             std::uint64_t sample_id_offset = 0,
                                                             CudaIcmTiming* timing = nullptr);

}  // namespace stlq
