#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "stlq/common/types.h"

namespace stlq {

struct PrecompLargeRoot;

// Batch ICM + optional ILS wrapper for dynamic_icm_with_ils_noabs_nonormal!
// (encode_noabs.jl:8-112, train_quantizer.jl:177-182).
void DynamicIcmWithIlsNoAbsNoNormal(const ColMajorMatrix<float>& X,
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
                                   std::uint64_t sample_id_offset = 0);

// Batch ICM + optional ILS wrapper for dynamic_icm_with_ils_abs_nonormal!
// (encode_abs.jl:379-512).
void DynamicIcmWithIlsAbsNoNormal(const ColMajorMatrix<float>& X,
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
                                 std::uint64_t sample_id_offset = 0);

// Large-root variant: root codes are stored in `cluster_id` (uint32), and
// small-layer codes are stored in `B_small` ((m-1)×n).
void DynamicIcmWithIlsNoAbsNoNormalLargeRoot(const ColMajorMatrix<float>& X,
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
                                            std::uint64_t sample_id_offset = 0);

void DynamicIcmWithIlsAbsNoNormalLargeRoot(const ColMajorMatrix<float>& X,
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
                                          std::uint64_t sample_id_offset = 0);

}  // namespace stlq
