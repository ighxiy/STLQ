#pragma once

#include <vector>

#include "stlq/common/types.h"

namespace stlq {

struct Precomp;
struct PrecompLargeRoot;

bool SolveLeastSquaresSample(const Precomp& precomp,
                             const ColMajorMatrix<float>& xC,
                             const ColMajorMatrix<FullCode>& B,
                             int sample,
                             std::vector<float>* A,
                             std::vector<float>* b,
                             std::vector<float>* work,
                             float* a_out);

void SolveLeastSquaresAll(const Precomp& precomp,
                          const ColMajorMatrix<float>& xC,
                          const ColMajorMatrix<FullCode>& B,
                          ColMajorMatrix<float>* a);

// Large-root variant (layer0 code is stored separately as `cluster_id`):
// - `cluster_id[i]` is in [0, h0)
// - `B_small` is (m-1)×n for layers 1..m-1.
bool SolveLeastSquaresSampleLargeRoot(const PrecompLargeRoot& pre,
                                      const ColMajorMatrix<float>& X,
                                      const ColMajorMatrix<float>& xC_small,
                                      const std::vector<std::uint32_t>& cluster_id,
                                      const ColMajorMatrix<Code>& B_small,
                                      int sample,
                                      std::vector<float>* A,
                                      std::vector<float>* b,
                                      std::vector<float>* work,
                                      float* a_out);

// Same as SolveLeastSquaresSampleLargeRoot, but uses a precomputed b0 = dot(C0[:,cid], x).
// This is useful when `cluster_id` is fixed (e.g., basic encoding where layer0 codes are fixed)
// and we call the solver many times (ICM/ILS inner loops).
bool SolveLeastSquaresSampleLargeRootWithB0(const PrecompLargeRoot& pre,
                                            float b0,
                                            const ColMajorMatrix<float>& xC_small,
                                            const std::vector<std::uint32_t>& cluster_id,
                                            const ColMajorMatrix<Code>& B_small,
                                            int sample,
                                            std::vector<float>* A,
                                            std::vector<float>* b,
                                            std::vector<float>* work,
                                            float* a_out);

void SolveLeastSquaresAllLargeRoot(const PrecompLargeRoot& pre,
                                  const ColMajorMatrix<float>& X,
                                  const ColMajorMatrix<float>& xC_small,
                                  const std::vector<std::uint32_t>& cluster_id,
                                  const ColMajorMatrix<Code>& B_small,
                                  ColMajorMatrix<float>* a);

}  // namespace stlq
