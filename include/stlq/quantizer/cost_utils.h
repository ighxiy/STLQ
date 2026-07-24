#pragma once

#include <cstdint>
#include <vector>

#include "stlq/common/types.h"

namespace stlq {

struct PrecompLargeRoot;

void ComputeXNorm2(const ColMajorMatrix<float>& X, std::vector<float>* X_norm2);

float SampleCost(const Precomp& precomp,
                 const ColMajorMatrix<float>& xC,
                 const ColMajorMatrix<FullCode>& B,
                 const ColMajorMatrix<float>& a,
                 int sample,
                 float x_norm2);

void ComputeCosts(const Precomp& precomp,
                  const ColMajorMatrix<float>& xC,
                  const ColMajorMatrix<FullCode>& B,
                  const ColMajorMatrix<float>& a,
                  const std::vector<float>& X_norm2,
                  std::vector<float>* out_cost);

float SampleCostLargeRoot(const PrecompLargeRoot& pre,
                          const ColMajorMatrix<float>& X,
                          const ColMajorMatrix<float>& xC_small,
                          const std::vector<std::uint32_t>& cluster_id,
                          const ColMajorMatrix<Code>& B_small,
                          const ColMajorMatrix<float>& a,
                          int sample,
                          float x_norm2);

void ComputeCostsLargeRoot(const PrecompLargeRoot& pre,
                           const ColMajorMatrix<float>& X,
                           const ColMajorMatrix<float>& xC_small,
                           const std::vector<std::uint32_t>& cluster_id,
                           const ColMajorMatrix<Code>& B_small,
                           const ColMajorMatrix<float>& a,
                           const std::vector<float>& X_norm2,
                           std::vector<float>* out_cost);

}  // namespace stlq
