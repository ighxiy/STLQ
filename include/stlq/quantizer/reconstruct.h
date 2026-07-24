#pragma once

#include <vector>

#include "stlq/common/types.h"

namespace stlq {

void ReconstructAllInto(const std::vector<ColMajorMatrix<float>>& C,
                        const ColMajorMatrix<FullCode>& B,
                        const ColMajorMatrix<float>& a,
                        ColMajorMatrix<float>* out);

// Reconstruct all vectors: out[:,i] = sum_l a[l,i] * C[l][:, B[l,i]].
ColMajorMatrix<float> ReconstructAll(const std::vector<ColMajorMatrix<float>>& C,
                                     const ColMajorMatrix<FullCode>& B,
                                     const ColMajorMatrix<float>& a);

}  // namespace stlq
