#pragma once

#include <vector>

#include "stlq/common/types.h"

namespace stlq {

struct MseStats {
    float mean = 0.0f;
    float min = 0.0f;
    float max = 0.0f;
};

struct LinkageDepthStats {
    float linkage_ratio = 0.0f;
    int max_depth = 0;
    float mean_depth = 0.0f;
};

// Per-vector squared errors: out[i] = ||X[:,i] - R_full[:,i]||^2.
void SquaredErrorsFromRecon(const ColMajorMatrix<float>& X,
                            const ColMajorMatrix<float>& R_full,
                            std::vector<float>* out);

MseStats MseStatsFromRecon(const ColMajorMatrix<float>& X,
                           const ColMajorMatrix<float>& R_full);

float MeanMseFromRecon(const ColMajorMatrix<float>& X,
                       const ColMajorMatrix<float>& R_full);

float MeanMseFromCodebooks(const ColMajorMatrix<float>& X,
                           const std::vector<ColMajorMatrix<float>>& C,
                           const ColMajorMatrix<FullCode>& B,
                           const ColMajorMatrix<float>& a);

float MeanMseFromPrecomp(const ColMajorMatrix<float>& X,
                         const Precomp& precomp,
                         const ColMajorMatrix<float>& xC,
                         const ColMajorMatrix<FullCode>& B,
                         const ColMajorMatrix<float>& a);

LinkageDepthStats ComputeLinkageDepthStats(const LinkageStructure& linkage, int n_total);

// Converts a 1-based parent array (0 means root) into per-node depths.
// Mainly used when loading a flat parent array from disk.
std::vector<int> ComputeDepthsFromParent(const std::vector<int>& parent);

}  // namespace stlq
