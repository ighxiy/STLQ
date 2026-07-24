#pragma once

#include <string>

#include "stlq/common/config.h"
#include "stlq/common/types.h"

namespace stlq {

struct RecallResult {
    float recall = 0.0f;
    ColMajorMatrix<int> indices;
    ColMajorMatrix<float> dists;
};

bool EvaluateRecallBase(const Config& config,
                        const Dataset& dataset,
                        const CodebookPack& codebooks,
                        const BaseEncoding& base,
                        RecallResult* result,
                        std::string* error);

bool EvaluateRecallBaseIvf(const Config& config,
                           const Dataset& dataset,
                           const CodebookPack& codebooks,
                           const BaseEncoding& base,
                           RecallResult* result,
                           std::string* error);

bool EvaluateRecallLinkage(const Config& config,
                         const Dataset& dataset,
                         const CodebookPack& C_root,
                         const CodebookPack& C_one,
                         const BaseEncoding& base,
                         const LinkageStructure& linkage,
                         int n_base_real,
                         RecallResult* result,
                         std::string* error);

// Virtual-mode linkage recall:
// - real base encoding lives in `base_real` (size n_base_real)
// - virtual nodes encoding is in `virt` (global ids map by g - n_base_real)
// - `linkage` may contain both real and virtual ids; virtual ids are ignored in top-k outputs.
bool EvaluateRecallLinkageVirtual(const Config& config,
                                const Dataset& dataset,
                                const CodebookPack& C_root,
                                const CodebookPack& C_one,
                                const BaseEncoding& base_real,
                                const VirtualEncoding& virt,
                                const LinkageStructure& linkage,
                                int n_base_real,
                                RecallResult* result,
                                std::string* error);

bool EvaluateRecallLinkageVirtualFromParentWithClusterId(const Config& config,
                                                       const Dataset& dataset,
                                                       const CodebookPack& C_root,
                                                       const CodebookPack& C_one,
                                                       const BaseEncoding& base_real,
                                                       const VirtualEncoding& virt,
                                                       const std::vector<int>& parent,
                                                       const std::vector<int>& cluster_id,
                                                       int n_base_real,
                                                       RecallResult* result,
                                                       std::string* error);

bool EvaluateRecallLinkageFromParent(const Config& config,
                                   const Dataset& dataset,
                                   const CodebookPack& C_root,
                                   const CodebookPack& C_one,
                                   const BaseEncoding& base,
                                   const std::vector<int>& parent,
                                   int n_base_real,
                                   RecallResult* result,
                                   std::string* error);

// Same as EvaluateRecallLinkageFromParent, but uses the stored per-vector cluster ids
// (from HDF5) instead of inferring cluster membership from B(0,*) and parent linkages.
bool EvaluateRecallLinkageFromParentWithClusterId(const Config& config,
                                                const Dataset& dataset,
                                                const CodebookPack& C_root,
                                                const CodebookPack& C_one,
                                                const BaseEncoding& base,
                                                const std::vector<int>& parent,
                                                const std::vector<int>& cluster_id,
                                                int n_base_real,
                                                RecallResult* result,
                                                std::string* error);

}  // namespace stlq
