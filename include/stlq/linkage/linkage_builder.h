#pragma once

#include <string>
#include <vector>

#include "stlq/common/config.h"
#include "stlq/core/kernels.h"
#include "stlq/common/types.h"

namespace stlq {

bool BuildLinkageOneInit(const LinkageBuildConfig& linkage_cfg,
                const HnswConfig& hnsw_cfg,
                const ColMajorMatrix<float>& X,
                BaseEncoding* base,
                const CodebookPack& C_root,
                LinkageStructure* linkage,
                ColMajorMatrix<float>* R_full,
                std::string* error);

bool BuildLinkageTwoCodebook(const LinkageBuildConfig& linkage_cfg,
                           const HnswConfig& hnsw_cfg,
                           const ColMajorMatrix<float>& X,
                           BaseEncoding* base,
                           const CodebookPack& C_root,
                           const CodebookPack& C_one,
                           LinkageStructure* linkage,
                           ColMajorMatrix<float>* R_full,
                           std::string* error);



std::vector<int> BuildGlobalParent(const LinkageStructure& linkage, int n_total);

// Virtual-mode variant: serializes global parent pointers for REAL nodes while allowing parents to be
// virtual roots. Virtual global ids are assigned by concatenating per-cluster virtual nodes in
// cluster-id order: g = n_base_real + prefix[cid] + v_local.
std::vector<int> BuildGlobalParentVirtual(const LinkageStructure& linkage,
                                         int n_base_real,
                                         const VirtualEncoding& virt);



}  // namespace stlq
