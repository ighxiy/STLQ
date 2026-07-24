#pragma once

#include <string>
#include <vector>

#include "stlq/common/config.h"
#include "stlq/common/types.h"
#include "stlq/linkage/virtual_augment.h"

namespace stlq {

// Virtual-only linkage builder variants.
// These are kept separate so baseline (non-virtual) code never depends on virtual helpers.

// Build a two-codebook linkage only for selected IVF clusters (cluster-local semantics).
// - `cols_by_cluster[cid]` contains global column ids belonging to that cluster (0..h0-1).
// - `B/a` are updated in-place for processed clusters.
// - `R_full` is returned for all n columns, where unprocessed clusters remain at the base reconstruction.
    bool ProcessGoodClustersVirtual(const LinkageBuildConfig& linkage_cfg,
                                          const HnswConfig& hnsw_cfg,
                                          const ColMajorMatrix<float>& X,
                                          std::vector<std::vector<int>> cols_by_cluster,
                                          ColMajorMatrix<FullCode>* B,
                                          ColMajorMatrix<float>* a,
                                          const CodebookPack& C_root,
                                          const CodebookPack& C_one,
                                          LinkageStructure* linkage,
                                          ColMajorMatrix<float>* R_full,
                                          std::string* error);


    bool ProcessBadClustersVirtualForTrain(const Config& config,
                                   const ColMajorMatrix<float>& X_rot,
                                   const std::vector<std::vector<int>>& bad_cols_by_cid,
                                   const std::vector<int>& bad_cids,
                                   const LinkageBuildConfig& linkage_cfg,
                                   const CodebookPack& C_root,
                                   const CodebookPack& C_one,
                                   const Precomp& pre_root,
                                   const Precomp& pre_one,
                                   const ColMajorMatrix<float>& G_one_root,
                                   ColMajorMatrix<FullCode>* B,
                                   ColMajorMatrix<float>* a,
                                   ColMajorMatrix<float>* R_full_shared,  // shared reconstruction buffer (d x n)
                                   std::vector<BadClusterCache>* bad_cache,
                                   std::string* error);

// Build a two-codebook linkage for base encoding in virtual mode (good/bad cluster split).
//
// Semantics (Julia reference: main_virtual.jl + linkage_quantizer.jl virtual strategy):
// - Good clusters: regular inner-to-outer two-codebook linkageing on REAL points only.
// - Bad clusters: multi-center linkageing; virtual roots are used ONLY as parent candidates (depth==0)
//   while REAL points are processed/updated. Virtual nodes must not be processed as children.
//
// `n_real` is the number of real base points (columns [0..n_real)); columns [n_real..n) are
// virtual nodes are provided separately in `virt` (see agents2.md / PROJECT_OVERVIEW_AND_PROMPT.md).
    bool BuildLinkageTwoCodebookVirtual(const LinkageBuildConfig& linkage_cfg,
                                 const HnswConfig& hnsw_cfg,
                                 const VirtualConfig& vcfg,
                                 const ColMajorMatrix<float>& X_real,
                                 BaseEncoding* base_real,
                                 const VirtualEncoding& virt,
                                 const CodebookPack& C_root,
                                 const CodebookPack& C_one,
                                 const std::vector<bool>& is_bad_cluster,
                                 const BadClusterKnnCache* bad_knn_cache,
                                 LinkageStructure* linkage,
                                 ColMajorMatrix<float>* R_full,
                                 std::string* error);




}  // namespace stlq
