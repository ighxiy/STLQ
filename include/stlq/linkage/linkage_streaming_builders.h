#pragma once

#include <cstdint>
#include <string>

#include "stlq/linkage/linkage_summary.h"
#include "stlq/common/config.h"
#include "stlq/io/linkage_list_store.h"
#include "stlq/common/types.h"

namespace stlq {

class StreamKernelProvider;

struct VirtualUmapReencodeEncodeConfig {
    int ils_iters = 0;
    int icm_iters = 0;
    int perturb_k = 0;
    std::uint32_t seed = 0;
};

namespace io {
struct DatasetVectorReader;
class BaseListReader;
class IvfListsReader;
class LinkageListReader;
}  // namespace io

// Build virtual-mode linkage results cluster-by-cluster from:
// - IVF CSR lists (offsets+ids)
// - base_list store (codes_small + coeffs, list-order contiguous)
// - optional base_list raw_u8.bin (raw vectors aligned with list-order)
//
// Outputs are written via LinkageListWriter (per-cluster, depth-order contiguous).
bool BuildLinkageTwoCodebookVirtualStreamingByCluster(const Config& cfg,
                                                    const TrainResult& train,
                                                    const VirtualUmapReencodeEncodeConfig& umap_reencode_cfg,
                                                    const io::BaseListReader& base_list,
                                                    const io::IvfListsReader& ivf_lists,
                                                    const io::DatasetVectorReader* fallback_reader,
                                                    bool allow_random_fallback,
                                                    StreamKernelProvider* kernels,
                                                    const io::LinkageListStoreConfig& store_cfg,
                                                    LinkageDepthStats* base_depth_stats_out,
                                                    MseStats* base_linkage_mse_out,
                                                    std::string* err);

// Large-pipeline init-phase linkage build (Prompt 11): build a single-codebook linkage_list
// using only C_root (no C_one, no virtual roots). The output linkage_list is later used to:
// - compute bad cluster mask (using C_root-only reconstruction)
// - do the first exact-LS update of C_one (using root_to_one mapping for layer0)
bool BuildLinkageOneCodebookVirtualInitStreamingByCluster(const Config& cfg,
                                                        const TrainResult& train,
                                                        const io::BaseListReader& base_list,
                                                        const io::IvfListsReader& ivf_lists,
                                                        const io::DatasetVectorReader* fallback_reader,
                                                        bool allow_random_fallback,
                                                        StreamKernelProvider* kernels,
                                                        const io::LinkageListStoreConfig& store_cfg,
                                                        LinkageDepthStats* base_depth_stats_out,
                                                        MseStats* base_linkage_mse_out,
                                                        std::string* err);

// Rebuild (or build) the coefficient codec store under an existing `linkage_list/` directory,
// using float coefficients already stored in `linkage_list/` (no linkage rebuild, no base rebuild).
//
// This is used for recall-only runs when `large.linkage_coeff_codec.*` changes and we want to
// regenerate the Huffman+scale codec from float coefficients.
bool RebuildLinkageCoeffCodecFromLinkageListVirtual(const Config& cfg,
                                                const TrainResult& train,
                                                const io::LinkageListReader& linkage_list,
                                                std::string* err);

}  // namespace stlq
