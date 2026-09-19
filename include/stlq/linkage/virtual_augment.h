#pragma once

#include <string>
#include <vector>

#include "stlq/common/config.h"
#include "stlq/core/kernels.h"
#include "stlq/common/types.h"

namespace stlq {

class CudaStreamKernelsPool;

struct VirtualAnchorStats {
    bool has_euclidean_sse = false;
    std::uint64_t source_points = 0;
    std::uint64_t centers = 0;
    double euclidean_sse = 0.0;
    bool has_encoding_sse = false;
    double encoding_sse = 0.0;
};

bool BuildVirtualAnchorCentersFromKnnTables(
    const VirtualConfig& vcfg,
    const ColMajorMatrix<float>& X,
    const std::vector<int>& cluster_cols,
    const std::vector<std::uint32_t>& knn_ids_flat,
    const std::vector<float>& knn_dists_flat,
    int k_graph_total,
    int k_graph_umap,
    int k_virtual,
    int forced_root_code,
    std::uint32_t seed,
    ColMajorMatrix<float>* centers,
    std::vector<int>* forced_codes,
    std::string* error,
    VirtualAnchorStats* stats = nullptr);

// Builds and encodes virtual anchors according to VirtualConfig::anchor_policy.
// The UMAP policy consumes the pre-built in-cluster kNN table; classic
// sub-kmeans intentionally ignores it and minimizes Euclidean SSE directly.
// `knn_ids_flat` / `knn_dists_flat` are row-major [n_local x k_graph] and use local ids (0..n_local-1).
// Distances are Squared L2 (no sqrt), matching the internal HNSW L2 metric and avoiding sqrt overhead.
bool AddVirtualRootsReencodeClusterFromKnnTables(const VirtualConfig& vcfg,
                                                     const CodebookPack& codebooks,
                                                     const Precomp& precomp,
                                                     const RuntimeConfig* runtime_cfg,
                                                     CudaStreamKernelsPool* cuda_pool,
                                                     const ColMajorMatrix<float>& X,
                                                     const std::vector<int>& cluster_cols,
                                                     const std::vector<std::uint32_t>& knn_ids_flat,
                                                     const std::vector<float>& knn_dists_flat,
                                                     int k_graph_total,
                                                     int k_graph_umap,
                                                     int forced_root_code,
                                                     int k_virtual,
                                                     int ils_iters,
                                                     int icm_iters,
                                                     int perturb_k,
                                                     std::uint32_t seed,
                                                     ColMajorMatrix<float>* X_virtual,
                                                     ColMajorMatrix<FullCode>* B_virtual,
                                                     ColMajorMatrix<float>* a_virtual,
                                                     std::string* error,
                                                     VirtualAnchorStats* stats = nullptr);

bool AddVirtualNodes(const Config& config,
                     const CodebookPack& codebooks,
                     const Precomp& precomp,
                     const ColMajorMatrix<float>& X_real,
                     const BaseEncoding& base_real,
                     const std::vector<bool>& is_bad_cluster,
                     VirtualEncoding* virt_out,
                     BadClusterKnnCache* knn_cache_out,
                     std::string* error);

int ResolveVirtualCount(int n_local, const VirtualConfig& cfg);

}  // namespace stlq
