#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "stlq/eval/linkage_cluster_provider.h"
#include "stlq/eval/query_table_builder.h"

namespace stlq::eval::cuda {

struct DiskLinkageGpuScanStats {
    // CPU-side time to prepare/pack metadata for this GPU call (does not include CUDA copies).
    double host_pack_cpu_sec = 0.0;
    // CPU-side time spent in (re)allocating GPU buffers (cudaMalloc/cudaFree), if any.
    double alloc_sec = 0.0;
    double clusters_pack_h2d_sec = 0.0;
    double query_tables_h2d_sec = 0.0;
    double kernel_sec = 0.0;
    // Scan kernel internal phase breakdown (only when profile_timing is enabled).
    double kernel_roots_sec = 0.0;
    double kernel_depth_sec = 0.0;
    double kernel_topk_sec = 0.0;
    double out_d2h_sec = 0.0;
    // Host-side time waiting on stream completion (covers GPU runtime + async copies).
    double stream_sync_sec = 0.0;
    // Cache diagnostics (per-call). Meaningful only when `profile_timing=true`.
    int cache_enabled = 0;
    int cache_slots = 0;
    int cache_hits = 0;            // number of clusters served from cache
    int cache_misses = 0;          // number of clusters uploaded this call
    int cache_upload_clusters = 0; // same as cache_misses (explicit name)
    std::uint64_t cache_upload_bytes = 0;
    std::uint64_t task_cluster_idx_bytes = 0;
    std::uint64_t out_d2h_bytes = 0;
    int tasks_total = 0;
    int tasks_gpu = 0;
    int clusters_total = 0;
    int clusters_gpu = 0;
    int kernel_max_nc = 0;
};

// Returns the maximum `kernel_max_nc` supported by the current GPU for the scan kernels
// (based on available dynamic shared memory per block). Returns 0 if CUDA is not enabled.
int DiskLinkageGpuScanMaxNcSupported();

// Experimental: GPU computes per-(query,cluster) local top-k (for "lookup, int8 coeff codec" path),
// CPU merges across clusters using existing TopK heap.
//
// Contract:
// - `task_cluster_idx.size()` must be `qlen * nprobe_cap`.
// - Each task outputs exactly `k` entries (padded with (id=UINT32_MAX, dist=+inf) when needed).
// - Tasks with `cluster_idx < 0` are treated as empty.
//
// This is eval-only; it must not affect any on-disk store hashes.
bool ScanDiskLinkageLookupTopK(const STLQueryTables& qt,
                             int m,
                             int m_codes,
                             int nlist,
                             int root_small_total_cols,
                             int one_total_cols,
                             const int* offsets_root_small,  // length m (offsets_root_small[0]==0)
                             const int* offsets_one,         // length m (meta_one.offsets)
                             const std::vector<ClusterView>& active_views,
                             const std::vector<int>& task_cluster_idx,
                             int qlen,
                             int nprobe_cap,
                             int k,
                             int max_nc,
                             int cache_mb,
                             bool tile256,
                             std::vector<float>* out_dists,
                             std::vector<std::uint32_t>* out_ids,
                             DiskLinkageGpuScanStats* stats,
                             std::string* err);

}  // namespace stlq::eval::cuda
