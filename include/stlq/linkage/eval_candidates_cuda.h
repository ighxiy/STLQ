#pragma once

#include <string>
#include <vector>

#include "stlq/cuda/cuda_stream_kernels_pool.h"
#include "stlq/common/types.h"

namespace stlq {

struct CudaLinkageEvalStats {
    std::uint64_t calls = 0;
    double h2d_s = 0.0;
    double residual_s = 0.0;
    double encode_s = 0.0;
    double reduce_s = 0.0;
    double d2h_s = 0.0;
    double total_s = 0.0;
};

// Peak workspace stats observed while CUDA linkage eval profiling is enabled.
// These are best-effort estimates intended for capacity planning (e.g., SIFT1B).
struct CudaLinkageEvalWorkspaceStats {
    // Peak shapes seen.
    int peak_d = 0;
    int peak_m = 0;
    int peak_H = 0;          // pre_one.H
    int peak_B = 0;          // batch nodes
    int peak_npairs = 0;     // Npairs (candidate pairs encoded on GPU)
    int peak_rfull_cols = 0; // device R_full columns (cluster-local nc)

    // Estimated bytes for a single CUDA ctx at peak (includes encode workspace + eval workspace).
    std::uint64_t peak_bytes_per_ctx_est = 0;
};

// Additional CUDA linkage eval profiling stats (only meaningful when profiling is enabled).
struct CudaLinkageEvalShapeStats {
    std::uint64_t calls = 0;
    std::uint64_t d2h_bytes = 0;
    int max_B = 0;
    int max_npairs = 0;
    // Histograms (counts) for call shapes.
    // B buckets: [1], [2-3], [4-7], [8-15], [16-31], [32-63], [64-127], [128-255],
    //            [256-511], [512-1023], [1024-2047], [2048+]
    std::uint64_t B_hist[12] = {};
    // Npairs buckets: [1], [2-3], [4-7], [8-15], [16-31], [32-63], [64-127], [128-255],
    //                 [256-511], [512-1023], [1024-2047], [2048+]
    std::uint64_t npairs_hist[12] = {};
};

// Enable/disable GPU-linkage profiling for EvaluateParentCandidatesBatchCuda.
// When enabled, CUDA event timing is recorded and aggregated across threads.
void SetCudaLinkageEvalProfiling(bool enabled);
CudaLinkageEvalStats GetAndResetCudaLinkageEvalStats();
CudaLinkageEvalWorkspaceStats GetAndResetCudaLinkageEvalWorkspaceStats();
CudaLinkageEvalShapeStats GetAndResetCudaLinkageEvalShapeStats();

// Optional capacity controls for CUDA many-nodes evaluator.
// These do not change semantics (all pairs are still evaluated); they only affect internal batching.
//
// - chunk_max_pairs<=0 disables internal chunking (one-shot up to caller's Npairs).
// - When enabled, a single many-nodes call may be split into multiple device sub-batches.
void SetCudaLinkageEvalChunkMaxPairs(int chunk_max_pairs);
void SetCudaLinkageEvalMemBudgetMb(int mem_budget_mb);
// Optional: host pinned staging budget (MiB) per CUDA ctx for async eval overlap.
// 0 disables async staging.
void SetCudaLinkageEvalAsyncPinnedBudgetMb(int pinned_mb);

// Evaluate candidate parent reconstructions for a single node (CPU control flow, GPU numerics).
//
// Input:
// - xi:        pointer to d floats (host)
// - Rp:        d×Kp column-major matrix (host, preferably pinned)
// - cand:      Kp ints (host), cluster-local ids matching Rp columns
//
// Output:
// - best_parent_local: cand[argmin]
// - best_cost: cost[argmin] (squared error)
// - best_B / best_a: residual encoding for the chosen parent (size m each)
bool EvaluateParentCandidatesBatchCuda(CudaCtx* ctx,
                                       const Precomp& pre_one,
                                       const float* xi,
                                       int d,
                                       const float* Rp,
                                       const int* cand_parent_local,
                                       int Kp,
                                       int icm_iters,
                                       int ils_iters,
                                       int perturb_k,
                                       std::uint32_t seed,
                                       std::uint64_t sample_id_offset,
                                       int* best_parent_local,
                                       float* best_cost,
                                       std::vector<FullCode>* best_B,
                                       std::vector<float>* best_a,
                                       std::string* err);

// Stage-2: upload per-cluster reconstruction matrix R_full to GPU.
//
// Layout requirement:
// - R_full is column-major (d×n_cols), matching cluster-local ids.
// - For good clusters (no virtual): local id == column id.
// - For bad clusters with virtual-front: first n_virt columns are virtual nodes,
//   followed by n_real columns for real nodes (local = n_virt + real_idx).
bool CudaLinkageUploadClusterRfull(CudaCtx* ctx,
                                const float* R_full,  // host, d×n_cols col-major
                                int d,
                                int n_cols,
                                std::string* err);

// Stage-2: upload R_full as two contiguous parts into a single virtual-front device buffer.
// This is intended for bad clusters where virtual and real recon buffers are separate on host.
bool CudaLinkageUploadClusterRfullTwoParts(CudaCtx* ctx,
                                        const float* R_part0,  // host, d×n0
                                        int n0,
                                        const float* R_part1,  // host, d×n1
                                        int n1,
                                        int d,
                                        std::string* err);

// Stage-2: after accepting best_parent and updating host R_full[:, col], update the GPU mirror column.
bool CudaLinkageUpdateClusterRfullColumn(CudaCtx* ctx,
                                       int col,              // cluster-local column id (virtual-front)
                                       const float* r_col,   // host pointer to d floats
                                       int d,
                                       std::string* err);

// Stage-2: update multiple columns of the GPU mirror in one shot.
//
// Inputs:
// - cols: length B, each entry is a cluster-local column id (virtual-front)
// - r_cols: length (B*d), packed as B contiguous columns, each column is d floats
//
// This is primarily used to batch per-node commit updates and reduce CUDA launch/driver overhead.
bool CudaLinkageUpdateClusterRfullColumnsBatch(CudaCtx* ctx,
                                            const int* cols,        // host, B
                                            const float* r_cols,    // host, (B*d)
                                            int d,
                                            int B,
                                            std::string* err);

// Stage-2 evaluator: Rp is gathered from device-resident d_R_full using cand_parent_local.
//
// IMPORTANT: caller must have already uploaded R_full via CudaLinkageUploadClusterRfull*(...),
// and must keep it updated via CudaLinkageUpdateClusterRfullColumn(...)
// (or the batched CudaLinkageUpdateClusterRfullColumnsBatch(...)) after each commit.
//
// Input:
// - xi: d floats (host)
// - cand_parent_local: Kp ints (host), cluster-local ids (virtual-front for bad clusters)
//
// Output:
// - best_parent_local: cand[argmin]
// - best_cost: cost[argmin] (squared error)
// - best_B / best_a: residual encoding for the chosen parent (size m each)
bool EvaluateParentCandidatesBatchCudaDeviceRfull(CudaCtx* ctx,
                                                  const Precomp& pre_one,
                                                  const float* xi,
                                                  int d,
                                                  const int* cand_parent_local,
                                                  int Kp,
                                                  int icm_iters,
                                                  int ils_iters,
                                                  int perturb_k,
                                                  std::uint32_t seed,
                                                  std::uint64_t sample_id_offset,
                                                  int* best_parent_local,
                                                  float* best_cost,
                                                  std::vector<FullCode>* best_B,
                                                  std::vector<float>* best_a,
                                                  std::string* err);

// Stage-3: Evaluate candidates for B nodes in one call, using device-resident R_full.
// Inputs are host pointers unless noted.
// - X_block: d×B (col-major), host
// - cand_parent_local_flat: size Npairs, host
// - pair_node: size Npairs, host, each entry in [0, B)
// - cand_offsets: size (B+1), host, offsets into cand_parent_local_flat per node
//
// Outputs are host arrays:
// - best_parent_local_out: size B
// - best_cost_out: size B
// - best_B_out: size (B*m) in node-major order (node i contiguous m codes)
// - best_a_out: size (B*m) in node-major order
bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfull(
    CudaCtx* ctx,
    const Precomp& pre_one,
    const float* X_block,              // host d×B
    int d,
    int B,
    const int* cand_parent_local_flat, // host Npairs
    const int* pair_node,              // host Npairs
    const int* cand_offsets,           // host (B+1)
    int Npairs,
    int icm_iters,
    int ils_iters,
    int perturb_k,
    std::uint32_t seed,
    std::uint64_t sample_id_offset,  // same semantics as per-node API: sample_id_offset + i
    int* best_parent_local_out,       // host B
    float* best_cost_out,             // host B
    FullCode* best_B_out,             // host B*m
    float* best_a_out,                // host B*m
    std::string* err);

// Same as EvaluateParentCandidatesManyNodesBatchCudaDeviceRfull, but uses per-node base IDs to make
// ILS perturbation deterministic and independent of batching/flush order.
//
// Each candidate pair (node i, candidate index t) is assigned:
//   sample_id = node_sample_id_base[i] + t
// which matches the per-node `sample_id_offset + t` scheme, but is independent of batching/flush order.
bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBase(
    CudaCtx* ctx,
    const Precomp& pre_one,
    const float* X_block,                 // host d×B
    int d,
    int B,
    const int* cand_parent_local_flat,    // host Npairs
    const int* pair_node,                 // host Npairs
    const int* cand_offsets,              // host (B+1)
    int Npairs,
    int icm_iters,
    int ils_iters,
    int perturb_k,
    std::uint32_t seed,
    const std::uint64_t* node_sample_id_base, // host B
    int* best_parent_local_out,            // host B
    float* best_cost_out,                  // host B
    FullCode* best_B_out,                  // host B*m
    float* best_a_out,                     // host B*m
    std::string* err);

// Same as `EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBase`, but allows providing
// an explicit per-pair "t" (candidate position within the node's candidate list) used to build per-pair
// sample IDs = node_sample_id_base[node] + t.
//
// This is required when a node's candidates are evaluated across multiple flushes/batches (e.g. same-layer
// dynamic windows), to keep ILS perturbation deterministic and consistent with the sequential per-node path.
bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseEx(
    CudaCtx* ctx,
    const Precomp& pre_one,
    const float* X_block,                 // host d×B
    int d,
    int B,
    const int* cand_parent_local_flat,    // host Npairs
    const int* pair_node,                 // host Npairs
    const int* cand_offsets,              // host (B+1)
    int Npairs,
    int icm_iters,
    int ils_iters,
    int perturb_k,
    std::uint32_t seed,
    const std::uint64_t* node_sample_id_base, // host B
    const int* pair_t_override,               // host Npairs (optional, may be null)
    int* best_parent_local_out,            // host B
    float* best_cost_out,                  // host B
    FullCode* best_B_out,                  // host B*m
    float* best_a_out,                     // host B*m
    std::string* err);

// Async variant of `EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBase`
// (one in-flight job per CudaCtx).
//
// Enqueue stages inputs into pinned host buffers (subject to `runtime.cuda_linkage_eval_async_pinned_mb`)
// and enqueues GPU work + D2H into a pinned buffer, then returns without synchronizing the stream.
// Finish waits for completion and unpacks results into the provided output arrays.
//
// Notes:
// - Only the one-shot (no internal pair-chunking) path is supported; if internal chunking would be used,
//   Enqueue returns false and callers should fall back to the synchronous API.
// - While a job is in flight, the same `CudaCtx` must not be used for another linkage-eval call.
bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseExEnqueue(
    CudaCtx* ctx,
    const Precomp& pre_one,
    const float* X_block,                 // host d×B
    int d,
    int B,
    const int* cand_parent_local_flat,    // host Npairs
    const int* pair_node,                 // host Npairs
    const int* cand_offsets,              // host (B+1)
    int Npairs,
    int icm_iters,
    int ils_iters,
    int perturb_k,
    std::uint32_t seed,
    const std::uint64_t* node_sample_id_base, // host B
    const int* pair_t_override,               // host Npairs (optional, may be null)
    std::string* err);

bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseExFinish(
    CudaCtx* ctx,
    int* best_parent_local_out,           // host B
    float* best_cost_out,                 // host B
    FullCode* best_B_out,                 // host B*m
    float* best_a_out,                    // host B*m
    std::string* err);

// Forced-root variant of EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBase.
// Uses LinkageEncodeBatchCudaForcedRootWithSampleIds on GPU (skip-layer0 ICM/ILS semantics), with
// forced_root_code applied to every candidate pair.
//
// Intended for init-linkage stages where layer0 is fixed (e.g., C_root-only init linkage build).
bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBase(
    CudaCtx* ctx,
    const Precomp& pre_one,
    int forced_root_code,
    const float* X_block,                 // host d×B
    int d,
    int B,
    const int* cand_parent_local_flat,    // host Npairs
    const int* pair_node,                 // host Npairs
    const int* cand_offsets,              // host (B+1)
    int Npairs,
    int icm_iters,
    int ils_iters,
    int perturb_k,
    std::uint32_t seed,
    const std::uint64_t* node_sample_id_base, // host B
    int* best_parent_local_out,           // host B
    float* best_cost_out,                 // host B
    FullCode* best_B_out,                 // host B*m
    float* best_a_out,                    // host B*m
    std::string* err);

// Large-root/split-root forced-root variant:
// - avoids requiring full-precomp G(H×H) for large h0 by using the large-root linkage-encode CUDA path
// - root code is fixed to `forced_root_code` for every candidate pair
// Intended for init-linkage when `C_root` layer0 is the IVF routing centroid (can be very large).
bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBaseLargeRoot(
    CudaCtx* ctx,
    const CodebookPack& C_root,
    int forced_root_code,
    const float* X_block,                 // host d×B
    int d,
    int B,
    const int* cand_parent_local_flat,    // host Npairs
    const int* pair_node,                 // host Npairs
    const int* cand_offsets,              // host (B+1)
    int Npairs,
    int icm_iters,
    int ils_iters,
    int perturb_k,
    std::uint32_t seed,
    const std::uint64_t* node_sample_id_base, // host B
    int* best_parent_local_out,           // host B
    float* best_cost_out,                 // host B
    FullCode* best_B_out,                 // host B*m
    float* best_a_out,                    // host B*m
    std::string* err);

// Large-root/split-root variant without a forced cluster-id root:
// - selects layer0 code greedily per residual (see LinkageEncodeBatchCudaWithSampleIdsLargeRoot)
// - keeps deterministic per-sample ILS via explicit node_sample_id_base
bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRoot(
    CudaCtx* ctx,
    const CodebookPack& C_root,
    const float* X_block,                 // host d×B
    int d,
    int B,
    const int* cand_parent_local_flat,    // host Npairs
    const int* pair_node,                 // host Npairs
    const int* cand_offsets,              // host (B+1)
    int Npairs,
    int icm_iters,
    int ils_iters,
    int perturb_k,
    std::uint32_t seed,
    const std::uint64_t* node_sample_id_base, // host B
    int* best_parent_local_out,           // host B
    float* best_cost_out,                 // host B
    FullCode* best_B_out,                 // host B*m
    float* best_a_out,                    // host B*m
    std::string* err);

// Same as `EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRoot`, but writes codes
// as uint32 (host B*m). Required when `h0_root > 256` (root codes do not fit in `FullCode` which is now u8).
bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootU32(
    CudaCtx* ctx,
    const CodebookPack& C_root,
    const float* X_block,                 // host d×B
    int d,
    int B,
    const int* cand_parent_local_flat,    // host Npairs
    const int* pair_node,                 // host Npairs
    const int* cand_offsets,              // host (B+1)
    int Npairs,
    int icm_iters,
    int ils_iters,
    int perturb_k,
    std::uint32_t seed,
    const std::uint64_t* node_sample_id_base, // host B
    int* best_parent_local_out,           // host B
    float* best_cost_out,                 // host B
    std::uint32_t* best_B_out_u32,        // host B*m
    float* best_a_out,                    // host B*m
    std::string* err);

// Same as `EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRoot`, but allows
// providing an explicit per-pair "t" (candidate position within the node's candidate list) used to build
// per-pair sample IDs = node_sample_id_base[node] + t.
//
// This is required when a node's candidates are evaluated across multiple flushes/batches (e.g. same-layer
// dynamic windows), to keep ILS perturbation deterministic and consistent with the sequential per-node path.
bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootEx(
    CudaCtx* ctx,
    const CodebookPack& C_root,
    const float* X_block,                 // host d×B
    int d,
    int B,
    const int* cand_parent_local_flat,    // host Npairs
    const int* pair_node,                 // host Npairs
    const int* cand_offsets,              // host (B+1)
    int Npairs,
    int icm_iters,
    int ils_iters,
    int perturb_k,
    std::uint32_t seed,
    const std::uint64_t* node_sample_id_base, // host B
    const int* pair_t_override,           // host Npairs
    int* best_parent_local_out,           // host B
    float* best_cost_out,                 // host B
    FullCode* best_B_out,                 // host B*m
    float* best_a_out,                    // host B*m
    std::string* err);

// Same as `EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootEx`, but writes codes
// as uint32 (host B*m). Required when `h0_root > 256` (root codes do not fit in `FullCode` which is now u8).
bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExU32(
    CudaCtx* ctx,
    const CodebookPack& C_root,
    const float* X_block,                 // host d×B
    int d,
    int B,
    const int* cand_parent_local_flat,    // host Npairs
    const int* pair_node,                 // host Npairs
    const int* cand_offsets,              // host (B+1)
    int Npairs,
    int icm_iters,
    int ils_iters,
    int perturb_k,
    std::uint32_t seed,
    const std::uint64_t* node_sample_id_base, // host B
    const int* pair_t_override,           // host Npairs
    int* best_parent_local_out,           // host B
    float* best_cost_out,                 // host B
    std::uint32_t* best_B_out_u32,        // host B*m
    float* best_a_out,                    // host B*m
    std::string* err);

// Async variant of `EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootEx`
// (one in-flight job per CudaCtx).
//
// Enqueue stages inputs into pinned host buffers (subject to `runtime.cuda_linkage_eval_async_pinned_mb`)
// and enqueues GPU work + D2H into a pinned buffer, then returns without synchronizing the stream.
// Finish waits for completion and unpacks results into the provided output arrays.
//
// Notes:
// - Only the one-shot (no internal pair-chunking) path is supported; if internal chunking would be used,
//   Enqueue returns false and callers should fall back to the synchronous API.
// - While a job is in flight, the same `CudaCtx` must not be used for another linkage-eval call.
bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExEnqueue(
    CudaCtx* ctx,
    const CodebookPack& C_root,
    const float* X_block,                 // host d×B
    int d,
    int B,
    const int* cand_parent_local_flat,    // host Npairs
    const int* pair_node,                 // host Npairs
    const int* cand_offsets,              // host (B+1)
    int Npairs,
    int icm_iters,
    int ils_iters,
    int perturb_k,
    std::uint32_t seed,
    const std::uint64_t* node_sample_id_base, // host B
    const int* pair_t_override,               // host Npairs
    std::string* err);

bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExFinish(
    CudaCtx* ctx,
    int* best_parent_local_out,           // host B
    float* best_cost_out,                 // host B
    FullCode* best_B_out,                 // host B*m
    float* best_a_out,                    // host B*m
    std::string* err);

// Async variant of `EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExU32`.
// See the `...LargeRootExEnqueue/Finish` docs above for semantics/limitations.
bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExEnqueueU32(
    CudaCtx* ctx,
    const CodebookPack& C_root,
    const float* X_block,                 // host d×B
    int d,
    int B,
    const int* cand_parent_local_flat,    // host Npairs
    const int* pair_node,                 // host Npairs
    const int* cand_offsets,              // host (B+1)
    int Npairs,
    int icm_iters,
    int ils_iters,
    int perturb_k,
    std::uint32_t seed,
    const std::uint64_t* node_sample_id_base, // host B
    const int* pair_t_override,               // host Npairs
    std::string* err);

bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExFinishU32(
    CudaCtx* ctx,
    int* best_parent_local_out,           // host B
    float* best_cost_out,                 // host B
    std::uint32_t* best_B_out_u32,        // host B*m
    float* best_a_out,                    // host B*m
    std::string* err);

}  // namespace stlq
