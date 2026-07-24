#pragma once

#include <string>

#include "stlq/cuda/cuda_stream_kernels_pool.h"
#include "stlq/common/types.h"

namespace stlq {

struct LinkageEncodeBatchCudaWorkspace;

struct CudaLinkageEncodeStats {
    std::uint64_t calls = 0;
    double gemm_s = 0.0;
    double greedy_s = 0.0;
    double ls_cost_s = 0.0;
    double icm_s = 0.0;
    double ils_s = 0.0;
    // ILS internal breakdown (outer loop aggregate).
    double ils_copy_s = 0.0;
    double ils_solve_cost_s = 0.0;
    double ils_icm_s = 0.0;
    double ils_encode_s = 0.0;
    double ils_accept_s = 0.0;
    double total_s = 0.0;
};

struct CudaLinkageEncodeLastProfile {
    double gemm_s = 0.0;
    double greedy_s = 0.0;
    double ls_cost_s = 0.0;
    double icm_s = 0.0;
    double ils_s = 0.0;
    double total_s = 0.0;
    // ILS internal breakdown (single call; aggregate over outer loop).
    double ils_copy_s = 0.0;
    double ils_solve_cost_s = 0.0;
    double ils_icm_s = 0.0;
    double ils_encode_s = 0.0;
    double ils_accept_s = 0.0;
    bool valid = false;
};

// Shape/flow stats for the large-root encoder (split-root path).
// Collected only when CUDA linkage profiling is enabled.
struct CudaLinkageEncodeLargeRootStats {
    std::uint64_t calls = 0;
    // Root selection tiling over h0:
    // - root_chunks = ceil(h0 / root_chunk_h) per call
    std::uint64_t root_chunks_total = 0;
    std::uint64_t root_chunks_max = 0;
    std::uint64_t have_xc0_full_calls = 0;  // root_chunk_h == h0
    std::uint64_t root_chunk_h_min = 0;
    std::uint64_t root_chunk_h_max = 0;

    // Root ICM update (jlayer==0) fast path usage.
    std::uint64_t root_icm_fast_calls = 0;  // cached xC0_full + G0s_T path
    std::uint64_t root_icm_slow_calls = 0;  // fallback path (re0/GEMM or other)

    // Sample-dimension chunking due to xC0_full memory cap (xc0_chunk_mb).
    std::uint64_t sample_chunk_calls = 0;       // wrapper used chunking (n split)
    std::uint64_t sample_chunks_total = 0;      // total number of chunks across calls
    std::uint64_t sample_chunk_n_min = 0;
    std::uint64_t sample_chunk_n_max = 0;
};

// Host-side overhead stats for linkage-encode entrypoints.
// This is intended to explain gaps between:
// - caller-side CUDA event timing around the encode call, and
// - kernel-only breakdown measured inside the encode implementation.
//
// Collected only when CUDA linkage profiling is enabled.
struct CudaLinkageEncodeHostStats {
    std::uint64_t calls = 0;
    double ensure_s = 0.0;                // time spent in EnsurePrecomp/EnsureBatch (wall)
    std::uint64_t precomp_refresh_calls = 0;
    std::uint64_t batch_grow_calls = 0;   // any of the main device buffers grew
    std::uint64_t xC_grow_calls = 0;      // d_xC capacity grew (typically dominates allocations)
    std::uint64_t xC_grow_max_elems = 0;  // max allocated elems for d_xC during the profiled window
};

LinkageEncodeBatchCudaWorkspace* CreateLinkageEncodeBatchCudaWorkspace();
void DestroyLinkageEncodeBatchCudaWorkspace(LinkageEncodeBatchCudaWorkspace* ws);

// Optional profiling for LinkageEncodeBatchCuda internals.
// Intended to be enabled only under `large.profile_timing=true`.
void SetCudaLinkageEncodeProfiling(bool enabled);
CudaLinkageEncodeStats GetAndResetCudaLinkageEncodeStats();
CudaLinkageEncodeLargeRootStats GetAndResetCudaLinkageEncodeLargeRootStats();
CudaLinkageEncodeHostStats GetAndResetCudaLinkageEncodeHostStats();
void ConsumeCudaLinkageEncodeLastProfile(LinkageEncodeBatchCudaWorkspace* ws);
CudaLinkageEncodeLastProfile GetCudaLinkageEncodeLastProfile(LinkageEncodeBatchCudaWorkspace* ws);

// Clear process-wide shared precomp caches used by linkage-encode CUDA kernels.
// This is intended to be called at a safe boundary (e.g., between training rounds) to avoid
// VRAM growth when codebooks are updated (build_tag changes).
// If `device<0`, clears caches for all devices.
void ClearCudaLinkageEncodeSharedPrecompCaches(int device = -1);

// Large-root root-selection tiling cap (MiB).
// Controls the temporary `xC0_chunk` matrix size used by the large-root (split-root) encoder.
// Larger values reduce the number of tiled GEMMs for root selection (faster), but increase peak VRAM per CUDA ctx.
// This does not change semantics (still exact scan over all root codes).
void SetCudaLinkageEncodeLargeRootXc0ChunkMb(int mb);

// Large-root workspace budget (MiB) for the variable-root large-root encoder.
// When >0, large-root encoding will chunk along the sample dimension to keep temporary buffers bounded
// (e.g. xC_small/rC_small/g0s for Npairs), which prevents OOM in many-nodes candidate evaluation.
// Semantics are unchanged (independent samples => chunking only affects execution order).
void SetCudaLinkageEncodeLargeRootWorkspaceMb(int mb);

// Hybrid init-linkage ILS policy (thread-local, per calling thread):
// Controls how many of the initial ILS rounds use VarRoot mode (layer0 perturbable + root ICM GEMM)
// before switching to ConstRoot mode (layer0 frozen, skip root ICM GEMM) for the remaining rounds.
// This avoids the expensive tiled GEMM over all h0 root centroids in later ILS rounds while
// preserving high-quality root assignment from the first round.
//   -1 = all rounds VarRoot (default, current behavior)
//    0 = all rounds ConstRoot (skip root perturbation and root ICM in all ILS rounds)
//    N>0 = first N rounds VarRoot, remaining rounds ConstRoot
void SetCudaLinkageEncodeHybridVarRootIlsRounds(int rounds);
void ResetCudaLinkageEncodeHybridVarRootIlsRounds();
int  GetCudaLinkageEncodeHybridVarRootIlsRounds();

// Batch residual encoding for linkage candidate evaluation:
// - xC = C_all^T * residuals (GEMM)
// - greedy init (abs, no-normal)
// - least-squares solve per sample (Cholesky retry)
// - ICM refinement (abs)
// - optional ILS outer loop (perturb → LS → ICM → accept)
// - cost per sample via algebraic form (no reconstruct)
//
// All tensors are column-major.
bool LinkageEncodeBatchCuda(const CudaCtx& ctx,
                          const Precomp& pre_one,
                          const float* d_residuals,      // d×n (device)
                          const float* d_res_norm2,      // n (device)
                          int n,
                          int icm_iters,
                          int ils_iters,
                          int perturb_k,
                          std::uint32_t seed,
                          std::uint64_t sample_id_offset,
                          FullCode* d_B_out,             // m×n (device)
                          float* d_a_out,                // m×n (device)
                          float* d_cost_out,             // n (device)
                          LinkageEncodeBatchCudaWorkspace* ws,
                          std::string* err);

// Same as LinkageEncodeBatchCuda, but with a forced root code (layer 0) per sample:
// - layer0 is initialized to `forced_root[i]` and never modified by greedy/ICM/ILS.
// - Greedy init and ICM/ILS operate only on layers 1..m-1.
//
// `d_forced_root` is a device pointer of length n with values in [0, h_vec[0]).
bool LinkageEncodeBatchCudaForcedRoot(const CudaCtx& ctx,
                                   const Precomp& pre_one,
                                   const float* d_residuals,      // d×n (device)
                                   const float* d_res_norm2,      // n (device)
                                   const int* d_forced_root,      // n (device)
                                   int n,
                                   int icm_iters,
                                   int ils_iters,
                                   int perturb_k,
                                   std::uint32_t seed,
                                   std::uint64_t sample_id_offset,
                                   FullCode* d_B_out,             // m×n (device)
                                   float* d_a_out,                // m×n (device)
                                   float* d_cost_out,             // n (device)
                                   LinkageEncodeBatchCudaWorkspace* ws,
                                   std::string* err);

// Variant of LinkageEncodeBatchCuda that uses an explicit per-sample ID for ILS perturbation.
// This is used to make many-nodes batching deterministic and independent of batching/flush order.
// `d_sample_ids[i]` corresponds to the CPU-side `i_global` used by SplitMix64 (see icm.cpp).
bool LinkageEncodeBatchCudaWithSampleIds(const CudaCtx& ctx,
                                       const Precomp& pre_one,
                                       const float* d_residuals,      // d×n (device)
                                       const float* d_res_norm2,      // n (device)
                                       const std::uint64_t* d_sample_ids,  // n (device)
                                       int n,
                                       int icm_iters,
                                       int ils_iters,
                                       int perturb_k,
                                       std::uint32_t seed,
                                       FullCode* d_B_out,             // m×n (device)
                                       float* d_a_out,                // m×n (device)
                                       float* d_cost_out,             // n (device)
                                       LinkageEncodeBatchCudaWorkspace* ws,
                                       std::string* err);

// Forced-root + explicit per-sample IDs (see LinkageEncodeBatchCudaWithSampleIds).
bool LinkageEncodeBatchCudaForcedRootWithSampleIds(const CudaCtx& ctx,
                                                 const Precomp& pre_one,
                                                 const float* d_residuals,      // d×n (device)
                                                 const float* d_res_norm2,      // n (device)
                                                 const int* d_forced_root,      // n (device)
                                                 const std::uint64_t* d_sample_ids,  // n (device)
                                                 int n,
                                                 int icm_iters,
                                                 int ils_iters,
                                                 int perturb_k,
                                                 std::uint32_t seed,
                                                 FullCode* d_B_out,             // m×n (device)
                                                 float* d_a_out,                // m×n (device)
                                                 float* d_cost_out,             // n (device)
                                                 LinkageEncodeBatchCudaWorkspace* ws,
                                                 std::string* err);

// Forced-root + explicit per-sample IDs, but using a "large-root / split-root" CUDA path.
//
// This variant is intended for init-linkage stages where:
// - layer0 (root) is fixed to a known cluster id (`forced_root_code`) and must NOT be modified by ICM/ILS
// - h0 can be large (e.g. nlist=6144/65536), so building full-precomp G(H×H) is infeasible
// - small layers (1..m-1) are moderate (typically h_vec[l>=1] <= 256)
//
// Semantics match LinkageEncodeBatchCudaForcedRootWithSampleIds, but the implementation avoids allocating
// the full Gram matrix by working only on the flattened "small" layers and using the fixed root vector.
bool LinkageEncodeBatchCudaForcedRootWithSampleIdsLargeRootConstRoot(const CudaCtx& ctx,
                                                                   const CodebookPack& C_root,
                                                                   int forced_root_code,
                                                                   const float* d_residuals,      // d×n (device)
                                                                   const float* d_res_norm2,      // n (device)
                                                                   const std::uint64_t* d_sample_ids,  // n (device)
                                                                   int n,
                                                                   int icm_iters,
                                                                   int ils_iters,
                                                                   int perturb_k,
                                                                   std::uint32_t seed,
                                                                   FullCode* d_B_out,             // m×n (device)
                                                                   float* d_a_out,                // m×n (device)
                                                                   float* d_cost_out,             // n (device)
                                                                   LinkageEncodeBatchCudaWorkspace* ws,
                                                                   std::string* err);

// Same as `LinkageEncodeBatchCudaForcedRootWithSampleIdsLargeRootConstRoot`, but writes codes as uint32 (m×n).
// Required when `h0_root > 256` (root codes do not fit in `FullCode` which is now u8).
bool LinkageEncodeBatchCudaForcedRootWithSampleIdsLargeRootConstRootU32(const CudaCtx& ctx,
                                                                      const CodebookPack& C_root,
                                                                      int forced_root_code,
                                                                      const float* d_residuals,      // d×n (device)
                                                                      const float* d_res_norm2,      // n (device)
                                                                      const std::uint64_t* d_sample_ids,  // n (device)
                                                                      int n,
                                                                      int icm_iters,
                                                                      int ils_iters,
                                                                      int perturb_k,
                                                                      std::uint32_t seed,
                                                                      std::uint32_t* d_B_out_u32,    // m×n (device)
                                                                      float* d_a_out,                // m×n (device)
                                                                      float* d_cost_out,             // n (device)
                                                                      LinkageEncodeBatchCudaWorkspace* ws,
                                                                      std::string* err);

// Large-root / split-root variant (no forced cluster-id root):
// - supports large h0 without building full Gram G(H×H)
// - selects layer0 code greedily (abs/nonormal) and keeps it fixed during ICM/ILS
// - greedy/ICM/ILS operate on layers 1..m-1
bool LinkageEncodeBatchCudaWithSampleIdsLargeRoot(const CudaCtx& ctx,
                                               const CodebookPack& C_root,
                                               const float* d_residuals,      // d×n (device)
                                               const float* d_res_norm2,      // n (device)
                                               const std::uint64_t* d_sample_ids,  // n (device)
                                               int n,
                                               int icm_iters,
                                               int ils_iters,
                                               int perturb_k,
                                               std::uint32_t seed,
                                               FullCode* d_B_out,             // m×n (device)
                                               float* d_a_out,                // m×n (device)
                                               float* d_cost_out,             // n (device)
                                               LinkageEncodeBatchCudaWorkspace* ws,
                                               std::string* err);

// Same as `LinkageEncodeBatchCudaWithSampleIdsLargeRoot`, but writes codes as uint32 (m×n).
// Required when `h0_root > 256` (root codes do not fit in `FullCode` which is now u8).
bool LinkageEncodeBatchCudaWithSampleIdsLargeRootU32(const CudaCtx& ctx,
                                                   const CodebookPack& C_root,
                                                   const float* d_residuals,      // d×n (device)
                                                   const float* d_res_norm2,      // n (device)
                                                   const std::uint64_t* d_sample_ids,  // n (device)
                                                   int n,
                                                   int icm_iters,
                                                   int ils_iters,
                                                   int perturb_k,
                                                   std::uint32_t seed,
                                                   std::uint32_t* d_B_out_u32,    // m×n (device)
                                                   float* d_a_out,                // m×n (device)
                                                   float* d_cost_out,             // n (device)
                                                   LinkageEncodeBatchCudaWorkspace* ws,
                                                   std::string* err);

// Forced-root variant that starts from a precomputed xC matrix on device:
//   xC = C_all^T * residuals  (H×n, column-major with leading dimension H).
// This is primarily used for UMAP-center encoding to match CPU baseline xC exactly.
bool LinkageEncodeBatchCudaForcedRootDeviceXc(const CudaCtx& ctx,
                                           const Precomp& pre_one,
                                           const float* d_xC,             // H×n (device)
                                           const float* d_res_norm2,      // n (device)
                                           const int* d_forced_root,      // n (device)
                                           int n,
                                           int icm_iters,
                                           int ils_iters,
                                           int perturb_k,
                                           std::uint32_t seed,
                                           std::uint64_t sample_id_offset,
                                           FullCode* d_B_out,             // m×n (device)
                                           float* d_a_out,                // m×n (device)
                                           float* d_cost_out,             // n (device)
                                           LinkageEncodeBatchCudaWorkspace* ws,
                                           std::string* err);

// Same as LinkageEncodeBatchCudaForcedRootDeviceXc, but assumes the caller has already initialized
// `d_B_out` (m×n) with valid codes (including the forced root at layer0). This skips the internal
// greedy-init stage and starts from the provided codes.
//
// Intended for UMAP-center encoding: use CPU greedy init (exact baseline) + GPU ICM/ILS refinement.
bool LinkageEncodeBatchCudaForcedRootDeviceXcFromInitB(const CudaCtx& ctx,
                                                    const Precomp& pre_one,
                                                    const float* d_xC,             // H×n (device)
                                                    const float* d_res_norm2,      // n (device)
                                                    const int* d_forced_root,      // n (device)
                                                    int n,
                                                    int icm_iters,
                                                    int ils_iters,
                                                    int perturb_k,
                                                    std::uint32_t seed,
                                                    std::uint64_t sample_id_offset,
                                                    FullCode* d_B_out,             // m×n (device, prefilled)
                                                    float* d_a_out,                // m×n (device)
                                                    float* d_cost_out,             // n (device)
                                                    LinkageEncodeBatchCudaWorkspace* ws,
                                                    std::string* err);

}  // namespace stlq
