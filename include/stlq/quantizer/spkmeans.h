#pragma once

#include <random>
#include <string>
#include <vector>

#include "stlq/io/col_block_reader.h"
#include "stlq/quantizer/rvq_layer_store.h"
#include "stlq/common/types.h"

namespace stlq {

class StreamKernelProvider;

struct KmeansResult {
    ColMajorMatrix<float> centers;
    std::vector<int> assignments;
    std::vector<float> weights;
};

struct KmeansTiming {
    double total_s = 0.0;
    double normalize_s = 0.0;
    // Optional breakdown of normalize_s (GPU streaming path only).
    // - normalize_stage_s: optional host-side packing into a pinned staging buffer (only when needed)
    // - normalize_h2d_s: device timeline for H2D copy (events)
    // - normalize_kernel_s: device timeline for normalize kernel (events)
    double normalize_stage_s = 0.0;
    double normalize_h2d_s = 0.0;
    double normalize_kernel_s = 0.0;
    // Optional breakdown for out-of-core streaming profiling (best-effort; may be 0 when disabled).
    double reader_io_s = 0.0;             // CPU time spent inside reader->ReadNextInto/ReadNext
    double reader_file_s = 0.0;           // best-effort: file seek+read time inside reader
    double reader_unpack_s = 0.0;         // best-effort: unpack/transpose time inside reader
    double rvq_project_s = 0.0;           // compute-stream time for Prompt-06 residual projection (layer>0)
    double assign_update_compute_s = 0.0; // compute-stream time for assign+accumulate (excludes rvq_project_s)
    double emit_codes_d2h_s = 0.0;        // copy-stream time for Prompt-10 D2H assignments when emitting codes
    // Best-effort GPU wait times (CUDA stream waits).
    // - wait_copy_s: copy stream waiting for slot free (compute finished reading previous slot)
    // - wait_compute_s: compute stream waiting for slot ready (upload finished)
    double wait_copy_s = 0.0;
    double wait_compute_s = 0.0;
    double init_centers_s = 0.0;
    // Optional breakdown for init_centers_s (best-effort; 0 when not applicable).
    double init_sample_s = 0.0; // build the normalized init sample (BuildNormalizedSample*)
    double init_pick_s = 0.0;   // pick/init centers from sample (InitializeCentersFast)
    double assign_cpu_s = 0.0;
    double assign_gemm_s = 0.0;
    double assign_argmax_s = 0.0;
    double weights_s = 0.0;
    double update_s = 0.0;
    // Optional finer breakdown for streaming mode (not always populated).
    double accumulate_s = 0.0;
    double finalize_s = 0.0;
    // Streaming spill/IO that is not included above.
    double spill_s = 0.0;              // pass-1: write costs/assign to disk
    double update_read_weights_s = 0.0; // pass-2b: read weights for accumulation
    // Metric: mse = mean(||r||^2) accumulated on GPU during the last assignment scan (no code/coeff saving).
    // In RVQ init, this corresponds to the squared residual norm after the current layer projection.
    double mse_proxy = -1.0;

    // Diagnostics (set by implementation; useful to interpret normalize_* behavior).
    // Note: `used_pinned_host_x` is best-effort and means "some pinned host memory was used for X uploads"
    // (either a full pinned host mirror, or per-block pinned staging in streaming k-means).
    int used_device_xnorm_cache = 0; // 1 if full X_norm was cached on device
    int used_pinned_host_x = 0;      // 1 if pinned host memory was used for X uploads
    // 1 if the streaming reader path used copy/compute overlap via ping-pong slots.
    int used_overlap_pass1 = 0;
    int used_overlap_pass2 = 0;

    // ---- Large-K hier2 assignment diagnostics (A2) ----
    // Filled best-effort by the k-means implementation for the last run.
    int hier2_enable = 0;
    int hier2_train = 0;
    int hier2_encode = 0;
    int hier2_threshold = 0;
    int hier2_topL = 0;
    int used_hier2 = 0;
    int hier2_K = 0;
    int hier2_K1 = 0;
    int hier2_K2 = 0;

    // ---- Annealing/adaptive weights diagnostics ----
    int anneal_enabled = 0;
    int anneal_no_spill = 0;
};

enum class KmeansAnnealMode {
    kOnePass = 0,
    kTwoPass = 1,
};

struct KmeansConfig {
    int max_iters = 100;
    float tol = 1e-6f;
    // Progress logging: when >0, print a one-line progress log every N k-means iterations.
    // This is used by tools that otherwise "look stuck" for long streaming scans.
    int progress_every = 0;
    std::string init_method = "random";
    int init_samples = 2000;
    // kmeans|| seeding (scalable kmeans++), GPU-first. Used only when init_method == "kmeans||".
    bool kmeansll_gpu_enable = true;
    int kmeansll_rounds = 4;
    float kmeansll_oversample = 2.0f;
    int kmeansll_candidate_cap = 0;      // 0 => default (e.g. min(16*K, 1'000'000))
    std::uint64_t kmeansll_seed = 0;     // 0 => derive from rng
    bool kmeansll_hier_enable = true;    // future: hierarchical seeding for very large K
    int kmeansll_hier_threshold = 16384;
    int kmeansll_hier_fixed_k1 = 256;
    float initial_weight = 1.0f;
    float min_weight = 0.1f;
    float outlier_quantile = 0.95f;
    float cost_threshold = 0.8f;
    float annealing_factor = 0.5f;
    int warmup_iters = 5;
    // When >0, use a fixed-range histogram on cost = (1 - dot) in [0,2] to approximate
    // the outlier quantile threshold. This avoids `nth_element` over O(n) copies and is
    // intended for very large N (streaming mode).
    int quantile_bins = 65536;
    // Optional temp dir for streaming k-means. When annealing is enabled, the implementation may
    // use this directory to spill per-sample weights/costs to disk instead of holding O(n) arrays.
    std::string tmp_dir;
    // When streaming+annealing is active, prefer a no-spill two-pass algorithm:
    // - Pass 1: scan X, accumulate GPU/CPU histogram to estimate the outlier quantile threshold.
    // - Pass 2: rescan X, recompute argmax, update weights, and accumulate centers.
    //
    // If false, the legacy spill-based implementation may be used as a fallback.
    bool anneal_no_spill = true;
    // Adaptive-weight anneal scan mode:
    // - kOnePass: update weights using previous iteration's threshold and build histogram for next (1 scan/iter).
    // - kTwoPass: legacy behavior (Pass1 histogram, Pass2 rescan+update+accumulate).
    // One-pass is only used when device-resident weights are available.
    KmeansAnnealMode anneal_mode = KmeansAnnealMode::kOnePass;
    // Prefer device-resident global weights for adaptive annealing (avoids per-block bestdot D2H + weight H2D).
    bool anneal_weights_device = true;
    // Debug/testing only: force-disable device weights even if available (exercise fallback paths).
    bool force_disable_device_weights = false;
    bool display = false;
    // When true, avoid materializing X_norm; use blockwise normalization + a streaming update.
    // This is required for very large N (1e7/1e8+) and makes the algorithm compatible with
    // out-of-core data sources. When adaptive weights/annealing is enabled, the implementation uses
    // a two-pass streaming algorithm (Pass1 histogram, Pass2 rescan+accumulate).
    bool streaming = false;
    // CUDA streaming behavior controls (used only when `streaming=true` and GPU kernels are available).
    // - cache_xnorm_device: if true, try to cache the full X_norm on device when it fits (fast path).
    // - pin_host_x: if true and cache_xnorm_device fails/disabled, try to pin a host mirror of X for faster H2D.
    bool cache_xnorm_device = true;
    // Partial device cache (MiB) for streaming k-means when full caching is disabled/unavailable.
    int device_cache_mb = 0;
    bool pin_host_x = true;
    // When true (SIFT/BIGANN bvecs/siftbin inputs), treat the (host) training matrix as uint8-valued
    // for the purpose of GPU upload+normalize, to reduce H2D bandwidth. This is only applied for
    // layer-0 in RVQ init (callers should disable for residual layers).
    bool bvecs_use_u8 = false;
    // Block size (number of columns) used by streaming kmeans. 0 => implementation default.
    int block_cols = 0;

    // When false, skip all GPU metric accumulation and downloads (Prompt-03 MSE totals).
    // This avoids per-block metric kernels and the final tiny D2H of (sum,count).
    bool collect_metrics = true;

    // ---- Large-K hierarchical assignment (A2) ----
    // When K >= large_k_threshold, allow using a 2-level coarse→fine assignment
    // to reduce scoring cost. This is intended for RVQ-like init of very large IVF
    // layers (e.g. K=65536) and produces a flat center array of size K.
    int large_k_threshold = 8192;        // default trigger threshold
    bool large_k_hier2_enable = true;    // master enable

    // Split rule controls for K -> (K1, K2). If fixed_k1 > 0, use it directly.
    int large_k_fixed_k1 = 0;            // 0 => auto; else user-specified K1
    int large_k_k1_min = 128;
    int large_k_k1_max = 1024;
    bool large_k_k1_pow2 = true;         // round K1 to power-of-two (GPU-friendly)

    // Optional quality knob: consider top-L coarse groups per point (L>=1).
    // NOTE: current CUDA kernels clamp L to at most 2.
    int large_k_top_coarse = 1;          // 1 => strict hier; 2 improves quality

    // Optional: apply hier2 only for assignment (train/encode), still output flat K centers/codes.
    bool large_k_hier2_train = true;     // used in k-means training assignment
    bool large_k_hier2_encode = true;    // used in RVQ encode pass assignment
};

KmeansResult SphericalKmeans(const ColMajorMatrix<float>& X,
                             int k,
                             const KmeansConfig& cfg,
                             std::mt19937* rng,
                             StreamKernelProvider* stream_kernels,
                             bool profile_timing,
                             KmeansTiming* timing);

// In-memory implementation (materializes X as a matrix). This is the current baseline and is kept
// for regression testing; streaming k-means will be added as a separate entrypoint.
KmeansResult SphericalKmeansInMemory(const ColMajorMatrix<float>& X,
                                     int k,
                                     const KmeansConfig& cfg,
                                     std::mt19937* rng,
                                     StreamKernelProvider* stream_kernels,
                                     bool profile_timing,
                                     KmeansTiming* timing);

// Centers-only streaming k-means (for out-of-core training).
//
// This API intentionally does NOT return per-sample assignments/weights, so it is suitable for
// very large N (1e7/1e8+) where storing assignments is infeasible. When adaptive weights/annealing
// is enabled, a two-pass streaming algorithm is required (to be implemented later).
ColMajorMatrix<float> SphericalKmeansCentersOnlyStreamingOnePass(const ColMajorMatrix<float>& X,
                                                                 int k,
                                                                 const KmeansConfig& cfg,
                                                                 int block_cols,
                                                                 std::mt19937* rng,
                                                                 StreamKernelProvider* stream_kernels,
                                                                 bool profile_timing,
                                                                 KmeansTiming* timing);

// Centers-only streaming spherical k-means over an out-of-core block reader.
//
// This API is suitable for 1e7/1e8-scale training because it does not allocate O(N) arrays
// (no assignments/weights vectors). When annealing is enabled, this implementation uses a
// two-pass full rescan algorithm (Pass1 histogram, Pass2 rescan+update+accumulate).
ColMajorMatrix<float> SphericalKmeansCentersOnlyStreamingReader(io::IColBlockReader* reader,
                                                                int k,
                                                                const KmeansConfig& cfg,
                                                                std::mt19937* rng,
                                                                StreamKernelProvider* stream_kernels,
                                                                bool profile_timing,
                                                                KmeansTiming* timing,
                                                                std::string* err);

// Prompt-10: emit the last-iteration assignments as codes during the k-means update scan,
// removing the need for a separate encode-only full dataset pass.
ColMajorMatrix<float> SphericalKmeansCentersOnlyStreamingReaderEmitCodes(io::IColBlockReader* reader,
                                                                         int k,
                                                                         const KmeansConfig& cfg,
                                                                         std::mt19937* rng,
                                                                         StreamKernelProvider* stream_kernels,
                                                                         RvqInitCodesInMemory* codes_out,
                                                                         int out_layer,
                                                                         bool profile_timing,
                                                                         KmeansTiming* timing,
                                                                         std::string* err);

// RVQ init: centers-only streaming k-means over residuals reconstructed on GPU per block.
ColMajorMatrix<float> SphericalKmeansCentersOnlyStreamingReaderRvqResidualGpu(
    io::IColBlockReader* reader,
    int k,
    const KmeansConfig& cfg,
    std::mt19937* rng,
    StreamKernelProvider* stream_kernels,
    const RvqInitCodesInMemory* prior_codes,
    const std::vector<ColMajorMatrix<float>>* prior_codebooks,
    int upto_layer,
    bool profile_timing,
    KmeansTiming* timing,
    std::string* err);

ColMajorMatrix<float> SphericalKmeansCentersOnlyStreamingReaderRvqResidualGpuEmitCodes(
    io::IColBlockReader* reader,
    int k,
    const KmeansConfig& cfg,
    std::mt19937* rng,
    StreamKernelProvider* stream_kernels,
    const RvqInitCodesInMemory* prior_codes,
    const std::vector<ColMajorMatrix<float>>* prior_codebooks,
    int upto_layer,
    RvqInitCodesInMemory* codes_out,
    int out_layer,
    bool profile_timing,
    KmeansTiming* timing,
    std::string* err);

}  // namespace stlq
