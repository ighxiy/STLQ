#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

#include "stlq/core/kernel_provider.h"

namespace stlq {

// CUDA implementation of StreamKernelProvider (basic streaming path).
// This is the entry point for GPU acceleration of the basic (base) stage:
// - uint8 -> float conversion
// - optional rotation (GEMM)
// - GEMM for query/codebook tables
//
// Build requirement: configure with -DSTLQ_ENABLE_CUDA=ON.
class CudaStreamKernels final : public StreamKernelProvider {
public:
    // `cublas_workspace_mb` controls the explicit cuBLAS workspace size (MiB).
    // It can materially affect GEMM algorithm selection and performance stability.
    CudaStreamKernels(int device, bool allow_tf32, int cublas_workspace_mb);
    ~CudaStreamKernels() override;

    CudaStreamKernels(const CudaStreamKernels&) = delete;
    CudaStreamKernels& operator=(const CudaStreamKernels&) = delete;

    bool IsGpu() const override { return true; }

    // Force-release all CUDA allocations held by this provider (device buffers, pinned staging, caches),
    // then recreate internal CUDA resources (streams/cuBLAS handle).
    //
    // Intended for large-scale pipelines where later stages (e.g. init-linkage) need most VRAM and
    // this basic-encode provider is temporarily unused.
    //
    // NOTE: This is a heavyweight operation; call only at phase boundaries (never in hot loops).
    void ResetCudaState();

    void ConvertU8ToF32(const ColMajorMatrix<std::uint8_t>& in,
                        ColMajorMatrix<float>* out) override;

    void ConvertU8ToF32AndRotate(const ColMajorMatrix<std::uint8_t>& in,
                                 const ColMajorMatrix<float>& R,
                                 ColMajorMatrix<float>* out) override;

    void ConvertU8ToF32AndRotatePtr(const std::uint8_t* in,
                                   int ld_in,
                                   int rows,
                                   int cols,
                                   const ColMajorMatrix<float>& R,
                                   ColMajorMatrix<float>* out) override;

    void Gemm(bool transA, bool transB,
              float alpha,
              const ColMajorMatrix<float>& A,
              const ColMajorMatrix<float>& B,
              float beta,
              ColMajorMatrix<float>* C) override;

    void GemmDevice(bool transA, bool transB,
                    float alpha,
                    const ColMajorMatrix<float>& A,
                    const ColMajorMatrix<float>& B,
                    float beta,
                    DeviceMatF32View* out) override;

    void GemmDeviceHostPtrB(bool transA, bool transB,
                            float alpha,
                            const ColMajorMatrix<float>& A,
                            const float* B,
                            int ldB,
                            int rowsB,
                            int colsB,
                            float beta,
                            DeviceMatF32View* out) override;

    bool EnsureDeviceF32(const ColMajorMatrix<float>& host,
                         DeviceMatF32View* out) override;

    void GemmDeviceDevicePtrB(bool transA, bool transB,
                              float alpha,
                              const ColMajorMatrix<float>& A,
                              const float* dB,
                              int ldB,
                              int rowsB,
                              int colsB,
                              float beta,
                              DeviceMatF32View* out) override;

    void DownloadF32(const DeviceMatF32View& view,
                     ColMajorMatrix<float>* out) override;

    bool TryGetCachedDeviceF32(const ColMajorMatrix<float>& host,
                               DeviceMatF32View* out) override;

    void ArgmaxColsF32(const DeviceMatF32View& scores,
                       std::vector<int>* assignments,
                       std::vector<float>* best_values) override;

    void ArgmaxColsF32Device(const DeviceMatF32View& scores,
                             DeviceVecI32View* d_assign,
                             DeviceVecF32View* d_best) override;

    void Sync() override;
    void SyncCompute() override;

    // -------- K-means streaming helpers (GPU) --------
    // Enable/disable GPU metric collection (Prompt-03 totals).
    // When disabled, skip norm2 bookkeeping and per-block metric kernels.
    void KmeansSetCollectMetrics(bool enable);

    // Upload a host tile (column-major) to device and normalize each column to unit L2.
    // Returns a device view backed by an internal temporary buffer that is overwritten
    // by subsequent calls.
    DeviceMatF32View KmeansUploadAndNormalizeHostPtrB(const float* B,
                                                      int ldB,
                                                      int rowsB,
                                                      int colsB);

    // Same as KmeansUploadAndNormalizeHostPtrB, but returns a breakdown of the GPU timeline
    // (H2D copy time and normalize-kernel time) in seconds. Only intended for profiling.
    //
    // If `stage_to_pinned` is true, the input tile is first packed into an internal pinned host buffer
    // (to avoid pageable-memory staging costs). This adds host-side memcpy time, but can dramatically
    // reduce total normalize time when the source pointer is not already pinned.
    DeviceMatF32View KmeansUploadAndNormalizeHostPtrBTimed(const float* B,
                                                           int ldB,
                                                           int rowsB,
                                                           int colsB,
                                                           bool stage_to_pinned,
                                                           double* stage_s,
                                                           double* h2d_s,
                                                           double* kernel_s);

    // Async upload+normalize for streaming k-means, using a fixed 2-slot ping-pong buffer.
    // - Work is enqueued onto an internal copy stream.
    // - The caller must insert a dependency on the compute stream via `KmeansComputeWaitForUpload(slot)`
    //   before consuming the returned device view.
    //
    // If `h2d_s` / `kernel_s` are non-null, this function may wait for completion to measure event time.
    // For overlap mode, pass nullptr timing pointers to avoid synchronization.
    DeviceMatF32View KmeansUploadAndNormalizeHostPtrBAsync(int slot,
                                                           const float* B,
                                                           int ldB,
                                                           int rowsB,
                                                           int colsB,
                                                           bool stage_to_pinned,
                                                           double* stage_s,
                                                           double* h2d_s,
                                                           double* kernel_s);

    // Optional: expose pinned stage buffers so out-of-core readers can fill them directly
    // (avoids doing pageable->pinned packing inside the upload call).
    //
    // Returns a column-major packed buffer with ld == rowsB.
    float* KmeansGetPinnedStageBuffer(int slot, int rowsB, int colsB);

    // -------- K-means uint8 streaming helpers (GPU) --------
    // Upload a uint8 block and L2-normalize columns on GPU (fused u8->f32 + normalize).
    // Same async/slot semantics as the float32 upload APIs above.
    DeviceMatF32View KmeansUploadU8AndNormalizeHostPtrBAsync(int slot,
                                                             const std::uint8_t* B,
                                                             int ldB,
                                                             int rowsB,
                                                             int colsB,
                                                             bool stage_to_pinned,
                                                             double* stage_s,
                                                             double* h2d_s,
                                                             double* kernel_s);

    // Returns a column-major packed pinned buffer with ld == rowsB.
    std::uint8_t* KmeansGetPinnedStageBufferU8(int slot, int rowsB, int colsB);

    // Like KmeansUploadU8AndNormalizeHostPtrBAsync, but the input pointer is assumed to already be pinned
    // and column-major packed. No host-side packing is performed.
    DeviceMatF32View KmeansUploadU8AndNormalizePinnedAsync(int slot,
                                                           const std::uint8_t* pinned_B,
                                                           int ldB,
                                                           int rowsB,
                                                           int colsB,
                                                           double* h2d_s,
                                                           double* kernel_s);

    // Like KmeansUploadAndNormalizeHostPtrBAsync, but the input pointer is assumed to already be pinned
    // and column-major packed. No host-side packing is performed.
    DeviceMatF32View KmeansUploadAndNormalizePinnedAsync(int slot,
                                                         const float* pinned_B,
                                                         int ldB,
                                                         int rowsB,
                                                         int colsB,
                                                         double* h2d_s,
                                                         double* kernel_s);

    // Insert a wait on the compute stream for the given slot's ready event recorded by the copy stream.
    void KmeansComputeWaitForUpload(int slot);

    // Mark a ping-pong upload slot as "consumed" by the compute stream.
    // This allows the copy stream to safely reuse the same slot buffers for a future upload.
    //
    // Call this after the last kernel that reads the corresponding `KmeansUploadAndNormalize*Async(slot)` output.
    void KmeansMarkUploadSlotConsumed(int slot);

    // -------- K-means streaming profiling (CUDA events; no NVTX) --------
    // These timings are best-effort and intended for diagnosing out-of-core performance.
    // They are designed to avoid per-block synchronization so overlap is preserved.
    void KmeansTimingEnable(bool enable);
    void KmeansTimingReset();
    void KmeansTimingComputeSectionBegin(int slot);
    void KmeansTimingComputeSectionEnd(int slot);
    void KmeansTimingFlush(double* out_upload_h2d_s,
                           double* out_upload_kernel_s,
                           double* out_rvq_project_s,
                           double* out_assign_update_compute_s,
                           double* out_emit_codes_d2h_s,
                           double* out_wait_copy_s,
                           double* out_wait_compute_s);

    // -------- kmeans|| seeding helpers (GPU-resident init_samples) --------
    // These helpers are intended for kmeans|| initialization (seeding only), and operate on internal
    // seeding scratch that is independent from the streaming k-means X caches.
    DeviceMatF32View KmeansllEnsureSampleBuffer(int rows, int cols);
    void KmeansllReleaseSampleCache();
    DeviceMatF32View KmeansllSampleView() const;
    // Copy the currently active streaming slot matrix into the internal sample buffer at [dst_col0, dst_col0+cols).
    // Intended usage: after `KmeansComputeWaitForUpload(slot)` (and optional RVQ projection), call this to materialize
    // the normalized (or residual) init sample on device.
    void KmeansllCopyActiveSlotToSample(int slot, int dst_col0, int cols);

    DeviceVecF32View KmeansllEnsureBestDot(int n);
    void KmeansllSetBestDot(float value);
    void KmeansllUpdateBestDotMaxOffset(const DeviceVecF32View& tile_best, int offset);
    float KmeansllComputePhiFromBestDot();

    // Sample candidate indices using the most recently computed D2 buffer (from phi computation).
    // Returns a device view of the selected indices. `*out_count_host` is set synchronously.
    DeviceVecI32View KmeansllSampleCandidates(float oversample_l,
                                             float phi,
                                             std::uint64_t seed,
                                             int max_out,
                                             int* out_count_host);
    void KmeansllDownloadLastSampled(std::vector<int>* out);

    // Utility: upload a host index list to an internal device buffer (used for gather/download paths).
    DeviceVecI32View KmeansllUploadIndicesHost(const int* idx_host, int m);

    // Gather columns: out(:,j) = X(:, idx[j]) for j in [0,m).
    DeviceMatF32View KmeansllGatherColumns(const DeviceMatF32View& X,
                                          const DeviceVecI32View& idx,
                                          int m);

    // -------- RVQ init residual helpers (Prompt-06) --------
    // Upload the prior-layer codebooks [0, upto_layer) to device (internal buffers).
    // Intended for RVQ-like streaming init where each layer's k-means runs on the residual:
    //   r = x; for l < upto_layer: r -= (dot(r, C_l[code_l])) * C_l[code_l]
    void RvqSetPriorCodebooks(const std::vector<ColMajorMatrix<float>>& codebooks,
                              int upto_layer);

    // Apply prior-layer residual projections on the given uploaded/normalized slot matrix.
    //
    // This runs on the copy stream after `KmeansUpload*Async(slot, ...)` and then re-records the
    // slot-ready event so that `KmeansComputeWaitForUpload(slot)` covers the full residual build.
    //
    // `codes_host[l]` points to the host codes for this block (length = cols) and must remain valid
    // until the copy stream finishes the H2D copies.
    // `code_bytes_per_code[l]` must be 1, 2, or 4.
    void RvqProjectXnormBlockInCopyStreamAfterUpload(int slot,
                                                     const DeviceMatF32View& Xnorm_slot,
                                                     int cols,
                                                     int upto_layer,
                                                     const void* const* codes_host,
                                                     const int* code_bytes_per_code);

    // High-performance RVQ residual pipeline (Prompt-06):
    // - Stage codes (H2D) on the copy stream and extend the slot-ready event.
    // - Run residual projection on the compute stream after `KmeansComputeWaitForUpload(slot)`.
    void RvqStageCodesBlockInCopyStreamAfterUpload(int slot,
                                                   int cols,
                                                   int upto_layer,
                                                   const void* const* codes_host,
                                                   const int* code_bytes_per_code);

    void RvqProjectXnormBlockOnComputeStream(int slot,
                                             const DeviceMatF32View& Xnorm_slot,
                                             int cols,
                                             int upto_layer);

    // Best-effort: cache the full normalized X on device for k-means (X_norm columns).
    // When it succeeds, assignment/update can reuse it across iterations and avoid repeated H2D+normalize.
    // Returns false if memory is insufficient or inputs are invalid.
    bool KmeansTryCacheXnormOnDevice(const ColMajorMatrix<float>& X,
                                     DeviceMatF32View* out_full);

    // Best-effort: create a pinned host mirror of X (float32) for faster repeated H2D copies in k-means
    // when X does not fit on device. This avoids cudaHostRegister (no "guard" behavior) and copies X once.
    // Returns false if X is too large to pin or pinning fails.
    bool KmeansTryPinHostX(const ColMajorMatrix<float>& X,
                           const float** out_pinned_base);

    // Release the full X_norm cache used by k-means, if any. This helps keep GPU memory
    // available for later pipeline stages (beam/ICM/linkage).
    void KmeansReleaseXnormCache();

    // -------- K-means partial device cache (streaming) --------
    // Best-effort: cache a prefix window of normalized X_norm on device and reuse it across iterations.
    // This is intended for large out-of-core k-means where disk IO + H2D dominates, but full-device cache
    // does not fit. The cache is filled once per layer and then copied (D2D) into the normal ping-pong
    // slot buffers on demand so existing overlap/metrics pipelines keep working.
    //
    // Contract:
    // - The cached data is normalized X_norm (float32) and the corresponding base ||x||^2 (float32).
    // - The caller chooses the cached prefix window; this class only stores and copies.
    DeviceMatF32View KmeansEnsurePartialXnormCache(int rows, int cols);
    DeviceMatF32View KmeansPartialXnormCacheView() const;
    void KmeansCopyActiveSlotToPartialCache(int slot, int dst_col0, int cols);
    DeviceMatF32View KmeansCopyPartialCacheToSlot(int slot, int src_col0, int cols);
    void KmeansSetActiveSlot(int slot);
    void KmeansReleasePartialXnormCache();

    // Allocate/resize and zero per-cluster sum buffers for k-means accumulation.
    void KmeansResetSums(int d, int k);

    // -------- K-means argmax helpers (GPU) --------
    // Allocate/resize per-block argmax buffers (assignments + best dot) for `cols` samples.
    // These buffers live on device and are overwritten by subsequent calls.
    void KmeansEnsureBlockArgmax(int cols);

    // Copy the latest tile argmax outputs (device-only) into the block buffers at [col_base, col_base+tcols).
    void KmeansCopyTileArgmaxToBlock(const DeviceVecI32View& tile_assign,
                                     const DeviceVecF32View& tile_best,
                                     int col_base);

    // Views for the current block buffers.
    DeviceVecI32View KmeansBlockAssignView() const;
    DeviceVecF32View KmeansBlockBestView() const;

    // Download the current block buffers to host (one D2H per buffer; used for cost/hist/spill).
    void KmeansDownloadBlockArgmax(std::vector<int>* assign,
                                   std::vector<float>* best);

    // Download only best-dot for the current block (device -> host).
    // This is useful when CPU-side logic (e.g. anneal weight update) needs best-dot,
    // but we want to avoid downloading assignments.
    void KmeansDownloadBlockBestDot(std::vector<float>* best);

    // Async download of the current block assignments (int32) using a compute->copy staging path.
    //
    // This exists for Prompt-10 (emit last-iteration assignments as RVQ codes) without a hard
    // cudaStreamSynchronize on the compute stream.
    //
    // Contract:
    // - `slot` must be 0 or 1, and you must call Sync(slot) before reusing that slot.
    // - The returned pointer from Sync(slot) remains valid until the next Async(slot).
    void KmeansDownloadBlockAssignAsync(int slot);
    void KmeansDownloadBlockAssignSync(int slot, const int** out_assign_i32, int* out_cols);

    // Upload per-block weights to device and return a view (length must equal current block cols).
    DeviceVecF32View KmeansUploadBlockWeights(const std::vector<float>& weights);

    // -------- K-means anneal weights (device-resident, global) --------
    // Best-effort: allocate a global device weight array of length `n` and initialize to `initial_weight`.
    // Returns false on OOM (no throw); other failures may still throw.
    bool KmeansInitAnnealWeightsAll(std::int64_t n, float initial_weight);

    // View of the global weights for the current block range [col0, col0+cols).
    DeviceVecF32View KmeansAnnealWeightsBlockView(std::int64_t col0, int cols) const;

    // Reset per-iteration anneal totals on device:
    // - totals[0] = total_w
    // - totals[1] = total_cost = Σ w*(1-bestdot)
    void KmeansResetAnnealTotals();

    // Update global weights for the current block in-place using `KmeansBlockBestView()` and
    // accumulate anneal totals on device. Does not synchronize.
    void KmeansAnnealUpdateWeightsAndTotalsFromBlockBest(std::int64_t col0, int cols,
                                                         float threshold,
                                                         float annealed_factor,
                                                         float min_weight);
    // Fallback helper: accumulate anneal totals using the current block best-dot and a provided
    // device weights vector (length = cols). Does not synchronize.
    void KmeansAccumulateAnnealTotalsFromBlockBestDotAndWeights(const DeviceVecF32View& d_weights);

    // Synchronous totals download (16 bytes). Intended to be called once per iteration.
    void KmeansDownloadAnnealTotals(double* total_w, double* total_cost);
    // Async totals download on copy stream; pair with Sync*.
    void KmeansDownloadAnnealTotalsAsync();
    void KmeansSyncAnnealTotalsDownload(double* total_w, double* total_cost);

    // -------- K-means metric totals (device-only bestdot metrics) --------
    // Reset per-iteration metric totals:
    // - totals[0] = residual_l2_sum = Σ ||r||^2
    // - totals[1] = count = Σ 1
    void KmeansResetMetricTotals();
    // Accumulate mse_sum/mse_count from `KmeansBlockBestView()` (length must match `cols`).
    void KmeansAccumulateMseFromBlockBestDot(int cols);
    // Synchronous metrics download (16 bytes). Intended to be called once per iteration.
    void KmeansDownloadMetricTotals(double* mse_sum, double* mse_count);
    // Async metrics download on copy stream; pair with Sync*.
    void KmeansDownloadMetricTotalsAsync();
    void KmeansSyncMetricTotalsDownload(double* mse_sum, double* mse_count);

    // -------- K-means anneal fallback: outlier mask (device->host) --------
    // Compute an outlier mask from `KmeansBlockBestView()`:
    // mask[j] = (1-bestdot[j] > threshold) ? 1 : 0
    void KmeansComputeOutlierMaskFromBlockBest(int cols, float threshold);
    void KmeansDownloadOutlierMaskAsync(int cols);
    void KmeansSyncOutlierMaskDownload(std::uint8_t** host_ptr, int* host_bytes);

    // Accumulate sum_x += Xnorm(:,j), sum_w += 1 using atomic adds on GPU.
    // `assignments` are host-side, length = Xnorm.cols.
    void KmeansAccumulateUnitWeights(const DeviceMatF32View& Xnorm,
                                     const std::vector<int>& assignments);

    // Accumulate using device-side assignments (length = Xnorm.cols).
    void KmeansAccumulateUnitWeightsDeviceAssign(const DeviceMatF32View& Xnorm,
                                                 const DeviceVecI32View& d_assign);

    // Accumulate sum_x/sum_w using device-side assignments without per-element atomics:
    // sort-by-assignment + run-length encode + per-segment reduction.
    //
    // Intended for large-K streaming k-means where atomicAdd becomes the bottleneck.
    void KmeansAccumulateUnitWeightsDeviceAssignNoAtomic(const DeviceMatF32View& Xnorm,
                                                         const DeviceVecI32View& d_assign);

    // Accumulate sum_x += w[j] * Xnorm(:,j), sum_w += w[j] using atomic adds on GPU.
    // `assignments` and `weights` are host-side, length = Xnorm.cols.
    void KmeansAccumulateWeights(const DeviceMatF32View& Xnorm,
                                 const std::vector<int>& assignments,
                                 const std::vector<float>& weights);

    // Accumulate using device-side assignments and weights (length = Xnorm.cols).
    void KmeansAccumulateWeightsDeviceAssign(const DeviceMatF32View& Xnorm,
                                             const DeviceVecI32View& d_assign,
                                             const DeviceVecF32View& d_weights);

    void KmeansAccumulateWeightsDeviceAssignNoAtomic(const DeviceMatF32View& Xnorm,
                                                     const DeviceVecI32View& d_assign,
                                                     const DeviceVecF32View& d_weights);

    // Download current sums to host.
    void KmeansDownloadSums(ColMajorMatrix<float>* sum_x,
                            std::vector<float>* sum_w);

    // -------- K-means cost histogram helpers (GPU) --------
    // Reset/allocate a uint32 histogram with `bins` buckets on device.
    void KmeansResetCostHistogram(int bins);

    // Accumulate a cost histogram from device best-dot values (length = n samples).
    // cost = clamp(1 - dot, [0,2]) mapped into `bins` buckets.
    void KmeansAccumulateCostHistogramFromBestDot(const DeviceVecF32View& d_best_dot,
                                                  int bins);

    // Download histogram to host (size = bins).
    void KmeansDownloadCostHistogram(std::vector<std::uint32_t>* hist_u32);

    // Async histogram download helpers (device -> pinned -> host).
    // These exist to avoid a hard cudaStreamSynchronize on the compute stream.
    //
    // Usage:
    //   std::vector<uint32_t> hist(bins);
    //   KmeansDownloadCostHistogramAsync(hist.data(), bins);
    //   ... do unrelated CPU work ...
    //   KmeansDownloadCostHistogramSync();
    //
    // Contract: you must not call Async again before calling Sync.
    void KmeansDownloadCostHistogramAsync(std::uint32_t* host_dst, int bins);
    void KmeansDownloadCostHistogramSync();

    // -------- K-means large-K hier2 assignment (GPU) --------
    // Hier2 assignment for k-means / RVQ init:
    // 1) coarse_scores = C_coarse^T * Xnorm  (GEMM, on device)
    // 2) select top-L coarse ids per column (GPU kernel)
    // 3) scan only the corresponding fine center groups and output best flat id + dot (GPU kernel)
    //
    // Outputs are returned on host.
    void KmeansHier2AssignColsF32(const DeviceMatF32View& Xnorm,
                                 const ColMajorMatrix<float>& coarse_centers,
                                 const ColMajorMatrix<float>& fine_centers,
                                 int K_valid,
                                 int K1,
                                 int K2,
                                 int topL,
                                 std::vector<int>* assignments,
                                 std::vector<float>* best_values);

    // Same hier2 assignment as above, but leaves results on device:
    // - writes into the internal per-block argmax buffers (assign + best dot)
    // - does NOT copy anything back to host
    //
    // Use `KmeansDownloadBlockArgmax()` if host results are required.
    void KmeansHier2AssignColsF32Device(const DeviceMatF32View& Xnorm,
                                        const ColMajorMatrix<float>& coarse_centers,
                                        const ColMajorMatrix<float>& fine_centers,
                                        int K_valid,
                                        int K1,
                                        int K2,
                                        int topL);

private:
    int device_ = 0;
    bool allow_tf32_ = false;
    int cublas_workspace_mb_ = 0;
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace stlq
