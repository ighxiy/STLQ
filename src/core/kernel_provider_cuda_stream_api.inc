namespace stlq {

CudaStreamKernels::CudaStreamKernels(int device, bool allow_tf32, int cublas_workspace_mb) {
    device_ = device;
    allow_tf32_ = allow_tf32;
    cublas_workspace_mb_ = cublas_workspace_mb;
    try {
        impl_ = new Impl(device, allow_tf32, static_cast<std::size_t>(std::max(0, cublas_workspace_mb)));
    } catch (...) {
        delete impl_;
        impl_ = nullptr;
        throw;
    }
    // One-time log: report GEMM math mode for this provider (suppress repeated messages).
    {
        static std::once_flag s_ks_log_once;
        std::call_once(s_ks_log_once, [&]() {
            LogInfo(std::string("CudaStreamKernels: device=") + std::to_string(device) +
                    " cuBLAS math_mode=" + (allow_tf32 ? "TF32" : "DEFAULT_FP32") +
                    " workspace_mb=" + std::to_string(cublas_workspace_mb));
        });
    }
}

CudaStreamKernels::~CudaStreamKernels() {
    delete impl_;
    impl_ = nullptr;
}

void CudaStreamKernels::ResetCudaState() {
    // Make sure outstanding work is complete before tearing down CUDA resources.
    if (impl_) {
        try {
            impl_->Sync();
        } catch (...) {
            // Best-effort: continue with teardown.
        }
    }
    delete impl_;
    impl_ = nullptr;
    impl_ = new Impl(device_, allow_tf32_, static_cast<std::size_t>(std::max(0, cublas_workspace_mb_)));
}

void CudaStreamKernels::ConvertU8ToF32(const ColMajorMatrix<std::uint8_t>& in,
                                      ColMajorMatrix<float>* out) {
    impl_->ConvertU8ToF32(in, out);
}

void CudaStreamKernels::ConvertU8ToF32AndRotate(const ColMajorMatrix<std::uint8_t>& in,
                                                const ColMajorMatrix<float>& R,
                                                ColMajorMatrix<float>* out) {
    impl_->ConvertU8ToF32AndRotate(in, R, out);
}

void CudaStreamKernels::ConvertU8ToF32AndRotatePtr(const std::uint8_t* in,
                                                   int ld_in,
                                                   int rows,
                                                   int cols,
                                                   const ColMajorMatrix<float>& R,
                                                   ColMajorMatrix<float>* out) {
    impl_->ConvertU8ToF32AndRotatePtr(in, ld_in, rows, cols, R, out);
}

void CudaStreamKernels::Gemm(bool transA, bool transB,
                             float alpha,
                             const ColMajorMatrix<float>& A,
                             const ColMajorMatrix<float>& B,
                             float beta,
                             ColMajorMatrix<float>* C) {
    impl_->Gemm(transA, transB, alpha, A, B, beta, C);
}

void CudaStreamKernels::GemmDevice(bool transA, bool transB,
                                  float alpha,
                                  const ColMajorMatrix<float>& A,
                                  const ColMajorMatrix<float>& B,
                                  float beta,
                                  DeviceMatF32View* out) {
    impl_->GemmDevice(transA, transB, alpha, A, B, beta, out);
}

void CudaStreamKernels::GemmDeviceHostPtrB(bool transA, bool transB,
                                          float alpha,
                                          const ColMajorMatrix<float>& A,
                                          const float* B,
                                          int ldB,
                                          int rowsB,
                                          int colsB,
                                          float beta,
                                          DeviceMatF32View* out) {
    impl_->GemmDeviceHostPtrB(transA, transB, alpha, A, B, ldB, rowsB, colsB, beta, out);
}

bool CudaStreamKernels::EnsureDeviceF32(const ColMajorMatrix<float>& host,
                                       DeviceMatF32View* out) {
    return impl_->EnsureDeviceF32(host, out);
}

void CudaStreamKernels::GemmDeviceDevicePtrB(bool transA, bool transB,
                                             float alpha,
                                             const ColMajorMatrix<float>& A,
                                             const float* dB,
                                             int ldB,
                                             int rowsB,
                                             int colsB,
                                             float beta,
                                             DeviceMatF32View* out) {
    impl_->GemmDeviceDevicePtrB(transA, transB, alpha, A, dB, ldB, rowsB, colsB, beta, out);
}

void CudaStreamKernels::DownloadF32(const DeviceMatF32View& view,
                                   ColMajorMatrix<float>* out) {
    impl_->DownloadF32(view, out);
}

bool CudaStreamKernels::TryGetCachedDeviceF32(const ColMajorMatrix<float>& host,
                                              DeviceMatF32View* out) {
    if (out) {
        out->ptr = nullptr;
        out->rows = 0;
        out->cols = 0;
        out->ld = 0;
    }
    if (!impl_ || host.rows <= 0 || host.cols <= 0 || host.data.empty()) {
        return false;
    }
    const std::uint64_t stamp = CachedDeviceMatF32::Stamp(host);
    if (impl_->cached_host_ptr == host.data.data() &&
        impl_->cached_rows == host.rows &&
        impl_->cached_cols == host.cols &&
        impl_->cached_host_stamp == stamp &&
        impl_->d_cached.ptr) {
        if (out) {
            out->ptr = static_cast<const float*>(impl_->d_cached.ptr);
            out->rows = host.rows;
            out->cols = host.cols;
            out->ld = host.rows;
        }
        return true;
    }

    // Also allow returning pointers from the internal input caches (A/B/X caches).
    // This is important for large-root basic encoding where we:
    //  1) compute xC_small on device via GemmDevice(pre.C_small, X)
    //  2) immediately run CUDA beam search / ICM that can consume X from device.
    // GemmDevice uploads `B` into `B_cache`, so expose it here to avoid an extra H2D upload.
    auto cache_matches = [&](const CachedDeviceMatF32& cache) -> bool {
        return cache.host_ptr == host.data.data() &&
               cache.rows == host.rows &&
               cache.cols == host.cols &&
               cache.host_stamp == stamp &&
               cache.buf.ptr;
    };
    if (cache_matches(impl_->B_cache)) {
        if (out) {
            out->ptr = static_cast<const float*>(impl_->B_cache.buf.ptr);
            out->rows = host.rows;
            out->cols = host.cols;
            out->ld = host.rows;
        }
        return true;
    }
    if (cache_matches(impl_->X_cache)) {
        if (out) {
            out->ptr = static_cast<const float*>(impl_->X_cache.buf.ptr);
            out->rows = host.rows;
            out->cols = host.cols;
            out->ld = host.rows;
        }
        return true;
    }
    return false;
}

void CudaStreamKernels::ArgmaxColsF32(const DeviceMatF32View& scores,
                                      std::vector<int>* assignments,
                                      std::vector<float>* best_values) {
    impl_->ArgmaxColsF32(scores, assignments, best_values);
}

void CudaStreamKernels::ArgmaxColsF32Device(const DeviceMatF32View& scores,
                                           DeviceVecI32View* d_assign,
                                           DeviceVecF32View* d_best) {
    impl_->ArgmaxColsF32Device(scores, d_assign, d_best);
}

void CudaStreamKernels::Sync() {
    impl_->Sync();
}

void CudaStreamKernels::SyncCompute() {
    // Compute-stream-only sync: do not block on the auxiliary copy stream.
    // This is important for profiling k-means assignment/argmax, where the copy stream may
    // be concurrently uploading the next block (overlap), and a full Sync() would attribute
    // that copy work to unrelated kernel timers.
    if (!impl_) return;
    ThrowIf(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize (compute stream)");
}

void CudaStreamKernels::KmeansSetCollectMetrics(bool enable) {
    impl_->KmeansSetCollectMetrics(enable);
}

DeviceMatF32View CudaStreamKernels::KmeansUploadAndNormalizeHostPtrB(const float* B,
                                                                     int ldB,
                                                                     int rowsB,
                                                                     int colsB) {
    return impl_->KmeansUploadAndNormalizeHostPtrB(B, ldB, rowsB, colsB);
}

DeviceMatF32View CudaStreamKernels::KmeansUploadAndNormalizeHostPtrBTimed(const float* B,
                                                                          int ldB,
                                                                          int rowsB,
                                                                          int colsB,
                                                                          bool stage_to_pinned,
                                                                          double* stage_s,
                                                                          double* h2d_s,
                                                                          double* kernel_s) {
    return impl_->KmeansUploadAndNormalizeHostPtrBTimed(B, ldB, rowsB, colsB,
                                                        stage_to_pinned, stage_s, h2d_s, kernel_s);
}

DeviceMatF32View CudaStreamKernels::KmeansUploadAndNormalizeHostPtrBAsync(int slot,
                                                                          const float* B,
                                                                          int ldB,
                                                                          int rowsB,
                                                                          int colsB,
                                                                          bool stage_to_pinned,
                                                                          double* stage_s,
                                                                          double* h2d_s,
                                                                          double* kernel_s) {
    return impl_->KmeansUploadAndNormalizeHostPtrBAsync(slot, B, ldB, rowsB, colsB,
                                                        stage_to_pinned, stage_s, h2d_s, kernel_s);
}

float* CudaStreamKernels::KmeansGetPinnedStageBuffer(int slot, int rowsB, int colsB) {
    return impl_->KmeansGetPinnedStageBuffer(slot, rowsB, colsB);
}

DeviceMatF32View CudaStreamKernels::KmeansUploadU8AndNormalizeHostPtrBAsync(int slot,
                                                                            const std::uint8_t* B,
                                                                            int ldB,
                                                                            int rowsB,
                                                                            int colsB,
                                                                            bool stage_to_pinned,
                                                                            double* stage_s,
                                                                            double* h2d_s,
                                                                            double* kernel_s) {
    return impl_->KmeansUploadU8AndNormalizeHostPtrBAsync(slot, B, ldB, rowsB, colsB,
                                                          stage_to_pinned, stage_s, h2d_s, kernel_s);
}

std::uint8_t* CudaStreamKernels::KmeansGetPinnedStageBufferU8(int slot, int rowsB, int colsB) {
    return impl_->KmeansGetPinnedStageBufferU8(slot, rowsB, colsB);
}

DeviceMatF32View CudaStreamKernels::KmeansUploadU8AndNormalizePinnedAsync(int slot,
                                                                          const std::uint8_t* pinned_B,
                                                                          int ldB,
                                                                          int rowsB,
                                                                          int colsB,
                                                                          double* h2d_s,
                                                                          double* kernel_s) {
    return impl_->KmeansUploadU8AndNormalizePinnedAsync(slot, pinned_B, ldB, rowsB, colsB,
                                                        h2d_s, kernel_s);
}

DeviceMatF32View CudaStreamKernels::KmeansUploadAndNormalizePinnedAsync(int slot,
                                                                        const float* pinned_B,
                                                                        int ldB,
                                                                        int rowsB,
                                                                        int colsB,
                                                                        double* h2d_s,
                                                                        double* kernel_s) {
    return impl_->KmeansUploadAndNormalizePinnedAsync(slot, pinned_B, ldB, rowsB, colsB,
                                                      h2d_s, kernel_s);
}

void CudaStreamKernels::KmeansComputeWaitForUpload(int slot) {
    impl_->KmeansComputeWaitForUpload(slot);
}

void CudaStreamKernels::KmeansMarkUploadSlotConsumed(int slot) {
    impl_->KmeansMarkUploadSlotConsumed(slot);
}

void CudaStreamKernels::KmeansTimingEnable(bool enable) { impl_->KmeansTimingEnable(enable); }
void CudaStreamKernels::KmeansTimingReset() { impl_->KmeansTimingReset(); }
void CudaStreamKernels::KmeansTimingComputeSectionBegin(int slot) { impl_->KmeansTimingComputeSectionBegin(slot); }
void CudaStreamKernels::KmeansTimingComputeSectionEnd(int slot) { impl_->KmeansTimingComputeSectionEnd(slot); }
void CudaStreamKernels::KmeansTimingFlush(double* out_upload_h2d_s,
                                          double* out_upload_kernel_s,
                                          double* out_rvq_project_s,
                                          double* out_assign_update_compute_s,
                                          double* out_emit_codes_d2h_s,
                                          double* out_wait_copy_s,
                                          double* out_wait_compute_s) {
    impl_->KmeansTimingFlush(out_upload_h2d_s, out_upload_kernel_s,
                             out_rvq_project_s, out_assign_update_compute_s,
                             out_emit_codes_d2h_s, out_wait_copy_s, out_wait_compute_s);
}

DeviceMatF32View CudaStreamKernels::KmeansllEnsureSampleBuffer(int rows, int cols) {
    return impl_->KmeansllEnsureSampleBuffer(rows, cols);
}

void CudaStreamKernels::KmeansllReleaseSampleCache() { impl_->KmeansllReleaseSampleCache(); }

DeviceMatF32View CudaStreamKernels::KmeansllSampleView() const { return impl_->KmeansllSampleView(); }

void CudaStreamKernels::KmeansllCopyActiveSlotToSample(int slot, int dst_col0, int cols) {
    impl_->KmeansllCopyActiveSlotToSample(slot, dst_col0, cols);
}

DeviceVecF32View CudaStreamKernels::KmeansllEnsureBestDot(int n) { return impl_->KmeansllEnsureBestDot(n); }

void CudaStreamKernels::KmeansllSetBestDot(float value) { impl_->KmeansllSetBestDot(value); }

void CudaStreamKernels::KmeansllUpdateBestDotMaxOffset(const DeviceVecF32View& tile_best, int offset) {
    impl_->KmeansllUpdateBestDotMaxOffset(tile_best, offset);
}

float CudaStreamKernels::KmeansllComputePhiFromBestDot() { return impl_->KmeansllComputePhiFromBestDot(); }

DeviceVecI32View CudaStreamKernels::KmeansllSampleCandidates(float oversample_l,
                                                             float phi,
                                                             std::uint64_t seed,
                                                             int max_out,
                                                             int* out_count_host) {
    return impl_->KmeansllSampleCandidates(oversample_l, phi, seed, max_out, out_count_host);
}

void CudaStreamKernels::KmeansllDownloadLastSampled(std::vector<int>* out) {
    impl_->KmeansllDownloadLastSampled(out);
}

DeviceVecI32View CudaStreamKernels::KmeansllUploadIndicesHost(const int* idx_host, int m) {
    return impl_->KmeansllUploadIndicesHost(idx_host, m);
}

DeviceMatF32View CudaStreamKernels::KmeansllGatherColumns(const DeviceMatF32View& X,
                                                          const DeviceVecI32View& idx,
                                                          int m) {
    return impl_->KmeansllGatherColumns(X, idx, m);
}

void CudaStreamKernels::RvqSetPriorCodebooks(const std::vector<ColMajorMatrix<float>>& codebooks,
                                             int upto_layer) {
    impl_->RvqSetPriorCodebooks(codebooks, upto_layer);
}

void CudaStreamKernels::RvqProjectXnormBlockInCopyStreamAfterUpload(int slot,
                                                                    const DeviceMatF32View& Xnorm_slot,
                                                                    int cols,
                                                                    int upto_layer,
                                                                    const void* const* codes_host,
                                                                    const int* code_bytes_per_code) {
    impl_->RvqProjectXnormBlockInCopyStreamAfterUpload(slot, Xnorm_slot, cols, upto_layer,
                                                       codes_host, code_bytes_per_code);
}

void CudaStreamKernels::RvqStageCodesBlockInCopyStreamAfterUpload(int slot,
                                                                  int cols,
                                                                  int upto_layer,
                                                                  const void* const* codes_host,
                                                                  const int* code_bytes_per_code) {
    impl_->RvqStageCodesBlockInCopyStreamAfterUpload(slot, cols, upto_layer, codes_host, code_bytes_per_code);
}

void CudaStreamKernels::RvqProjectXnormBlockOnComputeStream(int slot,
                                                            const DeviceMatF32View& Xnorm_slot,
                                                            int cols,
                                                            int upto_layer) {
    impl_->RvqProjectXnormBlockOnComputeStream(slot, Xnorm_slot, cols, upto_layer);
}

bool CudaStreamKernels::KmeansTryCacheXnormOnDevice(const ColMajorMatrix<float>& X,
                                                   DeviceMatF32View* out_full) {
    return impl_->KmeansTryCacheXnormOnDevice(X, out_full);
}

bool CudaStreamKernels::KmeansTryPinHostX(const ColMajorMatrix<float>& X,
                                         const float** out_pinned_base) {
    return impl_->KmeansTryPinHostX(X, out_pinned_base);
}

void CudaStreamKernels::KmeansReleaseXnormCache() {
    impl_->KmeansReleaseXnormCache();
}

DeviceMatF32View CudaStreamKernels::KmeansEnsurePartialXnormCache(int rows, int cols) {
    return impl_->KmeansEnsurePartialXnormCache(rows, cols);
}

DeviceMatF32View CudaStreamKernels::KmeansPartialXnormCacheView() const {
    return impl_->KmeansPartialXnormCacheView();
}

void CudaStreamKernels::KmeansCopyActiveSlotToPartialCache(int slot, int dst_col0, int cols) {
    impl_->KmeansCopyActiveSlotToPartialCache(slot, dst_col0, cols);
}

DeviceMatF32View CudaStreamKernels::KmeansCopyPartialCacheToSlot(int slot, int src_col0, int cols) {
    return impl_->KmeansCopyPartialCacheToSlot(slot, src_col0, cols);
}

void CudaStreamKernels::KmeansSetActiveSlot(int slot) {
    impl_->KmeansSetActiveSlot(slot);
}

void CudaStreamKernels::KmeansReleasePartialXnormCache() {
    impl_->KmeansReleasePartialXnormCache();
}

void CudaStreamKernels::KmeansEnsureBlockArgmax(int cols) {
    impl_->KmeansEnsureBlockArgmax(cols);
}

void CudaStreamKernels::KmeansCopyTileArgmaxToBlock(const DeviceVecI32View& tile_assign,
                                                    const DeviceVecF32View& tile_best,
                                                    int col_base) {
    impl_->KmeansCopyTileArgmaxToBlock(tile_assign, tile_best, col_base);
}

DeviceVecI32View CudaStreamKernels::KmeansBlockAssignView() const {
    return impl_->KmeansBlockAssignView();
}

DeviceVecF32View CudaStreamKernels::KmeansBlockBestView() const {
    return impl_->KmeansBlockBestView();
}

void CudaStreamKernels::KmeansDownloadBlockArgmax(std::vector<int>* assign,
                                                  std::vector<float>* best) {
    impl_->KmeansDownloadBlockArgmax(assign, best);
}

void CudaStreamKernels::KmeansDownloadBlockBestDot(std::vector<float>* best) {
    impl_->KmeansDownloadBlockBestDot(best);
}

void CudaStreamKernels::KmeansDownloadBlockAssignAsync(int slot) {
    impl_->KmeansDownloadBlockAssignAsync(slot);
}

void CudaStreamKernels::KmeansDownloadBlockAssignSync(int slot, const int** out_assign_i32, int* out_cols) {
    impl_->KmeansDownloadBlockAssignSync(slot, out_assign_i32, out_cols);
}

DeviceVecF32View CudaStreamKernels::KmeansUploadBlockWeights(const std::vector<float>& weights) {
    return impl_->KmeansUploadBlockWeights(weights);
}

bool CudaStreamKernels::KmeansInitAnnealWeightsAll(std::int64_t n, float initial_weight) {
    return impl_->KmeansInitAnnealWeightsAll(n, initial_weight);
}

DeviceVecF32View CudaStreamKernels::KmeansAnnealWeightsBlockView(std::int64_t col0, int cols) const {
    return impl_->KmeansAnnealWeightsBlockView(col0, cols);
}

void CudaStreamKernels::KmeansResetAnnealTotals() {
    impl_->KmeansResetAnnealTotals();
}

void CudaStreamKernels::KmeansAnnealUpdateWeightsAndTotalsFromBlockBest(std::int64_t col0, int cols,
                                                                        float threshold,
                                                                        float annealed_factor,
                                                                        float min_weight) {
    impl_->KmeansAnnealUpdateWeightsAndTotalsFromBlockBest(col0, cols, threshold, annealed_factor, min_weight);
}

void CudaStreamKernels::KmeansAccumulateAnnealTotalsFromBlockBestDotAndWeights(const DeviceVecF32View& d_weights) {
    impl_->KmeansAccumulateAnnealTotalsFromBlockBestDotAndWeights(d_weights);
}

void CudaStreamKernels::KmeansDownloadAnnealTotals(double* total_w, double* total_cost) {
    impl_->KmeansDownloadAnnealTotals(total_w, total_cost);
}

void CudaStreamKernels::KmeansDownloadAnnealTotalsAsync() {
    impl_->KmeansDownloadAnnealTotalsAsync();
}

void CudaStreamKernels::KmeansSyncAnnealTotalsDownload(double* total_w, double* total_cost) {
    impl_->KmeansSyncAnnealTotalsDownload(total_w, total_cost);
}

void CudaStreamKernels::KmeansResetMetricTotals() {
    impl_->KmeansResetMetricTotals();
}

void CudaStreamKernels::KmeansAccumulateMseFromBlockBestDot(int cols) {
    impl_->KmeansAccumulateMseFromBlockBestDot(cols);
}

void CudaStreamKernels::KmeansDownloadMetricTotals(double* mse_sum, double* mse_count) {
    impl_->KmeansDownloadMetricTotals(mse_sum, mse_count);
}

void CudaStreamKernels::KmeansDownloadMetricTotalsAsync() {
    impl_->KmeansDownloadMetricTotalsAsync();
}

void CudaStreamKernels::KmeansSyncMetricTotalsDownload(double* mse_sum, double* mse_count) {
    impl_->KmeansSyncMetricTotalsDownload(mse_sum, mse_count);
}

void CudaStreamKernels::KmeansComputeOutlierMaskFromBlockBest(int cols, float threshold) {
    impl_->KmeansComputeOutlierMaskFromBlockBest(cols, threshold);
}

void CudaStreamKernels::KmeansDownloadOutlierMaskAsync(int cols) {
    impl_->KmeansDownloadOutlierMaskAsync(cols);
}

void CudaStreamKernels::KmeansSyncOutlierMaskDownload(std::uint8_t** host_ptr, int* host_bytes) {
    impl_->KmeansSyncOutlierMaskDownload(host_ptr, host_bytes);
}

void CudaStreamKernels::KmeansResetCostHistogram(int bins) {
    impl_->KmeansResetCostHistogram(bins);
}

void CudaStreamKernels::KmeansAccumulateCostHistogramFromBestDot(const DeviceVecF32View& d_best_dot,
                                                                 int bins) {
    impl_->KmeansAccumulateCostHistogramFromBestDot(d_best_dot, bins);
}

void CudaStreamKernels::KmeansDownloadCostHistogram(std::vector<std::uint32_t>* hist_u32) {
    impl_->KmeansDownloadCostHistogram(hist_u32);
}

void CudaStreamKernels::KmeansDownloadCostHistogramAsync(std::uint32_t* host_dst, int bins) {
    impl_->KmeansDownloadCostHistogramAsync(host_dst, bins);
}

void CudaStreamKernels::KmeansDownloadCostHistogramSync() {
    impl_->KmeansDownloadCostHistogramSync();
}

void CudaStreamKernels::KmeansResetSums(int d, int k) {
    impl_->KmeansResetSums(d, k);
}

void CudaStreamKernels::KmeansAccumulateUnitWeights(const DeviceMatF32View& Xnorm,
                                                    const std::vector<int>& assignments) {
    impl_->KmeansAccumulateUnitWeights(Xnorm, assignments);
}

void CudaStreamKernels::KmeansAccumulateUnitWeightsDeviceAssign(const DeviceMatF32View& Xnorm,
                                                                const DeviceVecI32View& d_assign) {
    impl_->KmeansAccumulateUnitWeightsDeviceAssign(Xnorm, d_assign);
}

void CudaStreamKernels::KmeansAccumulateUnitWeightsDeviceAssignNoAtomic(const DeviceMatF32View& Xnorm,
                                                                        const DeviceVecI32View& d_assign) {
    impl_->KmeansAccumulateUnitWeightsDeviceAssignNoAtomic(Xnorm, d_assign);
}

void CudaStreamKernels::KmeansAccumulateWeights(const DeviceMatF32View& Xnorm,
                                                const std::vector<int>& assignments,
                                                const std::vector<float>& weights) {
    impl_->KmeansAccumulateWeights(Xnorm, assignments, weights);
}

void CudaStreamKernels::KmeansAccumulateWeightsDeviceAssign(const DeviceMatF32View& Xnorm,
                                                            const DeviceVecI32View& d_assign,
                                                            const DeviceVecF32View& d_weights) {
    impl_->KmeansAccumulateWeightsDeviceAssign(Xnorm, d_assign, d_weights);
}

void CudaStreamKernels::KmeansAccumulateWeightsDeviceAssignNoAtomic(const DeviceMatF32View& Xnorm,
                                                                    const DeviceVecI32View& d_assign,
                                                                    const DeviceVecF32View& d_weights) {
    impl_->KmeansAccumulateWeightsDeviceAssignNoAtomic(Xnorm, d_assign, d_weights);
}

void CudaStreamKernels::KmeansDownloadSums(ColMajorMatrix<float>* sum_x,
                                          std::vector<float>* sum_w) {
    impl_->KmeansDownloadSums(sum_x, sum_w);
}

void CudaStreamKernels::KmeansHier2AssignColsF32(const DeviceMatF32View& Xnorm,
                                                const ColMajorMatrix<float>& coarse_centers,
                                                const ColMajorMatrix<float>& fine_centers,
                                                int K_valid,
                                                int K1,
                                                int K2,
                                                int topL,
                                                std::vector<int>* assignments,
                                                std::vector<float>* best_values) {
    if (!assignments || !best_values) {
        throw std::runtime_error("KmeansHier2AssignColsF32: output is null.");
    }
    assignments->clear();
    best_values->clear();

    if (!Xnorm.ptr || Xnorm.rows <= 0 || Xnorm.cols <= 0 || Xnorm.ld < Xnorm.rows) {
        throw std::runtime_error("KmeansHier2AssignColsF32: invalid Xnorm device view.");
    }
    if (K_valid <= 0 || K1 <= 0 || K2 <= 0 || topL <= 0) {
        throw std::runtime_error("KmeansHier2AssignColsF32: invalid K/K1/K2/topL.");
    }
    if (coarse_centers.rows != Xnorm.rows || coarse_centers.cols != K1) {
        throw std::runtime_error("KmeansHier2AssignColsF32: coarse center shape mismatch.");
    }
    if (fine_centers.rows != Xnorm.rows || fine_centers.cols != K_valid) {
        throw std::runtime_error("KmeansHier2AssignColsF32: fine center shape mismatch.");
    }
    if (topL > 2) topL = 2;

    // Reuse the device-path implementation, then download.
    KmeansHier2AssignColsF32Device(Xnorm, coarse_centers, fine_centers,
                                  K_valid, K1, K2, topL);
    // Download from the standard per-block buffers (uses internal pinned host buffers).
    KmeansDownloadBlockArgmax(assignments, best_values);
}

void CudaStreamKernels::KmeansHier2AssignColsF32Device(const DeviceMatF32View& Xnorm,
                                                      const ColMajorMatrix<float>& coarse_centers,
                                                      const ColMajorMatrix<float>& fine_centers,
                                                      int K_valid,
                                                      int K1,
                                                      int K2,
                                                      int topL) {
    if (!Xnorm.ptr || Xnorm.rows <= 0 || Xnorm.cols <= 0 || Xnorm.ld < Xnorm.rows) {
        throw std::runtime_error("KmeansHier2AssignColsF32Device: invalid Xnorm device view.");
    }
    if (K_valid <= 0 || K1 <= 0 || K2 <= 0 || topL <= 0) {
        throw std::runtime_error("KmeansHier2AssignColsF32Device: invalid K/K1/K2/topL.");
    }
    if (coarse_centers.rows != Xnorm.rows || coarse_centers.cols != K1) {
        throw std::runtime_error("KmeansHier2AssignColsF32Device: coarse center shape mismatch.");
    }
    if (fine_centers.rows != Xnorm.rows || fine_centers.cols != K_valid) {
        throw std::runtime_error("KmeansHier2AssignColsF32Device: fine center shape mismatch.");
    }
    if (topL > 2) topL = 2;

    const int d = Xnorm.rows;
    const int cols = Xnorm.cols;

    // Ensure fine centers on device (reuse existing cached upload path).
    impl_->B_cache.UploadIfNeeded(fine_centers, impl_->stream);
    const float* d_fine = impl_->B_cache.dptr();

    // Coarse GEMM: scores = C_coarse^T * Xnorm  => (K1 x cols)
    DeviceMatF32View scores;
    impl_->GemmDeviceDevicePtrB(/*transA=*/true, /*transB=*/false,
                                /*alpha=*/1.0f,
                                coarse_centers,
                                Xnorm.ptr,
                                /*ldB=*/Xnorm.ld,
                                /*rowsB=*/d,
                                /*colsB=*/cols,
                                /*beta=*/0.0f,
                                &scores);

    // Top-L coarse selection.
    const std::size_t idx_bytes =
        static_cast<std::size_t>(cols) * static_cast<std::size_t>(topL) * sizeof(int);
    const std::size_t val_bytes =
        static_cast<std::size_t>(cols) * static_cast<std::size_t>(topL) * sizeof(float);
    impl_->kmeans_hier2_topk_idx.Ensure(idx_bytes);
    impl_->kmeans_hier2_topk_val.Ensure(val_bytes);
    int* d_topk_idx = static_cast<int*>(impl_->kmeans_hier2_topk_idx.ptr);
    float* d_topk_val = static_cast<float*>(impl_->kmeans_hier2_topk_val.ptr);

    if (topL == 1) {
        // Reuse argmax kernel: out_idx/out_val length = cols.
        constexpr int threads = 256;
        ArgmaxColsKernel<<<cols, threads, 0, impl_->stream>>>(scores.ptr, scores.ld, K1, cols,
                                                              d_topk_idx, d_topk_val);
    } else {
        constexpr int threads = 256;
        const std::size_t shmem =
            static_cast<std::size_t>(threads) * (sizeof(float) * 2 + sizeof(int) * 2);
        Top2ColsKernel<<<cols, threads, shmem, impl_->stream>>>(scores.ptr, scores.ld, K1, cols,
                                                                d_topk_idx, d_topk_val);
    }
    ThrowIf(cudaGetLastError(), "TopL coarse kernel launch (device)");

    // Fine scan.
    impl_->d_assign.Ensure(static_cast<std::size_t>(cols) * sizeof(int));
    impl_->d_best.Ensure(static_cast<std::size_t>(cols) * sizeof(float));
    int* d_assign = static_cast<int*>(impl_->d_assign.ptr);
    float* d_best = static_cast<float*>(impl_->d_best.ptr);

    constexpr int threads = 128;  // 4 warps per block
    if (topL == 1) {
        Hier2FineAssignKernel<1><<<cols, threads, 0, impl_->stream>>>(
            Xnorm.ptr, Xnorm.ld, d, cols, d_fine, /*ldF=*/d, K_valid, d_topk_idx, K2, d_assign,
            d_best);
    } else {
        Hier2FineAssignKernel<2><<<cols, threads, 0, impl_->stream>>>(
            Xnorm.ptr, Xnorm.ld, d, cols, d_fine, /*ldF=*/d, K_valid, d_topk_idx, K2, d_assign,
            d_best);
    }
    ThrowIf(cudaGetLastError(), "Hier2FineAssignKernel launch (device)");

    // Publish device results into the standard per-block argmax buffers so downstream helpers
    // (histogram/weights updates) can reuse them without extra D2H.
    impl_->KmeansEnsureBlockArgmax(cols);
    DeviceVecI32View v_assign;
    v_assign.ptr = d_assign;
    v_assign.n = cols;
    DeviceVecF32View v_best;
    v_best.ptr = d_best;
    v_best.n = cols;
    impl_->KmeansCopyTileArgmaxToBlock(v_assign, v_best, /*col_base=*/0);
}

}  // namespace stlq
