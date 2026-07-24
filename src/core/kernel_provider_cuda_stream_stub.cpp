#include "stlq/core/kernel_provider_cuda_stream.h"

#include <stdexcept>
#include <string>

namespace stlq {

namespace {
[[noreturn]] void ThrowCudaDisabled(const char* fn) {
    throw std::runtime_error(std::string(fn) + ": this build has STLQ_ENABLE_CUDA=OFF.");
}

DeviceMatF32View EmptyMat() { return DeviceMatF32View{}; }
DeviceVecI32View EmptyVecI32() { return DeviceVecI32View{}; }
DeviceVecF32View EmptyVecF32() { return DeviceVecF32View{}; }
}  // namespace

CudaStreamKernels::CudaStreamKernels(int device, bool allow_tf32, int cublas_workspace_mb) {
    device_ = device;
    allow_tf32_ = allow_tf32;
    cublas_workspace_mb_ = cublas_workspace_mb;
    impl_ = nullptr;
    ThrowCudaDisabled("CudaStreamKernels::CudaStreamKernels");
}

CudaStreamKernels::~CudaStreamKernels() {
    impl_ = nullptr;
}

void CudaStreamKernels::ResetCudaState() {
    ThrowCudaDisabled("CudaStreamKernels::ResetCudaState");
}

void CudaStreamKernels::ConvertU8ToF32(const ColMajorMatrix<std::uint8_t>& in,
                                      ColMajorMatrix<float>* out) {
    (void)in;
    (void)out;
    ThrowCudaDisabled("CudaStreamKernels::ConvertU8ToF32");
}

void CudaStreamKernels::ConvertU8ToF32AndRotate(const ColMajorMatrix<std::uint8_t>& in,
                                               const ColMajorMatrix<float>& R,
                                               ColMajorMatrix<float>* out) {
    (void)in;
    (void)R;
    (void)out;
    ThrowCudaDisabled("CudaStreamKernels::ConvertU8ToF32AndRotate");
}

void CudaStreamKernels::ConvertU8ToF32AndRotatePtr(const std::uint8_t* in,
                                                  int ld_in,
                                                  int rows,
                                                  int cols,
                                                  const ColMajorMatrix<float>& R,
                                                  ColMajorMatrix<float>* out) {
    (void)in;
    (void)ld_in;
    (void)rows;
    (void)cols;
    (void)R;
    (void)out;
    ThrowCudaDisabled("CudaStreamKernels::ConvertU8ToF32AndRotatePtr");
}

void CudaStreamKernels::Gemm(bool transA,
                             bool transB,
                             float alpha,
                             const ColMajorMatrix<float>& A,
                             const ColMajorMatrix<float>& B,
                             float beta,
                             ColMajorMatrix<float>* C) {
    (void)transA;
    (void)transB;
    (void)alpha;
    (void)A;
    (void)B;
    (void)beta;
    (void)C;
    ThrowCudaDisabled("CudaStreamKernels::Gemm");
}

void CudaStreamKernels::GemmDevice(bool transA,
                                  bool transB,
                                  float alpha,
                                  const ColMajorMatrix<float>& A,
                                  const ColMajorMatrix<float>& B,
                                  float beta,
                                  DeviceMatF32View* out) {
    (void)transA;
    (void)transB;
    (void)alpha;
    (void)A;
    (void)B;
    (void)beta;
    if (out) *out = EmptyMat();
    ThrowCudaDisabled("CudaStreamKernels::GemmDevice");
}

void CudaStreamKernels::GemmDeviceHostPtrB(bool transA,
                                          bool transB,
                                          float alpha,
                                          const ColMajorMatrix<float>& A,
                                          const float* B,
                                          int ldB,
                                          int rowsB,
                                          int colsB,
                                          float beta,
                                          DeviceMatF32View* out) {
    (void)transA;
    (void)transB;
    (void)alpha;
    (void)A;
    (void)B;
    (void)ldB;
    (void)rowsB;
    (void)colsB;
    (void)beta;
    if (out) *out = EmptyMat();
    ThrowCudaDisabled("CudaStreamKernels::GemmDeviceHostPtrB");
}

bool CudaStreamKernels::EnsureDeviceF32(const ColMajorMatrix<float>& host, DeviceMatF32View* out) {
    (void)host;
    if (out) *out = EmptyMat();
    ThrowCudaDisabled("CudaStreamKernels::EnsureDeviceF32");
}

void CudaStreamKernels::GemmDeviceDevicePtrB(bool transA,
                                             bool transB,
                                             float alpha,
                                             const ColMajorMatrix<float>& A,
                                             const float* dB,
                                             int ldB,
                                             int rowsB,
                                             int colsB,
                                             float beta,
                                             DeviceMatF32View* out) {
    (void)transA;
    (void)transB;
    (void)alpha;
    (void)A;
    (void)dB;
    (void)ldB;
    (void)rowsB;
    (void)colsB;
    (void)beta;
    if (out) *out = EmptyMat();
    ThrowCudaDisabled("CudaStreamKernels::GemmDeviceDevicePtrB");
}

void CudaStreamKernels::DownloadF32(const DeviceMatF32View& view, ColMajorMatrix<float>* out) {
    (void)view;
    (void)out;
    ThrowCudaDisabled("CudaStreamKernels::DownloadF32");
}

bool CudaStreamKernels::TryGetCachedDeviceF32(const ColMajorMatrix<float>& host, DeviceMatF32View* out) {
    (void)host;
    if (out) *out = EmptyMat();
    ThrowCudaDisabled("CudaStreamKernels::TryGetCachedDeviceF32");
}

void CudaStreamKernels::ArgmaxColsF32(const DeviceMatF32View& scores,
                                     std::vector<int>* assignments,
                                     std::vector<float>* best_values) {
    (void)scores;
    (void)assignments;
    (void)best_values;
    ThrowCudaDisabled("CudaStreamKernels::ArgmaxColsF32");
}

void CudaStreamKernels::ArgmaxColsF32Device(const DeviceMatF32View& scores,
                                           DeviceVecI32View* d_assign,
                                           DeviceVecF32View* d_best) {
    (void)scores;
    if (d_assign) *d_assign = EmptyVecI32();
    if (d_best) *d_best = EmptyVecF32();
    ThrowCudaDisabled("CudaStreamKernels::ArgmaxColsF32Device");
}

void CudaStreamKernels::Sync() { ThrowCudaDisabled("CudaStreamKernels::Sync"); }
void CudaStreamKernels::SyncCompute() { ThrowCudaDisabled("CudaStreamKernels::SyncCompute"); }

void CudaStreamKernels::KmeansSetCollectMetrics(bool enable) {
    (void)enable;
    ThrowCudaDisabled("CudaStreamKernels::KmeansSetCollectMetrics");
}

DeviceMatF32View CudaStreamKernels::KmeansUploadAndNormalizeHostPtrB(const float* B,
                                                                     int ldB,
                                                                     int rowsB,
                                                                     int colsB) {
    (void)B;
    (void)ldB;
    (void)rowsB;
    (void)colsB;
    ThrowCudaDisabled("CudaStreamKernels::KmeansUploadAndNormalizeHostPtrB");
}

DeviceMatF32View CudaStreamKernels::KmeansUploadAndNormalizeHostPtrBTimed(const float* B,
                                                                          int ldB,
                                                                          int rowsB,
                                                                          int colsB,
                                                                          bool stage_to_pinned,
                                                                          double* stage_s,
                                                                          double* h2d_s,
                                                                          double* kernel_s) {
    (void)B;
    (void)ldB;
    (void)rowsB;
    (void)colsB;
    (void)stage_to_pinned;
    if (stage_s) *stage_s = 0.0;
    if (h2d_s) *h2d_s = 0.0;
    if (kernel_s) *kernel_s = 0.0;
    ThrowCudaDisabled("CudaStreamKernels::KmeansUploadAndNormalizeHostPtrBTimed");
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
    (void)slot;
    (void)B;
    (void)ldB;
    (void)rowsB;
    (void)colsB;
    (void)stage_to_pinned;
    if (stage_s) *stage_s = 0.0;
    if (h2d_s) *h2d_s = 0.0;
    if (kernel_s) *kernel_s = 0.0;
    ThrowCudaDisabled("CudaStreamKernels::KmeansUploadAndNormalizeHostPtrBAsync");
}

float* CudaStreamKernels::KmeansGetPinnedStageBuffer(int slot, int rowsB, int colsB) {
    (void)slot;
    (void)rowsB;
    (void)colsB;
    ThrowCudaDisabled("CudaStreamKernels::KmeansGetPinnedStageBuffer");
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
    (void)slot;
    (void)B;
    (void)ldB;
    (void)rowsB;
    (void)colsB;
    (void)stage_to_pinned;
    if (stage_s) *stage_s = 0.0;
    if (h2d_s) *h2d_s = 0.0;
    if (kernel_s) *kernel_s = 0.0;
    ThrowCudaDisabled("CudaStreamKernels::KmeansUploadU8AndNormalizeHostPtrBAsync");
}

std::uint8_t* CudaStreamKernels::KmeansGetPinnedStageBufferU8(int slot, int rowsB, int colsB) {
    (void)slot;
    (void)rowsB;
    (void)colsB;
    ThrowCudaDisabled("CudaStreamKernels::KmeansGetPinnedStageBufferU8");
}

DeviceMatF32View CudaStreamKernels::KmeansUploadU8AndNormalizePinnedAsync(int slot,
                                                                         const std::uint8_t* pinned_B,
                                                                         int ldB,
                                                                         int rowsB,
                                                                         int colsB,
                                                                         double* h2d_s,
                                                                         double* kernel_s) {
    (void)slot;
    (void)pinned_B;
    (void)ldB;
    (void)rowsB;
    (void)colsB;
    if (h2d_s) *h2d_s = 0.0;
    if (kernel_s) *kernel_s = 0.0;
    ThrowCudaDisabled("CudaStreamKernels::KmeansUploadU8AndNormalizePinnedAsync");
}

DeviceMatF32View CudaStreamKernels::KmeansUploadAndNormalizePinnedAsync(int slot,
                                                                       const float* pinned_B,
                                                                       int ldB,
                                                                       int rowsB,
                                                                       int colsB,
                                                                       double* h2d_s,
                                                                       double* kernel_s) {
    (void)slot;
    (void)pinned_B;
    (void)ldB;
    (void)rowsB;
    (void)colsB;
    if (h2d_s) *h2d_s = 0.0;
    if (kernel_s) *kernel_s = 0.0;
    ThrowCudaDisabled("CudaStreamKernels::KmeansUploadAndNormalizePinnedAsync");
}

void CudaStreamKernels::KmeansComputeWaitForUpload(int slot) {
    (void)slot;
    ThrowCudaDisabled("CudaStreamKernels::KmeansComputeWaitForUpload");
}

void CudaStreamKernels::KmeansMarkUploadSlotConsumed(int slot) {
    (void)slot;
    ThrowCudaDisabled("CudaStreamKernels::KmeansMarkUploadSlotConsumed");
}

void CudaStreamKernels::KmeansTimingEnable(bool enable) {
    (void)enable;
    ThrowCudaDisabled("CudaStreamKernels::KmeansTimingEnable");
}

void CudaStreamKernels::KmeansTimingReset() { ThrowCudaDisabled("CudaStreamKernels::KmeansTimingReset"); }

void CudaStreamKernels::KmeansTimingComputeSectionBegin(int slot) {
    (void)slot;
    ThrowCudaDisabled("CudaStreamKernels::KmeansTimingComputeSectionBegin");
}

void CudaStreamKernels::KmeansTimingComputeSectionEnd(int slot) {
    (void)slot;
    ThrowCudaDisabled("CudaStreamKernels::KmeansTimingComputeSectionEnd");
}

void CudaStreamKernels::KmeansTimingFlush(double* out_upload_h2d_s,
                                         double* out_upload_kernel_s,
                                         double* out_rvq_project_s,
                                         double* out_assign_update_compute_s,
                                         double* out_emit_codes_d2h_s,
                                         double* out_wait_copy_s,
                                         double* out_wait_compute_s) {
    if (out_upload_h2d_s) *out_upload_h2d_s = 0.0;
    if (out_upload_kernel_s) *out_upload_kernel_s = 0.0;
    if (out_rvq_project_s) *out_rvq_project_s = 0.0;
    if (out_assign_update_compute_s) *out_assign_update_compute_s = 0.0;
    if (out_emit_codes_d2h_s) *out_emit_codes_d2h_s = 0.0;
    if (out_wait_copy_s) *out_wait_copy_s = 0.0;
    if (out_wait_compute_s) *out_wait_compute_s = 0.0;
    ThrowCudaDisabled("CudaStreamKernels::KmeansTimingFlush");
}

DeviceMatF32View CudaStreamKernels::KmeansllEnsureSampleBuffer(int rows, int cols) {
    (void)rows;
    (void)cols;
    ThrowCudaDisabled("CudaStreamKernels::KmeansllEnsureSampleBuffer");
}

void CudaStreamKernels::KmeansllReleaseSampleCache() { ThrowCudaDisabled("CudaStreamKernels::KmeansllReleaseSampleCache"); }

DeviceMatF32View CudaStreamKernels::KmeansllSampleView() const { ThrowCudaDisabled("CudaStreamKernels::KmeansllSampleView"); }

void CudaStreamKernels::KmeansllCopyActiveSlotToSample(int slot, int dst_col0, int cols) {
    (void)slot;
    (void)dst_col0;
    (void)cols;
    ThrowCudaDisabled("CudaStreamKernels::KmeansllCopyActiveSlotToSample");
}

DeviceVecF32View CudaStreamKernels::KmeansllEnsureBestDot(int n) {
    (void)n;
    ThrowCudaDisabled("CudaStreamKernels::KmeansllEnsureBestDot");
}

void CudaStreamKernels::KmeansllSetBestDot(float value) {
    (void)value;
    ThrowCudaDisabled("CudaStreamKernels::KmeansllSetBestDot");
}

void CudaStreamKernels::KmeansllUpdateBestDotMaxOffset(const DeviceVecF32View& tile_best, int offset) {
    (void)tile_best;
    (void)offset;
    ThrowCudaDisabled("CudaStreamKernels::KmeansllUpdateBestDotMaxOffset");
}

float CudaStreamKernels::KmeansllComputePhiFromBestDot() { ThrowCudaDisabled("CudaStreamKernels::KmeansllComputePhiFromBestDot"); }

DeviceVecI32View CudaStreamKernels::KmeansllSampleCandidates(float oversample_l,
                                                            float phi,
                                                            std::uint64_t seed,
                                                            int max_out,
                                                            int* out_count_host) {
    (void)oversample_l;
    (void)phi;
    (void)seed;
    (void)max_out;
    if (out_count_host) *out_count_host = 0;
    ThrowCudaDisabled("CudaStreamKernels::KmeansllSampleCandidates");
}

void CudaStreamKernels::KmeansllDownloadLastSampled(std::vector<int>* out) {
    if (out) out->clear();
    ThrowCudaDisabled("CudaStreamKernels::KmeansllDownloadLastSampled");
}

DeviceVecI32View CudaStreamKernels::KmeansllUploadIndicesHost(const int* idx_host, int m) {
    (void)idx_host;
    (void)m;
    ThrowCudaDisabled("CudaStreamKernels::KmeansllUploadIndicesHost");
}

DeviceMatF32View CudaStreamKernels::KmeansllGatherColumns(const DeviceMatF32View& X,
                                                         const DeviceVecI32View& idx,
                                                         int m) {
    (void)X;
    (void)idx;
    (void)m;
    ThrowCudaDisabled("CudaStreamKernels::KmeansllGatherColumns");
}

void CudaStreamKernels::RvqSetPriorCodebooks(const std::vector<ColMajorMatrix<float>>& codebooks, int upto_layer) {
    (void)codebooks;
    (void)upto_layer;
    ThrowCudaDisabled("CudaStreamKernels::RvqSetPriorCodebooks");
}

void CudaStreamKernels::RvqProjectXnormBlockInCopyStreamAfterUpload(int slot,
                                                                    const DeviceMatF32View& Xnorm_slot,
                                                                    int cols,
                                                                    int upto_layer,
                                                                    const void* const* codes_host,
                                                                    const int* code_bytes_per_code) {
    (void)slot;
    (void)Xnorm_slot;
    (void)cols;
    (void)upto_layer;
    (void)codes_host;
    (void)code_bytes_per_code;
    ThrowCudaDisabled("CudaStreamKernels::RvqProjectXnormBlockInCopyStreamAfterUpload");
}

void CudaStreamKernels::RvqStageCodesBlockInCopyStreamAfterUpload(int slot,
                                                                  int cols,
                                                                  int upto_layer,
                                                                  const void* const* codes_host,
                                                                  const int* code_bytes_per_code) {
    (void)slot;
    (void)cols;
    (void)upto_layer;
    (void)codes_host;
    (void)code_bytes_per_code;
    ThrowCudaDisabled("CudaStreamKernels::RvqStageCodesBlockInCopyStreamAfterUpload");
}

void CudaStreamKernels::RvqProjectXnormBlockOnComputeStream(int slot,
                                                           const DeviceMatF32View& Xnorm_slot,
                                                           int cols,
                                                           int upto_layer) {
    (void)slot;
    (void)Xnorm_slot;
    (void)cols;
    (void)upto_layer;
    ThrowCudaDisabled("CudaStreamKernels::RvqProjectXnormBlockOnComputeStream");
}

bool CudaStreamKernels::KmeansTryCacheXnormOnDevice(const ColMajorMatrix<float>& X, DeviceMatF32View* out_full) {
    (void)X;
    if (out_full) *out_full = EmptyMat();
    ThrowCudaDisabled("CudaStreamKernels::KmeansTryCacheXnormOnDevice");
}

bool CudaStreamKernels::KmeansTryPinHostX(const ColMajorMatrix<float>& X, const float** out_pinned_base) {
    (void)X;
    if (out_pinned_base) *out_pinned_base = nullptr;
    ThrowCudaDisabled("CudaStreamKernels::KmeansTryPinHostX");
}

void CudaStreamKernels::KmeansReleaseXnormCache() { ThrowCudaDisabled("CudaStreamKernels::KmeansReleaseXnormCache"); }

DeviceMatF32View CudaStreamKernels::KmeansEnsurePartialXnormCache(int rows, int cols) {
    (void)rows;
    (void)cols;
    ThrowCudaDisabled("CudaStreamKernels::KmeansEnsurePartialXnormCache");
}

DeviceMatF32View CudaStreamKernels::KmeansPartialXnormCacheView() const {
    ThrowCudaDisabled("CudaStreamKernels::KmeansPartialXnormCacheView");
}

void CudaStreamKernels::KmeansCopyActiveSlotToPartialCache(int slot, int dst_col0, int cols) {
    (void)slot;
    (void)dst_col0;
    (void)cols;
    ThrowCudaDisabled("CudaStreamKernels::KmeansCopyActiveSlotToPartialCache");
}

DeviceMatF32View CudaStreamKernels::KmeansCopyPartialCacheToSlot(int slot, int src_col0, int cols) {
    (void)slot;
    (void)src_col0;
    (void)cols;
    ThrowCudaDisabled("CudaStreamKernels::KmeansCopyPartialCacheToSlot");
}

void CudaStreamKernels::KmeansSetActiveSlot(int slot) {
    (void)slot;
    ThrowCudaDisabled("CudaStreamKernels::KmeansSetActiveSlot");
}

void CudaStreamKernels::KmeansReleasePartialXnormCache() { ThrowCudaDisabled("CudaStreamKernels::KmeansReleasePartialXnormCache"); }

void CudaStreamKernels::KmeansResetSums(int d, int k) {
    (void)d;
    (void)k;
    ThrowCudaDisabled("CudaStreamKernels::KmeansResetSums");
}

void CudaStreamKernels::KmeansEnsureBlockArgmax(int cols) {
    (void)cols;
    ThrowCudaDisabled("CudaStreamKernels::KmeansEnsureBlockArgmax");
}

void CudaStreamKernels::KmeansCopyTileArgmaxToBlock(const DeviceVecI32View& tile_assign,
                                                    const DeviceVecF32View& tile_best,
                                                    int col_base) {
    (void)tile_assign;
    (void)tile_best;
    (void)col_base;
    ThrowCudaDisabled("CudaStreamKernels::KmeansCopyTileArgmaxToBlock");
}

DeviceVecI32View CudaStreamKernels::KmeansBlockAssignView() const { ThrowCudaDisabled("CudaStreamKernels::KmeansBlockAssignView"); }
DeviceVecF32View CudaStreamKernels::KmeansBlockBestView() const { ThrowCudaDisabled("CudaStreamKernels::KmeansBlockBestView"); }

void CudaStreamKernels::KmeansDownloadBlockArgmax(std::vector<int>* assign, std::vector<float>* best) {
    (void)assign;
    (void)best;
    ThrowCudaDisabled("CudaStreamKernels::KmeansDownloadBlockArgmax");
}

void CudaStreamKernels::KmeansDownloadBlockBestDot(std::vector<float>* best) {
    (void)best;
    ThrowCudaDisabled("CudaStreamKernels::KmeansDownloadBlockBestDot");
}

void CudaStreamKernels::KmeansDownloadBlockAssignAsync(int slot) {
    (void)slot;
    ThrowCudaDisabled("CudaStreamKernels::KmeansDownloadBlockAssignAsync");
}

void CudaStreamKernels::KmeansDownloadBlockAssignSync(int slot, const int** out_assign_i32, int* out_cols) {
    (void)slot;
    if (out_assign_i32) *out_assign_i32 = nullptr;
    if (out_cols) *out_cols = 0;
    ThrowCudaDisabled("CudaStreamKernels::KmeansDownloadBlockAssignSync");
}

DeviceVecF32View CudaStreamKernels::KmeansUploadBlockWeights(const std::vector<float>& weights) {
    (void)weights;
    ThrowCudaDisabled("CudaStreamKernels::KmeansUploadBlockWeights");
}

bool CudaStreamKernels::KmeansInitAnnealWeightsAll(std::int64_t n, float initial_weight) {
    (void)n;
    (void)initial_weight;
    ThrowCudaDisabled("CudaStreamKernels::KmeansInitAnnealWeightsAll");
}

DeviceVecF32View CudaStreamKernels::KmeansAnnealWeightsBlockView(std::int64_t col0, int cols) const {
    (void)col0;
    (void)cols;
    ThrowCudaDisabled("CudaStreamKernels::KmeansAnnealWeightsBlockView");
}

void CudaStreamKernels::KmeansResetAnnealTotals() { ThrowCudaDisabled("CudaStreamKernels::KmeansResetAnnealTotals"); }

void CudaStreamKernels::KmeansAnnealUpdateWeightsAndTotalsFromBlockBest(std::int64_t col0,
                                                                       int cols,
                                                                       float threshold,
                                                                       float annealed_factor,
                                                                       float min_weight) {
    (void)col0;
    (void)cols;
    (void)threshold;
    (void)annealed_factor;
    (void)min_weight;
    ThrowCudaDisabled("CudaStreamKernels::KmeansAnnealUpdateWeightsAndTotalsFromBlockBest");
}

void CudaStreamKernels::KmeansAccumulateAnnealTotalsFromBlockBestDotAndWeights(const DeviceVecF32View& d_weights) {
    (void)d_weights;
    ThrowCudaDisabled("CudaStreamKernels::KmeansAccumulateAnnealTotalsFromBlockBestDotAndWeights");
}

void CudaStreamKernels::KmeansDownloadAnnealTotals(double* total_w, double* total_cost) {
    if (total_w) *total_w = 0.0;
    if (total_cost) *total_cost = 0.0;
    ThrowCudaDisabled("CudaStreamKernels::KmeansDownloadAnnealTotals");
}

void CudaStreamKernels::KmeansDownloadAnnealTotalsAsync() {
    ThrowCudaDisabled("CudaStreamKernels::KmeansDownloadAnnealTotalsAsync");
}

void CudaStreamKernels::KmeansSyncAnnealTotalsDownload(double* total_w, double* total_cost) {
    if (total_w) *total_w = 0.0;
    if (total_cost) *total_cost = 0.0;
    ThrowCudaDisabled("CudaStreamKernels::KmeansSyncAnnealTotalsDownload");
}

void CudaStreamKernels::KmeansResetMetricTotals() { ThrowCudaDisabled("CudaStreamKernels::KmeansResetMetricTotals"); }

void CudaStreamKernels::KmeansAccumulateMseFromBlockBestDot(int cols) {
    (void)cols;
    ThrowCudaDisabled("CudaStreamKernels::KmeansAccumulateMseFromBlockBestDot");
}

void CudaStreamKernels::KmeansDownloadMetricTotals(double* mse_sum, double* mse_count) {
    if (mse_sum) *mse_sum = 0.0;
    if (mse_count) *mse_count = 0.0;
    ThrowCudaDisabled("CudaStreamKernels::KmeansDownloadMetricTotals");
}

void CudaStreamKernels::KmeansDownloadMetricTotalsAsync() {
    ThrowCudaDisabled("CudaStreamKernels::KmeansDownloadMetricTotalsAsync");
}

void CudaStreamKernels::KmeansSyncMetricTotalsDownload(double* mse_sum, double* mse_count) {
    if (mse_sum) *mse_sum = 0.0;
    if (mse_count) *mse_count = 0.0;
    ThrowCudaDisabled("CudaStreamKernels::KmeansSyncMetricTotalsDownload");
}

void CudaStreamKernels::KmeansComputeOutlierMaskFromBlockBest(int cols, float threshold) {
    (void)cols;
    (void)threshold;
    ThrowCudaDisabled("CudaStreamKernels::KmeansComputeOutlierMaskFromBlockBest");
}

void CudaStreamKernels::KmeansDownloadOutlierMaskAsync(int cols) {
    (void)cols;
    ThrowCudaDisabled("CudaStreamKernels::KmeansDownloadOutlierMaskAsync");
}

void CudaStreamKernels::KmeansSyncOutlierMaskDownload(std::uint8_t** host_ptr, int* host_bytes) {
    if (host_ptr) *host_ptr = nullptr;
    if (host_bytes) *host_bytes = 0;
    ThrowCudaDisabled("CudaStreamKernels::KmeansSyncOutlierMaskDownload");
}

void CudaStreamKernels::KmeansAccumulateUnitWeights(const DeviceMatF32View& Xnorm,
                                                    const std::vector<int>& assignments) {
    (void)Xnorm;
    (void)assignments;
    ThrowCudaDisabled("CudaStreamKernels::KmeansAccumulateUnitWeights");
}

void CudaStreamKernels::KmeansAccumulateUnitWeightsDeviceAssign(const DeviceMatF32View& Xnorm,
                                                                const DeviceVecI32View& d_assign) {
    (void)Xnorm;
    (void)d_assign;
    ThrowCudaDisabled("CudaStreamKernels::KmeansAccumulateUnitWeightsDeviceAssign");
}

void CudaStreamKernels::KmeansAccumulateUnitWeightsDeviceAssignNoAtomic(const DeviceMatF32View& Xnorm,
                                                                        const DeviceVecI32View& d_assign) {
    (void)Xnorm;
    (void)d_assign;
    ThrowCudaDisabled("CudaStreamKernels::KmeansAccumulateUnitWeightsDeviceAssignNoAtomic");
}

void CudaStreamKernels::KmeansAccumulateWeights(const DeviceMatF32View& Xnorm,
                                               const std::vector<int>& assignments,
                                               const std::vector<float>& weights) {
    (void)Xnorm;
    (void)assignments;
    (void)weights;
    ThrowCudaDisabled("CudaStreamKernels::KmeansAccumulateWeights");
}

void CudaStreamKernels::KmeansAccumulateWeightsDeviceAssign(const DeviceMatF32View& Xnorm,
                                                           const DeviceVecI32View& d_assign,
                                                           const DeviceVecF32View& d_weights) {
    (void)Xnorm;
    (void)d_assign;
    (void)d_weights;
    ThrowCudaDisabled("CudaStreamKernels::KmeansAccumulateWeightsDeviceAssign");
}

void CudaStreamKernels::KmeansAccumulateWeightsDeviceAssignNoAtomic(const DeviceMatF32View& Xnorm,
                                                                   const DeviceVecI32View& d_assign,
                                                                   const DeviceVecF32View& d_weights) {
    (void)Xnorm;
    (void)d_assign;
    (void)d_weights;
    ThrowCudaDisabled("CudaStreamKernels::KmeansAccumulateWeightsDeviceAssignNoAtomic");
}

void CudaStreamKernels::KmeansDownloadSums(ColMajorMatrix<float>* sum_x, std::vector<float>* sum_w) {
    (void)sum_x;
    (void)sum_w;
    ThrowCudaDisabled("CudaStreamKernels::KmeansDownloadSums");
}

void CudaStreamKernels::KmeansResetCostHistogram(int bins) {
    (void)bins;
    ThrowCudaDisabled("CudaStreamKernels::KmeansResetCostHistogram");
}

void CudaStreamKernels::KmeansAccumulateCostHistogramFromBestDot(const DeviceVecF32View& d_best_dot, int bins) {
    (void)d_best_dot;
    (void)bins;
    ThrowCudaDisabled("CudaStreamKernels::KmeansAccumulateCostHistogramFromBestDot");
}

void CudaStreamKernels::KmeansDownloadCostHistogram(std::vector<std::uint32_t>* hist_u32) {
    (void)hist_u32;
    ThrowCudaDisabled("CudaStreamKernels::KmeansDownloadCostHistogram");
}

void CudaStreamKernels::KmeansDownloadCostHistogramAsync(std::uint32_t* host_dst, int bins) {
    (void)host_dst;
    (void)bins;
    ThrowCudaDisabled("CudaStreamKernels::KmeansDownloadCostHistogramAsync");
}

void CudaStreamKernels::KmeansDownloadCostHistogramSync() {
    ThrowCudaDisabled("CudaStreamKernels::KmeansDownloadCostHistogramSync");
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
    (void)Xnorm;
    (void)coarse_centers;
    (void)fine_centers;
    (void)K_valid;
    (void)K1;
    (void)K2;
    (void)topL;
    (void)assignments;
    (void)best_values;
    ThrowCudaDisabled("CudaStreamKernels::KmeansHier2AssignColsF32");
}

void CudaStreamKernels::KmeansHier2AssignColsF32Device(const DeviceMatF32View& Xnorm,
                                                       const ColMajorMatrix<float>& coarse_centers,
                                                       const ColMajorMatrix<float>& fine_centers,
                                                       int K_valid,
                                                       int K1,
                                                       int K2,
                                                       int topL) {
    (void)Xnorm;
    (void)coarse_centers;
    (void)fine_centers;
    (void)K_valid;
    (void)K1;
    (void)K2;
    (void)topL;
    ThrowCudaDisabled("CudaStreamKernels::KmeansHier2AssignColsF32Device");
}

}  // namespace stlq
