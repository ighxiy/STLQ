#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "stlq/core/kernel_provider_cuda_stream.h"

namespace {

stlq::ColMajorMatrix<float> MakeCodebook(int d, int axis0, int axis1) {
    stlq::ColMajorMatrix<float> codebook(d, 2);
    codebook.data.assign(static_cast<std::size_t>(d) * 2, 0.0f);
    codebook(axis0, 0) = 1.0f;
    codebook(axis1, 1) = 1.0f;
    return codebook;
}

}  // namespace

int main() {
    constexpr int d = 4;
    constexpr int full_cols = 4;
    constexpr int tail_cols = 2;

    std::vector<stlq::ColMajorMatrix<float>> codebooks;
    codebooks.push_back(MakeCodebook(d, /*axis0=*/0, /*axis1=*/1));
    codebooks.push_back(MakeCodebook(d, /*axis0=*/2, /*axis1=*/3));

    stlq::ColMajorMatrix<float> full(d, full_cols);
    stlq::ColMajorMatrix<float> tail(d, tail_cols);
    full.data.assign(static_cast<std::size_t>(d) * full_cols, 1.0f);
    tail.data.assign(static_cast<std::size_t>(d) * tail_cols, 1.0f);

    const std::vector<std::uint8_t> full_codes0(full_cols, 0);
    const std::vector<std::uint8_t> full_codes1(full_cols, 1);
    const std::vector<std::uint8_t> tail_codes0(tail_cols, 1);
    const std::vector<std::uint8_t> tail_codes1(tail_cols, 0);
    const void* full_code_ptrs[] = {full_codes0.data(), full_codes1.data()};
    const void* tail_code_ptrs[] = {tail_codes0.data(), tail_codes1.data()};
    const int code_bytes[] = {1, 1};

    stlq::CudaStreamKernels cuda(/*device=*/0, /*allow_tf32=*/false, /*cublas_workspace_mb=*/0);
    cuda.KmeansSetCollectMetrics(false);
    cuda.RvqSetPriorCodebooks(codebooks, /*upto_layer=*/2);

    const stlq::DeviceMatF32View full_view =
        cuda.KmeansUploadAndNormalizeHostPtrBAsync(
            /*slot=*/0, full.data.data(), d, d, full_cols,
            /*stage_to_pinned=*/false, nullptr, nullptr, nullptr);
    cuda.RvqStageCodesBlockInCopyStreamAfterUpload(
        /*slot=*/0, full_cols, /*upto_layer=*/2, full_code_ptrs, code_bytes);

    const stlq::DeviceMatF32View tail_view =
        cuda.KmeansUploadAndNormalizeHostPtrBAsync(
            /*slot=*/1, tail.data.data(), d, d, tail_cols,
            /*stage_to_pinned=*/false, nullptr, nullptr, nullptr);
    cuda.RvqStageCodesBlockInCopyStreamAfterUpload(
        /*slot=*/1, tail_cols, /*upto_layer=*/2, tail_code_ptrs, code_bytes);

    // Make the tail block's metadata upload complete before projecting slot 0.
    // A shared metadata buffer would now describe the two-column tail layout.
    cuda.Sync();
    cuda.KmeansComputeWaitForUpload(/*slot=*/0);
    cuda.RvqProjectXnormBlockOnComputeStream(/*slot=*/0, full_view, full_cols, /*upto_layer=*/2);
    cuda.KmeansMarkUploadSlotConsumed(/*slot=*/0);

    stlq::ColMajorMatrix<float> projected;
    cuda.DownloadF32(full_view, &projected);

    const float expected = 1.0f / std::sqrt(2.0f);
    constexpr float tolerance = 1e-5f;
    for (int col = 0; col < full_cols; ++col) {
        const float expected_col[] = {0.0f, expected, expected, 0.0f};
        for (int row = 0; row < d; ++row) {
            const float actual = projected(row, col);
            if (std::abs(actual - expected_col[row]) > tolerance) {
                throw std::runtime_error(
                    "RVQ projection used metadata from the other upload slot.");
            }
        }
    }

    (void)tail_view;
    return 0;
}
