#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "stlq/core/kernel_provider_cpu.h"
#include "stlq/core/kernel_provider_cuda_stream.h"
#include "stlq/quantizer/beam_search.h"
#include "stlq/quantizer/cost_utils.h"
#include "stlq/quantizer/least_squares.h"
#include "stlq/quantizer/precomp_large_root.h"

namespace {

stlq::CodebookPack MakeRandomCodebooks(int d, const std::vector<int>& h_vec, std::mt19937* rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    stlq::CodebookPack pack;
    pack.d = d;
    pack.h_vec = h_vec;
    pack.books.resize(h_vec.size());
    for (std::size_t l = 0; l < h_vec.size(); ++l) {
        const int h = h_vec[l];
        stlq::ColMajorMatrix<float> C(d, h);
        for (int j = 0; j < h; ++j) {
            float norm2 = 0.0f;
            float* col = C.Col(j);
            for (int r = 0; r < d; ++r) {
                const float v = nd(*rng);
                col[r] = v;
                norm2 += v * v;
            }
            const float inv = 1.0f / std::sqrt(std::max(1e-12f, norm2));
            for (int r = 0; r < d; ++r) col[r] *= inv;
        }
        pack.books[l] = std::move(C);
    }
    return pack;
}

stlq::ColMajorMatrix<float> MakeRandomX(int d, int n, std::mt19937* rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    stlq::ColMajorMatrix<float> X(d, n);
    for (int j = 0; j < n; ++j) {
        float* col = X.Col(j);
        for (int r = 0; r < d; ++r) col[r] = nd(*rng);
    }
    return X;
}

}  // namespace

int main() {
#if !defined(STLQ_ENABLE_CUDA) || !STLQ_ENABLE_CUDA
    std::cerr << "SKIP: built without CUDA\n";
    return 0;
#else
    const int d = 32;
    const int n = 512;
    const int m = 5;
    const int h0 = 256;
    const int hs = 64;

    std::mt19937 rng(123);
    std::vector<int> h_vec(static_cast<std::size_t>(m), hs);
    h_vec[0] = h0;
    stlq::CodebookPack pack = MakeRandomCodebooks(d, h_vec, &rng);
    stlq::ColMajorMatrix<float> X = MakeRandomX(d, n, &rng);

    stlq::CpuStreamKernels cpu;
    stlq::PrecompLargeRoot pre;
    std::string err;
    stlq::PrecompLargeRootBuildOptions opts;
    if (!stlq::BuildPrecompLargeRoot(pack, &cpu, opts, &pre, &err)) {
        std::cerr << "BuildPrecompLargeRoot failed: " << err << "\n";
        return 2;
    }

    // xC_small on GPU.
    stlq::CudaStreamKernels cuda_kernels(/*device=*/0, /*allow_tf32=*/false, /*cublas_workspace_mb=*/0);
    stlq::DeviceMatF32View xC_dev{};
    cuda_kernels.GemmDevice(true, false, 1.0f, pre.C_small, X, 0.0f, &xC_dev);

    stlq::DeviceMatF32View X_dev{};
    if (!cuda_kernels.TryGetCachedDeviceF32(X, &X_dev)) {
        std::cerr << "TryGetCachedDeviceF32(X) failed: expected X to be cached on device after GemmDevice\n";
        return 4;
    }

    std::vector<std::uint32_t> cluster_id;
    stlq::ColMajorMatrix<stlq::Code> B_small;
    std::vector<float> best_err;
    stlq::BeamSearchPrefixLSLargeRootCudaH2DeviceXcDeviceXWithBestErr(
        X_dev, pre, xC_dev, /*allow_tf32=*/false,
        &cluster_id, &B_small, &best_err,
        /*timing=*/nullptr);

    // Compute beam costs on CPU for the selected (cluster_id, B_small).
    stlq::ColMajorMatrix<float> xC_small;
    stlq::ComputeXcSmall(pre, X, &cpu, &xC_small);
    stlq::ColMajorMatrix<float> a(m, n);
    stlq::SolveLeastSquaresAllLargeRoot(pre, X, xC_small, cluster_id, B_small, &a);
    std::vector<float> X_norm2;
    stlq::ComputeXNorm2(X, &X_norm2);
    std::vector<float> cost_cpu;
    stlq::ComputeCostsLargeRoot(pre, X, xC_small, cluster_id, B_small, a, X_norm2, &cost_cpu);

    if (static_cast<int>(best_err.size()) != n || static_cast<int>(cost_cpu.size()) != n) {
        std::cerr << "size mismatch\n";
        return 3;
    }

    double max_abs = 0.0;
    double mean_abs = 0.0;
    for (int i = 0; i < n; ++i) {
        const double diff = std::fabs(static_cast<double>(best_err[static_cast<std::size_t>(i)]) -
                                      static_cast<double>(cost_cpu[static_cast<std::size_t>(i)]));
        max_abs = std::max(max_abs, diff);
        mean_abs += diff;
    }
    mean_abs /= static_cast<double>(n);

    std::cout << "test_beam_large_root_best_err_matches_cost:\n";
    std::cout << "  mean_abs_diff=" << mean_abs << "\n";
    std::cout << "  max_abs_diff=" << max_abs << "\n";

    // Loose tolerance: CUDA beam uses float math and can differ slightly from CPU cost computation.
    // This is only used for metrics, not for selecting parents/encoding outputs.
    if (max_abs > 1e-2) {
        std::cerr << "FAIL: best_err mismatch too large\n";
        return 1;
    }
    return 0;
#endif
}
