#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "stlq/common/logger.h"
#include "stlq/quantizer/beam_search.h"
#include "stlq/quantizer/encoder.h"
#include "stlq/quantizer/icm.h"
#include "stlq/quantizer/icm_cuda.h"
#include "stlq/quantizer/least_squares.h"

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
            for (int r = 0; r < d; ++r) {
                col[r] *= inv;
            }
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
        for (int r = 0; r < d; ++r) {
            col[r] = nd(*rng);
        }
    }
    return X;
}

stlq::ColMajorMatrix<float> ComputeXcNaive(const stlq::Precomp& pre, const stlq::ColMajorMatrix<float>& X) {
    const int d = X.rows;
    const int n = X.cols;
    const int H = pre.H;
    stlq::ColMajorMatrix<float> xC(H, n);
    for (int i = 0; i < n; ++i) {
        const float* x = X.Col(i);
        float* out = xC.Col(i);
        for (int j = 0; j < H; ++j) {
            const float* c = pre.C_all.Col(j);
            float s = 0.0f;
            for (int r = 0; r < d; ++r) {
                s += c[r] * x[r];
            }
            out[j] = s;
        }
    }
    return xC;
}

}  // namespace

int main() {
#if !defined(STLQ_ENABLE_CUDA) || !STLQ_ENABLE_CUDA
    std::cerr << "SKIP: built without CUDA\n";
    return 0;
#else
    const int d = 64;
    const int n = 256;
    const int m = 5;
    const int icm_iters = 4;
    const int ils_iters = 10;
    const int perturb_k = 3;
    const std::uint32_t seed = 12345;

    std::mt19937 rng(7);
    std::vector<int> h_vec(static_cast<std::size_t>(m), 64);
    stlq::CodebookPack pack = MakeRandomCodebooks(d, h_vec, &rng);
    stlq::Precomp pre;
    if (!stlq::BuildPrecomp(pack, &pre)) {
        std::cerr << "BuildPrecomp failed\n";
        return 2;
    }
    stlq::ColMajorMatrix<float> X = MakeRandomX(d, n, &rng);
    stlq::ColMajorMatrix<float> xC = ComputeXcNaive(pre, X);

    // Start point: beam search + LS (CPU) to match the real pipeline.
    stlq::ColMajorMatrix<stlq::FullCode> B0(m, n);
    stlq::ColMajorMatrix<float> a0(m, n);
    stlq::BeamSearchPrefixLS(X, pre, xC, /*H_beam=*/2, &B0);
    stlq::SolveLeastSquaresAll(pre, xC, B0, &a0);

    stlq::ColMajorMatrix<stlq::FullCode> B_cpu = B0;
    stlq::ColMajorMatrix<float> a_cpu = a0;
    std::vector<float> Xn2_cpu;
    std::vector<float> cost_cpu;
    stlq::DynamicIcmWithIlsAbsNoNormal(X, pre, xC, icm_iters, ils_iters, perturb_k, seed,
                                         &B_cpu, &a_cpu, &Xn2_cpu, &cost_cpu,
                                         /*print_progress=*/false,
                                         /*sample_id_offset=*/0);

    // Nested-parallel call: this simulates the "outer OpenMP region" case (e.g. cluster-parallel code),
    // where the CPU ICM implementation should take the serial fast-path and still match results.
    stlq::ColMajorMatrix<stlq::FullCode> B_cpu_nested = B0;
    stlq::ColMajorMatrix<float> a_cpu_nested = a0;
    std::vector<float> Xn2_cpu_nested;
    std::vector<float> cost_cpu_nested;
    #pragma omp parallel default(none) shared(X, pre, xC, icm_iters, ils_iters, perturb_k, seed, B_cpu_nested, a_cpu_nested, Xn2_cpu_nested, cost_cpu_nested)
    {
        #pragma omp single
        {
            stlq::DynamicIcmWithIlsAbsNoNormal(X, pre, xC, icm_iters, ils_iters, perturb_k, seed,
                                                 &B_cpu_nested, &a_cpu_nested,
                                                 &Xn2_cpu_nested, &cost_cpu_nested,
                                                 /*print_progress=*/false,
                                                 /*sample_id_offset=*/0);
        }
    }

    stlq::ColMajorMatrix<stlq::FullCode> B_gpu = B0;
    stlq::ColMajorMatrix<float> a_gpu = a0;
    std::vector<float> Xn2_gpu;
    std::vector<float> cost_gpu;
    stlq::DynamicIcmWithIlsAbsNoNormalCuda(X, pre, xC, icm_iters, ils_iters, perturb_k, seed,
                                             &B_gpu, &a_gpu, &Xn2_gpu, &cost_gpu,
                                             /*print_progress=*/false,
                                             /*sample_id_offset=*/0);

    // Device-xC path: upload xC to device and ensure results match the normal CUDA entrypoint.
    float* d_xC = nullptr;
    const std::size_t xC_elems = xC.data.size();
    if (cudaMalloc(&d_xC, sizeof(float) * xC_elems) != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
        return 3;
    }
    if (cudaMemcpy(d_xC, xC.data.data(), sizeof(float) * xC_elems, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "cudaMemcpy H2D failed\n";
        cudaFree(d_xC);
        return 3;
    }
    stlq::DeviceMatF32View xC_dev;
    xC_dev.ptr = d_xC;
    xC_dev.rows = pre.H;
    xC_dev.cols = n;
    xC_dev.ld = pre.H;

    stlq::ColMajorMatrix<stlq::FullCode> B_gpu_devxc = B0;
    stlq::ColMajorMatrix<float> a_gpu_devxc = a0;
    std::vector<float> Xn2_gpu_devxc;
    std::vector<float> cost_gpu_devxc;
    stlq::DynamicIcmWithIlsAbsNoNormalCudaDeviceXc(X, pre, xC_dev, icm_iters, ils_iters, perturb_k, seed,
                                                     &B_gpu_devxc, &a_gpu_devxc, &Xn2_gpu_devxc, &cost_gpu_devxc,
                                                     /*print_progress=*/false,
                                                     /*sample_id_offset=*/0);
    cudaFree(d_xC);

    double mean_cpu = 0.0;
    double mean_cpu_nested = 0.0;
    double mean_gpu = 0.0;
    double mean_gpu_devxc = 0.0;
    for (int i = 0; i < n; ++i) {
        mean_cpu += static_cast<double>(cost_cpu[static_cast<std::size_t>(i)]);
        mean_cpu_nested += static_cast<double>(cost_cpu_nested[static_cast<std::size_t>(i)]);
        mean_gpu += static_cast<double>(cost_gpu[static_cast<std::size_t>(i)]);
        mean_gpu_devxc += static_cast<double>(cost_gpu_devxc[static_cast<std::size_t>(i)]);
    }
    mean_cpu /= static_cast<double>(n);
    mean_cpu_nested /= static_cast<double>(n);
    mean_gpu /= static_cast<double>(n);
    mean_gpu_devxc /= static_cast<double>(n);
    const double diff = std::fabs(mean_gpu - mean_cpu);
    const double diff_devxc = std::fabs(mean_gpu_devxc - mean_gpu);
    const double diff_nested = std::fabs(mean_cpu_nested - mean_cpu);

    int mism_B = 0;
    int mism_B_nested = 0;
    int mism_B_devxc = 0;
    for (int i = 0; i < n; ++i) {
        for (int l = 0; l < m; ++l) {
            mism_B += (B_cpu(l, i) != B_gpu(l, i)) ? 1 : 0;
            mism_B_nested += (B_cpu(l, i) != B_cpu_nested(l, i)) ? 1 : 0;
            mism_B_devxc += (B_gpu_devxc(l, i) != B_gpu(l, i)) ? 1 : 0;
        }
    }

    std::cout << "test_icm_cuda_strict:\n";
    std::cout << "  mean_cost_cpu=" << mean_cpu << "\n";
    std::cout << "  mean_cost_cpu_nested=" << mean_cpu_nested << "\n";
    std::cout << "  mean_cost_gpu=" << mean_gpu << "\n";
    std::cout << "  mean_cost_gpu_devxc=" << mean_gpu_devxc << "\n";
    std::cout << "  abs_diff=" << diff << "\n";
    std::cout << "  abs_diff_devxc=" << diff_devxc << "\n";
    std::cout << "  abs_diff_nested=" << diff_nested << "\n";
    std::cout << "  mismatched_codes=" << mism_B << " / " << (n * m) << "\n";
    std::cout << "  mismatched_codes_devxc=" << mism_B_devxc << " / " << (n * m) << "\n";
    std::cout << "  mismatched_codes_nested=" << mism_B_nested << " / " << (n * m) << "\n";

    // We only require mean cost to be very close in "strict" mode.
    if (diff > 1e-2) {
        std::cerr << "FAIL: mean cost mismatch too large\n";
        return 1;
    }
    if (diff_nested > 1e-4 || mism_B_nested != 0) {
        std::cerr << "FAIL: nested-parallel CPU path mismatch\n";
        return 1;
    }
    if (diff_devxc > 1e-4 || mism_B_devxc != 0) {
        std::cerr << "FAIL: device-xC CUDA path mismatch\n";
        return 1;
    }
    return 0;
#endif
}
