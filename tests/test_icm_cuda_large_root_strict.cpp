#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "stlq/core/kernel_provider_cpu.h"
#include "stlq/common/logger.h"
#include "stlq/quantizer/icm.h"
#include "stlq/quantizer/icm_cuda.h"
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

}  // namespace

int main() {
#if !defined(STLQ_ENABLE_CUDA) || !STLQ_ENABLE_CUDA
    std::cerr << "SKIP: built without CUDA\n";
    return 0;
#else
    const int d = 64;
    const int n = 512;
    const int m = 5;
    const int h0 = 64;
    const int hs = 64;
    const int icm_iters = 3;
    const int ils_iters = 10;
    const int perturb_k = 3;
    const std::uint32_t seed = 77;

    std::mt19937 rng(9);
    std::vector<int> h_vec(static_cast<std::size_t>(m), hs);
    h_vec[0] = h0;
    stlq::CodebookPack pack = MakeRandomCodebooks(d, h_vec, &rng);
    stlq::ColMajorMatrix<float> X = MakeRandomX(d, n, &rng);

    stlq::CpuStreamKernels cpu_stream;
    stlq::PrecompLargeRoot pre;
    std::string err;
    stlq::PrecompLargeRootBuildOptions pre_opts;
    pre_opts.build_g0s_transpose = true; // should not change semantics
    if (!stlq::BuildPrecompLargeRoot(pack, &cpu_stream, pre_opts, &pre, &err)) {
        std::cerr << "BuildPrecompLargeRoot failed: " << err << "\n";
        return 2;
    }
    stlq::ColMajorMatrix<float> xC_small;
    stlq::ComputeXcSmall(pre, X, &cpu_stream, &xC_small);

    std::uniform_int_distribution<int> cid_dist(0, h0 - 1);
    std::vector<std::uint32_t> cluster_id(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) cluster_id[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(cid_dist(rng));

    stlq::ColMajorMatrix<stlq::Code> B_small(std::max(0, m - 1), n);
    std::uniform_int_distribution<int> code_dist(0, hs - 1);
    for (int i = 0; i < n; ++i) {
        for (int l = 1; l < m; ++l) {
            B_small(l - 1, i) = static_cast<stlq::Code>(code_dist(rng));
        }
    }

    stlq::ColMajorMatrix<float> a_cpu(m, n);
    stlq::ColMajorMatrix<float> a_gpu(m, n);
    std::vector<float> Xn2_cpu;
    std::vector<float> Xn2_gpu;
    std::vector<float> cost_cpu;
    std::vector<float> cost_gpu;
    stlq::ColMajorMatrix<stlq::Code> B_cpu = B_small;
    stlq::ColMajorMatrix<stlq::Code> B_gpu = B_small;

    stlq::DynamicIcmWithIlsAbsNoNormalLargeRoot(X, pre, xC_small,
                                                  icm_iters, ils_iters, perturb_k,
                                                  seed,
                                                  cluster_id,
                                                  &B_cpu, &a_cpu, &Xn2_cpu, &cost_cpu,
                                                  /*print_progress=*/false,
                                                  /*sample_id_offset=*/0);
    stlq::DynamicIcmWithIlsAbsNoNormalLargeRootCuda(X, pre, xC_small,
                                                      icm_iters, ils_iters, perturb_k,
                                                      seed,
                                                      cluster_id,
                                                      &B_gpu, &a_gpu, &Xn2_gpu, &cost_gpu,
                                                      /*print_progress=*/false,
                                                      /*sample_id_offset=*/0);

    double mean_cpu = 0.0;
    double mean_gpu = 0.0;
    for (int i = 0; i < n; ++i) {
        mean_cpu += static_cast<double>(cost_cpu[static_cast<std::size_t>(i)]);
        mean_gpu += static_cast<double>(cost_gpu[static_cast<std::size_t>(i)]);
    }
    mean_cpu /= static_cast<double>(n);
    mean_gpu /= static_cast<double>(n);
    const double diff = std::fabs(mean_gpu - mean_cpu);

    int mism_B = 0;
    for (int i = 0; i < n; ++i) {
        for (int l = 1; l < m; ++l) {
            mism_B += (B_cpu(l - 1, i) != B_gpu(l - 1, i)) ? 1 : 0;
        }
    }

    std::cout << "test_icm_cuda_large_root_strict:\n";
    std::cout << "  mean_cost_cpu=" << mean_cpu << "\n";
    std::cout << "  mean_cost_gpu=" << mean_gpu << "\n";
    std::cout << "  abs_diff=" << diff << "\n";
    std::cout << "  mismatched_codes=" << mism_B << " / " << (n * (m - 1)) << "\n";

    if (diff > 1e-2) {
        std::cerr << "FAIL: mean cost mismatch too large\n";
        return 1;
    }
    return 0;
#endif
}
