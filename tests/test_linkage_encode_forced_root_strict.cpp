#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "stlq/linkage/linkage_encode_cuda.h"
#include "stlq/quantizer/encoder.h"
#include "stlq/quantizer/icm.h"

#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

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

void ComputeXNorm2(const stlq::ColMajorMatrix<float>& X, std::vector<float>* out) {
    const int d = X.rows;
    const int n = X.cols;
    out->assign(static_cast<std::size_t>(std::max(0, n)), 0.0f);
    for (int i = 0; i < n; ++i) {
        const float* x = X.Col(i);
        float s = 0.0f;
        for (int r = 0; r < d; ++r) {
            s += x[r] * x[r];
        }
        (*out)[static_cast<std::size_t>(i)] = s;
    }
}

// CPU greedy init used by the UMAP forced-root path:
// - layer0 fixed to forced code
// - remaining layers chosen by max |rC[flat]*invnorm[flat]| among unused layers.
void GreedyInitAbsForcedRootCpu(const stlq::Precomp& pre,
                                const stlq::ColMajorMatrix<float>& xC,
                                const std::vector<int>& forced_root,
                                stlq::ColMajorMatrix<stlq::FullCode>* B,
                                stlq::ColMajorMatrix<float>* a) {
    const int H = pre.H;
    const int m = pre.m;
    const int n = xC.cols;
    B->rows = m;
    B->cols = n;
    B->data.assign(static_cast<std::size_t>(m) * static_cast<std::size_t>(n), 0);
    a->rows = m;
    a->cols = n;
    a->data.assign(static_cast<std::size_t>(m) * static_cast<std::size_t>(n), 0.0f);

    std::vector<float> rC(static_cast<std::size_t>(H), 0.0f);
    std::vector<std::uint8_t> available(static_cast<std::size_t>(m), 1);
    for (int i = 0; i < n; ++i) {
        const float* xCi = xC.Col(i);
        std::memcpy(rC.data(), xCi, sizeof(float) * static_cast<std::size_t>(H));
        std::fill(available.begin(), available.end(), 1);

        const int root_code = forced_root[static_cast<std::size_t>(i)];
        const int root_flat = pre.offsets[0] + root_code;
        const float invn_root = pre.invnorm_flat[static_cast<std::size_t>(root_flat)];
        const float alpha_root = rC[static_cast<std::size_t>(root_flat)] * invn_root * invn_root;
        (*B)(0, i) = static_cast<stlq::FullCode>(root_code);
        (*a)(0, i) = alpha_root;

        const float* Gcol_root = pre.G.Col(root_flat);
        for (int p = 0; p < H; ++p) {
            rC[static_cast<std::size_t>(p)] -= alpha_root * Gcol_root[p];
        }
        available[0] = 0;

        for (int t = 1; t < m; ++t) {
            int best_flat = 0;
            float best_abs = -std::numeric_limits<float>::infinity();
            float best_adj = 0.0f;
            for (int flat = 0; flat < H; ++flat) {
                const int layer = pre.flat_layer[static_cast<std::size_t>(flat)];
                if (!available[static_cast<std::size_t>(layer)]) {
                    continue;
                }
                const float adj = rC[static_cast<std::size_t>(flat)] *
                                  pre.invnorm_flat[static_cast<std::size_t>(flat)];
                const float aval = std::abs(adj);
                if (aval > best_abs) {
                    best_abs = aval;
                    best_adj = adj;
                    best_flat = flat;
                }
            }
            const int layer = pre.flat_layer[static_cast<std::size_t>(best_flat)];
            const int code = best_flat - pre.offsets[layer];
            const float invn = pre.invnorm_flat[static_cast<std::size_t>(best_flat)];
            const float alpha = best_adj * invn;

            (*B)(layer, i) = static_cast<stlq::FullCode>(code);
            (*a)(layer, i) = alpha;

            const float* Gcol = pre.G.Col(best_flat);
            for (int p = 0; p < H; ++p) {
                rC[static_cast<std::size_t>(p)] -= alpha * Gcol[p];
            }
            available[static_cast<std::size_t>(layer)] = 0;
        }
    }
}

}  // namespace

int main() {
#if !defined(STLQ_ENABLE_CUDA) || !STLQ_ENABLE_CUDA
    std::cerr << "SKIP: built without CUDA\n";
    return 0;
#else
    const int d = 64;
    const int n = 128;
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
    std::vector<float> X_norm2;
    ComputeXNorm2(X, &X_norm2);

    std::uniform_int_distribution<int> uni_root(0, h_vec[0] - 1);
    std::vector<int> forced(static_cast<std::size_t>(n), 0);
    for (int i = 0; i < n; ++i) {
        forced[static_cast<std::size_t>(i)] = uni_root(rng);
    }

    // GPU: compute xC via cuBLAS so CPU/GPU see the same xC values.
    stlq::CudaPoolConfig pcfg;
    pcfg.device = 0;
    pcfg.allow_tf32 = false;
    stlq::CudaStreamKernelsPool pool(/*num_ctx=*/1, pcfg);
    stlq::CudaCtx* ctx = pool.Acquire();
    if (!ctx) {
        std::cerr << "Acquire CUDA ctx failed\n";
        return 3;
    }

    float* d_X = nullptr;
    float* d_norm2 = nullptr;
    int* d_forced = nullptr;
    float* d_C = nullptr;
    float* d_xC = nullptr;
    stlq::FullCode* d_B = nullptr;
    float* d_a = nullptr;
    float* d_cost = nullptr;
    std::string err;
    const int H = pre.H;

    const std::size_t X_bytes = sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(n);
    const std::size_t n_bytes = sizeof(float) * static_cast<std::size_t>(n);
    const std::size_t forced_bytes = sizeof(int) * static_cast<std::size_t>(n);
    const std::size_t C_bytes = sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(H);
    const std::size_t xC_bytes = sizeof(float) * static_cast<std::size_t>(H) * static_cast<std::size_t>(n);
    const std::size_t B_bytes = sizeof(stlq::FullCode) * static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
    const std::size_t a_bytes = sizeof(float) * static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
    const std::size_t cost_bytes = sizeof(float) * static_cast<std::size_t>(n);

    auto cleanup = [&]() {
        if (d_cost) cudaFree(d_cost);
        if (d_a) cudaFree(d_a);
        if (d_B) cudaFree(d_B);
        if (d_xC) cudaFree(d_xC);
        if (d_C) cudaFree(d_C);
        if (d_forced) cudaFree(d_forced);
        if (d_norm2) cudaFree(d_norm2);
        if (d_X) cudaFree(d_X);
        pool.Release(ctx);
    };

    if (cudaSetDevice(ctx->device) != cudaSuccess) {
        std::cerr << "cudaSetDevice failed\n";
        cleanup();
        return 4;
    }
    if (cudaMalloc(&d_X, X_bytes) != cudaSuccess ||
        cudaMalloc(&d_norm2, n_bytes) != cudaSuccess ||
        cudaMalloc(&d_forced, forced_bytes) != cudaSuccess ||
        cudaMalloc(&d_C, C_bytes) != cudaSuccess ||
        cudaMalloc(&d_xC, xC_bytes) != cudaSuccess ||
        cudaMalloc(&d_B, B_bytes) != cudaSuccess ||
        cudaMalloc(&d_a, a_bytes) != cudaSuccess ||
        cudaMalloc(&d_cost, cost_bytes) != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
        cleanup();
        return 5;
    }

    if (cudaMemcpyAsync(d_X, X.data.data(), X_bytes, cudaMemcpyHostToDevice, ctx->stream) != cudaSuccess ||
        cudaMemcpyAsync(d_norm2, X_norm2.data(), n_bytes, cudaMemcpyHostToDevice, ctx->stream) != cudaSuccess ||
        cudaMemcpyAsync(d_forced, forced.data(), forced_bytes, cudaMemcpyHostToDevice, ctx->stream) != cudaSuccess ||
        cudaMemcpyAsync(d_C, pre.C_all.data.data(), C_bytes, cudaMemcpyHostToDevice, ctx->stream) != cudaSuccess) {
        std::cerr << "cudaMemcpyAsync inputs failed\n";
        cleanup();
        return 6;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (cublasSetStream(ctx->cublas, ctx->stream) != CUBLAS_STATUS_SUCCESS ||
        cublasSgemm(ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    H, n, d,
                    &alpha,
                    d_C, d,
                    d_X, d,
                    &beta,
                    d_xC, H) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasSgemm failed\n";
        cleanup();
        return 7;
    }

    stlq::ColMajorMatrix<float> xC_gpu(H, n);
    if (cudaMemcpyAsync(xC_gpu.data.data(), d_xC, xC_bytes, cudaMemcpyDeviceToHost, ctx->stream) != cudaSuccess ||
        cudaStreamSynchronize(ctx->stream) != cudaSuccess) {
        std::cerr << "copy xC back failed\n";
        cleanup();
        return 8;
    }

    // CPU reference: greedy forced-root init + DynamicIcmWithIlsAbsNoNormal using the same xC.
    stlq::ColMajorMatrix<stlq::FullCode> B_cpu;
    stlq::ColMajorMatrix<float> a_cpu;
    GreedyInitAbsForcedRootCpu(pre, xC_gpu, forced, &B_cpu, &a_cpu);
    std::vector<float> Xn2_cpu;
    std::vector<float> cost_cpu;
    stlq::DynamicIcmWithIlsAbsNoNormal(X, pre, xC_gpu, icm_iters, ils_iters, perturb_k, seed,
                                         &B_cpu, &a_cpu, &Xn2_cpu, &cost_cpu,
                                         /*print_progress=*/false,
                                         /*sample_id_offset=*/0);

    // GPU target.
    stlq::LinkageEncodeBatchCudaWorkspace* ws = stlq::CreateLinkageEncodeBatchCudaWorkspace();
    if (!ws) {
        std::cerr << "CreateLinkageEncodeBatchCudaWorkspace failed\n";
        cleanup();
        return 9;
    }
    stlq::SetCudaLinkageEncodeProfiling(false);
    if (!stlq::LinkageEncodeBatchCudaForcedRoot(*ctx, pre, d_X, d_norm2, d_forced,
                                                n, icm_iters, ils_iters, perturb_k, seed, 0,
                                                d_B, d_a, d_cost, ws, &err)) {
        std::cerr << "LinkageEncodeBatchCudaForcedRoot failed: " << err << "\n";
        stlq::DestroyLinkageEncodeBatchCudaWorkspace(ws);
        cleanup();
        return 10;
    }

    stlq::ColMajorMatrix<stlq::FullCode> B_gpu(m, n);
    std::vector<float> cost_gpu(static_cast<std::size_t>(n), 0.0f);
    if (cudaMemcpyAsync(B_gpu.data.data(), d_B, B_bytes, cudaMemcpyDeviceToHost, ctx->stream) != cudaSuccess ||
        cudaMemcpyAsync(cost_gpu.data(), d_cost, cost_bytes, cudaMemcpyDeviceToHost, ctx->stream) != cudaSuccess ||
        cudaStreamSynchronize(ctx->stream) != cudaSuccess) {
        std::cerr << "copy results failed\n";
        stlq::DestroyLinkageEncodeBatchCudaWorkspace(ws);
        cleanup();
        return 11;
    }

    stlq::DestroyLinkageEncodeBatchCudaWorkspace(ws);
    cleanup();

    int mism_B = 0;
    for (int i = 0; i < n; ++i) {
        for (int l = 0; l < m; ++l) {
            mism_B += (B_cpu(l, i) != B_gpu(l, i)) ? 1 : 0;
        }
    }
    double mean_cpu = 0.0;
    double mean_gpu = 0.0;
    for (int i = 0; i < n; ++i) {
        mean_cpu += static_cast<double>(cost_cpu[static_cast<std::size_t>(i)]);
        mean_gpu += static_cast<double>(cost_gpu[static_cast<std::size_t>(i)]);
    }
    mean_cpu /= static_cast<double>(n);
    mean_gpu /= static_cast<double>(n);
    const double diff = std::fabs(mean_gpu - mean_cpu);

    std::cout << "test_linkage_encode_forced_root_strict:\n";
    std::cout << "  mean_cost_cpu=" << mean_cpu << "\n";
    std::cout << "  mean_cost_gpu=" << mean_gpu << "\n";
    std::cout << "  abs_diff=" << diff << "\n";
    std::cout << "  mismatched_codes=" << mism_B << " / " << (n * m) << "\n";

    // This is intended to be a strict semantics-alignment test.
    if (diff > 1e-3 || mism_B != 0) {
        std::cerr << "FAIL: forced-root CPU/GPU mismatch\n";
        return 1;
    }
    return 0;
#endif
}
