#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "stlq/linkage/linkage_encode_cuda.h"
#include "stlq/core/blas.h"
#include "stlq/quantizer/encoder.h"

#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
#include <cuda_runtime.h>
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
        for (int r = 0; r < d; ++r) col[r] = nd(*rng);
    }
    return X;
}

std::vector<float> ComputeXNorm2(const stlq::ColMajorMatrix<float>& X) {
    std::vector<float> out(static_cast<std::size_t>(std::max(0, X.cols)), 0.0f);
    for (int i = 0; i < X.cols; ++i) {
        const float* x = X.Col(i);
        float s = 0.0f;
        for (int r = 0; r < X.rows; ++r) s += x[r] * x[r];
        out[static_cast<std::size_t>(i)] = s;
    }
    return out;
}

#if !defined(STLQ_ENABLE_CUDA) || !STLQ_ENABLE_CUDA
}  // namespace

int main() {
    std::cerr << "SKIP: built without CUDA\n";
    return 0;
}
#else

int RunCase(int m) {
    const int d = 64;
    const int n = 256;
    const int h0 = 256;  // Must fit in FullCode (u8). Large-root path tests the algorithm, not the type range.
    const int h_small = 256;
    const int icm_iters = 4;
    const int ils_iters = 6;
    const int perturb_k = 3;
    const std::uint32_t seed = 12345;
    const int forced_root_code = 200;

    std::mt19937 rng(7);
    std::vector<int> h_vec(static_cast<std::size_t>(m), h_small);
    h_vec[0] = h0;
    stlq::CodebookPack pack = MakeRandomCodebooks(d, h_vec, &rng);

    stlq::Precomp pre;
    if (!stlq::BuildPrecomp(pack, &pre)) {
        std::cerr << "BuildPrecomp failed\n";
        return 2;
    }

    stlq::ColMajorMatrix<float> X = MakeRandomX(d, n, &rng);
    const std::vector<float> X_norm2 = ComputeXNorm2(X);

    std::vector<int> forced(static_cast<std::size_t>(n), forced_root_code);
    std::vector<std::uint64_t> sample_ids(static_cast<std::size_t>(n), 0);
    for (int i = 0; i < n; ++i) sample_ids[static_cast<std::size_t>(i)] = static_cast<std::uint64_t>(i);

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
    std::uint64_t* d_sample_ids = nullptr;
    stlq::FullCode* d_B_full = nullptr;
    float* d_a_full = nullptr;
    float* d_cost_full = nullptr;
    stlq::FullCode* d_B_lr = nullptr;
    float* d_a_lr = nullptr;
    float* d_cost_lr = nullptr;

    auto cleanup = [&]() {
        if (d_cost_lr) cudaFree(d_cost_lr);
        if (d_a_lr) cudaFree(d_a_lr);
        if (d_B_lr) cudaFree(d_B_lr);
        if (d_cost_full) cudaFree(d_cost_full);
        if (d_a_full) cudaFree(d_a_full);
        if (d_B_full) cudaFree(d_B_full);
        if (d_sample_ids) cudaFree(d_sample_ids);
        if (d_forced) cudaFree(d_forced);
        if (d_norm2) cudaFree(d_norm2);
        if (d_X) cudaFree(d_X);
        d_cost_lr = nullptr;
        d_a_lr = nullptr;
        d_B_lr = nullptr;
        d_cost_full = nullptr;
        d_a_full = nullptr;
        d_B_full = nullptr;
        d_sample_ids = nullptr;
        d_forced = nullptr;
        d_norm2 = nullptr;
        d_X = nullptr;
    };

    const std::size_t X_bytes = sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(n);
    const std::size_t n_f32_bytes = sizeof(float) * static_cast<std::size_t>(n);
    const std::size_t forced_bytes = sizeof(int) * static_cast<std::size_t>(n);
    const std::size_t ids_bytes = sizeof(std::uint64_t) * static_cast<std::size_t>(n);
    const std::size_t B_bytes = sizeof(stlq::FullCode) * static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
    const std::size_t a_bytes = sizeof(float) * static_cast<std::size_t>(m) * static_cast<std::size_t>(n);

    if (cudaMalloc(&d_X, X_bytes) != cudaSuccess) {
        std::cerr << "cudaMalloc(d_X) failed\n";
        cleanup();
        return 4;
    }
    if (cudaMalloc(&d_norm2, n_f32_bytes) != cudaSuccess ||
        cudaMalloc(&d_forced, forced_bytes) != cudaSuccess ||
        cudaMalloc(&d_sample_ids, ids_bytes) != cudaSuccess ||
        cudaMalloc(&d_B_full, B_bytes) != cudaSuccess ||
        cudaMalloc(&d_a_full, a_bytes) != cudaSuccess ||
        cudaMalloc(&d_cost_full, n_f32_bytes) != cudaSuccess ||
        cudaMalloc(&d_B_lr, B_bytes) != cudaSuccess ||
        cudaMalloc(&d_a_lr, a_bytes) != cudaSuccess ||
        cudaMalloc(&d_cost_lr, n_f32_bytes) != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
        cleanup();
        return 5;
    }

    if (cudaMemcpy(d_X, X.data.data(), X_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_norm2, X_norm2.data(), n_f32_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_forced, forced.data(), forced_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_sample_ids, sample_ids.data(), ids_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "cudaMemcpy inputs failed\n";
        cleanup();
        return 6;
    }

    std::string err_full;
    std::string err_lr;
    stlq::LinkageEncodeBatchCudaWorkspace* ws_full = stlq::CreateLinkageEncodeBatchCudaWorkspace();
    stlq::LinkageEncodeBatchCudaWorkspace* ws_lr = stlq::CreateLinkageEncodeBatchCudaWorkspace();
    if (!ws_full || !ws_lr) {
        std::cerr << "CreateLinkageEncodeBatchCudaWorkspace failed\n";
        cleanup();
        return 7;
    }

    const bool ok_full =
        stlq::LinkageEncodeBatchCudaForcedRootWithSampleIds(*ctx, pre, d_X, d_norm2, d_forced, d_sample_ids,
                                                           n, icm_iters, ils_iters, perturb_k, seed,
                                                           d_B_full, d_a_full, d_cost_full, ws_full, &err_full);
    const bool ok_lr =
        stlq::LinkageEncodeBatchCudaForcedRootWithSampleIdsLargeRootConstRoot(*ctx, pack, forced_root_code,
                                                                             d_X, d_norm2, d_sample_ids,
                                                                             n, icm_iters, ils_iters, perturb_k, seed,
                                                                             d_B_lr, d_a_lr, d_cost_lr, ws_lr, &err_lr);
    if (!ok_full || !ok_lr) {
        std::cerr << "Encode failed: full=" << ok_full << " lr=" << ok_lr << "\n";
        if (!err_full.empty()) std::cerr << "full err: " << err_full << "\n";
        if (!err_lr.empty()) std::cerr << "lr err: " << err_lr << "\n";
        stlq::DestroyLinkageEncodeBatchCudaWorkspace(ws_full);
        stlq::DestroyLinkageEncodeBatchCudaWorkspace(ws_lr);
        cleanup();
        return 8;
    }

    std::vector<stlq::FullCode> B_full(static_cast<std::size_t>(m) * static_cast<std::size_t>(n), 0);
    std::vector<stlq::FullCode> B_lr(static_cast<std::size_t>(m) * static_cast<std::size_t>(n), 0);
    std::vector<float> a_full(static_cast<std::size_t>(m) * static_cast<std::size_t>(n), 0.0f);
    std::vector<float> a_lr(static_cast<std::size_t>(m) * static_cast<std::size_t>(n), 0.0f);
    std::vector<float> cost_full(static_cast<std::size_t>(n), 0.0f);
    std::vector<float> cost_lr(static_cast<std::size_t>(n), 0.0f);

    if (cudaMemcpy(B_full.data(), d_B_full, B_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(B_lr.data(), d_B_lr, B_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(a_full.data(), d_a_full, a_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(a_lr.data(), d_a_lr, a_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(cost_full.data(), d_cost_full, n_f32_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(cost_lr.data(), d_cost_lr, n_f32_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "cudaMemcpy outputs failed\n";
        stlq::DestroyLinkageEncodeBatchCudaWorkspace(ws_full);
        stlq::DestroyLinkageEncodeBatchCudaWorkspace(ws_lr);
        cleanup();
        return 9;
    }

    // Compare.
    int root_mismatch = 0;
    int samples_mismatch = 0;
    int elems_mismatch = 0;
    std::vector<std::string> examples;
    for (int i = 0; i < n; ++i) {
        const std::size_t base = static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
        if (static_cast<int>(B_full[base + 0]) != forced_root_code ||
            static_cast<int>(B_lr[base + 0]) != forced_root_code) {
            std::cerr << "Root code mismatch at i=" << i << "\n";
            root_mismatch++;
            break;
        }
        bool any = false;
        for (int l = 1; l < m; ++l) {
            if (B_full[base + static_cast<std::size_t>(l)] != B_lr[base + static_cast<std::size_t>(l)]) {
                elems_mismatch++;
                any = true;
                if (examples.size() < 4) {
                    examples.push_back("B mismatch at i=" + std::to_string(i) + " l=" + std::to_string(l) +
                                       " full=" + std::to_string(static_cast<int>(B_full[base + static_cast<std::size_t>(l)])) +
                                       " lr=" + std::to_string(static_cast<int>(B_lr[base + static_cast<std::size_t>(l)])));
                }
            }
        }
        if (any) samples_mismatch++;
    }

    double max_da = 0.0;
    for (std::size_t i = 0; i < a_lr.size(); ++i) {
        if (!std::isfinite(a_lr[i])) {
            std::cerr << "Non-finite a_lr at idx=" << i << "\n";
            root_mismatch++;
            break;
        }
    }
    for (std::size_t i = 0; i < a_full.size(); ++i) {
        max_da = std::max(max_da, std::abs(static_cast<double>(a_full[i]) - static_cast<double>(a_lr[i])));
    }
    double max_dcost = 0.0;
    double max_rel_dcost = 0.0;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(cost_full[static_cast<std::size_t>(i)]) || !std::isfinite(cost_lr[static_cast<std::size_t>(i)])) {
            std::cerr << "Non-finite cost at i=" << i << "\n";
            root_mismatch++;
            break;
        }
        const auto cf = static_cast<double>(cost_full[static_cast<std::size_t>(i)]);
        const auto cl = static_cast<double>(cost_lr[static_cast<std::size_t>(i)]);
        const double diff = std::abs(cf - cl);
        max_dcost = std::max(max_dcost, diff);
        const double denom = std::max(1.0, std::abs(cf));
        max_rel_dcost = std::max(max_rel_dcost, diff / denom);
    }

    // Exact bitwise matching is not expected here:
    // - full-precomp uses a single GEMM for xC over (h0+Hs) rows;
    // - large-root uses split GEMMs for (Hs) and (1) rows.
    // Different GEMM tiling/accumulation can cause small acceptance/tie-break drift.
    const double max_rel_cost_tol = (m == 7) ? 0.15 : 0.05;
    const double sample_mismatch_tol = 0.20;   // <=20% samples may differ in codes
    if (root_mismatch > 0 ||
        max_rel_dcost > max_rel_cost_tol ||
        static_cast<double>(samples_mismatch) > sample_mismatch_tol * static_cast<double>(n)) {
        for (const auto& s : examples) {
            std::cerr << s << "\n";
        }
        std::cerr << "Mismatch: root_mismatch=" << root_mismatch
                  << " samples_mismatch=" << samples_mismatch << "/" << n
                  << " elems_mismatch=" << elems_mismatch << "/" << (n * (m - 1))
                  << " max|da|=" << max_da
                  << " max|dcost|=" << max_dcost
                  << " max_rel_dcost=" << max_rel_dcost << " (tol " << max_rel_cost_tol << ")\n";
        stlq::DestroyLinkageEncodeBatchCudaWorkspace(ws_full);
        stlq::DestroyLinkageEncodeBatchCudaWorkspace(ws_lr);
        cleanup();
        return 10;
    }

    std::string err_variable_root;
    const bool ok_variable_root = stlq::LinkageEncodeBatchCudaWithSampleIdsLargeRoot(
        *ctx, pack, d_X, d_norm2, d_sample_ids, n,
        icm_iters, ils_iters, perturb_k, seed,
        d_B_lr, d_a_lr, d_cost_lr, ws_lr, &err_variable_root);
    if (!ok_variable_root || cudaStreamSynchronize(ctx->stream) != cudaSuccess ||
        cudaMemcpy(B_lr.data(), d_B_lr, B_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(a_lr.data(), d_a_lr, a_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(cost_lr.data(), d_cost_lr, n_f32_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "Variable-root large-root encode failed for m=" << m
                  << ": " << err_variable_root << "\n";
        stlq::DestroyLinkageEncodeBatchCudaWorkspace(ws_full);
        stlq::DestroyLinkageEncodeBatchCudaWorkspace(ws_lr);
        cleanup();
        return 11;
    }
    for (int i = 0; i < n; ++i) {
        const std::size_t base = static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
        if (!std::isfinite(cost_lr[static_cast<std::size_t>(i)])) {
            std::cerr << "Non-finite variable-root cost for m=" << m << " sample=" << i << "\n";
            stlq::DestroyLinkageEncodeBatchCudaWorkspace(ws_full);
            stlq::DestroyLinkageEncodeBatchCudaWorkspace(ws_lr);
            cleanup();
            return 12;
        }
        for (int l = 0; l < m; ++l) {
            if (static_cast<int>(B_lr[base + static_cast<std::size_t>(l)]) >= h_vec[static_cast<std::size_t>(l)] ||
                !std::isfinite(a_lr[base + static_cast<std::size_t>(l)])) {
                std::cerr << "Invalid variable-root output for m=" << m
                          << " sample=" << i << " layer=" << l << "\n";
                stlq::DestroyLinkageEncodeBatchCudaWorkspace(ws_full);
                stlq::DestroyLinkageEncodeBatchCudaWorkspace(ws_lr);
                cleanup();
                return 13;
            }
        }
    }

    stlq::DestroyLinkageEncodeBatchCudaWorkspace(ws_full);
    stlq::DestroyLinkageEncodeBatchCudaWorkspace(ws_lr);
    cleanup();
    std::cout << "OK m=" << m << "\n";
    return 0;
}

}  // namespace

int main() {
    if (const int rc = RunCase(5); rc != 0) return rc;
    if (const int rc = RunCase(6); rc != 0) return rc;
    if (const int rc = RunCase(7); rc != 0) return rc;
    return 0;
}
#endif
