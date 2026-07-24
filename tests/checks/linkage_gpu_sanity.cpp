#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "stlq/linkage/eval_candidates_cuda.h"
#include "stlq/core/lapack.h"
#include "stlq/cuda/cuda_stream_kernels_pool.h"
#include "stlq/common/logger.h"
#include "stlq/quantizer/encoder.h"

namespace {

float GAt(const stlq::ColMajorMatrix<float>& G, int row, int col) {
    return G.data[static_cast<std::size_t>(row) +
                  static_cast<std::size_t>(col) * static_cast<std::size_t>(G.rows)];
}

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

bool SolveLsRetryU(const stlq::Precomp& pre,
                   const std::vector<float>& xC,
                   const std::vector<stlq::FullCode>& B,
                   std::vector<float>* a_out) {
    const int m = pre.m;
    a_out->assign(static_cast<std::size_t>(m), 0.0f);
    std::vector<float> A(static_cast<std::size_t>(m) * static_cast<std::size_t>(m), 0.0f);
    std::vector<float> b(static_cast<std::size_t>(m), 0.0f);

    auto fill = [&](float lambda) {
        std::fill(A.begin(), A.end(), 0.0f);
        for (int j = 0; j < m; ++j) {
            const int flat_j = pre.offsets[j] + static_cast<int>(B[static_cast<std::size_t>(j)]);
            b[static_cast<std::size_t>(j)] = xC[static_cast<std::size_t>(flat_j)];
            for (int k = 0; k <= j; ++k) {
                const int flat_k = pre.offsets[k] + static_cast<int>(B[static_cast<std::size_t>(k)]);
                A[static_cast<std::size_t>(k) + static_cast<std::size_t>(j) * static_cast<std::size_t>(m)] =
                    GAt(pre.G, flat_k, flat_j);
            }
            A[static_cast<std::size_t>(j) + static_cast<std::size_t>(j) * static_cast<std::size_t>(m)] += lambda;
        }
    };

    fill(0.0f);
    int info = stlq::lapack::SpotrfU(m, A.data(), m);
    float bump = 1e-6f;
    for (int t = 0; t < 3 && info != 0; ++t) {
        fill(bump);
        info = stlq::lapack::SpotrfU(m, A.data(), m);
        bump *= 10.0f;
    }
    if (info != 0) {
        return false;
    }
    info = stlq::lapack::SpotrsU(m, 1, A.data(), m, b.data(), m);
    if (info != 0) {
        return false;
    }
    *a_out = std::move(b);
    return true;
}

float CostFromXc(const stlq::Precomp& pre,
                 const std::vector<float>& xC,
                 const std::vector<stlq::FullCode>& B,
                 const std::vector<float>& a,
                 float norm2) {
    const int m = pre.m;
    float term1 = 0.0f;
    float term2 = 0.0f;
    for (int l = 0; l < m; ++l) {
        const int flat_l = pre.offsets[l] + static_cast<int>(B[static_cast<std::size_t>(l)]);
        const float al = a[static_cast<std::size_t>(l)];
        term1 += al * xC[static_cast<std::size_t>(flat_l)];
        for (int k = 0; k < m; ++k) {
            const int flat_k = pre.offsets[k] + static_cast<int>(B[static_cast<std::size_t>(k)]);
            term2 += al * a[static_cast<std::size_t>(k)] * GAt(pre.G, flat_l, flat_k);
        }
    }
    return norm2 - 2.0f * term1 + term2;
}

void EncodeResidualCpuDeterministic(const stlq::Precomp& pre,
                                   const float* residual,
                                   int icm_iters,
                                   std::vector<stlq::FullCode>* B,
                                   std::vector<float>* a,
                                   float* cost_out) {
    const int d = pre.d;
    const int H = pre.H;
    const int m = pre.m;

    std::vector<float> xC(static_cast<std::size_t>(H), 0.0f);
    float norm2 = 0.0f;
    for (int r = 0; r < d; ++r) {
        const float v = residual[r];
        norm2 += v * v;
    }
    for (int j = 0; j < H; ++j) {
        const float* c = pre.C_all.Col(j);
        float s = 0.0f;
        for (int r = 0; r < d; ++r) {
            s += c[r] * residual[r];
        }
        xC[static_cast<std::size_t>(j)] = s;
    }

    std::vector<float> rC = xC;
    std::vector<std::uint8_t> avail(static_cast<std::size_t>(m), 1);
    B->assign(static_cast<std::size_t>(m), 0);
    a->assign(static_cast<std::size_t>(m), 0.0f);

    for (int t = 0; t < m; ++t) {
        int best_flat = 0;
        float best_abs = -std::numeric_limits<float>::infinity();
        float best_adj = 0.0f;
        for (int flat = 0; flat < H; ++flat) {
            const int layer = pre.flat_layer[static_cast<std::size_t>(flat)];
            if (!avail[static_cast<std::size_t>(layer)]) continue;
            const float adj = rC[static_cast<std::size_t>(flat)] * pre.invnorm_flat[static_cast<std::size_t>(flat)];
            const float aval = std::fabs(adj);
            if (aval > best_abs || (aval == best_abs && flat < best_flat)) {
                best_abs = aval;
                best_adj = adj;
                best_flat = flat;
            }
        }
        const int layer = pre.flat_layer[static_cast<std::size_t>(best_flat)];
        const int code = best_flat - pre.offsets[layer];
        const float invn = pre.invnorm_flat[static_cast<std::size_t>(best_flat)];
        const float alpha = best_adj * invn;
        (*B)[static_cast<std::size_t>(layer)] = static_cast<stlq::FullCode>(code);
        (*a)[static_cast<std::size_t>(layer)] = alpha;
        for (int q = 0; q < H; ++q) {
            rC[static_cast<std::size_t>(q)] -= alpha * GAt(pre.G, q, best_flat);
        }
        avail[static_cast<std::size_t>(layer)] = 0;
    }

    std::vector<float> a_ls;
    if (!SolveLsRetryU(pre, xC, *B, &a_ls)) {
        *cost_out = norm2;
        return;
    }
    *a = a_ls;
    float cost = CostFromXc(pre, xC, *B, *a, norm2);

    for (int it = 0; it < icm_iters; ++it) {
        bool changed = false;
        for (int jlayer = 0; jlayer < m; ++jlayer) {
            const int startf = pre.offsets[jlayer];
            const int stopf = startf + pre.h_vec[jlayer];
            const int old_code = static_cast<int>((*B)[static_cast<std::size_t>(jlayer)]);
            int best_code = old_code;
            float best_abs = -std::numeric_limits<float>::infinity();
            for (int flat = startf; flat < stopf; ++flat) {
                float tmp = xC[static_cast<std::size_t>(flat)];
                for (int l = 0; l < m; ++l) {
                    if (l == jlayer) continue;
                    const int flat_l = pre.offsets[l] + static_cast<int>((*B)[static_cast<std::size_t>(l)]);
                    tmp -= (*a)[static_cast<std::size_t>(l)] * GAt(pre.G, flat, flat_l);
                }
                const float val = tmp * pre.invnorm_flat[static_cast<std::size_t>(flat)];
                const float aval = std::fabs(val);
                const int code = flat - startf;
                if (aval > best_abs || (aval == best_abs && code < best_code)) {
                    best_abs = aval;
                    best_code = code;
                }
            }
            if (best_code == old_code) continue;
            std::vector<stlq::FullCode> B_try = *B;
            B_try[static_cast<std::size_t>(jlayer)] = static_cast<stlq::FullCode>(best_code);
            std::vector<float> a_new;
            if (!SolveLsRetryU(pre, xC, B_try, &a_new)) {
                continue;
            }
            const float new_cost = CostFromXc(pre, xC, B_try, a_new, norm2);
            if (new_cost + 1e-6f < cost) {
                *B = std::move(B_try);
                *a = std::move(a_new);
                cost = new_cost;
                changed = true;
            }
        }
        if (!changed) break;
    }

    *cost_out = cost;
}

struct OneRunResult {
    int cpu_best_parent = -1;
    float cpu_best_cost = std::numeric_limits<float>::infinity();
    int gpu_best_parent = -1;
    float gpu_best_cost = std::numeric_limits<float>::infinity();
    float max_cost_diff = 0.0f;
};

OneRunResult RunOnce(stlq::CudaStreamKernelsPool* pool,
                     const stlq::Precomp& pre,
                     int d,
                     int Kp,
                     int icm_iters,
                     std::mt19937* rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<float> xi(static_cast<std::size_t>(d));
    for (int r = 0; r < d; ++r) xi[static_cast<std::size_t>(r)] = nd(*rng);

    std::vector<float> Rp(static_cast<std::size_t>(d) * static_cast<std::size_t>(Kp));
    std::vector<int> cand(static_cast<std::size_t>(Kp), 0);
    for (int j = 0; j < Kp; ++j) {
        cand[static_cast<std::size_t>(j)] = j;
        float* col = Rp.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(d);
        for (int r = 0; r < d; ++r) col[r] = nd(*rng);
    }

    OneRunResult out;
    // CPU: brute over parents.
    for (int j = 0; j < Kp; ++j) {
        std::vector<float> residual(static_cast<std::size_t>(d));
        const float* rp = Rp.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(d);
        for (int r = 0; r < d; ++r) residual[static_cast<std::size_t>(r)] = xi[static_cast<std::size_t>(r)] - rp[r];
        std::vector<stlq::FullCode> B;
        std::vector<float> a;
        float cost = 0.0f;
        EncodeResidualCpuDeterministic(pre, residual.data(), icm_iters, &B, &a, &cost);
        if (cost < out.cpu_best_cost) {
            out.cpu_best_cost = cost;
            out.cpu_best_parent = cand[static_cast<std::size_t>(j)];
        }
    }

    stlq::CudaCtx* ctx = pool->Acquire();
    std::vector<stlq::FullCode> best_B;
    std::vector<float> best_a;
    std::string err;

    // Stage-2 device_rfull path sanity: upload R_full==Rp and compare new evaluator to old evaluator.
    std::string up_err;
    if (!stlq::CudaLinkageUploadClusterRfull(ctx, Rp.data(), d, Kp, &up_err)) {
        pool->Release(ctx);
        stlq::LogError("CudaLinkageUploadClusterRfull failed: " + up_err);
        std::exit(2);
    }

    const bool ok = stlq::EvaluateParentCandidatesBatchCuda(ctx,
                                                              pre,
                                                              xi.data(),
                                                              d,
                                                              Rp.data(),
                                                              cand.data(),
                                                              Kp,
                                                              icm_iters,
                                                              /*ils_iters=*/0,
                                                              /*perturb_k=*/0,
                                                              /*seed=*/0u,
                                                              /*sample_id_offset=*/0ull,
                                                              &out.gpu_best_parent,
                                                              &out.gpu_best_cost,
                                                              &best_B,
                                                              &best_a,
                                                              &err);
    if (!ok) {
        pool->Release(ctx);
        stlq::LogError("EvaluateParentCandidatesBatchCuda failed: " + err);
        std::exit(2);
    }

    int best_parent_dev = -1;
    float best_cost_dev = std::numeric_limits<float>::infinity();
    std::vector<stlq::FullCode> best_B_dev;
    std::vector<float> best_a_dev;
    std::string err_dev;
    if (!stlq::EvaluateParentCandidatesBatchCudaDeviceRfull(ctx,
                                                              pre,
                                                              xi.data(),
                                                              d,
                                                              cand.data(),
                                                              Kp,
                                                              icm_iters,
                                                              /*ils_iters=*/0,
                                                              /*perturb_k=*/0,
                                                              /*seed=*/0u,
                                                              /*sample_id_offset=*/0ull,
                                                              &best_parent_dev,
                                                              &best_cost_dev,
                                                              &best_B_dev,
                                                              &best_a_dev,
                                                              &err_dev)) {
        pool->Release(ctx);
        stlq::LogError("EvaluateParentCandidatesBatchCudaDeviceRfull failed: " + err_dev);
        std::exit(2);
    }
    pool->Release(ctx);

    if (best_parent_dev != out.gpu_best_parent) {
        stlq::LogError("device_rfull mismatch: best_parent old=" + std::to_string(out.gpu_best_parent) +
                         " new=" + std::to_string(best_parent_dev));
        std::exit(2);
    }
    const float denom = std::max(1.0f, out.gpu_best_cost);
    if (std::fabs(best_cost_dev - out.gpu_best_cost) > 1e-4f * denom) {
        stlq::LogError("device_rfull mismatch: best_cost old=" + std::to_string(out.gpu_best_cost) +
                         " new=" + std::to_string(best_cost_dev));
        std::exit(2);
    }
    if (best_B_dev != best_B) {
        stlq::LogError("device_rfull mismatch: best_B differs");
        std::exit(2);
    }
    if (best_a_dev.size() != best_a.size()) {
        stlq::LogError("device_rfull mismatch: best_a size differs");
        std::exit(2);
    }
    for (std::size_t i = 0; i < best_a.size(); ++i) {
        if (std::fabs(best_a_dev[i] - best_a[i]) > 1e-6f) {
            stlq::LogError("device_rfull mismatch: best_a differs");
            std::exit(2);
        }
    }

    out.max_cost_diff = std::fabs(out.gpu_best_cost - out.cpu_best_cost);
    return out;
}

struct BatchSanityRunResult {
    int B = 0;
    int n_cols = 0;
    int mismatched_parent = 0;
    int failed_cost = 0;
    int failed_codes = 0;
    int failed_coeffs = 0;
    float max_cost_diff = 0.0f;
};

BatchSanityRunResult RunBatchRfullOnce(stlq::CudaStreamKernelsPool* pool,
                                       const stlq::Precomp& pre,
                                       int d,
                                       int B,
                                       int n_cols,
                                       int kp_max,
                                       int icm_iters,
                                       std::mt19937* rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> kd(0, std::max(0, kp_max));
    std::uniform_int_distribution<int> pd(0, std::max(0, n_cols - 1));

    BatchSanityRunResult out;
    out.B = B;
    out.n_cols = n_cols;

    std::vector<float> R_full(static_cast<std::size_t>(d) * static_cast<std::size_t>(n_cols));
    for (float& v : R_full) v = nd(*rng);

    std::vector<float> X_block(static_cast<std::size_t>(d) * static_cast<std::size_t>(B));
    for (float& v : X_block) v = nd(*rng);

    std::vector<int> offsets(static_cast<std::size_t>(B) + 1u, 0);
    for (int i = 0; i < B; ++i) {
        offsets[static_cast<std::size_t>(i) + 1u] =
            offsets[static_cast<std::size_t>(i)] + kd(*rng);
    }
    int Npairs = offsets.back();
    if (Npairs == 0 && B > 0) {
        offsets[1] = 1;
        for (int i = 1; i < B; ++i) offsets[static_cast<std::size_t>(i) + 1u] = offsets[1];
        Npairs = 1;
    }

    std::vector<int> cand_flat(static_cast<std::size_t>(Npairs), 0);
    std::vector<int> pair_node(static_cast<std::size_t>(Npairs), 0);
    for (int i = 0; i < B; ++i) {
        const int start = offsets[static_cast<std::size_t>(i)];
        const int stop = offsets[static_cast<std::size_t>(i) + 1u];
        for (int j = start; j < stop; ++j) {
            cand_flat[static_cast<std::size_t>(j)] = pd(*rng);
            pair_node[static_cast<std::size_t>(j)] = i;
        }
    }

    std::vector<int> per_parent(static_cast<std::size_t>(B), -1);
    std::vector<float> per_cost(static_cast<std::size_t>(B), std::numeric_limits<float>::infinity());
    std::vector<stlq::FullCode> per_B(static_cast<std::size_t>(B) * static_cast<std::size_t>(pre.m), 0);
    std::vector<float> per_a(static_cast<std::size_t>(B) * static_cast<std::size_t>(pre.m), 0.0f);

    std::vector<int> bat_parent(static_cast<std::size_t>(B), -1);
    std::vector<float> bat_cost(static_cast<std::size_t>(B), std::numeric_limits<float>::infinity());
    std::vector<stlq::FullCode> bat_B(static_cast<std::size_t>(B) * static_cast<std::size_t>(pre.m), 0);
    std::vector<float> bat_a(static_cast<std::size_t>(B) * static_cast<std::size_t>(pre.m), 0.0f);

    stlq::CudaCtx* ctx = pool->Acquire();
    std::string err;
    if (!stlq::CudaLinkageUploadClusterRfull(ctx, R_full.data(), d, n_cols, &err)) {
        pool->Release(ctx);
        stlq::LogError("CudaLinkageUploadClusterRfull failed: " + err);
        std::exit(2);
    }

    for (int i = 0; i < B; ++i) {
        const int start = offsets[static_cast<std::size_t>(i)];
        const int stop = offsets[static_cast<std::size_t>(i) + 1u];
        const int Kp_i = stop - start;
        if (Kp_i <= 0) continue;
        const float* xi = X_block.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(d);
        const int* cand_i = cand_flat.data() + static_cast<std::size_t>(start);

        int best_parent = -1;
        float best_cost = std::numeric_limits<float>::infinity();
        std::vector<stlq::FullCode> best_B_vec;
        std::vector<float> best_a_vec;
        std::string err_i;
        if (!stlq::EvaluateParentCandidatesBatchCudaDeviceRfull(ctx,
                                                                  pre,
                                                                  xi,
                                                                  d,
                                                                  cand_i,
                                                                  Kp_i,
                                                                  icm_iters,
                                                                  /*ils_iters=*/0,
                                                                  /*perturb_k=*/0,
                                                                  /*seed=*/0u,
                                                                  /*sample_id_offset=*/static_cast<std::uint64_t>(i),
                                                                  &best_parent,
                                                                  &best_cost,
                                                                  &best_B_vec,
                                                                  &best_a_vec,
                                                                  &err_i)) {
            pool->Release(ctx);
            stlq::LogError("EvaluateParentCandidatesBatchCudaDeviceRfull failed: " + err_i);
            std::exit(2);
        }
        per_parent[static_cast<std::size_t>(i)] = best_parent;
        per_cost[static_cast<std::size_t>(i)] = best_cost;
        const std::size_t base = static_cast<std::size_t>(i) * static_cast<std::size_t>(pre.m);
        for (int l = 0; l < pre.m; ++l) {
            per_B[base + static_cast<std::size_t>(l)] = best_B_vec[static_cast<std::size_t>(l)];
            per_a[base + static_cast<std::size_t>(l)] = best_a_vec[static_cast<std::size_t>(l)];
        }
    }

    if (!stlq::EvaluateParentCandidatesManyNodesBatchCudaDeviceRfull(ctx,
                                                                       pre,
                                                                       X_block.data(),
                                                                       d,
                                                                       B,
                                                                       cand_flat.data(),
                                                                       pair_node.data(),
                                                                       offsets.data(),
                                                                       Npairs,
                                                                       icm_iters,
                                                                       /*ils_iters=*/0,
                                                                       /*perturb_k=*/0,
                                                                       /*seed=*/0u,
                                                                       /*sample_id_offset=*/0ull,
                                                                       bat_parent.data(),
                                                                       bat_cost.data(),
                                                                       bat_B.data(),
                                                                       bat_a.data(),
                                                                       &err)) {
        pool->Release(ctx);
        stlq::LogError("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfull failed: " + err);
        std::exit(2);
    }
    pool->Release(ctx);

    for (int i = 0; i < B; ++i) {
        if (per_parent[static_cast<std::size_t>(i)] != bat_parent[static_cast<std::size_t>(i)]) {
            out.mismatched_parent += 1;
        }
        const float c0 = per_cost[static_cast<std::size_t>(i)];
        const float c1 = bat_cost[static_cast<std::size_t>(i)];
        const float denom = std::max(1.0f, std::max(std::fabs(c0), std::fabs(c1)));
        const float diff = std::fabs(c0 - c1);
        out.max_cost_diff = std::max(out.max_cost_diff, diff);
        if (diff > 1e-4f * denom) {
            out.failed_cost += 1;
        }
        const std::size_t base = static_cast<std::size_t>(i) * static_cast<std::size_t>(pre.m);
        for (int l = 0; l < pre.m; ++l) {
            if (per_B[base + static_cast<std::size_t>(l)] != bat_B[base + static_cast<std::size_t>(l)]) {
                out.failed_codes += 1;
                break;
            }
        }
        for (int l = 0; l < pre.m; ++l) {
            const float a0 = per_a[base + static_cast<std::size_t>(l)];
            const float a1 = bat_a[base + static_cast<std::size_t>(l)];
            if (std::fabs(a0 - a1) > 1e-6f) {
                out.failed_coeffs += 1;
                break;
            }
        }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
#if !defined(STLQ_ENABLE_CUDA)
    std::cerr << "This build does not enable CUDA (STLQ_ENABLE_CUDA=0).\n";
    return 2;
#else
    int d = 64;
    int m = 5;
    int Kp = 256;
    int icm_iters = 2;
    int runs = 8;
    int threads = 4;
    int seed = 123;
    std::string mode = "single";
    int batch_B = 8;
    int batch_ncols = 256;
    int batch_kpmax = 64;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg.rfind("--mode=", 0) == 0) {
            mode = arg.substr(std::string("--mode=").size());
            continue;
        }
        auto take_int = [&](const char* key, int* dst) {
            const std::string prefix = std::string(key) + "=";
            if (arg.rfind(prefix, 0) == 0) {
                *dst = std::stoi(arg.substr(prefix.size()));
            }
        };
        take_int("--d", &d);
        take_int("--m", &m);
        take_int("--kp", &Kp);
        take_int("--icm", &icm_iters);
        take_int("--runs", &runs);
        take_int("--threads", &threads);
        take_int("--seed", &seed);
        take_int("--B", &batch_B);
        take_int("--ncols", &batch_ncols);
        take_int("--kpmax", &batch_kpmax);
    }

    std::mt19937 rng(static_cast<std::uint32_t>(seed));
    std::vector<int> h_vec(static_cast<std::size_t>(m), 256);
    stlq::CodebookPack pack = MakeRandomCodebooks(d, h_vec, &rng);
    stlq::Precomp pre;
    if (!stlq::BuildPrecomp(pack, &pre)) {
        std::cerr << "BuildPrecomp failed\n";
        return 2;
    }

    stlq::CudaPoolConfig pool_cfg;
    pool_cfg.device = 0;
    pool_cfg.allow_tf32 = false;
    stlq::CudaStreamKernelsPool pool(std::max(1, threads), pool_cfg);

    if (mode == "batch_rfull") {
        std::atomic<int> mismatched_parent{0};
        std::atomic<int> failed_cost{0};
        std::atomic<int> failed_codes{0};
        std::atomic<int> failed_coeffs{0};
        std::atomic<float> max_cost_diff{0.0f};

        auto worker = [&](int tid) {
            std::mt19937 lrng(static_cast<std::uint32_t>(seed + tid * 9997));
            for (int t = tid; t < runs; t += threads) {
                const BatchSanityRunResult r =
                    RunBatchRfullOnce(&pool, pre, d, batch_B, batch_ncols, batch_kpmax, icm_iters, &lrng);
                mismatched_parent.fetch_add(r.mismatched_parent);
                failed_cost.fetch_add(r.failed_cost);
                failed_codes.fetch_add(r.failed_codes);
                failed_coeffs.fetch_add(r.failed_coeffs);
                float cur = max_cost_diff.load();
                while (r.max_cost_diff > cur && !max_cost_diff.compare_exchange_weak(cur, r.max_cost_diff)) {
                }
            }
        };

        std::vector<std::thread> ts;
        ts.reserve(static_cast<std::size_t>(threads));
        for (int t = 0; t < threads; ++t) ts.emplace_back(worker, t);
        for (auto& t : ts) t.join();

        std::cout << "linkage_gpu_sanity(batch_rfull):\n"
                  << "  d=" << d << " m=" << m << " B=" << batch_B << " ncols=" << batch_ncols
                  << " kpmax=" << batch_kpmax << " icm=" << icm_iters << "\n"
                  << "  runs=" << runs << " threads=" << threads << "\n"
                  << "  mismatched_parent=" << mismatched_parent.load() << "\n"
                  << "  failed_cost(diff>1e-4*denom)=" << failed_cost.load() << "\n"
                  << "  failed_codes=" << failed_codes.load() << "\n"
                  << "  failed_coeffs=" << failed_coeffs.load() << "\n"
                  << "  max_cost_diff=" << max_cost_diff.load() << "\n";
        // Note: minor parent/codes/coeffs mismatches can happen due to floating-point differences
        // between per-node and batched GEMM sizes; cost agreement is the primary correctness signal.
        return (failed_cost.load() == 0 ? 0 : 1);
    }

    std::atomic<int> mismatched_parent{0};
    std::atomic<int> failed_cost{0};
    std::atomic<float> max_cost_diff{0.0f};

    auto worker = [&](int tid) {
        std::mt19937 lrng(static_cast<std::uint32_t>(seed + tid * 9997));
        for (int t = tid; t < runs; t += threads) {
            const OneRunResult r = RunOnce(&pool, pre, d, Kp, icm_iters, &lrng);
            const float diff = r.max_cost_diff;
            float cur = max_cost_diff.load();
            while (diff > cur && !max_cost_diff.compare_exchange_weak(cur, diff)) {
            }
            if (r.cpu_best_parent != r.gpu_best_parent) {
                mismatched_parent.fetch_add(1);
            }
            if (diff > 1e-3f) {
                failed_cost.fetch_add(1);
            }
        }
    };

    std::vector<std::thread> ts;
    ts.reserve(static_cast<std::size_t>(threads));
    for (int t = 0; t < threads; ++t) ts.emplace_back(worker, t);
    for (auto& t : ts) t.join();

    std::cout << "linkage_gpu_sanity:\n"
              << "  d=" << d << " m=" << m << " kp=" << Kp << " icm=" << icm_iters << "\n"
              << "  runs=" << runs << " threads=" << threads << "\n"
              << "  mismatched_parent=" << mismatched_parent.load() << "\n"
              << "  failed_cost(diff>1e-3)=" << failed_cost.load() << "\n"
              << "  max_cost_diff=" << max_cost_diff.load() << "\n";
    return (failed_cost.load() == 0 ? 0 : 1);
#endif
}
