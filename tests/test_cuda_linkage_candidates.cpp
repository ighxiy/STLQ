#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
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
            const int old_flat = startf + old_code;
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
            std::vector<float> a_try = *a;
            B_try[static_cast<std::size_t>(jlayer)] = static_cast<stlq::FullCode>(best_code);
            std::vector<float> a_new;
            if (!SolveLsRetryU(pre, xC, B_try, &a_new)) {
                continue;
            }
            a_try = a_new;
            const float new_cost = CostFromXc(pre, xC, B_try, a_try, norm2);
            if (new_cost + 1e-6f < cost) {
                *B = std::move(B_try);
                *a = std::move(a_try);
                cost = new_cost;
                changed = true;
            }
            (void)old_flat;
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

    // CPU reference.
    std::vector<float> re(static_cast<std::size_t>(d));
    for (int j = 0; j < Kp; ++j) {
        const float* rp = Rp.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(d);
        for (int r = 0; r < d; ++r) re[static_cast<std::size_t>(r)] = xi[static_cast<std::size_t>(r)] - rp[r];
        std::vector<stlq::FullCode> B;
        std::vector<float> a;
        float cost = 0.0f;
        EncodeResidualCpuDeterministic(pre, re.data(), icm_iters, &B, &a, &cost);
        if (cost < out.cpu_best_cost) {
            out.cpu_best_cost = cost;
            out.cpu_best_parent = j;
        }
    }

    // GPU.
    stlq::CudaCtx* ctx = pool->Acquire();
    std::vector<stlq::FullCode> Bbest;
    std::vector<float> abest;
    std::string err;
    if (!stlq::EvaluateParentCandidatesBatchCuda(ctx,
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
                                                   &Bbest,
                                                   &abest,
                                                   &err)) {
        pool->Release(ctx);
        throw std::runtime_error(err);
    }
    pool->Release(ctx);
    return out;
}

}  // namespace

int main() {
    using namespace stlq;

    // Build a small random codebook pack.
    CodebookPack pack;
    pack.d = 32;
    pack.h_vec = {64, 64, 64, 64, 64};
    pack.books.resize(pack.h_vec.size());
    std::mt19937 rng(123);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (std::size_t l = 0; l < pack.books.size(); ++l) {
        const int h = pack.h_vec[l];
        pack.books[l] = ColMajorMatrix<float>(pack.d, h);
        for (int j = 0; j < h; ++j) {
            float* col = pack.books[l].Col(j);
            for (int r = 0; r < pack.d; ++r) {
                col[r] = nd(rng);
            }
        }
    }

    Precomp pre;
    if (!BuildPrecomp(pack, &pre)) {
        std::cerr << "BuildPrecomp failed\n";
        return 1;
    }

    // Concurrency sanity: multiple contexts + threads.
    CudaPoolConfig pcfg;
    pcfg.device = 0;
    pcfg.allow_tf32 = false;
    CudaStreamKernelsPool pool(/*num_ctx=*/4, pcfg);

    std::atomic<int> failures{0};
    constexpr int kThreads = 4;
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            try {
                std::mt19937 local_rng(1000 + t);
                const int Kp = 128;
                const int icm_iters = 1;
                OneRunResult res = RunOnce(&pool, pre, pack.d, Kp, icm_iters, &local_rng);
                const float rel = std::fabs(res.cpu_best_cost - res.gpu_best_cost) /
                                  std::max(1.0f, std::fabs(res.cpu_best_cost));
                if (res.cpu_best_parent != res.gpu_best_parent && rel > 1e-3f) {
                    failures.fetch_add(1);
                }
            } catch (...) {
                failures.fetch_add(1);
            }
        });
    }
    for (auto& th : threads) th.join();

    if (failures.load() != 0) {
        std::cerr << "test_cuda_linkage_candidates: failures=" << failures.load() << "\n";
        return 1;
    }
    return 0;
}
