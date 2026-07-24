#include "stlq/quantizer/icm.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include <omp.h>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include "stlq/common/logger.h"
#include "stlq/quantizer/cost_utils.h"
#include "stlq/quantizer/least_squares.h"
#include "stlq/quantizer/linear_algebra.h"
#include "stlq/quantizer/precomp_large_root.h"

namespace stlq {

namespace {

constexpr float kEps = 1e-6f;
constexpr int kIcmLargeRootSampleParallelMinH0 = 1024;

struct IcmTls {
    std::vector<float> A;
    std::vector<float> b;
    std::vector<float> work;
    std::vector<float> backup;
    std::vector<float> tmp_vals;

    void Ensure(int m, int max_h) {
        const std::size_t mm = static_cast<std::size_t>(m) * static_cast<std::size_t>(m);
        A.resize(mm);
        b.resize(static_cast<std::size_t>(m));
        work.resize(static_cast<std::size_t>(m));
        backup.resize(static_cast<std::size_t>(m));
        tmp_vals.resize(static_cast<std::size_t>(max_h));
    }
};

inline IcmTls& GetIcmTls(int m, int max_h) {
    static thread_local IcmTls tls;
    tls.Ensure(m, max_h);
    return tls;
}

struct SplitMix64 {
    std::uint64_t state = 0;
    explicit SplitMix64(std::uint64_t seed) : state(seed) {}

    std::uint64_t NextU64() {
        std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31U);
    }
};

inline float DotF32(const float* a, const float* b, int d) {
    float s = 0.0f;
    #pragma omp simd reduction(+:s)
    for (int i = 0; i < d; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

inline std::uint32_t UniformBelow(SplitMix64* rng, std::uint32_t bound) {
    // Lemire-style multiply-high (no rejection): fast, good enough for ILS perturbations.
    const std::uint64_t x = rng->NextU64();
#if defined(_MSC_VER)
    unsigned __int64 hi = 0;
    (void)_umul128(static_cast<unsigned __int64>(x), static_cast<unsigned __int64>(bound), &hi);
    return static_cast<std::uint32_t>(hi);
#else
    const __uint128_t prod = static_cast<__uint128_t>(x) * static_cast<__uint128_t>(bound);
    return static_cast<std::uint32_t>(prod >> 64);
#endif
}

template <bool UseAbs>
void RunIcm(const ColMajorMatrix<float>& X,
            const Precomp& precomp,
            const ColMajorMatrix<float>& xC,
            int icm_iters,
            std::vector<uint8_t>* active,
            std::vector<uint8_t>* changed_iter,
            std::vector<int>* order,
            int max_h,
            ColMajorMatrix<FullCode>* B,
            ColMajorMatrix<float>* a,
            std::vector<float>* X_norm2,
            std::vector<float>* cur_cost,
            std::mt19937* rng) {
    // Julia reference: dynamic_icm_encoding_noabs_nonormal_clean_fast_for_ils1!
    // (encode_noabs.jl:119-213).
    const int n = X.cols;
    const int m = precomp.m;
    int ls_failures = 0;  // count rollback-triggering LS solve failures; logged at exit

    SolveLeastSquaresAll(precomp, xC, *B, a);

    if (X_norm2->empty()) {
        ComputeXNorm2(X, X_norm2);
    }
    ComputeCosts(precomp, xC, *B, *a, *X_norm2, cur_cost);

    if (!active || !changed_iter || !order) {
        return;
    }
    active->assign(static_cast<std::size_t>(n), 1);
    changed_iter->assign(static_cast<std::size_t>(n), 0);
    order->resize(static_cast<std::size_t>(m));
    std::iota(order->begin(), order->end(), 0);

    // When called from an outer OpenMP region (e.g. cluster-parallel linkage build),
    // nested parallelism is typically disabled. In that case the `#pragma omp parallel`
    // region below would serialize anyway but still costs runtime overhead.
    // Use a serial implementation (semantics identical).
    if (omp_in_parallel()) {
        IcmTls& tls = GetIcmTls(m, max_h);
        std::vector<float>& A = tls.A;
        std::vector<float>& b = tls.b;
        std::vector<float>& work = tls.work;
        std::vector<float>& backup = tls.backup;
        std::vector<float>& tmp_vals = tls.tmp_vals;

        for (int iter = 0; iter < icm_iters; ++iter) {
            std::shuffle(order->begin(), order->end(), *rng);
            for (int i = 0; i < n; ++i) {
                (*changed_iter)[static_cast<std::size_t>(i)] = 0;
            }

            for (int jlayer : *order) {
                if (jlayer == 0) {
                    continue;
                }
                const int startf = precomp.offsets[jlayer];
                const int hj = precomp.h_vec[jlayer];
                if (hj <= 1) {
                    continue;
                }

                for (int i = 0; i < n; ++i) {
                    if (!(*active)[static_cast<std::size_t>(i)]) {
                        continue;
                    }

                    const int old_code = static_cast<int>((*B)(jlayer, i));

                    const float* xC_i = xC.Col(i) + startf;
                    #pragma omp simd
                    for (int t = 0; t < hj; ++t) {
                        tmp_vals[t] = xC_i[t];
                    }
                    for (int l = 0; l < m; ++l) {
                        if (l == jlayer) {
                            continue;
                        }
                        const float alpha = (*a)(l, i);
                        const int flat_l = precomp.offsets[l] + static_cast<int>((*B)(l, i));
                        const float* G_col = precomp.G.Col(flat_l) + startf;
                        #pragma omp simd
                        for (int t = 0; t < hj; ++t) {
                            tmp_vals[t] -= alpha * G_col[t];
                        }
                    }

                    int best_code = old_code;
                    float best_score = -std::numeric_limits<float>::infinity();
                    const float* inv = precomp.invnorm_flat.data() + startf;
                    for (int t = 0; t < hj; ++t) {
                        const float val = tmp_vals[t] * inv[t];
                        const float score = UseAbs ? std::abs(val) : val;
                        if (score > best_score) {
                            best_score = score;
                            best_code = t;
                        }
                    }

                    if (best_code == old_code) {
                        continue;
                    }

                    float* a_i = a->Col(i);
                    std::memcpy(backup.data(), a_i, sizeof(float) * static_cast<std::size_t>(m));

                    (*B)(jlayer, i) = static_cast<FullCode>(best_code);

                    if (!SolveLeastSquaresSample(precomp, xC, *B, i, &A, &b, &work, a_i)) {
                        (*B)(jlayer, i) = static_cast<FullCode>(old_code);
                        std::memcpy(a_i, backup.data(), sizeof(float) * static_cast<std::size_t>(m));
                        ++ls_failures;
                        continue;
                    }

                    const float new_cost = SampleCost(precomp, xC, *B, *a, i, (*X_norm2)[i]);
                    if (new_cost + kEps < (*cur_cost)[i]) {
                        (*cur_cost)[i] = new_cost;
                        (*changed_iter)[static_cast<std::size_t>(i)] = 1;
                    } else {
                        (*B)(jlayer, i) = static_cast<FullCode>(old_code);
                        std::memcpy(a_i, backup.data(), sizeof(float) * static_cast<std::size_t>(m));
                    }
                }
            }

            for (int i = 0; i < n; ++i) {
                const auto ii = static_cast<std::size_t>(i);
                if ((*active)[ii] && !(*changed_iter)[ii]) {
                    (*active)[ii] = 0;
                }
            }
        }
        if (ls_failures > 0) {
            LogWarn("RunIcm: " + std::to_string(ls_failures) +
                    " LS solve failures (serial path); codebook may be ill-conditioned.");
        }
        return;
    }

    #pragma omp parallel default(none) shared(precomp, xC, active, changed_iter, order, B, a, X_norm2, cur_cost, rng) firstprivate(n, m, max_h, icm_iters) reduction(+:ls_failures)
    {
        IcmTls& tls = GetIcmTls(m, max_h);
        std::vector<float>& A = tls.A;
        std::vector<float>& b = tls.b;
        std::vector<float>& work = tls.work;
        std::vector<float>& backup = tls.backup;
        std::vector<float>& tmp_vals = tls.tmp_vals;

        for (int iter = 0; iter < icm_iters; ++iter) {
            #pragma omp single
            {
                std::shuffle(order->begin(), order->end(), *rng);
            }
            #pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) {
                (*changed_iter)[static_cast<std::size_t>(i)] = 0;
            }

            for (int jlayer : *order) {
                if (jlayer == 0) {
                    continue;
                }
                const int startf = precomp.offsets[jlayer];
                const int hj = precomp.h_vec[jlayer];
                if (hj <= 1) {
                    continue;
                }

                #pragma omp for schedule(static)
                for (int i = 0; i < n; ++i) {
                    if (!(*active)[static_cast<std::size_t>(i)]) {
                        continue;
                    }

                    const int old_code = static_cast<int>((*B)(jlayer, i));

                    // Pick best code by maximizing
                    //   val(flat) = ( xC(flat,i) - Σ_{l≠j} a_l * G(flat, flat_l) ) * invnorm_flat[flat]
                    // which matches Julia's:
                    //   val = (rC[flat,i] + a[j,i]*G[flat,old_flat]) * invnorm_flat[flat]
                    // in dynamic_icm_encoding_noabs_nonormal_clean_fast_for_ils1! (encode_noabs.jl:175-193).
                    const float* xC_i = xC.Col(i) + startf;
                    #pragma omp simd
                    for (int t = 0; t < hj; ++t) {
                        tmp_vals[t] = xC_i[t];
                    }
                    for (int l = 0; l < m; ++l) {
                        if (l == jlayer) {
                            continue;
                        }
                        const float alpha = (*a)(l, i);
                        const int flat_l = precomp.offsets[l] + static_cast<int>((*B)(l, i));
                        const float* G_col = precomp.G.Col(flat_l) + startf;
                        #pragma omp simd
                        for (int t = 0; t < hj; ++t) {
                            tmp_vals[t] -= alpha * G_col[t];
                        }
                    }

                    int best_code = old_code;
                    float best_score = -std::numeric_limits<float>::infinity();
                    const float* inv = precomp.invnorm_flat.data() + startf;
                    for (int t = 0; t < hj; ++t) {
                        const float val = tmp_vals[t] * inv[t];
                        const float score = UseAbs ? std::abs(val) : val;
                        if (score > best_score) {
                            best_score = score;
                            best_code = t;
                        }
                    }

                    if (best_code == old_code) {
                        continue;
                    }

                    float* a_i = a->Col(i);
                    std::memcpy(backup.data(), a_i, sizeof(float) * static_cast<std::size_t>(m));

                    (*B)(jlayer, i) = static_cast<FullCode>(best_code);

                    // Julia reference: optimized_least_squares_single! (encode_abs.jl:305-340),
                    // called by dynamic_icm_encoding_noabs_nonormal_clean_fast_for_ils1! (encode_noabs.jl:198-202).
                    if (!SolveLeastSquaresSample(precomp, xC, *B, i, &A, &b, &work, a_i)) {
                        (*B)(jlayer, i) = static_cast<FullCode>(old_code);
                        std::memcpy(a_i, backup.data(), sizeof(float) * static_cast<std::size_t>(m));
                        ++ls_failures;
                        continue;
                    }

                    const float new_cost = SampleCost(precomp, xC, *B, *a, i, (*X_norm2)[i]);
                    if (new_cost + kEps < (*cur_cost)[i]) {
                        (*cur_cost)[i] = new_cost;
                        (*changed_iter)[static_cast<std::size_t>(i)] = 1;
                    } else {
                        (*B)(jlayer, i) = static_cast<FullCode>(old_code);
                        std::memcpy(a_i, backup.data(), sizeof(float) * static_cast<std::size_t>(m));
                    }
                }
            }

            #pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) {
                const auto ii = static_cast<std::size_t>(i);
                if ((*active)[ii] && !(*changed_iter)[ii]) {
                    (*active)[ii] = 0;
                }
            }
        }
    }
    if (ls_failures > 0) {
        LogWarn("RunIcm: " + std::to_string(ls_failures) +
                " LS solve failures (parallel path); codebook may be ill-conditioned.");
    }
}

template <bool UseAbs>
void RunIcmLargeRoot(const ColMajorMatrix<float>& X,
                     const PrecompLargeRoot& pre,
                     const ColMajorMatrix<float>& xC_small,
                     const std::vector<std::uint32_t>& cluster_id,
                     int icm_iters,
                     std::vector<uint8_t>* active,
                     std::vector<uint8_t>* changed_iter,
                     std::vector<int>* order,
                     int max_h,
                     ColMajorMatrix<Code>* B_small,
                     ColMajorMatrix<float>* a,
                     std::vector<float>* X_norm2,
                     std::vector<float>* cur_cost,
                     std::mt19937* rng) {
    const int n = X.cols;
    const int m = pre.m;
    int ls_failures = 0;  // count rollback-triggering LS solve failures; logged at exit
    const bool has_g0s_t = !pre.G0S_T.data.empty();

    SolveLeastSquaresAllLargeRoot(pre, X, xC_small, cluster_id, *B_small, a);

    if (X_norm2->empty()) {
        ComputeXNorm2(X, X_norm2);
    }
    ComputeCostsLargeRoot(pre, X, xC_small, cluster_id, *B_small, *a, *X_norm2, cur_cost);

    if (!active || !changed_iter || !order) {
        return;
    }
    active->assign(static_cast<std::size_t>(n), 1);
    order->resize(static_cast<std::size_t>(m));
    std::iota(order->begin(), order->end(), 0);

    // Cache b0 = dot(C0[:,cid], x) once per sample (cluster_id is fixed in basic encoding).
    std::vector<float> b0_cache;
    b0_cache.resize(static_cast<std::size_t>(n));
    #pragma omp parallel for default(none) schedule(static) shared(cluster_id, pre, X, b0_cache) firstprivate(n)
    for (int i = 0; i < n; ++i) {
        const std::uint32_t cid = cluster_id[static_cast<std::size_t>(i)];
        const float* c0 = pre.C0.Col(static_cast<int>(cid));
        const float* x = X.Col(i);
        b0_cache[static_cast<std::size_t>(i)] = DotF32(c0, x, pre.d);
    }

    // Active index list (shrinks quickly after a few ICM iters).
    std::vector<int> active_idx;
    active_idx.reserve(static_cast<std::size_t>(n));

    const bool use_sample_parallel = (pre.h_vec.empty() ? false : (pre.h_vec[0] >= kIcmLargeRootSampleParallelMinH0));

    if (omp_in_parallel()) {
        IcmTls& tls = GetIcmTls(m, max_h);
        std::vector<float>& A_sys = tls.A;
        std::vector<float>& b_sys = tls.b;
        std::vector<float>& work = tls.work;
        std::vector<float>& backup = tls.backup;

        for (int iter = 0; iter < icm_iters; ++iter) {
            std::shuffle(order->begin(), order->end(), *rng);
            active_idx.clear();
            for (int i = 0; i < n; ++i) {
                if ((*active)[static_cast<std::size_t>(i)]) {
                    active_idx.push_back(i);
                }
            }

            if (use_sample_parallel) {
                // Process one sample at a time (all layers), to avoid per-layer barriers and reduce load imbalance.
                for (int i : active_idx) {
                    bool changed = false;
                    for (int jlayer : *order) {
                        if (jlayer == 0) continue;
                        const int startf = pre.small_offsets[jlayer];
                        const int hj = pre.h_vec[jlayer];
                        if (hj <= 1) continue;

                        const int old_code = static_cast<int>((*B_small)(jlayer - 1, i));
                        int best_code = old_code;
                        float best_score = -std::numeric_limits<float>::infinity();
                        const float* inv = pre.invnorm_small_flat.data() + startf;
                        const float* xC_i = xC_small.Col(i) + startf;

                        const float a0 = (*a)(0, i);
                        const std::uint32_t cid = cluster_id[static_cast<std::size_t>(i)];
                        const float* g0s_t_col = has_g0s_t ? pre.G0S_T.Col(static_cast<int>(cid)) : nullptr;
                        const float* g0s_col = has_g0s_t ? nullptr : pre.G0S.Col(startf);

                        // Small contributions pointers.
                        if (m == 5) {
                            const float a1 = (*a)(1, i);
                            const float a2 = (*a)(2, i);
                            const float a3 = (*a)(3, i);
                            const float a4 = (*a)(4, i);
                            const int flat1 = pre.small_offsets[1] + static_cast<int>((*B_small)(0, i));
                            const int flat2 = pre.small_offsets[2] + static_cast<int>((*B_small)(1, i));
                            const int flat3 = pre.small_offsets[3] + static_cast<int>((*B_small)(2, i));
                            const int flat4 = pre.small_offsets[4] + static_cast<int>((*B_small)(3, i));
                            const float* G1 = pre.G_small.Col(flat1) + startf;
                            const float* G2 = pre.G_small.Col(flat2) + startf;
                            const float* G3 = pre.G_small.Col(flat3) + startf;
                            const float* G4 = pre.G_small.Col(flat4) + startf;

                            for (int t = 0; t < hj; ++t) {
                                float v = xC_i[t];
                                const float g0s = has_g0s_t
                                                      ? g0s_t_col[static_cast<std::size_t>(startf + t)]
                                                      : g0s_col[static_cast<std::size_t>(t) * pre.G0S.rows +
                                                                static_cast<std::size_t>(cid)];
                                v -= a0 * g0s;
                                if (jlayer != 1) v -= a1 * G1[t];
                                if (jlayer != 2) v -= a2 * G2[t];
                                if (jlayer != 3) v -= a3 * G3[t];
                                if (jlayer != 4) v -= a4 * G4[t];
                                const float val = v * inv[t];
                                const float score = UseAbs ? std::abs(val) : val;
                                if (score > best_score) {
                                    best_score = score;
                                    best_code = t;
                                }
                            }
                        } else {
                            // Generic path: keep the old subtract loop (still reads root via G0S_T when available).
                            std::vector<float>& tmp_vals = tls.tmp_vals;
                            #pragma omp simd
                            for (int t = 0; t < hj; ++t) {
                                tmp_vals[t] = xC_i[t];
                            }
                            if (has_g0s_t) {
                                const int base = startf;
                                #pragma omp simd
                                for (int t = 0; t < hj; ++t) {
                                    tmp_vals[t] -= a0 * g0s_t_col[static_cast<std::size_t>(base + t)];
                                }
                            } else {
                                #pragma omp simd
                                for (int t = 0; t < hj; ++t) {
                                    tmp_vals[t] -= a0 * g0s_col[static_cast<std::size_t>(t) * pre.G0S.rows +
                                                                static_cast<std::size_t>(cid)];
                                }
                            }
                            for (int l = 1; l < m; ++l) {
                                if (l == jlayer) continue;
                                const float alpha = (*a)(l, i);
                                const int flat_l = pre.small_offsets[l] + static_cast<int>((*B_small)(l - 1, i));
                                const float* G_col = pre.G_small.Col(flat_l) + startf;
                                #pragma omp simd
                                for (int t = 0; t < hj; ++t) {
                                    tmp_vals[t] -= alpha * G_col[t];
                                }
                            }
                            for (int t = 0; t < hj; ++t) {
                                const float val = tmp_vals[t] * inv[t];
                                const float score = UseAbs ? std::abs(val) : val;
                                if (score > best_score) {
                                    best_score = score;
                                    best_code = t;
                                }
                            }
                        }

                        if (best_code == old_code) continue;

                        float* a_i = a->Col(i);
                        std::memcpy(backup.data(), a_i, sizeof(float) * static_cast<std::size_t>(m));
                        (*B_small)(jlayer - 1, i) = static_cast<Code>(best_code);
                        if (!SolveLeastSquaresSampleLargeRootWithB0(pre,
                                                                    b0_cache[static_cast<std::size_t>(i)],
                                                                    xC_small, cluster_id, *B_small,
                                                                    i, &A_sys, &b_sys, &work, a_i)) {
                            (*B_small)(jlayer - 1, i) = static_cast<Code>(old_code);
                            std::memcpy(a_i, backup.data(), sizeof(float) * static_cast<std::size_t>(m));
                            ++ls_failures;
                            continue;
                        }

                        const float new_cost = SampleCostLargeRoot(pre, X, xC_small, cluster_id, *B_small,
                                                                  *a, i, (*X_norm2)[static_cast<std::size_t>(i)]);
                        if (new_cost + kEps < (*cur_cost)[static_cast<std::size_t>(i)]) {
                            (*cur_cost)[static_cast<std::size_t>(i)] = new_cost;
                            changed = true;
                        } else {
                            (*B_small)(jlayer - 1, i) = static_cast<Code>(old_code);
                            std::memcpy(a_i, backup.data(), sizeof(float) * static_cast<std::size_t>(m));
                        }
                    }
                    if (!changed) {
                        (*active)[static_cast<std::size_t>(i)] = 0;
                    }
                }
            } else {
                // Small h0: keep layer-parallel update order (often faster due to better work distribution).
                changed_iter->assign(static_cast<std::size_t>(n), 0);
                for (int jlayer : *order) {
                    if (jlayer == 0) continue;
                    const int startf = pre.small_offsets[jlayer];
                    const int hj = pre.h_vec[jlayer];
                    if (hj <= 1) continue;

                    for (int i : active_idx) {
                        const int old_code = static_cast<int>((*B_small)(jlayer - 1, i));
                        std::vector<float>& tmp_vals = tls.tmp_vals;

                        const float* xC_i = xC_small.Col(i) + startf;
                        #pragma omp simd
                        for (int t = 0; t < hj; ++t) tmp_vals[t] = xC_i[t];

                        const float a0 = (*a)(0, i);
                        const std::uint32_t cid = cluster_id[static_cast<std::size_t>(i)];
                        if (has_g0s_t) {
                            const float* g0s_t_col = pre.G0S_T.Col(static_cast<int>(cid));
                            const int base = startf;
                            #pragma omp simd
                            for (int t = 0; t < hj; ++t) tmp_vals[t] -= a0 * g0s_t_col[static_cast<std::size_t>(base + t)];
                        } else {
                            const float* g0s_col = pre.G0S.Col(startf);
                            #pragma omp simd
                            for (int t = 0; t < hj; ++t) {
                                tmp_vals[t] -= a0 * g0s_col[static_cast<std::size_t>(t) * pre.G0S.rows +
                                                            static_cast<std::size_t>(cid)];
                            }
                        }
                        for (int l = 1; l < m; ++l) {
                            if (l == jlayer) continue;
                            const float alpha = (*a)(l, i);
                            const int flat_l = pre.small_offsets[l] + static_cast<int>((*B_small)(l - 1, i));
                            const float* G_col = pre.G_small.Col(flat_l) + startf;
                            #pragma omp simd
                            for (int t = 0; t < hj; ++t) tmp_vals[t] -= alpha * G_col[t];
                        }

                        int best_code = old_code;
                        float best_score = -std::numeric_limits<float>::infinity();
                        const float* inv = pre.invnorm_small_flat.data() + startf;
                        for (int t = 0; t < hj; ++t) {
                            const float val = tmp_vals[t] * inv[t];
                            const float score = UseAbs ? std::abs(val) : val;
                            if (score > best_score) {
                                best_score = score;
                                best_code = t;
                            }
                        }
                        if (best_code == old_code) continue;

                        float* a_i = a->Col(i);
                        std::memcpy(backup.data(), a_i, sizeof(float) * static_cast<std::size_t>(m));
                        (*B_small)(jlayer - 1, i) = static_cast<Code>(best_code);
                        if (!SolveLeastSquaresSampleLargeRootWithB0(pre,
                                                                    b0_cache[static_cast<std::size_t>(i)],
                                                                    xC_small, cluster_id, *B_small,
                                                                    i, &A_sys, &b_sys, &work, a_i)) {
                            (*B_small)(jlayer - 1, i) = static_cast<Code>(old_code);
                            std::memcpy(a_i, backup.data(), sizeof(float) * static_cast<std::size_t>(m));
                            ++ls_failures;
                            continue;
                        }
                        const float new_cost = SampleCostLargeRoot(pre, X, xC_small, cluster_id, *B_small,
                                                                  *a, i, (*X_norm2)[static_cast<std::size_t>(i)]);
                        if (new_cost + kEps < (*cur_cost)[static_cast<std::size_t>(i)]) {
                            (*cur_cost)[static_cast<std::size_t>(i)] = new_cost;
                            (*changed_iter)[static_cast<std::size_t>(i)] = 1;
                        } else {
                            (*B_small)(jlayer - 1, i) = static_cast<Code>(old_code);
                            std::memcpy(a_i, backup.data(), sizeof(float) * static_cast<std::size_t>(m));
                        }
                    }
                }
                for (int i : active_idx) {
                    const auto ii = static_cast<std::size_t>(i);
                    if ((*active)[ii] && !(*changed_iter)[ii]) {
                        (*active)[ii] = 0;
                    }
                }
            }
        }
        if (ls_failures > 0) {
            LogWarn("RunIcmLargeRoot: " + std::to_string(ls_failures) +
                    " LS solve failures (serial path); codebook may be ill-conditioned.");
        }
        return;
    }

    #pragma omp parallel default(none) shared(pre, X, xC_small, cluster_id, active, changed_iter, order, active_idx, B_small, a, X_norm2, cur_cost, b0_cache, rng) firstprivate(n, m, max_h, icm_iters, use_sample_parallel, has_g0s_t) reduction(+:ls_failures)
    {
        IcmTls& tls = GetIcmTls(m, max_h);
        std::vector<float>& A_sys = tls.A;
        std::vector<float>& b_sys = tls.b;
        std::vector<float>& work = tls.work;
        std::vector<float>& backup = tls.backup;

        for (int iter = 0; iter < icm_iters; ++iter) {
            #pragma omp single
            {
                std::shuffle(order->begin(), order->end(), *rng);
                if (use_sample_parallel) {
                    active_idx.clear();
                    for (int i = 0; i < n; ++i) {
                        if ((*active)[static_cast<std::size_t>(i)]) {
                            active_idx.push_back(i);
                        }
                    }
                } else {
                    active_idx.clear();
                    active_idx.reserve(static_cast<std::size_t>(n));
                    for (int i = 0; i < n; ++i) {
                        if ((*active)[static_cast<std::size_t>(i)]) {
                            active_idx.push_back(i);
                        }
                    }
                    // changed_iter is reused as a scratch array in the layer-parallel path.
                    changed_iter->assign(static_cast<std::size_t>(n), 0);
                }
            }

            if (use_sample_parallel) {
                #pragma omp for schedule(static)
                for (int idx = 0; idx < static_cast<int>(active_idx.size()); ++idx) {
                    const int i = active_idx[static_cast<std::size_t>(idx)];
                    bool changed = false;
                    for (int jlayer : *order) {
                        if (jlayer == 0) continue;
                        const int startf = pre.small_offsets[jlayer];
                        const int hj = pre.h_vec[jlayer];
                        if (hj <= 1) continue;

                        const int old_code = static_cast<int>((*B_small)(jlayer - 1, i));
                        int best_code = old_code;
                        float best_score = -std::numeric_limits<float>::infinity();
                        const float* inv = pre.invnorm_small_flat.data() + startf;
                        const float* xC_i = xC_small.Col(i) + startf;

                        const float a0 = (*a)(0, i);
                        const std::uint32_t cid = cluster_id[static_cast<std::size_t>(i)];
                        const float* g0s_t_col = has_g0s_t ? pre.G0S_T.Col(static_cast<int>(cid)) : nullptr;
                        const float* g0s_col = has_g0s_t ? nullptr : pre.G0S.Col(startf);

                        if (m == 5) {
                            const float a1 = (*a)(1, i);
                            const float a2 = (*a)(2, i);
                            const float a3 = (*a)(3, i);
                            const float a4 = (*a)(4, i);
                            const int flat1 = pre.small_offsets[1] + static_cast<int>((*B_small)(0, i));
                            const int flat2 = pre.small_offsets[2] + static_cast<int>((*B_small)(1, i));
                            const int flat3 = pre.small_offsets[3] + static_cast<int>((*B_small)(2, i));
                            const int flat4 = pre.small_offsets[4] + static_cast<int>((*B_small)(3, i));
                            const float* G1 = pre.G_small.Col(flat1) + startf;
                            const float* G2 = pre.G_small.Col(flat2) + startf;
                            const float* G3 = pre.G_small.Col(flat3) + startf;
                            const float* G4 = pre.G_small.Col(flat4) + startf;

                            for (int t = 0; t < hj; ++t) {
                                float v = xC_i[t];
                                const float g0s = has_g0s_t
                                                      ? g0s_t_col[static_cast<std::size_t>(startf + t)]
                                                      : g0s_col[static_cast<std::size_t>(t) * pre.G0S.rows +
                                                                static_cast<std::size_t>(cid)];
                                v -= a0 * g0s;
                                if (jlayer != 1) v -= a1 * G1[t];
                                if (jlayer != 2) v -= a2 * G2[t];
                                if (jlayer != 3) v -= a3 * G3[t];
                                if (jlayer != 4) v -= a4 * G4[t];
                                const float val = v * inv[t];
                                const float score = UseAbs ? std::abs(val) : val;
                                if (score > best_score) {
                                    best_score = score;
                                    best_code = t;
                                }
                            }
                        } else {
                            std::vector<float>& tmp_vals = tls.tmp_vals;
                            #pragma omp simd
                            for (int t = 0; t < hj; ++t) tmp_vals[t] = xC_i[t];
                            if (has_g0s_t) {
                                const int base = startf;
                                #pragma omp simd
                                for (int t = 0; t < hj; ++t) tmp_vals[t] -= a0 * g0s_t_col[static_cast<std::size_t>(base + t)];
                            } else {
                                #pragma omp simd
                                for (int t = 0; t < hj; ++t) {
                                    tmp_vals[t] -= a0 * g0s_col[static_cast<std::size_t>(t) * pre.G0S.rows +
                                                                static_cast<std::size_t>(cid)];
                                }
                            }
                            for (int l = 1; l < m; ++l) {
                                if (l == jlayer) continue;
                                const float alpha = (*a)(l, i);
                                const int flat_l = pre.small_offsets[l] + static_cast<int>((*B_small)(l - 1, i));
                                const float* G_col = pre.G_small.Col(flat_l) + startf;
                                #pragma omp simd
                                for (int t = 0; t < hj; ++t) tmp_vals[t] -= alpha * G_col[t];
                            }
                            for (int t = 0; t < hj; ++t) {
                                const float val = tmp_vals[t] * inv[t];
                                const float score = UseAbs ? std::abs(val) : val;
                                if (score > best_score) {
                                    best_score = score;
                                    best_code = t;
                                }
                            }
                        }

                        if (best_code == old_code) continue;

                        float* a_i = a->Col(i);
                        std::memcpy(backup.data(), a_i, sizeof(float) * static_cast<std::size_t>(m));
                        (*B_small)(jlayer - 1, i) = static_cast<Code>(best_code);
                        if (!SolveLeastSquaresSampleLargeRootWithB0(pre,
                                                                    b0_cache[static_cast<std::size_t>(i)],
                                                                    xC_small, cluster_id, *B_small,
                                                                    i, &A_sys, &b_sys, &work, a_i)) {
                            (*B_small)(jlayer - 1, i) = static_cast<Code>(old_code);
                            std::memcpy(a_i, backup.data(), sizeof(float) * static_cast<std::size_t>(m));
                            ++ls_failures;
                            continue;
                        }

                        const float new_cost = SampleCostLargeRoot(pre, X, xC_small, cluster_id, *B_small,
                                                                  *a, i, (*X_norm2)[static_cast<std::size_t>(i)]);
                        if (new_cost + kEps < (*cur_cost)[static_cast<std::size_t>(i)]) {
                            (*cur_cost)[static_cast<std::size_t>(i)] = new_cost;
                            changed = true;
                        } else {
                            (*B_small)(jlayer - 1, i) = static_cast<Code>(old_code);
                            std::memcpy(a_i, backup.data(), sizeof(float) * static_cast<std::size_t>(m));
                        }
                    }
                    if (!changed) {
                        (*active)[static_cast<std::size_t>(i)] = 0;
                    }
                }
            } else {
                for (int jlayer : *order) {
                    if (jlayer == 0) continue;
                    const int startf = pre.small_offsets[jlayer];
                    const int hj = pre.h_vec[jlayer];
                    if (hj <= 1) continue;

                    #pragma omp for schedule(static)
                    for (int idx = 0; idx < static_cast<int>(active_idx.size()); ++idx) {
                        const int i = active_idx[static_cast<std::size_t>(idx)];
                        const int old_code = static_cast<int>((*B_small)(jlayer - 1, i));
                        std::vector<float>& tmp_vals = tls.tmp_vals;

                        const float* xC_i = xC_small.Col(i) + startf;
                        #pragma omp simd
                        for (int t = 0; t < hj; ++t) tmp_vals[t] = xC_i[t];

                        const float a0 = (*a)(0, i);
                        const std::uint32_t cid = cluster_id[static_cast<std::size_t>(i)];
                        if (has_g0s_t) {
                            const float* g0s_t_col = pre.G0S_T.Col(static_cast<int>(cid));
                            const int base = startf;
                            #pragma omp simd
                            for (int t = 0; t < hj; ++t) tmp_vals[t] -= a0 * g0s_t_col[static_cast<std::size_t>(base + t)];
                        } else {
                            const float* g0s_col = pre.G0S.Col(startf);
                            #pragma omp simd
                            for (int t = 0; t < hj; ++t) {
                                tmp_vals[t] -= a0 * g0s_col[static_cast<std::size_t>(t) * pre.G0S.rows +
                                                            static_cast<std::size_t>(cid)];
                            }
                        }
                        for (int l = 1; l < m; ++l) {
                            if (l == jlayer) continue;
                            const float alpha = (*a)(l, i);
                            const int flat_l = pre.small_offsets[l] + static_cast<int>((*B_small)(l - 1, i));
                            const float* G_col = pre.G_small.Col(flat_l) + startf;
                            #pragma omp simd
                            for (int t = 0; t < hj; ++t) tmp_vals[t] -= alpha * G_col[t];
                        }
                        int best_code = old_code;
                        float best_score = -std::numeric_limits<float>::infinity();
                        const float* inv = pre.invnorm_small_flat.data() + startf;
                        for (int t = 0; t < hj; ++t) {
                            const float val = tmp_vals[t] * inv[t];
                            const float score = UseAbs ? std::abs(val) : val;
                            if (score > best_score) {
                                best_score = score;
                                best_code = t;
                            }
                        }
                        if (best_code == old_code) continue;

                        float* a_i = a->Col(i);
                        std::memcpy(backup.data(), a_i, sizeof(float) * static_cast<std::size_t>(m));
                        (*B_small)(jlayer - 1, i) = static_cast<Code>(best_code);
                        if (!SolveLeastSquaresSampleLargeRootWithB0(pre,
                                                                    b0_cache[static_cast<std::size_t>(i)],
                                                                    xC_small, cluster_id, *B_small,
                                                                    i, &A_sys, &b_sys, &work, a_i)) {
                            (*B_small)(jlayer - 1, i) = static_cast<Code>(old_code);
                            std::memcpy(a_i, backup.data(), sizeof(float) * static_cast<std::size_t>(m));
                            ++ls_failures;
                            continue;
                        }
                        const float new_cost = SampleCostLargeRoot(pre, X, xC_small, cluster_id, *B_small,
                                                                  *a, i, (*X_norm2)[static_cast<std::size_t>(i)]);
                        if (new_cost + kEps < (*cur_cost)[static_cast<std::size_t>(i)]) {
                            (*cur_cost)[static_cast<std::size_t>(i)] = new_cost;
                            (*changed_iter)[static_cast<std::size_t>(i)] = 1;
                        } else {
                            (*B_small)(jlayer - 1, i) = static_cast<Code>(old_code);
                            std::memcpy(a_i, backup.data(), sizeof(float) * static_cast<std::size_t>(m));
                        }
                    }
                }

                #pragma omp for schedule(static)
                for (int idx = 0; idx < static_cast<int>(active_idx.size()); ++idx) {
                    const int i = active_idx[static_cast<std::size_t>(idx)];
                    const auto ii = static_cast<std::size_t>(i);
                    if ((*active)[ii] && !(*changed_iter)[ii]) {
                        (*active)[ii] = 0;
                    }
                }
            }
        }
    }
    if (ls_failures > 0) {
        LogWarn("RunIcmLargeRoot: " + std::to_string(ls_failures) +
                " LS solve failures (parallel path); codebook may be ill-conditioned.");
    }
}

}  // namespace

template <bool UseAbs>
void DynamicIcmWithIlsImpl(const ColMajorMatrix<float>& X,
                           const Precomp& precomp,
                           const ColMajorMatrix<float>& xC,
                           int icm_iters,
                           int ils_iters,
                           int perturb_k,
                           std::uint32_t seed,
                           std::uint64_t sample_id_offset,
                           ColMajorMatrix<FullCode>* B,
                           ColMajorMatrix<float>* a,
                           std::vector<float>* X_norm2,
                           std::vector<float>* cost,
                           bool print_progress) {
    if (!B || !a) {
        return;
    }

    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    std::vector<uint8_t> active;
    std::vector<uint8_t> changed_iter;
    std::vector<int> order;
    const int max_h = *std::max_element(precomp.h_vec.begin(), precomp.h_vec.end());
    RunIcm<UseAbs>(X, precomp, xC, icm_iters, &active, &changed_iter, &order, max_h, B, a, X_norm2, cost, &rng);

    if (ils_iters <= 0 || perturb_k <= 0 || precomp.m <= 1) {
        return;
    }

    const int n = B->cols;
    const int ksel = std::min(perturb_k, precomp.m);
    const int layer_min = 1;
    const int layer_max = precomp.m - 1;
    const int layer_span = layer_max - layer_min + 1;

    // Candidate buffers:
    // - In nested OpenMP (outer cluster-parallel region), we don't spawn inner threads, so we can reuse TLS buffers
    //   to reduce allocator churn.
    // - Otherwise (top-level call), we may parallelize inside this function, so buffers must be shared across threads
    //   => keep them as local allocations.
    ColMajorMatrix<FullCode> B_cand_local;
    ColMajorMatrix<float> a_cand_local;
    std::vector<float> cand_cost_local;
    static thread_local ColMajorMatrix<FullCode> B_cand_tls;
    static thread_local ColMajorMatrix<float> a_cand_tls;
    static thread_local std::vector<float> cand_cost_tls;
    ColMajorMatrix<FullCode>* B_cand = nullptr;
    ColMajorMatrix<float>* a_cand = nullptr;
    std::vector<float>* cand_cost = nullptr;
    if (omp_in_parallel()) {
        B_cand_tls.rows = B->rows;
        B_cand_tls.cols = B->cols;
        B_cand_tls.data.resize(B->data.size());
        a_cand_tls.rows = a->rows;
        a_cand_tls.cols = a->cols;
        a_cand_tls.data.resize(a->data.size());
        cand_cost_tls.resize(static_cast<std::size_t>(std::max(0, n)));
        B_cand = &B_cand_tls;
        a_cand = &a_cand_tls;
        cand_cost = &cand_cost_tls;
    } else {
        B_cand_local.rows = B->rows;
        B_cand_local.cols = B->cols;
        B_cand_local.data.resize(B->data.size());
        a_cand_local.rows = a->rows;
        a_cand_local.cols = a->cols;
        a_cand_local.data.resize(a->data.size());
        cand_cost_local.resize(static_cast<std::size_t>(std::max(0, n)));
        B_cand = &B_cand_local;
        a_cand = &a_cand_local;
        cand_cost = &cand_cost_local;
    }

    // Julia reference: ILS outer loop (encode_abs.jl:425-512, encode_noabs.jl:52-112).
    for (int outer = 0; outer < ils_iters; ++outer) {
        std::memcpy(B_cand->data.data(), B->data.data(),
                    sizeof(FullCode) * B->data.size());

        // Julia reference: Lmat/New are generated outside threads (encode_abs.jl:423-460),
        // so the perturbation randomness should not depend on thread count.
        // Here we generate one deterministic RNG stream per sample and per outer iteration.
        #pragma omp parallel for default(none) if(!omp_in_parallel()) schedule(static) shared(precomp, B_cand) firstprivate(n, sample_id_offset, seed, outer, ksel, layer_min, layer_span)
        for (int i = 0; i < n; ++i) {
            const std::uint64_t i_global =
                sample_id_offset + static_cast<std::uint64_t>(i);
            SplitMix64 local_rng(static_cast<std::uint64_t>(seed) ^
                                 (static_cast<std::uint64_t>(outer + 1) * 0x9e3779b97f4a7c15ULL) ^
                                 ((i_global + 1) * 0xbf58476d1ce4e5b9ULL));
            for (int r = 0; r < ksel; ++r) {
                const int layer = layer_min + static_cast<int>(
                    UniformBelow(&local_rng, static_cast<std::uint32_t>(layer_span)));
                const int h = precomp.h_vec[layer];
                if (h <= 1) {
                    continue;
                }
                const int old_code = static_cast<int>((*B_cand)(layer, i));
                const int raw = static_cast<int>(UniformBelow(&local_rng, static_cast<std::uint32_t>(h - 1)));
                const int new_code = (raw >= old_code) ? (raw + 1) : raw;
                (*B_cand)(layer, i) = static_cast<FullCode>(new_code);
            }
        }

        RunIcm<UseAbs>(X, precomp, xC, icm_iters, &active, &changed_iter, &order, max_h,
                       B_cand, a_cand, X_norm2, cand_cost, &rng);

        long long accepted = 0;
        double sum_cost = 0.0;
        #pragma omp parallel for default(none) if(!omp_in_parallel()) schedule(static) reduction(+:accepted,sum_cost) shared(precomp, cand_cost, cost, B, B_cand, a, a_cand) firstprivate(n)
        for (int i = 0; i < n; ++i) {
            if ((*cand_cost)[i] + kEps < (*cost)[i]) {
                std::memcpy(B->Col(i), B_cand->Col(i),
                            sizeof(FullCode) * static_cast<std::size_t>(precomp.m));
                std::memcpy(a->Col(i), a_cand->Col(i),
                            sizeof(float) * static_cast<std::size_t>(precomp.m));
                (*cost)[i] = (*cand_cost)[i];
                accepted += 1;
            }
            sum_cost += static_cast<double>((*cost)[i]);
        }

        const double avg_cost = (n > 0) ? (sum_cost / static_cast<double>(n)) : 0.0;
        if (print_progress) {
            std::fprintf(stderr,
                         "\rILS+ICM round %d/%d accepted %lld / %d samples, avg mse %.6f",
                         outer + 1, ils_iters, accepted, n, avg_cost);
            if (outer + 1 == ils_iters) {
                std::fprintf(stderr, "\n");
            }
            std::fflush(stderr);
        }
    }
}

template <bool UseAbs>
void DynamicIcmWithIlsLargeRootImpl(const ColMajorMatrix<float>& X,
                                    const PrecompLargeRoot& pre,
                                    const ColMajorMatrix<float>& xC_small,
                                    int icm_iters,
                                    int ils_iters,
                                    int perturb_k,
                                    std::uint32_t seed,
                                    std::uint64_t sample_id_offset,
                                    const std::vector<std::uint32_t>& cluster_id,
                                    ColMajorMatrix<Code>* B_small,
                                    ColMajorMatrix<float>* a,
                                    std::vector<float>* X_norm2,
                                    std::vector<float>* cost,
                                    bool print_progress) {
    if (!B_small || !a) {
        return;
    }

    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    std::vector<uint8_t> active;
    std::vector<uint8_t> changed_iter;
    std::vector<int> order;
    const int max_h = *std::max_element(pre.h_vec.begin(), pre.h_vec.end());
    RunIcmLargeRoot<UseAbs>(X, pre, xC_small, cluster_id, icm_iters,
                            &active, &changed_iter, &order, max_h, B_small, a, X_norm2, cost, &rng);

    if (ils_iters <= 0 || perturb_k <= 0 || pre.m <= 1) {
        return;
    }

    const int n = B_small->cols;
    const int ksel = std::min(perturb_k, pre.m);
    const int layer_min = 1;
    const int layer_max = pre.m - 1;
    const int layer_span = layer_max - layer_min + 1;

    ColMajorMatrix<Code> B_cand_local;
    ColMajorMatrix<float> a_cand_local;
    std::vector<float> cand_cost_local;
    static thread_local ColMajorMatrix<Code> B_cand_tls;
    static thread_local ColMajorMatrix<float> a_cand_tls;
    static thread_local std::vector<float> cand_cost_tls;
    ColMajorMatrix<Code>* B_cand = nullptr;
    ColMajorMatrix<float>* a_cand = nullptr;
    std::vector<float>* cand_cost = nullptr;
    if (omp_in_parallel()) {
        B_cand_tls.rows = B_small->rows;
        B_cand_tls.cols = B_small->cols;
        B_cand_tls.data.resize(B_small->data.size());
        a_cand_tls.rows = a->rows;
        a_cand_tls.cols = a->cols;
        a_cand_tls.data.resize(a->data.size());
        cand_cost_tls.resize(static_cast<std::size_t>(std::max(0, n)));
        B_cand = &B_cand_tls;
        a_cand = &a_cand_tls;
        cand_cost = &cand_cost_tls;
    } else {
        B_cand_local.rows = B_small->rows;
        B_cand_local.cols = B_small->cols;
        B_cand_local.data.resize(B_small->data.size());
        a_cand_local.rows = a->rows;
        a_cand_local.cols = a->cols;
        a_cand_local.data.resize(a->data.size());
        cand_cost_local.resize(static_cast<std::size_t>(std::max(0, n)));
        B_cand = &B_cand_local;
        a_cand = &a_cand_local;
        cand_cost = &cand_cost_local;
    }

    for (int outer = 0; outer < ils_iters; ++outer) {
        std::memcpy(B_cand->data.data(), B_small->data.data(),
                    sizeof(Code) * B_small->data.size());

        #pragma omp parallel for default(none) if(!omp_in_parallel()) schedule(static) shared(pre, B_cand) firstprivate(n, sample_id_offset, seed, outer, ksel, layer_min, layer_span)
        for (int i = 0; i < n; ++i) {
            const std::uint64_t i_global =
                sample_id_offset + static_cast<std::uint64_t>(i);
            SplitMix64 local_rng(static_cast<std::uint64_t>(seed) ^
                                 (static_cast<std::uint64_t>(outer + 1) * 0x9e3779b97f4a7c15ULL) ^
                                 ((i_global + 1) * 0xbf58476d1ce4e5b9ULL));
            for (int r = 0; r < ksel; ++r) {
                const int layer = layer_min + static_cast<int>(
                    UniformBelow(&local_rng, static_cast<std::uint32_t>(layer_span)));
                const int h = pre.h_vec[layer];
                if (h <= 1) {
                    continue;
                }
                const int old_code = static_cast<int>((*B_cand)(layer - 1, i));
                const int raw = static_cast<int>(UniformBelow(&local_rng, static_cast<std::uint32_t>(h - 1)));
                const int new_code = (raw >= old_code) ? (raw + 1) : raw;
                (*B_cand)(layer - 1, i) = static_cast<Code>(new_code);
            }
        }

        RunIcmLargeRoot<UseAbs>(X, pre, xC_small, cluster_id, icm_iters,
                                &active, &changed_iter, &order, max_h,
                                B_cand, a_cand, X_norm2, cand_cost, &rng);

        long long accepted = 0;
        double sum_cost = 0.0;
        #pragma omp parallel for default(none) if(!omp_in_parallel()) schedule(static) reduction(+:accepted,sum_cost) shared(pre, cand_cost, cost, B_small, B_cand, a, a_cand) firstprivate(n)
        for (int i = 0; i < n; ++i) {
            if ((*cand_cost)[static_cast<std::size_t>(i)] + kEps < (*cost)[static_cast<std::size_t>(i)]) {
                std::memcpy(B_small->Col(i), B_cand->Col(i),
                            sizeof(Code) * static_cast<std::size_t>(std::max(0, pre.m - 1)));
                std::memcpy(a->Col(i), a_cand->Col(i),
                            sizeof(float) * static_cast<std::size_t>(pre.m));
                (*cost)[static_cast<std::size_t>(i)] = (*cand_cost)[static_cast<std::size_t>(i)];
                accepted += 1;
            }
            sum_cost += static_cast<double>((*cost)[static_cast<std::size_t>(i)]);
        }

        const double avg_cost = (n > 0) ? (sum_cost / static_cast<double>(n)) : 0.0;
        if (print_progress) {
            std::fprintf(stderr,
                         "\rILS+ICM round %d/%d accepted %lld / %d samples, avg mse %.6f",
                         outer + 1, ils_iters, accepted, n, avg_cost);
            if (outer + 1 == ils_iters) {
                std::fprintf(stderr, "\n");
            }
            std::fflush(stderr);
        }
    }
}

void DynamicIcmWithIlsNoAbsNoNormal(const ColMajorMatrix<float>& X,
                                   const Precomp& precomp,
                                   const ColMajorMatrix<float>& xC,
                                   int icm_iters,
                                   int ils_iters,
                                   int perturb_k,
                                   std::uint32_t seed,
                                   ColMajorMatrix<FullCode>* B,
                                   ColMajorMatrix<float>* a,
                                   std::vector<float>* X_norm2,
                                   std::vector<float>* cost,
                                   bool print_progress,
                                   std::uint64_t sample_id_offset) {
    // `cost` is an optional output. When metrics are disabled, callers may pass nullptr.
    // Internally we still maintain costs for acceptance decisions; we just skip returning them.
    static thread_local std::vector<float> cost_scratch;
    std::vector<float>* cost_out = cost ? cost : &cost_scratch;
    DynamicIcmWithIlsImpl<false>(X, precomp, xC, icm_iters, ils_iters, perturb_k,
                                 seed, sample_id_offset, B, a, X_norm2, cost_out, print_progress);
}

void DynamicIcmWithIlsAbsNoNormal(const ColMajorMatrix<float>& X,
                                 const Precomp& precomp,
                                 const ColMajorMatrix<float>& xC,
                                 int icm_iters,
                                 int ils_iters,
                                 int perturb_k,
                                 std::uint32_t seed,
                                 ColMajorMatrix<FullCode>* B,
                                 ColMajorMatrix<float>* a,
                                 std::vector<float>* X_norm2,
                                 std::vector<float>* cost,
                                 bool print_progress,
                                 std::uint64_t sample_id_offset) {
    static thread_local std::vector<float> cost_scratch;
    std::vector<float>* cost_out = cost ? cost : &cost_scratch;
    DynamicIcmWithIlsImpl<true>(X, precomp, xC, icm_iters, ils_iters, perturb_k,
                                seed, sample_id_offset, B, a, X_norm2, cost_out, print_progress);
}

void DynamicIcmWithIlsNoAbsNoNormalLargeRoot(const ColMajorMatrix<float>& X,
                                            const PrecompLargeRoot& pre,
                                            const ColMajorMatrix<float>& xC_small,
                                            int icm_iters,
                                            int ils_iters,
                                            int perturb_k,
                                            std::uint32_t seed,
                                            const std::vector<std::uint32_t>& cluster_id,
                                            ColMajorMatrix<Code>* B_small,
                                            ColMajorMatrix<float>* a,
                                            std::vector<float>* X_norm2,
                                            std::vector<float>* cost,
                                            bool print_progress,
                                            std::uint64_t sample_id_offset) {
    static thread_local std::vector<float> cost_scratch;
    std::vector<float>* cost_out = cost ? cost : &cost_scratch;
    DynamicIcmWithIlsLargeRootImpl<false>(X, pre, xC_small, icm_iters, ils_iters, perturb_k,
                                         seed, sample_id_offset, cluster_id, B_small, a, X_norm2, cost_out,
                                         print_progress);
}

void DynamicIcmWithIlsAbsNoNormalLargeRoot(const ColMajorMatrix<float>& X,
                                          const PrecompLargeRoot& pre,
                                          const ColMajorMatrix<float>& xC_small,
                                          int icm_iters,
                                          int ils_iters,
                                          int perturb_k,
                                          std::uint32_t seed,
                                          const std::vector<std::uint32_t>& cluster_id,
                                          ColMajorMatrix<Code>* B_small,
                                          ColMajorMatrix<float>* a,
                                          std::vector<float>* X_norm2,
                                          std::vector<float>* cost,
                                          bool print_progress,
                                          std::uint64_t sample_id_offset) {
    static thread_local std::vector<float> cost_scratch;
    std::vector<float>* cost_out = cost ? cost : &cost_scratch;
    DynamicIcmWithIlsLargeRootImpl<true>(X, pre, xC_small, icm_iters, ils_iters, perturb_k,
                                        seed, sample_id_offset, cluster_id, B_small, a, X_norm2, cost_out,
                                        print_progress);
}

}  // namespace stlq
