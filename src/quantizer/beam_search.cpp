#include "stlq/quantizer/beam_search.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include <omp.h>

#include "stlq/core/blas.h"
#include "stlq/core/threading.h"
#include "stlq/common/logger.h"
#include "stlq/common/timer.h"
#include "stlq/quantizer/linear_algebra.h"
#include "stlq/quantizer/precomp_large_root.h"

#if defined(_MSC_VER)
#define STLQ_RESTRICT __restrict
#else
#define STLQ_RESTRICT __restrict__
#endif

namespace stlq {

namespace {

struct BeamWorkspace {
    int m = 0;
    int H = 0;
    std::vector<FullCode> codes_curr;
    std::vector<FullCode> codes_next;
    std::vector<float> errs_curr;
    std::vector<float> errs_next;
    std::vector<float> a_curr;
    std::vector<float> a_next;
    std::vector<float> chol_curr;
    std::vector<float> chol_next;
    std::vector<float> tmp_a;
    std::vector<float> tmp_v;
    std::vector<float> tmp_y;
    std::vector<float> A_small;
    std::vector<int> sel_src_h;
    std::vector<int> sel_k;
    std::vector<float> sel_denom;
    std::vector<float> sel_alast;
    std::vector<float> sel_v;
    std::vector<float> sel_y;

    BeamWorkspace(int m_in, int H_in)
        : m(m_in),
          H(H_in),
          codes_curr(static_cast<std::size_t>(m_in) * H_in),
          codes_next(static_cast<std::size_t>(m_in) * H_in),
          errs_curr(H_in),
          errs_next(H_in),
          a_curr(static_cast<std::size_t>(m_in) * H_in),
          a_next(static_cast<std::size_t>(m_in) * H_in),
          chol_curr(static_cast<std::size_t>(m_in) * m_in * H_in),
          chol_next(static_cast<std::size_t>(m_in) * m_in * H_in),
          tmp_a(m_in),
          tmp_v(m_in),
          tmp_y(m_in),
          A_small(static_cast<std::size_t>(m_in) * m_in),
          sel_src_h(H_in, -1),
          sel_k(H_in, -1),
          sel_denom(H_in, 0.0f),
          sel_alast(H_in, 0.0f),
          sel_v(static_cast<std::size_t>(m_in) * H_in, 0.0f),
          sel_y(static_cast<std::size_t>(m_in) * H_in, 0.0f) {}

    FullCode& CodeAt(std::vector<FullCode>& buf, int layer, int beam) const {
        return buf[static_cast<std::size_t>(beam) * static_cast<std::size_t>(m) +
                   static_cast<std::size_t>(layer)];
    }
    float& ACoeff(std::vector<float>& buf, int layer, int beam) const {
        return buf[static_cast<std::size_t>(beam) * m + layer];
    }
    float* Chol(std::vector<float>& buf, int beam) const {
        return buf.data() + static_cast<std::size_t>(beam) * m * m;
    }
};

inline int WorstIdx(const float* errs, int H_beam) {
    if (H_beam <= 1) return 0;
    if (H_beam == 2) {
        // Match the existing scan semantics: worst_idx stays 0 on ties.
        return (errs[1] > errs[0]) ? 1 : 0;
    }
    int worst_idx = 0;
    float worst_err = errs[0];
    for (int h = 1; h < H_beam; ++h) {
        if (errs[h] > worst_err) {
            worst_err = errs[h];
            worst_idx = h;
        }
    }
    return worst_idx;
}

inline int RecomputeWorst(const float* errs, int H_beam, float* worst_err) {
    int worst_idx = 0;
    float we = errs[0];
    for (int h = 1; h < H_beam; ++h) {
        const float e = errs[h];
        if (e > we) {
            we = e;
            worst_idx = h;
        }
    }
    *worst_err = we;
    return worst_idx;
}

inline int CountFinite(const float* errs, int H_beam) {
    if (H_beam <= 0) return 0;
    const float inf = std::numeric_limits<float>::infinity();
    if (H_beam == 1) return errs[0] < inf ? 1 : 0;
    if (H_beam == 2) return (errs[0] < inf) + (errs[1] < inf);
    int n = 0;
    for (int h = 0; h < H_beam; ++h) {
        n += (errs[h] < inf);
    }
    return n;
}

template <int N>
inline void SolveCholUpperRawFixed(const float* STLQ_RESTRICT R,
                                   const float* STLQ_RESTRICT b,
                                   float* STLQ_RESTRICT x,
                                   float* STLQ_RESTRICT y,
                                   int stride) {
    static_assert(N >= 1, "N must be >= 1");
    // Forward solve U^T y = b
#if defined(__CUDACC__)
    #pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
        float s = b[i];
#if defined(__CUDACC__)
        #pragma unroll
#endif
        for (int k = 0; k < i; ++k) {
            s -= R[static_cast<std::size_t>(k) * static_cast<std::size_t>(stride) + i] * y[k];
        }
        y[i] = s / R[static_cast<std::size_t>(i) * static_cast<std::size_t>(stride) + i];
    }
    // Backward solve U x = y
#if defined(__CUDACC__)
    #pragma unroll
#endif
    for (int i = N - 1; i >= 0; --i) {
        float s = y[i];
#if defined(__CUDACC__)
        #pragma unroll
#endif
        for (int k = i + 1; k < N; ++k) {
            s -= R[static_cast<std::size_t>(i) * static_cast<std::size_t>(stride) + k] * x[k];
        }
        x[i] = s / R[static_cast<std::size_t>(i) * static_cast<std::size_t>(stride) + i];
    }
}

inline void SolveCholUpperRawDispatch(const float* STLQ_RESTRICT R,
                                      int n,
                                      const float* STLQ_RESTRICT b,
                                      float* STLQ_RESTRICT x,
                                      float* STLQ_RESTRICT y,
                                      int stride) {
    // Beam uses small n (<= m <= 16). Dispatch to enable compile-time unrolling.
    switch (n) {
        case 1: SolveCholUpperRawFixed<1>(R, b, x, y, stride); return;
        case 2: SolveCholUpperRawFixed<2>(R, b, x, y, stride); return;
        case 3: SolveCholUpperRawFixed<3>(R, b, x, y, stride); return;
        case 4: SolveCholUpperRawFixed<4>(R, b, x, y, stride); return;
        case 5: SolveCholUpperRawFixed<5>(R, b, x, y, stride); return;
        case 6: SolveCholUpperRawFixed<6>(R, b, x, y, stride); return;
        case 7: SolveCholUpperRawFixed<7>(R, b, x, y, stride); return;
        case 8: SolveCholUpperRawFixed<8>(R, b, x, y, stride); return;
        case 9: SolveCholUpperRawFixed<9>(R, b, x, y, stride); return;
        case 10: SolveCholUpperRawFixed<10>(R, b, x, y, stride); return;
        case 11: SolveCholUpperRawFixed<11>(R, b, x, y, stride); return;
        case 12: SolveCholUpperRawFixed<12>(R, b, x, y, stride); return;
        case 13: SolveCholUpperRawFixed<13>(R, b, x, y, stride); return;
        case 14: SolveCholUpperRawFixed<14>(R, b, x, y, stride); return;
        case 15: SolveCholUpperRawFixed<15>(R, b, x, y, stride); return;
        case 16: SolveCholUpperRawFixed<16>(R, b, x, y, stride); return;
        default: break;
    }
    // Fallback (shouldn't happen).
    SolveCholUpperRaw(R, n, b, x, y, stride);
}

}  // namespace

void BeamSearchPrefixLS(const ColMajorMatrix<float>& X,
                        const Precomp& precomp,
                        const ColMajorMatrix<float>& xC,
                        int H_beam,
                        ColMajorMatrix<FullCode>* B) {
    const int n = X.cols;
    const int d = X.rows;
    const int m = precomp.m;

    B->rows = m;
    B->cols = n;
    B->data.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));

    #pragma omp parallel default(none) shared(X, precomp, xC, B) firstprivate(H_beam, n, d, m)
    {
        BeamWorkspace ws(m, H_beam);

        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            const float* Xi = X.Col(i);
            const float* xCi = xC.Col(i);

            float norm_x2 = 0.0f;
            #pragma omp simd reduction(+:norm_x2)
            for (int t = 0; t < d; ++t) {
                float v = Xi[t];
                norm_x2 += v * v;
            }

            std::fill(ws.errs_curr.begin(), ws.errs_curr.end(), std::numeric_limits<float>::infinity());
            std::fill(ws.errs_next.begin(), ws.errs_next.end(), std::numeric_limits<float>::infinity());

            int curr_H = 0;
            const int K1 = precomp.h_vec[0];
            const int off1 = precomp.offsets[0];

            // Init Cholesky slots once (avoid per-candidate clearing inside the K loop).
            for (int h = 0; h < H_beam; ++h) {
                float* R = ws.Chol(ws.chol_next, h);
                std::fill(R, R + static_cast<std::size_t>(m) * static_cast<std::size_t>(m), 0.0f);
            }

            float worst_err = ws.errs_next[0];
            int worst_idx_cached = RecomputeWorst(ws.errs_next.data(), H_beam, &worst_err);
            for (int k = 0; k < K1; ++k) {
                int flat = off1 + k;
                float alpha = GAt(precomp.G, flat, flat);
                // Julia reference: only guard denom==0 (Beam_search.jl:327-331).
                // Keep the same semantics here to avoid skipping valid-but-small denominators.
                if (alpha == 0.0f) {
                    continue;
                }
                float gamma = xCi[flat];
                float a1 = gamma / alpha;
                float e1 = norm_x2 - gamma * a1;

                if (e1 >= worst_err) continue;
                ws.errs_next[worst_idx_cached] = e1;
                ws.CodeAt(ws.codes_next, 0, worst_idx_cached) = static_cast<FullCode>(k);
                ws.ACoeff(ws.a_next, 0, worst_idx_cached) = a1;

                float* R = ws.Chol(ws.chol_next, worst_idx_cached);
                R[0] = std::sqrt(alpha);

                worst_idx_cached = RecomputeWorst(ws.errs_next.data(), H_beam, &worst_err);
            }

            curr_H = CountFinite(ws.errs_next.data(), H_beam);

            // Julia reference: after L=1 init, swap curr/next buffers before expanding L=2..m
            // (Beam_search.jl:275-283).
            std::swap(ws.codes_curr, ws.codes_next);
            std::swap(ws.errs_curr, ws.errs_next);
            std::swap(ws.a_curr, ws.a_next);
            std::swap(ws.chol_curr, ws.chol_next);

            for (int layer = 1; layer < m; ++layer) {
                std::fill(ws.errs_next.begin(), ws.errs_next.end(), std::numeric_limits<float>::infinity());
                float worst_err = ws.errs_next[0];
                int worst_idx_cached = RecomputeWorst(ws.errs_next.data(), H_beam, &worst_err);
                std::fill(ws.sel_src_h.begin(), ws.sel_src_h.end(), -1);

                const int off = precomp.offsets[layer];
                const int K = precomp.h_vec[layer];
                int new_H = 0;

                for (int h = 0; h < curr_H; ++h) {
                    float* R_pref = ws.Chol(ws.chol_curr, h);
                    const float* a_old = ws.a_curr.data() + static_cast<std::size_t>(h) * m;
                    const float E_old = ws.errs_curr[h];
                    for (int k = 0; k < K; ++k) {
                        int flat = off + k;
                        float alpha = GAt(precomp.G, flat, flat);
                        float gamma = xCi[flat];

                        for (int t = 0; t < layer; ++t) {
                            int flat_t = precomp.offsets[t] +
                                         static_cast<int>(ws.CodeAt(ws.codes_curr, t, h));
                            ws.tmp_a[t] = GAt(precomp.G, flat_t, flat);
                        }

                        SolveCholUpperRawDispatch(R_pref, layer, ws.tmp_a.data(),
                                                  ws.tmp_v.data(), ws.tmp_y.data(), m);
                        float delta = 0.0f;
                        float eta = 0.0f;
                        for (int t = 0; t < layer; ++t) {
                            float at = ws.tmp_a[t];
                            delta += at * ws.tmp_v[t];
                            eta += at * a_old[t];
                        }

                        float denom = alpha - delta;
                        // Julia reference: denom==0 check only (Beam_search.jl:327-331).
                        if (denom == 0.0f) {
                            continue;
                        }

                        float diff = gamma - eta;
                        float a_last = diff / denom;
                        float e = E_old - diff * a_last;
                        if (e < 0.0f) {
                            e = 0.0f;
                        }

                        if (e >= worst_err) continue;
                        ws.errs_next[worst_idx_cached] = e;
                        ws.sel_src_h[static_cast<std::size_t>(worst_idx_cached)] = h;
                        ws.sel_k[static_cast<std::size_t>(worst_idx_cached)] = k;
                        ws.sel_denom[static_cast<std::size_t>(worst_idx_cached)] = denom;
                        ws.sel_alast[static_cast<std::size_t>(worst_idx_cached)] = a_last;
                        for (int t = 0; t < layer; ++t) {
                            ws.sel_v[static_cast<std::size_t>(t) * static_cast<std::size_t>(H_beam) +
                                     static_cast<std::size_t>(worst_idx_cached)] = ws.tmp_v[t];
                            ws.sel_y[static_cast<std::size_t>(t) * static_cast<std::size_t>(H_beam) +
                                     static_cast<std::size_t>(worst_idx_cached)] = ws.tmp_y[t];
                        }

                        worst_idx_cached = RecomputeWorst(ws.errs_next.data(), H_beam, &worst_err);
                    }
                }

                const float inf = std::numeric_limits<float>::infinity();
                for (int out = 0; out < H_beam; ++out) {
                    if (ws.errs_next[out] >= inf) continue;
                    const int src_h = ws.sel_src_h[static_cast<std::size_t>(out)];
                    // Finite errs_next slots are produced by the candidate loop, which writes
                    // sel_src_h from h in [0, curr_H). CUDA path has no matching guard.
                    // if (src_h < 0 || src_h >= curr_H) {
                    //     LogWarn("BeamSearchPrefixLS: beam slot " + std::to_string(out) +
                    //             " has valid err but src_h=" + std::to_string(src_h) +
                    //             " out of range [0," + std::to_string(curr_H) +
                    //             "); beam state corrupt. Skipping slot.");
                    //     continue;
                    // }

                    const int k = ws.sel_k[static_cast<std::size_t>(out)];
                    const float denom = ws.sel_denom[static_cast<std::size_t>(out)];
                    const float a_last = ws.sel_alast[static_cast<std::size_t>(out)];

                    for (int t = 0; t < layer; ++t) {
                        ws.CodeAt(ws.codes_next, t, out) =
                            ws.CodeAt(ws.codes_curr, t, src_h);
                    }
                    ws.CodeAt(ws.codes_next, layer, out) = static_cast<FullCode>(k);

                    const float* a_old =
                        ws.a_curr.data() + static_cast<std::size_t>(src_h) * static_cast<std::size_t>(m);
                    for (int t = 0; t < layer; ++t) {
                        const float vtk =
                            ws.sel_v[static_cast<std::size_t>(t) * static_cast<std::size_t>(H_beam) +
                                     static_cast<std::size_t>(out)];
                        ws.ACoeff(ws.a_next, t, out) = a_old[t] - vtk * a_last;
                    }
                    ws.ACoeff(ws.a_next, layer, out) = a_last;

                    const float* R_pref = ws.Chol(ws.chol_curr, src_h);
                    float* R_new = ws.Chol(ws.chol_next, out);
                    std::memcpy(R_new, R_pref,
                                sizeof(float) * static_cast<std::size_t>(layer) *
                                    static_cast<std::size_t>(m));
                    for (int t = 0; t < layer; ++t) {
                        const float ytk =
                            ws.sel_y[static_cast<std::size_t>(t) * static_cast<std::size_t>(H_beam) +
                                     static_cast<std::size_t>(out)];
                        R_new[static_cast<std::size_t>(t) * m + layer] = ytk;
                    }
                    R_new[static_cast<std::size_t>(layer) * m + layer] = std::sqrt(denom);
                }

                new_H = CountFinite(ws.errs_next.data(), H_beam);
                if (new_H == 0) {
                    // Julia reference: fallback when all candidates are skipped due to denom==0
                    // (Beam_search.jl:378-389).
                    new_H = std::min(curr_H, H_beam);
                    for (int h = 0; h < new_H; ++h) {
                        ws.errs_next[h] = ws.errs_curr[h];
                        for (int t = 0; t <= layer; ++t) {
                            ws.CodeAt(ws.codes_next, t, h) = ws.CodeAt(ws.codes_curr, t, h);
                            ws.ACoeff(ws.a_next, t, h) = ws.ACoeff(ws.a_curr, t, h);
                        }
                        float* R_src = ws.Chol(ws.chol_curr, h);
                        float* R_dst = ws.Chol(ws.chol_next, h);
                        std::copy(R_src, R_src + m * m, R_dst);
                    }
                }

                curr_H = std::max(1, new_H);
                std::swap(ws.codes_curr, ws.codes_next);
                std::swap(ws.errs_curr, ws.errs_next);
                std::swap(ws.a_curr, ws.a_next);
                std::swap(ws.chol_curr, ws.chol_next);
            }

            int best_idx = 0;
            float best_err = ws.errs_curr[0];
            for (int h = 1; h < curr_H; ++h) {
                if (ws.errs_curr[h] < best_err) {
                    best_err = ws.errs_curr[h];
                    best_idx = h;
                }
            }
            for (int L = 0; L < m; ++L) {
                (*B)(L, i) = ws.CodeAt(ws.codes_curr, L, best_idx);
            }
        }
    }
}

namespace {

struct BeamWorkspaceLargeRoot {
    int m = 0;
    int H = 0;
    int maxK = 0;
    // root codes are large (e.g. 16384) => store as int.
    std::vector<int> root_curr;
    std::vector<int> root_next;

    // small codes (layers 1..m-1) fit in uint8.
    std::vector<Code> codes_curr;
    std::vector<Code> codes_next;

    std::vector<float> errs_curr;
    std::vector<float> errs_next;
    std::vector<float> a_curr;
    std::vector<float> a_next;
    std::vector<float> chol_curr;
    std::vector<float> chol_next;
    std::vector<float> tmp_a;
    std::vector<float> tmp_v;
    std::vector<float> tmp_y;

    // Per-layer batched buffers (stride = maxK).
    std::vector<float> matA;
    std::vector<float> matY;
    std::vector<float> matV;
    std::vector<float> vec_alpha;
    std::vector<float> vec_gamma;
    std::vector<float> vec_eta;
    std::vector<float> vec_delta;

    // Selection metadata for deferred write-back (size H).
    std::vector<int> sel_src_h;
    std::vector<int> sel_k;
    std::vector<int> sel_root;
    std::vector<float> sel_denom;
    std::vector<float> sel_alast;
    // Per-selection vectors (size m*H, only first `layer` rows are used).
    std::vector<float> sel_v;
    std::vector<float> sel_y;

    BeamWorkspaceLargeRoot(int m_in, int H_in, int maxK_in)
        : m(m_in),
          H(H_in),
          maxK(maxK_in),
          root_curr(H_in, 0),
          root_next(H_in, 0),
          codes_curr(static_cast<std::size_t>(std::max(0, m_in - 1)) * H_in),
          codes_next(static_cast<std::size_t>(std::max(0, m_in - 1)) * H_in),
          errs_curr(H_in),
          errs_next(H_in),
          a_curr(static_cast<std::size_t>(m_in) * H_in),
          a_next(static_cast<std::size_t>(m_in) * H_in),
          chol_curr(static_cast<std::size_t>(m_in) * m_in * H_in),
          chol_next(static_cast<std::size_t>(m_in) * m_in * H_in),
          tmp_a(m_in),
          tmp_v(m_in),
          tmp_y(m_in),
          matA(static_cast<std::size_t>(m_in) * static_cast<std::size_t>(std::max(1, maxK_in)), 0.0f),
          matY(static_cast<std::size_t>(m_in) * static_cast<std::size_t>(std::max(1, maxK_in)), 0.0f),
          matV(static_cast<std::size_t>(m_in) * static_cast<std::size_t>(std::max(1, maxK_in)), 0.0f),
          vec_alpha(static_cast<std::size_t>(std::max(1, maxK_in)), 0.0f),
          vec_gamma(static_cast<std::size_t>(std::max(1, maxK_in)), 0.0f),
          vec_eta(static_cast<std::size_t>(std::max(1, maxK_in)), 0.0f),
          vec_delta(static_cast<std::size_t>(std::max(1, maxK_in)), 0.0f),
          sel_src_h(static_cast<std::size_t>(H_in), -1),
          sel_k(static_cast<std::size_t>(H_in), 0),
          sel_root(static_cast<std::size_t>(H_in), 0),
          sel_denom(static_cast<std::size_t>(H_in), 0.0f),
          sel_alast(static_cast<std::size_t>(H_in), 0.0f),
          sel_v(static_cast<std::size_t>(m_in) * static_cast<std::size_t>(H_in), 0.0f),
          sel_y(static_cast<std::size_t>(m_in) * static_cast<std::size_t>(H_in), 0.0f) {}

    Code& CodeSmallAt(std::vector<Code>& buf, int layer, int beam) const {
        // layer in [1, m)
        return buf[static_cast<std::size_t>(beam) * static_cast<std::size_t>(m - 1) +
                   static_cast<std::size_t>(layer - 1)];
    }
    float& ACoeff(std::vector<float>& buf, int layer, int beam) const {
        return buf[static_cast<std::size_t>(beam) * static_cast<std::size_t>(m) +
                   static_cast<std::size_t>(layer)];
    }
    float* Chol(std::vector<float>& buf, int beam) const {
        return buf.data() + static_cast<std::size_t>(beam) * m * m;
    }
};

}  // namespace

namespace {

inline void SolveCholUpperRawBatched(const float* STLQ_RESTRICT R,
                                     int n,
                                     int strideR,
                                     const float* STLQ_RESTRICT A,  // n×K row-major (strideK)
                                     int strideK,
                                     float* STLQ_RESTRICT Y,        // n×K row-major (strideK)
                                     float* STLQ_RESTRICT V,        // n×K row-major (strideK)
                                     int K) {
    // Copy A -> Y (only the K columns we use).
    for (int i = 0; i < n; ++i) {
        const float* src = A + static_cast<std::size_t>(i) * static_cast<std::size_t>(strideK);
        float* dst = Y + static_cast<std::size_t>(i) * static_cast<std::size_t>(strideK);
        for (int k = 0; k < K; ++k) {
            dst[k] = src[k];
        }
    }

    // Forward solve (R^T) * Y = A  (R upper => R^T lower).
    for (int i = 0; i < n; ++i) {
        float* Yi = Y + static_cast<std::size_t>(i) * static_cast<std::size_t>(strideK);
        for (int j = 0; j < i; ++j) {
            const float rij = R[static_cast<std::size_t>(j) * static_cast<std::size_t>(strideR) + i];
            const float* Yj = Y + static_cast<std::size_t>(j) * static_cast<std::size_t>(strideK);
            for (int k = 0; k < K; ++k) {
                Yi[k] -= rij * Yj[k];
            }
        }
        const float diag = R[static_cast<std::size_t>(i) * static_cast<std::size_t>(strideR) + i];
        const float inv_diag = 1.0f / diag;
        for (int k = 0; k < K; ++k) {
            Yi[k] *= inv_diag;
        }
    }

    // Copy Y -> V then backward solve R * V = Y.
    for (int i = 0; i < n; ++i) {
        const float* src = Y + static_cast<std::size_t>(i) * static_cast<std::size_t>(strideK);
        float* dst = V + static_cast<std::size_t>(i) * static_cast<std::size_t>(strideK);
        for (int k = 0; k < K; ++k) {
            dst[k] = src[k];
        }
    }

    for (int i = n - 1; i >= 0; --i) {
        float* Vi = V + static_cast<std::size_t>(i) * static_cast<std::size_t>(strideK);
        for (int j = i + 1; j < n; ++j) {
            const float rij = R[static_cast<std::size_t>(i) * static_cast<std::size_t>(strideR) + j];
            const float* Vj = V + static_cast<std::size_t>(j) * static_cast<std::size_t>(strideK);
            for (int k = 0; k < K; ++k) {
                Vi[k] -= rij * Vj[k];
            }
        }
        const float diag = R[static_cast<std::size_t>(i) * static_cast<std::size_t>(strideR) + i];
        const float inv_diag = 1.0f / diag;
        for (int k = 0; k < K; ++k) {
            Vi[k] *= inv_diag;
        }
    }
}

}  // namespace

void BeamSearchPrefixLSLargeRoot(const ColMajorMatrix<float>& X,
                                 const PrecompLargeRoot& pre,
                                 const ColMajorMatrix<float>& xC_small,
                                 int H_beam,
                                 std::vector<std::uint32_t>* cluster_id,
                                 ColMajorMatrix<Code>* B_small,
                                 BeamSearchLargeRootTiming* timing) {
    const int n = X.cols;
    const int d = X.rows;
    const int m = pre.m;
    const int h0 = pre.h_vec.empty() ? 0 : pre.h_vec[0];
    const int Hs = pre.H_small;

    if (!cluster_id || !B_small || n <= 0) {
        return;
    }
    cluster_id->resize(static_cast<std::size_t>(n));
    B_small->rows = std::max(0, m - 1);
    B_small->cols = n;
    B_small->data.resize(static_cast<std::size_t>(B_small->rows) * static_cast<std::size_t>(n));
    if (!pre.ready || d <= 0 || m <= 0 || h0 <= 0 || xC_small.rows != Hs || xC_small.cols != n) {
        return;
    }

    // Tune blocks for CPU BLAS throughput: prefer larger query blocks (fewer GEMM calls)
    // while keeping the score buffer size reasonable.
    const int q_block = std::min(8192, std::max(1, n));
    const int r_block = 512;

    struct Scratch {
        std::vector<float> norm_x2;
        std::vector<float> root_errs;
        std::vector<int> root_codes;
        std::vector<float> root_a0;
        std::vector<float> root_r00;
        ColMajorMatrix<float> scores;
    };
    Scratch scratch;
    scratch.scores.data.resize(static_cast<std::size_t>(r_block) * static_cast<std::size_t>(q_block));

    for (int q0 = 0; q0 < n; q0 += q_block) {
        const int qlen = std::min(q_block, n - q0);

        // norm_x2 for this block.
        Timer tseg;
        if (timing) tseg.Reset();
        scratch.norm_x2.assign(static_cast<std::size_t>(qlen), 0.0f);
        #pragma omp parallel for default(none) shared(scratch, X) firstprivate(q0, qlen, d) schedule(static)
        for (int j = 0; j < qlen; ++j) {
            const float* x = X.Col(q0 + j);
            float s = 0.0f;
            #pragma omp simd reduction(+:s)
            for (int t = 0; t < d; ++t) {
                const float v = x[t];
                s += v * v;
            }
            scratch.norm_x2[static_cast<std::size_t>(j)] = s;
        }
        if (timing) timing->norm_x2 += tseg.ElapsedSeconds();

        // Root init arrays: per-sample top-H candidates.
        scratch.root_errs.assign(static_cast<std::size_t>(qlen) * static_cast<std::size_t>(H_beam),
                                 std::numeric_limits<float>::infinity());
        scratch.root_codes.assign(static_cast<std::size_t>(qlen) * static_cast<std::size_t>(H_beam), 0);
        scratch.root_a0.assign(static_cast<std::size_t>(qlen) * static_cast<std::size_t>(H_beam), 0.0f);
        scratch.root_r00.assign(static_cast<std::size_t>(qlen) * static_cast<std::size_t>(H_beam), 0.0f);

        // score buffer (rlen × qlen)
        ColMajorMatrix<float>& scores = scratch.scores;

        // Compute root dot scores in blocks: scores = C0_block' * X_block
        for (int r0 = 0; r0 < h0; r0 += r_block) {
            const int rlen = std::min(r_block, h0 - r0);
            scores.rows = rlen;
            scores.cols = qlen;
            const std::size_t need_scores =
                static_cast<std::size_t>(rlen) * static_cast<std::size_t>(qlen);
            if (scores.data.size() < need_scores) {
                scores.data.resize(need_scores);
            }

            const float* A = pre.C0.data.data() + static_cast<std::size_t>(r0) * d;
            const float* B = X.data.data() + static_cast<std::size_t>(q0) * d;
            if (timing) tseg.Reset();
            {
                ScopedBlasThreads blas_scope(OmpMaxThreads());
                GemmRaw(true, false,
                        rlen, qlen, d,
                        1.0f,
                        A, d,
                        B, d,
                        0.0f,
                        scores.data.data(), rlen);
            }
            if (timing) timing->root_gemm += tseg.ElapsedSeconds();

            if (timing) tseg.Reset();
            #pragma omp parallel for default(none) shared(scratch, scores, pre) firstprivate(qlen, H_beam, r0, rlen) schedule(static)
            for (int j = 0; j < qlen; ++j) {
                float* errs = scratch.root_errs.data() +
                              static_cast<std::size_t>(j) * static_cast<std::size_t>(H_beam);
                int* codes = scratch.root_codes.data() +
                             static_cast<std::size_t>(j) * static_cast<std::size_t>(H_beam);
                float* a0 = scratch.root_a0.data() +
                            static_cast<std::size_t>(j) * static_cast<std::size_t>(H_beam);
                float* r00 = scratch.root_r00.data() +
                             static_cast<std::size_t>(j) * static_cast<std::size_t>(H_beam);
                const float nx2 = scratch.norm_x2[static_cast<std::size_t>(j)];

                float worst_err = errs[0];
                int worst_idx_cached = RecomputeWorst(errs, H_beam, &worst_err);
                for (int kk = 0; kk < rlen; ++kk) {
                    const int code = r0 + kk;
                    const float alpha = pre.norm0[static_cast<std::size_t>(code)];
                    if (alpha == 0.0f) {
                        continue;
                    }
                    const float gamma = scores(kk, j);
                    const float a1 = gamma / alpha;
                    const float e1 = nx2 - gamma * a1;

                    if (e1 >= worst_err) continue;
                    errs[worst_idx_cached] = e1;
                    codes[worst_idx_cached] = code;
                    a0[worst_idx_cached] = a1;
                    r00[worst_idx_cached] = std::sqrt(alpha);
                    worst_idx_cached = RecomputeWorst(errs, H_beam, &worst_err);
                }
            }
            if (timing) timing->root_update += tseg.ElapsedSeconds();
        }

        if (timing) tseg.Reset();
        BeamSearchPrefixLSLargeRootExpandFromRootCandidates(
            pre,
            xC_small,
            q0,
            qlen,
            H_beam,
            scratch.root_errs.data(),
            scratch.root_codes.data(),
            scratch.root_a0.data(),
            scratch.root_r00.data(),
            cluster_id,
            B_small);
        if (timing) timing->expand += tseg.ElapsedSeconds();
    }
}

void BeamSearchPrefixLSLargeRootExpandFromRootCandidates(
    const PrecompLargeRoot& pre,
    const ColMajorMatrix<float>& xC_small,
    int q0,
    int qlen,
    int H_beam,
    const float* root_errs,
    const int* root_codes,
    const float* root_a0,
    const float* root_r00,
    std::vector<std::uint32_t>* cluster_id,
    ColMajorMatrix<Code>* B_small) {
    if (!cluster_id || !B_small || qlen <= 0 || H_beam <= 0) return;
    const int n = xC_small.cols;
    const int m = pre.m;
    if (q0 < 0 || q0 + qlen > n) return;

    int maxK = 0;
    for (int layer = 1; layer < m; ++layer) {
        maxK = std::max(maxK, pre.h_vec[layer]);
    }
    maxK = std::max(1, maxK);

    #pragma omp parallel default(none) shared(pre, xC_small, root_errs, root_codes, root_a0, root_r00, cluster_id, B_small) firstprivate(q0, qlen, H_beam, m, maxK)
    {
        BeamWorkspaceLargeRoot ws(m, H_beam, maxK);

        #pragma omp for schedule(static)
        for (int j = 0; j < qlen; ++j) {
            const int i = q0 + j;
            const float* xCi_small = xC_small.Col(i);

            std::fill(ws.errs_curr.begin(), ws.errs_curr.end(), std::numeric_limits<float>::infinity());
            std::fill(ws.errs_next.begin(), ws.errs_next.end(), std::numeric_limits<float>::infinity());
            // NOTE: We rely on root candidates being packed first (finite entries at [0..curr_H)).
            // Avoid bulk-clearing the full workspace: H_beam is small but this runs at massive scale.

            int curr_H = 0;
            const float* errs0 =
                root_errs + static_cast<std::size_t>(j) * static_cast<std::size_t>(H_beam);
            const int* codes0 =
                root_codes + static_cast<std::size_t>(j) * static_cast<std::size_t>(H_beam);
            const float* a00 =
                root_a0 + static_cast<std::size_t>(j) * static_cast<std::size_t>(H_beam);
            const float* r000 =
                root_r00 + static_cast<std::size_t>(j) * static_cast<std::size_t>(H_beam);
            for (int h = 0; h < H_beam; ++h) {
                if (errs0[h] >= std::numeric_limits<float>::infinity()) {
                    continue;
                }
                ws.errs_curr[h] = errs0[h];
                ws.root_curr[h] = codes0[h];
                {
                    float* arow = ws.a_curr.data() +
                                  static_cast<std::size_t>(h) * static_cast<std::size_t>(m);
                    std::fill(arow, arow + m, 0.0f);
                    arow[0] = a00[h];
                }
                {
                    float* R = ws.Chol(ws.chol_curr, h);
                    std::fill(R, R + static_cast<std::size_t>(m) * static_cast<std::size_t>(m), 0.0f);
                    R[0] = r000[h];
                }
                ++curr_H;
            }
            curr_H = std::max(1, curr_H);

            const int H_small = pre.G_small.rows;
            const float* __restrict Gs = pre.G_small.data.data();
            const float* __restrict G0S = pre.G0S.data.data();
            const int G0S_rows = pre.G0S.rows;

            for (int layer = 1; layer < m; ++layer) {
                std::fill(ws.errs_next.begin(), ws.errs_next.end(), std::numeric_limits<float>::infinity());
                float worst_err = ws.errs_next[0];
                int worst_idx_cached = RecomputeWorst(ws.errs_next.data(), H_beam, &worst_err);
                int new_H = 0;

                // Reset selection metadata for this layer.
                std::fill(ws.sel_src_h.begin(), ws.sel_src_h.end(), -1);

                const int K = pre.h_vec[layer];
                const int off = pre.small_offsets[layer];
                const int strideK = ws.maxK;

                for (int h = 0; h < curr_H; ++h) {
                    const float* R_pref = ws.Chol(ws.chol_curr, h);
                    const float* a_old =
                        ws.a_curr.data() + static_cast<std::size_t>(h) * static_cast<std::size_t>(m);
                    const float E_old = ws.errs_curr[h];
                    const int root_code = ws.root_curr[h];

                    // Build A(:,k), alpha(k), gamma(k) for all codes k in this layer.
                    for (int k = 0; k < K; ++k) {
                        const int flat = off + k;
                        const float* __restrict Gs_col =
                            Gs + static_cast<std::size_t>(flat) * static_cast<std::size_t>(H_small);
                        ws.vec_alpha[static_cast<std::size_t>(k)] = Gs_col[flat];
                        ws.vec_gamma[static_cast<std::size_t>(k)] = xCi_small[flat];
                        ws.matA[static_cast<std::size_t>(0) * static_cast<std::size_t>(strideK) +
                                static_cast<std::size_t>(k)] =
                            (G0S + static_cast<std::size_t>(flat) * static_cast<std::size_t>(G0S_rows))[root_code];
                        for (int t = 1; t < layer; ++t) {
                            const int flat_t =
                                pre.small_offsets[t] + static_cast<int>(ws.CodeSmallAt(ws.codes_curr, t, h));
                            ws.matA[static_cast<std::size_t>(t) * static_cast<std::size_t>(strideK) +
                                    static_cast<std::size_t>(k)] = Gs_col[flat_t];
                        }
                    }

                    // Solve v = A_pref^{-1} * A and y = R^{-T} * A in batch.
                    SolveCholUpperRawBatched(R_pref, layer, m,
                                             ws.matA.data(), strideK,
                                             ws.matY.data(), ws.matV.data(), K);

                    // eta(k)=A(:,k)^T a_old, delta(k)=A(:,k)^T v(:,k)
                    std::fill(ws.vec_eta.begin(), ws.vec_eta.begin() + K, 0.0f);
                    std::fill(ws.vec_delta.begin(), ws.vec_delta.begin() + K, 0.0f);
                    for (int t = 0; t < layer; ++t) {
                        const float* Arow =
                            ws.matA.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(strideK);
                        const float* Vrow =
                            ws.matV.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(strideK);
                        const float aot = a_old[t];
                        float* eta = ws.vec_eta.data();
                        float* del = ws.vec_delta.data();
                        #pragma omp simd
                        for (int k = 0; k < K; ++k) {
                            const float a = Arow[k];
                            eta[k] += a * aot;
                            del[k] += a * Vrow[k];
                        }
                    }

                    for (int k = 0; k < K; ++k) {
                        const float alpha = ws.vec_alpha[static_cast<std::size_t>(k)];
                        const float denom = alpha - ws.vec_delta[static_cast<std::size_t>(k)];
                        if (denom == 0.0f) continue;
                        const float diff = ws.vec_gamma[static_cast<std::size_t>(k)] -
                                           ws.vec_eta[static_cast<std::size_t>(k)];
                        const float a_last = diff / denom;
                        float e = E_old - diff * a_last;
                        if (e < 0.0f) e = 0.0f;
                        if (e >= worst_err) continue;

                        ws.errs_next[worst_idx_cached] = e;
                        ws.sel_src_h[static_cast<std::size_t>(worst_idx_cached)] = h;
                        ws.sel_k[static_cast<std::size_t>(worst_idx_cached)] = k;
                        ws.sel_root[static_cast<std::size_t>(worst_idx_cached)] = root_code;
                        ws.sel_denom[static_cast<std::size_t>(worst_idx_cached)] = denom;
                        ws.sel_alast[static_cast<std::size_t>(worst_idx_cached)] = a_last;

                        // Snapshot v(:,k) and y(:,k) for this candidate (only 0..layer-1 used).
                        for (int t = 0; t < layer; ++t) {
                            ws.sel_v[static_cast<std::size_t>(t) * static_cast<std::size_t>(H_beam) +
                                     static_cast<std::size_t>(worst_idx_cached)] =
                                ws.matV[static_cast<std::size_t>(t) * static_cast<std::size_t>(strideK) +
                                        static_cast<std::size_t>(k)];
                            ws.sel_y[static_cast<std::size_t>(t) * static_cast<std::size_t>(H_beam) +
                                     static_cast<std::size_t>(worst_idx_cached)] =
                                ws.matY[static_cast<std::size_t>(t) * static_cast<std::size_t>(strideK) +
                                        static_cast<std::size_t>(k)];
                        }

                        worst_idx_cached = RecomputeWorst(ws.errs_next.data(), H_beam, &worst_err);
                    }
                }

                // Materialize selected candidates for this layer into next buffers.
                const float inf = std::numeric_limits<float>::infinity();
                for (int out = 0; out < H_beam; ++out) {
                    if (ws.errs_next[out] >= inf) continue;
                    const int src_h = ws.sel_src_h[static_cast<std::size_t>(out)];
                    // Finite errs_next slots are produced by the candidate loop, which writes
                    // sel_src_h from h in [0, curr_H). CUDA path has no matching guard.
                    // if (src_h < 0 || src_h >= curr_H) {
                    //     LogWarn("BeamSearchPrefixLS: beam slot " + std::to_string(out) +
                    //             " has valid err but src_h=" + std::to_string(src_h) +
                    //             " out of range [0," + std::to_string(curr_H) +
                    //             "); beam state corrupt. Skipping slot.");
                    //     continue;
                    // }

                    const int root_code = ws.sel_root[static_cast<std::size_t>(out)];
                    const int k = ws.sel_k[static_cast<std::size_t>(out)];
                    const float denom = ws.sel_denom[static_cast<std::size_t>(out)];
                    const float a_last = ws.sel_alast[static_cast<std::size_t>(out)];

                    ws.root_next[out] = root_code;
                    for (int t = 1; t < layer; ++t) {
                        ws.CodeSmallAt(ws.codes_next, t, out) =
                            ws.CodeSmallAt(ws.codes_curr, t, src_h);
                    }
                    ws.CodeSmallAt(ws.codes_next, layer, out) = static_cast<Code>(k);

                    const float* a_old =
                        ws.a_curr.data() + static_cast<std::size_t>(src_h) * static_cast<std::size_t>(m);
                    for (int t = 0; t < layer; ++t) {
                        const float vtk =
                            ws.sel_v[static_cast<std::size_t>(t) * static_cast<std::size_t>(H_beam) +
                                     static_cast<std::size_t>(out)];
                        ws.ACoeff(ws.a_next, t, out) = a_old[t] - vtk * a_last;
                    }
                    ws.ACoeff(ws.a_next, layer, out) = a_last;

                    const float* R_pref = ws.Chol(ws.chol_curr, src_h);
                    float* R_new = ws.Chol(ws.chol_next, out);
                    std::memcpy(R_new, R_pref,
                                sizeof(float) * static_cast<std::size_t>(layer) *
                                    static_cast<std::size_t>(m));
                    for (int t = 0; t < layer; ++t) {
                        const float ytk =
                            ws.sel_y[static_cast<std::size_t>(t) * static_cast<std::size_t>(H_beam) +
                                     static_cast<std::size_t>(out)];
                        R_new[static_cast<std::size_t>(t) * static_cast<std::size_t>(m) +
                              static_cast<std::size_t>(layer)] = ytk;
                    }
                    R_new[static_cast<std::size_t>(layer) * static_cast<std::size_t>(m) +
                          static_cast<std::size_t>(layer)] = std::sqrt(denom);
                }

                new_H = CountFinite(ws.errs_next.data(), H_beam);
                if (new_H == 0) {
                    new_H = std::min(curr_H, H_beam);
                    for (int h = 0; h < new_H; ++h) {
                        ws.errs_next[h] = ws.errs_curr[h];
                        ws.root_next[h] = ws.root_curr[h];
                        for (int t = 1; t <= layer; ++t) {
                            ws.CodeSmallAt(ws.codes_next, t, h) =
                                ws.CodeSmallAt(ws.codes_curr, t, h);
                        }
                        for (int t = 0; t <= layer; ++t) {
                            ws.ACoeff(ws.a_next, t, h) = ws.ACoeff(ws.a_curr, t, h);
                        }
                        const float* R_src = ws.Chol(ws.chol_curr, h);
                        float* R_dst = ws.Chol(ws.chol_next, h);
                        std::copy(R_src, R_src + m * m, R_dst);
                    }
                }

                curr_H = std::max(1, new_H);
                std::swap(ws.root_curr, ws.root_next);
                std::swap(ws.codes_curr, ws.codes_next);
                std::swap(ws.errs_curr, ws.errs_next);
                std::swap(ws.a_curr, ws.a_next);
                std::swap(ws.chol_curr, ws.chol_next);
            }

            int best_idx = 0;
            float best_err = ws.errs_curr[0];
            for (int h = 1; h < curr_H; ++h) {
                if (ws.errs_curr[h] < best_err) {
                    best_err = ws.errs_curr[h];
                    best_idx = h;
                }
            }

            (*cluster_id)[static_cast<std::size_t>(i)] =
                static_cast<std::uint32_t>(ws.root_curr[best_idx]);
            for (int layer = 1; layer < m; ++layer) {
                (*B_small)(layer - 1, i) = ws.CodeSmallAt(ws.codes_curr, layer, best_idx);
            }
        }
    }
}

}  // namespace stlq
