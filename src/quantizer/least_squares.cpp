#include "stlq/quantizer/least_squares.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <omp.h>

#include "stlq/core/lapack.h"
#include "stlq/common/logger.h"
#include "stlq/quantizer/linear_algebra.h"
#include "stlq/quantizer/precomp_large_root.h"

namespace stlq {

namespace {

constexpr float kEps = 1e-6f;

inline float& ACol(std::vector<float>& A, int m, int r, int c) {
    return A[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * m];
}

template <int M>
inline float& AMat(float* A, int r, int c) {
    return A[static_cast<std::size_t>(r) * static_cast<std::size_t>(M) + static_cast<std::size_t>(c)];
}

template <int M>
inline const float& AMatConst(const float* A, int r, int c) {
    return A[static_cast<std::size_t>(r) * static_cast<std::size_t>(M) + static_cast<std::size_t>(c)];
}

inline float Dot(const float* a, const float* b, int d) {
    float s = 0.0f;
    #pragma omp simd reduction(+:s)
    for (int i = 0; i < d; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

template <int M>
bool CholeskyUpperFixed(float* A) {
    for (int k = 0; k < M; ++k) {
        float sum = AMat<M>(A, k, k);
        for (int j = 0; j < k; ++j) {
            const float v = AMat<M>(A, j, k);
            sum -= v * v;
        }
        if (sum <= kEps) {
            return false;
        }
        const float diag = std::sqrt(sum);
        AMat<M>(A, k, k) = diag;
        for (int i = k + 1; i < M; ++i) {
            float s = AMat<M>(A, k, i);
            for (int j = 0; j < k; ++j) {
                s -= AMat<M>(A, j, k) * AMat<M>(A, j, i);
            }
            AMat<M>(A, k, i) = s / diag;
        }
    }
    return true;
}

template <int M>
void SolveCholUpperFixed(const float* R, const float* b, float* x, float* y) {
    for (int i = 0; i < M; ++i) {
        float s = b[i];
        for (int k = 0; k < i; ++k) {
            s -= AMatConst<M>(R, k, i) * y[k];
        }
        y[i] = s / AMatConst<M>(R, i, i);
    }
    for (int i = M - 1; i >= 0; --i) {
        float s = y[i];
        for (int k = i + 1; k < M; ++k) {
            s -= AMatConst<M>(R, i, k) * x[k];
        }
        x[i] = s / AMatConst<M>(R, i, i);
    }
}

template <int M>
bool SolveLeastSquaresSampleFixed(const Precomp& precomp,
                                 const ColMajorMatrix<float>& xC,
                                 const ColMajorMatrix<FullCode>& B,
                                 int sample,
                                 float* a_out) {
    float A[M * M];
    float b[M];
    float x[M];
    float y[M];

    auto fill_system = [&](float lambda) {
        for (int idx = 0; idx < M * M; ++idx) {
            A[idx] = 0.0f;
        }
        for (int j = 0; j < M; ++j) {
            const int flat_j = precomp.offsets[j] + static_cast<int>(B(j, sample));
            b[j] = xC(flat_j, sample);
            for (int k = 0; k <= j; ++k) {
                const int flat_k = precomp.offsets[k] + static_cast<int>(B(k, sample));
                AMat<M>(A, k, j) = GAt(precomp.G, flat_k, flat_j);
            }
            AMat<M>(A, j, j) += lambda;
        }
    };

    fill_system(0.0f);
    bool ok = CholeskyUpperFixed<M>(A);
    if (!ok) {
        float bump = kEps;
        for (int attempt = 0; attempt < 3 && !ok; ++attempt) {
            fill_system(bump);
            ok = CholeskyUpperFixed<M>(A);
            bump *= 10.0f;
        }
    }

    if (!ok) {
        return false;
    }
    SolveCholUpperFixed<M>(A, b, x, y);
    std::copy(x, x + M, a_out);
    return true;
}

template <int M>
bool SolveLeastSquaresSampleLargeRootFixed(const PrecompLargeRoot& pre,
                                          const ColMajorMatrix<float>& X,
                                          const ColMajorMatrix<float>& xC_small,
                                          const std::vector<std::uint32_t>& cluster_id,
                                          const ColMajorMatrix<Code>& B_small,
                                          int sample,
                                          float* a_out) {
    float A[M * M];
    float b[M];
    float x[M];
    float y[M];

    const int d = pre.d;
    const std::uint32_t cid = cluster_id[static_cast<std::size_t>(sample)];
    // Hot-path LS callers provide cluster_id from validated root assignments.
    // if (cid >= static_cast<std::uint32_t>(pre.h_vec[0])) {
    //     return false;
    // }
    const bool has_g0s_t = !pre.G0S_T.data.empty();
    const float* g0s_t_col = has_g0s_t ? pre.G0S_T.Col(static_cast<int>(cid)) : nullptr;
    const float* xvec = X.Col(sample);
    const float* c0 = pre.C0.Col(static_cast<int>(cid));

    auto fill_system = [&](float lambda) {
        for (int idx = 0; idx < M * M; ++idx) {
            A[idx] = 0.0f;
        }
        b[0] = Dot(c0, xvec, d);
        AMat<M>(A, 0, 0) = pre.norm0[static_cast<std::size_t>(cid)] + lambda;
        for (int j = 1; j < M; ++j) {
            const int flat_j = pre.small_offsets[j] + static_cast<int>(B_small(j - 1, sample));
            b[j] = xC_small(flat_j, sample);
            AMat<M>(A, 0, j) = has_g0s_t ? g0s_t_col[static_cast<std::size_t>(flat_j)]
                                         : GAt(pre.G0S, static_cast<int>(cid), flat_j);
            for (int k = 1; k <= j; ++k) {
                const int flat_k = pre.small_offsets[k] + static_cast<int>(B_small(k - 1, sample));
                AMat<M>(A, k, j) = GAt(pre.G_small, flat_k, flat_j);
            }
            AMat<M>(A, j, j) += lambda;
        }
    };

    fill_system(0.0f);
    bool ok = CholeskyUpperFixed<M>(A);
    if (!ok) {
        float bump = kEps;
        for (int attempt = 0; attempt < 3 && !ok; ++attempt) {
            fill_system(bump);
            ok = CholeskyUpperFixed<M>(A);
            bump *= 10.0f;
        }
    }
    if (!ok) {
        return false;
    }
    SolveCholUpperFixed<M>(A, b, x, y);
    std::copy(x, x + M, a_out);
    return true;
}

template <int M>
bool SolveLeastSquaresSampleLargeRootFixedWithB0(const PrecompLargeRoot& pre,
                                                float b0,
                                                const ColMajorMatrix<float>& xC_small,
                                                const std::vector<std::uint32_t>& cluster_id,
                                                const ColMajorMatrix<Code>& B_small,
                                                int sample,
                                                float* a_out) {
    float A[M * M];
    float b[M];
    float x[M];
    float y[M];

    const std::uint32_t cid = cluster_id[static_cast<std::size_t>(sample)];
    // Hot-path LS callers provide cluster_id from validated root assignments.
    // if (cid >= static_cast<std::uint32_t>(pre.h_vec[0])) {
    //     return false;
    // }
    const bool has_g0s_t = !pre.G0S_T.data.empty();
    const float* g0s_t_col = has_g0s_t ? pre.G0S_T.Col(static_cast<int>(cid)) : nullptr;

    auto fill_system = [&](float lambda) {
        for (int idx = 0; idx < M * M; ++idx) {
            A[idx] = 0.0f;
        }
        b[0] = b0;
        AMat<M>(A, 0, 0) = pre.norm0[static_cast<std::size_t>(cid)] + lambda;
        for (int j = 1; j < M; ++j) {
            const int flat_j = pre.small_offsets[j] + static_cast<int>(B_small(j - 1, sample));
            b[j] = xC_small(flat_j, sample);
            AMat<M>(A, 0, j) = has_g0s_t ? g0s_t_col[static_cast<std::size_t>(flat_j)]
                                         : GAt(pre.G0S, static_cast<int>(cid), flat_j);
            for (int k = 1; k <= j; ++k) {
                const int flat_k = pre.small_offsets[k] + static_cast<int>(B_small(k - 1, sample));
                AMat<M>(A, k, j) = GAt(pre.G_small, flat_k, flat_j);
            }
            AMat<M>(A, j, j) += lambda;
        }
    };

    fill_system(0.0f);
    bool ok = CholeskyUpperFixed<M>(A);
    if (!ok) {
        float bump = kEps;
        for (int attempt = 0; attempt < 3 && !ok; ++attempt) {
            fill_system(bump);
            ok = CholeskyUpperFixed<M>(A);
            bump *= 10.0f;
        }
    }
    if (!ok) {
        return false;
    }
    SolveCholUpperFixed<M>(A, b, x, y);
    std::copy(x, x + M, a_out);
    return true;
}

}  // namespace

bool SolveLeastSquaresSample(const Precomp& precomp,
                             const ColMajorMatrix<float>& xC,
                             const ColMajorMatrix<FullCode>& B,
                             int sample,
                             std::vector<float>* A,
                             std::vector<float>* b,
                             std::vector<float>* work,
                             float* a_out) {
    // Julia reference: optimized_least_squares_single! (encode_abs.jl:249+).
    const int m = precomp.m;
    (void)work;
    // Hot-path ICM callers pass thread-local workspaces already sized by GetIcmTls().
    // if (!A || !b || !work ||
    //     static_cast<int>(A->size()) != m * m ||
    //     static_cast<int>(b->size()) != m ||
    //     static_cast<int>(work->size()) != m) {
    //     return false;
    // }

    if (m == 5) {
        return SolveLeastSquaresSampleFixed<5>(precomp, xC, B, sample, a_out);
    }
    if (m == 10) {
        return SolveLeastSquaresSampleFixed<10>(precomp, xC, B, sample, a_out);
    }

    std::fill(A->begin(), A->end(), 0.0f);
    for (int j = 0; j < m; ++j) {
        int flat_j = precomp.offsets[j] + static_cast<int>(B(j, sample));
        (*b)[j] = xC(flat_j, sample);
        for (int k = 0; k <= j; ++k) {
            int flat_k = precomp.offsets[k] + static_cast<int>(B(k, sample));
            float val = GAt(precomp.G, flat_k, flat_j);
            ACol(*A, m, k, j) = val;
        }
    }

    int info = lapack::SpotrfU(m, A->data(), m);
    if (info != 0) {
        float bump = kEps;
        for (int attempt = 0; attempt < 3 && info != 0; ++attempt) {
            std::fill(A->begin(), A->end(), 0.0f);
            for (int j = 0; j < m; ++j) {
                int flat_j = precomp.offsets[j] + static_cast<int>(B(j, sample));
                (*b)[j] = xC(flat_j, sample);
                for (int k = 0; k <= j; ++k) {
                    int flat_k = precomp.offsets[k] + static_cast<int>(B(k, sample));
                    float val = GAt(precomp.G, flat_k, flat_j);
                    ACol(*A, m, k, j) = val;
                }
                ACol(*A, m, j, j) += bump;
            }
            info = lapack::SpotrfU(m, A->data(), m);
            bump *= 10.0f;
        }
    }

    if (info != 0) {
        return false;
    }
    info = lapack::SpotrsU(m, 1, A->data(), m, b->data(), m);
    if (info != 0) {
        return false;
    }
    std::copy(b->begin(), b->end(), a_out);
    return true;
}

bool SolveLeastSquaresSampleLargeRootWithB0(const PrecompLargeRoot& pre,
                                            float b0,
                                            const ColMajorMatrix<float>& xC_small,
                                            const std::vector<std::uint32_t>& cluster_id,
                                            const ColMajorMatrix<Code>& B_small,
                                            int sample,
                                            std::vector<float>* A,
                                            std::vector<float>* b,
                                            std::vector<float>* work,
                                            float* a_out) {
    const int m = pre.m;
    (void)work;
    // Hot-path ICM callers provide ready precomp, valid sample ranges, and sized workspaces.
    // if (!pre.ready || !A || !b || !work ||
    //     static_cast<int>(A->size()) != m * m ||
    //     static_cast<int>(b->size()) != m ||
    //     static_cast<int>(work->size()) != m ||
    //     sample < 0 || sample >= xC_small.cols ||
    //     B_small.cols != xC_small.cols ||
    //     B_small.rows != std::max(0, m - 1) ||
    //     cluster_id.size() != static_cast<std::size_t>(xC_small.cols)) {
    //     return false;
    // }

    if (m == 5) {
        return SolveLeastSquaresSampleLargeRootFixedWithB0<5>(pre, b0, xC_small, cluster_id, B_small, sample, a_out);
    }
    if (m == 10) {
        return SolveLeastSquaresSampleLargeRootFixedWithB0<10>(pre, b0, xC_small, cluster_id, B_small, sample, a_out);
    }

    const std::uint32_t cid = cluster_id[static_cast<std::size_t>(sample)];
    // Hot-path LS callers provide cluster_id from validated root assignments.
    // if (cid >= static_cast<std::uint32_t>(pre.h_vec[0])) {
    //     return false;
    // }
    const bool has_g0s_t = !pre.G0S_T.data.empty();
    const float* g0s_t_col = has_g0s_t ? pre.G0S_T.Col(static_cast<int>(cid)) : nullptr;

    std::fill(A->begin(), A->end(), 0.0f);
    (*b)[0] = b0;
    ACol(*A, m, 0, 0) = pre.norm0[static_cast<std::size_t>(cid)];
    for (int j = 1; j < m; ++j) {
        const int flat_j = pre.small_offsets[j] + static_cast<int>(B_small(j - 1, sample));
        (*b)[j] = xC_small(flat_j, sample);
        ACol(*A, m, 0, j) = has_g0s_t ? g0s_t_col[static_cast<std::size_t>(flat_j)]
                                      : GAt(pre.G0S, static_cast<int>(cid), flat_j);
        for (int k = 1; k <= j; ++k) {
            const int flat_k = pre.small_offsets[k] + static_cast<int>(B_small(k - 1, sample));
            ACol(*A, m, k, j) = GAt(pre.G_small, flat_k, flat_j);
        }
    }

    int info = lapack::SpotrfU(m, A->data(), m);
    if (info != 0) {
        float bump = kEps;
        for (int attempt = 0; attempt < 3 && info != 0; ++attempt) {
            std::fill(A->begin(), A->end(), 0.0f);
            (*b)[0] = b0;
            ACol(*A, m, 0, 0) = pre.norm0[static_cast<std::size_t>(cid)] + bump;
            for (int j = 1; j < m; ++j) {
                const int flat_j = pre.small_offsets[j] + static_cast<int>(B_small(j - 1, sample));
                (*b)[j] = xC_small(flat_j, sample);
                ACol(*A, m, 0, j) = has_g0s_t ? g0s_t_col[static_cast<std::size_t>(flat_j)]
                                              : GAt(pre.G0S, static_cast<int>(cid), flat_j);
                for (int k = 1; k <= j; ++k) {
                    const int flat_k = pre.small_offsets[k] + static_cast<int>(B_small(k - 1, sample));
                    ACol(*A, m, k, j) = GAt(pre.G_small, flat_k, flat_j);
                }
                ACol(*A, m, j, j) += bump;
            }
            info = lapack::SpotrfU(m, A->data(), m);
            bump *= 10.0f;
        }
    }
    if (info != 0) {
        return false;
    }
    info = lapack::SpotrsU(m, 1, A->data(), m, b->data(), m);
    if (info != 0) {
        return false;
    }
    std::copy(b->begin(), b->end(), a_out);
    return true;
}

void SolveLeastSquaresAll(const Precomp& precomp,
                          const ColMajorMatrix<float>& xC,
                          const ColMajorMatrix<FullCode>& B,
                          ColMajorMatrix<float>* a) {
    // Julia reference: optimized_least_squares_all! (train_quantizer.jl:166-168).
    const int n = B.cols;
    const int m = precomp.m;
    a->rows = m;
    a->cols = n;
    a->data.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));

    if (m == 5 || m == 10) {
        int fail_count = 0;
        if (m == 5) {
            #pragma omp parallel default(none) shared(precomp, xC, B, a) firstprivate(n) reduction(+:fail_count)
            {
                float A_sys[5 * 5];
                float b_sys[5];
                float x_sys[5];
                float y_sys[5];

                #pragma omp for schedule(static)
                for (int i = 0; i < n; ++i) {
                    auto fill_system = [&](float lambda) {
                        for (int idx = 0; idx < 25; ++idx) A_sys[idx] = 0.0f;
                        for (int j = 0; j < 5; ++j) {
                            const int flat_j = precomp.offsets[j] + static_cast<int>(B(j, i));
                            b_sys[j] = xC(flat_j, i);
                            for (int k = 0; k <= j; ++k) {
                                const int flat_k = precomp.offsets[k] + static_cast<int>(B(k, i));
                                AMat<5>(A_sys, k, j) = GAt(precomp.G, flat_k, flat_j);
                            }
                            AMat<5>(A_sys, j, j) += lambda;
                        }
                    };

                    fill_system(0.0f);
                    bool ok = CholeskyUpperFixed<5>(A_sys);
                    if (!ok) {
                        float bump = kEps;
                        for (int attempt = 0; attempt < 3 && !ok; ++attempt) {
                            fill_system(bump);
                            ok = CholeskyUpperFixed<5>(A_sys);
                            bump *= 10.0f;
                        }
                    }
                    if (!ok) {
                        ++fail_count;
                        std::fill(a->Col(i), a->Col(i) + 5, 0.0f);
                        continue;
                    }
                    SolveCholUpperFixed<5>(A_sys, b_sys, x_sys, y_sys);
                    std::copy(x_sys, x_sys + 5, a->Col(i));
                }
            }
        } else {
            #pragma omp parallel default(none) shared(precomp, xC, B, a) firstprivate(n) reduction(+:fail_count)
            {
                float A_sys[10 * 10];
                float b_sys[10];
                float x_sys[10];
                float y_sys[10];

                #pragma omp for schedule(static)
                for (int i = 0; i < n; ++i) {
                    auto fill_system = [&](float lambda) {
                        for (int idx = 0; idx < 100; ++idx) A_sys[idx] = 0.0f;
                        for (int j = 0; j < 10; ++j) {
                            const int flat_j = precomp.offsets[j] + static_cast<int>(B(j, i));
                            b_sys[j] = xC(flat_j, i);
                            for (int k = 0; k <= j; ++k) {
                                const int flat_k = precomp.offsets[k] + static_cast<int>(B(k, i));
                                AMat<10>(A_sys, k, j) = GAt(precomp.G, flat_k, flat_j);
                            }
                            AMat<10>(A_sys, j, j) += lambda;
                        }
                    };

                    fill_system(0.0f);
                    bool ok = CholeskyUpperFixed<10>(A_sys);
                    if (!ok) {
                        float bump = kEps;
                        for (int attempt = 0; attempt < 3 && !ok; ++attempt) {
                            fill_system(bump);
                            ok = CholeskyUpperFixed<10>(A_sys);
                            bump *= 10.0f;
                        }
                    }
                    if (!ok) {
                        ++fail_count;
                        std::fill(a->Col(i), a->Col(i) + 10, 0.0f);
                        continue;
                    }
                    SolveCholUpperFixed<10>(A_sys, b_sys, x_sys, y_sys);
                    std::copy(x_sys, x_sys + 10, a->Col(i));
                }
            }
        }

        if (fail_count > 0) {
            LogWarn("SolveLeastSquaresAll: failures for " + std::to_string(fail_count) +
                    " samples (zeroed coefficients).");
        }
        return;
    }

    int fail_count = 0;
    #pragma omp parallel default(none) shared(precomp, xC, B, a) firstprivate(n, m) reduction(+:fail_count)
    {
        std::vector<float> A(static_cast<std::size_t>(m) * m, 0.0f);
        std::vector<float> b(static_cast<std::size_t>(m), 0.0f);

        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            const auto fill_system = [&](float lambda) {
                std::fill(A.begin(), A.end(), 0.0f);
                for (int j = 0; j < m; ++j) {
                    int flat_j = precomp.offsets[j] + static_cast<int>(B(j, i));
                    b[j] = xC(flat_j, i);
                    for (int k = 0; k <= j; ++k) {
                        int flat_k = precomp.offsets[k] + static_cast<int>(B(k, i));
                        float val = GAt(precomp.G, flat_k, flat_j);
                        ACol(A, m, k, j) = val;
                    }
                    ACol(A, m, j, j) += lambda;
                }
            };

            fill_system(0.0f);
            int info = lapack::SpotrfU(m, A.data(), m);
            if (info != 0) {
                float bump = kEps;
                for (int attempt = 0; attempt < 3 && info != 0; ++attempt) {
                    fill_system(bump);
                    info = lapack::SpotrfU(m, A.data(), m);
                    bump *= 10.0f;
                }
            }

            if (info != 0) {
                ++fail_count;
                std::fill(a->Col(i), a->Col(i) + m, 0.0f);
                continue;
            }

            info = lapack::SpotrsU(m, 1, A.data(), m, b.data(), m);
            if (info != 0) {
                ++fail_count;
                std::fill(a->Col(i), a->Col(i) + m, 0.0f);
                continue;
            }
            std::copy(b.begin(), b.end(), a->Col(i));
        }
    }

    if (fail_count > 0) {
        LogWarn("SolveLeastSquaresAll: LAPACK failures for " + std::to_string(fail_count) +
                " samples (zeroed coefficients).");
    }
}

bool SolveLeastSquaresSampleLargeRoot(const PrecompLargeRoot& pre,
                                      const ColMajorMatrix<float>& X,
                                      const ColMajorMatrix<float>& xC_small,
                                      const std::vector<std::uint32_t>& cluster_id,
                                      const ColMajorMatrix<Code>& B_small,
                                      int sample,
                                      std::vector<float>* A,
                                      std::vector<float>* b,
                                      std::vector<float>* work,
                                      float* a_out) {
    const int m = pre.m;
    (void)work;
    // Hot-path callers provide ready precomp, valid sample ranges, and sized workspaces.
    // if (!pre.ready || !A || !b || !work ||
    //     static_cast<int>(A->size()) != m * m ||
    //     static_cast<int>(b->size()) != m ||
    //     static_cast<int>(work->size()) != m ||
    //     sample < 0 || sample >= X.cols ||
    //     xC_small.cols != X.cols ||
    //     B_small.cols != X.cols ||
    //     B_small.rows != std::max(0, m - 1) ||
    //     cluster_id.size() != static_cast<std::size_t>(X.cols)) {
    //     return false;
    // }

    if (m == 5) {
        return SolveLeastSquaresSampleLargeRootFixed<5>(pre, X, xC_small, cluster_id, B_small, sample, a_out);
    }
    if (m == 10) {
        return SolveLeastSquaresSampleLargeRootFixed<10>(pre, X, xC_small, cluster_id, B_small, sample, a_out);
    }

    const int d = pre.d;
    const std::uint32_t cid = cluster_id[static_cast<std::size_t>(sample)];
    // Hot-path LS callers provide cluster_id from validated root assignments.
    // if (cid >= static_cast<std::uint32_t>(pre.h_vec[0])) {
    //     return false;
    // }
    const bool has_g0s_t = !pre.G0S_T.data.empty();
    const float* g0s_t_col = has_g0s_t ? pre.G0S_T.Col(static_cast<int>(cid)) : nullptr;

    std::fill(A->begin(), A->end(), 0.0f);

    // b0 = dot(C0[:,cid], x)
    const float* x = X.Col(sample);
    const float* c0 = pre.C0.Col(static_cast<int>(cid));
    (*b)[0] = Dot(c0, x, d);

    // Fill b and upper-triangle of A using:
    // - root-small via G0S
    // - small-small via G_small
    // - diag(root) via norm0
    ACol(*A, m, 0, 0) = pre.norm0[static_cast<std::size_t>(cid)];
    for (int j = 1; j < m; ++j) {
        const int flat_j = pre.small_offsets[j] + static_cast<int>(B_small(j - 1, sample));
        (*b)[j] = xC_small(flat_j, sample);

        // root-small (row=0, col=j)
        ACol(*A, m, 0, j) = has_g0s_t ? g0s_t_col[static_cast<std::size_t>(flat_j)]
                                      : GAt(pre.G0S, static_cast<int>(cid), flat_j);
        for (int k = 1; k <= j; ++k) {
            const int flat_k = pre.small_offsets[k] + static_cast<int>(B_small(k - 1, sample));
            ACol(*A, m, k, j) = GAt(pre.G_small, flat_k, flat_j);
        }
    }

    int info = lapack::SpotrfU(m, A->data(), m);
    if (info != 0) {
        float bump = kEps;
        for (int attempt = 0; attempt < 3 && info != 0; ++attempt) {
            // Rebuild with diag regularization.
            std::fill(A->begin(), A->end(), 0.0f);
            (*b)[0] = Dot(c0, x, d);
            ACol(*A, m, 0, 0) = pre.norm0[static_cast<std::size_t>(cid)] + bump;
            for (int j = 1; j < m; ++j) {
                const int flat_j = pre.small_offsets[j] + static_cast<int>(B_small(j - 1, sample));
                (*b)[j] = xC_small(flat_j, sample);
                ACol(*A, m, 0, j) = has_g0s_t ? g0s_t_col[static_cast<std::size_t>(flat_j)]
                                              : GAt(pre.G0S, static_cast<int>(cid), flat_j);
                for (int k = 1; k <= j; ++k) {
                    const int flat_k = pre.small_offsets[k] + static_cast<int>(B_small(k - 1, sample));
                    ACol(*A, m, k, j) = GAt(pre.G_small, flat_k, flat_j);
                }
                ACol(*A, m, j, j) += bump;
            }
            info = lapack::SpotrfU(m, A->data(), m);
            bump *= 10.0f;
        }
    }
    if (info != 0) {
        return false;
    }
    info = lapack::SpotrsU(m, 1, A->data(), m, b->data(), m);
    if (info != 0) {
        return false;
    }
    std::copy(b->begin(), b->end(), a_out);
    return true;
}

void SolveLeastSquaresAllLargeRoot(const PrecompLargeRoot& pre,
                                  const ColMajorMatrix<float>& X,
                                  const ColMajorMatrix<float>& xC_small,
                                  const std::vector<std::uint32_t>& cluster_id,
                                  const ColMajorMatrix<Code>& B_small,
                                  ColMajorMatrix<float>* a) {
    const int n = X.cols;
    const int m = pre.m;
    a->rows = m;
    a->cols = n;
    a->data.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
    if (n <= 0) {
        return;
    }

    if (m == 5 || m == 10) {
        int fail_count = 0;
        if (m == 5) {
            #pragma omp parallel default(none) shared(pre, X, xC_small, cluster_id, B_small, a) firstprivate(n) reduction(+:fail_count)
            {
                float a_out[5];
                #pragma omp for schedule(static)
                for (int i = 0; i < n; ++i) {
                    if (!SolveLeastSquaresSampleLargeRootFixed<5>(pre, X, xC_small, cluster_id, B_small,
                                                                 i, a_out)) {
                        ++fail_count;
                        std::fill(a->Col(i), a->Col(i) + 5, 0.0f);
                    } else {
                        std::copy(a_out, a_out + 5, a->Col(i));
                    }
                }
            }
        } else {
            #pragma omp parallel default(none) shared(pre, X, xC_small, cluster_id, B_small, a) firstprivate(n) reduction(+:fail_count)
            {
                float a_out[10];
                #pragma omp for schedule(static)
                for (int i = 0; i < n; ++i) {
                    if (!SolveLeastSquaresSampleLargeRootFixed<10>(pre, X, xC_small, cluster_id, B_small,
                                                                  i, a_out)) {
                        ++fail_count;
                        std::fill(a->Col(i), a->Col(i) + 10, 0.0f);
                    } else {
                        std::copy(a_out, a_out + 10, a->Col(i));
                    }
                }
            }
        }

        if (fail_count > 0) {
            LogWarn("SolveLeastSquaresAllLargeRoot: failures for " + std::to_string(fail_count) +
                    " samples (zeroed coefficients).");
        }
        return;
    }

    int fail_count = 0;
    #pragma omp parallel default(none) shared(pre, X, xC_small, cluster_id, B_small, a) firstprivate(n, m) reduction(+:fail_count)
    {
        std::vector<float> A(static_cast<std::size_t>(m) * m, 0.0f);
        std::vector<float> b(static_cast<std::size_t>(m), 0.0f);
        std::vector<float> work(static_cast<std::size_t>(m), 0.0f);

        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            float* ai = a->Col(i);
            if (!SolveLeastSquaresSampleLargeRoot(pre, X, xC_small, cluster_id, B_small,
                                                  i, &A, &b, &work, ai)) {
                ++fail_count;
                std::fill(ai, ai + m, 0.0f);
            }
        }
    }

    if (fail_count > 0) {
        LogWarn("SolveLeastSquaresAllLargeRoot: LAPACK failures for " + std::to_string(fail_count) +
                " samples (zeroed coefficients).");
    }
}

}  // namespace stlq
