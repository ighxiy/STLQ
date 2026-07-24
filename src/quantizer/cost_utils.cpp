#include "stlq/quantizer/cost_utils.h"

#include <algorithm>
#include <cstddef>

#include <omp.h>

#include "stlq/quantizer/linear_algebra.h"
#include "stlq/quantizer/precomp_large_root.h"

namespace stlq {

namespace {

template <int M>
float SampleCostFixed(const Precomp& precomp,
                      const ColMajorMatrix<float>& xC,
                      const ColMajorMatrix<FullCode>& B,
                      const ColMajorMatrix<float>& a,
                      int sample,
                      float x_norm2) {
    float term1 = 0.0f;
    float term2 = 0.0f;
    int flat[M];
    float alpha[M];
    for (int l = 0; l < M; ++l) {
        flat[l] = precomp.offsets[l] + static_cast<int>(B(l, sample));
        alpha[l] = a(l, sample);
        term1 += alpha[l] * xC(flat[l], sample);
    }
    for (int l = 0; l < M; ++l) {
        const int flat_l = flat[l];
        const float al = alpha[l];
        for (int k = 0; k < M; ++k) {
            term2 += al * alpha[k] * GAt(precomp.G, flat_l, flat[k]);
        }
    }
    return x_norm2 - 2.0f * term1 + term2;
}

template <int M>
float SampleCostLargeRootFixed(const PrecompLargeRoot& pre,
                               const ColMajorMatrix<float>& X,
                               const ColMajorMatrix<float>& xC_small,
                               const std::vector<std::uint32_t>& cluster_id,
                               const ColMajorMatrix<Code>& B_small,
                               const ColMajorMatrix<float>& a,
                               int sample,
                               float x_norm2) {
    const int d = pre.d;
    const std::uint32_t cid = cluster_id[static_cast<std::size_t>(sample)];

    // xC0 = dot(C0[:,cid], x)
    const float* x = X.Col(sample);
    const float* c0 = pre.C0.Col(static_cast<int>(cid));
    float xC0 = 0.0f;
    #pragma omp simd reduction(+:xC0)
    for (int r = 0; r < d; ++r) {
        xC0 += c0[r] * x[r];
    }

    float term1 = a(0, sample) * xC0;
    int flat[M];
    float alpha[M];
    flat[0] = 0;
    alpha[0] = a(0, sample);
    for (int l = 1; l < M; ++l) {
        flat[l] = pre.small_offsets[l] + static_cast<int>(B_small(l - 1, sample));
        alpha[l] = a(l, sample);
        term1 += alpha[l] * xC_small(flat[l], sample);
    }

    float term2 = 0.0f;
    term2 += alpha[0] * alpha[0] * pre.norm0[static_cast<std::size_t>(cid)];
    for (int j = 1; j < M; ++j) {
        const float aj = alpha[j];
        const int flat_j = flat[j];
        term2 += 2.0f * alpha[0] * aj * GAt(pre.G0S, static_cast<int>(cid), flat_j);
        for (int k = 1; k < M; ++k) {
            term2 += aj * alpha[k] * GAt(pre.G_small, flat_j, flat[k]);
        }
    }
    return x_norm2 - 2.0f * term1 + term2;
}

}  // namespace

void ComputeXNorm2(const ColMajorMatrix<float>& X, std::vector<float>* X_norm2) {
    if (!X_norm2) {
        return;
    }
    const int d = X.rows;
    const int n = X.cols;
    X_norm2->resize(static_cast<std::size_t>(std::max(0, n)));
    if (d <= 0 || n <= 0) {
        return;
    }

    #pragma omp parallel for default(none) schedule(static) shared(X, X_norm2) firstprivate(d, n)
    for (int j = 0; j < n; ++j) {
        const float* x = X.Col(j);
        float s = 0.0f;
        #pragma omp simd reduction(+:s)
        for (int i = 0; i < d; ++i) {
            const float v = x[i];
            s += v * v;
        }
        (*X_norm2)[static_cast<std::size_t>(j)] = s;
    }
}

float SampleCost(const Precomp& precomp,
                 const ColMajorMatrix<float>& xC,
                 const ColMajorMatrix<FullCode>& B,
                 const ColMajorMatrix<float>& a,
                 int sample,
                 float x_norm2) {
    const int m = precomp.m;
    if (m == 5) {
        return SampleCostFixed<5>(precomp, xC, B, a, sample, x_norm2);
    }
    if (m == 10) {
        return SampleCostFixed<10>(precomp, xC, B, a, sample, x_norm2);
    }
    float term1 = 0.0f;
    float term2 = 0.0f;
    for (int l = 0; l < m; ++l) {
        const int flat_l = precomp.offsets[l] + static_cast<int>(B(l, sample));
        const float alpha_l = a(l, sample);
        term1 += alpha_l * xC(flat_l, sample);
        for (int k = 0; k < m; ++k) {
            const int flat_k = precomp.offsets[k] + static_cast<int>(B(k, sample));
            term2 += alpha_l * a(k, sample) * GAt(precomp.G, flat_l, flat_k);
        }
    }
    return x_norm2 - 2.0f * term1 + term2;
}

void ComputeCosts(const Precomp& precomp,
                  const ColMajorMatrix<float>& xC,
                  const ColMajorMatrix<FullCode>& B,
                  const ColMajorMatrix<float>& a,
                  const std::vector<float>& X_norm2,
                  std::vector<float>* out_cost) {
    if (!out_cost) {
        return;
    }
    const int n = B.cols;
    out_cost->resize(static_cast<std::size_t>(std::max(0, n)));
    if (n <= 0) {
        return;
    }
    #pragma omp parallel for default(none) schedule(static) shared(precomp, xC, B, a, X_norm2, out_cost) firstprivate(n)
    for (int j = 0; j < n; ++j) {
        (*out_cost)[static_cast<std::size_t>(j)] =
            SampleCost(precomp, xC, B, a, j, X_norm2[static_cast<std::size_t>(j)]);
    }
}

float SampleCostLargeRoot(const PrecompLargeRoot& pre,
                          const ColMajorMatrix<float>& X,
                          const ColMajorMatrix<float>& xC_small,
                          const std::vector<std::uint32_t>& cluster_id,
                          const ColMajorMatrix<Code>& B_small,
                          const ColMajorMatrix<float>& a,
                          int sample,
                          float x_norm2) {
    const int m = pre.m;
    if (m == 5) {
        return SampleCostLargeRootFixed<5>(pre, X, xC_small, cluster_id, B_small, a, sample, x_norm2);
    }
    if (m == 10) {
        return SampleCostLargeRootFixed<10>(pre, X, xC_small, cluster_id, B_small, a, sample, x_norm2);
    }
    const int d = pre.d;
    const std::uint32_t cid = cluster_id[static_cast<std::size_t>(sample)];

    // xC0 = dot(C0[:,cid], x)
    const float* x = X.Col(sample);
    const float* c0 = pre.C0.Col(static_cast<int>(cid));
    float xC0 = 0.0f;
    #pragma omp simd reduction(+:xC0)
    for (int r = 0; r < d; ++r) {
        xC0 += c0[r] * x[r];
    }

    float term1 = a(0, sample) * xC0;
    for (int l = 1; l < m; ++l) {
        const int flat_l = pre.small_offsets[l] + static_cast<int>(B_small(l - 1, sample));
        term1 += a(l, sample) * xC_small(flat_l, sample);
    }

    float term2 = 0.0f;
    // root-root
    term2 += a(0, sample) * a(0, sample) * pre.norm0[static_cast<std::size_t>(cid)];
    // root-small + small-small
    for (int j = 1; j < m; ++j) {
        const float aj = a(j, sample);
        const int flat_j = pre.small_offsets[j] + static_cast<int>(B_small(j - 1, sample));
        term2 += 2.0f * a(0, sample) * aj * GAt(pre.G0S, static_cast<int>(cid), flat_j);
        for (int k = 1; k < m; ++k) {
            const float ak = a(k, sample);
            const int flat_k = pre.small_offsets[k] + static_cast<int>(B_small(k - 1, sample));
            term2 += aj * ak * GAt(pre.G_small, flat_j, flat_k);
        }
    }

    return x_norm2 - 2.0f * term1 + term2;
}

void ComputeCostsLargeRoot(const PrecompLargeRoot& pre,
                           const ColMajorMatrix<float>& X,
                           const ColMajorMatrix<float>& xC_small,
                           const std::vector<std::uint32_t>& cluster_id,
                           const ColMajorMatrix<Code>& B_small,
                           const ColMajorMatrix<float>& a,
                           const std::vector<float>& X_norm2,
                           std::vector<float>* out_cost) {
    if (!out_cost) {
        return;
    }
    const int n = B_small.cols;
    out_cost->resize(static_cast<std::size_t>(std::max(0, n)));
    if (n <= 0) {
        return;
    }
    #pragma omp parallel for default(none) schedule(static) shared(pre, X, xC_small, cluster_id, B_small, a, X_norm2, out_cost) firstprivate(n)
    for (int j = 0; j < n; ++j) {
        (*out_cost)[static_cast<std::size_t>(j)] =
            SampleCostLargeRoot(pre, X, xC_small, cluster_id, B_small, a, j,
                                X_norm2[static_cast<std::size_t>(j)]);
    }
}

}  // namespace stlq
