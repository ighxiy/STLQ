#include "stlq/quantizer/reconstruct.h"

#include <algorithm>

#include <omp.h>

namespace stlq {

void ReconstructAllInto(const std::vector<ColMajorMatrix<float>>& C,
                        const ColMajorMatrix<FullCode>& B,
                        const ColMajorMatrix<float>& a,
                        ColMajorMatrix<float>* out) {
    if (!out) {
        return;
    }
    const int d = C.empty() ? 0 : C.front().rows;
    const int n = B.cols;
    const int m = static_cast<int>(C.size());
    out->rows = d;
    out->cols = n;
    const std::size_t need = static_cast<std::size_t>(d) * static_cast<std::size_t>(n);
    if (out->data.size() < need) {
        out->data.resize(need);
    }
    if (need > 0) {
        std::fill_n(out->data.data(), need, 0.0f);
    }

    #pragma omp parallel for default(none) if (!omp_in_parallel()) schedule(static) shared(C, B, a, out) firstprivate(d, n, m)
    for (int i = 0; i < n; ++i) {
        float* dst = out->Col(i);
        for (int l = 0; l < m; ++l) {
            const int code = static_cast<int>(B(l, i));
            const float coeff = a(l, i);
            const float* center = C[static_cast<std::size_t>(l)].Col(code);
            #pragma omp simd
            for (int r = 0; r < d; ++r) {
                dst[r] += coeff * center[r];
            }
        }
    }
}

ColMajorMatrix<float> ReconstructAll(const std::vector<ColMajorMatrix<float>>& C,
                                     const ColMajorMatrix<FullCode>& B,
                                     const ColMajorMatrix<float>& a) {
    ColMajorMatrix<float> out;
    ReconstructAllInto(C, B, a, &out);
    return out;
}

}  // namespace stlq
