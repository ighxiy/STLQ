#include "stlq/core/kernels_cpu.h"

#include "stlq/core/blas.h"
#include "stlq/common/logger.h"

namespace stlq {

void CpuKernels::GemmXc(const ColMajorMatrix<float>& C_all,
                        const ColMajorMatrix<float>& X,
                        ColMajorMatrix<float>* xC) {
    Gemm(true, false, 1.0f, C_all, X, 0.0f, xC);
}

void CpuKernels::UpdateRc(const ColMajorMatrix<float>& xC,
                          const ColMajorMatrix<float>& G,
                          const ColMajorMatrix<Index>& B,
                          const ColMajorMatrix<float>& a,
                          const std::vector<int>& offsets,
                          ColMajorMatrix<float>* rC) {
    const int H = xC.rows;
    const int n = xC.cols;
    const int m = B.rows;

    rC->rows = H;
    rC->cols = n;
    rC->data.resize(static_cast<std::size_t>(H) * n);

    for (int j = 0; j < n; ++j) {
        const float* xC_col = xC.Col(j);
        float* rC_col = rC->Col(j);
        for (int p = 0; p < H; ++p) {
            rC_col[p] = xC_col[p];
        }
        for (int l = 0; l < m; ++l) {
            const int offset = offsets[l];
            const Index code = B(l, j);
            const int col = offset + static_cast<int>(code);
            const float alpha = a(l, j);
            const float* G_col = G.Col(col);
            for (int p = 0; p < H; ++p) {
                rC_col[p] -= alpha * G_col[p];
            }
        }
    }
}

void CpuKernels::SolveLeastSquares(const ColMajorMatrix<float>& X,
                                   const ColMajorMatrix<float>& C_all,
                                   const ColMajorMatrix<Index>& B,
                                   const std::vector<int>& offsets,
                                   ColMajorMatrix<float>* a) {
    (void)X;
    (void)C_all;
    (void)B;
    (void)offsets;
    if (!a) {
        return;
    }
    std::fill(a->data.begin(), a->data.end(), 0.0f);
    LogWarn("SolveLeastSquares is a stub in CpuKernels.");
}

void CpuKernels::IcmStep(const ColMajorMatrix<float>& X,
                         const ColMajorMatrix<float>& C_all,
                         const ColMajorMatrix<float>& G,
                         const std::vector<int>& offsets,
                         ColMajorMatrix<Index>* B,
                         ColMajorMatrix<float>* a) {
    (void)X;
    (void)C_all;
    (void)G;
    (void)offsets;
    (void)B;
    (void)a;
    LogWarn("IcmStep is a stub in CpuKernels.");
}

}  // namespace stlq
