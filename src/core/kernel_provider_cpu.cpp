#include "stlq/core/kernel_provider_cpu.h"

#include <algorithm>
#include <cstdint>

#include <omp.h>

#include "stlq/core/blas.h"
#include "stlq/core/threading.h"

namespace stlq {

void CpuStreamKernels::ConvertU8ToF32(const ColMajorMatrix<std::uint8_t>& in,
                                     ColMajorMatrix<float>* out) {
    const int d = in.rows;
    const int n = in.cols;
    out->rows = d;
    out->cols = n;
    out->data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(n));

    #pragma omp parallel for default(none) if (!omp_in_parallel()) schedule(static) shared(in, out) firstprivate(d, n)
    for (int j = 0; j < n; ++j) {
        const std::uint8_t* src = in.Col(j);
        float* dst = out->Col(j);
        #pragma omp simd
        for (int r = 0; r < d; ++r) {
            dst[r] = static_cast<float>(src[r]);
        }
    }
}

void CpuStreamKernels::ConvertU8ToF32AndRotate(const ColMajorMatrix<std::uint8_t>& in,
                                              const ColMajorMatrix<float>& R,
                                              ColMajorMatrix<float>* out) {
    // tmp_ = float(in)
    ConvertU8ToF32(in, &tmp_);
    // out = R * tmp_
    out->rows = R.rows;
    out->cols = tmp_.cols;
    out->data.resize(static_cast<std::size_t>(out->rows) * static_cast<std::size_t>(out->cols));
    {
        const int nt = omp_in_parallel() ? 1 : OmpMaxThreads();
        ScopedBlasThreads blas_scope(nt);
        Gemm(false, false, 1.0f, R, tmp_, 0.0f, out);
    }
}

void CpuStreamKernels::ConvertU8ToF32AndRotatePtr(const std::uint8_t* in,
                                                  int ld_in,
                                                  int rows,
                                                  int cols,
                                                  const ColMajorMatrix<float>& R,
                                                  ColMajorMatrix<float>* out) {
    if (!out) return;
    if (!in || rows <= 0 || cols <= 0) {
        out->rows = 0;
        out->cols = 0;
        out->data.clear();
        return;
    }
    if (R.cols != rows) {
        throw std::runtime_error("CpuStreamKernels::ConvertU8ToF32AndRotatePtr: dim mismatch (R.cols != rows).");
    }

    // tmp_ = float(in)   (column-major)
    tmp_.rows = rows;
    tmp_.cols = cols;
    tmp_.data.resize(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
    #pragma omp parallel for default(none) if (!omp_in_parallel()) schedule(static) shared(in, out) firstprivate(ld_in, rows, cols)
    for (int j = 0; j < cols; ++j) {
        const std::uint8_t* src = in + static_cast<std::size_t>(j) * static_cast<std::size_t>(ld_in);
        float* dst = tmp_.Col(j);
        #pragma omp simd
        for (int r = 0; r < rows; ++r) {
            dst[r] = static_cast<float>(src[r]);
        }
    }

    // out = R * tmp_
    out->rows = R.rows;
    out->cols = cols;
    out->data.resize(static_cast<std::size_t>(out->rows) * static_cast<std::size_t>(out->cols));
    {
        const int nt = omp_in_parallel() ? 1 : OmpMaxThreads();
        ScopedBlasThreads blas_scope(nt);
        Gemm(false, false, 1.0f, R, tmp_, 0.0f, out);
    }
}

void CpuStreamKernels::Gemm(bool transA, bool transB,
                           float alpha,
                           const ColMajorMatrix<float>& A,
                           const ColMajorMatrix<float>& B,
                           float beta,
                           ColMajorMatrix<float>* C) {
    const int nt = omp_in_parallel() ? 1 : OmpMaxThreads();
    ScopedBlasThreads blas_scope(nt);
    stlq::Gemm(transA, transB, alpha, A, B, beta, C);
}

}  // namespace stlq
