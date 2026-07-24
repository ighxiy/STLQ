#pragma once

#include "stlq/common/types.h"

namespace stlq {

void BlasSetThreads(int nthreads);

void Gemm(bool trans_a,
          bool trans_b,
          float alpha,
          const ColMajorMatrix<float>& A,
          const ColMajorMatrix<float>& B,
          float beta,
          ColMajorMatrix<float>* C);

// Raw-pointer GEMM for submatrix views (column-major).
// Computes C = alpha * op(A) * op(B) + beta * C.
void GemmRaw(bool trans_a,
             bool trans_b,
             int m,
             int n,
             int k,
             float alpha,
             const float* A,
             int lda,
             const float* B,
             int ldb,
             float beta,
             float* C,
             int ldc);

}  // namespace stlq
