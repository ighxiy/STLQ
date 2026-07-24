#include "stlq/core/blas.h"

#if defined(STLQ_USE_MKL)
#include <mkl.h>
#else
#include <cblas.h>
#endif

namespace stlq {

namespace {
int g_last_threads = -1;

#if defined(STLQ_USE_OPENBLAS)
extern "C" void openblas_set_num_threads(int);
#endif
}  // namespace

void BlasSetThreads(int nthreads) {
    if (nthreads == g_last_threads) {
        return;
    }
    g_last_threads = nthreads;
#if defined(STLQ_USE_MKL)
    mkl_set_num_threads(nthreads);
#elif defined(STLQ_USE_OPENBLAS)
    openblas_set_num_threads(nthreads);
#else
    (void)nthreads;  // Fallback: relies on env vars (e.g. OPENBLAS_NUM_THREADS)
#endif
}

void Gemm(bool trans_a,
          bool trans_b,
          float alpha,
          const ColMajorMatrix<float>& A,
          const ColMajorMatrix<float>& B,
          float beta,
          ColMajorMatrix<float>* C) {
    const CBLAS_TRANSPOSE ta = trans_a ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE tb = trans_b ? CblasTrans : CblasNoTrans;

    const int m = trans_a ? A.cols : A.rows;
    const int k = trans_a ? A.rows : A.cols;
    const int n = trans_b ? B.rows : B.cols;

    cblas_sgemm(CblasColMajor, ta, tb,
                m, n, k,
                alpha,
                A.data.data(), A.rows,
                B.data.data(), B.rows,
                beta,
                C->data.data(), C->rows);
}

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
             int ldc) {
    const CBLAS_TRANSPOSE ta = trans_a ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE tb = trans_b ? CblasTrans : CblasNoTrans;
    cblas_sgemm(CblasColMajor, ta, tb,
                m, n, k,
                alpha,
                A, lda,
                B, ldb,
                beta,
                C, ldc);
}

}  // namespace stlq
