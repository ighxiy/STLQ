#pragma once

#include <algorithm>
#include <vector>

#if defined(STLQ_USE_MKL)
  #include <mkl_lapacke.h>
#elif defined(STLQ_HAS_LAPACKE)
  #include <lapacke.h>
#endif

namespace stlq::lapack {

// We only need a tiny subset of LAPACK:
// - potrf/potrs for small SPD solves (LS coefficients)
// - gesvd for OPQ/Procrustes rotation update
//
// Backend policy:
// - Prefer LAPACKE (C interface) when available (MKL or system lapacke).
// - Otherwise fall back to calling LAPACK Fortran symbols directly.

#if defined(STLQ_USE_MKL) || defined(STLQ_HAS_LAPACKE)
inline int SpotrfU(int n, float* A, int lda) {
    return static_cast<int>(LAPACKE_spotrf(LAPACK_COL_MAJOR, 'U', n, A, lda));
}

inline int SpotrsU(int n, int nrhs, float* A, int lda, float* B, int ldb) {
    return static_cast<int>(LAPACKE_spotrs(LAPACK_COL_MAJOR, 'U', n, nrhs, A, lda, B, ldb));
}

inline int DpotrfU(int n, double* A, int lda) {
    return static_cast<int>(LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', n, A, lda));
}

inline int DpotrsU(int n, int nrhs, double* A, int lda, double* B, int ldb) {
    return static_cast<int>(LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'U', n, nrhs, A, lda, B, ldb));
}

inline int Sgesvd(char jobu,
                  char jobvt,
                  int m,
                  int n,
                  float* A,
                  int lda,
                  float* S,
                  float* U,
                  int ldu,
                  float* VT,
                  int ldvt) {
    std::vector<float> superb(static_cast<std::size_t>(std::max(1, std::min(m, n) - 1)), 0.0f);
    return static_cast<int>(LAPACKE_sgesvd(LAPACK_COL_MAJOR, jobu, jobvt, m, n,
                                          A, lda, S, U, ldu, VT, ldvt, superb.data()));
}

#else

extern "C" {
void spotrf_(const char* uplo, const int* n, float* a, const int* lda, int* info);
void spotrs_(const char* uplo, const int* n, const int* nrhs, const float* a, const int* lda,
             float* b, const int* ldb, int* info);
void dpotrf_(const char* uplo, const int* n, double* a, const int* lda, int* info);
void dpotrs_(const char* uplo, const int* n, const int* nrhs, const double* a, const int* lda,
             double* b, const int* ldb, int* info);
void sgesvd_(const char* jobu, const char* jobvt, const int* m, const int* n,
             float* a, const int* lda, float* s, float* u, const int* ldu,
             float* vt, const int* ldvt, float* work, const int* lwork, int* info);
}

inline int SpotrfU(int n, float* A, int lda) {
    const char uplo = 'U';
    int info = 0;
    spotrf_(&uplo, &n, A, &lda, &info);
    return info;
}

inline int SpotrsU(int n, int nrhs, float* A, int lda, float* B, int ldb) {
    const char uplo = 'U';
    int info = 0;
    spotrs_(&uplo, &n, &nrhs, A, &lda, B, &ldb, &info);
    return info;
}

inline int DpotrfU(int n, double* A, int lda) {
    const char uplo = 'U';
    int info = 0;
    dpotrf_(&uplo, &n, A, &lda, &info);
    return info;
}

inline int DpotrsU(int n, int nrhs, double* A, int lda, double* B, int ldb) {
    const char uplo = 'U';
    int info = 0;
    dpotrs_(&uplo, &n, &nrhs, A, &lda, B, &ldb, &info);
    return info;
}

inline int Sgesvd(char jobu,
                  char jobvt,
                  int m,
                  int n,
                  float* A,
                  int lda,
                  float* S,
                  float* U,
                  int ldu,
                  float* VT,
                  int ldvt) {
    int info = 0;

    int lwork = -1;
    float work_query = 0.0f;
    sgesvd_(&jobu, &jobvt, &m, &n, A, &lda, S, U, &ldu, VT, &ldvt, &work_query, &lwork, &info);
    if (info != 0) {
        return info;
    }

    lwork = std::max(1, static_cast<int>(work_query));
    std::vector<float> work(static_cast<std::size_t>(lwork), 0.0f);
    sgesvd_(&jobu, &jobvt, &m, &n, A, &lda, S, U, &ldu, VT, &ldvt, work.data(), &lwork, &info);
    return info;
}

#endif

}  // namespace stlq::lapack
