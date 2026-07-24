#include "stlq/eval/query_table_builder.h"

#include "stlq/core/blas.h"
#include "stlq/common/timer.h"

#include <omp.h>

namespace stlq {

STLQueryTables BuildSTLQueryTables(const float* Xq,
                                       int ldXq,
                                       int d,
                                       int qlen,
                                       const ColMajorMatrix<float>& C_root0,
                                       const CodebookMeta& meta_root_small,
                                       const CodebookMeta& meta_one,
                                       int blas_threads,
                                       double* out_coarse_gemm_sec,
                                       double* out_root_small_gemm_sec,
                                       double* out_one_gemm_sec) {
    STLQueryTables out;
    out.xCq_root0 = ColMajorMatrix<float>(C_root0.cols, qlen);
    out.xCq_root_small = ColMajorMatrix<float>(meta_root_small.total_cols, qlen);
    out.xCq_one = ColMajorMatrix<float>(meta_one.total_cols, qlen);

    {
        ScopedBlasThreads scope(blas_threads);
        {
            // xCq_root0 = C_root0^T * Xq
            if (out_coarse_gemm_sec) {
                const double t0 = omp_get_wtime();
                GemmRaw(true, false,
                        /*m=*/C_root0.cols,
                        /*n=*/qlen,
                        /*k=*/d,
                        /*alpha=*/1.0f,
                        /*A=*/C_root0.data.data(),
                        /*lda=*/C_root0.rows,
                        /*B=*/Xq,
                        /*ldb=*/ldXq,
                        /*beta=*/0.0f,
                        /*C=*/out.xCq_root0.data.data(),
                        /*ldc=*/out.xCq_root0.rows);
                *out_coarse_gemm_sec = omp_get_wtime() - t0;
            } else {
                GemmRaw(true, false,
                        /*m=*/C_root0.cols,
                        /*n=*/qlen,
                        /*k=*/d,
                        /*alpha=*/1.0f,
                        /*A=*/C_root0.data.data(),
                        /*lda=*/C_root0.rows,
                        /*B=*/Xq,
                        /*ldb=*/ldXq,
                        /*beta=*/0.0f,
                        /*C=*/out.xCq_root0.data.data(),
                        /*ldc=*/out.xCq_root0.rows);
            }
        }
        {
            // xCq_root_small = meta_root_small.flat^T * Xq
            if (out_root_small_gemm_sec) {
                const double t0 = omp_get_wtime();
                GemmRaw(true, false,
                        /*m=*/meta_root_small.total_cols,
                        /*n=*/qlen,
                        /*k=*/d,
                        /*alpha=*/1.0f,
                        /*A=*/meta_root_small.flat.data.data(),
                        /*lda=*/meta_root_small.flat.rows,
                        /*B=*/Xq,
                        /*ldb=*/ldXq,
                        /*beta=*/0.0f,
                        /*C=*/out.xCq_root_small.data.data(),
                        /*ldc=*/out.xCq_root_small.rows);
                *out_root_small_gemm_sec = omp_get_wtime() - t0;
            } else {
                GemmRaw(true, false,
                        /*m=*/meta_root_small.total_cols,
                        /*n=*/qlen,
                        /*k=*/d,
                        /*alpha=*/1.0f,
                        /*A=*/meta_root_small.flat.data.data(),
                        /*lda=*/meta_root_small.flat.rows,
                        /*B=*/Xq,
                        /*ldb=*/ldXq,
                        /*beta=*/0.0f,
                        /*C=*/out.xCq_root_small.data.data(),
                        /*ldc=*/out.xCq_root_small.rows);
            }

            // xCq_one = meta_one.flat^T * Xq
            if (out_one_gemm_sec) {
                const double t0 = omp_get_wtime();
                GemmRaw(true, false,
                        /*m=*/meta_one.total_cols,
                        /*n=*/qlen,
                        /*k=*/d,
                        /*alpha=*/1.0f,
                        /*A=*/meta_one.flat.data.data(),
                        /*lda=*/meta_one.flat.rows,
                        /*B=*/Xq,
                        /*ldb=*/ldXq,
                        /*beta=*/0.0f,
                        /*C=*/out.xCq_one.data.data(),
                        /*ldc=*/out.xCq_one.rows);
                *out_one_gemm_sec = omp_get_wtime() - t0;
            } else {
                GemmRaw(true, false,
                        /*m=*/meta_one.total_cols,
                        /*n=*/qlen,
                        /*k=*/d,
                        /*alpha=*/1.0f,
                        /*A=*/meta_one.flat.data.data(),
                        /*lda=*/meta_one.flat.rows,
                        /*B=*/Xq,
                        /*ldb=*/ldXq,
                        /*beta=*/0.0f,
                        /*C=*/out.xCq_one.data.data(),
                        /*ldc=*/out.xCq_one.rows);
            }
        }
    }
    return out;
}

STLQueryTables BuildSTLQueryTablesSkipCoarse(const float* Xq,
                                                 int ldXq,
                                                 int d,
                                                 int qlen,
                                                 int h0,
                                                 const CodebookMeta& meta_root_small,
                                                 const CodebookMeta& meta_one,
                                                 int blas_threads,
                                                 double* out_root_small_gemm_sec,
                                                 double* out_one_gemm_sec) {
    STLQueryTables out;
    // Allocate xCq_root0 (h0 x qlen) zero-filled; caller fills entries sparsely.
    out.xCq_root0.rows = h0;
    out.xCq_root0.cols = qlen;
    out.xCq_root0.data.assign(static_cast<std::size_t>(h0) * static_cast<std::size_t>(qlen), 0.0f);

    out.xCq_root_small = ColMajorMatrix<float>(meta_root_small.total_cols, qlen);
    out.xCq_one = ColMajorMatrix<float>(meta_one.total_cols, qlen);

    {
        ScopedBlasThreads scope(blas_threads);
        // xCq_root_small
        if (out_root_small_gemm_sec) {
            const double t0 = omp_get_wtime();
            GemmRaw(true, false,
                /*m=*/meta_root_small.total_cols,
                /*n=*/qlen,
                /*k=*/d,
                /*alpha=*/1.0f,
                /*A=*/meta_root_small.flat.data.data(),
                /*lda=*/meta_root_small.flat.rows,
                /*B=*/Xq,
                /*ldb=*/ldXq,
                /*beta=*/0.0f,
                /*C=*/out.xCq_root_small.data.data(),
                /*ldc=*/out.xCq_root_small.rows);
            *out_root_small_gemm_sec = omp_get_wtime() - t0;
        } else {
            GemmRaw(true, false,
                /*m=*/meta_root_small.total_cols,
                /*n=*/qlen,
                /*k=*/d,
                /*alpha=*/1.0f,
                /*A=*/meta_root_small.flat.data.data(),
                /*lda=*/meta_root_small.flat.rows,
                /*B=*/Xq,
                /*ldb=*/ldXq,
                /*beta=*/0.0f,
                /*C=*/out.xCq_root_small.data.data(),
                /*ldc=*/out.xCq_root_small.rows);
        }

        // xCq_one
        if (out_one_gemm_sec) {
            const double t0 = omp_get_wtime();
            GemmRaw(true, false,
                /*m=*/meta_one.total_cols,
                /*n=*/qlen,
                /*k=*/d,
                /*alpha=*/1.0f,
                /*A=*/meta_one.flat.data.data(),
                /*lda=*/meta_one.flat.rows,
                /*B=*/Xq,
                /*ldb=*/ldXq,
                /*beta=*/0.0f,
                /*C=*/out.xCq_one.data.data(),
                /*ldc=*/out.xCq_one.rows);
            *out_one_gemm_sec = omp_get_wtime() - t0;
        } else {
            GemmRaw(true, false,
                /*m=*/meta_one.total_cols,
                /*n=*/qlen,
                /*k=*/d,
                /*alpha=*/1.0f,
                /*A=*/meta_one.flat.data.data(),
                /*lda=*/meta_one.flat.rows,
                /*B=*/Xq,
                /*ldb=*/ldXq,
                /*beta=*/0.0f,
                /*C=*/out.xCq_one.data.data(),
                /*ldc=*/out.xCq_one.rows);
        }
    }
    return out;
}

}  // namespace stlq
