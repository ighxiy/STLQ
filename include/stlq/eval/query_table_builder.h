#pragma once

#include "stlq/core/threading.h"
#include "stlq/quantizer/codebook_meta.h"
#include "stlq/common/types.h"

namespace stlq {

struct STLQueryTables {
    ColMajorMatrix<float> xCq_root0;       // h0 × qlen (h0=nlist)
    ColMajorMatrix<float> xCq_root_small;  // Hrs × qlen (layers 1..m-1)
    ColMajorMatrix<float> xCq_one;         // Ho × qlen (layers 0..m-1)
};

// Builds xCq tables for one query block.
// Threading policy:
// - This function may call BLAS and is intended to run outside any OpenMP parallel region.
// - `blas_threads` controls BLAS threads (e.g., OmpMaxThreads()).
//
// This overload avoids copying by taking a raw pointer to a contiguous col-major
// block of queries (d×qlen) with leading dimension `ldXq` (typically ldXq=d).
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
                                       double* out_one_gemm_sec);

// Builds only the root_small and one table GEMMs; xCq_root0 is allocated (h0 x qlen) but
// zero-filled.  The caller is expected to fill the needed entries sparsely (e.g. after hier2
// cluster selection).  This avoids the O(h0 * d * qlen) coarse GEMM when h0 is very large.
STLQueryTables BuildSTLQueryTablesSkipCoarse(const float* Xq,
                                                 int ldXq,
                                                 int d,
                                                 int qlen,
                                                 int h0,
                                                 const CodebookMeta& meta_root_small,
                                                 const CodebookMeta& meta_one,
                                                 int blas_threads,
                                                 double* out_root_small_gemm_sec,
                                                 double* out_one_gemm_sec);

}  // namespace stlq
