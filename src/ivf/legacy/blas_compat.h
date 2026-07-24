#pragma once

#if defined(STLQ_USE_MKL)
#include <mkl.h>
#include <mkl_cblas.h>
inline void SetBlasThreads(int nthreads) { mkl_set_num_threads(nthreads); }
#else
#include <cblas.h>
inline void SetBlasThreads(int) {}
#endif
