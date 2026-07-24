#pragma once

#include <algorithm>
#include <atomic>

#include <omp.h>

#include "stlq/core/blas.h"

namespace stlq {

// OpenMP "default" max threads for this process (best-effort).
//
// Rationale:
// - This project occasionally runs OpenMP-parallel kernels inside helper std::threads (e.g. hybrid pipelines).
// - On some OpenMP runtimes (notably MSVC vcomp), calling `omp_set_num_threads()` from a worker thread can
//   affect the process-wide setting. In those cases, restoring via `omp_get_max_threads()` inside the worker
//   may observe an incorrect value (often 1), causing later phases to run effectively single-threaded.
// - We store the intended "default" thread count once (typically from main after parsing config) and use it
//   as a fallback restore target when a worker thread observes `prev_=1` while the process default is >1.
inline std::atomic<int>& OmpDefaultThreadsVar() {
    static std::atomic<int> v{0};
    return v;
}

inline void SetOmpDefaultThreads(int nthreads) {
    OmpDefaultThreadsVar().store(std::max(1, nthreads), std::memory_order_relaxed);
}

inline int GetOmpDefaultThreads() {
    const int v = OmpDefaultThreadsVar().load(std::memory_order_relaxed);
    return (v > 0) ? v : std::max(1, omp_get_max_threads());
}

// RAII helper: temporarily set BLAS threads (and restore to 1).
// This matches the project's constraint that BLAS must be single-threaded inside outer OpenMP regions.
class ScopedBlasThreads {
public:
    explicit ScopedBlasThreads(int nthreads) : active_(false) {
        if (nthreads > 1) {
            BlasSetThreads(nthreads);
            active_ = true;
        }
    }
    ScopedBlasThreads(const ScopedBlasThreads&) = delete;
    ScopedBlasThreads& operator=(const ScopedBlasThreads&) = delete;
    ~ScopedBlasThreads() {
        if (active_) {
            BlasSetThreads(1);
        }
    }

private:
    bool active_;
};

// RAII helper: temporarily set OpenMP max threads and restore the previous value.
// Useful for making small helper routines deterministic (e.g. k-means on codebook centers).
class ScopedOmpThreads {
public:
    explicit ScopedOmpThreads(int nthreads) : prev_(omp_get_max_threads()), active_(false) {
        // Heuristic fix for MSVC vcomp: if a worker thread observes prev_=1 while the process default is >1,
        // restore to the process default instead of 1 to avoid accidentally "locking" later phases to 1 thread.
        const int def = GetOmpDefaultThreads();
        if (prev_ == 1 && def > 1) {
            prev_ = def;
        }
        const int n = std::max(1, nthreads);
        if (n != prev_) {
            omp_set_num_threads(n);
            active_ = true;
        }
    }
    ScopedOmpThreads(const ScopedOmpThreads&) = delete;
    ScopedOmpThreads& operator=(const ScopedOmpThreads&) = delete;
    ~ScopedOmpThreads() {
        if (active_) {
            omp_set_num_threads(prev_);
        }
    }

private:
    int prev_;
    bool active_;
};

inline int OmpMaxThreads() { return std::max(1, omp_get_max_threads()); }

}  // namespace stlq
