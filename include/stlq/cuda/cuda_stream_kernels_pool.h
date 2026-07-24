#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <vector>

#if defined(STLQ_ENABLE_CUDA)
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

namespace stlq {

struct CudaPoolConfig {
    int device = 0;
    bool allow_tf32 = false;
};

struct CudaCtx {
#if defined(STLQ_ENABLE_CUDA)
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas = nullptr;
#else
    void* stream = nullptr;
    void* cublas = nullptr;
#endif
    int device = 0;
    void* user = nullptr;
    void (*user_deleter)(void*) = nullptr;
};

class CudaStreamKernelsPool {
public:
    CudaStreamKernelsPool(int num_ctx, const CudaPoolConfig& cfg);
    ~CudaStreamKernelsPool();

    CudaStreamKernelsPool(const CudaStreamKernelsPool&) = delete;
    CudaStreamKernelsPool& operator=(const CudaStreamKernelsPool&) = delete;

    // Try to acquire a context without blocking.
    // Returns nullptr if the pool is exhausted.
    CudaCtx* TryAcquire();

    CudaCtx* Acquire();
    void Release(CudaCtx* ctx);

    int size() const { return static_cast<int>(ctx_.size()); }

private:
    CudaPoolConfig cfg_;
    std::vector<CudaCtx> ctx_;
    std::vector<int> free_;
    std::mutex mu_;
    std::condition_variable cv_;
};

}  // namespace stlq
