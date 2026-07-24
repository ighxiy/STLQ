#include "stlq/cuda/cuda_stream_kernels_pool.h"

#include <mutex>
#include <stdexcept>
#include <string>

#include "stlq/common/logger.h"

namespace stlq {

namespace {

inline void ThrowIf(cudaError_t st, const char* what) {
    if (st == cudaSuccess) {
        return;
    }
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(st));
}

inline void ThrowIf(cublasStatus_t st, const char* what) {
    if (st == CUBLAS_STATUS_SUCCESS) {
        return;
    }
    throw std::runtime_error(std::string(what) + ": cublasStatus=" +
                             std::to_string(static_cast<int>(st)));
}

}  // namespace

CudaStreamKernelsPool::CudaStreamKernelsPool(int num_ctx, const CudaPoolConfig& cfg) : cfg_(cfg) {
    if (num_ctx <= 0) {
        throw std::runtime_error("CudaStreamKernelsPool: num_ctx must be > 0.");
    }
    ThrowIf(cudaSetDevice(cfg_.device), "cudaSetDevice");

    ctx_.resize(static_cast<std::size_t>(num_ctx));
    free_.reserve(static_cast<std::size_t>(num_ctx));

    for (int i = 0; i < num_ctx; ++i) {
        CudaCtx& ctx = ctx_[static_cast<std::size_t>(i)];
        ctx.device = cfg_.device;
        ThrowIf(cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking), "cudaStreamCreate");
        ThrowIf(cublasCreate(&ctx.cublas), "cublasCreate");
        ThrowIf(cublasSetStream(ctx.cublas, ctx.stream), "cublasSetStream");
        if (cfg_.allow_tf32) {
            ThrowIf(cublasSetMathMode(ctx.cublas, CUBLAS_TF32_TENSOR_OP_MATH), "cublasSetMathMode(TF32)");
        } else {
            ThrowIf(cublasSetMathMode(ctx.cublas, CUBLAS_DEFAULT_MATH), "cublasSetMathMode(DEFAULT)");
        }
        free_.push_back(i);
    }
    // One-time log: report GEMM math mode for the pool (suppress repeated messages).
    {
        static std::once_flag s_pool_log_once;
        std::call_once(s_pool_log_once, [&]() {
            LogInfo(std::string("CudaStreamKernelsPool: ") + std::to_string(num_ctx) +
                    " ctx(s) on device " + std::to_string(cfg_.device) +
                    ", cuBLAS math_mode=" + (cfg_.allow_tf32 ? "TF32" : "DEFAULT_FP32"));
        });
    }
}

CudaStreamKernelsPool::~CudaStreamKernelsPool() {
    try {
        ThrowIf(cudaSetDevice(cfg_.device), "cudaSetDevice");
        for (CudaCtx& ctx : ctx_) {
            if (ctx.user && ctx.user_deleter) {
                ctx.user_deleter(ctx.user);
                ctx.user = nullptr;
                ctx.user_deleter = nullptr;
            }
            if (ctx.cublas) {
                cublasDestroy(ctx.cublas);
                ctx.cublas = nullptr;
            }
            if (ctx.stream) {
                cudaStreamDestroy(ctx.stream);
                ctx.stream = nullptr;
            }
        }
    } catch (const std::exception& e) {
        LogError(std::string("CudaStreamKernelsPool::~CudaStreamKernelsPool: ") + e.what());
    }
}

CudaCtx* CudaStreamKernelsPool::TryAcquire() {
    std::lock_guard<std::mutex> lock(mu_);
    if (free_.empty()) {
        return nullptr;
    }
    const int idx = free_.back();
    free_.pop_back();
    return &ctx_[static_cast<std::size_t>(idx)];
}

CudaCtx* CudaStreamKernelsPool::Acquire() {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [&]() { return !free_.empty(); });
    const int idx = free_.back();
    free_.pop_back();
    return &ctx_[static_cast<std::size_t>(idx)];
}

void CudaStreamKernelsPool::Release(CudaCtx* ctx) {
    if (!ctx) {
        return;
    }
    const std::ptrdiff_t idx = ctx - ctx_.data();
    if (idx < 0 || idx >= static_cast<std::ptrdiff_t>(ctx_.size())) {
        throw std::runtime_error("CudaStreamKernelsPool::Release: ctx not from this pool.");
    }
    {
        std::lock_guard<std::mutex> lock(mu_);
        free_.push_back(static_cast<int>(idx));
    }
    cv_.notify_one();
}

}  // namespace stlq
