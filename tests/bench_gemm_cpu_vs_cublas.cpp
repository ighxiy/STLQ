#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "stlq/core/blas.h"
#include "stlq/core/threading.h"
#include "stlq/common/logger.h"
#include "stlq/common/timer.h"
#include "stlq/common/types.h"

#if defined(STLQ_ENABLE_CUDA)
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

namespace {

struct Args {
    int m = 65536;   // rows of C
    int n = 64;      // cols of C
    int k = 96;      // shared dim
    int warmup = 5;
    int repeat = 20;
    int blas_threads = 1;
    int cuda_device = 0;
    bool run_cpu = true;
    bool run_gpu_fp32 = true;
    bool run_gpu_tf32 = true;
    bool check = false;
};

bool ParseIntArg(const std::string& s, const std::string& key, int* out) {
    if (s.rfind(key + "=", 0) != 0) return false;
    *out = std::atoi(s.c_str() + key.size() + 1);
    return true;
}

Args ParseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s(argv[i]);
        if (ParseIntArg(s, "--m", &a.m)) continue;
        if (ParseIntArg(s, "--n", &a.n)) continue;
        if (ParseIntArg(s, "--k", &a.k)) continue;
        if (ParseIntArg(s, "--warmup", &a.warmup)) continue;
        if (ParseIntArg(s, "--repeat", &a.repeat)) continue;
        if (ParseIntArg(s, "--blas_threads", &a.blas_threads)) continue;
        if (ParseIntArg(s, "--cuda_device", &a.cuda_device)) continue;
        if (s == "--no_cpu") {
            a.run_cpu = false;
            continue;
        }
        if (s == "--no_gpu_fp32") {
            a.run_gpu_fp32 = false;
            continue;
        }
        if (s == "--no_gpu_tf32") {
            a.run_gpu_tf32 = false;
            continue;
        }
        if (s == "--check") {
            a.check = true;
            continue;
        }
        if (s == "--help" || s == "-h") {
            std::cout
                << "bench_gemm_cpu_vs_cublas\n"
                << "  CPU: cblas_sgemm (MKL/OpenBLAS)\n"
                << "  GPU: cuBLAS sgemm (FP32) and GemmEx (TF32)\n\n"
                << "Args:\n"
                << "  --m=65536 --n=64 --k=96\n"
                << "  --warmup=5 --repeat=20\n"
                << "  --blas_threads=1\n"
                << "  --cuda_device=0\n"
                << "  --no_cpu --no_gpu_fp32 --no_gpu_tf32\n"
                << "  --check (compare CPU vs GPU outputs; TF32 will differ)\n";
            std::exit(0);
        }
    }
    return a;
}

double GflopsSgemm(int m, int n, int k) {
    // GEMM flops ~= 2*m*n*k
    return (2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k)) / 1e9;
}

double MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    const std::size_t n = std::min(a.size(), b.size());
    double mx = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        mx = std::max(mx, static_cast<double>(std::abs(a[i] - b[i])));
    }
    return mx;
}

#if defined(STLQ_ENABLE_CUDA)
inline void ThrowIf(cudaError_t st, const char* what) {
    if (st == cudaSuccess) return;
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(st));
}
inline void ThrowIf(cublasStatus_t st, const char* what) {
    if (st == CUBLAS_STATUS_SUCCESS) return;
    throw std::runtime_error(std::string(what) + ": cublasStatus=" + std::to_string(static_cast<int>(st)));
}
#endif

}  // namespace

int main(int argc, char** argv) {
    const Args args = ParseArgs(argc, argv);
    const int m = std::max(1, args.m);
    const int n = std::max(1, args.n);
    const int k = std::max(1, args.k);
    const int warmup = std::max(0, args.warmup);
    const int repeat = std::max(1, args.repeat);

    stlq::LogInfo("===== GEMM bench (C = A^T * B) =====");
    stlq::LogInfo("Shape: m=" + std::to_string(m) + " n=" + std::to_string(n) + " k=" + std::to_string(k) +
                    " (A: k×m, B: k×n, C: m×n; col-major)");
    stlq::LogInfo("Warmup=" + std::to_string(warmup) + " Repeat=" + std::to_string(repeat));
    stlq::LogInfo(std::string("BLAS backend: ") +
#if defined(STLQ_USE_MKL)
                    "MKL"
#elif defined(STLQ_USE_OPENBLAS)
                    "OpenBLAS"
#else
                    "cblas (unknown vendor)"
#endif
    );

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Column-major A (k×m), B (k×n), C (m×n)
    std::vector<float> A(static_cast<std::size_t>(k) * static_cast<std::size_t>(m));
    std::vector<float> B(static_cast<std::size_t>(k) * static_cast<std::size_t>(n));
    std::vector<float> C_cpu(static_cast<std::size_t>(m) * static_cast<std::size_t>(n), 0.0f);
    for (float& v : A) v = dist(rng);
    for (float& v : B) v = dist(rng);

    const double gflop = GflopsSgemm(m, n, k);

    if (args.run_cpu) {
        stlq::ScopedBlasThreads blas_scope(std::max(1, args.blas_threads));
        stlq::Timer t;
        for (int i = 0; i < warmup; ++i) {
            stlq::GemmRaw(/*trans_a=*/true,
                            /*trans_b=*/false,
                            m, n, k,
                            1.0f,
                            A.data(), /*lda=*/k,
                            B.data(), /*ldb=*/k,
                            0.0f,
                            C_cpu.data(), /*ldc=*/m);
        }
        const stlq::Timer tr;
        for (int i = 0; i < repeat; ++i) {
            stlq::GemmRaw(/*trans_a=*/true,
                            /*trans_b=*/false,
                            m, n, k,
                            1.0f,
                            A.data(), /*lda=*/k,
                            B.data(), /*ldb=*/k,
                            0.0f,
                            C_cpu.data(), /*ldc=*/m);
        }
        const double sec = tr.ElapsedSeconds();
        const double gflops = (sec > 0.0) ? (gflop * static_cast<double>(repeat) / sec) : 0.0;
        stlq::LogInfo("CPU GEMM: sec=" + std::to_string(sec) +
                        " avg_ms=" + std::to_string(1000.0 * sec / static_cast<double>(repeat)) +
                        " GF/s=" + std::to_string(gflops) +
                        " (blas_threads=" + std::to_string(std::max(1, args.blas_threads)) + ")");
        (void)t;
    }

#if defined(STLQ_ENABLE_CUDA)
    auto run_gpu = [&](bool tf32, std::vector<float>* out_C) {
        ThrowIf(cudaSetDevice(args.cuda_device), "cudaSetDevice");
        // Ensure context init cost is not included.
        ThrowIf(cudaFree(nullptr), "cudaFree(nullptr)");

        cublasHandle_t handle = nullptr;
        cudaStream_t stream = nullptr;
        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        try {
            ThrowIf(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
            ThrowIf(cublasCreate(&handle), "cublasCreate");
            ThrowIf(cublasSetStream(handle, stream), "cublasSetStream");
            ThrowIf(cublasSetMathMode(handle, tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH),
                    "cublasSetMathMode");

            const std::size_t bytesA = sizeof(float) * static_cast<std::size_t>(k) * static_cast<std::size_t>(m);
            const std::size_t bytesB = sizeof(float) * static_cast<std::size_t>(k) * static_cast<std::size_t>(n);
            const std::size_t bytesC = sizeof(float) * static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
            ThrowIf(cudaMalloc(&dA, bytesA), "cudaMalloc(A)");
            ThrowIf(cudaMalloc(&dB, bytesB), "cudaMalloc(B)");
            ThrowIf(cudaMalloc(&dC, bytesC), "cudaMalloc(C)");
            ThrowIf(cudaMemcpyAsync(dA, A.data(), bytesA, cudaMemcpyHostToDevice, stream), "h2d(A)");
            ThrowIf(cudaMemcpyAsync(dB, B.data(), bytesB, cudaMemcpyHostToDevice, stream), "h2d(B)");
            ThrowIf(cudaMemsetAsync(dC, 0, bytesC, stream), "memset(C)");
            ThrowIf(cudaStreamSynchronize(stream), "sync(h2d)");

            const float alpha = 1.0f;
            const float beta = 0.0f;

            auto gemm_once = [&]() {
                if (tf32) {
                    // C = A^T * B (all FP32 inputs/outputs; TF32 compute).
                    ThrowIf(cublasGemmEx(handle,
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_N,
                                         m, n, k,
                                         &alpha,
                                         dA, CUDA_R_32F, /*lda=*/k,
                                         dB, CUDA_R_32F, /*ldb=*/k,
                                         &beta,
                                         dC, CUDA_R_32F, /*ldc=*/m,
                                         CUBLAS_COMPUTE_32F_FAST_TF32,
                                         CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                            "cublasGemmEx(TF32)");
                } else {
                    ThrowIf(cublasSgemm(handle,
                                        CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        m, n, k,
                                        &alpha,
                                        dA, /*lda=*/k,
                                        dB, /*ldb=*/k,
                                        &beta,
                                        dC, /*ldc=*/m),
                            "cublasSgemm");
                }
            };

            for (int i = 0; i < warmup; ++i) {
                gemm_once();
            }
            ThrowIf(cudaStreamSynchronize(stream), "sync(warmup)");

            cudaEvent_t ev0 = nullptr, ev1 = nullptr;
            ThrowIf(cudaEventCreate(&ev0), "cudaEventCreate(ev0)");
            ThrowIf(cudaEventCreate(&ev1), "cudaEventCreate(ev1)");

            // Wall time (includes launches + sync behavior in this stream).
            const stlq::Timer wall;
            // Kernel time via CUDA events.
            float ms_sum = 0.0f;
            for (int i = 0; i < repeat; ++i) {
                ThrowIf(cudaEventRecord(ev0, stream), "cudaEventRecord(ev0)");
                gemm_once();
                ThrowIf(cudaEventRecord(ev1, stream), "cudaEventRecord(ev1)");
                ThrowIf(cudaEventSynchronize(ev1), "cudaEventSynchronize(ev1)");
                float ms = 0.0f;
                ThrowIf(cudaEventElapsedTime(&ms, ev0, ev1), "cudaEventElapsedTime");
                ms_sum += ms;
            }
            ThrowIf(cudaStreamSynchronize(stream), "sync(repeat)");
            const double wall_sec = wall.ElapsedSeconds();

            const double event_sec = static_cast<double>(ms_sum) / 1000.0;
            const double gflops_wall = (wall_sec > 0.0) ? (gflop * static_cast<double>(repeat) / wall_sec) : 0.0;
            const double gflops_evt = (event_sec > 0.0) ? (gflop * static_cast<double>(repeat) / event_sec) : 0.0;
            stlq::LogInfo(std::string("GPU GEMM ") + (tf32 ? "TF32" : "FP32") +
                            ": wall_sec=" + std::to_string(wall_sec) +
                            " avg_wall_ms=" + std::to_string(1000.0 * wall_sec / static_cast<double>(repeat)) +
                            " wall_GF/s=" + std::to_string(gflops_wall) +
                            " event_ms=" + std::to_string(ms_sum) +
                            " event_GF/s=" + std::to_string(gflops_evt) +
                            " (cuda_device=" + std::to_string(args.cuda_device) + ")");

            if (out_C) {
                out_C->assign(static_cast<std::size_t>(m) * static_cast<std::size_t>(n), 0.0f);
                ThrowIf(cudaMemcpyAsync(out_C->data(), dC, bytesC, cudaMemcpyDeviceToHost, stream), "d2h(C)");
                ThrowIf(cudaStreamSynchronize(stream), "sync(d2h)");
            }

            ThrowIf(cudaEventDestroy(ev0), "cudaEventDestroy(ev0)");
            ThrowIf(cudaEventDestroy(ev1), "cudaEventDestroy(ev1)");
            ThrowIf(cublasDestroy(handle), "cublasDestroy");
            ThrowIf(cudaStreamDestroy(stream), "cudaStreamDestroy");
            ThrowIf(cudaFree(dA), "cudaFree(A)");
            ThrowIf(cudaFree(dB), "cudaFree(B)");
            ThrowIf(cudaFree(dC), "cudaFree(C)");
        } catch (...) {
            if (dA) cudaFree(dA);
            if (dB) cudaFree(dB);
            if (dC) cudaFree(dC);
            if (handle) cublasDestroy(handle);
            if (stream) cudaStreamDestroy(stream);
            throw;
        }
    };

    std::vector<float> C_gpu_fp32;
    std::vector<float> C_gpu_tf32;

    if (args.run_gpu_fp32) {
        run_gpu(/*tf32=*/false, args.check ? &C_gpu_fp32 : nullptr);
    }
    if (args.run_gpu_tf32) {
        run_gpu(/*tf32=*/true, args.check ? &C_gpu_tf32 : nullptr);
    }

    if (args.check && args.run_cpu) {
        if (!C_gpu_fp32.empty()) {
            stlq::LogInfo("Check CPU vs GPU(FP32): max_abs_diff=" +
                            std::to_string(MaxAbsDiff(C_cpu, C_gpu_fp32)));
        }
        if (!C_gpu_tf32.empty()) {
            stlq::LogInfo("Check CPU vs GPU(TF32): max_abs_diff=" +
                            std::to_string(MaxAbsDiff(C_cpu, C_gpu_tf32)) +
                            " (TF32 expected to differ)");
        }
    }
#else
    if (args.run_gpu_fp32 || args.run_gpu_tf32) {
        stlq::LogWarn("STLQ_ENABLE_CUDA=0: GPU bench disabled at build time.");
    }
#endif

    return 0;
}

