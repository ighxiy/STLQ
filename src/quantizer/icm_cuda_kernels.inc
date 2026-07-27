#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#include <intrin.h>
#endif

namespace {

constexpr float kEps = 1e-6f;
constexpr int kMaxM = kMaxSupportedModelM;

inline void CudaCheck(cudaError_t st, const char* msg) {
    if (st == cudaSuccess) {
        return;
    }
    throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(st));
}

template <typename T>
struct DeviceBuf {
    T* ptr = nullptr;
    std::size_t cap = 0;

    ~DeviceBuf() { Reset(); }

    void Reset() {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
            cap = 0;
        }
    }

    void Ensure(std::size_t n) {
        if (n <= cap) {
            return;
        }
        Reset();
        const cudaError_t st = cudaMalloc(&ptr, sizeof(T) * n);
        CudaCheck(st, "cudaMalloc");
        cap = ptr ? n : 0;
    }
};

struct CudaEventPair {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    bool ready = false;

    ~CudaEventPair() {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
    }

    void Ensure() {
        if (ready) return;
        CudaCheck(cudaEventCreate(&start), "cudaEventCreate(start)");
        CudaCheck(cudaEventCreate(&stop), "cudaEventCreate(stop)");
        ready = true;
    }

    template <typename F>
    double MeasureSeconds(F&& fn) {
        Ensure();
        CudaCheck(cudaEventRecord(start, 0), "cudaEventRecord(start)");
        fn();
        CudaCheck(cudaEventRecord(stop, 0), "cudaEventRecord(stop)");
        CudaCheck(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
        float ms = 0.0f;
        CudaCheck(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
        return static_cast<double>(ms) * 1e-3;
    }
};

__device__ __forceinline__ int UpkIndex(int r, int c) {
    // packed upper triangle (r<=c)
    return r + (c * (c + 1)) / 2;
}

__device__ __forceinline__ float UpkSymAt(const float* upk, int r, int c) {
    // Symmetric access for packed upper triangle.
    return (r <= c) ? upk[UpkIndex(r, c)] : upk[UpkIndex(c, r)];
}

__device__ __forceinline__ std::uint32_t SplitMixNextU32(std::uint64_t* state) {
    *state += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = *state;
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    z = z ^ (z >> 31U);
    return static_cast<std::uint32_t>(z);
}

__device__ __forceinline__ std::uint64_t SplitMixNextU64(std::uint64_t* state) {
    *state += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = *state;
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
}

__device__ __forceinline__ std::uint32_t UniformBelow64(std::uint64_t* state, std::uint32_t bound) {
    // Match CPU's Lemire multiply-high mapping: floor(x * bound / 2^64).
    const std::uint64_t x = SplitMixNextU64(state);
    return static_cast<std::uint32_t>(__umul64hi(x, static_cast<std::uint64_t>(bound)));
}

template <int MaxM>
__device__ bool CholUpperPacked(float* A_upk, int m) {
    // A_upk stores upper triangle, overwritten by U where A = U^T U.
    for (int c = 0; c < m; ++c) {
        for (int r = 0; r <= c; ++r) {
            float sum = A_upk[UpkIndex(r, c)];
            for (int k = 0; k < r; ++k) {
                sum -= A_upk[UpkIndex(k, r)] * A_upk[UpkIndex(k, c)];
            }
            if (r == c) {
                if (!(sum > 0.0f)) {
                    return false;
                }
                A_upk[UpkIndex(r, c)] = sqrtf(sum);
            } else {
                const float diag = A_upk[UpkIndex(r, r)];
                A_upk[UpkIndex(r, c)] = sum / diag;
            }
        }
    }
    return true;
}

template <int MaxM>
__device__ void SolveCholUpperPacked(const float* U_upk, int m, float* b_inout) {
    // Solve (U^T U) x = b. Overwrites b with x.
    float y[MaxM];
    // Forward solve U^T y = b
    for (int i = 0; i < m; ++i) {
        float sum = b_inout[i];
        for (int k = 0; k < i; ++k) {
            sum -= U_upk[UpkIndex(k, i)] * y[k];
        }
        y[i] = sum / U_upk[UpkIndex(i, i)];
    }
    // Backward solve U x = y
    for (int i = m - 1; i >= 0; --i) {
        float sum = y[i];
        for (int k = i + 1; k < m; ++k) {
            sum -= U_upk[UpkIndex(i, k)] * b_inout[k];
        }
        b_inout[i] = sum / U_upk[UpkIndex(i, i)];
    }
}

template <int M>
__device__ bool CholUpperPackedFixed(float* A_upk) {
    #pragma unroll
    for (int c = 0; c < M; ++c) {
        #pragma unroll
        for (int r = 0; r <= c; ++r) {
            float sum = A_upk[UpkIndex(r, c)];
            #pragma unroll
            for (int k = 0; k < r; ++k) {
                sum -= A_upk[UpkIndex(k, r)] * A_upk[UpkIndex(k, c)];
            }
            if (r == c) {
                if (!(sum > 0.0f)) {
                    return false;
                }
                A_upk[UpkIndex(r, c)] = sqrtf(sum);
            } else {
                const float diag = A_upk[UpkIndex(r, r)];
                A_upk[UpkIndex(r, c)] = sum / diag;
            }
        }
    }
    return true;
}

template <int M>
__device__ void SolveCholUpperPackedFixed(const float* U_upk, float* b_inout) {
    float y[M];
    // Forward solve U^T y = b
    #pragma unroll
    for (int i = 0; i < M; ++i) {
        float sum = b_inout[i];
        #pragma unroll
        for (int k = 0; k < i; ++k) {
            sum -= U_upk[UpkIndex(k, i)] * y[k];
        }
        y[i] = sum / U_upk[UpkIndex(i, i)];
    }
    // Backward solve U x = y
    #pragma unroll
    for (int i = M - 1; i >= 0; --i) {
        float sum = y[i];
        #pragma unroll
        for (int k = i + 1; k < M; ++k) {
            sum -= U_upk[UpkIndex(i, k)] * b_inout[k];
        }
        b_inout[i] = sum / U_upk[UpkIndex(i, i)];
    }
}

__global__ void InitActiveChanged(int n, std::uint8_t* active, std::uint8_t* changed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    active[i] = 1;
    changed[i] = 0;
}

__global__ void ClearChanged(int n, std::uint8_t* changed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    changed[i] = 0;
}

__global__ void UpdateActive(int n, std::uint8_t* active, const std::uint8_t* changed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (active[i] && !changed[i]) {
        active[i] = 0;
    }
}

// ---------------- Full-precomp (small h0) ----------------

__device__ __forceinline__ float GAtDev(const float* G, int rows, int row, int col) {
    return G[static_cast<std::size_t>(col) * static_cast<std::size_t>(rows) + static_cast<std::size_t>(row)];
}

template <bool UseAbs>
__global__ void IcmLayerFull(int H, int m,
                             const int* __restrict__ offsets,
                             const int* __restrict__ h_vec,
                             const float* __restrict__ invnorm_flat,
                             const float* __restrict__ xC,  // H×n (col-major)
                             const float* __restrict__ G,   // H×H
                             const float* __restrict__ X_norm2,
                             std::uint8_t* __restrict__ active,
                             std::uint8_t* __restrict__ changed,
                             int jlayer,
                             FullCode* __restrict__ B,      // m×n
                             float* __restrict__ a,         // m×n
                             float* __restrict__ cost,      // n
                             int n) {
    // block-per-sample to make G/xC segment accesses coalesced across threads.
    const int i = blockIdx.x;
    if (i >= n) return;
    if (!active[i]) return;
    // Host RunIcmDevice supplies jlayer from validated order_host and skips layer 0.
    // if (jlayer <= 0 || jlayer >= m) {
    //     return;
    // }

    const int tid = threadIdx.x;
    // Non-root hj==256 is validated once in FullIcmCudaSession::EnsurePrecomp.
    (void)h_vec;
    const int startf = offsets[jlayer];

    FullCode* B_col = B + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    float* a_col = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    const float* xC_col = xC + static_cast<std::size_t>(i) * static_cast<std::size_t>(H);

    __shared__ FullCode B_sh[kMaxM];
    __shared__ float a_sh[kMaxM];
    if (tid < m) {
        B_sh[tid] = B_col[tid];
        a_sh[tid] = a_col[tid];
    }
    __syncthreads();

    const int old_code = static_cast<int>(B_sh[jlayer]);

    float score = -INFINITY;
    int code = 0;
    const int row = startf + tid;
    float tmp = xC_col[row];
    #pragma unroll
    for (int l = 0; l < kMaxM; ++l) {
        if (l >= m) break;
        if (l == jlayer) continue;
        const float alpha = a_sh[l];
        const int flat_l = offsets[l] + static_cast<int>(B_sh[l]);
        tmp -= alpha * GAtDev(G, H, row, flat_l);
    }
    const float val = tmp * invnorm_flat[row];
    score = UseAbs ? fabsf(val) : val;
    code = tid;

    // Warp-level reduction.
    unsigned mask = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) {
        const float v = __shfl_down_sync(mask, score, off);
        const int c = __shfl_down_sync(mask, code, off);
        // CPU scans codes in increasing order and keeps the first max (smallest code).
        if (v > score || (v == score && c < code)) {
            score = v;
            code = c;
        }
    }

    // Reduce warp winners.
    __shared__ float warp_best_val[8];
    __shared__ int warp_best_code[8];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) {
        warp_best_val[warp] = score;
        warp_best_code[warp] = code;
    }
    __syncthreads();

    int best_code = old_code;
    if (warp == 0) {
        float v = (lane < 8) ? warp_best_val[lane] : -INFINITY;
        int c = (lane < 8) ? warp_best_code[lane] : 0;
        for (int off = 16; off > 0; off >>= 1) {
            const float vv = __shfl_down_sync(mask, v, off);
            const int cc = __shfl_down_sync(mask, c, off);
            if (vv > v || (vv == v && cc < c)) {
                v = vv;
                c = cc;
            }
        }
        if (lane == 0) {
            best_code = c;
            warp_best_code[0] = best_code;
        }
    }
    __syncthreads();
    best_code = warp_best_code[0];

    if (tid != 0) {
        return;
    }
    if (best_code == old_code) return;

    // Backup coefficients.
    float a_old[kMaxM];
    for (int l = 0; l < m; ++l) {
        a_old[l] = a_col[l];
    }

    // Tentatively apply code update.
    B_col[jlayer] = static_cast<FullCode>(best_code);

    // Solve LS for this sample (Cholesky with retry).
    float A0[(kMaxM * (kMaxM + 1)) / 2];
    float A[(kMaxM * (kMaxM + 1)) / 2];
    float b0[kMaxM];
    float bvec[kMaxM];

    // Build b and base A upper triangle.
    for (int j = 0; j < m; ++j) {
        const int flat_j = offsets[j] + static_cast<int>(B_col[j]);
        const float bx = xC_col[flat_j];
        b0[j] = bx;
        bvec[j] = bx;
        for (int k = 0; k <= j; ++k) {
            const int flat_k = offsets[k] + static_cast<int>(B_col[k]);
            A0[UpkIndex(k, j)] = GAtDev(G, H, flat_k, flat_j);
        }
    }

    bool ok = false;
    float bump = 0.0f;
    for (int attempt = 0; attempt < 4 && !ok; ++attempt) {
        // copy base
        const int upk = (m * (m + 1)) / 2;
        for (int t = 0; t < upk; ++t) {
            A[t] = A0[t];
        }
        if (bump > 0.0f) {
            for (int d0 = 0; d0 < m; ++d0) {
                A[UpkIndex(d0, d0)] += bump;
            }
        }
        ok = CholUpperPacked<kMaxM>(A, m);
        if (!ok) {
            bump = (attempt == 0) ? kEps : bump * 10.0f;
        }
    }

    if (!ok) {
        // Reject update.
        B_col[jlayer] = static_cast<FullCode>(old_code);
        for (int l = 0; l < m; ++l) {
            a_col[l] = a_old[l];
        }
        return;
    }

    SolveCholUpperPacked<kMaxM>(A, m, bvec);
    for (int l = 0; l < m; ++l) {
        a_col[l] = bvec[l];
    }

    // Compute cost for acceptance.
    float term1 = 0.0f;
    float term2 = 0.0f;
    for (int l = 0; l < m; ++l) {
        const float alpha_l = bvec[l];
        term1 += alpha_l * b0[l];
        for (int k = 0; k < m; ++k) {
            term2 += alpha_l * bvec[k] * UpkSymAt(A0, l, k);
        }
    }
    const float new_cost = X_norm2[i] - 2.0f * term1 + term2;
#if defined(STLQ_ICM_VALIDATE_COST)
    {
        const std::uint32_t key = static_cast<std::uint32_t>(i) * 2654435761u + static_cast<std::uint32_t>(jlayer);
        if ((key & 1023u) == 0u) {
            float term1_ref = 0.0f;
            float term2_ref = 0.0f;
            for (int l = 0; l < m; ++l) {
                const int flat_l = offsets[l] + static_cast<int>(B_col[l]);
                const float al = a_col[l];
                term1_ref += al * xC_col[flat_l];
                for (int k = 0; k < m; ++k) {
                    const int flat_k = offsets[k] + static_cast<int>(B_col[k]);
                    term2_ref += al * a_col[k] * GAtDev(G, H, flat_l, flat_k);
                }
            }
            const float ref = X_norm2[i] - 2.0f * term1_ref + term2_ref;
            const float diff = fabsf(new_cost - ref);
            const float tol = 1e-5f * fmaxf(1.0f, fabsf(ref));
            if (!(diff <= tol)) {
                asm("trap;");
            }
        }
    }
#endif
    if (new_cost + kEps < cost[i]) {
        cost[i] = new_cost;
        changed[i] = 1;
    } else {
        // Reject update.
        B_col[jlayer] = static_cast<FullCode>(old_code);
        for (int l = 0; l < m; ++l) {
            a_col[l] = a_old[l];
        }
    }
}

template <bool UseAbs, int M>
__global__ void IcmLayerFullFixed(int H,
                                 const int* __restrict__ offsets,
                                 const int* __restrict__ h_vec,
                                 const float* __restrict__ invnorm_flat,
                                 const float* __restrict__ xC,  // H×n (col-major)
                                 const float* __restrict__ G,   // H×H
                                 const float* __restrict__ X_norm2,
                                 std::uint8_t* __restrict__ active,
                                 std::uint8_t* __restrict__ changed,
                                 int jlayer,
                                 FullCode* __restrict__ B,      // m×n
                                 float* __restrict__ a,         // m×n
                                 float* __restrict__ cost,      // n
                                 int n) {
    static_assert(M > 0, "M must be positive");
    static_assert(M <= kMaxM, "M exceeds kMaxM");
    const int i = blockIdx.x;
    if (i >= n) return;
    if (!active[i]) return;
    // Host RunIcmDevice supplies jlayer from validated order_host and skips layer 0.
    // if (jlayer <= 0 || jlayer >= M) {
    //     return;
    // }

    const int tid = threadIdx.x;
    // Non-root hj==256 is validated once in FullIcmCudaSession::EnsurePrecomp.
    (void)h_vec;
    const int startf = offsets[jlayer];

    FullCode* B_col = B + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    float* a_col = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    const float* xC_col = xC + static_cast<std::size_t>(i) * static_cast<std::size_t>(H);

    __shared__ FullCode B_sh[M];
    __shared__ float a_sh[M];
    if (tid < M) {
        B_sh[tid] = B_col[tid];
        a_sh[tid] = a_col[tid];
    }
    __syncthreads();

    const int old_code = static_cast<int>(B_sh[jlayer]);

    float score = -INFINITY;
    int code = 0;
    const int row = startf + tid;
    float tmp = xC_col[row];
    #pragma unroll
    for (int l = 0; l < M; ++l) {
        if (l == jlayer) continue;
        const float alpha = a_sh[l];
        const int flat_l = offsets[l] + static_cast<int>(B_sh[l]);
        tmp -= alpha * GAtDev(G, H, row, flat_l);
    }
    const float val = tmp * invnorm_flat[row];
    score = UseAbs ? fabsf(val) : val;
    code = tid;

    unsigned mask = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) {
        const float v = __shfl_down_sync(mask, score, off);
        const int c = __shfl_down_sync(mask, code, off);
        if (v > score || (v == score && c < code)) {
            score = v;
            code = c;
        }
    }

    __shared__ float warp_best_val[8];
    __shared__ int warp_best_code[8];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) {
        warp_best_val[warp] = score;
        warp_best_code[warp] = code;
    }
    __syncthreads();

    int best_code = old_code;
    if (warp == 0) {
        float v = (lane < 8) ? warp_best_val[lane] : -INFINITY;
        int c = (lane < 8) ? warp_best_code[lane] : 0;
        for (int off = 16; off > 0; off >>= 1) {
            const float vv = __shfl_down_sync(mask, v, off);
            const int cc = __shfl_down_sync(mask, c, off);
            if (vv > v || (vv == v && cc < c)) {
                v = vv;
                c = cc;
            }
        }
        if (lane == 0) {
            best_code = c;
            warp_best_code[0] = best_code;
        }
    }
    __syncthreads();
    best_code = warp_best_code[0];

    if (tid != 0) {
        return;
    }
    if (best_code == old_code) return;

    float a_old[M];
    #pragma unroll
    for (int l = 0; l < M; ++l) {
        a_old[l] = a_col[l];
    }

    B_col[jlayer] = static_cast<FullCode>(best_code);

    float A0[(M * (M + 1)) / 2];
    float A[(M * (M + 1)) / 2];
    float b0[M];
    float bvec[M];

    #pragma unroll
    for (int j = 0; j < M; ++j) {
        const int flat_j = offsets[j] + static_cast<int>(B_col[j]);
        const float bx = xC_col[flat_j];
        b0[j] = bx;
        bvec[j] = bx;
        #pragma unroll
        for (int k = 0; k <= j; ++k) {
            const int flat_k = offsets[k] + static_cast<int>(B_col[k]);
            A0[UpkIndex(k, j)] = GAtDev(G, H, flat_k, flat_j);
        }
    }

    bool ok = false;
    float bump = 0.0f;
    for (int attempt = 0; attempt < 4 && !ok; ++attempt) {
        constexpr int upk = (M * (M + 1)) / 2;
        #pragma unroll
        for (int t = 0; t < upk; ++t) {
            A[t] = A0[t];
        }
        if (bump > 0.0f) {
            #pragma unroll
            for (int d0 = 0; d0 < M; ++d0) {
                A[UpkIndex(d0, d0)] += bump;
            }
        }
        ok = CholUpperPackedFixed<M>(A);
        if (!ok) {
            bump = (attempt == 0) ? kEps : bump * 10.0f;
        }
    }

    if (!ok) {
        B_col[jlayer] = static_cast<FullCode>(old_code);
        #pragma unroll
        for (int l = 0; l < M; ++l) {
            a_col[l] = a_old[l];
        }
        return;
    }

    SolveCholUpperPackedFixed<M>(A, bvec);
    #pragma unroll
    for (int l = 0; l < M; ++l) {
        a_col[l] = bvec[l];
    }

    float term1 = 0.0f;
    float term2 = 0.0f;
    #pragma unroll
    for (int l = 0; l < M; ++l) {
        const float alpha_l = bvec[l];
        term1 += alpha_l * b0[l];
        #pragma unroll
        for (int k = 0; k < M; ++k) {
            term2 += alpha_l * bvec[k] * UpkSymAt(A0, l, k);
        }
    }
    const float new_cost = X_norm2[i] - 2.0f * term1 + term2;
#if defined(STLQ_ICM_VALIDATE_COST)
    {
        const std::uint32_t key = static_cast<std::uint32_t>(i) * 2654435761u + static_cast<std::uint32_t>(jlayer);
        if ((key & 1023u) == 0u) {
            float term1_ref = 0.0f;
            float term2_ref = 0.0f;
            #pragma unroll
            for (int l = 0; l < M; ++l) {
                const int flat_l = offsets[l] + static_cast<int>(B_col[l]);
                const float al = a_col[l];
                term1_ref += al * xC_col[flat_l];
                #pragma unroll
                for (int k = 0; k < M; ++k) {
                    const int flat_k = offsets[k] + static_cast<int>(B_col[k]);
                    term2_ref += al * a_col[k] * GAtDev(G, H, flat_l, flat_k);
                }
            }
            const float ref = X_norm2[i] - 2.0f * term1_ref + term2_ref;
            const float diff = fabsf(new_cost - ref);
            const float tol = 1e-5f * fmaxf(1.0f, fabsf(ref));
            if (!(diff <= tol)) {
                asm("trap;");
            }
        }
    }
#endif
    if (new_cost + kEps < cost[i]) {
        cost[i] = new_cost;
        changed[i] = 1;
    } else {
        B_col[jlayer] = static_cast<FullCode>(old_code);
        #pragma unroll
        for (int l = 0; l < M; ++l) {
            a_col[l] = a_old[l];
        }
    }
}

__global__ void SolveAllFull(int H, int m,
                             const int* __restrict__ offsets,
                             const float* __restrict__ xC,
                             const float* __restrict__ G,
                             const FullCode* __restrict__ B,
                             float* __restrict__ a,
                             int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const FullCode* B_col = B + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    float* a_col = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    const float* xC_col = xC + static_cast<std::size_t>(i) * static_cast<std::size_t>(H);

    float A0[(kMaxM * (kMaxM + 1)) / 2];
    float A[(kMaxM * (kMaxM + 1)) / 2];
    float b[kMaxM];

    for (int j = 0; j < m; ++j) {
        const int flat_j = offsets[j] + static_cast<int>(B_col[j]);
        b[j] = xC_col[flat_j];
        for (int k = 0; k <= j; ++k) {
            const int flat_k = offsets[k] + static_cast<int>(B_col[k]);
            A0[UpkIndex(k, j)] = GAtDev(G, H, flat_k, flat_j);
        }
    }

    bool ok = false;
    float bump = 0.0f;
    for (int attempt = 0; attempt < 4 && !ok; ++attempt) {
        const int upk = (m * (m + 1)) / 2;
        for (int t = 0; t < upk; ++t) {
            A[t] = A0[t];
        }
        if (bump > 0.0f) {
            for (int d0 = 0; d0 < m; ++d0) {
                A[UpkIndex(d0, d0)] += bump;
            }
        }
        ok = CholUpperPacked<kMaxM>(A, m);
        if (!ok) {
            bump = (attempt == 0) ? kEps : bump * 10.0f;
        }
    }

    if (!ok) {
        for (int l = 0; l < m; ++l) {
            a_col[l] = 0.0f;
        }
        return;
    }

    SolveCholUpperPacked<kMaxM>(A, m, b);
    for (int l = 0; l < m; ++l) {
        a_col[l] = b[l];
    }
}

template <int M>
__global__ void SolveAllFullFixed(int H,
                                  const int* __restrict__ offsets,
                                  const float* __restrict__ xC,
                                  const float* __restrict__ G,
                                  const FullCode* __restrict__ B,
                                  float* __restrict__ a,
                                  int n) {
    static_assert(M > 0, "M must be positive");
    static_assert(M <= kMaxM, "M exceeds kMaxM");
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const FullCode* B_col = B + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    float* a_col = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    const float* xC_col = xC + static_cast<std::size_t>(i) * static_cast<std::size_t>(H);

    float A0[(M * (M + 1)) / 2];
    float A[(M * (M + 1)) / 2];
    float b[M];

    #pragma unroll
    for (int j = 0; j < M; ++j) {
        const int flat_j = offsets[j] + static_cast<int>(B_col[j]);
        b[j] = xC_col[flat_j];
        #pragma unroll
        for (int k = 0; k <= j; ++k) {
            const int flat_k = offsets[k] + static_cast<int>(B_col[k]);
            A0[UpkIndex(k, j)] = GAtDev(G, H, flat_k, flat_j);
        }
    }

    bool ok = false;
    float bump = 0.0f;
    for (int attempt = 0; attempt < 4 && !ok; ++attempt) {
        constexpr int upk = (M * (M + 1)) / 2;
        #pragma unroll
        for (int t = 0; t < upk; ++t) {
            A[t] = A0[t];
        }
        if (bump > 0.0f) {
            #pragma unroll
            for (int d0 = 0; d0 < M; ++d0) {
                A[UpkIndex(d0, d0)] += bump;
            }
        }
        ok = CholUpperPackedFixed<M>(A);
        if (!ok) {
            bump = (attempt == 0) ? kEps : bump * 10.0f;
        }
    }

    if (!ok) {
        #pragma unroll
        for (int l = 0; l < M; ++l) {
            a_col[l] = 0.0f;
        }
        return;
    }

    SolveCholUpperPackedFixed<M>(A, b);
    #pragma unroll
    for (int l = 0; l < M; ++l) {
        a_col[l] = b[l];
    }
}

__global__ void ComputeCostsFull(int H, int m,
                                 const int* __restrict__ offsets,
                                 const float* __restrict__ xC,
                                 const float* __restrict__ G,
                                 const FullCode* __restrict__ B,
                                 const float* __restrict__ a,
                                 const float* __restrict__ X_norm2,
                                 float* __restrict__ out_cost,
                                 int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const FullCode* B_col = B + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    const float* a_col = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    const float* xC_col = xC + static_cast<std::size_t>(i) * static_cast<std::size_t>(H);

    float term1 = 0.0f;
    float term2 = 0.0f;
    for (int l = 0; l < m; ++l) {
        const int flat_l = offsets[l] + static_cast<int>(B_col[l]);
        const float alpha_l = a_col[l];
        term1 += alpha_l * xC_col[flat_l];
        for (int k = 0; k < m; ++k) {
            const int flat_k = offsets[k] + static_cast<int>(B_col[k]);
            term2 += alpha_l * a_col[k] * GAtDev(G, H, flat_l, flat_k);
        }
    }
    out_cost[i] = X_norm2[i] - 2.0f * term1 + term2;
}

template <int M>
__global__ void ComputeCostsFullFixed(int H,
                                      const int* __restrict__ offsets,
                                      const float* __restrict__ xC,
                                      const float* __restrict__ G,
                                      const FullCode* __restrict__ B,
                                      const float* __restrict__ a,
                                      const float* __restrict__ X_norm2,
                                      float* __restrict__ out_cost,
                                      int n) {
    static_assert(M > 0, "M must be positive");
    static_assert(M <= kMaxM, "M exceeds kMaxM");
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const FullCode* B_col = B + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    const float* a_col = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    const float* xC_col = xC + static_cast<std::size_t>(i) * static_cast<std::size_t>(H);

    float term1 = 0.0f;
    float term2 = 0.0f;
    #pragma unroll
    for (int l = 0; l < M; ++l) {
        const int flat_l = offsets[l] + static_cast<int>(B_col[l]);
        const float alpha_l = a_col[l];
        term1 += alpha_l * xC_col[flat_l];
        #pragma unroll
        for (int k = 0; k < M; ++k) {
            const int flat_k = offsets[k] + static_cast<int>(B_col[k]);
            term2 += alpha_l * a_col[k] * GAtDev(G, H, flat_l, flat_k);
        }
    }
    out_cost[i] = X_norm2[i] - 2.0f * term1 + term2;
}

__global__ void PerturbFullCodes(int m,
                                const int* __restrict__ h_vec,
                                std::uint32_t seed,
                                int outer_iter_1based,
                                std::uint64_t sample_id_offset,
                                int ksel,
                                FullCode* __restrict__ B,
                                int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int layer_min = 1;
    const int layer_max = m - 1;
    const int layer_span = layer_max - layer_min + 1;
    if (layer_span <= 0) return;

    // SplitMix64-like state (deterministic per (seed,outer,sample)).
    const std::uint64_t i_global = sample_id_offset + static_cast<std::uint64_t>(i);
    std::uint64_t state =
        static_cast<std::uint64_t>(seed) ^
        (static_cast<std::uint64_t>(outer_iter_1based) * 0x9e3779b97f4a7c15ULL) ^
        ((i_global + 1) * 0xbf58476d1ce4e5b9ULL);

    FullCode* B_col = B + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    for (int r = 0; r < ksel; ++r) {
        const int layer = layer_min + static_cast<int>(
            UniformBelow64(&state, static_cast<std::uint32_t>(layer_span)));
        const int h = h_vec[layer];
        if (h <= 1) continue;
        const int old_code = static_cast<int>(B_col[layer]);
        const int raw = static_cast<int>(UniformBelow64(&state, static_cast<std::uint32_t>(h - 1)));
        const int new_code = (raw >= old_code) ? (raw + 1) : raw;
        B_col[layer] = static_cast<FullCode>(new_code);
    }
}

__global__ void CopyAndPerturbFullCodes(int m,
                                       const int* __restrict__ h_vec,
                                       std::uint32_t seed,
                                       int outer_iter_1based,
                                       std::uint64_t sample_id_offset,
                                       int ksel,
                                       const FullCode* __restrict__ B_src,
                                       FullCode* __restrict__ B_dst,
                                       int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const FullCode* src = B_src + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    FullCode* dst = B_dst + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    #pragma unroll
    for (int l = 0; l < kMaxM; ++l) {
        if (l >= m) break;
        dst[l] = src[l];
    }

    const int layer_min = 1;
    const int layer_max = m - 1;
    const int layer_span = layer_max - layer_min + 1;
    if (layer_span <= 0) return;

    const std::uint64_t i_global = sample_id_offset + static_cast<std::uint64_t>(i);
    std::uint64_t state =
        static_cast<std::uint64_t>(seed) ^
        (static_cast<std::uint64_t>(outer_iter_1based) * 0x9e3779b97f4a7c15ULL) ^
        ((i_global + 1) * 0xbf58476d1ce4e5b9ULL);

    for (int r = 0; r < ksel; ++r) {
        const int layer = layer_min + static_cast<int>(
            UniformBelow64(&state, static_cast<std::uint32_t>(layer_span)));
        const int h = h_vec[layer];
        if (h <= 1) continue;
        const int old_code = static_cast<int>(dst[layer]);
        const int raw = static_cast<int>(UniformBelow64(&state, static_cast<std::uint32_t>(h - 1)));
        const int new_code = (raw >= old_code) ? (raw + 1) : raw;
        dst[layer] = static_cast<FullCode>(new_code);
    }
}

__global__ void PerturbSmallCodes(int m,
                                 const int* __restrict__ h_vec,
                                 std::uint32_t seed,
                                 int outer_iter_1based,
                                 std::uint64_t sample_id_offset,
                                 int ksel,
                                 Code* __restrict__ B_small,
                                 int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int layer_min = 1;
    const int layer_max = m - 1;
    const int layer_span = layer_max - layer_min + 1;
    if (layer_span <= 0) return;

    const std::uint64_t i_global = sample_id_offset + static_cast<std::uint64_t>(i);
    std::uint64_t state =
        static_cast<std::uint64_t>(seed) ^
        (static_cast<std::uint64_t>(outer_iter_1based) * 0x9e3779b97f4a7c15ULL) ^
        ((i_global + 1) * 0xbf58476d1ce4e5b9ULL);

    Code* B_col = B_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(m - 1);
    for (int r = 0; r < ksel; ++r) {
        const int layer = layer_min + static_cast<int>(
            UniformBelow64(&state, static_cast<std::uint32_t>(layer_span)));
        const int h = h_vec[layer];
        if (h <= 1) continue;
        const int idx = layer - 1;
        const int old_code = static_cast<int>(B_col[idx]);
        const int raw = static_cast<int>(UniformBelow64(&state, static_cast<std::uint32_t>(h - 1)));
        const int new_code = (raw >= old_code) ? (raw + 1) : raw;
        B_col[idx] = static_cast<Code>(new_code);
    }
}

__global__ void CopyAndPerturbSmallCodes(int m,
                                        const int* __restrict__ h_vec,
                                        std::uint32_t seed,
                                        int outer_iter_1based,
                                        std::uint64_t sample_id_offset,
                                        int ksel,
                                        const Code* __restrict__ B_src,
                                        Code* __restrict__ B_dst,
                                        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int m_codes = m - 1;
    const Code* src = B_src + static_cast<std::size_t>(i) * static_cast<std::size_t>(m_codes);
    Code* dst = B_dst + static_cast<std::size_t>(i) * static_cast<std::size_t>(m_codes);
    #pragma unroll
    for (int l = 0; l < kMaxM; ++l) {
        if (l >= m_codes) break;
        dst[l] = src[l];
    }

    const int layer_min = 1;
    const int layer_max = m - 1;
    const int layer_span = layer_max - layer_min + 1;
    if (layer_span <= 0) return;

    const std::uint64_t i_global = sample_id_offset + static_cast<std::uint64_t>(i);
    std::uint64_t state =
        static_cast<std::uint64_t>(seed) ^
        (static_cast<std::uint64_t>(outer_iter_1based) * 0x9e3779b97f4a7c15ULL) ^
        ((i_global + 1) * 0xbf58476d1ce4e5b9ULL);

    for (int r = 0; r < ksel; ++r) {
        const int layer = layer_min + static_cast<int>(
            UniformBelow64(&state, static_cast<std::uint32_t>(layer_span)));
        const int h = h_vec[layer];
        if (h <= 1) continue;
        const int idx = layer - 1;
        const int old_code = static_cast<int>(dst[idx]);
        const int raw = static_cast<int>(UniformBelow64(&state, static_cast<std::uint32_t>(h - 1)));
        const int new_code = (raw >= old_code) ? (raw + 1) : raw;
        dst[idx] = static_cast<Code>(new_code);
    }
}

template <int M>
__global__ void CopyAndPerturbSmallCodesFixed(const int* __restrict__ h_vec,
                                             std::uint32_t seed,
                                             int outer_iter_1based,
                                             std::uint64_t sample_id_offset,
                                             int ksel,
                                             const Code* __restrict__ B_src,
                                             Code* __restrict__ B_dst,
                                             int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    constexpr int m_codes = M - 1;
    const Code* src = B_src + static_cast<std::size_t>(i) * static_cast<std::size_t>(m_codes);
    Code* dst = B_dst + static_cast<std::size_t>(i) * static_cast<std::size_t>(m_codes);
    #pragma unroll
    for (int l = 0; l < m_codes; ++l) {
        dst[l] = src[l];
    }

    const int layer_min = 1;
    const int layer_max = M - 1;
    const int layer_span = layer_max - layer_min + 1;

    const std::uint64_t i_global = sample_id_offset + static_cast<std::uint64_t>(i);
    std::uint64_t state =
        static_cast<std::uint64_t>(seed) ^
        (static_cast<std::uint64_t>(outer_iter_1based) * 0x9e3779b97f4a7c15ULL) ^
        ((i_global + 1) * 0xbf58476d1ce4e5b9ULL);

    for (int r = 0; r < ksel; ++r) {
        const int layer = layer_min + static_cast<int>(
            UniformBelow64(&state, static_cast<std::uint32_t>(layer_span)));
        const int h = h_vec[layer];
        if (h <= 1) continue;
        const int idx = layer - 1;
        const int old_code = static_cast<int>(dst[idx]);
        const int raw = static_cast<int>(UniformBelow64(&state, static_cast<std::uint32_t>(h - 1)));
        const int new_code = (raw >= old_code) ? (raw + 1) : raw;
        dst[idx] = static_cast<Code>(new_code);
    }
}

__global__ void AcceptIfBetterFull(int m,
                                  const FullCode* __restrict__ B_cand,
                                  const float* __restrict__ a_cand,
                                  const float* __restrict__ cost_cand,
                                  FullCode* __restrict__ B,
                                  float* __restrict__ a,
                                  float* __restrict__ cost,
                                  int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float cand = cost_cand[i];
    if (cand + kEps < cost[i]) {
        const FullCode* srcB = B_cand + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
        FullCode* dstB = B + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
        const float* srcA = a_cand + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
        float* dstA = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
        for (int l = 0; l < m; ++l) {
            dstB[l] = srcB[l];
            dstA[l] = srcA[l];
        }
        cost[i] = cand;
    }
}

__global__ void AcceptIfBetterSmall(int m,
                                   const Code* __restrict__ B_cand,
                                   const float* __restrict__ a_cand,
                                   const float* __restrict__ cost_cand,
                                   Code* __restrict__ B,
                                   float* __restrict__ a,
                                   float* __restrict__ cost,
                                   int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float cand = cost_cand[i];
    if (cand + kEps < cost[i]) {
        const Code* srcB = B_cand + static_cast<std::size_t>(i) * static_cast<std::size_t>(m - 1);
        Code* dstB = B + static_cast<std::size_t>(i) * static_cast<std::size_t>(m - 1);
        const float* srcA = a_cand + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
        float* dstA = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
        for (int l = 1; l < m; ++l) {
            dstB[l - 1] = srcB[l - 1];
        }
        for (int l = 0; l < m; ++l) {
            dstA[l] = srcA[l];
        }
        cost[i] = cand;
    }
}

template <int M>
__global__ void AcceptIfBetterSmallFixed(const Code* __restrict__ B_cand,
                                        const float* __restrict__ a_cand,
                                        const float* __restrict__ cost_cand,
                                        Code* __restrict__ B,
                                        float* __restrict__ a,
                                        float* __restrict__ cost,
                                        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float cand = cost_cand[i];
    if (cand + kEps < cost[i]) {
        constexpr int m_codes = M - 1;
        const Code* srcB = B_cand + static_cast<std::size_t>(i) * static_cast<std::size_t>(m_codes);
        Code* dstB = B + static_cast<std::size_t>(i) * static_cast<std::size_t>(m_codes);
        #pragma unroll
        for (int l = 0; l < m_codes; ++l) {
            dstB[l] = srcB[l];
        }
        const float* srcA = a_cand + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
        float* dstA = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
        #pragma unroll
        for (int l = 0; l < M; ++l) {
            dstA[l] = srcA[l];
        }
        cost[i] = cand;
    }
}

// ---------------- Large-root variant ----------------

__device__ __forceinline__ float DotDev(const float* a, const float* b, int d) {
    float s = 0.0f;
    for (int i = 0; i < d; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

// Transpose G0S from (h0×Hs) column-major to (Hs×h0) column-major.
// src layout: src[cid + flat*h0]
// dst layout: dst[flat + cid*Hs]
__global__ void TransposeG0S_H0xHs_To_HsXH0(int h0, int Hs,
                                            const float* __restrict__ src,
                                            float* __restrict__ dst) {
    const int flat = blockIdx.x * blockDim.x + threadIdx.x;
    const int cid = blockIdx.y * blockDim.y + threadIdx.y;
    if (flat >= Hs || cid >= h0) return;
    dst[static_cast<std::size_t>(flat) + static_cast<std::size_t>(cid) * static_cast<std::size_t>(Hs)] =
        src[static_cast<std::size_t>(cid) + static_cast<std::size_t>(flat) * static_cast<std::size_t>(h0)];
}

__global__ void ComputeXc0FromXAndC0(int d, int h0,
                                     const float* __restrict__ X, int ldX,
                                     const float* __restrict__ C0, int ldC0,
                                     const std::uint32_t* __restrict__ cluster_id,
                                     float* __restrict__ out_xC0,
                                     int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const std::uint32_t cid = cluster_id[i];
    // Large-root callers provide cluster_id from validated root assignments.
    // if (cid >= static_cast<std::uint32_t>(h0)) {
    //     out_xC0[i] = 0.0f;
    //     return;
    // }
    const float* x = X + static_cast<std::size_t>(i) * static_cast<std::size_t>(ldX);
    const float* c0 = C0 + static_cast<std::size_t>(cid) * static_cast<std::size_t>(ldC0);
    float sum = 0.0f;
    for (int r = 0; r < d; ++r) {
        sum += x[r] * c0[r];
    }
    out_xC0[i] = sum;
}

template <bool UseAbs>
__global__ void IcmLayerLargeRoot(int d, int h0, int Hs, int m,
                                  const int* __restrict__ small_offsets,
                                  const int* __restrict__ h_vec,
                                  const float* __restrict__ invnorm_small,
                                  const float* __restrict__ C0,     // d×h0
                                  const float* __restrict__ norm0,  // h0
                                  const float* __restrict__ xC_small,  // Hs×n
                                  const float* __restrict__ G_small,   // Hs×Hs
                                  const float* __restrict__ G0S_T,     // Hs×h0 (column-major), see TransposeG0S_H0xHs_To_HsXH0
                                  const float* __restrict__ X,         // d×n
                                  const float* __restrict__ X_norm2,   // n
                                  const std::uint32_t* __restrict__ cluster_id,
                                  std::uint8_t* __restrict__ active,
                                  std::uint8_t* __restrict__ changed,
                                  int jlayer,
                                  Code* __restrict__ B_small,      // (m-1)×n
                                  float* __restrict__ a,           // m×n
                                  float* __restrict__ cost,        // n
                                  int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!active[i]) return;
    // Host RunIcmDevice supplies jlayer from validated order_host and skips layer 0.
    // if (jlayer <= 0 || jlayer >= m) return;

    const int hj = h_vec[jlayer];
    if (hj <= 1) return;
    const int startf = small_offsets[jlayer];

    const std::uint32_t cid = cluster_id[i];
    // Large-root callers provide cluster_id from validated root assignments.
    // if (cid >= static_cast<std::uint32_t>(h0)) return;

    Code* B_col = B_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(m - 1);
    float* a_col = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    const float* xC_col = xC_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(Hs);

    const int old_code = static_cast<int>(B_col[jlayer - 1]);
    int best_code = old_code;
    float best_score = -INFINITY;

    const float a0 = a_col[0];

    // Pick best code for this layer.
    for (int t = 0; t < hj; ++t) {
        const int row = startf + t;
        float tmp = xC_col[row];
        // subtract root contribution
        tmp -= a0 * G0S_T[static_cast<std::size_t>(row) +
                          static_cast<std::size_t>(cid) * static_cast<std::size_t>(Hs)];
        // subtract small contributions
        for (int l = 1; l < m; ++l) {
            if (l == jlayer) continue;
            const float alpha = a_col[l];
            const int flat_l = small_offsets[l] + static_cast<int>(B_col[l - 1]);
            tmp -= alpha * GAtDev(G_small, Hs, row, flat_l);
        }
        const float val = tmp * invnorm_small[row];
        const float score = UseAbs ? fabsf(val) : val;
        // CPU scans codes in increasing order and keeps the first max (smallest code).
        if (score > best_score || (score == best_score && t < best_code)) {
            best_score = score;
            best_code = t;
        }
    }

    if (best_code == old_code) {
        return;
    }

    float a_old[kMaxM];
    for (int l = 0; l < m; ++l) {
        a_old[l] = a_col[l];
    }
    B_col[jlayer - 1] = static_cast<Code>(best_code);

    // Build LS system.
    float A0[(kMaxM * (kMaxM + 1)) / 2];
    float A[(kMaxM * (kMaxM + 1)) / 2];
    float b0[kMaxM];
    float bvec[kMaxM];

    const float* x = X + static_cast<std::size_t>(i) * static_cast<std::size_t>(d);
    const float* c0 = C0 + static_cast<std::size_t>(cid) * static_cast<std::size_t>(d);
    const float xC0 = DotDev(c0, x, d);
    b0[0] = xC0;
    bvec[0] = xC0;
    A0[UpkIndex(0, 0)] = norm0[cid];

    for (int j = 1; j < m; ++j) {
        const int flat_j = small_offsets[j] + static_cast<int>(B_col[j - 1]);
        const float bx = xC_col[flat_j];
        b0[j] = bx;
        bvec[j] = bx;
        // root-small
        A0[UpkIndex(0, j)] = G0S_T[static_cast<std::size_t>(flat_j) +
                                   static_cast<std::size_t>(cid) * static_cast<std::size_t>(Hs)];
        for (int k = 1; k <= j; ++k) {
            const int flat_k = small_offsets[k] + static_cast<int>(B_col[k - 1]);
            A0[UpkIndex(k, j)] = GAtDev(G_small, Hs, flat_k, flat_j);
        }
    }

    bool ok = false;
    float bump = 0.0f;
    for (int attempt = 0; attempt < 4 && !ok; ++attempt) {
        const int upk = (m * (m + 1)) / 2;
        for (int t = 0; t < upk; ++t) {
            A[t] = A0[t];
        }
        if (bump > 0.0f) {
            for (int dd = 0; dd < m; ++dd) {
                A[UpkIndex(dd, dd)] += bump;
            }
        }
        ok = CholUpperPacked<kMaxM>(A, m);
        if (!ok) {
            bump = (attempt == 0) ? kEps : bump * 10.0f;
        }
    }

    if (!ok) {
        B_col[jlayer - 1] = static_cast<Code>(old_code);
        for (int l = 0; l < m; ++l) {
            a_col[l] = a_old[l];
        }
        return;
    }

    SolveCholUpperPacked<kMaxM>(A, m, bvec);
    for (int l = 0; l < m; ++l) {
        a_col[l] = bvec[l];
    }

    // Compute cost.
    float term1 = 0.0f;
    float term2 = 0.0f;
    for (int l = 0; l < m; ++l) {
        const float alpha_l = bvec[l];
        term1 += alpha_l * b0[l];
        for (int k = 0; k < m; ++k) {
            term2 += alpha_l * bvec[k] * UpkSymAt(A0, l, k);
        }
    }
    const float new_cost = X_norm2[i] - 2.0f * term1 + term2;
    if (new_cost + kEps < cost[i]) {
        cost[i] = new_cost;
        changed[i] = 1;
    } else {
        B_col[jlayer - 1] = static_cast<Code>(old_code);
        for (int l = 0; l < m; ++l) {
            a_col[l] = a_old[l];
        }
    }
}

template <bool UseAbs>
__global__ void IcmLayerLargeRootBlock(int d, int h0, int Hs, int m,
                                       const int* __restrict__ small_offsets,
                                       const int* __restrict__ h_vec,
                                       const float* __restrict__ invnorm_small,
                                       const float* __restrict__ norm0,  // h0
                                       const float* __restrict__ xC_small,  // Hs×n
                                       const float* __restrict__ G_small,   // Hs×Hs
                                       const float* __restrict__ G0S_T,     // Hs×h0 (column-major)
                                       const float* __restrict__ xC0,       // n (dot(C0[:,cid], X[:,i]))
                                       const float* __restrict__ X_norm2,   // n
                                       const std::uint32_t* __restrict__ cluster_id,
                                       std::uint8_t* __restrict__ active,
                                       std::uint8_t* __restrict__ changed,
                                       int jlayer,
                                       Code* __restrict__ B_small,  // (m-1)×n
                                       float* __restrict__ a,       // m×n
                                       float* __restrict__ cost,    // n
                                       int n) {
    // block-per-sample for parallel scan over codes (hj up to 256).
    const int i = blockIdx.x;
    if (i >= n) return;
    if (!active[i]) return;
    // Host RunIcmDevice supplies jlayer from validated order_host and skips layer 0.
    // if (jlayer <= 0 || jlayer >= m) return;

    const int tid = threadIdx.x;
    // Non-root hj==256 is validated once in LargeRootIcmCudaSession::EnsurePrecomp.
    (void)h_vec;
    const int startf = small_offsets[jlayer];

    const std::uint32_t cid = cluster_id[i];
    // Large-root callers provide cluster_id from validated root assignments.
    // if (cid >= static_cast<std::uint32_t>(h0)) return;

    Code* B_col = B_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(m - 1);
    float* a_col = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    const float* xC_col = xC_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(Hs);

    __shared__ Code B_sh[kMaxM];
    __shared__ float a_sh[kMaxM];
    if (tid < (m - 1)) {
        B_sh[tid + 1] = B_col[tid];
    }
    if (tid < m) {
        a_sh[tid] = a_col[tid];
    }
    if (tid == 0) {
        B_sh[0] = static_cast<Code>(cid);
    }
    __syncthreads();

    const int old_code = static_cast<int>(B_col[jlayer - 1]);
    float score = -INFINITY;
    int code = 0;

    const int row = startf + tid;
    float tmp = xC_col[row];
    // subtract root contribution
    tmp -= a_sh[0] * G0S_T[static_cast<std::size_t>(row) +
                           static_cast<std::size_t>(cid) * static_cast<std::size_t>(Hs)];
    #pragma unroll
    for (int l = 1; l < kMaxM; ++l) {
        if (l >= m) break;
        if (l == jlayer) continue;
        const float alpha = a_sh[l];
        const int flat_l = small_offsets[l] + static_cast<int>(B_sh[l]);
        tmp -= alpha * GAtDev(G_small, Hs, row, flat_l);
    }
    const float val = tmp * invnorm_small[row];
    score = UseAbs ? fabsf(val) : val;
    code = tid;

    // Warp reduction.
    unsigned mask = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) {
        const float v = __shfl_down_sync(mask, score, off);
        const int c = __shfl_down_sync(mask, code, off);
        if (v > score || (v == score && c < code)) {
            score = v;
            code = c;
        }
    }

    __shared__ float warp_best_val[8];
    __shared__ int warp_best_code[8];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) {
        warp_best_val[warp] = score;
        warp_best_code[warp] = code;
    }
    __syncthreads();

    int best_code = old_code;
    if (warp == 0) {
        float v = (lane < 8) ? warp_best_val[lane] : -INFINITY;
        int c = (lane < 8) ? warp_best_code[lane] : 0;
        for (int off = 16; off > 0; off >>= 1) {
            const float vv = __shfl_down_sync(mask, v, off);
            const int cc = __shfl_down_sync(mask, c, off);
            if (vv > v || (vv == v && cc < c)) {
                v = vv;
                c = cc;
            }
        }
        if (lane == 0) {
            best_code = c;
            warp_best_code[0] = best_code;
        }
    }
    __syncthreads();
    best_code = warp_best_code[0];

    if (tid != 0) return;
    if (best_code == old_code) return;

    // Keep candidate state local and only commit to global memory if accepted.
    int codes[kMaxM];
    codes[0] = static_cast<int>(cid);
    #pragma unroll
    for (int l = 1; l < kMaxM; ++l) {
        if (l >= m) break;
        codes[l] = static_cast<int>(B_col[l - 1]);
    }
    codes[jlayer] = best_code;

    float A0[(kMaxM * (kMaxM + 1)) / 2];
    float A[(kMaxM * (kMaxM + 1)) / 2];
    float b0[kMaxM];
    float bvec[kMaxM];

    b0[0] = xC0[i];
    bvec[0] = b0[0];
    A0[UpkIndex(0, 0)] = norm0[cid];
    for (int j = 1; j < m; ++j) {
        const int flat_j = small_offsets[j] + codes[j];
        const float bx = xC_col[flat_j];
        b0[j] = bx;
        bvec[j] = bx;
        A0[UpkIndex(0, j)] = G0S_T[static_cast<std::size_t>(flat_j) +
                                  static_cast<std::size_t>(cid) * static_cast<std::size_t>(Hs)];
        for (int k = 1; k <= j; ++k) {
            const int flat_k = small_offsets[k] + codes[k];
            A0[UpkIndex(k, j)] = GAtDev(G_small, Hs, flat_k, flat_j);
        }
    }

    bool ok = false;
    float bump = 0.0f;
    for (int attempt = 0; attempt < 4 && !ok; ++attempt) {
        const int upk = (m * (m + 1)) / 2;
        for (int t = 0; t < upk; ++t) {
            A[t] = A0[t];
        }
        for (int l = 0; l < m; ++l) {
            bvec[l] = b0[l];
        }
        if (bump > 0.0f) {
            for (int dd = 0; dd < m; ++dd) {
                A[UpkIndex(dd, dd)] += bump;
            }
        }
        ok = CholUpperPacked<kMaxM>(A, m);
        if (!ok) {
            bump = (attempt == 0) ? kEps : bump * 10.0f;
        }
    }

    if (!ok) {
        return;
    }

    SolveCholUpperPacked<kMaxM>(A, m, bvec);

    float term1 = 0.0f;
    float term2 = 0.0f;
    for (int l = 0; l < m; ++l) {
        const float alpha_l = bvec[l];
        term1 += alpha_l * b0[l];
        for (int k = 0; k < m; ++k) {
            term2 += alpha_l * bvec[k] * UpkSymAt(A0, l, k);
        }
    }
    const float new_cost = X_norm2[i] - 2.0f * term1 + term2;
    if (new_cost + kEps < cost[i]) {
        // Commit candidate update.
        B_col[jlayer - 1] = static_cast<Code>(best_code);
        #pragma unroll
        for (int l = 0; l < kMaxM; ++l) {
            if (l >= m) break;
            a_col[l] = bvec[l];
        }
        cost[i] = new_cost;
        changed[i] = 1;
    }
}

template <bool UseAbs, int M>
__global__ void IcmLayerLargeRootBlockFixed(int d, int h0, int Hs,
                                            const int* __restrict__ small_offsets,
                                            const int* __restrict__ h_vec,
                                            const float* __restrict__ invnorm_small,
                                            const float* __restrict__ norm0,  // h0
                                            const float* __restrict__ xC_small,  // Hs×n
                                            const float* __restrict__ G_small,   // Hs×Hs
                                            const float* __restrict__ G0S_T,     // Hs×h0 (column-major)
                                            const float* __restrict__ xC0,       // n (dot(C0[:,cid], X[:,i]))
                                            const float* __restrict__ X_norm2,   // n
                                            const std::uint32_t* __restrict__ cluster_id,
                                            std::uint8_t* __restrict__ active,
                                            std::uint8_t* __restrict__ changed,
                                            int jlayer,
                                            Code* __restrict__ B_small,  // (M-1)×n
                                            float* __restrict__ a,       // M×n
                                            float* __restrict__ cost,    // n
                                            int n) {
    const int i = blockIdx.x;
    if (i >= n) return;
    if (!active[i]) return;
    // Host RunIcmDevice supplies jlayer from validated order_host and skips layer 0.
    // if (jlayer <= 0 || jlayer >= M) return;

    // specialization assumes full 256-way scan for layers 1..M-1; dispatch ensures this.
    (void)h_vec;
    const int startf = small_offsets[jlayer];

    const std::uint32_t cid = cluster_id[i];
    // Large-root callers provide cluster_id from validated root assignments.
    // if (cid >= static_cast<std::uint32_t>(h0)) return;

    const int tid = threadIdx.x;
    Code* B_col = B_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(M - 1);
    float* a_col = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    const float* xC_col = xC_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(Hs);

    __shared__ Code B_sh[M];
    __shared__ float a_sh[M];
    if (tid < (M - 1)) {
        B_sh[tid + 1] = B_col[tid];
    }
    if (tid < M) {
        a_sh[tid] = a_col[tid];
    }
    if (tid == 0) {
        B_sh[0] = static_cast<Code>(cid);
    }
    __syncthreads();

    const int old_code = static_cast<int>(B_col[jlayer - 1]);
    float score = -INFINITY;
    int code = 0;

    // Scan 256 codes in parallel.
    const int row = startf + tid;
    float tmp = xC_col[row];
    tmp -= a_sh[0] * G0S_T[static_cast<std::size_t>(row) +
                           static_cast<std::size_t>(cid) * static_cast<std::size_t>(Hs)];
    #pragma unroll
    for (int l = 1; l < M; ++l) {
        if (l == jlayer) continue;
        const float alpha = a_sh[l];
        const int flat_l = small_offsets[l] + static_cast<int>(B_sh[l]);
        tmp -= alpha * GAtDev(G_small, Hs, row, flat_l);
    }
    const float val = tmp * invnorm_small[row];
    score = UseAbs ? fabsf(val) : val;
    code = tid;

    // Warp reduction (keep smallest code on ties, matching CPU).
    unsigned mask = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) {
        const float v = __shfl_down_sync(mask, score, off);
        const int c = __shfl_down_sync(mask, code, off);
        if (v > score || (v == score && c < code)) {
            score = v;
            code = c;
        }
    }

    __shared__ float warp_best_val[8];
    __shared__ int warp_best_code[8];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) {
        warp_best_val[warp] = score;
        warp_best_code[warp] = code;
    }
    __syncthreads();

    int best_code = old_code;
    if (warp == 0) {
        float v = (lane < 8) ? warp_best_val[lane] : -INFINITY;
        int c = (lane < 8) ? warp_best_code[lane] : 0;
        for (int off = 16; off > 0; off >>= 1) {
            const float vv = __shfl_down_sync(mask, v, off);
            const int cc = __shfl_down_sync(mask, c, off);
            if (vv > v || (vv == v && cc < c)) {
                v = vv;
                c = cc;
            }
        }
        if (lane == 0) {
            warp_best_code[0] = c;
        }
    }
    __syncthreads();
    best_code = warp_best_code[0];

    if (tid != 0) return;
    if (best_code == old_code) return;

    int codes[M];
    codes[0] = static_cast<int>(cid);
    #pragma unroll
    for (int l = 1; l < M; ++l) {
        codes[l] = static_cast<int>(B_col[l - 1]);
    }
    codes[jlayer] = best_code;

    float A0[(M * (M + 1)) / 2];
    float A[(M * (M + 1)) / 2];
    float b0[M];
    float bvec[M];

    b0[0] = xC0[i];
    bvec[0] = b0[0];
    A0[UpkIndex(0, 0)] = norm0[cid];
    #pragma unroll
    for (int j = 1; j < M; ++j) {
        const int flat_j = small_offsets[j] + codes[j];
        const float bx = xC_col[flat_j];
        b0[j] = bx;
        bvec[j] = bx;
        A0[UpkIndex(0, j)] = G0S_T[static_cast<std::size_t>(flat_j) +
                                  static_cast<std::size_t>(cid) * static_cast<std::size_t>(Hs)];
        #pragma unroll
        for (int k = 1; k <= j; ++k) {
            const int flat_k = small_offsets[k] + codes[k];
            A0[UpkIndex(k, j)] = GAtDev(G_small, Hs, flat_k, flat_j);
        }
    }

    bool ok = false;
    float bump = 0.0f;
    #pragma unroll
    for (int attempt = 0; attempt < 4 && !ok; ++attempt) {
        constexpr int upk = (M * (M + 1)) / 2;
        #pragma unroll
        for (int t = 0; t < upk; ++t) {
            A[t] = A0[t];
        }
        #pragma unroll
        for (int l = 0; l < M; ++l) {
            bvec[l] = b0[l];
        }
        if (bump > 0.0f) {
            #pragma unroll
            for (int dd = 0; dd < M; ++dd) {
                A[UpkIndex(dd, dd)] += bump;
            }
        }
        ok = CholUpperPackedFixed<M>(A);
        if (!ok) {
            bump = (attempt == 0) ? kEps : bump * 10.0f;
        }
    }
    if (!ok) return;

    SolveCholUpperPackedFixed<M>(A, bvec);

    float term1 = 0.0f;
    float term2 = 0.0f;
    #pragma unroll
    for (int l = 0; l < M; ++l) {
        const float alpha_l = bvec[l];
        term1 += alpha_l * b0[l];
        #pragma unroll
        for (int k = 0; k < M; ++k) {
            term2 += alpha_l * bvec[k] * UpkSymAt(A0, l, k);
        }
    }

    const float new_cost = X_norm2[i] - 2.0f * term1 + term2;
    if (new_cost + kEps < cost[i]) {
        B_col[jlayer - 1] = static_cast<Code>(best_code);
        #pragma unroll
        for (int l = 0; l < M; ++l) {
            a_col[l] = bvec[l];
        }
        cost[i] = new_cost;
        changed[i] = 1;
    }
}

__global__ void SolveAllLargeRoot(int h0, int Hs, int m,
                                  const int* __restrict__ small_offsets,
                                  const float* __restrict__ norm0,
                                  const float* __restrict__ xC_small,
                                  const float* __restrict__ G_small,
                                  const float* __restrict__ G0S_T,
                                  const float* __restrict__ xC0,
                                  const std::uint32_t* __restrict__ cluster_id,
                                  const Code* __restrict__ B_small,
                                  float* __restrict__ a,
                                  int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const std::uint32_t cid = cluster_id[i];
    // Large-root callers provide cluster_id from validated root assignments.
    // if (cid >= static_cast<std::uint32_t>(h0)) {
    //     float* a_col = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    //     for (int l = 0; l < m; ++l) a_col[l] = 0.0f;
    //     return;
    // }

    const Code* B_col = B_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(m - 1);
    float* a_col = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    const float* xC_col = xC_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(Hs);

    float A0[(kMaxM * (kMaxM + 1)) / 2];
    float A[(kMaxM * (kMaxM + 1)) / 2];
    float b0[kMaxM];
    float bvec[kMaxM];

    b0[0] = xC0[i];
    bvec[0] = b0[0];
    A0[UpkIndex(0, 0)] = norm0[cid];
    for (int j = 1; j < m; ++j) {
        const int flat_j = small_offsets[j] + static_cast<int>(B_col[j - 1]);
        const float bx = xC_col[flat_j];
        b0[j] = bx;
        bvec[j] = bx;
        A0[UpkIndex(0, j)] = G0S_T[static_cast<std::size_t>(flat_j) +
                                  static_cast<std::size_t>(cid) * static_cast<std::size_t>(Hs)];
        for (int k = 1; k <= j; ++k) {
            const int flat_k = small_offsets[k] + static_cast<int>(B_col[k - 1]);
            A0[UpkIndex(k, j)] = GAtDev(G_small, Hs, flat_k, flat_j);
        }
    }

    bool ok = false;
    float bump = 0.0f;
    for (int attempt = 0; attempt < 4 && !ok; ++attempt) {
        const int upk = (m * (m + 1)) / 2;
        for (int t = 0; t < upk; ++t) {
            A[t] = A0[t];
        }
        for (int l = 0; l < m; ++l) {
            bvec[l] = b0[l];
        }
        if (bump > 0.0f) {
            for (int dd = 0; dd < m; ++dd) {
                A[UpkIndex(dd, dd)] += bump;
            }
        }
        ok = CholUpperPacked<kMaxM>(A, m);
        if (!ok) {
            bump = (attempt == 0) ? kEps : bump * 10.0f;
        }
    }

    if (!ok) {
        for (int l = 0; l < m; ++l) {
            a_col[l] = 0.0f;
        }
        return;
    }

    SolveCholUpperPacked<kMaxM>(A, m, bvec);
    for (int l = 0; l < m; ++l) {
        a_col[l] = bvec[l];
    }
}

__global__ void ComputeCostsLargeRoot(int h0, int Hs, int m,
                                      const int* __restrict__ small_offsets,
                                      const float* __restrict__ norm0,
                                      const float* __restrict__ xC_small,
                                      const float* __restrict__ G_small,
                                      const float* __restrict__ G0S_T,
                                      const float* __restrict__ xC0,
                                      const float* __restrict__ X_norm2,
                                      const std::uint32_t* __restrict__ cluster_id,
                                      const Code* __restrict__ B_small,
                                      const float* __restrict__ a,
                                      float* __restrict__ out_cost,
                                      int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const std::uint32_t cid = cluster_id[i];
    // Large-root callers provide cluster_id from validated root assignments.
    // if (cid >= static_cast<std::uint32_t>(h0)) {
    //     out_cost[i] = X_norm2[i];
    //     return;
    // }

    const Code* B_col = B_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(m - 1);
    const float* a_col = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    const float* xC_col = xC_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(Hs);

    const float xC0_i = xC0[i];
    float term1 = a_col[0] * xC0_i;
    for (int l = 1; l < m; ++l) {
        const int flat_l = small_offsets[l] + static_cast<int>(B_col[l - 1]);
        term1 += a_col[l] * xC_col[flat_l];
    }

    float term2 = a_col[0] * a_col[0] * norm0[cid];
    for (int j = 1; j < m; ++j) {
        const float aj = a_col[j];
        const int flat_j = small_offsets[j] + static_cast<int>(B_col[j - 1]);
        term2 += 2.0f * a_col[0] * aj *
                 G0S_T[static_cast<std::size_t>(flat_j) +
                       static_cast<std::size_t>(cid) * static_cast<std::size_t>(Hs)];
        for (int k = 1; k < m; ++k) {
            const float ak = a_col[k];
            const int flat_k = small_offsets[k] + static_cast<int>(B_col[k - 1]);
            term2 += aj * ak * GAtDev(G_small, Hs, flat_j, flat_k);
        }
    }

    out_cost[i] = X_norm2[i] - 2.0f * term1 + term2;
}

__global__ void SolveCostInitLargeRoot(int h0, int Hs, int m,
                                       const int* __restrict__ small_offsets,
                                       const float* __restrict__ norm0,
                                       const float* __restrict__ xC_small,
                                       const float* __restrict__ G_small,
                                       const float* __restrict__ G0S_T,
                                       const float* __restrict__ xC0,
                                       const float* __restrict__ X_norm2,
                                       const std::uint32_t* __restrict__ cluster_id,
                                       const Code* __restrict__ B_small,
                                       float* __restrict__ a,
                                       float* __restrict__ out_cost,
                                       std::uint8_t* __restrict__ active,
                                       std::uint8_t* __restrict__ changed,
                                       int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const std::uint32_t cid = cluster_id[i];
    active[i] = 1;
    changed[i] = 0;

    float* a_col = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    // Large-root callers provide cluster_id from validated root assignments.
    // if (cid >= static_cast<std::uint32_t>(h0)) {
    //     for (int l = 0; l < m; ++l) a_col[l] = 0.0f;
    //     out_cost[i] = X_norm2[i];
    //     return;
    // }

    const Code* B_col = B_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(m - 1);
    const float* xC_col = xC_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(Hs);

    float A0[(kMaxM * (kMaxM + 1)) / 2];
    float A[(kMaxM * (kMaxM + 1)) / 2];
    float b0[kMaxM];
    float bvec[kMaxM];

    b0[0] = xC0[i];
    bvec[0] = b0[0];
    A0[UpkIndex(0, 0)] = norm0[cid];
    for (int j = 1; j < m; ++j) {
        const int flat_j = small_offsets[j] + static_cast<int>(B_col[j - 1]);
        const float bx = xC_col[flat_j];
        b0[j] = bx;
        bvec[j] = bx;
        A0[UpkIndex(0, j)] = G0S_T[static_cast<std::size_t>(flat_j) +
                                  static_cast<std::size_t>(cid) * static_cast<std::size_t>(Hs)];
        for (int k = 1; k <= j; ++k) {
            const int flat_k = small_offsets[k] + static_cast<int>(B_col[k - 1]);
            A0[UpkIndex(k, j)] = GAtDev(G_small, Hs, flat_k, flat_j);
        }
    }

    bool ok = false;
    float bump = 0.0f;
    for (int attempt = 0; attempt < 4 && !ok; ++attempt) {
        const int upk = (m * (m + 1)) / 2;
        for (int t = 0; t < upk; ++t) {
            A[t] = A0[t];
        }
        for (int l = 0; l < m; ++l) {
            bvec[l] = b0[l];
        }
        if (bump > 0.0f) {
            for (int dd = 0; dd < m; ++dd) {
                A[UpkIndex(dd, dd)] += bump;
            }
        }
        ok = CholUpperPacked<kMaxM>(A, m);
        if (!ok) {
            bump = (attempt == 0) ? kEps : bump * 10.0f;
        }
    }

    if (!ok) {
        for (int l = 0; l < m; ++l) {
            a_col[l] = 0.0f;
        }
        out_cost[i] = X_norm2[i];
        return;
    }

    SolveCholUpperPacked<kMaxM>(A, m, bvec);
    for (int l = 0; l < m; ++l) {
        a_col[l] = bvec[l];
    }

    // Initial cost.
    float term1 = 0.0f;
    float term2 = 0.0f;
    for (int l = 0; l < m; ++l) {
        const float alpha_l = bvec[l];
        term1 += alpha_l * b0[l];
        for (int k = 0; k < m; ++k) {
            term2 += alpha_l * bvec[k] * UpkSymAt(A0, l, k);
        }
    }
    out_cost[i] = X_norm2[i] - 2.0f * term1 + term2;
}

template <int M>
__global__ void SolveCostInitLargeRootFixed(int h0, int Hs,
                                           const int* __restrict__ small_offsets,
                                           const float* __restrict__ norm0,
                                           const float* __restrict__ xC_small,
                                           const float* __restrict__ G_small,
                                           const float* __restrict__ G0S_T,
                                           const float* __restrict__ xC0,
                                           const float* __restrict__ X_norm2,
                                           const std::uint32_t* __restrict__ cluster_id,
                                           const Code* __restrict__ B_small,
                                           float* __restrict__ a,
                                           float* __restrict__ out_cost,
                                           std::uint8_t* __restrict__ active,
                                           std::uint8_t* __restrict__ changed,
                                           int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const std::uint32_t cid = cluster_id[i];
    active[i] = 1;
    changed[i] = 0;

    float* a_col = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    // Large-root callers provide cluster_id from validated root assignments.
    // if (cid >= static_cast<std::uint32_t>(h0)) {
    //     #pragma unroll
    //     for (int l = 0; l < M; ++l) a_col[l] = 0.0f;
    //     out_cost[i] = X_norm2[i];
    //     return;
    // }

    constexpr int m_codes = M - 1;
    const Code* B_col = B_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(m_codes);
    const float* xC_col = xC_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(Hs);

    float A0[(M * (M + 1)) / 2];
    float A[(M * (M + 1)) / 2];
    float b0[M];
    float bvec[M];

    b0[0] = xC0[i];
    bvec[0] = b0[0];
    A0[UpkIndex(0, 0)] = norm0[cid];
    #pragma unroll
    for (int j = 1; j < M; ++j) {
        const int flat_j = small_offsets[j] + static_cast<int>(B_col[j - 1]);
        const float bx = xC_col[flat_j];
        b0[j] = bx;
        bvec[j] = bx;
        A0[UpkIndex(0, j)] = G0S_T[static_cast<std::size_t>(flat_j) +
                                  static_cast<std::size_t>(cid) * static_cast<std::size_t>(Hs)];
        #pragma unroll
        for (int k = 1; k <= j; ++k) {
            const int flat_k = small_offsets[k] + static_cast<int>(B_col[k - 1]);
            A0[UpkIndex(k, j)] = GAtDev(G_small, Hs, flat_k, flat_j);
        }
    }

    bool ok = false;
    float bump = 0.0f;
    #pragma unroll
    for (int attempt = 0; attempt < 4 && !ok; ++attempt) {
        constexpr int upk = (M * (M + 1)) / 2;
        #pragma unroll
        for (int t = 0; t < upk; ++t) {
            A[t] = A0[t];
        }
        #pragma unroll
        for (int l = 0; l < M; ++l) {
            bvec[l] = b0[l];
        }
        if (bump > 0.0f) {
            #pragma unroll
            for (int dd = 0; dd < M; ++dd) {
                A[UpkIndex(dd, dd)] += bump;
            }
        }
        ok = CholUpperPackedFixed<M>(A);
        if (!ok) {
            bump = (attempt == 0) ? kEps : bump * 10.0f;
        }
    }

    if (!ok) {
        #pragma unroll
        for (int l = 0; l < M; ++l) {
            a_col[l] = 0.0f;
        }
        out_cost[i] = X_norm2[i];
        return;
    }

    SolveCholUpperPackedFixed<M>(A, bvec);
    #pragma unroll
    for (int l = 0; l < M; ++l) {
        a_col[l] = bvec[l];
    }

    float term1 = 0.0f;
    float term2 = 0.0f;
    #pragma unroll
    for (int l = 0; l < M; ++l) {
        const float alpha_l = bvec[l];
        term1 += alpha_l * b0[l];
        #pragma unroll
        for (int k = 0; k < M; ++k) {
            term2 += alpha_l * bvec[k] * UpkSymAt(A0, l, k);
        }
    }

    out_cost[i] = X_norm2[i] - 2.0f * term1 + term2;
}

struct FullIcmCudaSession {
    int m = 0;
    int H = 0;
    int d = 0;
    int n = 0;

    DeviceBuf<float> d_xC;
    DeviceBuf<float> d_G;
    DeviceBuf<float> d_inv;
    DeviceBuf<int> d_offsets;
    DeviceBuf<int> d_hvec;

    DeviceBuf<FullCode> d_B;
    DeviceBuf<float> d_a;
    DeviceBuf<float> d_cost;
    DeviceBuf<FullCode> d_B_cand;
    DeviceBuf<float> d_a_cand;
    DeviceBuf<float> d_cost_cand;
    DeviceBuf<float> d_xn2;
    DeviceBuf<std::uint8_t> d_active;
    DeviceBuf<std::uint8_t> d_changed;

    // Host-side layer permutation buffer (avoid per-call allocations).
    std::vector<int> order_host;

    // Optional external device pointer to xC (H x n, col-major, ld==H). When set, we avoid uploading host xC.
    const float* xC_ptr = nullptr;

    std::uint64_t precomp_tag = 0;
    bool precomp_uploaded = false;
    CudaEventPair ev;

    void EnsurePrecomp(const Precomp& pre) {
        for (int layer = 1; layer < pre.m; ++layer) {
            if (layer >= static_cast<int>(pre.h_vec.size()) || pre.h_vec[static_cast<std::size_t>(layer)] != 256) {
                throw std::runtime_error(
                    "GPU ICM currently supports hj==256 for non-root layers only (layer>=1). "
                    "Got hj=" + std::to_string((layer < static_cast<int>(pre.h_vec.size())) ? pre.h_vec[static_cast<std::size_t>(layer)] : -1) +
                    " at layer=" + std::to_string(layer) + ". See stlq_gpu/README.md.");
            }
        }
        m = pre.m;
        d = pre.d;
        H = pre.H;
        const std::uint64_t new_tag =
            static_cast<std::uint64_t>(pre.build_tag) ^
            (static_cast<std::uint64_t>(H) << 1) ^
            (static_cast<std::uint64_t>(m) << 33);
        const bool same_precomp = (new_tag == precomp_tag) && precomp_uploaded;
        precomp_tag = new_tag;

        d_G.Ensure(static_cast<std::size_t>(H) * static_cast<std::size_t>(H));
        d_inv.Ensure(static_cast<std::size_t>(H));
        d_offsets.Ensure(static_cast<std::size_t>(m));
        d_hvec.Ensure(static_cast<std::size_t>(m));

        if (same_precomp) {
            return;
        }
        CudaCheck(cudaMemcpy(d_G.ptr, pre.G.data.data(),
                             sizeof(float) * static_cast<std::size_t>(H) * static_cast<std::size_t>(H),
                             cudaMemcpyHostToDevice),
                  "Memcpy G");
        CudaCheck(cudaMemcpy(d_inv.ptr, pre.invnorm_flat.data(),
                             sizeof(float) * static_cast<std::size_t>(H),
                             cudaMemcpyHostToDevice),
                  "Memcpy invnorm");
        CudaCheck(cudaMemcpy(d_offsets.ptr, pre.offsets.data(),
                             sizeof(int) * static_cast<std::size_t>(m),
                             cudaMemcpyHostToDevice),
                  "Memcpy offsets");
        CudaCheck(cudaMemcpy(d_hvec.ptr, pre.h_vec.data(),
                             sizeof(int) * static_cast<std::size_t>(m),
                             cudaMemcpyHostToDevice),
                  "Memcpy hvec");
        precomp_uploaded = true;
    }

    void EnsureBatch(int n_in) {
        n = n_in;
        d_xC.Ensure(static_cast<std::size_t>(H) * static_cast<std::size_t>(n));
        d_B.Ensure(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
        d_a.Ensure(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
        d_cost.Ensure(static_cast<std::size_t>(n));
        d_B_cand.Ensure(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
        d_a_cand.Ensure(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
        d_cost_cand.Ensure(static_cast<std::size_t>(n));
        d_xn2.Ensure(static_cast<std::size_t>(n));
        d_active.Ensure(static_cast<std::size_t>(n));
        d_changed.Ensure(static_cast<std::size_t>(n));
    }

    void UploadXcAndNorms(const ColMajorMatrix<float>& xC, const std::vector<float>& X_norm2) {
        xC_ptr = nullptr;
        CudaCheck(cudaMemcpy(d_xC.ptr, xC.data.data(),
                             sizeof(float) * static_cast<std::size_t>(H) * static_cast<std::size_t>(n),
                             cudaMemcpyHostToDevice),
                  "Memcpy xC");
        CudaCheck(cudaMemcpy(d_xn2.ptr, X_norm2.data(),
                             sizeof(float) * static_cast<std::size_t>(n),
                             cudaMemcpyHostToDevice),
                  "Memcpy X_norm2");
    }

    void UploadFixedDeviceXc(const DeviceMatF32View& xC_dev, const std::vector<float>& X_norm2) {
        if (!xC_dev.ptr || xC_dev.rows != H || xC_dev.cols != n || xC_dev.ld != H) {
            throw std::runtime_error("FullIcmCudaSession::UploadFixedDeviceXc: invalid xC_dev view.");
        }
        xC_ptr = xC_dev.ptr;
        CudaCheck(cudaMemcpy(d_xn2.ptr, X_norm2.data(),
                             sizeof(float) * static_cast<std::size_t>(n),
                             cudaMemcpyHostToDevice),
                  "Memcpy X_norm2");
    }

    void RunIcmDevice(bool use_abs,
                      int icm_iters,
                      FullCode* dB,
                      float* dA,
                      float* dCost,
                      std::mt19937* rng) {
        const int threads = 128;
        const int blocks = (n + threads - 1) / threads;
        const float* xc = xC_ptr ? xC_ptr : d_xC.ptr;
        if (m == 5) {
            SolveAllFullFixed<5><<<blocks, threads>>>(H, d_offsets.ptr, xc, d_G.ptr, dB, dA, n);
            ComputeCostsFullFixed<5><<<blocks, threads>>>(H, d_offsets.ptr, xc, d_G.ptr, dB,
                                                          dA, d_xn2.ptr, dCost, n);
        } else if (m == 10) {
            SolveAllFullFixed<10><<<blocks, threads>>>(H, d_offsets.ptr, xc, d_G.ptr, dB, dA, n);
            ComputeCostsFullFixed<10><<<blocks, threads>>>(H, d_offsets.ptr, xc, d_G.ptr, dB,
                                                           dA, d_xn2.ptr, dCost, n);
        } else {
            SolveAllFull<<<blocks, threads>>>(H, m, d_offsets.ptr, xc, d_G.ptr, dB, dA, n);
            ComputeCostsFull<<<blocks, threads>>>(H, m, d_offsets.ptr, xc, d_G.ptr, dB,
                                                  dA, d_xn2.ptr, dCost, n);
        }
        InitActiveChanged<<<blocks, threads>>>(n, d_active.ptr, d_changed.ptr);
        CudaCheck(cudaGetLastError(), "Launch init/solve/cost");

        if (static_cast<int>(order_host.size()) != m) {
            order_host.resize(static_cast<std::size_t>(m));
        }
        // Match CPU semantics: start each RunIcmDevice call from the canonical 0..m-1 order.
        std::iota(order_host.begin(), order_host.end(), 0);

        for (int it = 0; it < icm_iters; ++it) {
            std::shuffle(order_host.begin(), order_host.end(), *rng);
            ClearChanged<<<blocks, threads>>>(n, d_changed.ptr);
            for (int oi = 0; oi < m; ++oi) {
                const int jlayer = order_host[oi];
                if (jlayer == 0) continue;
                dim3 grid_icm(n);
                dim3 block_icm(256);
                if (use_abs) {
                    if (m == 5) {
                        IcmLayerFullFixed<true, 5><<<grid_icm, block_icm>>>(H, d_offsets.ptr, d_hvec.ptr,
                                                                           d_inv.ptr, xc, d_G.ptr, d_xn2.ptr,
                                                                           d_active.ptr, d_changed.ptr, jlayer,
                                                                           dB, dA, dCost, n);
                    } else if (m == 10) {
                        IcmLayerFullFixed<true, 10><<<grid_icm, block_icm>>>(H, d_offsets.ptr, d_hvec.ptr,
                                                                            d_inv.ptr, xc, d_G.ptr, d_xn2.ptr,
                                                                            d_active.ptr, d_changed.ptr, jlayer,
                                                                            dB, dA, dCost, n);
                    } else {
                        IcmLayerFull<true><<<grid_icm, block_icm>>>(H, m, d_offsets.ptr, d_hvec.ptr,
                                                                   d_inv.ptr, xc, d_G.ptr, d_xn2.ptr,
                                                                   d_active.ptr, d_changed.ptr, jlayer,
                                                                   dB, dA, dCost, n);
                    }
                } else {
                    if (m == 5) {
                        IcmLayerFullFixed<false, 5><<<grid_icm, block_icm>>>(H, d_offsets.ptr, d_hvec.ptr,
                                                                            d_inv.ptr, xc, d_G.ptr, d_xn2.ptr,
                                                                            d_active.ptr, d_changed.ptr, jlayer,
                                                                            dB, dA, dCost, n);
                    } else if (m == 10) {
                        IcmLayerFullFixed<false, 10><<<grid_icm, block_icm>>>(H, d_offsets.ptr, d_hvec.ptr,
                                                                             d_inv.ptr, xc, d_G.ptr, d_xn2.ptr,
                                                                             d_active.ptr, d_changed.ptr, jlayer,
                                                                             dB, dA, dCost, n);
                    } else {
                        IcmLayerFull<false><<<grid_icm, block_icm>>>(H, m, d_offsets.ptr, d_hvec.ptr,
                                                                    d_inv.ptr, xc, d_G.ptr, d_xn2.ptr,
                                                                    d_active.ptr, d_changed.ptr, jlayer,
                                                                    dB, dA, dCost, n);
                    }
                }
            }
            UpdateActive<<<blocks, threads>>>(n, d_active.ptr, d_changed.ptr);
        }
        CudaCheck(cudaGetLastError(), "Launch ICM loop");
    }

    void UploadB(const FullCode* B_host) {
        CudaCheck(cudaMemcpy(d_B.ptr, B_host,
                             sizeof(FullCode) * static_cast<std::size_t>(m) * static_cast<std::size_t>(n),
                             cudaMemcpyHostToDevice),
                  "Memcpy B to device");
    }

    void DownloadAll(FullCode* B_host, float* a_host, float* cost_host) {
        CudaCheck(cudaMemcpy(B_host, d_B.ptr,
                             sizeof(FullCode) * static_cast<std::size_t>(m) * static_cast<std::size_t>(n),
                             cudaMemcpyDeviceToHost),
                  "Memcpy B back");
        CudaCheck(cudaMemcpy(a_host, d_a.ptr,
                             sizeof(float) * static_cast<std::size_t>(m) * static_cast<std::size_t>(n),
                             cudaMemcpyDeviceToHost),
                  "Memcpy a back");
        CudaCheck(cudaMemcpy(cost_host, d_cost.ptr,
                             sizeof(float) * static_cast<std::size_t>(n),
                             cudaMemcpyDeviceToHost),
                  "Memcpy cost back");
    }
};

struct LargeRootIcmCudaSession {
    int d = 0;
    int m = 0;
    int h0 = 0;
    int Hs = 0;
    int n = 0;

    DeviceBuf<float> d_C0;
    DeviceBuf<float> d_norm0;
    DeviceBuf<float> d_Gs;
    // G0S is consumed as a transposed table for coalesced access inside ICM kernels.
    DeviceBuf<float> d_G0S_T;   // Hs×h0
    DeviceBuf<float> d_G0S_src; // h0×Hs (temporary upload buffer, freed after transpose)
    DeviceBuf<float> d_inv;
    DeviceBuf<int> d_small_offsets;
    DeviceBuf<int> d_hvec;

    DeviceBuf<float> d_xC;
    DeviceBuf<float> d_xn2;
    DeviceBuf<float> d_xC0;
    DeviceBuf<std::uint32_t> d_cid;

    DeviceBuf<Code> d_B;
    DeviceBuf<float> d_a;
    DeviceBuf<float> d_cost;
    DeviceBuf<std::uint8_t> d_active;
    DeviceBuf<std::uint8_t> d_changed;
    DeviceBuf<Code> d_B_cand;
    DeviceBuf<float> d_a_cand;
    DeviceBuf<float> d_cost_cand;

    std::uint64_t pre_tag = 0;
    bool precomp_uploaded = false;
    CudaEventPair ev;

    // Host-side layer permutation buffer (avoid per-call allocations).
    std::vector<int> order_host;
    const float* xC_ptr = nullptr;  // points to d_xC.ptr or an external device buffer

    void EnsurePrecomp(const PrecompLargeRoot& pre) {
        for (int layer = 1; layer < pre.m; ++layer) {
            if (layer >= static_cast<int>(pre.h_vec.size()) || pre.h_vec[static_cast<std::size_t>(layer)] != 256) {
                throw std::runtime_error(
                    "GPU ICM currently supports hj==256 for non-root layers only (layer>=1). "
                    "Got hj=" + std::to_string((layer < static_cast<int>(pre.h_vec.size())) ? pre.h_vec[static_cast<std::size_t>(layer)] : -1) +
                    " at layer=" + std::to_string(layer) + ". See stlq_gpu/README.md.");
            }
        }
        d = pre.d;
        m = pre.m;
        h0 = pre.h_vec.empty() ? 0 : pre.h_vec[0];
        Hs = pre.H_small;
        const std::uint64_t new_tag =
            static_cast<std::uint64_t>(pre.build_tag) ^
            (static_cast<std::uint64_t>(h0) << 1) ^
            (static_cast<std::uint64_t>(Hs) << 33);
        const bool same_precomp = (new_tag == pre_tag) && precomp_uploaded;
        pre_tag = new_tag;

        d_C0.Ensure(static_cast<std::size_t>(d) * static_cast<std::size_t>(h0));
        d_norm0.Ensure(static_cast<std::size_t>(h0));
        d_Gs.Ensure(static_cast<std::size_t>(Hs) * static_cast<std::size_t>(Hs));
        d_G0S_T.Ensure(static_cast<std::size_t>(h0) * static_cast<std::size_t>(Hs));
        d_inv.Ensure(static_cast<std::size_t>(Hs));
        d_small_offsets.Ensure(static_cast<std::size_t>(m));
        d_hvec.Ensure(static_cast<std::size_t>(m));

        if (same_precomp) {
            return;
        }
        CudaCheck(cudaMemcpy(d_C0.ptr, pre.C0.data.data(),
                             sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(h0),
                             cudaMemcpyHostToDevice),
                  "Memcpy C0");
        CudaCheck(cudaMemcpy(d_norm0.ptr, pre.norm0.data(),
                             sizeof(float) * static_cast<std::size_t>(h0),
                             cudaMemcpyHostToDevice),
                  "Memcpy norm0");
        CudaCheck(cudaMemcpy(d_Gs.ptr, pre.G_small.data.data(),
                             sizeof(float) * static_cast<std::size_t>(Hs) * static_cast<std::size_t>(Hs),
                             cudaMemcpyHostToDevice),
                  "Memcpy G_small");
        // Upload h0×Hs then transpose to Hs×h0 for better memory coalescing (fixed cid, varying flat).
        d_G0S_src.Ensure(static_cast<std::size_t>(h0) * static_cast<std::size_t>(Hs));
        CudaCheck(cudaMemcpy(d_G0S_src.ptr, pre.G0S.data.data(),
                             sizeof(float) * static_cast<std::size_t>(h0) * static_cast<std::size_t>(Hs),
                             cudaMemcpyHostToDevice),
                  "Memcpy G0S(src)");
        {
            dim3 block(32, 8);
            dim3 grid((Hs + block.x - 1) / block.x, (h0 + block.y - 1) / block.y);
            TransposeG0S_H0xHs_To_HsXH0<<<grid, block>>>(h0, Hs, d_G0S_src.ptr, d_G0S_T.ptr);
            CudaCheck(cudaGetLastError(), "Launch TransposeG0S");
        }
        // Free temporary to keep peak device memory low.
        d_G0S_src.Reset();
        CudaCheck(cudaMemcpy(d_inv.ptr, pre.invnorm_small_flat.data(),
                             sizeof(float) * static_cast<std::size_t>(Hs),
                             cudaMemcpyHostToDevice),
                  "Memcpy invnorm_small");
        CudaCheck(cudaMemcpy(d_small_offsets.ptr, pre.small_offsets.data(),
                             sizeof(int) * static_cast<std::size_t>(m),
                             cudaMemcpyHostToDevice),
                  "Memcpy small_offsets");
        CudaCheck(cudaMemcpy(d_hvec.ptr, pre.h_vec.data(),
                             sizeof(int) * static_cast<std::size_t>(m),
                             cudaMemcpyHostToDevice),
                  "Memcpy hvec");
        precomp_uploaded = true;
    }

    void EnsureBatch(int n_in, bool need_xc_buffer = true) {
        n = n_in;
        if (need_xc_buffer) {
            d_xC.Ensure(static_cast<std::size_t>(Hs) * static_cast<std::size_t>(n));
        } else {
            d_xC.Reset();
        }
        d_xn2.Ensure(static_cast<std::size_t>(n));
        d_xC0.Ensure(static_cast<std::size_t>(n));
        d_cid.Ensure(static_cast<std::size_t>(n));

        d_B.Ensure(static_cast<std::size_t>(m - 1) * static_cast<std::size_t>(n));
        d_a.Ensure(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
        d_cost.Ensure(static_cast<std::size_t>(n));
        d_active.Ensure(static_cast<std::size_t>(n));
        d_changed.Ensure(static_cast<std::size_t>(n));
        d_B_cand.Ensure(static_cast<std::size_t>(m - 1) * static_cast<std::size_t>(n));
        d_a_cand.Ensure(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
        d_cost_cand.Ensure(static_cast<std::size_t>(n));
    }

    void UploadFixed(const std::vector<float>& xC0,
                     const ColMajorMatrix<float>& xC_small,
                     const std::vector<float>& X_norm2,
                     const std::vector<std::uint32_t>& cluster_id) {
        CudaCheck(cudaMemcpy(d_xC.ptr, xC_small.data.data(),
                             sizeof(float) * static_cast<std::size_t>(Hs) * static_cast<std::size_t>(n),
                             cudaMemcpyHostToDevice),
                  "Memcpy xC_small");
        xC_ptr = d_xC.ptr;
        CudaCheck(cudaMemcpy(d_xn2.ptr, X_norm2.data(),
                             sizeof(float) * static_cast<std::size_t>(n),
                             cudaMemcpyHostToDevice),
                  "Memcpy X_norm2");
        CudaCheck(cudaMemcpy(d_xC0.ptr, xC0.data(),
                             sizeof(float) * static_cast<std::size_t>(n),
                             cudaMemcpyHostToDevice),
                  "Memcpy xC0");
        CudaCheck(cudaMemcpy(d_cid.ptr, cluster_id.data(),
                             sizeof(std::uint32_t) * static_cast<std::size_t>(n),
                             cudaMemcpyHostToDevice),
                  "Memcpy cluster_id");
    }

    void UploadFixedDeviceXc(const std::vector<float>& xC0,
                             const DeviceMatF32View& xC_small_dev,
                             const std::vector<float>& X_norm2,
                             const std::vector<std::uint32_t>& cluster_id) {
        if (!xC_small_dev.ptr || xC_small_dev.rows != Hs || xC_small_dev.cols != n || xC_small_dev.ld != Hs) {
            throw std::runtime_error("LargeRootIcmCudaSession::UploadFixedDeviceXc: invalid xC_small_dev view.");
        }
        xC_ptr = xC_small_dev.ptr;
        CudaCheck(cudaMemcpy(d_xn2.ptr, X_norm2.data(),
                             sizeof(float) * static_cast<std::size_t>(n),
                             cudaMemcpyHostToDevice),
                  "Memcpy X_norm2");
        CudaCheck(cudaMemcpy(d_xC0.ptr, xC0.data(),
                             sizeof(float) * static_cast<std::size_t>(n),
                             cudaMemcpyHostToDevice),
                  "Memcpy xC0");
        CudaCheck(cudaMemcpy(d_cid.ptr, cluster_id.data(),
                             sizeof(std::uint32_t) * static_cast<std::size_t>(n),
                             cudaMemcpyHostToDevice),
                  "Memcpy cluster_id");
    }

    void UploadFixedDeviceXcNoXc0(const DeviceMatF32View& xC_small_dev,
                                  const std::vector<float>& X_norm2,
                                  const std::vector<std::uint32_t>& cluster_id) {
        if (!xC_small_dev.ptr || xC_small_dev.rows != Hs || xC_small_dev.cols != n || xC_small_dev.ld != Hs) {
            throw std::runtime_error("LargeRootIcmCudaSession::UploadFixedDeviceXcNoXc0: invalid xC_small_dev view.");
        }
        xC_ptr = xC_small_dev.ptr;
        CudaCheck(cudaMemcpy(d_xn2.ptr, X_norm2.data(),
                             sizeof(float) * static_cast<std::size_t>(n),
                             cudaMemcpyHostToDevice),
                  "Memcpy X_norm2");
        CudaCheck(cudaMemcpy(d_cid.ptr, cluster_id.data(),
                             sizeof(std::uint32_t) * static_cast<std::size_t>(n),
                             cudaMemcpyHostToDevice),
                  "Memcpy cluster_id");
    }

    void UploadB(const Code* B_host) {
        CudaCheck(cudaMemcpy(d_B.ptr, B_host,
                             sizeof(Code) * static_cast<std::size_t>(m - 1) * static_cast<std::size_t>(n),
                             cudaMemcpyHostToDevice),
                  "Memcpy B_small to device");
    }

    void DownloadAll(Code* B_host, float* a_host, float* cost_host) {
        CudaCheck(cudaMemcpy(B_host, d_B.ptr,
                             sizeof(Code) * static_cast<std::size_t>(m - 1) * static_cast<std::size_t>(n),
                             cudaMemcpyDeviceToHost),
                  "Memcpy B_small back");
        CudaCheck(cudaMemcpy(a_host, d_a.ptr,
                             sizeof(float) * static_cast<std::size_t>(m) * static_cast<std::size_t>(n),
                             cudaMemcpyDeviceToHost),
                  "Memcpy a back");
        CudaCheck(cudaMemcpy(cost_host, d_cost.ptr,
                             sizeof(float) * static_cast<std::size_t>(n),
                             cudaMemcpyDeviceToHost),
                  "Memcpy cost back");
    }

    void RunIcmDevice(bool use_abs,
                      int icm_iters,
                      Code* dB,
                      float* dA,
                      float* dCost,
                      std::mt19937* rng) {
        const int threads = 128;
        const int blocks = (n + threads - 1) / threads;
        if (m == 5) {
            SolveCostInitLargeRootFixed<5><<<blocks, threads>>>(
                h0, Hs, d_small_offsets.ptr,
                d_norm0.ptr, xC_ptr ? xC_ptr : d_xC.ptr, d_Gs.ptr, d_G0S_T.ptr,
                d_xC0.ptr, d_xn2.ptr,
                d_cid.ptr, dB, dA, dCost,
                d_active.ptr, d_changed.ptr, n);
        } else if (m == 10) {
            SolveCostInitLargeRootFixed<10><<<blocks, threads>>>(
                h0, Hs, d_small_offsets.ptr,
                d_norm0.ptr, xC_ptr ? xC_ptr : d_xC.ptr, d_Gs.ptr, d_G0S_T.ptr,
                d_xC0.ptr, d_xn2.ptr,
                d_cid.ptr, dB, dA, dCost,
                d_active.ptr, d_changed.ptr, n);
        } else {
            SolveCostInitLargeRoot<<<blocks, threads>>>(h0, Hs, m, d_small_offsets.ptr,
                                                        d_norm0.ptr, xC_ptr ? xC_ptr : d_xC.ptr, d_Gs.ptr, d_G0S_T.ptr,
                                                        d_xC0.ptr, d_xn2.ptr,
                                                        d_cid.ptr, dB, dA, dCost,
                                                        d_active.ptr, d_changed.ptr, n);
        }

        if (static_cast<int>(order_host.size()) != m) {
            order_host.resize(static_cast<std::size_t>(m));
        }
        // Match CPU semantics: start each RunIcmDevice call from the canonical 0..m-1 order.
        std::iota(order_host.begin(), order_host.end(), 0);
        for (int it = 0; it < icm_iters; ++it) {
            std::shuffle(order_host.begin(), order_host.end(), *rng);
            ClearChanged<<<blocks, threads>>>(n, d_changed.ptr);
            for (int oi = 0; oi < m; ++oi) {
                const int jlayer = order_host[oi];
                if (jlayer == 0) continue;
                dim3 grid_icm(n);
                dim3 block_icm(256);
                // Specialized fast paths for common m (5/10) with full hj=256 layers.
                if (m == 5) {
                    if (use_abs) {
                        IcmLayerLargeRootBlockFixed<true, 5><<<grid_icm, block_icm>>>(
                            d, h0, Hs, d_small_offsets.ptr, d_hvec.ptr, d_inv.ptr,
                            d_norm0.ptr, xC_ptr ? xC_ptr : d_xC.ptr, d_Gs.ptr, d_G0S_T.ptr,
                            d_xC0.ptr, d_xn2.ptr, d_cid.ptr, d_active.ptr, d_changed.ptr,
                            jlayer, dB, dA, dCost, n);
                    } else {
                        IcmLayerLargeRootBlockFixed<false, 5><<<grid_icm, block_icm>>>(
                            d, h0, Hs, d_small_offsets.ptr, d_hvec.ptr, d_inv.ptr,
                            d_norm0.ptr, xC_ptr ? xC_ptr : d_xC.ptr, d_Gs.ptr, d_G0S_T.ptr,
                            d_xC0.ptr, d_xn2.ptr, d_cid.ptr, d_active.ptr, d_changed.ptr,
                            jlayer, dB, dA, dCost, n);
                    }
                } else if (m == 10) {
                    if (use_abs) {
                        IcmLayerLargeRootBlockFixed<true, 10><<<grid_icm, block_icm>>>(
                            d, h0, Hs, d_small_offsets.ptr, d_hvec.ptr, d_inv.ptr,
                            d_norm0.ptr, xC_ptr ? xC_ptr : d_xC.ptr, d_Gs.ptr, d_G0S_T.ptr,
                            d_xC0.ptr, d_xn2.ptr, d_cid.ptr, d_active.ptr, d_changed.ptr,
                            jlayer, dB, dA, dCost, n);
                    } else {
                        IcmLayerLargeRootBlockFixed<false, 10><<<grid_icm, block_icm>>>(
                            d, h0, Hs, d_small_offsets.ptr, d_hvec.ptr, d_inv.ptr,
                            d_norm0.ptr, xC_ptr ? xC_ptr : d_xC.ptr, d_Gs.ptr, d_G0S_T.ptr,
                            d_xC0.ptr, d_xn2.ptr, d_cid.ptr, d_active.ptr, d_changed.ptr,
                            jlayer, dB, dA, dCost, n);
                    }
                } else {
                    if (use_abs) {
                        IcmLayerLargeRootBlock<true><<<grid_icm, block_icm>>>(
                            d, h0, Hs, m, d_small_offsets.ptr, d_hvec.ptr, d_inv.ptr,
                            d_norm0.ptr, xC_ptr ? xC_ptr : d_xC.ptr, d_Gs.ptr, d_G0S_T.ptr,
                            d_xC0.ptr, d_xn2.ptr, d_cid.ptr, d_active.ptr, d_changed.ptr,
                            jlayer, dB, dA, dCost, n);
                    } else {
                        IcmLayerLargeRootBlock<false><<<grid_icm, block_icm>>>(
                            d, h0, Hs, m, d_small_offsets.ptr, d_hvec.ptr, d_inv.ptr,
                            d_norm0.ptr, xC_ptr ? xC_ptr : d_xC.ptr, d_Gs.ptr, d_G0S_T.ptr,
                            d_xC0.ptr, d_xn2.ptr, d_cid.ptr, d_active.ptr, d_changed.ptr,
                            jlayer, dB, dA, dCost, n);
                    }
                }
            }
            UpdateActive<<<blocks, threads>>>(n, d_active.ptr, d_changed.ptr);
        }
        CudaCheck(cudaGetLastError(), "Launch ICM large-root loop");
    }

    void RunIcm(bool use_abs,
                int icm_iters,
                Code* B_host,
                float* a_host,
                float* cost_host,
                std::mt19937* rng) {
        UploadB(B_host);
        RunIcmDevice(use_abs, icm_iters, d_B.ptr, d_a.ptr, d_cost.ptr, rng);
        DownloadAll(B_host, a_host, cost_host);
    }
};

// CPU-side SplitMix64 utilities copied from icm.cpp (kept deterministic across thread counts).
struct SplitMix64 {
    std::uint64_t state = 0;
    explicit SplitMix64(std::uint64_t seed) : state(seed) {}

    std::uint64_t NextU64() {
        std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31U);
    }
};

inline std::uint32_t UniformBelow(SplitMix64* rng, std::uint32_t bound) {
    const std::uint64_t x = rng->NextU64();
#if defined(__CUDA_ARCH__)
    // Device compilation (nvcc): avoid non-standard __uint128_t on MSVC toollinkages.
    return static_cast<std::uint32_t>(__umul64hi(x, static_cast<std::uint64_t>(bound)));
#elif defined(_MSC_VER)
    unsigned __int64 hi = 0;
    (void)_umul128(static_cast<unsigned __int64>(x), static_cast<unsigned __int64>(bound), &hi);
    return static_cast<std::uint32_t>(hi);
#else
    const __uint128_t prod = static_cast<__uint128_t>(x) * static_cast<__uint128_t>(bound);
    return static_cast<std::uint32_t>(prod >> 64);
#endif
}

template <bool UseAbs>
void DynamicIcmWithIlsCudaImpl(const ColMajorMatrix<float>& X,
                               const Precomp& precomp,
                               const ColMajorMatrix<float>& xC,
                               int icm_iters,
                               int ils_iters,
                               int perturb_k,
                               std::uint32_t seed,
                               std::uint64_t sample_id_offset,
                               ColMajorMatrix<FullCode>* B,
                               ColMajorMatrix<float>* a,
                               std::vector<float>* X_norm2,
                               std::vector<float>* cost,
                               bool print_progress,
                               CudaIcmTiming* timing) {
    if (!B || !a || !X_norm2) {
        return;
    }
    static thread_local std::vector<float> cost_scratch;
    std::vector<float>* cost_out = cost ? cost : &cost_scratch;
    // Keep the CUDA path quiet; the high-level pipeline already logs per-block progress.
    // If progress printing is requested, fall back to the CPU implementation.
    if (print_progress) {
        if constexpr (UseAbs) {
            DynamicIcmWithIlsAbsNoNormal(X, precomp, xC, icm_iters, ils_iters, perturb_k,
                                         seed, B, a, X_norm2, cost_out, print_progress, sample_id_offset);
        } else {
            DynamicIcmWithIlsNoAbsNoNormal(X, precomp, xC, icm_iters, ils_iters, perturb_k,
                                           seed, B, a, X_norm2, cost_out, print_progress, sample_id_offset);
        }
        return;
    }
    const int n = B->cols;
    if (n <= 0) {
        return;
    }
    if (precomp.m <= 1 || precomp.m > kMaxM) {
        if constexpr (UseAbs) {
            DynamicIcmWithIlsAbsNoNormal(X, precomp, xC, icm_iters, ils_iters, perturb_k,
                                         seed, B, a, X_norm2, cost, print_progress, sample_id_offset);
        } else {
            DynamicIcmWithIlsNoAbsNoNormal(X, precomp, xC, icm_iters, ils_iters, perturb_k,
                                           seed, B, a, X_norm2, cost, print_progress, sample_id_offset);
        }
        return;
    }
    if (static_cast<int>(X_norm2->size()) != n) {
        ComputeXNorm2(X, X_norm2);
    }

    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));

    static thread_local FullIcmCudaSession session;
    session.EnsurePrecomp(precomp);
    session.EnsureBatch(n);
    if (timing) {
        Timer tt;
        session.UploadXcAndNorms(xC, *X_norm2);
        timing->upload_fixed += tt.ElapsedSeconds();
    } else {
        session.UploadXcAndNorms(xC, *X_norm2);
    }

    a->rows = precomp.m;
    a->cols = n;
    a->data.resize(static_cast<std::size_t>(precomp.m) * static_cast<std::size_t>(n));
    cost_out->resize(static_cast<std::size_t>(n));

    if (timing) {
        Timer tt;
        session.UploadB(B->data.data());
        timing->upload_B += tt.ElapsedSeconds();
        timing->icm_init += session.ev.MeasureSeconds([&] {
            session.RunIcmDevice(/*use_abs=*/UseAbs, icm_iters,
                                 session.d_B.ptr, session.d_a.ptr, session.d_cost.ptr, &rng);
        });
    } else {
        session.UploadB(B->data.data());
        session.RunIcmDevice(/*use_abs=*/UseAbs, icm_iters,
                             session.d_B.ptr, session.d_a.ptr, session.d_cost.ptr, &rng);
    }

    if (ils_iters <= 0 || perturb_k <= 0 || precomp.m <= 1) {
        if (timing) {
            Timer tt;
            session.DownloadAll(B->data.data(), a->data.data(), cost->data());
            timing->download += tt.ElapsedSeconds();
        } else {
            session.DownloadAll(B->data.data(), a->data.data(), cost->data());
        }
        return;
    }

    const int ksel = std::min(perturb_k, precomp.m);
    const int threads = 128;
    const int blocks = (n + threads - 1) / threads;

    for (int outer = 0; outer < ils_iters; ++outer) {
        if (timing) {
            timing->ils_copy_perturb += session.ev.MeasureSeconds([&] {
                CopyAndPerturbFullCodes<<<blocks, threads>>>(precomp.m,
                                                            session.d_hvec.ptr,
                                                            seed,
                                                            outer + 1,
                                                            sample_id_offset,
                                                            ksel,
                                                            session.d_B.ptr,
                                                            session.d_B_cand.ptr,
                                                            n);
                CudaCheck(cudaGetLastError(), "Launch CopyAndPerturbFullCodes");
            });
            timing->ils_icm += session.ev.MeasureSeconds([&] {
                session.RunIcmDevice(/*use_abs=*/UseAbs, icm_iters,
                                     session.d_B_cand.ptr,
                                     session.d_a_cand.ptr,
                                     session.d_cost_cand.ptr,
                                     &rng);
            });
            timing->ils_accept += session.ev.MeasureSeconds([&] {
                AcceptIfBetterFull<<<blocks, threads>>>(precomp.m,
                                                       session.d_B_cand.ptr,
                                                       session.d_a_cand.ptr,
                                                       session.d_cost_cand.ptr,
                                                       session.d_B.ptr,
                                                       session.d_a.ptr,
                                                       session.d_cost.ptr,
                                                       n);
                CudaCheck(cudaGetLastError(), "Launch AcceptIfBetterFull");
            });
        } else {
            CopyAndPerturbFullCodes<<<blocks, threads>>>(precomp.m,
                                                        session.d_hvec.ptr,
                                                        seed,
                                                        outer + 1,
                                                        sample_id_offset,
                                                        ksel,
                                                        session.d_B.ptr,
                                                        session.d_B_cand.ptr,
                                                        n);

            session.RunIcmDevice(/*use_abs=*/UseAbs, icm_iters,
                                 session.d_B_cand.ptr,
                                 session.d_a_cand.ptr,
                                 session.d_cost_cand.ptr,
                                 &rng);

            AcceptIfBetterFull<<<blocks, threads>>>(precomp.m,
                                                   session.d_B_cand.ptr,
                                                   session.d_a_cand.ptr,
                                                   session.d_cost_cand.ptr,
                                                   session.d_B.ptr,
                                                   session.d_a.ptr,
                                                   session.d_cost.ptr,
                                                   n);
        }
    }

    if (timing) {
        Timer tt;
        if (cost) {
            session.DownloadAll(B->data.data(), a->data.data(), cost_out->data());
        } else {
            // Skip downloading costs when not needed for metrics.
            CudaCheck(cudaMemcpy(B->data.data(), session.d_B.ptr,
                                 sizeof(FullCode) * static_cast<std::size_t>(precomp.m) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      "Memcpy B back (no cost)");
            CudaCheck(cudaMemcpy(a->data.data(), session.d_a.ptr,
                                 sizeof(float) * static_cast<std::size_t>(precomp.m) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      "Memcpy a back (no cost)");
        }
        timing->download += tt.ElapsedSeconds();
    } else {
        if (cost) {
            session.DownloadAll(B->data.data(), a->data.data(), cost_out->data());
        } else {
            CudaCheck(cudaMemcpy(B->data.data(), session.d_B.ptr,
                                 sizeof(FullCode) * static_cast<std::size_t>(precomp.m) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      "Memcpy B back (no cost)");
            CudaCheck(cudaMemcpy(a->data.data(), session.d_a.ptr,
                                 sizeof(float) * static_cast<std::size_t>(precomp.m) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      "Memcpy a back (no cost)");
        }
    }
}

template <bool UseAbs>
void DynamicIcmWithIlsCudaImplDeviceXc(const ColMajorMatrix<float>& X,
                                      const Precomp& precomp,
                                      const DeviceMatF32View& xC_dev,
                                      int icm_iters,
                                      int ils_iters,
                                      int perturb_k,
                                      std::uint32_t seed,
                                      std::uint64_t sample_id_offset,
                                      ColMajorMatrix<FullCode>* B,
                                      ColMajorMatrix<float>* a,
                                      std::vector<float>* X_norm2,
                                      std::vector<float>* cost,
                                      bool print_progress,
                                      CudaIcmTiming* timing) {
    if (!B || !a || !X_norm2) {
        return;
    }
    static thread_local std::vector<float> cost_scratch;
    std::vector<float>* cost_out = cost ? cost : &cost_scratch;
    const int n = B->cols;
    if (n <= 0) {
        return;
    }
    if (print_progress) {
        throw std::runtime_error("DynamicIcmWithIlsCudaImplDeviceXc: print_progress is not supported.");
    }
    if (!xC_dev.ptr || xC_dev.rows != precomp.H || xC_dev.cols != n || xC_dev.ld != precomp.H) {
        throw std::runtime_error("DynamicIcmWithIlsCudaImplDeviceXc: invalid xC_dev view.");
    }
    if (precomp.m <= 1 || precomp.m > kMaxM) {
        throw std::runtime_error("DynamicIcmWithIlsCudaImplDeviceXc: unsupported m for CUDA device-xC path.");
    }
    if (static_cast<int>(X_norm2->size()) != n) {
        ComputeXNorm2(X, X_norm2);
    }

    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));

    static thread_local FullIcmCudaSession session;
    session.EnsurePrecomp(precomp);
    session.EnsureBatch(n);
    if (timing) {
        Timer tt;
        session.UploadFixedDeviceXc(xC_dev, *X_norm2);
        timing->upload_fixed += tt.ElapsedSeconds();
    } else {
        session.UploadFixedDeviceXc(xC_dev, *X_norm2);
    }

    a->rows = precomp.m;
    a->cols = n;
    a->data.resize(static_cast<std::size_t>(precomp.m) * static_cast<std::size_t>(n));
    cost_out->resize(static_cast<std::size_t>(n));

    if (timing) {
        Timer tt;
        session.UploadB(B->data.data());
        timing->upload_B += tt.ElapsedSeconds();
        timing->icm_init += session.ev.MeasureSeconds([&] {
            session.RunIcmDevice(/*use_abs=*/UseAbs, icm_iters,
                                 session.d_B.ptr, session.d_a.ptr, session.d_cost.ptr, &rng);
        });
    } else {
        session.UploadB(B->data.data());
        session.RunIcmDevice(/*use_abs=*/UseAbs, icm_iters,
                             session.d_B.ptr, session.d_a.ptr, session.d_cost.ptr, &rng);
    }

    if (ils_iters <= 0 || perturb_k <= 0 || precomp.m <= 1) {
        if (timing) {
            Timer tt;
            session.DownloadAll(B->data.data(), a->data.data(), cost_out->data());
            timing->download += tt.ElapsedSeconds();
        } else {
            session.DownloadAll(B->data.data(), a->data.data(), cost_out->data());
        }
        return;
    }

    const int ksel = std::min(perturb_k, precomp.m);
    const int threads = 128;
    const int blocks = (n + threads - 1) / threads;

    for (int outer = 0; outer < ils_iters; ++outer) {
        if (timing) {
            timing->ils_copy_perturb += session.ev.MeasureSeconds([&] {
                CopyAndPerturbFullCodes<<<blocks, threads>>>(precomp.m,
                                                            session.d_hvec.ptr,
                                                            seed,
                                                            outer + 1,
                                                            sample_id_offset,
                                                            ksel,
                                                            session.d_B.ptr,
                                                            session.d_B_cand.ptr,
                                                            n);
                CudaCheck(cudaGetLastError(), "Launch CopyAndPerturbFullCodes");
            });
            timing->ils_icm += session.ev.MeasureSeconds([&] {
                session.RunIcmDevice(/*use_abs=*/UseAbs, icm_iters,
                                     session.d_B_cand.ptr,
                                     session.d_a_cand.ptr,
                                     session.d_cost_cand.ptr,
                                     &rng);
            });
            timing->ils_accept += session.ev.MeasureSeconds([&] {
                AcceptIfBetterFull<<<blocks, threads>>>(precomp.m,
                                                       session.d_B_cand.ptr,
                                                       session.d_a_cand.ptr,
                                                       session.d_cost_cand.ptr,
                                                       session.d_B.ptr,
                                                       session.d_a.ptr,
                                                       session.d_cost.ptr,
                                                       n);
                CudaCheck(cudaGetLastError(), "Launch AcceptIfBetterFull");
            });
        } else {
            CopyAndPerturbFullCodes<<<blocks, threads>>>(precomp.m,
                                                        session.d_hvec.ptr,
                                                        seed,
                                                        outer + 1,
                                                        sample_id_offset,
                                                        ksel,
                                                        session.d_B.ptr,
                                                        session.d_B_cand.ptr,
                                                        n);
            CudaCheck(cudaGetLastError(), "Launch CopyAndPerturbFullCodes");

            session.RunIcmDevice(/*use_abs=*/UseAbs, icm_iters,
                                 session.d_B_cand.ptr,
                                 session.d_a_cand.ptr,
                                 session.d_cost_cand.ptr,
                                 &rng);

            AcceptIfBetterFull<<<blocks, threads>>>(precomp.m,
                                                   session.d_B_cand.ptr,
                                                   session.d_a_cand.ptr,
                                                   session.d_cost_cand.ptr,
                                                   session.d_B.ptr,
                                                   session.d_a.ptr,
                                                   session.d_cost.ptr,
                                                   n);
            CudaCheck(cudaGetLastError(), "Launch AcceptIfBetterFull");
        }
    }

    if (timing) {
        Timer tt;
        if (cost) {
            session.DownloadAll(B->data.data(), a->data.data(), cost_out->data());
        } else {
            CudaCheck(cudaMemcpy(B->data.data(), session.d_B.ptr,
                                 sizeof(FullCode) * static_cast<std::size_t>(precomp.m) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      "Memcpy B back (no cost)");
            CudaCheck(cudaMemcpy(a->data.data(), session.d_a.ptr,
                                 sizeof(float) * static_cast<std::size_t>(precomp.m) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      "Memcpy a back (no cost)");
        }
        timing->download += tt.ElapsedSeconds();
    } else {
        if (cost) {
            session.DownloadAll(B->data.data(), a->data.data(), cost_out->data());
        } else {
            CudaCheck(cudaMemcpy(B->data.data(), session.d_B.ptr,
                                 sizeof(FullCode) * static_cast<std::size_t>(precomp.m) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      "Memcpy B back (no cost)");
            CudaCheck(cudaMemcpy(a->data.data(), session.d_a.ptr,
                                 sizeof(float) * static_cast<std::size_t>(precomp.m) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      "Memcpy a back (no cost)");
        }
    }
}

template <bool UseAbs>
void DynamicIcmWithIlsLargeRootCudaImpl(const ColMajorMatrix<float>& X,
                                        const PrecompLargeRoot& pre,
                                        const ColMajorMatrix<float>& xC_small,
                                        int icm_iters,
                                        int ils_iters,
                                        int perturb_k,
                                        std::uint32_t seed,
                                        std::uint64_t sample_id_offset,
                                        const std::vector<std::uint32_t>& cluster_id,
                                        ColMajorMatrix<Code>* B_small,
                                        ColMajorMatrix<float>* a,
                                        std::vector<float>* X_norm2,
                                        std::vector<float>* cost,
                                        bool print_progress,
                                        CudaIcmTiming* timing) {
    if (!B_small || !a || !X_norm2) {
        return;
    }
    static thread_local std::vector<float> cost_scratch;
    std::vector<float>* cost_out = cost ? cost : &cost_scratch;
    const int n = B_small->cols;
    if (n <= 0) {
        return;
    }
    if (pre.m <= 1 || pre.m > kMaxM) {
        if constexpr (UseAbs) {
            DynamicIcmWithIlsAbsNoNormalLargeRoot(X, pre, xC_small, icm_iters, ils_iters, perturb_k,
                                                 seed, cluster_id, B_small, a, X_norm2, cost,
                                                 print_progress, sample_id_offset);
        } else {
            DynamicIcmWithIlsNoAbsNoNormalLargeRoot(X, pre, xC_small, icm_iters, ils_iters, perturb_k,
                                                   seed, cluster_id, B_small, a, X_norm2, cost,
                                                   print_progress, sample_id_offset);
        }
        return;
    }
    if (static_cast<int>(X_norm2->size()) != n) {
        ComputeXNorm2(X, X_norm2);
    }

    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));

    auto DotScalar = [](const float* a, const float* b, int d) -> float {
        float s = 0.0f;
        for (int i = 0; i < d; ++i) {
            s += a[i] * b[i];
        }
        return s;
    };

    static thread_local LargeRootIcmCudaSession session;
    session.EnsurePrecomp(pre);
    session.EnsureBatch(n);
    static thread_local std::vector<float> xC0_host;
    xC0_host.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const std::uint32_t cid = cluster_id[static_cast<std::size_t>(i)];
        if (cid >= static_cast<std::uint32_t>(pre.h_vec.empty() ? 0 : pre.h_vec[0])) {
            xC0_host[static_cast<std::size_t>(i)] = 0.0f;
            continue;
        }
        const float* x = X.data.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(pre.d);
        const float* c0 = pre.C0.data.data() + static_cast<std::size_t>(cid) * static_cast<std::size_t>(pre.d);
        xC0_host[static_cast<std::size_t>(i)] = DotScalar(c0, x, pre.d);
    }

    if (timing) {
        Timer tt;
        session.UploadFixed(xC0_host, xC_small, *X_norm2, cluster_id);
        timing->upload_fixed += tt.ElapsedSeconds();
    } else {
        session.UploadFixed(xC0_host, xC_small, *X_norm2, cluster_id);
    }

    a->rows = pre.m;
    a->cols = n;
    a->data.resize(static_cast<std::size_t>(pre.m) * static_cast<std::size_t>(n));
    cost_out->resize(static_cast<std::size_t>(n));

    if (timing) {
        Timer tt;
        session.UploadB(B_small->data.data());
        timing->upload_B += tt.ElapsedSeconds();
        timing->icm_init += session.ev.MeasureSeconds([&] {
            session.RunIcmDevice(/*use_abs=*/UseAbs,
                                 icm_iters,
                                 session.d_B.ptr,
                                 session.d_a.ptr,
                                 session.d_cost.ptr,
                                 &rng);
        });
    } else {
        session.UploadB(B_small->data.data());
        session.RunIcmDevice(/*use_abs=*/UseAbs,
                             icm_iters,
                             session.d_B.ptr,
                             session.d_a.ptr,
                             session.d_cost.ptr,
                             &rng);
    }

    if (ils_iters <= 0 || perturb_k <= 0 || pre.m <= 1) {
        if (timing) {
            Timer tt;
            if (cost) {
                session.DownloadAll(B_small->data.data(), a->data.data(), cost_out->data());
            } else {
                CudaCheck(cudaMemcpy(B_small->data.data(), session.d_B.ptr,
                                     sizeof(Code) * static_cast<std::size_t>(pre.m - 1) * static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "Memcpy B_small back (no cost)");
                CudaCheck(cudaMemcpy(a->data.data(), session.d_a.ptr,
                                     sizeof(float) * static_cast<std::size_t>(pre.m) * static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "Memcpy a back (no cost)");
            }
            timing->download += tt.ElapsedSeconds();
        } else {
            if (cost) {
                session.DownloadAll(B_small->data.data(), a->data.data(), cost_out->data());
            } else {
                CudaCheck(cudaMemcpy(B_small->data.data(), session.d_B.ptr,
                                     sizeof(Code) * static_cast<std::size_t>(pre.m - 1) * static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "Memcpy B_small back (no cost)");
                CudaCheck(cudaMemcpy(a->data.data(), session.d_a.ptr,
                                     sizeof(float) * static_cast<std::size_t>(pre.m) * static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "Memcpy a back (no cost)");
            }
        }
        return;
    }

    const int ksel = std::min(perturb_k, pre.m);
    const int threads = 128;
    const int blocks = (n + threads - 1) / threads;

    for (int outer = 0; outer < ils_iters; ++outer) {
        if (timing) {
            timing->ils_copy_perturb += session.ev.MeasureSeconds([&] {
                if (pre.m == 5) {
                    CopyAndPerturbSmallCodesFixed<5><<<blocks, threads>>>(session.d_hvec.ptr,
                                                                         seed,
                                                                         outer + 1,
                                                                         sample_id_offset,
                                                                         ksel,
                                                                         session.d_B.ptr,
                                                                         session.d_B_cand.ptr,
                                                                         n);
                } else if (pre.m == 10) {
                    CopyAndPerturbSmallCodesFixed<10><<<blocks, threads>>>(session.d_hvec.ptr,
                                                                          seed,
                                                                          outer + 1,
                                                                          sample_id_offset,
                                                                          ksel,
                                                                          session.d_B.ptr,
                                                                          session.d_B_cand.ptr,
                                                                          n);
                } else {
                    CopyAndPerturbSmallCodes<<<blocks, threads>>>(pre.m,
                                                                 session.d_hvec.ptr,
                                                                 seed,
                                                                 outer + 1,
                                                                 sample_id_offset,
                                                                 ksel,
                                                                 session.d_B.ptr,
                                                                 session.d_B_cand.ptr,
                                                                 n);
                }
                CudaCheck(cudaGetLastError(), "Launch CopyAndPerturbSmallCodes");
            });
            timing->ils_icm += session.ev.MeasureSeconds([&] {
                session.RunIcmDevice(/*use_abs=*/UseAbs,
                                     icm_iters,
                                     session.d_B_cand.ptr,
                                     session.d_a_cand.ptr,
                                     session.d_cost_cand.ptr,
                                     &rng);
            });
            timing->ils_accept += session.ev.MeasureSeconds([&] {
                if (pre.m == 5) {
                    AcceptIfBetterSmallFixed<5><<<blocks, threads>>>(session.d_B_cand.ptr,
                                                                    session.d_a_cand.ptr,
                                                                    session.d_cost_cand.ptr,
                                                                    session.d_B.ptr,
                                                                    session.d_a.ptr,
                                                                    session.d_cost.ptr,
                                                                    n);
                } else if (pre.m == 10) {
                    AcceptIfBetterSmallFixed<10><<<blocks, threads>>>(session.d_B_cand.ptr,
                                                                     session.d_a_cand.ptr,
                                                                     session.d_cost_cand.ptr,
                                                                     session.d_B.ptr,
                                                                     session.d_a.ptr,
                                                                     session.d_cost.ptr,
                                                                     n);
                } else {
                    AcceptIfBetterSmall<<<blocks, threads>>>(pre.m,
                                                            session.d_B_cand.ptr,
                                                            session.d_a_cand.ptr,
                                                            session.d_cost_cand.ptr,
                                                            session.d_B.ptr,
                                                            session.d_a.ptr,
                                                            session.d_cost.ptr,
                                                            n);
                }
                CudaCheck(cudaGetLastError(), "Launch AcceptIfBetterSmall");
            });
        } else {
            if (pre.m == 5) {
                CopyAndPerturbSmallCodesFixed<5><<<blocks, threads>>>(session.d_hvec.ptr,
                                                                     seed,
                                                                     outer + 1,
                                                                     sample_id_offset,
                                                                     ksel,
                                                                     session.d_B.ptr,
                                                                     session.d_B_cand.ptr,
                                                                     n);
            } else if (pre.m == 10) {
                CopyAndPerturbSmallCodesFixed<10><<<blocks, threads>>>(session.d_hvec.ptr,
                                                                      seed,
                                                                      outer + 1,
                                                                      sample_id_offset,
                                                                      ksel,
                                                                      session.d_B.ptr,
                                                                      session.d_B_cand.ptr,
                                                                      n);
            } else {
                CopyAndPerturbSmallCodes<<<blocks, threads>>>(pre.m,
                                                             session.d_hvec.ptr,
                                                             seed,
                                                             outer + 1,
                                                             sample_id_offset,
                                                             ksel,
                                                             session.d_B.ptr,
                                                             session.d_B_cand.ptr,
                                                             n);
            }

            session.RunIcmDevice(/*use_abs=*/UseAbs,
                                 icm_iters,
                                 session.d_B_cand.ptr,
                                 session.d_a_cand.ptr,
                                 session.d_cost_cand.ptr,
                                 &rng);

            if (pre.m == 5) {
                AcceptIfBetterSmallFixed<5><<<blocks, threads>>>(session.d_B_cand.ptr,
                                                                session.d_a_cand.ptr,
                                                                session.d_cost_cand.ptr,
                                                                session.d_B.ptr,
                                                                session.d_a.ptr,
                                                                session.d_cost.ptr,
                                                                n);
            } else if (pre.m == 10) {
                AcceptIfBetterSmallFixed<10><<<blocks, threads>>>(session.d_B_cand.ptr,
                                                                 session.d_a_cand.ptr,
                                                                 session.d_cost_cand.ptr,
                                                                 session.d_B.ptr,
                                                                 session.d_a.ptr,
                                                                 session.d_cost.ptr,
                                                                 n);
            } else {
                AcceptIfBetterSmall<<<blocks, threads>>>(pre.m,
                                                        session.d_B_cand.ptr,
                                                        session.d_a_cand.ptr,
                                                        session.d_cost_cand.ptr,
                                                        session.d_B.ptr,
                                                        session.d_a.ptr,
                                                        session.d_cost.ptr,
                                                        n);
            }
        }
    }

    if (timing) {
        Timer tt;
        if (cost) {
            session.DownloadAll(B_small->data.data(), a->data.data(), cost_out->data());
        } else {
            CudaCheck(cudaMemcpy(B_small->data.data(), session.d_B.ptr,
                                 sizeof(Code) * static_cast<std::size_t>(pre.m - 1) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      "Memcpy B_small back (no cost)");
            CudaCheck(cudaMemcpy(a->data.data(), session.d_a.ptr,
                                 sizeof(float) * static_cast<std::size_t>(pre.m) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      "Memcpy a back (no cost)");
        }
        timing->download += tt.ElapsedSeconds();
    } else {
        if (cost) {
            session.DownloadAll(B_small->data.data(), a->data.data(), cost_out->data());
        } else {
            CudaCheck(cudaMemcpy(B_small->data.data(), session.d_B.ptr,
                                 sizeof(Code) * static_cast<std::size_t>(pre.m - 1) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      "Memcpy B_small back (no cost)");
            CudaCheck(cudaMemcpy(a->data.data(), session.d_a.ptr,
                                 sizeof(float) * static_cast<std::size_t>(pre.m) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      "Memcpy a back (no cost)");
        }
    }
}

}  // namespace
