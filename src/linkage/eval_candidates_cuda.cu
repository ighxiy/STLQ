#include "stlq/linkage/eval_candidates_cuda.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "stlq/linkage/linkage_encode_cuda.h"

namespace stlq {

namespace {

std::atomic<bool> g_profile_enabled{false};
std::atomic<int> g_chunk_max_pairs{0};
std::atomic<int> g_mem_budget_mb{0};
std::atomic<int> g_async_pinned_budget_mb{0};
std::atomic<std::uint64_t> g_calls{0};
std::atomic<std::int64_t> g_h2d_ns{0};
std::atomic<std::int64_t> g_residual_ns{0};
std::atomic<std::int64_t> g_encode_ns{0};
std::atomic<std::int64_t> g_reduce_ns{0};
std::atomic<std::int64_t> g_d2h_ns{0};
std::atomic<std::int64_t> g_total_ns{0};

std::atomic<std::uint64_t> g_d2h_bytes{0};
std::atomic<std::uint64_t> g_shape_calls{0};
std::atomic<int> g_max_B{0};
std::atomic<int> g_max_npairs{0};
std::atomic<std::uint64_t> g_hist_B[12] = {};
std::atomic<std::uint64_t> g_hist_npairs[12] = {};

std::atomic<int> g_peak_d{0};
std::atomic<int> g_peak_m{0};
std::atomic<int> g_peak_H{0};
std::atomic<int> g_peak_B{0};
std::atomic<int> g_peak_npairs{0};
std::atomic<int> g_peak_rfull_cols{0};
std::atomic<std::uint64_t> g_peak_bytes_per_ctx_est{0};

inline void ThrowIf(cudaError_t st, const char* what) {
    if (st == cudaSuccess) return;
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(st));
}

inline void AtomicMax(std::atomic<int>* dst, int v) {
    int cur = dst->load(std::memory_order_relaxed);
    while (v > cur && !dst->compare_exchange_weak(cur, v, std::memory_order_relaxed)) {
    }
}

inline void AtomicMaxU64(std::atomic<std::uint64_t>* dst, std::uint64_t v) {
    std::uint64_t cur = dst->load(std::memory_order_relaxed);
    while (v > cur && !dst->compare_exchange_weak(cur, v, std::memory_order_relaxed)) {
    }
}

struct HostPinnedBuf {
    void* ptr = nullptr;
    std::size_t bytes = 0;

    ~HostPinnedBuf() { Reset(); }

    void Reset() {
        if (ptr) {
            cudaFreeHost(ptr);
            ptr = nullptr;
        }
        bytes = 0;
    }

    void Ensure(std::size_t need_bytes) {
        if (need_bytes <= bytes) return;
        Reset();
        ThrowIf(cudaHostAlloc(&ptr, need_bytes, cudaHostAllocPortable), "cudaHostAlloc");
        bytes = need_bytes;
    }

    [[nodiscard]] std::uint8_t* u8() const { return static_cast<std::uint8_t*>(ptr); }
};

std::uint64_t EstimateBytesPerCtxWithCodeBytes(int d,
                                               int H,
                                               int m,
                                               int B,
                                               int npairs,
                                               int rfull_cols,
                                               int code_bytes) {
    const std::uint64_t u64_d = static_cast<std::uint64_t>(std::max(0, d));
    const std::uint64_t u64_H = static_cast<std::uint64_t>(std::max(0, H));
    const std::uint64_t u64_m = static_cast<std::uint64_t>(std::max(0, m));
    const std::uint64_t u64_B = static_cast<std::uint64_t>(std::max(0, B));
    const std::uint64_t u64_n = static_cast<std::uint64_t>(std::max(0, npairs));
    const std::uint64_t u64_cols = static_cast<std::uint64_t>(std::max(0, rfull_cols));
    const std::uint64_t u64_code_bytes = static_cast<std::uint64_t>(std::max(1, code_bytes));

    std::uint64_t bytes = 0;

    // Eval workspace.
    bytes += u64_d * u64_cols * sizeof(float);           // d_R_full
    bytes += u64_d * u64_n * sizeof(float);              // d_Rp
    bytes += u64_n * sizeof(float);                      // d_norm2
    bytes += u64_n * sizeof(float);                      // d_cost
    bytes += u64_m * u64_n * u64_code_bytes;             // d_B (FullCode or u32)
    bytes += u64_m * u64_n * sizeof(float);              // d_a
    bytes += u64_d * u64_B * sizeof(float);              // d_xblock
    bytes += u64_n * sizeof(int);                        // d_cand
    bytes += u64_n * sizeof(int);                        // d_pair_node
    bytes += (u64_B + 1u) * sizeof(int);                 // d_offsets
    bytes += u64_B * sizeof(std::uint64_t);              // d_node_sample_id_base
    bytes += u64_n * sizeof(std::uint64_t);              // d_sample_ids
    bytes += u64_n * sizeof(int);                        // d_forced_root (forced-root path only; worst-case)
    bytes += u64_B * sizeof(float);                      // d_best_cost_batch
    bytes += u64_B * sizeof(int);                        // d_best_pair_batch
    bytes += u64_B * sizeof(int);                        // d_best_parent_batch
    bytes += u64_B * u64_m * u64_code_bytes;             // d_best_B_batch (FullCode or u32)
    bytes += u64_B * u64_m * sizeof(float);              // d_best_a_batch
    // d_best_pack: packed best parent/cost/B/a for B nodes (worst-case layout).
    bytes += u64_B * (16u + u64_m * (u64_code_bytes + sizeof(float)));

    // LinkageEncodeBatchCudaWorkspace (rough, based on batch sizes).
    // NOTE: precomp buffers (C/G/offsets/...) are now shared across CUDA ctxs and excluded here.
    bytes += u64_H * u64_n * sizeof(float);              // d_xC
    bytes += u64_H * u64_n * sizeof(float);              // d_rC
    bytes += u64_n * sizeof(std::uint8_t);               // d_active
    bytes += u64_n * sizeof(std::uint8_t);               // d_changed
    bytes += u64_m * u64_n * u64_code_bytes;             // d_B_cand (FullCode or u32)
    bytes += u64_m * u64_n * sizeof(float);              // d_a_cand
    bytes += u64_n * sizeof(float);                      // d_cost_cand

    return bytes;
}

std::uint64_t EstimateBytesPerCtx(int d, int H, int m, int B, int npairs, int rfull_cols) {
    return EstimateBytesPerCtxWithCodeBytes(d, H, m, B, npairs, rfull_cols, static_cast<int>(sizeof(FullCode)));
}

inline int Bucket12Pow2ish(int x) {
    if (x <= 1) return 0;
    if (x <= 3) return 1;
    if (x <= 7) return 2;
    if (x <= 15) return 3;
    if (x <= 31) return 4;
    if (x <= 63) return 5;
    if (x <= 127) return 6;
    if (x <= 255) return 7;
    if (x <= 511) return 8;
    if (x <= 1023) return 9;
    if (x <= 2047) return 10;
    return 11;
}

inline void ProfileShapeAndD2HBytes(int B, int npairs, int m, bool profile) {
    if (!profile) return;
    g_shape_calls.fetch_add(1, std::memory_order_relaxed);
    // Best-effort D2H bytes: parent(int) + cost(float) per node, plus best codes/coeffs per node.
    const std::uint64_t u64_B = static_cast<std::uint64_t>(std::max(0, B));
    const std::uint64_t u64_m = static_cast<std::uint64_t>(std::max(0, m));
    const std::uint64_t bytes =
        u64_B * sizeof(int) +
        u64_B * sizeof(float) +
        u64_B * u64_m * sizeof(FullCode) +
        u64_B * u64_m * sizeof(float);
    g_d2h_bytes.fetch_add(bytes, std::memory_order_relaxed);
    AtomicMax(&g_max_B, B);
    AtomicMax(&g_max_npairs, npairs);
    g_hist_B[Bucket12Pow2ish(B)].fetch_add(1, std::memory_order_relaxed);
    g_hist_npairs[Bucket12Pow2ish(npairs)].fetch_add(1, std::memory_order_relaxed);
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
        }
        cap = 0;
    }

	    void Ensure(std::size_t n) {
	        if (n <= cap) return;
	        const std::size_t old_cap = cap;
	        std::size_t new_cap = (old_cap > 0) ? old_cap : n;
	        while (new_cap < n) {
	            const std::size_t grow = (new_cap >> 1) + 1;
	            const std::size_t next = new_cap + grow;
	            new_cap = (next > new_cap) ? next : n;
	        }
	        Reset();
	        ThrowIf(cudaMalloc(&ptr, sizeof(T) * new_cap), "cudaMalloc");
	        cap = new_cap;
	    }
	};

template <typename T>
struct PinnedBuf {
    T* ptr = nullptr;
    std::size_t cap = 0;

    ~PinnedBuf() { Reset(); }

    void Reset() {
        if (ptr) {
            cudaFreeHost(ptr);
            ptr = nullptr;
        }
        cap = 0;
    }

	    void Ensure(std::size_t n) {
	        if (n <= cap) return;
	        const std::size_t old_cap = cap;
	        std::size_t new_cap = (old_cap > 0) ? old_cap : n;
	        while (new_cap < n) {
	            const std::size_t grow = (new_cap >> 1) + 1;
	            const std::size_t next = new_cap + grow;
	            new_cap = (next > new_cap) ? next : n;
	        }
	        Reset();
	        ThrowIf(cudaMallocHost(&ptr, sizeof(T) * new_cap), "cudaMallocHost");
	        cap = new_cap;
	    }
	};

__global__ void ResidualInplaceAndNorm2(const float* __restrict xi,
                                       float* __restrict Rp_residual,  // d×Kp, overwritten
                                       float* __restrict norm2,        // Kp
                                       int d,
                                       int Kp) {
    // One block per column.
    const int col = blockIdx.x;
    if (col >= Kp) return;
    const int r = threadIdx.x;
    __shared__ float sh[256];

    float* col_ptr = Rp_residual + static_cast<std::size_t>(col) * static_cast<std::size_t>(d);
    float sum = 0.0f;
    for (int rr = r; rr < d; rr += blockDim.x) {
        const float v = xi[rr] - col_ptr[rr];
        col_ptr[rr] = v;
        sum += v * v;
    }
    sh[r] = sum;
    __syncthreads();

    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (r < off) {
            sh[r] += sh[r + off];
        }
        __syncthreads();
    }
    if (r == 0) {
        norm2[col] = sh[0];
    }
}

__global__ void GatherResidualAndNorm2FromRfull(const float* __restrict xi,
                                                const float* __restrict R_full,   // d×n_cols
                                                const int* __restrict cand,       // Kp
                                                float* __restrict residual,       // d×Kp (col-major)
                                                float* __restrict norm2,          // Kp
                                                int d,
                                                int Kp) {
    // One block per candidate column.
    const int col = blockIdx.x;
    if (col >= Kp) return;
    const int tid = threadIdx.x;
    __shared__ float sh[256];

    const int p = cand[col];
    const float* rp = R_full + static_cast<std::size_t>(p) * static_cast<std::size_t>(d);
    float* out = residual + static_cast<std::size_t>(col) * static_cast<std::size_t>(d);

    float sum = 0.0f;
    for (int r = tid; r < d; r += blockDim.x) {
        const float v = xi[r] - rp[r];
        out[r] = v;
        sum += v * v;
    }
    sh[tid] = sum;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) {
            sh[tid] += sh[tid + off];
        }
        __syncthreads();
    }
    if (tid == 0) {
        norm2[col] = sh[0];
    }
}

__global__ void GatherResidualAndNorm2FromRfullPairs(const float* __restrict X_block,  // d×B
                                                     const float* __restrict R_full,   // d×n_cols
                                                     const int* __restrict cand,       // Npairs
                                                     const int* __restrict pair_node,  // Npairs
                                                     float* __restrict residual,       // d×Npairs (col-major)
                                                     float* __restrict norm2,          // Npairs
                                                     int d,
                                                     int Npairs) {
    // One block per pair.
    const int col = blockIdx.x;
    if (col >= Npairs) return;
    const int tid = threadIdx.x;
    __shared__ float sh[256];

    const int node = pair_node[col];
    const float* xi = X_block + static_cast<std::size_t>(node) * static_cast<std::size_t>(d);

    const int p = cand[col];
    const float* rp = R_full + static_cast<std::size_t>(p) * static_cast<std::size_t>(d);

    float* out = residual + static_cast<std::size_t>(col) * static_cast<std::size_t>(d);

    float sum = 0.0f;
    for (int r = tid; r < d; r += blockDim.x) {
        const float v = xi[r] - rp[r];
        out[r] = v;
        sum += v * v;
    }
    sh[tid] = sum;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) {
            sh[tid] += sh[tid + off];
        }
        __syncthreads();
    }
    if (tid == 0) {
        norm2[col] = sh[0];
    }
}

__global__ void BuildPairSampleIdsFromNodeBase(const std::uint64_t* __restrict node_base,  // B
                                               const int* __restrict pair_node,           // Npairs
                                               const int* __restrict offsets,             // B+1
                                               const int* __restrict pair_t_override,     // Npairs (optional)
                                               std::uint64_t* __restrict out_ids,         // Npairs
                                               int B,
                                               int Npairs) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Npairs) return;
    const int node = pair_node[idx];
#ifndef NDEBUG
    if (node < 0 || node >= B) return;
#endif
    const std::uint64_t base = node_base[node];
    const std::uint64_t t = pair_t_override
                                ? static_cast<std::uint64_t>(pair_t_override[idx])
                                : static_cast<std::uint64_t>(idx - offsets[node]);
    // Match the per-node API's `sample_id_offset + t` behavior, but make it independent of
    // batching/flush order by using a per-node base derived from the node's stable local id.
    out_ids[idx] = base + t;
}

__global__ void FillI32(int* __restrict out, int n, int v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = v;
}

__global__ void SegmentedArgMinCost(const float* __restrict cost,   // Npairs
                                    const int* __restrict offsets, // B+1
                                    int B,
                                    float* __restrict out_cost,   // B
                                    int* __restrict out_pair_idx  // B (global pair idx)
) {
    const int node = blockIdx.x;
    if (node >= B) return;
    const int tid = threadIdx.x;

    const int start = offsets[node];
    const int stop = offsets[node + 1];
    if (start >= stop) {
        if (tid == 0) {
            out_cost[node] = INFINITY;
            out_pair_idx[node] = -1;
        }
        return;
    }

    float best = INFINITY;
    int best_i = -1;
    for (int i = start + tid; i < stop; i += blockDim.x) {
        const float c = cost[i];
        // Stable tie-breaking: if costs are equal, prefer the smaller pair index.
        // This matches a sequential CPU scan with `if (c < best)` semantics ("first seen wins").
        if (c < best || (c == best && (best_i < 0 || i < best_i))) {
            best = c;
            best_i = i;
        }
    }

    __shared__ float sh_cost[256];
    __shared__ int sh_idx[256];
    sh_cost[tid] = best;
    sh_idx[tid] = best_i;
    __syncthreads();

    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) {
            const float c2 = sh_cost[tid + off];
            const int i2 = sh_idx[tid + off];
            const float c1 = sh_cost[tid];
            const int i1 = sh_idx[tid];
            if (c2 < c1 || (c2 == c1 && i2 >= 0 && (i1 < 0 || i2 < i1))) {
                sh_cost[tid] = c2;
                sh_idx[tid] = i2;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_cost[node] = sh_cost[0];
        out_pair_idx[node] = sh_idx[0];
    }
}

__global__ void GatherBestParentPerNode(const int* __restrict best_pair_idx,  // B
                                       const int* __restrict cand,           // Npairs
                                       int B,
                                       int* __restrict out_parent) {         // B
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    const int pi = best_pair_idx[i];
    out_parent[i] = (pi >= 0) ? cand[pi] : -1;
}

__global__ void GatherBestCodesPerNode(int B,
                                      int m,
                                      const int* __restrict best_pair_idx,  // B
                                      const FullCode* __restrict B_all,     // Npairs*m
                                      const float* __restrict a_all,        // Npairs*m
                                      FullCode* __restrict B_out,           // B*m
                                      float* __restrict a_out) {            // B*m
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * m;
    if (i >= total) return;
    const int node = i / m;
    const int l = i - node * m;
    const int pi = best_pair_idx[node];
    if (pi < 0) {
        B_out[i] = 0;
        a_out[i] = 0.0f;
        return;
    }
    const std::size_t base = static_cast<std::size_t>(pi) * static_cast<std::size_t>(m);
    B_out[i] = B_all[base + static_cast<std::size_t>(l)];
    a_out[i] = a_all[base + static_cast<std::size_t>(l)];
}

__global__ void GatherBestCodesPerNodeU32(int B,
                                         int m,
                                         const int* __restrict best_pair_idx,     // B
                                         const std::uint32_t* __restrict B_all,   // Npairs*m
                                         const float* __restrict a_all,           // Npairs*m
                                         std::uint32_t* __restrict B_out,         // B*m
                                         float* __restrict a_out) {               // B*m
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * m;
    if (i >= total) return;
    const int node = i / m;
    const int l = i - node * m;
    const int pi = best_pair_idx[node];
    if (pi < 0) {
        B_out[i] = 0;
        a_out[i] = 0.0f;
        return;
    }
    const std::size_t base = static_cast<std::size_t>(pi) * static_cast<std::size_t>(m);
    B_out[i] = B_all[base + static_cast<std::size_t>(l)];
    a_out[i] = a_all[base + static_cast<std::size_t>(l)];
}

__global__ void PackManyNodesBestKernel(int B,
                                       int m,
                                       int a_off,
                                       int stride,
                                       const int* __restrict best_parent,     // B
                                       const float* __restrict best_cost,     // B
                                       const FullCode* __restrict best_B,     // B*m (node-major)
                                       const float* __restrict best_a,        // B*m (node-major)
                                       std::uint8_t* __restrict out_bytes) {  // B*stride
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    std::uint8_t* dst = out_bytes + static_cast<std::size_t>(i) * static_cast<std::size_t>(stride);
    *reinterpret_cast<int*>(dst + 0) = best_parent[i];
    *reinterpret_cast<float*>(dst + 4) = best_cost[i];

    auto* dst_B = reinterpret_cast<FullCode*>(dst + 8);
    const FullCode* src_B = best_B + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    for (int l = 0; l < m; ++l) {
        dst_B[l] = src_B[l];
    }

    auto* dst_a = reinterpret_cast<float*>(dst + a_off);
    const float* src_a = best_a + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    for (int l = 0; l < m; ++l) {
        dst_a[l] = src_a[l];
    }
}

__global__ void PackManyNodesBestKernelU32(int B,
                                          int m,
                                          int a_off,
                                          int stride,
                                          const int* __restrict best_parent,           // B
                                          const float* __restrict best_cost,           // B
                                          const std::uint32_t* __restrict best_B,      // B*m (node-major)
                                          const float* __restrict best_a,              // B*m (node-major)
                                          std::uint8_t* __restrict out_bytes) {        // B*stride
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    std::uint8_t* dst = out_bytes + static_cast<std::size_t>(i) * static_cast<std::size_t>(stride);
    *reinterpret_cast<int*>(dst + 0) = best_parent[i];
    *reinterpret_cast<float*>(dst + 4) = best_cost[i];

    auto* dst_B = reinterpret_cast<std::uint32_t*>(dst + 8);
    const std::uint32_t* src_B = best_B + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    for (int l = 0; l < m; ++l) {
        dst_B[l] = src_B[l];
    }

    auto* dst_a = reinterpret_cast<float*>(dst + a_off);
    const float* src_a = best_a + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    for (int l = 0; l < m; ++l) {
        dst_a[l] = src_a[l];
    }
}

__global__ void PackSingleBestKernel(int m,
                                    int a_off,
                                    int stride,
                                    const int* __restrict best_idx,       // 1
                                    const float* __restrict best_cost,    // 1
                                    const FullCode* __restrict best_B,    // m
                                    const float* __restrict best_a,       // m
                                    std::uint8_t* __restrict out_bytes) { // stride
    const int tid = threadIdx.x;
    if (tid == 0) {
        *reinterpret_cast<int*>(out_bytes + 0) = best_idx[0];
        *reinterpret_cast<float*>(out_bytes + 4) = best_cost[0];
    }
    if (tid < m) {
        reinterpret_cast<FullCode*>(out_bytes + 8)[tid] = best_B[tid];
        reinterpret_cast<float*>(out_bytes + a_off)[tid] = best_a[tid];
    }
    (void)stride;
}

__global__ void ReduceMinBlocks(const float* __restrict cost,
                               int n,
                               float* __restrict out_cost,
                               int* __restrict out_idx) {
    const int block = blockIdx.x;
    const int tid = threadIdx.x;
    const int base = block * blockDim.x;
    float best = INFINITY;
    int best_i = -1;
    const int i = base + tid;
    if (i < n) {
        best = cost[i];
        best_i = i;
    }
    __shared__ float sh_cost[256];
    __shared__ int sh_idx[256];
    sh_cost[tid] = best;
    sh_idx[tid] = best_i;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) {
            const float other = sh_cost[tid + off];
            const int other_i = sh_idx[tid + off];
            if (other < sh_cost[tid]) {
                sh_cost[tid] = other;
                sh_idx[tid] = other_i;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_cost[block] = sh_cost[0];
        out_idx[block] = sh_idx[0];
    }
}

__global__ void ReduceMinFinal(const float* __restrict in_cost,
                              const int* __restrict in_idx,
                              int n,
                              float* __restrict out_cost,
                              int* __restrict out_idx) {
    const int tid = threadIdx.x;
    float best = INFINITY;
    int best_i = -1;
    if (tid < n) {
        best = in_cost[tid];
        best_i = in_idx[tid];
    }
    __shared__ float sh_cost[256];
    __shared__ int sh_idx[256];
    sh_cost[tid] = best;
    sh_idx[tid] = best_i;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) {
            const float other = sh_cost[tid + off];
            const int other_i = sh_idx[tid + off];
            if (other < sh_cost[tid]) {
                sh_cost[tid] = other;
                sh_idx[tid] = other_i;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        *out_cost = sh_cost[0];
        *out_idx = sh_idx[0];
    }
}

__global__ void GatherBest(int m,
                           const FullCode* __restrict B_all,  // m×Kp
                           const float* __restrict a_all,     // m×Kp
                           const int* __restrict best_idx,
                           FullCode* __restrict B_best,       // m
                           float* __restrict a_best) {        // m
    const int tid = threadIdx.x;
    if (tid >= m) return;
    const int idx = *best_idx;
    if (idx < 0) return;
    const std::size_t base = static_cast<std::size_t>(idx) * static_cast<std::size_t>(m);
    B_best[tid] = B_all[base + static_cast<std::size_t>(tid)];
    a_best[tid] = a_all[base + static_cast<std::size_t>(tid)];
}

struct EvalWorkspace {
    int d = 0;
    int max_kp = 0;

    DeviceBuf<float> d_xi;
    DeviceBuf<float> d_Rp;    // d×Kp, overwritten to residual
    DeviceBuf<float> d_norm2;
    DeviceBuf<FullCode> d_B;
    DeviceBuf<std::uint32_t> d_B_u32;
    DeviceBuf<float> d_a;
    DeviceBuf<float> d_cost;

    DeviceBuf<int> d_cand;

    // Stage-2: device mirror of cluster-local R_full (col-major, d×rfull_cols).
    DeviceBuf<float> d_R_full;
    int rfull_d = 0;
    int rfull_cols = 0;

    // Stage-3: batch inner-candidates across many nodes.
    DeviceBuf<float> d_xblock;            // d×B
    DeviceBuf<int> d_pair_node;           // Npairs
    DeviceBuf<int> d_offsets;             // (B+1)
	// Optional: per-pair sample IDs for deterministic ILS perturbation (Npairs),
	// built from a per-node base ID (B).
	DeviceBuf<std::uint64_t> d_node_sample_id_base;  // B
	DeviceBuf<int> d_pair_t_override;                // Npairs (optional)
	DeviceBuf<std::uint64_t> d_sample_ids;           // Npairs
    // Optional: forced-root code for each candidate pair (Npairs), used by init-linkage forced-root encoding.
    DeviceBuf<int> d_forced_root;         // Npairs
    DeviceBuf<int> d_best_pair_batch;     // B
    DeviceBuf<float> d_best_cost_batch;   // B
    DeviceBuf<int> d_best_parent_batch;   // B
    DeviceBuf<FullCode> d_best_B_batch;   // B*m (node-major)
    DeviceBuf<std::uint32_t> d_best_B_batch_u32;   // B*m (node-major, u32 codes)
    DeviceBuf<float> d_best_a_batch;      // B*m (node-major)

    DeviceBuf<float> d_partial_cost;
    DeviceBuf<int> d_partial_idx;
    DeviceBuf<float> d_best_cost;
    DeviceBuf<int> d_best_idx;
    DeviceBuf<FullCode> d_best_B;
    DeviceBuf<float> d_best_a;

    // Packed many-nodes output (reduces the number of cudaMemcpy calls on D2H).
    DeviceBuf<std::uint8_t> d_best_pack;
    HostPinnedBuf h_best_pack;
    int best_pack_stride = 0;
    int best_pack_a_off = 0;
    int best_pack_code_bytes = 0;

	LinkageEncodeBatchCudaWorkspace* encode_ws = nullptr;
	bool events_inited = false;
	cudaEvent_t ev[6] = {};
	// Async overlap support (one in-flight job per CudaCtx).
	bool async_in_flight = false;
	bool async_profile = false;
	int async_d = 0;
	int async_B = 0;
	int async_Npairs = 0;
	int async_m = 0;
	cudaEvent_t async_done = nullptr;  // only used when profiling is disabled
	bool async_done_inited = false;
	PinnedBuf<float> h_async_X;                   // d×B
	PinnedBuf<int> h_async_cand;                  // Npairs
	PinnedBuf<int> h_async_pair_node;             // Npairs
	PinnedBuf<int> h_async_offsets;               // (B+1)
	PinnedBuf<std::uint64_t> h_async_node_base;   // B
	PinnedBuf<int> h_async_pair_t_override;       // Npairs (optional)

    // Stage-2: per-column R_full updates are tiny (d floats) and are issued very frequently (per-node commit).
    // `cudaMemcpyAsync` from pageable host memory can implicitly stage and block the CPU, which hurts overlap.
    // We therefore stage updates into a small pinned ring and enqueue H2D copies from pinned memory.
    static constexpr int kRfullUpdateRingSlots = 16;
    PinnedBuf<float> h_rfull_update_ring;  // (kRfullUpdateRingSlots * d)
    int rfull_update_ring_d = 0;
    int rfull_update_ring_pos = 0;
    cudaEvent_t rfull_update_ev[kRfullUpdateRingSlots] = {};
    bool rfull_update_ev_inited[kRfullUpdateRingSlots] = {};

    // Stage-2: batched R_full column updates (pack+scatter).
    // - Host: small pinned rings to avoid CPU stalls on H2D staging.
    // - Device: staging buffers (reused per ctx/stream; stream ordering guarantees safety).
    // NOTE: updates are often enqueued much faster than the stream drains (interleaved with heavy candidate-eval kernels).
    // If the ring is too small, we end up `cudaEventSynchronize`-ing frequently just to reuse a staging slot, which
    // artificially inflates "rfull_update" wall time and reduces overlap across many CUDA contexts.
    static constexpr int kRfullUpdateBatchRingSlots = 32;
    PinnedBuf<int> h_rfull_update_batch_cols[kRfullUpdateBatchRingSlots];
    PinnedBuf<float> h_rfull_update_batch_data[kRfullUpdateBatchRingSlots];
    int rfull_update_batch_pos = 0;
    cudaEvent_t rfull_update_batch_ev[kRfullUpdateBatchRingSlots] = {};
    bool rfull_update_batch_ev_inited[kRfullUpdateBatchRingSlots] = {};
    DeviceBuf<int> d_rfull_update_batch_cols;
    DeviceBuf<float> d_rfull_update_batch_data;

    ~EvalWorkspace() { Reset(); }

    void Reset() {
        if (async_done_inited) {
            if (async_done) cudaEventDestroy(async_done);
            async_done = nullptr;
            async_done_inited = false;
        }
        async_in_flight = false;
        async_profile = false;
        async_d = 0;
        async_B = 0;
        async_Npairs = 0;
        async_m = 0;
        h_async_X.Reset();
        h_async_cand.Reset();
        h_async_pair_node.Reset();
        h_async_offsets.Reset();
        h_async_node_base.Reset();
        h_async_pair_t_override.Reset();
        h_rfull_update_ring.Reset();
        rfull_update_ring_d = 0;
        rfull_update_ring_pos = 0;
        for (int i = 0; i < kRfullUpdateRingSlots; ++i) {
            if (rfull_update_ev_inited[i]) {
                if (rfull_update_ev[i]) cudaEventDestroy(rfull_update_ev[i]);
                rfull_update_ev[i] = nullptr;
                rfull_update_ev_inited[i] = false;
            }
        }
        for (int i = 0; i < kRfullUpdateBatchRingSlots; ++i) {
            h_rfull_update_batch_cols[i].Reset();
            h_rfull_update_batch_data[i].Reset();
            if (rfull_update_batch_ev_inited[i]) {
                if (rfull_update_batch_ev[i]) cudaEventDestroy(rfull_update_batch_ev[i]);
                rfull_update_batch_ev[i] = nullptr;
                rfull_update_batch_ev_inited[i] = false;
            }
        }
        rfull_update_batch_pos = 0;
        d_rfull_update_batch_cols.Reset();
        d_rfull_update_batch_data.Reset();
        if (events_inited) {
            for (cudaEvent_t& e : ev) {
                if (e) cudaEventDestroy(e);
                e = nullptr;
            }
            events_inited = false;
        }
        if (encode_ws) {
            DestroyLinkageEncodeBatchCudaWorkspace(encode_ws);
            encode_ws = nullptr;
        }
        d_xi.Reset();
        d_Rp.Reset();
        d_norm2.Reset();
        d_B.Reset();
        d_B_u32.Reset();
        d_a.Reset();
        d_cost.Reset();
        d_cand.Reset();
        d_R_full.Reset();
        rfull_d = 0;
        rfull_cols = 0;
        d_xblock.Reset();
        d_pair_node.Reset();
	        d_offsets.Reset();
	        d_node_sample_id_base.Reset();
	        d_pair_t_override.Reset();
	        d_sample_ids.Reset();
        d_best_pair_batch.Reset();
        d_best_cost_batch.Reset();
        d_best_parent_batch.Reset();
        d_best_B_batch.Reset();
        d_best_B_batch_u32.Reset();
        d_best_a_batch.Reset();
        d_partial_cost.Reset();
        d_partial_idx.Reset();
			d_best_cost.Reset();
			d_best_idx.Reset();
			d_best_B.Reset();
	        d_best_a.Reset();
            d_best_pack.Reset();
            h_best_pack.Reset();
            best_pack_stride = 0;
            best_pack_a_off = 0;
            best_pack_code_bytes = 0;
			d = 0;
			max_kp = 0;
		}

    void EnsureRfullUpdateRing(int d_in) {
        if (d_in <= 0) return;
        const std::size_t need =
            static_cast<std::size_t>(kRfullUpdateRingSlots) * static_cast<std::size_t>(d_in);
        if (rfull_update_ring_d == d_in && h_rfull_update_ring.cap >= need) {
            return;
        }
        h_rfull_update_ring.Reset();
        rfull_update_ring_d = d_in;
        rfull_update_ring_pos = 0;
        h_rfull_update_ring.Ensure(need);
        for (int i = 0; i < kRfullUpdateRingSlots; ++i) {
            if (rfull_update_ev_inited[i]) {
                if (rfull_update_ev[i]) cudaEventDestroy(rfull_update_ev[i]);
                rfull_update_ev[i] = nullptr;
                rfull_update_ev_inited[i] = false;
            }
        }
    }

    void EnsureBestPack(const CudaCtx& ctx, int B, int m, int code_bytes) {
        ThrowIf(cudaSetDevice(ctx.device), "cudaSetDevice");
        const int cb = std::max(1, code_bytes);
        // Pack layout:
        //   [0..3]   parent (int32)
        //   [4..7]   cost (float)
        //   [8..]    B[m] (cb bytes each)
        //   [a_off..]a[m] (float), 4-byte aligned
        const int b_bytes = 8 + m * cb;
	        const int a_off = (b_bytes + 3) & ~3;
	        const int stride = a_off + m * static_cast<int>(sizeof(float));
	        if (best_pack_code_bytes == cb && best_pack_stride == stride && best_pack_a_off == a_off &&
	            d_best_pack.cap >= static_cast<std::size_t>(B) * static_cast<std::size_t>(stride) &&
	            h_best_pack.bytes >= static_cast<std::size_t>(B) * static_cast<std::size_t>(stride)) {
	            return;
	        }
        best_pack_code_bytes = cb;
        best_pack_stride = stride;
        best_pack_a_off = a_off;
        d_best_pack.Ensure(static_cast<std::size_t>(B) * static_cast<std::size_t>(stride));
        h_best_pack.Ensure(static_cast<std::size_t>(B) * static_cast<std::size_t>(stride));
    }

	void Ensure(const CudaCtx& ctx, int d_in, int kp_in, int m) {
        if (!encode_ws) {
            encode_ws = CreateLinkageEncodeBatchCudaWorkspace();
        }
        d = d_in;
        max_kp = kp_in;

        ThrowIf(cudaSetDevice(ctx.device), "cudaSetDevice");
        d_xi.Ensure(static_cast<std::size_t>(d));
        d_Rp.Ensure(static_cast<std::size_t>(d) * static_cast<std::size_t>(kp_in));
        d_norm2.Ensure(static_cast<std::size_t>(kp_in));
        d_B.Ensure(static_cast<std::size_t>(m) * static_cast<std::size_t>(kp_in));
        d_a.Ensure(static_cast<std::size_t>(m) * static_cast<std::size_t>(kp_in));
        d_cost.Ensure(static_cast<std::size_t>(kp_in));
        d_cand.Ensure(static_cast<std::size_t>(kp_in));

        const int blocks = (kp_in + 255) / 256;
        d_partial_cost.Ensure(static_cast<std::size_t>(blocks));
        d_partial_idx.Ensure(static_cast<std::size_t>(blocks));
        d_best_cost.Ensure(1);
        d_best_idx.Ensure(1);
        d_best_B.Ensure(static_cast<std::size_t>(m));
        d_best_a.Ensure(static_cast<std::size_t>(m));
        EnsureBestPack(ctx, /*B=*/1, m, static_cast<int>(sizeof(FullCode)));
	}

	void EnsureEvents(const CudaCtx& ctx) {
	    if (events_inited) return;
	    ThrowIf(cudaSetDevice(ctx.device), "cudaSetDevice");
	    for (cudaEvent_t& e : ev) {
	        ThrowIf(cudaEventCreateWithFlags(&e, cudaEventDefault), "cudaEventCreate");
	    }
	    events_inited = true;
	}

	void EnsureAsyncDoneEvent(const CudaCtx& ctx) {
	    if (async_done_inited) return;
	    ThrowIf(cudaSetDevice(ctx.device), "cudaSetDevice");
	    ThrowIf(cudaEventCreateWithFlags(&async_done, cudaEventDisableTiming), "cudaEventCreate(async_done)");
	    async_done_inited = true;
	}

		    void EnsureBatch(const CudaCtx& ctx, int d_in, int B, int Npairs, int m) {
		        Ensure(ctx, d_in, Npairs, m);
		        d_xblock.Ensure(static_cast<std::size_t>(d_in) * static_cast<std::size_t>(B));
		        d_pair_node.Ensure(static_cast<std::size_t>(Npairs));
		        d_offsets.Ensure(static_cast<std::size_t>(B + 1));
		        d_node_sample_id_base.Ensure(static_cast<std::size_t>(B));
		        d_pair_t_override.Ensure(static_cast<std::size_t>(Npairs));
		        d_sample_ids.Ensure(static_cast<std::size_t>(Npairs));
		        d_forced_root.Ensure(static_cast<std::size_t>(Npairs));
		        d_best_pair_batch.Ensure(static_cast<std::size_t>(B));
		        d_best_cost_batch.Ensure(static_cast<std::size_t>(B));
		        d_best_parent_batch.Ensure(static_cast<std::size_t>(B));
		        d_best_B_batch.Ensure(static_cast<std::size_t>(B) * static_cast<std::size_t>(m));
		        d_best_a_batch.Ensure(static_cast<std::size_t>(B) * static_cast<std::size_t>(m));
		        EnsureBestPack(ctx, /*B=*/B, m, static_cast<int>(sizeof(FullCode)));
		    }

		    void EnsureBatchU32(const CudaCtx& ctx, int d_in, int B, int Npairs, int m) {
		        EnsureBatch(ctx, d_in, B, Npairs, m);
		        d_B_u32.Ensure(static_cast<std::size_t>(m) * static_cast<std::size_t>(Npairs));
		        d_best_B_batch_u32.Ensure(static_cast<std::size_t>(B) * static_cast<std::size_t>(m));
		        EnsureBestPack(ctx, /*B=*/B, m, /*code_bytes=*/static_cast<int>(sizeof(std::uint32_t)));
		    }
		};

void DeleteEvalWorkspace(void* ptr) {
    delete static_cast<EvalWorkspace*>(ptr);
}

EvalWorkspace* GetWorkspace(CudaCtx* ctx) {
    if (!ctx->user) {
        ctx->user = new EvalWorkspace();
        ctx->user_deleter = &DeleteEvalWorkspace;
    }
    return static_cast<EvalWorkspace*>(ctx->user);
}

}  // namespace

void SetCudaLinkageEvalProfiling(bool enabled) {
    g_profile_enabled.store(enabled, std::memory_order_relaxed);
    SetCudaLinkageEncodeProfiling(enabled);
}

void SetCudaLinkageEvalChunkMaxPairs(int chunk_max_pairs) {
    g_chunk_max_pairs.store(chunk_max_pairs, std::memory_order_relaxed);
}

void SetCudaLinkageEvalMemBudgetMb(int mem_budget_mb) {
    g_mem_budget_mb.store(mem_budget_mb, std::memory_order_relaxed);
}

void SetCudaLinkageEvalAsyncPinnedBudgetMb(int pinned_mb) {
    g_async_pinned_budget_mb.store(pinned_mb, std::memory_order_relaxed);
}

static int EffectiveChunkMaxPairsFromBudgetWithCodeBytes(int d,
                                                         int H,
                                                         int m,
                                                         int B,
                                                         int rfull_cols,
                                                         int mem_budget_mb,
                                                         int code_bytes) {
    if (mem_budget_mb <= 0) return 0;
    const std::uint64_t budget_bytes =
        static_cast<std::uint64_t>(mem_budget_mb) * static_cast<std::uint64_t>(1024u * 1024u);
    // Conservative: ignore precomp sharing and treat the estimate as per-ctx peak.
    // This yields a lower cap, which is safe against OOM.
    int lo = 1;
    int hi = 1;
    while (hi < (1 << 26)) {
        const std::uint64_t bytes = EstimateBytesPerCtxWithCodeBytes(d, H, m, B, hi, rfull_cols, code_bytes);
        if (bytes >= budget_bytes) break;
        hi *= 2;
    }
    int best = 0;
    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        const std::uint64_t bytes = EstimateBytesPerCtxWithCodeBytes(d, H, m, B, mid, rfull_cols, code_bytes);
        if (bytes <= budget_bytes) {
            best = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    // If the budget is too small to fit even npairs=1 in our estimate, still return 1 so we at least
    // force pair-chunking and avoid a one-shot call that is more likely to OOM.
    return std::max(1, best);
}

static int EffectiveChunkMaxPairsFromBudget(int d, int H, int m, int B, int rfull_cols, int mem_budget_mb) {
    return EffectiveChunkMaxPairsFromBudgetWithCodeBytes(d, H, m, B, rfull_cols, mem_budget_mb,
                                                         static_cast<int>(sizeof(FullCode)));
}

CudaLinkageEvalStats GetAndResetCudaLinkageEvalStats() {
    CudaLinkageEvalStats out;
    out.calls = g_calls.exchange(0, std::memory_order_relaxed);
    const auto take_ns = [](std::atomic<std::int64_t>* v) -> double {
        const std::int64_t ns = v->exchange(0, std::memory_order_relaxed);
        return static_cast<double>(ns) * 1e-9;
    };
    out.h2d_s = take_ns(&g_h2d_ns);
    out.residual_s = take_ns(&g_residual_ns);
    out.encode_s = take_ns(&g_encode_ns);
    out.reduce_s = take_ns(&g_reduce_ns);
    out.d2h_s = take_ns(&g_d2h_ns);
    out.total_s = take_ns(&g_total_ns);
    return out;
}

CudaLinkageEvalWorkspaceStats GetAndResetCudaLinkageEvalWorkspaceStats() {
    CudaLinkageEvalWorkspaceStats out;
    out.peak_d = g_peak_d.exchange(0, std::memory_order_relaxed);
    out.peak_m = g_peak_m.exchange(0, std::memory_order_relaxed);
    out.peak_H = g_peak_H.exchange(0, std::memory_order_relaxed);
    out.peak_B = g_peak_B.exchange(0, std::memory_order_relaxed);
    out.peak_npairs = g_peak_npairs.exchange(0, std::memory_order_relaxed);
    out.peak_rfull_cols = g_peak_rfull_cols.exchange(0, std::memory_order_relaxed);
    out.peak_bytes_per_ctx_est = g_peak_bytes_per_ctx_est.exchange(0, std::memory_order_relaxed);
    return out;
}

CudaLinkageEvalShapeStats GetAndResetCudaLinkageEvalShapeStats() {
    CudaLinkageEvalShapeStats out;
    out.calls = g_shape_calls.exchange(0, std::memory_order_relaxed);
    out.d2h_bytes = g_d2h_bytes.exchange(0, std::memory_order_relaxed);
    out.max_B = g_max_B.exchange(0, std::memory_order_relaxed);
    out.max_npairs = g_max_npairs.exchange(0, std::memory_order_relaxed);
    for (int i = 0; i < 12; ++i) {
        out.B_hist[i] = g_hist_B[i].exchange(0, std::memory_order_relaxed);
        out.npairs_hist[i] = g_hist_npairs[i].exchange(0, std::memory_order_relaxed);
    }
    return out;
}

bool CudaLinkageUploadClusterRfull(CudaCtx* ctx,
                                const float* R_full,
                                int d,
                                int n_cols,
                                std::string* err) {
    try {
        if (!ctx) {
            throw std::runtime_error("CudaLinkageUploadClusterRfull: ctx is null.");
        }
        if (d <= 0 || n_cols < 0) {
            throw std::runtime_error("CudaLinkageUploadClusterRfull: invalid shape.");
        }
        if (n_cols > 0 && !R_full) {
            throw std::runtime_error("CudaLinkageUploadClusterRfull: R_full is null.");
        }
        ThrowIf(cudaSetDevice(ctx->device), "cudaSetDevice");
        EvalWorkspace* ws = GetWorkspace(ctx);
        ws->d_R_full.Ensure(static_cast<std::size_t>(d) * static_cast<std::size_t>(n_cols));
        ws->rfull_d = d;
        ws->rfull_cols = n_cols;
        if (n_cols > 0) {
            ThrowIf(cudaMemcpyAsync(ws->d_R_full.ptr, R_full,
                                    sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(n_cols),
                                    cudaMemcpyHostToDevice, ctx->stream),
                    "Memcpy R_full H2D");
        }
        return true;
    } catch (const std::exception& e) {
        if (err) {
            *err = e.what();
        }
        return false;
    }
}

bool CudaLinkageUploadClusterRfullTwoParts(CudaCtx* ctx,
                                        const float* R_part0,
                                        int n0,
                                        const float* R_part1,
                                        int n1,
                                        int d,
                                        std::string* err) {
    try {
        if (!ctx) {
            throw std::runtime_error("CudaLinkageUploadClusterRfullTwoParts: ctx is null.");
        }
        if (d <= 0 || n0 < 0 || n1 < 0) {
            throw std::runtime_error("CudaLinkageUploadClusterRfullTwoParts: invalid shape.");
        }
        if (n0 > 0 && !R_part0) {
            throw std::runtime_error("CudaLinkageUploadClusterRfullTwoParts: R_part0 is null.");
        }
        if (n1 > 0 && !R_part1) {
            throw std::runtime_error("CudaLinkageUploadClusterRfullTwoParts: R_part1 is null.");
        }
        const int n_cols = n0 + n1;
        ThrowIf(cudaSetDevice(ctx->device), "cudaSetDevice");
        EvalWorkspace* ws = GetWorkspace(ctx);
        ws->d_R_full.Ensure(static_cast<std::size_t>(d) * static_cast<std::size_t>(n_cols));
        ws->rfull_d = d;
        ws->rfull_cols = n_cols;
        if (n0 > 0) {
            ThrowIf(cudaMemcpyAsync(ws->d_R_full.ptr, R_part0,
                                    sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(n0),
                                    cudaMemcpyHostToDevice, ctx->stream),
                    "Memcpy R_part0 H2D");
        }
        if (n1 > 0) {
            float* dst = ws->d_R_full.ptr + static_cast<std::size_t>(d) * static_cast<std::size_t>(n0);
            ThrowIf(cudaMemcpyAsync(dst, R_part1,
                                    sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(n1),
                                    cudaMemcpyHostToDevice, ctx->stream),
                    "Memcpy R_part1 H2D");
        }
        return true;
    } catch (const std::exception& e) {
        if (err) {
            *err = e.what();
        }
        return false;
    }
}

bool CudaLinkageUpdateClusterRfullColumn(CudaCtx* ctx,
                                       int col,
                                       const float* r_col,
                                       int d,
                                       std::string* err) {
    try {
        if (!ctx) {
            throw std::runtime_error("CudaLinkageUpdateClusterRfullColumn: ctx is null.");
        }
        if (!r_col || d <= 0) {
            throw std::runtime_error("CudaLinkageUpdateClusterRfullColumn: invalid input.");
        }
        ThrowIf(cudaSetDevice(ctx->device), "cudaSetDevice");
        EvalWorkspace* ws = GetWorkspace(ctx);
        if (!ws->d_R_full.ptr || ws->rfull_cols <= 0) {
            throw std::runtime_error("CudaLinkageUpdateClusterRfullColumn: device R_full not uploaded.");
        }
        if (ws->rfull_d != d) {
            throw std::runtime_error("CudaLinkageUpdateClusterRfullColumn: d mismatch.");
        }
        if (col < 0 || col >= ws->rfull_cols) {
            throw std::runtime_error("CudaLinkageUpdateClusterRfullColumn: col out of range.");
        }
        float* dst = ws->d_R_full.ptr + static_cast<std::size_t>(col) * static_cast<std::size_t>(d);

        // Stage into pinned ring to avoid pageable-host implicit staging in `cudaMemcpyAsync`.
        ws->EnsureRfullUpdateRing(d);
        const std::size_t bytes = sizeof(float) * static_cast<std::size_t>(d);

        int slot = ws->rfull_update_ring_pos;
        for (int tries = 0; tries < EvalWorkspace::kRfullUpdateRingSlots; ++tries) {
            if (!ws->rfull_update_ev_inited[slot]) {
                break;
            }
            const cudaError_t q = cudaEventQuery(ws->rfull_update_ev[slot]);
            if (q == cudaSuccess) {
                break;
            }
            if (q != cudaErrorNotReady) {
                ThrowIf(q, "cudaEventQuery(rfull_update)");
            }
            slot = (slot + 1) % EvalWorkspace::kRfullUpdateRingSlots;
        }
        if (ws->rfull_update_ev_inited[slot]) {
            const cudaError_t q = cudaEventQuery(ws->rfull_update_ev[slot]);
            if (q == cudaErrorNotReady) {
                ThrowIf(cudaEventSynchronize(ws->rfull_update_ev[slot]), "cudaEventSynchronize(rfull_update)");
            } else if (q != cudaSuccess) {
                ThrowIf(q, "cudaEventQuery(rfull_update)");
            }
        }
        if (!ws->rfull_update_ev_inited[slot]) {
            ThrowIf(cudaEventCreateWithFlags(&ws->rfull_update_ev[slot], cudaEventDisableTiming),
                    "cudaEventCreateWithFlags(rfull_update)");
            ws->rfull_update_ev_inited[slot] = true;
        }

        float* staging =
            ws->h_rfull_update_ring.ptr + static_cast<std::size_t>(slot) * static_cast<std::size_t>(d);
        std::memcpy(staging, r_col, bytes);
        ThrowIf(cudaMemcpyAsync(dst, staging, bytes, cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy R_full column H2D (pinned staging)");
        ThrowIf(cudaEventRecord(ws->rfull_update_ev[slot], ctx->stream), "cudaEventRecord(rfull_update)");
        ws->rfull_update_ring_pos = (slot + 1) % EvalWorkspace::kRfullUpdateRingSlots;
        return true;
    } catch (const std::exception& e) {
        if (err) {
            *err = e.what();
        }
        return false;
    }
}

namespace {

__global__ void ScatterRfullColumnsKernel(float* __restrict d_R_full,   // d×n_cols, col-major
                                         const int* __restrict cols,   // B
                                         const float* __restrict data, // (B*d), packed columns
                                         int d,
                                         int B) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = B * d;
    if (tid >= total) return;
    const int b = tid / d;
    const int r = tid - b * d;
    const int col = cols[b];
    d_R_full[static_cast<std::size_t>(col) * static_cast<std::size_t>(d) + static_cast<std::size_t>(r)] =
        data[static_cast<std::size_t>(b) * static_cast<std::size_t>(d) + static_cast<std::size_t>(r)];
}

}  // namespace

bool CudaLinkageUpdateClusterRfullColumnsBatch(CudaCtx* ctx,
                                            const int* cols,
                                            const float* r_cols,
                                            int d,
                                            int B,
                                            std::string* err) {
    try {
        if (!ctx) {
            throw std::runtime_error("CudaLinkageUpdateClusterRfullColumnsBatch: ctx is null.");
        }
        if (d <= 0 || B < 0) {
            throw std::runtime_error("CudaLinkageUpdateClusterRfullColumnsBatch: invalid shape.");
        }
        if (B == 0) {
            return true;
        }
        if (!cols || !r_cols) {
            throw std::runtime_error("CudaLinkageUpdateClusterRfullColumnsBatch: null input.");
        }
        ThrowIf(cudaSetDevice(ctx->device), "cudaSetDevice");
        EvalWorkspace* ws = GetWorkspace(ctx);
        if (!ws->d_R_full.ptr || ws->rfull_cols <= 0) {
            throw std::runtime_error("CudaLinkageUpdateClusterRfullColumnsBatch: device R_full not uploaded.");
        }
        if (ws->rfull_d != d) {
            throw std::runtime_error("CudaLinkageUpdateClusterRfullColumnsBatch: d mismatch.");
        }
        for (int i = 0; i < B; ++i) {
            const int col = cols[i];
            if (col < 0 || col >= ws->rfull_cols) {
                throw std::runtime_error("CudaLinkageUpdateClusterRfullColumnsBatch: col out of range.");
            }
        }

        // Select a free pinned slot (avoid overwriting memory still in-flight on the stream).
        int slot = ws->rfull_update_batch_pos;
        for (int tries = 0; tries < EvalWorkspace::kRfullUpdateBatchRingSlots; ++tries) {
            if (!ws->rfull_update_batch_ev_inited[slot]) {
                break;
            }
            const cudaError_t q = cudaEventQuery(ws->rfull_update_batch_ev[slot]);
            if (q == cudaSuccess) {
                break;
            }
            if (q != cudaErrorNotReady) {
                ThrowIf(q, "cudaEventQuery(rfull_update_batch)");
            }
            slot = (slot + 1) % EvalWorkspace::kRfullUpdateBatchRingSlots;
        }
        if (ws->rfull_update_batch_ev_inited[slot]) {
            const cudaError_t q = cudaEventQuery(ws->rfull_update_batch_ev[slot]);
            if (q == cudaErrorNotReady) {
                ThrowIf(cudaEventSynchronize(ws->rfull_update_batch_ev[slot]),
                        "cudaEventSynchronize(rfull_update_batch)");
            } else if (q != cudaSuccess) {
                ThrowIf(q, "cudaEventQuery(rfull_update_batch)");
            }
        }
        if (!ws->rfull_update_batch_ev_inited[slot]) {
            ThrowIf(cudaEventCreateWithFlags(&ws->rfull_update_batch_ev[slot], cudaEventDisableTiming),
                    "cudaEventCreateWithFlags(rfull_update_batch)");
            ws->rfull_update_batch_ev_inited[slot] = true;
        }

        const auto u64_B = static_cast<std::size_t>(B);
        const auto u64_d = static_cast<std::size_t>(d);
        ws->h_rfull_update_batch_cols[slot].Ensure(u64_B);
        ws->h_rfull_update_batch_data[slot].Ensure(u64_B * u64_d);
        std::memcpy(ws->h_rfull_update_batch_cols[slot].ptr, cols, sizeof(int) * u64_B);
        std::memcpy(ws->h_rfull_update_batch_data[slot].ptr, r_cols, sizeof(float) * u64_B * u64_d);

        ws->d_rfull_update_batch_cols.Ensure(u64_B);
        ws->d_rfull_update_batch_data.Ensure(u64_B * u64_d);
        ThrowIf(cudaMemcpyAsync(ws->d_rfull_update_batch_cols.ptr,
                                ws->h_rfull_update_batch_cols[slot].ptr,
                                sizeof(int) * u64_B,
                                cudaMemcpyHostToDevice,
                                ctx->stream),
                "Memcpy R_full batch cols H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_rfull_update_batch_data.ptr,
                                ws->h_rfull_update_batch_data[slot].ptr,
                                sizeof(float) * u64_B * u64_d,
                                cudaMemcpyHostToDevice,
                                ctx->stream),
                "Memcpy R_full batch data H2D");

        const int total = B * d;
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;
        ScatterRfullColumnsKernel<<<blocks, threads, 0, ctx->stream>>>(
            ws->d_R_full.ptr, ws->d_rfull_update_batch_cols.ptr, ws->d_rfull_update_batch_data.ptr, d, B);
        ThrowIf(cudaGetLastError(), "ScatterRfullColumnsKernel");
        ThrowIf(cudaEventRecord(ws->rfull_update_batch_ev[slot], ctx->stream),
                "cudaEventRecord(rfull_update_batch)");
        ws->rfull_update_batch_pos = (slot + 1) % EvalWorkspace::kRfullUpdateBatchRingSlots;
        return true;
    } catch (const std::exception& e) {
        if (err) {
            *err = e.what();
        }
        return false;
    }
}

bool EvaluateParentCandidatesBatchCuda(CudaCtx* ctx,
                                       const Precomp& pre_one,
                                       const float* xi,
                                       int d,
                                       const float* Rp,
                                       const int* cand_parent_local,
                                       int Kp,
                                       int icm_iters,
                                       int ils_iters,
                                       int perturb_k,
                                       std::uint32_t seed,
                                       std::uint64_t sample_id_offset,
                                       int* best_parent_local,
                                       float* best_cost,
                                       std::vector<FullCode>* best_B,
                                       std::vector<float>* best_a,
                                       std::string* err) {
    try {
        if (!ctx) {
            throw std::runtime_error("EvaluateParentCandidatesBatchCuda: ctx is null.");
        }
        if (Kp <= 0) {
            if (best_parent_local) *best_parent_local = -1;
            if (best_cost) *best_cost = std::numeric_limits<float>::infinity();
            if (best_B) best_B->clear();
            if (best_a) best_a->clear();
            return true;
        }
        if (!xi || !Rp || !cand_parent_local) {
            throw std::runtime_error("EvaluateParentCandidatesBatchCuda: null input pointer.");
        }
        if (d != pre_one.d) {
            throw std::runtime_error("EvaluateParentCandidatesBatchCuda: d mismatch.");
        }
        const int m = pre_one.m;
        if (m <= 0) {
            throw std::runtime_error("EvaluateParentCandidatesBatchCuda: invalid m.");
        }

        ThrowIf(cudaSetDevice(ctx->device), "cudaSetDevice");

        EvalWorkspace* ws = GetWorkspace(ctx);
        ws->Ensure(*ctx, d, Kp, m);

        const bool profile = g_profile_enabled.load(std::memory_order_relaxed);
        if (profile) {
            ws->EnsureEvents(*ctx);
            ThrowIf(cudaEventRecord(ws->ev[0], ctx->stream), "cudaEventRecord(ev0)");
        }

        ThrowIf(cudaMemcpyAsync(ws->d_xi.ptr, xi, sizeof(float) * static_cast<std::size_t>(d),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy xi H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_Rp.ptr, Rp,
                                sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(Kp),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy Rp H2D");
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[1], ctx->stream), "cudaEventRecord(ev1)");
        }

        // Residuals in-place + norm2.
        ResidualInplaceAndNorm2<<<Kp, 256, 0, ctx->stream>>>(ws->d_xi.ptr, ws->d_Rp.ptr, ws->d_norm2.ptr, d, Kp);
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[2], ctx->stream), "cudaEventRecord(ev2)");
        }

        // Encode residuals + cost on GPU.
        std::string encode_err;
        if (!LinkageEncodeBatchCuda(*ctx, pre_one,
                                  ws->d_Rp.ptr, ws->d_norm2.ptr, Kp, icm_iters,
                                  ils_iters, perturb_k, seed, sample_id_offset,
                                  ws->d_B.ptr, ws->d_a.ptr, ws->d_cost.ptr,
                                  ws->encode_ws, &encode_err)) {
            throw std::runtime_error(std::string("LinkageEncodeBatchCuda failed: ") + encode_err);
        }
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[3], ctx->stream), "cudaEventRecord(ev3)");
        }

        // Argmin cost.
        const int reduce_threads = 256;
        const int reduce_blocks = (Kp + reduce_threads - 1) / reduce_threads;
        ReduceMinBlocks<<<reduce_blocks, reduce_threads, 0, ctx->stream>>>(
            ws->d_cost.ptr, Kp, ws->d_partial_cost.ptr, ws->d_partial_idx.ptr);
        ReduceMinFinal<<<1, 256, 0, ctx->stream>>>(
            ws->d_partial_cost.ptr, ws->d_partial_idx.ptr, reduce_blocks,
            ws->d_best_cost.ptr, ws->d_best_idx.ptr);
        GatherBest<<<1, 256, 0, ctx->stream>>>(m, ws->d_B.ptr, ws->d_a.ptr,
                                               ws->d_best_idx.ptr, ws->d_best_B.ptr, ws->d_best_a.ptr);
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[4], ctx->stream), "cudaEventRecord(ev4)");
        }

        PackSingleBestKernel<<<1, 256, 0, ctx->stream>>>(
            m,
            ws->best_pack_a_off,
            ws->best_pack_stride,
            ws->d_best_idx.ptr,
            ws->d_best_cost.ptr,
            ws->d_best_B.ptr,
            ws->d_best_a.ptr,
            ws->d_best_pack.ptr);
        ThrowIf(cudaMemcpyAsync(ws->h_best_pack.u8(), ws->d_best_pack.ptr,
                                static_cast<std::size_t>(ws->best_pack_stride),
                                cudaMemcpyDeviceToHost, ctx->stream),
                "Memcpy best_pack D2H");
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[5], ctx->stream), "cudaEventRecord(ev5)");
        }
        ThrowIf(cudaStreamSynchronize(ctx->stream), "cudaStreamSynchronize");
		if (profile) {
			ConsumeCudaLinkageEncodeLastProfile(ws->encode_ws);
			ProfileShapeAndD2HBytes(/*B=*/1, /*npairs=*/Kp, m, profile);
		}

        if (profile) {
            float ms01 = 0, ms12 = 0, ms23 = 0, ms34 = 0, ms45 = 0, ms05 = 0;
            ThrowIf(cudaEventElapsedTime(&ms01, ws->ev[0], ws->ev[1]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms12, ws->ev[1], ws->ev[2]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms23, ws->ev[2], ws->ev[3]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms34, ws->ev[3], ws->ev[4]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms45, ws->ev[4], ws->ev[5]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms05, ws->ev[0], ws->ev[5]), "cudaEventElapsedTime");
            const auto add_ns = [](std::atomic<std::int64_t>* dst, float ms) {
                const auto ns = static_cast<std::int64_t>(static_cast<double>(ms) * 1e6);
                dst->fetch_add(ns, std::memory_order_relaxed);
            };
            g_calls.fetch_add(1, std::memory_order_relaxed);
            add_ns(&g_h2d_ns, ms01);
            add_ns(&g_residual_ns, ms12);
            add_ns(&g_encode_ns, ms23);
            add_ns(&g_reduce_ns, ms34);
            add_ns(&g_d2h_ns, ms45);
            add_ns(&g_total_ns, ms05);
        }

        const std::uint8_t* src = ws->h_best_pack.u8();
        const int best_idx_host = *reinterpret_cast<const int*>(src + 0);
        const float best_cost_host = *reinterpret_cast<const float*>(src + 4);

        if (best_idx_host < 0 || best_idx_host >= Kp) {
            throw std::runtime_error("EvaluateParentCandidatesBatchCuda: invalid best_idx.");
        }
        if (best_parent_local) {
            *best_parent_local = cand_parent_local[best_idx_host];
        }
        if (best_cost) {
            *best_cost = best_cost_host;
        }
        if (best_B) {
            best_B->assign(static_cast<std::size_t>(m), 0);
            std::memcpy(best_B->data(), src + 8,
                        sizeof(FullCode) * static_cast<std::size_t>(m));
        }
        if (best_a) {
            best_a->assign(static_cast<std::size_t>(m), 0.0f);
            std::memcpy(best_a->data(), src + static_cast<std::size_t>(ws->best_pack_a_off),
                        sizeof(float) * static_cast<std::size_t>(m));
        }
        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

bool EvaluateParentCandidatesBatchCudaDeviceRfull(CudaCtx* ctx,
                                                  const Precomp& pre_one,
                                                  const float* xi,
                                                  int d,
                                                  const int* cand_parent_local,
                                                  int Kp,
                                                  int icm_iters,
                                                  int ils_iters,
                                                  int perturb_k,
                                                  std::uint32_t seed,
                                                  std::uint64_t sample_id_offset,
                                                  int* best_parent_local,
                                                  float* best_cost,
                                                  std::vector<FullCode>* best_B,
                                                  std::vector<float>* best_a,
                                                  std::string* err) {
    try {
        if (!ctx) {
            throw std::runtime_error("EvaluateParentCandidatesBatchCudaDeviceRfull: ctx is null.");
        }
        if (Kp <= 0) {
            if (best_parent_local) *best_parent_local = -1;
            if (best_cost) *best_cost = std::numeric_limits<float>::infinity();
            if (best_B) best_B->clear();
            if (best_a) best_a->clear();
            return true;
        }
        if (!xi || !cand_parent_local) {
            throw std::runtime_error("EvaluateParentCandidatesBatchCudaDeviceRfull: null input pointer.");
        }
        if (d != pre_one.d) {
            throw std::runtime_error("EvaluateParentCandidatesBatchCudaDeviceRfull: d mismatch.");
        }
        const int m = pre_one.m;
        if (m <= 0) {
            throw std::runtime_error("EvaluateParentCandidatesBatchCudaDeviceRfull: invalid m.");
        }

        ThrowIf(cudaSetDevice(ctx->device), "cudaSetDevice");

        EvalWorkspace* ws = GetWorkspace(ctx);
        ws->Ensure(*ctx, d, Kp, m);

        if (!ws->d_R_full.ptr || ws->rfull_cols <= 0 || ws->rfull_d != d) {
            throw std::runtime_error("EvaluateParentCandidatesBatchCudaDeviceRfull: device R_full not uploaded.");
        }

        const bool profile = g_profile_enabled.load(std::memory_order_relaxed);
        if (profile) {
            ws->EnsureEvents(*ctx);
            const std::uint64_t bytes_est =
                EstimateBytesPerCtx(d, pre_one.H, m, /*B=*/1, /*npairs=*/Kp, /*rfull_cols=*/ws->rfull_cols);
            AtomicMax(&g_peak_d, d);
            AtomicMax(&g_peak_m, m);
            AtomicMax(&g_peak_H, pre_one.H);
            AtomicMax(&g_peak_B, 1);
            AtomicMax(&g_peak_npairs, Kp);
            AtomicMax(&g_peak_rfull_cols, ws->rfull_cols);
            AtomicMaxU64(&g_peak_bytes_per_ctx_est, bytes_est);
        }
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[0], ctx->stream), "cudaEventRecord(ev0)");
        }

        ThrowIf(cudaMemcpyAsync(ws->d_xi.ptr, xi, sizeof(float) * static_cast<std::size_t>(d),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy xi H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_cand.ptr, cand_parent_local,
                                sizeof(int) * static_cast<std::size_t>(Kp),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy cand H2D");
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[1], ctx->stream), "cudaEventRecord(ev1)");
        }

        // Gather residuals + norm2.
        GatherResidualAndNorm2FromRfull<<<Kp, 256, 0, ctx->stream>>>(
            ws->d_xi.ptr, ws->d_R_full.ptr, ws->d_cand.ptr,
            ws->d_Rp.ptr, ws->d_norm2.ptr, d, Kp);
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[2], ctx->stream), "cudaEventRecord(ev2)");
        }

        // Encode residuals + cost on GPU.
        std::string encode_err;
        if (!LinkageEncodeBatchCuda(*ctx, pre_one,
                                  ws->d_Rp.ptr, ws->d_norm2.ptr, Kp, icm_iters,
                                  ils_iters, perturb_k, seed, sample_id_offset,
                                  ws->d_B.ptr, ws->d_a.ptr, ws->d_cost.ptr,
                                  ws->encode_ws, &encode_err)) {
            throw std::runtime_error(std::string("LinkageEncodeBatchCuda failed: ") + encode_err);
        }
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[3], ctx->stream), "cudaEventRecord(ev3)");
        }

        // Argmin cost.
        const int reduce_threads = 256;
        const int reduce_blocks = (Kp + reduce_threads - 1) / reduce_threads;
        ReduceMinBlocks<<<reduce_blocks, reduce_threads, 0, ctx->stream>>>(
            ws->d_cost.ptr, Kp, ws->d_partial_cost.ptr, ws->d_partial_idx.ptr);
        ReduceMinFinal<<<1, 256, 0, ctx->stream>>>(
            ws->d_partial_cost.ptr, ws->d_partial_idx.ptr, reduce_blocks,
            ws->d_best_cost.ptr, ws->d_best_idx.ptr);
        GatherBest<<<1, 256, 0, ctx->stream>>>(m, ws->d_B.ptr, ws->d_a.ptr,
                                               ws->d_best_idx.ptr, ws->d_best_B.ptr, ws->d_best_a.ptr);
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[4], ctx->stream), "cudaEventRecord(ev4)");
        }

        PackSingleBestKernel<<<1, 256, 0, ctx->stream>>>(
            m,
            ws->best_pack_a_off,
            ws->best_pack_stride,
            ws->d_best_idx.ptr,
            ws->d_best_cost.ptr,
            ws->d_best_B.ptr,
            ws->d_best_a.ptr,
            ws->d_best_pack.ptr);
        ThrowIf(cudaMemcpyAsync(ws->h_best_pack.u8(), ws->d_best_pack.ptr,
                                static_cast<std::size_t>(ws->best_pack_stride),
                                cudaMemcpyDeviceToHost, ctx->stream),
                "Memcpy best_pack D2H");
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[5], ctx->stream), "cudaEventRecord(ev5)");
        }
	        ThrowIf(cudaStreamSynchronize(ctx->stream), "cudaStreamSynchronize");
	        if (profile) {
	            ConsumeCudaLinkageEncodeLastProfile(ws->encode_ws);
	            ProfileShapeAndD2HBytes(/*B=*/1, /*npairs=*/Kp, m, profile);
	        }

        if (profile) {
            float ms01 = 0, ms12 = 0, ms23 = 0, ms34 = 0, ms45 = 0, ms05 = 0;
            ThrowIf(cudaEventElapsedTime(&ms01, ws->ev[0], ws->ev[1]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms12, ws->ev[1], ws->ev[2]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms23, ws->ev[2], ws->ev[3]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms34, ws->ev[3], ws->ev[4]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms45, ws->ev[4], ws->ev[5]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms05, ws->ev[0], ws->ev[5]), "cudaEventElapsedTime");
            const auto add_ns = [](std::atomic<std::int64_t>* dst, float ms) {
                const auto ns = static_cast<std::int64_t>(static_cast<double>(ms) * 1e6);
                dst->fetch_add(ns, std::memory_order_relaxed);
            };
            g_calls.fetch_add(1, std::memory_order_relaxed);
            add_ns(&g_h2d_ns, ms01);
            add_ns(&g_residual_ns, ms12);
            add_ns(&g_encode_ns, ms23);
            add_ns(&g_reduce_ns, ms34);
            add_ns(&g_d2h_ns, ms45);
            add_ns(&g_total_ns, ms05);
        }

        const std::uint8_t* src = ws->h_best_pack.u8();
        const int best_idx_host = *reinterpret_cast<const int*>(src + 0);
        const float best_cost_host = *reinterpret_cast<const float*>(src + 4);
        if (best_idx_host < 0 || best_idx_host >= Kp) {
            throw std::runtime_error("EvaluateParentCandidatesBatchCudaDeviceRfull: invalid best_idx.");
        }
        if (best_parent_local) {
            *best_parent_local = cand_parent_local[best_idx_host];
        }
        if (best_cost) {
            *best_cost = best_cost_host;
        }
        if (best_B) {
            best_B->assign(static_cast<std::size_t>(m), 0);
            std::memcpy(best_B->data(), src + 8,
                        sizeof(FullCode) * static_cast<std::size_t>(m));
        }
        if (best_a) {
            best_a->assign(static_cast<std::size_t>(m), 0.0f);
            std::memcpy(best_a->data(), src + static_cast<std::size_t>(ws->best_pack_a_off),
                        sizeof(float) * static_cast<std::size_t>(m));
        }
        return true;
    } catch (const std::exception& e) {
        if (err) {
            *err = e.what();
        }
        return false;
    }
}

bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfull(CudaCtx* ctx,
                                                          const Precomp& pre_one,
                                                          const float* X_block,
                                                          int d,
                                                          int B,
                                                          const int* cand_parent_local_flat,
                                                          const int* pair_node,
                                                          const int* cand_offsets,
                                                          int Npairs,
                                                          int icm_iters,
                                                          int ils_iters,
                                                          int perturb_k,
                                                          std::uint32_t seed,
                                                          std::uint64_t sample_id_offset,
                                                          int* best_parent_local_out,
                                                          float* best_cost_out,
                                                          FullCode* best_B_out,
                                                          float* best_a_out,
                                                          std::string* err) {
    try {
        if (!ctx) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfull: ctx is null.");
        }
        if (d <= 0 || B <= 0) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfull: invalid shape.");
        }
        if (!X_block || !cand_parent_local_flat || !pair_node || !cand_offsets ||
            !best_parent_local_out || !best_cost_out || !best_B_out || !best_a_out) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfull: null input pointer.");
        }
        if (d != pre_one.d) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfull: d mismatch.");
        }
        const int m = pre_one.m;
        if (m <= 0) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfull: invalid m.");
        }
        if (Npairs < 0) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfull: invalid Npairs.");
        }
        const int expected_npairs = cand_offsets[B];
        if (expected_npairs != Npairs) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfull: Npairs != cand_offsets[B].");
        }
        if (Npairs == 0) {
            for (int i = 0; i < B; ++i) {
                best_parent_local_out[i] = -1;
                best_cost_out[i] = std::numeric_limits<float>::infinity();
            }
            std::fill(best_B_out, best_B_out + static_cast<std::size_t>(B) * static_cast<std::size_t>(m), 0);
            std::fill(best_a_out, best_a_out + static_cast<std::size_t>(B) * static_cast<std::size_t>(m), 0.0f);
            return true;
        }

        ThrowIf(cudaSetDevice(ctx->device), "cudaSetDevice");

        EvalWorkspace* ws = GetWorkspace(ctx);
        ws->EnsureBatch(*ctx, d, B, Npairs, m);

        if (!ws->d_R_full.ptr || ws->rfull_cols <= 0 || ws->rfull_d != d) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfull: device R_full not uploaded.");
        }

        const bool profile = g_profile_enabled.load(std::memory_order_relaxed);
        if (profile) {
            ws->EnsureEvents(*ctx);
            const std::uint64_t bytes_est =
                EstimateBytesPerCtx(d, pre_one.H, m, /*B=*/B, /*npairs=*/Npairs, /*rfull_cols=*/ws->rfull_cols);
            AtomicMax(&g_peak_d, d);
            AtomicMax(&g_peak_m, m);
            AtomicMax(&g_peak_H, pre_one.H);
            AtomicMax(&g_peak_B, B);
            AtomicMax(&g_peak_npairs, Npairs);
            AtomicMax(&g_peak_rfull_cols, ws->rfull_cols);
            AtomicMaxU64(&g_peak_bytes_per_ctx_est, bytes_est);
            ThrowIf(cudaEventRecord(ws->ev[0], ctx->stream), "cudaEventRecord(ev0)");
        }

        ThrowIf(cudaMemcpyAsync(ws->d_xblock.ptr, X_block,
                                sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(B),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy X_block H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_cand.ptr, cand_parent_local_flat,
                                sizeof(int) * static_cast<std::size_t>(Npairs),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy cand_flat H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_pair_node.ptr, pair_node,
                                sizeof(int) * static_cast<std::size_t>(Npairs),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy pair_node H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_offsets.ptr, cand_offsets,
                                sizeof(int) * static_cast<std::size_t>(B + 1),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy offsets H2D");

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[1], ctx->stream), "cudaEventRecord(ev1)");
        }

        GatherResidualAndNorm2FromRfullPairs<<<Npairs, 256, 0, ctx->stream>>>(
            ws->d_xblock.ptr, ws->d_R_full.ptr, ws->d_cand.ptr, ws->d_pair_node.ptr,
            ws->d_Rp.ptr, ws->d_norm2.ptr, d, Npairs);

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[2], ctx->stream), "cudaEventRecord(ev2)");
        }

        std::string encode_err;
        if (!LinkageEncodeBatchCuda(*ctx, pre_one,
                                  ws->d_Rp.ptr, ws->d_norm2.ptr, Npairs, icm_iters,
                                  ils_iters, perturb_k, seed, sample_id_offset,
                                  ws->d_B.ptr, ws->d_a.ptr, ws->d_cost.ptr,
                                  ws->encode_ws, &encode_err)) {
            throw std::runtime_error(std::string("LinkageEncodeBatchCuda failed: ") + encode_err);
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[3], ctx->stream), "cudaEventRecord(ev3)");
        }

        SegmentedArgMinCost<<<B, 256, 0, ctx->stream>>>(
            ws->d_cost.ptr, ws->d_offsets.ptr, B,
            ws->d_best_cost_batch.ptr, ws->d_best_pair_batch.ptr);

        const int parent_blocks = (B + 255) / 256;
        GatherBestParentPerNode<<<parent_blocks, 256, 0, ctx->stream>>>(
            ws->d_best_pair_batch.ptr, ws->d_cand.ptr, B, ws->d_best_parent_batch.ptr);

        const int total = B * m;
        const int codes_blocks = (total + 255) / 256;
        GatherBestCodesPerNode<<<codes_blocks, 256, 0, ctx->stream>>>(
            B, m, ws->d_best_pair_batch.ptr, ws->d_B.ptr, ws->d_a.ptr,
            ws->d_best_B_batch.ptr, ws->d_best_a_batch.ptr);

        const int pack_blocks = (B + 255) / 256;
        PackManyNodesBestKernel<<<pack_blocks, 256, 0, ctx->stream>>>(
            B, m,
            ws->best_pack_a_off,
            ws->best_pack_stride,
            ws->d_best_parent_batch.ptr,
            ws->d_best_cost_batch.ptr,
            ws->d_best_B_batch.ptr,
            ws->d_best_a_batch.ptr,
            ws->d_best_pack.ptr);

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[4], ctx->stream), "cudaEventRecord(ev4)");
        }

        ThrowIf(cudaMemcpyAsync(ws->h_best_pack.u8(), ws->d_best_pack.ptr,
                                static_cast<std::size_t>(B) * static_cast<std::size_t>(ws->best_pack_stride),
                                cudaMemcpyDeviceToHost, ctx->stream),
                "Memcpy best_pack D2H");
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[5], ctx->stream), "cudaEventRecord(ev5)");
        }
        ThrowIf(cudaStreamSynchronize(ctx->stream), "cudaStreamSynchronize");

        // Unpack on CPU.
        for (int i = 0; i < B; ++i) {
            const std::uint8_t* src =
                ws->h_best_pack.u8() +
                static_cast<std::size_t>(i) * static_cast<std::size_t>(ws->best_pack_stride);
            best_parent_local_out[i] = *reinterpret_cast<const int*>(src + 0);
            best_cost_out[i] = *reinterpret_cast<const float*>(src + 4);
            const auto* src_B = reinterpret_cast<const FullCode*>(src + 8);
            FullCode* dst_B = best_B_out + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
            for (int l = 0; l < m; ++l) {
                dst_B[l] = src_B[l];
            }
            const auto* src_a = reinterpret_cast<const float*>(src + ws->best_pack_a_off);
            float* dst_a = best_a_out + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
            for (int l = 0; l < m; ++l) {
                dst_a[l] = src_a[l];
            }
        }

        if (profile) {
            ConsumeCudaLinkageEncodeLastProfile(ws->encode_ws);
            ProfileShapeAndD2HBytes(/*B=*/B, /*npairs=*/Npairs, m, profile);
        }

        if (profile) {
            float ms01 = 0, ms12 = 0, ms23 = 0, ms34 = 0, ms45 = 0, ms05 = 0;
            ThrowIf(cudaEventElapsedTime(&ms01, ws->ev[0], ws->ev[1]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms12, ws->ev[1], ws->ev[2]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms23, ws->ev[2], ws->ev[3]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms34, ws->ev[3], ws->ev[4]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms45, ws->ev[4], ws->ev[5]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms05, ws->ev[0], ws->ev[5]), "cudaEventElapsedTime");
            const auto add_ns = [](std::atomic<std::int64_t>* dst, float ms) {
                const auto ns = static_cast<std::int64_t>(static_cast<double>(ms) * 1e6);
                dst->fetch_add(ns, std::memory_order_relaxed);
            };
            g_calls.fetch_add(1, std::memory_order_relaxed);
            add_ns(&g_h2d_ns, ms01);
            add_ns(&g_residual_ns, ms12);
            add_ns(&g_encode_ns, ms23);
            add_ns(&g_reduce_ns, ms34);
            add_ns(&g_d2h_ns, ms45);
            add_ns(&g_total_ns, ms05);
        }
        return true;
    } catch (const std::exception& e) {
        if (err) {
            *err = e.what();
        }
        return false;
    }
}

static bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseOneShot(
    CudaCtx* ctx,
    const Precomp& pre_one,
    const float* X_block,
    int d,
    int B,
    const int* cand_parent_local_flat,
    const int* pair_node,
    const int* cand_offsets,
    int Npairs,
    int icm_iters,
    int ils_iters,
    int perturb_k,
    std::uint32_t seed,
    const std::uint64_t* node_sample_id_base,
    const int* pair_t_override,
    int* best_parent_local_out,
    float* best_cost_out,
    FullCode* best_B_out,
    float* best_a_out,
    std::string* err) {
    try {
        if (!ctx) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBase: ctx is null.");
        }
        if (d <= 0 || B <= 0) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBase: invalid shape.");
        }
        if (!X_block || !cand_parent_local_flat || !pair_node || !cand_offsets || !node_sample_id_base ||
            !best_parent_local_out || !best_cost_out || !best_B_out || !best_a_out) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBase: null input pointer.");
        }
        if (d != pre_one.d) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBase: d mismatch.");
        }
        const int m = pre_one.m;
        if (m <= 0) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBase: invalid m.");
        }
        if (Npairs < 0) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBase: invalid Npairs.");
        }
        const int expected_npairs = cand_offsets[B];
        if (expected_npairs != Npairs) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBase: Npairs != cand_offsets[B].");
        }
        if (Npairs == 0) {
            for (int i = 0; i < B; ++i) {
                best_parent_local_out[i] = -1;
                best_cost_out[i] = std::numeric_limits<float>::infinity();
            }
            std::fill(best_B_out, best_B_out + static_cast<std::size_t>(B) * static_cast<std::size_t>(m), 0);
            std::fill(best_a_out, best_a_out + static_cast<std::size_t>(B) * static_cast<std::size_t>(m), 0.0f);
            return true;
        }

        ThrowIf(cudaSetDevice(ctx->device), "cudaSetDevice");

        EvalWorkspace* ws = GetWorkspace(ctx);
        ws->EnsureBatch(*ctx, d, B, Npairs, m);

        if (!ws->d_R_full.ptr || ws->rfull_cols <= 0 || ws->rfull_d != d) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBase: device R_full not uploaded.");
        }

        const bool profile = g_profile_enabled.load(std::memory_order_relaxed);
        if (profile) {
            ws->EnsureEvents(*ctx);
            const std::uint64_t bytes_est =
                EstimateBytesPerCtx(d, pre_one.H, m, /*B=*/B, /*npairs=*/Npairs, /*rfull_cols=*/ws->rfull_cols) +
                static_cast<std::uint64_t>(sizeof(std::uint64_t)) * static_cast<std::uint64_t>(B) +
                static_cast<std::uint64_t>(sizeof(std::uint64_t)) * static_cast<std::uint64_t>(Npairs);
            AtomicMax(&g_peak_d, d);
            AtomicMax(&g_peak_m, m);
            AtomicMax(&g_peak_H, pre_one.H);
            AtomicMax(&g_peak_B, B);
            AtomicMax(&g_peak_npairs, Npairs);
            AtomicMax(&g_peak_rfull_cols, ws->rfull_cols);
            AtomicMaxU64(&g_peak_bytes_per_ctx_est, bytes_est);
            ThrowIf(cudaEventRecord(ws->ev[0], ctx->stream), "cudaEventRecord(ev0)");
        }

        ThrowIf(cudaMemcpyAsync(ws->d_xblock.ptr, X_block,
                                sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(B),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy X_block H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_cand.ptr, cand_parent_local_flat,
                                sizeof(int) * static_cast<std::size_t>(Npairs),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy cand_flat H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_pair_node.ptr, pair_node,
                                sizeof(int) * static_cast<std::size_t>(Npairs),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy pair_node H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_offsets.ptr, cand_offsets,
                                sizeof(int) * static_cast<std::size_t>(B + 1),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy offsets H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_node_sample_id_base.ptr, node_sample_id_base,
                                sizeof(std::uint64_t) * static_cast<std::size_t>(B),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy node_sample_id_base H2D");
        if (pair_t_override) {
            ThrowIf(cudaMemcpyAsync(ws->d_pair_t_override.ptr, pair_t_override,
                                    sizeof(int) * static_cast<std::size_t>(Npairs),
                                    cudaMemcpyHostToDevice, ctx->stream),
                    "Memcpy pair_t_override H2D");
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[1], ctx->stream), "cudaEventRecord(ev1)");
        }

        // Build residuals + norm2.
        GatherResidualAndNorm2FromRfullPairs<<<Npairs, 256, 0, ctx->stream>>>(
            ws->d_xblock.ptr, ws->d_R_full.ptr, ws->d_cand.ptr, ws->d_pair_node.ptr,
            ws->d_Rp.ptr, ws->d_norm2.ptr, d, Npairs);

        // Build per-pair sample IDs so ILS perturbation is stable independent of batching.
        {
            const int threads = 256;
            const int blocks = (Npairs + threads - 1) / threads;
            BuildPairSampleIdsFromNodeBase<<<blocks, threads, 0, ctx->stream>>>(
                ws->d_node_sample_id_base.ptr,
                ws->d_pair_node.ptr,
                ws->d_offsets.ptr,
                /*pair_t_override=*/pair_t_override ? ws->d_pair_t_override.ptr : nullptr,
                ws->d_sample_ids.ptr,
                B,
                Npairs);
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[2], ctx->stream), "cudaEventRecord(ev2)");
        }

        std::string encode_err;
        if (!LinkageEncodeBatchCudaWithSampleIds(*ctx, pre_one,
                                               ws->d_Rp.ptr, ws->d_norm2.ptr,
                                               ws->d_sample_ids.ptr,
                                               Npairs, icm_iters,
                                               ils_iters, perturb_k, seed,
                                               ws->d_B.ptr, ws->d_a.ptr, ws->d_cost.ptr,
                                               ws->encode_ws, &encode_err)) {
            throw std::runtime_error(std::string("LinkageEncodeBatchCudaWithSampleIds failed: ") + encode_err);
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[3], ctx->stream), "cudaEventRecord(ev3)");
        }

        SegmentedArgMinCost<<<B, 256, 0, ctx->stream>>>(
            ws->d_cost.ptr, ws->d_offsets.ptr, B,
            ws->d_best_cost_batch.ptr, ws->d_best_pair_batch.ptr);

        const int parent_blocks = (B + 255) / 256;
        GatherBestParentPerNode<<<parent_blocks, 256, 0, ctx->stream>>>(
            ws->d_best_pair_batch.ptr, ws->d_cand.ptr, B, ws->d_best_parent_batch.ptr);

        const int total = B * m;
        const int codes_blocks = (total + 255) / 256;
        GatherBestCodesPerNode<<<codes_blocks, 256, 0, ctx->stream>>>(
            B, m, ws->d_best_pair_batch.ptr, ws->d_B.ptr, ws->d_a.ptr,
            ws->d_best_B_batch.ptr, ws->d_best_a_batch.ptr);

        const int pack_blocks = (B + 255) / 256;
        PackManyNodesBestKernel<<<pack_blocks, 256, 0, ctx->stream>>>(
            B, m,
            ws->best_pack_a_off,
            ws->best_pack_stride,
            ws->d_best_parent_batch.ptr,
            ws->d_best_cost_batch.ptr,
            ws->d_best_B_batch.ptr,
            ws->d_best_a_batch.ptr,
            ws->d_best_pack.ptr);

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[4], ctx->stream), "cudaEventRecord(ev4)");
        }

        ThrowIf(cudaMemcpyAsync(ws->h_best_pack.u8(), ws->d_best_pack.ptr,
                                static_cast<std::size_t>(B) * static_cast<std::size_t>(ws->best_pack_stride),
                                cudaMemcpyDeviceToHost, ctx->stream),
                "Memcpy best_pack D2H");
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[5], ctx->stream), "cudaEventRecord(ev5)");
        }
        ThrowIf(cudaStreamSynchronize(ctx->stream), "cudaStreamSynchronize");

        // Unpack on CPU.
        for (int i = 0; i < B; ++i) {
            const std::uint8_t* src =
                ws->h_best_pack.u8() +
                static_cast<std::size_t>(i) * static_cast<std::size_t>(ws->best_pack_stride);
            best_parent_local_out[i] = *reinterpret_cast<const int*>(src + 0);
            best_cost_out[i] = *reinterpret_cast<const float*>(src + 4);
            const auto* src_B = reinterpret_cast<const FullCode*>(src + 8);
            FullCode* dst_B = best_B_out + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
            for (int l = 0; l < m; ++l) {
                dst_B[l] = src_B[l];
            }
            const auto* src_a = reinterpret_cast<const float*>(src + ws->best_pack_a_off);
            float* dst_a = best_a_out + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
            for (int l = 0; l < m; ++l) {
                dst_a[l] = src_a[l];
            }
        }

        if (profile) {
            ConsumeCudaLinkageEncodeLastProfile(ws->encode_ws);
            ProfileShapeAndD2HBytes(/*B=*/B, /*npairs=*/Npairs, m, profile);
        }

        if (profile) {
            float ms01 = 0, ms12 = 0, ms23 = 0, ms34 = 0, ms45 = 0, ms05 = 0;
            ThrowIf(cudaEventElapsedTime(&ms01, ws->ev[0], ws->ev[1]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms12, ws->ev[1], ws->ev[2]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms23, ws->ev[2], ws->ev[3]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms34, ws->ev[3], ws->ev[4]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms45, ws->ev[4], ws->ev[5]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms05, ws->ev[0], ws->ev[5]), "cudaEventElapsedTime");
            const auto add_ns = [](std::atomic<std::int64_t>* dst, float ms) {
                const auto ns = static_cast<std::int64_t>(static_cast<double>(ms) * 1e6);
                dst->fetch_add(ns, std::memory_order_relaxed);
            };
            g_calls.fetch_add(1, std::memory_order_relaxed);
            add_ns(&g_h2d_ns, ms01);
            add_ns(&g_residual_ns, ms12);
            add_ns(&g_encode_ns, ms23);
            add_ns(&g_reduce_ns, ms34);
            add_ns(&g_d2h_ns, ms45);
            add_ns(&g_total_ns, ms05);
        }
        return true;
    } catch (const std::exception& e) {
        if (err) {
            *err = e.what();
        }
        return false;
    }
}

bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBase(
    CudaCtx* ctx,
    const Precomp& pre_one,
    const float* X_block,
    int d,
    int B,
    const int* cand_parent_local_flat,
    const int* pair_node,
    const int* cand_offsets,
    int Npairs,
    int icm_iters,
    int ils_iters,
    int perturb_k,
    std::uint32_t seed,
    const std::uint64_t* node_sample_id_base,
    int* best_parent_local_out,
    float* best_cost_out,
    FullCode* best_B_out,
    float* best_a_out,
    std::string* err) {
    return EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseEx(
        ctx,
        pre_one,
        X_block,
        d,
        B,
        cand_parent_local_flat,
        pair_node,
        cand_offsets,
        Npairs,
        icm_iters,
        ils_iters,
        perturb_k,
        seed,
        node_sample_id_base,
        /*pair_t_override=*/nullptr,
        best_parent_local_out,
        best_cost_out,
        best_B_out,
        best_a_out,
        err);
}

bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseEx(
    CudaCtx* ctx,
    const Precomp& pre_one,
    const float* X_block,
    int d,
    int B,
    const int* cand_parent_local_flat,
    const int* pair_node,
    const int* cand_offsets,
    int Npairs,
    int icm_iters,
    int ils_iters,
    int perturb_k,
    std::uint32_t seed,
    const std::uint64_t* node_sample_id_base,
    const int* pair_t_override,
    int* best_parent_local_out,
    float* best_cost_out,
    FullCode* best_B_out,
    float* best_a_out,
    std::string* err) {
    int chunk_max_pairs_cfg = g_chunk_max_pairs.load(std::memory_order_relaxed);
    if (chunk_max_pairs_cfg <= 0) {
        const int mem_budget_mb = g_mem_budget_mb.load(std::memory_order_relaxed);
        int rfull_cols = 0;
        if (ctx) {
            EvalWorkspace* ws = GetWorkspace(ctx);
            if (ws && ws->rfull_d == d && ws->rfull_cols > 0) {
                rfull_cols = ws->rfull_cols;
            }
        }
        const int B_guess = std::max(1, std::min(B, 64));
        chunk_max_pairs_cfg = EffectiveChunkMaxPairsFromBudget(d, pre_one.H, pre_one.m, B_guess, rfull_cols, mem_budget_mb);
    }
	    if (chunk_max_pairs_cfg <= 0 || Npairs <= chunk_max_pairs_cfg) {
	        return EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseOneShot(
	            ctx, pre_one, X_block, d, B, cand_parent_local_flat, pair_node, cand_offsets, Npairs,
	            icm_iters, ils_iters, perturb_k, seed, node_sample_id_base, pair_t_override,
	            best_parent_local_out, best_cost_out, best_B_out, best_a_out, err);
	    }
	    if (B <= 0 || !cand_offsets || cand_offsets[B] != Npairs) {
	        return EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseOneShot(
	            ctx, pre_one, X_block, d, B, cand_parent_local_flat, pair_node, cand_offsets, Npairs,
	            icm_iters, ils_iters, perturb_k, seed, node_sample_id_base, pair_t_override,
	            best_parent_local_out, best_cost_out, best_B_out, best_a_out, err);
	    }

    thread_local std::vector<int> tls_offsets;
    thread_local std::vector<int> tls_pair_node;
    thread_local std::vector<FullCode> tls_tmp_B;
    thread_local std::vector<float> tls_tmp_a;

    const int m = pre_one.m;
    tls_tmp_B.resize(static_cast<std::size_t>(std::max(0, m)));
    tls_tmp_a.resize(static_cast<std::size_t>(std::max(0, m)));

    int node0 = 0;
    while (node0 < B) {
        const int pairs_begin = cand_offsets[node0];
        int node1 = node0;
        while (node1 + 1 <= B) {
            const int pairs_end_next = cand_offsets[node1 + 1];
            if (pairs_end_next - pairs_begin > chunk_max_pairs_cfg) {
                break;
            }
            ++node1;
        }
        node1 = std::max(node1, node0 + 1);

        const int pairs_end = cand_offsets[node1];
        const int npairs_chunk = pairs_end - pairs_begin;
        const int Bchunk = node1 - node0;
        if (npairs_chunk <= 0) {
            for (int i = node0; i < node1; ++i) {
                best_parent_local_out[i] = -1;
                best_cost_out[i] = std::numeric_limits<float>::infinity();
            }
            std::fill(best_B_out + static_cast<std::size_t>(node0) * static_cast<std::size_t>(m),
                      best_B_out + static_cast<std::size_t>(node1) * static_cast<std::size_t>(m),
                      0);
            std::fill(best_a_out + static_cast<std::size_t>(node0) * static_cast<std::size_t>(m),
                      best_a_out + static_cast<std::size_t>(node1) * static_cast<std::size_t>(m),
                      0.0f);
            node0 = node1;
            continue;
        }

        if (npairs_chunk <= chunk_max_pairs_cfg) {
            tls_offsets.resize(static_cast<std::size_t>(Bchunk + 1));
            for (int i = 0; i <= Bchunk; ++i) {
                tls_offsets[static_cast<std::size_t>(i)] = cand_offsets[node0 + i] - pairs_begin;
            }
            tls_pair_node.resize(static_cast<std::size_t>(npairs_chunk));
            for (int i = 0; i < Bchunk; ++i) {
                const int s = tls_offsets[static_cast<std::size_t>(i)];
                const int t = tls_offsets[static_cast<std::size_t>(i + 1)];
                for (int p = s; p < t; ++p) {
                    tls_pair_node[static_cast<std::size_t>(p)] = i;
                }
            }

            std::string local_err;
            const bool ok = EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseOneShot(
                ctx, pre_one,
                X_block + static_cast<std::size_t>(node0) * static_cast<std::size_t>(d), d, Bchunk,
                cand_parent_local_flat + pairs_begin,
                tls_pair_node.data(),
                tls_offsets.data(),
                npairs_chunk,
                icm_iters, ils_iters, perturb_k, seed,
                node_sample_id_base + static_cast<std::size_t>(node0),
                pair_t_override ? (pair_t_override + pairs_begin) : nullptr,
                best_parent_local_out + node0,
                best_cost_out + node0,
                best_B_out + static_cast<std::size_t>(node0) * static_cast<std::size_t>(m),
                best_a_out + static_cast<std::size_t>(node0) * static_cast<std::size_t>(m),
                &local_err);
            if (!ok) {
                if (err) *err = local_err;
                return false;
            }
            node0 = node1;
            continue;
        }

        // Single-node chunk still exceeds cap: split by pairs (exact semantics; preserve tie-breaking by processing order).
        if (Bchunk != 1) {
            return EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseOneShot(
                ctx, pre_one, X_block, d, B, cand_parent_local_flat, pair_node, cand_offsets, Npairs,
                icm_iters, ils_iters, perturb_k, seed, node_sample_id_base, pair_t_override,
                best_parent_local_out, best_cost_out, best_B_out, best_a_out, err);
        }

        const int node_idx = node0;
        best_parent_local_out[node_idx] = -1;
        best_cost_out[node_idx] = std::numeric_limits<float>::infinity();
        std::fill(best_B_out + static_cast<std::size_t>(node_idx) * static_cast<std::size_t>(m),
                  best_B_out + static_cast<std::size_t>(node_idx + 1) * static_cast<std::size_t>(m),
                  0);
        std::fill(best_a_out + static_cast<std::size_t>(node_idx) * static_cast<std::size_t>(m),
                  best_a_out + static_cast<std::size_t>(node_idx + 1) * static_cast<std::size_t>(m),
                  0.0f);

        tls_pair_node.resize(static_cast<std::size_t>(chunk_max_pairs_cfg));
        std::fill(tls_pair_node.begin(), tls_pair_node.end(), 0);
        int p0 = 0;
        while (p0 < npairs_chunk) {
            const int len = std::min(chunk_max_pairs_cfg, npairs_chunk - p0);
            const int off2[2] = {0, len};
            std::string local_err;
            int best_parent_tmp = -1;
            float best_cost_tmp = std::numeric_limits<float>::infinity();
            const bool ok = EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseOneShot(
                ctx, pre_one,
                X_block + static_cast<std::size_t>(node_idx) * static_cast<std::size_t>(d), d, 1,
                cand_parent_local_flat + pairs_begin + p0,
                tls_pair_node.data(),
                off2,
                len,
                icm_iters, ils_iters, perturb_k, seed,
                node_sample_id_base + static_cast<std::size_t>(node_idx),
                pair_t_override ? (pair_t_override + pairs_begin + p0) : nullptr,
                &best_parent_tmp,
                &best_cost_tmp,
                tls_tmp_B.data(),
                tls_tmp_a.data(),
                &local_err);
            if (!ok) {
                if (err) *err = local_err;
                return false;
            }
            if (best_cost_tmp < best_cost_out[node_idx]) {
                best_parent_local_out[node_idx] = best_parent_tmp;
                best_cost_out[node_idx] = best_cost_tmp;
                std::copy(tls_tmp_B.begin(), tls_tmp_B.end(),
                          best_B_out + static_cast<std::size_t>(node_idx) * static_cast<std::size_t>(m));
                std::copy(tls_tmp_a.begin(), tls_tmp_a.end(),
                          best_a_out + static_cast<std::size_t>(node_idx) * static_cast<std::size_t>(m));
            }
            p0 += len;
        }
        node0 = node1;
    }
    return true;
}

static bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBaseOneShot(
    CudaCtx* ctx,
    const Precomp& pre_one,
    int forced_root_code,
    const float* X_block,
    int d,
    int B,
    const int* cand_parent_local_flat,
    const int* pair_node,
    const int* cand_offsets,
    int Npairs,
    int icm_iters,
    int ils_iters,
    int perturb_k,
    std::uint32_t seed,
    const std::uint64_t* node_sample_id_base,
    int* best_parent_local_out,
    float* best_cost_out,
    FullCode* best_B_out,
    float* best_a_out,
    std::string* err) {
    try {
        if (!ctx) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBase: ctx is null.");
        }
        if (d <= 0 || B <= 0) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBase: invalid shape.");
        }
        if (d != pre_one.d) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBase: d mismatch.");
        }
        const int m = pre_one.m;
        if (m <= 0) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBase: invalid m.");
        }
        if (Npairs < 0) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBase: invalid Npairs.");
        }
        if (!cand_offsets || !best_parent_local_out || !best_cost_out || !best_B_out || !best_a_out) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBase: null input pointer.");
        }
        const int expected_npairs = cand_offsets[B];
        if (expected_npairs != Npairs) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBase: Npairs != cand_offsets[B].");
        }
        if (Npairs == 0) {
            for (int i = 0; i < B; ++i) {
                best_parent_local_out[i] = -1;
                best_cost_out[i] = std::numeric_limits<float>::infinity();
            }
            std::fill(best_B_out, best_B_out + static_cast<std::size_t>(B) * static_cast<std::size_t>(m), 0);
            std::fill(best_a_out, best_a_out + static_cast<std::size_t>(B) * static_cast<std::size_t>(m), 0.0f);
            return true;
        }
        if (!X_block || !cand_parent_local_flat || !pair_node || !node_sample_id_base) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBase: null input pointer.");
        }

        ThrowIf(cudaSetDevice(ctx->device), "cudaSetDevice");

        EvalWorkspace* ws = GetWorkspace(ctx);
        ws->EnsureBatch(*ctx, d, B, Npairs, m);

        if (!ws->d_R_full.ptr || ws->rfull_cols <= 0 || ws->rfull_d != d) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBase: device R_full not uploaded.");
        }

        const bool profile = g_profile_enabled.load(std::memory_order_relaxed);
        if (profile) {
            ws->EnsureEvents(*ctx);
            const std::uint64_t bytes_est =
                EstimateBytesPerCtx(d, pre_one.H, m, /*B=*/B, /*npairs=*/Npairs, /*rfull_cols=*/ws->rfull_cols) +
                static_cast<std::uint64_t>(sizeof(std::uint64_t)) * static_cast<std::uint64_t>(B) +
                static_cast<std::uint64_t>(sizeof(std::uint64_t)) * static_cast<std::uint64_t>(Npairs) +
                static_cast<std::uint64_t>(sizeof(int)) * static_cast<std::uint64_t>(Npairs);
            AtomicMax(&g_peak_d, d);
            AtomicMax(&g_peak_m, m);
            AtomicMax(&g_peak_H, pre_one.H);
            AtomicMax(&g_peak_B, B);
            AtomicMax(&g_peak_npairs, Npairs);
            AtomicMax(&g_peak_rfull_cols, ws->rfull_cols);
            AtomicMaxU64(&g_peak_bytes_per_ctx_est, bytes_est);
            ThrowIf(cudaEventRecord(ws->ev[0], ctx->stream), "cudaEventRecord(ev0)");
        }

        ThrowIf(cudaMemcpyAsync(ws->d_xblock.ptr, X_block,
                                sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(B),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy X_block H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_cand.ptr, cand_parent_local_flat,
                                sizeof(int) * static_cast<std::size_t>(Npairs),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy cand_flat H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_pair_node.ptr, pair_node,
                                sizeof(int) * static_cast<std::size_t>(Npairs),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy pair_node H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_offsets.ptr, cand_offsets,
                                sizeof(int) * static_cast<std::size_t>(B + 1),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy offsets H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_node_sample_id_base.ptr, node_sample_id_base,
                                sizeof(std::uint64_t) * static_cast<std::size_t>(B),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy node_sample_id_base H2D");

        // forced_root is constant, fill on device.
        {
            const int threads = 256;
            const int blocks = (Npairs + threads - 1) / threads;
            FillI32<<<blocks, threads, 0, ctx->stream>>>(ws->d_forced_root.ptr, Npairs, forced_root_code);
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[1], ctx->stream), "cudaEventRecord(ev1)");
        }

        // Build residuals + norm2.
        GatherResidualAndNorm2FromRfullPairs<<<Npairs, 256, 0, ctx->stream>>>(
            ws->d_xblock.ptr, ws->d_R_full.ptr, ws->d_cand.ptr, ws->d_pair_node.ptr,
            ws->d_Rp.ptr, ws->d_norm2.ptr, d, Npairs);

        // Build per-pair sample IDs so ILS perturbation is stable independent of batching.
        {
            const int threads = 256;
            const int blocks = (Npairs + threads - 1) / threads;
            BuildPairSampleIdsFromNodeBase<<<blocks, threads, 0, ctx->stream>>>(
                ws->d_node_sample_id_base.ptr,
                ws->d_pair_node.ptr,
                ws->d_offsets.ptr,
                /*pair_t_override=*/nullptr,
                ws->d_sample_ids.ptr,
                B,
                Npairs);
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[2], ctx->stream), "cudaEventRecord(ev2)");
        }

        std::string encode_err;
        if (!LinkageEncodeBatchCudaForcedRootWithSampleIds(*ctx, pre_one,
                                                         ws->d_Rp.ptr, ws->d_norm2.ptr,
                                                         ws->d_forced_root.ptr,
                                                         ws->d_sample_ids.ptr,
                                                         Npairs, icm_iters,
                                                         ils_iters, perturb_k, seed,
                                                         ws->d_B.ptr, ws->d_a.ptr, ws->d_cost.ptr,
                                                         ws->encode_ws, &encode_err)) {
            throw std::runtime_error(std::string("LinkageEncodeBatchCudaForcedRootWithSampleIds failed: ") + encode_err);
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[3], ctx->stream), "cudaEventRecord(ev3)");
        }

        SegmentedArgMinCost<<<B, 256, 0, ctx->stream>>>(
            ws->d_cost.ptr, ws->d_offsets.ptr, B,
            ws->d_best_cost_batch.ptr, ws->d_best_pair_batch.ptr);

        const int parent_blocks = (B + 255) / 256;
        GatherBestParentPerNode<<<parent_blocks, 256, 0, ctx->stream>>>(
            ws->d_best_pair_batch.ptr, ws->d_cand.ptr, B, ws->d_best_parent_batch.ptr);

        const int total = B * m;
        const int codes_blocks = (total + 255) / 256;
        GatherBestCodesPerNode<<<codes_blocks, 256, 0, ctx->stream>>>(
            B, m, ws->d_best_pair_batch.ptr, ws->d_B.ptr, ws->d_a.ptr,
            ws->d_best_B_batch.ptr, ws->d_best_a_batch.ptr);

        const int pack_blocks = (B + 255) / 256;
        PackManyNodesBestKernel<<<pack_blocks, 256, 0, ctx->stream>>>(
            B, m,
            ws->best_pack_a_off,
            ws->best_pack_stride,
            ws->d_best_parent_batch.ptr,
            ws->d_best_cost_batch.ptr,
            ws->d_best_B_batch.ptr,
            ws->d_best_a_batch.ptr,
            ws->d_best_pack.ptr);

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[4], ctx->stream), "cudaEventRecord(ev4)");
        }

        ThrowIf(cudaMemcpyAsync(ws->h_best_pack.u8(), ws->d_best_pack.ptr,
                                static_cast<std::size_t>(B) * static_cast<std::size_t>(ws->best_pack_stride),
                                cudaMemcpyDeviceToHost, ctx->stream),
                "Memcpy best_pack D2H");
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[5], ctx->stream), "cudaEventRecord(ev5)");
        }
        ThrowIf(cudaStreamSynchronize(ctx->stream), "cudaStreamSynchronize");

        for (int i = 0; i < B; ++i) {
            const std::uint8_t* src =
                ws->h_best_pack.u8() +
                static_cast<std::size_t>(i) * static_cast<std::size_t>(ws->best_pack_stride);
            best_parent_local_out[i] = *reinterpret_cast<const int*>(src + 0);
            best_cost_out[i] = *reinterpret_cast<const float*>(src + 4);
            const auto* src_B = reinterpret_cast<const FullCode*>(src + 8);
            FullCode* dst_B = best_B_out + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
            for (int l = 0; l < m; ++l) {
                dst_B[l] = src_B[l];
            }
            const auto* src_a = reinterpret_cast<const float*>(src + ws->best_pack_a_off);
            float* dst_a = best_a_out + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
            for (int l = 0; l < m; ++l) {
                dst_a[l] = src_a[l];
            }
        }

        if (profile) {
            ConsumeCudaLinkageEncodeLastProfile(ws->encode_ws);
            ProfileShapeAndD2HBytes(/*B=*/B, /*npairs=*/Npairs, m, profile);
        }

        if (profile) {
            float ms01 = 0, ms12 = 0, ms23 = 0, ms34 = 0, ms45 = 0, ms05 = 0;
            ThrowIf(cudaEventElapsedTime(&ms01, ws->ev[0], ws->ev[1]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms12, ws->ev[1], ws->ev[2]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms23, ws->ev[2], ws->ev[3]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms34, ws->ev[3], ws->ev[4]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms45, ws->ev[4], ws->ev[5]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms05, ws->ev[0], ws->ev[5]), "cudaEventElapsedTime");
            const auto add_ns = [](std::atomic<std::int64_t>* dst, float ms) {
                const auto ns = static_cast<std::int64_t>(static_cast<double>(ms) * 1e6);
                dst->fetch_add(ns, std::memory_order_relaxed);
            };
            g_calls.fetch_add(1, std::memory_order_relaxed);
            add_ns(&g_h2d_ns, ms01);
            add_ns(&g_residual_ns, ms12);
            add_ns(&g_encode_ns, ms23);
            add_ns(&g_reduce_ns, ms34);
            add_ns(&g_d2h_ns, ms45);
            add_ns(&g_total_ns, ms05);
        }
        return true;
    } catch (const std::exception& e) {
        if (err) {
            *err = e.what();
        }
        return false;
    }
}

bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBase(
    CudaCtx* ctx,
    const Precomp& pre_one,
    int forced_root_code,
    const float* X_block,
    int d,
    int B,
    const int* cand_parent_local_flat,
    const int* pair_node,
    const int* cand_offsets,
    int Npairs,
    int icm_iters,
    int ils_iters,
    int perturb_k,
    std::uint32_t seed,
    const std::uint64_t* node_sample_id_base,
    int* best_parent_local_out,
    float* best_cost_out,
    FullCode* best_B_out,
    float* best_a_out,
    std::string* err) {
    int chunk_max_pairs_cfg = g_chunk_max_pairs.load(std::memory_order_relaxed);
    if (chunk_max_pairs_cfg <= 0) {
        const int mem_budget_mb = g_mem_budget_mb.load(std::memory_order_relaxed);
        int rfull_cols = 0;
        if (ctx) {
            EvalWorkspace* ws = GetWorkspace(ctx);
            if (ws && ws->rfull_d == d && ws->rfull_cols > 0) {
                rfull_cols = ws->rfull_cols;
            }
        }
        const int B_guess = std::max(1, std::min(B, 64));
        chunk_max_pairs_cfg = EffectiveChunkMaxPairsFromBudget(d, pre_one.H, pre_one.m, B_guess, rfull_cols, mem_budget_mb);
    }
    if (chunk_max_pairs_cfg <= 0 || Npairs <= chunk_max_pairs_cfg) {
        return EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBaseOneShot(
            ctx, pre_one, forced_root_code, X_block, d, B, cand_parent_local_flat, pair_node, cand_offsets, Npairs,
            icm_iters, ils_iters, perturb_k, seed, node_sample_id_base,
            best_parent_local_out, best_cost_out, best_B_out, best_a_out, err);
    }
    if (B <= 0 || !cand_offsets || cand_offsets[B] != Npairs) {
        return EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBaseOneShot(
            ctx, pre_one, forced_root_code, X_block, d, B, cand_parent_local_flat, pair_node, cand_offsets, Npairs,
            icm_iters, ils_iters, perturb_k, seed, node_sample_id_base,
            best_parent_local_out, best_cost_out, best_B_out, best_a_out, err);
    }

    thread_local std::vector<int> tls_offsets;
    thread_local std::vector<int> tls_pair_node;
    thread_local std::vector<FullCode> tls_tmp_B;
    thread_local std::vector<float> tls_tmp_a;

    const int m = pre_one.m;
    tls_tmp_B.resize(static_cast<std::size_t>(std::max(0, m)));
    tls_tmp_a.resize(static_cast<std::size_t>(std::max(0, m)));

    int node0 = 0;
    while (node0 < B) {
        const int pairs_begin = cand_offsets[node0];
        int node1 = node0;
        while (node1 + 1 <= B) {
            const int pairs_end_next = cand_offsets[node1 + 1];
            if (pairs_end_next - pairs_begin > chunk_max_pairs_cfg) {
                break;
            }
            ++node1;
        }
        node1 = std::max(node1, node0 + 1);

        const int pairs_end = cand_offsets[node1];
        const int npairs_chunk = pairs_end - pairs_begin;
        const int Bchunk = node1 - node0;
        if (npairs_chunk <= 0) {
            for (int i = node0; i < node1; ++i) {
                best_parent_local_out[i] = -1;
                best_cost_out[i] = std::numeric_limits<float>::infinity();
            }
            std::fill(best_B_out + static_cast<std::size_t>(node0) * static_cast<std::size_t>(m),
                      best_B_out + static_cast<std::size_t>(node1) * static_cast<std::size_t>(m),
                      0);
            std::fill(best_a_out + static_cast<std::size_t>(node0) * static_cast<std::size_t>(m),
                      best_a_out + static_cast<std::size_t>(node1) * static_cast<std::size_t>(m),
                      0.0f);
            node0 = node1;
            continue;
        }

        if (npairs_chunk <= chunk_max_pairs_cfg) {
            tls_offsets.resize(static_cast<std::size_t>(Bchunk + 1));
            for (int i = 0; i <= Bchunk; ++i) {
                tls_offsets[static_cast<std::size_t>(i)] = cand_offsets[node0 + i] - pairs_begin;
            }
            tls_pair_node.resize(static_cast<std::size_t>(npairs_chunk));
            for (int i = 0; i < Bchunk; ++i) {
                const int s = tls_offsets[static_cast<std::size_t>(i)];
                const int t = tls_offsets[static_cast<std::size_t>(i + 1)];
                for (int p = s; p < t; ++p) {
                    tls_pair_node[static_cast<std::size_t>(p)] = i;
                }
            }

            std::string local_err;
            const bool ok = EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBaseOneShot(
                ctx, pre_one, forced_root_code,
                X_block + static_cast<std::size_t>(node0) * static_cast<std::size_t>(d), d, Bchunk,
                cand_parent_local_flat + pairs_begin,
                tls_pair_node.data(),
                tls_offsets.data(),
                npairs_chunk,
                icm_iters, ils_iters, perturb_k, seed,
                node_sample_id_base + static_cast<std::size_t>(node0),
                best_parent_local_out + node0,
                best_cost_out + node0,
                best_B_out + static_cast<std::size_t>(node0) * static_cast<std::size_t>(m),
                best_a_out + static_cast<std::size_t>(node0) * static_cast<std::size_t>(m),
                &local_err);
            if (!ok) {
                if (err) *err = local_err;
                return false;
            }
            node0 = node1;
            continue;
        }

        if (Bchunk != 1) {
            return EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBaseOneShot(
                ctx, pre_one, forced_root_code, X_block, d, B, cand_parent_local_flat, pair_node, cand_offsets, Npairs,
                icm_iters, ils_iters, perturb_k, seed, node_sample_id_base,
                best_parent_local_out, best_cost_out, best_B_out, best_a_out, err);
        }

        const int node_idx = node0;
        best_parent_local_out[node_idx] = -1;
        best_cost_out[node_idx] = std::numeric_limits<float>::infinity();
        std::fill(best_B_out + static_cast<std::size_t>(node_idx) * static_cast<std::size_t>(m),
                  best_B_out + static_cast<std::size_t>(node_idx + 1) * static_cast<std::size_t>(m),
                  0);
        std::fill(best_a_out + static_cast<std::size_t>(node_idx) * static_cast<std::size_t>(m),
                  best_a_out + static_cast<std::size_t>(node_idx + 1) * static_cast<std::size_t>(m),
                  0.0f);

        tls_pair_node.resize(static_cast<std::size_t>(chunk_max_pairs_cfg));
        std::fill(tls_pair_node.begin(), tls_pair_node.end(), 0);
        int p0 = 0;
        while (p0 < npairs_chunk) {
            const int len = std::min(chunk_max_pairs_cfg, npairs_chunk - p0);
            const int off2[2] = {0, len};
            std::string local_err;
            int best_parent_tmp = -1;
            float best_cost_tmp = std::numeric_limits<float>::infinity();
            const bool ok = EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBaseOneShot(
                ctx, pre_one, forced_root_code,
                X_block + static_cast<std::size_t>(node_idx) * static_cast<std::size_t>(d), d, 1,
                cand_parent_local_flat + pairs_begin + p0,
                tls_pair_node.data(),
                off2,
                len,
                icm_iters, ils_iters, perturb_k, seed,
                node_sample_id_base + static_cast<std::size_t>(node_idx),
                &best_parent_tmp,
                &best_cost_tmp,
                tls_tmp_B.data(),
                tls_tmp_a.data(),
                &local_err);
            if (!ok) {
                if (err) *err = local_err;
                return false;
            }
            if (best_cost_tmp < best_cost_out[node_idx]) {
                best_parent_local_out[node_idx] = best_parent_tmp;
                best_cost_out[node_idx] = best_cost_tmp;
                std::copy(tls_tmp_B.begin(), tls_tmp_B.end(),
                          best_B_out + static_cast<std::size_t>(node_idx) * static_cast<std::size_t>(m));
                std::copy(tls_tmp_a.begin(), tls_tmp_a.end(),
                          best_a_out + static_cast<std::size_t>(node_idx) * static_cast<std::size_t>(m));
            }
            p0 += len;
        }
        node0 = node1;
    }
    return true;
}

static bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBaseLargeRootOneShot(
    CudaCtx* ctx,
    const CodebookPack& C_root,
    int forced_root_code,
    const float* X_block,
    int d,
    int B,
    const int* cand_parent_local_flat,
    const int* pair_node,
    const int* cand_offsets,
    int Npairs,
    int icm_iters,
    int ils_iters,
    int perturb_k,
    std::uint32_t seed,
    const std::uint64_t* node_sample_id_base,
    int* best_parent_local_out,
    float* best_cost_out,
    FullCode* best_B_out,
    float* best_a_out,
    std::string* err) {
    try {
        if (!ctx) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBaseLargeRoot: ctx is null.");
        }
        if (d <= 0 || B <= 0) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBaseLargeRoot: invalid shape.");
        }
        const int m = static_cast<int>(C_root.books.size());
        if (m <= 1) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBaseLargeRoot: invalid m.");
        }
        if (C_root.books.empty() || C_root.books[0].rows != d) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBaseLargeRoot: d mismatch.");
        }
        if (Npairs < 0) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBaseLargeRoot: invalid Npairs.");
        }
        if (!cand_offsets || !best_parent_local_out || !best_cost_out || !best_B_out || !best_a_out) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBaseLargeRoot: null input pointer.");
        }
        const int expected_npairs = cand_offsets[B];
        if (expected_npairs != Npairs) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBaseLargeRoot: Npairs != cand_offsets[B].");
        }
        if (Npairs == 0) {
            for (int i = 0; i < B; ++i) {
                best_parent_local_out[i] = -1;
                best_cost_out[i] = std::numeric_limits<float>::infinity();
            }
            std::fill(best_B_out, best_B_out + static_cast<std::size_t>(B) * static_cast<std::size_t>(m), 0);
            std::fill(best_a_out, best_a_out + static_cast<std::size_t>(B) * static_cast<std::size_t>(m), 0.0f);
            return true;
        }
        if (!X_block || !cand_parent_local_flat || !pair_node || !node_sample_id_base) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBaseLargeRoot: null input pointer.");
        }
        const int h0 = C_root.books[0].cols;
        if (forced_root_code < 0 || forced_root_code >= h0) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBaseLargeRoot: forced_root_code out of range.");
        }

        ThrowIf(cudaSetDevice(ctx->device), "cudaSetDevice");
        EvalWorkspace* ws = GetWorkspace(ctx);
        const bool need_root_u32 = (C_root.books.front().cols > 256);
        if (!need_root_u32) {
            ws->EnsureBatch(*ctx, d, B, Npairs, m);
        } else {
            ws->EnsureBatchU32(*ctx, d, B, Npairs, m);
        }

        if (!ws->d_R_full.ptr || ws->rfull_cols <= 0 || ws->rfull_d != d) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBaseLargeRoot: device R_full not uploaded.");
        }

        int Hs = 0;
        for (int l = 1; l < m; ++l) {
            Hs += C_root.books[static_cast<std::size_t>(l)].cols;
        }

        const bool profile = g_profile_enabled.load(std::memory_order_relaxed);
        if (profile) {
            ws->EnsureEvents(*ctx);
            const std::uint64_t bytes_est =
                EstimateBytesPerCtx(d, Hs, m, /*B=*/B, /*npairs=*/Npairs, /*rfull_cols=*/ws->rfull_cols) +
                static_cast<std::uint64_t>(sizeof(std::uint64_t)) * static_cast<std::uint64_t>(B) +
                static_cast<std::uint64_t>(sizeof(std::uint64_t)) * static_cast<std::uint64_t>(Npairs) +
                static_cast<std::uint64_t>(sizeof(int)) * static_cast<std::uint64_t>(Npairs);
            AtomicMax(&g_peak_d, d);
            AtomicMax(&g_peak_m, m);
            AtomicMax(&g_peak_H, Hs);
            AtomicMax(&g_peak_B, B);
            AtomicMax(&g_peak_npairs, Npairs);
            AtomicMax(&g_peak_rfull_cols, ws->rfull_cols);
            AtomicMaxU64(&g_peak_bytes_per_ctx_est, bytes_est);
            ThrowIf(cudaEventRecord(ws->ev[0], ctx->stream), "cudaEventRecord(ev0)");
        }

        ThrowIf(cudaMemcpyAsync(ws->d_xblock.ptr, X_block,
                                sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(B),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy X_block H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_cand.ptr, cand_parent_local_flat,
                                sizeof(int) * static_cast<std::size_t>(Npairs),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy cand_flat H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_pair_node.ptr, pair_node,
                                sizeof(int) * static_cast<std::size_t>(Npairs),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy pair_node H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_offsets.ptr, cand_offsets,
                                sizeof(int) * static_cast<std::size_t>(B + 1),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy offsets H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_node_sample_id_base.ptr, node_sample_id_base,
                                sizeof(std::uint64_t) * static_cast<std::size_t>(B),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy node_sample_id_base H2D");

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[1], ctx->stream), "cudaEventRecord(ev1)");
        }

        // Build residuals + norm2.
        GatherResidualAndNorm2FromRfullPairs<<<Npairs, 256, 0, ctx->stream>>>(
            ws->d_xblock.ptr, ws->d_R_full.ptr, ws->d_cand.ptr, ws->d_pair_node.ptr,
            ws->d_Rp.ptr, ws->d_norm2.ptr, d, Npairs);

        // Build per-pair sample IDs so ILS perturbation is stable independent of batching.
        {
            const int threads = 256;
            const int blocks = (Npairs + threads - 1) / threads;
            BuildPairSampleIdsFromNodeBase<<<blocks, threads, 0, ctx->stream>>>(
                ws->d_node_sample_id_base.ptr,
                ws->d_pair_node.ptr,
                ws->d_offsets.ptr,
                /*pair_t_override=*/nullptr,
                ws->d_sample_ids.ptr,
                B,
                Npairs);
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[2], ctx->stream), "cudaEventRecord(ev2)");
        }

        std::string encode_err;
        if (!need_root_u32) {
            if (!LinkageEncodeBatchCudaForcedRootWithSampleIdsLargeRootConstRoot(*ctx, C_root, forced_root_code,
                                                                               ws->d_Rp.ptr, ws->d_norm2.ptr,
                                                                               ws->d_sample_ids.ptr,
                                                                               Npairs, icm_iters,
                                                                               ils_iters, perturb_k, seed,
                                                                               ws->d_B.ptr, ws->d_a.ptr, ws->d_cost.ptr,
                                                                               ws->encode_ws, &encode_err)) {
                throw std::runtime_error(std::string("LinkageEncodeBatchCudaForcedRootWithSampleIdsLargeRootConstRoot failed: ") + encode_err);
            }
        } else {
            if (!LinkageEncodeBatchCudaForcedRootWithSampleIdsLargeRootConstRootU32(*ctx, C_root, forced_root_code,
                                                                                  ws->d_Rp.ptr, ws->d_norm2.ptr,
                                                                                  ws->d_sample_ids.ptr,
                                                                                  Npairs, icm_iters,
                                                                                  ils_iters, perturb_k, seed,
                                                                                  ws->d_B_u32.ptr, ws->d_a.ptr, ws->d_cost.ptr,
                                                                                  ws->encode_ws, &encode_err)) {
                throw std::runtime_error(std::string("LinkageEncodeBatchCudaForcedRootWithSampleIdsLargeRootConstRootU32 failed: ") + encode_err);
            }
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[3], ctx->stream), "cudaEventRecord(ev3)");
        }

        SegmentedArgMinCost<<<B, 256, 0, ctx->stream>>>(
            ws->d_cost.ptr, ws->d_offsets.ptr, B,
            ws->d_best_cost_batch.ptr, ws->d_best_pair_batch.ptr);

        const int parent_blocks = (B + 255) / 256;
        GatherBestParentPerNode<<<parent_blocks, 256, 0, ctx->stream>>>(
            ws->d_best_pair_batch.ptr, ws->d_cand.ptr, B, ws->d_best_parent_batch.ptr);

        const int total = B * m;
        const int codes_blocks = (total + 255) / 256;
        if (!need_root_u32) {
            GatherBestCodesPerNode<<<codes_blocks, 256, 0, ctx->stream>>>(
                B, m, ws->d_best_pair_batch.ptr, ws->d_B.ptr, ws->d_a.ptr,
                ws->d_best_B_batch.ptr, ws->d_best_a_batch.ptr);
        } else {
            GatherBestCodesPerNodeU32<<<codes_blocks, 256, 0, ctx->stream>>>(
                B, m, ws->d_best_pair_batch.ptr, ws->d_B_u32.ptr, ws->d_a.ptr,
                ws->d_best_B_batch_u32.ptr, ws->d_best_a_batch.ptr);
        }

        const int pack_blocks = (B + 255) / 256;
        if (!need_root_u32) {
            PackManyNodesBestKernel<<<pack_blocks, 256, 0, ctx->stream>>>(
                B, m,
                ws->best_pack_a_off,
                ws->best_pack_stride,
                ws->d_best_parent_batch.ptr,
                ws->d_best_cost_batch.ptr,
                ws->d_best_B_batch.ptr,
                ws->d_best_a_batch.ptr,
                ws->d_best_pack.ptr);
        } else {
            PackManyNodesBestKernelU32<<<pack_blocks, 256, 0, ctx->stream>>>(
                B, m,
                ws->best_pack_a_off,
                ws->best_pack_stride,
                ws->d_best_parent_batch.ptr,
                ws->d_best_cost_batch.ptr,
                ws->d_best_B_batch_u32.ptr,
                ws->d_best_a_batch.ptr,
                ws->d_best_pack.ptr);
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[4], ctx->stream), "cudaEventRecord(ev4)");
        }

        ThrowIf(cudaMemcpyAsync(ws->h_best_pack.u8(), ws->d_best_pack.ptr,
                                static_cast<std::size_t>(B) * static_cast<std::size_t>(ws->best_pack_stride),
                                cudaMemcpyDeviceToHost, ctx->stream),
                "Memcpy best_pack D2H");
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[5], ctx->stream), "cudaEventRecord(ev5)");
        }
        ThrowIf(cudaStreamSynchronize(ctx->stream), "cudaStreamSynchronize");

        for (int i = 0; i < B; ++i) {
            const std::uint8_t* src =
                ws->h_best_pack.u8() +
                static_cast<std::size_t>(i) * static_cast<std::size_t>(ws->best_pack_stride);
            best_parent_local_out[i] = *reinterpret_cast<const int*>(src + 0);
            best_cost_out[i] = *reinterpret_cast<const float*>(src + 4);
            FullCode* dst_B = best_B_out + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
            if (!need_root_u32) {
                const auto* src_B = reinterpret_cast<const FullCode*>(src + 8);
                for (int l = 0; l < m; ++l) {
                    dst_B[l] = src_B[l];
                }
            } else {
                // For u32-root forced-root encoding, caller already knows the forced root code.
                // Keep layer0 as a dummy 0 in the FullCode output and cast layers 1..m-1.
                const auto* src_B = reinterpret_cast<const std::uint32_t*>(src + 8);
                if (m > 0) {
                    dst_B[0] = static_cast<FullCode>(0);
                }
                for (int l = 1; l < m; ++l) {
                    dst_B[l] = static_cast<FullCode>(src_B[l]);
                }
            }
            const auto* src_a = reinterpret_cast<const float*>(src + ws->best_pack_a_off);
            float* dst_a = best_a_out + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
            for (int l = 0; l < m; ++l) {
                dst_a[l] = src_a[l];
            }
        }

        if (profile) {
            ConsumeCudaLinkageEncodeLastProfile(ws->encode_ws);
            ProfileShapeAndD2HBytes(/*B=*/B, /*npairs=*/Npairs, m, profile);
        }

        if (profile) {
            float ms01 = 0, ms12 = 0, ms23 = 0, ms34 = 0, ms45 = 0, ms05 = 0;
            ThrowIf(cudaEventElapsedTime(&ms01, ws->ev[0], ws->ev[1]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms12, ws->ev[1], ws->ev[2]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms23, ws->ev[2], ws->ev[3]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms34, ws->ev[3], ws->ev[4]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms45, ws->ev[4], ws->ev[5]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms05, ws->ev[0], ws->ev[5]), "cudaEventElapsedTime");
            const auto add_ns = [](std::atomic<std::int64_t>* dst, float ms) {
                const auto ns = static_cast<std::int64_t>(static_cast<double>(ms) * 1e6);
                dst->fetch_add(ns, std::memory_order_relaxed);
            };
            g_calls.fetch_add(1, std::memory_order_relaxed);
            add_ns(&g_h2d_ns, ms01);
            add_ns(&g_residual_ns, ms12);
            add_ns(&g_encode_ns, ms23);
            add_ns(&g_reduce_ns, ms34);
            add_ns(&g_d2h_ns, ms45);
            add_ns(&g_total_ns, ms05);
        }
        return true;
    } catch (const std::exception& e) {
        if (err) {
            *err = e.what();
        }
        return false;
    }
}

bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBaseLargeRoot(
    CudaCtx* ctx,
    const CodebookPack& C_root,
    int forced_root_code,
    const float* X_block,
    int d,
    int B,
    const int* cand_parent_local_flat,
    const int* pair_node,
    const int* cand_offsets,
    int Npairs,
    int icm_iters,
    int ils_iters,
    int perturb_k,
    std::uint32_t seed,
    const std::uint64_t* node_sample_id_base,
    int* best_parent_local_out,
    float* best_cost_out,
    FullCode* best_B_out,
    float* best_a_out,
    std::string* err) {
    int chunk_max_pairs_cfg = g_chunk_max_pairs.load(std::memory_order_relaxed);
    const int m = static_cast<int>(C_root.books.size());
    int Hs = 0;
    for (int l = 1; l < m; ++l) {
        Hs += C_root.books[static_cast<std::size_t>(l)].cols;
    }
    if (chunk_max_pairs_cfg <= 0) {
        const int mem_budget_mb = g_mem_budget_mb.load(std::memory_order_relaxed);
        int rfull_cols = 0;
        if (ctx) {
            EvalWorkspace* ws = GetWorkspace(ctx);
            if (ws && ws->rfull_d == d && ws->rfull_cols > 0) {
                rfull_cols = ws->rfull_cols;
            }
        }
        const int B_guess = std::max(1, std::min(B, 64));
        chunk_max_pairs_cfg = EffectiveChunkMaxPairsFromBudget(d, Hs, m, B_guess, rfull_cols, mem_budget_mb);
    }
    if (chunk_max_pairs_cfg <= 0 || Npairs <= chunk_max_pairs_cfg) {
        return EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBaseLargeRootOneShot(
            ctx, C_root, forced_root_code, X_block, d, B, cand_parent_local_flat, pair_node, cand_offsets, Npairs,
            icm_iters, ils_iters, perturb_k, seed, node_sample_id_base,
            best_parent_local_out, best_cost_out, best_B_out, best_a_out, err);
    }
    if (B <= 0 || !cand_offsets || cand_offsets[B] != Npairs) {
        return EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBaseLargeRootOneShot(
            ctx, C_root, forced_root_code, X_block, d, B, cand_parent_local_flat, pair_node, cand_offsets, Npairs,
            icm_iters, ils_iters, perturb_k, seed, node_sample_id_base,
            best_parent_local_out, best_cost_out, best_B_out, best_a_out, err);
    }

    thread_local std::vector<int> tls_offsets;
    thread_local std::vector<int> tls_pair_node;
    thread_local std::vector<FullCode> tls_tmp_B;
    thread_local std::vector<float> tls_tmp_a;

    tls_tmp_B.resize(static_cast<std::size_t>(std::max(0, m)));
    tls_tmp_a.resize(static_cast<std::size_t>(std::max(0, m)));

    int node0 = 0;
    while (node0 < B) {
        const int pairs_begin = cand_offsets[node0];
        int node1 = node0;
        while (node1 + 1 <= B) {
            const int pairs_end_next = cand_offsets[node1 + 1];
            if (pairs_end_next - pairs_begin > chunk_max_pairs_cfg) {
                break;
            }
            ++node1;
        }
        node1 = std::max(node1, node0 + 1);

        const int pairs_end = cand_offsets[node1];
        const int npairs_chunk = pairs_end - pairs_begin;
        const int Bchunk = node1 - node0;
        if (npairs_chunk <= 0) {
            for (int i = node0; i < node1; ++i) {
                best_parent_local_out[i] = -1;
                best_cost_out[i] = std::numeric_limits<float>::infinity();
            }
            std::fill(best_B_out + static_cast<std::size_t>(node0) * static_cast<std::size_t>(m),
                      best_B_out + static_cast<std::size_t>(node1) * static_cast<std::size_t>(m),
                      0);
            std::fill(best_a_out + static_cast<std::size_t>(node0) * static_cast<std::size_t>(m),
                      best_a_out + static_cast<std::size_t>(node1) * static_cast<std::size_t>(m),
                      0.0f);
            node0 = node1;
            continue;
        }

        if (npairs_chunk <= chunk_max_pairs_cfg) {
            tls_offsets.resize(static_cast<std::size_t>(Bchunk + 1));
            for (int i = 0; i <= Bchunk; ++i) {
                tls_offsets[static_cast<std::size_t>(i)] = cand_offsets[node0 + i] - pairs_begin;
            }
            tls_pair_node.resize(static_cast<std::size_t>(npairs_chunk));
            for (int i = 0; i < Bchunk; ++i) {
                const int s = tls_offsets[static_cast<std::size_t>(i)];
                const int t = tls_offsets[static_cast<std::size_t>(i + 1)];
                for (int p = s; p < t; ++p) {
                    tls_pair_node[static_cast<std::size_t>(p)] = i;
                }
            }

            std::string local_err;
            const bool ok = EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBaseLargeRootOneShot(
                ctx, C_root, forced_root_code,
                X_block + static_cast<std::size_t>(node0) * static_cast<std::size_t>(d), d, Bchunk,
                cand_parent_local_flat + pairs_begin,
                tls_pair_node.data(),
                tls_offsets.data(),
                npairs_chunk,
                icm_iters, ils_iters, perturb_k, seed,
                node_sample_id_base + static_cast<std::size_t>(node0),
                best_parent_local_out + node0,
                best_cost_out + node0,
                best_B_out + static_cast<std::size_t>(node0) * static_cast<std::size_t>(m),
                best_a_out + static_cast<std::size_t>(node0) * static_cast<std::size_t>(m),
                &local_err);
            if (!ok) {
                if (err) *err = local_err;
                return false;
            }
            node0 = node1;
            continue;
        }

        if (Bchunk != 1) {
            return EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBaseLargeRootOneShot(
                ctx, C_root, forced_root_code, X_block, d, B, cand_parent_local_flat, pair_node, cand_offsets, Npairs,
                icm_iters, ils_iters, perturb_k, seed, node_sample_id_base,
                best_parent_local_out, best_cost_out, best_B_out, best_a_out, err);
        }

        const int node_idx = node0;
        best_parent_local_out[node_idx] = -1;
        best_cost_out[node_idx] = std::numeric_limits<float>::infinity();
        std::fill(best_B_out + static_cast<std::size_t>(node_idx) * static_cast<std::size_t>(m),
                  best_B_out + static_cast<std::size_t>(node_idx + 1) * static_cast<std::size_t>(m),
                  0);
        std::fill(best_a_out + static_cast<std::size_t>(node_idx) * static_cast<std::size_t>(m),
                  best_a_out + static_cast<std::size_t>(node_idx + 1) * static_cast<std::size_t>(m),
                  0.0f);

        tls_pair_node.resize(static_cast<std::size_t>(chunk_max_pairs_cfg));
        std::fill(tls_pair_node.begin(), tls_pair_node.end(), 0);
        int p0 = 0;
        while (p0 < npairs_chunk) {
            const int len = std::min(chunk_max_pairs_cfg, npairs_chunk - p0);
            const int off2[2] = {0, len};
            std::string local_err;
            int best_parent_tmp = -1;
            float best_cost_tmp = std::numeric_limits<float>::infinity();
            const bool ok = EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullForcedRootWithNodeSampleIdBaseLargeRootOneShot(
                ctx, C_root, forced_root_code,
                X_block + static_cast<std::size_t>(node_idx) * static_cast<std::size_t>(d), d, 1,
                cand_parent_local_flat + pairs_begin + p0,
                tls_pair_node.data(),
                off2,
                len,
                icm_iters, ils_iters, perturb_k, seed,
                node_sample_id_base + static_cast<std::size_t>(node_idx),
                &best_parent_tmp,
                &best_cost_tmp,
                tls_tmp_B.data(),
                tls_tmp_a.data(),
                &local_err);
            if (!ok) {
                if (err) *err = local_err;
                return false;
            }
            if (best_cost_tmp < best_cost_out[node_idx]) {
                best_parent_local_out[node_idx] = best_parent_tmp;
                best_cost_out[node_idx] = best_cost_tmp;
                std::copy(tls_tmp_B.begin(), tls_tmp_B.end(),
                          best_B_out + static_cast<std::size_t>(node_idx) * static_cast<std::size_t>(m));
                std::copy(tls_tmp_a.begin(), tls_tmp_a.end(),
                          best_a_out + static_cast<std::size_t>(node_idx) * static_cast<std::size_t>(m));
            }
            p0 += len;
        }
        node0 = node1;
    }
    return true;
}

static bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootOneShot(
    CudaCtx* ctx,
    const CodebookPack& C_root,
    const float* X_block,
    int d,
    int B,
    const int* cand_parent_local_flat,
    const int* pair_node,
    const int* cand_offsets,
    int Npairs,
    int icm_iters,
    int ils_iters,
    int perturb_k,
    std::uint32_t seed,
    const std::uint64_t* node_sample_id_base,
    const int* pair_t_override,
    int* best_parent_local_out,
    float* best_cost_out,
    FullCode* best_B_out,
    float* best_a_out,
    std::string* err) {
    try {
        if (!ctx) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRoot: ctx is null.");
        }
        if (d <= 0 || B <= 0) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRoot: invalid shape.");
        }
        const int m = static_cast<int>(C_root.books.size());
        if (m <= 1) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRoot: invalid m.");
        }
        if (C_root.books.empty() || C_root.books[0].rows != d) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRoot: d mismatch.");
        }
        if (Npairs < 0) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRoot: invalid Npairs.");
        }
        if (!cand_offsets || !best_parent_local_out || !best_cost_out || !best_B_out || !best_a_out) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRoot: null input pointer.");
        }
        const int expected_npairs = cand_offsets[B];
        if (expected_npairs != Npairs) {
            throw std::runtime_error("EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRoot: Npairs != cand_offsets[B].");
        }
        if (Npairs == 0) {
            for (int i = 0; i < B; ++i) {
                best_parent_local_out[i] = -1;
                best_cost_out[i] = std::numeric_limits<float>::infinity();
            }
            std::fill(best_B_out, best_B_out + static_cast<std::size_t>(B) * static_cast<std::size_t>(m), 0);
            std::fill(best_a_out, best_a_out + static_cast<std::size_t>(B) * static_cast<std::size_t>(m), 0.0f);
            return true;
        }

        EvalWorkspace* ws = GetWorkspace(ctx);
        ws->EnsureBatch(*ctx, d, B, Npairs, m);

        const bool profile = g_profile_enabled.load(std::memory_order_relaxed);
        if (profile) {
            ws->EnsureEvents(*ctx);
            ThrowIf(cudaEventRecord(ws->ev[0], ctx->stream), "cudaEventRecord(ev0)");
        }

        ThrowIf(cudaMemcpyAsync(ws->d_xblock.ptr, X_block,
                                sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(B),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy X_block H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_cand.ptr, cand_parent_local_flat,
                                sizeof(int) * static_cast<std::size_t>(Npairs),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy cand_flat H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_pair_node.ptr, pair_node,
                                sizeof(int) * static_cast<std::size_t>(Npairs),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy pair_node H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_offsets.ptr, cand_offsets,
                                sizeof(int) * static_cast<std::size_t>(B + 1),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy offsets H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_node_sample_id_base.ptr, node_sample_id_base,
                                sizeof(std::uint64_t) * static_cast<std::size_t>(B),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy node_sample_id_base H2D");
        if (pair_t_override) {
            ThrowIf(cudaMemcpyAsync(ws->d_pair_t_override.ptr, pair_t_override,
                                    sizeof(int) * static_cast<std::size_t>(Npairs),
                                    cudaMemcpyHostToDevice, ctx->stream),
                    "Memcpy pair_t_override H2D");
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[1], ctx->stream), "cudaEventRecord(ev1)");
        }

        GatherResidualAndNorm2FromRfullPairs<<<Npairs, 256, 0, ctx->stream>>>(
            ws->d_xblock.ptr, ws->d_R_full.ptr, ws->d_cand.ptr, ws->d_pair_node.ptr,
            ws->d_Rp.ptr, ws->d_norm2.ptr, d, Npairs);

        {
            const int threads = 256;
            const int blocks = (Npairs + threads - 1) / threads;
            BuildPairSampleIdsFromNodeBase<<<blocks, threads, 0, ctx->stream>>>(
                ws->d_node_sample_id_base.ptr,
                ws->d_pair_node.ptr,
                ws->d_offsets.ptr,
                /*pair_t_override=*/pair_t_override ? ws->d_pair_t_override.ptr : nullptr,
                ws->d_sample_ids.ptr,
                B,
                Npairs);
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[2], ctx->stream), "cudaEventRecord(ev2)");
        }

        std::string encode_err;
        if (!LinkageEncodeBatchCudaWithSampleIdsLargeRoot(*ctx, C_root,
                                                        ws->d_Rp.ptr, ws->d_norm2.ptr,
                                                        ws->d_sample_ids.ptr,
                                                        Npairs, icm_iters,
                                                        ils_iters, perturb_k, seed,
                                                        ws->d_B.ptr, ws->d_a.ptr, ws->d_cost.ptr,
                                                        ws->encode_ws, &encode_err)) {
            throw std::runtime_error(std::string("LinkageEncodeBatchCudaWithSampleIdsLargeRoot failed: ") + encode_err);
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[3], ctx->stream), "cudaEventRecord(ev3)");
        }

        SegmentedArgMinCost<<<B, 256, 0, ctx->stream>>>(
            ws->d_cost.ptr, ws->d_offsets.ptr, B,
            ws->d_best_cost_batch.ptr, ws->d_best_pair_batch.ptr);

        const int parent_blocks = (B + 255) / 256;
        GatherBestParentPerNode<<<parent_blocks, 256, 0, ctx->stream>>>(
            ws->d_best_pair_batch.ptr, ws->d_cand.ptr, B, ws->d_best_parent_batch.ptr);

        const int total = B * m;
        const int codes_blocks = (total + 255) / 256;
        GatherBestCodesPerNode<<<codes_blocks, 256, 0, ctx->stream>>>(
            B, m, ws->d_best_pair_batch.ptr, ws->d_B.ptr, ws->d_a.ptr,
            ws->d_best_B_batch.ptr, ws->d_best_a_batch.ptr);

        const int pack_blocks = (B + 255) / 256;
        PackManyNodesBestKernel<<<pack_blocks, 256, 0, ctx->stream>>>(
            B, m,
            ws->best_pack_a_off,
            ws->best_pack_stride,
            ws->d_best_parent_batch.ptr,
            ws->d_best_cost_batch.ptr,
            ws->d_best_B_batch.ptr,
            ws->d_best_a_batch.ptr,
            ws->d_best_pack.ptr);

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[4], ctx->stream), "cudaEventRecord(ev4)");
        }

        ThrowIf(cudaMemcpyAsync(ws->h_best_pack.u8(), ws->d_best_pack.ptr,
                                static_cast<std::size_t>(B) * static_cast<std::size_t>(ws->best_pack_stride),
                                cudaMemcpyDeviceToHost, ctx->stream),
                "Memcpy best_pack D2H");
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[5], ctx->stream), "cudaEventRecord(ev5)");
        }
        ThrowIf(cudaStreamSynchronize(ctx->stream), "cudaStreamSynchronize");

        for (int i = 0; i < B; ++i) {
            const std::uint8_t* src =
                ws->h_best_pack.u8() +
                static_cast<std::size_t>(i) * static_cast<std::size_t>(ws->best_pack_stride);
            best_parent_local_out[i] = *reinterpret_cast<const int*>(src + 0);
            best_cost_out[i] = *reinterpret_cast<const float*>(src + 4);
            const auto* src_B = reinterpret_cast<const FullCode*>(src + 8);
            FullCode* dst_B = best_B_out + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
            for (int l = 0; l < m; ++l) {
                dst_B[l] = src_B[l];
            }
            const auto* src_a = reinterpret_cast<const float*>(src + ws->best_pack_a_off);
            float* dst_a = best_a_out + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
            for (int l = 0; l < m; ++l) {
                dst_a[l] = src_a[l];
            }
        }

        if (profile) {
            ConsumeCudaLinkageEncodeLastProfile(ws->encode_ws);
            ProfileShapeAndD2HBytes(/*B=*/B, /*npairs=*/Npairs, m, profile);

            float ms01 = 0, ms12 = 0, ms23 = 0, ms34 = 0, ms45 = 0, ms05 = 0;
            ThrowIf(cudaEventElapsedTime(&ms01, ws->ev[0], ws->ev[1]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms12, ws->ev[1], ws->ev[2]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms23, ws->ev[2], ws->ev[3]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms34, ws->ev[3], ws->ev[4]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms45, ws->ev[4], ws->ev[5]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms05, ws->ev[0], ws->ev[5]), "cudaEventElapsedTime");
            const auto add_ns = [](std::atomic<std::int64_t>* dst, float ms) {
                const auto ns = static_cast<std::int64_t>(static_cast<double>(ms) * 1e6);
                dst->fetch_add(ns, std::memory_order_relaxed);
            };
            g_calls.fetch_add(1, std::memory_order_relaxed);
            add_ns(&g_h2d_ns, ms01);
            add_ns(&g_residual_ns, ms12);
            add_ns(&g_encode_ns, ms23);
            add_ns(&g_reduce_ns, ms34);
            add_ns(&g_d2h_ns, ms45);
            add_ns(&g_total_ns, ms05);
        }

        return true;
    } catch (const std::exception& e) {
        if (err) {
            *err = e.what();
        }
        return false;
	    }
	}

	static bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootOneShotU32(
	    CudaCtx* ctx,
	    const CodebookPack& C_root,
	    const float* X_block,
	    int d,
	    int B,
	    const int* cand_parent_local_flat,
	    const int* pair_node,
	    const int* cand_offsets,
	    int Npairs,
	    int icm_iters,
	    int ils_iters,
	    int perturb_k,
	    std::uint32_t seed,
	    const std::uint64_t* node_sample_id_base,
	    const int* pair_t_override,
	    int* best_parent_local_out,
	    float* best_cost_out,
	    std::uint32_t* best_B_out_u32,
	    float* best_a_out,
	    std::string* err) {
	    try {
	        if (!ctx) {
	            throw std::runtime_error(
	                "EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootU32: ctx is null.");
	        }
	        if (d <= 0 || B <= 0) {
	            throw std::runtime_error(
	                "EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootU32: invalid shape.");
	        }
	        const int m = static_cast<int>(C_root.books.size());
	        if (m <= 1) {
	            throw std::runtime_error(
	                "EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootU32: invalid m.");
	        }
	        if (C_root.books.empty() || C_root.books[0].rows != d) {
	            throw std::runtime_error(
	                "EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootU32: d mismatch.");
	        }
	        if (Npairs < 0) {
	            throw std::runtime_error(
	                "EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootU32: invalid Npairs.");
	        }
	        if (!cand_offsets || !best_parent_local_out || !best_cost_out || !best_B_out_u32 || !best_a_out) {
	            throw std::runtime_error(
	                "EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootU32: null input pointer.");
	        }
	        const int expected_npairs = cand_offsets[B];
	        if (expected_npairs != Npairs) {
	            throw std::runtime_error(
	                "EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootU32: Npairs != cand_offsets[B].");
	        }
	        if (Npairs == 0) {
	            for (int i = 0; i < B; ++i) {
	                best_parent_local_out[i] = -1;
	                best_cost_out[i] = std::numeric_limits<float>::infinity();
	            }
	            std::fill(best_B_out_u32,
	                      best_B_out_u32 + static_cast<std::size_t>(B) * static_cast<std::size_t>(m),
	                      0u);
	            std::fill(best_a_out,
	                      best_a_out + static_cast<std::size_t>(B) * static_cast<std::size_t>(m),
	                      0.0f);
	            return true;
	        }
	        if (!X_block || !cand_parent_local_flat || !pair_node || !node_sample_id_base) {
	            throw std::runtime_error(
	                "EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootU32: null input pointer.");
	        }

	        EvalWorkspace* ws = GetWorkspace(ctx);
	        ws->EnsureBatchU32(*ctx, d, B, Npairs, m);

	        const bool profile = g_profile_enabled.load(std::memory_order_relaxed);
	        if (profile) {
	            int Hs = 0;
	            for (int l = 1; l < m; ++l) {
	                Hs += C_root.books[static_cast<std::size_t>(l)].cols;
	            }
	            const std::uint64_t bytes_est =
	                EstimateBytesPerCtxWithCodeBytes(d, Hs, m, /*B=*/B, /*npairs=*/Npairs, /*rfull_cols=*/ws->rfull_cols,
	                                                 /*code_bytes=*/static_cast<int>(sizeof(std::uint32_t)));
	            AtomicMax(&g_peak_d, d);
	            AtomicMax(&g_peak_m, m);
	            AtomicMax(&g_peak_H, Hs);
	            AtomicMax(&g_peak_B, B);
	            AtomicMax(&g_peak_npairs, Npairs);
	            AtomicMax(&g_peak_rfull_cols, ws->rfull_cols);
	            AtomicMaxU64(&g_peak_bytes_per_ctx_est, bytes_est);
	            ws->EnsureEvents(*ctx);
	            ThrowIf(cudaEventRecord(ws->ev[0], ctx->stream), "cudaEventRecord(ev0)");
	        }

	        ThrowIf(cudaMemcpyAsync(ws->d_xblock.ptr, X_block,
	                                sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(B),
	                                cudaMemcpyHostToDevice, ctx->stream),
	                "Memcpy X_block H2D");
	        ThrowIf(cudaMemcpyAsync(ws->d_cand.ptr, cand_parent_local_flat,
	                                sizeof(int) * static_cast<std::size_t>(Npairs),
	                                cudaMemcpyHostToDevice, ctx->stream),
	                "Memcpy cand_flat H2D");
	        ThrowIf(cudaMemcpyAsync(ws->d_pair_node.ptr, pair_node,
	                                sizeof(int) * static_cast<std::size_t>(Npairs),
	                                cudaMemcpyHostToDevice, ctx->stream),
	                "Memcpy pair_node H2D");
	        ThrowIf(cudaMemcpyAsync(ws->d_offsets.ptr, cand_offsets,
	                                sizeof(int) * static_cast<std::size_t>(B + 1),
	                                cudaMemcpyHostToDevice, ctx->stream),
	                "Memcpy offsets H2D");
	        ThrowIf(cudaMemcpyAsync(ws->d_node_sample_id_base.ptr, node_sample_id_base,
	                                sizeof(std::uint64_t) * static_cast<std::size_t>(B),
	                                cudaMemcpyHostToDevice, ctx->stream),
	                "Memcpy node_sample_id_base H2D");
	        if (pair_t_override) {
	            ThrowIf(cudaMemcpyAsync(ws->d_pair_t_override.ptr, pair_t_override,
	                                    sizeof(int) * static_cast<std::size_t>(Npairs),
	                                    cudaMemcpyHostToDevice, ctx->stream),
	                    "Memcpy pair_t_override H2D");
	        }

	        if (profile) {
	            ThrowIf(cudaEventRecord(ws->ev[1], ctx->stream), "cudaEventRecord(ev1)");
	        }

	        GatherResidualAndNorm2FromRfullPairs<<<Npairs, 256, 0, ctx->stream>>>(
	            ws->d_xblock.ptr, ws->d_R_full.ptr, ws->d_cand.ptr, ws->d_pair_node.ptr,
	            ws->d_Rp.ptr, ws->d_norm2.ptr, d, Npairs);

	        {
	            const int threads = 256;
	            const int blocks = (Npairs + threads - 1) / threads;
	            BuildPairSampleIdsFromNodeBase<<<blocks, threads, 0, ctx->stream>>>(
	                ws->d_node_sample_id_base.ptr,
	                ws->d_pair_node.ptr,
	                ws->d_offsets.ptr,
	                /*pair_t_override=*/pair_t_override ? ws->d_pair_t_override.ptr : nullptr,
	                ws->d_sample_ids.ptr,
	                B,
	                Npairs);
	        }

	        if (profile) {
	            ThrowIf(cudaEventRecord(ws->ev[2], ctx->stream), "cudaEventRecord(ev2)");
	        }

	        std::string encode_err;
	        if (!LinkageEncodeBatchCudaWithSampleIdsLargeRootU32(*ctx, C_root,
	                                                           ws->d_Rp.ptr, ws->d_norm2.ptr,
	                                                           ws->d_sample_ids.ptr,
	                                                           Npairs, icm_iters,
	                                                           ils_iters, perturb_k, seed,
	                                                           ws->d_B_u32.ptr, ws->d_a.ptr, ws->d_cost.ptr,
	                                                           ws->encode_ws, &encode_err)) {
	            throw std::runtime_error(std::string("LinkageEncodeBatchCudaWithSampleIdsLargeRootU32 failed: ") + encode_err);
	        }

	        if (profile) {
	            ThrowIf(cudaEventRecord(ws->ev[3], ctx->stream), "cudaEventRecord(ev3)");
	        }

	        SegmentedArgMinCost<<<B, 256, 0, ctx->stream>>>(
	            ws->d_cost.ptr, ws->d_offsets.ptr, B,
	            ws->d_best_cost_batch.ptr, ws->d_best_pair_batch.ptr);

	        const int parent_blocks = (B + 255) / 256;
	        GatherBestParentPerNode<<<parent_blocks, 256, 0, ctx->stream>>>(
	            ws->d_best_pair_batch.ptr, ws->d_cand.ptr, B, ws->d_best_parent_batch.ptr);

	        const int total = B * m;
	        const int codes_blocks = (total + 255) / 256;
	        GatherBestCodesPerNodeU32<<<codes_blocks, 256, 0, ctx->stream>>>(
	            B, m, ws->d_best_pair_batch.ptr, ws->d_B_u32.ptr, ws->d_a.ptr,
	            ws->d_best_B_batch_u32.ptr, ws->d_best_a_batch.ptr);

	        const int pack_blocks = (B + 255) / 256;
	        PackManyNodesBestKernelU32<<<pack_blocks, 256, 0, ctx->stream>>>(
	            B, m,
	            ws->best_pack_a_off,
	            ws->best_pack_stride,
	            ws->d_best_parent_batch.ptr,
	            ws->d_best_cost_batch.ptr,
	            ws->d_best_B_batch_u32.ptr,
	            ws->d_best_a_batch.ptr,
	            ws->d_best_pack.ptr);

	        if (profile) {
	            ThrowIf(cudaEventRecord(ws->ev[4], ctx->stream), "cudaEventRecord(ev4)");
	        }

	        ThrowIf(cudaMemcpyAsync(ws->h_best_pack.u8(), ws->d_best_pack.ptr,
	                                static_cast<std::size_t>(B) * static_cast<std::size_t>(ws->best_pack_stride),
	                                cudaMemcpyDeviceToHost, ctx->stream),
	                "Memcpy best_pack D2H");
	        if (profile) {
	            ThrowIf(cudaEventRecord(ws->ev[5], ctx->stream), "cudaEventRecord(ev5)");
	        }
	        ThrowIf(cudaStreamSynchronize(ctx->stream), "cudaStreamSynchronize");

	        for (int i = 0; i < B; ++i) {
	            const std::uint8_t* src =
	                ws->h_best_pack.u8() +
	                static_cast<std::size_t>(i) * static_cast<std::size_t>(ws->best_pack_stride);
	            best_parent_local_out[i] = *reinterpret_cast<const int*>(src + 0);
	            best_cost_out[i] = *reinterpret_cast<const float*>(src + 4);
	            const auto* src_B = reinterpret_cast<const std::uint32_t*>(src + 8);
	            std::uint32_t* dst_B = best_B_out_u32 + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
	            for (int l = 0; l < m; ++l) {
	                dst_B[l] = src_B[l];
	            }
	            const auto* src_a = reinterpret_cast<const float*>(src + ws->best_pack_a_off);
	            float* dst_a = best_a_out + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
	            for (int l = 0; l < m; ++l) {
	                dst_a[l] = src_a[l];
	            }
	        }

	        if (profile) {
	            ConsumeCudaLinkageEncodeLastProfile(ws->encode_ws);
	            ProfileShapeAndD2HBytes(/*B=*/B, /*npairs=*/Npairs, m, /*profile=*/true);

	            float ms01 = 0, ms12 = 0, ms23 = 0, ms34 = 0, ms45 = 0, ms05 = 0;
	            ThrowIf(cudaEventElapsedTime(&ms01, ws->ev[0], ws->ev[1]), "cudaEventElapsedTime");
	            ThrowIf(cudaEventElapsedTime(&ms12, ws->ev[1], ws->ev[2]), "cudaEventElapsedTime");
	            ThrowIf(cudaEventElapsedTime(&ms23, ws->ev[2], ws->ev[3]), "cudaEventElapsedTime");
	            ThrowIf(cudaEventElapsedTime(&ms34, ws->ev[3], ws->ev[4]), "cudaEventElapsedTime");
	            ThrowIf(cudaEventElapsedTime(&ms45, ws->ev[4], ws->ev[5]), "cudaEventElapsedTime");
	            ThrowIf(cudaEventElapsedTime(&ms05, ws->ev[0], ws->ev[5]), "cudaEventElapsedTime");
	            const auto add_ns = [](std::atomic<std::int64_t>* dst, float ms) {
	                const auto ns = static_cast<std::int64_t>(static_cast<double>(ms) * 1e6);
	                dst->fetch_add(ns, std::memory_order_relaxed);
	            };
	            g_calls.fetch_add(1, std::memory_order_relaxed);
	            add_ns(&g_h2d_ns, ms01);
	            add_ns(&g_residual_ns, ms12);
	            add_ns(&g_encode_ns, ms23);
	            add_ns(&g_reduce_ns, ms34);
	            add_ns(&g_d2h_ns, ms45);
	            add_ns(&g_total_ns, ms05);
	        }

	        return true;
	    } catch (const std::exception& e) {
	        if (err) *err = e.what();
	        return false;
	    }
	}

bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootEx(
	    CudaCtx* ctx,
	    const CodebookPack& C_root,
    const float* X_block,
    int d,
    int B,
    const int* cand_parent_local_flat,
    const int* pair_node,
    const int* cand_offsets,
    int Npairs,
    int icm_iters,
    int ils_iters,
    int perturb_k,
    std::uint32_t seed,
    const std::uint64_t* node_sample_id_base,
    const int* pair_t_override,
    int* best_parent_local_out,
    float* best_cost_out,
    FullCode* best_B_out,
    float* best_a_out,
    std::string* err) {
    int chunk_max_pairs_cfg = g_chunk_max_pairs.load(std::memory_order_relaxed);
    const int m = static_cast<int>(C_root.books.size());
    int Hs = 0;
    for (int l = 1; l < m; ++l) {
        Hs += C_root.books[static_cast<std::size_t>(l)].cols;
    }
    if (chunk_max_pairs_cfg <= 0) {
        const int mem_budget_mb = g_mem_budget_mb.load(std::memory_order_relaxed);
        int rfull_cols = 0;
        if (ctx) {
            EvalWorkspace* ws = GetWorkspace(ctx);
            if (ws && ws->rfull_d == d && ws->rfull_cols > 0) {
                rfull_cols = ws->rfull_cols;
            }
        }
        const int B_guess = std::max(1, std::min(B, 64));
        chunk_max_pairs_cfg = EffectiveChunkMaxPairsFromBudget(d, Hs, m, B_guess, rfull_cols, mem_budget_mb);
    }
    if (chunk_max_pairs_cfg <= 0 || Npairs <= chunk_max_pairs_cfg) {
        return EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootOneShot(
            ctx, C_root, X_block, d, B, cand_parent_local_flat, pair_node, cand_offsets, Npairs,
            icm_iters, ils_iters, perturb_k, seed, node_sample_id_base, pair_t_override,
            best_parent_local_out, best_cost_out, best_B_out, best_a_out, err);
    }
    if (B <= 0 || !cand_offsets || cand_offsets[B] != Npairs) {
        return EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootOneShot(
            ctx, C_root, X_block, d, B, cand_parent_local_flat, pair_node, cand_offsets, Npairs,
            icm_iters, ils_iters, perturb_k, seed, node_sample_id_base, pair_t_override,
            best_parent_local_out, best_cost_out, best_B_out, best_a_out, err);
    }

    thread_local std::vector<int> tls_offsets;
    thread_local std::vector<int> tls_pair_node;
    thread_local std::vector<FullCode> tls_tmp_B;
    thread_local std::vector<float> tls_tmp_a;

    tls_tmp_B.resize(static_cast<std::size_t>(std::max(0, m)));
    tls_tmp_a.resize(static_cast<std::size_t>(std::max(0, m)));

    int node0 = 0;
    while (node0 < B) {
        int node1 = node0 + 1;
        while (node1 < B && cand_offsets[node1] == cand_offsets[node0]) {
            ++node1;
        }
        const int pairs_begin = cand_offsets[node0];
        const int pairs_end = cand_offsets[node1];
        const int npairs_chunk = pairs_end - pairs_begin;
        const int Bchunk = node1 - node0;
        if (npairs_chunk <= chunk_max_pairs_cfg) {
            tls_offsets.resize(static_cast<std::size_t>(Bchunk + 1));
            tls_offsets[0] = 0;
            for (int i = 0; i < Bchunk; ++i) {
                tls_offsets[static_cast<std::size_t>(i + 1)] = cand_offsets[node0 + i + 1] - pairs_begin;
            }
            tls_pair_node.resize(static_cast<std::size_t>(npairs_chunk));
            for (int i = 0; i < Bchunk; ++i) {
                const int s = tls_offsets[static_cast<std::size_t>(i)];
                const int t = tls_offsets[static_cast<std::size_t>(i + 1)];
                for (int p = s; p < t; ++p) {
                    tls_pair_node[static_cast<std::size_t>(p)] = i;
                }
            }
            std::string local_err;
            const bool ok = EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootOneShot(
                ctx, C_root,
                X_block + static_cast<std::size_t>(node0) * static_cast<std::size_t>(d), d, Bchunk,
                cand_parent_local_flat + pairs_begin,
                tls_pair_node.data(),
                tls_offsets.data(),
                npairs_chunk,
                icm_iters, ils_iters, perturb_k, seed,
                node_sample_id_base + static_cast<std::size_t>(node0),
                pair_t_override ? (pair_t_override + pairs_begin) : nullptr,
                best_parent_local_out + node0,
                best_cost_out + node0,
                best_B_out + static_cast<std::size_t>(node0) * static_cast<std::size_t>(m),
                best_a_out + static_cast<std::size_t>(node0) * static_cast<std::size_t>(m),
                &local_err);
            if (!ok) {
                if (err) *err = local_err;
                return false;
            }
            node0 = node1;
            continue;
        }

        if (Bchunk != 1) {
            return EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootOneShot(
                ctx, C_root, X_block, d, B, cand_parent_local_flat, pair_node, cand_offsets, Npairs,
                icm_iters, ils_iters, perturb_k, seed, node_sample_id_base, pair_t_override,
                best_parent_local_out, best_cost_out, best_B_out, best_a_out, err);
        }

        const int node_idx = node0;
        best_parent_local_out[node_idx] = -1;
        best_cost_out[node_idx] = std::numeric_limits<float>::infinity();
        std::fill(best_B_out + static_cast<std::size_t>(node_idx) * static_cast<std::size_t>(m),
                  best_B_out + static_cast<std::size_t>(node_idx + 1) * static_cast<std::size_t>(m),
                  0);
        std::fill(best_a_out + static_cast<std::size_t>(node_idx) * static_cast<std::size_t>(m),
                  best_a_out + static_cast<std::size_t>(node_idx + 1) * static_cast<std::size_t>(m),
                  0.0f);

        tls_pair_node.resize(static_cast<std::size_t>(chunk_max_pairs_cfg));
        std::fill(tls_pair_node.begin(), tls_pair_node.end(), 0);
        int p0 = 0;
        while (p0 < npairs_chunk) {
            const int len = std::min(chunk_max_pairs_cfg, npairs_chunk - p0);
            const int off2[2] = {0, len};
            std::string local_err;
            int best_parent_tmp = -1;
            float best_cost_tmp = std::numeric_limits<float>::infinity();
            const bool ok = EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootOneShot(
                ctx, C_root,
                X_block + static_cast<std::size_t>(node_idx) * static_cast<std::size_t>(d), d, 1,
                cand_parent_local_flat + pairs_begin + p0,
                tls_pair_node.data(),
                off2,
                len,
                icm_iters, ils_iters, perturb_k, seed,
                node_sample_id_base + static_cast<std::size_t>(node_idx),
                pair_t_override ? (pair_t_override + pairs_begin + p0) : nullptr,
                &best_parent_tmp,
                &best_cost_tmp,
                tls_tmp_B.data(),
                tls_tmp_a.data(),
                &local_err);
            if (!ok) {
                if (err) *err = local_err;
                return false;
            }
            if (best_cost_tmp < best_cost_out[node_idx]) {
                best_parent_local_out[node_idx] = best_parent_tmp;
                best_cost_out[node_idx] = best_cost_tmp;
                std::copy(tls_tmp_B.begin(), tls_tmp_B.end(),
                          best_B_out + static_cast<std::size_t>(node_idx) * static_cast<std::size_t>(m));
                std::copy(tls_tmp_a.begin(), tls_tmp_a.end(),
                          best_a_out + static_cast<std::size_t>(node_idx) * static_cast<std::size_t>(m));
            }
            p0 += len;
        }
        node0 = node1;
    }
    return true;
}

bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExU32(
    CudaCtx* ctx,
    const CodebookPack& C_root,
    const float* X_block,
    int d,
    int B,
    const int* cand_parent_local_flat,
    const int* pair_node,
    const int* cand_offsets,
    int Npairs,
    int icm_iters,
    int ils_iters,
    int perturb_k,
    std::uint32_t seed,
    const std::uint64_t* node_sample_id_base,
    const int* pair_t_override,
    int* best_parent_local_out,
    float* best_cost_out,
    std::uint32_t* best_B_out_u32,
    float* best_a_out,
    std::string* err) {
    int chunk_max_pairs_cfg = g_chunk_max_pairs.load(std::memory_order_relaxed);
    const int m = static_cast<int>(C_root.books.size());
    int Hs = 0;
    for (int l = 1; l < m; ++l) {
        Hs += C_root.books[static_cast<std::size_t>(l)].cols;
    }
    if (chunk_max_pairs_cfg <= 0) {
        const int mem_budget_mb = g_mem_budget_mb.load(std::memory_order_relaxed);
        int rfull_cols = 0;
        if (ctx) {
            EvalWorkspace* ws = GetWorkspace(ctx);
            if (ws && ws->rfull_d == d && ws->rfull_cols > 0) {
                rfull_cols = ws->rfull_cols;
            }
        }
        const int B_guess = std::max(1, std::min(B, 64));
        chunk_max_pairs_cfg = EffectiveChunkMaxPairsFromBudgetWithCodeBytes(d, Hs, m, B_guess, rfull_cols, mem_budget_mb,
                                                                            static_cast<int>(sizeof(std::uint32_t)));
    }
    if (chunk_max_pairs_cfg <= 0 || Npairs <= chunk_max_pairs_cfg) {
        return EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootOneShotU32(
            ctx, C_root, X_block, d, B, cand_parent_local_flat, pair_node, cand_offsets, Npairs,
            icm_iters, ils_iters, perturb_k, seed, node_sample_id_base, pair_t_override,
            best_parent_local_out, best_cost_out, best_B_out_u32, best_a_out, err);
    }
    if (B <= 0 || !cand_offsets || cand_offsets[B] != Npairs) {
        return EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootOneShotU32(
            ctx, C_root, X_block, d, B, cand_parent_local_flat, pair_node, cand_offsets, Npairs,
            icm_iters, ils_iters, perturb_k, seed, node_sample_id_base, pair_t_override,
            best_parent_local_out, best_cost_out, best_B_out_u32, best_a_out, err);
    }

    thread_local std::vector<int> tls_offsets;
    thread_local std::vector<int> tls_pair_node;
    thread_local std::vector<std::uint32_t> tls_tmp_B;
    thread_local std::vector<float> tls_tmp_a;

    tls_tmp_B.resize(static_cast<std::size_t>(std::max(0, m)));
    tls_tmp_a.resize(static_cast<std::size_t>(std::max(0, m)));

    int node0 = 0;
    while (node0 < B) {
        int node1 = node0 + 1;
        while (node1 < B && cand_offsets[node1] == cand_offsets[node0]) {
            ++node1;
        }
        const int pairs_begin = cand_offsets[node0];
        const int pairs_end = cand_offsets[node1];
        const int npairs_chunk = pairs_end - pairs_begin;
        const int Bchunk = node1 - node0;
        if (npairs_chunk <= chunk_max_pairs_cfg) {
            tls_offsets.resize(static_cast<std::size_t>(Bchunk + 1));
            tls_offsets[0] = 0;
            for (int i = 0; i < Bchunk; ++i) {
                tls_offsets[static_cast<std::size_t>(i + 1)] = cand_offsets[node0 + i + 1] - pairs_begin;
            }
            tls_pair_node.resize(static_cast<std::size_t>(npairs_chunk));
            for (int i = 0; i < Bchunk; ++i) {
                const int s = tls_offsets[static_cast<std::size_t>(i)];
                const int t = tls_offsets[static_cast<std::size_t>(i + 1)];
                for (int p = s; p < t; ++p) {
                    tls_pair_node[static_cast<std::size_t>(p)] = i;
                }
            }
            std::string local_err;
            const bool ok = EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootOneShotU32(
                ctx, C_root,
                X_block + static_cast<std::size_t>(node0) * static_cast<std::size_t>(d), d, Bchunk,
                cand_parent_local_flat + pairs_begin,
                tls_pair_node.data(),
                tls_offsets.data(),
                npairs_chunk,
                icm_iters, ils_iters, perturb_k, seed,
                node_sample_id_base + static_cast<std::size_t>(node0),
                pair_t_override ? (pair_t_override + pairs_begin) : nullptr,
                best_parent_local_out + node0,
                best_cost_out + node0,
                best_B_out_u32 + static_cast<std::size_t>(node0) * static_cast<std::size_t>(m),
                best_a_out + static_cast<std::size_t>(node0) * static_cast<std::size_t>(m),
                &local_err);
            if (!ok) {
                if (err) *err = local_err;
                return false;
            }
            node0 = node1;
            continue;
        }

        if (Bchunk != 1) {
            return EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootOneShotU32(
                ctx, C_root, X_block, d, B, cand_parent_local_flat, pair_node, cand_offsets, Npairs,
                icm_iters, ils_iters, perturb_k, seed, node_sample_id_base, pair_t_override,
                best_parent_local_out, best_cost_out, best_B_out_u32, best_a_out, err);
        }

        const int node_idx = node0;
        best_parent_local_out[node_idx] = -1;
        best_cost_out[node_idx] = std::numeric_limits<float>::infinity();
        std::fill(best_B_out_u32 + static_cast<std::size_t>(node_idx) * static_cast<std::size_t>(m),
                  best_B_out_u32 + static_cast<std::size_t>(node_idx + 1) * static_cast<std::size_t>(m),
                  0u);
        std::fill(best_a_out + static_cast<std::size_t>(node_idx) * static_cast<std::size_t>(m),
                  best_a_out + static_cast<std::size_t>(node_idx + 1) * static_cast<std::size_t>(m),
                  0.0f);

        tls_pair_node.resize(static_cast<std::size_t>(chunk_max_pairs_cfg));
        std::fill(tls_pair_node.begin(), tls_pair_node.end(), 0);
        int p0 = 0;
        while (p0 < npairs_chunk) {
            const int len = std::min(chunk_max_pairs_cfg, npairs_chunk - p0);
            const int off2[2] = {0, len};
            std::string local_err;
            int best_parent_tmp = -1;
            float best_cost_tmp = std::numeric_limits<float>::infinity();
            const bool ok = EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootOneShotU32(
                ctx, C_root,
                X_block + static_cast<std::size_t>(node_idx) * static_cast<std::size_t>(d), d, 1,
                cand_parent_local_flat + pairs_begin + p0,
                tls_pair_node.data(),
                off2,
                len,
                icm_iters, ils_iters, perturb_k, seed,
                node_sample_id_base + static_cast<std::size_t>(node_idx),
                pair_t_override ? (pair_t_override + pairs_begin + p0) : nullptr,
                &best_parent_tmp,
                &best_cost_tmp,
                tls_tmp_B.data(),
                tls_tmp_a.data(),
                &local_err);
            if (!ok) {
                if (err) *err = local_err;
                return false;
            }
            if (best_cost_tmp < best_cost_out[node_idx]) {
                best_parent_local_out[node_idx] = best_parent_tmp;
                best_cost_out[node_idx] = best_cost_tmp;
                std::copy(tls_tmp_B.begin(), tls_tmp_B.end(),
                          best_B_out_u32 + static_cast<std::size_t>(node_idx) * static_cast<std::size_t>(m));
                std::copy(tls_tmp_a.begin(), tls_tmp_a.end(),
                          best_a_out + static_cast<std::size_t>(node_idx) * static_cast<std::size_t>(m));
            }
            p0 += len;
        }
        node0 = node1;
    }
    return true;
}

bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootU32(
    CudaCtx* ctx,
    const CodebookPack& C_root,
    const float* X_block,
    int d,
    int B,
    const int* cand_parent_local_flat,
    const int* pair_node,
    const int* cand_offsets,
    int Npairs,
    int icm_iters,
    int ils_iters,
    int perturb_k,
    std::uint32_t seed,
    const std::uint64_t* node_sample_id_base,
    int* best_parent_local_out,
    float* best_cost_out,
    std::uint32_t* best_B_out_u32,
    float* best_a_out,
    std::string* err) {
    return EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExU32(
        ctx, C_root, X_block, d, B, cand_parent_local_flat, pair_node, cand_offsets, Npairs,
        icm_iters, ils_iters, perturb_k, seed, node_sample_id_base,
        /*pair_t_override=*/nullptr,
        best_parent_local_out, best_cost_out, best_B_out_u32, best_a_out, err);
}

bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExEnqueue(
    CudaCtx* ctx,
    const CodebookPack& C_root,
    const float* X_block,
    int d,
    int B,
    const int* cand_parent_local_flat,
    const int* pair_node,
    const int* cand_offsets,
    int Npairs,
    int icm_iters,
    int ils_iters,
    int perturb_k,
    std::uint32_t seed,
    const std::uint64_t* node_sample_id_base,
    const int* pair_t_override,
    std::string* err) {
    try {
        const int pinned_mb = g_async_pinned_budget_mb.load(std::memory_order_relaxed);
        if (pinned_mb <= 0) {
            throw std::runtime_error("async eval disabled (cuda_linkage_eval_async_pinned_mb<=0).");
        }
        if (!ctx) {
            throw std::runtime_error("ctx is null.");
        }
        if (d <= 0 || B <= 0) {
            throw std::runtime_error("invalid shape.");
        }
        const int m = static_cast<int>(C_root.books.size());
        if (m <= 1) {
            throw std::runtime_error("invalid m.");
        }
        if (C_root.books.empty() || C_root.books[0].rows != d) {
            throw std::runtime_error("d mismatch.");
        }
        if (Npairs <= 0) {
            throw std::runtime_error("Npairs<=0 (nothing to enqueue).");
        }
        if (!cand_parent_local_flat || !pair_node || !cand_offsets || !node_sample_id_base) {
            throw std::runtime_error("null input pointer.");
        }
        if (cand_offsets[B] != Npairs) {
            throw std::runtime_error("Npairs != cand_offsets[B].");
        }

        // Reject if internal pair-chunking would be used (async API supports one-shot only).
        int chunk_max_pairs_cfg = g_chunk_max_pairs.load(std::memory_order_relaxed);
        int Hs = 0;
        for (int l = 1; l < m; ++l) {
            Hs += C_root.books[static_cast<std::size_t>(l)].cols;
        }
        if (chunk_max_pairs_cfg <= 0) {
            const int mem_budget_mb = g_mem_budget_mb.load(std::memory_order_relaxed);
            int rfull_cols = 0;
            EvalWorkspace* ws0 = GetWorkspace(ctx);
            if (ws0 && ws0->rfull_d == d && ws0->rfull_cols > 0) {
                rfull_cols = ws0->rfull_cols;
            }
            const int B_guess = std::max(1, std::min(B, 64));
            chunk_max_pairs_cfg = EffectiveChunkMaxPairsFromBudget(d, Hs, m, B_guess, rfull_cols, mem_budget_mb);
        }
        if (chunk_max_pairs_cfg > 0 && Npairs > chunk_max_pairs_cfg) {
            throw std::runtime_error("async enqueue does not support internal pair chunking (Npairs too large).");
        }

        EvalWorkspace* ws = GetWorkspace(ctx);
        if (ws->async_in_flight) {
            throw std::runtime_error("async enqueue called while another job is in flight for this ctx.");
        }
        ws->EnsureBatch(*ctx, d, B, Npairs, m);

        const std::uint64_t budget_bytes =
            static_cast<std::uint64_t>(pinned_mb) * static_cast<std::uint64_t>(1024u * 1024u);
        const std::uint64_t need_bytes =
            static_cast<std::uint64_t>(sizeof(float)) * static_cast<std::uint64_t>(d) * static_cast<std::uint64_t>(B) +
            static_cast<std::uint64_t>(sizeof(int)) * static_cast<std::uint64_t>(Npairs) * 2u +
            static_cast<std::uint64_t>(sizeof(int)) * static_cast<std::uint64_t>(B + 1) +
            static_cast<std::uint64_t>(sizeof(std::uint64_t)) * static_cast<std::uint64_t>(B) +
            (pair_t_override ? static_cast<std::uint64_t>(sizeof(int)) * static_cast<std::uint64_t>(Npairs) : 0u);
        if (need_bytes > budget_bytes) {
            throw std::runtime_error("async enqueue pinned budget too small for this call.");
        }

        ws->h_async_X.Ensure(static_cast<std::size_t>(d) * static_cast<std::size_t>(B));
        ws->h_async_cand.Ensure(static_cast<std::size_t>(Npairs));
        ws->h_async_pair_node.Ensure(static_cast<std::size_t>(Npairs));
        ws->h_async_offsets.Ensure(static_cast<std::size_t>(B + 1));
        ws->h_async_node_base.Ensure(static_cast<std::size_t>(B));
        if (pair_t_override) {
            ws->h_async_pair_t_override.Ensure(static_cast<std::size_t>(Npairs));
        }

        std::memcpy(ws->h_async_X.ptr, X_block,
                    sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(B));
        std::memcpy(ws->h_async_cand.ptr, cand_parent_local_flat, sizeof(int) * static_cast<std::size_t>(Npairs));
        std::memcpy(ws->h_async_pair_node.ptr, pair_node, sizeof(int) * static_cast<std::size_t>(Npairs));
        std::memcpy(ws->h_async_offsets.ptr, cand_offsets, sizeof(int) * static_cast<std::size_t>(B + 1));
        std::memcpy(ws->h_async_node_base.ptr, node_sample_id_base,
                    sizeof(std::uint64_t) * static_cast<std::size_t>(B));
        if (pair_t_override) {
            std::memcpy(ws->h_async_pair_t_override.ptr, pair_t_override,
                        sizeof(int) * static_cast<std::size_t>(Npairs));
        }

        const bool profile = g_profile_enabled.load(std::memory_order_relaxed);
        ws->async_profile = profile;
        ws->async_d = d;
        ws->async_B = B;
        ws->async_Npairs = Npairs;
        ws->async_m = m;

        if (profile) {
            const std::uint64_t bytes_est = EstimateBytesPerCtx(d, Hs, m, /*B=*/B, /*npairs=*/Npairs, /*rfull_cols=*/ws->rfull_cols);
            AtomicMax(&g_peak_d, d);
            AtomicMax(&g_peak_m, m);
            AtomicMax(&g_peak_H, Hs);
            AtomicMax(&g_peak_B, B);
            AtomicMax(&g_peak_npairs, Npairs);
            AtomicMax(&g_peak_rfull_cols, ws->rfull_cols);
            AtomicMaxU64(&g_peak_bytes_per_ctx_est, bytes_est);
            ws->EnsureEvents(*ctx);
            ThrowIf(cudaEventRecord(ws->ev[0], ctx->stream), "cudaEventRecord(ev0)");
        }

        ThrowIf(cudaMemcpyAsync(ws->d_xblock.ptr, ws->h_async_X.ptr,
                                sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(B),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy X_block H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_cand.ptr, ws->h_async_cand.ptr,
                                sizeof(int) * static_cast<std::size_t>(Npairs),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy cand_flat H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_pair_node.ptr, ws->h_async_pair_node.ptr,
                                sizeof(int) * static_cast<std::size_t>(Npairs),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy pair_node H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_offsets.ptr, ws->h_async_offsets.ptr,
                                sizeof(int) * static_cast<std::size_t>(B + 1),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy offsets H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_node_sample_id_base.ptr, ws->h_async_node_base.ptr,
                                sizeof(std::uint64_t) * static_cast<std::size_t>(B),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy node_sample_id_base H2D");
        if (pair_t_override) {
            ThrowIf(cudaMemcpyAsync(ws->d_pair_t_override.ptr, ws->h_async_pair_t_override.ptr,
                                    sizeof(int) * static_cast<std::size_t>(Npairs),
                                    cudaMemcpyHostToDevice, ctx->stream),
                    "Memcpy pair_t_override H2D");
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[1], ctx->stream), "cudaEventRecord(ev1)");
        }

        GatherResidualAndNorm2FromRfullPairs<<<Npairs, 256, 0, ctx->stream>>>(
            ws->d_xblock.ptr, ws->d_R_full.ptr, ws->d_cand.ptr, ws->d_pair_node.ptr,
            ws->d_Rp.ptr, ws->d_norm2.ptr, d, Npairs);

        {
            const int threads = 256;
            const int blocks = (Npairs + threads - 1) / threads;
            BuildPairSampleIdsFromNodeBase<<<blocks, threads, 0, ctx->stream>>>(
                ws->d_node_sample_id_base.ptr,
                ws->d_pair_node.ptr,
                ws->d_offsets.ptr,
                /*pair_t_override=*/pair_t_override ? ws->d_pair_t_override.ptr : nullptr,
                ws->d_sample_ids.ptr,
                B,
                Npairs);
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[2], ctx->stream), "cudaEventRecord(ev2)");
        }

        std::string encode_err;
        if (!LinkageEncodeBatchCudaWithSampleIdsLargeRoot(*ctx, C_root,
                                                        ws->d_Rp.ptr, ws->d_norm2.ptr,
                                                        ws->d_sample_ids.ptr,
                                                        Npairs, icm_iters,
                                                        ils_iters, perturb_k, seed,
                                                        ws->d_B.ptr, ws->d_a.ptr, ws->d_cost.ptr,
                                                        ws->encode_ws, &encode_err)) {
            throw std::runtime_error(std::string("LinkageEncodeBatchCudaWithSampleIdsLargeRoot failed: ") + encode_err);
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[3], ctx->stream), "cudaEventRecord(ev3)");
        }

        SegmentedArgMinCost<<<B, 256, 0, ctx->stream>>>(
            ws->d_cost.ptr, ws->d_offsets.ptr, B,
            ws->d_best_cost_batch.ptr, ws->d_best_pair_batch.ptr);

        const int parent_blocks = (B + 255) / 256;
        GatherBestParentPerNode<<<parent_blocks, 256, 0, ctx->stream>>>(
            ws->d_best_pair_batch.ptr, ws->d_cand.ptr, B, ws->d_best_parent_batch.ptr);

        const int total = B * m;
        const int codes_blocks = (total + 255) / 256;
        GatherBestCodesPerNode<<<codes_blocks, 256, 0, ctx->stream>>>(
            B, m, ws->d_best_pair_batch.ptr, ws->d_B.ptr, ws->d_a.ptr,
            ws->d_best_B_batch.ptr, ws->d_best_a_batch.ptr);

        const int pack_blocks = (B + 255) / 256;
        PackManyNodesBestKernel<<<pack_blocks, 256, 0, ctx->stream>>>(
            B, m,
            ws->best_pack_a_off,
            ws->best_pack_stride,
            ws->d_best_parent_batch.ptr,
            ws->d_best_cost_batch.ptr,
            ws->d_best_B_batch.ptr,
            ws->d_best_a_batch.ptr,
            ws->d_best_pack.ptr);

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[4], ctx->stream), "cudaEventRecord(ev4)");
        }

        ThrowIf(cudaMemcpyAsync(ws->h_best_pack.u8(), ws->d_best_pack.ptr,
                                static_cast<std::size_t>(B) * static_cast<std::size_t>(ws->best_pack_stride),
                                cudaMemcpyDeviceToHost, ctx->stream),
                "Memcpy best_pack D2H");
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[5], ctx->stream), "cudaEventRecord(ev5)");
        } else {
            ws->EnsureAsyncDoneEvent(*ctx);
            ThrowIf(cudaEventRecord(ws->async_done, ctx->stream), "cudaEventRecord(async_done)");
        }

        ws->async_in_flight = true;
        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExFinish(
    CudaCtx* ctx,
    int* best_parent_local_out,
    float* best_cost_out,
    FullCode* best_B_out,
    float* best_a_out,
    std::string* err) {
    try {
        if (!ctx) {
            throw std::runtime_error("ctx is null.");
        }
        EvalWorkspace* ws = GetWorkspace(ctx);
        if (!ws->async_in_flight) {
            return true;
        }
        if (!best_parent_local_out || !best_cost_out || !best_B_out || !best_a_out) {
            throw std::runtime_error("null output pointer.");
        }

        if (ws->async_profile) {
            ThrowIf(cudaEventSynchronize(ws->ev[5]), "cudaEventSynchronize(ev5)");
        } else {
            ThrowIf(cudaEventSynchronize(ws->async_done), "cudaEventSynchronize(async_done)");
        }

        const int B = ws->async_B;
        const int m = ws->async_m;
        for (int i = 0; i < B; ++i) {
            const std::uint8_t* src =
                ws->h_best_pack.u8() +
                static_cast<std::size_t>(i) * static_cast<std::size_t>(ws->best_pack_stride);
            best_parent_local_out[i] = *reinterpret_cast<const int*>(src + 0);
            best_cost_out[i] = *reinterpret_cast<const float*>(src + 4);
            const auto* src_B = reinterpret_cast<const FullCode*>(src + 8);
            FullCode* dst_B = best_B_out + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
            for (int l = 0; l < m; ++l) {
                dst_B[l] = src_B[l];
            }
            const auto* src_a = reinterpret_cast<const float*>(src + ws->best_pack_a_off);
            float* dst_a = best_a_out + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
            for (int l = 0; l < m; ++l) {
                dst_a[l] = src_a[l];
            }
        }

        if (ws->async_profile) {
            ConsumeCudaLinkageEncodeLastProfile(ws->encode_ws);
            ProfileShapeAndD2HBytes(/*B=*/B, /*npairs=*/ws->async_Npairs, m, /*profile=*/true);

            float ms01 = 0, ms12 = 0, ms23 = 0, ms34 = 0, ms45 = 0, ms05 = 0;
            ThrowIf(cudaEventElapsedTime(&ms01, ws->ev[0], ws->ev[1]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms12, ws->ev[1], ws->ev[2]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms23, ws->ev[2], ws->ev[3]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms34, ws->ev[3], ws->ev[4]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms45, ws->ev[4], ws->ev[5]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms05, ws->ev[0], ws->ev[5]), "cudaEventElapsedTime");
            const auto add_ns = [](std::atomic<std::int64_t>* dst, float ms) {
                const auto ns = static_cast<std::int64_t>(static_cast<double>(ms) * 1e6);
                dst->fetch_add(ns, std::memory_order_relaxed);
            };
            g_calls.fetch_add(1, std::memory_order_relaxed);
            add_ns(&g_h2d_ns, ms01);
            add_ns(&g_residual_ns, ms12);
            add_ns(&g_encode_ns, ms23);
            add_ns(&g_reduce_ns, ms34);
            add_ns(&g_d2h_ns, ms45);
            add_ns(&g_total_ns, ms05);
        }

        ws->async_in_flight = false;
        ws->async_profile = false;
        ws->async_d = 0;
        ws->async_B = 0;
        ws->async_Npairs = 0;
        ws->async_m = 0;
        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExEnqueueU32(
    CudaCtx* ctx,
    const CodebookPack& C_root,
    const float* X_block,
    int d,
    int B,
    const int* cand_parent_local_flat,
    const int* pair_node,
    const int* cand_offsets,
    int Npairs,
    int icm_iters,
    int ils_iters,
    int perturb_k,
    std::uint32_t seed,
    const std::uint64_t* node_sample_id_base,
    const int* pair_t_override,
    std::string* err) {
    try {
        const int pinned_mb = g_async_pinned_budget_mb.load(std::memory_order_relaxed);
        if (pinned_mb <= 0) {
            throw std::runtime_error("async eval disabled (cuda_linkage_eval_async_pinned_mb<=0).");
        }
        if (!ctx) {
            throw std::runtime_error("ctx is null.");
        }
        if (d <= 0 || B <= 0) {
            throw std::runtime_error("invalid shape.");
        }
        const int m = static_cast<int>(C_root.books.size());
        if (m <= 1) {
            throw std::runtime_error("invalid m.");
        }
        if (C_root.books.empty() || C_root.books[0].rows != d) {
            throw std::runtime_error("d mismatch.");
        }
        if (Npairs <= 0) {
            throw std::runtime_error("Npairs<=0 (nothing to enqueue).");
        }
        if (!cand_parent_local_flat || !pair_node || !cand_offsets || !node_sample_id_base) {
            throw std::runtime_error("null input pointer.");
        }
        if (cand_offsets[B] != Npairs) {
            throw std::runtime_error("Npairs != cand_offsets[B].");
        }

        // Reject if internal pair-chunking would be used (async API supports one-shot only).
        int chunk_max_pairs_cfg = g_chunk_max_pairs.load(std::memory_order_relaxed);
        int Hs = 0;
        for (int l = 1; l < m; ++l) {
            Hs += C_root.books[static_cast<std::size_t>(l)].cols;
        }
        if (chunk_max_pairs_cfg <= 0) {
            const int mem_budget_mb = g_mem_budget_mb.load(std::memory_order_relaxed);
            int rfull_cols = 0;
            EvalWorkspace* ws0 = GetWorkspace(ctx);
            if (ws0 && ws0->rfull_d == d && ws0->rfull_cols > 0) {
                rfull_cols = ws0->rfull_cols;
            }
            const int B_guess = std::max(1, std::min(B, 64));
            chunk_max_pairs_cfg = EffectiveChunkMaxPairsFromBudgetWithCodeBytes(d, Hs, m, B_guess, rfull_cols, mem_budget_mb,
                                                                                static_cast<int>(sizeof(std::uint32_t)));
        }
        if (chunk_max_pairs_cfg > 0 && Npairs > chunk_max_pairs_cfg) {
            throw std::runtime_error("async enqueue does not support internal pair chunking (Npairs too large).");
        }

        EvalWorkspace* ws = GetWorkspace(ctx);
        if (ws->async_in_flight) {
            throw std::runtime_error("async enqueue called while another job is in flight for this ctx.");
        }
        ws->EnsureBatchU32(*ctx, d, B, Npairs, m);

        const std::uint64_t budget_bytes =
            static_cast<std::uint64_t>(pinned_mb) * static_cast<std::uint64_t>(1024u * 1024u);
        const std::uint64_t need_bytes =
            static_cast<std::uint64_t>(sizeof(float)) * static_cast<std::uint64_t>(d) * static_cast<std::uint64_t>(B) +
            static_cast<std::uint64_t>(sizeof(int)) * static_cast<std::uint64_t>(Npairs) * 2u +
            static_cast<std::uint64_t>(sizeof(int)) * static_cast<std::uint64_t>(B + 1) +
            static_cast<std::uint64_t>(sizeof(std::uint64_t)) * static_cast<std::uint64_t>(B) +
            (pair_t_override ? static_cast<std::uint64_t>(sizeof(int)) * static_cast<std::uint64_t>(Npairs) : 0u);
        if (need_bytes > budget_bytes) {
            throw std::runtime_error("async enqueue pinned budget too small for this call.");
        }

        ws->h_async_X.Ensure(static_cast<std::size_t>(d) * static_cast<std::size_t>(B));
        ws->h_async_cand.Ensure(static_cast<std::size_t>(Npairs));
        ws->h_async_pair_node.Ensure(static_cast<std::size_t>(Npairs));
        ws->h_async_offsets.Ensure(static_cast<std::size_t>(B + 1));
        ws->h_async_node_base.Ensure(static_cast<std::size_t>(B));
        if (pair_t_override) {
            ws->h_async_pair_t_override.Ensure(static_cast<std::size_t>(Npairs));
        }

        std::memcpy(ws->h_async_X.ptr, X_block,
                    sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(B));
        std::memcpy(ws->h_async_cand.ptr, cand_parent_local_flat, sizeof(int) * static_cast<std::size_t>(Npairs));
        std::memcpy(ws->h_async_pair_node.ptr, pair_node, sizeof(int) * static_cast<std::size_t>(Npairs));
        std::memcpy(ws->h_async_offsets.ptr, cand_offsets, sizeof(int) * static_cast<std::size_t>(B + 1));
        std::memcpy(ws->h_async_node_base.ptr, node_sample_id_base,
                    sizeof(std::uint64_t) * static_cast<std::size_t>(B));
        if (pair_t_override) {
            std::memcpy(ws->h_async_pair_t_override.ptr, pair_t_override,
                        sizeof(int) * static_cast<std::size_t>(Npairs));
        }

        const bool profile = g_profile_enabled.load(std::memory_order_relaxed);
        ws->async_profile = profile;
        ws->async_d = d;
        ws->async_B = B;
        ws->async_Npairs = Npairs;
        ws->async_m = m;

        if (profile) {
            const std::uint64_t bytes_est =
                EstimateBytesPerCtxWithCodeBytes(d, Hs, m, /*B=*/B, /*npairs=*/Npairs, /*rfull_cols=*/ws->rfull_cols,
                                                 static_cast<int>(sizeof(std::uint32_t)));
            AtomicMax(&g_peak_d, d);
            AtomicMax(&g_peak_m, m);
            AtomicMax(&g_peak_H, Hs);
            AtomicMax(&g_peak_B, B);
            AtomicMax(&g_peak_npairs, Npairs);
            AtomicMax(&g_peak_rfull_cols, ws->rfull_cols);
            AtomicMaxU64(&g_peak_bytes_per_ctx_est, bytes_est);
            ws->EnsureEvents(*ctx);
            ThrowIf(cudaEventRecord(ws->ev[0], ctx->stream), "cudaEventRecord(ev0)");
        }

        ThrowIf(cudaMemcpyAsync(ws->d_xblock.ptr, ws->h_async_X.ptr,
                                sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(B),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy X_block H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_cand.ptr, ws->h_async_cand.ptr,
                                sizeof(int) * static_cast<std::size_t>(Npairs),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy cand_flat H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_pair_node.ptr, ws->h_async_pair_node.ptr,
                                sizeof(int) * static_cast<std::size_t>(Npairs),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy pair_node H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_offsets.ptr, ws->h_async_offsets.ptr,
                                sizeof(int) * static_cast<std::size_t>(B + 1),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy offsets H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_node_sample_id_base.ptr, ws->h_async_node_base.ptr,
                                sizeof(std::uint64_t) * static_cast<std::size_t>(B),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy node_sample_id_base H2D");
        if (pair_t_override) {
            ThrowIf(cudaMemcpyAsync(ws->d_pair_t_override.ptr, ws->h_async_pair_t_override.ptr,
                                    sizeof(int) * static_cast<std::size_t>(Npairs),
                                    cudaMemcpyHostToDevice, ctx->stream),
                    "Memcpy pair_t_override H2D");
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[1], ctx->stream), "cudaEventRecord(ev1)");
        }

        GatherResidualAndNorm2FromRfullPairs<<<Npairs, 256, 0, ctx->stream>>>(
            ws->d_xblock.ptr, ws->d_R_full.ptr, ws->d_cand.ptr, ws->d_pair_node.ptr,
            ws->d_Rp.ptr, ws->d_norm2.ptr, d, Npairs);

        {
            const int threads = 256;
            const int blocks = (Npairs + threads - 1) / threads;
            BuildPairSampleIdsFromNodeBase<<<blocks, threads, 0, ctx->stream>>>(
                ws->d_node_sample_id_base.ptr,
                ws->d_pair_node.ptr,
                ws->d_offsets.ptr,
                /*pair_t_override=*/pair_t_override ? ws->d_pair_t_override.ptr : nullptr,
                ws->d_sample_ids.ptr,
                B,
                Npairs);
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[2], ctx->stream), "cudaEventRecord(ev2)");
        }

        std::string encode_err;
        if (!LinkageEncodeBatchCudaWithSampleIdsLargeRootU32(*ctx, C_root,
                                                           ws->d_Rp.ptr, ws->d_norm2.ptr,
                                                           ws->d_sample_ids.ptr,
                                                           Npairs, icm_iters,
                                                           ils_iters, perturb_k, seed,
                                                           ws->d_B_u32.ptr, ws->d_a.ptr, ws->d_cost.ptr,
                                                           ws->encode_ws, &encode_err)) {
            throw std::runtime_error(std::string("LinkageEncodeBatchCudaWithSampleIdsLargeRootU32 failed: ") + encode_err);
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[3], ctx->stream), "cudaEventRecord(ev3)");
        }

        SegmentedArgMinCost<<<B, 256, 0, ctx->stream>>>(
            ws->d_cost.ptr, ws->d_offsets.ptr, B,
            ws->d_best_cost_batch.ptr, ws->d_best_pair_batch.ptr);

        const int parent_blocks = (B + 255) / 256;
        GatherBestParentPerNode<<<parent_blocks, 256, 0, ctx->stream>>>(
            ws->d_best_pair_batch.ptr, ws->d_cand.ptr, B, ws->d_best_parent_batch.ptr);

        const int total = B * m;
        const int codes_blocks = (total + 255) / 256;
        GatherBestCodesPerNodeU32<<<codes_blocks, 256, 0, ctx->stream>>>(
            B, m, ws->d_best_pair_batch.ptr, ws->d_B_u32.ptr, ws->d_a.ptr,
            ws->d_best_B_batch_u32.ptr, ws->d_best_a_batch.ptr);

        const int pack_blocks = (B + 255) / 256;
        PackManyNodesBestKernelU32<<<pack_blocks, 256, 0, ctx->stream>>>(
            B, m,
            ws->best_pack_a_off,
            ws->best_pack_stride,
            ws->d_best_parent_batch.ptr,
            ws->d_best_cost_batch.ptr,
            ws->d_best_B_batch_u32.ptr,
            ws->d_best_a_batch.ptr,
            ws->d_best_pack.ptr);

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[4], ctx->stream), "cudaEventRecord(ev4)");
        }

        ThrowIf(cudaMemcpyAsync(ws->h_best_pack.u8(), ws->d_best_pack.ptr,
                                static_cast<std::size_t>(B) * static_cast<std::size_t>(ws->best_pack_stride),
                                cudaMemcpyDeviceToHost, ctx->stream),
                "Memcpy best_pack D2H");
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[5], ctx->stream), "cudaEventRecord(ev5)");
        } else {
            ws->EnsureAsyncDoneEvent(*ctx);
            ThrowIf(cudaEventRecord(ws->async_done, ctx->stream), "cudaEventRecord(async_done)");
        }

        ws->async_in_flight = true;
        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExFinishU32(
    CudaCtx* ctx,
    int* best_parent_local_out,
    float* best_cost_out,
    std::uint32_t* best_B_out_u32,
    float* best_a_out,
    std::string* err) {
    try {
        if (!ctx) {
            throw std::runtime_error("ctx is null.");
        }
        EvalWorkspace* ws = GetWorkspace(ctx);
        if (!ws->async_in_flight) {
            return true;
        }
        if (!best_parent_local_out || !best_cost_out || !best_B_out_u32 || !best_a_out) {
            throw std::runtime_error("null output pointer.");
        }

        if (ws->async_profile) {
            ThrowIf(cudaEventSynchronize(ws->ev[5]), "cudaEventSynchronize(ev5)");
        } else {
            ThrowIf(cudaEventSynchronize(ws->async_done), "cudaEventSynchronize(async_done)");
        }

        const int B = ws->async_B;
        const int m = ws->async_m;
        for (int i = 0; i < B; ++i) {
            const std::uint8_t* src =
                ws->h_best_pack.u8() +
                static_cast<std::size_t>(i) * static_cast<std::size_t>(ws->best_pack_stride);
            best_parent_local_out[i] = *reinterpret_cast<const int*>(src + 0);
            best_cost_out[i] = *reinterpret_cast<const float*>(src + 4);
            const auto* src_B = reinterpret_cast<const std::uint32_t*>(src + 8);
            std::uint32_t* dst_B = best_B_out_u32 + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
            for (int l = 0; l < m; ++l) {
                dst_B[l] = src_B[l];
            }
            const auto* src_a = reinterpret_cast<const float*>(src + ws->best_pack_a_off);
            float* dst_a = best_a_out + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
            for (int l = 0; l < m; ++l) {
                dst_a[l] = src_a[l];
            }
        }

        if (ws->async_profile) {
            ConsumeCudaLinkageEncodeLastProfile(ws->encode_ws);
            ProfileShapeAndD2HBytes(/*B=*/B, /*npairs=*/ws->async_Npairs, m, /*profile=*/true);

            float ms01 = 0, ms12 = 0, ms23 = 0, ms34 = 0, ms45 = 0, ms05 = 0;
            ThrowIf(cudaEventElapsedTime(&ms01, ws->ev[0], ws->ev[1]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms12, ws->ev[1], ws->ev[2]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms23, ws->ev[2], ws->ev[3]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms34, ws->ev[3], ws->ev[4]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms45, ws->ev[4], ws->ev[5]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms05, ws->ev[0], ws->ev[5]), "cudaEventElapsedTime");
            const auto add_ns = [](std::atomic<std::int64_t>* dst, float ms) {
                const auto ns = static_cast<std::int64_t>(static_cast<double>(ms) * 1e6);
                dst->fetch_add(ns, std::memory_order_relaxed);
            };
            g_calls.fetch_add(1, std::memory_order_relaxed);
            add_ns(&g_h2d_ns, ms01);
            add_ns(&g_residual_ns, ms12);
            add_ns(&g_encode_ns, ms23);
            add_ns(&g_reduce_ns, ms34);
            add_ns(&g_d2h_ns, ms45);
            add_ns(&g_total_ns, ms05);
        }

        ws->async_in_flight = false;
        ws->async_profile = false;
        ws->async_d = 0;
        ws->async_B = 0;
        ws->async_Npairs = 0;
        ws->async_m = 0;
        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseExEnqueue(
    CudaCtx* ctx,
    const Precomp& pre_one,
    const float* X_block,
    int d,
    int B,
    const int* cand_parent_local_flat,
    const int* pair_node,
    const int* cand_offsets,
    int Npairs,
    int icm_iters,
    int ils_iters,
    int perturb_k,
    std::uint32_t seed,
    const std::uint64_t* node_sample_id_base,
    const int* pair_t_override,
    std::string* err) {
    try {
        const int pinned_mb = g_async_pinned_budget_mb.load(std::memory_order_relaxed);
        if (pinned_mb <= 0) {
            throw std::runtime_error("async eval disabled (cuda_linkage_eval_async_pinned_mb<=0).");
        }
        if (!ctx) {
            throw std::runtime_error("ctx is null.");
        }
        if (d <= 0 || B <= 0) {
            throw std::runtime_error("invalid shape.");
        }
        if (d != pre_one.d) {
            throw std::runtime_error("d mismatch.");
        }
        const int m = pre_one.m;
        if (m <= 0) {
            throw std::runtime_error("invalid m.");
        }
        if (Npairs <= 0) {
            throw std::runtime_error("Npairs<=0 (nothing to enqueue).");
        }
        if (!cand_parent_local_flat || !pair_node || !cand_offsets || !node_sample_id_base) {
            throw std::runtime_error("null input pointer.");
        }
        if (cand_offsets[B] != Npairs) {
            throw std::runtime_error("Npairs != cand_offsets[B].");
        }

        // Reject if internal pair-chunking would be used (async API supports one-shot only).
        int chunk_max_pairs_cfg = g_chunk_max_pairs.load(std::memory_order_relaxed);
        if (chunk_max_pairs_cfg <= 0) {
            const int mem_budget_mb = g_mem_budget_mb.load(std::memory_order_relaxed);
            int rfull_cols = 0;
            EvalWorkspace* ws0 = GetWorkspace(ctx);
            if (ws0 && ws0->rfull_d == d && ws0->rfull_cols > 0) {
                rfull_cols = ws0->rfull_cols;
            }
            const int B_guess = std::max(1, std::min(B, 64));
            chunk_max_pairs_cfg =
                EffectiveChunkMaxPairsFromBudget(d, pre_one.H, m, B_guess, rfull_cols, mem_budget_mb);
        }
        if (chunk_max_pairs_cfg > 0 && Npairs > chunk_max_pairs_cfg) {
            throw std::runtime_error("async enqueue does not support internal pair chunking (Npairs too large).");
        }

        EvalWorkspace* ws = GetWorkspace(ctx);
        if (ws->async_in_flight) {
            throw std::runtime_error("async enqueue called while another job is in flight for this ctx.");
        }
        ws->EnsureBatch(*ctx, d, B, Npairs, m);

        const std::uint64_t budget_bytes =
            static_cast<std::uint64_t>(pinned_mb) * static_cast<std::uint64_t>(1024u * 1024u);
        const std::uint64_t need_bytes =
            static_cast<std::uint64_t>(sizeof(float)) * static_cast<std::uint64_t>(d) * static_cast<std::uint64_t>(B) +
            static_cast<std::uint64_t>(sizeof(int)) * static_cast<std::uint64_t>(Npairs) * 2u +
            static_cast<std::uint64_t>(sizeof(int)) * static_cast<std::uint64_t>(B + 1) +
            static_cast<std::uint64_t>(sizeof(std::uint64_t)) * static_cast<std::uint64_t>(B) +
            (pair_t_override ? static_cast<std::uint64_t>(sizeof(int)) * static_cast<std::uint64_t>(Npairs) : 0u);
        if (need_bytes > budget_bytes) {
            throw std::runtime_error("async enqueue pinned budget too small for this call.");
        }

        ws->h_async_X.Ensure(static_cast<std::size_t>(d) * static_cast<std::size_t>(B));
        ws->h_async_cand.Ensure(static_cast<std::size_t>(Npairs));
        ws->h_async_pair_node.Ensure(static_cast<std::size_t>(Npairs));
        ws->h_async_offsets.Ensure(static_cast<std::size_t>(B + 1));
        ws->h_async_node_base.Ensure(static_cast<std::size_t>(B));
        if (pair_t_override) {
            ws->h_async_pair_t_override.Ensure(static_cast<std::size_t>(Npairs));
        }

        std::memcpy(ws->h_async_X.ptr, X_block,
                    sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(B));
        std::memcpy(ws->h_async_cand.ptr, cand_parent_local_flat, sizeof(int) * static_cast<std::size_t>(Npairs));
        std::memcpy(ws->h_async_pair_node.ptr, pair_node, sizeof(int) * static_cast<std::size_t>(Npairs));
        std::memcpy(ws->h_async_offsets.ptr, cand_offsets, sizeof(int) * static_cast<std::size_t>(B + 1));
        std::memcpy(ws->h_async_node_base.ptr, node_sample_id_base,
                    sizeof(std::uint64_t) * static_cast<std::size_t>(B));
        if (pair_t_override) {
            std::memcpy(ws->h_async_pair_t_override.ptr, pair_t_override,
                        sizeof(int) * static_cast<std::size_t>(Npairs));
        }

        const bool profile = g_profile_enabled.load(std::memory_order_relaxed);
        ws->async_profile = profile;
        ws->async_d = d;
        ws->async_B = B;
        ws->async_Npairs = Npairs;
        ws->async_m = m;

        if (profile) {
            const std::uint64_t bytes_est =
                EstimateBytesPerCtx(d, pre_one.H, m, /*B=*/B, /*npairs=*/Npairs, /*rfull_cols=*/ws->rfull_cols);
            AtomicMax(&g_peak_d, d);
            AtomicMax(&g_peak_m, m);
            AtomicMax(&g_peak_H, pre_one.H);
            AtomicMax(&g_peak_B, B);
            AtomicMax(&g_peak_npairs, Npairs);
            AtomicMax(&g_peak_rfull_cols, ws->rfull_cols);
            AtomicMaxU64(&g_peak_bytes_per_ctx_est, bytes_est);
            ws->EnsureEvents(*ctx);
            ThrowIf(cudaEventRecord(ws->ev[0], ctx->stream), "cudaEventRecord(ev0)");
        }

        ThrowIf(cudaMemcpyAsync(ws->d_xblock.ptr, ws->h_async_X.ptr,
                                sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(B),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy X_block H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_cand.ptr, ws->h_async_cand.ptr,
                                sizeof(int) * static_cast<std::size_t>(Npairs),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy cand_flat H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_pair_node.ptr, ws->h_async_pair_node.ptr,
                                sizeof(int) * static_cast<std::size_t>(Npairs),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy pair_node H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_offsets.ptr, ws->h_async_offsets.ptr,
                                sizeof(int) * static_cast<std::size_t>(B + 1),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy offsets H2D");
        ThrowIf(cudaMemcpyAsync(ws->d_node_sample_id_base.ptr, ws->h_async_node_base.ptr,
                                sizeof(std::uint64_t) * static_cast<std::size_t>(B),
                                cudaMemcpyHostToDevice, ctx->stream),
                "Memcpy node_sample_id_base H2D");
        if (pair_t_override) {
            ThrowIf(cudaMemcpyAsync(ws->d_pair_t_override.ptr, ws->h_async_pair_t_override.ptr,
                                    sizeof(int) * static_cast<std::size_t>(Npairs),
                                    cudaMemcpyHostToDevice, ctx->stream),
                    "Memcpy pair_t_override H2D");
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[1], ctx->stream), "cudaEventRecord(ev1)");
        }

        GatherResidualAndNorm2FromRfullPairs<<<Npairs, 256, 0, ctx->stream>>>(
            ws->d_xblock.ptr, ws->d_R_full.ptr, ws->d_cand.ptr, ws->d_pair_node.ptr,
            ws->d_Rp.ptr, ws->d_norm2.ptr, d, Npairs);

        {
            const int threads = 256;
            const int blocks = (Npairs + threads - 1) / threads;
            BuildPairSampleIdsFromNodeBase<<<blocks, threads, 0, ctx->stream>>>(
                ws->d_node_sample_id_base.ptr,
                ws->d_pair_node.ptr,
                ws->d_offsets.ptr,
                /*pair_t_override=*/pair_t_override ? ws->d_pair_t_override.ptr : nullptr,
                ws->d_sample_ids.ptr,
                B,
                Npairs);
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[2], ctx->stream), "cudaEventRecord(ev2)");
        }

        std::string encode_err;
        if (!LinkageEncodeBatchCudaWithSampleIds(*ctx, pre_one,
                                               ws->d_Rp.ptr, ws->d_norm2.ptr,
                                               ws->d_sample_ids.ptr,
                                               Npairs, icm_iters,
                                               ils_iters, perturb_k, seed,
                                               ws->d_B.ptr, ws->d_a.ptr, ws->d_cost.ptr,
                                               ws->encode_ws, &encode_err)) {
            throw std::runtime_error(std::string("LinkageEncodeBatchCudaWithSampleIds failed: ") + encode_err);
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[3], ctx->stream), "cudaEventRecord(ev3)");
        }

        SegmentedArgMinCost<<<B, 256, 0, ctx->stream>>>(
            ws->d_cost.ptr, ws->d_offsets.ptr, B,
            ws->d_best_cost_batch.ptr, ws->d_best_pair_batch.ptr);

        const int parent_blocks = (B + 255) / 256;
        GatherBestParentPerNode<<<parent_blocks, 256, 0, ctx->stream>>>(
            ws->d_best_pair_batch.ptr, ws->d_cand.ptr, B, ws->d_best_parent_batch.ptr);

        const int total = B * m;
        const int codes_blocks = (total + 255) / 256;
        GatherBestCodesPerNode<<<codes_blocks, 256, 0, ctx->stream>>>(
            B, m, ws->d_best_pair_batch.ptr, ws->d_B.ptr, ws->d_a.ptr,
            ws->d_best_B_batch.ptr, ws->d_best_a_batch.ptr);

        const int pack_blocks = (B + 255) / 256;
        PackManyNodesBestKernel<<<pack_blocks, 256, 0, ctx->stream>>>(
            B, m,
            ws->best_pack_a_off,
            ws->best_pack_stride,
            ws->d_best_parent_batch.ptr,
            ws->d_best_cost_batch.ptr,
            ws->d_best_B_batch.ptr,
            ws->d_best_a_batch.ptr,
            ws->d_best_pack.ptr);

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[4], ctx->stream), "cudaEventRecord(ev4)");
        }

        ThrowIf(cudaMemcpyAsync(ws->h_best_pack.u8(), ws->d_best_pack.ptr,
                                static_cast<std::size_t>(B) * static_cast<std::size_t>(ws->best_pack_stride),
                                cudaMemcpyDeviceToHost, ctx->stream),
                "Memcpy best_pack D2H");
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[5], ctx->stream), "cudaEventRecord(ev5)");
        } else {
            ws->EnsureAsyncDoneEvent(*ctx);
            ThrowIf(cudaEventRecord(ws->async_done, ctx->stream), "cudaEventRecord(async_done)");
        }

        ws->async_in_flight = true;
        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseExFinish(
    CudaCtx* ctx,
    int* best_parent_local_out,
    float* best_cost_out,
    FullCode* best_B_out,
    float* best_a_out,
    std::string* err) {
    try {
        if (!ctx) {
            throw std::runtime_error("ctx is null.");
        }
        EvalWorkspace* ws = GetWorkspace(ctx);
        if (!ws->async_in_flight) {
            return true;
        }
        if (!best_parent_local_out || !best_cost_out || !best_B_out || !best_a_out) {
            throw std::runtime_error("null output pointer.");
        }

        if (ws->async_profile) {
            ThrowIf(cudaEventSynchronize(ws->ev[5]), "cudaEventSynchronize(ev5)");
        } else {
            ThrowIf(cudaEventSynchronize(ws->async_done), "cudaEventSynchronize(async_done)");
        }

        const int B = ws->async_B;
        const int m = ws->async_m;
        for (int i = 0; i < B; ++i) {
            const std::uint8_t* src =
                ws->h_best_pack.u8() +
                static_cast<std::size_t>(i) * static_cast<std::size_t>(ws->best_pack_stride);
            best_parent_local_out[i] = *reinterpret_cast<const int*>(src + 0);
            best_cost_out[i] = *reinterpret_cast<const float*>(src + 4);
            const auto* src_B = reinterpret_cast<const FullCode*>(src + 8);
            FullCode* dst_B = best_B_out + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
            for (int l = 0; l < m; ++l) {
                dst_B[l] = src_B[l];
            }
            const auto* src_a = reinterpret_cast<const float*>(src + ws->best_pack_a_off);
            float* dst_a = best_a_out + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
            for (int l = 0; l < m; ++l) {
                dst_a[l] = src_a[l];
            }
        }

        if (ws->async_profile) {
            ConsumeCudaLinkageEncodeLastProfile(ws->encode_ws);
            ProfileShapeAndD2HBytes(/*B=*/B, /*npairs=*/ws->async_Npairs, m, /*profile=*/true);

            float ms01 = 0, ms12 = 0, ms23 = 0, ms34 = 0, ms45 = 0, ms05 = 0;
            ThrowIf(cudaEventElapsedTime(&ms01, ws->ev[0], ws->ev[1]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms12, ws->ev[1], ws->ev[2]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms23, ws->ev[2], ws->ev[3]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms34, ws->ev[3], ws->ev[4]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms45, ws->ev[4], ws->ev[5]), "cudaEventElapsedTime");
            ThrowIf(cudaEventElapsedTime(&ms05, ws->ev[0], ws->ev[5]), "cudaEventElapsedTime");
            const auto add_ns = [](std::atomic<std::int64_t>* dst, float ms) {
                const auto ns = static_cast<std::int64_t>(static_cast<double>(ms) * 1e6);
                dst->fetch_add(ns, std::memory_order_relaxed);
            };
            g_calls.fetch_add(1, std::memory_order_relaxed);
            add_ns(&g_h2d_ns, ms01);
            add_ns(&g_residual_ns, ms12);
            add_ns(&g_encode_ns, ms23);
            add_ns(&g_reduce_ns, ms34);
            add_ns(&g_d2h_ns, ms45);
            add_ns(&g_total_ns, ms05);
        }

        ws->async_in_flight = false;
        ws->async_profile = false;
        ws->async_d = 0;
        ws->async_B = 0;
        ws->async_Npairs = 0;
        ws->async_m = 0;
        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

bool EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRoot(
    CudaCtx* ctx,
    const CodebookPack& C_root,
    const float* X_block,
    int d,
    int B,
    const int* cand_parent_local_flat,
    const int* pair_node,
    const int* cand_offsets,
    int Npairs,
    int icm_iters,
    int ils_iters,
    int perturb_k,
    std::uint32_t seed,
    const std::uint64_t* node_sample_id_base,
    int* best_parent_local_out,
    float* best_cost_out,
    FullCode* best_B_out,
    float* best_a_out,
    std::string* err) {
    return EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootEx(
        ctx, C_root, X_block, d, B, cand_parent_local_flat, pair_node, cand_offsets, Npairs,
        icm_iters, ils_iters, perturb_k, seed, node_sample_id_base,
        /*pair_t_override=*/nullptr,
        best_parent_local_out, best_cost_out, best_B_out, best_a_out, err);
}

}  // namespace stlq
