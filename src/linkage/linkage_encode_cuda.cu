#include "stlq/linkage/linkage_encode_cuda.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "stlq/core/blas.h"
#include "stlq/core/model_limits.h"
#include "stlq/core/threading.h"
#include "stlq/common/logger.h"

namespace stlq {

std::atomic<int> g_lr_xc0_chunk_mb{256};
std::atomic<int> g_lr_workspace_mb{0};

void SetCudaLinkageEncodeLargeRootXc0ChunkMb(int mb) {
    g_lr_xc0_chunk_mb.store(std::max(0, mb), std::memory_order_relaxed);
}

void SetCudaLinkageEncodeLargeRootWorkspaceMb(int mb) {
    g_lr_workspace_mb.store(std::max(0, mb), std::memory_order_relaxed);
}

// Hybrid init-linkage ILS policy: process-wide variable controlling how many initial ILS rounds
// use VarRoot mode (layer0 perturbable + root ICM GEMM) before switching to ConstRoot mode.
// -1 = all VarRoot (default).  See SetCudaLinkageEncodeHybridVarRootIlsRounds() doc in header.
static std::atomic<int> g_hybrid_varroot_ils_rounds{-1};

void SetCudaLinkageEncodeHybridVarRootIlsRounds(int rounds) { g_hybrid_varroot_ils_rounds.store(rounds, std::memory_order_relaxed); }
void ResetCudaLinkageEncodeHybridVarRootIlsRounds()          { g_hybrid_varroot_ils_rounds.store(-1, std::memory_order_relaxed); }
int  GetCudaLinkageEncodeHybridVarRootIlsRounds()             { return g_hybrid_varroot_ils_rounds.load(std::memory_order_relaxed); }

namespace {

constexpr float kEps = 1e-6f;
constexpr int kMaxM = kMaxSupportedModelM;

template <typename Fn>
inline void DispatchM2To20(int m, const char* what, Fn&& fn) {
    switch (m) {
        case 2: fn(std::integral_constant<int, 2>{}); return;
        case 3: fn(std::integral_constant<int, 3>{}); return;
        case 4: fn(std::integral_constant<int, 4>{}); return;
        case 5: fn(std::integral_constant<int, 5>{}); return;
        case 6: fn(std::integral_constant<int, 6>{}); return;
        case 7: fn(std::integral_constant<int, 7>{}); return;
        case 8: fn(std::integral_constant<int, 8>{}); return;
        case 9: fn(std::integral_constant<int, 9>{}); return;
        case 10: fn(std::integral_constant<int, 10>{}); return;
        case 11: fn(std::integral_constant<int, 11>{}); return;
        case 12: fn(std::integral_constant<int, 12>{}); return;
        case 13: fn(std::integral_constant<int, 13>{}); return;
        case 14: fn(std::integral_constant<int, 14>{}); return;
        case 15: fn(std::integral_constant<int, 15>{}); return;
        case 16: fn(std::integral_constant<int, 16>{}); return;
        case 17: fn(std::integral_constant<int, 17>{}); return;
        case 18: fn(std::integral_constant<int, 18>{}); return;
        case 19: fn(std::integral_constant<int, 19>{}); return;
        case 20: fn(std::integral_constant<int, 20>{}); return;
        default:
            throw std::runtime_error(std::string(what) + ": unsupported m=" + std::to_string(m));
    }
}

std::atomic<bool> g_profile_enabled{false};
std::atomic<std::uint64_t> g_calls{0};
std::atomic<std::int64_t> g_gemm_ns{0};
std::atomic<std::int64_t> g_greedy_ns{0};
std::atomic<std::int64_t> g_ls_cost_ns{0};
std::atomic<std::int64_t> g_icm_ns{0};
std::atomic<std::int64_t> g_ils_ns{0};
std::atomic<std::int64_t> g_ils_copy_ns{0};
std::atomic<std::int64_t> g_ils_solve_cost_ns{0};
std::atomic<std::int64_t> g_ils_icm_ns{0};
std::atomic<std::int64_t> g_ils_encode_ns{0};
std::atomic<std::int64_t> g_ils_accept_ns{0};
std::atomic<std::int64_t> g_total_ns{0};

// Large-root encode stats (shape/flow), collected only under profiling.
std::atomic<std::uint64_t> g_lr_calls{0};
std::atomic<std::uint64_t> g_lr_root_chunks_total{0};
std::atomic<std::uint64_t> g_lr_root_chunks_max{0};
std::atomic<std::uint64_t> g_lr_have_xc0_full_calls{0};
std::atomic<std::uint64_t> g_lr_root_chunk_h_min{0};
std::atomic<std::uint64_t> g_lr_root_chunk_h_max{0};
std::atomic<std::uint64_t> g_lr_root_icm_fast_calls{0};
std::atomic<std::uint64_t> g_lr_root_icm_slow_calls{0};
std::atomic<std::uint64_t> g_lr_sample_chunk_calls{0};
std::atomic<std::uint64_t> g_lr_sample_chunks_total{0};
std::atomic<std::uint64_t> g_lr_sample_chunk_n_min{0};
std::atomic<std::uint64_t> g_lr_sample_chunk_n_max{0};

// Host-side overhead stats (EnsurePrecomp / EnsureBatch / allocations).
std::atomic<std::uint64_t> g_host_calls{0};
std::atomic<std::int64_t> g_host_ensure_ns{0};
std::atomic<std::uint64_t> g_host_precomp_refresh_calls{0};
std::atomic<std::uint64_t> g_host_batch_grow_calls{0};
std::atomic<std::uint64_t> g_host_xc_grow_calls{0};
std::atomic<std::uint64_t> g_host_xc_grow_max_elems{0};

inline void AtomicMaxU64(std::atomic<std::uint64_t>* dst, std::uint64_t v) {
    std::uint64_t cur = dst->load(std::memory_order_relaxed);
    while (cur < v && !dst->compare_exchange_weak(cur, v, std::memory_order_relaxed)) {
    }
}

inline void AtomicMinU64(std::atomic<std::uint64_t>* dst, std::uint64_t v) {
    std::uint64_t cur = dst->load(std::memory_order_relaxed);
    if (cur == 0) {
        // Use 0 as "unset" sentinel; first writer wins.
        dst->compare_exchange_strong(cur, v, std::memory_order_relaxed);
        cur = dst->load(std::memory_order_relaxed);
    }
    while (cur > v && !dst->compare_exchange_weak(cur, v, std::memory_order_relaxed)) {
    }
}

inline void ThrowIf(cudaError_t st, const char* what) {
    if (st == cudaSuccess) return;
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(st));
}

inline void ThrowIf(cublasStatus_t st, const char* what) {
    if (st == CUBLAS_STATUS_SUCCESS) return;
    throw std::runtime_error(std::string(what) + ": cublasStatus=" + std::to_string(static_cast<int>(st)));
}

inline void AddNs(std::atomic<std::int64_t>* dst, float ms) {
    const auto ns = static_cast<std::int64_t>(static_cast<double>(ms) * 1e6);
    dst->fetch_add(ns, std::memory_order_relaxed);
}

inline std::int64_t NowNs() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch()).count();
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
	            // Geometric growth to reduce cudaMalloc/cudaFree churn across varying batch sizes.
	            // 1.5× tends to be a good balance between fewer reallocations and VRAM overhead.
	            const std::size_t grow = (new_cap >> 1) + 1;
	            const std::size_t next = new_cap + grow;
	            new_cap = (next > new_cap) ? next : n;  // overflow-safe fallback
	        }
	        Reset();
	        const std::size_t bytes = sizeof(T) * new_cap;
	        const cudaError_t st = cudaMalloc(&ptr, bytes);
	        if (st != cudaSuccess) {
	            throw std::runtime_error(std::string("cudaMalloc(") + std::to_string(bytes) + " bytes) failed: " +
	                                     cudaGetErrorString(st));
	        }
	        cap = new_cap;
	    }
	};

struct SharedPrecompDevice {
    int device = 0;
    std::uint64_t tag = 0;
    std::uint64_t build_tag = 0;
    int d = 0;
    int m = 0;
    int H = 0;

    DeviceBuf<float> d_C;
    DeviceBuf<float> d_G;
    DeviceBuf<float> d_inv;
    DeviceBuf<int> d_offsets;
    DeviceBuf<int> d_hvec;
    DeviceBuf<int> d_flat_layer;
};

std::mutex g_shared_precomp_mu;
std::vector<std::unique_ptr<SharedPrecompDevice>> g_shared_precomp;

struct SharedLargeRootPrecompDevice {
    int device = 0;
    std::uint64_t tag = 0;
    int d = 0;
    int m = 0;
    int h0 = 0;
    int Hs = 0;

    // Root layer0 codebook (C0): d×h0 (col-major).
    DeviceBuf<float> d_C0;
    // Root invnorm (1/sqrt(norm2)): h0.
    DeviceBuf<float> d_inv_root;
    // Flattened small layers 1..m-1 (C_small): d×Hs (col-major).
    DeviceBuf<float> d_C_small;
    // Cross table (G0s): Hs×h0 (col-major), where G0s[:,c] = C_small^T * C0[:,c].
    DeviceBuf<float> d_G0s;
    // Transposed cross table (G0s_T): h0×Hs (col-major), where G0s_T[:,r] = G0s[r,:]^T.
    // This layout makes scanning over root codes (code-major) coalesced for a fixed flattened-small index `r`.
    DeviceBuf<float> d_G0s_T;
    // Small-small Gram (G_small): Hs×Hs (col-major).
    DeviceBuf<float> d_G_small;
    // Small invnorm (1/sqrt(norm2)): Hs.
    DeviceBuf<float> d_inv_small;
    // Small offsets (within flattened small): size m, small_offsets[0]=0, small_offsets[1]=0.
    DeviceBuf<int> d_small_offsets;
    // Full h_vec: size m.
    DeviceBuf<int> d_hvec;
};

std::mutex g_shared_lr_mu;
std::vector<std::unique_ptr<SharedLargeRootPrecompDevice>> g_shared_lr_precomp;

std::uint64_t PrecompDeviceTag(const Precomp& pre) {
    return static_cast<std::uint64_t>(pre.build_tag) ^
           (static_cast<std::uint64_t>(pre.H) << 1) ^
           (static_cast<std::uint64_t>(pre.m) << 33);
}

SharedPrecompDevice* GetOrCreateSharedPrecomp(const CudaCtx& ctx, const Precomp& pre) {
    const std::uint64_t tag = PrecompDeviceTag(pre);
    std::lock_guard<std::mutex> lock(g_shared_precomp_mu);
    for (const auto& p : g_shared_precomp) {
        if (!p || p->device != ctx.device) continue;
        if (p->d != pre.d || p->m != pre.m || p->H != pre.H) continue;
        if (p->build_tag != pre.build_tag) continue;
        // IMPORTANT: do not refresh in-place on build_tag mismatch. Multiple different precomps
        // (e.g. C_root vs C_one) can share the same shape and may be used concurrently across threads.
        return p.get();
    }

    auto entry = std::make_unique<SharedPrecompDevice>();
    entry->device = ctx.device;
    entry->tag = tag;
    entry->build_tag = pre.build_tag;
    entry->d = pre.d;
    entry->m = pre.m;
    entry->H = pre.H;

    ThrowIf(cudaSetDevice(ctx.device), "cudaSetDevice");
    entry->d_C.Ensure(static_cast<std::size_t>(pre.d) * static_cast<std::size_t>(pre.H));
    entry->d_G.Ensure(static_cast<std::size_t>(pre.H) * static_cast<std::size_t>(pre.H));
    entry->d_inv.Ensure(static_cast<std::size_t>(pre.H));
    entry->d_offsets.Ensure(static_cast<std::size_t>(pre.m));
    entry->d_hvec.Ensure(static_cast<std::size_t>(pre.m));
    entry->d_flat_layer.Ensure(static_cast<std::size_t>(pre.H));

    ThrowIf(cudaMemcpy(entry->d_C.ptr, pre.C_all.data.data(),
                       sizeof(float) * static_cast<std::size_t>(pre.d) * static_cast<std::size_t>(pre.H),
                       cudaMemcpyHostToDevice),
            "Memcpy C_all");
    ThrowIf(cudaMemcpy(entry->d_G.ptr, pre.G.data.data(),
                       sizeof(float) * static_cast<std::size_t>(pre.H) * static_cast<std::size_t>(pre.H),
                       cudaMemcpyHostToDevice),
            "Memcpy G");
    ThrowIf(cudaMemcpy(entry->d_inv.ptr, pre.invnorm_flat.data(),
                       sizeof(float) * static_cast<std::size_t>(pre.H),
                       cudaMemcpyHostToDevice),
            "Memcpy invnorm");
    ThrowIf(cudaMemcpy(entry->d_offsets.ptr, pre.offsets.data(),
                       sizeof(int) * static_cast<std::size_t>(pre.m),
                       cudaMemcpyHostToDevice),
            "Memcpy offsets");
    ThrowIf(cudaMemcpy(entry->d_hvec.ptr, pre.h_vec.data(),
                       sizeof(int) * static_cast<std::size_t>(pre.m),
                       cudaMemcpyHostToDevice),
            "Memcpy hvec");
    ThrowIf(cudaMemcpy(entry->d_flat_layer.ptr, pre.flat_layer.data(),
                       sizeof(int) * static_cast<std::size_t>(pre.H),
                       cudaMemcpyHostToDevice),
            "Memcpy flat_layer");

    g_shared_precomp.push_back(std::move(entry));
    return g_shared_precomp.back().get();
}

std::uint64_t LargeRootDeviceTag(const CodebookPack& C_root) {
    // Cache key for shared large-root precomp.
    //
    // IMPORTANT:
    // - Must be stable across benign CodebookPack copies (pointer addresses may change).
    // - Must change when codebook contents change (caller is responsible for bumping build_tag).
    //
    // We therefore key primarily on `build_tag` (monotonic across codebook updates) plus shapes.
    // If build_tag is 0, caching becomes unsafe; treat that as a distinct (but still stable) tag
    // using only shapes (caller should ensure build_tag is set for correctness across updates).
    std::uint64_t tag = 0x9e3779b97f4a7c15ULL ^ static_cast<std::uint64_t>(C_root.build_tag);
    tag ^= static_cast<std::uint64_t>(C_root.d) * 0xbf58476d1ce4e5b9ULL;
    const int m = static_cast<int>(C_root.books.size());
    tag ^= static_cast<std::uint64_t>(m) * 0x94d049bb133111ebULL;
    for (int l = 0; l < m; ++l) {
        const auto& book = C_root.books[static_cast<std::size_t>(l)];
        tag ^= static_cast<std::uint64_t>(book.rows) + (static_cast<std::uint64_t>(book.cols) << 32);
    }
    return tag;
}


SharedLargeRootPrecompDevice* GetOrCreateSharedLargeRootPrecomp(const CudaCtx& ctx, const CodebookPack& C_root) {
    const int m = static_cast<int>(C_root.books.size());
    if (m <= 0) {
        throw std::runtime_error("GetOrCreateSharedLargeRootPrecomp: empty codebooks.");
    }
    const int d = C_root.books.front().rows;
    const int h0 = C_root.books.front().cols;
    int Hs = 0;
    for (int l = 1; l < m; ++l) {
        Hs += C_root.books[static_cast<std::size_t>(l)].cols;
    }
    const std::uint64_t tag = LargeRootDeviceTag(C_root);

    std::lock_guard<std::mutex> lock(g_shared_lr_mu);
    for (const auto& p : g_shared_lr_precomp) {
        if (!p || p->device != ctx.device) continue;
        if (p->tag != tag) continue;
        if (p->d != d || p->m != m || p->h0 != h0 || p->Hs != Hs) continue;
        return p.get();
    }

    auto entry = std::make_unique<SharedLargeRootPrecompDevice>();
    entry->device = ctx.device;
    entry->tag = tag;
    entry->d = d;
    entry->m = m;
    entry->h0 = h0;
    entry->Hs = Hs;

    ThrowIf(cudaSetDevice(ctx.device), "cudaSetDevice");
    entry->d_C0.Ensure(static_cast<std::size_t>(d) * static_cast<std::size_t>(h0));
    entry->d_inv_root.Ensure(static_cast<std::size_t>(h0));
    entry->d_C_small.Ensure(static_cast<std::size_t>(d) * static_cast<std::size_t>(Hs));
    entry->d_G0s.Ensure(static_cast<std::size_t>(Hs) * static_cast<std::size_t>(h0));
    entry->d_G0s_T.Ensure(static_cast<std::size_t>(h0) * static_cast<std::size_t>(Hs));
    entry->d_G_small.Ensure(static_cast<std::size_t>(Hs) * static_cast<std::size_t>(Hs));
    entry->d_inv_small.Ensure(static_cast<std::size_t>(Hs));
    entry->d_small_offsets.Ensure(static_cast<std::size_t>(m));
    entry->d_hvec.Ensure(static_cast<std::size_t>(m));

    // Upload C0.
    ThrowIf(cudaMemcpy(entry->d_C0.ptr,
                       C_root.books[0].data.data(),
                       sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(h0),
                       cudaMemcpyHostToDevice),
            "Memcpy C0");
    // Upload invnorm for C0.
    {
        std::vector<float> inv_root_host;
        inv_root_host.resize(static_cast<std::size_t>(h0));
        for (int c = 0; c < h0; ++c) {
            const float* src = C_root.books[0].Col(c);
            float n2 = 0.0f;
            for (int r = 0; r < d; ++r) {
                const float v = src[r];
                n2 += v * v;
            }
            inv_root_host[static_cast<std::size_t>(c)] = n2 > 0.0f ? 1.0f / std::sqrt(n2) : 0.0f;
        }
        ThrowIf(cudaMemcpy(entry->d_inv_root.ptr,
                           inv_root_host.data(),
                           sizeof(float) * static_cast<std::size_t>(h0),
                           cudaMemcpyHostToDevice),
                "Memcpy inv_root");
    }

    // Flatten and upload C_small + invnorm_small.
    std::vector<float> C_small_host;
    C_small_host.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(Hs));
    std::vector<float> inv_small_host;
    inv_small_host.resize(static_cast<std::size_t>(Hs));
    int col_off = 0;
    for (int l = 1; l < m; ++l) {
        const auto& book = C_root.books[static_cast<std::size_t>(l)];
        if (book.rows != d) {
            throw std::runtime_error("GetOrCreateSharedLargeRootPrecomp: d mismatch across books.");
        }
        for (int c = 0; c < book.cols; ++c) {
            const float* src = book.Col(c);
            float* dst = C_small_host.data() + static_cast<std::size_t>(col_off + c) * static_cast<std::size_t>(d);
            float n2 = 0.0f;
            for (int r = 0; r < d; ++r) {
                const float v = src[r];
                dst[r] = v;
                n2 += v * v;
            }
            inv_small_host[static_cast<std::size_t>(col_off + c)] =
                n2 > 0.0f ? 1.0f / std::sqrt(n2) : 0.0f;
        }
        col_off += book.cols;
    }
    ThrowIf(cudaMemcpy(entry->d_C_small.ptr,
                       C_small_host.data(),
                       sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(Hs),
                       cudaMemcpyHostToDevice),
            "Memcpy C_small");
    ThrowIf(cudaMemcpy(entry->d_inv_small.ptr,
                       inv_small_host.data(),
                       sizeof(float) * static_cast<std::size_t>(Hs),
                       cudaMemcpyHostToDevice),
            "Memcpy inv_small");

    // Build cross table G0s = C_small^T * C0 on GPU (Hs×h0). This avoids redoing GEMM for every batch.
    // Synchronize before publishing the shared entry to avoid stream races across threads.
    {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        ThrowIf(cublasSetStream(ctx.cublas, ctx.stream), "cublasSetStream(G0s)");
        ThrowIf(cublasSgemm(ctx.cublas,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            /*rows(C)=*/Hs, /*cols(C)=*/h0, /*k=*/d,
                            &alpha,
                            entry->d_C_small.ptr, d,
                            entry->d_C0.ptr, d,
                            &beta,
                            entry->d_G0s.ptr, /*ldc=*/Hs),
                "cublasSgemm(G0s)");
        // Also build a transposed view for fast root scans: G0s_T = G0s^T (h0×Hs).
        // Use GEAM transpose to preserve exact values (avoid recomputing via a second GEMM).
        // Note: even with beta=0, cuBLAS validates B/ldb against op(B) shape. Use the same
        // matrix and transpose for B to satisfy shape constraints (B is ignored by beta=0).
        ThrowIf(cublasSgeam(ctx.cublas,
                            CUBLAS_OP_T, CUBLAS_OP_T,
                            /*rows(C)=*/h0, /*cols(C)=*/Hs,
                            &alpha,
                            entry->d_G0s.ptr, /*lda=*/Hs,
                            &beta,
                            entry->d_G0s.ptr, /*ldb=*/Hs,
                            entry->d_G0s_T.ptr, /*ldc=*/h0),
                "cublasSgeam(G0s_T)");
        ThrowIf(cudaStreamSynchronize(ctx.stream), "cudaStreamSynchronize(G0s)");
    }

    ColMajorMatrix<float> C_small_mat;
    C_small_mat.rows = d;
    C_small_mat.cols = Hs;
    C_small_mat.data = C_small_host;

    // Upload offsets and hvec.
    std::vector<int> h_vec_host;
    h_vec_host.resize(static_cast<std::size_t>(m));
    for (int l = 0; l < m; ++l) {
        h_vec_host[static_cast<std::size_t>(l)] = C_root.books[static_cast<std::size_t>(l)].cols;
    }
    std::vector<int> small_offsets_host;
    small_offsets_host.assign(static_cast<std::size_t>(m), 0);
    int off = 0;
    for (int l = 1; l < m; ++l) {
        small_offsets_host[static_cast<std::size_t>(l)] = off;
        off += h_vec_host[static_cast<std::size_t>(l)];
    }
    ThrowIf(cudaMemcpy(entry->d_hvec.ptr,
                       h_vec_host.data(),
                       sizeof(int) * static_cast<std::size_t>(m),
                       cudaMemcpyHostToDevice),
            "Memcpy hvec");
    ThrowIf(cudaMemcpy(entry->d_small_offsets.ptr,
                       small_offsets_host.data(),
                       sizeof(int) * static_cast<std::size_t>(m),
                       cudaMemcpyHostToDevice),
            "Memcpy small_offsets");

    // Build G_small on CPU to match full-precomp numerical behavior (avoid cublas/TF32 drift).
    ColMajorMatrix<float> G_small_host(Hs, Hs);
    {
        ScopedBlasThreads blas_scope(OmpMaxThreads());
        Gemm(true, false, 1.0f, C_small_mat, C_small_mat, 0.0f, &G_small_host);
    }
    ThrowIf(cudaMemcpy(entry->d_G_small.ptr,
                       G_small_host.data.data(),
                       sizeof(float) * static_cast<std::size_t>(Hs) * static_cast<std::size_t>(Hs),
                       cudaMemcpyHostToDevice),
            "Memcpy G_small");

    g_shared_lr_precomp.push_back(std::move(entry));
    return g_shared_lr_precomp.back().get();
}

__global__ void CopyF32(const float* __restrict src, float* __restrict dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = src[i];
}

__global__ void InitBestRootState(int n,
                                 float* __restrict best_score,   // n
                                 RootCode* __restrict best_code, // n
                                 float* __restrict best_xC0,     // n
                                 float* __restrict best_norm0) { // n
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    best_score[i] = -INFINITY;
    best_code[i] = static_cast<RootCode>(0);
    best_xC0[i] = 0.0f;
    best_norm0[i] = 0.0f;
}

__global__ void UpdateBestRootFromChunk(int h0_start,
                                       int chunk_h,
                                       int h0_total,
                                       int n,
                                       const float* __restrict xC0_chunk,   // chunk_h×n
                                       const float* __restrict inv_root,    // h0_total
                                       float* __restrict best_score,        // n
                                       RootCode* __restrict best_code,      // n
                                       float* __restrict best_xC0,          // n
                                       float* __restrict best_norm0) {      // n
    // One block per sample, 256 threads: parallel scan over codes in this chunk.
    const int i = blockIdx.x;
    if (i >= n) return;
    const int tid = threadIdx.x;

    float best_s = -INFINITY;
    int best_c = 0;
    float best_xc = 0.0f;
    float best_n0 = 0.0f;

    const float* col = xC0_chunk + static_cast<std::size_t>(i) * static_cast<std::size_t>(chunk_h);
    for (int k = tid; k < chunk_h; k += blockDim.x) {
        const int code = h0_start + k;
        if (code >= h0_total) break;
        const float invn = inv_root[code];
        const float xc = col[k];
        const float score = fabsf(xc * invn);
        if (score > best_s || (score == best_s && code < best_c)) {
            best_s = score;
            best_c = code;
            best_xc = xc;
            best_n0 = (invn > 0.0f) ? (1.0f / (invn * invn)) : 0.0f;
        }
    }

    __shared__ float sh_s[256];
    __shared__ int sh_c[256];
    __shared__ float sh_xc[256];
    __shared__ float sh_n0[256];
    sh_s[tid] = best_s;
    sh_c[tid] = best_c;
    sh_xc[tid] = best_xc;
    sh_n0[tid] = best_n0;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) {
            const float s2 = sh_s[tid + off];
            const int c2 = sh_c[tid + off];
            const float s1 = sh_s[tid];
            const int c1 = sh_c[tid];
            if (s2 > s1 || (s2 == s1 && c2 < c1)) {
                sh_s[tid] = s2;
                sh_c[tid] = c2;
                sh_xc[tid] = sh_xc[tid + off];
                sh_n0[tid] = sh_n0[tid + off];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float cur_best = best_score[i];
        const int cur_code = static_cast<int>(best_code[i]);
        const float cand_best = sh_s[0];
        const int cand_code = sh_c[0];
        if (cand_best > cur_best || (cand_best == cur_best && cand_code < cur_code)) {
            best_score[i] = cand_best;
            best_code[i] = static_cast<RootCode>(cand_code);
            best_xC0[i] = sh_xc[0];
            best_norm0[i] = sh_n0[0];
        }
    }
}

__global__ void InitBestRootCandState(int n,
                                     float* __restrict best_score,    // n
                                     RootCode* __restrict best_code) { // n
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    best_score[i] = -INFINITY;
    best_code[i] = static_cast<RootCode>(0);
}

__global__ void UpdateBestRootCandFromChunk(int h0_start,
                                           int chunk_h,
                                           int h0_total,
                                           int n,
                                           const float* __restrict xC0_chunk,   // chunk_h×n
                                           const float* __restrict inv_root,    // h0_total (1/||c||)
                                           float* __restrict best_score,        // n
                                           RootCode* __restrict best_code) {    // n
    const int i = blockIdx.x;
    if (i >= n) return;
    const int tid = threadIdx.x;

    float best_s = -INFINITY;
    int best_c = 0;

    const float* col = xC0_chunk + static_cast<std::size_t>(i) * static_cast<std::size_t>(chunk_h);
    for (int k = tid; k < chunk_h; k += blockDim.x) {
        const int code = h0_start + k;
        if (code >= h0_total) break;
        const float invn = inv_root[code];
        const float xc = col[k];
        const float score = fabsf(xc * invn);
        if (score > best_s || (score == best_s && code < best_c)) {
            best_s = score;
            best_c = code;
        }
    }

    __shared__ float sh_s[256];
    __shared__ int sh_c[256];
    sh_s[tid] = best_s;
    sh_c[tid] = best_c;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) {
            const float s2 = sh_s[tid + off];
            const int c2 = sh_c[tid + off];
            const float s1 = sh_s[tid];
            const int c1 = sh_c[tid];
            if (s2 > s1 || (s2 == s1 && c2 < c1)) {
                sh_s[tid] = s2;
                sh_c[tid] = c2;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float cur_best = best_score[i];
        const int cur_code = static_cast<int>(best_code[i]);
        const float cand_best = sh_s[0];
        const int cand_code = sh_c[0];
        if (cand_best > cur_best || (cand_best == cur_best && cand_code < cur_code)) {
            best_score[i] = cand_best;
            best_code[i] = static_cast<RootCode>(cand_code);
        }
    }
}

template <int M>
__global__ void FindBestRootCandFromXc0FullAndSmallFixed(int h0,
                                                        const float* __restrict xC0_full,  // h0×n (col-major)
                                                        const float* __restrict inv_root, // h0
                                                        const float* __restrict G0s_T,    // h0×Hs (col-major)
                                                        const int* __restrict small_offsets, // size M
                                                        const Code* __restrict B_small,      // (M-1)×n (per-sample contiguous)
                                                        const float* __restrict a,           // M×n (per-sample contiguous)
                                                        RootCode* __restrict out_code,       // n
                                                        int n) {
    static_assert(M > 1, "M must be > 1");
    const int i = blockIdx.x;
    if (i >= n) return;
    const int tid = threadIdx.x;

    const Code* B_col = B_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(M - 1);
    const float* a_col = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);

    int flats[M - 1];
    float alphas[M - 1];
    #pragma unroll
    for (int l = 1; l < M; ++l) {
        flats[l - 1] = small_offsets[l] + static_cast<int>(B_col[l - 1]);
        alphas[l - 1] = a_col[l];
    }

    float best_s = -INFINITY;
    int best_c = 0;

    const float* xc_col = xC0_full + static_cast<std::size_t>(i) * static_cast<std::size_t>(h0);
    for (int code = tid; code < h0; code += blockDim.x) {
        const float invn = inv_root[code];
        float xc = xc_col[code];
        float corr = 0.0f;
        #pragma unroll
        for (int t = 0; t < M - 1; ++t) {
            const int flat = flats[t];
            const float alpha = alphas[t];
            corr += alpha * G0s_T[static_cast<std::size_t>(code) + static_cast<std::size_t>(flat) * static_cast<std::size_t>(h0)];
        }
        xc -= corr;
        const float score = fabsf(xc * invn);
        if (score > best_s || (score == best_s && code < best_c)) {
            best_s = score;
            best_c = code;
        }
    }

    __shared__ float sh_s[256];
    __shared__ int sh_c[256];
    sh_s[tid] = best_s;
    sh_c[tid] = best_c;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) {
            const float s2 = sh_s[tid + off];
            const int c2 = sh_c[tid + off];
            const float s1 = sh_s[tid];
            const int c1 = sh_c[tid];
            if (s2 > s1 || (s2 == s1 && c2 < c1)) {
                sh_s[tid] = s2;
                sh_c[tid] = c2;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_code[i] = static_cast<RootCode>(sh_c[0]);
    }
}

template <int M>
__global__ void BuildRe0FromSmallCodesLargeRootFixed(int d,
                                                     const int* __restrict__ small_offsets, // size M
                                                     const float* __restrict__ C_small,     // d×Hs (col-major)
                                                     const Code* __restrict__ B_small,      // (M-1)×n
                                                     const float* __restrict__ a,           // M×n (col-major)
                                                     const float* __restrict__ residuals,   // d×n (col-major)
                                                     float* __restrict__ re0,               // d×n (col-major)
                                                     int n) {
    static_assert(M > 1, "M must be > 1");
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = d * n;
    if (idx >= total) return;
    const int r = idx % d;
    const int i = idx / d;

    float val = residuals[static_cast<std::size_t>(r) + static_cast<std::size_t>(i) * static_cast<std::size_t>(d)];
    const Code* B_col = B_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(M - 1);
    const float* a_col = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    #pragma unroll
    for (int l = 1; l < M; ++l) {
        const int flat = small_offsets[l] + static_cast<int>(B_col[l - 1]);
        const float alpha = a_col[l];
        val -= alpha * C_small[static_cast<std::size_t>(r) + static_cast<std::size_t>(flat) * static_cast<std::size_t>(d)];
    }
    re0[static_cast<std::size_t>(r) + static_cast<std::size_t>(i) * static_cast<std::size_t>(d)] = val;
}

__global__ void GatherG0sFromTable(int Hs,
                                  int n,
                                  const float* __restrict G0s,        // Hs×h0 (col-major)
                                  const RootCode* __restrict codes,   // n
                                  float* __restrict g0s_out) {        // Hs×n (col-major)
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = Hs * n;
    if (idx >= total) return;
    const int q = idx % Hs;
    const int i = idx / Hs;
    const int code = static_cast<int>(codes[i]);
    g0s_out[static_cast<std::size_t>(q) + static_cast<std::size_t>(i) * static_cast<std::size_t>(Hs)] =
        G0s[static_cast<std::size_t>(q) + static_cast<std::size_t>(code) * static_cast<std::size_t>(Hs)];
}

__global__ void ComputeXc0AndNorm0FromRootCodes(int d,
                                                int n,
                                                const float* __restrict__ residuals, // d×n (col-major)
                                                const float* __restrict__ C0,        // d×h0 (col-major)
                                                const RootCode* __restrict__ codes,  // n
                                                float* __restrict__ xC0,             // n
                                                float* __restrict__ norm0) {         // n
    const int i = blockIdx.x;
    if (i >= n) return;
    const int code = static_cast<int>(codes[i]);
    const float* c0 = C0 + static_cast<std::size_t>(code) * static_cast<std::size_t>(d);
    float dot = 0.0f;
    float norm = 0.0f;
    for (int rr = threadIdx.x; rr < d; rr += blockDim.x) {
        const float r = residuals[static_cast<std::size_t>(rr) + static_cast<std::size_t>(i) * static_cast<std::size_t>(d)];
        const float c = c0[rr];
        dot += r * c;
        norm += c * c;
    }
    __shared__ float sh_dot[256];
    __shared__ float sh_norm[256];
    sh_dot[threadIdx.x] = dot;
    sh_norm[threadIdx.x] = norm;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            sh_dot[threadIdx.x] += sh_dot[threadIdx.x + off];
            sh_norm[threadIdx.x] += sh_norm[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        xC0[i] = sh_dot[0];
        norm0[i] = sh_norm[0];
    }
}

__global__ void ApplyRootProjectionToRcSmall(int Hs,
                                            int n,
                                            const float* __restrict xC_small, // Hs×n
                                            const float* __restrict g0s,      // Hs×n
                                            const float* __restrict xC0,      // n
                                            const float* __restrict norm0,    // n
                                            float* __restrict rC_small) {     // Hs×n
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = Hs * n;
    if (idx >= total) return;
    const int q = idx % Hs;
    const int i = idx / Hs;
    const float n0 = norm0[i];
    const float alpha0 = (n0 > 0.0f) ? (xC0[i] / n0) : 0.0f;
    const std::size_t off = static_cast<std::size_t>(q) + static_cast<std::size_t>(i) * static_cast<std::size_t>(Hs);
    rC_small[off] = xC_small[off] - alpha0 * g0s[off];
}

__device__ __forceinline__ std::uint64_t SplitMixNextU64(std::uint64_t& state) {
    std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ std::uint64_t UniformBelowU64(std::uint64_t& state, std::uint64_t bound) {
    // Multiply-high mapping: floor(x * bound / 2^64). Deterministic and fast.
    // Assumes bound > 0.
    const std::uint64_t x = SplitMixNextU64(state);
    return __umul64hi(x, bound);
}

__global__ void LinkageCopyAndPerturbFullCodes(int m,
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

    FullCode* dst = B_dst + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    const FullCode* src = B_src + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    for (int l = 0; l < m; ++l) {
        dst[l] = src[l];
    }

    if (m <= 1 || ksel <= 0) return;
    const std::uint64_t i_global = sample_id_offset + static_cast<std::uint64_t>(i);
    auto state = static_cast<std::uint64_t>(seed);
    state ^= static_cast<std::uint64_t>(outer_iter_1based) * 0x9e3779b97f4a7c15ULL;
    state ^= (i_global + 1ULL) * 0xbf58476d1ce4e5b9ULL;

    // Match CPU linkage semantics (DynamicIcmEncodingSingleForResidual): allow perturbing any layer (including layer 0).
    for (int t = 0; t < ksel; ++t) {
        const int layer = static_cast<int>(UniformBelowU64(state, static_cast<std::uint64_t>(m)));
        const int h = h_vec[layer];
        if (h <= 1) continue;
        const int old_code = static_cast<int>(dst[layer]);
        // Uniform over all codes except old_code.
        const int raw = static_cast<int>(UniformBelowU64(state, static_cast<std::uint64_t>(h - 1)));
        const int new_code = (raw >= old_code) ? (raw + 1) : raw;
        dst[layer] = static_cast<FullCode>(new_code);
    }
}

template <int M>
__global__ void LinkageCopyAndPerturbFullCodesFixed(const int* __restrict__ h_vec,
                                                 std::uint32_t seed,
                                                 int outer_iter_1based,
                                                 std::uint64_t sample_id_offset,
                                                 int ksel,
                                                 const FullCode* __restrict__ B_src,
                                                 FullCode* __restrict__ B_dst,
                                                 int n) {
    static_assert(M > 0, "M must be positive");
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    FullCode* dst = B_dst + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    const FullCode* src = B_src + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    #pragma unroll
    for (int l = 0; l < M; ++l) {
        dst[l] = src[l];
    }

    if (M <= 1 || ksel <= 0) return;
    const std::uint64_t i_global = sample_id_offset + static_cast<std::uint64_t>(i);
    auto state = static_cast<std::uint64_t>(seed);
    state ^= static_cast<std::uint64_t>(outer_iter_1based) * 0x9e3779b97f4a7c15ULL;
    state ^= (i_global + 1ULL) * 0xbf58476d1ce4e5b9ULL;

    // Match CPU linkage semantics (DynamicIcmEncodingSingleForResidual): allow perturbing any layer (including layer 0).
    for (int t = 0; t < ksel; ++t) {
        const int layer = static_cast<int>(UniformBelowU64(state, static_cast<std::uint64_t>(M)));
        const int h = h_vec[layer];
        if (h <= 1) continue;
        const int old_code = static_cast<int>(dst[layer]);
        const int raw = static_cast<int>(UniformBelowU64(state, static_cast<std::uint64_t>(h - 1)));
        const int new_code = (raw >= old_code) ? (raw + 1) : raw;
        dst[layer] = static_cast<FullCode>(new_code);
    }
}

__global__ void LinkageCopyAndPerturbFullCodesSkipLayer0(int m,
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

    FullCode* dst = B_dst + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    const FullCode* src = B_src + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    for (int l = 0; l < m; ++l) {
        dst[l] = src[l];
    }

    if (m <= 1 || ksel <= 0) return;

    const std::uint64_t i_global = sample_id_offset + static_cast<std::uint64_t>(i);
    auto state = static_cast<std::uint64_t>(seed);
    state ^= static_cast<std::uint64_t>(outer_iter_1based) * 0x9e3779b97f4a7c15ULL;
    state ^= (i_global + 1ULL) * 0xbf58476d1ce4e5b9ULL;

    for (int t = 0; t < ksel; ++t) {
        const int layer = 1 + static_cast<int>(UniformBelowU64(state, static_cast<std::uint64_t>(m - 1)));
        const int h = h_vec[layer];
        if (h <= 1) continue;
        const int old_code = static_cast<int>(dst[layer]);
        const int raw = static_cast<int>(UniformBelowU64(state, static_cast<std::uint64_t>(h - 1)));
        const int new_code = (raw >= old_code) ? (raw + 1) : raw;
        dst[layer] = static_cast<FullCode>(new_code);
    }
}

template <int M>
__global__ void LinkageCopyAndPerturbFullCodesSkipLayer0Fixed(const int* __restrict__ h_vec,
                                                           std::uint32_t seed,
                                                           int outer_iter_1based,
                                                           std::uint64_t sample_id_offset,
                                                           int ksel,
                                                           const FullCode* __restrict__ B_src,
                                                           FullCode* __restrict__ B_dst,
                                                           int n) {
    static_assert(M > 0, "M must be positive");
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    FullCode* dst = B_dst + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    const FullCode* src = B_src + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    #pragma unroll
    for (int l = 0; l < M; ++l) {
        dst[l] = src[l];
    }

    if (M <= 1 || ksel <= 0) return;

    const std::uint64_t i_global = sample_id_offset + static_cast<std::uint64_t>(i);
    auto state = static_cast<std::uint64_t>(seed);
    state ^= static_cast<std::uint64_t>(outer_iter_1based) * 0x9e3779b97f4a7c15ULL;
    state ^= (i_global + 1ULL) * 0xbf58476d1ce4e5b9ULL;

    for (int t = 0; t < ksel; ++t) {
        const int layer = 1 + static_cast<int>(UniformBelowU64(state, static_cast<std::uint64_t>(M - 1)));
        const int h = h_vec[layer];
        if (h <= 1) continue;
        const int old_code = static_cast<int>(dst[layer]);
        const int raw = static_cast<int>(UniformBelowU64(state, static_cast<std::uint64_t>(h - 1)));
        const int new_code = (raw >= old_code) ? (raw + 1) : raw;
        dst[layer] = static_cast<FullCode>(new_code);
    }
}

__global__ void LinkageCopyAndPerturbFullCodesWithSampleIds(int m,
                                                         const int* __restrict__ h_vec,
                                                         std::uint32_t seed,
                                                         int outer_iter_1based,
                                                         const std::uint64_t* __restrict__ sample_ids,
                                                         int ksel,
                                                         const FullCode* __restrict__ B_src,
                                                         FullCode* __restrict__ B_dst,
                                                         int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    FullCode* dst = B_dst + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    const FullCode* src = B_src + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    for (int l = 0; l < m; ++l) {
        dst[l] = src[l];
    }

    if (m <= 1 || ksel <= 0) return;
    const std::uint64_t i_global = sample_ids[static_cast<std::size_t>(i)];
    auto state = static_cast<std::uint64_t>(seed);
    state ^= static_cast<std::uint64_t>(outer_iter_1based) * 0x9e3779b97f4a7c15ULL;
    state ^= (i_global + 1ULL) * 0xbf58476d1ce4e5b9ULL;

    // Match CPU linkage semantics (DynamicIcmEncodingSingleForResidual): allow perturbing any layer (including layer 0).
    for (int t = 0; t < ksel; ++t) {
        const int layer = static_cast<int>(UniformBelowU64(state, static_cast<std::uint64_t>(m)));
        const int h = h_vec[layer];
        if (h <= 1) continue;
        const int old_code = static_cast<int>(dst[layer]);
        const int raw = static_cast<int>(UniformBelowU64(state, static_cast<std::uint64_t>(h - 1)));
        const int new_code = (raw >= old_code) ? (raw + 1) : raw;
        dst[layer] = static_cast<FullCode>(new_code);
    }
}

template <int M>
__global__ void LinkageCopyAndPerturbFullCodesWithSampleIdsFixed(const int* __restrict__ h_vec,
                                                              std::uint32_t seed,
                                                              int outer_iter_1based,
                                                              const std::uint64_t* __restrict__ sample_ids,
                                                              int ksel,
                                                              const FullCode* __restrict__ B_src,
                                                              FullCode* __restrict__ B_dst,
                                                              int n) {
    static_assert(M > 0, "M must be positive");
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    FullCode* dst = B_dst + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    const FullCode* src = B_src + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    #pragma unroll
    for (int l = 0; l < M; ++l) {
        dst[l] = src[l];
    }

    if (M <= 1 || ksel <= 0) return;
    const std::uint64_t i_global = sample_ids[static_cast<std::size_t>(i)];
    auto state = static_cast<std::uint64_t>(seed);
    state ^= static_cast<std::uint64_t>(outer_iter_1based) * 0x9e3779b97f4a7c15ULL;
    state ^= (i_global + 1ULL) * 0xbf58476d1ce4e5b9ULL;

    // Match CPU linkage semantics (DynamicIcmEncodingSingleForResidual): allow perturbing any layer (including layer 0).
    for (int t = 0; t < ksel; ++t) {
        const int layer = static_cast<int>(UniformBelowU64(state, static_cast<std::uint64_t>(M)));
        const int h = h_vec[layer];
        if (h <= 1) continue;
        const int old_code = static_cast<int>(dst[layer]);
        const int raw = static_cast<int>(UniformBelowU64(state, static_cast<std::uint64_t>(h - 1)));
        const int new_code = (raw >= old_code) ? (raw + 1) : raw;
        dst[layer] = static_cast<FullCode>(new_code);
    }
}

__global__ void LinkageCopyAndPerturbFullCodesSkipLayer0WithSampleIds(int m,
                                                                   const int* __restrict__ h_vec,
                                                                   std::uint32_t seed,
                                                                   int outer_iter_1based,
                                                                   const std::uint64_t* __restrict__ sample_ids,
                                                                   int ksel,
                                                                   const FullCode* __restrict__ B_src,
                                                                   FullCode* __restrict__ B_dst,
                                                                   int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    FullCode* dst = B_dst + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    const FullCode* src = B_src + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    for (int l = 0; l < m; ++l) {
        dst[l] = src[l];
    }

    if (m <= 1 || ksel <= 0) return;
    const std::uint64_t i_global = sample_ids[static_cast<std::size_t>(i)];
    auto state = static_cast<std::uint64_t>(seed);
    state ^= static_cast<std::uint64_t>(outer_iter_1based) * 0x9e3779b97f4a7c15ULL;
    state ^= (i_global + 1ULL) * 0xbf58476d1ce4e5b9ULL;

    for (int t = 0; t < ksel; ++t) {
        const int layer = 1 + static_cast<int>(UniformBelowU64(state, static_cast<std::uint64_t>(m - 1)));
        const int h = h_vec[layer];
        if (h <= 1) continue;
        const int old_code = static_cast<int>(dst[layer]);
        const int raw = static_cast<int>(UniformBelowU64(state, static_cast<std::uint64_t>(h - 1)));
        const int new_code = (raw >= old_code) ? (raw + 1) : raw;
        dst[layer] = static_cast<FullCode>(new_code);
    }
}

template <int M>
__global__ void LinkageCopyAndPerturbFullCodesSkipLayer0WithSampleIdsFixed(
    const int* __restrict__ h_vec,
    std::uint32_t seed,
    int outer_iter_1based,
    const std::uint64_t* __restrict__ sample_ids,
    int ksel,
    const FullCode* __restrict__ B_src,
    FullCode* __restrict__ B_dst,
    int n) {
    static_assert(M > 0, "M must be positive");
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    FullCode* dst = B_dst + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    const FullCode* src = B_src + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    #pragma unroll
    for (int l = 0; l < M; ++l) {
        dst[l] = src[l];
    }

    if (M <= 1 || ksel <= 0) return;
    const std::uint64_t i_global = sample_ids[static_cast<std::size_t>(i)];
    auto state = static_cast<std::uint64_t>(seed);
    state ^= static_cast<std::uint64_t>(outer_iter_1based) * 0x9e3779b97f4a7c15ULL;
    state ^= (i_global + 1ULL) * 0xbf58476d1ce4e5b9ULL;

    for (int t = 0; t < ksel; ++t) {
        const int layer = 1 + static_cast<int>(UniformBelowU64(state, static_cast<std::uint64_t>(M - 1)));
        const int h = h_vec[layer];
        if (h <= 1) continue;
        const int old_code = static_cast<int>(dst[layer]);
        const int raw = static_cast<int>(UniformBelowU64(state, static_cast<std::uint64_t>(h - 1)));
        const int new_code = (raw >= old_code) ? (raw + 1) : raw;
        dst[layer] = static_cast<FullCode>(new_code);
    }
}

template <int M>
__global__ void LinkageCopyAndPerturbSmallCodesSkipLayer0WithSampleIdsFixed(
    const int* __restrict__ h_vec,
    std::uint32_t seed,
    int outer_iter_1based,
    const std::uint64_t* __restrict__ sample_ids,
    int ksel,
    const Code* __restrict__ B_src_small,  // (M-1)×n, sample-major
    Code* __restrict__ B_dst_small,        // (M-1)×n, sample-major
    int n) {
    static_assert(M > 1, "M must be > 1");
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Code* dst = B_dst_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(M - 1);
    const Code* src = B_src_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(M - 1);
    #pragma unroll
    for (int l = 0; l < M - 1; ++l) {
        dst[l] = src[l];
    }

    if (ksel <= 0) return;
    const std::uint64_t i_global = sample_ids[static_cast<std::size_t>(i)];
    auto state = static_cast<std::uint64_t>(seed);
    state ^= static_cast<std::uint64_t>(outer_iter_1based) * 0x9e3779b97f4a7c15ULL;
    state ^= (i_global + 1ULL) * 0xbf58476d1ce4e5b9ULL;

    for (int t = 0; t < ksel; ++t) {
        const int layer = 1 + static_cast<int>(UniformBelowU64(state, static_cast<std::uint64_t>(M - 1)));
        const int h = h_vec[layer];
        if (h <= 1) continue;
        const int j = layer - 1;  // index in B_small
        const int old_code = static_cast<int>(dst[j]);
        const int raw = static_cast<int>(UniformBelowU64(state, static_cast<std::uint64_t>(h - 1)));
        const int new_code = (raw >= old_code) ? (raw + 1) : raw;
        dst[j] = static_cast<Code>(new_code);
    }
}

template <int M>
__global__ void LinkageCopyAndPerturbLargeRootCodesWithSampleIdsFixed(
    const int* __restrict__ h_vec,
    std::uint32_t seed,
    int outer_iter_1based,
    const std::uint64_t* __restrict__ sample_ids,
    int ksel,
    const RootCode* __restrict__ B0_src,   // n
    const Code* __restrict__ B_small_src,  // (M-1)×n, sample-major
    RootCode* __restrict__ B0_dst,         // n
    Code* __restrict__ B_small_dst,        // (M-1)×n, sample-major
    int n) {
    static_assert(M > 1, "M must be > 1");
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    B0_dst[i] = B0_src[i];
    Code* dst = B_small_dst + static_cast<std::size_t>(i) * static_cast<std::size_t>(M - 1);
    const Code* src = B_small_src + static_cast<std::size_t>(i) * static_cast<std::size_t>(M - 1);
    #pragma unroll
    for (int l = 0; l < M - 1; ++l) {
        dst[l] = src[l];
    }

    if (ksel <= 0) return;
    const std::uint64_t i_global = sample_ids[static_cast<std::size_t>(i)];
    auto state = static_cast<std::uint64_t>(seed);
    state ^= static_cast<std::uint64_t>(outer_iter_1based) * 0x9e3779b97f4a7c15ULL;
    state ^= (i_global + 1ULL) * 0xbf58476d1ce4e5b9ULL;

    for (int t = 0; t < ksel; ++t) {
        const int layer = static_cast<int>(UniformBelowU64(state, static_cast<std::uint64_t>(M)));
        const int h = h_vec[layer];
        if (h <= 1) continue;
        if (layer == 0) {
            const int old_code = static_cast<int>(B0_dst[i]);
            const int raw = static_cast<int>(UniformBelowU64(state, static_cast<std::uint64_t>(h - 1)));
            const int new_code = (raw >= old_code) ? (raw + 1) : raw;
            B0_dst[i] = static_cast<RootCode>(new_code);
        } else {
            const int j = layer - 1;
            const int old_code = static_cast<int>(dst[j]);
            const int raw = static_cast<int>(UniformBelowU64(state, static_cast<std::uint64_t>(h - 1)));
            const int new_code = (raw >= old_code) ? (raw + 1) : raw;
            dst[j] = static_cast<Code>(new_code);
        }
    }
}

// Hybrid ConstRoot variant: same as LinkageCopyAndPerturbLargeRootCodesWithSampleIdsFixed but
// layer 0 (root) is frozen — perturbation only affects layers 1..M-1.
// Used in ILS rounds after the first VarRoot round to avoid the expensive root ICM GEMM.
template <int M>
__global__ void LinkageCopyAndPerturbLargeRootCodesSkipRootWithSampleIdsFixed(
    const int* __restrict__ h_vec,
    std::uint32_t seed,
    int outer_iter_1based,
    const std::uint64_t* __restrict__ sample_ids,
    int ksel,
    const RootCode* __restrict__ B0_src,   // n
    const Code* __restrict__ B_small_src,  // (M-1)×n, sample-major
    RootCode* __restrict__ B0_dst,         // n
    Code* __restrict__ B_small_dst,        // (M-1)×n, sample-major
    int n) {
    static_assert(M > 1, "M must be > 1");
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Copy root code unchanged.
    B0_dst[i] = B0_src[i];
    Code* dst = B_small_dst + static_cast<std::size_t>(i) * static_cast<std::size_t>(M - 1);
    const Code* src = B_small_src + static_cast<std::size_t>(i) * static_cast<std::size_t>(M - 1);
    #pragma unroll
    for (int l = 0; l < M - 1; ++l) {
        dst[l] = src[l];
    }

    if (ksel <= 0 || M <= 2) return;
    const std::uint64_t i_global = sample_ids[static_cast<std::size_t>(i)];
    auto state = static_cast<std::uint64_t>(seed);
    state ^= static_cast<std::uint64_t>(outer_iter_1based) * 0x9e3779b97f4a7c15ULL;
    state ^= (i_global + 1ULL) * 0xbf58476d1ce4e5b9ULL;

    for (int t = 0; t < ksel; ++t) {
        // Pick layer from [1, M-1] only (skip root layer 0).
        const int layer = 1 + static_cast<int>(UniformBelowU64(state, static_cast<std::uint64_t>(M - 1)));
        const int h = h_vec[layer];
        if (h <= 1) continue;
        const int j = layer - 1;
        const int old_code = static_cast<int>(dst[j]);
        const int raw = static_cast<int>(UniformBelowU64(state, static_cast<std::uint64_t>(h - 1)));
        const int new_code = (raw >= old_code) ? (raw + 1) : raw;
        dst[j] = static_cast<Code>(new_code);
    }
}

template <int M>
__global__ void LinkageAcceptIfBetterLargeRootFixedRootFixed(
    const Code* __restrict__ B_cand_small,    // (M-1)×n, sample-major
    const float* __restrict__ a_cand,         // M×n, sample-major
    const float* __restrict__ cost_cand,      // n
    Code* __restrict__ B_small,               // (M-1)×n, sample-major
    float* __restrict__ a,                    // M×n, sample-major
    float* __restrict__ cost,                 // n
    int n) {
    static_assert(M > 1, "M must be > 1");
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float c_new = cost_cand[i];
    const float c_old = cost[i];
    if (!(c_new + kEps < c_old)) return;
    cost[i] = c_new;
    {
        Code* dstB = B_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(M - 1);
        const Code* srcB = B_cand_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(M - 1);
        #pragma unroll
        for (int l = 0; l < M - 1; ++l) {
            dstB[l] = srcB[l];
        }
    }
    {
        float* dstA = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
        const float* srcA = a_cand + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
        #pragma unroll
        for (int l = 0; l < M; ++l) {
            dstA[l] = srcA[l];
        }
    }
}

template <int M>
__global__ void LinkageAcceptIfBetterLargeRootVarRootFixed(
    const RootCode* __restrict__ B0_cand,       // n
    const float* __restrict__ xC0_cand,         // n
    const float* __restrict__ norm0_cand,       // n
    const Code* __restrict__ B_small_cand,      // (M-1)×n, sample-major
    const float* __restrict__ a_cand,           // M×n, sample-major
    const float* __restrict__ cost_cand,        // n
    RootCode* __restrict__ B0,                  // n
    float* __restrict__ xC0,                    // n
    float* __restrict__ norm0,                  // n
    Code* __restrict__ B_small,                 // (M-1)×n, sample-major
    float* __restrict__ a,                      // M×n, sample-major
    float* __restrict__ cost,                   // n
    int n) {
    static_assert(M > 1, "M must be > 1");
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float c_new = cost_cand[i];
    const float c_old = cost[i];
    if (!(c_new + kEps < c_old)) return;
    cost[i] = c_new;
    B0[i] = B0_cand[i];
    xC0[i] = xC0_cand[i];
    norm0[i] = norm0_cand[i];
    {
        Code* dstB = B_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(M - 1);
        const Code* srcB = B_small_cand + static_cast<std::size_t>(i) * static_cast<std::size_t>(M - 1);
        #pragma unroll
        for (int l = 0; l < M - 1; ++l) {
            dstB[l] = srcB[l];
        }
    }
    {
        float* dstA = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
        const float* srcA = a_cand + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
        #pragma unroll
        for (int l = 0; l < M; ++l) {
            dstA[l] = srcA[l];
        }
    }
}

// Templatized write-codes kernels: OutT ∈ {FullCode(u8), uint16_t, uint32_t}.
// Merges the former WriteFullCodes* / WriteU32Codes* duplicated pairs.
template <typename OutT, int M>
__global__ void WriteCodesFromFixedRootAndSmallFixed(int forced_root_code,
                                                    const Code* __restrict__ B_small,  // (M-1)×n
                                                    OutT* __restrict__ B_full,         // M×n
                                                    int n) {
    static_assert(M > 1, "M must be > 1");
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Code* src = B_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(M - 1);
    OutT* dst = B_full + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    dst[0] = static_cast<OutT>(forced_root_code);
    #pragma unroll
    for (int l = 1; l < M; ++l) {
        dst[l] = static_cast<OutT>(src[l - 1]);
    }
}

template <typename OutT, int M>
__global__ void WriteCodesFromPerSampleRootAndSmallFixed(const RootCode* __restrict__ root_codes, // n
                                                        const Code* __restrict__ B_small,       // (M-1)×n
                                                        OutT* __restrict__ B_full,              // M×n
                                                        int n) {
    static_assert(M > 1, "M must be > 1");
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Code* src = B_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(M - 1);
    OutT* dst = B_full + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    dst[0] = static_cast<OutT>(root_codes[i]);
    #pragma unroll
    for (int l = 1; l < M; ++l) {
        dst[l] = static_cast<OutT>(src[l - 1]);
    }
}

__global__ void LinkageAcceptIfBetterFull(int m,
                                       const FullCode* __restrict__ B_cand,
                                       const float* __restrict__ a_cand,
                                       const float* __restrict__ cost_cand,
                                       FullCode* __restrict__ B,
                                       float* __restrict__ a,
                                       float* __restrict__ cost,
                                       int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float c_new = cost_cand[i];
    const float c_old = cost[i];
    if (!(c_new + kEps < c_old)) return;
    cost[i] = c_new;
    const FullCode* Bc = B_cand + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    FullCode* Bo = B + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    const float* ac = a_cand + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    float* ao = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    for (int l = 0; l < m; ++l) {
        Bo[l] = Bc[l];
        ao[l] = ac[l];
    }
}

template <int M>
__global__ void LinkageAcceptIfBetterFullFixed(const FullCode* __restrict__ B_cand,
                                            const float* __restrict__ a_cand,
                                            const float* __restrict__ cost_cand,
                                            FullCode* __restrict__ B,
                                            float* __restrict__ a,
                                            float* __restrict__ cost,
                                            int n) {
    static_assert(M > 0, "M must be positive");
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float c_new = cost_cand[i];
    const float c_old = cost[i];
    if (!(c_new + kEps < c_old)) return;
    cost[i] = c_new;
    const FullCode* Bc = B_cand + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    FullCode* Bo = B + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    const float* ac = a_cand + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    float* ao = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    #pragma unroll
    for (int l = 0; l < M; ++l) {
        Bo[l] = Bc[l];
        ao[l] = ac[l];
    }
}

__global__ void ForceLayer0Codes(int m,
                                 const int* __restrict__ forced_root,
                                 FullCode* __restrict__ B,
                                 int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    FullCode* Bi = B + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    Bi[0] = static_cast<FullCode>(forced_root[i]);
}

__device__ __forceinline__ int UpkIndex(int r, int c) {
    return r + (c * (c + 1)) / 2;
}

template <int MaxM>
__device__ bool CholUpperPacked(float* A_upk, int m) {
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

template <int M>
__device__ bool CholUpperPackedFixed(float* A_upk) {
    static_assert(M > 0, "M must be positive");
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

template <int MaxM>
__device__ void SolveCholUpperPacked(const float* U_upk, int m, float* b_inout) {
    float y[MaxM];
    for (int i = 0; i < m; ++i) {
        float sum = b_inout[i];
        for (int k = 0; k < i; ++k) {
            sum -= U_upk[UpkIndex(k, i)] * y[k];
        }
        y[i] = sum / U_upk[UpkIndex(i, i)];
    }
    for (int i = m - 1; i >= 0; --i) {
        float sum = y[i];
        for (int k = i + 1; k < m; ++k) {
            sum -= U_upk[UpkIndex(i, k)] * b_inout[k];
        }
        b_inout[i] = sum / U_upk[UpkIndex(i, i)];
    }
}

template <int M>
__device__ void SolveCholUpperPackedFixed(const float* U_upk, float* b_inout) {
    static_assert(M > 0, "M must be positive");
    float y[M];
    #pragma unroll
    for (int i = 0; i < M; ++i) {
        float sum = b_inout[i];
        #pragma unroll
        for (int k = 0; k < i; ++k) {
            sum -= U_upk[UpkIndex(k, i)] * y[k];
        }
        y[i] = sum / U_upk[UpkIndex(i, i)];
    }
    for (int i = M - 1; i >= 0; --i) {
        float sum = y[i];
        for (int k = i + 1; k < M; ++k) {
            sum -= U_upk[UpkIndex(i, k)] * b_inout[k];
        }
        b_inout[i] = sum / U_upk[UpkIndex(i, i)];
    }
}

__device__ __forceinline__ float GAtDev(const float* G, int rows, int row, int col) {
    return G[static_cast<std::size_t>(col) * static_cast<std::size_t>(rows) + static_cast<std::size_t>(row)];
}

template <int M>
__global__ void TryAcceptRootIcmLargeRootPerSampleFixed(int d,
                                                        int Hs,
                                                        const int* __restrict__ small_offsets, // size M
                                                        const float* __restrict__ C0,          // d×h0 (col-major)
                                                        const float* __restrict__ C_small,     // d×Hs (col-major)
                                                        const float* __restrict__ xC_small,    // Hs×n
                                                        const float* __restrict__ G_small,     // Hs×Hs
                                                        const float* __restrict__ residuals,   // d×n
                                                        const float* __restrict__ X_norm2,     // n
                                                        const RootCode* __restrict__ B0_cand,  // n
                                                        const Code* __restrict__ B_small,      // (M-1)×n
                                                        const std::uint8_t* __restrict__ active, // n
                                                        RootCode* __restrict__ B0,             // n
                                                        float* __restrict__ xC0,               // n
                                                        float* __restrict__ norm0,             // n
                                                        float* __restrict__ a,                 // M×n
                                                        float* __restrict__ cost,              // n
                                                        std::uint8_t* __restrict__ changed,    // n
                                                        int n) {
    static_assert(M > 1, "M must be > 1");
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!active[i]) return;
    const int old_code = static_cast<int>(B0[i]);
    const int new_code = static_cast<int>(B0_cand[i]);
    if (new_code == old_code) return;

    const float* r_i = residuals + static_cast<std::size_t>(i) * static_cast<std::size_t>(d);
    const float* c0 = C0 + static_cast<std::size_t>(new_code) * static_cast<std::size_t>(d);

    float norm0_new = 0.0f;
    float xC0_new = 0.0f;
    for (int rr = 0; rr < d; ++rr) {
        const float cv = c0[rr];
        norm0_new += cv * cv;
        xC0_new += r_i[rr] * cv;
    }

    const Code* B_col = B_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(M - 1);
    const float* xC_col = xC_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(Hs);
    float* a_col = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);

    int codes[M];
    codes[0] = 0;
    #pragma unroll
    for (int l = 1; l < M; ++l) {
        codes[l] = static_cast<int>(B_col[l - 1]);
    }

    float A0[(M * (M + 1)) / 2];
    float A[(M * (M + 1)) / 2];
    float b0[M];
    float bvec[M];

    b0[0] = xC0_new;
    bvec[0] = xC0_new;
    A0[UpkIndex(0, 0)] = norm0_new;

    #pragma unroll
    for (int j = 1; j < M; ++j) {
        const int flat_j = small_offsets[j] + codes[j];
        const float bx = xC_col[flat_j];
        b0[j] = bx;
        bvec[j] = bx;
        const float* cs = C_small + static_cast<std::size_t>(flat_j) * static_cast<std::size_t>(d);
        float g0 = 0.0f;
        for (int rr = 0; rr < d; ++rr) {
            g0 += cs[rr] * c0[rr];
        }
        A0[UpkIndex(0, j)] = g0;
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
            const float g = (k <= l) ? A0[UpkIndex(k, l)] : A0[UpkIndex(l, k)];
            term2 += alpha_l * bvec[k] * g;
        }
    }
    const float new_cost = X_norm2[i] - 2.0f * term1 + term2;
    if (new_cost + kEps < cost[i]) {
        B0[i] = static_cast<RootCode>(new_code);
        xC0[i] = xC0_new;
        norm0[i] = norm0_new;
        #pragma unroll
        for (int l = 0; l < M; ++l) {
            a_col[l] = bvec[l];
        }
        cost[i] = new_cost;
        changed[i] = 1;
    }
}

template <int M>
__global__ void GreedyInitAbsSmallFixedRootFixed(int Hs,
                                                const int* __restrict__ small_offsets,  // size M
                                                const int* __restrict__ h_vec,          // size M
                                                const float* __restrict__ invnorm_small,  // Hs
                                                const float* __restrict__ G_small,        // Hs×Hs
                                                float* __restrict__ rC_small,             // Hs×n (col-major), updated in-place
                                                Code* __restrict__ B_small,               // (M-1)×n (sample-major)
                                                int n) {
    static_assert(M > 1, "M must be > 1");
    const int i = blockIdx.x;
    if (i >= n) return;
    const int tid = threadIdx.x;

    float* rC_col = rC_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(Hs);
    Code* B_col = B_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(M - 1);

    if (tid < (M - 1)) {
        B_col[tid] = 0;
    }

    __shared__ int avail_mask_sh;
    __shared__ int best_flat_sh;
    __shared__ float best_alpha_sh;
    __shared__ float best_score_sh[256];
    __shared__ int best_flat_thread_sh[256];

    if (tid == 0) {
        // Only layers 1..M-1 are available.
        avail_mask_sh = ((1 << M) - 1) & ~1;
    }
    __syncthreads();

    // Select one code per small layer.
    for (int t = 1; t < M; ++t) {
        float best_score = -INFINITY;
        int best_flat = 0;

        const int avail_mask = avail_mask_sh;
        for (int flat = tid; flat < Hs; flat += blockDim.x) {
            int layer = 1;
            // small_offsets[1] == 0 by construction.
            #pragma unroll
            for (int l = 1; l < M; ++l) {
                const int start = small_offsets[l];
                const int stop = start + h_vec[l];
                if (flat >= start && flat < stop) {
                    layer = l;
                    break;
                }
            }
            if (((avail_mask >> layer) & 1) == 0) {
                continue;
            }
            const float adj = rC_col[flat] * invnorm_small[flat];
            const float score = fabsf(adj);
            if (score > best_score || (score == best_score && flat < best_flat)) {
                best_score = score;
                best_flat = flat;
            }
        }

        best_score_sh[tid] = best_score;
        best_flat_thread_sh[tid] = best_flat;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                const float other = best_score_sh[tid + offset];
                const int other_flat = best_flat_thread_sh[tid + offset];
                const int cur_flat = best_flat_thread_sh[tid];
                if (other > best_score_sh[tid] ||
                    (other == best_score_sh[tid] && other_flat < cur_flat)) {
                    best_score_sh[tid] = other;
                    best_flat_thread_sh[tid] = other_flat;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            best_flat_sh = best_flat_thread_sh[0];
            int layer = 1;
            #pragma unroll
            for (int l = 1; l < M; ++l) {
                const int start = small_offsets[l];
                const int stop = start + h_vec[l];
                if (best_flat_sh >= start && best_flat_sh < stop) {
                    layer = l;
                    break;
                }
            }
            const float invn = invnorm_small[best_flat_sh];
            const float adj = rC_col[best_flat_sh] * invn;
            best_alpha_sh = adj * invn;
            const int code = best_flat_sh - small_offsets[layer];
            B_col[layer - 1] = static_cast<Code>(code);
            avail_mask_sh &= ~(1 << layer);
        }
        __syncthreads();

        const float alpha = best_alpha_sh;
        const int best_flat2 = best_flat_sh;
        for (int q = tid; q < Hs; q += blockDim.x) {
            rC_col[q] -= alpha * G_small[static_cast<std::size_t>(best_flat2) * static_cast<std::size_t>(Hs) +
                                         static_cast<std::size_t>(q)];
        }
        __syncthreads();
    }
}

template <int M>
__global__ void SolveCostInitLargeRootFixedRootFixed(int Hs,
                                                     const int* __restrict__ small_offsets,  // size M
                                                     float norm0_root,
                                                     const float* __restrict__ xC_small,     // Hs×n
                                                     const float* __restrict__ G_small,      // Hs×Hs
                                                     const float* __restrict__ g0s_root,     // Hs
                                                     const float* __restrict__ xC0,          // n
                                                     const float* __restrict__ X_norm2,      // n
                                                     const Code* __restrict__ B_small,       // (M-1)×n
                                                     float* __restrict__ a,                  // M×n
                                                     float* __restrict__ out_cost,           // n
                                                     std::uint8_t* __restrict__ active,      // n
                                                     std::uint8_t* __restrict__ changed,     // n
                                                     int n) {
    static_assert(M > 1, "M must be > 1");
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    active[i] = 1;
    changed[i] = 0;

    float* a_col = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    const Code* B_col = B_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(M - 1);
    const float* xC_col = xC_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(Hs);

    // Build packed Gram A0 and RHS b0 in the same way as the full-precomp kernels, to minimize
    // acceptance/tie-break drift (important for exact-match regression tests).
    float A0[(M * (M + 1)) / 2];
    float A[(M * (M + 1)) / 2];
    float b0[M];
    float bvec[M];

    b0[0] = xC0[i];
    bvec[0] = xC0[i];
    A0[UpkIndex(0, 0)] = norm0_root;
    #pragma unroll
    for (int j = 1; j < M; ++j) {
        const int flat_j = small_offsets[j] + static_cast<int>(B_col[j - 1]);
        const float bx = xC_col[flat_j];
        b0[j] = bx;
        bvec[j] = bx;
        A0[UpkIndex(0, j)] = g0s_root[flat_j];
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

    // Cost: match accumulation order from SolveAllAndCostsFullFixed (l-major, k-minor), using packed A0.
    float term1 = 0.0f;
    float term2 = 0.0f;
    #pragma unroll
    for (int l = 0; l < M; ++l) {
        const float alpha_l = bvec[l];
        term1 += alpha_l * b0[l];
        #pragma unroll
        for (int k = 0; k < M; ++k) {
            const float g = (k <= l) ? A0[UpkIndex(k, l)] : A0[UpkIndex(l, k)];
            term2 += alpha_l * bvec[k] * g;
        }
    }
    out_cost[i] = X_norm2[i] - 2.0f * term1 + term2;
}

template <int M>
__global__ void SolveCostInitLargeRootFixedRootPerSampleFixed(int Hs,
                                                              const int* __restrict__ small_offsets,  // size M
                                                              const float* __restrict__ norm0,        // n
                                                              const float* __restrict__ xC_small,     // Hs×n
                                                              const float* __restrict__ G_small,      // Hs×Hs
                                                              const float* __restrict__ g0s,          // Hs×n
                                                              const float* __restrict__ xC0,          // n
                                                              const float* __restrict__ X_norm2,      // n
                                                              const Code* __restrict__ B_small,       // (M-1)×n
                                                              float* __restrict__ a,                  // M×n
                                                              float* __restrict__ out_cost,           // n
                                                              std::uint8_t* __restrict__ active,      // n
                                                              std::uint8_t* __restrict__ changed,     // n
                                                              int n) {
    static_assert(M > 1, "M must be > 1");
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    active[i] = 1;
    changed[i] = 0;

    const float norm0_i = norm0[i];
    const float* g0s_col = g0s + static_cast<std::size_t>(i) * static_cast<std::size_t>(Hs);

    float* a_col = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    const Code* B_col = B_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(M - 1);
    const float* xC_col = xC_small + static_cast<std::size_t>(i) * static_cast<std::size_t>(Hs);

    float A0[(M * (M + 1)) / 2];
    float A[(M * (M + 1)) / 2];
    float b0[M];
    float bvec[M];

    b0[0] = xC0[i];
    bvec[0] = xC0[i];
    A0[UpkIndex(0, 0)] = norm0_i;
    #pragma unroll
    for (int j = 1; j < M; ++j) {
        const int flat_j = small_offsets[j] + static_cast<int>(B_col[j - 1]);
        const float bx = xC_col[flat_j];
        b0[j] = bx;
        bvec[j] = bx;
        A0[UpkIndex(0, j)] = g0s_col[flat_j];
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
            const float g = (k <= l) ? A0[UpkIndex(k, l)] : A0[UpkIndex(l, k)];
            term2 += alpha_l * bvec[k] * g;
        }
    }
    out_cost[i] = X_norm2[i] - 2.0f * term1 + term2;
}

template <bool UseAbs, int M>
__global__ void IcmLayerLargeRootFixedRootBlockFixed(int Hs,
                                                     const int* __restrict__ small_offsets,  // size M
                                                     const int* __restrict__ h_vec,          // size M
                                                     const float* __restrict__ invnorm_small,  // Hs
                                                     float norm0_root,
                                                     const float* __restrict__ xC_small,   // Hs×n
                                                     const float* __restrict__ G_small,    // Hs×Hs
                                                     const float* __restrict__ g0s_root,   // Hs
                                                     const float* __restrict__ xC0,        // n
                                                     const float* __restrict__ X_norm2,    // n
                                                     const std::uint8_t* __restrict__ active,    // n
                                                     std::uint8_t* __restrict__ changed,   // n
                                                     int jlayer,
                                                     Code* __restrict__ B_small,           // (M-1)×n
                                                     float* __restrict__ a,                // M×n
                                                     float* __restrict__ cost,             // n
                                                     int n) {
    static_assert(M > 1, "M must be > 1");
    const int i = blockIdx.x;
    if (i >= n) return;
    if (!active[i]) return;
    // Host ICM order only launches non-root layers for the large-root path.
    // if (jlayer <= 0 || jlayer >= M) return;

    const int tid = threadIdx.x;
    const int hj = h_vec[jlayer];
    if (hj <= 1) return;
    const int startf = small_offsets[jlayer];

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
        B_sh[0] = 0;
    }
    __syncthreads();

    const int old_code = static_cast<int>(B_col[jlayer - 1]);
    float best_score = -INFINITY;
    int best_code = old_code;

    // Scan all codes in this layer in parallel (supports hj > blockDim.x).
    for (int cand = tid; cand < hj; cand += blockDim.x) {
        const int row = startf + cand;
        float tmp = xC_col[row];
        tmp -= a_sh[0] * g0s_root[row];
        #pragma unroll
        for (int l = 1; l < M; ++l) {
            if (l == jlayer) continue;
            const float alpha = a_sh[l];
            const int flat_l = small_offsets[l] + static_cast<int>(B_sh[l]);
            tmp -= alpha * GAtDev(G_small, Hs, row, flat_l);
        }
        const float val = tmp * invnorm_small[row];
        const float score = UseAbs ? fabsf(val) : val;
        if (score > best_score || (score == best_score && cand < best_code)) {
            best_score = score;
            best_code = cand;
        }
    }

    __shared__ float best_score_sh[256];
    __shared__ int best_code_sh[256];
    best_score_sh[tid] = best_score;
    best_code_sh[tid] = best_code;
    __syncthreads();

    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) {
            const float other = best_score_sh[tid + off];
            const int other_code = best_code_sh[tid + off];
            const int cur_code = best_code_sh[tid];
            if (other > best_score_sh[tid] ||
                (other == best_score_sh[tid] && other_code < cur_code)) {
                best_score_sh[tid] = other;
                best_code_sh[tid] = other_code;
            }
        }
        __syncthreads();
    }
    best_code = best_code_sh[0];

    if (tid != 0) return;
    if (best_code == old_code) return;

    int codes[M];
    codes[0] = 0;
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
    bvec[0] = xC0[i];
    A0[UpkIndex(0, 0)] = norm0_root;
    #pragma unroll
    for (int j = 1; j < M; ++j) {
        const int flat_j = small_offsets[j] + codes[j];
        const float bx = xC_col[flat_j];
        b0[j] = bx;
        bvec[j] = bx;
        A0[UpkIndex(0, j)] = g0s_root[flat_j];
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
            const float g =
                (k <= l) ? A0[UpkIndex(k, l)] : A0[UpkIndex(l, k)];
            term2 += alpha_l * bvec[k] * g;
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

template <bool UseAbs, int M>
__global__ void IcmLayerLargeRootFixedRootPerSampleBlockFixed(int Hs,
                                                              const int* __restrict__ small_offsets,  // size M
                                                              const int* __restrict__ h_vec,          // size M
                                                              const float* __restrict__ invnorm_small,  // Hs
                                                              const float* __restrict__ norm0,         // n
                                                              const float* __restrict__ xC_small,      // Hs×n
                                                              const float* __restrict__ G_small,       // Hs×Hs
                                                              const float* __restrict__ g0s,           // Hs×n
                                                              const float* __restrict__ xC0,           // n
                                                              const float* __restrict__ X_norm2,       // n
                                                              const std::uint8_t* __restrict__ active,       // n
                                                              std::uint8_t* __restrict__ changed,      // n
                                                              int jlayer,
                                                              Code* __restrict__ B_small,              // (M-1)×n
                                                              float* __restrict__ a,                   // M×n
                                                              float* __restrict__ cost,                // n
                                                              int n) {
    static_assert(M > 1, "M must be > 1");
    const int i = blockIdx.x;
    if (i >= n) return;
    if (!active[i]) return;
    // Host ICM order only launches non-root layers for the large-root path.
    // if (jlayer <= 0 || jlayer >= M) return;

    const int tid = threadIdx.x;
    const int hj = h_vec[jlayer];
    if (hj <= 1) return;
    const int startf = small_offsets[jlayer];

    const float norm0_i = norm0[i];
    const float* g0s_col = g0s + static_cast<std::size_t>(i) * static_cast<std::size_t>(Hs);

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
        B_sh[0] = 0;
    }
    __syncthreads();

    const int old_code = static_cast<int>(B_col[jlayer - 1]);
    float best_score = -INFINITY;
    int best_code = old_code;

    for (int cand = tid; cand < hj; cand += blockDim.x) {
        const int row = startf + cand;
        float tmp = xC_col[row];
        tmp -= a_sh[0] * g0s_col[row];
        #pragma unroll
        for (int l = 1; l < M; ++l) {
            if (l == jlayer) continue;
            const float alpha = a_sh[l];
            const int flat_l = small_offsets[l] + static_cast<int>(B_sh[l]);
            tmp -= alpha * GAtDev(G_small, Hs, row, flat_l);
        }
        const float val = tmp * invnorm_small[row];
        const float score = UseAbs ? fabsf(val) : val;
        if (score > best_score || (score == best_score && cand < best_code)) {
            best_score = score;
            best_code = cand;
        }
    }

    __shared__ float best_score_sh[256];
    __shared__ int best_code_sh[256];
    best_score_sh[tid] = best_score;
    best_code_sh[tid] = best_code;
    __syncthreads();

    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) {
            const float other = best_score_sh[tid + off];
            const int other_code = best_code_sh[tid + off];
            const int cur_code = best_code_sh[tid];
            if (other > best_score_sh[tid] ||
                (other == best_score_sh[tid] && other_code < cur_code)) {
                best_score_sh[tid] = other;
                best_code_sh[tid] = other_code;
            }
        }
        __syncthreads();
    }
    best_code = best_code_sh[0];

    if (tid != 0) return;
    if (best_code == old_code) return;

    int codes[M];
    codes[0] = 0;
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
    bvec[0] = xC0[i];
    A0[UpkIndex(0, 0)] = norm0_i;
    #pragma unroll
    for (int j = 1; j < M; ++j) {
        const int flat_j = small_offsets[j] + codes[j];
        const float bx = xC_col[flat_j];
        b0[j] = bx;
        bvec[j] = bx;
        A0[UpkIndex(0, j)] = g0s_col[flat_j];
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
            const float g = (k <= l) ? A0[UpkIndex(k, l)] : A0[UpkIndex(l, k)];
            term2 += alpha_l * bvec[k] * g;
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

__global__ void SolveAllAndCostsFull(int H, int m,
                                    const int* __restrict offsets,
                                    const float* __restrict xC,
                                    const float* __restrict G,
                                    const FullCode* __restrict B,
                                    float* __restrict a,
                                    const float* __restrict X_norm2,
                                    float* __restrict out_cost,
                                    int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const FullCode* B_col = B + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    float* a_col = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    const float* xC_col = xC + static_cast<std::size_t>(i) * static_cast<std::size_t>(H);

    float A0[(kMaxM * (kMaxM + 1)) / 2];
    float A[(kMaxM * (kMaxM + 1)) / 2];
    float b0[kMaxM];
    float bvec[kMaxM];

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
        out_cost[i] = X_norm2[i];
        return;
    }

    SolveCholUpperPacked<kMaxM>(A, m, bvec);
    for (int l = 0; l < m; ++l) {
        a_col[l] = bvec[l];
    }

    // Match `ComputeCostsFull` accumulation order (l-major, k-minor), but use the already-built
    // packed Gram `A0` to avoid extra global reads.
    float term1 = 0.0f;
    float term2 = 0.0f;
    for (int l = 0; l < m; ++l) {
        const float alpha_l = bvec[l];
        term1 += alpha_l * b0[l];
        for (int k = 0; k < m; ++k) {
            const float g =
                (k <= l) ? A0[UpkIndex(k, l)] : A0[UpkIndex(l, k)];
            term2 += alpha_l * bvec[k] * g;
        }
    }
    out_cost[i] = X_norm2[i] - 2.0f * term1 + term2;
}

template <int M>
__global__ void SolveAllAndCostsFullFixed(int H,
                                         const int* __restrict offsets,
                                         const float* __restrict xC,
                                         const float* __restrict G,
                                         const FullCode* __restrict B,
                                         float* __restrict a,
                                         const float* __restrict X_norm2,
                                         float* __restrict out_cost,
                                         int n) {
    static_assert(M > 0, "M must be positive");
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const FullCode* B_col = B + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    float* a_col = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    const float* xC_col = xC + static_cast<std::size_t>(i) * static_cast<std::size_t>(H);

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
            const float g =
                (k <= l) ? A0[UpkIndex(k, l)] : A0[UpkIndex(l, k)];
            term2 += alpha_l * bvec[k] * g;
        }
    }
    out_cost[i] = X_norm2[i] - 2.0f * term1 + term2;
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

__global__ void GreedyInitAbs(int H, int m,
                              const int* __restrict offsets,
                              const int* __restrict flat_layer,
                              const float* __restrict invnorm_flat,
                              const float* __restrict G,     // H×H
                              float* __restrict rC,          // H×n (col-major), updated in-place
                              FullCode* __restrict B,        // m×n (col-major)
                              float* __restrict a,           // m×n (col-major)
                              int n) {
    // One block per sample. Assumes H is moderate (<= few thousand).
    const int i = blockIdx.x;
    if (i >= n) return;
    const int tid = threadIdx.x;

    float* rC_col = rC + static_cast<std::size_t>(i) * static_cast<std::size_t>(H);
    FullCode* B_col = B + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    float* a_col = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);

    if (tid < m) {
        B_col[tid] = 0;
        a_col[tid] = 0.0f;
    }

    __shared__ int avail_mask_sh;
    __shared__ int best_flat_sh;
    __shared__ float best_alpha_sh;
    __shared__ float best_score_sh[256];
    __shared__ int best_flat_thread_sh[256];

    if (tid == 0) {
        avail_mask_sh = (m >= 31) ? -1 : ((1 << m) - 1);
    }
    __syncthreads();

    for (int t = 0; t < m; ++t) {
        float best_score = -INFINITY;
        int best_flat = 0;

        const int avail_mask = avail_mask_sh;
        for (int flat = tid; flat < H; flat += blockDim.x) {
            const int layer = flat_layer[flat];
            if (((avail_mask >> layer) & 1) == 0) {
                continue;
            }
            const float adj = rC_col[flat] * invnorm_flat[flat];
            const float score = fabsf(adj);
            if (score > best_score || (score == best_score && flat < best_flat)) {
                best_score = score;
                best_flat = flat;
            }
        }

        best_score_sh[tid] = best_score;
        best_flat_thread_sh[tid] = best_flat;
        __syncthreads();

        // Reduce to find max score.
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                const float other = best_score_sh[tid + offset];
                const int other_flat = best_flat_thread_sh[tid + offset];
                const int cur_flat = best_flat_thread_sh[tid];
                if (other > best_score_sh[tid] ||
                    (other == best_score_sh[tid] && other_flat < cur_flat)) {
                    best_score_sh[tid] = other;
                    best_flat_thread_sh[tid] = other_flat;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            best_flat_sh = best_flat_thread_sh[0];
            const int layer = flat_layer[best_flat_sh];
            const float invn = invnorm_flat[best_flat_sh];
            const float adj = rC_col[best_flat_sh] * invn;
            best_alpha_sh = adj * invn;
            const int code = best_flat_sh - offsets[layer];
            B_col[layer] = static_cast<FullCode>(code);
            a_col[layer] = best_alpha_sh;
            avail_mask_sh &= ~(1 << layer);
        }
        __syncthreads();

        // Update rC[:,i] -= alpha * G[:,best_flat].
        const float alpha = best_alpha_sh;
        const int best_flat2 = best_flat_sh;
        for (int q = tid; q < H; q += blockDim.x) {
            rC_col[q] -= alpha * G[static_cast<std::size_t>(best_flat2) * static_cast<std::size_t>(H) +
                                   static_cast<std::size_t>(q)];
        }
        __syncthreads();
    }
}

template <int M>
__global__ void GreedyInitAbsFixed(int H,
                                  const int* __restrict offsets,
                                  const int* __restrict flat_layer,
                                  const float* __restrict invnorm_flat,
                                  const float* __restrict G,     // H×H
                                  float* __restrict rC,          // H×n (col-major), updated in-place
                                  FullCode* __restrict B,        // m×n (col-major)
                                  float* __restrict a,           // m×n (col-major)
                                  int n) {
    static_assert(M > 0, "M must be positive");
    const int i = blockIdx.x;
    if (i >= n) return;
    const int tid = threadIdx.x;

    float* rC_col = rC + static_cast<std::size_t>(i) * static_cast<std::size_t>(H);
    FullCode* B_col = B + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    float* a_col = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);

    if (tid < M) {
        B_col[tid] = 0;
        a_col[tid] = 0.0f;
    }

    __shared__ int avail_mask_sh;
    __shared__ int best_flat_sh;
    __shared__ float best_alpha_sh;
    __shared__ float best_score_sh[256];
    __shared__ int best_flat_thread_sh[256];

    if (tid == 0) {
        avail_mask_sh = (M >= 31) ? -1 : ((1 << M) - 1);
    }
    __syncthreads();

    for (int t = 0; t < M; ++t) {
        float best_score = -INFINITY;
        int best_flat = 0;

        const int avail_mask = avail_mask_sh;
        for (int flat = tid; flat < H; flat += blockDim.x) {
            const int layer = flat_layer[flat];
            if (((avail_mask >> layer) & 1) == 0) {
                continue;
            }
            const float adj = rC_col[flat] * invnorm_flat[flat];
            const float score = fabsf(adj);
            if (score > best_score || (score == best_score && flat < best_flat)) {
                best_score = score;
                best_flat = flat;
            }
        }

        best_score_sh[tid] = best_score;
        best_flat_thread_sh[tid] = best_flat;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                const float other = best_score_sh[tid + offset];
                const int other_flat = best_flat_thread_sh[tid + offset];
                const int cur_flat = best_flat_thread_sh[tid];
                if (other > best_score_sh[tid] ||
                    (other == best_score_sh[tid] && other_flat < cur_flat)) {
                    best_score_sh[tid] = other;
                    best_flat_thread_sh[tid] = other_flat;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            best_flat_sh = best_flat_thread_sh[0];
            const int layer = flat_layer[best_flat_sh];
            const float invn = invnorm_flat[best_flat_sh];
            const float adj = rC_col[best_flat_sh] * invn;
            best_alpha_sh = adj * invn;
            const int code = best_flat_sh - offsets[layer];
            B_col[layer] = static_cast<FullCode>(code);
            a_col[layer] = best_alpha_sh;
            avail_mask_sh &= ~(1 << layer);
        }
        __syncthreads();

        const float alpha = best_alpha_sh;
        const int best_flat2 = best_flat_sh;
        for (int q = tid; q < H; q += blockDim.x) {
            rC_col[q] -= alpha * G[static_cast<std::size_t>(best_flat2) * static_cast<std::size_t>(H) +
                                   static_cast<std::size_t>(q)];
        }
        __syncthreads();
    }
}

__global__ void GreedyInitAbsForcedRoot(int H, int m,
                                       const int* __restrict offsets,
                                       const int* __restrict flat_layer,
                                       const float* __restrict invnorm_flat,
                                       const float* __restrict G,        // H×H
                                       const int* __restrict forced_root, // n
                                       float* __restrict rC,              // H×n
                                       FullCode* __restrict B,            // m×n
                                       float* __restrict a,               // m×n
                                       int n) {
    const int i = blockIdx.x;
    if (i >= n) return;
    const int tid = threadIdx.x;

    float* rC_col = rC + static_cast<std::size_t>(i) * static_cast<std::size_t>(H);
    FullCode* B_col = B + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
    float* a_col = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);

    if (tid < m) {
        B_col[tid] = 0;
        a_col[tid] = 0.0f;
    }

    __shared__ int avail_mask_sh;
    __shared__ int best_flat_sh;
    __shared__ float best_alpha_sh;
    __shared__ float best_score_sh[256];
    __shared__ int best_flat_thread_sh[256];

    if (tid == 0) {
        avail_mask_sh = (m >= 31) ? -1 : ((1 << m) - 1);
    }
    __syncthreads();

    // Forced root selection (layer 0).
    const int root_code = forced_root[i];
    const int root_flat = offsets[0] + root_code;
    const float invn_root = invnorm_flat[root_flat];
    const float adj_root = rC_col[root_flat] * invn_root;
    const float alpha_root = adj_root * invn_root;
    if (tid == 0) {
        B_col[0] = static_cast<FullCode>(root_code);
        a_col[0] = alpha_root;
        avail_mask_sh &= ~1;  // clear layer 0
    }
    __syncthreads();

    for (int q = tid; q < H; q += blockDim.x) {
        rC_col[q] -= alpha_root * G[static_cast<std::size_t>(root_flat) * static_cast<std::size_t>(H) +
                                    static_cast<std::size_t>(q)];
    }
    __syncthreads();

    // Remaining greedy picks for layers 1..m-1.
    for (int t = 1; t < m; ++t) {
        float best_score = -INFINITY;
        int best_flat = 0;

        const int avail_mask = avail_mask_sh;
        for (int flat = tid; flat < H; flat += blockDim.x) {
            const int layer = flat_layer[flat];
            if (((avail_mask >> layer) & 1) == 0) {
                continue;
            }
            const float adj = rC_col[flat] * invnorm_flat[flat];
            const float score = fabsf(adj);
            if (score > best_score || (score == best_score && flat < best_flat)) {
                best_score = score;
                best_flat = flat;
            }
        }

        best_score_sh[tid] = best_score;
        best_flat_thread_sh[tid] = best_flat;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                const float other = best_score_sh[tid + offset];
                const int other_flat = best_flat_thread_sh[tid + offset];
                const int cur_flat = best_flat_thread_sh[tid];
                if (other > best_score_sh[tid] ||
                    (other == best_score_sh[tid] && other_flat < cur_flat)) {
                    best_score_sh[tid] = other;
                    best_flat_thread_sh[tid] = other_flat;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            best_flat_sh = best_flat_thread_sh[0];
            const int layer = flat_layer[best_flat_sh];
            const float invn = invnorm_flat[best_flat_sh];
            const float adj = rC_col[best_flat_sh] * invn;
            best_alpha_sh = adj * invn;
            const int code = best_flat_sh - offsets[layer];
            B_col[layer] = static_cast<FullCode>(code);
            a_col[layer] = best_alpha_sh;
            avail_mask_sh &= ~(1 << layer);
        }
        __syncthreads();

        const float alpha = best_alpha_sh;
        const int best_flat2 = best_flat_sh;
        for (int q = tid; q < H; q += blockDim.x) {
            rC_col[q] -= alpha * G[static_cast<std::size_t>(best_flat2) * static_cast<std::size_t>(H) +
                                   static_cast<std::size_t>(q)];
        }
        __syncthreads();
    }
}

template <int M>
__global__ void GreedyInitAbsForcedRootFixed(int H,
                                            const int* __restrict offsets,
                                            const int* __restrict flat_layer,
                                            const float* __restrict invnorm_flat,
                                            const float* __restrict G,        // H×H
                                            const int* __restrict forced_root, // n
                                            float* __restrict rC,              // H×n
                                            FullCode* __restrict B,            // m×n
                                            float* __restrict a,               // m×n
                                            int n) {
    static_assert(M > 0, "M must be positive");
    const int i = blockIdx.x;
    if (i >= n) return;
    const int tid = threadIdx.x;

    float* rC_col = rC + static_cast<std::size_t>(i) * static_cast<std::size_t>(H);
    FullCode* B_col = B + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);
    float* a_col = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(M);

    if (tid < M) {
        B_col[tid] = 0;
        a_col[tid] = 0.0f;
    }

    __shared__ int avail_mask_sh;
    __shared__ int best_flat_sh;
    __shared__ float best_alpha_sh;
    __shared__ float best_score_sh[256];
    __shared__ int best_flat_thread_sh[256];

    if (tid == 0) {
        avail_mask_sh = (M >= 31) ? -1 : ((1 << M) - 1);
    }
    __syncthreads();

    const int root_code = forced_root[i];
    const int root_flat = offsets[0] + root_code;
    const float invn_root = invnorm_flat[root_flat];
    const float adj_root = rC_col[root_flat] * invn_root;
    const float alpha_root = adj_root * invn_root;
    if (tid == 0) {
        B_col[0] = static_cast<FullCode>(root_code);
        a_col[0] = alpha_root;
        avail_mask_sh &= ~1;
    }
    __syncthreads();

    for (int q = tid; q < H; q += blockDim.x) {
        rC_col[q] -= alpha_root * G[static_cast<std::size_t>(root_flat) * static_cast<std::size_t>(H) +
                                    static_cast<std::size_t>(q)];
    }
    __syncthreads();

    for (int t = 1; t < M; ++t) {
        float best_score = -INFINITY;
        int best_flat = 0;

        const int avail_mask = avail_mask_sh;
        for (int flat = tid; flat < H; flat += blockDim.x) {
            const int layer = flat_layer[flat];
            if (((avail_mask >> layer) & 1) == 0) {
                continue;
            }
            const float adj = rC_col[flat] * invnorm_flat[flat];
            const float score = fabsf(adj);
            if (score > best_score || (score == best_score && flat < best_flat)) {
                best_score = score;
                best_flat = flat;
            }
        }

        best_score_sh[tid] = best_score;
        best_flat_thread_sh[tid] = best_flat;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                const float other = best_score_sh[tid + offset];
                const int other_flat = best_flat_thread_sh[tid + offset];
                const int cur_flat = best_flat_thread_sh[tid];
                if (other > best_score_sh[tid] ||
                    (other == best_score_sh[tid] && other_flat < cur_flat)) {
                    best_score_sh[tid] = other;
                    best_flat_thread_sh[tid] = other_flat;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            best_flat_sh = best_flat_thread_sh[0];
            const int layer = flat_layer[best_flat_sh];
            const float invn = invnorm_flat[best_flat_sh];
            const float adj = rC_col[best_flat_sh] * invn;
            best_alpha_sh = adj * invn;
            const int code = best_flat_sh - offsets[layer];
            B_col[layer] = static_cast<FullCode>(code);
            a_col[layer] = best_alpha_sh;
            avail_mask_sh &= ~(1 << layer);
        }
        __syncthreads();

        const float alpha = best_alpha_sh;
        const int best_flat2 = best_flat_sh;
        for (int q = tid; q < H; q += blockDim.x) {
            rC_col[q] -= alpha * G[static_cast<std::size_t>(best_flat2) * static_cast<std::size_t>(H) +
                                   static_cast<std::size_t>(q)];
        }
        __syncthreads();
    }
}

template <bool UseAbs>
__global__ void IcmLayerFull(int H, int m,
                             const int* __restrict offsets,
                             const int* __restrict h_vec,
                             const float* __restrict invnorm_flat,
                             const float* __restrict xC,   // H×n
                             const float* __restrict G,    // H×H
                             const float* __restrict X_norm2,
                             const std::uint8_t* __restrict active,
                             std::uint8_t* __restrict changed,
                             int jlayer,
                             FullCode* __restrict B,       // m×n
                             float* __restrict a,          // m×n
                             float* __restrict cost,
                             int n) {
    const int i = blockIdx.x;
    if (i >= n) return;
    if (!active[i]) return;
    // Host ICM order is generated from the valid layer range [0, m).
    // if (jlayer < 0 || jlayer >= m) return;

    const int tid = threadIdx.x;
    const int hj = h_vec[jlayer];
    if (hj <= 1) return;
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

    float best_score = -INFINITY;
    int best_code = old_code;

    for (int code = tid; code < hj; code += blockDim.x) {
        const int row = startf + code;
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
        const float score = UseAbs ? fabsf(val) : val;
        if (score > best_score || (score == best_score && code < best_code)) {
            best_score = score;
            best_code = code;
        }
    }

    // Warp-level reduction to avoid O(logN) full-block sync steps.
    // Tie-breaker: smaller code if scores equal.
    constexpr unsigned kFullMask = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) {
        const float other_score = __shfl_down_sync(kFullMask, best_score, off);
        const int other_code = __shfl_down_sync(kFullMask, best_code, off);
        if (other_score > best_score || (other_score == best_score && other_code < best_code)) {
            best_score = other_score;
            best_code = other_code;
        }
    }

    __shared__ float warp_score_sh[8];
    __shared__ int warp_code_sh[8];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) {
        warp_score_sh[warp] = best_score;
        warp_code_sh[warp] = best_code;
    }
    __syncthreads();

    float block_best_score = -INFINITY;
    int block_best_code = old_code;
    if (warp == 0) {
        if (lane < 8) {
            block_best_score = warp_score_sh[lane];
            block_best_code = warp_code_sh[lane];
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float other_score = __shfl_down_sync(kFullMask, block_best_score, off);
            const int other_code = __shfl_down_sync(kFullMask, block_best_code, off);
            if (other_score > block_best_score ||
                (other_score == block_best_score && other_code < block_best_code)) {
                block_best_score = other_score;
                block_best_code = other_code;
            }
        }
        if (lane == 0) {
            warp_code_sh[0] = block_best_code;
        }
    }
    __syncthreads();

    const int new_code = warp_code_sh[0];
    if (tid != 0) {
        return;
    }
    if (new_code == old_code) return;

    // Backup coefficients.
    float a_old[kMaxM];
    for (int l = 0; l < m; ++l) {
        a_old[l] = a_col[l];
    }

    // Tentatively apply code update.
    B_col[jlayer] = static_cast<FullCode>(new_code);

    // Solve LS for this sample (Cholesky retry).
        float A0[(kMaxM * (kMaxM + 1)) / 2];
        float A[(kMaxM * (kMaxM + 1)) / 2];
        float b0[kMaxM];
        float bvec[kMaxM];

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

    // Reuse A0 (G submatrix, packed upper) + b0 (xC at selected flats) to avoid
    // extra global reads for cost evaluation.
    //
    // IMPORTANT: Match CPU `SampleCost` accumulation order exactly:
    //   for l: term1 += a_l * xC(flat_l)
    //          for k: term2 += a_l * a_k * G(flat_l, flat_k)
    // This is mathematically equivalent to the symmetric form, but produces closer
    // floating behavior and reduces acceptance/tie-break drift.
    float term1 = 0.0f;
    float term2 = 0.0f;
    for (int l = 0; l < m; ++l) {
        const float alpha_l = bvec[l];
        term1 += alpha_l * b0[l];
        for (int k = 0; k < m; ++k) {
            const float g =
                (k <= l) ? A0[UpkIndex(k, l)] : A0[UpkIndex(l, k)];
            term2 += alpha_l * bvec[k] * g;
        }
    }
    const float new_cost = X_norm2[i] - 2.0f * term1 + term2;
    if (new_cost + kEps < cost[i]) {
        cost[i] = new_cost;
        changed[i] = 1;
    } else {
        B_col[jlayer] = static_cast<FullCode>(old_code);
        for (int l = 0; l < m; ++l) {
            a_col[l] = a_old[l];
        }
    }
}

template <bool UseAbs, int M>
__global__ void IcmLayerFullFixed(int H,
                                 const int* __restrict offsets,
                                 const int* __restrict h_vec,
                                 const float* __restrict invnorm_flat,
                                 const float* __restrict xC,
                                 const float* __restrict G,
                                 const float* __restrict X_norm2,
                                 const std::uint8_t* __restrict active,
                                 std::uint8_t* __restrict changed,
                                 int jlayer,
                                 FullCode* __restrict B,
                                 float* __restrict a,
                                 float* __restrict cost,
                                 int n) {
    static_assert(M > 0, "M must be positive");
    const int i = blockIdx.x;
    if (i >= n) return;
    if (!active[i]) return;
    // Host ICM order is generated from the valid layer range [0, M).
    // if (jlayer < 0 || jlayer >= M) return;

    const int tid = threadIdx.x;
    const int hj = h_vec[jlayer];
    if (hj <= 1) return;
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

    float best_score = -INFINITY;
    int best_code = old_code;

    for (int code = tid; code < hj; code += blockDim.x) {
        const int row = startf + code;
        float tmp = xC_col[row];
        #pragma unroll
        for (int l = 0; l < M; ++l) {
            if (l == jlayer) continue;
            const float alpha = a_sh[l];
            const int flat_l = offsets[l] + static_cast<int>(B_sh[l]);
            tmp -= alpha * GAtDev(G, H, row, flat_l);
        }
        const float val = tmp * invnorm_flat[row];
        const float score = UseAbs ? fabsf(val) : val;
        if (score > best_score || (score == best_score && code < best_code)) {
            best_score = score;
            best_code = code;
        }
    }

    constexpr unsigned kFullMask = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) {
        const float other_score = __shfl_down_sync(kFullMask, best_score, off);
        const int other_code = __shfl_down_sync(kFullMask, best_code, off);
        if (other_score > best_score || (other_score == best_score && other_code < best_code)) {
            best_score = other_score;
            best_code = other_code;
        }
    }

    __shared__ float warp_score_sh[8];
    __shared__ int warp_code_sh[8];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) {
        warp_score_sh[warp] = best_score;
        warp_code_sh[warp] = best_code;
    }
    __syncthreads();

    float block_best_score = -INFINITY;
    int block_best_code = old_code;
    if (warp == 0) {
        if (lane < 8) {
            block_best_score = warp_score_sh[lane];
            block_best_code = warp_code_sh[lane];
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float other_score = __shfl_down_sync(kFullMask, block_best_score, off);
            const int other_code = __shfl_down_sync(kFullMask, block_best_code, off);
            if (other_score > block_best_score ||
                (other_score == block_best_score && other_code < block_best_code)) {
                block_best_score = other_score;
                block_best_code = other_code;
            }
        }
        if (lane == 0) {
            warp_code_sh[0] = block_best_code;
        }
    }
    __syncthreads();

    const int new_code = warp_code_sh[0];
    if (tid != 0) {
        return;
    }
    if (new_code == old_code) return;

    float a_old[M];
    #pragma unroll
    for (int l = 0; l < M; ++l) {
        a_old[l] = a_col[l];
    }

    B_col[jlayer] = static_cast<FullCode>(new_code);

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
            const float g =
                (k <= l) ? A0[UpkIndex(k, l)] : A0[UpkIndex(l, k)];
            term2 += alpha_l * bvec[k] * g;
        }
    }
    const float new_cost = X_norm2[i] - 2.0f * term1 + term2;
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

inline void LaunchGreedyInitAbs(cudaStream_t stream,
                               int H,
                               int m,
                               const int* offsets,
                               const int* flat_layer,
                               const float* invnorm_flat,
                               const float* G,
                               float* rC,
                               FullCode* B,
                               float* a,
                               int n) {
    const int threads = 256;
    switch (m) {
        case 5:
            GreedyInitAbsFixed<5><<<n, threads, 0, stream>>>(H, offsets, flat_layer, invnorm_flat, G, rC, B, a, n);
            break;
        case 10:
            GreedyInitAbsFixed<10><<<n, threads, 0, stream>>>(H, offsets, flat_layer, invnorm_flat, G, rC, B, a, n);
            break;
        case 15:
            GreedyInitAbsFixed<15><<<n, threads, 0, stream>>>(H, offsets, flat_layer, invnorm_flat, G, rC, B, a, n);
            break;
        default:
            GreedyInitAbs<<<n, threads, 0, stream>>>(H, m, offsets, flat_layer, invnorm_flat, G, rC, B, a, n);
            break;
    }
}

inline void LaunchGreedyInitAbsForcedRoot(cudaStream_t stream,
                                         int H,
                                         int m,
                                         const int* offsets,
                                         const int* flat_layer,
                                         const float* invnorm_flat,
                                         const float* G,
                                         const int* forced_root,
                                         float* rC,
                                         FullCode* B,
                                         float* a,
                                         int n) {
    const int threads = 256;
    switch (m) {
        case 5:
            GreedyInitAbsForcedRootFixed<5>
                <<<n, threads, 0, stream>>>(H, offsets, flat_layer, invnorm_flat, G, forced_root, rC, B, a, n);
            break;
        case 10:
            GreedyInitAbsForcedRootFixed<10>
                <<<n, threads, 0, stream>>>(H, offsets, flat_layer, invnorm_flat, G, forced_root, rC, B, a, n);
            break;
        case 15:
            GreedyInitAbsForcedRootFixed<15>
                <<<n, threads, 0, stream>>>(H, offsets, flat_layer, invnorm_flat, G, forced_root, rC, B, a, n);
            break;
        default:
            GreedyInitAbsForcedRoot<<<n, threads, 0, stream>>>(
                H, m, offsets, flat_layer, invnorm_flat, G, forced_root, rC, B, a, n);
            break;
    }
}

inline void LaunchSolveAllAndCostsFull(cudaStream_t stream,
                                      int H,
                                      int m,
                                      const int* offsets,
                                      const float* xC,
                                      const float* G,
                                      const FullCode* B,
                                      float* a,
                                      const float* X_norm2,
                                      float* out_cost,
                                      int n,
                                      int blocks,
                                      int threads) {
    switch (m) {
        case 5:
            SolveAllAndCostsFullFixed<5><<<blocks, threads, 0, stream>>>(
                H, offsets, xC, G, B, a, X_norm2, out_cost, n);
            break;
        case 10:
            SolveAllAndCostsFullFixed<10><<<blocks, threads, 0, stream>>>(
                H, offsets, xC, G, B, a, X_norm2, out_cost, n);
            break;
        case 15:
            SolveAllAndCostsFullFixed<15><<<blocks, threads, 0, stream>>>(
                H, offsets, xC, G, B, a, X_norm2, out_cost, n);
            break;
        default:
            SolveAllAndCostsFull<<<blocks, threads, 0, stream>>>(
                H, m, offsets, xC, G, B, a, X_norm2, out_cost, n);
            break;
    }
}

inline void LaunchIcmLayerFullAbs(cudaStream_t stream,
                                 int H,
                                 int m,
                                 const int* offsets,
                                 const int* h_vec,
                                 const float* invnorm_flat,
                                 const float* xC,
                                 const float* G,
                                 const float* X_norm2,
                                 std::uint8_t* active,
                                 std::uint8_t* changed,
                                 int jlayer,
                                 FullCode* B,
                                 float* a,
                                 float* cost,
                                 int n) {
    dim3 grid(n);
    dim3 block(256);
    switch (m) {
        case 5:
            IcmLayerFullFixed<true, 5><<<grid, block, 0, stream>>>(
                H, offsets, h_vec, invnorm_flat, xC, G, X_norm2, active, changed, jlayer, B, a, cost, n);
            break;
        case 10:
            IcmLayerFullFixed<true, 10><<<grid, block, 0, stream>>>(
                H, offsets, h_vec, invnorm_flat, xC, G, X_norm2, active, changed, jlayer, B, a, cost, n);
            break;
        case 15:
            IcmLayerFullFixed<true, 15><<<grid, block, 0, stream>>>(
                H, offsets, h_vec, invnorm_flat, xC, G, X_norm2, active, changed, jlayer, B, a, cost, n);
            break;
        default:
            IcmLayerFull<true><<<grid, block, 0, stream>>>(
                H, m, offsets, h_vec, invnorm_flat, xC, G, X_norm2, active, changed, jlayer, B, a, cost, n);
            break;
    }
}

inline void LaunchLinkageCopyAndPerturbFullCodes(cudaStream_t stream,
                                              int m,
                                              const int* h_vec,
                                              std::uint32_t seed,
                                              int outer_iter_1based,
                                              std::uint64_t sample_id_offset,
                                              int ksel,
                                              const FullCode* B_src,
                                              FullCode* B_dst,
                                              int n,
                                              int blocks,
                                              int threads) {
    switch (m) {
        case 5:
            LinkageCopyAndPerturbFullCodesFixed<5><<<blocks, threads, 0, stream>>>(
                h_vec, seed, outer_iter_1based, sample_id_offset, ksel, B_src, B_dst, n);
            break;
        case 10:
            LinkageCopyAndPerturbFullCodesFixed<10><<<blocks, threads, 0, stream>>>(
                h_vec, seed, outer_iter_1based, sample_id_offset, ksel, B_src, B_dst, n);
            break;
        case 15:
            LinkageCopyAndPerturbFullCodesFixed<15><<<blocks, threads, 0, stream>>>(
                h_vec, seed, outer_iter_1based, sample_id_offset, ksel, B_src, B_dst, n);
            break;
        default:
            LinkageCopyAndPerturbFullCodes<<<blocks, threads, 0, stream>>>(
                m, h_vec, seed, outer_iter_1based, sample_id_offset, ksel, B_src, B_dst, n);
            break;
    }
}

inline void LaunchLinkageCopyAndPerturbFullCodesWithSampleIds(cudaStream_t stream,
                                                           int m,
                                                           const int* h_vec,
                                                           std::uint32_t seed,
                                                           int outer_iter_1based,
                                                           const std::uint64_t* sample_ids,
                                                           int ksel,
                                                           const FullCode* B_src,
                                                           FullCode* B_dst,
                                                           int n,
                                                           int blocks,
                                                           int threads) {
    switch (m) {
        case 5:
            LinkageCopyAndPerturbFullCodesWithSampleIdsFixed<5><<<blocks, threads, 0, stream>>>(
                h_vec, seed, outer_iter_1based, sample_ids, ksel, B_src, B_dst, n);
            break;
        case 10:
            LinkageCopyAndPerturbFullCodesWithSampleIdsFixed<10><<<blocks, threads, 0, stream>>>(
                h_vec, seed, outer_iter_1based, sample_ids, ksel, B_src, B_dst, n);
            break;
        case 15:
            LinkageCopyAndPerturbFullCodesWithSampleIdsFixed<15><<<blocks, threads, 0, stream>>>(
                h_vec, seed, outer_iter_1based, sample_ids, ksel, B_src, B_dst, n);
            break;
        default:
            LinkageCopyAndPerturbFullCodesWithSampleIds<<<blocks, threads, 0, stream>>>(
                m, h_vec, seed, outer_iter_1based, sample_ids, ksel, B_src, B_dst, n);
            break;
    }
}

inline void LaunchLinkageCopyAndPerturbFullCodesSkipLayer0(cudaStream_t stream,
                                                        int m,
                                                        const int* h_vec,
                                                        std::uint32_t seed,
                                                        int outer_iter_1based,
                                                        std::uint64_t sample_id_offset,
                                                        int ksel,
                                                        const FullCode* B_src,
                                                        FullCode* B_dst,
                                                        int n,
                                                        int blocks,
                                                        int threads) {
    switch (m) {
        case 5:
            LinkageCopyAndPerturbFullCodesSkipLayer0Fixed<5><<<blocks, threads, 0, stream>>>(
                h_vec, seed, outer_iter_1based, sample_id_offset, ksel, B_src, B_dst, n);
            break;
        case 10:
            LinkageCopyAndPerturbFullCodesSkipLayer0Fixed<10><<<blocks, threads, 0, stream>>>(
                h_vec, seed, outer_iter_1based, sample_id_offset, ksel, B_src, B_dst, n);
            break;
        case 15:
            LinkageCopyAndPerturbFullCodesSkipLayer0Fixed<15><<<blocks, threads, 0, stream>>>(
                h_vec, seed, outer_iter_1based, sample_id_offset, ksel, B_src, B_dst, n);
            break;
        default:
            LinkageCopyAndPerturbFullCodesSkipLayer0<<<blocks, threads, 0, stream>>>(
                m, h_vec, seed, outer_iter_1based, sample_id_offset, ksel, B_src, B_dst, n);
            break;
    }
}

inline void LaunchLinkageCopyAndPerturbFullCodesSkipLayer0WithSampleIds(
    cudaStream_t stream,
    int m,
    const int* h_vec,
    std::uint32_t seed,
    int outer_iter_1based,
    const std::uint64_t* sample_ids,
    int ksel,
    const FullCode* B_src,
    FullCode* B_dst,
    int n,
    int blocks,
    int threads) {
    switch (m) {
        case 5:
            LinkageCopyAndPerturbFullCodesSkipLayer0WithSampleIdsFixed<5><<<blocks, threads, 0, stream>>>(
                h_vec, seed, outer_iter_1based, sample_ids, ksel, B_src, B_dst, n);
            break;
        case 10:
            LinkageCopyAndPerturbFullCodesSkipLayer0WithSampleIdsFixed<10><<<blocks, threads, 0, stream>>>(
                h_vec, seed, outer_iter_1based, sample_ids, ksel, B_src, B_dst, n);
            break;
        case 15:
            LinkageCopyAndPerturbFullCodesSkipLayer0WithSampleIdsFixed<15><<<blocks, threads, 0, stream>>>(
                h_vec, seed, outer_iter_1based, sample_ids, ksel, B_src, B_dst, n);
            break;
        default:
            LinkageCopyAndPerturbFullCodesSkipLayer0WithSampleIds<<<blocks, threads, 0, stream>>>(
                m, h_vec, seed, outer_iter_1based, sample_ids, ksel, B_src, B_dst, n);
            break;
    }
}

inline void LaunchLinkageAcceptIfBetterFull(cudaStream_t stream,
                                         int m,
                                         const FullCode* B_cand,
                                         const float* a_cand,
                                         const float* cost_cand,
                                         FullCode* B,
                                         float* a,
                                         float* cost,
                                         int n,
                                         int blocks,
                                         int threads) {
    switch (m) {
        case 5:
            LinkageAcceptIfBetterFullFixed<5><<<blocks, threads, 0, stream>>>(B_cand, a_cand, cost_cand, B, a, cost, n);
            break;
        case 10:
            LinkageAcceptIfBetterFullFixed<10><<<blocks, threads, 0, stream>>>(B_cand, a_cand, cost_cand, B, a, cost, n);
            break;
        case 15:
            LinkageAcceptIfBetterFullFixed<15><<<blocks, threads, 0, stream>>>(B_cand, a_cand, cost_cand, B, a, cost, n);
            break;
        default:
            LinkageAcceptIfBetterFull<<<blocks, threads, 0, stream>>>(m, B_cand, a_cand, cost_cand, B, a, cost, n);
            break;
    }
}

inline void LaunchGreedyInitAbsSmallFixedRoot(cudaStream_t stream,
                                             int m,
                                             int Hs,
                                             const int* small_offsets,
                                             const int* h_vec,
                                             const float* invnorm_small,
                                             const float* G_small,
                                             float* rC_small,
                                             Code* B_small,
                                             int n) {
    const int threads = 256;
    const int blocks = n;
    DispatchM2To20(m, "LaunchGreedyInitAbsSmallFixedRoot", [&](auto m_tag) {
        constexpr int M = decltype(m_tag)::value;
        GreedyInitAbsSmallFixedRootFixed<M><<<blocks, threads, 0, stream>>>(
            Hs, small_offsets, h_vec, invnorm_small, G_small, rC_small, B_small, n);
    });
}

inline void LaunchSolveCostInitLargeRootFixedRoot(cudaStream_t stream,
                                                  int m,
                                                  int Hs,
                                                  const int* small_offsets,
                                                  float norm0_root,
                                                  const float* xC_small,
                                                  const float* G_small,
                                                  const float* g0s_root,
                                                  const float* xC0,
                                                  const float* X_norm2,
                                                  const Code* B_small,
                                                  float* a,
                                                  float* out_cost,
                                                  std::uint8_t* active,
                                                  std::uint8_t* changed,
                                                  int n,
                                                  int blocks,
                                                  int threads) {
    DispatchM2To20(m, "LaunchSolveCostInitLargeRootFixedRoot", [&](auto m_tag) {
        constexpr int M = decltype(m_tag)::value;
        SolveCostInitLargeRootFixedRootFixed<M><<<blocks, threads, 0, stream>>>(
            Hs, small_offsets, norm0_root, xC_small, G_small, g0s_root, xC0, X_norm2,
            B_small, a, out_cost, active, changed, n);
    });
}

inline void LaunchSolveCostInitLargeRootFixedRootPerSample(cudaStream_t stream,
                                                          int m,
                                                          int Hs,
                                                          const int* small_offsets,
                                                          const float* norm0,
                                                          const float* xC_small,
                                                          const float* G_small,
                                                          const float* g0s,
                                                          const float* xC0,
                                                          const float* X_norm2,
                                                          const Code* B_small,
                                                          float* a,
                                                          float* out_cost,
                                                          std::uint8_t* active,
                                                          std::uint8_t* changed,
                                                          int n,
                                                          int blocks,
                                                          int threads) {
    DispatchM2To20(m, "LaunchSolveCostInitLargeRootFixedRootPerSample", [&](auto m_tag) {
        constexpr int M = decltype(m_tag)::value;
        SolveCostInitLargeRootFixedRootPerSampleFixed<M><<<blocks, threads, 0, stream>>>(
            Hs, small_offsets, norm0, xC_small, G_small, g0s, xC0, X_norm2,
            B_small, a, out_cost, active, changed, n);
    });
}

inline void LaunchIcmLayerLargeRootFixedRootBlockAbs(cudaStream_t stream,
                                                    int m,
                                                    int Hs,
                                                    const int* small_offsets,
                                                    const int* h_vec,
                                                    const float* invnorm_small,
                                                    float norm0_root,
                                                    const float* xC_small,
                                                    const float* G_small,
                                                    const float* g0s_root,
                                                    const float* xC0,
                                                    const float* X_norm2,
                                                    const std::uint8_t* active,
                                                    std::uint8_t* changed,
                                                    int jlayer,
                                                    Code* B_small,
                                                    float* a,
                                                    float* cost,
                                                    int n) {
    dim3 grid(n);
    dim3 block(256);
    DispatchM2To20(m, "LaunchIcmLayerLargeRootFixedRootBlockAbs", [&](auto m_tag) {
        constexpr int M = decltype(m_tag)::value;
        IcmLayerLargeRootFixedRootBlockFixed<true, M><<<grid, block, 0, stream>>>(
            Hs, small_offsets, h_vec, invnorm_small, norm0_root, xC_small, G_small,
            g0s_root, xC0, X_norm2, active, changed, jlayer, B_small, a, cost, n);
    });
}

inline void LaunchIcmLayerLargeRootFixedRootPerSampleBlockAbs(cudaStream_t stream,
                                                             int m,
                                                             int Hs,
                                                             const int* small_offsets,
                                                             const int* h_vec,
                                                             const float* invnorm_small,
                                                             const float* norm0,
                                                             const float* xC_small,
                                                             const float* G_small,
                                                             const float* g0s,
                                                             const float* xC0,
                                                             const float* X_norm2,
                                                             const std::uint8_t* active,
                                                             std::uint8_t* changed,
                                                             int jlayer,
                                                             Code* B_small,
                                                             float* a,
                                                             float* cost,
                                                             int n) {
    dim3 grid(n);
    dim3 block(256);
    DispatchM2To20(m, "LaunchIcmLayerLargeRootFixedRootPerSampleBlockAbs", [&](auto m_tag) {
        constexpr int M = decltype(m_tag)::value;
        IcmLayerLargeRootFixedRootPerSampleBlockFixed<true, M><<<grid, block, 0, stream>>>(
            Hs, small_offsets, h_vec, invnorm_small, norm0, xC_small, G_small,
            g0s, xC0, X_norm2, active, changed, jlayer, B_small, a, cost, n);
    });
}

inline void LaunchLinkageCopyAndPerturbSmallCodesSkipLayer0WithSampleIds(
    cudaStream_t stream,
    int m,
    const int* h_vec,
    std::uint32_t seed,
    int outer_iter_1based,
    const std::uint64_t* sample_ids,
    int ksel,
    const Code* B_src_small,
    Code* B_dst_small,
    int n,
    int blocks,
    int threads) {
    DispatchM2To20(m, "LaunchLinkageCopyAndPerturbSmallCodesSkipLayer0WithSampleIds", [&](auto m_tag) {
        constexpr int M = decltype(m_tag)::value;
        LinkageCopyAndPerturbSmallCodesSkipLayer0WithSampleIdsFixed<M><<<blocks, threads, 0, stream>>>(
            h_vec, seed, outer_iter_1based, sample_ids, ksel, B_src_small, B_dst_small, n);
    });
}

inline void LaunchFindBestRootCandFromXc0FullAndSmall(cudaStream_t stream,
                                                     int m,
                                                     int h0,
                                                     const float* xC0_full,
                                                     const float* inv_root,
                                                     const float* G0s_T,
                                                     const int* small_offsets,
                                                     const Code* B_small,
                                                     const float* a,
                                                     RootCode* out_code,
                                                     int n,
                                                     int threads) {
    DispatchM2To20(m, "LaunchFindBestRootCandFromXc0FullAndSmall", [&](auto m_tag) {
        constexpr int M = decltype(m_tag)::value;
        FindBestRootCandFromXc0FullAndSmallFixed<M><<<n, threads, 0, stream>>>(
            h0, xC0_full, inv_root, G0s_T, small_offsets, B_small, a, out_code, n);
    });
}

inline void LaunchBuildRe0FromSmallCodesLargeRoot(cudaStream_t stream,
                                                 int m,
                                                 int d,
                                                 const int* small_offsets,
                                                 const float* C_small,
                                                 const Code* B_small,
                                                 const float* a,
                                                 const float* residuals,
                                                 float* re0,
                                                 int n,
                                                 int blocks,
                                                 int threads) {
    DispatchM2To20(m, "LaunchBuildRe0FromSmallCodesLargeRoot", [&](auto m_tag) {
        constexpr int M = decltype(m_tag)::value;
        BuildRe0FromSmallCodesLargeRootFixed<M><<<blocks, threads, 0, stream>>>(
            d, small_offsets, C_small, B_small, a, residuals, re0, n);
    });
}

inline void LaunchTryAcceptRootIcmLargeRootPerSample(cudaStream_t stream,
                                                    int m,
                                                    int d,
                                                    int Hs,
                                                    const int* small_offsets,
                                                    const float* C0,
                                                    const float* C_small,
                                                    const float* xC_small,
                                                    const float* G_small,
                                                    const float* residuals,
                                                    const float* X_norm2,
                                                    const RootCode* B0_cand,
                                                    const Code* B_small,
                                                    const std::uint8_t* active,
                                                    RootCode* B0,
                                                    float* xC0,
                                                    float* norm0,
                                                    float* a,
                                                    float* cost,
                                                    std::uint8_t* changed,
                                                    int n,
                                                    int blocks,
                                                    int threads) {
    DispatchM2To20(m, "LaunchTryAcceptRootIcmLargeRootPerSample", [&](auto m_tag) {
        constexpr int M = decltype(m_tag)::value;
        TryAcceptRootIcmLargeRootPerSampleFixed<M><<<blocks, threads, 0, stream>>>(
            d, Hs, small_offsets, C0, C_small, xC_small, G_small, residuals, X_norm2,
            B0_cand, B_small, active, B0, xC0, norm0, a, cost, changed, n);
    });
}

inline void LaunchLinkageCopyAndPerturbLargeRootCodesWithSampleIds(
    cudaStream_t stream,
    int m,
    const int* h_vec,
    std::uint32_t seed,
    int outer_iter_1based,
    const std::uint64_t* sample_ids,
    int ksel,
    const RootCode* B0_src,
    const Code* B_small_src,
    RootCode* B0_dst,
    Code* B_small_dst,
    int n,
    int blocks,
    int threads) {
    DispatchM2To20(m, "LaunchLinkageCopyAndPerturbLargeRootCodesWithSampleIds", [&](auto m_tag) {
        constexpr int M = decltype(m_tag)::value;
        LinkageCopyAndPerturbLargeRootCodesWithSampleIdsFixed<M><<<blocks, threads, 0, stream>>>(
            h_vec, seed, outer_iter_1based, sample_ids, ksel,
            B0_src, B_small_src, B0_dst, B_small_dst, n);
    });
}

inline void LaunchLinkageCopyAndPerturbLargeRootCodesSkipRootWithSampleIds(
    cudaStream_t stream,
    int m,
    const int* h_vec,
    std::uint32_t seed,
    int outer_iter_1based,
    const std::uint64_t* sample_ids,
    int ksel,
    const RootCode* B0_src,
    const Code* B_small_src,
    RootCode* B0_dst,
    Code* B_small_dst,
    int n,
    int blocks,
    int threads) {
    DispatchM2To20(m, "LaunchLinkageCopyAndPerturbLargeRootCodesSkipRootWithSampleIds", [&](auto m_tag) {
        constexpr int M = decltype(m_tag)::value;
        LinkageCopyAndPerturbLargeRootCodesSkipRootWithSampleIdsFixed<M><<<blocks, threads, 0, stream>>>(
            h_vec, seed, outer_iter_1based, sample_ids, ksel,
            B0_src, B_small_src, B0_dst, B_small_dst, n);
    });
}

inline void LaunchLinkageAcceptIfBetterLargeRootFixedRoot(cudaStream_t stream,
                                                         int m,
                                                         const Code* B_cand_small,
                                                         const float* a_cand,
                                                         const float* cost_cand,
                                                         Code* B_small,
                                                         float* a,
                                                         float* cost,
                                                         int n,
                                                         int blocks,
                                                         int threads) {
    DispatchM2To20(m, "LaunchLinkageAcceptIfBetterLargeRootFixedRoot", [&](auto m_tag) {
        constexpr int M = decltype(m_tag)::value;
        LinkageAcceptIfBetterLargeRootFixedRootFixed<M><<<blocks, threads, 0, stream>>>(
            B_cand_small, a_cand, cost_cand, B_small, a, cost, n);
    });
}

inline void LaunchLinkageAcceptIfBetterLargeRootVarRoot(cudaStream_t stream,
                                                       int m,
                                                       const RootCode* B0_cand,
                                                       const float* xC0_cand,
                                                       const float* norm0_cand,
                                                       const Code* B_small_cand,
                                                       const float* a_cand,
                                                       const float* cost_cand,
                                                       RootCode* B0,
                                                       float* xC0,
                                                       float* norm0,
                                                       Code* B_small,
                                                       float* a,
                                                       float* cost,
                                                       int n,
                                                       int blocks,
                                                       int threads) {
    DispatchM2To20(m, "LaunchLinkageAcceptIfBetterLargeRootVarRoot", [&](auto m_tag) {
        constexpr int M = decltype(m_tag)::value;
        LinkageAcceptIfBetterLargeRootVarRootFixed<M><<<blocks, threads, 0, stream>>>(
            B0_cand, xC0_cand, norm0_cand, B_small_cand, a_cand, cost_cand,
            B0, xC0, norm0, B_small, a, cost, n);
    });
}

template <typename OutCodeT>
inline void LaunchWriteCodesFromFixedRootAndSmall(cudaStream_t stream,
                                                  int m,
                                                  int forced_root_code,
                                                  const Code* B_small,
                                                  OutCodeT* B_full,
                                                  int n,
                                                  int blocks,
                                                  int threads) {
    DispatchM2To20(m, "LaunchWriteCodesFromFixedRootAndSmall", [&](auto m_tag) {
        constexpr int M = decltype(m_tag)::value;
        WriteCodesFromFixedRootAndSmallFixed<OutCodeT, M><<<blocks, threads, 0, stream>>>(
            forced_root_code, B_small, B_full, n);
    });
}

template <typename OutCodeT>
inline void LaunchWriteCodesFromPerSampleRootAndSmall(cudaStream_t stream,
                                                      int m,
                                                      const RootCode* root_codes,
                                                      const Code* B_small,
                                                      OutCodeT* B_full,
                                                      int n,
                                                      int blocks,
                                                      int threads) {
    DispatchM2To20(m, "LaunchWriteCodesFromPerSampleRootAndSmall", [&](auto m_tag) {
        constexpr int M = decltype(m_tag)::value;
        WriteCodesFromPerSampleRootAndSmallFixed<OutCodeT, M><<<blocks, threads, 0, stream>>>(
            root_codes, B_small, B_full, n);
    });
}

}  // namespace

void ClearCudaLinkageEncodeSharedPrecompCaches(int device) {
    // NOTE: This must be called only at a safe point when no linkage-encode CUDA kernels are
    // in-flight, because workspaces keep raw pointers into shared cache entries.
    std::lock_guard<std::mutex> lock_pre(g_shared_precomp_mu);
    std::lock_guard<std::mutex> lock_lr(g_shared_lr_mu);

    auto clear_vec_for_device = [&](auto* vec_ptr) {
        auto& vec = *vec_ptr;
        if (device >= 0) {
            ThrowIf(cudaSetDevice(device), "cudaSetDevice");
            ThrowIf(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            for (auto& p : vec) {
                if (p && p->device == device) {
                    p.reset();
                }
            }
            vec.erase(std::remove(vec.begin(), vec.end(), nullptr), vec.end());
            return;
        }

        // device < 0: clear for all devices present in the cache.
        std::vector<int> devices;
        devices.reserve(vec.size());
        for (const auto& p : vec) {
            if (!p) continue;
            devices.push_back(p->device);
        }
        std::sort(devices.begin(), devices.end());
        devices.erase(std::unique(devices.begin(), devices.end()), devices.end());

        for (int dev : devices) {
            ThrowIf(cudaSetDevice(dev), "cudaSetDevice");
            ThrowIf(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            for (auto& p : vec) {
                if (p && p->device == dev) {
                    p.reset();
                }
            }
        }
        vec.clear();
    };

    clear_vec_for_device(&g_shared_precomp);
    clear_vec_for_device(&g_shared_lr_precomp);
}

	struct LinkageEncodeBatchCudaWorkspace {
	    int m = 0;
	    int H = 0;
	    int d = 0;
	    int n = 0;
	    int n_cap = 0;

    // Large-root (split-root) metadata for forced-root encoding.
    int lr_m = 0;
    int lr_d = 0;
	    int lr_h0 = 0;
	    int lr_Hs = 0;
	    int lr_n = 0;
	    int lr_n_cap = 0;
	    int lr_root_chunk_h = 0;

    template <typename T>
    struct DeviceView {
        const T* ptr = nullptr;
    };

    DeviceView<float> d_C;
    DeviceView<float> d_G;
    DeviceView<float> d_inv;
    DeviceView<int> d_offsets;
    DeviceView<int> d_hvec;
    DeviceView<int> d_flat_layer;
    std::uint64_t precomp_tag = 0;

    // Large-root shared precomp views (C0 + small-only precomp).
    DeviceView<float> lr_d_C0;
    DeviceView<float> lr_d_inv_root;
    DeviceView<float> lr_d_C_small;
    DeviceView<float> lr_d_G0s;
    DeviceView<float> lr_d_G0s_T;
    DeviceView<float> lr_d_G_small;
    DeviceView<float> lr_d_inv_small;
    DeviceView<int> lr_d_small_offsets;
    DeviceView<int> lr_d_hvec;
    std::uint64_t lr_precomp_tag = 0;

    // Per-ctx fixed-root cross vector (g0s_root = C_small^T * C0[:,root]) for the current forced_root_code.
    int lr_forced_root_code = -1;
    std::uint64_t lr_forced_root_tag = 0;
    int lr_norm0_root_code = -1;
    std::uint64_t lr_norm0_root_tag = 0;
    float lr_norm0_root = 0.0f;
    DeviceBuf<float> lr_d_g0s_root;

    DeviceBuf<float> d_xC;
    DeviceBuf<float> d_rC;
    DeviceBuf<std::uint8_t> d_active;
    DeviceBuf<std::uint8_t> d_changed;

    DeviceBuf<FullCode> d_B_cand;
    DeviceBuf<float> d_a_cand;
    DeviceBuf<float> d_cost_cand;

    // Large-root batch buffers.
    DeviceBuf<float> lr_d_xC_small;   // Hs×n
    DeviceBuf<float> lr_d_rC_small;   // Hs×n
    DeviceBuf<float> lr_d_xC0;        // n
    DeviceBuf<Code> lr_d_B_small;     // (m-1)×n
    DeviceBuf<Code> lr_d_B_small_cand;// (m-1)×n
	    // Variable-root batch buffers (used by init-linkage no-forced-root path).
	    DeviceBuf<float> lr_d_xC0_chunk;  // chunk_h×n
	    DeviceBuf<float> lr_d_xC0_full;   // h0×n (optional cache when it fits)
	    DeviceBuf<float> lr_d_g0s;        // Hs×n
	    DeviceBuf<float> lr_d_g0s_cand;   // Hs×n
	    DeviceBuf<float> lr_d_best_score; // n
    DeviceBuf<RootCode> lr_d_B0;      // n
    DeviceBuf<RootCode> lr_d_B0_cand; // n
    DeviceBuf<RootCode> lr_d_B0_tmp;  // n
    DeviceBuf<float> lr_d_norm0;      // n
    DeviceBuf<float> lr_d_xC0_cand;   // n
    DeviceBuf<float> lr_d_norm0_cand; // n
    DeviceBuf<float> lr_d_re0;        // d×n

    bool events_inited = false;
    bool last_profile_valid = false;
    cudaEvent_t ev[6] = {};  // segment boundaries for profiling
    // ILS internal profiling (per call; aggregate over outer loop).
    // ils_ev layout when enabled:
    //   [0]               = loop start
    //   [1 + 3*i + 0]     = after copy/perturb
    //   [1 + 4*i + 1]     = after solve+cost
    //   [1 + 4*i + 2]     = after ICM (if any; otherwise equals solve+cost point)
    //   [1 + 4*i + 3]     = after accept
    std::vector<cudaEvent_t> ils_ev = {};
    int ils_profile_iters = 0;
    bool ils_profile_valid = false;

    void EnsurePrecomp(const CudaCtx& ctx, const Precomp& pre) {
        const std::uint64_t tag = PrecompDeviceTag(pre);
        if (precomp_tag == tag && d_C.ptr && d_G.ptr && d_inv.ptr && d_offsets.ptr && d_hvec.ptr && d_flat_layer.ptr) {
            return;
        }
        m = pre.m;
        d = pre.d;
        H = pre.H;
        SharedPrecompDevice* shared = GetOrCreateSharedPrecomp(ctx, pre);
        precomp_tag = shared->tag;
        d_C.ptr = shared->d_C.ptr;
        d_G.ptr = shared->d_G.ptr;
        d_inv.ptr = shared->d_inv.ptr;
        d_offsets.ptr = shared->d_offsets.ptr;
        d_hvec.ptr = shared->d_hvec.ptr;
        d_flat_layer.ptr = shared->d_flat_layer.ptr;
    }

	    void EnsureLargeRootPrecomp(const CudaCtx& ctx, const CodebookPack& C_root) {
	        const std::uint64_t tag = LargeRootDeviceTag(C_root);
	        if (lr_precomp_tag == tag &&
	            lr_d_C0.ptr && lr_d_inv_root.ptr && lr_d_C_small.ptr && lr_d_G0s.ptr && lr_d_G_small.ptr && lr_d_inv_small.ptr &&
	            lr_d_small_offsets.ptr && lr_d_hvec.ptr) {
	            return;
	        }
	        SharedLargeRootPrecompDevice* shared = GetOrCreateSharedLargeRootPrecomp(ctx, C_root);
	        lr_precomp_tag = shared->tag;
	        lr_m = shared->m;
	        lr_d = shared->d;
	        lr_h0 = shared->h0;
	        lr_Hs = shared->Hs;
        lr_d_C0.ptr = shared->d_C0.ptr;
        lr_d_inv_root.ptr = shared->d_inv_root.ptr;
        lr_d_C_small.ptr = shared->d_C_small.ptr;
        lr_d_G0s.ptr = shared->d_G0s.ptr;
        lr_d_G0s_T.ptr = shared->d_G0s_T.ptr;
        lr_d_G_small.ptr = shared->d_G_small.ptr;
        lr_d_inv_small.ptr = shared->d_inv_small.ptr;
        lr_d_small_offsets.ptr = shared->d_small_offsets.ptr;
        lr_d_hvec.ptr = shared->d_hvec.ptr;
        // Invalidate per-root cache.
        lr_forced_root_code = -1;
        lr_forced_root_tag = 0;
	        lr_norm0_root_code = -1;
	        lr_norm0_root_tag = 0;
	        lr_norm0_root = 0.0f;
	        lr_n_cap = 0;
	    }

	    static int RoundUpPow2Int(int x) {
	        if (x <= 1) return 1;
	        auto v = static_cast<std::uint32_t>(x - 1);
	        v |= v >> 1;
	        v |= v >> 2;
	        v |= v >> 4;
	        v |= v >> 8;
	        v |= v >> 16;
	        return static_cast<int>(v + 1);
	    }

	    static int PickAllocN(int n_in) {
	        // Reduce cudaMalloc/cudaFree churn by rounding `n` up to larger quanta.
	        // This is a pure performance tradeoff (slightly more VRAM for fewer reallocations).
	        if (n_in <= 0) return 0;
	        if (n_in <= 8192) {
	            return RoundUpPow2Int(n_in);
	        }
	        if (n_in <= 65536) {
	            constexpr int k = 8192;
	            return ((n_in + k - 1) / k) * k;
	        }
	        return n_in;
	    }

	    void EnsureBatch(int n_in) {
	        n = n_in;
	        const int want = PickAllocN(n_in);
	        n_cap = std::max(n_cap, want);
	        const int n_alloc = std::max(n, n_cap);
	        d_xC.Ensure(static_cast<std::size_t>(H) * static_cast<std::size_t>(n_alloc));
	        d_rC.Ensure(static_cast<std::size_t>(H) * static_cast<std::size_t>(n_alloc));
	        d_active.Ensure(static_cast<std::size_t>(n_alloc));
	        d_changed.Ensure(static_cast<std::size_t>(n_alloc));
	        d_B_cand.Ensure(static_cast<std::size_t>(m) * static_cast<std::size_t>(n_alloc));
	        d_a_cand.Ensure(static_cast<std::size_t>(m) * static_cast<std::size_t>(n_alloc));
	        d_cost_cand.Ensure(static_cast<std::size_t>(n_alloc));
	    }

		    void EnsureLargeRootBatch(int n_in) {
		        lr_n = n_in;
		        const int want = PickAllocN(n_in);
		        // IMPORTANT: do not update `lr_n_cap` before successful allocation.
		        // The large-root public wrapper may back off on OOM and retry with a smaller `n`.
		        // If we eagerly raise `lr_n_cap` and then fail, subsequent retries would still allocate
		        // at the larger (failed) cap and OOM again, defeating the backoff logic.
		        const int n_alloc = std::max(lr_n, std::max(lr_n_cap, want));
		        lr_d_xC_small.Ensure(static_cast<std::size_t>(lr_Hs) * static_cast<std::size_t>(n_alloc));
		        lr_d_rC_small.Ensure(static_cast<std::size_t>(lr_Hs) * static_cast<std::size_t>(n_alloc));
		        lr_d_xC0.Ensure(static_cast<std::size_t>(n_alloc));
		        lr_d_B_small.Ensure(static_cast<std::size_t>(std::max(0, lr_m - 1)) * static_cast<std::size_t>(n_alloc));
		        lr_d_B_small_cand.Ensure(static_cast<std::size_t>(std::max(0, lr_m - 1)) * static_cast<std::size_t>(n_alloc));
		        d_active.Ensure(static_cast<std::size_t>(n_alloc));
		        d_changed.Ensure(static_cast<std::size_t>(n_alloc));
		        d_a_cand.Ensure(static_cast<std::size_t>(lr_m) * static_cast<std::size_t>(n_alloc));
		        d_cost_cand.Ensure(static_cast<std::size_t>(n_alloc));
		        lr_d_g0s_root.Ensure(static_cast<std::size_t>(lr_Hs));
		        lr_n_cap = std::max(lr_n_cap, want);
		    }

		    void EnsureLargeRootBatchVarRoot(int n_in, int root_chunk_h) {
		        EnsureLargeRootBatch(n_in);
		        lr_root_chunk_h = root_chunk_h;
		        const int n_alloc = std::max(lr_n, lr_n_cap);
		        lr_d_xC0_chunk.Ensure(static_cast<std::size_t>(lr_root_chunk_h) * static_cast<std::size_t>(n_alloc));
	        lr_d_g0s.Ensure(static_cast<std::size_t>(lr_Hs) * static_cast<std::size_t>(n_alloc));
	        lr_d_g0s_cand.Ensure(static_cast<std::size_t>(lr_Hs) * static_cast<std::size_t>(n_alloc));
	        lr_d_best_score.Ensure(static_cast<std::size_t>(n_alloc));
	        lr_d_B0.Ensure(static_cast<std::size_t>(n_alloc));
	        lr_d_B0_cand.Ensure(static_cast<std::size_t>(n_alloc));
	        lr_d_B0_tmp.Ensure(static_cast<std::size_t>(n_alloc));
	        lr_d_norm0.Ensure(static_cast<std::size_t>(n_alloc));
	        lr_d_xC0_cand.Ensure(static_cast<std::size_t>(n_alloc));
		        lr_d_norm0_cand.Ensure(static_cast<std::size_t>(n_alloc));
		        lr_d_re0.Ensure(static_cast<std::size_t>(lr_d) * static_cast<std::size_t>(n_alloc));
		    }

		    // OOM backoff helper: drop all large-root batch buffers so a retry with smaller `n` can truly allocate smaller.
		    // This is only used on error paths; hot-path performance is unaffected.
		    void ReleaseLargeRootBatchBuffers() {
		        lr_n = 0;
		        lr_n_cap = 0;
		        lr_root_chunk_h = 0;
		        lr_d_xC_small.Reset();
		        lr_d_rC_small.Reset();
		        lr_d_xC0.Reset();
		        lr_d_B_small.Reset();
		        lr_d_B_small_cand.Reset();
		        lr_d_xC0_chunk.Reset();
		        lr_d_xC0_full.Reset();
		        lr_d_g0s.Reset();
		        lr_d_g0s_cand.Reset();
		        lr_d_best_score.Reset();
		        lr_d_B0.Reset();
		        lr_d_B0_cand.Reset();
		        lr_d_B0_tmp.Reset();
		        lr_d_norm0.Reset();
		        lr_d_xC0_cand.Reset();
		        lr_d_norm0_cand.Reset();
		        lr_d_re0.Reset();
		        // Keep d_active/d_changed/d_a_cand/d_cost_cand: they are small and shared with other paths.
		    }

    void EnsureEvents() {
        if (events_inited) return;
        for (cudaEvent_t& e : ev) {
            ThrowIf(cudaEventCreateWithFlags(&e, cudaEventDefault), "cudaEventCreate");
        }
        events_inited = true;
    }

    void ResetIlsEvents() {
        for (cudaEvent_t& e : ils_ev) {
            if (e) cudaEventDestroy(e);
            e = nullptr;
        }
        ils_ev.clear();
        ils_profile_iters = 0;
        ils_profile_valid = false;
    }

    void EnsureIlsEvents(int iters) {
        if (iters <= 0) {
            ResetIlsEvents();
            return;
        }
        const auto need = static_cast<std::size_t>(1 + 4 * iters);
        if (ils_ev.size() == need) {
            ils_profile_iters = iters;
            return;
        }
        ResetIlsEvents();
        ils_ev.resize(need);
        for (cudaEvent_t& e : ils_ev) {
            ThrowIf(cudaEventCreateWithFlags(&e, cudaEventDefault), "cudaEventCreate(ils)");
        }
        ils_profile_iters = iters;
    }

    ~LinkageEncodeBatchCudaWorkspace() {
        if (events_inited) {
            for (cudaEvent_t& e : ev) {
                if (e) cudaEventDestroy(e);
                e = nullptr;
            }
            events_inited = false;
        }
        ResetIlsEvents();
    }
};

LinkageEncodeBatchCudaWorkspace* CreateLinkageEncodeBatchCudaWorkspace() {
    return new LinkageEncodeBatchCudaWorkspace();
}

void DestroyLinkageEncodeBatchCudaWorkspace(LinkageEncodeBatchCudaWorkspace* ws) {
    delete ws;
}

bool LinkageEncodeBatchCuda(const CudaCtx& ctx,
                          const Precomp& pre_one,
                          const float* d_residuals,
                          const float* d_res_norm2,
                          int n,
                          int icm_iters,
                          int ils_iters,
                          int perturb_k,
                          std::uint32_t seed,
                          std::uint64_t sample_id_offset,
                          FullCode* d_B_out,
                          float* d_a_out,
                          float* d_cost_out,
                          LinkageEncodeBatchCudaWorkspace* ws,
                          std::string* err) {
    try {
        if (!ws) {
            throw std::runtime_error("LinkageEncodeBatchCuda: workspace is null.");
        }
	        if (n <= 0) {
	            return true;
	        }
		        ThrowIf(cudaSetDevice(ctx.device), "cudaSetDevice");
		        const bool profile = g_profile_enabled.load(std::memory_order_relaxed);
		        if (profile) {
		            // Clear any sticky CUDA error from previous profiling work on this host thread.
		            // This prevents misleading failures when we check cudaGetLastError() after our own launches.
		            (void)cudaGetLastError();
		        }
	        std::int64_t t_ensure0 = 0;
	        std::uint64_t pre_tag0 = 0;
	        std::size_t cap_xc0 = 0, cap_rc0 = 0, cap_B0 = 0, cap_a0 = 0, cap_cost0 = 0;
	        if (profile) {
	            t_ensure0 = NowNs();
	            pre_tag0 = ws->precomp_tag;
	            cap_xc0 = ws->d_xC.cap;
	            cap_rc0 = ws->d_rC.cap;
	            cap_B0 = ws->d_B_cand.cap;
	            cap_a0 = ws->d_a_cand.cap;
	            cap_cost0 = ws->d_cost_cand.cap;
	        }
	        ws->EnsurePrecomp(ctx, pre_one);
	        ws->EnsureBatch(n);
	        if (profile) {
	            const std::int64_t dt = NowNs() - t_ensure0;
	            g_host_calls.fetch_add(1, std::memory_order_relaxed);
	            g_host_ensure_ns.fetch_add(dt, std::memory_order_relaxed);
	            if (ws->precomp_tag != pre_tag0) {
	                g_host_precomp_refresh_calls.fetch_add(1, std::memory_order_relaxed);
	            }
	            const std::size_t Hn = static_cast<std::size_t>(pre_one.H) * static_cast<std::size_t>(n);
	            const std::size_t Mn = static_cast<std::size_t>(pre_one.m) * static_cast<std::size_t>(n);
	            const bool xc_grew = (Hn > cap_xc0);
	            const bool any_grew =
	                xc_grew ||
	                (Hn > cap_rc0) ||
	                (Mn > cap_B0) ||
	                (Mn > cap_a0) ||
	                (static_cast<std::size_t>(n) > cap_cost0);
	            if (any_grew) {
	                g_host_batch_grow_calls.fetch_add(1, std::memory_order_relaxed);
	            }
	            if (xc_grew) {
	                g_host_xc_grow_calls.fetch_add(1, std::memory_order_relaxed);
	            }
	            AtomicMaxU64(&g_host_xc_grow_max_elems, static_cast<std::uint64_t>(ws->d_xC.cap));
	        }

	        const int H = pre_one.H;
	        const int d = pre_one.d;
		        const int m = pre_one.m;
		        if (m <= 0 || m > kMaxM) {
		            throw std::runtime_error("LinkageEncodeBatchCuda: unsupported m.");
		        }

		        if (profile) {
		            ws->EnsureEvents();
		            ws->last_profile_valid = false;
		            ws->ils_profile_valid = false;
		            ThrowIf(cudaEventRecord(ws->ev[0], ctx.stream), "cudaEventRecord(enc0)");
		        }

	        // xC = C_all^T * residuals (H×n).
	        const float alpha = 1.0f;
	        const float beta = 0.0f;
	        ThrowIf(cublasSgemm(ctx.cublas,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            H, n, d,
                            &alpha,
                            ws->d_C.ptr, d,
                            d_residuals, d,
	                            &beta,
	                            ws->d_xC.ptr, H),
	                "cublasSgemm(C^T * re)");
	        if (profile) {
	            ThrowIf(cudaEventRecord(ws->ev[1], ctx.stream), "cudaEventRecord(enc1)");
	        }

	        // rC = xC (copy).
	        {
	            const int total = H * n;
	            const int threads = 256;
            const int blocks = (total + threads - 1) / threads;
            CopyF32<<<blocks, threads, 0, ctx.stream>>>(ws->d_xC.ptr, ws->d_rC.ptr, total);
        }

        // Greedy init (abs) + update rC.
        {
            LaunchGreedyInitAbs(ctx.stream, H, m,
                                ws->d_offsets.ptr,
                                ws->d_flat_layer.ptr,
                                ws->d_inv.ptr,
                                ws->d_G.ptr,
                                ws->d_rC.ptr,
                                d_B_out,
                                d_a_out,
                                n);
	        }
	        if (profile) {
	            ThrowIf(cudaEventRecord(ws->ev[2], ctx.stream), "cudaEventRecord(enc2)");
	        }

	        // LS solve and initial costs.
	        {
	            const int threads = 128;
	            const int blocks = (n + threads - 1) / threads;
                LaunchSolveAllAndCostsFull(ctx.stream, H, m,
                                           ws->d_offsets.ptr,
                                           ws->d_xC.ptr,
                                           ws->d_G.ptr,
                                           d_B_out,
                                           d_a_out,
                                           d_res_norm2,
                                           d_cost_out,
                                           n,
                                           blocks,
                                           threads);
	        }
	        if (profile) {
	            ThrowIf(cudaEventRecord(ws->ev[3], ctx.stream), "cudaEventRecord(enc3)");
	        }

        // CPU alignment: `DynamicIcmEncodingSingleForResidual` shuffles the per-iteration layer order (0..m-1)
        // using a single RNG stream seeded by `seed`.
        const bool do_icm = (icm_iters > 0 && m > 1);
        std::mt19937 icm_rng;
        std::array<int, kMaxM> icm_order{};
        const auto reset_icm_order = [&]() {
            for (int j = 0; j < m; ++j) icm_order[static_cast<std::size_t>(j)] = j;
        };
        if (do_icm) {
            icm_rng = std::mt19937(static_cast<std::mt19937::result_type>(seed));
        }

        if (do_icm) {
            const int threads = 128;
            const int blocks = (n + threads - 1) / threads;
            InitActiveChanged<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_active.ptr, ws->d_changed.ptr);

            reset_icm_order();
            for (int it = 0; it < icm_iters; ++it) {
                std::shuffle(icm_order.begin(), icm_order.begin() + m, icm_rng);
                ClearChanged<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_changed.ptr);
                for (int ord = 0; ord < m; ++ord) {
                    const int jlayer = icm_order[static_cast<std::size_t>(ord)];
                    LaunchIcmLayerFullAbs(ctx.stream, H, m,
                                          ws->d_offsets.ptr,
                                          ws->d_hvec.ptr,
                                          ws->d_inv.ptr,
                                          ws->d_xC.ptr,
                                          ws->d_G.ptr,
                                          d_res_norm2,
                                          ws->d_active.ptr,
                                          ws->d_changed.ptr,
                                          jlayer,
                                          d_B_out,
                                          d_a_out,
                                          d_cost_out,
                                          n);
                }
                UpdateActive<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_active.ptr, ws->d_changed.ptr);
            }
        }
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[4], ctx.stream), "cudaEventRecord(enc4)");
        }

	        const int ksel = std::min(std::max(0, perturb_k), m);
	        if (ils_iters > 0 && ksel > 0 && m > 1) {
	            const int threads = 128;
	            const int blocks = (n + threads - 1) / threads;
	            const int copy_threads = 256;
	            const int copy_blocks = (n + copy_threads - 1) / copy_threads;
	            if (profile) {
	                ws->EnsureIlsEvents(ils_iters);
	                ThrowIf(cudaEventRecord(ws->ils_ev[0], ctx.stream), "cudaEventRecord(ils0)");
	            }
	            for (int outer = 0; outer < ils_iters; ++outer) {
	                LaunchLinkageCopyAndPerturbFullCodes(ctx.stream, m, ws->d_hvec.ptr,
                                                       seed,
                                                       outer + 1,
                                                       sample_id_offset,
                                                       ksel,
                                                       d_B_out,
                                                       ws->d_B_cand.ptr,
                                                       n,
                                                       copy_blocks,
                                                       copy_threads);
	                if (profile) {
	                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 0)], ctx.stream),
	                            "cudaEventRecord(ils_copy)");
	                }

	                LaunchSolveAllAndCostsFull(ctx.stream, H, m,
                                           ws->d_offsets.ptr,
                                           ws->d_xC.ptr,
                                           ws->d_G.ptr,
                                           ws->d_B_cand.ptr,
                                           ws->d_a_cand.ptr,
                                           d_res_norm2,
                                           ws->d_cost_cand.ptr,
                                           n,
                                           blocks,
                                           threads);
	                if (profile) {
	                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 1)], ctx.stream),
	                            "cudaEventRecord(ils_solve_cost)");
	                }

                if (do_icm) {
                    InitActiveChanged<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_active.ptr, ws->d_changed.ptr);
                    reset_icm_order();
                    for (int it = 0; it < icm_iters; ++it) {
                        ClearChanged<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_changed.ptr);
                        std::shuffle(icm_order.begin(), icm_order.begin() + m, icm_rng);
                        for (int ord = 0; ord < m; ++ord) {
                            const int jlayer = icm_order[static_cast<std::size_t>(ord)];
                            LaunchIcmLayerFullAbs(ctx.stream, H, m,
                                                  ws->d_offsets.ptr,
                                                  ws->d_hvec.ptr,
                                                  ws->d_inv.ptr,
                                                  ws->d_xC.ptr,
                                                  ws->d_G.ptr,
                                                  d_res_norm2,
                                                  ws->d_active.ptr,
                                                  ws->d_changed.ptr,
                                                  jlayer,
                                                  ws->d_B_cand.ptr,
                                                  ws->d_a_cand.ptr,
                                                  ws->d_cost_cand.ptr,
                                                  n);
                        }
                        UpdateActive<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_active.ptr, ws->d_changed.ptr);
                    }
                }
                if (profile) {
                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 2)], ctx.stream),
                            "cudaEventRecord(ils_icm)");
                }

	                LaunchLinkageAcceptIfBetterFull(ctx.stream, m,
                                              ws->d_B_cand.ptr,
                                              ws->d_a_cand.ptr,
                                              ws->d_cost_cand.ptr,
                                              d_B_out,
                                              d_a_out,
                                              d_cost_out,
                                              n,
                                              blocks,
                                              threads);
	                if (profile) {
	                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 3)], ctx.stream),
	                            "cudaEventRecord(ils_accept)");
	                }
	            }
	            if (profile) {
	                ws->ils_profile_valid = true;
	            }
	        }
	
	        if (profile) {
	            ThrowIf(cudaEventRecord(ws->ev[5], ctx.stream), "cudaEventRecord(enc5)");
	            ws->last_profile_valid = true;
	        }

	        ThrowIf(cudaGetLastError(), "LinkageEncodeBatchCuda launch");
	        return true;
    } catch (const std::exception& e) {
        if (err) {
            *err = e.what();
        }
        return false;
    }
}

bool LinkageEncodeBatchCudaWithSampleIds(const CudaCtx& ctx,
                                      const Precomp& pre_one,
                                      const float* d_residuals,
                                      const float* d_res_norm2,
                                      const std::uint64_t* d_sample_ids,
                                      int n,
                                      int icm_iters,
                                      int ils_iters,
                                      int perturb_k,
                                      std::uint32_t seed,
                                      FullCode* d_B_out,
                                      float* d_a_out,
                                      float* d_cost_out,
                                      LinkageEncodeBatchCudaWorkspace* ws,
                                      std::string* err) {
    try {
        if (!ws) {
            throw std::runtime_error("LinkageEncodeBatchCudaWithSampleIds: workspace is null.");
        }
        if (n <= 0) {
            return true;
        }
	        if (!d_sample_ids && ils_iters > 0 && perturb_k > 0) {
	            throw std::runtime_error("LinkageEncodeBatchCudaWithSampleIds: d_sample_ids is null.");
	        }
	        ThrowIf(cudaSetDevice(ctx.device), "cudaSetDevice");
	        const bool profile = g_profile_enabled.load(std::memory_order_relaxed);
	        if (profile) {
	            (void)cudaGetLastError();
	        }
	        std::int64_t t_ensure0 = 0;
	        std::uint64_t pre_tag0 = 0;
	        std::size_t cap_xc0 = 0, cap_rc0 = 0, cap_B0 = 0, cap_a0 = 0, cap_cost0 = 0;
	        if (profile) {
	            t_ensure0 = NowNs();
	            pre_tag0 = ws->precomp_tag;
	            cap_xc0 = ws->d_xC.cap;
	            cap_rc0 = ws->d_rC.cap;
	            cap_B0 = ws->d_B_cand.cap;
	            cap_a0 = ws->d_a_cand.cap;
	            cap_cost0 = ws->d_cost_cand.cap;
	        }
	        ws->EnsurePrecomp(ctx, pre_one);
	        ws->EnsureBatch(n);
	        if (profile) {
	            const std::int64_t dt = NowNs() - t_ensure0;
	            g_host_calls.fetch_add(1, std::memory_order_relaxed);
	            g_host_ensure_ns.fetch_add(dt, std::memory_order_relaxed);
	            if (ws->precomp_tag != pre_tag0) {
	                g_host_precomp_refresh_calls.fetch_add(1, std::memory_order_relaxed);
	            }
	            const std::size_t Hn = static_cast<std::size_t>(pre_one.H) * static_cast<std::size_t>(n);
	            const std::size_t Mn = static_cast<std::size_t>(pre_one.m) * static_cast<std::size_t>(n);
	            const bool xc_grew = (Hn > cap_xc0);
	            const bool any_grew =
	                xc_grew ||
	                (Hn > cap_rc0) ||
	                (Mn > cap_B0) ||
	                (Mn > cap_a0) ||
	                (static_cast<std::size_t>(n) > cap_cost0);
	            if (any_grew) {
	                g_host_batch_grow_calls.fetch_add(1, std::memory_order_relaxed);
	            }
	            if (xc_grew) {
	                g_host_xc_grow_calls.fetch_add(1, std::memory_order_relaxed);
	            }
	            AtomicMaxU64(&g_host_xc_grow_max_elems, static_cast<std::uint64_t>(ws->d_xC.cap));
	        }

	        const int H = pre_one.H;
	        const int d = pre_one.d;
	        const int m = pre_one.m;
	        if (m <= 0 || m > kMaxM) {
	            throw std::runtime_error("LinkageEncodeBatchCudaWithSampleIds: unsupported m.");
	        }

	        if (profile) {
	            ws->EnsureEvents();
	            ws->last_profile_valid = false;
	            ws->ils_profile_valid = false;
	            ThrowIf(cudaEventRecord(ws->ev[0], ctx.stream), "cudaEventRecord(enc0)");
	        }

        // xC = C_all^T * residuals (H×n).
        const float alpha = 1.0f;
        const float beta = 0.0f;
        ThrowIf(cublasSgemm(ctx.cublas,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            H, n, d,
                            &alpha,
                            ws->d_C.ptr, d,
                            d_residuals, d,
                            &beta,
                            ws->d_xC.ptr, H),
                "cublasSgemm(C^T * re)");
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[1], ctx.stream), "cudaEventRecord(enc1)");
        }

        // rC = xC (copy).
        {
            const int total = H * n;
            const int threads = 256;
            const int blocks = (total + threads - 1) / threads;
            CopyF32<<<blocks, threads, 0, ctx.stream>>>(ws->d_xC.ptr, ws->d_rC.ptr, total);
        }

        // Greedy init (abs) + update rC.
        {
            LaunchGreedyInitAbs(ctx.stream, H, m,
                                ws->d_offsets.ptr,
                                ws->d_flat_layer.ptr,
                                ws->d_inv.ptr,
                                ws->d_G.ptr,
                                ws->d_rC.ptr,
                                d_B_out,
                                d_a_out,
                                n);
        }
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[2], ctx.stream), "cudaEventRecord(enc2)");
        }

        // LS solve and initial costs.
        {
            const int threads = 128;
            const int blocks = (n + threads - 1) / threads;
            LaunchSolveAllAndCostsFull(ctx.stream, H, m,
                                       ws->d_offsets.ptr,
                                       ws->d_xC.ptr,
                                       ws->d_G.ptr,
                                       d_B_out,
                                       d_a_out,
                                       d_res_norm2,
                                       d_cost_out,
                                       n,
                                       blocks,
                                       threads);
        }
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[3], ctx.stream), "cudaEventRecord(enc3)");
        }

        const bool do_icm = (icm_iters > 0 && m > 1);
        std::mt19937 icm_rng;
        std::array<int, kMaxM> icm_order{};
        const auto reset_icm_order = [&]() {
            for (int j = 0; j < m; ++j) icm_order[static_cast<std::size_t>(j)] = j;
        };
        if (do_icm) {
            icm_rng = std::mt19937(static_cast<std::mt19937::result_type>(seed));
        }

        if (do_icm) {
            const int threads = 128;
            const int blocks = (n + threads - 1) / threads;
            InitActiveChanged<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_active.ptr, ws->d_changed.ptr);

            reset_icm_order();
            for (int it = 0; it < icm_iters; ++it) {
                std::shuffle(icm_order.begin(), icm_order.begin() + m, icm_rng);
                ClearChanged<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_changed.ptr);
                for (int ord = 0; ord < m; ++ord) {
                    const int jlayer = icm_order[static_cast<std::size_t>(ord)];
                    LaunchIcmLayerFullAbs(ctx.stream, H, m,
                                          ws->d_offsets.ptr,
                                          ws->d_hvec.ptr,
                                          ws->d_inv.ptr,
                                          ws->d_xC.ptr,
                                          ws->d_G.ptr,
                                          d_res_norm2,
                                          ws->d_active.ptr,
                                          ws->d_changed.ptr,
                                          jlayer,
                                          d_B_out,
                                          d_a_out,
                                          d_cost_out,
                                          n);
                }
                UpdateActive<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_active.ptr, ws->d_changed.ptr);
            }
        }
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[4], ctx.stream), "cudaEventRecord(enc4)");
        }

        const int ksel = std::min(std::max(0, perturb_k), m);
        if (ils_iters > 0 && ksel > 0 && m > 1) {
            const int threads = 128;
            const int blocks = (n + threads - 1) / threads;
            const int copy_threads = 256;
            const int copy_blocks = (n + copy_threads - 1) / copy_threads;
            if (profile) {
                ws->EnsureIlsEvents(ils_iters);
                ThrowIf(cudaEventRecord(ws->ils_ev[0], ctx.stream), "cudaEventRecord(ils0)");
            }
            for (int outer = 0; outer < ils_iters; ++outer) {
                LaunchLinkageCopyAndPerturbFullCodesWithSampleIds(ctx.stream, m, ws->d_hvec.ptr,
                                                               seed,
                                                               outer + 1,
                                                               d_sample_ids,
                                                               ksel,
                                                               d_B_out,
                                                               ws->d_B_cand.ptr,
                                                               n,
                                                               copy_blocks,
                                                               copy_threads);
                if (profile) {
                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 0)], ctx.stream),
                            "cudaEventRecord(ils_copy)");
                }

                LaunchSolveAllAndCostsFull(ctx.stream, H, m,
                                           ws->d_offsets.ptr,
                                           ws->d_xC.ptr,
                                           ws->d_G.ptr,
                                           ws->d_B_cand.ptr,
                                           ws->d_a_cand.ptr,
                                           d_res_norm2,
                                           ws->d_cost_cand.ptr,
                                           n,
                                           blocks,
                                           threads);
                if (profile) {
                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 1)], ctx.stream),
                            "cudaEventRecord(ils_solve_cost)");
                }

                if (do_icm) {
                    InitActiveChanged<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_active.ptr, ws->d_changed.ptr);
                    reset_icm_order();
                    for (int it = 0; it < icm_iters; ++it) {
                        std::shuffle(icm_order.begin(), icm_order.begin() + m, icm_rng);
                        ClearChanged<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_changed.ptr);
                        for (int ord = 0; ord < m; ++ord) {
                            const int jlayer = icm_order[static_cast<std::size_t>(ord)];
                            LaunchIcmLayerFullAbs(ctx.stream, H, m,
                                                  ws->d_offsets.ptr,
                                                  ws->d_hvec.ptr,
                                                  ws->d_inv.ptr,
                                                  ws->d_xC.ptr,
                                                  ws->d_G.ptr,
                                                  d_res_norm2,
                                                  ws->d_active.ptr,
                                                  ws->d_changed.ptr,
                                                  jlayer,
                                                  ws->d_B_cand.ptr,
                                                  ws->d_a_cand.ptr,
                                                  ws->d_cost_cand.ptr,
                                                  n);
                        }
                        UpdateActive<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_active.ptr, ws->d_changed.ptr);
                    }
                }
                if (profile) {
                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 2)], ctx.stream),
                            "cudaEventRecord(ils_icm)");
                }

                LaunchLinkageAcceptIfBetterFull(ctx.stream, m,
                                              ws->d_B_cand.ptr,
                                              ws->d_a_cand.ptr,
                                              ws->d_cost_cand.ptr,
                                              d_B_out,
                                              d_a_out,
                                              d_cost_out,
                                              n,
                                              blocks,
                                              threads);
                if (profile) {
                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 3)], ctx.stream),
                            "cudaEventRecord(ils_accept)");
                }
            }
            if (profile) {
                ws->ils_profile_valid = true;
            }
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[5], ctx.stream), "cudaEventRecord(enc5)");
            ws->last_profile_valid = true;
        }

        ThrowIf(cudaGetLastError(), "LinkageEncodeBatchCudaWithSampleIds launch");
        return true;
    } catch (const std::exception& e) {
        if (err) {
            *err = e.what();
        }
        return false;
    }
}

bool LinkageEncodeBatchCudaForcedRoot(const CudaCtx& ctx,
                                   const Precomp& pre_one,
                                   const float* d_residuals,
                                   const float* d_res_norm2,
                                   const int* d_forced_root,
                                   int n,
                                   int icm_iters,
                                   int ils_iters,
                                   int perturb_k,
                                   std::uint32_t seed,
                                   std::uint64_t sample_id_offset,
                                   FullCode* d_B_out,
                                   float* d_a_out,
                                   float* d_cost_out,
                                   LinkageEncodeBatchCudaWorkspace* ws,
                                   std::string* err) {
    try {
        if (!ws) {
            throw std::runtime_error("LinkageEncodeBatchCudaForcedRoot: workspace is null.");
        }
        if (n <= 0) {
            return true;
        }
        if (!d_forced_root) {
            throw std::runtime_error("LinkageEncodeBatchCudaForcedRoot: forced_root is null.");
        }
        ThrowIf(cudaSetDevice(ctx.device), "cudaSetDevice");

        ws->EnsurePrecomp(ctx, pre_one);
        ws->EnsureBatch(n);

        const int H = pre_one.H;
        const int d = pre_one.d;
        const int m = pre_one.m;
        if (m <= 0 || m > kMaxM) {
            throw std::runtime_error("LinkageEncodeBatchCudaForcedRoot: unsupported m.");
        }
        if (m <= 1) {
            // Trivial codebook: only forced root.
            // Still compute xC/LS/cost so downstream can use `a` and `cost`.
            // (Greedy init will set only layer0.)
        }

        const bool profile = g_profile_enabled.load(std::memory_order_relaxed);
        if (profile) {
            ws->EnsureEvents();
            ws->last_profile_valid = false;
            ws->ils_profile_valid = false;
            ThrowIf(cudaEventRecord(ws->ev[0], ctx.stream), "cudaEventRecord(enc0)");
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;
        ThrowIf(cublasSgemm(ctx.cublas,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            H, n, d,
                            &alpha,
                            ws->d_C.ptr, d,
                            d_residuals, d,
                            &beta,
                            ws->d_xC.ptr, H),
                "cublasSgemm(C^T * re)");
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[1], ctx.stream), "cudaEventRecord(enc1)");
        }

        // rC = xC
        {
            const int total = H * n;
            const int threads = 256;
            const int blocks = (total + threads - 1) / threads;
            CopyF32<<<blocks, threads, 0, ctx.stream>>>(ws->d_xC.ptr, ws->d_rC.ptr, total);
        }

        // Greedy init with forced root (abs) + update rC.
        {
            LaunchGreedyInitAbsForcedRoot(ctx.stream, H, m,
                                          ws->d_offsets.ptr,
                                          ws->d_flat_layer.ptr,
                                          ws->d_inv.ptr,
                                          ws->d_G.ptr,
                                          d_forced_root,
                                          ws->d_rC.ptr,
                                          d_B_out,
                                          d_a_out,
                                          n);
        }
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[2], ctx.stream), "cudaEventRecord(enc2)");
        }

        // LS solve and initial costs.
        {
            const int threads = 128;
            const int blocks = (n + threads - 1) / threads;
            LaunchSolveAllAndCostsFull(ctx.stream, H, m,
                                       ws->d_offsets.ptr,
                                       ws->d_xC.ptr,
                                       ws->d_G.ptr,
                                       d_B_out,
                                       d_a_out,
                                       d_res_norm2,
                                       d_cost_out,
                                       n,
                                       blocks,
                                       threads);
        }
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[3], ctx.stream), "cudaEventRecord(enc3)");
        }

        // Match CPU `DynamicIcmWithIlsAbsNoNormal`: per-ICM-iter shuffled layer order (0..m-1),
        // while skipping layer0 updates (forced root). This only affects the forced-root path
        // (used for UMAP/bad-cluster virtual roots) and is a semantics/correctness alignment.
        std::mt19937 icm_rng;
        std::array<int, kMaxM> icm_order{};
        const auto reset_icm_order = [&]() {
            for (int j = 0; j < m; ++j) icm_order[static_cast<std::size_t>(j)] = j;
        };
        if (icm_iters > 0 && m > 1) {
            icm_rng = std::mt19937(static_cast<std::mt19937::result_type>(seed));
            reset_icm_order();
            const int threads = 128;
            const int blocks = (n + threads - 1) / threads;
            InitActiveChanged<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_active.ptr, ws->d_changed.ptr);

            for (int it = 0; it < icm_iters; ++it) {
                std::shuffle(icm_order.begin(), icm_order.begin() + m, icm_rng);
                ClearChanged<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_changed.ptr);
                for (int ord = 0; ord < m; ++ord) {
                    const int jlayer = icm_order[static_cast<std::size_t>(ord)];
                    if (jlayer == 0) continue;
                    LaunchIcmLayerFullAbs(ctx.stream, H, m,
                                          ws->d_offsets.ptr,
                                          ws->d_hvec.ptr,
                                          ws->d_inv.ptr,
                                          ws->d_xC.ptr,
                                          ws->d_G.ptr,
                                          d_res_norm2,
                                          ws->d_active.ptr,
                                          ws->d_changed.ptr,
                                          jlayer,
                                          d_B_out,
                                          d_a_out,
                                          d_cost_out,
                                          n);
                }
                UpdateActive<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_active.ptr, ws->d_changed.ptr);
            }
        }
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[4], ctx.stream), "cudaEventRecord(enc4)");
        }

        const int ksel = std::min(std::max(0, perturb_k), m);
        if (ils_iters > 0 && ksel > 0 && m > 1) {
            const int threads = 128;
            const int blocks = (n + threads - 1) / threads;
            const int copy_threads = 256;
            const int copy_blocks = (n + copy_threads - 1) / copy_threads;
            if (profile) {
                ws->EnsureIlsEvents(ils_iters);
                ThrowIf(cudaEventRecord(ws->ils_ev[0], ctx.stream), "cudaEventRecord(ils0)");
            }
            for (int outer = 0; outer < ils_iters; ++outer) {
                LaunchLinkageCopyAndPerturbFullCodesSkipLayer0(ctx.stream, m, ws->d_hvec.ptr,
                                                            seed,
                                                            outer + 1,
                                                            sample_id_offset,
                                                            ksel,
                                                            d_B_out,
                                                            ws->d_B_cand.ptr,
                                                            n,
                                                            copy_blocks,
                                                            copy_threads);
                if (profile) {
                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 0)], ctx.stream),
                            "cudaEventRecord(ils_copy)");
                }

                LaunchSolveAllAndCostsFull(ctx.stream, H, m,
                                           ws->d_offsets.ptr,
                                           ws->d_xC.ptr,
                                           ws->d_G.ptr,
                                           ws->d_B_cand.ptr,
                                           ws->d_a_cand.ptr,
                                           d_res_norm2,
                                           ws->d_cost_cand.ptr,
                                           n,
                                           blocks,
                                           threads);
                if (profile) {
                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 1)], ctx.stream),
                            "cudaEventRecord(ils_solve_cost)");
                }

                if (icm_iters > 0 && m > 1) {
                    // Match CPU `RunIcm`: same RNG stream as the initial ICM (seeded once), but reset
                    // the order to 0..m-1 once per ICM call (i.e., once per ILS outer iteration).
                    reset_icm_order();
                    InitActiveChanged<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_active.ptr, ws->d_changed.ptr);
                    for (int it = 0; it < icm_iters; ++it) {
                        std::shuffle(icm_order.begin(), icm_order.begin() + m, icm_rng);
                        ClearChanged<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_changed.ptr);
                        for (int ord = 0; ord < m; ++ord) {
                            const int jlayer = icm_order[static_cast<std::size_t>(ord)];
                            if (jlayer == 0) continue;
                            LaunchIcmLayerFullAbs(ctx.stream, H, m,
                                                  ws->d_offsets.ptr,
                                                  ws->d_hvec.ptr,
                                                  ws->d_inv.ptr,
                                                  ws->d_xC.ptr,
                                                  ws->d_G.ptr,
                                                  d_res_norm2,
                                                  ws->d_active.ptr,
                                                  ws->d_changed.ptr,
                                                  jlayer,
                                                  ws->d_B_cand.ptr,
                                                  ws->d_a_cand.ptr,
                                                  ws->d_cost_cand.ptr,
                                                  n);
                        }
                        UpdateActive<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_active.ptr, ws->d_changed.ptr);
                    }
                }
                if (profile) {
                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 2)], ctx.stream),
                            "cudaEventRecord(ils_icm)");
                }

                LaunchLinkageAcceptIfBetterFull(ctx.stream, m,
                                              ws->d_B_cand.ptr,
                                              ws->d_a_cand.ptr,
                                              ws->d_cost_cand.ptr,
                                              d_B_out,
                                              d_a_out,
                                              d_cost_out,
                                              n,
                                              blocks,
                                              threads);
                if (profile) {
                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 3)], ctx.stream),
                            "cudaEventRecord(ils_accept)");
                }
            }
            if (profile) {
                ws->ils_profile_valid = true;
            }
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[5], ctx.stream), "cudaEventRecord(enc5)");
            ws->last_profile_valid = true;
        }

        ThrowIf(cudaGetLastError(), "LinkageEncodeBatchCudaForcedRoot launch");
        return true;
    } catch (const std::exception& e) {
        if (err) {
            *err = e.what();
        }
        return false;
    }
}

bool LinkageEncodeBatchCudaForcedRootWithSampleIds(const CudaCtx& ctx,
                                                const Precomp& pre_one,
                                                const float* d_residuals,
                                                const float* d_res_norm2,
                                                const int* d_forced_root,
                                                const std::uint64_t* d_sample_ids,
                                                int n,
                                                int icm_iters,
                                                int ils_iters,
                                                int perturb_k,
                                                std::uint32_t seed,
                                                FullCode* d_B_out,
                                                float* d_a_out,
                                                float* d_cost_out,
                                                LinkageEncodeBatchCudaWorkspace* ws,
                                                std::string* err) {
    try {
        if (!ws) {
            throw std::runtime_error("LinkageEncodeBatchCudaForcedRootWithSampleIds: workspace is null.");
        }
        if (n <= 0) {
            return true;
        }
        if (!d_forced_root) {
            throw std::runtime_error("LinkageEncodeBatchCudaForcedRootWithSampleIds: forced_root is null.");
        }
        if (!d_sample_ids && ils_iters > 0 && perturb_k > 0) {
            throw std::runtime_error("LinkageEncodeBatchCudaForcedRootWithSampleIds: d_sample_ids is null.");
        }
        ThrowIf(cudaSetDevice(ctx.device), "cudaSetDevice");

        ws->EnsurePrecomp(ctx, pre_one);
        ws->EnsureBatch(n);

        const int H = pre_one.H;
        const int d = pre_one.d;
        const int m = pre_one.m;
        if (m <= 0 || m > kMaxM) {
            throw std::runtime_error("LinkageEncodeBatchCudaForcedRootWithSampleIds: unsupported m.");
        }

        const bool profile = g_profile_enabled.load(std::memory_order_relaxed);
        if (profile) {
            ws->EnsureEvents();
            ws->last_profile_valid = false;
            ws->ils_profile_valid = false;
            ThrowIf(cudaEventRecord(ws->ev[0], ctx.stream), "cudaEventRecord(enc0)");
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;
        ThrowIf(cublasSgemm(ctx.cublas,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            H, n, d,
                            &alpha,
                            ws->d_C.ptr, d,
                            d_residuals, d,
                            &beta,
                            ws->d_xC.ptr, H),
                "cublasSgemm(C^T * re)");
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[1], ctx.stream), "cudaEventRecord(enc1)");
        }

        {
            const int total = H * n;
            const int threads = 256;
            const int blocks = (total + threads - 1) / threads;
            CopyF32<<<blocks, threads, 0, ctx.stream>>>(ws->d_xC.ptr, ws->d_rC.ptr, total);
        }

        {
            LaunchGreedyInitAbsForcedRoot(ctx.stream, H, m,
                                          ws->d_offsets.ptr,
                                          ws->d_flat_layer.ptr,
                                          ws->d_inv.ptr,
                                          ws->d_G.ptr,
                                          d_forced_root,
                                          ws->d_rC.ptr,
                                          d_B_out,
                                          d_a_out,
                                          n);
        }
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[2], ctx.stream), "cudaEventRecord(enc2)");
        }

        {
            const int threads = 128;
            const int blocks = (n + threads - 1) / threads;
            LaunchSolveAllAndCostsFull(ctx.stream, H, m,
                                       ws->d_offsets.ptr,
                                       ws->d_xC.ptr,
                                       ws->d_G.ptr,
                                       d_B_out,
                                       d_a_out,
                                       d_res_norm2,
                                       d_cost_out,
                                       n,
                                       blocks,
                                       threads);
        }
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[3], ctx.stream), "cudaEventRecord(enc3)");
        }

        std::mt19937 icm_rng;
        std::array<int, kMaxM> icm_order{};
        const auto reset_icm_order = [&]() {
            for (int j = 0; j < m; ++j) icm_order[static_cast<std::size_t>(j)] = j;
        };
        if (icm_iters > 0 && m > 1) {
            icm_rng = std::mt19937(static_cast<std::mt19937::result_type>(seed));
            reset_icm_order();
            const int threads = 128;
            const int blocks = (n + threads - 1) / threads;
            InitActiveChanged<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_active.ptr, ws->d_changed.ptr);

            for (int it = 0; it < icm_iters; ++it) {
                std::shuffle(icm_order.begin(), icm_order.begin() + m, icm_rng);
                ClearChanged<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_changed.ptr);
                for (int ord = 0; ord < m; ++ord) {
                    const int jlayer = icm_order[static_cast<std::size_t>(ord)];
                    if (jlayer == 0) continue;
                    LaunchIcmLayerFullAbs(ctx.stream, H, m,
                                          ws->d_offsets.ptr,
                                          ws->d_hvec.ptr,
                                          ws->d_inv.ptr,
                                          ws->d_xC.ptr,
                                          ws->d_G.ptr,
                                          d_res_norm2,
                                          ws->d_active.ptr,
                                          ws->d_changed.ptr,
                                          jlayer,
                                          d_B_out,
                                          d_a_out,
                                          d_cost_out,
                                          n);
                }
                UpdateActive<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_active.ptr, ws->d_changed.ptr);
            }
        }
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[4], ctx.stream), "cudaEventRecord(enc4)");
        }

        const int ksel = std::min(std::max(0, perturb_k), m);
        if (ils_iters > 0 && ksel > 0 && m > 1) {
            const int threads = 128;
            const int blocks = (n + threads - 1) / threads;
            const int copy_threads = 256;
            const int copy_blocks = (n + copy_threads - 1) / copy_threads;
            if (profile) {
                ws->EnsureIlsEvents(ils_iters);
                ThrowIf(cudaEventRecord(ws->ils_ev[0], ctx.stream), "cudaEventRecord(ils0)");
            }
            for (int outer = 0; outer < ils_iters; ++outer) {
                LaunchLinkageCopyAndPerturbFullCodesSkipLayer0WithSampleIds(ctx.stream, m, ws->d_hvec.ptr,
                                                                         seed,
                                                                         outer + 1,
                                                                         d_sample_ids,
                                                                         ksel,
                                                                         d_B_out,
                                                                         ws->d_B_cand.ptr,
                                                                         n,
                                                                         copy_blocks,
                                                                         copy_threads);
                if (profile) {
                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 0)], ctx.stream),
                            "cudaEventRecord(ils_copy)");
                }

                LaunchSolveAllAndCostsFull(ctx.stream, H, m,
                                           ws->d_offsets.ptr,
                                           ws->d_xC.ptr,
                                           ws->d_G.ptr,
                                           ws->d_B_cand.ptr,
                                           ws->d_a_cand.ptr,
                                           d_res_norm2,
                                           ws->d_cost_cand.ptr,
                                           n,
                                           blocks,
                                           threads);
                if (profile) {
                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 1)], ctx.stream),
                            "cudaEventRecord(ils_solve_cost)");
                }

                if (icm_iters > 0 && m > 1) {
                    reset_icm_order();
                    InitActiveChanged<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_active.ptr, ws->d_changed.ptr);
                    for (int it = 0; it < icm_iters; ++it) {
                        std::shuffle(icm_order.begin(), icm_order.begin() + m, icm_rng);
                        ClearChanged<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_changed.ptr);
                        for (int ord = 0; ord < m; ++ord) {
                            const int jlayer = icm_order[static_cast<std::size_t>(ord)];
                            if (jlayer == 0) continue;
                            LaunchIcmLayerFullAbs(ctx.stream, H, m,
                                                  ws->d_offsets.ptr,
                                                  ws->d_hvec.ptr,
                                                  ws->d_inv.ptr,
                                                  ws->d_xC.ptr,
                                                  ws->d_G.ptr,
                                                  d_res_norm2,
                                                  ws->d_active.ptr,
                                                  ws->d_changed.ptr,
                                                  jlayer,
                                                  ws->d_B_cand.ptr,
                                                  ws->d_a_cand.ptr,
                                                  ws->d_cost_cand.ptr,
                                                  n);
                        }
                        UpdateActive<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_active.ptr, ws->d_changed.ptr);
                    }
                }
                if (profile) {
                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 2)], ctx.stream),
                            "cudaEventRecord(ils_icm)");
                }

                LaunchLinkageAcceptIfBetterFull(ctx.stream, m,
                                              ws->d_B_cand.ptr,
                                              ws->d_a_cand.ptr,
                                              ws->d_cost_cand.ptr,
                                              d_B_out,
                                              d_a_out,
                                              d_cost_out,
                                              n,
                                              blocks,
                                              threads);
                if (profile) {
                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 3)], ctx.stream),
                            "cudaEventRecord(ils_accept)");
                }
            }
            if (profile) {
                ws->ils_profile_valid = true;
            }
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[5], ctx.stream), "cudaEventRecord(enc5)");
            ws->last_profile_valid = true;
        }

        ThrowIf(cudaGetLastError(), "LinkageEncodeBatchCudaForcedRootWithSampleIds launch");
        return true;
    } catch (const std::exception& e) {
        if (err) {
            *err = e.what();
        }
        return false;
    }
}

template <typename OutCodeT>
static bool LinkageEncodeBatchCudaForcedRootWithSampleIdsLargeRootConstRootImpl(
    const CudaCtx& ctx,
    const CodebookPack& C_root,
    int forced_root_code,
    const float* d_residuals,
    const float* d_res_norm2,
    const std::uint64_t* d_sample_ids,
    int n,
    int icm_iters,
    int ils_iters,
    int perturb_k,
    std::uint32_t seed,
    OutCodeT* d_B_out,
    float* d_a_out,
    float* d_cost_out,
    LinkageEncodeBatchCudaWorkspace* ws,
    std::string* err) {
    try {
        if (!ws) {
            throw std::runtime_error("LinkageEncodeBatchCudaForcedRootWithSampleIdsLargeRootConstRoot: workspace is null.");
        }
        if (n <= 0) {
            return true;
        }
        if (!d_residuals || !d_res_norm2 || !d_B_out || !d_a_out || !d_cost_out) {
            throw std::runtime_error("LinkageEncodeBatchCudaForcedRootWithSampleIdsLargeRootConstRoot: null input pointer.");
        }
        if (!d_sample_ids && ils_iters > 0 && perturb_k > 0) {
            throw std::runtime_error("LinkageEncodeBatchCudaForcedRootWithSampleIdsLargeRootConstRoot: d_sample_ids is null.");
        }
	        ThrowIf(cudaSetDevice(ctx.device), "cudaSetDevice");
	        if (g_profile_enabled.load(std::memory_order_relaxed)) {
	            (void)cudaGetLastError();
	        }

	        ws->EnsureLargeRootPrecomp(ctx, C_root);
        ws->EnsureLargeRootBatch(n);

        const int d = ws->lr_d;
        const int m = ws->lr_m;
        const int h0 = ws->lr_h0;
        const int Hs = ws->lr_Hs;
        if (m <= 1 || m > kMaxM) {
            throw std::runtime_error("LinkageEncodeBatchCudaForcedRootWithSampleIdsLargeRootConstRoot: unsupported m.");
        }
        if constexpr (std::is_same_v<OutCodeT, FullCode>) {
            // `FullCode` is uint8_t; it can represent codes in [0, 255], i.e. h0 up to 256.
            if (h0 > 256) {
                throw std::runtime_error(
                    "LinkageEncodeBatchCudaForcedRootWithSampleIdsLargeRootConstRoot: h0_root>256 requires U32 output.");
            }
        }
        if (forced_root_code < 0 || forced_root_code >= h0) {
            throw std::runtime_error("LinkageEncodeBatchCudaForcedRootWithSampleIdsLargeRootConstRoot: forced_root_code out of range.");
        }
        if (d != C_root.books.front().rows) {
            throw std::runtime_error("LinkageEncodeBatchCudaForcedRootWithSampleIdsLargeRootConstRoot: d mismatch.");
        }

        // Cache norm0(root) on host per forced_root_code.
        if (ws->lr_norm0_root_code != forced_root_code || ws->lr_norm0_root_tag != ws->lr_precomp_tag) {
            const float* c0 = C_root.books[0].Col(forced_root_code);
            double n2 = 0.0;
            for (int r = 0; r < d; ++r) {
                const auto v = static_cast<double>(c0[r]);
                n2 += v * v;
            }
            ws->lr_norm0_root = static_cast<float>(n2);
            ws->lr_norm0_root_code = forced_root_code;
            ws->lr_norm0_root_tag = ws->lr_precomp_tag;
        }
        const float norm0_root = ws->lr_norm0_root;
        if (!(norm0_root > 0.0f)) {
            // Degenerate root centroid: make root contribution a no-op (still deterministic).
            // The downstream LS solver will handle the ill-conditioned case via regularization.
        }

        const bool profile = g_profile_enabled.load(std::memory_order_relaxed);
        if (profile) {
            ws->EnsureEvents();
            ws->last_profile_valid = false;
            ws->ils_profile_valid = false;
            ThrowIf(cudaEventRecord(ws->ev[0], ctx.stream), "cudaEventRecord(enc0)");
        }

        // xC_small = C_small^T * residuals  (Hs×n).
        {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            ThrowIf(cublasSgemm(ctx.cublas,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                Hs, n, d,
                                &alpha,
                                ws->lr_d_C_small.ptr, d,
                                d_residuals, d,
                                &beta,
                                ws->lr_d_xC_small.ptr, Hs),
                    "cublasSgemm(C_small^T * re)");
        }
        // rC_small = xC_small (copy).
        {
            const int total = Hs * n;
            const int threads = 256;
            const int blocks = (total + threads - 1) / threads;
            CopyF32<<<blocks, threads, 0, ctx.stream>>>(ws->lr_d_xC_small.ptr, ws->lr_d_rC_small.ptr, total);
        }

        // xC0 = C0_root^T * residuals  (1×n).
        const float* d_C0_root = ws->lr_d_C0.ptr + static_cast<std::size_t>(forced_root_code) * static_cast<std::size_t>(d);
        {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            ThrowIf(cublasSgemm(ctx.cublas,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                /*rows(C)=*/1, /*cols(C)=*/n, /*k=*/d,
                                &alpha,
                                d_C0_root, d,
                                d_residuals, d,
                                &beta,
                                ws->lr_d_xC0.ptr, /*ldc=*/1),
                    "cublasSgemm(C0_root^T * re)");
        }

        // g0s_root = C_small^T * C0_root  (Hs×1), cached per forced_root_code.
        const float* d_g0s_root = ws->lr_d_g0s_root.ptr;
        if (ws->lr_forced_root_code != forced_root_code || ws->lr_forced_root_tag != ws->lr_precomp_tag) {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            ThrowIf(cublasSgemm(ctx.cublas,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                /*rows(C)=*/Hs, /*cols(C)=*/1, /*k=*/d,
                                &alpha,
                                ws->lr_d_C_small.ptr, d,
                                d_C0_root, d,
                                &beta,
                                ws->lr_d_g0s_root.ptr, /*ldc=*/Hs),
                    "cublasSgemm(g0s_root)");
            ws->lr_forced_root_code = forced_root_code;
            ws->lr_forced_root_tag = ws->lr_precomp_tag;
        }

        // Apply fixed-root contribution: rC_small -= (xC0 / norm0_root) * g0s_root
        // Implemented as a rank-1 update: rC_small += alpha * g0s_root * xC0_row, alpha = -1/norm0_root.
        if (norm0_root > 0.0f) {
            const float alpha = -1.0f / norm0_root;
            const float beta = 1.0f;
            ThrowIf(cublasSgemm(ctx.cublas,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                /*rows(C)=*/Hs, /*cols(C)=*/n, /*k=*/1,
                                &alpha,
                                d_g0s_root, /*lda=*/Hs,
                                ws->lr_d_xC0.ptr, /*ldb=*/1,
                                &beta,
                                ws->lr_d_rC_small.ptr, /*ldc=*/Hs),
                    "cublasSgemm(rC_small -= g0s_root * xC0/norm0)");
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[1], ctx.stream), "cudaEventRecord(enc1)");
        }

        // Greedy init (abs) on small layers only.
        {
            LaunchGreedyInitAbsSmallFixedRoot(ctx.stream, m, Hs,
                                              ws->lr_d_small_offsets.ptr,
                                              ws->lr_d_hvec.ptr,
                                              ws->lr_d_inv_small.ptr,
                                              ws->lr_d_G_small.ptr,
                                              ws->lr_d_rC_small.ptr,
                                              ws->lr_d_B_small.ptr,
                                              n);
        }
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[2], ctx.stream), "cudaEventRecord(enc2)");
        }

        // LS solve + initial costs.
        {
            const int threads = 128;
            const int blocks = (n + threads - 1) / threads;
            LaunchSolveCostInitLargeRootFixedRoot(ctx.stream, m, Hs,
                                                  ws->lr_d_small_offsets.ptr, norm0_root,
                                                  ws->lr_d_xC_small.ptr,
                                                  ws->lr_d_G_small.ptr,
                                                  d_g0s_root,
                                                  ws->lr_d_xC0.ptr,
                                                  d_res_norm2,
                                                  ws->lr_d_B_small.ptr,
                                                  d_a_out,
                                                  d_cost_out,
                                                  ws->d_active.ptr,
                                                  ws->d_changed.ptr,
                                                  n,
                                                  blocks,
                                                  threads);
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[3], ctx.stream), "cudaEventRecord(enc3)");
        }

        // ICM (skip layer0): shuffle the full 0..m-1 order, but skip jlayer==0 (match full-precomp path).
        const bool do_icm = (icm_iters > 0 && m > 1);
        std::mt19937 icm_rng;
        std::array<int, kMaxM> icm_order{};
        const auto reset_icm_order = [&]() {
            for (int j = 0; j < m; ++j) icm_order[static_cast<std::size_t>(j)] = j;
        };
        if (do_icm) {
            icm_rng = std::mt19937(static_cast<std::mt19937::result_type>(seed));
        }

        if (do_icm) {
            const int blocks = (n + 127) / 128;
            const int threads = 128;
            for (int it = 0; it < icm_iters; ++it) {
                reset_icm_order();
                std::shuffle(icm_order.begin(), icm_order.begin() + m, icm_rng);
                ClearChanged<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_changed.ptr);
                for (int ord = 0; ord < m; ++ord) {
                    const int jlayer = icm_order[static_cast<std::size_t>(ord)];
                    if (jlayer == 0) continue;
                    LaunchIcmLayerLargeRootFixedRootBlockAbs(ctx.stream, m, Hs,
                                                            ws->lr_d_small_offsets.ptr,
                                                            ws->lr_d_hvec.ptr,
                                                            ws->lr_d_inv_small.ptr,
                                                            norm0_root,
                                                            ws->lr_d_xC_small.ptr,
                                                            ws->lr_d_G_small.ptr,
                                                            d_g0s_root,
                                                            ws->lr_d_xC0.ptr,
                                                            d_res_norm2,
                                                            ws->d_active.ptr,
                                                            ws->d_changed.ptr,
                                                            jlayer,
                                                            ws->lr_d_B_small.ptr,
                                                            d_a_out,
                                                            d_cost_out,
                                                            n);
                }
                UpdateActive<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_active.ptr, ws->d_changed.ptr);
            }
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[4], ctx.stream), "cudaEventRecord(enc4)");
        }

        // ILS outer loop (perturb → LS+cost → ICM → accept), operating on small codes only.
        const bool do_ils = (m > 1 && ils_iters > 0 && perturb_k > 0);
        if (do_ils) {
            ws->EnsureIlsEvents(ils_iters);
            if (profile && ws->ils_ev.empty()) {
                // EnsureIlsEvents already set; this is just defensive.
                throw std::runtime_error("LinkageEncodeBatchCudaForcedRootWithSampleIdsLargeRootConstRoot: ils events missing.");
            }
            const int copy_threads = 256;
            const int copy_blocks = (n + copy_threads - 1) / copy_threads;
            const int threads = 128;
            const int blocks = (n + threads - 1) / threads;
            if (profile) {
                ThrowIf(cudaEventRecord(ws->ils_ev[0], ctx.stream), "cudaEventRecord(ils0)");
            }

            for (int outer = 0; outer < ils_iters; ++outer) {
                const int ksel = std::min(std::max(0, perturb_k), m - 1);
                LaunchLinkageCopyAndPerturbSmallCodesSkipLayer0WithSampleIds(
                    ctx.stream, m, ws->lr_d_hvec.ptr, seed, outer + 1, d_sample_ids, ksel,
                    ws->lr_d_B_small.ptr, ws->lr_d_B_small_cand.ptr, n, copy_blocks, copy_threads);
                if (profile) {
                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 0)], ctx.stream),
                            "cudaEventRecord(ils_copy)");
                }

                LaunchSolveCostInitLargeRootFixedRoot(ctx.stream, m, Hs,
                                                      ws->lr_d_small_offsets.ptr, norm0_root,
                                                      ws->lr_d_xC_small.ptr,
                                                      ws->lr_d_G_small.ptr,
                                                      d_g0s_root,
                                                      ws->lr_d_xC0.ptr,
                                                      d_res_norm2,
                                                      ws->lr_d_B_small_cand.ptr,
                                                      ws->d_a_cand.ptr,
                                                      ws->d_cost_cand.ptr,
                                                      ws->d_active.ptr,
                                                      ws->d_changed.ptr,
                                                      n,
                                                      blocks,
                                                      threads);

                if (profile) {
                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 1)], ctx.stream),
                            "cudaEventRecord(ils_solve_cost)");
                }

                if (do_icm) {
                    for (int it = 0; it < icm_iters; ++it) {
                        reset_icm_order();
                        std::shuffle(icm_order.begin(), icm_order.begin() + m, icm_rng);
                        ClearChanged<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_changed.ptr);
                        for (int ord = 0; ord < m; ++ord) {
                            const int jlayer = icm_order[static_cast<std::size_t>(ord)];
                            if (jlayer == 0) continue;
                            LaunchIcmLayerLargeRootFixedRootBlockAbs(ctx.stream, m, Hs,
                                                                    ws->lr_d_small_offsets.ptr,
                                                                    ws->lr_d_hvec.ptr,
                                                                    ws->lr_d_inv_small.ptr,
                                                                    norm0_root,
                                                                    ws->lr_d_xC_small.ptr,
                                                                    ws->lr_d_G_small.ptr,
                                                                    d_g0s_root,
                                                                    ws->lr_d_xC0.ptr,
                                                                    d_res_norm2,
                                                                    ws->d_active.ptr,
                                                                    ws->d_changed.ptr,
                                                                    jlayer,
                                                                    ws->lr_d_B_small_cand.ptr,
                                                                    ws->d_a_cand.ptr,
                                                                    ws->d_cost_cand.ptr,
                                                                    n);
                        }
                        UpdateActive<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_active.ptr, ws->d_changed.ptr);
                    }
                }

                if (profile) {
                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 2)], ctx.stream),
                            "cudaEventRecord(ils_icm)");
                }

                LaunchLinkageAcceptIfBetterLargeRootFixedRoot(ctx.stream, m,
                                                              ws->lr_d_B_small_cand.ptr,
                                                              ws->d_a_cand.ptr,
                                                              ws->d_cost_cand.ptr,
                                                              ws->lr_d_B_small.ptr,
                                                              d_a_out,
                                                              d_cost_out,
                                                              n,
                                                              blocks,
                                                              threads);

                if (profile) {
                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 3)], ctx.stream),
                            "cudaEventRecord(ils_accept)");
                }
            }
            if (profile) {
                ws->ils_profile_valid = true;
            }
        } else {
            ws->EnsureIlsEvents(0);
        }

        // Materialize full codes (include forced root) for the caller.
        {
            const int threads = 256;
            const int blocks = (n + threads - 1) / threads;
            LaunchWriteCodesFromFixedRootAndSmall<OutCodeT>(ctx.stream, m, forced_root_code,
                                                            ws->lr_d_B_small.ptr,
                                                            d_B_out,
                                                            n,
                                                            blocks,
                                                            threads);
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[5], ctx.stream), "cudaEventRecord(enc5)");
            ws->last_profile_valid = true;
        }

        ThrowIf(cudaGetLastError(), "LinkageEncodeBatchCudaForcedRootWithSampleIdsLargeRootConstRoot launch");
        return true;
    } catch (const std::exception& e) {
        if (err) {
            *err = e.what();
        }
        return false;
    }
}

bool LinkageEncodeBatchCudaForcedRootWithSampleIdsLargeRootConstRoot(const CudaCtx& ctx,
                                                                  const CodebookPack& C_root,
                                                                  int forced_root_code,
                                                                  const float* d_residuals,
                                                                  const float* d_res_norm2,
                                                                  const std::uint64_t* d_sample_ids,
                                                                  int n,
                                                                  int icm_iters,
                                                                  int ils_iters,
                                                                  int perturb_k,
                                                                  std::uint32_t seed,
                                                                  FullCode* d_B_out,
                                                                  float* d_a_out,
                                                                  float* d_cost_out,
                                                                  LinkageEncodeBatchCudaWorkspace* ws,
                                                                  std::string* err) {
    return LinkageEncodeBatchCudaForcedRootWithSampleIdsLargeRootConstRootImpl<FullCode>(
        ctx, C_root, forced_root_code, d_residuals, d_res_norm2, d_sample_ids, n, icm_iters, ils_iters, perturb_k,
        seed, d_B_out, d_a_out, d_cost_out, ws, err);
}

bool LinkageEncodeBatchCudaForcedRootWithSampleIdsLargeRootConstRootU32(const CudaCtx& ctx,
                                                                     const CodebookPack& C_root,
                                                                     int forced_root_code,
                                                                     const float* d_residuals,
                                                                     const float* d_res_norm2,
                                                                     const std::uint64_t* d_sample_ids,
                                                                     int n,
                                                                     int icm_iters,
                                                                     int ils_iters,
                                                                     int perturb_k,
                                                                     std::uint32_t seed,
                                                                     std::uint32_t* d_B_out_u32,
                                                                     float* d_a_out,
                                                                     float* d_cost_out,
                                                                     LinkageEncodeBatchCudaWorkspace* ws,
                                                                     std::string* err) {
    return LinkageEncodeBatchCudaForcedRootWithSampleIdsLargeRootConstRootImpl<std::uint32_t>(
        ctx, C_root, forced_root_code, d_residuals, d_res_norm2, d_sample_ids, n, icm_iters, ils_iters, perturb_k,
        seed, d_B_out_u32, d_a_out, d_cost_out, ws, err);
}

template <typename OutCodeT>
static bool LinkageEncodeBatchCudaWithSampleIdsLargeRootSingleImpl(const CudaCtx& ctx,
                                                                 const CodebookPack& C_root,
                                                                 const float* d_residuals,
                                                                 const float* d_res_norm2,
                                                                 const std::uint64_t* d_sample_ids,
                                                                 int n,
                                                                 int icm_iters,
                                                                 int ils_iters,
                                                                 int perturb_k,
                                                                 std::uint32_t seed,
                                                                 OutCodeT* d_B_out,
                                                                 float* d_a_out,
                                                                 float* d_cost_out,
                                                                 LinkageEncodeBatchCudaWorkspace* ws,
                                                                 std::string* err) {
    try {
        if (!ws) {
            throw std::runtime_error("LinkageEncodeBatchCudaWithSampleIdsLargeRoot: workspace is null.");
        }
        if (n <= 0) {
            return true;
        }
        if (!d_residuals || !d_res_norm2 || !d_B_out || !d_a_out || !d_cost_out) {
            throw std::runtime_error("LinkageEncodeBatchCudaWithSampleIdsLargeRoot: null input pointer.");
        }
	        if (!d_sample_ids && ils_iters > 0 && perturb_k > 0) {
	            throw std::runtime_error("LinkageEncodeBatchCudaWithSampleIdsLargeRoot: d_sample_ids is null.");
	        }
	        ThrowIf(cudaSetDevice(ctx.device), "cudaSetDevice");

	        const bool profile = g_profile_enabled.load(std::memory_order_relaxed);
	        std::int64_t t_ensure0 = 0;
	        std::uint64_t pre_tag0 = 0;
	        std::size_t cap_xc_small0 = 0;
	        std::size_t cap_xc0_chunk0 = 0;
	        std::size_t cap_best0 = 0;
	        std::size_t cap_B_small0 = 0;
	        std::size_t cap_a0 = 0;
	        std::size_t cap_cost0 = 0;
	        if (profile) {
	            t_ensure0 = NowNs();
	            pre_tag0 = ws->lr_precomp_tag;
	            cap_xc_small0 = ws->lr_d_xC_small.cap;
	            cap_xc0_chunk0 = ws->lr_d_xC0_chunk.cap;
	            cap_best0 = ws->lr_d_best_score.cap;
	            cap_B_small0 = ws->lr_d_B_small.cap;
	            cap_a0 = ws->d_a_cand.cap;
	            cap_cost0 = ws->d_cost_cand.cap;
	        }

	        ws->EnsureLargeRootPrecomp(ctx, C_root);

        const int d = ws->lr_d;
        const int m = ws->lr_m;
        const int h0 = ws->lr_h0;
        const int Hs = ws->lr_Hs;
        if (m <= 1 || m > kMaxM) {
            throw std::runtime_error("LinkageEncodeBatchCudaWithSampleIdsLargeRoot: unsupported m.");
        }
        if constexpr (std::is_same_v<OutCodeT, FullCode>) {
            // `FullCode` is uint8_t; it can represent codes in [0, 255], i.e. h0 up to 256.
            if (h0 > 256) {
                throw std::runtime_error("LinkageEncodeBatchCudaWithSampleIdsLargeRoot: h0_root>256 requires U32 output.");
            }
        }
        if (d != C_root.books.front().rows) {
            throw std::runtime_error("LinkageEncodeBatchCudaWithSampleIdsLargeRoot: d mismatch.");
        }
        if (h0 <= 0) {
            throw std::runtime_error("LinkageEncodeBatchCudaWithSampleIdsLargeRoot: invalid h0.");
        }

	        // Root selection chunk size: keep xC0_chunk under a configurable cap to avoid OOM when (h0×n) is large.
        // This is a strict exact scan over all h0 codes, just tiled.
        //
        // When a large-root workspace budget is active, also cap xC0_chunk to a fraction of that budget so
        // we don't accidentally blow VRAM by choosing a huge root_chunk_h when `n` is small.
        const int xc0_chunk_mb = g_lr_xc0_chunk_mb.load(std::memory_order_relaxed);
        const std::size_t kMaxXc0ChunkBytesCfg =
            static_cast<std::size_t>(std::max(0, xc0_chunk_mb)) * 1024ull * 1024ull;
        std::size_t kMaxXc0ChunkBytes = (kMaxXc0ChunkBytesCfg > 0) ? kMaxXc0ChunkBytesCfg : 0ull;
        const int ws_mb = g_lr_workspace_mb.load(std::memory_order_relaxed);
        if (ws_mb > 0) {
            const std::size_t ws_bytes = static_cast<std::size_t>(ws_mb) * 1024ull * 1024ull;
            // xC0_chunk is only for root selection (GEMM output) and can be much smaller than the full workspace.
            // Keep it a small fraction of the budget to avoid OOM when many CUDA ctx exist.
            const std::size_t ws_xc0_cap = ws_bytes / 16u;
            kMaxXc0ChunkBytes = (kMaxXc0ChunkBytes > 0) ? std::min(kMaxXc0ChunkBytes, ws_xc0_cap) : ws_xc0_cap;
        }
        int root_chunk_h = static_cast<int>(
            std::min<std::size_t>(static_cast<std::size_t>(h0),
                                  std::max<std::size_t>(256u,
                                                        (kMaxXc0ChunkBytes > 0)
                                                            ? (kMaxXc0ChunkBytes / (sizeof(float) * static_cast<std::size_t>(n)))
                                                            : 256u)));
        // Round down to a multiple of 256 for more stable GEMM throughput when possible.
        root_chunk_h = std::max(1, (root_chunk_h / 256) * 256);
        root_chunk_h = std::min(root_chunk_h, h0);
        const bool have_xc0_full = (root_chunk_h == h0);
        if (profile) {
            g_lr_calls.fetch_add(1, std::memory_order_relaxed);
            const std::uint64_t root_chunks =
                static_cast<std::uint64_t>((h0 + root_chunk_h - 1) / std::max(1, root_chunk_h));
            g_lr_root_chunks_total.fetch_add(root_chunks, std::memory_order_relaxed);
            AtomicMaxU64(&g_lr_root_chunks_max, root_chunks);
            if (have_xc0_full) {
                g_lr_have_xc0_full_calls.fetch_add(1, std::memory_order_relaxed);
            }
            AtomicMinU64(&g_lr_root_chunk_h_min, static_cast<std::uint64_t>(root_chunk_h));
            AtomicMaxU64(&g_lr_root_chunk_h_max, static_cast<std::uint64_t>(root_chunk_h));
        }

	        ws->EnsureLargeRootBatchVarRoot(n, root_chunk_h);
	        if (profile) {
	            const std::int64_t dt = NowNs() - t_ensure0;
	            g_host_calls.fetch_add(1, std::memory_order_relaxed);
	            g_host_ensure_ns.fetch_add(dt, std::memory_order_relaxed);
	            if (ws->lr_precomp_tag != pre_tag0) {
	                g_host_precomp_refresh_calls.fetch_add(1, std::memory_order_relaxed);
	            }
	            const std::size_t Hs_n = static_cast<std::size_t>(Hs) * static_cast<std::size_t>(n);
	            const std::size_t h0chunk_n = static_cast<std::size_t>(root_chunk_h) * static_cast<std::size_t>(n);
	            const std::size_t small_codes = static_cast<std::size_t>(std::max(0, m - 1)) * static_cast<std::size_t>(n);
	            const std::size_t a_elems = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
	            const bool xc_grew = (Hs_n > cap_xc_small0) || (h0chunk_n > cap_xc0_chunk0);
	            const bool any_grew =
	                xc_grew ||
	                (static_cast<std::size_t>(n) > cap_best0) ||
	                (small_codes > cap_B_small0) ||
	                (a_elems > cap_a0) ||
	                (static_cast<std::size_t>(n) > cap_cost0);
	            if (any_grew) {
	                g_host_batch_grow_calls.fetch_add(1, std::memory_order_relaxed);
	            }
	            if (xc_grew) {
	                g_host_xc_grow_calls.fetch_add(1, std::memory_order_relaxed);
	            }
	            const auto cap_xc =
	                static_cast<std::uint64_t>(std::max(ws->lr_d_xC_small.cap, ws->lr_d_xC0_chunk.cap));
	            AtomicMaxU64(&g_host_xc_grow_max_elems, cap_xc);
	        }
	        if (profile) {
	            ws->EnsureEvents();
	            ws->last_profile_valid = false;
	            ws->ils_profile_valid = false;
	            ThrowIf(cudaEventRecord(ws->ev[0], ctx.stream), "cudaEventRecord(enc0)");
        }

        // xC_small = C_small^T * residuals  (Hs×n).
        {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            ThrowIf(cublasSgemm(ctx.cublas,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                Hs, n, d,
                                &alpha,
                                ws->lr_d_C_small.ptr, d,
                                d_residuals, d,
                                &beta,
                                ws->lr_d_xC_small.ptr, Hs),
                    "cublasSgemm(C_small^T * re)");
        }

		        // Root greedy selection: scan all h0 codes using chunked GEMM.
		        // If `have_xc0_full`, this loop runs once and leaves a full xC0 cache in `lr_d_xC0_chunk`.
		        {
	            const int threads = 256;
	            const int blocks_init = (n + threads - 1) / threads;
	            InitBestRootState<<<blocks_init, threads, 0, ctx.stream>>>(
	                n, ws->lr_d_best_score.ptr, ws->lr_d_B0.ptr, ws->lr_d_xC0.ptr, ws->lr_d_norm0.ptr);

	            const float alpha = 1.0f;
	            const float beta = 0.0f;
	            for (int s = 0; s < h0; s += root_chunk_h) {
	                const int len = std::min(root_chunk_h, h0 - s);
	                const float* d_C0_chunk = ws->lr_d_C0.ptr + static_cast<std::size_t>(s) * static_cast<std::size_t>(d);
	                ThrowIf(cublasSgemm(ctx.cublas,
	                                    CUBLAS_OP_T, CUBLAS_OP_N,
	                                    /*rows(C)=*/len, /*cols(C)=*/n, /*k=*/d,
	                                    &alpha,
	                                    d_C0_chunk, d,
	                                    d_residuals, d,
	                                    &beta,
	                                    ws->lr_d_xC0_chunk.ptr, /*ldc=*/len),
	                        "cublasSgemm(C0_chunk^T * re)");
	                UpdateBestRootFromChunk<<<n, threads, 0, ctx.stream>>>(
	                    s, len, h0, n,
	                    ws->lr_d_xC0_chunk.ptr,
	                    ws->lr_d_inv_root.ptr,
	                    ws->lr_d_best_score.ptr,
	                    ws->lr_d_B0.ptr,
	                    ws->lr_d_xC0.ptr,
	                    ws->lr_d_norm0.ptr);
	            }
	        }

        // Gather selected root vectors and build g0s = C_small^T * C0_sel (Hs×n).
        {
            const int total = Hs * n;
            const int threads = 256;
            const int blocks = (total + threads - 1) / threads;
            GatherG0sFromTable<<<blocks, threads, 0, ctx.stream>>>(
                Hs, n, ws->lr_d_G0s.ptr, ws->lr_d_B0.ptr, ws->lr_d_g0s.ptr);
        }

        // rC_small = xC_small - (xC0/norm0) * g0s  (Hs×n).
        {
            const int total = Hs * n;
            const int threads = 256;
            const int blocks = (total + threads - 1) / threads;
            ApplyRootProjectionToRcSmall<<<blocks, threads, 0, ctx.stream>>>(
                Hs, n, ws->lr_d_xC_small.ptr, ws->lr_d_g0s.ptr, ws->lr_d_xC0.ptr, ws->lr_d_norm0.ptr, ws->lr_d_rC_small.ptr);
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[1], ctx.stream), "cudaEventRecord(enc1)");
        }

        // Greedy init (abs) on small layers only (root is fixed after greedy selection above).
        {
            LaunchGreedyInitAbsSmallFixedRoot(ctx.stream, m, Hs,
                                              ws->lr_d_small_offsets.ptr,
                                              ws->lr_d_hvec.ptr,
                                              ws->lr_d_inv_small.ptr,
                                              ws->lr_d_G_small.ptr,
                                              ws->lr_d_rC_small.ptr,
                                              ws->lr_d_B_small.ptr,
                                              n);
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[2], ctx.stream), "cudaEventRecord(enc2)");
        }

        const int threads = 256;
        const int blocks = (n + threads - 1) / threads;
        const bool do_icm = icm_iters > 0;
        const int ils_iters_eff = std::max(0, ils_iters);
        const int ksel = std::max(0, perturb_k);

	        // Solve + initial costs.
	        LaunchSolveCostInitLargeRootFixedRootPerSample(ctx.stream, m, Hs,
	                                                       ws->lr_d_small_offsets.ptr,
	                                                       ws->lr_d_norm0.ptr,
	                                                       ws->lr_d_xC_small.ptr,
	                                                       ws->lr_d_G_small.ptr,
	                                                       ws->lr_d_g0s.ptr,
	                                                       ws->lr_d_xC0.ptr,
	                                                       d_res_norm2,
	                                                       ws->lr_d_B_small.ptr,
	                                                       d_a_out,
	                                                       d_cost_out,
	                                                       ws->d_active.ptr,
	                                                       ws->d_changed.ptr,
	                                                       n,
	                                                       blocks,
	                                                       threads);

	        if (profile) {
	            // Solve+cost segment boundary (matches full-precomp profiling semantics).
	            ThrowIf(cudaEventRecord(ws->ev[3], ctx.stream), "cudaEventRecord(enc3)");
	        }

		        if (do_icm) {
		            // Deterministic per-call RNG on host for layer order (kept small).
		            std::mt19937 icm_rng(static_cast<std::mt19937::result_type>(seed) ^ 0xA17C9E5u);
	            std::vector<int> icm_order(static_cast<std::size_t>(m));
            auto reset_icm_order = [&]() {
                for (int l = 0; l < m; ++l) icm_order[static_cast<std::size_t>(l)] = l;
            };
            for (int it = 0; it < icm_iters; ++it) {
                reset_icm_order();
                std::shuffle(icm_order.begin(), icm_order.begin() + m, icm_rng);
                ClearChanged<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_changed.ptr);
	                for (int ord = 0; ord < m; ++ord) {
	                    const int jlayer = icm_order[static_cast<std::size_t>(ord)];
	                    if (jlayer == 0) {
	                        // Root ICM update (exact scan over h0) using current small-layer codes + coefficients.
	                        {
		                            const int threads0 = 256;
		                            const int blocks0 = (n + threads0 - 1) / threads0;
		                            if (have_xc0_full && ws->lr_d_G0s_T.ptr) {
		                                if (profile) {
		                                    g_lr_root_icm_fast_calls.fetch_add(1, std::memory_order_relaxed);
			                                }
			                                // Fast path: use cached xC0_full + transposed cross table to compute
			                                // C0^T*(x - sum_{l>0} a_l*C_l) without building re0 or running GEMM.
			                                LaunchFindBestRootCandFromXc0FullAndSmall(
			                                    ctx.stream, m, h0,
			                                    ws->lr_d_xC0_chunk.ptr,
			                                    ws->lr_d_inv_root.ptr,
			                                    ws->lr_d_G0s_T.ptr,
			                                    ws->lr_d_small_offsets.ptr,
			                                    ws->lr_d_B_small.ptr,
			                                    d_a_out,
			                                    ws->lr_d_B0_cand.ptr,
			                                    n,
			                                    threads0);
			                            } else {
			                                if (profile) {
			                                    g_lr_root_icm_slow_calls.fetch_add(1, std::memory_order_relaxed);
			                                }
			                                const int total = d * n;
			                                const int re_threads = 256;
			                                const int re_blocks = (total + re_threads - 1) / re_threads;
			                                LaunchBuildRe0FromSmallCodesLargeRoot(
			                                    ctx.stream, m, d,
			                                    ws->lr_d_small_offsets.ptr,
			                                    ws->lr_d_C_small.ptr,
			                                    ws->lr_d_B_small.ptr,
			                                    d_a_out,
			                                    d_residuals,
			                                    ws->lr_d_re0.ptr,
			                                    n,
			                                    re_blocks,
			                                    re_threads);

	                                InitBestRootCandState<<<blocks0, threads0, 0, ctx.stream>>>(
	                                    n, ws->lr_d_best_score.ptr, ws->lr_d_B0_cand.ptr);
	                                const float alpha = 1.0f;
	                                const float beta = 0.0f;
	                                for (int s = 0; s < h0; s += root_chunk_h) {
	                                    const int len = std::min(root_chunk_h, h0 - s);
	                                    const float* d_C0_chunk = ws->lr_d_C0.ptr + static_cast<std::size_t>(s) * static_cast<std::size_t>(d);
	                                    ThrowIf(cublasSgemm(ctx.cublas,
	                                                        CUBLAS_OP_T, CUBLAS_OP_N,
	                                                        /*rows(C)=*/len, /*cols(C)=*/n, /*k=*/d,
	                                                        &alpha,
	                                                        d_C0_chunk, d,
	                                                        ws->lr_d_re0.ptr, d,
	                                                        &beta,
	                                                        ws->lr_d_xC0_chunk.ptr, /*ldc=*/len),
	                                            "cublasSgemm(C0_chunk^T * re0)");
	                                    UpdateBestRootCandFromChunk<<<n, threads0, 0, ctx.stream>>>(
	                                        s, len, h0, n,
	                                        ws->lr_d_xC0_chunk.ptr,
	                                        ws->lr_d_inv_root.ptr,
	                                        ws->lr_d_best_score.ptr,
	                                        ws->lr_d_B0_cand.ptr);
	                                }
	                            }
			                            LaunchTryAcceptRootIcmLargeRootPerSample(
			                                ctx.stream, m, d, Hs,
			                                ws->lr_d_small_offsets.ptr,
			                                ws->lr_d_C0.ptr,
			                                ws->lr_d_C_small.ptr,
			                                ws->lr_d_xC_small.ptr,
			                                ws->lr_d_G_small.ptr,
			                                d_residuals,
			                                d_res_norm2,
			                                ws->lr_d_B0_cand.ptr,
			                                ws->lr_d_B_small.ptr,
			                                ws->d_active.ptr,
			                                ws->lr_d_B0.ptr,
			                                ws->lr_d_xC0.ptr,
			                                ws->lr_d_norm0.ptr,
			                                d_a_out,
			                                d_cost_out,
			                                ws->d_changed.ptr,
			                                n,
			                                blocks0,
			                                threads0);
                            // Rebuild per-sample g0s for the (possibly updated) root codes.
                            {
                                const int total = Hs * n;
                                const int threads_g = 256;
                                const int blocks_g = (total + threads_g - 1) / threads_g;
                                GatherG0sFromTable<<<blocks_g, threads_g, 0, ctx.stream>>>(
                                    Hs, n, ws->lr_d_G0s.ptr, ws->lr_d_B0.ptr, ws->lr_d_g0s.ptr);
                            }
                        }
                        continue;
                    }
		                    LaunchIcmLayerLargeRootFixedRootPerSampleBlockAbs(
		                        ctx.stream, m, Hs,
		                        ws->lr_d_small_offsets.ptr,
		                        ws->lr_d_hvec.ptr,
		                        ws->lr_d_inv_small.ptr,
		                        ws->lr_d_norm0.ptr,
		                        ws->lr_d_xC_small.ptr,
		                        ws->lr_d_G_small.ptr,
		                        ws->lr_d_g0s.ptr,
		                        ws->lr_d_xC0.ptr,
		                        d_res_norm2,
		                        ws->d_active.ptr,
		                        ws->d_changed.ptr,
		                        jlayer,
		                        ws->lr_d_B_small.ptr,
		                        d_a_out,
		                        d_cost_out,
		                        n);
	                }
		                UpdateActive<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_active.ptr, ws->d_changed.ptr);
		            }
		        }

	        if (profile) {
	            // ICM segment boundary (matches full-precomp profiling semantics).
	            ThrowIf(cudaEventRecord(ws->ev[4], ctx.stream), "cudaEventRecord(enc4)");
	        }

	        if (ils_iters_eff > 0 && ksel > 0) {
	            ws->EnsureIlsEvents(ils_iters_eff);
            if (profile) {
                ThrowIf(cudaEventRecord(ws->ils_ev[0], ctx.stream), "cudaEventRecord(ils0)");
            }
            std::mt19937 icm_rng(static_cast<std::mt19937::result_type>(seed) ^ 0x51C2E2Du);
            std::vector<int> icm_order(static_cast<std::size_t>(m));
            auto reset_icm_order = [&]() {
                for (int l = 0; l < m; ++l) icm_order[static_cast<std::size_t>(l)] = l;
            };
	            for (int outer = 0; outer < ils_iters_eff; ++outer) {
	                // Hybrid init-linkage ILS policy: first `vr_rounds` use VarRoot (layer0 perturbable +
	                // root ICM GEMM), remaining rounds use ConstRoot (layer0 frozen, skip root ICM GEMM).
	                const int vr_rounds = g_hybrid_varroot_ils_rounds.load(std::memory_order_relaxed);
	                const bool varroot_this = (vr_rounds < 0 || outer < vr_rounds);
	                {
		                    const int copy_threads = 256;
		                    const int copy_blocks = (n + copy_threads - 1) / copy_threads;
		                    if (varroot_this) {
		                    LaunchLinkageCopyAndPerturbLargeRootCodesWithSampleIds(
		                        ctx.stream, m, ws->lr_d_hvec.ptr, seed, outer + 1, d_sample_ids, ksel,
		                        ws->lr_d_B0.ptr, ws->lr_d_B_small.ptr,
		                        ws->lr_d_B0_cand.ptr, ws->lr_d_B_small_cand.ptr,
		                        n, copy_blocks, copy_threads);
		                    } else {
		                        // Hybrid ConstRoot: freeze root, perturb only small layers.
		                        LaunchLinkageCopyAndPerturbLargeRootCodesSkipRootWithSampleIds(
		                            ctx.stream, m, ws->lr_d_hvec.ptr, seed, outer + 1, d_sample_ids, ksel,
		                            ws->lr_d_B0.ptr, ws->lr_d_B_small.ptr,
		                            ws->lr_d_B0_cand.ptr, ws->lr_d_B_small_cand.ptr,
		                            n, copy_blocks, copy_threads);
		                    }
		                }
	                if (profile) {
	                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 0)], ctx.stream),
	                            "cudaEventRecord(ils_copy)");
	                }

	                // Build candidate root-dependent terms (xC0/norm0/g0s) for the perturbed root codes.
	                if (varroot_this) {
	                    ComputeXc0AndNorm0FromRootCodes<<<n, 256, 0, ctx.stream>>>(
	                        d, n, d_residuals, ws->lr_d_C0.ptr, ws->lr_d_B0_cand.ptr, ws->lr_d_xC0_cand.ptr, ws->lr_d_norm0_cand.ptr);
	                    const int total = Hs * n;
	                    const int threads_g = 256;
	                    const int blocks_g = (total + threads_g - 1) / threads_g;
	                    GatherG0sFromTable<<<blocks_g, threads_g, 0, ctx.stream>>>(
	                        Hs, n, ws->lr_d_G0s.ptr, ws->lr_d_B0_cand.ptr, ws->lr_d_g0s_cand.ptr);
	                }

	                // For ConstRoot rounds, root-dependent terms are unchanged — read directly from
	                // current (accepted) buffers instead of copying to candidate buffers via D2D memcpy.
	                const float* d_norm0_rd = varroot_this ? ws->lr_d_norm0_cand.ptr : ws->lr_d_norm0.ptr;
	                const float* d_g0s_rd  = varroot_this ? ws->lr_d_g0s_cand.ptr  : ws->lr_d_g0s.ptr;
	                const float* d_xC0_rd  = varroot_this ? ws->lr_d_xC0_cand.ptr  : ws->lr_d_xC0.ptr;

		                LaunchSolveCostInitLargeRootFixedRootPerSample(
		                    ctx.stream, m, Hs,
		                    ws->lr_d_small_offsets.ptr,
		                    d_norm0_rd,
		                    ws->lr_d_xC_small.ptr,
		                    ws->lr_d_G_small.ptr,
		                    d_g0s_rd,
		                    d_xC0_rd,
		                    d_res_norm2,
		                    ws->lr_d_B_small_cand.ptr,
		                    ws->d_a_cand.ptr,
		                    ws->d_cost_cand.ptr,
		                    ws->d_active.ptr,
		                    ws->d_changed.ptr,
		                    n,
		                    blocks,
		                    threads);
                if (profile) {
                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 1)], ctx.stream),
                            "cudaEventRecord(ils_solve_cost)");
                }

                if (do_icm) {
                    for (int it = 0; it < icm_iters; ++it) {
                        reset_icm_order();
                        std::shuffle(icm_order.begin(), icm_order.begin() + m, icm_rng);
	                        ClearChanged<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_changed.ptr);
	                        for (int ord = 0; ord < m; ++ord) {
	                            const int jlayer = icm_order[static_cast<std::size_t>(ord)];
	                            if (jlayer == 0) {
	                                // Hybrid ConstRoot: skip root ICM (the expensive tiled GEMM over h0).
	                                if (!varroot_this) continue;
	                                // Root ICM update for candidate (exact scan over h0).
	                                {
		                                    const int threads0 = 256;
		                                    const int blocks0 = (n + threads0 - 1) / threads0;
		                                    if (have_xc0_full && ws->lr_d_G0s_T.ptr) {
		                                        LaunchFindBestRootCandFromXc0FullAndSmall(
		                                            ctx.stream, m, h0,
		                                            ws->lr_d_xC0_chunk.ptr,
		                                            ws->lr_d_inv_root.ptr,
		                                            ws->lr_d_G0s_T.ptr,
		                                            ws->lr_d_small_offsets.ptr,
		                                            ws->lr_d_B_small_cand.ptr,
		                                            ws->d_a_cand.ptr,
		                                            ws->lr_d_B0_tmp.ptr,
		                                            n,
		                                            threads0);
		                                    } else {
		                                        const int total = d * n;
		                                        const int re_threads = 256;
		                                        const int re_blocks = (total + re_threads - 1) / re_threads;
		                                        LaunchBuildRe0FromSmallCodesLargeRoot(
		                                            ctx.stream, m, d,
		                                            ws->lr_d_small_offsets.ptr,
		                                            ws->lr_d_C_small.ptr,
		                                            ws->lr_d_B_small_cand.ptr,
		                                            ws->d_a_cand.ptr,
		                                            d_residuals,
		                                            ws->lr_d_re0.ptr,
		                                            n,
		                                            re_blocks,
		                                            re_threads);

	                                        InitBestRootCandState<<<blocks0, threads0, 0, ctx.stream>>>(
	                                            n, ws->lr_d_best_score.ptr, ws->lr_d_B0_tmp.ptr);
	                                        const float alpha = 1.0f;
	                                        const float beta = 0.0f;
	                                        for (int s = 0; s < h0; s += root_chunk_h) {
	                                            const int len = std::min(root_chunk_h, h0 - s);
	                                            const float* d_C0_chunk = ws->lr_d_C0.ptr + static_cast<std::size_t>(s) * static_cast<std::size_t>(d);
	                                            ThrowIf(cublasSgemm(ctx.cublas,
	                                                                CUBLAS_OP_T, CUBLAS_OP_N,
	                                                                /*rows(C)=*/len, /*cols(C)=*/n, /*k=*/d,
	                                                                &alpha,
	                                                                d_C0_chunk, d,
	                                                                ws->lr_d_re0.ptr, d,
	                                                                &beta,
	                                                                ws->lr_d_xC0_chunk.ptr, /*ldc=*/len),
	                                                    "cublasSgemm(C0_chunk^T * re0 cand)");
	                                            UpdateBestRootCandFromChunk<<<n, threads0, 0, ctx.stream>>>(
	                                                s, len, h0, n,
	                                                ws->lr_d_xC0_chunk.ptr,
	                                                ws->lr_d_inv_root.ptr,
	                                                ws->lr_d_best_score.ptr,
	                                                ws->lr_d_B0_tmp.ptr);
	                                        }
	                                    }
		                                    LaunchTryAcceptRootIcmLargeRootPerSample(
		                                        ctx.stream, m, d, Hs,
		                                        ws->lr_d_small_offsets.ptr,
		                                        ws->lr_d_C0.ptr,
		                                        ws->lr_d_C_small.ptr,
		                                        ws->lr_d_xC_small.ptr,
		                                        ws->lr_d_G_small.ptr,
		                                        d_residuals,
		                                        d_res_norm2,
		                                        ws->lr_d_B0_tmp.ptr,
		                                        ws->lr_d_B_small_cand.ptr,
		                                        ws->d_active.ptr,
		                                        ws->lr_d_B0_cand.ptr,
		                                        ws->lr_d_xC0_cand.ptr,
		                                        ws->lr_d_norm0_cand.ptr,
		                                        ws->d_a_cand.ptr,
		                                        ws->d_cost_cand.ptr,
		                                        ws->d_changed.ptr,
		                                        n,
		                                        blocks0,
		                                        threads0);

	                                    // Rebuild candidate g0s for updated candidate root codes.
	                                    {
	                                        const int total = Hs * n;
	                                        const int threads_g = 256;
	                                        const int blocks_g = (total + threads_g - 1) / threads_g;
	                                        GatherG0sFromTable<<<blocks_g, threads_g, 0, ctx.stream>>>(
	                                            Hs, n, ws->lr_d_G0s.ptr, ws->lr_d_B0_cand.ptr, ws->lr_d_g0s_cand.ptr);
	                                    }
	                                }
	                                continue;
	                            }
		                            LaunchIcmLayerLargeRootFixedRootPerSampleBlockAbs(
		                                ctx.stream, m, Hs,
		                                ws->lr_d_small_offsets.ptr,
		                                ws->lr_d_hvec.ptr,
		                                ws->lr_d_inv_small.ptr,
		                                d_norm0_rd,
		                                ws->lr_d_xC_small.ptr,
		                                ws->lr_d_G_small.ptr,
		                                d_g0s_rd,
		                                d_xC0_rd,
		                                d_res_norm2,
		                                ws->d_active.ptr,
		                                ws->d_changed.ptr,
		                                jlayer,
		                                ws->lr_d_B_small_cand.ptr,
		                                ws->d_a_cand.ptr,
		                                ws->d_cost_cand.ptr,
		                                n);
                        }
                        UpdateActive<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_active.ptr, ws->d_changed.ptr);
                    }
                }

                if (profile) {
                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 2)], ctx.stream),
                            "cudaEventRecord(ils_icm)");
                }

		                if (varroot_this) {
		                    LaunchLinkageAcceptIfBetterLargeRootVarRoot(
		                        ctx.stream, m,
		                        ws->lr_d_B0_cand.ptr,
		                        ws->lr_d_xC0_cand.ptr,
		                        ws->lr_d_norm0_cand.ptr,
		                        ws->lr_d_B_small_cand.ptr,
		                        ws->d_a_cand.ptr,
		                        ws->d_cost_cand.ptr,
		                        ws->lr_d_B0.ptr,
		                        ws->lr_d_xC0.ptr,
		                        ws->lr_d_norm0.ptr,
		                        ws->lr_d_B_small.ptr,
		                        d_a_out,
		                        d_cost_out,
		                        n,
		                        blocks,
		                        threads);
		                } else {
		                    // Hybrid ConstRoot: root is unchanged — use lighter FixedRoot accept that
		                    // skips writing B0/xC0/norm0 (they are already up-to-date in current buffers).
		                    LaunchLinkageAcceptIfBetterLargeRootFixedRoot(
		                        ctx.stream, m,
		                        ws->lr_d_B_small_cand.ptr,
		                        ws->d_a_cand.ptr,
		                        ws->d_cost_cand.ptr,
		                        ws->lr_d_B_small.ptr,
		                        d_a_out,
		                        d_cost_out,
		                        n,
		                        blocks,
		                        threads);
		                }

	                // Root may change during ILS; rebuild current g0s for correctness.
	                // Hybrid ConstRoot rounds: root is unchanged, skip g0s rebuild.
	                if (varroot_this) {
	                    const int total = Hs * n;
	                    const int threads_g = 256;
	                    const int blocks_g = (total + threads_g - 1) / threads_g;
	                    GatherG0sFromTable<<<blocks_g, threads_g, 0, ctx.stream>>>(
	                        Hs, n, ws->lr_d_G0s.ptr, ws->lr_d_B0.ptr, ws->lr_d_g0s.ptr);
	                }

	                if (profile) {
	                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 3)], ctx.stream),
	                            "cudaEventRecord(ils_accept)");
	                }
	            }
	            if (profile) {
                ws->ils_profile_valid = true;
            }
        } else {
            ws->EnsureIlsEvents(0);
        }

	        // Materialize full codes (include selected root) for the caller.
	        {
	            const int threads = 256;
	            const int blocks = (n + threads - 1) / threads;
	            LaunchWriteCodesFromPerSampleRootAndSmall<OutCodeT>(ctx.stream, m,
	                                                                ws->lr_d_B0.ptr,
	                                                                ws->lr_d_B_small.ptr,
	                                                                d_B_out,
	                                                                n,
	                                                                blocks,
	                                                                threads);
	        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[5], ctx.stream), "cudaEventRecord(enc5)");
            ws->last_profile_valid = true;
        }

        ThrowIf(cudaGetLastError(), "LinkageEncodeBatchCudaWithSampleIdsLargeRoot launch");
        return true;
    } catch (const std::exception& e) {
        if (err) {
            *err = e.what();
        }
        return false;
    }
}

bool LinkageEncodeBatchCudaWithSampleIdsLargeRoot(const CudaCtx& ctx,
                                               const CodebookPack& C_root,
                                               const float* d_residuals,
                                               const float* d_res_norm2,
                                               const std::uint64_t* d_sample_ids,
                                               int n,
                                               int icm_iters,
                                               int ils_iters,
                                               int perturb_k,
                                               std::uint32_t seed,
                                               FullCode* d_B_out,
                                               float* d_a_out,
                                               float* d_cost_out,
                                               LinkageEncodeBatchCudaWorkspace* ws,
                                               std::string* err) {
    try {
        if (!ws) {
            throw std::runtime_error("LinkageEncodeBatchCudaWithSampleIdsLargeRoot: workspace is null.");
        }
        if (n <= 0) {
            return true;
        }
        if (!d_residuals || !d_res_norm2 || !d_B_out || !d_a_out || !d_cost_out) {
            throw std::runtime_error("LinkageEncodeBatchCudaWithSampleIdsLargeRoot: null input pointer.");
        }
        if (!d_sample_ids && ils_iters > 0 && perturb_k > 0) {
            throw std::runtime_error("LinkageEncodeBatchCudaWithSampleIdsLargeRoot: d_sample_ids is null.");
        }

        const int d = C_root.books.front().rows;
        const int m = static_cast<int>(C_root.books.size());
        const int h0 = C_root.books.front().cols;
        int Hs = 0;
        for (int l = 1; l < m; ++l) {
            Hs += C_root.books[static_cast<std::size_t>(l)].cols;
        }

        const auto is_oom = [](const std::string& msg) -> bool {
            return (msg.find("out of memory") != std::string::npos) || (msg.find("cudaMalloc(") != std::string::npos);
        };
        auto round_down_chunk = [](int v) -> int {
            if (v <= 1) return 1;
            if (v > 8192) {
                return (v / 8192) * 8192;
            }
            if (v >= 256) {
                return (v / 256) * 256;
            }
            return v;
        };

        // Match the allocator rounding used by the workspace (see PickAllocN in LinkageEncodeBatchCudaWorkspace).
        const auto round_alloc_n = [](int n_in) -> int {
            if (n_in <= 0) return 0;
            if (n_in <= 8192) {
                auto v = static_cast<std::uint32_t>(n_in - 1);
                v |= v >> 1;
                v |= v >> 2;
                v |= v >> 4;
                v |= v >> 8;
                v |= v >> 16;
                return static_cast<int>(v + 1);
            }
            if (n_in <= 65536) {
                constexpr int k = 8192;
                return ((n_in + k - 1) / k) * k;
            }
            return n_in;
        };

        // Heuristic: to enable the fast root-update path (reuse full xC0 + G0s_T, no re0/GEMM inside root ICM/ILS),
        // we need root_chunk_h==h0, which implies:
        //   kMaxXc0ChunkBytes / (sizeof(float) * n_chunk) >= h0
        // => n_chunk <= kMaxXc0ChunkBytes / (sizeof(float) * h0)
        const int xc0_chunk_mb = g_lr_xc0_chunk_mb.load(std::memory_order_relaxed);
        const std::size_t kMaxXc0ChunkBytes =
            static_cast<std::size_t>(std::max(0, xc0_chunk_mb)) * 1024ull * 1024ull;
        std::size_t max_n_full_xc0 = 0;
        if (kMaxXc0ChunkBytes > 0 && h0 > 0) {
            max_n_full_xc0 = kMaxXc0ChunkBytes / (sizeof(float) * static_cast<std::size_t>(h0));
        }
        int chunk_n = n;
        if (max_n_full_xc0 > 0 && static_cast<std::size_t>(n) > max_n_full_xc0) {
            chunk_n = static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(n), max_n_full_xc0));
            // Round down for steadier GEMM performance when possible.
            if (chunk_n >= 256) {
                chunk_n = (chunk_n / 256) * 256;
            }
            chunk_n = std::max(1, chunk_n);
        }

        // Additionally, bound temporary large-root workspace (xC_small/rC_small/g0s/etc) by a MiB budget.
        // This matters when `n` is Npairs (many-nodes evaluator) and can be large.
        const int ws_mb = g_lr_workspace_mb.load(std::memory_order_relaxed);
        if (ws_mb > 0 && Hs > 0) {
            // Use only a fraction of the budget for the encoder itself; the evaluator also needs VRAM.
            const std::size_t budget_bytes_full = static_cast<std::size_t>(ws_mb) * 1024ull * 1024ull;
            const std::size_t budget_bytes = budget_bytes_full / 2u;
            // Conservative estimate per sample (float arrays dominate):
            //   xC_small + rC_small + g0s + g0s_cand ~= 4 * Hs floats
            //   re0 ~= d floats
            //   xC0_chunk ~= min(256,h0) floats (root-chunk is separately capped by workspace budget in the single-call impl)
            // plus small overhead for codes/cost/flags.
            const std::size_t root_min = static_cast<std::size_t>(std::min(256, std::max(1, h0)));
            const std::size_t floats_per_sample =
                4ull * static_cast<std::size_t>(Hs) + static_cast<std::size_t>(d) + root_min + 64ull;
            const std::size_t bytes_per_sample = floats_per_sample * sizeof(float);
            if (bytes_per_sample > 0) {
                std::size_t max_n_by_budget = budget_bytes / bytes_per_sample;
                if (max_n_by_budget == 0) max_n_by_budget = 1;
                int cand = static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(chunk_n), max_n_by_budget));
                // Make the allocator rounding predictable (avoid 8192-multiple rounding up past the budget).
                if (cand > 8192) {
                    cand = (cand / 8192) * 8192;
                } else if (cand >= 256) {
                    cand = (cand / 256) * 256;
                }
                cand = std::max(1, cand);

                // Adjust down until the rounded allocation fits within the budget.
                while (cand > 1) {
                    const int n_alloc = round_alloc_n(cand);
                    const std::size_t need = static_cast<std::size_t>(n_alloc) * bytes_per_sample;
                    if (need <= budget_bytes) break;
                    if (cand > 8192) {
                        cand -= 8192;
                    } else if (cand >= 256) {
                        cand -= 256;
                    } else {
                        cand -= 1;
                    }
                }

                chunk_n = std::min(chunk_n, cand);
            }
        }

        // If we can process in one shot, keep the old behavior in the common case.
        // However, still handle rare OOM/fragmentation peaks by falling back to the chunked backoff path.
	        if (chunk_n >= n) {
	            std::string local_err;
	            const bool ok = LinkageEncodeBatchCudaWithSampleIdsLargeRootSingleImpl<FullCode>(
	                ctx, C_root,
	                d_residuals, d_res_norm2, d_sample_ids,
	                n, icm_iters, ils_iters, perturb_k, seed,
	                d_B_out, d_a_out, d_cost_out, ws, &local_err);
	            if (ok) return true;
	            if (!is_oom(local_err)) {
	                if (err) *err = local_err;
	                return false;
	            }
	            // OOM: drop any previously grown large-root batch buffers so backoff can actually shrink allocations.
	            ws->ReleaseLargeRootBatchBuffers();
	            // OOM: fall through into the backoff loop with a smaller initial chunk.
	            chunk_n = round_down_chunk(std::max(1, n / 2));
	        }

        if (g_profile_enabled.load(std::memory_order_relaxed)) {
            g_lr_sample_chunk_calls.fetch_add(1, std::memory_order_relaxed);
            const std::uint64_t chunks =
                static_cast<std::uint64_t>((n + chunk_n - 1) / std::max(1, chunk_n));
            g_lr_sample_chunks_total.fetch_add(chunks, std::memory_order_relaxed);
            AtomicMinU64(&g_lr_sample_chunk_n_min, static_cast<std::uint64_t>(chunk_n));
            AtomicMaxU64(&g_lr_sample_chunk_n_max, static_cast<std::uint64_t>(chunk_n));
        }

        // Chunk along sample dimension (independent samples => semantics unchanged).
        // Note: profiling events inside the single-call implementation are per-call; with chunking enabled
        // the detailed per-stage timing becomes less meaningful, but correctness is preserved.
        int i0 = 0;
        while (i0 < n) {
            int nc = std::min(chunk_n, n - i0);
            // Robust backoff on OOM: if allocations still fail due to fragmentation / concurrent ctx peaks,
            // retry the same chunk with smaller `nc` until it fits.
            while (true) {
                const float* d_res_chunk = d_residuals + static_cast<std::size_t>(i0) * static_cast<std::size_t>(d);
                const float* d_n2_chunk = d_res_norm2 + i0;
                const std::uint64_t* d_sid_chunk = d_sample_ids ? (d_sample_ids + i0) : nullptr;
                // B_out is column-major (rows=m, cols=n). Chunk offset must advance by whole columns.
                FullCode* d_B_chunk =
                    d_B_out + static_cast<std::size_t>(i0) * static_cast<std::size_t>(m);
                float* d_a_chunk = d_a_out + static_cast<std::size_t>(i0) * static_cast<std::size_t>(m);
                float* d_cost_chunk = d_cost_out + i0;

                std::string local_err;
                const bool ok = LinkageEncodeBatchCudaWithSampleIdsLargeRootSingleImpl<FullCode>(
                    ctx, C_root,
                    d_res_chunk, d_n2_chunk, d_sid_chunk,
                    nc, icm_iters, ils_iters, perturb_k, seed,
                    d_B_chunk, d_a_chunk, d_cost_chunk,
                    ws, &local_err);
                if (ok) break;

	                if (!is_oom(local_err) || nc <= 1) {
	                    if (err) {
	                        *err = "LinkageEncodeBatchCudaWithSampleIdsLargeRoot chunk failed at i0=" +
	                               std::to_string(i0) + " nc=" + std::to_string(nc) + ": " + local_err;
	                    }
	                    return false;
	                }

	                // OOM: free large-root batch buffers so the next retry can allocate smaller.
	                ws->ReleaseLargeRootBatchBuffers();

	                int next = nc / 2;
	                next = round_down_chunk(next);
	                next = std::max(1, next);
	                if (next >= nc) {
	                    next = std::max(1, nc - 1);
                }
                nc = next;
            }
            i0 += nc;
        }
        return true;
    } catch (const std::exception& e) {
        if (err) {
            *err = e.what();
        }
        return false;
    }
}

bool LinkageEncodeBatchCudaWithSampleIdsLargeRootU32(const CudaCtx& ctx,
                                                  const CodebookPack& C_root,
                                                  const float* d_residuals,
                                                  const float* d_res_norm2,
                                                  const std::uint64_t* d_sample_ids,
                                                  int n,
                                                  int icm_iters,
                                                  int ils_iters,
                                                  int perturb_k,
                                                  std::uint32_t seed,
                                                  std::uint32_t* d_B_out_u32,
                                                  float* d_a_out,
                                                  float* d_cost_out,
                                                  LinkageEncodeBatchCudaWorkspace* ws,
                                                  std::string* err) {
    try {
        if (!ws) {
            throw std::runtime_error("LinkageEncodeBatchCudaWithSampleIdsLargeRootU32: workspace is null.");
        }
        if (n <= 0) {
            return true;
        }
        if (!d_residuals || !d_res_norm2 || !d_B_out_u32 || !d_a_out || !d_cost_out) {
            throw std::runtime_error("LinkageEncodeBatchCudaWithSampleIdsLargeRootU32: null input pointer.");
        }
        if (!d_sample_ids && ils_iters > 0 && perturb_k > 0) {
            throw std::runtime_error("LinkageEncodeBatchCudaWithSampleIdsLargeRootU32: d_sample_ids is null.");
        }

        const int d = C_root.books.front().rows;
        const int m = static_cast<int>(C_root.books.size());
        const int h0 = C_root.books.front().cols;
        int Hs = 0;
        for (int l = 1; l < m; ++l) {
            Hs += C_root.books[static_cast<std::size_t>(l)].cols;
        }

        const auto is_oom = [](const std::string& msg) -> bool {
            return (msg.find("out of memory") != std::string::npos) || (msg.find("cudaMalloc(") != std::string::npos);
        };
        auto round_down_chunk = [](int v) -> int {
            if (v <= 1) return 1;
            if (v > 8192) {
                return (v / 8192) * 8192;
            }
            if (v >= 256) {
                return (v / 256) * 256;
            }
            return v;
        };
        const auto round_alloc_n = [](int n_in) -> int {
            if (n_in <= 0) return 0;
            if (n_in <= 8192) {
                auto v = static_cast<std::uint32_t>(n_in - 1);
                v |= v >> 1;
                v |= v >> 2;
                v |= v >> 4;
                v |= v >> 8;
                v |= v >> 16;
                return static_cast<int>(v + 1);
            }
            if (n_in <= 65536) {
                constexpr int k = 8192;
                return ((n_in + k - 1) / k) * k;
            }
            return n_in;
        };

        const int xc0_chunk_mb = g_lr_xc0_chunk_mb.load(std::memory_order_relaxed);
        const std::size_t kMaxXc0ChunkBytes =
            static_cast<std::size_t>(std::max(0, xc0_chunk_mb)) * 1024ull * 1024ull;
        std::size_t max_n_full_xc0 = 0;
        if (kMaxXc0ChunkBytes > 0 && h0 > 0) {
            max_n_full_xc0 = kMaxXc0ChunkBytes / (sizeof(float) * static_cast<std::size_t>(h0));
        }
        int chunk_n = n;
        if (max_n_full_xc0 > 0 && static_cast<std::size_t>(n) > max_n_full_xc0) {
            chunk_n = static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(n), max_n_full_xc0));
            if (chunk_n >= 256) {
                chunk_n = (chunk_n / 256) * 256;
            }
            chunk_n = std::max(1, chunk_n);
        }

        const int ws_mb = g_lr_workspace_mb.load(std::memory_order_relaxed);
        if (ws_mb > 0 && Hs > 0) {
            const std::size_t budget_bytes_full = static_cast<std::size_t>(ws_mb) * 1024ull * 1024ull;
            const std::size_t budget_bytes = budget_bytes_full / 2u;
            const std::size_t root_min = static_cast<std::size_t>(std::min(256, std::max(1, h0)));
            const std::size_t floats_per_sample =
                4ull * static_cast<std::size_t>(Hs) + static_cast<std::size_t>(d) + root_min + 64ull;
            const std::size_t bytes_per_sample = floats_per_sample * sizeof(float);
            if (bytes_per_sample > 0) {
                std::size_t max_n_by_budget = budget_bytes / bytes_per_sample;
                if (max_n_by_budget == 0) max_n_by_budget = 1;
                int cand = static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(chunk_n), max_n_by_budget));
                if (cand > 8192) {
                    cand = (cand / 8192) * 8192;
                } else if (cand >= 256) {
                    cand = (cand / 256) * 256;
                }
                cand = std::max(1, cand);
                while (cand > 1) {
                    const int n_alloc = round_alloc_n(cand);
                    const std::size_t need = static_cast<std::size_t>(n_alloc) * bytes_per_sample;
                    if (need <= budget_bytes) break;
                    if (cand > 8192) {
                        cand -= 8192;
                    } else if (cand >= 256) {
                        cand -= 256;
                    } else {
                        cand -= 1;
                    }
                }
                chunk_n = std::min(chunk_n, cand);
            }
        }

        if (chunk_n >= n) {
            std::string local_err;
            const bool ok = LinkageEncodeBatchCudaWithSampleIdsLargeRootSingleImpl<std::uint32_t>(
                ctx, C_root,
                d_residuals, d_res_norm2, d_sample_ids,
                n, icm_iters, ils_iters, perturb_k, seed,
                d_B_out_u32, d_a_out, d_cost_out, ws, &local_err);
            if (ok) return true;
            if (!is_oom(local_err)) {
                if (err) *err = local_err;
                return false;
            }
            ws->ReleaseLargeRootBatchBuffers();
            chunk_n = round_down_chunk(std::max(1, n / 2));
        }

        if (g_profile_enabled.load(std::memory_order_relaxed)) {
            g_lr_sample_chunk_calls.fetch_add(1, std::memory_order_relaxed);
            const std::uint64_t chunks =
                static_cast<std::uint64_t>((n + chunk_n - 1) / std::max(1, chunk_n));
            g_lr_sample_chunks_total.fetch_add(chunks, std::memory_order_relaxed);
            AtomicMinU64(&g_lr_sample_chunk_n_min, static_cast<std::uint64_t>(chunk_n));
            AtomicMaxU64(&g_lr_sample_chunk_n_max, static_cast<std::uint64_t>(chunk_n));
        }

        int i0 = 0;
        while (i0 < n) {
            int nc = std::min(chunk_n, n - i0);
            while (true) {
                const float* d_res_chunk = d_residuals + static_cast<std::size_t>(i0) * static_cast<std::size_t>(d);
                const float* d_n2_chunk = d_res_norm2 + i0;
                const std::uint64_t* d_sid_chunk = d_sample_ids ? (d_sample_ids + i0) : nullptr;
                std::uint32_t* d_B_chunk =
                    d_B_out_u32 + static_cast<std::size_t>(i0) * static_cast<std::size_t>(m);
                float* d_a_chunk = d_a_out + static_cast<std::size_t>(i0) * static_cast<std::size_t>(m);
                float* d_cost_chunk = d_cost_out + i0;

                std::string local_err;
                const bool ok = LinkageEncodeBatchCudaWithSampleIdsLargeRootSingleImpl<std::uint32_t>(
                    ctx, C_root,
                    d_res_chunk, d_n2_chunk, d_sid_chunk,
                    nc, icm_iters, ils_iters, perturb_k, seed,
                    d_B_chunk, d_a_chunk, d_cost_chunk,
                    ws, &local_err);
                if (ok) break;

                if (!is_oom(local_err) || nc <= 1) {
                    if (err) {
                        *err = "LinkageEncodeBatchCudaWithSampleIdsLargeRootU32 chunk failed at i0=" +
                               std::to_string(i0) + " nc=" + std::to_string(nc) + ": " + local_err;
                    }
                    return false;
                }
                ws->ReleaseLargeRootBatchBuffers();
                int next = nc / 2;
                next = round_down_chunk(next);
                next = std::max(1, next);
                if (next >= nc) {
                    next = std::max(1, nc - 1);
                }
                nc = next;
            }
            i0 += nc;
        }
        return true;
    } catch (const std::exception& e) {
        if (err) {
            *err = e.what();
        }
        return false;
    }
}

bool LinkageEncodeBatchCudaForcedRootDeviceXc(const CudaCtx& ctx,
                                           const Precomp& pre_one,
                                           const float* d_xC,
                                           const float* d_res_norm2,
                                           const int* d_forced_root,
                                           int n,
                                           int icm_iters,
                                           int ils_iters,
                                           int perturb_k,
                                           std::uint32_t seed,
                                           std::uint64_t sample_id_offset,
                                           FullCode* d_B_out,
                                           float* d_a_out,
                                           float* d_cost_out,
                                           LinkageEncodeBatchCudaWorkspace* ws,
                                           std::string* err) {
    try {
        if (!ws) {
            throw std::runtime_error("LinkageEncodeBatchCudaForcedRootDeviceXc: workspace is null.");
        }
        if (n <= 0) {
            return true;
        }
        if (!d_xC) {
            throw std::runtime_error("LinkageEncodeBatchCudaForcedRootDeviceXc: d_xC is null.");
        }
        ThrowIf(cudaSetDevice(ctx.device), "cudaSetDevice");

        ws->EnsurePrecomp(ctx, pre_one);
        ws->EnsureBatch(n);

        const int H = pre_one.H;
        const int m = pre_one.m;
        if (m <= 0 || m > kMaxM) {
            throw std::runtime_error("LinkageEncodeBatchCudaForcedRootDeviceXc: unsupported m.");
        }

        const bool profile = g_profile_enabled.load(std::memory_order_relaxed);
        if (profile) {
            ws->EnsureEvents();
            ws->last_profile_valid = false;
            ws->ils_profile_valid = false;
            ThrowIf(cudaEventRecord(ws->ev[0], ctx.stream), "cudaEventRecord(enc0)");
        }

        // xC is precomputed by the caller; copy into workspace so downstream kernels see the same layout.
        ThrowIf(cudaMemcpyAsync(ws->d_xC.ptr,
                                d_xC,
                                sizeof(float) * static_cast<std::size_t>(H) * static_cast<std::size_t>(n),
                                cudaMemcpyDeviceToDevice,
                                ctx.stream),
                "cudaMemcpyAsync(xC D2D)");
        if (profile) {
            // Keep the profiling structure consistent: "gemm" time includes this copy (cheap).
            ThrowIf(cudaEventRecord(ws->ev[1], ctx.stream), "cudaEventRecord(enc1)");
        }

        // rC = xC
        {
            const int total = H * n;
            const int threads = 256;
            const int blocks = (total + threads - 1) / threads;
            CopyF32<<<blocks, threads, 0, ctx.stream>>>(ws->d_xC.ptr, ws->d_rC.ptr, total);
        }

        // Greedy init with forced root (abs) + update rC.
        {
            LaunchGreedyInitAbsForcedRoot(ctx.stream, H, m,
                                          ws->d_offsets.ptr,
                                          ws->d_flat_layer.ptr,
                                          ws->d_inv.ptr,
                                          ws->d_G.ptr,
                                          d_forced_root,
                                          ws->d_rC.ptr,
                                          d_B_out,
                                          d_a_out,
                                          n);
        }
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[2], ctx.stream), "cudaEventRecord(enc2)");
        }

        // LS solve and initial costs.
        {
            const int threads = 128;
            const int blocks = (n + threads - 1) / threads;
            LaunchSolveAllAndCostsFull(ctx.stream, H, m,
                                       ws->d_offsets.ptr,
                                       ws->d_xC.ptr,
                                       ws->d_G.ptr,
                                       d_B_out,
                                       d_a_out,
                                       d_res_norm2,
                                       d_cost_out,
                                       n,
                                       blocks,
                                       threads);
        }
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[3], ctx.stream), "cudaEventRecord(enc3)");
        }

        // Match CPU forced-root path: shuffled layer order per ICM iter, skip layer0.
        std::mt19937 icm_rng;
        std::array<int, kMaxM> icm_order{};
        const auto reset_icm_order = [&]() {
            for (int j = 0; j < m; ++j) icm_order[static_cast<std::size_t>(j)] = j;
        };
        if (icm_iters > 0 && m > 1) {
            icm_rng = std::mt19937(static_cast<std::mt19937::result_type>(seed));
            reset_icm_order();
            const int threads = 128;
            const int blocks = (n + threads - 1) / threads;
            InitActiveChanged<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_active.ptr, ws->d_changed.ptr);

            for (int it = 0; it < icm_iters; ++it) {
                std::shuffle(icm_order.begin(), icm_order.begin() + m, icm_rng);
                ClearChanged<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_changed.ptr);
                for (int ord = 0; ord < m; ++ord) {
                    const int jlayer = icm_order[static_cast<std::size_t>(ord)];
                    if (jlayer == 0) continue;
                    LaunchIcmLayerFullAbs(ctx.stream, H, m,
                                          ws->d_offsets.ptr,
                                          ws->d_hvec.ptr,
                                          ws->d_inv.ptr,
                                          ws->d_xC.ptr,
                                          ws->d_G.ptr,
                                          d_res_norm2,
                                          ws->d_active.ptr,
                                          ws->d_changed.ptr,
                                          jlayer,
                                          d_B_out,
                                          d_a_out,
                                          d_cost_out,
                                          n);
                }
                UpdateActive<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_active.ptr, ws->d_changed.ptr);
            }
        }
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[4], ctx.stream), "cudaEventRecord(enc4)");
        }

        const int ksel = std::min(std::max(0, perturb_k), m);
        if (ils_iters > 0 && ksel > 0 && m > 1) {
            const int threads = 128;
            const int blocks = (n + threads - 1) / threads;
            const int copy_threads = 256;
            const int copy_blocks = (n + copy_threads - 1) / copy_threads;
            if (profile) {
                ws->EnsureIlsEvents(ils_iters);
                ThrowIf(cudaEventRecord(ws->ils_ev[0], ctx.stream), "cudaEventRecord(ils0)");
            }
            for (int outer = 0; outer < ils_iters; ++outer) {
                LaunchLinkageCopyAndPerturbFullCodesSkipLayer0(ctx.stream, m, ws->d_hvec.ptr,
                                                            seed,
                                                            outer + 1,
                                                            sample_id_offset,
                                                            ksel,
                                                            d_B_out,
                                                            ws->d_B_cand.ptr,
                                                            n,
                                                            copy_blocks,
                                                            copy_threads);
                if (profile) {
                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 0)], ctx.stream),
                            "cudaEventRecord(ils_copy)");
                }

                LaunchSolveAllAndCostsFull(ctx.stream, H, m,
                                           ws->d_offsets.ptr,
                                           ws->d_xC.ptr,
                                           ws->d_G.ptr,
                                           ws->d_B_cand.ptr,
                                           ws->d_a_cand.ptr,
                                           d_res_norm2,
                                           ws->d_cost_cand.ptr,
                                           n,
                                           blocks,
                                           threads);
                if (profile) {
                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 1)], ctx.stream),
                            "cudaEventRecord(ils_solve_cost)");
                }

                if (icm_iters > 0 && m > 1) {
                    reset_icm_order();
                    InitActiveChanged<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_active.ptr, ws->d_changed.ptr);
                    for (int it = 0; it < icm_iters; ++it) {
                        std::shuffle(icm_order.begin(), icm_order.begin() + m, icm_rng);
                        ClearChanged<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_changed.ptr);
                        for (int ord = 0; ord < m; ++ord) {
                            const int jlayer = icm_order[static_cast<std::size_t>(ord)];
                            if (jlayer == 0) continue;
                            LaunchIcmLayerFullAbs(ctx.stream, H, m,
                                                  ws->d_offsets.ptr,
                                                  ws->d_hvec.ptr,
                                                  ws->d_inv.ptr,
                                                  ws->d_xC.ptr,
                                                  ws->d_G.ptr,
                                                  d_res_norm2,
                                                  ws->d_active.ptr,
                                                  ws->d_changed.ptr,
                                                  jlayer,
                                                  ws->d_B_cand.ptr,
                                                  ws->d_a_cand.ptr,
                                                  ws->d_cost_cand.ptr,
                                                  n);
                        }
                        UpdateActive<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_active.ptr, ws->d_changed.ptr);
                    }
                }
                if (profile) {
                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 2)], ctx.stream),
                            "cudaEventRecord(ils_icm)");
                }

                LaunchLinkageAcceptIfBetterFull(ctx.stream, m,
                                              ws->d_B_cand.ptr,
                                              ws->d_a_cand.ptr,
                                              ws->d_cost_cand.ptr,
                                              d_B_out,
                                              d_a_out,
                                              d_cost_out,
                                              n,
                                              blocks,
                                              threads);
                if (profile) {
                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 3)], ctx.stream),
                            "cudaEventRecord(ils_accept)");
                }
            }
            if (profile) {
                ws->ils_profile_valid = true;
            }
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[5], ctx.stream), "cudaEventRecord(enc5)");
            ws->last_profile_valid = true;
        }

        ThrowIf(cudaGetLastError(), "LinkageEncodeBatchCudaForcedRootDeviceXc launch");
        return true;
    } catch (const std::exception& e) {
        if (err) {
            *err = e.what();
        }
        return false;
    }
}

bool LinkageEncodeBatchCudaForcedRootDeviceXcFromInitB(const CudaCtx& ctx,
                                                    const Precomp& pre_one,
                                                    const float* d_xC,
                                                    const float* d_res_norm2,
                                                    const int* d_forced_root,
                                                    int n,
                                                    int icm_iters,
                                                    int ils_iters,
                                                    int perturb_k,
                                                    std::uint32_t seed,
                                                    std::uint64_t sample_id_offset,
                                                    FullCode* d_B_out,
                                                    float* d_a_out,
                                                    float* d_cost_out,
                                                    LinkageEncodeBatchCudaWorkspace* ws,
                                                    std::string* err) {
    try {
        if (!ws) {
            throw std::runtime_error("LinkageEncodeBatchCudaForcedRootDeviceXcFromInitB: workspace is null.");
        }
        if (n <= 0) {
            return true;
        }
        if (!d_xC) {
            throw std::runtime_error("LinkageEncodeBatchCudaForcedRootDeviceXcFromInitB: d_xC is null.");
        }
        if (!d_B_out) {
            throw std::runtime_error("LinkageEncodeBatchCudaForcedRootDeviceXcFromInitB: d_B_out is null.");
        }
        ThrowIf(cudaSetDevice(ctx.device), "cudaSetDevice");

        ws->EnsurePrecomp(ctx, pre_one);
        ws->EnsureBatch(n);

        const int H = pre_one.H;
        const int m = pre_one.m;
        if (m <= 0 || m > kMaxM) {
            throw std::runtime_error("LinkageEncodeBatchCudaForcedRootDeviceXcFromInitB: unsupported m.");
        }

        const bool profile = g_profile_enabled.load(std::memory_order_relaxed);
        if (profile) {
            ws->EnsureEvents();
            ws->last_profile_valid = false;
            ws->ils_profile_valid = false;
            ThrowIf(cudaEventRecord(ws->ev[0], ctx.stream), "cudaEventRecord(enc0)");
        }

        // xC is precomputed by the caller; copy into workspace so downstream kernels see the same layout.
        ThrowIf(cudaMemcpyAsync(ws->d_xC.ptr,
                                d_xC,
                                sizeof(float) * static_cast<std::size_t>(H) * static_cast<std::size_t>(n),
                                cudaMemcpyDeviceToDevice,
                                ctx.stream),
                "cudaMemcpyAsync(xC D2D)");
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[1], ctx.stream), "cudaEventRecord(enc1)");
            // Skip greedy stage: record immediately so ms12 ~ 0.
            ThrowIf(cudaEventRecord(ws->ev[2], ctx.stream), "cudaEventRecord(enc2)");
        }

        // Enforce forced root codes (layer0) defensively.
        {
            const int threads = 256;
            const int blocks = (n + threads - 1) / threads;
            ForceLayer0Codes<<<blocks, threads, 0, ctx.stream>>>(m, d_forced_root, d_B_out, n);
        }

        // LS solve and initial costs from the provided codes.
        {
            const int threads = 128;
            const int blocks = (n + threads - 1) / threads;
            LaunchSolveAllAndCostsFull(ctx.stream, H, m,
                                       ws->d_offsets.ptr,
                                       ws->d_xC.ptr,
                                       ws->d_G.ptr,
                                       d_B_out,
                                       d_a_out,
                                       d_res_norm2,
                                       d_cost_out,
                                       n,
                                       blocks,
                                       threads);
        }
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[3], ctx.stream), "cudaEventRecord(enc3)");
        }

        // Match CPU ICM: per-iter shuffled layer order, skipping layer0.
        std::mt19937 icm_rng;
        std::array<int, kMaxM> icm_order{};
        const auto reset_icm_order = [&]() {
            for (int j = 0; j < m; ++j) icm_order[static_cast<std::size_t>(j)] = j;
        };
        if (icm_iters > 0 && m > 1) {
            icm_rng = std::mt19937(static_cast<std::mt19937::result_type>(seed));
            reset_icm_order();
            const int threads = 128;
            const int blocks = (n + threads - 1) / threads;
            InitActiveChanged<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_active.ptr, ws->d_changed.ptr);

            for (int it = 0; it < icm_iters; ++it) {
                std::shuffle(icm_order.begin(), icm_order.begin() + m, icm_rng);
                ClearChanged<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_changed.ptr);
                for (int ord = 0; ord < m; ++ord) {
                    const int jlayer = icm_order[static_cast<std::size_t>(ord)];
                    if (jlayer == 0) continue;
                    LaunchIcmLayerFullAbs(ctx.stream, H, m,
                                          ws->d_offsets.ptr,
                                          ws->d_hvec.ptr,
                                          ws->d_inv.ptr,
                                          ws->d_xC.ptr,
                                          ws->d_G.ptr,
                                          d_res_norm2,
                                          ws->d_active.ptr,
                                          ws->d_changed.ptr,
                                          jlayer,
                                          d_B_out,
                                          d_a_out,
                                          d_cost_out,
                                          n);
                }
                UpdateActive<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_active.ptr, ws->d_changed.ptr);
            }
        }
        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[4], ctx.stream), "cudaEventRecord(enc4)");
        }

        const int ksel = std::min(std::max(0, perturb_k), m);
        if (ils_iters > 0 && ksel > 0 && m > 1) {
            const int threads = 128;
            const int blocks = (n + threads - 1) / threads;
            const int copy_threads = 256;
            const int copy_blocks = (n + copy_threads - 1) / copy_threads;
            if (profile) {
                ws->EnsureIlsEvents(ils_iters);
                ThrowIf(cudaEventRecord(ws->ils_ev[0], ctx.stream), "cudaEventRecord(ils0)");
            }
            for (int outer = 0; outer < ils_iters; ++outer) {
                LaunchLinkageCopyAndPerturbFullCodesSkipLayer0(ctx.stream, m, ws->d_hvec.ptr,
                                                            seed,
                                                            outer + 1,
                                                            sample_id_offset,
                                                            ksel,
                                                            d_B_out,
                                                            ws->d_B_cand.ptr,
                                                            n,
                                                            copy_blocks,
                                                            copy_threads);
                if (profile) {
                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 0)], ctx.stream),
                            "cudaEventRecord(ils_copy)");
                }

                LaunchSolveAllAndCostsFull(ctx.stream, H, m,
                                           ws->d_offsets.ptr,
                                           ws->d_xC.ptr,
                                           ws->d_G.ptr,
                                           ws->d_B_cand.ptr,
                                           ws->d_a_cand.ptr,
                                           d_res_norm2,
                                           ws->d_cost_cand.ptr,
                                           n,
                                           blocks,
                                           threads);
                if (profile) {
                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 1)], ctx.stream),
                            "cudaEventRecord(ils_solve_cost)");
                }

                if (icm_iters > 0 && m > 1) {
                    reset_icm_order();
                    InitActiveChanged<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_active.ptr, ws->d_changed.ptr);
                    for (int it = 0; it < icm_iters; ++it) {
                        std::shuffle(icm_order.begin(), icm_order.begin() + m, icm_rng);
                        ClearChanged<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_changed.ptr);
                        for (int ord = 0; ord < m; ++ord) {
                            const int jlayer = icm_order[static_cast<std::size_t>(ord)];
                            if (jlayer == 0) continue;
                            LaunchIcmLayerFullAbs(ctx.stream, H, m,
                                                  ws->d_offsets.ptr,
                                                  ws->d_hvec.ptr,
                                                  ws->d_inv.ptr,
                                                  ws->d_xC.ptr,
                                                  ws->d_G.ptr,
                                                  d_res_norm2,
                                                  ws->d_active.ptr,
                                                  ws->d_changed.ptr,
                                                  jlayer,
                                                  ws->d_B_cand.ptr,
                                                  ws->d_a_cand.ptr,
                                                  ws->d_cost_cand.ptr,
                                                  n);
                        }
                        UpdateActive<<<blocks, threads, 0, ctx.stream>>>(n, ws->d_active.ptr, ws->d_changed.ptr);
                    }
                }
                if (profile) {
                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 2)], ctx.stream),
                            "cudaEventRecord(ils_icm)");
                }

                LaunchLinkageAcceptIfBetterFull(ctx.stream, m,
                                              ws->d_B_cand.ptr,
                                              ws->d_a_cand.ptr,
                                              ws->d_cost_cand.ptr,
                                              d_B_out,
                                              d_a_out,
                                              d_cost_out,
                                              n,
                                              blocks,
                                              threads);
                if (profile) {
                    ThrowIf(cudaEventRecord(ws->ils_ev[static_cast<std::size_t>(1 + 4 * outer + 3)], ctx.stream),
                            "cudaEventRecord(ils_accept)");
                }
            }
            if (profile) {
                ws->ils_profile_valid = true;
            }
        }

        if (profile) {
            ThrowIf(cudaEventRecord(ws->ev[5], ctx.stream), "cudaEventRecord(enc5)");
            ws->last_profile_valid = true;
        }

        ThrowIf(cudaGetLastError(), "LinkageEncodeBatchCudaForcedRootDeviceXcFromInitB launch");
        return true;
    } catch (const std::exception& e) {
        if (err) {
            *err = e.what();
        }
        return false;
    }
}

void SetCudaLinkageEncodeProfiling(bool enabled) {
    g_profile_enabled.store(enabled, std::memory_order_relaxed);
}

CudaLinkageEncodeStats GetAndResetCudaLinkageEncodeStats() {
    CudaLinkageEncodeStats out;
    out.calls = g_calls.exchange(0, std::memory_order_relaxed);
    const auto take_ns = [](std::atomic<std::int64_t>* v) -> double {
        const std::int64_t ns = v->exchange(0, std::memory_order_relaxed);
        return static_cast<double>(ns) * 1e-9;
    };
    out.gemm_s = take_ns(&g_gemm_ns);
    out.greedy_s = take_ns(&g_greedy_ns);
    out.ls_cost_s = take_ns(&g_ls_cost_ns);
    out.icm_s = take_ns(&g_icm_ns);
    out.ils_s = take_ns(&g_ils_ns);
    out.ils_copy_s = take_ns(&g_ils_copy_ns);
    out.ils_solve_cost_s = take_ns(&g_ils_solve_cost_ns);
    out.ils_icm_s = take_ns(&g_ils_icm_ns);
    out.ils_encode_s = take_ns(&g_ils_encode_ns);
    out.ils_accept_s = take_ns(&g_ils_accept_ns);
    out.total_s = take_ns(&g_total_ns);
    return out;
}

CudaLinkageEncodeLargeRootStats GetAndResetCudaLinkageEncodeLargeRootStats() {
    CudaLinkageEncodeLargeRootStats out;
    out.calls = g_lr_calls.exchange(0, std::memory_order_relaxed);
    out.root_chunks_total = g_lr_root_chunks_total.exchange(0, std::memory_order_relaxed);
    out.root_chunks_max = g_lr_root_chunks_max.exchange(0, std::memory_order_relaxed);
    out.have_xc0_full_calls = g_lr_have_xc0_full_calls.exchange(0, std::memory_order_relaxed);
    out.root_chunk_h_min = g_lr_root_chunk_h_min.exchange(0, std::memory_order_relaxed);
    out.root_chunk_h_max = g_lr_root_chunk_h_max.exchange(0, std::memory_order_relaxed);
    out.root_icm_fast_calls = g_lr_root_icm_fast_calls.exchange(0, std::memory_order_relaxed);
    out.root_icm_slow_calls = g_lr_root_icm_slow_calls.exchange(0, std::memory_order_relaxed);
    out.sample_chunk_calls = g_lr_sample_chunk_calls.exchange(0, std::memory_order_relaxed);
    out.sample_chunks_total = g_lr_sample_chunks_total.exchange(0, std::memory_order_relaxed);
    out.sample_chunk_n_min = g_lr_sample_chunk_n_min.exchange(0, std::memory_order_relaxed);
    out.sample_chunk_n_max = g_lr_sample_chunk_n_max.exchange(0, std::memory_order_relaxed);
    return out;
}

CudaLinkageEncodeHostStats GetAndResetCudaLinkageEncodeHostStats() {
    CudaLinkageEncodeHostStats out;
    out.calls = g_host_calls.exchange(0, std::memory_order_relaxed);
    const std::int64_t ns = g_host_ensure_ns.exchange(0, std::memory_order_relaxed);
    out.ensure_s = static_cast<double>(ns) * 1e-9;
    out.precomp_refresh_calls = g_host_precomp_refresh_calls.exchange(0, std::memory_order_relaxed);
    out.batch_grow_calls = g_host_batch_grow_calls.exchange(0, std::memory_order_relaxed);
    out.xC_grow_calls = g_host_xc_grow_calls.exchange(0, std::memory_order_relaxed);
    out.xC_grow_max_elems = g_host_xc_grow_max_elems.exchange(0, std::memory_order_relaxed);
    return out;
}

void ConsumeCudaLinkageEncodeLastProfile(LinkageEncodeBatchCudaWorkspace* ws_base) {
    auto* ws = static_cast<LinkageEncodeBatchCudaWorkspace*>(ws_base);
    if (!ws || !ws->events_inited || !ws->last_profile_valid) {
        return;
    }
    float ms01 = 0, ms12 = 0, ms23 = 0, ms34 = 0, ms45 = 0, ms05 = 0;
    // Caller is expected to have synchronized the stream; these events should be complete.
    const auto take_ms = [&](float* out, cudaEvent_t a, cudaEvent_t b) -> bool {
        const cudaError_t st = cudaEventElapsedTime(out, a, b);
        if (st == cudaSuccess) return true;
        // Avoid poisoning the caller thread with a sticky CUDA error.
        (void)cudaGetLastError();
        ws->last_profile_valid = false;
        ws->ils_profile_valid = false;
        return false;
    };
    if (!take_ms(&ms01, ws->ev[0], ws->ev[1])) return;
    if (!take_ms(&ms12, ws->ev[1], ws->ev[2])) return;
    if (!take_ms(&ms23, ws->ev[2], ws->ev[3])) return;
    if (!take_ms(&ms34, ws->ev[3], ws->ev[4])) return;
    if (!take_ms(&ms45, ws->ev[4], ws->ev[5])) return;
    if (!take_ms(&ms05, ws->ev[0], ws->ev[5])) return;

    g_calls.fetch_add(1, std::memory_order_relaxed);
    AddNs(&g_gemm_ns, ms01);
    AddNs(&g_greedy_ns, ms12);
    AddNs(&g_ls_cost_ns, ms23);
    AddNs(&g_icm_ns, ms34);
    AddNs(&g_ils_ns, ms45);
    AddNs(&g_total_ns, ms05);

    if (ws->ils_profile_valid && ws->ils_profile_iters > 0 &&
        ws->ils_ev.size() == static_cast<std::size_t>(1 + 4 * ws->ils_profile_iters)) {
        float ms_copy = 0.0f;
        float ms_solve_cost = 0.0f;
        float ms_icm = 0.0f;
        float ms_accept = 0.0f;
        cudaEvent_t prev = ws->ils_ev[0];
        for (int i = 0; i < ws->ils_profile_iters; ++i) {
            const cudaEvent_t e_copy = ws->ils_ev[static_cast<std::size_t>(1 + 4 * i + 0)];
            const cudaEvent_t e_solve = ws->ils_ev[static_cast<std::size_t>(1 + 4 * i + 1)];
            const cudaEvent_t e_icm = ws->ils_ev[static_cast<std::size_t>(1 + 4 * i + 2)];
            const cudaEvent_t e_acc = ws->ils_ev[static_cast<std::size_t>(1 + 4 * i + 3)];
            float tmp = 0.0f;
            if (cudaEventElapsedTime(&tmp, prev, e_copy) != cudaSuccess) break;
            ms_copy += tmp;
            if (cudaEventElapsedTime(&tmp, e_copy, e_solve) != cudaSuccess) break;
            ms_solve_cost += tmp;
            if (cudaEventElapsedTime(&tmp, e_solve, e_icm) != cudaSuccess) break;
            ms_icm += tmp;
            if (cudaEventElapsedTime(&tmp, e_icm, e_acc) != cudaSuccess) break;
            ms_accept += tmp;
            prev = e_acc;
        }
        const float ms_encode = ms_solve_cost + ms_icm;
        AddNs(&g_ils_copy_ns, ms_copy);
        AddNs(&g_ils_solve_cost_ns, ms_solve_cost);
        AddNs(&g_ils_icm_ns, ms_icm);
        AddNs(&g_ils_encode_ns, ms_encode);
        AddNs(&g_ils_accept_ns, ms_accept);
    }
    ws->last_profile_valid = false;
    ws->ils_profile_valid = false;
}

CudaLinkageEncodeLastProfile GetCudaLinkageEncodeLastProfile(LinkageEncodeBatchCudaWorkspace* ws_base) {
    CudaLinkageEncodeLastProfile out;
    auto* ws = static_cast<LinkageEncodeBatchCudaWorkspace*>(ws_base);
    if (!ws || !ws->events_inited || !ws->last_profile_valid) {
        out.valid = false;
        return out;
    }

    float ms01 = 0, ms12 = 0, ms23 = 0, ms34 = 0, ms45 = 0, ms05 = 0;
    const auto take_ms = [&](float* out_ms, cudaEvent_t a, cudaEvent_t b) -> bool {
        const cudaError_t st = cudaEventElapsedTime(out_ms, a, b);
        if (st == cudaSuccess) return true;
        (void)cudaGetLastError();
        return false;
    };
    if (!take_ms(&ms01, ws->ev[0], ws->ev[1])) return out;
    if (!take_ms(&ms12, ws->ev[1], ws->ev[2])) return out;
    if (!take_ms(&ms23, ws->ev[2], ws->ev[3])) return out;
    if (!take_ms(&ms34, ws->ev[3], ws->ev[4])) return out;
    if (!take_ms(&ms45, ws->ev[4], ws->ev[5])) return out;
    if (!take_ms(&ms05, ws->ev[0], ws->ev[5])) return out;

    out.gemm_s = static_cast<double>(ms01) * 1e-3;
    out.greedy_s = static_cast<double>(ms12) * 1e-3;
    out.ls_cost_s = static_cast<double>(ms23) * 1e-3;
    out.icm_s = static_cast<double>(ms34) * 1e-3;
    out.ils_s = static_cast<double>(ms45) * 1e-3;
    out.total_s = static_cast<double>(ms05) * 1e-3;

    if (ws->ils_profile_valid && ws->ils_profile_iters > 0 &&
        ws->ils_ev.size() == static_cast<std::size_t>(1 + 4 * ws->ils_profile_iters)) {
        float ms_copy = 0.0f;
        float ms_solve_cost = 0.0f;
        float ms_icm = 0.0f;
        float ms_accept = 0.0f;
        cudaEvent_t prev = ws->ils_ev[0];
        for (int i = 0; i < ws->ils_profile_iters; ++i) {
            const cudaEvent_t e_copy = ws->ils_ev[static_cast<std::size_t>(1 + 4 * i + 0)];
            const cudaEvent_t e_solve = ws->ils_ev[static_cast<std::size_t>(1 + 4 * i + 1)];
            const cudaEvent_t e_icm = ws->ils_ev[static_cast<std::size_t>(1 + 4 * i + 2)];
            const cudaEvent_t e_acc = ws->ils_ev[static_cast<std::size_t>(1 + 4 * i + 3)];
            float tmp = 0.0f;
            if (cudaEventElapsedTime(&tmp, prev, e_copy) != cudaSuccess) break;
            ms_copy += tmp;
            if (cudaEventElapsedTime(&tmp, e_copy, e_solve) != cudaSuccess) break;
            ms_solve_cost += tmp;
            if (cudaEventElapsedTime(&tmp, e_solve, e_icm) != cudaSuccess) break;
            ms_icm += tmp;
            if (cudaEventElapsedTime(&tmp, e_icm, e_acc) != cudaSuccess) break;
            ms_accept += tmp;
            prev = e_acc;
        }
        const float ms_encode = ms_solve_cost + ms_icm;
        out.ils_copy_s = static_cast<double>(ms_copy) * 1e-3;
        out.ils_solve_cost_s = static_cast<double>(ms_solve_cost) * 1e-3;
        out.ils_icm_s = static_cast<double>(ms_icm) * 1e-3;
        out.ils_encode_s = static_cast<double>(ms_encode) * 1e-3;
        out.ils_accept_s = static_cast<double>(ms_accept) * 1e-3;
    }

    out.valid = true;
    return out;
}

}  // namespace stlq
