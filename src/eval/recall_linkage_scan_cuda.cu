#include "stlq/eval/recall_linkage_scan_cuda.h"

#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>

#include <cuda_runtime.h>

#include <cub/block/block_radix_sort.cuh>

#include "stlq/common/timer.h"

namespace stlq::eval::cuda {

namespace {

inline void CudaCheck(cudaError_t st, const char* msg) {
    if (st != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(st));
    }
}

template <typename T>
struct DeviceBuf {
    T* ptr = nullptr;
    std::size_t cap = 0;
    DeviceBuf() = default;
    explicit DeviceBuf(std::size_t n_in) { Ensure(n_in); }
    DeviceBuf(const DeviceBuf&) = delete;
    DeviceBuf& operator=(const DeviceBuf&) = delete;
    ~DeviceBuf() { Free(); }
    void Free() {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
            cap = 0;
        }
    }
    void Ensure(std::size_t n_in) {
        if (n_in <= cap) return;
        Free();
        if (n_in) {
            CudaCheck(cudaMalloc(&ptr, sizeof(T) * n_in), "cudaMalloc");
            cap = n_in;
        }
    }
};

struct ClusterDescDev {
    int cid = -1;
    int n_real = 0;
    int n_virt = 0;
    int nc = 0;
    int n_root_real = 0;
    int depth_offsets_len = 0;
    int parent_elem_bytes = 4;
    // Stride (in int8 elements) between consecutive layers in q_layer_major.
    // Non-cache packing uses q_stride==nc; cache packing uses q_stride==max_nc.
    int q_stride = 0;

    std::uint32_t off_real_ids = 0;
    std::uint32_t off_parent = 0;          // bytes
    std::uint32_t off_depth_offsets = 0;   // u32, elements
    std::uint32_t off_codes_small = 0;      // bytes
    std::uint32_t off_code0_one = 0;        // bytes
    std::uint32_t off_virt_codes_small = 0; // bytes
    std::uint32_t off_q_layer_major = 0;    // int8, elements
    std::uint32_t off_scales_root = 0;      // float, elements (m)
    std::uint32_t off_scales_linkage = 0;     // float, elements (m)
    std::uint32_t off_r_norm2 = 0;          // float, elements
};

struct ScanContext {
    cudaStream_t stream = nullptr;
    bool stream_ready = false;

    bool events_ready = false;
    cudaEvent_t ev_pack_start = nullptr;
    cudaEvent_t ev_pack_end = nullptr;
    cudaEvent_t ev_tables_start = nullptr;
    cudaEvent_t ev_tables_end = nullptr;
    cudaEvent_t ev_kernel_start = nullptr;
    cudaEvent_t ev_kernel_end = nullptr;
    cudaEvent_t ev_d2h_start = nullptr;
    cudaEvent_t ev_d2h_end = nullptr;

    DeviceBuf<ClusterDescDev> d_clusters;
    DeviceBuf<std::uint32_t> d_real_ids;
    DeviceBuf<std::uint8_t> d_parent;
    DeviceBuf<std::uint32_t> d_depth_offsets;
    DeviceBuf<std::uint8_t> d_codes_small;
    DeviceBuf<std::uint8_t> d_code0_one;
    DeviceBuf<std::uint8_t> d_virt_codes_small;
    DeviceBuf<std::int8_t> d_q_layer_major;
    DeviceBuf<float> d_scales_root;
    DeviceBuf<float> d_scales_linkage;
    DeviceBuf<float> d_r_norm2;

    DeviceBuf<int> d_offsets_root_small;
    DeviceBuf<int> d_offsets_one;
    DeviceBuf<int> d_task_cluster_idx;

    DeviceBuf<float> d_xCq_root0;
    DeviceBuf<float> d_xCq_root_small;
    DeviceBuf<float> d_xCq_one;

    DeviceBuf<float> d_out_dists;
    DeviceBuf<std::uint32_t> d_out_ids;
    DeviceBuf<unsigned long long> d_kernel_phase_accum; // [roots, depth, topk, blocks]
    int dev_clock_khz = 0;

    // Optional per-cluster device cache (eval-only).
    // When enabled, we keep a small slot cache across query blocks to avoid repeated host packing + H2D.
    int cache_enabled = 0;
    int cache_nlist = 0;
    int cache_slots = 0;
    int cache_max_nreal = 0;
    int cache_max_nc = 0;
    int cache_m = 0;
    int cache_m_codes = 0;
    int cache_max_depth_offsets_len = 0;
    std::uint64_t cache_epoch = 1;
    std::vector<int> cid_to_slot;          // length cache_nlist, -1 if not cached
    std::vector<int> slot_to_cid;          // length cache_slots, -1 if free
    std::vector<std::uint64_t> slot_epoch; // LRU timestamp per slot

    // Host-side staging buffers (reused across calls to avoid per-qblk allocations).
    std::vector<ClusterDescDev> h_clusters;
    std::vector<std::uint32_t> h_real_ids;
    std::vector<std::uint8_t> h_parent;
    std::vector<std::uint32_t> h_depth_offsets;
    std::vector<std::uint8_t> h_codes_small;
    std::vector<std::uint8_t> h_code0_one;
    std::vector<std::uint8_t> h_virt_codes_small;
    std::vector<std::int8_t> h_q_layer_major;
    std::vector<float> h_scales_root;
    std::vector<float> h_scales_linkage;
    std::vector<float> h_r_norm2;
    std::vector<int> h_task_cluster_idx_slots;
    std::vector<int> h_offsets_root_small_cache;
    std::vector<int> h_offsets_one_cache;

    // Opt-in dynamic shared memory (needed when kernel_max_nc==16384 on most GPUs).
    int optin_16384_shmem_11 = 0;

    // Pinned host registrations (to make H2D/D2H truly async and faster).
    void* reg_clusters = nullptr;
    std::size_t reg_clusters_bytes = 0;
    void* reg_real_ids = nullptr;
    std::size_t reg_real_ids_bytes = 0;
    void* reg_parent = nullptr;
    std::size_t reg_parent_bytes = 0;
    void* reg_depth_offsets = nullptr;
    std::size_t reg_depth_offsets_bytes = 0;
    void* reg_codes_small = nullptr;
    std::size_t reg_codes_small_bytes = 0;
    void* reg_code0_one = nullptr;
    std::size_t reg_code0_one_bytes = 0;
    void* reg_virt_codes_small = nullptr;
    std::size_t reg_virt_codes_small_bytes = 0;
    void* reg_q_layer_major = nullptr;
    std::size_t reg_q_layer_major_bytes = 0;
    void* reg_scales_root = nullptr;
    std::size_t reg_scales_root_bytes = 0;
    void* reg_scales_linkage = nullptr;
    std::size_t reg_scales_linkage_bytes = 0;
    void* reg_r_norm2 = nullptr;
    std::size_t reg_r_norm2_bytes = 0;

    void* reg_task_cluster_idx = nullptr;
    std::size_t reg_task_cluster_idx_bytes = 0;

    void* reg_offsets_root_small = nullptr;
    std::size_t reg_offsets_root_small_bytes = 0;
    void* reg_offsets_one = nullptr;
    std::size_t reg_offsets_one_bytes = 0;

    void* reg_xCq_root0 = nullptr;
    std::size_t reg_xCq_root0_bytes = 0;
    void* reg_xCq_root_small = nullptr;
    std::size_t reg_xCq_root_small_bytes = 0;
    void* reg_xCq_one = nullptr;
    std::size_t reg_xCq_one_bytes = 0;

    void* reg_out_dists = nullptr;
    std::size_t reg_out_dists_bytes = 0;
    void* reg_out_ids = nullptr;
    std::size_t reg_out_ids_bytes = 0;

    ~ScanContext() {
        DestroyEvents();
        UnregisterAll();
        if (stream_ready && stream) {
            cudaStreamDestroy(stream);
            stream = nullptr;
            stream_ready = false;
        }
    }

    void EnsureStream() {
        if (stream_ready) return;
        CudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        stream_ready = true;
    }

    void EnsureDeviceClock() {
        if (dev_clock_khz > 0) return;
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return;
        int khz = 0;
        if (cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, dev) != cudaSuccess) return;
        dev_clock_khz = khz;
    }

    void EnsureEvents() {
        if (events_ready) return;
        CudaCheck(cudaEventCreate(&ev_pack_start), "cudaEventCreate pack_start");
        CudaCheck(cudaEventCreate(&ev_pack_end), "cudaEventCreate pack_end");
        CudaCheck(cudaEventCreate(&ev_tables_start), "cudaEventCreate tables_start");
        CudaCheck(cudaEventCreate(&ev_tables_end), "cudaEventCreate tables_end");
        CudaCheck(cudaEventCreate(&ev_kernel_start), "cudaEventCreate kernel_start");
        CudaCheck(cudaEventCreate(&ev_kernel_end), "cudaEventCreate kernel_end");
        CudaCheck(cudaEventCreate(&ev_d2h_start), "cudaEventCreate d2h_start");
        CudaCheck(cudaEventCreate(&ev_d2h_end), "cudaEventCreate d2h_end");
        events_ready = true;
    }

    void DestroyEvents() {
        if (!events_ready) return;
        cudaEventDestroy(ev_pack_start);
        cudaEventDestroy(ev_pack_end);
        cudaEventDestroy(ev_tables_start);
        cudaEventDestroy(ev_tables_end);
        cudaEventDestroy(ev_kernel_start);
        cudaEventDestroy(ev_kernel_end);
        cudaEventDestroy(ev_d2h_start);
        cudaEventDestroy(ev_d2h_end);
        ev_pack_start = ev_pack_end = nullptr;
        ev_tables_start = ev_tables_end = nullptr;
        ev_kernel_start = ev_kernel_end = nullptr;
        ev_d2h_start = ev_d2h_end = nullptr;
        events_ready = false;
    }

    static double ElapsedSeconds(cudaEvent_t a, cudaEvent_t b) {
        float ms = 0.0f;
        CudaCheck(cudaEventElapsedTime(&ms, a, b), "cudaEventElapsedTime");
        return static_cast<double>(ms) * 1e-3;
    }

    static void EnsurePinned(void* ptr, std::size_t bytes, void** reg_ptr, std::size_t* reg_bytes) {
        if (!ptr || bytes == 0) return;
        if (*reg_ptr == ptr && *reg_bytes == bytes) return;
        if (*reg_ptr) {
            cudaHostUnregister(*reg_ptr);
            *reg_ptr = nullptr;
            *reg_bytes = 0;
        }
        CudaCheck(cudaHostRegister(ptr, bytes, cudaHostRegisterPortable), "cudaHostRegister");
        *reg_ptr = ptr;
        *reg_bytes = bytes;
    }

    void UnregisterAll() {
        auto unreg = [](void** p, std::size_t* n) {
            if (*p) {
                cudaHostUnregister(*p);
                *p = nullptr;
                *n = 0;
            }
        };
        unreg(&reg_clusters, &reg_clusters_bytes);
        unreg(&reg_real_ids, &reg_real_ids_bytes);
        unreg(&reg_parent, &reg_parent_bytes);
        unreg(&reg_depth_offsets, &reg_depth_offsets_bytes);
        unreg(&reg_codes_small, &reg_codes_small_bytes);
        unreg(&reg_code0_one, &reg_code0_one_bytes);
        unreg(&reg_virt_codes_small, &reg_virt_codes_small_bytes);
        unreg(&reg_q_layer_major, &reg_q_layer_major_bytes);
        unreg(&reg_scales_root, &reg_scales_root_bytes);
        unreg(&reg_scales_linkage, &reg_scales_linkage_bytes);
        unreg(&reg_r_norm2, &reg_r_norm2_bytes);
        unreg(&reg_task_cluster_idx, &reg_task_cluster_idx_bytes);
        unreg(&reg_offsets_root_small, &reg_offsets_root_small_bytes);
        unreg(&reg_offsets_one, &reg_offsets_one_bytes);
        unreg(&reg_xCq_root0, &reg_xCq_root0_bytes);
        unreg(&reg_xCq_root_small, &reg_xCq_root_small_bytes);
        unreg(&reg_xCq_one, &reg_xCq_one_bytes);
        unreg(&reg_out_dists, &reg_out_dists_bytes);
        unreg(&reg_out_ids, &reg_out_ids_bytes);
    }

    void ResetCidCache(int nlist,
                       int cache_slots_in,
                       int max_nreal,
                       int max_nc,
                       int max_depth_offsets_len,
                       int m,
                       int m_codes) {
        cache_enabled = (cache_slots_in > 0) ? 1 : 0;
        cache_nlist = nlist;
        cache_slots = std::max(0, cache_slots_in);
        cache_max_nreal = max_nreal;
        cache_max_nc = max_nc;
        cache_max_depth_offsets_len = max_depth_offsets_len;
        cache_m = m;
        cache_m_codes = m_codes;
        cache_epoch = 1;
        cid_to_slot.assign(static_cast<std::size_t>(std::max(0, nlist)), -1);
        slot_to_cid.assign(static_cast<std::size_t>(cache_slots), -1);
        slot_epoch.assign(static_cast<std::size_t>(cache_slots), 0);
        h_clusters.assign(static_cast<std::size_t>(cache_slots), ClusterDescDev{});
        for (auto& cd : h_clusters) cd.cid = -1;
    }

    void EnsureOptin16384(std::size_t shmem_bytes);
};


__device__ __forceinline__ std::uint32_t FloatToOrderedUInt(float x) {
    const std::uint32_t u = __float_as_uint(x);
    const std::uint32_t mask = (u & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return u ^ mask;
}

__device__ __forceinline__ std::uint32_t ReadParent1BasedRaw(const std::uint8_t* parent_1based_raw,
                                                             int parent_elem_bytes,
                                                             int pos) {
    if (parent_elem_bytes == 2) {
        const auto* p16 = reinterpret_cast<const std::uint16_t*>(parent_1based_raw);
        return static_cast<std::uint32_t>(p16[pos]);
    }
    const auto* p32 = reinterpret_cast<const std::uint32_t*>(parent_1based_raw);
    return p32[pos];
}

__device__ __forceinline__ float OrderedUIntToFloat(std::uint32_t u) {
    const std::uint32_t mask = (u & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu;
    return __uint_as_float(u ^ mask);
}

__device__ __forceinline__ unsigned long long MakeKey(float dist, std::uint32_t id) {
    const auto hi = static_cast<unsigned long long>(FloatToOrderedUInt(dist));
    const auto lo = static_cast<unsigned long long>(id);
    return (hi << 32) | lo;
}

__device__ __forceinline__ float KeyDist(unsigned long long key) {
    return OrderedUIntToFloat(static_cast<std::uint32_t>(key >> 32));
}

__device__ __forceinline__ std::uint32_t KeyId(unsigned long long key) {
    return static_cast<std::uint32_t>(key & 0xFFFFFFFFull);
}

template <std::size_t A, std::size_t B>
struct ConstMax {
    static constexpr std::size_t value = (A > B) ? A : B;
};

template <std::size_t Align>
constexpr std::size_t AlignUp(std::size_t x) {
    return (x + (Align - 1)) & ~(Align - 1);
}

template <int MaxNC, int TileN>
constexpr std::size_t DiskLinkageLookupTopKTiledKernelShmemBytes() {
    constexpr int kBlockThreads = 256;
    constexpr int kGroupTiles = 2;
    constexpr int kMaxK = 256;
    using KeyT = unsigned long long;

    static_assert(TileN % kBlockThreads == 0, "TileN must be a multiple of block threads.");
    constexpr int kTileItemsPerThread = TileN / kBlockThreads;
    constexpr int kMergeItemsTotal = (kGroupTiles + 1) * kMaxK;
    static_assert(kMergeItemsTotal % kBlockThreads == 0, "Merge items must evenly divide block threads.");
    constexpr int kMergeItemsPerThread = kMergeItemsTotal / kBlockThreads;

    using TileSort = cub::BlockRadixSort<KeyT, kBlockThreads, kTileItemsPerThread>;
    using MergeSort = cub::BlockRadixSort<KeyT, kBlockThreads, kMergeItemsPerThread>;
    constexpr std::size_t kSortAlign =
        ConstMax<alignof(typename TileSort::TempStorage), alignof(typename MergeSort::TempStorage)>::value;
    constexpr std::size_t kSortBytes =
        ConstMax<sizeof(typename TileSort::TempStorage), sizeof(typename MergeSort::TempStorage)>::value;

    constexpr std::size_t base_bytes =
        sizeof(float) * static_cast<std::size_t>(MaxNC) +
        sizeof(KeyT) * static_cast<std::size_t>(kMaxK) +
        sizeof(KeyT) * static_cast<std::size_t>(kGroupTiles * kMaxK);

    constexpr std::size_t sort_off = AlignUp<kSortAlign>(base_bytes);
    return sort_off + kSortBytes;
}

template <int MaxNC, int TileN, int MaxK>
__global__ void DiskLinkageLookupTopKTiledKernel(const ClusterDescDev* __restrict clusters,
                                              int n_clusters,
                                              const std::uint32_t* __restrict real_ids_all,
                                              const std::uint8_t* __restrict parent_all,
                                              const std::uint32_t* __restrict depth_offsets_all,
                                              const std::uint8_t* __restrict codes_small_bytes_all,
                                              const std::uint8_t* __restrict code0_one_bytes_all,
                                              const std::uint8_t* __restrict virt_codes_small_bytes_all,
                                              const std::int8_t* __restrict q_layer_major_all,
                                              const float* __restrict scales_root_all,
                                              const float* __restrict scales_linkage_all,
                                              const float* __restrict r_norm2_all,
                                              const float* __restrict xCq_root0,      // nlist×qlen
                                              const float* __restrict xCq_root_small, // Hrs×qlen
                                              const float* __restrict xCq_one,        // Ho×qlen
                                              int nlist,
                                              int root_small_total_cols,
                                              int one_total_cols,
                                              const int* __restrict offsets_root_small, // m
                                              const int* __restrict offsets_one,        // m
                                              int m,
                                              int m_codes,
                                              const int* __restrict task_cluster_idx, // tasks_total
                                              int tasks_total,
                                              int qlen,
                                              int nprobe_cap,
                                              int k,
                                              int max_nreal,
                                              int max_nc,
                                              float* __restrict out_dists,          // tasks_total×k
                                              std::uint32_t* __restrict out_ids,   // tasks_total×k
                                              unsigned long long* __restrict phase_accum) {  // optional [4]
    constexpr int kBlockThreads = 256;
    constexpr int kGroupTiles = 2;
    static_assert(MaxK == 256, "GPU scan kernel assumes MaxK==256.");
    static_assert(kBlockThreads == 256, "GPU scan kernel assumes 256 threads.");
    static_assert(TileN % kBlockThreads == 0, "TileN must be a multiple of blockDim.x.");
    constexpr int kTileItemsPerThread = TileN / kBlockThreads;
    constexpr int kMergeItemsTotal = (kGroupTiles + 1) * MaxK;
    static_assert(kMergeItemsTotal % kBlockThreads == 0, "Merge items must evenly divide block threads.");
    constexpr int kMergeItemsPerThread = kMergeItemsTotal / kBlockThreads;

    using KeyT = unsigned long long;
    using TileSort = cub::BlockRadixSort<KeyT, kBlockThreads, kTileItemsPerThread>;
    using MergeSort = cub::BlockRadixSort<KeyT, kBlockThreads, kMergeItemsPerThread>;

    const int task = blockIdx.x;
    if (task >= tasks_total) return;
    const int cluster_idx = task_cluster_idx[task];
    const int qi = task / nprobe_cap;
    if (qi < 0 || qi >= qlen) return;

    const int out_base = task * k;
    const int tid = threadIdx.x;

    if (k > MaxK) {
        if (tid < k) {
            out_dists[out_base + tid] = INFINITY;
            out_ids[out_base + tid] = 0xFFFFFFFFu;
        }
        return;
    }

    if (cluster_idx < 0 || cluster_idx >= n_clusters) {
        if (tid < k) {
            out_dists[out_base + tid] = INFINITY;
            out_ids[out_base + tid] = 0xFFFFFFFFu;
        }
        return;
    }

    const ClusterDescDev cd = clusters[cluster_idx];
    if (cd.n_real <= 0 || cd.nc <= 0) {
        if (tid < k) {
            out_dists[out_base + tid] = INFINITY;
            out_ids[out_base + tid] = 0xFFFFFFFFu;
        }
        return;
    }
    if (cd.n_real > max_nreal || cd.nc > max_nc || cd.nc > MaxNC) {
        if (tid < k) {
            out_dists[out_base + tid] = INFINITY;
            out_ids[out_base + tid] = 0xFFFFFFFFu;
        }
        return;
    }
    if (cd.depth_offsets_len < 2) {
        if (tid < k) {
            out_dists[out_base + tid] = INFINITY;
            out_ids[out_base + tid] = 0xFFFFFFFFu;
        }
        return;
    }

    extern __shared__ unsigned char smem[];
    auto* dot_all = reinterpret_cast<float*>(smem); // MaxNC
    KeyT* best_keys = reinterpret_cast<KeyT*>(dot_all + MaxNC); // MaxK
    KeyT* group_keys = best_keys + MaxK; // kGroupTiles*MaxK

    auto* sort_ptr = reinterpret_cast<unsigned char*>(group_keys + kGroupTiles * MaxK);
    constexpr std::size_t kSortAlign =
        ConstMax<alignof(typename TileSort::TempStorage), alignof(typename MergeSort::TempStorage)>::value;
    const auto raw_off = static_cast<std::size_t>(sort_ptr - smem);
    const std::size_t sort_off = (raw_off + (kSortAlign - 1)) & ~(kSortAlign - 1);
    unsigned char* sort_base = smem + sort_off;
    auto* tile_temp = reinterpret_cast<typename TileSort::TempStorage*>(sort_base);
    auto* merge_temp = reinterpret_cast<typename MergeSort::TempStorage*>(sort_base);

    const std::uint32_t* __restrict real_ids = real_ids_all + cd.off_real_ids;
    const std::uint8_t* __restrict parent_1based_raw = parent_all + cd.off_parent;
    const std::uint32_t* __restrict depth_offsets = depth_offsets_all + cd.off_depth_offsets;
    const std::uint8_t* __restrict codes_small_bytes = codes_small_bytes_all + cd.off_codes_small;
    const std::uint8_t* __restrict code0_one_bytes = code0_one_bytes_all + cd.off_code0_one;
    const std::uint8_t* __restrict virt_codes_small_bytes = virt_codes_small_bytes_all + cd.off_virt_codes_small;
    const std::int8_t* __restrict q_layer_major = q_layer_major_all + cd.off_q_layer_major;
    const float* __restrict scales_root = scales_root_all + cd.off_scales_root;
    const float* __restrict scales_linkage = scales_linkage_all + cd.off_scales_linkage;
    const float* __restrict r_norm2 = r_norm2_all + cd.off_r_norm2;
    const int parent_elem_bytes = cd.parent_elem_bytes;

    const float* __restrict xCq_root0_col = xCq_root0 + static_cast<std::size_t>(qi) * static_cast<std::size_t>(nlist);
    const float* __restrict xCq_root_small_col =
        xCq_root_small + static_cast<std::size_t>(qi) * static_cast<std::size_t>(root_small_total_cols);
    const float* __restrict xCq_one_col =
        xCq_one + static_cast<std::size_t>(qi) * static_cast<std::size_t>(one_total_cols);

    unsigned long long t0 = 0;
    unsigned long long t1 = 0;
    unsigned long long t2 = 0;
    unsigned long long t3 = 0;
    if (phase_accum && tid == 0) t0 = clock64();

    // Initialize best-k to +inf (sorted).
    const KeyT inf_key = MakeKey(INFINITY, 0xFFFFFFFFu);
    for (int i = tid; i < MaxK; i += blockDim.x) {
        best_keys[i] = inf_key;
    }
    __syncthreads();

    const int n_real = cd.n_real;
    const int n_virt = cd.n_virt;
    const int nc = cd.nc;
    const int q_stride = (cd.q_stride > 0) ? cd.q_stride : nc;
    const int real_base = n_virt;
    const int n_root_real = cd.n_root_real;

    // dot_all DP (depth-parallel), identical math to CPU lookup path.
    const float root0_scaled = scales_root[0] * xCq_root0_col[cd.cid];
    const std::ptrdiff_t stride_codes = static_cast<std::ptrdiff_t>(m_codes);

    for (int v = tid; v < n_virt; v += blockDim.x) {
        float dot = root0_scaled * static_cast<float>(q_layer_major[v]);
        const std::uint8_t* row = virt_codes_small_bytes + static_cast<std::ptrdiff_t>(v) * stride_codes;
        for (int l = 1; l < m; ++l) {
            const std::int8_t* ql = q_layer_major + static_cast<std::ptrdiff_t>(l) * q_stride;
            const float* table = xCq_root_small_col + offsets_root_small[l];
            const float sl = scales_root[l];
            const int code = static_cast<int>(row[l - 1]);
            dot += static_cast<float>(ql[v]) * (sl * table[code]);
        }
        dot_all[v] = dot;
    }
    for (int pos = tid; pos < n_root_real; pos += blockDim.x) {
        const int local = real_base + pos;
        float dot = root0_scaled * static_cast<float>(q_layer_major[local]);
        const std::uint8_t* row = codes_small_bytes + static_cast<std::ptrdiff_t>(pos) * stride_codes;
        for (int l = 1; l < m; ++l) {
            const std::int8_t* ql = q_layer_major + static_cast<std::ptrdiff_t>(l) * q_stride;
            const float* table = xCq_root_small_col + offsets_root_small[l];
            const float sl = scales_root[l];
            const int code = static_cast<int>(row[l - 1]);
            dot += static_cast<float>(ql[local]) * (sl * table[code]);
        }
        dot_all[local] = dot;
    }
    __syncthreads();

    if (phase_accum && tid == 0) t1 = clock64();

    const int depth_len = cd.depth_offsets_len;
    for (int dep = 1; dep + 1 < depth_len; ++dep) {
        const int begin = static_cast<int>(depth_offsets[dep]);
        const int end = static_cast<int>(depth_offsets[dep + 1]);
        for (int pos = begin + tid; pos < end; pos += blockDim.x) {
            const int local = real_base + pos;
            const int p = static_cast<int>(ReadParent1BasedRaw(parent_1based_raw, parent_elem_bytes, pos)) - 1;
            float dot = dot_all[p];

            const int code0 = static_cast<int>(code0_one_bytes[pos]);
            const float s0 = scales_linkage[0];
            dot += static_cast<float>(q_layer_major[local]) * (s0 * xCq_one_col[offsets_one[0] + code0]);

            const std::uint8_t* row = codes_small_bytes + static_cast<std::ptrdiff_t>(pos) * stride_codes;
            for (int l = 1; l < m; ++l) {
                const std::int8_t* ql = q_layer_major + static_cast<std::ptrdiff_t>(l) * q_stride;
                const float* one_l = xCq_one_col + offsets_one[l];
                const float sl = scales_linkage[l];
                const int code = static_cast<int>(row[l - 1]);
                dot += static_cast<float>(ql[local]) * (sl * one_l[code]);
            }
            dot_all[local] = dot;
        }
        __syncthreads();
    }

    if (phase_accum && tid == 0) t2 = clock64();

    // Tile over real nodes:
    // - Per tile: compute keys=(dist,id), radix-sort, keep first MaxK ranks in group buffer.
    // - Every kGroupTiles tiles: merge (best + group) via radix-sort to update best.
    int group_count = 0;
    for (int base = 0; base < n_real; base += TileN) {
        KeyT keys[kTileItemsPerThread];
#pragma unroll
        for (int ii = 0; ii < kTileItemsPerThread; ++ii) {
            const int i = tid + ii * kBlockThreads;
            const int pos = base + i;
            if (pos < n_real) {
                const float dist = r_norm2[pos] - 2.0f * dot_all[real_base + pos];
                keys[ii] = MakeKey(dist, real_ids[pos]);
            } else {
                keys[ii] = inf_key;
            }
        }

        TileSort(*tile_temp).Sort(keys);
        __syncthreads();

#pragma unroll
        for (int ii = 0; ii < kTileItemsPerThread; ++ii) {
            const int rank = tid * kTileItemsPerThread + ii;
            if (rank < MaxK) {
                group_keys[group_count * MaxK + rank] = (rank < k) ? keys[ii] : inf_key;
            }
        }
        __syncthreads();

        ++group_count;
        if (group_count == kGroupTiles) {
            KeyT merge_keys[kMergeItemsPerThread];
#pragma unroll
            for (int ii = 0; ii < kMergeItemsPerThread; ++ii) {
                const int idx = tid * kMergeItemsPerThread + ii;
                if (idx < MaxK) {
                    merge_keys[ii] = best_keys[idx];
                } else {
                    merge_keys[ii] = group_keys[idx - MaxK];
                }
            }
            MergeSort(*merge_temp).Sort(merge_keys);
            __syncthreads();
#pragma unroll
            for (int ii = 0; ii < kMergeItemsPerThread; ++ii) {
                const int rank = tid * kMergeItemsPerThread + ii;
                if (rank < MaxK) {
                    best_keys[rank] = merge_keys[ii];
                }
            }
            __syncthreads();
            group_count = 0;
        }
    }

    if (group_count > 0) {
        for (int t = group_count; t < kGroupTiles; ++t) {
            for (int rank = tid; rank < MaxK; rank += blockDim.x) {
                group_keys[t * MaxK + rank] = inf_key;
            }
        }
        __syncthreads();

        KeyT merge_keys[kMergeItemsPerThread];
#pragma unroll
        for (int ii = 0; ii < kMergeItemsPerThread; ++ii) {
            const int idx = tid * kMergeItemsPerThread + ii;
            if (idx < MaxK) {
                merge_keys[ii] = best_keys[idx];
            } else {
                merge_keys[ii] = group_keys[idx - MaxK];
            }
        }
        MergeSort(*merge_temp).Sort(merge_keys);
        __syncthreads();
#pragma unroll
        for (int ii = 0; ii < kMergeItemsPerThread; ++ii) {
            const int rank = tid * kMergeItemsPerThread + ii;
            if (rank < MaxK) {
                best_keys[rank] = merge_keys[ii];
            }
        }
        __syncthreads();
    }

    if (phase_accum && tid == 0) {
        t3 = clock64();
        atomicAdd(&phase_accum[0], static_cast<unsigned long long>(t1 - t0));
        atomicAdd(&phase_accum[1], static_cast<unsigned long long>(t2 - t1));
        atomicAdd(&phase_accum[2], static_cast<unsigned long long>(t3 - t2));
        atomicAdd(&phase_accum[3], 1ull);
    }

    if (tid < k) {
        const KeyT key = best_keys[tid];
        out_dists[out_base + tid] = KeyDist(key);
        out_ids[out_base + tid] = KeyId(key);
    }
}

void ScanContext::EnsureOptin16384(std::size_t shmem_bytes) {
    if (shmem_bytes <= 48 * 1024) return;
    const int shmem_int = static_cast<int>(shmem_bytes);
    if (optin_16384_shmem_11 >= shmem_int) return;
    CudaCheck(cudaFuncSetAttribute(DiskLinkageLookupTopKTiledKernel<16384, 1024, 256>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   shmem_int),
              "cudaFuncSetAttribute (MaxDynamicSharedMemorySize)");
    // Tile256 is a separate kernel instantiation; it must opt-in too, otherwise launches can fail
    // with "invalid argument" when dynamic shared memory exceeds the default limit.
    CudaCheck(cudaFuncSetAttribute(DiskLinkageLookupTopKTiledKernel<16384, 256, 256>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   shmem_int),
              "cudaFuncSetAttribute (MaxDynamicSharedMemorySize)");
    optin_16384_shmem_11 = shmem_int;
}

inline std::size_t DiskLinkageLookupTopKShmemBytesFor(int kernel_max_nc, bool tile256) {
    const bool t256 = tile256;
    switch (kernel_max_nc) {
        case 512:
            return t256 ? DiskLinkageLookupTopKTiledKernelShmemBytes<512, 256>()
                        : DiskLinkageLookupTopKTiledKernelShmemBytes<512, 1024>();
        case 2048:
            return t256 ? DiskLinkageLookupTopKTiledKernelShmemBytes<2048, 256>()
                        : DiskLinkageLookupTopKTiledKernelShmemBytes<2048, 1024>();
        case 4096:
            return t256 ? DiskLinkageLookupTopKTiledKernelShmemBytes<4096, 256>()
                        : DiskLinkageLookupTopKTiledKernelShmemBytes<4096, 1024>();
        case 8192:
            return t256 ? DiskLinkageLookupTopKTiledKernelShmemBytes<8192, 256>()
                        : DiskLinkageLookupTopKTiledKernelShmemBytes<8192, 1024>();
        case 16384:
            return t256 ? DiskLinkageLookupTopKTiledKernelShmemBytes<16384, 256>()
                        : DiskLinkageLookupTopKTiledKernelShmemBytes<16384, 1024>();
        default:
            return t256 ? DiskLinkageLookupTopKTiledKernelShmemBytes<8192, 256>()
                        : DiskLinkageLookupTopKTiledKernelShmemBytes<8192, 1024>();
    }
}

}  // namespace

int DiskLinkageGpuScanMaxNcSupported() {
    // Conservative check: use the maximum shared-memory requirement across our supported TileN values.
    // If the device supports opting-in enough shared memory for MaxNC=16384, we expose 16384; otherwise 8192.
    constexpr std::size_t shmem_1024 = DiskLinkageLookupTopKTiledKernelShmemBytes<16384, 1024>();
    constexpr std::size_t shmem_256 = DiskLinkageLookupTopKTiledKernelShmemBytes<16384, 256>();
    const int want_shmem = static_cast<int>((shmem_1024 > shmem_256) ? shmem_1024 : shmem_256);

    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        return 8192;
    }
    int max_optin = 0;
    if (cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess) {
        return 8192;
    }
    return (max_optin >= want_shmem) ? 16384 : 8192;
}

bool ScanDiskLinkageLookupTopK(const STLQueryTables& qt,
                             int m,
                             int m_codes,
                             int nlist,
                             int root_small_total_cols,
                             int one_total_cols,
                             const int* offsets_root_small,
                             const int* offsets_one,
                             const std::vector<ClusterView>& active_views,
                             const std::vector<int>& task_cluster_idx,
                             int qlen,
                             int nprobe_cap,
	                             int k,
	                             int max_nc,
	                             int cache_mb,
	                             bool tile256,
                             std::vector<float>* out_dists,
                             std::vector<std::uint32_t>* out_ids,
	                             DiskLinkageGpuScanStats* stats,
	                             std::string* err) {
    try {
        if (!out_dists || !out_ids) {
            if (err) *err = "ScanDiskLinkageLookupTopK: output buffers are null.";
            return false;
        }
        if (m <= 1 || m_codes != m - 1) {
            if (err) *err = "ScanDiskLinkageLookupTopK: invalid m/m_codes.";
            return false;
        }
        if (qlen <= 0 || nprobe_cap <= 0 || k <= 0) {
            if (err) *err = "ScanDiskLinkageLookupTopK: invalid qlen/nprobe/k.";
            return false;
        }
        if (static_cast<int>(task_cluster_idx.size()) != qlen * nprobe_cap) {
            if (err) *err = "ScanDiskLinkageLookupTopK: task_cluster_idx size mismatch.";
            return false;
        }
        if (max_nc <= 0) {
            if (err) *err = "ScanDiskLinkageLookupTopK: invalid max_nc.";
            return false;
        }
        const int max_nreal = max_nc;  // nc = n_real + n_virt, and n_virt is always small; keep a single safety bound.
        if (k > 256) {
            if (err) *err = "ScanDiskLinkageLookupTopK: GPU path supports 1<=k<=256.";
            return false;
        }
        if (max_nc > 16384) {
            if (err) *err = "ScanDiskLinkageLookupTopK: max_nc out of supported range (1..16384).";
            return false;
        }
        const int device_max_nc = DiskLinkageGpuScanMaxNcSupported();
        if (device_max_nc > 0 && max_nc > device_max_nc) {
            if (err) {
                *err = "ScanDiskLinkageLookupTopK: max_nc exceeds device shared-memory supported max_nc (" +
                       std::to_string(device_max_nc) + ").";
            }
            return false;
        }
        if (cache_mb < 0) {
            if (err) *err = "ScanDiskLinkageLookupTopK: cache_mb must be >=0.";
            return false;
        }

        // Eval contract: codes are stored as uint8 payloads.
        constexpr int kSmallCodeBytes = 1;
        constexpr int kCode0Bytes = 1;
	
	        const int tasks_total = qlen * nprobe_cap;
	        out_dists->resize(static_cast<std::size_t>(tasks_total) * static_cast<std::size_t>(k));
	        out_ids->resize(static_cast<std::size_t>(tasks_total) * static_cast<std::size_t>(k));
	
	        if (stats) {
	            stats->tasks_total = tasks_total;
	            stats->tasks_gpu = 0;
	            stats->task_cluster_idx_bytes = sizeof(int) * static_cast<std::uint64_t>(task_cluster_idx.size());
	            stats->out_d2h_bytes =
	                static_cast<std::uint64_t>(tasks_total) * static_cast<std::uint64_t>(k) *
	                (sizeof(float) + sizeof(std::uint32_t));
	        }

        // Count GPU tasks.
        int tasks_gpu = 0;
        for (int t : task_cluster_idx) if (t >= 0) ++tasks_gpu;
        if (stats) stats->tasks_gpu = tasks_gpu;
        if (tasks_gpu == 0) {
            // No eligible tasks: return empty top-k.
            const std::size_t total_out = out_dists->size();
            for (std::size_t i = 0; i < total_out; ++i) {
                (*out_dists)[i] = INFINITY;
                (*out_ids)[i] = 0xFFFFFFFFu;
            }
            return true;
        }

        static thread_local ScanContext ctx;
        ctx.EnsureStream();
        const bool prof = (stats != nullptr);
        if (prof) ctx.EnsureEvents();
        if (stats) {
            stats->clusters_total = static_cast<int>(active_views.size());
            stats->clusters_gpu = 0;
            stats->kernel_max_nc = 0;
        }

        Timer t_host_pack;

        auto& clusters_h = ctx.h_clusters;

        int max_nc_needed = 0;
        int kernel_max_nc = 512;
        std::vector<std::uint8_t> used_active(static_cast<std::size_t>(active_views.size()), 0);
        for (int idx : task_cluster_idx) {
            if (idx >= 0 && idx < static_cast<int>(active_views.size())) {
                used_active[static_cast<std::size_t>(idx)] = 1;
            }
        }
        int clusters_gpu = 0;
        for (std::size_t i = 0; i < active_views.size(); ++i) {
            if (used_active[i]) {
                ++clusters_gpu;
                max_nc_needed = std::max(max_nc_needed, active_views[i].nc);
            }
        }
        if (max_nc_needed <= 512) kernel_max_nc = 512;
        else if (max_nc_needed <= 2048) kernel_max_nc = 2048;
        else if (max_nc_needed <= 4096) kernel_max_nc = 4096;
        else if (max_nc_needed <= 8192) kernel_max_nc = 8192;
        else kernel_max_nc = 16384;

        if (stats) {
            stats->clusters_gpu = clusters_gpu;
            stats->kernel_max_nc = kernel_max_nc;
        }

        constexpr int kCacheMaxDepthOffsetsLen = 512;
        // Cache policy:
        // - Controlled solely by `cache_mb` (MiB). 0 disables caching.
        // - Effective slots = min(nlist, floor(cache_mb / slot_mb)).
        int cache_slots = (cache_mb > 0) ? std::max(1, nlist) : 0;
        if (cache_mb > 0) {
            const std::uint64_t cache_bytes = static_cast<std::uint64_t>(cache_mb) * 1024ull * 1024ull;
            // Match the fixed-stride device layout used by the cache path.
            const std::uint64_t slot_bytes =
                static_cast<std::uint64_t>(max_nreal) * (4u + 4u + 4u) + // real_ids + parent + r_norm2
                static_cast<std::uint64_t>(kCacheMaxDepthOffsetsLen) * 4u + // depth_offsets (fixed cap)
                static_cast<std::uint64_t>(max_nreal) * static_cast<std::uint64_t>(m_codes) *
                    static_cast<std::uint64_t>(kSmallCodeBytes) + // codes_small
                static_cast<std::uint64_t>(max_nreal) * static_cast<std::uint64_t>(kCode0Bytes) + // code0_one
                static_cast<std::uint64_t>(max_nc) * static_cast<std::uint64_t>(m_codes) *
                    static_cast<std::uint64_t>(kSmallCodeBytes) + // virt_codes_small
                static_cast<std::uint64_t>(m) * static_cast<std::uint64_t>(max_nc) * 1u + // q_layer_major
                static_cast<std::uint64_t>(2u) * static_cast<std::uint64_t>(m) * 4u; // scales_root+linkage

            if (slot_bytes > 0) {
                const std::uint64_t slots_by_mb = cache_bytes / slot_bytes;
                cache_slots = static_cast<int>(std::min<std::uint64_t>(
                    static_cast<std::uint64_t>(std::max(0, cache_slots)),
                    std::min<std::uint64_t>(static_cast<std::uint64_t>(nlist), slots_by_mb)));
            }
        }
        bool use_cache = (cache_slots > 0);
        int n_clusters_for_kernel = static_cast<int>(active_views.size());
        std::vector<const ClusterView*> upload_cvs;

        if (use_cache) {
            // If a cluster has an unusually large depth_offsets array, fall back to the original packing path.
            for (std::size_t i = 0; i < active_views.size(); ++i) {
                if (!used_active[i]) continue;
                if (active_views[i].depth_offsets_len > kCacheMaxDepthOffsetsLen) {
                    use_cache = false;
                    break;
                }
            }
        }

        if (use_cache) {
            const bool cache_params_ok =
                (ctx.cache_enabled != 0) &&
                (ctx.cache_nlist == nlist) &&
                (ctx.cache_slots == cache_slots) &&
                (ctx.cache_max_nreal == max_nreal) &&
                (ctx.cache_max_nc == max_nc) &&
                (ctx.cache_max_depth_offsets_len == kCacheMaxDepthOffsetsLen) &&
                (ctx.cache_m == m) &&
                (ctx.cache_m_codes == m_codes);
            if (!cache_params_ok) {
                ctx.ResetCidCache(nlist, cache_slots, max_nreal, max_nc, kCacheMaxDepthOffsetsLen,
                                  m, m_codes);
            }

            n_clusters_for_kernel = ctx.cache_slots;
            upload_cvs.reserve(static_cast<std::size_t>(clusters_gpu));
            ctx.h_task_cluster_idx_slots.assign(task_cluster_idx.size(), -1);

            auto alloc_slot = [&]() -> int {
                // First try to find a free slot.
                for (int s = 0; s < ctx.cache_slots; ++s) {
                    if (ctx.slot_to_cid[static_cast<std::size_t>(s)] < 0) {
                        return s;
                    }
                }
                // Evict the least-recently-used slot.
                int best = 0;
                std::uint64_t best_epoch = ctx.slot_epoch.empty() ? 0 : ctx.slot_epoch[0];
                for (int s = 1; s < ctx.cache_slots; ++s) {
                    const std::uint64_t e = ctx.slot_epoch[static_cast<std::size_t>(s)];
                    if (e < best_epoch) {
                        best = s;
                        best_epoch = e;
                    }
                }
                const int old_cid = ctx.slot_to_cid[static_cast<std::size_t>(best)];
                if (old_cid >= 0 && old_cid < ctx.cache_nlist) {
                    ctx.cid_to_slot[static_cast<std::size_t>(old_cid)] = -1;
                }
                ctx.slot_to_cid[static_cast<std::size_t>(best)] = -1;
                ctx.slot_epoch[static_cast<std::size_t>(best)] = 0;
                ctx.h_clusters[static_cast<std::size_t>(best)] = ClusterDescDev{};
                ctx.h_clusters[static_cast<std::size_t>(best)].cid = -1;
                return best;
            };

            // Assign/lookup slots for tasks; collect first-seen clusters for upload.
            for (int task = 0; task < tasks_total; ++task) {
                const int idx = task_cluster_idx[static_cast<std::size_t>(task)];
                if (idx < 0 || idx >= static_cast<int>(active_views.size())) continue;
                const ClusterView& cv = active_views[static_cast<std::size_t>(idx)];
                const int cid = cv.cid;
                if (cid < 0 || cid >= nlist) continue;
                int slot = ctx.cid_to_slot[static_cast<std::size_t>(cid)];
                if (slot < 0) {
                    slot = alloc_slot();
                    ctx.cid_to_slot[static_cast<std::size_t>(cid)] = slot;
                    ctx.slot_to_cid[static_cast<std::size_t>(slot)] = cid;
                    upload_cvs.push_back(&cv);
                }
                ctx.slot_epoch[static_cast<std::size_t>(slot)] = ctx.cache_epoch++;
                ctx.h_task_cluster_idx_slots[static_cast<std::size_t>(task)] = slot;
            }

            const std::uint32_t stride_codes_small = static_cast<std::uint32_t>(max_nreal) *
                                                     static_cast<std::uint32_t>(m_codes) *
                                                     static_cast<std::uint32_t>(kSmallCodeBytes);
            const std::uint32_t stride_code0_one = static_cast<std::uint32_t>(max_nreal) *
                                                   static_cast<std::uint32_t>(kCode0Bytes);
            const std::uint32_t stride_virt_codes_small = static_cast<std::uint32_t>(max_nc) *
                                                          static_cast<std::uint32_t>(m_codes) *
                                                          static_cast<std::uint32_t>(kSmallCodeBytes);
            const std::uint32_t stride_q_layer_major = static_cast<std::uint32_t>(m) *
                                                       static_cast<std::uint32_t>(max_nc);

            // Update host cluster descriptors for new uploads.
            for (const ClusterView* cvp : upload_cvs) {
                const ClusterView& cv = *cvp;
                const int cid = cv.cid;
                const int slot = (cid >= 0 && cid < nlist) ? ctx.cid_to_slot[static_cast<std::size_t>(cid)] : -1;
                if (slot < 0) continue;

                ClusterDescDev cd;
                cd.cid = cid;
                cd.n_real = cv.n_real;
                cd.n_virt = cv.n_virt;
                cd.nc = cv.nc;
                cd.n_root_real = cv.n_root_real;
                cd.depth_offsets_len = cv.depth_offsets_len;
                cd.parent_elem_bytes = cv.parent_is_u16 ? static_cast<int>(sizeof(std::uint16_t))
                                                        : static_cast<int>(sizeof(std::uint32_t));
                cd.q_stride = max_nc;

                cd.off_real_ids = static_cast<std::uint32_t>(slot) * static_cast<std::uint32_t>(max_nreal);
                cd.off_parent = static_cast<std::uint32_t>(slot) * static_cast<std::uint32_t>(max_nreal) *
                                static_cast<std::uint32_t>(sizeof(std::uint32_t));
                cd.off_r_norm2 = static_cast<std::uint32_t>(slot) * static_cast<std::uint32_t>(max_nreal);
                cd.off_depth_offsets = static_cast<std::uint32_t>(slot) * static_cast<std::uint32_t>(kCacheMaxDepthOffsetsLen);
                cd.off_codes_small = static_cast<std::uint32_t>(slot) * stride_codes_small;
                cd.off_code0_one = static_cast<std::uint32_t>(slot) * stride_code0_one;
                cd.off_virt_codes_small = static_cast<std::uint32_t>(slot) * stride_virt_codes_small;
                cd.off_q_layer_major = static_cast<std::uint32_t>(slot) * stride_q_layer_major;
                cd.off_scales_root = static_cast<std::uint32_t>(slot) * static_cast<std::uint32_t>(m);
                cd.off_scales_linkage = static_cast<std::uint32_t>(slot) * static_cast<std::uint32_t>(m);

                ctx.h_clusters[static_cast<std::size_t>(slot)] = cd;
            }
        }

        std::uint64_t total_n_real = 0;
        std::uint64_t total_parent_bytes = 0;
        std::uint64_t total_depth_offsets = 0;
        std::uint64_t total_codes_small_bytes = 0;
        std::uint64_t total_code0_one_bytes = 0;
        std::uint64_t total_virt_codes_small_bytes = 0;
        std::uint64_t total_qbytes = 0;
        if (!use_cache) {
            clusters_h.resize(active_views.size());
            for (std::size_t i = 0; i < active_views.size(); ++i) {
                if (!used_active[i]) continue;
                const ClusterView& cv = active_views[i];
                total_n_real += static_cast<std::uint64_t>(cv.n_real);
                total_parent_bytes += static_cast<std::uint64_t>(cv.n_real) *
                                      static_cast<std::uint64_t>(cv.parent_is_u16 ? sizeof(std::uint16_t)
                                                                                  : sizeof(std::uint32_t));
                total_depth_offsets += static_cast<std::uint64_t>(cv.depth_offsets_len);
                total_codes_small_bytes += static_cast<std::uint64_t>(cv.n_real) *
                                           static_cast<std::uint64_t>(m_codes) *
                                           static_cast<std::uint64_t>(kSmallCodeBytes);
                total_code0_one_bytes += static_cast<std::uint64_t>(cv.n_real) *
                                         static_cast<std::uint64_t>(kCode0Bytes);
                total_virt_codes_small_bytes += static_cast<std::uint64_t>(cv.n_virt) *
                                                static_cast<std::uint64_t>(m_codes) *
                                                static_cast<std::uint64_t>(kSmallCodeBytes);
                total_qbytes += static_cast<std::uint64_t>(m) * static_cast<std::uint64_t>(cv.nc);
            }
        }

        if (!use_cache) {
            auto& real_ids_h = ctx.h_real_ids;
            auto& parent_h = ctx.h_parent;
            auto& depth_offsets_h = ctx.h_depth_offsets;
            auto& codes_small_h = ctx.h_codes_small;
            auto& code0_one_h = ctx.h_code0_one;
            auto& virt_codes_small_h = ctx.h_virt_codes_small;
            auto& q_layer_major_h = ctx.h_q_layer_major;
            auto& scales_root_h = ctx.h_scales_root;
            auto& scales_linkage_h = ctx.h_scales_linkage;
            auto& r_norm2_h = ctx.h_r_norm2;

            real_ids_h.resize(static_cast<std::size_t>(total_n_real));
            parent_h.resize(static_cast<std::size_t>(total_parent_bytes));
            r_norm2_h.resize(static_cast<std::size_t>(total_n_real));
            depth_offsets_h.resize(static_cast<std::size_t>(total_depth_offsets));
            codes_small_h.resize(static_cast<std::size_t>(total_codes_small_bytes));
            code0_one_h.resize(static_cast<std::size_t>(total_code0_one_bytes));
            virt_codes_small_h.resize(static_cast<std::size_t>(total_virt_codes_small_bytes));
            q_layer_major_h.resize(static_cast<std::size_t>(total_qbytes));
            scales_root_h.resize(static_cast<std::size_t>(clusters_gpu) * static_cast<std::size_t>(m));
            scales_linkage_h.resize(static_cast<std::size_t>(clusters_gpu) * static_cast<std::size_t>(m));

            std::uint64_t off_real_ids = 0, off_parent = 0, off_depth_offsets = 0, off_codes_small = 0, off_code0_one = 0;
            std::uint64_t off_virt_codes_small = 0, off_q_layer_major = 0;
            std::uint64_t off_scales_root = 0, off_scales_linkage = 0, off_r_norm2 = 0;

            for (std::size_t i = 0; i < active_views.size(); ++i) {
                const ClusterView& cv = active_views[i];
                if (!used_active[i]) {
                    ClusterDescDev cd;
                    cd.cid = cv.cid;
                    clusters_h[i] = cd;
                    continue;
                }
                ClusterDescDev cd;
                cd.cid = cv.cid;
                cd.n_real = cv.n_real;
                cd.n_virt = cv.n_virt;
                cd.nc = cv.nc;
                cd.n_root_real = cv.n_root_real;
                cd.depth_offsets_len = cv.depth_offsets_len;
                cd.parent_elem_bytes = cv.parent_is_u16 ? static_cast<int>(sizeof(std::uint16_t))
                                                        : static_cast<int>(sizeof(std::uint32_t));
                cd.q_stride = cv.nc;

                cd.off_real_ids = static_cast<std::uint32_t>(off_real_ids);
                cd.off_parent = static_cast<std::uint32_t>(off_parent);
                cd.off_depth_offsets = static_cast<std::uint32_t>(off_depth_offsets);
                cd.off_codes_small = static_cast<std::uint32_t>(off_codes_small);
                cd.off_code0_one = static_cast<std::uint32_t>(off_code0_one);
                cd.off_virt_codes_small = static_cast<std::uint32_t>(off_virt_codes_small);
                cd.off_q_layer_major = static_cast<std::uint32_t>(off_q_layer_major);
                cd.off_scales_root = static_cast<std::uint32_t>(off_scales_root);
                cd.off_scales_linkage = static_cast<std::uint32_t>(off_scales_linkage);
                cd.off_r_norm2 = static_cast<std::uint32_t>(off_r_norm2);

                clusters_h[i] = cd;

                if (cv.n_real <= 0) {
                    if (err) *err = "ScanDiskLinkageLookupTopK: GPU task references an empty cluster.";
                    return false;
                }
                {
                    if (cv.depth_offsets_len <= 0 || !cv.depth_offsets) {
                        if (err) *err = "ScanDiskLinkageLookupTopK: missing depth_offsets.";
                        return false;
                    }
                    std::memcpy(real_ids_h.data() + static_cast<std::size_t>(off_real_ids),
                                cv.real_ids,
                                sizeof(std::uint32_t) * static_cast<std::size_t>(cv.n_real));
                    if (cv.parent_is_u16) {
                        std::memcpy(parent_h.data() + static_cast<std::size_t>(off_parent),
                                    cv.parent_1based_u16,
                                    sizeof(std::uint16_t) * static_cast<std::size_t>(cv.n_real));
                    } else {
                        std::memcpy(parent_h.data() + static_cast<std::size_t>(off_parent),
                                    cv.parent_1based,
                                    sizeof(std::uint32_t) * static_cast<std::size_t>(cv.n_real));
                    }
                    std::memcpy(r_norm2_h.data() + static_cast<std::size_t>(off_r_norm2),
                                cv.r_norm2,
                                sizeof(float) * static_cast<std::size_t>(cv.n_real));
                    std::memcpy(depth_offsets_h.data() + static_cast<std::size_t>(off_depth_offsets),
                                cv.depth_offsets,
                                sizeof(std::uint32_t) * static_cast<std::size_t>(cv.depth_offsets_len));
                    off_real_ids += static_cast<std::uint64_t>(cv.n_real);
                    off_parent += static_cast<std::uint64_t>(cv.n_real) *
                                  static_cast<std::uint64_t>(cv.parent_is_u16 ? sizeof(std::uint16_t)
                                                                              : sizeof(std::uint32_t));
                    off_r_norm2 += static_cast<std::uint64_t>(cv.n_real);
                    off_depth_offsets += static_cast<std::uint64_t>(cv.depth_offsets_len);

                    const std::size_t cs_bytes =
                        static_cast<std::size_t>(cv.n_real) * static_cast<std::size_t>(m_codes) * static_cast<std::size_t>(kSmallCodeBytes);
                    std::memcpy(codes_small_h.data() + static_cast<std::size_t>(off_codes_small),
                                cv.codes_small_bytes,
                                sizeof(std::uint8_t) * cs_bytes);
                    off_codes_small += static_cast<std::uint64_t>(cs_bytes);

                    const std::size_t c0_bytes =
                        static_cast<std::size_t>(cv.n_real) * static_cast<std::size_t>(kCode0Bytes);
                    std::memcpy(code0_one_h.data() + static_cast<std::size_t>(off_code0_one),
                                cv.code0_one_bytes,
                                sizeof(std::uint8_t) * c0_bytes);
                    off_code0_one += static_cast<std::uint64_t>(c0_bytes);
                }
                if (cv.n_virt > 0) {
                    const std::size_t vbytes =
                        static_cast<std::size_t>(cv.n_virt) * static_cast<std::size_t>(m_codes) * static_cast<std::size_t>(kSmallCodeBytes);
                    std::memcpy(virt_codes_small_h.data() + static_cast<std::size_t>(off_virt_codes_small),
                                cv.virt_codes_small_bytes,
                                sizeof(std::uint8_t) * vbytes);
                    off_virt_codes_small += static_cast<std::uint64_t>(vbytes);
                }
                if (!cv.q_layer_major || !cv.scales_root || !cv.scales_linkage) {
                    if (err) *err = "ScanDiskLinkageLookupTopK: expected int8 coeff codec buffers (q_layer_major/scales).";
                    return false;
                }
                const std::size_t qbytes = static_cast<std::size_t>(m) * static_cast<std::size_t>(cv.nc);
                std::memcpy(q_layer_major_h.data() + static_cast<std::size_t>(off_q_layer_major),
                            cv.q_layer_major,
                            sizeof(std::int8_t) * qbytes);
                off_q_layer_major += static_cast<std::uint64_t>(qbytes);

                std::memcpy(scales_root_h.data() + static_cast<std::size_t>(off_scales_root),
                            cv.scales_root,
                            sizeof(float) * static_cast<std::size_t>(m));
                std::memcpy(scales_linkage_h.data() + static_cast<std::size_t>(off_scales_linkage),
                            cv.scales_linkage,
                            sizeof(float) * static_cast<std::size_t>(m));
                off_scales_root += static_cast<std::uint64_t>(m);
                off_scales_linkage += static_cast<std::uint64_t>(m);
            }

            // Register host staging buffers as pinned (only on capacity/size changes).
            stlq::eval::cuda::ScanContext::EnsurePinned(clusters_h.data(), sizeof(ClusterDescDev) * clusters_h.size(), &ctx.reg_clusters, &ctx.reg_clusters_bytes);
            stlq::eval::cuda::ScanContext::EnsurePinned(real_ids_h.data(), sizeof(std::uint32_t) * real_ids_h.size(), &ctx.reg_real_ids, &ctx.reg_real_ids_bytes);
            stlq::eval::cuda::ScanContext::EnsurePinned(parent_h.data(), sizeof(std::uint8_t) * parent_h.size(), &ctx.reg_parent, &ctx.reg_parent_bytes);
            stlq::eval::cuda::ScanContext::EnsurePinned(depth_offsets_h.data(), sizeof(std::uint32_t) * depth_offsets_h.size(), &ctx.reg_depth_offsets, &ctx.reg_depth_offsets_bytes);
            stlq::eval::cuda::ScanContext::EnsurePinned(codes_small_h.data(), sizeof(std::uint8_t) * codes_small_h.size(), &ctx.reg_codes_small, &ctx.reg_codes_small_bytes);
            stlq::eval::cuda::ScanContext::EnsurePinned(code0_one_h.data(), sizeof(std::uint8_t) * code0_one_h.size(), &ctx.reg_code0_one, &ctx.reg_code0_one_bytes);
            stlq::eval::cuda::ScanContext::EnsurePinned(virt_codes_small_h.data(), sizeof(std::uint8_t) * virt_codes_small_h.size(), &ctx.reg_virt_codes_small, &ctx.reg_virt_codes_small_bytes);
            stlq::eval::cuda::ScanContext::EnsurePinned(q_layer_major_h.data(), sizeof(std::int8_t) * q_layer_major_h.size(), &ctx.reg_q_layer_major, &ctx.reg_q_layer_major_bytes);
            stlq::eval::cuda::ScanContext::EnsurePinned(scales_root_h.data(), sizeof(float) * scales_root_h.size(), &ctx.reg_scales_root, &ctx.reg_scales_root_bytes);
            stlq::eval::cuda::ScanContext::EnsurePinned(scales_linkage_h.data(), sizeof(float) * scales_linkage_h.size(), &ctx.reg_scales_linkage, &ctx.reg_scales_linkage_bytes);
            stlq::eval::cuda::ScanContext::EnsurePinned(r_norm2_h.data(), sizeof(float) * r_norm2_h.size(), &ctx.reg_r_norm2, &ctx.reg_r_norm2_bytes);
        } else {
            // Cache mode: do not pin large per-qblk staging buffers; ensure old pins are released.
            auto unreg = [](void** p, std::size_t* n) {
                if (*p) {
                    cudaHostUnregister(*p);
                    *p = nullptr;
                    *n = 0;
                }
            };
            unreg(&ctx.reg_real_ids, &ctx.reg_real_ids_bytes);
            unreg(&ctx.reg_parent, &ctx.reg_parent_bytes);
            unreg(&ctx.reg_depth_offsets, &ctx.reg_depth_offsets_bytes);
            unreg(&ctx.reg_codes_small, &ctx.reg_codes_small_bytes);
            unreg(&ctx.reg_code0_one, &ctx.reg_code0_one_bytes);
            unreg(&ctx.reg_virt_codes_small, &ctx.reg_virt_codes_small_bytes);
            unreg(&ctx.reg_q_layer_major, &ctx.reg_q_layer_major_bytes);
            unreg(&ctx.reg_scales_root, &ctx.reg_scales_root_bytes);
            unreg(&ctx.reg_scales_linkage, &ctx.reg_scales_linkage_bytes);
            unreg(&ctx.reg_r_norm2, &ctx.reg_r_norm2_bytes);

            stlq::eval::cuda::ScanContext::EnsurePinned(clusters_h.data(), sizeof(ClusterDescDev) * clusters_h.size(), &ctx.reg_clusters, &ctx.reg_clusters_bytes);
            stlq::eval::cuda::ScanContext::EnsurePinned(ctx.h_task_cluster_idx_slots.data(),
                             sizeof(int) * ctx.h_task_cluster_idx_slots.size(),
                             &ctx.reg_task_cluster_idx,
                             &ctx.reg_task_cluster_idx_bytes);
        }

        if (stats) {
            stats->host_pack_cpu_sec = t_host_pack.ElapsedSeconds();
        }

        Timer t_alloc;
        auto& d_clusters = ctx.d_clusters;
        auto& d_real_ids = ctx.d_real_ids;
        auto& d_parent = ctx.d_parent;
        auto& d_depth_offsets = ctx.d_depth_offsets;
        auto& d_codes_small = ctx.d_codes_small;
        auto& d_code0_one = ctx.d_code0_one;
        auto& d_virt_codes_small = ctx.d_virt_codes_small;
        auto& d_q_layer_major = ctx.d_q_layer_major;
        auto& d_scales_root = ctx.d_scales_root;
        auto& d_scales_linkage = ctx.d_scales_linkage;
        auto& d_r_norm2 = ctx.d_r_norm2;

        auto& d_offsets_root_small = ctx.d_offsets_root_small;
        auto& d_offsets_one = ctx.d_offsets_one;
        auto& d_task_cluster_idx = ctx.d_task_cluster_idx;

        auto& d_xCq_root0 = ctx.d_xCq_root0;
        auto& d_xCq_root_small = ctx.d_xCq_root_small;
        auto& d_xCq_one = ctx.d_xCq_one;

	        auto& d_out_dists = ctx.d_out_dists;
	        auto& d_out_ids = ctx.d_out_ids;

        auto& real_ids_h = ctx.h_real_ids;
        auto& parent_h = ctx.h_parent;
        auto& depth_offsets_h = ctx.h_depth_offsets;
        auto& codes_small_h = ctx.h_codes_small;
        auto& code0_one_h = ctx.h_code0_one;
        auto& virt_codes_small_h = ctx.h_virt_codes_small;
        auto& q_layer_major_h = ctx.h_q_layer_major;
        auto& scales_root_h = ctx.h_scales_root;
        auto& scales_linkage_h = ctx.h_scales_linkage;
        auto& r_norm2_h = ctx.h_r_norm2;

        if (!use_cache) {
            d_clusters.Ensure(static_cast<std::size_t>(n_clusters_for_kernel));
            d_real_ids.Ensure(real_ids_h.size());
            d_parent.Ensure(parent_h.size());
            d_depth_offsets.Ensure(depth_offsets_h.size());
            d_codes_small.Ensure(codes_small_h.size());
            d_code0_one.Ensure(code0_one_h.size());
            d_virt_codes_small.Ensure(virt_codes_small_h.size());
            d_q_layer_major.Ensure(q_layer_major_h.size());
            d_scales_root.Ensure(scales_root_h.size());
            d_scales_linkage.Ensure(scales_linkage_h.size());
            d_r_norm2.Ensure(r_norm2_h.size());
        } else {
            d_clusters.Ensure(static_cast<std::size_t>(n_clusters_for_kernel));
            d_real_ids.Ensure(static_cast<std::size_t>(n_clusters_for_kernel) * static_cast<std::size_t>(max_nreal));
            d_parent.Ensure(static_cast<std::size_t>(n_clusters_for_kernel) *
                            static_cast<std::size_t>(max_nreal) * sizeof(std::uint32_t));
            d_r_norm2.Ensure(static_cast<std::size_t>(n_clusters_for_kernel) * static_cast<std::size_t>(max_nreal));
            d_depth_offsets.Ensure(static_cast<std::size_t>(n_clusters_for_kernel) * static_cast<std::size_t>(kCacheMaxDepthOffsetsLen));
            d_codes_small.Ensure(static_cast<std::size_t>(n_clusters_for_kernel) *
                                 static_cast<std::size_t>(max_nreal) *
                                 static_cast<std::size_t>(m_codes) *
                                 static_cast<std::size_t>(kSmallCodeBytes));
            d_code0_one.Ensure(static_cast<std::size_t>(n_clusters_for_kernel) *
                               static_cast<std::size_t>(max_nreal) *
                               static_cast<std::size_t>(kCode0Bytes));
            d_virt_codes_small.Ensure(static_cast<std::size_t>(n_clusters_for_kernel) *
                                      static_cast<std::size_t>(max_nc) *
                                      static_cast<std::size_t>(m_codes) *
                                      static_cast<std::size_t>(kSmallCodeBytes));
            d_q_layer_major.Ensure(static_cast<std::size_t>(n_clusters_for_kernel) *
                                   static_cast<std::size_t>(m) *
                                   static_cast<std::size_t>(max_nc));
            d_scales_root.Ensure(static_cast<std::size_t>(n_clusters_for_kernel) * static_cast<std::size_t>(m));
            d_scales_linkage.Ensure(static_cast<std::size_t>(n_clusters_for_kernel) * static_cast<std::size_t>(m));
        }

        d_offsets_root_small.Ensure(static_cast<std::size_t>(m));
        d_offsets_one.Ensure(static_cast<std::size_t>(m));
        d_task_cluster_idx.Ensure(task_cluster_idx.size());

        d_xCq_root0.Ensure(static_cast<std::size_t>(nlist) * static_cast<std::size_t>(qlen));
        d_xCq_root_small.Ensure(static_cast<std::size_t>(root_small_total_cols) * static_cast<std::size_t>(qlen));
        d_xCq_one.Ensure(static_cast<std::size_t>(one_total_cols) * static_cast<std::size_t>(qlen));

	        d_out_dists.Ensure(static_cast<std::size_t>(tasks_total) * static_cast<std::size_t>(k));
	        d_out_ids.Ensure(static_cast<std::size_t>(tasks_total) * static_cast<std::size_t>(k));

        if (stats) stats->alloc_sec = t_alloc.ElapsedSeconds();

        if (stats) {
            stats->cache_enabled = use_cache ? 1 : 0;
            stats->cache_slots = use_cache ? cache_slots : 0;
            if (use_cache) {
                const int misses = static_cast<int>(upload_cvs.size());
                const int hits = std::max(0, clusters_gpu - misses);
                std::uint64_t upload_bytes = 0;
                for (const ClusterView* cvp : upload_cvs) {
                    const ClusterView& cv = *cvp;
                    upload_bytes += static_cast<std::uint64_t>(cv.n_real) * 4u; // real_ids
                    upload_bytes += static_cast<std::uint64_t>(cv.n_real) *
                                    static_cast<std::uint64_t>(cv.parent_is_u16 ? sizeof(std::uint16_t)
                                                                                : sizeof(std::uint32_t));
                    upload_bytes += static_cast<std::uint64_t>(cv.n_real) * 4u; // r_norm2
                    upload_bytes += static_cast<std::uint64_t>(cv.depth_offsets_len) * 4u;
                    upload_bytes += static_cast<std::uint64_t>(cv.n_real) * static_cast<std::uint64_t>(m_codes) *
                                    static_cast<std::uint64_t>(kSmallCodeBytes);
                    upload_bytes += static_cast<std::uint64_t>(cv.n_real) * static_cast<std::uint64_t>(kCode0Bytes);
                    upload_bytes += static_cast<std::uint64_t>(cv.n_virt) * static_cast<std::uint64_t>(m_codes) *
                                    static_cast<std::uint64_t>(kSmallCodeBytes);
                    upload_bytes += static_cast<std::uint64_t>(m) * static_cast<std::uint64_t>(cv.nc); // q_layer_major
                    upload_bytes += static_cast<std::uint64_t>(2u) * static_cast<std::uint64_t>(m) * 4u; // scales_root+linkage
                }
                stats->cache_hits = hits;
                stats->cache_misses = misses;
                stats->cache_upload_clusters = misses;
                stats->cache_upload_bytes = upload_bytes;
            } else {
                stats->cache_hits = 0;
                stats->cache_misses = 0;
                stats->cache_upload_clusters = 0;
                stats->cache_upload_bytes = 0;
            }
        }

        if (prof) CudaCheck(cudaEventRecord(ctx.ev_pack_start, ctx.stream), "cudaEventRecord (pack_start)");

        if (use_cache && !upload_cvs.empty()) {
            const auto stride_real = static_cast<std::size_t>(max_nreal);
            const auto stride_parent_bytes = static_cast<std::size_t>(max_nreal) * sizeof(std::uint32_t);
            const auto stride_depth = static_cast<std::size_t>(kCacheMaxDepthOffsetsLen);
            const std::size_t stride_codes_small =
                static_cast<std::size_t>(max_nreal) * static_cast<std::size_t>(m_codes) * static_cast<std::size_t>(kSmallCodeBytes);
            const std::size_t stride_code0_one =
                static_cast<std::size_t>(max_nreal) * static_cast<std::size_t>(kCode0Bytes);
            const std::size_t stride_virt_codes_small =
                static_cast<std::size_t>(max_nc) * static_cast<std::size_t>(m_codes) * static_cast<std::size_t>(kSmallCodeBytes);
            const std::size_t stride_q_layer_major = static_cast<std::size_t>(m) * static_cast<std::size_t>(max_nc);
            const auto stride_scales = static_cast<std::size_t>(m);
            for (const ClusterView* cvp : upload_cvs) {
                const ClusterView& cv = *cvp;
                const int cid = cv.cid;
                if (cid < 0 || cid >= nlist) continue;
                const int slot = ctx.cid_to_slot[static_cast<std::size_t>(cid)];
                if (slot < 0 || slot >= n_clusters_for_kernel) continue;

                if (!cv.real_ids || (!cv.parent_is_u16 && !cv.parent_1based) ||
                    (cv.parent_is_u16 && !cv.parent_1based_u16) ||
                    !cv.depth_offsets || !cv.codes_small_bytes || !cv.code0_one_bytes || !cv.r_norm2) {
                    if (err) *err = "ScanDiskLinkageLookupTopK(cache): missing linkage_list buffers.";
                    return false;
                }
                if (!cv.q_layer_major || !cv.scales_root || !cv.scales_linkage) {
                    if (err) *err = "ScanDiskLinkageLookupTopK(cache): expected int8 coeff codec buffers.";
                    return false;
                }

                const std::size_t base_real = static_cast<std::size_t>(slot) * stride_real;
                const std::size_t base_parent = static_cast<std::size_t>(slot) * stride_parent_bytes;
                const std::size_t base_depth = static_cast<std::size_t>(slot) * stride_depth;
                const std::size_t base_codes_small = static_cast<std::size_t>(slot) * stride_codes_small;
                const std::size_t base_code0_one = static_cast<std::size_t>(slot) * stride_code0_one;
                const std::size_t base_virt_codes_small = static_cast<std::size_t>(slot) * stride_virt_codes_small;
                const std::size_t base_q = static_cast<std::size_t>(slot) * stride_q_layer_major;
                const std::size_t base_scales = static_cast<std::size_t>(slot) * stride_scales;

                CudaCheck(cudaMemcpyAsync(d_real_ids.ptr + base_real, cv.real_ids,
                                          sizeof(std::uint32_t) * static_cast<std::size_t>(cv.n_real),
                                          cudaMemcpyHostToDevice, ctx.stream),
                          "MemcpyAsync(cache) real_ids");
                CudaCheck(cudaMemcpyAsync(d_parent.ptr + base_parent,
                                          cv.parent_is_u16
                                              ? reinterpret_cast<const std::uint8_t*>(cv.parent_1based_u16)
                                              : reinterpret_cast<const std::uint8_t*>(cv.parent_1based),
                                          static_cast<std::size_t>(cv.n_real) *
                                              (cv.parent_is_u16 ? sizeof(std::uint16_t) : sizeof(std::uint32_t)),
                                          cudaMemcpyHostToDevice, ctx.stream),
                          "MemcpyAsync(cache) parent");
                CudaCheck(cudaMemcpyAsync(d_r_norm2.ptr + base_real, cv.r_norm2,
                                          sizeof(float) * static_cast<std::size_t>(cv.n_real),
                                          cudaMemcpyHostToDevice, ctx.stream),
                          "MemcpyAsync(cache) r_norm2");
                CudaCheck(cudaMemcpyAsync(d_depth_offsets.ptr + base_depth, cv.depth_offsets,
                                          sizeof(std::uint32_t) * static_cast<std::size_t>(cv.depth_offsets_len),
                                          cudaMemcpyHostToDevice, ctx.stream),
                          "MemcpyAsync(cache) depth_offsets");

                const std::size_t cs_bytes =
                    static_cast<std::size_t>(cv.n_real) * static_cast<std::size_t>(m_codes) * static_cast<std::size_t>(kSmallCodeBytes);
                CudaCheck(cudaMemcpyAsync(d_codes_small.ptr + base_codes_small, cv.codes_small_bytes,
                                          sizeof(std::uint8_t) * cs_bytes,
                                          cudaMemcpyHostToDevice, ctx.stream),
                          "MemcpyAsync(cache) codes_small");
                const std::size_t c0_bytes =
                    static_cast<std::size_t>(cv.n_real) * static_cast<std::size_t>(kCode0Bytes);
                CudaCheck(cudaMemcpyAsync(d_code0_one.ptr + base_code0_one, cv.code0_one_bytes,
                                          sizeof(std::uint8_t) * c0_bytes,
                                          cudaMemcpyHostToDevice, ctx.stream),
                          "MemcpyAsync(cache) code0_one");

                if (cv.n_virt > 0) {
                    if (!cv.virt_codes_small_bytes) {
                        if (err) *err = "ScanDiskLinkageLookupTopK(cache): missing virt_codes_small_bytes.";
                        return false;
                    }
                    const std::size_t vbytes =
                        static_cast<std::size_t>(cv.n_virt) * static_cast<std::size_t>(m_codes) * static_cast<std::size_t>(kSmallCodeBytes);
                    CudaCheck(cudaMemcpyAsync(d_virt_codes_small.ptr + base_virt_codes_small, cv.virt_codes_small_bytes,
                                              sizeof(std::uint8_t) * vbytes,
                                              cudaMemcpyHostToDevice, ctx.stream),
                              "MemcpyAsync(cache) virt_codes_small");
                }

                // Cache layout stores q_layer_major with per-layer stride max_nc (ClusterDescDev::q_stride).
                // Source buffer is packed tight with stride cv.nc.
                for (int l = 0; l < m; ++l) {
                    const std::size_t src_off = static_cast<std::size_t>(l) * static_cast<std::size_t>(cv.nc);
                    const std::size_t dst_off = static_cast<std::size_t>(l) * static_cast<std::size_t>(max_nc);
                    CudaCheck(cudaMemcpyAsync(d_q_layer_major.ptr + base_q + dst_off,
                                              cv.q_layer_major + src_off,
                                              sizeof(std::int8_t) * static_cast<std::size_t>(cv.nc),
                                              cudaMemcpyHostToDevice, ctx.stream),
                              "MemcpyAsync(cache) q_layer_major(layer)");
                }
                CudaCheck(cudaMemcpyAsync(d_scales_root.ptr + base_scales, cv.scales_root,
                                          sizeof(float) * static_cast<std::size_t>(m),
                                          cudaMemcpyHostToDevice, ctx.stream),
                          "MemcpyAsync(cache) scales_root");
                CudaCheck(cudaMemcpyAsync(d_scales_linkage.ptr + base_scales, cv.scales_linkage,
                                          sizeof(float) * static_cast<std::size_t>(m),
                                          cudaMemcpyHostToDevice, ctx.stream),
                          "MemcpyAsync(cache) scales_linkage");
            }
        }

        const bool need_clusters_h2d = (!use_cache) || (!upload_cvs.empty());
        if (need_clusters_h2d) {
            CudaCheck(cudaMemcpyAsync(d_clusters.ptr, clusters_h.data(),
                                      sizeof(ClusterDescDev) * clusters_h.size(),
                                      cudaMemcpyHostToDevice, ctx.stream),
                      "MemcpyAsync clusters");
        }
        if (!use_cache) {
            if (!real_ids_h.empty()) {
                CudaCheck(cudaMemcpyAsync(d_real_ids.ptr, real_ids_h.data(),
                                          sizeof(std::uint32_t) * real_ids_h.size(),
                                          cudaMemcpyHostToDevice, ctx.stream),
                          "MemcpyAsync real_ids");
                CudaCheck(cudaMemcpyAsync(d_parent.ptr, parent_h.data(),
                                          sizeof(std::uint8_t) * parent_h.size(),
                                          cudaMemcpyHostToDevice, ctx.stream),
                          "MemcpyAsync parent");
                CudaCheck(cudaMemcpyAsync(d_depth_offsets.ptr, depth_offsets_h.data(),
                                          sizeof(std::uint32_t) * depth_offsets_h.size(),
                                          cudaMemcpyHostToDevice, ctx.stream),
                          "MemcpyAsync depth_offsets");
                CudaCheck(cudaMemcpyAsync(d_r_norm2.ptr, r_norm2_h.data(),
                                          sizeof(float) * r_norm2_h.size(),
                                          cudaMemcpyHostToDevice, ctx.stream),
                          "MemcpyAsync r_norm2");
            }
            if (!codes_small_h.empty()) {
                CudaCheck(cudaMemcpyAsync(d_codes_small.ptr, codes_small_h.data(),
                                          sizeof(std::uint8_t) * codes_small_h.size(),
                                          cudaMemcpyHostToDevice, ctx.stream),
                          "MemcpyAsync codes_small");
            }
            if (!code0_one_h.empty()) {
                CudaCheck(cudaMemcpyAsync(d_code0_one.ptr, code0_one_h.data(),
                                          sizeof(std::uint8_t) * code0_one_h.size(),
                                          cudaMemcpyHostToDevice, ctx.stream),
                          "MemcpyAsync code0_one");
            }
            if (!virt_codes_small_h.empty()) {
                CudaCheck(cudaMemcpyAsync(d_virt_codes_small.ptr, virt_codes_small_h.data(),
                                          sizeof(std::uint8_t) * virt_codes_small_h.size(),
                                          cudaMemcpyHostToDevice, ctx.stream),
                          "MemcpyAsync virt_codes_small");
            }
            if (!q_layer_major_h.empty()) {
                CudaCheck(cudaMemcpyAsync(d_q_layer_major.ptr, q_layer_major_h.data(),
                                          sizeof(std::int8_t) * q_layer_major_h.size(),
                                          cudaMemcpyHostToDevice, ctx.stream),
                          "MemcpyAsync q_layer_major");
            }
            if (!scales_root_h.empty()) {
                CudaCheck(cudaMemcpyAsync(d_scales_root.ptr, scales_root_h.data(),
                                          sizeof(float) * scales_root_h.size(),
                                          cudaMemcpyHostToDevice, ctx.stream),
                          "MemcpyAsync scales_root");
                CudaCheck(cudaMemcpyAsync(d_scales_linkage.ptr, scales_linkage_h.data(),
                                          sizeof(float) * scales_linkage_h.size(),
                                          cudaMemcpyHostToDevice, ctx.stream),
                          "MemcpyAsync scales_linkage");
            }
        }
        const std::size_t offsets_bytes = sizeof(int) * static_cast<std::size_t>(m);
        const bool offsets_root_dirty =
            (ctx.h_offsets_root_small_cache.size() != static_cast<std::size_t>(m)) ||
            (std::memcmp(ctx.h_offsets_root_small_cache.data(), offsets_root_small, offsets_bytes) != 0);
        if (offsets_root_dirty) {
            ctx.h_offsets_root_small_cache.assign(offsets_root_small, offsets_root_small + m);
            CudaCheck(cudaMemcpyAsync(d_offsets_root_small.ptr, ctx.h_offsets_root_small_cache.data(),
                                      offsets_bytes,
                                      cudaMemcpyHostToDevice, ctx.stream),
                      "MemcpyAsync offsets_root_small");
        }
        const bool offsets_one_dirty =
            (ctx.h_offsets_one_cache.size() != static_cast<std::size_t>(m)) ||
            (std::memcmp(ctx.h_offsets_one_cache.data(), offsets_one, offsets_bytes) != 0);
        if (offsets_one_dirty) {
            ctx.h_offsets_one_cache.assign(offsets_one, offsets_one + m);
            CudaCheck(cudaMemcpyAsync(d_offsets_one.ptr, ctx.h_offsets_one_cache.data(),
                                      offsets_bytes,
                                      cudaMemcpyHostToDevice, ctx.stream),
                      "MemcpyAsync offsets_one");
        }
        // IMPORTANT: Only pin host buffers that are owned by ScanContext and are guaranteed to
        // outlive this call. Pinning external pointers (caller stack/temporaries) can break
        // repeat runs when those pointers become invalid.
        const int* task_cluster_src = nullptr;
        if (use_cache) {
            task_cluster_src = ctx.h_task_cluster_idx_slots.data();
        } else {
            ctx.h_task_cluster_idx_slots = task_cluster_idx;
            task_cluster_src = ctx.h_task_cluster_idx_slots.data();
        }
        const int* offsets_root_src = ctx.h_offsets_root_small_cache.data();
        const int* offsets_one_src = ctx.h_offsets_one_cache.data();

        stlq::eval::cuda::ScanContext::EnsurePinned(const_cast<int*>(task_cluster_src),
                         sizeof(int) * ctx.h_task_cluster_idx_slots.size(),
                         &ctx.reg_task_cluster_idx,
                         &ctx.reg_task_cluster_idx_bytes);
        stlq::eval::cuda::ScanContext::EnsurePinned(const_cast<int*>(offsets_root_src),
                         sizeof(int) * static_cast<std::size_t>(m),
                         &ctx.reg_offsets_root_small,
                         &ctx.reg_offsets_root_small_bytes);
        stlq::eval::cuda::ScanContext::EnsurePinned(const_cast<int*>(offsets_one_src),
                         sizeof(int) * static_cast<std::size_t>(m),
                         &ctx.reg_offsets_one,
                         &ctx.reg_offsets_one_bytes);

        CudaCheck(cudaMemcpyAsync(d_task_cluster_idx.ptr, task_cluster_src,
                                  sizeof(int) * task_cluster_idx.size(),
                                  cudaMemcpyHostToDevice, ctx.stream),
                  "MemcpyAsync task_cluster_idx");
        if (prof) CudaCheck(cudaEventRecord(ctx.ev_pack_end, ctx.stream), "cudaEventRecord (pack_end)");

        if (prof) CudaCheck(cudaEventRecord(ctx.ev_tables_start, ctx.stream), "cudaEventRecord (tables_start)");
        CudaCheck(cudaMemcpyAsync(d_xCq_root0.ptr, qt.xCq_root0.data.data(),
                                  sizeof(float) * static_cast<std::size_t>(nlist) * static_cast<std::size_t>(qlen),
                                  cudaMemcpyHostToDevice, ctx.stream),
                  "MemcpyAsync xCq_root0");
        CudaCheck(cudaMemcpyAsync(d_xCq_root_small.ptr, qt.xCq_root_small.data.data(),
                                  sizeof(float) * static_cast<std::size_t>(root_small_total_cols) * static_cast<std::size_t>(qlen),
                                  cudaMemcpyHostToDevice, ctx.stream),
                  "MemcpyAsync xCq_root_small");
        CudaCheck(cudaMemcpyAsync(d_xCq_one.ptr, qt.xCq_one.data.data(),
                                  sizeof(float) * static_cast<std::size_t>(one_total_cols) * static_cast<std::size_t>(qlen),
                                  cudaMemcpyHostToDevice, ctx.stream),
                  "MemcpyAsync xCq_one");
        if (prof) CudaCheck(cudaEventRecord(ctx.ev_tables_end, ctx.stream), "cudaEventRecord (tables_end)");

        const dim3 blocks(tasks_total);
        const dim3 threads(256);

        const std::size_t shmem = DiskLinkageLookupTopKShmemBytesFor(kernel_max_nc, tile256);
        unsigned long long* phase_ptr = nullptr;
        if (prof) {
            ctx.EnsureDeviceClock();
            ctx.d_kernel_phase_accum.Ensure(4);
            CudaCheck(cudaMemsetAsync(ctx.d_kernel_phase_accum.ptr,
                                      0,
                                      sizeof(unsigned long long) * 4,
                                      ctx.stream),
                      "cudaMemsetAsync kernel_phase_accum");
            phase_ptr = ctx.d_kernel_phase_accum.ptr;
        }
        if (prof) CudaCheck(cudaEventRecord(ctx.ev_kernel_start, ctx.stream), "cudaEventRecord (kernel_start)");
        if (kernel_max_nc == 16384) {
            ctx.EnsureOptin16384(shmem);
        }

        // Select TileN at compile time to avoid hot-branching inside the kernel.
#define STLQ_LAUNCH_TILE(TN)                                                                                          \
    switch (kernel_max_nc) {                                                                                            \
        case 512:                                                                                                       \
            DiskLinkageLookupTopKTiledKernel<512, TN, 256><<<blocks, threads, shmem, ctx.stream>>>(                       \
                d_clusters.ptr, n_clusters_for_kernel,                                                                  \
                d_real_ids.ptr, d_parent.ptr, d_depth_offsets.ptr,                                                      \
                d_codes_small.ptr, d_code0_one.ptr, d_virt_codes_small.ptr,                                             \
                d_q_layer_major.ptr, d_scales_root.ptr, d_scales_linkage.ptr, d_r_norm2.ptr,                              \
                d_xCq_root0.ptr, d_xCq_root_small.ptr, d_xCq_one.ptr,                                                   \
                nlist, root_small_total_cols, one_total_cols,                                                           \
                d_offsets_root_small.ptr, d_offsets_one.ptr,                                                            \
                m, m_codes,                                                                                             \
                d_task_cluster_idx.ptr,                                                                                 \
                tasks_total, qlen, nprobe_cap, k,                                                                       \
                max_nreal, max_nc,                                                                                      \
                d_out_dists.ptr, d_out_ids.ptr, phase_ptr);                                                             \
            break;                                                                                                      \
        case 2048:                                                                                                      \
            DiskLinkageLookupTopKTiledKernel<2048, TN, 256><<<blocks, threads, shmem, ctx.stream>>>(                      \
                d_clusters.ptr, n_clusters_for_kernel,                                                                  \
                d_real_ids.ptr, d_parent.ptr, d_depth_offsets.ptr,                                                      \
                d_codes_small.ptr, d_code0_one.ptr, d_virt_codes_small.ptr,                                             \
                d_q_layer_major.ptr, d_scales_root.ptr, d_scales_linkage.ptr, d_r_norm2.ptr,                              \
                d_xCq_root0.ptr, d_xCq_root_small.ptr, d_xCq_one.ptr,                                                   \
                nlist, root_small_total_cols, one_total_cols,                                                           \
                d_offsets_root_small.ptr, d_offsets_one.ptr,                                                            \
                m, m_codes,                                                                                             \
                d_task_cluster_idx.ptr,                                                                                 \
                tasks_total, qlen, nprobe_cap, k,                                                                       \
                max_nreal, max_nc,                                                                                      \
                d_out_dists.ptr, d_out_ids.ptr, phase_ptr);                                                             \
            break;                                                                                                      \
        case 4096:                                                                                                      \
            DiskLinkageLookupTopKTiledKernel<4096, TN, 256><<<blocks, threads, shmem, ctx.stream>>>(                      \
                d_clusters.ptr, n_clusters_for_kernel,                                                                  \
                d_real_ids.ptr, d_parent.ptr, d_depth_offsets.ptr,                                                      \
                d_codes_small.ptr, d_code0_one.ptr, d_virt_codes_small.ptr,                                             \
                d_q_layer_major.ptr, d_scales_root.ptr, d_scales_linkage.ptr, d_r_norm2.ptr,                              \
                d_xCq_root0.ptr, d_xCq_root_small.ptr, d_xCq_one.ptr,                                                   \
                nlist, root_small_total_cols, one_total_cols,                                                           \
                d_offsets_root_small.ptr, d_offsets_one.ptr,                                                            \
                m, m_codes,                                                                                             \
                d_task_cluster_idx.ptr,                                                                                 \
                tasks_total, qlen, nprobe_cap, k,                                                                       \
                max_nreal, max_nc,                                                                                      \
                d_out_dists.ptr, d_out_ids.ptr, phase_ptr);                                                             \
            break;                                                                                                      \
        case 8192:                                                                                                      \
            DiskLinkageLookupTopKTiledKernel<8192, TN, 256><<<blocks, threads, shmem, ctx.stream>>>(                      \
                d_clusters.ptr, n_clusters_for_kernel,                                                                  \
                d_real_ids.ptr, d_parent.ptr, d_depth_offsets.ptr,                                                      \
                d_codes_small.ptr, d_code0_one.ptr, d_virt_codes_small.ptr,                                             \
                d_q_layer_major.ptr, d_scales_root.ptr, d_scales_linkage.ptr, d_r_norm2.ptr,                              \
                d_xCq_root0.ptr, d_xCq_root_small.ptr, d_xCq_one.ptr,                                                   \
                nlist, root_small_total_cols, one_total_cols,                                                           \
                d_offsets_root_small.ptr, d_offsets_one.ptr,                                                            \
                m, m_codes,                                                                                             \
                d_task_cluster_idx.ptr,                                                                                 \
                tasks_total, qlen, nprobe_cap, k,                                                                       \
                max_nreal, max_nc,                                                                                      \
                d_out_dists.ptr, d_out_ids.ptr, phase_ptr);                                                             \
            break;                                                                                                      \
        default:                                                                                                        \
            DiskLinkageLookupTopKTiledKernel<16384, TN, 256><<<blocks, threads, shmem, ctx.stream>>>(                     \
                d_clusters.ptr, n_clusters_for_kernel,                                                                  \
                d_real_ids.ptr, d_parent.ptr, d_depth_offsets.ptr,                                                      \
                d_codes_small.ptr, d_code0_one.ptr, d_virt_codes_small.ptr,                                             \
                d_q_layer_major.ptr, d_scales_root.ptr, d_scales_linkage.ptr, d_r_norm2.ptr,                              \
                d_xCq_root0.ptr, d_xCq_root_small.ptr, d_xCq_one.ptr,                                                   \
                nlist, root_small_total_cols, one_total_cols,                                                           \
                d_offsets_root_small.ptr, d_offsets_one.ptr,                                                            \
                m, m_codes,                                                                                             \
                d_task_cluster_idx.ptr,                                                                                 \
                tasks_total, qlen, nprobe_cap, k,                                                                       \
                max_nreal, max_nc,                                                                                      \
                d_out_dists.ptr, d_out_ids.ptr, phase_ptr);                                                             \
            break;                                                                                                      \
    }

        if (tile256) {
            STLQ_LAUNCH_TILE(256);
        } else {
            STLQ_LAUNCH_TILE(1024);
        }

#undef STLQ_LAUNCH_TILE
	        CudaCheck(cudaGetLastError(), "Launch DiskLinkageLookupTopKKernel");
	        if (prof) CudaCheck(cudaEventRecord(ctx.ev_kernel_end, ctx.stream), "cudaEventRecord (kernel_end)");
	
	        if (prof) CudaCheck(cudaEventRecord(ctx.ev_d2h_start, ctx.stream), "cudaEventRecord (d2h_start)");
	        CudaCheck(cudaMemcpyAsync(out_dists->data(), d_out_dists.ptr,
	                                  sizeof(float) * out_dists->size(),
	                                  cudaMemcpyDeviceToHost, ctx.stream),
	                  "MemcpyAsync out_dists D2H");
	        CudaCheck(cudaMemcpyAsync(out_ids->data(), d_out_ids.ptr,
	                                  sizeof(std::uint32_t) * out_ids->size(),
	                                  cudaMemcpyDeviceToHost, ctx.stream),
	                  "MemcpyAsync out_ids D2H");
        if (prof) CudaCheck(cudaEventRecord(ctx.ev_d2h_end, ctx.stream), "cudaEventRecord (d2h_end)");

        Timer t_sync;
        CudaCheck(cudaStreamSynchronize(ctx.stream), "cudaStreamSynchronize (DiskLinkageLookupTopK)");
        if (stats) stats->stream_sync_sec = t_sync.ElapsedSeconds();

        if (prof && stats && phase_ptr && ctx.dev_clock_khz > 0) {
            unsigned long long h_phase[4] = {0, 0, 0, 0};
            CudaCheck(cudaMemcpy(h_phase,
                                 ctx.d_kernel_phase_accum.ptr,
                                 sizeof(h_phase),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy kernel_phase_accum");
            const unsigned long long blocks_done = h_phase[3];
            if (blocks_done > 0) {
                const double hz = static_cast<double>(ctx.dev_clock_khz) * 1000.0;
                const double denom = hz * static_cast<double>(blocks_done);
                stats->kernel_roots_sec = static_cast<double>(h_phase[0]) / denom;
                stats->kernel_depth_sec = static_cast<double>(h_phase[1]) / denom;
                stats->kernel_topk_sec = static_cast<double>(h_phase[2]) / denom;
            }
        }

	        if (prof && stats) {
	            stats->clusters_pack_h2d_sec = ScanContext::ElapsedSeconds(ctx.ev_pack_start, ctx.ev_pack_end);
	            stats->query_tables_h2d_sec = ScanContext::ElapsedSeconds(ctx.ev_tables_start, ctx.ev_tables_end);
	            stats->kernel_sec = ScanContext::ElapsedSeconds(ctx.ev_kernel_start, ctx.ev_kernel_end);
	            stats->out_d2h_sec = ScanContext::ElapsedSeconds(ctx.ev_d2h_start, ctx.ev_d2h_end);
	        }

        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

}  // namespace stlq::eval::cuda

#else

namespace stlq::eval::cuda {
bool ScanDiskLinkageLookupTopK(const STLQueryTables&,
                             int,
                             int,
                             int,
                             int,
                             int,
                             const int*,
                             const int*,
                             const std::vector<ClusterView>&,
                             const std::vector<int>&,
	                             int,
	                             int,
	                             int,
	                             int,
	                             int,
	                             bool,
	                             std::vector<float>*,
                             std::vector<std::uint32_t>*,
                             DiskLinkageGpuScanStats*,
                             std::string* err) {
    if (err) *err = "ScanDiskLinkageLookupTopK: STLQ_ENABLE_CUDA is not enabled.";
    return false;
}

int DiskLinkageGpuScanMaxNcSupported() {
    return 0;
}
}  // namespace stlq::eval::cuda

#endif
