#include "stlq/eval/linkage_norm_provider_cuda.h"

#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "stlq/common/timer.h"
#include "stlq/eval/linkage_norm_provider.h"

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
    std::size_t n = 0;
    DeviceBuf() = default;
    DeviceBuf(const DeviceBuf&) = delete;
    DeviceBuf& operator=(const DeviceBuf&) = delete;
    ~DeviceBuf() { Reset(0); }
    void Reset(std::size_t n_in) {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
            n = 0;
        }
        if (n_in) {
            CudaCheck(cudaMalloc(&ptr, sizeof(T) * n_in), "cudaMalloc");
            n = n_in;
        }
    }
    void Ensure(std::size_t n_in) {
        if (n_in <= n) return;
        Reset(n_in);
    }
};

inline float DotCpu(const float* a, const float* b, int d) {
    float s = 0.0f;
    for (int i = 0; i < d; ++i) s += a[i] * b[i];
    return s;
}

void BuildKmeans1DInitialCentersHost(const float* data,
                                     int n,
                                     int k,
                                     std::vector<float>* centers) {
    if (!centers) {
        return;
    }
    centers->assign(static_cast<std::size_t>(std::max(0, k)), 0.0f);
    if (!data || n <= 0 || k <= 0) {
        return;
    }
    if (k == 1) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += data[i];
        }
        (*centers)[0] = static_cast<float>(sum / static_cast<double>(n));
        return;
    }
    std::vector<float> sorted(static_cast<std::size_t>(n));
    std::copy(data, data + n, sorted.begin());
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < k; ++i) {
        const int idx = static_cast<int>((static_cast<long long>(i) * n) / k);
        (*centers)[static_cast<std::size_t>(i)] =
            sorted[static_cast<std::size_t>(std::min(idx, n - 1))];
    }
}

__device__ __forceinline__ float MatGet(const float* __restrict mat, int rows, int i, int j) {
    return mat[static_cast<std::size_t>(j) * static_cast<std::size_t>(rows) + static_cast<std::size_t>(i)];
}

__device__ __forceinline__ int ReadSmallCode(const std::uint8_t* __restrict codes_bytes,
                                             std::size_t pos,
                                             int m_codes,
                                             int idx) {
    return static_cast<int>(codes_bytes[pos * static_cast<std::size_t>(m_codes) + static_cast<std::size_t>(idx)]);
}

__device__ __forceinline__ int ReadCode0One(const std::uint8_t* __restrict code0_bytes, std::size_t pos) {
    return static_cast<int>(code0_bytes[pos]);
}

template <bool UseQuantized>
__device__ __forceinline__ float ReadRootCoeff(const std::int8_t* __restrict q_layer_major,
                                               const float* __restrict scales_root,
                                               const float* __restrict a0_real,
                                               const float* __restrict coeffs_small_real,
                                               const float* __restrict virt_a0,
                                               const float* __restrict virt_coeffs_small,
                                               int local,
                                               int nc,
                                               int m_codes,
                                               bool is_virtual,
                                               int pos,
                                               int layer) {
    if constexpr (UseQuantized) {
        const std::size_t qidx =
            static_cast<std::size_t>(layer) * static_cast<std::size_t>(nc) + static_cast<std::size_t>(local);
        return scales_root[layer] * static_cast<float>(q_layer_major[qidx]);
    } else {
        if (layer == 0) {
            return is_virtual ? virt_a0[static_cast<std::size_t>(pos)]
                              : a0_real[static_cast<std::size_t>(pos)];
        }
        const std::size_t off = static_cast<std::size_t>(pos) * static_cast<std::size_t>(m_codes) +
                                static_cast<std::size_t>(layer - 1);
        return is_virtual ? virt_coeffs_small[off] : coeffs_small_real[off];
    }
}

template <bool UseQuantized>
__device__ __forceinline__ float ReadLinkageCoeff(const std::int8_t* __restrict q_layer_major,
                                                const float* __restrict scales_linkage,
                                                const float* __restrict a0_real,
                                                const float* __restrict coeffs_small_real,
                                                int local,
                                                int nc,
                                                int m_codes,
                                                int pos,
                                                int layer) {
    if constexpr (UseQuantized) {
        const std::size_t qidx =
            static_cast<std::size_t>(layer) * static_cast<std::size_t>(nc) + static_cast<std::size_t>(local);
        return scales_linkage[layer] * static_cast<float>(q_layer_major[qidx]);
    } else {
        if (layer == 0) {
            return a0_real[static_cast<std::size_t>(pos)];
        }
        const std::size_t off = static_cast<std::size_t>(pos) * static_cast<std::size_t>(m_codes) +
                                static_cast<std::size_t>(layer - 1);
        return coeffs_small_real[off];
    }
}

__device__ __forceinline__ int ReadParent1BasedRaw(const std::uint8_t* __restrict parent_bytes,
                                                   std::size_t pos,
                                                   int parent_elem_bytes) {
    if (parent_elem_bytes == 2) {
        const std::size_t off = pos * sizeof(std::uint16_t);
        const std::uint16_t v =
            static_cast<std::uint16_t>(parent_bytes[off]) |
            (static_cast<std::uint16_t>(parent_bytes[off + 1]) << 8);
        return static_cast<int>(v);
    }
    const std::size_t off = pos * sizeof(std::uint32_t);
    std::uint32_t v = 0;
    v |= static_cast<std::uint32_t>(parent_bytes[off]);
    v |= static_cast<std::uint32_t>(parent_bytes[off + 1]) << 8;
    v |= static_cast<std::uint32_t>(parent_bytes[off + 2]) << 16;
    v |= static_cast<std::uint32_t>(parent_bytes[off + 3]) << 24;
    return static_cast<int>(v);
}

template <bool UseQuantized>
__global__ void NormVirtRootsKernel(int m,
                                   int m_codes,
                                   int n_virt,
                                   int nc,
                                   const std::uint8_t* __restrict virt_codes_small_bytes,
                                   const std::int8_t* __restrict q_layer_major,  // m*nc or null
                                   const float* __restrict scales_root,          // m or null
                                   const float* __restrict virt_a0,
                                   const float* __restrict virt_coeffs_small,
                                   const int* __restrict offsets_root_small,     // m
                                   float c0_norm2,
                                   const float* __restrict c0_dot_rootS,  // Hrs
                                   const float* __restrict D_rr,           // Hrs×Hrs (col-major)
                                   int D_rr_rows,
                                   float* __restrict norm2_local) {  // nc
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_virt) return;
    const int local = v;
    float a0 = ReadRootCoeff<UseQuantized>(
        q_layer_major, scales_root,
        nullptr, nullptr,
        virt_a0, virt_coeffs_small,
        local, nc, m_codes, /*is_virtual=*/true, v, 0);

    int idx_rs[16];
    float a_rs[16];
    const auto vpos = static_cast<std::size_t>(v);
    for (int l = 1; l < m; ++l) {
        const int code = ReadSmallCode(virt_codes_small_bytes, vpos, m_codes, l - 1);
        idx_rs[l - 1] = offsets_root_small[l] + code;
        a_rs[l - 1] = ReadRootCoeff<UseQuantized>(
            q_layer_major, scales_root,
            nullptr, nullptr,
            virt_a0, virt_coeffs_small,
            local, nc, m_codes, /*is_virtual=*/true, v, l);
    }

    float cross0 = 0.0f;
    for (int l = 1; l < m; ++l) {
        cross0 += a_rs[l - 1] * c0_dot_rootS[static_cast<std::size_t>(idx_rs[l - 1])];
    }
    float small_term = 0.0f;
    for (int j = 1; j < m; ++j) {
        const float aj = a_rs[j - 1];
        const int fj = idx_rs[j - 1];
        small_term += (aj * aj) * MatGet(D_rr, D_rr_rows, fj, fj);
        for (int k = j + 1; k < m; ++k) {
            small_term +=
                (2.0f * aj * a_rs[k - 1]) * MatGet(D_rr, D_rr_rows, fj, idx_rs[k - 1]);
        }
    }
    norm2_local[local] = a0 * a0 * c0_norm2 + 2.0f * a0 * cross0 + small_term;
}

template <bool UseQuantized>
__global__ void NormRealRootsKernel(int m,
                                   int m_codes,
                                   int n_root_real,
                                   int n_virt,
                                   int nc,
                                   const std::uint8_t* __restrict codes_small_bytes,
                                   const std::int8_t* __restrict q_layer_major,  // m*nc or null
                                   const float* __restrict scales_root,          // m or null
                                   const float* __restrict a0_real,
                                   const float* __restrict coeffs_small_real,
                                   const int* __restrict offsets_root_small,     // m
                                   float c0_norm2,
                                   const float* __restrict c0_dot_rootS,  // Hrs
                                   const float* __restrict D_rr,           // Hrs×Hrs
                                   int D_rr_rows,
                                   float* __restrict norm2_local,  // nc
                                   float* __restrict out_norm2_real) {  // n_real
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= n_root_real) return;
    const int local = n_virt + pos;
    const auto p = static_cast<std::size_t>(pos);
    float a0 = ReadRootCoeff<UseQuantized>(
        q_layer_major, scales_root,
        a0_real, coeffs_small_real,
        nullptr, nullptr,
        local, nc, m_codes, /*is_virtual=*/false, pos, 0);

    int idx_rs[16];
    float a_rs[16];
    for (int l = 1; l < m; ++l) {
        const int code = ReadSmallCode(codes_small_bytes, p, m_codes, l - 1);
        idx_rs[l - 1] = offsets_root_small[l] + code;
        a_rs[l - 1] = ReadRootCoeff<UseQuantized>(
            q_layer_major, scales_root,
            a0_real, coeffs_small_real,
            nullptr, nullptr,
            local, nc, m_codes, /*is_virtual=*/false, pos, l);
    }

    float cross0 = 0.0f;
    for (int l = 1; l < m; ++l) {
        cross0 += a_rs[l - 1] * c0_dot_rootS[static_cast<std::size_t>(idx_rs[l - 1])];
    }
    float small_term = 0.0f;
    for (int j = 1; j < m; ++j) {
        const float aj = a_rs[j - 1];
        const int fj = idx_rs[j - 1];
        small_term += (aj * aj) * MatGet(D_rr, D_rr_rows, fj, fj);
        for (int k = j + 1; k < m; ++k) {
            small_term +=
                (2.0f * aj * a_rs[k - 1]) * MatGet(D_rr, D_rr_rows, fj, idx_rs[k - 1]);
        }
    }
    const float norm2 = a0 * a0 * c0_norm2 + 2.0f * a0 * cross0 + small_term;
    norm2_local[local] = norm2;
    out_norm2_real[pos] = norm2;
}

template <bool UseQuantized>
__global__ void NormLinkageDepthKernel(int m,
                                    int m_codes,
                                    int begin_pos,
                                    int end_pos,
                                    int n_real,
                                    int n_root_real,
                                    int n_virt,
                                    int nc,
                                    const std::uint8_t* __restrict parent_1based_raw,
                                    int parent_elem_bytes,
                                    const std::uint8_t* __restrict codes_small_bytes,      // n_real*m_codes
                                    const std::uint8_t* __restrict code0_one_bytes,        // n_real
                                    const std::uint8_t* __restrict virt_codes_small_bytes, // n_virt*m_codes
                                    const std::int8_t* __restrict q_layer_major,  // m*nc or null
                                    const float* __restrict scales_root,          // m or null
                                    const float* __restrict scales_linkage,         // m or null
                                    const float* __restrict a0_real,
                                    const float* __restrict coeffs_small_real,
                                    const float* __restrict virt_a0,
                                    const float* __restrict virt_coeffs_small,
                                    const int* __restrict offsets_root_small,     // m
                                    const int* __restrict offsets_one,            // m
                                    float c0_norm2,
                                    const float* __restrict c0_dot_one,    // Ho
                                    const float* __restrict D_oo,          // Ho×Ho
                                    int D_oo_rows,
                                    const float* __restrict D_ro,          // Hrs×Ho
                                    int D_ro_rows,
                                    float* __restrict norm2_local,         // nc
                                    float* __restrict out_norm2_real) {    // n_real
    const int pos = begin_pos + (blockIdx.x * blockDim.x + threadIdx.x);
    if (pos >= end_pos) return;
    if (pos < n_root_real || pos >= n_real) return;

    const auto ppos = static_cast<std::size_t>(pos);
    const int local = n_virt + pos;
    const int p = ReadParent1BasedRaw(parent_1based_raw, ppos, parent_elem_bytes) - 1;

    int idx_i[16];
    float b_i[16];

    idx_i[0] = offsets_one[0] + ReadCode0One(code0_one_bytes, ppos);
    b_i[0] = ReadLinkageCoeff<UseQuantized>(
        q_layer_major, scales_linkage,
        a0_real, coeffs_small_real,
        local, nc, m_codes, pos, 0);
    for (int l = 1; l < m; ++l) {
        const int code = ReadSmallCode(codes_small_bytes, ppos, m_codes, l - 1);
        idx_i[l] = offsets_one[l] + code;
        b_i[l] = ReadLinkageCoeff<UseQuantized>(
            q_layer_major, scales_linkage,
            a0_real, coeffs_small_real,
            local, nc, m_codes, pos, l);
    }

    float res_norm2 = 0.0f;
    for (int j = 0; j < m; ++j) {
        const float aj = b_i[j];
        const int fj = idx_i[j];
        res_norm2 += (aj * aj) * MatGet(D_oo, D_oo_rows, fj, fj);
        for (int k = j + 1; k < m; ++k) {
            res_norm2 += (2.0f * aj * b_i[k]) * MatGet(D_oo, D_oo_rows, fj, idx_i[k]);
        }
    }

    float cross = 0.0f;
    int a = p;
    const int real_base = n_virt;
    while (a >= 0) {
        if (a < real_base) {
            // virt root
            const int v = a;
            const auto vp = static_cast<std::size_t>(v);
            const float a0 = ReadRootCoeff<UseQuantized>(
                q_layer_major, scales_root,
                a0_real, coeffs_small_real,
                virt_a0, virt_coeffs_small,
                v, nc, m_codes, /*is_virtual=*/true, v, 0);
            float dot0 = 0.0f;
            for (int k = 0; k < m; ++k) {
                dot0 += b_i[k] * c0_dot_one[static_cast<std::size_t>(idx_i[k])];
            }
            cross += a0 * dot0;
            for (int l = 1; l < m; ++l) {
                const int code = ReadSmallCode(virt_codes_small_bytes, vp, m_codes, l - 1);
                const int idx_rs = offsets_root_small[l] + code;
                const float al = ReadRootCoeff<UseQuantized>(
                    q_layer_major, scales_root,
                    a0_real, coeffs_small_real,
                    virt_a0, virt_coeffs_small,
                    v, nc, m_codes, /*is_virtual=*/true, v, l);
                float acc = 0.0f;
                for (int k = 0; k < m; ++k) {
                    acc += b_i[k] * MatGet(D_ro, D_ro_rows, idx_rs, idx_i[k]);
                }
                cross += al * acc;
            }
            break;
        }

        const int a_real = a - real_base;
        if (a_real >= 0 && a_real < n_root_real) {
            // real root
            const auto rp = static_cast<std::size_t>(a_real);
            const float a0 = ReadRootCoeff<UseQuantized>(
                q_layer_major, scales_root,
                a0_real, coeffs_small_real,
                virt_a0, virt_coeffs_small,
                a, nc, m_codes, /*is_virtual=*/false, a_real, 0);
            float dot0 = 0.0f;
            for (int k = 0; k < m; ++k) {
                dot0 += b_i[k] * c0_dot_one[static_cast<std::size_t>(idx_i[k])];
            }
            cross += a0 * dot0;
            for (int l = 1; l < m; ++l) {
                const int code = ReadSmallCode(codes_small_bytes, rp, m_codes, l - 1);
                const int idx_rs = offsets_root_small[l] + code;
                const float al = ReadRootCoeff<UseQuantized>(
                    q_layer_major, scales_root,
                    a0_real, coeffs_small_real,
                    virt_a0, virt_coeffs_small,
                    a, nc, m_codes, /*is_virtual=*/false, a_real, l);
                float acc = 0.0f;
                for (int k = 0; k < m; ++k) {
                    acc += b_i[k] * MatGet(D_ro, D_ro_rows, idx_rs, idx_i[k]);
                }
                cross += al * acc;
            }
            break;
        }

        if (a_real < 0 || a_real >= n_real) {
            break;
        }

        // linkage ancestor residual dot
        int idx_u[16];
        float b_u[16];
        const auto up = static_cast<std::size_t>(a_real);

        idx_u[0] = offsets_one[0] + ReadCode0One(code0_one_bytes, up);
        b_u[0] = ReadLinkageCoeff<UseQuantized>(
            q_layer_major, scales_linkage,
            a0_real, coeffs_small_real,
            a, nc, m_codes, a_real, 0);
        for (int l = 1; l < m; ++l) {
            const int code = ReadSmallCode(codes_small_bytes, up, m_codes, l - 1);
            idx_u[l] = offsets_one[l] + code;
            b_u[l] = ReadLinkageCoeff<UseQuantized>(
                q_layer_major, scales_linkage,
                a0_real, coeffs_small_real,
                a, nc, m_codes, a_real, l);
        }

        float dot = 0.0f;
        for (int j = 0; j < m; ++j) {
            const float aj = b_u[j];
            const int fj = idx_u[j];
            for (int k = 0; k < m; ++k) {
                dot += (aj * b_i[k]) * MatGet(D_oo, D_oo_rows, fj, idx_i[k]);
            }
        }
        cross += dot;
        a = ReadParent1BasedRaw(parent_1based_raw, up, parent_elem_bytes) - 1;
    }

    float parent_norm = 0.0f;
    if (p >= 0) {
        parent_norm = norm2_local[p];
    }

    const float norm2 = parent_norm + res_norm2 + 2.0f * cross;
    norm2_local[local] = norm2;
    out_norm2_real[pos] = norm2;
}

__global__ void InitCentersLinearKernel(int k,
                                        float min_v,
                                        float max_v,
                                        float* __restrict centers) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k) return;
    if (k <= 1) {
        centers[0] = min_v;
        return;
    }
    const float t = static_cast<float>(j) / static_cast<float>(k - 1);
    centers[j] = min_v + t * (max_v - min_v);
}

__global__ void AssignKmeans1DAndAccumulateKernel(const float* __restrict data,
                                                  int n,
                                                  const float* __restrict centers,
                                                  int k,
                                                  std::uint8_t* __restrict assigns,
                                                  double* __restrict sums,
                                                  int* __restrict counts) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = data[i];
    float best = fabsf(x - centers[0]);
    int best_k = 0;
    for (int j = 1; j < k; ++j) {
        const float dist = fabsf(x - centers[j]);
        if (dist < best) {
            best = dist;
            best_k = j;
        }
    }
    assigns[i] = static_cast<std::uint8_t>(best_k);
    atomicAdd(sums + best_k, static_cast<double>(x));
    atomicAdd(counts + best_k, 1);
}

__global__ void AssignKmeans1DFinalKernel(const float* __restrict data,
                                          int n,
                                          const float* __restrict centers,
                                          int k,
                                          std::uint8_t* __restrict assigns) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = data[i];
    float best = fabsf(x - centers[0]);
    int best_k = 0;
    for (int j = 1; j < k; ++j) {
        const float dist = fabsf(x - centers[j]);
        if (dist < best) {
            best = dist;
            best_k = j;
        }
    }
    assigns[i] = static_cast<std::uint8_t>(best_k);
}

__global__ void UpdateKmeans1DCentersKernel(const float* __restrict centers_prev,
                                            float* __restrict centers_next,
                                            const double* __restrict sums,
                                            const int* __restrict counts,
                                            int k) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k) return;
    centers_next[j] = (counts[j] > 0)
        ? static_cast<float>(sums[j] / static_cast<double>(counts[j]))
        : centers_prev[j];
}

}  // namespace

struct LinkageNormProviderLookupCuda::Impl {
    LinkageNormProviderLookup cpu_;
    const ColMajorMatrix<float>& C_root0_;
    const CodebookMeta& meta_root_small_;
    const CodebookMeta& meta_one_;
    int gpu_max_nc_ = 512;
    bool profile_breakdown_ = false;
    LinkageNormProviderLookupCuda::Stats last_stats_{};
    cudaStream_t stream_ = nullptr;

    std::vector<int> offsets_root_small_;
    std::vector<int> offsets_one_;

    DeviceBuf<float> d_D_rr_;
    DeviceBuf<float> d_D_oo_;
    DeviceBuf<float> d_D_ro_;
    int D_rr_rows_ = 0;
    int D_oo_rows_ = 0;
    int D_ro_rows_ = 0;

    DeviceBuf<int> d_offsets_root_small_;
    DeviceBuf<int> d_offsets_one_;

    DeviceBuf<std::uint8_t> d_parent_;
    DeviceBuf<std::uint8_t> d_codes_small_;
    DeviceBuf<std::uint8_t> d_code0_one_;
    DeviceBuf<std::uint8_t> d_virt_codes_small_;
    DeviceBuf<std::int8_t> d_q_layer_major_;
    DeviceBuf<float> d_scales_root_;
    DeviceBuf<float> d_scales_linkage_;
    DeviceBuf<float> d_a0_real_;
    DeviceBuf<float> d_coeffs_small_real_;
    DeviceBuf<float> d_virt_a0_;
    DeviceBuf<float> d_virt_coeffs_small_;

    DeviceBuf<float> d_c0_dot_rootS_;
    DeviceBuf<float> d_c0_dot_one_;

    DeviceBuf<float> d_norm2_local_;
    DeviceBuf<float> d_norm2_real_;
    DeviceBuf<float> d_kmeans_centers_;
    DeviceBuf<float> d_kmeans_centers_next_;
    DeviceBuf<double> d_kmeans_sums_;
    DeviceBuf<int> d_kmeans_counts_;
    DeviceBuf<std::uint8_t> d_kmeans_assign_u8_;
    DeviceBuf<float> d_reduce_tmp_;
    DeviceBuf<std::uint8_t> d_cub_tmp_;

    Impl(const ColMajorMatrix<float>& C_root0,
         const CodebookMeta& meta_root_small,
         const CodebookMeta& meta_one,
         int gpu_max_nc,
         bool profile_breakdown)
        : cpu_(C_root0, meta_root_small, meta_one),
          C_root0_(C_root0),
          meta_root_small_(meta_root_small),
          meta_one_(meta_one) {
        gpu_max_nc_ = std::max(1, gpu_max_nc);
        profile_breakdown_ = profile_breakdown;
        last_stats_.prof_enabled = profile_breakdown_;
        CudaCheck(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");

        const int m = meta_one_.m;
        offsets_root_small_.assign(static_cast<std::size_t>(m), 0);
        for (int l = 1; l < m; ++l) {
            offsets_root_small_[static_cast<std::size_t>(l)] =
                meta_root_small_.offsets[static_cast<std::size_t>(l - 1)];
        }
        offsets_one_ = meta_one_.offsets;

        const auto& D_rr = cpu_.D_rr();
        const auto& D_oo = cpu_.D_oo();
        const auto& D_ro = cpu_.D_ro();

        D_rr_rows_ = D_rr.rows;
        D_oo_rows_ = D_oo.rows;
        D_ro_rows_ = D_ro.rows;

        d_D_rr_.Reset(D_rr.Size());
        d_D_oo_.Reset(D_oo.Size());
        d_D_ro_.Reset(D_ro.Size());
        CudaCheck(cudaMemcpy(d_D_rr_.ptr, D_rr.data.data(), sizeof(float) * D_rr.Size(), cudaMemcpyHostToDevice),
                  "cudaMemcpy D_rr");
        CudaCheck(cudaMemcpy(d_D_oo_.ptr, D_oo.data.data(), sizeof(float) * D_oo.Size(), cudaMemcpyHostToDevice),
                  "cudaMemcpy D_oo");
        CudaCheck(cudaMemcpy(d_D_ro_.ptr, D_ro.data.data(), sizeof(float) * D_ro.Size(), cudaMemcpyHostToDevice),
                  "cudaMemcpy D_ro");

        d_offsets_root_small_.Reset(offsets_root_small_.size());
        d_offsets_one_.Reset(offsets_one_.size());
        CudaCheck(cudaMemcpy(d_offsets_root_small_.ptr,
                             offsets_root_small_.data(),
                             sizeof(int) * offsets_root_small_.size(),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy offsets_root_small");
        CudaCheck(cudaMemcpy(d_offsets_one_.ptr,
                             offsets_one_.data(),
                             sizeof(int) * offsets_one_.size(),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy offsets_one");

        d_c0_dot_rootS_.Reset(static_cast<std::size_t>(meta_root_small_.total_cols));
        d_c0_dot_one_.Reset(static_cast<std::size_t>(meta_one_.total_cols));
    }

    ~Impl() {
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    bool ComputeNorm2Gpu(const ClusterView& cv, std::vector<float>* out_r_norm2, std::string* err) {
        last_stats_ = LinkageNormProviderLookupCuda::Stats{};
        last_stats_.prof_enabled = profile_breakdown_;
        if (!out_r_norm2) return false;
        const int d = meta_one_.d;
        const int m = cv.m;
        const int m_codes = cv.m_codes;
        const bool use_quantized = (cv.q_layer_major != nullptr);
        if (cv.cid < 0 || cv.cid >= C_root0_.cols) {
            if (err) *err = "LinkageNormProviderLookupCuda: invalid cid.";
            return false;
        }
        if (cv.n_real <= 0 || cv.nc != cv.n_real + cv.n_virt) {
            if (err) *err = "LinkageNormProviderLookupCuda: invalid cluster sizes.";
            return false;
        }
        if (use_quantized) {
            if (!cv.scales_root || !cv.scales_linkage) {
                last_stats_.fallback_reason = 3;
                if (err) *err = "LinkageNormProviderLookupCuda: missing quantized coeff inputs.";
                return false;
            }
        } else {
            if (!cv.a0 || !cv.coeffs_small || (cv.n_virt > 0 && (!cv.virt_a0 || !cv.virt_coeffs_small))) {
                last_stats_.fallback_reason = 3;
                if (err) *err = "LinkageNormProviderLookupCuda: missing float coeff inputs.";
                return false;
            }
        }
        if (m <= 1 || m > 16 || m_codes != m - 1) {
            last_stats_.fallback_reason = 3;
            if (err) *err = "LinkageNormProviderLookupCuda: unsupported m.";
            return false;
        }
        constexpr int kCodeSizeBytes = 1;
        // Soft-limit: allow a small overflow to avoid frequent CPU fallbacks caused by
        // slightly imbalanced IVF lists (e.g. n_real=10097 with limit=10000).
        // This does not change semantics; it only controls whether GPU is used.
        constexpr int kSoftSlack = 256;
        if (cv.nc > gpu_max_nc_ + kSoftSlack) {
            last_stats_.fallback_reason = 2;
            if (err) *err = "LinkageNormProviderLookupCuda: cluster exceeds GPU limits.";
            return false;
        }
        if (cv.depth_offsets_len < 2 || !cv.depth_offsets) {
            if (err) *err = "LinkageNormProviderLookupCuda: depth_offsets missing.";
            return false;
        }
        if (((!cv.parent_is_u16) && !cv.parent_1based) ||
            (cv.parent_is_u16 && !cv.parent_1based_u16) ||
            !cv.codes_small_bytes || !cv.code0_one_bytes ||
            (cv.n_virt > 0 && !cv.virt_codes_small_bytes)) {
            last_stats_.fallback_reason = 3;
            if (err) *err = "LinkageNormProviderLookupCuda: missing linkage list buffers.";
            return false;
        }

        const int n_virt = cv.n_virt;
        const int n_real = cv.n_real;
        const int nc = cv.nc;
        const int n_root_real = cv.n_root_real;

        // Cluster-specific c0 dot vectors (CPU), then upload (tiny).
        Timer t_cpu_c0dot;
        const float* c0 = C_root0_.Col(cv.cid);
        const float c0_norm2 = DotCpu(c0, c0, d);

        std::vector<float> c0_dot_rootS(static_cast<std::size_t>(meta_root_small_.total_cols), 0.0f);
        for (int j = 0; j < meta_root_small_.total_cols; ++j) {
            c0_dot_rootS[static_cast<std::size_t>(j)] = DotCpu(c0, meta_root_small_.flat.Col(j), d);
        }
        std::vector<float> c0_dot_one(static_cast<std::size_t>(meta_one_.total_cols), 0.0f);
        for (int j = 0; j < meta_one_.total_cols; ++j) {
            c0_dot_one[static_cast<std::size_t>(j)] = DotCpu(c0, meta_one_.flat.Col(j), d);
        }
        if (profile_breakdown_) {
            last_stats_.cpu_c0dot_sec = t_cpu_c0dot.ElapsedSeconds();
        }

        Timer t_h2d;
        CudaCheck(cudaMemcpyAsync(d_c0_dot_rootS_.ptr,
                                  c0_dot_rootS.data(),
                                  sizeof(float) * c0_dot_rootS.size(),
                                  cudaMemcpyHostToDevice,
                                  stream_),
                  "cudaMemcpyAsync c0_dot_rootS");
        CudaCheck(cudaMemcpyAsync(d_c0_dot_one_.ptr,
                                  c0_dot_one.data(),
                                  sizeof(float) * c0_dot_one.size(),
                                  cudaMemcpyHostToDevice,
                                  stream_),
                  "cudaMemcpyAsync c0_dot_one");

        // Upload cluster inputs (bounded by gpu_max_*).
        const int parent_elem_bytes = cv.parent_is_u16 ? static_cast<int>(sizeof(std::uint16_t))
                                                       : static_cast<int>(sizeof(std::uint32_t));
        const std::size_t parent_bytes = static_cast<std::size_t>(n_real) * static_cast<std::size_t>(parent_elem_bytes);
        d_parent_.Ensure(parent_bytes);
        const void* parent_upload_ptr = cv.parent_is_u16
            ? static_cast<const void*>(cv.parent_1based_u16)
            : static_cast<const void*>(cv.parent_1based);
        CudaCheck(cudaMemcpyAsync(d_parent_.ptr,
                                  parent_upload_ptr,
                                  parent_bytes,
                                  cudaMemcpyHostToDevice,
                                  stream_),
                  "cudaMemcpyAsync parent");

        const std::size_t codes_small_bytes =
            static_cast<std::size_t>(n_real) * static_cast<std::size_t>(m_codes) *
            static_cast<std::size_t>(kCodeSizeBytes);
        const std::size_t virt_codes_small_bytes =
            static_cast<std::size_t>(n_virt) * static_cast<std::size_t>(m_codes) *
            static_cast<std::size_t>(kCodeSizeBytes);
        const std::size_t code0_one_bytes = static_cast<std::size_t>(n_real);

        d_codes_small_.Ensure(codes_small_bytes);
        d_code0_one_.Ensure(code0_one_bytes);
        if (virt_codes_small_bytes) {
            d_virt_codes_small_.Ensure(virt_codes_small_bytes);
        } else {
            // Keep a non-null device pointer (kernels won't dereference when n_virt==0).
            d_virt_codes_small_.Ensure(1);
        }
        CudaCheck(cudaMemcpyAsync(d_codes_small_.ptr,
                                  cv.codes_small_bytes,
                                  codes_small_bytes,
                                  cudaMemcpyHostToDevice,
                                  stream_),
                  "cudaMemcpyAsync codes_small");
        if (virt_codes_small_bytes) {
            CudaCheck(cudaMemcpyAsync(d_virt_codes_small_.ptr,
                                      cv.virt_codes_small_bytes,
                                      virt_codes_small_bytes,
                                      cudaMemcpyHostToDevice,
                                      stream_),
                      "cudaMemcpyAsync virt_codes_small");
        }
        CudaCheck(cudaMemcpyAsync(d_code0_one_.ptr,
                                  cv.code0_one_bytes,
                                  code0_one_bytes,
                                  cudaMemcpyHostToDevice,
                                  stream_),
                  "cudaMemcpyAsync code0_one");

        if (use_quantized) {
            const std::size_t q_layer_major_bytes =
                static_cast<std::size_t>(m) * static_cast<std::size_t>(nc);
            d_q_layer_major_.Ensure(q_layer_major_bytes);
            CudaCheck(cudaMemcpyAsync(d_q_layer_major_.ptr,
                                      cv.q_layer_major,
                                      q_layer_major_bytes,
                                      cudaMemcpyHostToDevice,
                                      stream_),
                      "cudaMemcpyAsync q_layer_major");

            d_scales_root_.Ensure(static_cast<std::size_t>(m));
            d_scales_linkage_.Ensure(static_cast<std::size_t>(m));
            CudaCheck(cudaMemcpyAsync(d_scales_root_.ptr,
                                      cv.scales_root,
                                      sizeof(float) * static_cast<std::size_t>(m),
                                      cudaMemcpyHostToDevice,
                                      stream_),
                      "cudaMemcpyAsync scales_root");
            CudaCheck(cudaMemcpyAsync(d_scales_linkage_.ptr,
                                      cv.scales_linkage,
                                      sizeof(float) * static_cast<std::size_t>(m),
                                      cudaMemcpyHostToDevice,
                                      stream_),
                      "cudaMemcpyAsync scales_linkage");
        } else {
            d_a0_real_.Ensure(static_cast<std::size_t>(n_real));
            d_coeffs_small_real_.Ensure(static_cast<std::size_t>(n_real) * static_cast<std::size_t>(m_codes));
            d_virt_a0_.Ensure(static_cast<std::size_t>(std::max(1, n_virt)));
            d_virt_coeffs_small_.Ensure(static_cast<std::size_t>(std::max(1, n_virt)) * static_cast<std::size_t>(m_codes));
            CudaCheck(cudaMemcpyAsync(d_a0_real_.ptr,
                                      cv.a0,
                                      sizeof(float) * static_cast<std::size_t>(n_real),
                                      cudaMemcpyHostToDevice,
                                      stream_),
                      "cudaMemcpyAsync a0_real");
            CudaCheck(cudaMemcpyAsync(d_coeffs_small_real_.ptr,
                                      cv.coeffs_small,
                                      sizeof(float) * static_cast<std::size_t>(n_real) * static_cast<std::size_t>(m_codes),
                                      cudaMemcpyHostToDevice,
                                      stream_),
                      "cudaMemcpyAsync coeffs_small_real");
            if (n_virt > 0) {
                CudaCheck(cudaMemcpyAsync(d_virt_a0_.ptr,
                                          cv.virt_a0,
                                          sizeof(float) * static_cast<std::size_t>(n_virt),
                                          cudaMemcpyHostToDevice,
                                          stream_),
                          "cudaMemcpyAsync virt_a0");
                CudaCheck(cudaMemcpyAsync(d_virt_coeffs_small_.ptr,
                                          cv.virt_coeffs_small,
                                          sizeof(float) * static_cast<std::size_t>(n_virt) * static_cast<std::size_t>(m_codes),
                                          cudaMemcpyHostToDevice,
                                          stream_),
                          "cudaMemcpyAsync virt_coeffs_small");
            }
        }
        if (profile_breakdown_) {
            last_stats_.h2d_sec = t_h2d.ElapsedSeconds();
        }

        d_norm2_local_.Ensure(static_cast<std::size_t>(nc));
        d_norm2_real_.Ensure(static_cast<std::size_t>(n_real));

        cudaEvent_t ev_start = nullptr;
        cudaEvent_t ev_stop = nullptr;
        if (profile_breakdown_) {
            CudaCheck(cudaEventCreate(&ev_start), "cudaEventCreate");
            CudaCheck(cudaEventCreate(&ev_stop), "cudaEventCreate");
            CudaCheck(cudaEventRecord(ev_start, stream_), "cudaEventRecord");
        }

        const int threads = 128;
        if (n_virt > 0) {
            const int blocks = (n_virt + threads - 1) / threads;
            if (use_quantized) {
                NormVirtRootsKernel<true><<<blocks, threads, 0, stream_>>>(
                    m, m_codes,
                    n_virt, nc,
                    d_virt_codes_small_.ptr,
                    d_q_layer_major_.ptr,
                    d_scales_root_.ptr,
                    nullptr,
                    nullptr,
                    d_offsets_root_small_.ptr,
                    c0_norm2,
                    d_c0_dot_rootS_.ptr,
                    d_D_rr_.ptr,
                    D_rr_rows_,
                    d_norm2_local_.ptr);
            } else {
                NormVirtRootsKernel<false><<<blocks, threads, 0, stream_>>>(
                    m, m_codes,
                    n_virt, nc,
                    d_virt_codes_small_.ptr,
                    nullptr,
                    nullptr,
                    d_virt_a0_.ptr,
                    d_virt_coeffs_small_.ptr,
                    d_offsets_root_small_.ptr,
                    c0_norm2,
                    d_c0_dot_rootS_.ptr,
                    d_D_rr_.ptr,
                    D_rr_rows_,
                    d_norm2_local_.ptr);
            }
            CudaCheck(cudaGetLastError(), "NormVirtRootsKernel launch");
        }

        if (n_root_real > 0) {
            const int blocks = (n_root_real + threads - 1) / threads;
            if (use_quantized) {
                NormRealRootsKernel<true><<<blocks, threads, 0, stream_>>>(
                    m, m_codes,
                    n_root_real, n_virt, nc,
                    d_codes_small_.ptr,
                    d_q_layer_major_.ptr,
                    d_scales_root_.ptr,
                    nullptr,
                    nullptr,
                    d_offsets_root_small_.ptr,
                    c0_norm2,
                    d_c0_dot_rootS_.ptr,
                    d_D_rr_.ptr,
                    D_rr_rows_,
                    d_norm2_local_.ptr,
                    d_norm2_real_.ptr);
            } else {
                NormRealRootsKernel<false><<<blocks, threads, 0, stream_>>>(
                    m, m_codes,
                    n_root_real, n_virt, nc,
                    d_codes_small_.ptr,
                    nullptr,
                    nullptr,
                    d_a0_real_.ptr,
                    d_coeffs_small_real_.ptr,
                    d_offsets_root_small_.ptr,
                    c0_norm2,
                    d_c0_dot_rootS_.ptr,
                    d_D_rr_.ptr,
                    D_rr_rows_,
                    d_norm2_local_.ptr,
                    d_norm2_real_.ptr);
            }
            CudaCheck(cudaGetLastError(), "NormRealRootsKernel launch");
        }

        // Linkage nodes by depth range.
        const int max_dep = cv.depth_offsets_len - 1;
        for (int dep = 1; dep < max_dep; ++dep) {
            const int begin = static_cast<int>(cv.depth_offsets[static_cast<std::size_t>(dep)]);
            const int end = static_cast<int>(cv.depth_offsets[static_cast<std::size_t>(dep + 1)]);
            if (begin >= end) continue;
            const int len = end - begin;
            const int blocks = (len + threads - 1) / threads;
            if (use_quantized) {
                NormLinkageDepthKernel<true><<<blocks, threads, 0, stream_>>>(
                    m, m_codes, begin, end,
                    n_real, n_root_real, n_virt, nc,
                    d_parent_.ptr,
                    parent_elem_bytes,
                    d_codes_small_.ptr,
                    d_code0_one_.ptr,
                    d_virt_codes_small_.ptr,
                    d_q_layer_major_.ptr,
                    d_scales_root_.ptr,
                    d_scales_linkage_.ptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    d_offsets_root_small_.ptr,
                    d_offsets_one_.ptr,
                    c0_norm2,
                    d_c0_dot_one_.ptr,
                    d_D_oo_.ptr,
                    D_oo_rows_,
                    d_D_ro_.ptr,
                    D_ro_rows_,
                    d_norm2_local_.ptr,
                    d_norm2_real_.ptr);
            } else {
                NormLinkageDepthKernel<false><<<blocks, threads, 0, stream_>>>(
                    m, m_codes, begin, end,
                    n_real, n_root_real, n_virt, nc,
                    d_parent_.ptr,
                    parent_elem_bytes,
                    d_codes_small_.ptr,
                    d_code0_one_.ptr,
                    d_virt_codes_small_.ptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    d_a0_real_.ptr,
                    d_coeffs_small_real_.ptr,
                    d_virt_a0_.ptr,
                    d_virt_coeffs_small_.ptr,
                    d_offsets_root_small_.ptr,
                    d_offsets_one_.ptr,
                    c0_norm2,
                    d_c0_dot_one_.ptr,
                    d_D_oo_.ptr,
                    D_oo_rows_,
                    d_D_ro_.ptr,
                    D_ro_rows_,
                    d_norm2_local_.ptr,
                    d_norm2_real_.ptr);
            }
            CudaCheck(cudaGetLastError(), "NormLinkageDepthKernel launch");
        }

        if (profile_breakdown_) {
            CudaCheck(cudaEventRecord(ev_stop, stream_), "cudaEventRecord(stop)");
            CudaCheck(cudaEventSynchronize(ev_stop), "cudaEventSynchronize(stop)");
            float ms = 0.0f;
            CudaCheck(cudaEventElapsedTime(&ms, ev_start, ev_stop), "cudaEventElapsedTime");
            last_stats_.kernel_sec = static_cast<double>(ms) / 1000.0;
            cudaEventDestroy(ev_start);
            cudaEventDestroy(ev_stop);
        }

        // Copy back.
        Timer t_d2h;
        out_r_norm2->assign(static_cast<std::size_t>(n_real), 0.0f);
        CudaCheck(cudaMemcpyAsync(out_r_norm2->data(),
                                  d_norm2_real_.ptr,
                                  sizeof(float) * static_cast<std::size_t>(n_real),
                                  cudaMemcpyDeviceToHost,
                                  stream_),
                  "cudaMemcpyAsync norm2_real D2H");
        CudaCheck(cudaStreamSynchronize(stream_), "cudaStreamSynchronize norm2_real");
        if (profile_breakdown_) {
            last_stats_.d2h_sec = t_d2h.ElapsedSeconds();
        }
        last_stats_.used_gpu = true;
        return true;
    }

    bool ComputeNorm2LutGpu(const ClusterView& cv,
                            int requested_centers,
                            int max_iter,
                            Norm2Lut* out,
                            std::string* err) {
        if (!out) {
            if (err) *err = "LinkageNormProviderLookupCuda: null LUT output.";
            return false;
        }
        *out = Norm2Lut{};
        std::vector<float> host_norm2;
        if (!ComputeNorm2Gpu(cv, &host_norm2, err)) {
            return false;
        }
        return ComputeNorm2LutFromHost(host_norm2.data(),
                                       static_cast<int>(host_norm2.size()),
                                       requested_centers,
                                       max_iter,
                                       out,
                                       err);
    }

    bool ComputeNorm2LutFromHost(const float* norm2,
                                 int n,
                                 int requested_centers,
                                 int max_iter,
                                 Norm2Lut* out,
                                 std::string* err) {
        if (!out) {
            if (err) *err = "LinkageNormProviderLookupCuda: null LUT output.";
            return false;
        }
        *out = Norm2Lut{};
        if (n < 0 || (!norm2 && n > 0)) {
            if (err) *err = "LinkageNormProviderLookupCuda: invalid host norm2 input.";
            return false;
        }
        if (n == 0) {
            return true;
        }
        const int k = std::min(std::max(1, requested_centers), std::min(n, 256));
        std::vector<float> init_centers_host;
        BuildKmeans1DInitialCentersHost(norm2, n, k, &init_centers_host);
        d_norm2_real_.Ensure(static_cast<std::size_t>(n));
        CudaCheck(cudaMemcpyAsync(d_norm2_real_.ptr,
                                  norm2,
                                  sizeof(float) * static_cast<std::size_t>(n),
                                  cudaMemcpyHostToDevice,
                                  stream_),
                  "cudaMemcpyAsync norm2_real H2D");
        last_stats_.used_gpu = true;
        return ComputeNorm2LutFromDeviceNorm2(n,
                                             requested_centers,
                                             max_iter,
                                             init_centers_host.empty() ? nullptr : init_centers_host.data(),
                                             out,
                                             err);
    }

    bool ComputeNorm2LutFromDeviceNorm2(int n,
                                        int requested_centers,
                                        int max_iter,
                                        const float* init_centers_host,
                                        Norm2Lut* out,
                                        std::string* err) {
        if (!out) {
            if (err) *err = "LinkageNormProviderLookupCuda: null LUT output.";
            return false;
        }
        *out = Norm2Lut{};
        if (n <= 0) {
            return true;
        }
        const int k = std::min(std::max(1, requested_centers), std::min(n, 256));
        d_kmeans_centers_.Ensure(static_cast<std::size_t>(k));
        d_kmeans_centers_next_.Ensure(static_cast<std::size_t>(k));
        d_kmeans_sums_.Ensure(static_cast<std::size_t>(k));
        d_kmeans_counts_.Ensure(static_cast<std::size_t>(k));
        d_kmeans_assign_u8_.Ensure(static_cast<std::size_t>(n));

        const int threads = 256;
        const int center_blocks = (k + threads - 1) / threads;
        if (init_centers_host) {
            CudaCheck(cudaMemcpyAsync(d_kmeans_centers_.ptr,
                                      init_centers_host,
                                      sizeof(float) * static_cast<std::size_t>(k),
                                      cudaMemcpyHostToDevice,
                                      stream_),
                      "cudaMemcpyAsync kmeans init centers");
        } else {
            d_reduce_tmp_.Ensure(2);
            std::size_t tmp_bytes = 0;
            cub::DeviceReduce::Min(nullptr, tmp_bytes, d_norm2_real_.ptr, d_reduce_tmp_.ptr, n, stream_);
            cub::DeviceReduce::Max(nullptr, tmp_bytes, d_norm2_real_.ptr, d_reduce_tmp_.ptr + 1, n, stream_);
            d_cub_tmp_.Ensure(tmp_bytes);
            cub::DeviceReduce::Min(d_cub_tmp_.ptr, tmp_bytes, d_norm2_real_.ptr, d_reduce_tmp_.ptr, n, stream_);
            cub::DeviceReduce::Max(d_cub_tmp_.ptr, tmp_bytes, d_norm2_real_.ptr, d_reduce_tmp_.ptr + 1, n, stream_);
            CudaCheck(cudaGetLastError(), "cub::DeviceReduce Min/Max(norm2)");
            float minmax_host[2] = {0.0f, 0.0f};
            CudaCheck(cudaMemcpyAsync(minmax_host,
                                      d_reduce_tmp_.ptr,
                                      sizeof(minmax_host),
                                      cudaMemcpyDeviceToHost,
                                      stream_),
                      "cudaMemcpyAsync norm2 minmax");
            CudaCheck(cudaStreamSynchronize(stream_), "cudaStreamSynchronize norm2 minmax");
            InitCentersLinearKernel<<<center_blocks, threads, 0, stream_>>>(
                k, minmax_host[0], minmax_host[1], d_kmeans_centers_.ptr);
            CudaCheck(cudaGetLastError(), "InitCentersLinearKernel launch");
        }

        Timer t_kmeans;
        const int blocks = (n + threads - 1) / threads;
        std::vector<float> prev_centers_host;
        std::vector<float> cur_centers_host;
        if (init_centers_host) {
            prev_centers_host.assign(init_centers_host, init_centers_host + k);
            cur_centers_host.resize(static_cast<std::size_t>(k));
        }
        for (int it = 0; it < std::max(1, max_iter); ++it) {
            CudaCheck(cudaMemset(d_kmeans_sums_.ptr, 0, sizeof(double) * static_cast<std::size_t>(k)),
                      "cudaMemset kmeans sums");
            CudaCheck(cudaMemset(d_kmeans_counts_.ptr, 0, sizeof(int) * static_cast<std::size_t>(k)),
                      "cudaMemset kmeans counts");
            AssignKmeans1DAndAccumulateKernel<<<blocks, threads, 0, stream_>>>(
                d_norm2_real_.ptr, n,
                d_kmeans_centers_.ptr, k,
                d_kmeans_assign_u8_.ptr,
                d_kmeans_sums_.ptr,
                d_kmeans_counts_.ptr);
            CudaCheck(cudaGetLastError(), "AssignKmeans1DAndAccumulateKernel launch");
            UpdateKmeans1DCentersKernel<<<center_blocks, threads, 0, stream_>>>(
                d_kmeans_centers_.ptr,
                d_kmeans_centers_next_.ptr,
                d_kmeans_sums_.ptr,
                d_kmeans_counts_.ptr,
                k);
            CudaCheck(cudaGetLastError(), "UpdateKmeans1DCentersKernel launch");
            std::swap(d_kmeans_centers_.ptr, d_kmeans_centers_next_.ptr);
            std::swap(d_kmeans_centers_.n, d_kmeans_centers_next_.n);
            if (!cur_centers_host.empty()) {
                CudaCheck(cudaMemcpyAsync(cur_centers_host.data(),
                                          d_kmeans_centers_.ptr,
                                          sizeof(float) * static_cast<std::size_t>(k),
                                          cudaMemcpyDeviceToHost,
                                          stream_),
                          "cudaMemcpyAsync kmeans centers iter");
                CudaCheck(cudaStreamSynchronize(stream_), "cudaStreamSynchronize kmeans iter");
                bool converged = true;
                for (int j = 0; j < k; ++j) {
                    if (std::fabs(cur_centers_host[static_cast<std::size_t>(j)] -
                                  prev_centers_host[static_cast<std::size_t>(j)]) > 1e-6f) {
                        converged = false;
                        break;
                    }
                }
                prev_centers_host.swap(cur_centers_host);
                if (converged) {
                    break;
                }
            }
        }
        AssignKmeans1DFinalKernel<<<blocks, threads, 0, stream_>>>(
            d_norm2_real_.ptr, n,
            d_kmeans_centers_.ptr, k,
            d_kmeans_assign_u8_.ptr);
        CudaCheck(cudaGetLastError(), "AssignKmeans1DFinalKernel launch");
        CudaCheck(cudaStreamSynchronize(stream_), "cudaStreamSynchronize kmeans1d");
        last_stats_.lut_kmeans_sec = t_kmeans.ElapsedSeconds();
        last_stats_.used_gpu_lut = true;

        out->centers.resize(static_cast<std::size_t>(k));
        out->codes_u8.resize(static_cast<std::size_t>(n));
        CudaCheck(cudaMemcpyAsync(out->centers.data(),
                                  d_kmeans_centers_.ptr,
                                  sizeof(float) * static_cast<std::size_t>(k),
                                  cudaMemcpyDeviceToHost,
                                  stream_),
                  "cudaMemcpyAsync kmeans centers");
        CudaCheck(cudaMemcpyAsync(out->codes_u8.data(),
                                  d_kmeans_assign_u8_.ptr,
                                  sizeof(std::uint8_t) * static_cast<std::size_t>(n),
                                  cudaMemcpyDeviceToHost,
                                  stream_),
                  "cudaMemcpyAsync kmeans assignments");
        CudaCheck(cudaStreamSynchronize(stream_), "cudaStreamSynchronize kmeans outputs");
        return true;
    }
};

LinkageNormProviderLookupCuda::LinkageNormProviderLookupCuda(const ColMajorMatrix<float>& C_root0,
                                                         const CodebookMeta& meta_root_small,
                                                         const CodebookMeta& meta_one,
                                                         int gpu_max_nc,
                                                         bool profile_breakdown)
    : impl_(std::make_unique<Impl>(C_root0, meta_root_small, meta_one, gpu_max_nc, profile_breakdown)) {}

LinkageNormProviderLookupCuda::~LinkageNormProviderLookupCuda() = default;

std::unique_ptr<IClusterNormProvider> LinkageNormProviderLookupCuda::CloneForParallelPrecompute() const {
    return std::make_unique<LinkageNormProviderLookupCuda>(impl_->C_root0_,
                                                         impl_->meta_root_small_,
                                                         impl_->meta_one_,
                                                         impl_->gpu_max_nc_,
                                                         impl_->profile_breakdown_);
}

bool LinkageNormProviderLookupCuda::ComputeNorm2(const ClusterView& cv,
                                               std::vector<float>* out_r_norm2,
                                               std::string* err) {
    if (!out_r_norm2) return false;
    try {
        std::string ignored;
        const bool ok = impl_->ComputeNorm2Gpu(cv, out_r_norm2, &ignored);
        if (ok) {
            return true;
        }
    } catch (...) {
        impl_->last_stats_.used_gpu = false;
        if (impl_->last_stats_.fallback_reason == 0) {
            impl_->last_stats_.fallback_reason = 4;
        }
    }
    if (impl_->last_stats_.fallback_reason == 0) {
        impl_->last_stats_.fallback_reason = 4;
    }
    return impl_->cpu_.ComputeNorm2(cv, out_r_norm2, err);
}

bool LinkageNormProviderLookupCuda::ComputeNorm2Lut(const ClusterView& cv,
                                                  int requested_centers,
                                                  int max_iter,
                                                  Norm2Lut* out,
                                                  std::string* err) {
    if (!out) return false;
    try {
        std::string ignored;
        if (impl_->ComputeNorm2LutGpu(cv, requested_centers, max_iter, out, &ignored)) {
            return true;
        }
    } catch (...) {
        impl_->last_stats_.used_gpu_lut = false;
        if (impl_->last_stats_.fallback_reason == 0) {
            impl_->last_stats_.fallback_reason = 4;
        }
    }
    std::vector<float> r_norm2;
    if (!impl_->cpu_.ComputeNorm2(cv, &r_norm2, err)) {
        return false;
    }
    return BuildNorm2Lut(r_norm2.data(), static_cast<int>(r_norm2.size()),
                         requested_centers, max_iter, out, err);
}

bool LinkageNormProviderLookupCuda::ComputeNorm2LutFromHost(const float* norm2,
                                                          int n,
                                                          int requested_centers,
                                                          int max_iter,
                                                          Norm2Lut* out,
                                                          std::string* err) {
    if (!out) return false;
    try {
        std::string ignored;
        if (impl_->ComputeNorm2LutFromHost(norm2, n, requested_centers, max_iter, out, &ignored)) {
            return true;
        }
    } catch (...) {
        impl_->last_stats_.used_gpu_lut = false;
        if (impl_->last_stats_.fallback_reason == 0) {
            impl_->last_stats_.fallback_reason = 4;
        }
    }
    return BuildNorm2Lut(norm2, n, requested_centers, max_iter, out, err);
}

const LinkageNormProviderLookupCuda::Stats& LinkageNormProviderLookupCuda::last_stats() const {
    return impl_->last_stats_;
}

}  // namespace stlq::eval::cuda

#endif  // STLQ_ENABLE_CUDA
