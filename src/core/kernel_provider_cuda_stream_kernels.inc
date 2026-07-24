namespace stlq {

namespace {

inline void ThrowIf(cudaError_t st, const char* what) {
    if (st != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(st));
    }
}

inline void ThrowIf(cublasStatus_t st, const char* what) {
    if (st == CUBLAS_STATUS_SUCCESS) return;
    throw std::runtime_error(std::string(what) + ": cublasStatus=" + std::to_string(static_cast<int>(st)));
}

__global__ void U8ToF32Kernel1D(const std::uint8_t* __restrict src,
                                int ld_src,
                                float* __restrict dst,
                                int ld_dst,
                                int rows,
                                int cols) {
    // Column-major: element (r,c) stored at base + r + c*ld.
    // Use a 1D launch to avoid grid.y limits when cols is large (e.g. 100k+).
    const std::int64_t total = static_cast<std::int64_t>(rows) * static_cast<std::int64_t>(cols);
    for (std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const int r = static_cast<int>(idx % rows);
        const int c = static_cast<int>(idx / rows);
        dst[r + c * ld_dst] = static_cast<float>(src[r + c * ld_src]);
    }
}

// -------- kmeans|| (kmeansll) seeding helpers --------

__global__ void SetF32Kernel(float* __restrict x, int n, float v) {
    for (int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<int>(gridDim.x) * blockDim.x) {
        x[i] = v;
    }
}

__global__ void IotaI32Kernel(int* __restrict out, int n) {
    for (int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<int>(gridDim.x) * blockDim.x) {
        out[i] = i;
    }
}

__global__ void MaxOffsetKernel(float* __restrict best,
                                const float* __restrict tile,
                                int n,
                                int off) {
    for (int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<int>(gridDim.x) * blockDim.x) {
        const float v = tile[i];
        float* p = best + (off + i);
        const float cur = *p;
        *p = (v > cur) ? v : cur;
    }
}

__global__ void KmeansllComputeD2Kernel(const float* __restrict best_dot,
                                       float* __restrict out_d2,
                                       int n) {
    for (int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<int>(gridDim.x) * blockDim.x) {
        float bd = best_dot[i];
        if (!(bd > -1e20f)) bd = -1.0f;
        bd = fminf(1.0f, fmaxf(-1.0f, bd));
        const float d2 = fmaxf(0.0f, 2.0f - 2.0f * bd);
        out_d2[i] = d2;
    }
}

__device__ inline std::uint32_t Xorshift32(std::uint32_t x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

__global__ void KmeansllSampleFlagsKernel(const float* __restrict d2,
                                         std::uint8_t* __restrict flags,
                                         int n,
                                         float inv_phi,
                                         float oversample_l,
                                         std::uint64_t seed) {
    constexpr float kInv2p32 = 1.0f / 4294967296.0f;
    for (int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<int>(gridDim.x) * blockDim.x) {
        const float p_raw = oversample_l * d2[i] * inv_phi;
        const float p = fminf(1.0f, fmaxf(0.0f, p_raw));
        if (p <= 0.0f) {
            flags[i] = 0;
            continue;
        }
        std::uint32_t r = static_cast<std::uint32_t>(seed ^ (static_cast<std::uint64_t>(i) * 0x9e3779b97f4a7c15ULL));
        r = Xorshift32(r);
        const float u = (static_cast<float>(r) + 1.0f) * kInv2p32;
        flags[i] = (u < p) ? 1 : 0;
    }
}

__global__ void GatherColsKernel1D(float* __restrict out,
                                  int ld_out,
                                  const float* __restrict X,
                                  int ldX,
                                  const int* __restrict idx,
                                  int rows,
                                  int cols) {
    const std::int64_t total = static_cast<std::int64_t>(rows) * static_cast<std::int64_t>(cols);
    for (std::int64_t t = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         t < total;
         t += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const int r = static_cast<int>(t % rows);
        const int c = static_cast<int>(t / rows);
        const int j = idx[c];
        out[r + c * ld_out] = X[r + j * ldX];
    }
}

__global__ void ArgmaxColsKernel(const float* __restrict scores,
                                 int ld,
                                 int rows,
                                 int cols,
                                 int* __restrict out_idx,
                                 float* __restrict out_val) {
    const int c = static_cast<int>(blockIdx.x);
    if (c >= cols) return;

    float best = -INFINITY;
    int best_i = 0;

    const float* col = scores + static_cast<std::size_t>(c) * static_cast<std::size_t>(ld);
    for (int r = threadIdx.x; r < rows; r += blockDim.x) {
        const float v = col[r];
        if (v > best || (v == best && r < best_i)) {
            best = v;
            best_i = r;
        }
    }

    // Warp-level reduction (faster than full shared reduction).
    const unsigned mask = 0xFFFFFFFFu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float bv = __shfl_down_sync(mask, best, offset);
        const int bi = __shfl_down_sync(mask, best_i, offset);
        if (bv > best || (bv == best && bi < best_i)) {
            best = bv;
            best_i = bi;
        }
    }

    __shared__ float warp_best[8];
    __shared__ int warp_idx[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_best[warp] = best;
        warp_idx[warp] = best_i;
    }
    __syncthreads();

    if (warp == 0) {
        float wbest = (lane < (blockDim.x >> 5)) ? warp_best[lane] : -INFINITY;
        int widx = (lane < (blockDim.x >> 5)) ? warp_idx[lane] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float bv = __shfl_down_sync(mask, wbest, offset);
            const int bi = __shfl_down_sync(mask, widx, offset);
            if (bv > wbest || (bv == wbest && bi < widx)) {
                wbest = bv;
                widx = bi;
            }
        }
        if (lane == 0) {
            out_idx[c] = widx;
            out_val[c] = wbest;
        }
    }
}

__device__ inline void Top2Update(float v, int i, float* v0, int* i0, float* v1, int* i1) {
    if (v > *v0 || (v == *v0 && i < *i0)) {
        *v1 = *v0;
        *i1 = *i0;
        *v0 = v;
        *i0 = i;
        return;
    }
    if (i == *i0) return;
    if (v > *v1 || (v == *v1 && i < *i1)) {
        *v1 = v;
        *i1 = i;
    }
}

__global__ void Top2ColsKernel(const float* __restrict scores,
                               int ld,
                               int rows,
                               int cols,
                               int* __restrict out_idx2,
                               float* __restrict out_val2) {
    const int c = static_cast<int>(blockIdx.x);
    if (c >= cols) return;

    const float* col = scores + static_cast<std::size_t>(c) * static_cast<std::size_t>(ld);
    float v0 = -INFINITY;
    float v1 = -INFINITY;
    int i0 = 0;
    int i1 = 0;
    for (int r = threadIdx.x; r < rows; r += blockDim.x) {
        const float v = col[r];
        Top2Update(v, r, &v0, &i0, &v1, &i1);
    }

    extern __shared__ unsigned char smem[];
    float* sv0 = reinterpret_cast<float*>(smem);
    float* sv1 = sv0 + blockDim.x;
    int* si0 = reinterpret_cast<int*>(sv1 + blockDim.x);
    int* si1 = si0 + blockDim.x;
    sv0[threadIdx.x] = v0;
    sv1[threadIdx.x] = v1;
    si0[threadIdx.x] = i0;
    si1[threadIdx.x] = i1;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float a0 = sv0[threadIdx.x];
            float a1 = sv1[threadIdx.x];
            int ai0 = si0[threadIdx.x];
            int ai1 = si1[threadIdx.x];

            const float b0 = sv0[threadIdx.x + stride];
            const float b1 = sv1[threadIdx.x + stride];
            const int bi0 = si0[threadIdx.x + stride];
            const int bi1 = si1[threadIdx.x + stride];

            // Merge four candidates into (a0,a1).
            Top2Update(b0, bi0, &a0, &ai0, &a1, &ai1);
            Top2Update(b1, bi1, &a0, &ai0, &a1, &ai1);

            sv0[threadIdx.x] = a0;
            sv1[threadIdx.x] = a1;
            si0[threadIdx.x] = ai0;
            si1[threadIdx.x] = ai1;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_idx2[2 * c + 0] = si0[0];
        out_val2[2 * c + 0] = sv0[0];
        out_idx2[2 * c + 1] = si1[0];
        out_val2[2 * c + 1] = sv1[0];
    }
}

template <int TOPL>
__global__ void Hier2FineAssignKernel(const float* __restrict Xnorm,
                                      int ldX,
                                      int d,
                                      int cols,
                                      const float* __restrict fine,
                                      int ldF,
                                      int K_valid,
                                      const int* __restrict topk_idx,
                                      int K2,
                                      int* __restrict out_assign,
                                      float* __restrict out_best) {
    const int col = static_cast<int>(blockIdx.x);
    if (col >= cols) return;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;  // 0..3 for 128 threads
    const float* x = Xnorm + static_cast<std::size_t>(col) * static_cast<std::size_t>(ldX);

    __shared__ float warp_dot[4];
    __shared__ int warp_id[4];

    float best_dot = -INFINITY;
    int best_id = 0;

    #pragma unroll
    for (int t = 0; t < TOPL; ++t) {
        const int coarse = topk_idx[static_cast<std::size_t>(TOPL) * static_cast<std::size_t>(col) +
                                    static_cast<std::size_t>(t)];
        const int start = coarse * K2;
        const int remain = K_valid - start;
        if (remain <= 0) continue;
        const int sz = (remain < K2) ? remain : K2;

        for (int base = 0; base < sz; base += 4) {
            const int local = base + warp;
            float dot = -INFINITY;
            int fid = -1;
            if (local < sz) {
                fid = start + local;
                const float* cptr = fine + static_cast<std::size_t>(fid) * static_cast<std::size_t>(ldF);
                float sum = 0.0f;
                for (int r = lane; r < d; r += 32) {
                    sum += x[r] * cptr[r];
                }
                const unsigned mask = 0xFFFFFFFFu;
                for (int offset = 16; offset > 0; offset >>= 1) {
                    sum += __shfl_down_sync(mask, sum, offset);
                }
                dot = __shfl_sync(mask, sum, 0);
            }
            if (lane == 0) {
                warp_dot[warp] = dot;
                warp_id[warp] = fid;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                for (int w = 0; w < 4; ++w) {
                    const float v = warp_dot[w];
                    const int id = warp_id[w];
                    if (id >= 0 && (v > best_dot || (v == best_dot && id < best_id))) {
                        best_dot = v;
                        best_id = id;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (threadIdx.x == 0) {
        out_assign[col] = best_id;
        out_best[col] = best_dot;
    }
}

template <int WARPS_PER_BLOCK>
__global__ void NormalizeColsWarpKernel(float* __restrict X,
                                        int ld,
                                        int rows,
                                        int cols) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;  // [0, WARPS_PER_BLOCK)
    const int c = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp_in_block;
    if (c >= cols) return;

    float* col = X + static_cast<std::size_t>(c) * static_cast<std::size_t>(ld);
    float sum = 0.0f;
    for (int r = lane; r < rows; r += 32) {
        const float v = col[r];
        sum += v * v;
    }
    const unsigned mask = 0xFFFFFFFFu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
    }
    const float inv = rsqrtf(fmaxf(__shfl_sync(mask, sum, 0), 1e-12f));
    for (int r = lane; r < rows; r += 32) {
        col[r] *= inv;
    }
}

template <int WARPS_PER_BLOCK>
__global__ void NormalizeColsWarpKernelWithNorm2(float* __restrict X,
                                                 int ld,
                                                 int rows,
                                                 int cols,
                                                 float* __restrict norm2_out) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;  // [0, WARPS_PER_BLOCK)
    const int c = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp_in_block;
    if (c >= cols) return;

    float* col = X + static_cast<std::size_t>(c) * static_cast<std::size_t>(ld);
    float sum = 0.0f;
    for (int r = lane; r < rows; r += 32) {
        const float v = col[r];
        sum += v * v;
    }
    const unsigned mask = 0xFFFFFFFFu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
    }
    sum = __shfl_sync(mask, sum, 0);
    if (lane == 0) {
        norm2_out[c] = sum;
    }
    const float inv = rsqrtf(fmaxf(sum, 1e-12f));
    for (int r = lane; r < rows; r += 32) {
        col[r] *= inv;
    }
}

template <int WARPS_PER_BLOCK>
__global__ void NormalizeColsWarpKernelU8(const std::uint8_t* __restrict in_u8,
                                          int ld_in,
                                          float* __restrict out_f32,
                                          int ld_out,
                                          int rows,
                                          int cols) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;  // [0, WARPS_PER_BLOCK)
    const int c = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp_in_block;
    if (c >= cols) return;

    const std::uint8_t* col_in =
        in_u8 + static_cast<std::size_t>(c) * static_cast<std::size_t>(ld_in);
    float* col_out =
        out_f32 + static_cast<std::size_t>(c) * static_cast<std::size_t>(ld_out);

    float sum = 0.0f;
    for (int r = lane; r < rows; r += 32) {
        const float v = static_cast<float>(col_in[r]);
        sum += v * v;
    }
    const unsigned mask = 0xFFFFFFFFu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
    }
    const float inv = rsqrtf(fmaxf(__shfl_sync(mask, sum, 0), 1e-12f));
    for (int r = lane; r < rows; r += 32) {
        col_out[r] = static_cast<float>(col_in[r]) * inv;
    }
}

template <int WARPS_PER_BLOCK>
__global__ void NormalizeColsWarpKernelU8WithNorm2(const std::uint8_t* __restrict in_u8,
                                                   int ld_in,
                                                   float* __restrict out_f32,
                                                   int ld_out,
                                                   int rows,
                                                   int cols,
                                                   float* __restrict norm2_out) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;  // [0, WARPS_PER_BLOCK)
    const int c = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp_in_block;
    if (c >= cols) return;

    const std::uint8_t* col_in =
        in_u8 + static_cast<std::size_t>(c) * static_cast<std::size_t>(ld_in);
    float* col_out =
        out_f32 + static_cast<std::size_t>(c) * static_cast<std::size_t>(ld_out);

    float sum = 0.0f;
    for (int r = lane; r < rows; r += 32) {
        const float v = static_cast<float>(col_in[r]);
        sum += v * v;
    }
    const unsigned mask = 0xFFFFFFFFu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
    }
    sum = __shfl_sync(mask, sum, 0);
    if (lane == 0) {
        norm2_out[c] = sum;
    }
    const float inv = rsqrtf(fmaxf(sum, 1e-12f));
    for (int r = lane; r < rows; r += 32) {
        col_out[r] = static_cast<float>(col_in[r]) * inv;
    }
}

__global__ void FillF32Kernel(float* __restrict out, int n, float v) {
    const int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int stride = static_cast<int>(blockDim.x) * static_cast<int>(gridDim.x);
    for (int i = tid; i < n; i += stride) {
        out[i] = v;
    }
}

template <int WARPS_PER_BLOCK>
__global__ void ScaleColsWarpKernelFromNorm2(float* __restrict X,
                                             int ld,
                                             int rows,
                                             int cols,
                                             const float* __restrict norm2) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;  // [0, WARPS_PER_BLOCK)
    const int c = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp_in_block;
    if (c >= cols) return;

    const float n2 = fmaxf(norm2[c], 1e-12f);
    const float inv = rsqrtf(n2);
    float* col = X + static_cast<std::size_t>(c) * static_cast<std::size_t>(ld);
    for (int r = lane; r < rows; r += 32) {
        col[r] *= inv;
    }
}

template <int WARPS_PER_BLOCK, typename CodeT>
__global__ void RvqProjectColsWarpKernel(float* __restrict Xnorm,
                                        int ldX,
                                        int rows,
                                        int cols,
                                        const float* __restrict C,
                                        int ldC,
                                        int K,
                                        const CodeT* __restrict codes) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;  // [0, WARPS_PER_BLOCK)
    const int c = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp_in_block;
    if (c >= cols) return;

    const int code_i = static_cast<int>(codes[c]);
    if (code_i < 0 || code_i >= K) return;

    float* x = Xnorm + static_cast<std::size_t>(c) * static_cast<std::size_t>(ldX);
    const float* center = C + static_cast<std::size_t>(code_i) * static_cast<std::size_t>(ldC);

    float dot = 0.0f;
    for (int r = lane; r < rows; r += 32) {
        dot += x[r] * center[r];
    }
    const unsigned mask = 0xFFFFFFFFu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(mask, dot, offset);
    }
    dot = __shfl_sync(mask, dot, 0);
    for (int r = lane; r < rows; r += 32) {
        x[r] -= dot * center[r];
    }
}

__device__ __forceinline__ int LoadRvqCode(const std::uint8_t* __restrict codes_base,
                                          int code_offset_bytes,
                                          int code_bytes,
                                          int col) {
    const std::uint8_t* p = codes_base + static_cast<std::size_t>(code_offset_bytes) +
                            static_cast<std::size_t>(col) * static_cast<std::size_t>(code_bytes);
    if (code_bytes == 1) {
        return static_cast<int>(p[0]);
    }
    if (code_bytes == 2) {
        const std::uint16_t v =
            static_cast<std::uint16_t>(p[0]) |
            (static_cast<std::uint16_t>(p[1]) << 8);
        return static_cast<int>(v);
    }
    const std::uint32_t v =
        static_cast<std::uint32_t>(p[0]) |
        (static_cast<std::uint32_t>(p[1]) << 8) |
        (static_cast<std::uint32_t>(p[2]) << 16) |
        (static_cast<std::uint32_t>(p[3]) << 24);
    return static_cast<int>(v);
}

// Fused prior-layer projection + normalization (Prompt-06).
// Applies sequential projections for layers [0, upto_layer):
//   x <- x - dot(x, c_l[code_l]) * c_l[code_l]
// then normalizes x to unit L2.
// Optionally writes hatnorm2_out[col] = ||x||^2 before normalization.
template <int WARPS_PER_BLOCK>
__global__ void RvqProjectAndNormalizeColsWarpKernel(float* __restrict Xnorm,
                                                     int ldX,
                                                     int rows,
                                                     int cols,
                                                     const float* __restrict Cbase,
                                                     int ldC,
                                                     const int* __restrict cb_offsets_f,
                                                     const int* __restrict cb_K,
                                                     const std::uint8_t* __restrict codes_base,
                                                     const int* __restrict code_offsets_bytes,
                                                     const int* __restrict code_bytes_per_code,
                                                     int upto_layer,
                                                     float* __restrict hatnorm2_out) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;  // [0, WARPS_PER_BLOCK)
    const int col = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp_in_block;
    if (col >= cols) return;

    float* x = Xnorm + static_cast<std::size_t>(col) * static_cast<std::size_t>(ldX);
    // Keep x in registers per-lane for rows <= 256 to avoid reloading/storing x for each layer.
    constexpr int kMaxValsPerLane = 8;  // 8*32 = 256 dims
    const int vals = (rows + 31) / 32;
    if (vals > kMaxValsPerLane) {
        // Fallback: operate directly on global memory (rare in this project).
        for (int l = 0; l < upto_layer; ++l) {
            const int code_bytes = code_bytes_per_code[l];
            const int code_i = LoadRvqCode(codes_base, code_offsets_bytes[l], code_bytes, col);
            const int K = cb_K[l];
            if (code_i < 0 || code_i >= K) continue;
            const float* center =
                Cbase + static_cast<std::size_t>(cb_offsets_f[l]) + static_cast<std::size_t>(code_i) * static_cast<std::size_t>(ldC);
            float dot = 0.0f;
            for (int r = lane; r < rows; r += 32) {
                dot += x[r] * center[r];
            }
            const unsigned mask = 0xFFFFFFFFu;
            for (int offset = 16; offset > 0; offset >>= 1) {
                dot += __shfl_down_sync(mask, dot, offset);
            }
            dot = __shfl_sync(mask, dot, 0);
            for (int r = lane; r < rows; r += 32) {
                x[r] -= dot * center[r];
            }
        }
        float sum = 0.0f;
        for (int r = lane; r < rows; r += 32) {
            const float v = x[r];
            sum += v * v;
        }
        const unsigned mask = 0xFFFFFFFFu;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(mask, sum, offset);
        }
        sum = __shfl_sync(mask, sum, 0);
        if (hatnorm2_out && lane == 0) {
            hatnorm2_out[col] = sum;
        }
        const float inv = rsqrtf(fmaxf(sum, 1e-12f));
        for (int r = lane; r < rows; r += 32) {
            x[r] *= inv;
        }
        return;
    }

    float x_vals[kMaxValsPerLane];
    #pragma unroll
    for (int t = 0; t < kMaxValsPerLane; ++t) {
        x_vals[t] = 0.0f;
    }
    for (int t = 0; t < vals; ++t) {
        const int r = lane + 32 * t;
        if (r < rows) x_vals[t] = x[r];
    }

    for (int l = 0; l < upto_layer; ++l) {
        const int code_bytes = code_bytes_per_code[l];
        const int code_i = LoadRvqCode(codes_base, code_offsets_bytes[l], code_bytes, col);
        const int K = cb_K[l];
        if (code_i < 0 || code_i >= K) continue;
        const float* center =
            Cbase + static_cast<std::size_t>(cb_offsets_f[l]) + static_cast<std::size_t>(code_i) * static_cast<std::size_t>(ldC);

        float center_vals[kMaxValsPerLane];
        #pragma unroll
        for (int t = 0; t < kMaxValsPerLane; ++t) {
            center_vals[t] = 0.0f;
        }
        float dot = 0.0f;
        for (int t = 0; t < vals; ++t) {
            const int r = lane + 32 * t;
            const float cv = (r < rows) ? center[r] : 0.0f;
            center_vals[t] = cv;
            dot += x_vals[t] * cv;
        }
        const unsigned mask = 0xFFFFFFFFu;
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(mask, dot, offset);
        }
        dot = __shfl_sync(mask, dot, 0);
        for (int t = 0; t < vals; ++t) {
            x_vals[t] -= dot * center_vals[t];
        }
    }

    float sum = 0.0f;
    for (int t = 0; t < vals; ++t) {
        const float v = x_vals[t];
        sum += v * v;
    }
    const unsigned mask = 0xFFFFFFFFu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
    }
    sum = __shfl_sync(mask, sum, 0);
    if (hatnorm2_out && lane == 0) {
        hatnorm2_out[col] = sum;
    }
    const float inv = rsqrtf(fmaxf(sum, 1e-12f));
    for (int t = 0; t < vals; ++t) {
        x_vals[t] *= inv;
    }
    for (int t = 0; t < vals; ++t) {
        const int r = lane + 32 * t;
        if (r < rows) x[r] = x_vals[t];
    }
}

template <int WARPS_PER_BLOCK, typename CodeT>
__global__ void RvqProjectColsWarpKernelWithNorm2(float* __restrict Xnorm,
                                                  int ldX,
                                                  int rows,
                                                  int cols,
                                                  const float* __restrict C,
                                                  int ldC,
                                                  int K,
                                                  const CodeT* __restrict codes,
                                                  float* __restrict hatnorm2) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;  // [0, WARPS_PER_BLOCK)
    const int c = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp_in_block;
    if (c >= cols) return;

    const int code_i = static_cast<int>(codes[c]);
    if (code_i < 0 || code_i >= K) return;

    float* x = Xnorm + static_cast<std::size_t>(c) * static_cast<std::size_t>(ldX);
    const float* center = C + static_cast<std::size_t>(code_i) * static_cast<std::size_t>(ldC);

    float dot = 0.0f;
    for (int r = lane; r < rows; r += 32) {
        dot += x[r] * center[r];
    }
    const unsigned mask = 0xFFFFFFFFu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(mask, dot, offset);
    }
    dot = __shfl_sync(mask, dot, 0);
    for (int r = lane; r < rows; r += 32) {
        x[r] -= dot * center[r];
    }
    if (lane == 0) {
        const float prev = hatnorm2[c];
        hatnorm2[c] = fmaxf(0.0f, prev - dot * dot);
    }
}

__global__ void KmeansAccumulateUnitWeightsKernel1D(const float* __restrict X,
                                                    int ldX,
                                                    int rows,
                                                    int cols,
                                                    const int* __restrict assign,
                                                    float* __restrict sum_x,
                                                    int ld_sum_x,
                                                    float* __restrict sum_w) {
    const std::int64_t total = static_cast<std::int64_t>(rows) * static_cast<std::int64_t>(cols);
    for (std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const int r = static_cast<int>(idx % rows);
        const int c = static_cast<int>(idx / rows);
        const int k = assign[c];
        atomicAdd(sum_x + r + k * ld_sum_x, X[r + c * ldX]);
        if (r == 0) {
            atomicAdd(sum_w + k, 1.0f);
        }
    }
}

__global__ void KmeansAccumulateWeightsKernel1D(const float* __restrict X,
                                                int ldX,
                                                int rows,
                                                int cols,
                                                const int* __restrict assign,
                                                const float* __restrict w,
                                                float* __restrict sum_x,
                                                int ld_sum_x,
                                                float* __restrict sum_w) {
    const std::int64_t total = static_cast<std::int64_t>(rows) * static_cast<std::int64_t>(cols);
    for (std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const int r = static_cast<int>(idx % rows);
        const int c = static_cast<int>(idx / rows);
        const int k = assign[c];
        const float ww = w[c];
        atomicAdd(sum_x + r + k * ld_sum_x, ww * X[r + c * ldX]);
        if (r == 0) {
            atomicAdd(sum_w + k, ww);
        }
    }
}

__global__ void FillKeysValsKernel(int cols,
                                   const int* __restrict assign,
                                   int* __restrict keys,
                                   int* __restrict vals) {
    const int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int stride = static_cast<int>(blockDim.x) * static_cast<int>(gridDim.x);
    for (int c = tid; c < cols; c += stride) {
        keys[c] = assign[c];
        vals[c] = c;
    }
}

__global__ void ReduceSegmentsAccumulateKernel(const float* __restrict X,
                                               int ldX,
                                               int d,
                                               int cols,
                                               const int* __restrict sorted_vals,
                                               const int* __restrict unique_keys,
                                               const int* __restrict seg_offsets,
                                               const int* __restrict seg_counts,
                                               const int* __restrict num_segments_ptr,
                                               const float* __restrict w_or_null,
                                               float* __restrict sum_x,
                                               int ld_sum_x,
                                               float* __restrict sum_w) {
    const int seg = static_cast<int>(blockIdx.x);
    const int num_segments = num_segments_ptr ? num_segments_ptr[0] : 0;
    if (seg >= num_segments) return;

    const int k = unique_keys[seg];
    const int begin = seg_offsets[seg];
    const int len = seg_counts[seg];
    if (len <= 0) return;

    // Weighted sum of weights for this segment.
    float wsum_local = 0.0f;
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        const int c = sorted_vals[begin + i];
        const float ww = w_or_null ? w_or_null[c] : 1.0f;
        wsum_local += ww;
    }

    __shared__ float wsum_sh[256];
    if (threadIdx.x < 256) {
        wsum_sh[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    if (threadIdx.x < 256) {
        wsum_sh[threadIdx.x] = wsum_local;
    }
    __syncthreads();
    for (int offset = 128; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            wsum_sh[threadIdx.x] += wsum_sh[threadIdx.x + offset];
        }
        __syncthreads();
    }
    const float wsum = wsum_sh[0];
    if (threadIdx.x == 0) {
        sum_w[k] += wsum;
    }

    for (int r = threadIdx.x; r < d; r += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < len; ++i) {
            const int c = sorted_vals[begin + i];
            const float ww = w_or_null ? w_or_null[c] : 1.0f;
            acc += ww * X[r + c * ldX];
        }
        sum_x[r + k * ld_sum_x] += acc;
    }
}

__global__ void KmeansHistCostFromBestDotWarpAggKernel(const float* __restrict best_dot,
                                                      int n,
                                                      std::uint32_t* __restrict hist,
                                                      int bins) {
    if (!best_dot || !hist || bins <= 0) return;
    const float lo = 0.0f;
    const float hi = 2.0f;
    const float inv_span = static_cast<float>(bins) / (hi - lo);

    const int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int stride = static_cast<int>(blockDim.x) * static_cast<int>(gridDim.x);

    for (int i = tid; i < n; i += stride) {
        const float dot = best_dot[i];
        float cost = 1.0f - dot;
        cost = fminf(hi, fmaxf(lo, cost));
        int b = static_cast<int>((cost - lo) * inv_span);
        if (b >= bins) b = bins - 1;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
        // Warp-aggregated atomic: group lanes that hit the same bin.
        const unsigned mask = __match_any_sync(0xffffffffu, b);
        const int leader_lane = __ffs(mask) - 1;  // [0,31]
        if ((static_cast<int>(threadIdx.x) & 31) == leader_lane) {
            const unsigned cnt = __popc(mask);
            atomicAdd(&hist[b], static_cast<std::uint32_t>(cnt));
        }
#else
        atomicAdd(&hist[b], 1u);
#endif
    }
}

__global__ void FillF32Kernel(float* __restrict dst, std::int64_t n, float value) {
    const std::int64_t tid = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * static_cast<std::int64_t>(gridDim.x);
    for (std::int64_t i = tid; i < n; i += stride) {
        dst[i] = value;
    }
}

__global__ void KmeansAnnealUpdateWeightsAndTotalsFromBestDotKernel(const float* __restrict best_dot,
                                                                    float* __restrict w_all,
                                                                    std::int64_t col0,
                                                                    int cols,
                                                                    float threshold,
                                                                    float annealed_factor,
                                                                    float min_weight,
                                                                    double* __restrict totals) {
    double sum_w = 0.0;
    double sum_cost = 0.0;
    const int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int stride = static_cast<int>(blockDim.x) * static_cast<int>(gridDim.x);
    for (int j = tid; j < cols; j += stride) {
        const float dot = best_dot[j];
        const float cost = 1.0f - dot;
        float w = w_all[col0 + static_cast<std::int64_t>(j)];
        if (cost > threshold) {
            w = fmaxf(min_weight, w * (1.0f - annealed_factor));
        } else {
            w = fminf(1.0f, w * (1.0f + 0.5f * annealed_factor));
        }
        w_all[col0 + static_cast<std::int64_t>(j)] = w;
        sum_w += static_cast<double>(w);
        sum_cost += static_cast<double>(w) * static_cast<double>(cost);
    }

    using BlockReduce = cub::BlockReduce<double, 256>;
    __shared__ typename BlockReduce::TempStorage temp;
    const double block_sum_w = BlockReduce(temp).Sum(sum_w);
    __syncthreads();
    const double block_sum_cost = BlockReduce(temp).Sum(sum_cost);
    if (threadIdx.x == 0) {
        atomicAdd(&totals[0], block_sum_w);
        atomicAdd(&totals[1], block_sum_cost);
    }
}

__global__ void KmeansAccumulateAnnealTotalsFromBestDotAndWeightsKernel(const float* __restrict best_dot,
                                                                       const float* __restrict w,
                                                                       int cols,
                                                                       double* __restrict totals) {
    double sum_w = 0.0;
    double sum_cost = 0.0;
    const int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int stride = static_cast<int>(blockDim.x) * static_cast<int>(gridDim.x);
    for (int j = tid; j < cols; j += stride) {
        const float dot = best_dot[j];
        const float cost = 1.0f - dot;
        const float ww = w[j];
        sum_w += static_cast<double>(ww);
        sum_cost += static_cast<double>(ww) * static_cast<double>(cost);
    }
    using BlockReduce = cub::BlockReduce<double, 256>;
    __shared__ typename BlockReduce::TempStorage temp;
    const double block_sum_w = BlockReduce(temp).Sum(sum_w);
    __syncthreads();
    const double block_sum_cost = BlockReduce(temp).Sum(sum_cost);
    if (threadIdx.x == 0) {
        atomicAdd(&totals[0], block_sum_w);
        atomicAdd(&totals[1], block_sum_cost);
    }
}

__global__ void KmeansAccumulateMseFromBestDotKernel(const float* __restrict best_dot,
                                                     const float* __restrict base_norm2,
                                                     const float* __restrict hatnorm2,
                                                     int cols,
                                                     double* __restrict metric_totals) {
    double sum_mse = 0.0;
    double cnt = 0.0;
    const int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int stride = static_cast<int>(blockDim.x) * static_cast<int>(gridDim.x);
    for (int j = tid; j < cols; j += stride) {
        const double bd = static_cast<double>(best_dot[j]);
        const double one_minus = 1.0 - bd * bd;
        const double n2 = static_cast<double>(base_norm2 ? base_norm2[j] : 1.0f);
        const double hn2 = static_cast<double>(hatnorm2 ? hatnorm2[j] : 1.0f);
        // residual_l2 = ||r_in||^2 * (1 - bestdot^2), where ||r_in||^2 = base_norm2 * hatnorm2.
        const double e = n2 * hn2 * one_minus;
        sum_mse += e;
        cnt += 1.0;
    }
    using BlockReduce = cub::BlockReduce<double, 256>;
    __shared__ typename BlockReduce::TempStorage temp;
    const double block_sum_mse = BlockReduce(temp).Sum(sum_mse);
    __syncthreads();
    const double block_sum_cnt = BlockReduce(temp).Sum(cnt);
    if (threadIdx.x == 0) {
        atomicAdd(&metric_totals[0], block_sum_mse);
        atomicAdd(&metric_totals[1], block_sum_cnt);
    }
}

__global__ void OutlierMaskFromBestDotKernel(const float* __restrict best_dot,
                                             int cols,
                                             float threshold,
                                             std::uint8_t* __restrict mask) {
    const int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int stride = static_cast<int>(blockDim.x) * static_cast<int>(gridDim.x);
    for (int j = tid; j < cols; j += stride) {
        const float cost = 1.0f - best_dot[j];
        mask[j] = static_cast<std::uint8_t>((cost > threshold) ? 1 : 0);
    }
}

struct DeviceBuffer {
    void* ptr = nullptr;
    std::size_t bytes = 0;

    ~DeviceBuffer() { Reset(); }

    void Reset() {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
        bytes = 0;
    }

    void Ensure(std::size_t need_bytes) {
        if (need_bytes <= bytes) return;
        Reset();
        ThrowIf(cudaMalloc(&ptr, need_bytes), "cudaMalloc");
        bytes = need_bytes;
    }
};

struct HostPinnedBuffer {
    void* ptr = nullptr;
    std::size_t bytes = 0;
    unsigned int flags = 0;

    ~HostPinnedBuffer() { Reset(); }

    void Reset() {
        if (ptr) {
            cudaFreeHost(ptr);
            ptr = nullptr;
        }
        bytes = 0;
        flags = 0;
    }

    void Ensure(std::size_t need_bytes) { EnsureWithFlags(need_bytes, cudaHostAllocPortable); }

    void EnsureWithFlags(std::size_t need_bytes, unsigned int alloc_flags) {
        if (need_bytes <= bytes && flags == alloc_flags) return;
        Reset();
        ThrowIf(cudaHostAlloc(&ptr, need_bytes, alloc_flags), "cudaHostAlloc");
        bytes = need_bytes;
        flags = alloc_flags;
    }
};

struct CachedDeviceMatF32 {
    const float* host_ptr = nullptr;
    int rows = 0;
    int cols = 0;
    std::uint64_t host_stamp = 0;
    DeviceBuffer buf;

    float* dptr() { return static_cast<float*>(buf.ptr); }
    const float* dptr() const { return static_cast<const float*>(buf.ptr); }

    static std::uint64_t Stamp(const ColMajorMatrix<float>& h) {
        // Lightweight content stamp to avoid stale-cache bugs when host storage is reused.
        // We sample a few elements across the matrix; this is cheap and reliable enough for
        // our usage (codebooks / rotations change broadly when updated).
        std::uint64_t x = 1469598103934665603ull;
        auto mix_u64 = [&](std::uint64_t v) {
            x ^= v;
            x *= 1099511628211ull;
        };
        mix_u64(static_cast<std::uint64_t>(h.rows));
        mix_u64(static_cast<std::uint64_t>(h.cols));
        const std::size_t n = h.data.size();
        mix_u64(static_cast<std::uint64_t>(n));
        if (n == 0) {
            return x;
        }
        const std::size_t samples = std::min<std::size_t>(32, n);
        const std::size_t stride = std::max<std::size_t>(1, n / samples);
        for (std::size_t i = 0; i < samples; ++i) {
            const std::size_t idx = std::min<std::size_t>(n - 1, i * stride);
            std::uint32_t bits = 0;
            std::memcpy(&bits, &h.data[idx], sizeof(bits));
            mix_u64(static_cast<std::uint64_t>(bits));
        }
        return x;
    }

    void UploadIfNeeded(const ColMajorMatrix<float>& h, cudaStream_t stream) {
        const std::uint64_t stamp = Stamp(h);
        const bool same =
            host_ptr == h.data.data() && rows == h.rows && cols == h.cols && host_stamp == stamp && !h.data.empty();
        if (same) return;
        host_ptr = h.data.data();
        rows = h.rows;
        cols = h.cols;
        host_stamp = stamp;
        const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
        buf.Ensure(n * sizeof(float));
        ThrowIf(cudaMemcpyAsync(buf.ptr, h.data.data(), n * sizeof(float),
                                cudaMemcpyHostToDevice, stream),
                "cudaMemcpyAsync H2D");
    }
};

}  // namespace

}  // namespace stlq
