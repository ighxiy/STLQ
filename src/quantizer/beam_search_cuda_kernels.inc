namespace {

constexpr int kMaxM = kMaxSupportedModelM;

inline void ThrowIf(cudaError_t st, const char* what) {
    if (st != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(st));
    }
}

inline void ThrowIf(cublasStatus_t st, const char* what) {
    if (st == CUBLAS_STATUS_SUCCESS) return;
    throw std::runtime_error(std::string(what) + ": cublasStatus=" + std::to_string(static_cast<int>(st)));
}

template <typename T>
struct DeviceBuf {
    T* ptr = nullptr;
    std::size_t n = 0;

    ~DeviceBuf() { Reset(); }

    void Reset() {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
        n = 0;
    }

    void Ensure(std::size_t n_need) {
        if (n_need <= n) return;
        Reset();
        ThrowIf(cudaMalloc(&ptr, sizeof(T) * n_need), "cudaMalloc");
        n = n_need;
    }
};

__global__ void Norm2ColsKernel(const float* __restrict X,
                                int ldX,
                                int d,
                                int n,
                                float* __restrict out_norm2) {
    const int col = blockIdx.x;
    if (col >= n) return;
    float sum = 0.0f;
    // d is small (e.g., 128); one block per column.
    for (int r = threadIdx.x; r < d; r += blockDim.x) {
        const float v = X[r + col * ldX];
        sum += v * v;
    }
    // block reduce
    __shared__ float sh[256];
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sh[threadIdx.x] += sh[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_norm2[col] = sh[0];
    }
}

struct Cand2 {
    float e0;
    int c0;
    float a0;
    float r0;
    float e1;
    int c1;
    float a1;
    float r1;
};

struct Cand4 {
    float e0;
    int c0;
    float a0;
    float r0;
    float e1;
    int c1;
    float a1;
    float r1;
    float e2;
    int c2;
    float a2;
    float r2;
    float e3;
    int c3;
    float a3;
    float r3;
};

__device__ inline bool Better(float e, int code, float best_e, int best_code) {
    // Primary: smaller error; tie: smaller code (matches earlier-scan behavior).
    return (e < best_e) || (e == best_e && code < best_code);
}

__device__ inline void Insert2(Cand2* st, float e, int code, float a, float r) {
    if (code == st->c0 || code == st->c1) return;
    if (Better(e, code, st->e0, st->c0)) {
        st->e1 = st->e0;
        st->c1 = st->c0;
        st->a1 = st->a0;
        st->r1 = st->r0;
        st->e0 = e;
        st->c0 = code;
        st->a0 = a;
        st->r0 = r;
        return;
    }
    if (Better(e, code, st->e1, st->c1)) {
        st->e1 = e;
        st->c1 = code;
        st->a1 = a;
        st->r1 = r;
    }
}

__device__ inline Cand2 Merge2(const Cand2& a, const Cand2& b) {
    Cand2 out = a;
    Insert2(&out, b.e0, b.c0, b.a0, b.r0);
    Insert2(&out, b.e1, b.c1, b.a1, b.r1);
    return out;
}

__device__ inline void Insert4(Cand4* st, float e, int code, float a, float r) {
    if (code == st->c0 || code == st->c1 || code == st->c2 || code == st->c3) return;
    if (Better(e, code, st->e0, st->c0)) {
        st->e3 = st->e2; st->c3 = st->c2; st->a3 = st->a2; st->r3 = st->r2;
        st->e2 = st->e1; st->c2 = st->c1; st->a2 = st->a1; st->r2 = st->r1;
        st->e1 = st->e0; st->c1 = st->c0; st->a1 = st->a0; st->r1 = st->r0;
        st->e0 = e; st->c0 = code; st->a0 = a; st->r0 = r;
        return;
    }
    if (Better(e, code, st->e1, st->c1)) {
        st->e3 = st->e2; st->c3 = st->c2; st->a3 = st->a2; st->r3 = st->r2;
        st->e2 = st->e1; st->c2 = st->c1; st->a2 = st->a1; st->r2 = st->r1;
        st->e1 = e; st->c1 = code; st->a1 = a; st->r1 = r;
        return;
    }
    if (Better(e, code, st->e2, st->c2)) {
        st->e3 = st->e2; st->c3 = st->c2; st->a3 = st->a2; st->r3 = st->r2;
        st->e2 = e; st->c2 = code; st->a2 = a; st->r2 = r;
        return;
    }
    if (Better(e, code, st->e3, st->c3)) {
        st->e3 = e; st->c3 = code; st->a3 = a; st->r3 = r;
    }
}

__device__ inline Cand4 Merge4(const Cand4& a, const Cand4& b) {
    Cand4 out = a;
    Insert4(&out, b.e0, b.c0, b.a0, b.r0);
    Insert4(&out, b.e1, b.c1, b.a1, b.r1);
    Insert4(&out, b.e2, b.c2, b.a2, b.r2);
    Insert4(&out, b.e3, b.c3, b.a3, b.r3);
    return out;
}

__global__ void UpdateTop2FromScoresKernel(const float* __restrict scores,
                                           int ld_scores,
                                           int rlen,
                                           int r0,
                                           const float* __restrict norm0,
                                           const float* __restrict norm_x2,
                                           int qlen,
                                           float* __restrict best_err0,
                                           int* __restrict best_code0,
                                           float* __restrict best_a0,
                                           float* __restrict best_r00,
                                           float* __restrict best_err1,
                                           int* __restrict best_code1,
                                           float* __restrict best_a1,
                                           float* __restrict best_r01) {
    const int j = blockIdx.x;
    if (j >= qlen) return;
    Cand2 local;
    local.e0 = INFINITY;
    local.c0 = -1;
    local.a0 = 0.0f;
    local.r0 = 0.0f;
    local.e1 = INFINITY;
    local.c1 = -1;
    local.a1 = 0.0f;
    local.r1 = 0.0f;

    const float nx2 = norm_x2[j];
    for (int kk = threadIdx.x; kk < rlen; kk += blockDim.x) {
        const int code = r0 + kk;
        const float alpha = norm0[code];
        if (alpha == 0.0f) continue;
        const float gamma = scores[kk + j * ld_scores];
        const float a = gamma / alpha;
        float e = nx2 - gamma * a;
        if (e < 0.0f) e = 0.0f;  // clamp: TF32/atomics may violate Cauchy-Schwarz, yielding e<0
        const float r = sqrtf(alpha);
        Insert2(&local, e, code, a, r);
    }

    // Reduce within block by merging candidates.
    __shared__ Cand2 sh[256];
    sh[threadIdx.x] = local;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sh[threadIdx.x] = Merge2(sh[threadIdx.x], sh[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    if (threadIdx.x != 0) return;

    // Merge chunk top2 into global top2.
    Cand2 global;
    global.e0 = best_err0[j];
    global.c0 = best_code0[j];
    global.a0 = best_a0[j];
    global.r0 = best_r00[j];
    global.e1 = best_err1[j];
    global.c1 = best_code1[j];
    global.a1 = best_a1[j];
    global.r1 = best_r01[j];

    global = Merge2(global, sh[0]);
    best_err0[j] = global.e0;
    best_code0[j] = global.c0;
    best_a0[j] = global.a0;
    best_r00[j] = global.r0;
    best_err1[j] = global.e1;
    best_code1[j] = global.c1;
    best_a1[j] = global.a1;
    best_r01[j] = global.r1;
}

__global__ void UpdateTop4FromScoresKernel(const float* __restrict scores,
                                           int ld_scores,
                                           int rlen,
                                           int r0,
                                           const float* __restrict norm0,
                                           const float* __restrict norm_x2,
                                           int qlen,
                                           float* __restrict best_err0,
                                           int* __restrict best_code0,
                                           float* __restrict best_a0,
                                           float* __restrict best_r00,
                                           float* __restrict best_err1,
                                           int* __restrict best_code1,
                                           float* __restrict best_a1,
                                           float* __restrict best_r01,
                                           float* __restrict best_err2,
                                           int* __restrict best_code2,
                                           float* __restrict best_a2,
                                           float* __restrict best_r02,
                                           float* __restrict best_err3,
                                           int* __restrict best_code3,
                                           float* __restrict best_a3,
                                           float* __restrict best_r03) {
    const int j = blockIdx.x;
    if (j >= qlen) return;
    Cand4 local;
    local.e0 = INFINITY; local.c0 = -1; local.a0 = 0.0f; local.r0 = 0.0f;
    local.e1 = INFINITY; local.c1 = -1; local.a1 = 0.0f; local.r1 = 0.0f;
    local.e2 = INFINITY; local.c2 = -1; local.a2 = 0.0f; local.r2 = 0.0f;
    local.e3 = INFINITY; local.c3 = -1; local.a3 = 0.0f; local.r3 = 0.0f;

    const float nx2 = norm_x2[j];
    for (int kk = threadIdx.x; kk < rlen; kk += blockDim.x) {
        const int code = r0 + kk;
        const float alpha = norm0[code];
        if (alpha == 0.0f) continue;
        const float gamma = scores[kk + j * ld_scores];
        const float a = gamma / alpha;
        float e = nx2 - gamma * a;
        if (e < 0.0f) e = 0.0f;  // clamp: TF32/atomics may violate Cauchy-Schwarz, yielding e<0
        const float r = sqrtf(alpha);
        Insert4(&local, e, code, a, r);
    }

    __shared__ Cand4 sh[256];
    sh[threadIdx.x] = local;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sh[threadIdx.x] = Merge4(sh[threadIdx.x], sh[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    if (threadIdx.x != 0) return;

    Cand4 global;
    global.e0 = best_err0[j]; global.c0 = best_code0[j]; global.a0 = best_a0[j]; global.r0 = best_r00[j];
    global.e1 = best_err1[j]; global.c1 = best_code1[j]; global.a1 = best_a1[j]; global.r1 = best_r01[j];
    global.e2 = best_err2[j]; global.c2 = best_code2[j]; global.a2 = best_a2[j]; global.r2 = best_r02[j];
    global.e3 = best_err3[j]; global.c3 = best_code3[j]; global.a3 = best_a3[j]; global.r3 = best_r03[j];

    global = Merge4(global, sh[0]);
    best_err0[j] = global.e0; best_code0[j] = global.c0; best_a0[j] = global.a0; best_r00[j] = global.r0;
    best_err1[j] = global.e1; best_code1[j] = global.c1; best_a1[j] = global.a1; best_r01[j] = global.r1;
    best_err2[j] = global.e2; best_code2[j] = global.c2; best_a2[j] = global.a2; best_r02[j] = global.r2;
    best_err3[j] = global.e3; best_code3[j] = global.c3; best_a3[j] = global.a3; best_r03[j] = global.r3;
}

struct CudaRootBeamSession {
    int d = 0;
    int h0 = 0;
    std::uint64_t tag = 0;
    bool uploaded = false;

    cublasHandle_t handle = nullptr;
    bool handle_tf32 = false;
    bool handle_atomics_allowed = true;

    DeviceBuf<float> d_C0;     // d*h0
    DeviceBuf<float> d_norm0;  // h0

    DeviceBuf<float> d_X;      // d*qlen
    DeviceBuf<float> d_scores; // rlen*qlen
    DeviceBuf<float> d_normx2; // qlen

    DeviceBuf<float> d_best_err0;
    DeviceBuf<int> d_best_code0;
    DeviceBuf<float> d_best_a0;
    DeviceBuf<float> d_best_r00;
    DeviceBuf<float> d_best_err1;
    DeviceBuf<int> d_best_code1;
    DeviceBuf<float> d_best_a1;
    DeviceBuf<float> d_best_r01;
    DeviceBuf<float> d_best_err2;
    DeviceBuf<int> d_best_code2;
    DeviceBuf<float> d_best_a2;
    DeviceBuf<float> d_best_r02;
    DeviceBuf<float> d_best_err3;
    DeviceBuf<int> d_best_code3;
    DeviceBuf<float> d_best_a3;
    DeviceBuf<float> d_best_r03;

    ~CudaRootBeamSession() {
        if (handle) {
            cublasDestroy(handle);
            handle = nullptr;
        }
    }

    void EnsureHandle(bool allow_tf32) {
        if (!handle) {
            ThrowIf(cublasCreate(&handle), "cublasCreate");
        }
        if (handle_tf32 != allow_tf32) {
            handle_tf32 = allow_tf32;
            if (allow_tf32) {
                ThrowIf(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH), "cublasSetMathMode(TF32)");
            } else {
                ThrowIf(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH), "cublasSetMathMode(DEFAULT)");
            }
        }

        // Reproducibility: disallow atomic-based GEMM algorithms when TF32 is disabled.
        // This primarily targets "strict" runs (TF32 off by default) where we prefer deterministic
        // reductions over maximum throughput.
        const bool want_atomics = allow_tf32;
        if (handle_atomics_allowed != want_atomics) {
            handle_atomics_allowed = want_atomics;
            ThrowIf(cublasSetAtomicsMode(handle,
                                         want_atomics ? CUBLAS_ATOMICS_ALLOWED : CUBLAS_ATOMICS_NOT_ALLOWED),
                    "cublasSetAtomicsMode");
        }
    }

    void EnsurePre(const PrecompLargeRoot& pre) {
        d = pre.d;
        h0 = pre.h_vec.empty() ? 0 : pre.h_vec[0];
        const std::uint64_t new_tag =
            static_cast<std::uint64_t>(pre.build_tag) ^
            (static_cast<std::uint64_t>(d) << 1) ^
            (static_cast<std::uint64_t>(h0) << 33);
        const bool same = uploaded && (new_tag == tag);
        tag = new_tag;
        d_C0.Ensure(static_cast<std::size_t>(d) * static_cast<std::size_t>(h0));
        d_norm0.Ensure(static_cast<std::size_t>(h0));
        if (same) return;
        ThrowIf(cudaMemcpy(d_C0.ptr, pre.C0.data.data(),
                           sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(h0),
                           cudaMemcpyHostToDevice),
                "cudaMemcpy C0 H2D");
        ThrowIf(cudaMemcpy(d_norm0.ptr, pre.norm0.data(),
                           sizeof(float) * static_cast<std::size_t>(h0),
                           cudaMemcpyHostToDevice),
                "cudaMemcpy norm0 H2D");
        uploaded = true;
    }

    void EnsureBatch(int qlen, int rlen) {
        d_X.Ensure(static_cast<std::size_t>(d) * static_cast<std::size_t>(qlen));
        d_scores.Ensure(static_cast<std::size_t>(rlen) * static_cast<std::size_t>(qlen));
        d_normx2.Ensure(static_cast<std::size_t>(qlen));

        d_best_err0.Ensure(static_cast<std::size_t>(qlen));
        d_best_code0.Ensure(static_cast<std::size_t>(qlen));
        d_best_a0.Ensure(static_cast<std::size_t>(qlen));
        d_best_r00.Ensure(static_cast<std::size_t>(qlen));
        d_best_err1.Ensure(static_cast<std::size_t>(qlen));
        d_best_code1.Ensure(static_cast<std::size_t>(qlen));
        d_best_a1.Ensure(static_cast<std::size_t>(qlen));
        d_best_r01.Ensure(static_cast<std::size_t>(qlen));
        d_best_err2.Ensure(static_cast<std::size_t>(qlen));
        d_best_code2.Ensure(static_cast<std::size_t>(qlen));
        d_best_a2.Ensure(static_cast<std::size_t>(qlen));
        d_best_r02.Ensure(static_cast<std::size_t>(qlen));
        d_best_err3.Ensure(static_cast<std::size_t>(qlen));
        d_best_code3.Ensure(static_cast<std::size_t>(qlen));
        d_best_a3.Ensure(static_cast<std::size_t>(qlen));
        d_best_r03.Ensure(static_cast<std::size_t>(qlen));
    }
};

// CUDA expand session for large-root beam with H_beam=2.
// Keeps large-root (layer0) candidates and expands layers 1..m-1 fully on GPU.
//
// Correctness: matches the CPU expand semantics (same scan order, denom==0 skip, e>=worst prune,
// tie behavior for H=2 worst selection).
struct CudaExpandH2Session {
    int m = 0;
    int Hs = 0;
    int h0 = 0;
    std::uint64_t tag = 0;
    bool uploaded = false;

    DeviceBuf<float> d_Gs;   // Hs*Hs (col-major)
    DeviceBuf<float> d_G0S;  // h0*Hs (col-major)
    DeviceBuf<int> d_small_offsets; // size m
    DeviceBuf<int> d_hvec;          // size m

    DeviceBuf<float> d_xC;  // Hs*qlen (col-major)

    // Current / next beam state (beam-major per sample: idx = (j*2 + b)).
    DeviceBuf<int> d_root_curr;
    DeviceBuf<int> d_root_next;
    DeviceBuf<Code> d_codes_curr; // (m-1)*qlen*2
    DeviceBuf<Code> d_codes_next; // (m-1)*qlen*2
    DeviceBuf<float> d_err_curr;  // qlen*2
    DeviceBuf<float> d_err_next;  // qlen*2
    DeviceBuf<float> d_a_curr;    // m*qlen*2
    DeviceBuf<float> d_a_next;    // m*qlen*2
    DeviceBuf<float> d_R_curr;    // (m*m)*qlen*2, row-major with stride=m
    DeviceBuf<float> d_R_next;    // (m*m)*qlen*2

    DeviceBuf<std::uint32_t> d_cluster_id; // qlen
    DeviceBuf<Code> d_B_small; // (m-1)*qlen

    void EnsurePre(const PrecompLargeRoot& pre) {
        m = pre.m;
        Hs = pre.H_small;
        h0 = pre.h_vec.empty() ? 0 : pre.h_vec[0];
        const std::uint64_t new_tag =
            static_cast<std::uint64_t>(pre.build_tag) ^
            (static_cast<std::uint64_t>(m) << 1) ^
            (static_cast<std::uint64_t>(Hs) << 21) ^
            (static_cast<std::uint64_t>(h0) << 43);
        const bool same = uploaded && (new_tag == tag);
        tag = new_tag;

        d_Gs.Ensure(static_cast<std::size_t>(Hs) * static_cast<std::size_t>(Hs));
        d_G0S.Ensure(static_cast<std::size_t>(h0) * static_cast<std::size_t>(Hs));
        d_small_offsets.Ensure(static_cast<std::size_t>(std::max(1, m)));
        d_hvec.Ensure(static_cast<std::size_t>(std::max(1, m)));
        if (same) return;

        if (!pre.G_small.data.empty()) {
            ThrowIf(cudaMemcpy(d_Gs.ptr, pre.G_small.data.data(),
                               sizeof(float) * static_cast<std::size_t>(Hs) * static_cast<std::size_t>(Hs),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy G_small H2D");
        }
        if (!pre.G0S.data.empty()) {
            ThrowIf(cudaMemcpy(d_G0S.ptr, pre.G0S.data.data(),
                               sizeof(float) * static_cast<std::size_t>(h0) * static_cast<std::size_t>(Hs),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy G0S H2D");
        }
        if (!pre.small_offsets.empty()) {
            ThrowIf(cudaMemcpy(d_small_offsets.ptr, pre.small_offsets.data(),
                               sizeof(int) * static_cast<std::size_t>(m),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy small_offsets H2D");
        }
        if (!pre.h_vec.empty()) {
            ThrowIf(cudaMemcpy(d_hvec.ptr, pre.h_vec.data(),
                               sizeof(int) * static_cast<std::size_t>(m),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy h_vec H2D");
        }
        uploaded = true;
    }

    void EnsureBatch(int qlen) {
        d_xC.Ensure(static_cast<std::size_t>(Hs) * static_cast<std::size_t>(qlen));

        d_root_curr.Ensure(static_cast<std::size_t>(qlen) * 2);
        d_root_next.Ensure(static_cast<std::size_t>(qlen) * 2);
        d_codes_curr.Ensure(static_cast<std::size_t>(std::max(0, m - 1)) * static_cast<std::size_t>(qlen) * 2);
        d_codes_next.Ensure(static_cast<std::size_t>(std::max(0, m - 1)) * static_cast<std::size_t>(qlen) * 2);
        d_err_curr.Ensure(static_cast<std::size_t>(qlen) * 2);
        d_err_next.Ensure(static_cast<std::size_t>(qlen) * 2);
        d_a_curr.Ensure(static_cast<std::size_t>(std::max(1, m)) * static_cast<std::size_t>(qlen) * 2);
        d_a_next.Ensure(static_cast<std::size_t>(std::max(1, m)) * static_cast<std::size_t>(qlen) * 2);
        d_R_curr.Ensure(static_cast<std::size_t>(std::max(1, m * m)) * static_cast<std::size_t>(qlen) * 2);
        d_R_next.Ensure(static_cast<std::size_t>(std::max(1, m * m)) * static_cast<std::size_t>(qlen) * 2);

        d_cluster_id.Ensure(static_cast<std::size_t>(qlen));
        d_B_small.Ensure(static_cast<std::size_t>(std::max(0, m - 1)) * static_cast<std::size_t>(qlen));
    }
};

struct CudaExpandH4Session {
    int m = 0;
    int Hs = 0;
    int h0 = 0;
    std::uint64_t tag = 0;
    bool uploaded = false;

    DeviceBuf<float> d_Gs;   // Hs*Hs
    DeviceBuf<float> d_G0S;  // h0*Hs
    DeviceBuf<int> d_small_offsets; // size m
    DeviceBuf<int> d_hvec;          // size m

    DeviceBuf<float> d_xC;  // Hs*qlen

    DeviceBuf<int> d_root_curr;
    DeviceBuf<int> d_root_next;
    DeviceBuf<Code> d_codes_curr; // (m-1)*qlen*4
    DeviceBuf<Code> d_codes_next;
    DeviceBuf<float> d_err_curr;  // qlen*4
    DeviceBuf<float> d_err_next;
    DeviceBuf<float> d_a_curr;    // m*qlen*4
    DeviceBuf<float> d_a_next;
    DeviceBuf<float> d_R_curr;    // (m*m)*qlen*4
    DeviceBuf<float> d_R_next;

    DeviceBuf<std::uint32_t> d_cluster_id; // qlen
    DeviceBuf<Code> d_B_small; // (m-1)*qlen

    void EnsurePre(const PrecompLargeRoot& pre) {
        m = pre.m;
        Hs = pre.H_small;
        h0 = pre.h_vec.empty() ? 0 : pre.h_vec[0];
        const std::uint64_t new_tag =
            static_cast<std::uint64_t>(pre.build_tag) ^
            (static_cast<std::uint64_t>(m) << 1) ^
            (static_cast<std::uint64_t>(Hs) << 21) ^
            (static_cast<std::uint64_t>(h0) << 43) ^
            0x4ull;
        const bool same = uploaded && (new_tag == tag);
        tag = new_tag;

        d_Gs.Ensure(static_cast<std::size_t>(Hs) * static_cast<std::size_t>(Hs));
        d_G0S.Ensure(static_cast<std::size_t>(h0) * static_cast<std::size_t>(Hs));
        d_small_offsets.Ensure(static_cast<std::size_t>(std::max(1, m)));
        d_hvec.Ensure(static_cast<std::size_t>(std::max(1, m)));
        if (same) return;

        if (!pre.G_small.data.empty()) {
            ThrowIf(cudaMemcpy(d_Gs.ptr, pre.G_small.data.data(),
                               sizeof(float) * static_cast<std::size_t>(Hs) * static_cast<std::size_t>(Hs),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy G_small H2D (H4)");
        }
        if (!pre.G0S.data.empty()) {
            ThrowIf(cudaMemcpy(d_G0S.ptr, pre.G0S.data.data(),
                               sizeof(float) * static_cast<std::size_t>(h0) * static_cast<std::size_t>(Hs),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy G0S H2D (H4)");
        }
        if (!pre.small_offsets.empty()) {
            ThrowIf(cudaMemcpy(d_small_offsets.ptr, pre.small_offsets.data(),
                               sizeof(int) * static_cast<std::size_t>(m),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy small_offsets H2D (H4)");
        }
        if (!pre.h_vec.empty()) {
            ThrowIf(cudaMemcpy(d_hvec.ptr, pre.h_vec.data(),
                               sizeof(int) * static_cast<std::size_t>(m),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy h_vec H2D (H4)");
        }
        uploaded = true;
    }

    void EnsureBatch(int qlen) {
        d_xC.Ensure(static_cast<std::size_t>(Hs) * static_cast<std::size_t>(qlen));

        d_root_curr.Ensure(static_cast<std::size_t>(qlen) * 4);
        d_root_next.Ensure(static_cast<std::size_t>(qlen) * 4);
        d_codes_curr.Ensure(static_cast<std::size_t>(std::max(0, m - 1)) * static_cast<std::size_t>(qlen) * 4);
        d_codes_next.Ensure(static_cast<std::size_t>(std::max(0, m - 1)) * static_cast<std::size_t>(qlen) * 4);
        d_err_curr.Ensure(static_cast<std::size_t>(qlen) * 4);
        d_err_next.Ensure(static_cast<std::size_t>(qlen) * 4);
        d_a_curr.Ensure(static_cast<std::size_t>(std::max(1, m)) * static_cast<std::size_t>(qlen) * 4);
        d_a_next.Ensure(static_cast<std::size_t>(std::max(1, m)) * static_cast<std::size_t>(qlen) * 4);
        d_R_curr.Ensure(static_cast<std::size_t>(std::max(1, m * m)) * static_cast<std::size_t>(qlen) * 4);
        d_R_next.Ensure(static_cast<std::size_t>(std::max(1, m * m)) * static_cast<std::size_t>(qlen) * 4);

        d_cluster_id.Ensure(static_cast<std::size_t>(qlen));
        d_B_small.Ensure(static_cast<std::size_t>(std::max(0, m - 1)) * static_cast<std::size_t>(qlen));
    }
};

template <int N>
__device__ inline void SolveCholUpperFixed(const float* __restrict__ R,
                                           const float* __restrict__ b,
                                           float* __restrict__ x,
                                           float* __restrict__ y,
                                           int stride) {
    // Forward solve U^T y = b
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        float s = b[i];
        #pragma unroll
        for (int k = 0; k < i; ++k) {
            s -= R[static_cast<std::size_t>(k) * static_cast<std::size_t>(stride) + i] * y[k];
        }
        y[i] = s / R[static_cast<std::size_t>(i) * static_cast<std::size_t>(stride) + i];
    }
    // Backward solve U x = y
    #pragma unroll
    for (int i = N - 1; i >= 0; --i) {
        float s = y[i];
        #pragma unroll
        for (int k = i + 1; k < N; ++k) {
            s -= R[static_cast<std::size_t>(i) * static_cast<std::size_t>(stride) + k] * x[k];
        }
        x[i] = s / R[static_cast<std::size_t>(i) * static_cast<std::size_t>(stride) + i];
    }
}

__device__ inline void SolveCholUpperDispatch(const float* __restrict__ R,
                                              int n,
                                              const float* __restrict__ b,
                                              float* __restrict__ x,
                                              float* __restrict__ y,
                                              int stride) {
    switch (n) {
        case 1: SolveCholUpperFixed<1>(R, b, x, y, stride); return;
        case 2: SolveCholUpperFixed<2>(R, b, x, y, stride); return;
        case 3: SolveCholUpperFixed<3>(R, b, x, y, stride); return;
        case 4: SolveCholUpperFixed<4>(R, b, x, y, stride); return;
        case 5: SolveCholUpperFixed<5>(R, b, x, y, stride); return;
        case 6: SolveCholUpperFixed<6>(R, b, x, y, stride); return;
        case 7: SolveCholUpperFixed<7>(R, b, x, y, stride); return;
        case 8: SolveCholUpperFixed<8>(R, b, x, y, stride); return;
        case 9: SolveCholUpperFixed<9>(R, b, x, y, stride); return;
        case 10: SolveCholUpperFixed<10>(R, b, x, y, stride); return;
        case 11: SolveCholUpperFixed<11>(R, b, x, y, stride); return;
        case 12: SolveCholUpperFixed<12>(R, b, x, y, stride); return;
        case 13: SolveCholUpperFixed<13>(R, b, x, y, stride); return;
        case 14: SolveCholUpperFixed<14>(R, b, x, y, stride); return;
        case 15: SolveCholUpperFixed<15>(R, b, x, y, stride); return;
        case 16: SolveCholUpperFixed<16>(R, b, x, y, stride); return;
        default: break;
    }
    // Layers above the fixed specializations use the generic solver.
    // Forward solve U^T y = b
    for (int i = 0; i < n; ++i) {
        float s = b[i];
        for (int k = 0; k < i; ++k) {
            s -= R[static_cast<std::size_t>(k) * static_cast<std::size_t>(stride) + i] * y[k];
        }
        y[i] = s / R[static_cast<std::size_t>(i) * static_cast<std::size_t>(stride) + i];
    }
    // Backward solve U x = y
    for (int i = n - 1; i >= 0; --i) {
        float s = y[i];
        for (int k = i + 1; k < n; ++k) {
            s -= R[static_cast<std::size_t>(i) * static_cast<std::size_t>(stride) + k] * x[k];
        }
        x[i] = s / R[static_cast<std::size_t>(i) * static_cast<std::size_t>(stride) + i];
    }
}

__device__ inline int WorstIdx2(float e0, float e1) {
    // Match CPU WorstIdx(H=2): worst stays 0 on ties.
    return (e1 > e0) ? 1 : 0;
}

__global__ void InitBeam2FromRootTop2Kernel(int m,
                                            int qlen,
                                            const float* __restrict__ best_err0,
                                            const int* __restrict__ best_code0,
                                            const float* __restrict__ best_a0,
                                            const float* __restrict__ best_r00,
                                            const float* __restrict__ best_err1,
                                            const int* __restrict__ best_code1,
                                            const float* __restrict__ best_a1,
                                            const float* __restrict__ best_r01,
                                            int* __restrict__ root_curr,
                                            Code* __restrict__ codes_curr,
                                            float* __restrict__ err_curr,
                                            float* __restrict__ a_curr,
                                            float* __restrict__ R_curr) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= qlen) return;
    const int beams = 2;
    const int codes_rows = (m > 1) ? (m - 1) : 0;

    // For beam 0 and beam 1.
    for (int b = 0; b < beams; ++b) {
        const int idx = j * beams + b;
        const bool is_b1 = (b == 1);
        const float e = is_b1 ? best_err1[j] : best_err0[j];
        const int rc = is_b1 ? best_code1[j] : best_code0[j];
        const float a0 = is_b1 ? best_a1[j] : best_a0[j];
        const float r00 = is_b1 ? best_r01[j] : best_r00[j];

        root_curr[idx] = rc;
        err_curr[idx] = e;

        // a_curr: zero then set a0 at layer0.
        float* arow = a_curr + static_cast<std::size_t>(idx) * static_cast<std::size_t>(m);
        for (int t = 0; t < m; ++t) arow[t] = 0.0f;
        arow[0] = a0;

        // R_curr: zero then set (0,0)=r00.
        float* R = R_curr + static_cast<std::size_t>(idx) * static_cast<std::size_t>(m) * static_cast<std::size_t>(m);
        for (int t = 0; t < m * m; ++t) R[t] = 0.0f;
        R[0] = r00;

        // codes_curr: initialize to 0 (layers 1..m-1).
        Code* c = codes_curr + static_cast<std::size_t>(idx) * static_cast<std::size_t>(codes_rows);
        for (int t = 0; t < codes_rows; ++t) c[t] = static_cast<Code>(0);
    }
}

__global__ void ExpandLayerBeam2Kernel(int m,
                                       int Hs,
                                       int G0S_rows,
                                       int layer,
                                       int qlen,
                                       const float* __restrict__ Gs,
                                       const float* __restrict__ G0S,
                                       const int* __restrict__ small_offsets,
                                       const int* __restrict__ h_vec,
                                       const float* __restrict__ xC_small,
                                       const int* __restrict__ root_curr,
                                       const Code* __restrict__ codes_curr,
                                       const float* __restrict__ err_curr,
                                       const float* __restrict__ a_curr,
                                       const float* __restrict__ R_curr,
                                       int* __restrict__ root_next,
                                       Code* __restrict__ codes_next,
                                       float* __restrict__ err_next,
                                       float* __restrict__ a_next,
                                       float* __restrict__ R_next) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= qlen) return;
    const int beams = 2;
    const int codes_rows = (m > 1) ? (m - 1) : 0;
    const int K = h_vec[layer];
    const int off = small_offsets[layer];
    const float inf = INFINITY;

    float best_e[2] = {inf, inf};
    int sel_src[2] = {-1, -1};
    int sel_k[2] = {0, 0};
    int sel_root[2] = {0, 0};
    float sel_denom[2] = {0.0f, 0.0f};
    float sel_alast[2] = {0.0f, 0.0f};
    float sel_v[2][kMaxM];
    float sel_y[2][kMaxM];
    #pragma unroll
    for (int b = 0; b < 2; ++b) {
        #pragma unroll
        for (int t = 0; t < kMaxM; ++t) {
            sel_v[b][t] = 0.0f;
            sel_y[b][t] = 0.0f;
        }
    }

    int worst_idx = 0;
    float worst_err = best_e[0];

    // Scan order matches CPU: h then k.
    for (int h = 0; h < beams; ++h) {
        const int idx = j * beams + h;
        const float E_old = err_curr[idx];
        if (!(E_old < inf)) continue;
        const int root_code = root_curr[idx];

        const float* __restrict a_old = a_curr + static_cast<std::size_t>(idx) * static_cast<std::size_t>(m);
        const float* __restrict R_pref = R_curr + static_cast<std::size_t>(idx) * static_cast<std::size_t>(m) *
                                             static_cast<std::size_t>(m);
        const Code* __restrict c_old = codes_curr + static_cast<std::size_t>(idx) * static_cast<std::size_t>(codes_rows);

        for (int k = 0; k < K; ++k) {
            const int flat = off + k;
            const float alpha = Gs[static_cast<std::size_t>(flat) +
                                   static_cast<std::size_t>(flat) * static_cast<std::size_t>(Hs)];
            const float gamma =
                xC_small[static_cast<std::size_t>(flat) + static_cast<std::size_t>(j) * static_cast<std::size_t>(Hs)];

            float A[kMaxM];
            float v[kMaxM];
            float y[kMaxM];
            #pragma unroll
            for (int t = 0; t < kMaxM; ++t) {
                A[t] = 0.0f;
                v[t] = 0.0f;
                y[t] = 0.0f;
            }

            // A[0] is root-vs-small cross term.
            A[0] = G0S[static_cast<std::size_t>(root_code) +
                       static_cast<std::size_t>(flat) * static_cast<std::size_t>(G0S_rows)];
            // A[t] for previous small layers (t=1..layer-1).
            for (int t = 1; t < layer; ++t) {
                const int flat_t = small_offsets[t] + static_cast<int>(c_old[t - 1]);
                A[t] = Gs[static_cast<std::size_t>(flat_t) +
                          static_cast<std::size_t>(flat) * static_cast<std::size_t>(Hs)];
            }

            SolveCholUpperDispatch(R_pref, layer, A, v, y, m);

            float delta = 0.0f;
            float eta = 0.0f;
            for (int t = 0; t < layer; ++t) {
                const float at = A[t];
                delta += at * v[t];
                eta += at * a_old[t];
            }

            const float denom = alpha - delta;
            if (denom == 0.0f) continue;  // Julia/CPU semantics.

            const float diff = gamma - eta;
            const float a_last = diff / denom;
            float e = E_old - diff * a_last;
            if (e < 0.0f) e = 0.0f;

            if (e >= worst_err) continue;
            const int out = worst_idx;
            best_e[out] = e;
            sel_src[out] = h;
            sel_k[out] = k;
            sel_root[out] = root_code;
            sel_denom[out] = denom;
            sel_alast[out] = a_last;
            for (int t = 0; t < layer; ++t) {
                sel_v[out][t] = v[t];
                sel_y[out][t] = y[t];
            }

            worst_idx = WorstIdx2(best_e[0], best_e[1]);
            worst_err = best_e[worst_idx];
        }
    }

    const int n_finite = (best_e[0] < inf) + (best_e[1] < inf);
    if (n_finite == 0) {
        // Fallback: copy min(curr_H,2) beams in order.
        int curr_H = 0;
        float e0 = err_curr[j * beams + 0];
        float e1 = err_curr[j * beams + 1];
        curr_H = (e0 < inf) + (e1 < inf);
        if (curr_H == 0) curr_H = 1;
        const int newH = (curr_H < 2) ? curr_H : 2;
        for (int out = 0; out < beams; ++out) {
            const int dst = j * beams + out;
            if (out < newH) {
                const int src = j * beams + out;
                err_next[dst] = err_curr[src];
                root_next[dst] = root_curr[src];
                // copy codes up to layer
                const Code* cs = codes_curr + static_cast<std::size_t>(src) * static_cast<std::size_t>(codes_rows);
                Code* cd = codes_next + static_cast<std::size_t>(dst) * static_cast<std::size_t>(codes_rows);
                for (int t = 0; t < layer; ++t) {
                    cd[t] = cs[t];
                }
                // copy a up to layer
                const float* as = a_curr + static_cast<std::size_t>(src) * static_cast<std::size_t>(m);
                float* ad = a_next + static_cast<std::size_t>(dst) * static_cast<std::size_t>(m);
                for (int t = 0; t <= layer; ++t) {
                    ad[t] = as[t];
                }
                // copy R full (m*m) for simplicity (small).
                const float* Rs = R_curr + static_cast<std::size_t>(src) * static_cast<std::size_t>(m) *
                                      static_cast<std::size_t>(m);
                float* Rd = R_next + static_cast<std::size_t>(dst) * static_cast<std::size_t>(m) *
                                static_cast<std::size_t>(m);
                for (int t = 0; t < m * m; ++t) {
                    Rd[t] = Rs[t];
                }
            } else {
                err_next[dst] = inf;
            }
        }
        return;
    }

    // Materialize selected beams.
    for (int out = 0; out < beams; ++out) {
        const int dst = j * beams + out;
        if (!(best_e[out] < inf)) {
            err_next[dst] = inf;
            continue;
        }
        const int src_h = sel_src[out];
        const int src = j * beams + src_h;
        err_next[dst] = best_e[out];
        root_next[dst] = sel_root[out];

        const Code* cs = codes_curr + static_cast<std::size_t>(src) * static_cast<std::size_t>(codes_rows);
        Code* cd = codes_next + static_cast<std::size_t>(dst) * static_cast<std::size_t>(codes_rows);
        // copy previous small codes (layers 1..layer-1)
        for (int t = 1; t < layer; ++t) {
            cd[t - 1] = cs[t - 1];
        }
        cd[layer - 1] = static_cast<Code>(sel_k[out]);

        const float* as = a_curr + static_cast<std::size_t>(src) * static_cast<std::size_t>(m);
        float* ad = a_next + static_cast<std::size_t>(dst) * static_cast<std::size_t>(m);
        const float a_last = sel_alast[out];
        for (int t = 0; t < layer; ++t) {
            ad[t] = as[t] - sel_v[out][t] * a_last;
        }
        ad[layer] = a_last;

        const float* Rs = R_curr + static_cast<std::size_t>(src) * static_cast<std::size_t>(m) *
                              static_cast<std::size_t>(m);
        float* Rd = R_next + static_cast<std::size_t>(dst) * static_cast<std::size_t>(m) *
                        static_cast<std::size_t>(m);
        // Copy row-major prefix rows [0..layer-1].
        for (int r = 0; r < layer; ++r) {
            const float* src_row = Rs + static_cast<std::size_t>(r) * static_cast<std::size_t>(m);
            float* dst_row = Rd + static_cast<std::size_t>(r) * static_cast<std::size_t>(m);
            for (int c = 0; c < m; ++c) {
                dst_row[c] = src_row[c];
            }
        }
        for (int t = 0; t < layer; ++t) {
            Rd[static_cast<std::size_t>(t) * static_cast<std::size_t>(m) + static_cast<std::size_t>(layer)] =
                sel_y[out][t];
        }
        Rd[static_cast<std::size_t>(layer) * static_cast<std::size_t>(m) + static_cast<std::size_t>(layer)] =
            sqrtf(sel_denom[out]);
    }
}

__global__ void FinalizeBestBeam2Kernel(int m,
                                        int qlen,
                                        const int* __restrict__ root_curr,
                                        const Code* __restrict__ codes_curr,
                                        const float* __restrict__ err_curr,
                                        std::uint32_t* __restrict__ cluster_id,
                                        Code* __restrict__ B_small) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= qlen) return;
    const int beams = 2;
    const int codes_rows = (m > 1) ? (m - 1) : 0;
    const int idx0 = j * beams + 0;
    const int idx1 = j * beams + 1;
    const float e0 = err_curr[idx0];
    const float e1 = err_curr[idx1];
    const int best = (e1 < e0) ? 1 : 0;  // tie => 0
    const int idx = j * beams + best;

    cluster_id[j] = static_cast<std::uint32_t>(root_curr[idx]);
    const Code* cs = codes_curr + static_cast<std::size_t>(idx) * static_cast<std::size_t>(codes_rows);
    Code* out = B_small + static_cast<std::size_t>(j) * static_cast<std::size_t>(codes_rows);
    for (int t = 0; t < codes_rows; ++t) {
        out[t] = cs[t];
    }
}

__device__ inline int WorstIdx4(const float* errs) {
    // Match CPU WorstIdx for H>2: strict '>' keeps earliest max on ties.
    int worst = 0;
    float we = errs[0];
    for (int i = 1; i < 4; ++i) {
        const float e = errs[i];
        if (e > we) {
            we = e;
            worst = i;
        }
    }
    return worst;
}

__global__ void InitBeam4FromRootTop4Kernel(int m,
                                            int qlen,
                                            const float* __restrict__ best_err0,
                                            const int* __restrict__ best_code0,
                                            const float* __restrict__ best_a0,
                                            const float* __restrict__ best_r00,
                                            const float* __restrict__ best_err1,
                                            const int* __restrict__ best_code1,
                                            const float* __restrict__ best_a1,
                                            const float* __restrict__ best_r01,
                                            const float* __restrict__ best_err2,
                                            const int* __restrict__ best_code2,
                                            const float* __restrict__ best_a2,
                                            const float* __restrict__ best_r02,
                                            const float* __restrict__ best_err3,
                                            const int* __restrict__ best_code3,
                                            const float* __restrict__ best_a3,
                                            const float* __restrict__ best_r03,
                                            int* __restrict__ root_curr,
                                            Code* __restrict__ codes_curr,
                                            float* __restrict__ err_curr,
                                            float* __restrict__ a_curr,
                                            float* __restrict__ R_curr) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= qlen) return;
    const int beams = 4;
    const int codes_rows = (m > 1) ? (m - 1) : 0;

    const float e_in[4] = {best_err0[j], best_err1[j], best_err2[j], best_err3[j]};
    const int c_in[4] = {best_code0[j], best_code1[j], best_code2[j], best_code3[j]};
    const float a_in[4] = {best_a0[j], best_a1[j], best_a2[j], best_a3[j]};
    const float r_in[4] = {best_r00[j], best_r01[j], best_r02[j], best_r03[j]};

    for (int b = 0; b < beams; ++b) {
        const int idx = j * beams + b;
        root_curr[idx] = c_in[b];
        err_curr[idx] = e_in[b];
        float* arow = a_curr + static_cast<std::size_t>(idx) * static_cast<std::size_t>(m);
        for (int t = 0; t < m; ++t) arow[t] = 0.0f;
        arow[0] = a_in[b];
        float* R = R_curr + static_cast<std::size_t>(idx) * static_cast<std::size_t>(m) * static_cast<std::size_t>(m);
        for (int t = 0; t < m * m; ++t) R[t] = 0.0f;
        R[0] = r_in[b];
        Code* c = codes_curr + static_cast<std::size_t>(idx) * static_cast<std::size_t>(codes_rows);
        for (int t = 0; t < codes_rows; ++t) c[t] = static_cast<Code>(0);
    }
}

__global__ void ExpandLayerBeam4Kernel(int m,
                                       int Hs,
                                       int G0S_rows,
                                       int layer,
                                       int qlen,
                                       const float* __restrict__ Gs,
                                       const float* __restrict__ G0S,
                                       const int* __restrict__ small_offsets,
                                       const int* __restrict__ h_vec,
                                       const float* __restrict__ xC_small,
                                       const int* __restrict__ root_curr,
                                       const Code* __restrict__ codes_curr,
                                       const float* __restrict__ err_curr,
                                       const float* __restrict__ a_curr,
                                       const float* __restrict__ R_curr,
                                       int* __restrict__ root_next,
                                       Code* __restrict__ codes_next,
                                       float* __restrict__ err_next,
                                       float* __restrict__ a_next,
                                       float* __restrict__ R_next) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= qlen) return;
    const int beams = 4;
    const int codes_rows = (m > 1) ? (m - 1) : 0;
    const int K = h_vec[layer];
    const int off = small_offsets[layer];
    const float inf = INFINITY;

    float best_e[4] = {inf, inf, inf, inf};
    int sel_src[4] = {-1, -1, -1, -1};
    int sel_k[4] = {0, 0, 0, 0};
    int sel_root[4] = {0, 0, 0, 0};
    float sel_denom[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sel_alast[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sel_v[4][kMaxM];
    float sel_y[4][kMaxM];
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        #pragma unroll
        for (int t = 0; t < kMaxM; ++t) {
            sel_v[b][t] = 0.0f;
            sel_y[b][t] = 0.0f;
        }
    }

    int worst_idx = 0;
    float worst_err = best_e[0];

    for (int h = 0; h < beams; ++h) {
        const int idx = j * beams + h;
        const float E_old = err_curr[idx];
        if (!(E_old < inf)) continue;
        const int root_code = root_curr[idx];

        const float* __restrict a_old = a_curr + static_cast<std::size_t>(idx) * static_cast<std::size_t>(m);
        const float* __restrict R_pref =
            R_curr + static_cast<std::size_t>(idx) * static_cast<std::size_t>(m) * static_cast<std::size_t>(m);
        const Code* __restrict c_old =
            codes_curr + static_cast<std::size_t>(idx) * static_cast<std::size_t>(codes_rows);

        for (int k = 0; k < K; ++k) {
            const int flat = off + k;
            const float alpha = Gs[static_cast<std::size_t>(flat) +
                                   static_cast<std::size_t>(flat) * static_cast<std::size_t>(Hs)];
            const float gamma =
                xC_small[static_cast<std::size_t>(flat) + static_cast<std::size_t>(j) * static_cast<std::size_t>(Hs)];

            float A[kMaxM];
            float v[kMaxM];
            float y[kMaxM];
            #pragma unroll
            for (int t = 0; t < kMaxM; ++t) {
                A[t] = 0.0f;
                v[t] = 0.0f;
                y[t] = 0.0f;
            }

            A[0] = G0S[static_cast<std::size_t>(root_code) +
                       static_cast<std::size_t>(flat) * static_cast<std::size_t>(G0S_rows)];
            for (int t = 1; t < layer; ++t) {
                const int flat_t = small_offsets[t] + static_cast<int>(c_old[t - 1]);
                A[t] = Gs[static_cast<std::size_t>(flat_t) +
                          static_cast<std::size_t>(flat) * static_cast<std::size_t>(Hs)];
            }

            SolveCholUpperDispatch(R_pref, layer, A, v, y, m);

            float delta = 0.0f;
            float eta = 0.0f;
            for (int t = 0; t < layer; ++t) {
                const float at = A[t];
                delta += at * v[t];
                eta += at * a_old[t];
            }

            const float denom = alpha - delta;
            if (denom == 0.0f) continue;
            const float diff = gamma - eta;
            const float a_last = diff / denom;
            float e = E_old - diff * a_last;
            if (e < 0.0f) e = 0.0f;
            if (e >= worst_err) continue;

            const int out = worst_idx;
            best_e[out] = e;
            sel_src[out] = h;
            sel_k[out] = k;
            sel_root[out] = root_code;
            sel_denom[out] = denom;
            sel_alast[out] = a_last;
            for (int t = 0; t < layer; ++t) {
                sel_v[out][t] = v[t];
                sel_y[out][t] = y[t];
            }

            worst_idx = WorstIdx4(best_e);
            worst_err = best_e[worst_idx];
        }
    }

    const int n_finite = (best_e[0] < inf) + (best_e[1] < inf) + (best_e[2] < inf) + (best_e[3] < inf);
    if (n_finite == 0) {
        int curr_H = 0;
        for (int h = 0; h < 4; ++h) {
            curr_H += (err_curr[j * beams + h] < inf);
        }
        if (curr_H == 0) curr_H = 1;
        const int newH = (curr_H < 4) ? curr_H : 4;
        for (int out = 0; out < beams; ++out) {
            const int dst = j * beams + out;
            if (out < newH) {
                const int src = j * beams + out;
                err_next[dst] = err_curr[src];
                root_next[dst] = root_curr[src];
                const Code* cs = codes_curr + static_cast<std::size_t>(src) * static_cast<std::size_t>(codes_rows);
                Code* cd = codes_next + static_cast<std::size_t>(dst) * static_cast<std::size_t>(codes_rows);
                for (int t = 0; t < layer; ++t) cd[t] = cs[t];
                const float* as = a_curr + static_cast<std::size_t>(src) * static_cast<std::size_t>(m);
                float* ad = a_next + static_cast<std::size_t>(dst) * static_cast<std::size_t>(m);
                for (int t = 0; t <= layer; ++t) ad[t] = as[t];
                const float* Rs = R_curr + static_cast<std::size_t>(src) * static_cast<std::size_t>(m) *
                                      static_cast<std::size_t>(m);
                float* Rd = R_next + static_cast<std::size_t>(dst) * static_cast<std::size_t>(m) *
                                static_cast<std::size_t>(m);
                for (int t = 0; t < m * m; ++t) Rd[t] = Rs[t];
            } else {
                err_next[dst] = inf;
            }
        }
        return;
    }

    for (int out = 0; out < beams; ++out) {
        const int dst = j * beams + out;
        if (!(best_e[out] < inf)) {
            err_next[dst] = inf;
            continue;
        }
        const int src_h = sel_src[out];
        const int src = j * beams + src_h;
        err_next[dst] = best_e[out];
        root_next[dst] = sel_root[out];

        const Code* cs = codes_curr + static_cast<std::size_t>(src) * static_cast<std::size_t>(codes_rows);
        Code* cd = codes_next + static_cast<std::size_t>(dst) * static_cast<std::size_t>(codes_rows);
        for (int t = 1; t < layer; ++t) {
            cd[t - 1] = cs[t - 1];
        }
        cd[layer - 1] = static_cast<Code>(sel_k[out]);

        const float* as = a_curr + static_cast<std::size_t>(src) * static_cast<std::size_t>(m);
        float* ad = a_next + static_cast<std::size_t>(dst) * static_cast<std::size_t>(m);
        const float a_last = sel_alast[out];
        for (int t = 0; t < layer; ++t) {
            ad[t] = as[t] - sel_v[out][t] * a_last;
        }
        ad[layer] = a_last;

        const float* Rs = R_curr + static_cast<std::size_t>(src) * static_cast<std::size_t>(m) *
                              static_cast<std::size_t>(m);
        float* Rd = R_next + static_cast<std::size_t>(dst) * static_cast<std::size_t>(m) *
                        static_cast<std::size_t>(m);
        for (int r = 0; r < layer; ++r) {
            const float* src_row = Rs + static_cast<std::size_t>(r) * static_cast<std::size_t>(m);
            float* dst_row = Rd + static_cast<std::size_t>(r) * static_cast<std::size_t>(m);
            for (int c = 0; c < m; ++c) dst_row[c] = src_row[c];
        }
        for (int t = 0; t < layer; ++t) {
            Rd[static_cast<std::size_t>(t) * static_cast<std::size_t>(m) + static_cast<std::size_t>(layer)] =
                sel_y[out][t];
        }
        Rd[static_cast<std::size_t>(layer) * static_cast<std::size_t>(m) + static_cast<std::size_t>(layer)] =
            sqrtf(sel_denom[out]);
    }
}

__global__ void FinalizeBestBeam4Kernel(int m,
                                        int qlen,
                                        const int* __restrict__ root_curr,
                                        const Code* __restrict__ codes_curr,
                                        const float* __restrict__ err_curr,
                                        std::uint32_t* __restrict__ cluster_id,
                                        Code* __restrict__ B_small) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= qlen) return;
    const int beams = 4;
    const int codes_rows = (m > 1) ? (m - 1) : 0;
    int best = 0;
    float be = err_curr[j * beams + 0];
    for (int b = 1; b < beams; ++b) {
        const float e = err_curr[j * beams + b];
        if (e < be) {
            be = e;
            best = b;
        }
    }
    const int idx = j * beams + best;
    cluster_id[j] = static_cast<std::uint32_t>(root_curr[idx]);
    const Code* cs = codes_curr + static_cast<std::size_t>(idx) * static_cast<std::size_t>(codes_rows);
    Code* out = B_small + static_cast<std::size_t>(j) * static_cast<std::size_t>(codes_rows);
    for (int t = 0; t < codes_rows; ++t) out[t] = cs[t];
}

}  // namespace
