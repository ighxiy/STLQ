
constexpr float kEps = 1e-6f;

template <int M>
float SampleCostFromXcFixed(const Precomp& precomp,
                            const std::vector<float>& xC,
                            const std::vector<FullCode>& B,
                            const std::vector<float>& a,
                            float x_norm2) {
    static_assert(M > 0, "M must be positive");
    float term1 = 0.0f;
    float term2 = 0.0f;
    for (int l = 0; l < M; ++l) {
        const int flat_l = precomp.offsets[l] + static_cast<int>(B[static_cast<std::size_t>(l)]);
        const float alpha_l = a[static_cast<std::size_t>(l)];
        term1 += alpha_l * xC[static_cast<std::size_t>(flat_l)];
        for (int k = 0; k < M; ++k) {
            const int flat_k = precomp.offsets[k] + static_cast<int>(B[static_cast<std::size_t>(k)]);
            term2 += alpha_l * a[static_cast<std::size_t>(k)] * GAt(precomp.G, flat_l, flat_k);
        }
    }
    return x_norm2 - 2.0f * term1 + term2;
}

template <int M>
bool SolveLeastSquaresVectorFixed(const Precomp& precomp,
                                 const std::vector<float>& xC,
                                 const std::vector<FullCode>& B,
                                 std::vector<float>* A,
                                 std::vector<float>* b,
                                 std::vector<float>* work,
                                 float* a_out) {
    static_assert(M > 0, "M must be positive");
    if (!A || !b || !work ||
        static_cast<int>(A->size()) != M * M ||
        static_cast<int>(b->size()) != M ||
        static_cast<int>(work->size()) != M) {
        return false;
    }

    const auto fill_system = [&](float lambda) {
        std::fill(A->begin(), A->end(), 0.0f);
        for (int j = 0; j < M; ++j) {
            const int flat_j = precomp.offsets[j] + static_cast<int>(B[static_cast<std::size_t>(j)]);
            (*b)[static_cast<std::size_t>(j)] = xC[static_cast<std::size_t>(flat_j)];
            for (int k = 0; k <= j; ++k) {
                const int flat_k = precomp.offsets[k] + static_cast<int>(B[static_cast<std::size_t>(k)]);
                const float val = GAt(precomp.G, flat_k, flat_j);
                (*A)[static_cast<std::size_t>(k) + static_cast<std::size_t>(j) * static_cast<std::size_t>(M)] = val;
            }
            (*A)[static_cast<std::size_t>(j) + static_cast<std::size_t>(j) * static_cast<std::size_t>(M)] += lambda;
        }
    };

    fill_system(0.0f);
    int info = lapack::SpotrfU(M, A->data(), M);
    if (info != 0) {
        float bump = kEps;
        for (int attempt = 0; attempt < 3 && info != 0; ++attempt) {
            fill_system(bump);
            info = lapack::SpotrfU(M, A->data(), M);
            bump *= 10.0f;
        }
    }
    if (info != 0) {
        return false;
    }
    info = lapack::SpotrsU(M, 1, A->data(), M, b->data(), M);
    if (info != 0) {
        return false;
    }
    std::copy_n(b->data(), M, a_out);
    return true;
}

template <int M>
bool SolveLeastSquaresVectorFromResidualFixed(const Precomp& precomp,
                                             const std::vector<float>& re,
                                             const std::vector<FullCode>& B,
                                             std::vector<float>* A,
                                             std::vector<float>* b,
                                             std::vector<float>* work,
                                             float* a_out) {
    static_assert(M > 0, "M must be positive");
    const int d = precomp.d;
    if (!A || !b || !work ||
        static_cast<int>(A->size()) != M * M ||
        static_cast<int>(b->size()) != M ||
        static_cast<int>(work->size()) != M) {
        return false;
    }

    const float* re_ptr = re.data();
    const auto fill_system = [&](float lambda) {
        std::fill(A->begin(), A->end(), 0.0f);
        for (int j = 0; j < M; ++j) {
            const int flat_j = precomp.offsets[j] + static_cast<int>(B[static_cast<std::size_t>(j)]);
            const float* cj = precomp.C_all.Col(flat_j);
            float dot = 0.0f;
            #pragma omp simd reduction(+:dot)
            for (int r = 0; r < d; ++r) {
                dot += cj[r] * re_ptr[r];
            }
            (*b)[static_cast<std::size_t>(j)] = dot;
            for (int k = 0; k <= j; ++k) {
                const int flat_k = precomp.offsets[k] + static_cast<int>(B[static_cast<std::size_t>(k)]);
                const float val = GAt(precomp.G, flat_k, flat_j);
                (*A)[static_cast<std::size_t>(k) + static_cast<std::size_t>(j) * static_cast<std::size_t>(M)] = val;
            }
            (*A)[static_cast<std::size_t>(j) + static_cast<std::size_t>(j) * static_cast<std::size_t>(M)] += lambda;
        }
    };

    fill_system(0.0f);
    int info = lapack::SpotrfU(M, A->data(), M);
    if (info != 0) {
        float bump = kEps;
        for (int attempt = 0; attempt < 3 && info != 0; ++attempt) {
            fill_system(bump);
            info = lapack::SpotrfU(M, A->data(), M);
            bump *= 10.0f;
        }
    }
    if (info != 0) {
        return false;
    }
    info = lapack::SpotrsU(M, 1, A->data(), M, b->data(), M);
    if (info != 0) {
        return false;
    }
    std::copy_n(b->data(), M, a_out);
    return true;
}

class ScopedBlasThreads {
public:
    explicit ScopedBlasThreads(int nthreads) : active_(false) {
        if (nthreads > 1) {
            BlasSetThreads(nthreads);
            active_ = true;
        }
    }
    ScopedBlasThreads(const ScopedBlasThreads&) = delete;
    ScopedBlasThreads& operator=(const ScopedBlasThreads&) = delete;
    ~ScopedBlasThreads() {
        if (active_) {
            BlasSetThreads(1);
        }
    }

private:
    bool active_;
};

[[maybe_unused]] std::string FormatFloatLocal(double value, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

[[maybe_unused]] void PrintClusterProgress(int done, int total) {
    std::fprintf(stderr, "\rprocessed %d/%d   ", done, total);
    if (done >= total) {
        std::fprintf(stderr, "\n");
    }
    std::fflush(stderr);
}

struct LinkageTls {
    std::vector<float> A;
    std::vector<float> b;
    std::vector<float> work;
    std::vector<float> R_self;
    std::vector<float> re;
    std::vector<FullCode> B_init;
    std::vector<float> a_init;
    std::vector<float> a_bak;
    std::vector<FullCode> B_best;
    std::vector<float> a_best;
    std::vector<float> xC_base_buf;
    std::vector<float> xC_buf;
    std::vector<float> rC_buf;
    std::vector<int> order_idx;
    std::vector<uint8_t> processed;
    std::vector<int> local_pos;
    std::vector<int> candidates;
    std::vector<int> candidates_inner;
    std::vector<int> candidates_same;
    std::vector<uint8_t> available;
    std::vector<FullCode> B_try;
    std::vector<float> a_try;
    std::vector<int> pert_layers;
    std::vector<int> new_flat;
};

LinkageTls MakeLinkageTls(int d, int m, int H) {
    LinkageTls tls;
    tls.A.resize(static_cast<std::size_t>(m) * m, 0.0f);
    tls.b.resize(static_cast<std::size_t>(m), 0.0f);
    tls.work.resize(static_cast<std::size_t>(m), 0.0f);
    tls.R_self.resize(static_cast<std::size_t>(d), 0.0f);
    tls.re.resize(static_cast<std::size_t>(d), 0.0f);
    tls.B_init.resize(static_cast<std::size_t>(m), 0);
    tls.a_init.resize(static_cast<std::size_t>(m), 0.0f);
    tls.a_bak.resize(static_cast<std::size_t>(m), 0.0f);
    tls.B_best.resize(static_cast<std::size_t>(m), 0);
    tls.a_best.resize(static_cast<std::size_t>(m), 0.0f);
    tls.xC_base_buf.resize(static_cast<std::size_t>(H), 0.0f);
    tls.xC_buf.resize(static_cast<std::size_t>(H), 0.0f);
    tls.rC_buf.resize(static_cast<std::size_t>(H), 0.0f);
    tls.available.resize(static_cast<std::size_t>(m), 0);
    tls.B_try.resize(static_cast<std::size_t>(m), 0);
    tls.a_try.resize(static_cast<std::size_t>(m), 0.0f);
    tls.pert_layers.resize(static_cast<std::size_t>(m), 0);
    tls.new_flat.resize(static_cast<std::size_t>(m), 0);
    return tls;
}

inline float& ACol(std::vector<float>& A, int m, int r, int c) {
    return A[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * m];
}

void ComputeXcBase(const Precomp& precomp, const float* x, std::vector<float>* xC) {
    // Julia reference: xC_base = C_all' * x (linkage_quantizer.jl:239-246).
    const int d = precomp.d;
    const int H = precomp.H;
    if (static_cast<int>(xC->size()) != H) {
        xC->resize(static_cast<std::size_t>(H));
    }
    for (int j = 0; j < H; ++j) {
        const float* c = precomp.C_all.Col(j);
        float s = 0.0f;
        #pragma omp simd reduction(+:s)
        for (int r = 0; r < d; ++r) {
            s += c[r] * x[r];
        }
        (*xC)[j] = s;
    }
}

void ComputeXcFromResidual(const Precomp& precomp, const std::vector<float>& re, std::vector<float>* xC) {
    // Julia reference: rC = C_all' * re in quantize_with_kmeans_abs_nonormal_single_extend_allcodes_fast!
    // (greedy_quantizer.jl:4-21).
    const int d = precomp.d;
    const int H = precomp.H;
    if (static_cast<int>(xC->size()) != H) {
        xC->resize(static_cast<std::size_t>(H));
    }
    for (int j = 0; j < H; ++j) {
        const float* c = precomp.C_all.Col(j);
        float s = 0.0f;
        const float* re_ptr = re.data();
        #pragma omp simd reduction(+:s)
        for (int r = 0; r < d; ++r) {
            s += c[r] * re_ptr[r];
        }
        (*xC)[j] = s;
    }
}

ColMajorMatrix<float> BuildOneRootCrossGram(const Precomp& pre_one, const Precomp& pre_root) {
    // Julia reference: build_one_root_cross_gram (linkage_encode_helper.jl:578-592).
    ColMajorMatrix<float> G_one_root(pre_one.H, pre_root.H);
    Gemm(true, false, 1.0f, pre_one.C_all, pre_root.C_all, 0.0f, &G_one_root);
    return G_one_root;
}

void BuildXcForParentSingleLocal(const Precomp& precomp,
                                 const std::vector<float>& xC_base,
                                 const ColMajorMatrix<FullCode>& B,
                                 const ColMajorMatrix<float>& a,
                                 const std::vector<int>& cluster_indices,
                                 const std::vector<int>& parent_local,
                                 int p_local,
                                 std::vector<float>* xC_res) {
    // Same math as BuildXcForParentSingle, but traverses the cluster-local parent linkage.
    const int H = precomp.H;
    const int m = precomp.m;
    if (static_cast<int>(xC_res->size()) != H) {
        xC_res->assign(static_cast<std::size_t>(H), 0.0f);
    }
    std::memcpy(xC_res->data(), xC_base.data(), sizeof(float) * static_cast<std::size_t>(H));

    int local = p_local;
    while (local >= 0) {
        const int g = cluster_indices[static_cast<std::size_t>(local)];
        for (int l = 0; l < m; ++l) {
            const float alpha = a(l, g);
            if (alpha == 0.0f) {
                continue;
            }
            const int code = static_cast<int>(B(l, g));
            const int flat = precomp.offsets[l] + code;
            const float* Gcol = precomp.G.Col(flat);
            float* dst = xC_res->data();
            #pragma omp simd
            for (int q = 0; q < H; ++q) {
                dst[q] -= alpha * Gcol[q];
            }
        }
        const int p1 = parent_local[static_cast<std::size_t>(local)];
        if (p1 == 0) {
            break;
        }
        local = p1 - 1;
    }
}

[[maybe_unused]] void BuildXcOneForParentSingle(const Precomp& pre_root,
                               const Precomp& pre_one,
                               const ColMajorMatrix<float>& G_one_root,
                               const std::vector<float>& xC_one_base,
                               const ColMajorMatrix<FullCode>& B,
                               const ColMajorMatrix<float>& a,
                               const std::vector<int>& parent,
                               const std::vector<int>& depths,
                               int p_idx,
                               std::vector<float>* xC_res,
                               int forced_root_code = -1) {
    // Julia reference: build_xC_one_for_parent! (linkage_encode_helper.jl:610-653).
    const int H_one = pre_one.H;
    const int m = pre_one.m;
    if (static_cast<int>(xC_res->size()) != H_one) {
        xC_res->assign(static_cast<std::size_t>(H_one), 0.0f);
    }
    std::memcpy(xC_res->data(), xC_one_base.data(), sizeof(float) * static_cast<std::size_t>(H_one));

    int idx = p_idx;
    int depth = depths[static_cast<std::size_t>(p_idx)];
    while (idx >= 0) {
        for (int l = 0; l < m; ++l) {
            float alpha = a(l, idx);
            if (alpha == 0.0f) {
                continue;
            }
            int code = static_cast<int>(B(l, idx));
            // When h0_root > 256, FullCode (u8) cannot represent the root code;
            // B(0,idx) is a dummy 0.  Use forced_root_code for layer-0 at depth=0.
            if (depth == 0 && l == 0 && forced_root_code >= 0) {
                code = forced_root_code;
            }
            if (depth == 0) {
                int flat_root = pre_root.offsets[l] + code;
                const float* Gcol = G_one_root.Col(flat_root);
                float* dst = xC_res->data();
                #pragma omp simd
                for (int q = 0; q < H_one; ++q) {
                    dst[q] -= alpha * Gcol[q];
                }
            } else {
                int flat_one = pre_one.offsets[l] + code;
                const float* Gcol = pre_one.G.Col(flat_one);
                float* dst = xC_res->data();
                #pragma omp simd
                for (int q = 0; q < H_one; ++q) {
                    dst[q] -= alpha * Gcol[q];
                }
            }
        }
        int p = parent[static_cast<std::size_t>(idx)];
        if (p == 0) {
            break;
        }
        idx = p - 1;
        --depth;
    }
}

void BuildXcOneForParentSingleCols(const Precomp& pre_root,
                                   const Precomp& pre_one,
                                   const ColMajorMatrix<float>& G_one_root,
                                   const std::vector<float>& xC_one_base,
                                   FullCode* const* B_cols,
                                   float* const* a_cols,
                                   const std::vector<int>& parent,
                                   const std::vector<int>& depths,
                                   int p_idx,
                                   std::vector<float>* xC_res,
                                   int forced_root_code = -1) {
    // Same math as BuildXcOneForParentSingle, but reads B/a from per-column pointers.
    const int H_one = pre_one.H;
    const int m = pre_one.m;
    if (static_cast<int>(xC_res->size()) != H_one) {
        xC_res->assign(static_cast<std::size_t>(H_one), 0.0f);
    }
    std::memcpy(xC_res->data(), xC_one_base.data(), sizeof(float) * static_cast<std::size_t>(H_one));

    int idx = p_idx;
    int depth = depths[static_cast<std::size_t>(p_idx)];
    while (idx >= 0) {
        const FullCode* bcol = B_cols[idx];
        const float* acol = a_cols[idx];
        for (int l = 0; l < m; ++l) {
            float alpha = acol[static_cast<std::size_t>(l)];
            if (alpha == 0.0f) {
                continue;
            }
            int code = static_cast<int>(bcol[static_cast<std::size_t>(l)]);
            // When h0_root > 256, FullCode (u8) cannot represent the root code;
            // bcol[0] is a dummy 0.  Use forced_root_code for layer-0 at depth=0.
            if (depth == 0 && l == 0 && forced_root_code >= 0) {
                code = forced_root_code;
            }
            if (depth == 0) {
                int flat_root = pre_root.offsets[l] + code;
                const float* Gcol = G_one_root.Col(flat_root);
                float* dst = xC_res->data();
                #pragma omp simd
                for (int q = 0; q < H_one; ++q) {
                    dst[q] -= alpha * Gcol[q];
                }
            } else {
                int flat_one = pre_one.offsets[l] + code;
                const float* Gcol = pre_one.G.Col(flat_one);
                float* dst = xC_res->data();
                #pragma omp simd
                for (int q = 0; q < H_one; ++q) {
                    dst[q] -= alpha * Gcol[q];
                }
            }
        }
        int p = parent[static_cast<std::size_t>(idx)];
        if (p == 0) {
            break;
        }
        idx = p - 1;
        --depth;
    }
}

template <typename GetBCol, typename GetACol>
void BuildXcOneForParentSingleGetCols(const Precomp& pre_root,
                                      const Precomp& pre_one,
                                      const ColMajorMatrix<float>& G_one_root,
                                      const std::vector<float>& xC_one_base,
                                      GetBCol get_bcol,
                                      GetACol get_acol,
                                      const std::vector<int>& parent,
                                      const std::vector<int>& depths,
                                      int p_idx,
                                      std::vector<float>* xC_res,
                                      int forced_root_code = -1) {
    // Same math as BuildXcOneForParentSingleCols, but retrieves B/a via callables.
    const int H_one = pre_one.H;
    const int m = pre_one.m;
    if (static_cast<int>(xC_res->size()) != H_one) {
        xC_res->assign(static_cast<std::size_t>(H_one), 0.0f);
    }
    std::memcpy(xC_res->data(), xC_one_base.data(), sizeof(float) * static_cast<std::size_t>(H_one));

    int idx = p_idx;
    int depth = depths[static_cast<std::size_t>(p_idx)];
    while (idx >= 0) {
        const FullCode* bcol = get_bcol(idx);
        const float* acol = get_acol(idx);
        for (int l = 0; l < m; ++l) {
            float alpha = acol[static_cast<std::size_t>(l)];
            if (alpha == 0.0f) {
                continue;
            }
            int code = static_cast<int>(bcol[static_cast<std::size_t>(l)]);
            // When h0_root > 256, FullCode (u8) cannot represent the root code;
            // bcol[0] is a dummy 0.  Use forced_root_code for layer-0 at depth=0.
            if (depth == 0 && l == 0 && forced_root_code >= 0) {
                code = forced_root_code;
            }
            if (depth == 0) {
                int flat_root = pre_root.offsets[l] + code;
                const float* Gcol = G_one_root.Col(flat_root);
                float* dst = xC_res->data();
                #pragma omp simd
                for (int q = 0; q < H_one; ++q) {
                    dst[q] -= alpha * Gcol[q];
                }
            } else {
                int flat_one = pre_one.offsets[l] + code;
                const float* Gcol = pre_one.G.Col(flat_one);
                float* dst = xC_res->data();
                #pragma omp simd
                for (int q = 0; q < H_one; ++q) {
                    dst[q] -= alpha * Gcol[q];
                }
            }
        }
        int p = parent[static_cast<std::size_t>(idx)];
        if (p == 0) {
            break;
        }
        idx = p - 1;
        --depth;
    }
}

float SampleCostFromXc(const Precomp& precomp,
                       const std::vector<float>& xC,
                       const std::vector<FullCode>& B,
                       const std::vector<float>& a,
                       float x_norm2) {
    // Julia reference: sample_cost_from_xC! inside dynamic_icm_encoding_single_for_residual!
    // (linkage_encode_helper.jl:760-789).
    const int m = precomp.m;
    if (m == 5) return SampleCostFromXcFixed<5>(precomp, xC, B, a, x_norm2);
    if (m == 10) return SampleCostFromXcFixed<10>(precomp, xC, B, a, x_norm2);
    float term1 = 0.0f;
    float term2 = 0.0f;
    for (int l = 0; l < m; ++l) {
        int flat_l = precomp.offsets[l] + static_cast<int>(B[static_cast<std::size_t>(l)]);
        float alpha_l = a[static_cast<std::size_t>(l)];
        term1 += alpha_l * xC[static_cast<std::size_t>(flat_l)];
        for (int k = 0; k < m; ++k) {
            int flat_k = precomp.offsets[k] + static_cast<int>(B[static_cast<std::size_t>(k)]);
            term2 += alpha_l * a[static_cast<std::size_t>(k)] * GAt(precomp.G, flat_l, flat_k);
        }
    }
    return x_norm2 - 2.0f * term1 + term2;
}

bool SolveLeastSquaresVector(const Precomp& precomp,
                             const std::vector<float>& xC,
                             const std::vector<FullCode>& B,
                             std::vector<float>* A,
                             std::vector<float>* b,
                             std::vector<float>* work,
                             float* a_out) {
    // Julia reference: optimized_least_squares_single! (encode_abs.jl:305-340).
    const int m = precomp.m;
    if (m == 5) return SolveLeastSquaresVectorFixed<5>(precomp, xC, B, A, b, work, a_out);
    if (m == 10) return SolveLeastSquaresVectorFixed<10>(precomp, xC, B, A, b, work, a_out);
    if (!A || !b || !work ||
        static_cast<int>(A->size()) != m * m ||
        static_cast<int>(b->size()) != m ||
        static_cast<int>(work->size()) != m) {
        return false;
    }

    const auto fill_system = [&](float lambda) {
        std::fill(A->begin(), A->end(), 0.0f);
        for (int j = 0; j < m; ++j) {
            int flat_j = precomp.offsets[j] + static_cast<int>(B[static_cast<std::size_t>(j)]);
            (*b)[j] = xC[static_cast<std::size_t>(flat_j)];
            for (int k = 0; k <= j; ++k) {
                int flat_k = precomp.offsets[k] + static_cast<int>(B[static_cast<std::size_t>(k)]);
                float val = GAt(precomp.G, flat_k, flat_j);
                ACol(*A, m, k, j) = val;
            }
            ACol(*A, m, j, j) += lambda;
        }
    };

    fill_system(0.0f);
    int info = lapack::SpotrfU(m, A->data(), m);
    if (info != 0) {
        float bump = kEps;
        for (int attempt = 0; attempt < 3 && info != 0; ++attempt) {
            fill_system(bump);
            info = lapack::SpotrfU(m, A->data(), m);
            bump *= 10.0f;
        }
    }
    if (info != 0) {
        return false;
    }
    info = lapack::SpotrsU(m, 1, A->data(), m, b->data(), m);
    if (info != 0) {
        return false;
    }
    std::copy(b->begin(), b->end(), a_out);
    return true;
}

bool SolveLeastSquaresVectorFromResidual(const Precomp& precomp,
                                         const std::vector<float>& re,
                                         const std::vector<FullCode>& B,
                                         std::vector<float>* A,
                                         std::vector<float>* b,
                                         std::vector<float>* work,
                                         float* a_out) {
    // Julia reference: optimized_least_squares_single! after greedy quantize in linkage loop
    // (linkage_quantizer.jl:241-244, encode_abs.jl:249+).
    const int m = precomp.m;
    if (m == 5) return SolveLeastSquaresVectorFromResidualFixed<5>(precomp, re, B, A, b, work, a_out);
    if (m == 10) return SolveLeastSquaresVectorFromResidualFixed<10>(precomp, re, B, A, b, work, a_out);
    const int d = precomp.d;
    if (!A || !b || !work ||
        static_cast<int>(A->size()) != m * m ||
        static_cast<int>(b->size()) != m ||
        static_cast<int>(work->size()) != m) {
        return false;
    }

    const float* re_ptr = re.data();

    const auto fill_system = [&](float lambda) {
        std::fill(A->begin(), A->end(), 0.0f);
        for (int j = 0; j < m; ++j) {
            int flat_j = precomp.offsets[j] + static_cast<int>(B[static_cast<std::size_t>(j)]);
            const float* cj = precomp.C_all.Col(flat_j);
            float dot = 0.0f;
            #pragma omp simd reduction(+:dot)
            for (int r = 0; r < d; ++r) {
                dot += cj[r] * re_ptr[r];
            }
            (*b)[j] = dot;
            for (int k = 0; k <= j; ++k) {
                int flat_k = precomp.offsets[k] + static_cast<int>(B[static_cast<std::size_t>(k)]);
                float val = GAt(precomp.G, flat_k, flat_j);
                ACol(*A, m, k, j) = val;
            }
            ACol(*A, m, j, j) += lambda;
        }
    };

    fill_system(0.0f);
    int info = lapack::SpotrfU(m, A->data(), m);
    if (info != 0) {
        float bump = kEps;
        for (int attempt = 0; attempt < 3 && info != 0; ++attempt) {
            fill_system(bump);
            info = lapack::SpotrfU(m, A->data(), m);
            bump *= 10.0f;
        }
    }
    if (info != 0) {
        return false;
    }
    info = lapack::SpotrsU(m, 1, A->data(), m, b->data(), m);
    if (info != 0) {
        return false;
    }
    std::copy(b->begin(), b->end(), a_out);
    return true;
}

void DynamicIcmEncodingSingleForResidual(const Precomp& precomp,
                                         const std::vector<float>& re,
                                         const std::vector<float>& xC_res,
                                         int max_iters,
                                         std::vector<FullCode>* B,
                                         std::vector<float>* a,
                                         std::vector<float>* a_bak,
                                         std::vector<float>* rC,
                                         std::vector<float>* A,
                                         std::vector<float>* b,
                                         std::vector<float>* work,
                                         std::mt19937* rng,
                                         int* ls_failures) {
    // Julia reference: dynamic_icm_encoding_single_for_residual!
    // (linkage_encode_helper.jl:726-840).
    if (max_iters <= 0) {
        return;
    }

    const int m = precomp.m;
    const int H = precomp.H;
    const int d = precomp.d;

    float x_norm2 = 0.0f;
    const float* re_ptr = re.data();
    #pragma omp simd reduction(+:x_norm2)
    for (int i = 0; i < d; ++i) {
        float v = re_ptr[i];
        x_norm2 += v * v;
    }

    if (static_cast<int>(rC->size()) != H) {
        rC->assign(static_cast<std::size_t>(H), 0.0f);
    }
    std::memcpy(rC->data(), xC_res.data(), sizeof(float) * static_cast<std::size_t>(H));
    for (int l = 0; l < m; ++l) {
        float alpha = (*a)[static_cast<std::size_t>(l)];
        if (alpha == 0.0f) {
            continue;
        }
        int flat = precomp.offsets[l] + static_cast<int>((*B)[static_cast<std::size_t>(l)]);
        const float* Gcol = precomp.G.Col(flat);
        float* dst = rC->data();
        #pragma omp simd
        for (int q = 0; q < H; ++q) {
            dst[q] -= alpha * Gcol[q];
        }
    }

    float cur_cost = SampleCostFromXc(precomp, xC_res, *B, *a, x_norm2);

    int order_stack[32];
    std::vector<int> order_heap;
    int* order = nullptr;
    if (m <= 32) {
        order = order_stack;
    } else {
        order_heap.resize(static_cast<std::size_t>(m), 0);
        order = order_heap.data();
    }
    for (int iter = 0; iter < max_iters; ++iter) {
        for (int j = 0; j < m; ++j) {
            order[j] = j;
        }
        std::shuffle(order, order + m, *rng);
        bool changed_any = false;

        for (int ord = 0; ord < m; ++ord) {
            const int jlayer = order[ord];
            int old_code = static_cast<int>((*B)[static_cast<std::size_t>(jlayer)]);
            int old_flat = precomp.offsets[jlayer] + old_code;

            int best_code = old_code;
            float best_abs = -std::numeric_limits<float>::infinity();
            int startf = precomp.offsets[jlayer];
            int stopf = startf + precomp.h_vec[jlayer];

            for (int flat = startf; flat < stopf; ++flat) {
                float val = (*rC)[static_cast<std::size_t>(flat)] +
                            (*a)[static_cast<std::size_t>(jlayer)] * GAt(precomp.G, flat, old_flat);
                val *= precomp.invnorm_flat[static_cast<std::size_t>(flat)];
                float aval = std::abs(val);
                if (aval > best_abs) {
                    best_abs = aval;
                    best_code = flat - startf;
                }
            }

            if (best_code == old_code) {
                continue;
            }

            for (int l = 0; l < m; ++l) {
                (*a_bak)[static_cast<std::size_t>(l)] = (*a)[static_cast<std::size_t>(l)];
            }

            (*B)[static_cast<std::size_t>(jlayer)] = static_cast<FullCode>(best_code);

            if (!SolveLeastSquaresVector(precomp, xC_res, *B, A, b, work, a->data())) {
                (*B)[static_cast<std::size_t>(jlayer)] = static_cast<FullCode>(old_code);
                for (int l = 0; l < m; ++l) {
                    (*a)[static_cast<std::size_t>(l)] = (*a_bak)[static_cast<std::size_t>(l)];
                }
                if (ls_failures) {
                    ++(*ls_failures);
                }
                continue;
            }

            std::memcpy(rC->data(), xC_res.data(), sizeof(float) * static_cast<std::size_t>(H));
            for (int l = 0; l < m; ++l) {
                float alpha = (*a)[static_cast<std::size_t>(l)];
                if (alpha == 0.0f) {
                    continue;
                }
                int flat = precomp.offsets[l] + static_cast<int>((*B)[static_cast<std::size_t>(l)]);
                const float* Gcol = precomp.G.Col(flat);
                float* dst = rC->data();
                #pragma omp simd
                for (int q = 0; q < H; ++q) {
                    dst[q] -= alpha * Gcol[q];
                }
            }

            float new_cost = SampleCostFromXc(precomp, xC_res, *B, *a, x_norm2);
            if (new_cost + kEps < cur_cost) {
                cur_cost = new_cost;
                changed_any = true;
            } else {
                (*B)[static_cast<std::size_t>(jlayer)] = static_cast<FullCode>(old_code);
                for (int l = 0; l < m; ++l) {
                    (*a)[static_cast<std::size_t>(l)] = (*a_bak)[static_cast<std::size_t>(l)];
                }
                std::memcpy(rC->data(), xC_res.data(), sizeof(float) * static_cast<std::size_t>(H));
                for (int l = 0; l < m; ++l) {
                    float alpha = (*a)[static_cast<std::size_t>(l)];
                    if (alpha == 0.0f) {
                        continue;
                    }
                    int flat = precomp.offsets[l] + static_cast<int>((*B)[static_cast<std::size_t>(l)]);
                    const float* Gcol = precomp.G.Col(flat);
                    float* dst = rC->data();
                    #pragma omp simd
                    for (int q = 0; q < H; ++q) {
                        dst[q] -= alpha * Gcol[q];
                    }
                }
            }
        }

        if (!changed_any) {
            break;
        }
    }
}

void QuantizeWithKmeansAbsNoNormalSingle(const Precomp& precomp,
                                         const std::vector<float>& re,
                                         std::vector<FullCode>* B,
                                         std::vector<float>* a,
                                         std::vector<float>* rC,
                                         std::vector<uint8_t>* available) {
    // Julia reference: quantize_with_kmeans_abs_nonormal_single_extend_allcodes_fast!
    // (greedy_quantizer.jl:4-58).
    const int d = precomp.d;
    const int H = precomp.H;
    const int m = precomp.m;

    if (static_cast<int>(available->size()) != m) {
        available->assign(static_cast<std::size_t>(m), 1);
    } else {
        std::fill(available->begin(), available->end(), 1);
    }
    if (static_cast<int>(rC->size()) != H) {
        rC->assign(static_cast<std::size_t>(H), 0.0f);
    }

    const float* re_ptr = re.data();
    for (int j = 0; j < H; ++j) {
        const float* c = precomp.C_all.Col(j);
        float s = 0.0f;
        #pragma omp simd reduction(+:s)
        for (int r = 0; r < d; ++r) {
            s += c[r] * re_ptr[r];
        }
        (*rC)[static_cast<std::size_t>(j)] = s;
    }

    if (m == 5 || m == 10) {
        std::uint32_t avail_mask = (m >= 31) ? 0xffffffffu : ((1u << static_cast<unsigned>(m)) - 1u);
        for (int t = 0; t < m; ++t) {
            int best_flat = 0;
            float best_abs = -std::numeric_limits<float>::infinity();
            float best_adj = 0.0f;
            for (int flat = 0; flat < H; ++flat) {
                const int layer = precomp.flat_layer[static_cast<std::size_t>(flat)];
                if (((avail_mask >> static_cast<unsigned>(layer)) & 1u) == 0u) {
                    continue;
                }
                const float adj =
                    (*rC)[static_cast<std::size_t>(flat)] * precomp.invnorm_flat[static_cast<std::size_t>(flat)];
                const float aval = std::abs(adj);
                if (aval > best_abs) {
                    best_abs = aval;
                    best_adj = adj;
                    best_flat = flat;
                }
            }

            const int layer = precomp.flat_layer[static_cast<std::size_t>(best_flat)];
            const int code = best_flat - precomp.offsets[layer];
            const float invn = precomp.invnorm_flat[static_cast<std::size_t>(best_flat)];
            const float alpha = best_adj * invn;

            (*B)[static_cast<std::size_t>(layer)] = static_cast<FullCode>(code);
            (*a)[static_cast<std::size_t>(layer)] = alpha;

            const float* Gcol = precomp.G.Col(best_flat);
            float* dst = rC->data();
            #pragma omp simd
            for (int j = 0; j < H; ++j) {
                dst[j] -= alpha * Gcol[j];
            }

            (*available)[static_cast<std::size_t>(layer)] = 0;
            avail_mask &= ~(1u << static_cast<unsigned>(layer));
        }
        return;
    }

    for (int t = 0; t < m; ++t) {
        int best_flat = 0;
        float best_abs = -std::numeric_limits<float>::infinity();
        float best_adj = 0.0f;
        for (int flat = 0; flat < H; ++flat) {
            int layer = precomp.flat_layer[static_cast<std::size_t>(flat)];
            if (!(*available)[static_cast<std::size_t>(layer)]) {
                continue;
            }
            float adj = (*rC)[static_cast<std::size_t>(flat)] *
                        precomp.invnorm_flat[static_cast<std::size_t>(flat)];
            float aval = std::abs(adj);
            if (aval > best_abs) {
                best_abs = aval;
                best_adj = adj;
                best_flat = flat;
            }
        }

        int layer = precomp.flat_layer[static_cast<std::size_t>(best_flat)];
        int code = best_flat - precomp.offsets[layer];
        float invn = precomp.invnorm_flat[static_cast<std::size_t>(best_flat)];
        float alpha = best_adj * invn;

        (*B)[static_cast<std::size_t>(layer)] = static_cast<FullCode>(code);
        (*a)[static_cast<std::size_t>(layer)] = alpha;

        const float* Gcol = precomp.G.Col(best_flat);
        float* dst = rC->data();
        #pragma omp simd
        for (int j = 0; j < H; ++j) {
            dst[j] -= alpha * Gcol[j];
        }

        (*available)[static_cast<std::size_t>(layer)] = 0;
    }
}

void ReconstructNode(const std::vector<ColMajorMatrix<float>>& C,
                     const std::vector<FullCode>& B,
                     const std::vector<float>& a,
                     std::vector<float>* out) {
    // Julia reference: reconstruct_node! (linkage_encode_helper.jl:503-517).
    const int d = C.front().rows;
    const int m = static_cast<int>(C.size());
    if (static_cast<int>(out->size()) != d) {
        out->assign(static_cast<std::size_t>(d), 0.0f);
    } else {
        std::fill(out->begin(), out->end(), 0.0f);
    }
    for (int l = 0; l < m; ++l) {
        int code = static_cast<int>(B[static_cast<std::size_t>(l)]);
        // if (code < 0 || code >= C[l].cols) continue;
        float coeff = a[static_cast<std::size_t>(l)];
        const float* center = C[l].Col(code);
        #pragma omp simd
        for (int r = 0; r < d; ++r) {
            (*out)[static_cast<std::size_t>(r)] += coeff * center[r];
        }
    }
}

void ReconstructAllInto(const std::vector<ColMajorMatrix<float>>& C,
                        const ColMajorMatrix<FullCode>& B,
                        const ColMajorMatrix<float>& a,
                        ColMajorMatrix<float>* out) {
    // Julia reference: compute_reconstructed (utils.jl:46-70).
    if (!out) {
        return;
    }
    const int d = C.empty() ? 0 : C.front().rows;
    const int n = B.cols;
    const int m = static_cast<int>(C.size());
    out->rows = d;
    out->cols = n;
    const std::size_t need = static_cast<std::size_t>(d) * static_cast<std::size_t>(n);
    if (out->data.size() < need) {
        out->data.resize(need);
    }
    if (need > 0) {
        std::fill_n(out->data.data(), need, 0.0f);
    }

    #pragma omp parallel for if (!omp_in_parallel()) schedule(static)
    for (int i = 0; i < n; ++i) {
        float* dst = out->Col(i);
        for (int l = 0; l < m; ++l) {
            int code = static_cast<int>(B(l, i));
            // if (code < 0 || code >= C[l].cols) continue;
            float coeff = a(l, i);
            const float* center = C[static_cast<std::size_t>(l)].Col(code);
            #pragma omp simd
            for (int r = 0; r < d; ++r) {
                dst[r] += coeff * center[r];
            }
        }
    }
}

ColMajorMatrix<float> ReconstructAll(const std::vector<ColMajorMatrix<float>>& C,
                                     const ColMajorMatrix<FullCode>& B,
                                     const ColMajorMatrix<float>& a) {
    ColMajorMatrix<float> out;
    ReconstructAllInto(C, B, a, &out);
    return out;
}

[[maybe_unused]] void ComputeCosineAngles(const ColMajorMatrix<float>& X,
                         const ColMajorMatrix<float>& C0,
                         const std::vector<std::vector<int>>& clusters,
                         std::vector<float>* cos_angles) {
    // Julia reference: compute_cosine_angles (linkage_encode_helper.jl:294-326).
    const int d = X.rows;
    const int n = X.cols;
    cos_angles->assign(static_cast<std::size_t>(n), 0.0f);

    std::vector<float> x_norms(static_cast<std::size_t>(n), 0.0f);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const float* x = X.Col(i);
        float s = 0.0f;
        #pragma omp simd reduction(+:s)
        for (int r = 0; r < d; ++r) {
            s += x[r] * x[r];
        }
        x_norms[static_cast<std::size_t>(i)] = std::sqrt(std::max(s, 1e-12f));
    }

    #pragma omp parallel for schedule(dynamic)
    for (int code = 0; code < static_cast<int>(clusters.size()); ++code) {
        const auto& idxs = clusters[static_cast<std::size_t>(code)];
        if (idxs.empty()) {
            continue;
        }
        const float* center = C0.Col(code);
        float cnorm = 0.0f;
        #pragma omp simd reduction(+:cnorm)
        for (int r = 0; r < d; ++r) {
            cnorm += center[r] * center[r];
        }
        cnorm = std::sqrt(std::max(cnorm, 1e-12f));
        for (int g : idxs) {
            const float* x = X.Col(static_cast<int>(g));
            float dot = 0.0f;
            #pragma omp simd reduction(+:dot)
            for (int r = 0; r < d; ++r) {
                dot += x[r] * center[r];
            }
            float denom = std::max(cnorm * x_norms[static_cast<std::size_t>(g)], 1e-12f);
            (*cos_angles)[static_cast<std::size_t>(g)] = dot / denom;
        }
    }
}

void PartitionClusterLayers(const std::vector<float>& cos_angles,
                            const std::vector<int>& cluster_indices,
                            int num_layers,
                            float root_percentile,
                            std::vector<std::vector<int>>* layers_local) {
    // Julia reference: partition_by_cosine_percentiles (linkage_encode_helper.jl:332-452).
    const int n_points = static_cast<int>(cluster_indices.size());
    layers_local->assign(static_cast<std::size_t>(num_layers), std::vector<int>());
    if (n_points == 0) {
        return;
    }

    std::vector<int> order(static_cast<std::size_t>(n_points), 0);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        int ga = cluster_indices[static_cast<std::size_t>(a)];
        int gb = cluster_indices[static_cast<std::size_t>(b)];
        return cos_angles[static_cast<std::size_t>(ga)] > cos_angles[static_cast<std::size_t>(gb)];
    });

    if (num_layers <= 1) {
        (*layers_local)[0] = order;
        return;
    }

    int n_root = static_cast<int>(std::floor(root_percentile * n_points));
    n_root = std::max(1, std::min(n_root, n_points));
    (*layers_local)[0].assign(order.begin(), order.begin() + n_root);

    int remaining = n_points - n_root;
    int base_layer_size = remaining / (num_layers - 1);
    int remainder = remaining % (num_layers - 1);

    int current_idx = n_root;
    for (int layer = 1; layer < num_layers; ++layer) {
        int layer_size = base_layer_size + (layer <= remainder ? 1 : 0);
        int end_idx = std::min(current_idx + layer_size, n_points);
        if (current_idx < end_idx) {
            (*layers_local)[static_cast<std::size_t>(layer)]
                .assign(order.begin() + current_idx, order.begin() + end_idx);
        }
        current_idx = end_idx;
    }
}

inline std::uint64_t CausalReferenceSelectionKey(int seed,
                                                 int cluster_id,
                                                 int node_local) {
    std::uint64_t x = static_cast<std::uint32_t>(seed);
    x ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(cluster_id)) << 32U;
    x ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(node_local)) *
         0x9e3779b97f4a7c15ULL;
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31U);
}

// Reduce the structured topology's admissible candidate set to one parent.
// This happens after radial/same-layer causal filtering but before residual
// evaluation. Depth-ineligible candidates are excluded, so the chosen parent
// is always a legal edge. The stateless random key deliberately does not
// consume the ICM/ILS random stream.
inline void SelectOneCausalReference(const std::string& policy,
                                     std::uint64_t selection_key,
                                     const float* query,
                                     const ColMajorMatrix<float>& X,
                                     const std::vector<int>& cluster_indices,
                                     const std::vector<int>& parent_depth,
                                     int max_depth,
                                     std::vector<int>* first,
                                     std::vector<int>* second = nullptr) {
    if (!IsCausalSingleParentPolicy(policy)) return;

    const auto FilterDepth = [&](std::vector<int>* candidates) {
        if (!candidates) return;
        std::size_t keep = 0;
        for (int candidate : *candidates) {
            if (candidate < 0 ||
                candidate >= static_cast<int>(parent_depth.size()) ||
                parent_depth[static_cast<std::size_t>(candidate)] + 1 > max_depth) {
                continue;
            }
            (*candidates)[keep++] = candidate;
        }
        candidates->resize(keep);
    };
    FilterDepth(first);
    FilterDepth(second);

    const std::size_t first_size = first ? first->size() : 0u;
    const std::size_t second_size = second ? second->size() : 0u;
    const std::size_t total = first_size + second_size;
    if (total == 0u) return;

    int chosen = -1;
    bool chosen_from_first = false;
    if (policy == "causal_random") {
        const std::size_t pick = static_cast<std::size_t>(selection_key % total);
        chosen_from_first = pick < first_size;
        chosen = chosen_from_first
                     ? (*first)[pick]
                     : (*second)[pick - first_size];
    } else {
        float best_distance = std::numeric_limits<float>::infinity();
        const auto Consider = [&](const std::vector<int>* candidates,
                                  bool from_first) {
            if (!candidates) return;
            for (int candidate : *candidates) {
                const int global = cluster_indices[static_cast<std::size_t>(candidate)];
                const float* parent = X.Col(global);
                float distance = 0.0f;
#pragma omp simd reduction(+:distance)
                for (int r = 0; r < X.rows; ++r) {
                    const float diff = query[r] - parent[r];
                    distance += diff * diff;
                }
                if (distance < best_distance ||
                    (distance == best_distance &&
                     (chosen < 0 || candidate < chosen))) {
                    best_distance = distance;
                    chosen = candidate;
                    chosen_from_first = from_first;
                }
            }
        };
        Consider(first, true);
        Consider(second, false);
    }

    if (first) first->clear();
    if (second) second->clear();
    if (chosen < 0) return;
    if (chosen_from_first || !second) {
        first->push_back(chosen);
    } else {
        second->push_back(chosen);
    }
}

void QueryCandidatesHnsw(const ClusterHnswIndex& index,
                         const float* query,
                         const std::vector<int>& cluster_indices,
                         int local_idx,
                         const std::vector<int>& layer_of_point,
                         const std::vector<int>& local_pos,
                         const std::vector<uint8_t>& processed,
                         int inner_lo,
                         int inner_hi,
                         int layer_idx,
                         int num_layers,
                         int knn_k,
                         int candidate_multiplier,
                         std::vector<int>* out) {
    // Julia reference: knn(tree_cluster, Xi, Kp, true) usage (linkage_quantizer.jl:148-166).
    // HNSW path preserves squared L2 semantics and applies the same layer filters.
    const int csize = static_cast<int>(cluster_indices.size());
    const int max_possible = std::max(0, csize - 1);
    if (max_possible <= 0) {
        out->clear();
        return;
    }

    const int depth_span = std::max(0, inner_hi - inner_lo + 1);
    const int mult = std::max(1, candidate_multiplier);
    // Julia reference: Kp = min(csize, max(32, 4 * (depth_span + 1) * knn_k)).
    // We use the configurable multiplier in place of the Julia constant 4.
    int kprime = (depth_span + 1) * std::max(1, knn_k) * mult;
    kprime = std::max(kprime, 32);
    kprime = std::min(kprime, max_possible);
    kprime = std::max(kprime, 1);

    out->clear();
    index.SetEf(kprime);
    // Hot path: avoid per-call allocations (this runs for every node).
    static thread_local std::vector<int> candidates;
    if (candidates.capacity() < static_cast<std::size_t>(kprime)) {
        candidates.reserve(static_cast<std::size_t>(kprime));
    }
    try {
        index.SearchLabelsInto(query, kprime, &candidates);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("HNSW search failed: ") + e.what());
    }

    // Julia reference: KNN candidate list is filtered by allowed parent layers, with a per-layer
    // budget of knn_k and an early-exit once the total budget is met
    // (linkage_quantizer.jl:161-210).
    static thread_local std::vector<int> per_layer_budget;
    const int num_layers_use = std::max(0, num_layers);
    if (static_cast<int>(per_layer_budget.size()) < num_layers_use) {
        per_layer_budget.resize(static_cast<std::size_t>(num_layers_use), 0);
    }
    const int knn_k_use = std::max(1, knn_k);
    if (num_layers_use > 0) {
        std::fill_n(per_layer_budget.data(),
                    static_cast<std::size_t>(num_layers_use),
                    knn_k_use);
    }
    const int want_total = (depth_span + 1) * std::max(1, knn_k);

    for (int c_rel : candidates) {
        if (c_rel == local_idx) {
            continue;
        }
        int p_global = cluster_indices[static_cast<std::size_t>(c_rel)];
        int lp = layer_of_point[static_cast<std::size_t>(p_global)];
        if (lp >= inner_lo && lp <= inner_hi) {
            // if (lp >= 0 && lp < num_layers &&
            //     per_layer_budget[static_cast<std::size_t>(lp)] > 0) {
            if (per_layer_budget[static_cast<std::size_t>(lp)] > 0) {
                out->push_back(c_rel);
                per_layer_budget[static_cast<std::size_t>(lp)] -= 1;
            }
            // }
        } else if (lp == layer_idx) {
            int pos = local_pos[static_cast<std::size_t>(c_rel)];
            // if (pos >= 0 && processed[static_cast<std::size_t>(pos)]) {
            if (processed[static_cast<std::size_t>(pos)]) {
                // if (lp >= 0 && lp < num_layers &&
                //     per_layer_budget[static_cast<std::size_t>(lp)] > 0) {
                if (per_layer_budget[static_cast<std::size_t>(lp)] > 0) {
                    out->push_back(c_rel);
                    per_layer_budget[static_cast<std::size_t>(lp)] -= 1;
                }
                // }
            }
            // }
        }
        if (static_cast<int>(out->size()) >= want_total) {
            break;
        }
    }
}

inline void QueryCandidatesHnswSplit(const ClusterHnswIndex& index,
                                    const float* query,
                                    const std::vector<int>& cluster_indices,
                                    int query_local,
                                    const std::vector<int>& layer_of_point,
                                    const std::vector<int>& local_pos,
                                    const std::vector<uint8_t>& processed,
                                    int inner_lo,
                                    int inner_hi,
                                    int layer_idx,
                                    int num_layers,
                                    int knn_k,
                                    int candidate_multiplier,
                                    std::vector<int>* out_inner,
                                    std::vector<int>* out_same) {
    const int csize = static_cast<int>(cluster_indices.size());
    const int max_possible = std::max(0, csize - 1);
    if (max_possible <= 0) {
        if (out_inner) out_inner->clear();
        if (out_same) out_same->clear();
        return;
    }

    const int depth_span = std::max(0, inner_hi - inner_lo + 1);
    const int knn_k_use = std::max(1, knn_k);
    const int mult = std::max(1, candidate_multiplier);
    const int want_total = (depth_span + 1) * knn_k_use;

    int kprime = std::max(32, want_total * mult);
    kprime = std::min(kprime, max_possible);
    kprime = std::max(kprime, 1);

    if (out_inner) out_inner->clear();
    if (out_same) out_same->clear();

    index.SetEf(kprime);
    // Hot path: avoid per-call allocations (this runs for every node).
    static thread_local std::vector<int> candidates;
    if (candidates.capacity() < static_cast<std::size_t>(kprime)) {
        candidates.reserve(static_cast<std::size_t>(kprime));
    }
    try {
        index.SearchLabelsInto(query, kprime, &candidates);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("HNSW search failed: ") + e.what());
    }

    static thread_local std::vector<int> per_layer_budget;
    const int num_layers_use = std::max(0, num_layers);
    if (static_cast<int>(per_layer_budget.size()) < num_layers_use) {
        per_layer_budget.resize(static_cast<std::size_t>(num_layers_use), 0);
    }
    if (num_layers_use > 0) {
        std::fill_n(per_layer_budget.data(),
                    static_cast<std::size_t>(num_layers_use),
                    0);
        const int lo = std::max(0, inner_lo);
        const int hi = std::min(num_layers_use - 1, inner_hi);
        for (int lp = lo; lp <= hi; ++lp) {
            per_layer_budget[static_cast<std::size_t>(lp)] = knn_k_use;
        }
        if (layer_idx >= 0 && layer_idx < num_layers_use) {
            per_layer_budget[static_cast<std::size_t>(layer_idx)] = knn_k_use;
        }
    }

    const int cap_inner = std::max(0, depth_span * knn_k_use);
    if (out_inner && out_inner->capacity() < static_cast<std::size_t>(cap_inner)) {
        out_inner->reserve(static_cast<std::size_t>(cap_inner));
    }
    if (out_same && out_same->capacity() < static_cast<std::size_t>(knn_k_use)) {
        out_same->reserve(static_cast<std::size_t>(knn_k_use));
    }

    for (int c_rel : candidates) {
        if (c_rel == query_local) {
            continue;
        }
        const int p_global = cluster_indices[static_cast<std::size_t>(c_rel)];
        const int lp = layer_of_point[static_cast<std::size_t>(p_global)];
        if (lp < inner_lo || lp > layer_idx) {
            continue;
        }
        if (lp == layer_idx) {
            const int pos_in_layer = local_pos[static_cast<std::size_t>(c_rel)];
            if (pos_in_layer < 0 || !processed[static_cast<std::size_t>(pos_in_layer)]) {
                continue;
            }
            if (lp >= 0 && lp < num_layers_use &&
                per_layer_budget[static_cast<std::size_t>(lp)] > 0) {
                per_layer_budget[static_cast<std::size_t>(lp)] -= 1;
                if (out_same) out_same->push_back(c_rel);
            }
        } else {
            if (lp >= 0 && lp < num_layers_use &&
                per_layer_budget[static_cast<std::size_t>(lp)] > 0) {
                per_layer_budget[static_cast<std::size_t>(lp)] -= 1;
                if (out_inner) out_inner->push_back(c_rel);
            }
        }
        const std::size_t got =
            (out_inner ? out_inner->size() : 0u) + (out_same ? out_same->size() : 0u);
        if (static_cast<int>(got) >= want_total) {
            break;
        }
    }
}
