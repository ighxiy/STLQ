
constexpr float kEps = 1e-6f;

void HierarchicalQuantize(const ColMajorMatrix<float>& X,
                          const std::vector<int>& h_vec,
                          const KmeansConfig& cfg,
                          std::mt19937* rng,
                          StreamKernelProvider* stream_kernels,
                          bool profile_timing,
                          KmeansTiming* kmeans_timing,
                          std::vector<ColMajorMatrix<float>>* codebooks,
                          ColMajorMatrix<FullCode>* B,
                          ColMajorMatrix<float>* a) {
    // C++ init for hierarchical_vector_quantization_spherical_extend
    // (train_quantizer.jl:390-486).
    const int d = X.rows;
    const int n = X.cols;
    const int m = static_cast<int>(h_vec.size());
    LogInfo("RVQ init: HierarchicalQuantize(in-memory) n=" + std::to_string(n) +
            " d=" + std::to_string(d) +
            " m=" + std::to_string(m) +
            " kmeans_streaming=" + std::to_string(cfg.streaming ? 1 : 0) +
            " gpu=" + std::to_string((stream_kernels && stream_kernels->IsGpu()) ? 1 : 0));

    codebooks->assign(m, ColMajorMatrix<float>());
    B->rows = m;
    B->cols = n;
    B->data.assign(static_cast<std::size_t>(m) * n, 0);
    a->rows = m;
    a->cols = n;
    a->data.assign(static_cast<std::size_t>(m) * n, 0.0f);

    ColMajorMatrix<float> residuals = X;
    std::vector<int> order(static_cast<std::size_t>(n), 0);

    for (int layer = 0; layer < m; ++layer) {
        int k = h_vec[layer];
        KmeansConfig layer_cfg = cfg;
        if (layer != 0) {
            layer_cfg.bvecs_use_u8 = false;
        }
        KmeansResult km = SphericalKmeans(residuals, k, layer_cfg, rng,
                                          stream_kernels,
                                          profile_timing,
                                          kmeans_timing);
        (*codebooks)[layer] = km.centers;

        std::vector<int> offsets(static_cast<std::size_t>(k) + 1, 0);
        for (int i = 0; i < n; ++i) {
            int code = km.assignments[static_cast<std::size_t>(i)];
            ++offsets[static_cast<std::size_t>(code) + 1];
        }
        for (int code = 0; code < k; ++code) {
            offsets[static_cast<std::size_t>(code) + 1] += offsets[static_cast<std::size_t>(code)];
        }
        std::vector<int> cursor = offsets;
        for (int i = 0; i < n; ++i) {
            int code = km.assignments[static_cast<std::size_t>(i)];
            int pos = cursor[static_cast<std::size_t>(code)]++;
            order[static_cast<std::size_t>(pos)] = i;
        }

        #pragma omp parallel for schedule(static)
        for (int code = 0; code < k; ++code) {
            const float* center = km.centers.Col(code);
            int begin = offsets[static_cast<std::size_t>(code)];
            int end = offsets[static_cast<std::size_t>(code) + 1];
            for (int p = begin; p < end; ++p) {
                int i = order[static_cast<std::size_t>(p)];
                float* res = residuals.Col(i);

                float proj = 0.0f;
                #pragma omp simd reduction(+:proj)
                for (int r = 0; r < d; ++r) {
                    proj += center[r] * res[r];
                }

                (*B)(layer, i) = static_cast<FullCode>(code);
                (*a)(layer, i) = proj;

                #pragma omp simd
                for (int r = 0; r < d; ++r) {
                    res[r] -= proj * center[r];
                }
            }
        }
    }
}

class RvqResidualReader final : public stlq::io::IColBlockReader {
public:
    RvqResidualReader(stlq::io::IColBlockReader* base,
                      stlq::RvqLayerStoreReader* store,
                      const std::vector<ColMajorMatrix<float>>* codebooks,
                      int upto_layer)
        : base_(base), store_(store), codebooks_(codebooks), upto_layer_(upto_layer) {}

    [[nodiscard]] int d() const override { return base_ ? base_->d() : 0; }
    [[nodiscard]] std::int64_t n() const override { return base_ ? base_->n() : 0; }

    bool Reset(std::string* err) override {
        if (!base_) {
            if (err) *err = "RvqResidualReader::Reset: base is null.";
            return false;
        }
        if (!base_->Reset(err)) return false;
        if (upto_layer_ > 0) {
            if (!store_) {
                if (err) *err = "RvqResidualReader::Reset: store is null.";
                return false;
            }
            if (!store_->ResetUpto(upto_layer_, err)) return false;
        }
        return true;
    }

    bool ReadNext(int max_cols, stlq::io::ColBlock* out, std::string* err) override {
        if (!out) {
            if (err) *err = "RvqResidualReader::ReadNext: out is null.";
            return false;
        }
        if (!base_) {
            if (err) *err = "RvqResidualReader::ReadNext: base is null.";
            return false;
        }
        if (!base_->ReadNext(max_cols, out, err)) return false;
        const int cols0 =
            (out->dtype == stlq::io::ColBlockDType::kU8) ? out->X_u8.cols : out->X_f32.cols;
        if (cols0 == 0) return true;
        if (upto_layer_ <= 0) return true;
        if (!store_ || !codebooks_) {
            if (err) *err = "RvqResidualReader::ReadNext: store/codebooks null.";
            return false;
        }
        if (out->dtype == stlq::io::ColBlockDType::kU8) {
            const int d0 = out->X_u8.rows;
            const int cols = out->X_u8.cols;
            out->X_f32 = ColMajorMatrix<float>(d0, cols);
            for (int t = 0; t < cols; ++t) {
                const std::uint8_t* src = out->X_u8.Col(t);
                float* dst = out->X_f32.Col(t);
                #pragma omp simd
                for (int r = 0; r < d0; ++r) dst[r] = static_cast<float>(src[r]);
            }
            out->dtype = stlq::io::ColBlockDType::kF32;
        }
        if (!store_->ReadNextUpto(upto_layer_, out->X_f32.cols, &codes_, &a_, err)) return false;

        const int d0 = out->X_f32.rows;
        const int cols = out->X_f32.cols;
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < cols; ++t) {
            float* rt = out->X_f32.Col(t);
            for (int j = 0; j < upto_layer_; ++j) {
                const std::uint16_t code = codes_[static_cast<std::size_t>(j)][static_cast<std::size_t>(t)];
                const float alpha = a_[static_cast<std::size_t>(j)][static_cast<std::size_t>(t)];
                const float* cj = (*codebooks_)[static_cast<std::size_t>(j)].Col(static_cast<int>(code));
                #pragma omp simd
                for (int r = 0; r < d0; ++r) {
                    rt[r] -= alpha * cj[r];
                }
            }
        }
        return true;
    }

private:
    stlq::io::IColBlockReader* base_ = nullptr;
    stlq::RvqLayerStoreReader* store_ = nullptr;
    const std::vector<ColMajorMatrix<float>>* codebooks_ = nullptr;
    int upto_layer_ = 0;
    std::vector<std::vector<std::uint16_t>> codes_;
    std::vector<std::vector<float>> a_;
};

std::vector<ColMajorMatrix<float>> UpdateCodebooksLS(const ColMajorMatrix<float>& X,
                                                     const ColMajorMatrix<FullCode>& B,
                                                     const ColMajorMatrix<float>& a,
                                                     const std::vector<int>& h_vec) {
    const int d = X.rows;
    const int n = X.cols;
    const int m = static_cast<int>(h_vec.size());

    int total_k = 0;
    std::vector<int> offsets(m, 0);
    for (int l = 0; l < m; ++l) {
        offsets[l] = total_k;
        total_k += h_vec[l];
    }

    // Memory-efficient determinism: build CSR lists of samples per "flat row" idx_l (layer,code),
    // then compute each G/T row independently in a fixed order. This avoids:
    //   - nthreads * total_k^2 temporary buffers (too large for m=10,h=256)
    //   - omp critical merge (nondeterministic order)
    std::vector<int> row_counts(static_cast<std::size_t>(total_k), 0);
    for (int i = 0; i < n; ++i) {
        const FullCode* bcol = B.Col(i);
        for (int l = 0; l < m; ++l) {
            const int idx_l = offsets[l] + static_cast<int>(bcol[l]);
            row_counts[static_cast<std::size_t>(idx_l)] += 1;
        }
    }
    std::vector<int> row_offsets(static_cast<std::size_t>(total_k) + 1, 0);
    for (int r = 0; r < total_k; ++r) {
        row_offsets[static_cast<std::size_t>(r) + 1] =
            row_offsets[static_cast<std::size_t>(r)] + row_counts[static_cast<std::size_t>(r)];
    }
    std::vector<int> row_flat(static_cast<std::size_t>(row_offsets.back()), 0);
    std::vector<int> cursor = row_offsets;
    for (int i = 0; i < n; ++i) {
        const FullCode* bcol = B.Col(i);
        for (int l = 0; l < m; ++l) {
            const int idx_l = offsets[l] + static_cast<int>(bcol[l]);
            const int pos = cursor[static_cast<std::size_t>(idx_l)]++;
            row_flat[static_cast<std::size_t>(pos)] = i;
        }
    }
    std::vector<int> row_layer(static_cast<std::size_t>(total_k), 0);
    for (int l = 0; l < m; ++l) {
        for (int code = 0; code < h_vec[l]; ++code) {
            row_layer[static_cast<std::size_t>(offsets[l] + code)] = l;
        }
    }

    std::vector<float> G(static_cast<std::size_t>(total_k) * total_k, 0.0f);   // row-major
    std::vector<float> T(static_cast<std::size_t>(total_k) * d, 0.0f);         // row-major (total_k rows, d cols)

    #pragma omp parallel
    {
        std::vector<double> grow(static_cast<std::size_t>(total_k), 0.0);
        std::vector<double> trow(static_cast<std::size_t>(d), 0.0);

        #pragma omp for schedule(static)
        for (int ridx = 0; ridx < total_k; ++ridx) {
            std::fill(grow.begin(), grow.end(), 0.0);
            std::fill(trow.begin(), trow.end(), 0.0);

            const int l = row_layer[static_cast<std::size_t>(ridx)];
            const int begin = row_offsets[static_cast<std::size_t>(ridx)];
            const int end = row_offsets[static_cast<std::size_t>(ridx) + 1];
            for (int p = begin; p < end; ++p) {
                const int i = row_flat[static_cast<std::size_t>(p)];
                const FullCode* bcol = B.Col(i);
                const float* acol = a.Col(i);
                const float alpha_l = acol[l];
                const float* xi = X.Col(i);

                for (int r = 0; r < d; ++r) {
                    trow[static_cast<std::size_t>(r)] += static_cast<double>(alpha_l) * static_cast<double>(xi[r]);
                }
                for (int kk = 0; kk < m; ++kk) {
                    const int idx_k = offsets[kk] + static_cast<int>(bcol[kk]);
                    grow[static_cast<std::size_t>(idx_k)] +=
                        static_cast<double>(alpha_l) * static_cast<double>(acol[kk]);
                }
            }

            float* Gdst = G.data() + static_cast<std::size_t>(ridx) * total_k;
            float* Tdst = T.data() + static_cast<std::size_t>(ridx) * d;
            #pragma omp simd
            for (int c = 0; c < total_k; ++c) {
                Gdst[c] = static_cast<float>(grow[static_cast<std::size_t>(c)]);
            }
            #pragma omp simd
            for (int r = 0; r < d; ++r) {
                Tdst[r] = static_cast<float>(trow[static_cast<std::size_t>(r)]);
            }
        }
    }

    // Solve (G * W = T) using LAPACK Cholesky (no hand-written Cholesky/Solve).
    // Julia reference: update_codebooks_ls_extend (train_quantizer.jl), which is LS solve per (layer, code).
    //
    // Convert accumulated row-major G into column-major upper-triangle for LAPACK.
    std::vector<float> Gcm(static_cast<std::size_t>(total_k) * total_k, 0.0f);
    for (int r = 0; r < total_k; ++r) {
        for (int c = 0; c < total_k; ++c) {
            // G is row-major: G[r,c] = G[r*total_k + c]
            Gcm[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * total_k] =
                G[static_cast<std::size_t>(r) * total_k + c];
        }
    }

    std::vector<float> Gbak = Gcm;
    int info = lapack::SpotrfU(total_k, Gcm.data(), total_k);
    if (info != 0) {
        float bump = kEps;
        for (int attempt = 0; attempt < 10 && info != 0; ++attempt) {
            Gcm = Gbak;
            for (int i = 0; i < total_k; ++i) {
                Gcm[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * total_k] += bump;
            }
            info = lapack::SpotrfU(total_k, Gcm.data(), total_k);
            bump *= 10.0f;
        }
        if (info != 0) {
            LogWarn("UpdateCodebooksLS: potrf failed (info=" + std::to_string(info) +
                    "), returning zero codebooks.");
            std::vector<ColMajorMatrix<float>> C(m);
            for (int l = 0; l < m; ++l) {
                C[l] = ColMajorMatrix<float>(d, h_vec[l]);
                std::fill(C[l].data.begin(), C[l].data.end(), 0.0f);
            }
            return C;
        }
    }

    ColMajorMatrix<float> Wcm(total_k, d);
    for (int idx = 0; idx < total_k; ++idx) {
        for (int r = 0; r < d; ++r) {
            Wcm(idx, r) = T[static_cast<std::size_t>(idx) * d + r];
        }
    }
    info = lapack::SpotrsU(total_k, d, Gcm.data(), total_k, Wcm.data.data(), total_k);
    if (info != 0) {
        LogWarn("UpdateCodebooksLS: potrs failed (info=" + std::to_string(info) +
                "), returning zero codebooks.");
        std::vector<ColMajorMatrix<float>> C(m);
        for (int l = 0; l < m; ++l) {
            C[l] = ColMajorMatrix<float>(d, h_vec[l]);
            std::fill(C[l].data.begin(), C[l].data.end(), 0.0f);
        }
        return C;
    }

    std::vector<ColMajorMatrix<float>> C(m);
    for (int l = 0; l < m; ++l) {
        int k = h_vec[l];
        C[l] = ColMajorMatrix<float>(d, k);
        for (int code = 0; code < k; ++code) {
            int idx = offsets[l] + code;
            float* dst = C[l].Col(code);
            for (int r = 0; r < d; ++r) {
                dst[r] = Wcm(idx, r);
            }
        }
    }
    return C;
}

// Update codebooks via LS from scattered residual samples without forming a contiguous X/B/a subset.
// Each sample j contributes:
//   x_res = X[:, idx[j]] - parent_recon[j]
// with per-sample weight w[j] applied as:
//   G += w * alpha_l * alpha_k
//   T += w * alpha_l * x_res
// which matches the "scale by sqrt(w)" behavior used elsewhere.
std::vector<ColMajorMatrix<float>> UpdateCodebooksLSFromResidualPtrs(
        const ColMajorMatrix<float>& X,
        const ColMajorMatrix<FullCode>& B,
        const ColMajorMatrix<float>& a,
        const std::vector<int>& idx,
        const std::vector<const float*>& parent_recon,
        const std::vector<float>& w,
        const std::vector<int>& h_vec,
        const std::uint16_t* b0_override) {
    const int d = X.rows;
    const int n = X.cols;
    const int k = static_cast<int>(idx.size());
    const int m = static_cast<int>(h_vec.size());
    if (k <= 0 || d <= 0 || n <= 0 || m <= 0) {
        return {};
    }

    int total_k = 0;
    std::vector<int> offsets(m, 0);
    for (int l = 0; l < m; ++l) {
        offsets[l] = total_k;
        total_k += h_vec[l];
    }

    // Memory-efficient determinism: CSR lists of subset-samples per flat row idx_l.
    std::vector<int> row_counts(static_cast<std::size_t>(total_k), 0);
    for (int j = 0; j < k; ++j) {
        const int gi = idx[static_cast<std::size_t>(j)];
        const FullCode* bcol = B.Col(gi);
        for (int l = 0; l < m; ++l) {
            const int code_l =
                (l == 0 && b0_override) ? static_cast<int>(b0_override[j])
                                        : static_cast<int>(bcol[l]);
            const int idx_l = offsets[l] + code_l;
            row_counts[static_cast<std::size_t>(idx_l)] += 1;
        }
    }
    std::vector<int> row_offsets(static_cast<std::size_t>(total_k) + 1, 0);
    for (int r = 0; r < total_k; ++r) {
        row_offsets[static_cast<std::size_t>(r) + 1] =
            row_offsets[static_cast<std::size_t>(r)] + row_counts[static_cast<std::size_t>(r)];
    }
    std::vector<int> row_flat(static_cast<std::size_t>(row_offsets.back()), 0);
    std::vector<int> cursor = row_offsets;
    for (int j = 0; j < k; ++j) {
        const int gi = idx[static_cast<std::size_t>(j)];
        const FullCode* bcol = B.Col(gi);
        for (int l = 0; l < m; ++l) {
            const int code_l =
                (l == 0 && b0_override) ? static_cast<int>(b0_override[j])
                                        : static_cast<int>(bcol[l]);
            const int idx_l = offsets[l] + code_l;
            const int pos = cursor[static_cast<std::size_t>(idx_l)]++;
            row_flat[static_cast<std::size_t>(pos)] = j;
        }
    }
    std::vector<int> row_layer(static_cast<std::size_t>(total_k), 0);
    for (int l = 0; l < m; ++l) {
        for (int code = 0; code < h_vec[l]; ++code) {
            row_layer[static_cast<std::size_t>(offsets[l] + code)] = l;
        }
    }

    std::vector<float> G(static_cast<std::size_t>(total_k) * total_k, 0.0f);   // row-major
    std::vector<float> T(static_cast<std::size_t>(total_k) * d, 0.0f);         // row-major

    #pragma omp parallel
    {
        std::vector<double> grow(static_cast<std::size_t>(total_k), 0.0);
        std::vector<double> trow(static_cast<std::size_t>(d), 0.0);

        #pragma omp for schedule(static)
        for (int ridx = 0; ridx < total_k; ++ridx) {
            std::fill(grow.begin(), grow.end(), 0.0);
            std::fill(trow.begin(), trow.end(), 0.0);

            const int l = row_layer[static_cast<std::size_t>(ridx)];
            const int begin = row_offsets[static_cast<std::size_t>(ridx)];
            const int end = row_offsets[static_cast<std::size_t>(ridx) + 1];
            for (int p = begin; p < end; ++p) {
                const int j = row_flat[static_cast<std::size_t>(p)];
                const int gi = idx[static_cast<std::size_t>(j)];
                const FullCode* bcol = B.Col(gi);
                const float* acol = a.Col(gi);
                const float alpha_l = acol[l];
                const float weight = w.empty() ? 1.0f : w[static_cast<std::size_t>(j)];
                const float* xi = X.Col(gi);
                const float* rp = parent_recon[static_cast<std::size_t>(j)];

                const double scale = static_cast<double>(weight) * static_cast<double>(alpha_l);
                for (int r = 0; r < d; ++r) {
                    trow[static_cast<std::size_t>(r)] += scale * (static_cast<double>(xi[r]) - static_cast<double>(rp[r]));
                }
                const double wl = static_cast<double>(weight) * static_cast<double>(alpha_l);
                for (int kk = 0; kk < m; ++kk) {
                    const int code_k =
                        (kk == 0 && b0_override) ? static_cast<int>(b0_override[j])
                                                 : static_cast<int>(bcol[kk]);
                    const int idx_k = offsets[kk] + code_k;
                    grow[static_cast<std::size_t>(idx_k)] += wl * static_cast<double>(acol[kk]);
                }
            }

            float* Gdst = G.data() + static_cast<std::size_t>(ridx) * total_k;
            float* Tdst = T.data() + static_cast<std::size_t>(ridx) * d;
            #pragma omp simd
            for (int c = 0; c < total_k; ++c) {
                Gdst[c] = static_cast<float>(grow[static_cast<std::size_t>(c)]);
            }
            #pragma omp simd
            for (int r = 0; r < d; ++r) {
                Tdst[r] = static_cast<float>(trow[static_cast<std::size_t>(r)]);
            }
        }
    }

    // Solve (G * W = T) using LAPACK Cholesky (same path as UpdateCodebooksLS).
    std::vector<float> Gcm(static_cast<std::size_t>(total_k) * total_k, 0.0f);
    for (int r = 0; r < total_k; ++r) {
        for (int c = 0; c < total_k; ++c) {
            Gcm[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * total_k] =
                G[static_cast<std::size_t>(r) * total_k + c];
        }
    }

    std::vector<float> Gbak = Gcm;
    int info = lapack::SpotrfU(total_k, Gcm.data(), total_k);
    if (info != 0) {
        float bump = kEps;
        for (int attempt = 0; attempt < 10 && info != 0; ++attempt) {
            Gcm = Gbak;
            for (int i = 0; i < total_k; ++i) {
                Gcm[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * total_k] += bump;
            }
            info = lapack::SpotrfU(total_k, Gcm.data(), total_k);
            bump *= 10.0f;
        }
        if (info != 0) {
            LogWarn("UpdateCodebooksLSFromResidualPtrs: potrf failed (info=" +
                    std::to_string(info) + "), returning zero codebooks.");
            std::vector<ColMajorMatrix<float>> C(m);
            for (int l = 0; l < m; ++l) {
                C[l] = ColMajorMatrix<float>(d, h_vec[l]);
                std::fill(C[l].data.begin(), C[l].data.end(), 0.0f);
            }
            return C;
        }
    }

    ColMajorMatrix<float> Wcm(total_k, d);
    for (int idxr = 0; idxr < total_k; ++idxr) {
        for (int r = 0; r < d; ++r) {
            Wcm(idxr, r) = T[static_cast<std::size_t>(idxr) * d + r];
        }
    }
    info = lapack::SpotrsU(total_k, d, Gcm.data(), total_k, Wcm.data.data(), total_k);
    if (info != 0) {
        LogWarn("UpdateCodebooksLSFromResidualPtrs: potrs failed (info=" +
                std::to_string(info) + "), returning zero codebooks.");
        std::vector<ColMajorMatrix<float>> C(m);
        for (int l = 0; l < m; ++l) {
            C[l] = ColMajorMatrix<float>(d, h_vec[l]);
            std::fill(C[l].data.begin(), C[l].data.end(), 0.0f);
        }
        return C;
    }

    std::vector<ColMajorMatrix<float>> C(m);
    for (int l = 0; l < m; ++l) {
        int kk = h_vec[l];
        C[l] = ColMajorMatrix<float>(d, kk);
        for (int code = 0; code < kk; ++code) {
            int row = offsets[l] + code;
            float* dst = C[l].Col(code);
            for (int r = 0; r < d; ++r) {
                dst[r] = Wcm(row, r);
            }
        }
    }
    return C;
}

template <typename T>
void GatherColumns(const ColMajorMatrix<T>& X,
                   const std::vector<int>& cols,
                   ColMajorMatrix<T>* out) {
    const int d = X.rows;
    const int n = static_cast<int>(cols.size());
    out->rows = d;
    out->cols = n;
    out->data.assign(static_cast<std::size_t>(d) * n, T{});

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n; ++j) {
        int src_col = cols[static_cast<std::size_t>(j)];
        const T* src = X.Col(src_col);
        T* dst = out->Col(j);
        for (int r = 0; r < d; ++r) {
            dst[r] = src[r];
        }
    }
}

bool BuildCOneFromLinkage(const Config& config,
                        const ColMajorMatrix<float>& X_rot,
                        const ColMajorMatrix<float>& R_full,
                        const LinkageStructure& linkage,
                        const ColMajorMatrix<FullCode>& B,
                        const ColMajorMatrix<float>& a,
                        const CodebookPack& C_root,
                        CodebookPack* C_one,
                        ColMajorMatrix<float>* R_in) {
    // Julia reference: build R_in + update_codebooks_ls_extend for C_one
    // (train_quantizer.jl:204-227).
    if (!C_one || !R_in) {
        return false;
    }
    const int d = X_rot.rows;
    const int n = X_rot.cols;
    const int m = config.model.m;
    if (d <= 0 || n <= 0 || m <= 0) {
        return false;
    }
    if (B.rows != m || a.rows != m || B.cols != n || a.cols != n) {
        return false;
    }

    R_in->rows = d;
    R_in->cols = n;
    R_in->data.assign(static_cast<std::size_t>(d) * n, 0.0f);

    // Only depth>0 nodes contribute to C_one update; root nodes are skipped.
    std::vector<std::pair<int, int>> one_pairs;
    one_pairs.reserve(static_cast<std::size_t>(n));

    for (const auto& cluster : linkage.clusters) {
        const int csize = static_cast<int>(cluster.indices.size());
        const int n_real = (cluster.n_real > 0) ? std::min(cluster.n_real, csize) : csize;
        if (n_real <= 0 || cluster.depth_offsets.size() < 2) {
            continue;
        }
        const int begin = cluster.depth_offsets[1];  // depth>0 starts after depth-0 block
        for (int local = begin; local < n_real; ++local) {
            const int g = cluster.indices[static_cast<std::size_t>(local)];
            const int p_local_1b = cluster.parent_local[static_cast<std::size_t>(local)];
            const int p_idx_local = static_cast<int>(p_local_1b) - 1;
            const int pg = cluster.indices[static_cast<std::size_t>(p_idx_local)];
            one_pairs.emplace_back(static_cast<int>(g), pg);
        }
    }

    auto h_vec_one = config.model.h_vec;
    if (!h_vec_one.empty()) {
        h_vec_one[0] = config.model.h0_one;
    }
    C_one->d = d;
    C_one->h_vec = h_vec_one;
    if (one_pairs.empty()) {
        LogWarn("Linkage training produced no depth>0 samples; C_one falls back to C_root.");
        C_one->books = C_root.books;
        if (!C_one->books.empty() && C_one->books[0].cols != config.model.h0_one) {
            C_one->books[0] = ColMajorMatrix<float>(d, std::max(1, config.model.h0_one));
        }
        return true;
    }

    const int h0_one = std::max(1, config.model.h0_one);
    const int h0_root = (!C_root.books.empty()) ? C_root.books.front().cols : 0;
    std::vector<std::uint16_t> root_to_one;
    if (h0_root > 0 && h0_one > 0 && h0_one != h0_root) {
        root_to_one.assign(static_cast<std::size_t>(h0_root), 0);
        {
            ScopedOmpThreads omp_scope(1);  // keep mapping stable across OMP settings
            KmeansConfig kcfg;
            kcfg.max_iters = 30;
            kcfg.tol = 1e-6f;
            kcfg.init_method = "kmeans++";
            kcfg.init_samples = std::max(1, h0_root);
            kcfg.initial_weight = 1.0f;
            kcfg.min_weight = 1.0f;
            kcfg.outlier_quantile = 1.0f;
            kcfg.cost_threshold = 1e9f;
            kcfg.annealing_factor = 0.0f;
            kcfg.warmup_iters = 0;
            std::mt19937 rng(static_cast<std::mt19937::result_type>(config.train.seed ^ 0xC001D00Du));
            const KmeansResult km = SphericalKmeans(C_root.books.front(), h0_one, kcfg, &rng,
                                                    /*stream_kernels=*/nullptr,
                                                    /*profile_timing=*/false,
                                                    /*timing=*/nullptr);
            for (int r = 0; r < h0_root; ++r) {
                root_to_one[static_cast<std::size_t>(r)] =
                    static_cast<std::uint16_t>(km.assignments[static_cast<std::size_t>(r)]);
            }
        }
    }

    ColMajorMatrix<float> R_in_sub(d, static_cast<int>(one_pairs.size()));
    ColMajorMatrix<FullCode> B_sub(m, static_cast<int>(one_pairs.size()));
    ColMajorMatrix<float> a_sub(m, static_cast<int>(one_pairs.size()));

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < static_cast<int>(one_pairs.size()); ++j) {
        const int gi = one_pairs[static_cast<std::size_t>(j)].first;
        const float* x = X_rot.Col(gi);
        float* rdst = R_in_sub.Col(j);

        const int p_global = one_pairs[static_cast<std::size_t>(j)].second;
        if (p_global < 0) {
            #pragma omp simd
            for (int r = 0; r < d; ++r) {
                rdst[r] = x[r];
            }
        } else {
            const float* rp = R_full.Col(p_global);
            #pragma omp simd
            for (int r = 0; r < d; ++r) {
                rdst[r] = x[r] - rp[r];
            }
        }

        std::memcpy(B_sub.Col(j), B.Col(gi), sizeof(FullCode) * static_cast<std::size_t>(m));
        std::memcpy(a_sub.Col(j), a.Col(gi), sizeof(float) * static_cast<std::size_t>(m));
        std::memcpy(R_in->Col(gi), rdst, sizeof(float) * static_cast<std::size_t>(d));

        if (!root_to_one.empty()) {
            const int root_id = static_cast<int>(B(0, gi));
            B_sub(0, j) = static_cast<FullCode>(root_to_one[static_cast<std::size_t>(root_id)]);
        }
    }
    C_one->books = UpdateCodebooksLS(R_in_sub, B_sub, a_sub, h_vec_one);
    return true;
}

std::string FormatFloat(float value, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

bool UpdateOpqRotation(const ColMajorMatrix<float>& Zhat,
                       const ColMajorMatrix<float>& X,
                       ColMajorMatrix<float>* R) {
    const int d = X.rows;
    ColMajorMatrix<float> M(d, d);
    {
        ScopedBlasThreads blas_scope(OmpMaxThreads());
        Gemm(false, true, 1.0f, Zhat, X, 0.0f, &M);
    }

    ColMajorMatrix<float> U(d, d);
    ColMajorMatrix<float> VT(d, d);
    std::vector<float> S(d, 0.0f);
    ColMajorMatrix<float> Mcopy = M;

    int info = 0;
    {
        ScopedBlasThreads blas_scope(1);
        info = lapack::Sgesvd('S', 'S', d, d,
                              Mcopy.data.data(), d, S.data(),
                              U.data.data(), d, VT.data.data(), d);
    }
    if (info != 0) {
        return false;
    }

    R->rows = d;
    R->cols = d;
    R->data.assign(static_cast<std::size_t>(d) * static_cast<std::size_t>(d), 0.0f);
    for (int c = 0; c < d; ++c) {
        float* dst = R->Col(c);
        for (int r = 0; r < d; ++r) {
            float acc = 0.0f;
            for (int k = 0; k < d; ++k) {
                acc += U(r, k) * VT(k, c);
            }
            dst[r] = acc;
        }
    }
    return true;
}
