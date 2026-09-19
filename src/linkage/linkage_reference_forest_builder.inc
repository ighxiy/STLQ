template <bool UseIls>
bool LinkageTwoCodebookReferenceForestOneCluster(
    const LinkageBuildConfig& linkage_cfg,
    const HnswConfig& hnsw_cfg,
    int forced_root_code,
    const ColMajorMatrix<float>& X,
    const CodebookPack& C_root,
    const Precomp& pre_root,
    const CodebookPack& C_one,
    const Precomp& pre_one,
    const ColMajorMatrix<float>& G_one_root,
    ColMajorMatrix<FullCode>* B,
    ColMajorMatrix<float>* a,
    LinkageStructure::Cluster* out_cluster,
    int* ls_failures_out,
    MseStats* linkaged_mse_out,
    double* linkaged_mse_sum_out,
    ReferenceForest* reference_forest_out,
    std::string* error) {
    if (!B || !a || !out_cluster) {
        if (error) *error = "Reference-forest linkage: null output.";
        return false;
    }
    const int d = X.rows;
    const int n = X.cols;
    const int m = pre_one.m;
    const int H = pre_one.H;
    if (d <= 0 || n < 0 || B->rows != m || B->cols != n ||
        a->rows != m || a->cols != n) {
        if (error) *error = "Reference-forest linkage: input shape mismatch.";
        return false;
    }
    if (pre_one.G.data.empty() || pre_one.C_all.data.empty()) {
        if (error) {
            *error = "Reference-forest linkage currently requires full C_one precomputation; "
                     "the large-root meta-only evaluator is unsupported.";
        }
        return false;
    }

    out_cluster->cluster_id = forced_root_code;
    out_cluster->n_real = n;
    out_cluster->n_virtual = 0;
    out_cluster->indices.resize(static_cast<std::size_t>(n));
    std::iota(out_cluster->indices.begin(), out_cluster->indices.end(), 0);
    out_cluster->parent_local.assign(static_cast<std::size_t>(n), 0);
    out_cluster->depth_offsets.clear();
    if (ls_failures_out) *ls_failures_out = 0;
    if (n == 0) {
        out_cluster->depth_offsets.assign(2, 0);
        if (linkaged_mse_out) *linkaged_mse_out = {};
        if (linkaged_mse_sum_out) *linkaged_mse_sum_out = 0.0;
        if (reference_forest_out) *reference_forest_out = {};
        return true;
    }

    std::vector<int> cluster_indices(static_cast<std::size_t>(n));
    std::iota(cluster_indices.begin(), cluster_indices.end(), 0);
    ReferenceForest forest;
    const std::uint64_t forest_seed =
        static_cast<std::uint64_t>(static_cast<std::uint32_t>(linkage_cfg.seed)) ^
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(forced_root_code)) << 32U);
    if (!BuildReferenceForest(linkage_cfg.reference_policy, X, cluster_indices,
                              hnsw_cfg, linkage_cfg.max_depth, forest_seed,
                              &forest, error)) {
        return false;
    }

    ColMajorMatrix<float> R_full(d, n);
    for (int i = 0; i < n; ++i) {
        float* dst = R_full.Col(i);
        std::fill(dst, dst + d, 0.0f);
        for (int l = 0; l < m; ++l) {
            const int code = (l == 0)
                                 ? forced_root_code
                                 : static_cast<int>((*B)(l, i));
            const float coeff = (*a)(l, i);
            const float* atom = C_root.books[static_cast<std::size_t>(l)].Col(code);
#pragma omp simd
            for (int r = 0; r < d; ++r) dst[r] += coeff * atom[r];
        }
    }
    std::vector<float> base_mse;
    SquaredErrorsFromRecon(X, R_full, &base_mse);

    LinkageTls tls = MakeLinkageTls(d, m, H);
    std::vector<int> parent_local(static_cast<std::size_t>(n), 0);
    std::vector<int> depth_local(static_cast<std::size_t>(n), 0);
    std::vector<FullCode*> B_cols(static_cast<std::size_t>(n), nullptr);
    std::vector<float*> a_cols(static_cast<std::size_t>(n), nullptr);
    for (int i = 0; i < n; ++i) {
        B_cols[static_cast<std::size_t>(i)] = B->Col(i);
        a_cols[static_cast<std::size_t>(i)] = a->Col(i);
    }

    const int icm_round = std::max(0, linkage_cfg.icm_round);
    const int ils_rounds = std::max(0, linkage_cfg.ils_rounds);
    const int ils_k = std::min(std::max(0, linkage_cfg.ils_perturb_layers), m);
    int ls_failures = 0;

    for (int local_idx : forest.order) {
        const int p_local = forest.parent[static_cast<std::size_t>(local_idx)];
        if (p_local < 0) continue;
        const float* xi = X.Col(local_idx);
        const float* rp = R_full.Col(p_local);
#pragma omp simd
        for (int r = 0; r < d; ++r) {
            tls.re[static_cast<std::size_t>(r)] = xi[r] - rp[r];
        }

        QuantizeWithKmeansAbsNoNormalSingle(pre_one, tls.re,
                                            &tls.B_init, &tls.a_init,
                                            &tls.rC_buf, &tls.available);
        if (icm_round > 0) {
            ComputeXcBase(pre_one, xi, &tls.xC_base_buf);
            BuildXcOneForParentSingleCols(pre_root, pre_one, G_one_root,
                                          tls.xC_base_buf,
                                          B_cols.data(), a_cols.data(),
                                          parent_local, depth_local,
                                          p_local, &tls.xC_buf,
                                          forced_root_code);
            if (!SolveLeastSquaresVector(pre_one, tls.xC_buf, tls.B_init,
                                         &tls.A, &tls.b, &tls.work,
                                         tls.a_init.data())) {
                ++ls_failures;
                continue;
            }
        } else if (!SolveLeastSquaresVectorFromResidual(pre_one, tls.re, tls.B_init,
                                                        &tls.A, &tls.b, &tls.work,
                                                        tls.a_init.data())) {
            ++ls_failures;
            continue;
        }

        std::mt19937 rng(static_cast<std::mt19937::result_type>(
            static_cast<std::uint32_t>(linkage_cfg.seed) ^
            static_cast<std::uint32_t>(forced_root_code * 2654435761U) ^
            static_cast<std::uint32_t>(local_idx * 2246822519U)));
        if (icm_round > 0) {
            DynamicIcmEncodingSingleForResidual(pre_one, tls.re, tls.xC_buf,
                                                icm_round,
                                                &tls.B_init, &tls.a_init,
                                                &tls.a_bak, &tls.rC_buf,
                                                &tls.A, &tls.b, &tls.work,
                                                &rng, &ls_failures);
        }

        ReconstructNode(C_one.books, tls.B_init, tls.a_init, &tls.R_self);
        float best_error = 0.0f;
#pragma omp simd reduction(+:best_error)
        for (int r = 0; r < d; ++r) {
            const float diff = tls.re[static_cast<std::size_t>(r)] -
                               tls.R_self[static_cast<std::size_t>(r)];
            best_error += diff * diff;
        }

        if constexpr (UseIls) {
            std::uniform_int_distribution<int> layer_dist(0, m - 1);
            std::vector<std::uniform_int_distribution<int>> code_dists;
            code_dists.reserve(static_cast<std::size_t>(m));
            for (int layer = 0; layer < m; ++layer) {
                code_dists.emplace_back(0, std::max(0, pre_one.h_vec[layer] - 2));
            }
            for (int il = 0; il < ils_rounds; ++il) {
                tls.B_try = tls.B_init;
                for (int t = 0; t < ils_k; ++t) {
                    const int layer = layer_dist(rng);
                    const int h = pre_one.h_vec[layer];
                    const int old_code = static_cast<int>(tls.B_try[static_cast<std::size_t>(layer)]);
                    int code = old_code;
                    if (h > 1) {
                        code = code_dists[static_cast<std::size_t>(layer)](rng);
                        if (code >= old_code) ++code;
                    }
                    tls.B_try[static_cast<std::size_t>(layer)] = static_cast<FullCode>(code);
                }
                tls.a_try = tls.a_init;
                const bool solved =
                    icm_round > 0
                        ? SolveLeastSquaresVector(pre_one, tls.xC_buf, tls.B_try,
                                                  &tls.A, &tls.b, &tls.work,
                                                  tls.a_try.data())
                        : SolveLeastSquaresVectorFromResidual(pre_one, tls.re, tls.B_try,
                                                              &tls.A, &tls.b, &tls.work,
                                                              tls.a_try.data());
                if (!solved) {
                    ++ls_failures;
                    continue;
                }
                if (icm_round > 0) {
                    DynamicIcmEncodingSingleForResidual(pre_one, tls.re, tls.xC_buf,
                                                        icm_round,
                                                        &tls.B_try, &tls.a_try,
                                                        &tls.a_bak, &tls.rC_buf,
                                                        &tls.A, &tls.b, &tls.work,
                                                        &rng, &ls_failures);
                }
                ReconstructNode(C_one.books, tls.B_try, tls.a_try, &tls.R_self);
                float trial_error = 0.0f;
#pragma omp simd reduction(+:trial_error)
                for (int r = 0; r < d; ++r) {
                    const float diff = tls.re[static_cast<std::size_t>(r)] -
                                       tls.R_self[static_cast<std::size_t>(r)];
                    trial_error += diff * diff;
                }
                if (trial_error < best_error) {
                    best_error = trial_error;
                    tls.B_init = tls.B_try;
                    tls.a_init = tls.a_try;
                }
            }
        }

        // A preferred reference is advisory: retain it only when the complete
        // reconstructed node strictly improves over the basic root encoding.
        if (!(best_error < base_mse[static_cast<std::size_t>(local_idx)])) continue;

        for (int l = 0; l < m; ++l) {
            (*B)(l, local_idx) = tls.B_init[static_cast<std::size_t>(l)];
            (*a)(l, local_idx) = tls.a_init[static_cast<std::size_t>(l)];
        }
        parent_local[static_cast<std::size_t>(local_idx)] = p_local + 1;
        depth_local[static_cast<std::size_t>(local_idx)] =
            depth_local[static_cast<std::size_t>(p_local)] + 1;
        ReconstructNode(C_one.books, tls.B_init, tls.a_init, &tls.R_self);
        float* dst = R_full.Col(local_idx);
#pragma omp simd
        for (int r = 0; r < d; ++r) {
            dst[r] = rp[r] + tls.R_self[static_cast<std::size_t>(r)];
        }
    }

    out_cluster->parent_local = std::move(parent_local);
    FinalizeLinkageClusterDepthOrderInPlaceFromDepth(out_cluster, depth_local.data());

    if (linkaged_mse_out || linkaged_mse_sum_out) {
        double sum = 0.0;
        double min_error = std::numeric_limits<double>::infinity();
        double max_error = 0.0;
        for (int i = 0; i < n; ++i) {
            const float* x = X.Col(i);
            const float* recon = R_full.Col(i);
            double e = 0.0;
#pragma omp simd reduction(+:e)
            for (int r = 0; r < d; ++r) {
                const double diff = static_cast<double>(x[r]) - static_cast<double>(recon[r]);
                e += diff * diff;
            }
            sum += e;
            min_error = std::min(min_error, e);
            max_error = std::max(max_error, e);
        }
        if (linkaged_mse_out) {
            linkaged_mse_out->mean = static_cast<float>(sum / static_cast<double>(n));
            linkaged_mse_out->min = static_cast<float>(min_error);
            linkaged_mse_out->max = static_cast<float>(max_error);
        }
        if (linkaged_mse_sum_out) *linkaged_mse_sum_out = sum;
    }
    if (ls_failures_out) *ls_failures_out = ls_failures;
    if (reference_forest_out) *reference_forest_out = std::move(forest);
    return true;
}
