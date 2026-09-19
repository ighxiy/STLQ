// Two-codebook linkage builder: inner-to-outer strategy (Julia: linkage_from_inner_to_outer_two_codebook_ils!).
// Baseline version: no virtual-specific masking logic.
template <bool UseIls>
void LinkageFromInnerToOuterTwoCodebookImplCore(const LinkageBuildConfig& linkage_cfg,
                                              const HnswConfig& hnsw_cfg,
                                              const ColMajorMatrix<float>& X,
                                              std::vector<std::vector<int>>& clusters,
                                              const CodebookPack& C_root,
                                              const Precomp& pre_root,
                                              const CodebookPack& C_one,
                                              const Precomp& pre_one,
                                              ColMajorMatrix<FullCode>* B,
                                              ColMajorMatrix<float>* a,
                                              LinkageStructure* linkage,
                                              ColMajorMatrix<float>* R_full_out,
                                              int ils_rounds,
                                              int ils_perturb_layers) {
    // Julia reference: linkage_from_inner_to_outer_two_codebook_ils!
    // (linkage_quantizer.jl:1131-1668).
    const int d = X.rows;
    const int n = X.cols;
    const int m = pre_one.m;
    const int H = pre_one.H;
    const int h0 = C_root.books.front().cols;

    const int num_layers = std::max(1, linkage_cfg.num_layers);
    const int knn_k = std::max(1, linkage_cfg.knn_k);
    const int depth_k = std::max(1, linkage_cfg.depth_k);
    const int max_depth = std::max(1, linkage_cfg.max_depth);
    const int icm_round = std::max(0, linkage_cfg.icm_round);
    const float root_percentile = static_cast<float>(linkage_cfg.root_percentile);
    const int ils_k = std::min(ils_perturb_layers, m);
    const bool causal_single_parent =
        IsCausalSingleParentPolicy(linkage_cfg.reference_policy);
    (void)ils_rounds;

    ColMajorMatrix<float> G_one_root;
    {
        ScopedBlasThreads blas_scope(std::max(1, omp_get_max_threads()));
        G_one_root = BuildOneRootCrossGram(pre_one, pre_root);
    }

    // Julia reference: compute_cosine_angles + partition_by_cosine_percentiles
    // (linkage_encode_helper.jl:294-452, linkage_quantizer.jl:1191-1204).
    std::vector<float> cos_angles;
    ComputeCosineAngles(X, C_root.books.front(), clusters, &cos_angles);
    std::vector<std::vector<std::vector<int>>> layers(static_cast<std::size_t>(h0));
    std::vector<int> layer_of_point(static_cast<std::size_t>(n), 0);

    for (int code = 0; code < h0; ++code) {
        const auto& idxs = clusters[static_cast<std::size_t>(code)];
        if (idxs.empty()) {
            continue;
        }
        std::vector<std::vector<int>> layer_local;
        PartitionClusterLayers(cos_angles, idxs, num_layers, root_percentile, &layer_local);
        layers[static_cast<std::size_t>(code)] = std::move(layer_local);
        for (int layer = 0; layer < num_layers; ++layer) {
            for (int local_idx : layers[static_cast<std::size_t>(code)][static_cast<std::size_t>(layer)]) {
                int g = idxs[static_cast<std::size_t>(local_idx)];
                layer_of_point[static_cast<std::size_t>(g)] = layer;
            }
        }
    }

    // Julia reference: compute_reconstructed + all_errors!
    // (utils.jl:46-70, linkage_quantizer.jl:1206-1212).
    ColMajorMatrix<float> R_full = ReconstructAll(C_root.books, *B, *a);
    std::vector<float> base_mse;
    SquaredErrorsFromRecon(X, R_full, &base_mse);

    int num_threads = omp_get_max_threads();
    std::vector<LinkageTls> tls_pool;
    tls_pool.reserve(static_cast<std::size_t>(num_threads));
    for (int t = 0; t < num_threads; ++t) {
        tls_pool.push_back(MakeLinkageTls(d, m, H));
    }

    std::vector<LinkageStructure::Cluster> cluster_out(static_cast<std::size_t>(h0));
    std::vector<uint8_t> cluster_used(static_cast<std::size_t>(h0), 0);

    int ls_failures = 0;
    const int clusters_total = static_cast<int>(std::count_if(
        clusters.begin(), clusters.end(),
        [](const std::vector<int>& c) { return !c.empty(); }));
    std::atomic<int> clusters_done{0};

    #pragma omp parallel for schedule(dynamic) reduction(+:ls_failures)
    for (int code = 0; code < h0; ++code) {
        auto& cidxs = clusters[static_cast<std::size_t>(code)];
        if (cidxs.empty()) {
            continue;
        }

        LinkageTls& tls = tls_pool[static_cast<std::size_t>(omp_get_thread_num())];
        std::mt19937 rng(static_cast<std::mt19937::result_type>(linkage_cfg.seed +
                                                                997 * code));

        const int csize = static_cast<int>(cidxs.size());
        std::vector<int> parent_local(static_cast<std::size_t>(csize), 0);
        std::vector<int> depth_local(static_cast<std::size_t>(csize), 0);
        std::vector<FullCode*> B_cols(static_cast<std::size_t>(csize), nullptr);
        std::vector<float*> a_cols(static_cast<std::size_t>(csize), nullptr);
        for (int local = 0; local < csize; ++local) {
            const int g = cidxs[static_cast<std::size_t>(local)];
            B_cols[static_cast<std::size_t>(local)] = B->Col(static_cast<int>(g));
            a_cols[static_cast<std::size_t>(local)] = a->Col(static_cast<int>(g));
        }
        const int candidate_multiplier = std::max(1, hnsw_cfg.candidate_multiplier_good);

        ClusterHnswIndex hnsw_index;
        const int desired_ef_construction = std::max(1, depth_k * candidate_multiplier * knn_k);
        const int ef_construction = std::max(1, std::min(desired_ef_construction, csize));
        hnsw_index.Build(X, cidxs, hnsw_cfg, ef_construction);

        const auto& layers_local = layers[static_cast<std::size_t>(code)];

        // Julia reference: per-cluster, per-layer processing (linkage_quantizer.jl:1250-1665).
        for (int layer_idx = 1; layer_idx < num_layers; ++layer_idx) {
            const auto& current_layer = layers_local[static_cast<std::size_t>(layer_idx)];
            if (current_layer.empty()) {
                continue;
            }

            tls.order_idx.resize(current_layer.size());
            std::iota(tls.order_idx.begin(), tls.order_idx.end(), 0);
            std::shuffle(tls.order_idx.begin(), tls.order_idx.end(), rng);

            tls.processed.resize(current_layer.size());
            std::fill(tls.processed.begin(), tls.processed.end(), 0);

            int inner_lo = std::max(0, layer_idx - depth_k);
            int inner_hi = layer_idx - 1;

            tls.local_pos.resize(static_cast<std::size_t>(csize));
            std::fill(tls.local_pos.begin(), tls.local_pos.end(), -1);
            for (std::size_t pos = 0; pos < current_layer.size(); ++pos) {
                int local_idx = current_layer[pos];
                tls.local_pos[static_cast<std::size_t>(local_idx)] = static_cast<int>(pos);
            }

            for (int pos_in_layer : tls.order_idx) {
                int local_idx = current_layer[static_cast<std::size_t>(pos_in_layer)];
                int i_global = cidxs[static_cast<std::size_t>(local_idx)];
                const float* xi = X.Col(static_cast<int>(i_global));

                QueryCandidatesHnsw(hnsw_index,
                                    xi,
                                    cidxs,
                                    local_idx,
                                    layer_of_point,
                                    tls.local_pos,
                                    tls.processed,
                                    inner_lo,
                                    inner_hi,
                                    layer_idx,
                                    num_layers,
                                    knn_k,
                                    candidate_multiplier,
                                    &tls.candidates);

                if (causal_single_parent) {
                    SelectOneCausalReference(
                        linkage_cfg.reference_policy,
                        CausalReferenceSelectionKey(linkage_cfg.seed, code, local_idx),
                        xi, X, cidxs, depth_local, max_depth,
                        &tls.candidates);
                }


                tls.processed[static_cast<std::size_t>(pos_in_layer)] = 1;
                if (tls.candidates.empty()) {
                    continue;
                }

                // Julia reference: xC_one_base precompute (linkage_quantizer.jl:1350-1354).
                if (icm_round > 0) {
                    ComputeXcBase(pre_one, xi, &tls.xC_base_buf);
                }

                float bestE2 = base_mse[static_cast<std::size_t>(i_global)];
                int best_parent_local = -1;

                for (int p_local : tls.candidates) {
                    int p_global = cidxs[static_cast<std::size_t>(p_local)];
                    int new_depth = depth_local[static_cast<std::size_t>(p_local)] + 1;
                    if (new_depth > max_depth) {
                        continue;
                    }

                    const float* rp = R_full.Col(static_cast<int>(p_global));
                    #pragma omp simd
                    for (int r = 0; r < d; ++r) {
                        tls.re[static_cast<std::size_t>(r)] = xi[r] - rp[r];
                    }

                    // Julia reference: quantize_with_kmeans_abs_nonormal_single_extend_allcodes_fast!
                    // (greedy_quantizer.jl:4-58), using C_one.
                    QuantizeWithKmeansAbsNoNormalSingle(pre_one, tls.re,
                                                        &tls.B_init, &tls.a_init,
                                                        &tls.rC_buf, &tls.available);

                    // Julia reference: build_xC_one_for_parent! + dynamic_icm_encoding_single_for_residual!
                    // (linkage_encode_helper.jl:610-653).
                    if (icm_round > 0) {
                        BuildXcOneForParentSingleCols(pre_root, pre_one, G_one_root,
                                                      tls.xC_base_buf,
                                                      B_cols.data(), a_cols.data(),
                                                      parent_local, depth_local,
                                                      p_local,
                                                      &tls.xC_buf,
                                                      code);
                        if (!SolveLeastSquaresVector(pre_one, tls.xC_buf, tls.B_init,
                                                     &tls.A, &tls.b, &tls.work,
                                                     tls.a_init.data())) {
                            ++ls_failures;
                            continue;
                        }
                        DynamicIcmEncodingSingleForResidual(pre_one, tls.re, tls.xC_buf,
                                                            icm_round,
                                                            &tls.B_init, &tls.a_init,
                                                            &tls.a_bak, &tls.rC_buf,
                                                            &tls.A, &tls.b, &tls.work,
                                                            &rng, &ls_failures);
                    } else {
                        // Julia reference: optimized_least_squares_single! on re (linkage_quantizer.jl:1373-1379).
                        // When icm_round==0 we don't need the full xC_one table; only the m selected dot-products.
                        if (!SolveLeastSquaresVectorFromResidual(pre_one, tls.re, tls.B_init,
                                                                 &tls.A, &tls.b, &tls.work,
                                                                 tls.a_init.data())) {
                            ++ls_failures;
                            continue;
                        }
                    }

                    ReconstructNode(C_one.books, tls.B_init, tls.a_init, &tls.R_self);
                    float E2_best = 0.0f;
                    const float* re_ptr = tls.re.data();
                    const float* rself_ptr = tls.R_self.data();
                    #pragma omp simd reduction(+:E2_best)
                    for (int r = 0; r < d; ++r) {
                        float diff = re_ptr[r] - rself_ptr[r];
                        E2_best += diff * diff;
                    }

                    // Julia reference: ILS perturbation loop (linkage_quantizer.jl:1416-1482).
                    if constexpr (UseIls) {
                        std::uniform_int_distribution<int> layer_dist(0, m - 1);
                        std::vector<std::uniform_int_distribution<int>> code_dists;
                        code_dists.reserve(static_cast<std::size_t>(m));
                        for (int layer = 0; layer < m; ++layer) {
                            const int h = pre_one.h_vec[layer];
                            code_dists.emplace_back(0, std::max(0, h - 2));
                        }
                        for (int il = 0; il < ils_rounds; ++il) {
                            for (int t = 0; t < ils_k; ++t) {
                                int layer = layer_dist(rng);
                                int h = pre_one.h_vec[layer];
                                int old_code = static_cast<int>(tls.B_init[static_cast<std::size_t>(layer)]);
                                int new_code = old_code;
                                if (h > 1) {
                                    int raw = code_dists[static_cast<std::size_t>(layer)](rng);
                                    new_code = raw;
                                    if (new_code >= old_code) {
                                        new_code += 1;
                                    }
                                }
                                int startf = pre_one.offsets[layer];
                                tls.pert_layers[static_cast<std::size_t>(t)] = layer;
                                tls.new_flat[static_cast<std::size_t>(t)] = startf + new_code;
                            }

                            tls.B_try = tls.B_init;
                            for (int t = 0; t < ils_k; ++t) {
                                int layer = tls.pert_layers[static_cast<std::size_t>(t)];
                                int flat = tls.new_flat[static_cast<std::size_t>(t)];
                                int code = flat - pre_one.offsets[layer];
                                tls.B_try[static_cast<std::size_t>(layer)] = static_cast<FullCode>(code);
                            }

                            tls.a_try = tls.a_init;
                            if (icm_round > 0) {
                                if (!SolveLeastSquaresVector(pre_one, tls.xC_buf, tls.B_try,
                                                             &tls.A, &tls.b, &tls.work,
                                                             tls.a_try.data())) {
                                    ++ls_failures;
                                    continue;
                                }
                                DynamicIcmEncodingSingleForResidual(pre_one, tls.re, tls.xC_buf,
                                                                    icm_round,
                                                                    &tls.B_try, &tls.a_try,
                                                                    &tls.a_bak, &tls.rC_buf,
                                                                    &tls.A, &tls.b, &tls.work,
                                                                    &rng, &ls_failures);
                            } else {
                                if (!SolveLeastSquaresVectorFromResidual(pre_one, tls.re, tls.B_try,
                                                                         &tls.A, &tls.b, &tls.work,
                                                                         tls.a_try.data())) {
                                    ++ls_failures;
                                    continue;
                                }
                            }

                            ReconstructNode(C_one.books, tls.B_try, tls.a_try, &tls.R_self);
                            float E2_try = 0.0f;
                            const float* rself_try = tls.R_self.data();
                            #pragma omp simd reduction(+:E2_try)
                            for (int r = 0; r < d; ++r) {
                                float diff = re_ptr[r] - rself_try[r];
                                E2_try += diff * diff;
                            }

                            if (E2_try < E2_best) {
                                E2_best = E2_try;
                                tls.B_init = tls.B_try;
                                tls.a_init = tls.a_try;
                            }
                        }
                    }

                    if (E2_best < bestE2) {
                        bestE2 = E2_best;
                        best_parent_local = p_local;
                        tls.B_best = tls.B_init;
                        tls.a_best = tls.a_init;
                    }
                }

                // Julia reference: apply best parent + update R_full (linkage_quantizer.jl:1484-1509).
                if (best_parent_local >= 0) {
                    int p_global = cidxs[static_cast<std::size_t>(best_parent_local)];
                    for (int l = 0; l < m; ++l) {
                        (*B)(l, static_cast<int>(i_global)) = tls.B_best[static_cast<std::size_t>(l)];
                        (*a)(l, static_cast<int>(i_global)) = tls.a_best[static_cast<std::size_t>(l)];
                    }
                    parent_local[static_cast<std::size_t>(local_idx)] =
                        best_parent_local + 1;
                    depth_local[static_cast<std::size_t>(local_idx)] =
                        depth_local[static_cast<std::size_t>(best_parent_local)] + 1;

                    ReconstructNode(C_one.books, tls.B_best, tls.a_best, &tls.R_self);
                    const float* rp = R_full.Col(static_cast<int>(p_global));
                    float* rdst = R_full.Col(static_cast<int>(i_global));
                    const float* rself_best = tls.R_self.data();
                    #pragma omp simd
                    for (int r = 0; r < d; ++r) {
                        rdst[r] = rp[r] + rself_best[r];
                    }
                }
            }
        }

        LinkageStructure::Cluster cluster;
        cluster.cluster_id = code;
        cluster.n_real = csize;
        cluster.n_virtual = 0;
        cluster.indices = std::move(cidxs);
        cluster.parent_local = std::move(parent_local);
        FinalizeLinkageClusterDepthOrderInPlaceFromDepth(&cluster, depth_local.data());

        cluster_out[static_cast<std::size_t>(code)] = std::move(cluster);
        cluster_used[static_cast<std::size_t>(code)] = 1;

        int done = clusters_done.fetch_add(1, std::memory_order_relaxed) + 1;
        if (done % 10 == 0 || done == clusters_total) {
            PrintClusterProgress(done, clusters_total);
        }
    }

    linkage->clusters.clear();
    linkage->max_depth = 0;
    for (int code = 0; code < h0; ++code) {
        if (!cluster_used[static_cast<std::size_t>(code)]) {
            continue;
        }
        linkage->max_depth = std::max(linkage->max_depth, static_cast<int>(
            cluster_out[static_cast<std::size_t>(code)].depth_offsets.size()) - 2);
        linkage->clusters.push_back(std::move(cluster_out[static_cast<std::size_t>(code)]));
    }

    if (R_full_out) {
        *R_full_out = std::move(R_full);
    }

    if (ls_failures > 0) {
        LogWarn("Linkage LS failures: " + std::to_string(ls_failures));
    }
}
