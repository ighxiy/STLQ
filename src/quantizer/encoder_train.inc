namespace virtual_train {

void PrintClusterProgress(int done, int total) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> guard(mutex);
    std::cerr << "\rprocessed " << done << "/" << total << "   ";
    if (done >= total) {
        std::cerr << "\n";
    }
    std::cerr.flush();
}

std::vector<int> ExtractFirstLayerCodes(const ColMajorMatrix<FullCode>& B) {
    const int n = B.cols;
    std::vector<int> out(static_cast<std::size_t>(n), 0);
    for (int i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] = static_cast<int>(B(0, i));
    }
    return out;
}



std::vector<float> ComputePerSampleSquaredErrors(const ColMajorMatrix<float>& X,
                                                 const ColMajorMatrix<float>& R_full) {
    const int d = X.rows;
    const int n = X.cols;
    std::vector<float> out(static_cast<std::size_t>(n), 0.0f);
    if (d <= 0 || n <= 0 || R_full.rows != d || R_full.cols != n) {
        return out;
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const float* x = X.Col(i);
        const float* r = R_full.Col(i);
        float s = 0.0f;
        #pragma omp simd reduction(+:s)
        for (int j = 0; j < d; ++j) {
            float diff = x[j] - r[j];
            s += diff * diff;
        }
        out[static_cast<std::size_t>(i)] = s;
    }
    return out;
}

std::vector<bool> BuildBadClusterMaskQuantile(const std::vector<int>& cluster_id,
                                              const std::vector<float>& mse_vec,
                                              int h1,
                                              double good_fraction,
                                              double* out_baseline) {
    if (out_baseline) {
        *out_baseline = 0.0;
    }
    const int n = static_cast<int>(cluster_id.size());
    std::vector<bool> is_bad(static_cast<std::size_t>(std::max(0, h1)), false);
    if (n <= 0 || h1 <= 0 || static_cast<int>(mse_vec.size()) != n) {
        return is_bad;
    }

    std::vector<double> sum(static_cast<std::size_t>(h1), 0.0);
    std::vector<int> cnt(static_cast<std::size_t>(h1), 0);
    for (int i = 0; i < n; ++i) {
        int c = cluster_id[static_cast<std::size_t>(i)];
        // if (static_cast<unsigned>(c) >= static_cast<unsigned>(h1)) {
        //     continue;
        // }
        sum[static_cast<std::size_t>(c)] += static_cast<double>(mse_vec[static_cast<std::size_t>(i)]);
        cnt[static_cast<std::size_t>(c)] += 1;
    }

    std::vector<double> avgs;
    avgs.reserve(static_cast<std::size_t>(h1));
    for (int c = 0; c < h1; ++c) {
        if (cnt[static_cast<std::size_t>(c)] > 0) {
            avgs.push_back(sum[static_cast<std::size_t>(c)] /
                           static_cast<double>(cnt[static_cast<std::size_t>(c)]));
        }
    }
    if (avgs.empty()) {
        return is_bad;
    }
    std::sort(avgs.begin(), avgs.end());

    const double q = std::max(0.0, std::min(1.0, good_fraction));
    double baseline = avgs.back();
    if (q <= 0.0) {
        baseline = avgs.front();
    } else if (q >= 1.0) {
        baseline = avgs.back();
    } else {
        const int n_used = static_cast<int>(avgs.size());
        int pos = static_cast<int>(std::llround(q * static_cast<double>(n_used)));
        pos = std::max(1, std::min(pos, n_used));
        baseline = avgs[static_cast<std::size_t>(pos - 1)];
    }
    if (out_baseline) {
        *out_baseline = baseline;
    }

    for (int c = 0; c < h1; ++c) {
        if (cnt[static_cast<std::size_t>(c)] <= 0) {
            continue;
        }
        double avg = sum[static_cast<std::size_t>(c)] /
                     static_cast<double>(cnt[static_cast<std::size_t>(c)]);
        if (avg > baseline) {
            is_bad[static_cast<std::size_t>(c)] = true;
        }
    }
    return is_bad;
}


bool UpdateCOneWeighted(const Config& config,
                        const ColMajorMatrix<float>& X_rot,
                        const ColMajorMatrix<float>& R_full,
                        const ColMajorMatrix<FullCode>& B,
                        const ColMajorMatrix<float>& a,
                        const CodebookPack& C_root,
                        const LinkageStructure& linkage,
                        const std::vector<int>& cluster_id,
                        const std::vector<bool>& is_bad_cluster,
                        CodebookPack* C_one,
                        std::string* error) {
    if (!C_one) {
        if (error) {
            *error = "UpdateCOneWeighted: null output.";
        }
        return false;
    }
    if (cluster_id.size() != static_cast<std::size_t>(X_rot.cols)) {
        if (error) {
            *error = "UpdateCOneWeighted: length mismatch.";
        }
        return false;
    }

    const int d = X_rot.rows;
    const int n = X_rot.cols;
    const int m = config.model.m;
    if (d <= 0 || n <= 0 || m <= 0) {
        if (error) {
            *error = "UpdateCOneWeighted: invalid shapes.";
        }
        return false;
    }
    if (B.rows != m || a.rows != m || B.cols != n || a.cols != n) {
        if (error) {
            *error = "UpdateCOneWeighted: B/a shape mismatch.";
        }
        return false;
    }

    // Only depth>0 nodes contribute to C_one update; root nodes are skipped.
    std::vector<std::pair<int, int>> one_pairs;
    one_pairs.reserve(static_cast<std::size_t>(n));

    for (const auto& cluster : linkage.clusters) {
        const int csize = static_cast<int>(cluster.indices.size());
        const int n_real = (cluster.n_real > 0) ? std::min(cluster.n_real, csize) : csize;
        if (n_real <= 0 || cluster.depth_offsets.size() < 2) {
            continue;
        }
        const int begin = cluster.depth_offsets[1];
        const int real_base = std::max(0, cluster.n_virtual);
        for (int real_idx = begin; real_idx < n_real; ++real_idx) {
            const Index g = cluster.indices[static_cast<std::size_t>(real_idx)];
            const int local = real_base + real_idx;
            const int p_local_1b = cluster.parent_local[static_cast<std::size_t>(local)];
            const int p_local = static_cast<int>(p_local_1b) - 1;
            if (p_local < real_base) {
                // Parent is a virtual root (no global id); skip for weighted update.
                continue;
            }
            const int p_real = p_local - real_base;
            if (p_real < 0 || p_real >= n_real) {
                LogWarn("UpdateCOneWeighted: cluster parent index p_real=" + std::to_string(p_real) +
                        " out of range [0," + std::to_string(n_real) + ") at real_idx=" +
                        std::to_string(real_idx) + "; linkage data corrupt.");
                continue;
            }
            const Index pg = cluster.indices[static_cast<std::size_t>(p_real)];
            one_pairs.emplace_back(static_cast<int>(g), static_cast<int>(pg));
        }
    }

    C_one->d = d;
    auto h_vec_one = config.model.h_vec;
    if (!h_vec_one.empty()) {
        h_vec_one[0] = config.model.h0_one;
    }
    C_one->h_vec = h_vec_one;
    if (one_pairs.empty()) {
        if (error) {
            *error = "UpdateCOneWeighted: no depth>0 samples for C_one update.";
        }
        return false;
    }

    const int h0_one = std::max(1, config.model.h0_one);
    const int h0_root = (!C_root.books.empty()) ? C_root.books.front().cols : 0;
    if (h0_root <= 0) {
        if (error) {
            *error = "UpdateCOneWeighted: empty C_root[0] for root->one mapping.";
        }
        return false;
    }
    if (h0_one > h0_root) {
        if (error) {
            *error = "UpdateCOneWeighted: h0_one must be <= h0_root.";
        }
        return false;
    }

    // Geometry-aware root_id -> one_id mapping: only needed when h0_one < h0_root.
    // In the common non-large case (h0_one == h0_root), keep CPU semantics by using B0 as-is.
    std::vector<std::uint16_t> root_to_one;
    if (h0_one < h0_root) {
        root_to_one.assign(static_cast<std::size_t>(h0_root), 0);
        ScopedOmpThreads omp_scope(1);  // determinism: keep mapping independent of OMP threads
        KmeansConfig kcfg;
        kcfg.max_iters = 30;
        kcfg.tol = 1e-6f;
        kcfg.init_method = "kmeans++";
        kcfg.init_samples = std::max(1, h0_root);
        // Disable adaptive weights for stable mapping.
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
        if (static_cast<int>(km.assignments.size()) != h0_root) {
            if (error) {
                *error = "UpdateCOneWeighted: unexpected root kmeans assignment size.";
            }
            return false;
        }
        for (int r = 0; r < h0_root; ++r) {
            root_to_one[static_cast<std::size_t>(r)] =
                static_cast<std::uint16_t>(km.assignments[static_cast<std::size_t>(r)]);
        }
    }

    // std::sort(one_pairs.begin(), one_pairs.end(),
    //           [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    std::vector<int> idx;
    std::vector<const float*> parent_recon;
    std::vector<float> weights;
    idx.resize(one_pairs.size());
    parent_recon.resize(one_pairs.size(), nullptr);
    weights.resize(one_pairs.size(), 1.0f);
    std::vector<std::uint16_t> b0_one;
    const std::uint16_t* b0_one_ptr = nullptr;
    if (h0_one < h0_root) {
        b0_one.resize(one_pairs.size(), 0);
        b0_one_ptr = b0_one.data();
    }

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < static_cast<int>(one_pairs.size()); ++j) {
        const int gi = one_pairs[static_cast<std::size_t>(j)].first;
        const int p_global = one_pairs[static_cast<std::size_t>(j)].second;
        const float* rp = R_full.Col(p_global);

        // Apply weights: bad clusters get alpha_bad, good clusters weight=1.
        int cid = cluster_id[static_cast<std::size_t>(gi)];
        float ww = 1.0f;
        if (is_bad_cluster[static_cast<std::size_t>(cid)]) {
            ww = config.virtual_cfg.alpha_bad;
        }
        idx[static_cast<std::size_t>(j)] = gi;
        parent_recon[static_cast<std::size_t>(j)] = rp;
        weights[static_cast<std::size_t>(j)] = ww;
        if (b0_one_ptr) {
            b0_one[static_cast<std::size_t>(j)] = root_to_one[static_cast<std::size_t>(cid)];
        }
    }

    C_one->books = UpdateCodebooksLSFromResidualPtrs(X_rot, B, a,
                                                     idx, parent_recon, weights,
                                                     h_vec_one,
                                                     b0_one_ptr);
    return true;
}



bool UpdateCOneFromLinkagesVirtualNoCopy(const Config& config,
                                       const ColMajorMatrix<float>& X_rot,
                                       const ColMajorMatrix<FullCode>& B,
                                       const ColMajorMatrix<float>& a,
                                       const LinkageStructure& linkage_good,
                                       const ColMajorMatrix<float>& R_full_shared,
                                       const std::vector<BadClusterCache>& bad_cache,
                                       CodebookPack* C_one,
                                       std::string* error) {
    if (!C_one) {
        if (error) {
            *error = "UpdateCOneFromLinkagesVirtualNoCopy: null output.";
        }
        return false;
    }
    const int d = X_rot.rows;
    const int n = X_rot.cols;
    const int m = config.model.m;
    if (d <= 0 || n <= 0 || m <= 0) {
        if (error) {
            *error = "UpdateCOneFromLinkagesVirtualNoCopy: invalid shapes.";
        }
        return false;
    }

    std::vector<int> idx;
    std::vector<const float*> parent_recon;
    std::vector<float> weights;
    idx.reserve(static_cast<std::size_t>(n));
    parent_recon.reserve(static_cast<std::size_t>(n));
    weights.reserve(static_cast<std::size_t>(n));

    // Bad clusters (weight alpha_bad), real nodes only, parents may be virtual.
    for (const auto& cache : bad_cache) {
        const int n_aug = cache.n_real + cache.n_virt;
        if (cache.cluster_id < 0 || cache.n_real <= 0 || n_aug <= 0 || cache.linkage_aug.clusters.empty()) {
            continue;
        }
        const auto& cl = cache.linkage_aug.clusters.front();
        const int csize = static_cast<int>(cl.indices.size());
        const int n_real = (cl.n_real > 0) ? std::min(cl.n_real, csize) : csize;
        if (n_real <= 0 || cl.depth_offsets.size() < 2) {
            continue;
        }
        const int real_base = std::max(0, cl.n_virtual);
        for (int pos = cl.depth_offsets[1]; pos < n_real; ++pos) {
            const int li = cl.indices[static_cast<std::size_t>(pos)];
            const int gi = cache.cols_global[static_cast<std::size_t>(li)];

            const int local = real_base + pos;
            const int ppos = static_cast<int>(cl.parent_local[static_cast<std::size_t>(local)]) - 1;
            const float* rp = nullptr;
            if (ppos >= 0 && ppos < real_base) {
                rp = cache.R_virt.Col(ppos);
            } else if (ppos >= real_base) {
                const int p_real = ppos - real_base;
                if (p_real >= 0 && p_real < n_real) {
                    const int pli = cl.indices[static_cast<std::size_t>(p_real)];
                    const int pgi = cache.cols_global[static_cast<std::size_t>(pli)];
                    rp = R_full_shared.Col(pgi);
                }
            }
            if (!rp) {
                LogWarn("UpdateCOneFromLinkages bad cluster [cid=" + std::to_string(cache.cluster_id) +
                        " pos=" + std::to_string(pos) + "]: ppos=" + std::to_string(ppos) +
                        " maps to no valid parent (n_real=" + std::to_string(n_real) +
                        " n_virt=" + std::to_string(real_base) + "); linkage data corrupt.");
                continue;
            }

            idx.push_back(gi);
            parent_recon.push_back(rp);
            weights.push_back(config.virtual_cfg.alpha_bad);
        }
    }

    // Good clusters (weight 1), parents are always real.
    for (const auto& cluster : linkage_good.clusters) {
        const int csize = static_cast<int>(cluster.indices.size());
        const int n_real = (cluster.n_real > 0) ? std::min(cluster.n_real, csize) : csize;
        if (n_real <= 0 || cluster.depth_offsets.size() < 2) {
            continue;
        }
        const int real_base = std::max(0, cluster.n_virtual);
        for (int real_idx = cluster.depth_offsets[1]; real_idx < n_real; ++real_idx) {
            const Index g = cluster.indices[static_cast<std::size_t>(real_idx)];
            const int local = real_base + real_idx;
            const int p_local =
                static_cast<int>(cluster.parent_local[static_cast<std::size_t>(local)]) - 1;
            const int p_real = p_local - real_base;
            if (p_real < 0 || p_real >= n_real) {
                LogWarn("UpdateCOneFromLinkages good cluster parent p_real=" + std::to_string(p_real) +
                        " out of range [0," + std::to_string(n_real) + ") at real_idx=" +
                        std::to_string(real_idx) + "; linkage data corrupt.");
                continue;
            }
            const Index pg = cluster.indices[static_cast<std::size_t>(p_real)];
            idx.push_back(static_cast<int>(g));
            parent_recon.push_back(R_full_shared.Col(static_cast<int>(pg)));
            weights.push_back(1.0f);
        }
    }

    if (idx.empty()) {
        if (error) {
            *error = "UpdateCOneFromLinkagesVirtualNoCopy: no depth>0 samples.";
        }
        return false;
    }

    auto h_vec_one = config.model.h_vec;
    if (!h_vec_one.empty()) {
        h_vec_one[0] = config.model.h0_one;
    }
    C_one->d = d;
    C_one->h_vec = h_vec_one;
    C_one->books = UpdateCodebooksLSFromResidualPtrs(X_rot, B, a,
                                                     idx, parent_recon, weights,
                                                     h_vec_one,
                                                     /*b0_override=*/nullptr);
    return true;
}

bool TrainQuantizerAndLinkageVirtualBeam(const Config& config,
                                      const ColMajorMatrix<float>& X_rot,
                                      StreamKernelProvider* stream_kernels,
                                      TrainState* state,
                                      std::string* error) {
    if (!state) {
        if (error) {
            *error = "TrainState output is null.";
        }
        return false;
    }

    const int d = X_rot.rows;
    const int n = X_rot.cols;
    if (d <= 0 || n <= 0) {
        if (error) {
            *error = "Empty training data for train_quantizer_and_linkage_virtual_beam.";
        }
        return false;
    }

    const int m = config.model.m;
    if (m <= 0 || static_cast<int>(config.model.h_vec.size()) != m) {
        if (error) {
            *error = "Model h_vec does not match m.";
        }
        return false;
    }

    std::mt19937 rng(static_cast<std::mt19937::result_type>(config.train.seed));

    KmeansConfig kcfg;
    kcfg.max_iters = std::max(1, config.train.kmeans_iters);
    kcfg.tol = static_cast<float>(config.train.kmeans_tol);
    kcfg.init_method = config.train.kmeans_init;
    kcfg.init_samples = std::max(1, config.train.init_samples);
    kcfg.kmeansll_gpu_enable = config.train.kmeansll_gpu_enable;
    kcfg.kmeansll_rounds = std::min(16, std::max(1, config.train.kmeansll_rounds));
    kcfg.kmeansll_oversample = static_cast<float>(std::min(16.0, std::max(1.0, config.train.kmeansll_oversample)));
    kcfg.kmeansll_candidate_cap = std::max(0, config.train.kmeansll_candidate_cap);
    kcfg.kmeansll_seed = config.train.kmeansll_seed;
    kcfg.kmeansll_hier_enable = config.train.kmeansll_hier_enable;
    kcfg.kmeansll_hier_threshold = std::max(1, config.train.kmeansll_hier_threshold);
    kcfg.kmeansll_hier_fixed_k1 = config.train.kmeansll_hier_fixed_k1;
    kcfg.initial_weight = static_cast<float>(config.train.kmeans_initial_weight);
    kcfg.min_weight = static_cast<float>(config.train.kmeans_min_weight);
    kcfg.outlier_quantile = static_cast<float>(config.train.kmeans_outlier_quantile);
    kcfg.cost_threshold = static_cast<float>(config.train.kmeans_cost_threshold);
    kcfg.annealing_factor = static_cast<float>(config.train.kmeans_annealing_factor);
    kcfg.warmup_iters = std::max(0, config.train.kmeans_warmup_iters);
    kcfg.quantile_bins = std::max(1, config.train.kmeans_quantile_bins);
    kcfg.anneal_no_spill = config.train.kmeans_anneal_no_spill;
    kcfg.anneal_mode = (config.train.kmeans_anneal_mode == "twopass") ? KmeansAnnealMode::kTwoPass
                                                                      : KmeansAnnealMode::kOnePass;
    kcfg.anneal_weights_device = config.train.kmeans_anneal_weights_device;
    kcfg.force_disable_device_weights = config.train.kmeans_force_disable_device_weights;
    kcfg.streaming = config.train.kmeans_streaming;
    kcfg.cache_xnorm_device = config.train.kmeans_cache_xnorm_device;
    kcfg.device_cache_mb = std::max(0, config.train.kmeans_device_cache_mb);
    kcfg.pin_host_x = config.train.kmeans_pin_host_x;
    kcfg.bvecs_use_u8 = (config.dataset.name == "SIFT1B") && config.train.kmeans_bvecs_use_u8;
    kcfg.block_cols = std::max(1, config.large.train_block);
    kcfg.tmp_dir = !config.train.kmeans_tmp_dir.empty() ? config.train.kmeans_tmp_dir : config.large.tmp_dir;
    kcfg.collect_metrics = config.train.log_metrics;
    kcfg.large_k_threshold = std::max(1, config.train.kmeans_large_k_threshold);
    kcfg.large_k_hier2_enable = config.train.kmeans_large_k_hier2_enable;
    kcfg.large_k_fixed_k1 = std::max(0, config.train.kmeans_large_k_fixed_k1);
    kcfg.large_k_k1_min = std::max(1, config.train.kmeans_large_k_k1_min);
    kcfg.large_k_k1_max = std::max(kcfg.large_k_k1_min, config.train.kmeans_large_k_k1_max);
    kcfg.large_k_k1_pow2 = config.train.kmeans_large_k_k1_pow2;
    kcfg.large_k_top_coarse = std::max(1, config.train.kmeans_large_k_top_coarse);
    kcfg.large_k_hier2_train = config.train.kmeans_large_k_hier2_train;
    kcfg.large_k_hier2_encode = config.train.kmeans_large_k_hier2_encode;

    // Julia reference: train_quantizer_and_linkage_virtual_beam! (train_quantizer.jl:587-742).
    std::vector<ColMajorMatrix<float>> C_root;
    ColMajorMatrix<FullCode> B;
    ColMajorMatrix<float> a;
    LogInfo("(1)RVQ-like init by Spherical kmeans with annealing factor...");
    KmeansTiming kmeans_timing{};
    // NOTE: `X_rot` is already materialized in-memory in this training path.
    // Even when `train.kmeans_streaming=true`, we keep the hierarchical quantizer in-memory
    // and rely on `kcfg.streaming` to select the streaming k-means kernels internally.
    HierarchicalQuantize(X_rot, config.model.h_vec, kcfg, &rng,
                         stream_kernels,
                         /*profile_timing=*/false,
                         &kmeans_timing,
                         &C_root, &B, &a);

    float init_mse = MeanMseFromCodebooks(X_rot, C_root, B, a);
    if (config.train.log_metrics) {
        LogInfo("RVQ-like Init mse error: " + FormatFloat(init_mse, 6));
    }
    {
        std::string tag = "cpu";
        if (stream_kernels && stream_kernels->IsGpu()) tag = "gpu";
	        if (config.large.profile_timing) {
	            const double norm_total_s =
	                ((kmeans_timing.normalize_stage_s + kmeans_timing.normalize_h2d_s + kmeans_timing.normalize_kernel_s) > 0.0)
	                    ? (kmeans_timing.normalize_stage_s + kmeans_timing.normalize_h2d_s + kmeans_timing.normalize_kernel_s)
	                    : kmeans_timing.normalize_s;
	            const double accounted =
	                kmeans_timing.reader_io_s +
	                norm_total_s +
	                kmeans_timing.init_centers_s +
	                kmeans_timing.assign_cpu_s +
	                kmeans_timing.assign_gemm_s +
	                kmeans_timing.assign_argmax_s +
	                kmeans_timing.rvq_project_s +
	                kmeans_timing.assign_update_compute_s +
	                kmeans_timing.emit_codes_d2h_s +
	                kmeans_timing.wait_copy_s +
	                kmeans_timing.wait_compute_s +
	                kmeans_timing.weights_s +
	                kmeans_timing.update_s +
	                kmeans_timing.spill_s +
	                kmeans_timing.update_read_weights_s;
	            const double other = std::max(0.0, kmeans_timing.total_s - accounted);
			            LogInfo("RVQ kmeans timing(" + tag + ", s): total=" + FormatFloat(static_cast<float>(kmeans_timing.total_s), 4) +
			                    " kmeans_streaming=" + std::to_string(kcfg.streaming ? 1 : 0) +
			                    " cache_xnorm_dev=" + std::to_string(kcfg.cache_xnorm_device ? 1 : 0) +
			                    " pin_host_x=" + std::to_string(kcfg.pin_host_x ? 1 : 0) +
			                    " bvecs_u8=" + std::to_string(kcfg.bvecs_use_u8 ? 1 : 0) +
			                    " tf32_eff=" + std::to_string((stream_kernels && stream_kernels->IsGpu() && EffectiveCudaAllowTf32(config.runtime)) ? 1 : 0) +
			                    " ws_mb=" + std::to_string((stream_kernels && stream_kernels->IsGpu()) ? config.runtime.cuda_cublas_workspace_mb : 0) +
			                    " overlap_p1=" + std::to_string(kmeans_timing.used_overlap_pass1) +
			                    " overlap_p2=" + std::to_string(kmeans_timing.used_overlap_pass2) +
			                    " hier2_en=" + std::to_string(kmeans_timing.hier2_enable) +
		                    " hier2_thr=" + std::to_string(kmeans_timing.hier2_threshold) +
	                    " hier2_used=" + std::to_string(kmeans_timing.used_hier2) +
	                    " hier2_K=" + std::to_string(kmeans_timing.hier2_K) +
	                    " hier2_K1=" + std::to_string(kmeans_timing.hier2_K1) +
	                    " hier2_K2=" + std::to_string(kmeans_timing.hier2_K2) +
	                    " hier2_topL=" + std::to_string(kmeans_timing.hier2_topL) +
	                    " anneal_en=" + std::to_string(kmeans_timing.anneal_enabled) +
	                    " anneal_no_spill=" + std::to_string(kmeans_timing.anneal_no_spill) +
	                    " io=" + FormatFloat(static_cast<float>(kmeans_timing.reader_io_s), 4) +
	                    " io_file=" + FormatFloat(static_cast<float>(kmeans_timing.reader_file_s), 4) +
	                    " io_unpack=" + FormatFloat(static_cast<float>(kmeans_timing.reader_unpack_s), 4) +
	                    " norm=" + FormatFloat(static_cast<float>(norm_total_s), 4) +
	                    " norm_stage=" + FormatFloat(static_cast<float>(kmeans_timing.normalize_stage_s), 4) +
	                    " norm_h2d=" + FormatFloat(static_cast<float>(kmeans_timing.normalize_h2d_s), 4) +
	                    " norm_ker=" + FormatFloat(static_cast<float>(kmeans_timing.normalize_kernel_s), 4) +
		                    " proj=" + FormatFloat(static_cast<float>(kmeans_timing.rvq_project_s), 4) +
		                    " assign_upd=" + FormatFloat(static_cast<float>(kmeans_timing.assign_update_compute_s), 4) +
		                    " codes_d2h=" + FormatFloat(static_cast<float>(kmeans_timing.emit_codes_d2h_s), 4) +
		                    " wait_copy=" + FormatFloat(static_cast<float>(kmeans_timing.wait_copy_s), 4) +
		                    " wait_compute=" + FormatFloat(static_cast<float>(kmeans_timing.wait_compute_s), 4) +
		                    " init=" + FormatFloat(static_cast<float>(kmeans_timing.init_centers_s), 4) +
		                    " init_sample=" + FormatFloat(static_cast<float>(kmeans_timing.init_sample_s), 4) +
		                    " init_pick=" + FormatFloat(static_cast<float>(kmeans_timing.init_pick_s), 4) +
		                    " assign_cpu=" + FormatFloat(static_cast<float>(kmeans_timing.assign_cpu_s), 4) +
		                    " assign_gemm=" + FormatFloat(static_cast<float>(kmeans_timing.assign_gemm_s), 4) +
		                    " assign_argmax=" + FormatFloat(static_cast<float>(kmeans_timing.assign_argmax_s), 4) +
		                    " weights=" + FormatFloat(static_cast<float>(kmeans_timing.weights_s), 4) +
                    " update=" + FormatFloat(static_cast<float>(kmeans_timing.update_s), 4) +
                    " spill=" + FormatFloat(static_cast<float>(kmeans_timing.spill_s), 4) +
                    " upd_wread=" + FormatFloat(static_cast<float>(kmeans_timing.update_read_weights_s), 4) +
                    " other=" + FormatFloat(static_cast<float>(other), 4) +
                    " xnorm_dev_used=" + std::to_string(kmeans_timing.used_device_xnorm_cache) +
                    " x_pin_used=" + std::to_string(kmeans_timing.used_pinned_host_x));
        } else if (config.train.log_metrics) {
			            LogInfo("RVQ kmeans timing(" + tag + ", s): total=" +
			                    FormatFloat(static_cast<float>(kmeans_timing.total_s), 4) +
			                    " kmeans_streaming=" + std::to_string(kcfg.streaming ? 1 : 0) +
		                    " cache_xnorm_dev=" + std::to_string(kcfg.cache_xnorm_device ? 1 : 0) +
		                    " pin_host_x=" + std::to_string(kcfg.pin_host_x ? 1 : 0) +
		                    " bvecs_u8=" + std::to_string(kcfg.bvecs_use_u8 ? 1 : 0) +
		                    " tf32_eff=" + std::to_string((stream_kernels && stream_kernels->IsGpu() && EffectiveCudaAllowTf32(config.runtime)) ? 1 : 0) +
		                    " ws_mb=" + std::to_string((stream_kernels && stream_kernels->IsGpu()) ? config.runtime.cuda_cublas_workspace_mb : 0) +
		                    " overlap_p1=" + std::to_string(kmeans_timing.used_overlap_pass1) +
			                    " overlap_p2=" + std::to_string(kmeans_timing.used_overlap_pass2) +
			                    " xnorm_dev_used=" + std::to_string(kmeans_timing.used_device_xnorm_cache) +
		                    " x_pin_used=" + std::to_string(kmeans_timing.used_pinned_host_x));
		        }
    }
    state->init_mse = init_mse;

    CodebookPack root_pack;
    root_pack.d = d;
    root_pack.h_vec = config.model.h_vec;
    root_pack.books = C_root;

    Precomp pre_root;
    if (!BuildPrecomp(root_pack, &pre_root)) {
        if (error) {
            *error = "Failed to build precomp for initial root codebooks.";
        }
        return false;
    }
    ColMajorMatrix<float> xC(pre_root.H, n);
    {
        ScopedBlasThreads blas_scope(OmpMaxThreads());
        Gemm(true, false, 1.0f, pre_root.C_all, X_rot, 0.0f, &xC);
    }

    BeamSearchPrefixLS(X_rot, pre_root, xC, std::max(1, config.base.encode.H_beam), &B);
    SolveLeastSquaresAll(pre_root, xC, B, &a);
    float beam_mse = MeanMseFromPrecomp(X_rot, pre_root, xC, B, a);
    if (config.train.log_metrics) {
        LogInfo("(2)Basic encode init with beam search mse error: " + FormatFloat(beam_mse, 6));
    }
    state->beam_mse = beam_mse;

    std::vector<float> X_norm2;
    std::vector<float> train_cost;
#if defined(STLQ_ENABLE_CUDA)
    if (config.runtime.use_cuda) {
        DynamicIcmWithIlsNoAbsNoNormalCuda(X_rot, pre_root, xC,
                                           config.train.icm_iters,
                                           config.train.ils_iters,
                                           config.train.perturb_k,
                                           static_cast<std::uint32_t>(config.train.seed),
                                           &B, &a, &X_norm2, &train_cost,
                                           /*print_progress=*/false,
                                           /*sample_id_offset=*/0);
    } else
#endif
    {
        DynamicIcmWithIlsNoAbsNoNormal(X_rot, pre_root, xC,
                                       config.train.icm_iters,
                                       config.train.ils_iters,
                                       config.train.perturb_k,
                                       static_cast<std::uint32_t>(config.train.seed),
                                       &B, &a, &X_norm2, &train_cost);
    }

    // Save root cluster ids before linkageing (Julia: cluster_id = Int.(vec(B[1,:]))).
    const std::vector<int> cluster_id = ExtractFirstLayerCodes(B);

    C_root = UpdateCodebooksLS(X_rot, B, a, config.model.h_vec);
    float icm_mse = MeanMseFromCodebooks(X_rot, C_root, B, a);
    if (config.train.log_metrics) {
        LogInfo("(3)Basic encode ILS+ICM mse error after updating C_root: " + FormatFloat(icm_mse, 6));
    }
    state->icm_mse = icm_mse;
    root_pack.books = C_root;

    // Initial linkage build uses the ILS variant in Julia.
    LinkageBuildConfig linkage_cfg = config.train.linkage;

    BaseEncoding train_base;
    train_base.B = std::move(B);
    train_base.a = std::move(a);
    LinkageStructure train_linkage;
    ColMajorMatrix<float> R_full;
    if (!BuildLinkageOneInit(linkage_cfg, config.hnsw, X_rot, &train_base, root_pack,
                    &train_linkage, &R_full, error)) {
        return false;
    }
    B = std::move(train_base.B);
    a = std::move(train_base.a);

    const std::vector<float> errors_linkage = ComputePerSampleSquaredErrors(X_rot, R_full);
    if (config.train.log_metrics && config.train.log_linkage_pre_c1) {
        float e_min = std::numeric_limits<float>::infinity();
        float e_max = -std::numeric_limits<float>::infinity();
        double e_sum = 0.0;
        for (float e : errors_linkage) {
            e_min = std::min(e_min, e);
            e_max = std::max(e_max, e);
            e_sum += static_cast<double>(e);
        }
        const float e_mean = (n > 0) ? static_cast<float>(e_sum / static_cast<double>(n)) : 0.0f;
        LogInfo("(4)Linkage encode analysis before updating C_one:");
        LinkageDepthStats linkage_stats = ComputeLinkageDepthStats(train_linkage, n);
        LogInfo("  Linkage Ratio: " + FormatFloat(100.0f * linkage_stats.linkage_ratio, 2) + "%");
        LogInfo("  Max depth: " + std::to_string(linkage_stats.max_depth) +
                " Mean depth: " + FormatFloat(linkage_stats.mean_depth, 2));
        LogInfo("  Linkage Error: Max=" + FormatFloat(e_max, 4) +
                ", Min=" + FormatFloat(e_min, 4) +
                ", Mean=" + FormatFloat(e_mean, 4));
    }
    double baseline = 0.0;
    const int h0 = root_pack.books.empty() ? 0 : root_pack.books.front().cols;
    std::vector<bool> is_bad_cluster = BuildBadClusterMaskQuantile(
        cluster_id, errors_linkage, h0,
        config.virtual_cfg.good_fraction, &baseline);
    int bad_cnt = 0;
    for (bool v : is_bad_cluster) {
        bad_cnt += v ? 1 : 0;
    }
    LogInfo("Virtual init: num bad clusters = " + std::to_string(bad_cnt) +
            " (baseline=" + FormatFloat(static_cast<float>(baseline), 6) + ")");

    CodebookPack one_pack;
    std::string local_error;
    if (!UpdateCOneWeighted(config, X_rot, R_full,
                            B, a,
                            root_pack,
                            train_linkage,
                            cluster_id, is_bad_cluster,
                            &one_pack, &local_error)) {
        if (error) {
            *error = local_error;
        }
        return false;
    }

    ColMajorMatrix<float> R_full_updated;
    try {
        R_full_updated = ComputeLinkagedReconstructionMultiBook(root_pack, one_pack, B, a, train_linkage);
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Virtual init: failed to recompute linkaged reconstruction: ") + e.what();
        }
        return false;
    }

    if (config.train.log_metrics) {
        LogInfo("(4)Linkage encode analysis after updating C_one:");
        LinkageDepthStats linkage_stats = ComputeLinkageDepthStats(train_linkage, n);
        MseStats mse_stats = MseStatsFromRecon(X_rot, R_full_updated);
        LogInfo("  Linkage Ratio: " + FormatFloat(100.0f * linkage_stats.linkage_ratio, 2) + "%");
        LogInfo("  Max depth: " + std::to_string(linkage_stats.max_depth) +
                " Mean depth: " + FormatFloat(linkage_stats.mean_depth, 2));
        LogInfo("  Linkage Error: Max=" + FormatFloat(mse_stats.max, 4) +
                ", Min=" + FormatFloat(mse_stats.min, 4) +
                ", Mean=" + FormatFloat(mse_stats.mean, 4));
        state->linkage_mse_mean = mse_stats.mean;
        state->linkage_ratio = linkage_stats.linkage_ratio;
        state->linkage_max_depth = linkage_stats.max_depth;
        state->linkage_mean_depth = linkage_stats.mean_depth;
    }

    state->C_root = root_pack;
    state->C_one = one_pack;
    state->B = B;
    state->a = a;
    state->R_full = std::move(R_full_updated);
    state->R_in = {};
    state->train_linkage = std::move(train_linkage);
    state->is_bad_cluster = std::move(is_bad_cluster);
    return true;
}

bool LinkageRefineOnRotatedDataVirtualBeam(const Config& config,
                                        const ColMajorMatrix<float>& X_rot,
                                        TrainState* state,
                                        std::string* error) {
    if (!state) {
        if (error) {
            *error = "TrainState output is null.";
        }
        return false;
    }
    if (state->C_root.books.empty() || state->C_one.books.empty()) {
        if (error) {
            *error = "TrainState codebooks are empty.";
        }
        return false;
    }
    if (state->is_bad_cluster.empty()) {
        if (error) {
            *error = "Virtual refine requires is_bad_cluster from initialization.";
        }
        return false;
    }

    const int d = X_rot.rows;
    const int n = X_rot.cols;
    if (d <= 0 || n <= 0) {
        if (error) {
            *error = "Empty training data for linkage_refine_on_rotated_data_virtual_beam.";
        }
        return false;
    }
    if (state->B.cols != n || state->a.cols != n) {
        state->B = ColMajorMatrix<FullCode>(config.model.m, n);
        state->a = ColMajorMatrix<float>(config.model.m, n);
    }

    Precomp pre_root;
    if (!BuildPrecomp(state->C_root, &pre_root)) {
        if (error) {
            *error = "Failed to build precomp for C_root.";
        }
        return false;
    }

    ColMajorMatrix<float> xC(pre_root.H, n);
    {
        ScopedBlasThreads blas_scope(OmpMaxThreads());
        Gemm(true, false, 1.0f, pre_root.C_all, X_rot, 0.0f, &xC);
    }

    if (config.train.log_metrics) {
        LogInfo("doing beam search...");
    }
    BeamSearchPrefixLS(X_rot, pre_root, xC, std::max(1, config.base.encode.H_beam), &state->B);
    SolveLeastSquaresAll(pre_root, xC, state->B, &state->a);
    float beam_mse = MeanMseFromPrecomp(X_rot, pre_root, xC, state->B, state->a);
    if (config.train.log_metrics) {
        LogInfo("Beam_search mse error (initial): " + FormatFloat(beam_mse, 6));
    }
    state->beam_mse = beam_mse;

    std::vector<float> X_norm2;
    std::vector<float> train_cost;
#if defined(STLQ_ENABLE_CUDA)
    if (config.runtime.use_cuda) {
        DynamicIcmWithIlsNoAbsNoNormalCuda(X_rot, pre_root, xC,
                                           config.train.icm_iters,
                                           config.train.ils_iters,
                                           config.train.perturb_k,
                                           static_cast<std::uint32_t>(config.train.seed),
                                           &state->B, &state->a, &X_norm2, &train_cost,
                                           /*print_progress=*/false,
                                           /*sample_id_offset=*/0);
    } else
#endif
    {
        DynamicIcmWithIlsNoAbsNoNormal(X_rot, pre_root, xC,
                                       config.train.icm_iters,
                                       config.train.ils_iters,
                                       config.train.perturb_k,
                                       static_cast<std::uint32_t>(config.train.seed),
                                       &state->B, &state->a, &X_norm2, &train_cost);
    }

    const std::vector<int> cluster_id = ExtractFirstLayerCodes(state->B);

    state->C_root.books = UpdateCodebooksLS(X_rot, state->B, state->a, config.model.h_vec);
    float pre_linkage_mse = MeanMseFromCodebooks(X_rot, state->C_root.books, state->B, state->a);
    if (config.train.log_metrics) {
        LogInfo("Trainset mse error after update C_root: " + FormatFloat(pre_linkage_mse, 6));
    }
    state->icm_mse = pre_linkage_mse;

    // Rebuild precomps after codebook updates (encoding used the earlier pre_root/xC).
    if (!BuildPrecomp(state->C_root, &pre_root)) {
        if (error) {
            *error = "Failed to rebuild precomp for updated C_root.";
        }
        return false;
    }
    Precomp pre_one;
    if (!BuildPrecomp(state->C_one, &pre_one)) {
        if (error) {
            *error = "Failed to build precomp for C_one.";
        }
        return false;
    }
    ColMajorMatrix<float> G_one_root(pre_one.H, pre_root.H);
    {
        ScopedBlasThreads blas_scope(OmpMaxThreads());
        Gemm(true, false, 1.0f, pre_one.C_all, pre_root.C_all, 0.0f, &G_one_root);
    }

    // Dual-strategy linkageing (virtual):
    // 1) Split cluster ids into good/bad once.
    // 2) Good clusters: standard two-codebook linkage (no masking, no copies).
    // 3) Bad clusters: fused UMAP-like augment -> multi-center linkage (per cluster, parallel).
    LinkageBuildConfig linkage_cfg = config.train.linkage;

    const int h1 = config.model.h_vec.front();
    std::vector<std::vector<int>> good_cols_by_cid(static_cast<std::size_t>(h1));
    std::vector<std::vector<int>> bad_cols_by_cid(static_cast<std::size_t>(h1));
    for (int i = 0; i < n; ++i) {
        const int cid = cluster_id[static_cast<std::size_t>(i)];
        if (state->is_bad_cluster[static_cast<std::size_t>(cid)]) {
            bad_cols_by_cid[static_cast<std::size_t>(cid)].push_back(i);
        } else {
            good_cols_by_cid[static_cast<std::size_t>(cid)].push_back(i);
        }
    }
    std::vector<int> bad_cids;
    bad_cids.reserve(static_cast<std::size_t>(h1));
    for (int cid = 0; cid < h1; ++cid) {
        if (!bad_cols_by_cid[static_cast<std::size_t>(cid)].empty()) {
            bad_cids.push_back(cid);
        }
    }

    LinkageStructure linkage_good;
    ColMajorMatrix<float> R_full_shared;
    if (!ProcessGoodClustersVirtual(linkage_cfg, config.hnsw,
                                               X_rot,
                                               std::move(good_cols_by_cid),
                                               &state->B,
                                               &state->a,
                                               state->C_root,
                                               state->C_one,
                                               &linkage_good,
                                               &R_full_shared,
                                               error)) {
        return false;
    }

    std::vector<BadClusterCache> bad_cache;
    if (!ProcessBadClustersVirtualForTrain(config, X_rot,
                                   bad_cols_by_cid,
                                   bad_cids,
                                   linkage_cfg,
                                   state->C_root,
                                   state->C_one,
                                   pre_root,
                                   pre_one,
                                   G_one_root,
                                   &state->B,
                                   &state->a,
                                   &R_full_shared,
                                   &bad_cache,
                                   error)) {
        return false;
    }

    if (config.train.log_metrics && config.train.log_linkage_pre_c1) {
        LogInfo("Train Linkage encode analysis before updating C_one:");
        long long linkaged = 0;
        long long depth_sum = 0;
        int depth_max = 0;
        for (const auto& cluster : linkage_good.clusters) {
            const int cid = static_cast<int>(cluster.cluster_id);
            if (cid >= 0 && cid < static_cast<int>(state->is_bad_cluster.size()) &&
                state->is_bad_cluster[static_cast<std::size_t>(cid)]) {
                continue;
            }
            const int csize = static_cast<int>(cluster.indices.size());
            const int n_real = (cluster.n_real > 0) ? std::min(cluster.n_real, csize) : csize;
            if (n_real <= 0 || cluster.depth_offsets.size() < 2) {
                continue;
            }
            const int max_d = static_cast<int>(cluster.depth_offsets.size()) - 2;
            depth_max = std::max(depth_max, max_d);
            const int depth0 = cluster.depth_offsets[1] - cluster.depth_offsets[0];
            linkaged += static_cast<long long>(n_real - depth0);
            for (int d = 0; d <= max_d; ++d) {
                const int cnt = cluster.depth_offsets[static_cast<std::size_t>(d + 1)] -
                                cluster.depth_offsets[static_cast<std::size_t>(d)];
                depth_sum += static_cast<long long>(d) * static_cast<long long>(cnt);
            }
        }
        for (const auto& cache : bad_cache) {
            if (cache.cluster_id < 0 || cache.n_real <= 0 || cache.linkage_aug.clusters.empty()) {
                continue;
            }
            const auto& cl = cache.linkage_aug.clusters.front();
            if (cl.depth_offsets.size() < 2) {
                continue;
            }
            const int max_d = static_cast<int>(cl.depth_offsets.size()) - 2;
            depth_max = std::max(depth_max, max_d);
            const int depth0 = cl.depth_offsets[1] - cl.depth_offsets[0];
            linkaged += static_cast<long long>(cache.n_real - depth0);
            for (int d = 0; d <= max_d; ++d) {
                const int cnt = cl.depth_offsets[static_cast<std::size_t>(d + 1)] -
                                cl.depth_offsets[static_cast<std::size_t>(d)];
                depth_sum += static_cast<long long>(d) * static_cast<long long>(cnt);
            }
        }

        const float linkage_ratio = (n > 0)
                                      ? static_cast<float>(static_cast<double>(linkaged) /
                                                           static_cast<double>(n))
                                      : 0.0f;
        const float mean_depth = (n > 0)
                                     ? static_cast<float>(static_cast<double>(depth_sum) /
                                                          static_cast<double>(n))
                                     : 0.0f;
        MseStats mse_stats = MseStatsFromRecon(X_rot, R_full_shared);
        LogInfo("  Linkage Ratio: " + FormatFloat(100.0f * linkage_ratio, 2) + "%");
        LogInfo("  Max depth: " + std::to_string(depth_max) +
                " Mean depth: " + FormatFloat(mean_depth, 2));
        LogInfo("  Linkage Error: Max=" + FormatFloat(mse_stats.max, 4) +
                ", Min=" + FormatFloat(mse_stats.min, 4) +
                ", Mean=" + FormatFloat(mse_stats.mean, 4));
    }

    if (!UpdateCOneFromLinkagesVirtualNoCopy(config, X_rot,
                                           state->B, state->a,
                                           linkage_good,
                                           R_full_shared,
                                           bad_cache,
                                           &state->C_one,
                                           error)) {
        return false;
    }

    auto RebuildRfullAfterCOneUpdate = [&](ColMajorMatrix<float>* out) -> bool {
        try {
            *out = ReconstructAll(state->C_root.books, state->B, state->a);

            // Good clusters: overwrite linkaged nodes in-place (global indices).
            ApplyLinkagedReconstructionMultiBookInPlace(state->C_root, state->C_one,
                                                       state->B, state->a,
                                                       linkage_good,
                                                       out);

            // Bad clusters: augmented (real + virtual) cluster-local indices.
            // Write real nodes directly into `out` and keep virtual nodes in a temporary buffer.
            for (const auto& cache : bad_cache) {
                const int n_real = cache.n_real;
                const int n_virt = cache.n_virt;
                const int n_aug = n_real + n_virt;
                if (cache.cluster_id < 0 || n_aug <= 0 || cache.linkage_aug.clusters.empty()) {
                    continue;
                }

                // Reconstruct only real nodes; virtual nodes are roots (depth==0) with cached recon.
                const auto& cluster = cache.linkage_aug.clusters.front();
                const int csize = static_cast<int>(cluster.indices.size());
                if (csize <= 0) {
                    continue;
                }
                if (cluster.depth_offsets.size() < 2) {
                    continue;
                }
                const auto& offsets = cluster.depth_offsets;
                const int max_depth = static_cast<int>(offsets.size()) - 2;
                const int d = out->rows;
                const int m = config.model.m;
                for (int dep = 0; dep <= max_depth; ++dep) {
                    const int begin = offsets[static_cast<std::size_t>(dep)];
                    const int end = offsets[static_cast<std::size_t>(dep + 1)];
                    for (int pos = begin; pos < end; ++pos) {
                        const int li = cluster.indices[static_cast<std::size_t>(pos)];
                        const int gi = cache.cols_global[static_cast<std::size_t>(li)];
                        float* dst = out->Col(gi);
                        std::fill(dst, dst + d, 0.0f);

                        const auto& books = (dep == 0) ? state->C_root.books : state->C_one.books;
                    for (int l = 0; l < m; ++l) {
                        const int code = static_cast<int>(state->B(l, gi));
                        const float coeff = state->a(l, gi);
                        const float* center = books[l].Col(code);
                            #pragma omp simd
                            for (int r = 0; r < d; ++r) {
                                dst[r] += coeff * center[r];
                            }
                        }

                        const int real_base = n_virt;
                        const int local = real_base + pos;
                        const int p_local1 = cluster.parent_local[static_cast<std::size_t>(local)];
                        if (p_local1 != 0) {
                            const int p_local = p_local1 - 1;
                            const float* rp = nullptr;
                            if (p_local < real_base) {
                                rp = cache.R_virt.Col(p_local);
                            } else {
                                const int p_real = p_local - real_base;
                                if (p_real < 0 || p_real >= n_real) {
                                    LogWarn("RebuildRfull[cluster] real parent p_real=" + std::to_string(p_real) +
                                            " out of range [0," + std::to_string(n_real) + ") at pos=" +
                                            std::to_string(pos) + "; linkage data corrupt.");
                                    continue;
                                }
                                const int pli = cluster.indices[static_cast<std::size_t>(p_real)];
                                const int pgi = cache.cols_global[static_cast<std::size_t>(pli)];
                                rp = out->Col(pgi);
                            }
                            if (!rp) {
                                LogWarn("RebuildRfull[cluster] virtual parent p_local=" + std::to_string(p_local) +
                                        " has null rp (R_virt not precomputed or out of range); linkage data corrupt.");
                                continue;
                            }
                            #pragma omp simd
                            for (int r = 0; r < d; ++r) {
                                dst[r] += rp[r];
                            }
                        }
                    }
                }
            }
        } catch (const std::exception& e) {
            if (error) {
                *error = std::string("Virtual refine: failed to rebuild reconstruction: ") + e.what();
            }
            return false;
        }
        return true;
    };

    // After updating C_one, rebuild reconstruction only for reporting / next iteration.
    // Start from the base reconstruction (depth=0 uses C_root), then overwrite:
    // - good clusters via standard linkage reconstruction
    // - bad clusters via augmented-linkage reconstruction (real nodes only)
    ColMajorMatrix<float> R_full_updated;
    if (!RebuildRfullAfterCOneUpdate(&R_full_updated)) {
        return false;
    }

    {
        LogInfo("(3)Linkage encode analysis after updating C_one:");
        long long linkaged = 0;
        long long depth_sum = 0;
        int depth_max = 0;
        for (const auto& cluster : linkage_good.clusters) {
            const int cid = static_cast<int>(cluster.cluster_id);
            if (cid >= 0 && cid < static_cast<int>(state->is_bad_cluster.size()) &&
                state->is_bad_cluster[static_cast<std::size_t>(cid)]) {
                continue;
            }
            const int csize = static_cast<int>(cluster.indices.size());
            const int n_real = (cluster.n_real > 0) ? std::min(cluster.n_real, csize) : csize;
            if (n_real <= 0 || cluster.depth_offsets.size() < 2) {
                continue;
            }
            const int max_d = static_cast<int>(cluster.depth_offsets.size()) - 2;
            depth_max = std::max(depth_max, max_d);
            const int depth0 = cluster.depth_offsets[1] - cluster.depth_offsets[0];
            linkaged += static_cast<long long>(n_real - depth0);
            for (int d = 0; d <= max_d; ++d) {
                const int cnt = cluster.depth_offsets[static_cast<std::size_t>(d + 1)] -
                                cluster.depth_offsets[static_cast<std::size_t>(d)];
                depth_sum += static_cast<long long>(d) * static_cast<long long>(cnt);
            }
        }
        for (const auto& cache : bad_cache) {
            if (cache.cluster_id < 0 || cache.n_real <= 0 || cache.linkage_aug.clusters.empty()) {
                continue;
            }
            const auto& cl = cache.linkage_aug.clusters.front();
            if (cl.depth_offsets.size() < 2) {
                continue;
            }
            const int max_d = static_cast<int>(cl.depth_offsets.size()) - 2;
            depth_max = std::max(depth_max, max_d);
            const int depth0 = cl.depth_offsets[1] - cl.depth_offsets[0];
            linkaged += static_cast<long long>(cache.n_real - depth0);
            for (int d = 0; d <= max_d; ++d) {
                const int cnt = cl.depth_offsets[static_cast<std::size_t>(d + 1)] -
                                cl.depth_offsets[static_cast<std::size_t>(d)];
                depth_sum += static_cast<long long>(d) * static_cast<long long>(cnt);
            }
        }

        const float linkage_ratio = (n > 0)
                                      ? static_cast<float>(static_cast<double>(linkaged) /
                                                           static_cast<double>(n))
                                      : 0.0f;
        const float mean_depth = (n > 0)
                                     ? static_cast<float>(static_cast<double>(depth_sum) /
                                                          static_cast<double>(n))
                                     : 0.0f;
        MseStats mse_stats = MseStatsFromRecon(X_rot, R_full_updated);
        LogInfo("  Linkage Ratio: " + FormatFloat(100.0f * linkage_ratio, 2) + "%");
        LogInfo("  Max depth: " + std::to_string(depth_max) +
                " Mean depth: " + FormatFloat(mean_depth, 2));
        LogInfo("  Linkage Error: Max=" + FormatFloat(mse_stats.max, 4) +
                ", Min=" + FormatFloat(mse_stats.min, 4) +
                ", Mean=" + FormatFloat(mse_stats.mean, 4));
        state->linkage_mse_mean = mse_stats.mean;
        state->linkage_ratio = linkage_ratio;
        state->linkage_mean_depth = mean_depth;
        state->linkage_max_depth = depth_max;
    }

    state->R_full = std::move(R_full_updated);
    state->R_in = {};
    state->train_linkage = {};
    return true;
}

}  // namespace virtual_train



bool TrainQuantizerAndLinkageBeam(const Config& config,
                                const ColMajorMatrix<float>& X_rot,
                                StreamKernelProvider* stream_kernels,
                                TrainState* state,
                                std::string* error) {
    if (!state) {
        if (error) {
            *error = "TrainState output is null.";
        }
        return false;
    }

    const int d = X_rot.rows;
    const int n = X_rot.cols;
    if (d <= 0 || n <= 0) {
        if (error) {
            *error = "Empty training data for train_quantizer_and_linkage_beam.";
        }
        return false;
    }

    const int m = config.model.m;
    if (m <= 0 || static_cast<int>(config.model.h_vec.size()) != m) {
        if (error) {
            *error = "Model h_vec does not match m.";
        }
        return false;
    }

    std::mt19937 rng(static_cast<std::mt19937::result_type>(config.train.seed));

    KmeansConfig kcfg;
    kcfg.max_iters = std::max(1, config.train.kmeans_iters);
    kcfg.tol = static_cast<float>(config.train.kmeans_tol);
    kcfg.init_method = config.train.kmeans_init;
    kcfg.init_samples = std::max(1, config.train.init_samples);
    kcfg.kmeansll_gpu_enable = config.train.kmeansll_gpu_enable;
    kcfg.kmeansll_rounds = std::min(16, std::max(1, config.train.kmeansll_rounds));
    kcfg.kmeansll_oversample = static_cast<float>(std::min(16.0, std::max(1.0, config.train.kmeansll_oversample)));
    kcfg.kmeansll_candidate_cap = std::max(0, config.train.kmeansll_candidate_cap);
    kcfg.kmeansll_seed = config.train.kmeansll_seed;
    kcfg.kmeansll_hier_enable = config.train.kmeansll_hier_enable;
    kcfg.kmeansll_hier_threshold = std::max(1, config.train.kmeansll_hier_threshold);
    kcfg.kmeansll_hier_fixed_k1 = config.train.kmeansll_hier_fixed_k1;
    kcfg.initial_weight = static_cast<float>(config.train.kmeans_initial_weight);
    kcfg.min_weight = static_cast<float>(config.train.kmeans_min_weight);
    kcfg.outlier_quantile = static_cast<float>(config.train.kmeans_outlier_quantile);
    kcfg.cost_threshold = static_cast<float>(config.train.kmeans_cost_threshold);
    kcfg.annealing_factor = static_cast<float>(config.train.kmeans_annealing_factor);
    kcfg.warmup_iters = std::max(0, config.train.kmeans_warmup_iters);
    kcfg.quantile_bins = std::max(1, config.train.kmeans_quantile_bins);
    kcfg.anneal_no_spill = config.train.kmeans_anneal_no_spill;
    kcfg.anneal_mode = (config.train.kmeans_anneal_mode == "twopass") ? KmeansAnnealMode::kTwoPass
                                                                      : KmeansAnnealMode::kOnePass;
    kcfg.anneal_weights_device = config.train.kmeans_anneal_weights_device;
    kcfg.force_disable_device_weights = config.train.kmeans_force_disable_device_weights;
    kcfg.streaming = config.train.kmeans_streaming;
    kcfg.cache_xnorm_device = config.train.kmeans_cache_xnorm_device;
    kcfg.device_cache_mb = std::max(0, config.train.kmeans_device_cache_mb);
    kcfg.pin_host_x = config.train.kmeans_pin_host_x;
    kcfg.bvecs_use_u8 = (config.dataset.name == "SIFT1B") && config.train.kmeans_bvecs_use_u8;
    kcfg.block_cols = std::max(1, config.large.train_block);
    kcfg.tmp_dir = !config.train.kmeans_tmp_dir.empty() ? config.train.kmeans_tmp_dir : config.large.tmp_dir;
    kcfg.collect_metrics = config.train.log_metrics;

    // Julia reference: train_quantizer_and_linkage_beam! (train_quantizer.jl:118-188).
    std::vector<ColMajorMatrix<float>> C_root;
    ColMajorMatrix<FullCode> B;
    ColMajorMatrix<float> a;
    // Step 1a: hierarchical_vector_quantization_spherical_extend
    // (train_quantizer.jl:118-125).
    LogInfo("(1)RVQ-like init by Spherical kmeans with annealing factor...");
    KmeansTiming kmeans_timing{};
    // Same rationale as the virtual version above: for in-memory `X_rot`, keep the
    // hierarchical quantizer in-memory, and let `kcfg.streaming` pick the streaming
    // k-means kernels internally. Avoid the `rvq_layer_store` path here.
    HierarchicalQuantize(X_rot, config.model.h_vec, kcfg, &rng,
                         stream_kernels,
                         /*profile_timing=*/false,
                         &kmeans_timing,
                         &C_root, &B, &a);
    float init_mse = MeanMseFromCodebooks(X_rot, C_root, B, a);
    if (config.train.log_metrics) {
        LogInfo("RVQ-like Init mse error: " + FormatFloat(init_mse, 6));
    }
    {
        std::string tag = "cpu";
        if (stream_kernels && stream_kernels->IsGpu()) tag = "gpu";
        if (config.large.profile_timing) {
			            LogInfo("RVQ kmeans timing(" + tag + ", s): total=" + FormatFloat(static_cast<float>(kmeans_timing.total_s), 4) +
			                    " kmeans_streaming=" + std::to_string(kcfg.streaming ? 1 : 0) +
			                    " cache_xnorm_dev=" + std::to_string(kcfg.cache_xnorm_device ? 1 : 0) +
		                    " pin_host_x=" + std::to_string(kcfg.pin_host_x ? 1 : 0) +
		                    " bvecs_u8=" + std::to_string(kcfg.bvecs_use_u8 ? 1 : 0) +
		                    " tf32_eff=" + std::to_string((stream_kernels && stream_kernels->IsGpu() && EffectiveCudaAllowTf32(config.runtime)) ? 1 : 0) +
		                    " ws_mb=" + std::to_string((stream_kernels && stream_kernels->IsGpu()) ? config.runtime.cuda_cublas_workspace_mb : 0) +
		                    " overlap_p1=" + std::to_string(kmeans_timing.used_overlap_pass1) +
		                    " overlap_p2=" + std::to_string(kmeans_timing.used_overlap_pass2) +
		                    " norm=" + FormatFloat(static_cast<float>(kmeans_timing.normalize_s), 4) +
	                    " norm_h2d=" + FormatFloat(static_cast<float>(kmeans_timing.normalize_h2d_s), 4) +
	                    " norm_ker=" + FormatFloat(static_cast<float>(kmeans_timing.normalize_kernel_s), 4) +
	                    " init=" + FormatFloat(static_cast<float>(kmeans_timing.init_centers_s), 4) +
	                    " init_sample=" + FormatFloat(static_cast<float>(kmeans_timing.init_sample_s), 4) +
	                    " init_pick=" + FormatFloat(static_cast<float>(kmeans_timing.init_pick_s), 4) +
	                    " assign_cpu=" + FormatFloat(static_cast<float>(kmeans_timing.assign_cpu_s), 4) +
	                    " assign_gemm=" + FormatFloat(static_cast<float>(kmeans_timing.assign_gemm_s), 4) +
	                    " assign_argmax=" + FormatFloat(static_cast<float>(kmeans_timing.assign_argmax_s), 4) +
	                    " weights=" + FormatFloat(static_cast<float>(kmeans_timing.weights_s), 4) +
                    " update=" + FormatFloat(static_cast<float>(kmeans_timing.update_s), 4) +
                    " spill=" + FormatFloat(static_cast<float>(kmeans_timing.spill_s), 4) +
	                    " upd_wread=" + FormatFloat(static_cast<float>(kmeans_timing.update_read_weights_s), 4) +
	                    " xnorm_dev_used=" + std::to_string(kmeans_timing.used_device_xnorm_cache) +
	                    " x_pin_used=" + std::to_string(kmeans_timing.used_pinned_host_x));
        } else if (config.train.log_metrics) {
			            LogInfo("RVQ kmeans timing(" + tag + ", s): total=" +
			                    FormatFloat(static_cast<float>(kmeans_timing.total_s), 4) +
			                    " kmeans_streaming=" + std::to_string(kcfg.streaming ? 1 : 0) +
		                    " cache_xnorm_dev=" + std::to_string(kcfg.cache_xnorm_device ? 1 : 0) +
		                    " pin_host_x=" + std::to_string(kcfg.pin_host_x ? 1 : 0) +
		                    " bvecs_u8=" + std::to_string(kcfg.bvecs_use_u8 ? 1 : 0) +
		                    " tf32_eff=" + std::to_string((stream_kernels && stream_kernels->IsGpu() && EffectiveCudaAllowTf32(config.runtime)) ? 1 : 0) +
		                    " ws_mb=" + std::to_string((stream_kernels && stream_kernels->IsGpu()) ? config.runtime.cuda_cublas_workspace_mb : 0) +
		                    " overlap_p1=" + std::to_string(kmeans_timing.used_overlap_pass1) +
		                    " overlap_p2=" + std::to_string(kmeans_timing.used_overlap_pass2) +
		                    " hier2_en=" + std::to_string(kmeans_timing.hier2_enable) +
	                    " hier2_thr=" + std::to_string(kmeans_timing.hier2_threshold) +
                    " hier2_used=" + std::to_string(kmeans_timing.used_hier2) +
                    " hier2_K=" + std::to_string(kmeans_timing.hier2_K) +
                    " hier2_K1=" + std::to_string(kmeans_timing.hier2_K1) +
                    " hier2_K2=" + std::to_string(kmeans_timing.hier2_K2) +
                    " hier2_topL=" + std::to_string(kmeans_timing.hier2_topL) +
                    " anneal_en=" + std::to_string(kmeans_timing.anneal_enabled) +
	                    " anneal_no_spill=" + std::to_string(kmeans_timing.anneal_no_spill) +
	                    " xnorm_dev_used=" + std::to_string(kmeans_timing.used_device_xnorm_cache) +
	                    " x_pin_used=" + std::to_string(kmeans_timing.used_pinned_host_x));
        }
    }
    state->init_mse = init_mse;

    CodebookPack root_pack;
    root_pack.d = d;
    root_pack.h_vec = config.model.h_vec;
    root_pack.books = C_root;

    // Step 1b: build precomp + xC (train_quantizer.jl:138-149).
    Precomp pre_root;
    if (!BuildPrecomp(root_pack, &pre_root)) {
        if (error) {
            *error = "Failed to build precomp for initial root codebooks.";
        }
        return false;
    }
    ColMajorMatrix<float> xC(pre_root.H, n);
    {
        ScopedBlasThreads blas_scope(OmpMaxThreads());
        Gemm(true, false, 1.0f, pre_root.C_all, X_rot, 0.0f, &xC);
    }

    // Step 1c: beam_quantize_prefix_ls_incremental! (train_quantizer.jl:142-159).
    BeamSearchPrefixLS(X_rot, pre_root, xC, std::max(1, config.base.encode.H_beam), &B);

    // Step 1d: optimized_least_squares_all! (train_quantizer.jl:166-168).
    SolveLeastSquaresAll(pre_root, xC, B, &a);
    float beam_mse = MeanMseFromPrecomp(X_rot, pre_root, xC, B, a);
    if (config.train.log_metrics) {
        LogInfo("(2)Basic encode init with beam search mse error: " + FormatFloat(beam_mse, 6));
    }
    state->beam_mse = beam_mse;

    // Step 2: dynamic_icm_with_ils_noabs_nonormal!
    // (train_quantizer.jl:177-182, encode_abs.jl:430-512).
    std::vector<float> X_norm2;
    std::vector<float> train_cost;
#if defined(STLQ_ENABLE_CUDA)
    if (config.runtime.use_cuda) {
        DynamicIcmWithIlsNoAbsNoNormalCuda(X_rot, pre_root, xC,
                                           config.train.icm_iters,
                                           config.train.ils_iters,
                                           config.train.perturb_k,
                                           static_cast<std::uint32_t>(config.train.seed),
                                           &B, &a, &X_norm2, &train_cost,
                                           /*print_progress=*/false,
                                           /*sample_id_offset=*/0);
    } else
#endif
    {
        DynamicIcmWithIlsNoAbsNoNormal(X_rot, pre_root, xC,
                                       config.train.icm_iters,
                                       config.train.ils_iters,
                                       config.train.perturb_k,
                                       static_cast<std::uint32_t>(config.train.seed),
                                       &B, &a, &X_norm2, &train_cost);
    }

    // Step 3: update_codebooks_ls_extend (train_quantizer.jl:184-189).
    C_root = UpdateCodebooksLS(X_rot, B, a, config.model.h_vec);
    float icm_mse = MeanMseFromCodebooks(X_rot, C_root, B, a);
    if (config.train.log_metrics) {
        LogInfo("(3)Basic encode ILS+ICM mse error after updating C_root: " + FormatFloat(icm_mse, 6));
    }
    state->icm_mse = icm_mse;
    root_pack.books = C_root;

    // Step 4: linkage_from_inner_to_outer_one_codebook_ils!
    // (train_quantizer.jl:196-213, linkage_quantizer.jl:380-768).
    BaseEncoding train_base;
    train_base.B = std::move(B);
    train_base.a = std::move(a);
    LinkageStructure train_linkage;
    ColMajorMatrix<float> R_full;

    if (!BuildLinkageOneInit(config.train.linkage, config.hnsw,
                    X_rot, &train_base, root_pack, &train_linkage, &R_full, error)) {
        return false;
    }
    B = std::move(train_base.B);
    a = std::move(train_base.a);

    if (config.train.log_metrics && config.train.log_linkage_pre_c1) {
        LogInfo("(4)Linkage encode analysis before updating C_one:");
        LinkageDepthStats linkage_stats = ComputeLinkageDepthStats(train_linkage, n);
        MseStats mse_stats = MseStatsFromRecon(X_rot, R_full);
        LogInfo("  Linkage Ratio: " + FormatFloat(100.0f * linkage_stats.linkage_ratio, 2) + "%");
        LogInfo("  Max depth: " + std::to_string(linkage_stats.max_depth) +
                " Mean depth: " + FormatFloat(linkage_stats.mean_depth, 2));
        LogInfo("  Linkage Error: Max=" + FormatFloat(mse_stats.max, 4) +
                ", Min=" + FormatFloat(mse_stats.min, 4) +
                ", Mean=" + FormatFloat(mse_stats.mean, 4));
    }

    CodebookPack one_pack;
    ColMajorMatrix<float> R_in;
    if (!BuildCOneFromLinkage(config, X_rot, R_full, train_linkage, B, a,
                            root_pack, &one_pack, &R_in)) {
        if (error) {
            *error = "Failed to build C_one from linkage residuals.";
        }
        return false;
    }

    // Julia reference: analyze_linkaged_reconstruction_itself_twobook1 (utils.jl:116-162).
    ColMajorMatrix<float> R_full_updated;
    try {
        R_full_updated = ComputeLinkagedReconstructionMultiBook(root_pack, one_pack, B, a, train_linkage);
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Failed to recompute linkaged reconstruction after C_one update: ") + e.what();
        }
        return false;
    }
    if (config.train.log_metrics) {
        LogInfo("(4)Linkage encode analysis after updating C_one:");
        LinkageDepthStats linkage_stats = ComputeLinkageDepthStats(train_linkage, n);
        MseStats mse_stats = MseStatsFromRecon(X_rot, R_full_updated);
        LogInfo("  Linkage Ratio: " + FormatFloat(100.0f * linkage_stats.linkage_ratio, 2) + "%");
        LogInfo("  Max depth: " + std::to_string(linkage_stats.max_depth) +
                " Mean depth: " + FormatFloat(linkage_stats.mean_depth, 2));
        LogInfo("  Linkage Error: Max=" + FormatFloat(mse_stats.max, 4) +
                ", Min=" + FormatFloat(mse_stats.min, 4) +
                ", Mean=" + FormatFloat(mse_stats.mean, 4));
        state->linkage_mse_mean = mse_stats.mean;
        state->linkage_ratio = linkage_stats.linkage_ratio;
        state->linkage_max_depth = linkage_stats.max_depth;
        state->linkage_mean_depth = linkage_stats.mean_depth;
    }
    R_full = std::move(R_full_updated);

    state->C_root = root_pack;
    state->C_one = one_pack;
    state->B = B;
    state->a = a;
    state->R_full = std::move(R_full);
    state->R_in = std::move(R_in);
    state->train_linkage = std::move(train_linkage);
    return true;
}

bool LinkageRefineOnRotatedDataBeam(const Config& config,
                                  const ColMajorMatrix<float>& X_rot,
                                  TrainState* state,
                                  std::string* error) {
    if (!state) {
        if (error) {
            *error = "TrainState output is null.";
        }
        return false;
    }
    if (state->C_root.books.empty() || state->C_one.books.empty()) {
        if (error) {
            *error = "TrainState codebooks are empty.";
        }
        return false;
    }

    const int d = X_rot.rows;
    const int n = X_rot.cols;
    if (d <= 0 || n <= 0) {
        if (error) {
            *error = "Empty training data for linkage_refine_on_rotated_data_beam.";
        }
        return false;
    }
    if (state->B.cols != n || state->a.cols != n) {
        if (error) {
            *error = "TrainState B/a do not match rotated data columns.";
        }
        return false;
    }

    // Julia reference: linkage_refine_on_rotated_data_beam!
    // (train_quantizer.jl:247-360).
    Precomp pre_root;
    if (!BuildPrecomp(state->C_root, &pre_root)) {
        if (error) {
            *error = "Failed to build precomp for C_root.";
        }
        return false;
    }

    ColMajorMatrix<float> xC(pre_root.H, n);
    {
        ScopedBlasThreads blas_scope(OmpMaxThreads());
        Gemm(true, false, 1.0f, pre_root.C_all, X_rot, 0.0f, &xC);
    }

    // Beam init (train_quantizer.jl:271-291).
    BeamSearchPrefixLS(X_rot, pre_root, xC, std::max(1, config.base.encode.H_beam), &state->B);
    SolveLeastSquaresAll(pre_root, xC, state->B, &state->a);
    float beam_mse = MeanMseFromPrecomp(X_rot, pre_root, xC, state->B, state->a);
    if (config.train.log_metrics) {
        LogInfo("(1)Basic encode init with beam search mse error : " + FormatFloat(beam_mse, 6));
    }
    state->beam_mse = beam_mse;

    // ICM/ILS refinement (train_quantizer.jl:299-307).
    std::vector<float> X_norm2;
    std::vector<float> train_cost;
#if defined(STLQ_ENABLE_CUDA)
    if (config.runtime.use_cuda) {
        DynamicIcmWithIlsNoAbsNoNormalCuda(X_rot, pre_root, xC,
                                           config.train.icm_iters,
                                           config.train.ils_iters,
                                           config.train.perturb_k,
                                           static_cast<std::uint32_t>(config.train.seed),
                                           &state->B, &state->a, &X_norm2, &train_cost,
                                           /*print_progress=*/false,
                                           /*sample_id_offset=*/0);
    } else
#endif
    {
        DynamicIcmWithIlsNoAbsNoNormal(X_rot, pre_root, xC,
                                       config.train.icm_iters,
                                       config.train.ils_iters,
                                       config.train.perturb_k,
                                       static_cast<std::uint32_t>(config.train.seed),
                                       &state->B, &state->a, &X_norm2, &train_cost);
    }

    // Update C_root (train_quantizer.jl:311-319).
    state->C_root.books = UpdateCodebooksLS(X_rot, state->B, state->a, config.model.h_vec);
    float pre_linkage_mse = MeanMseFromCodebooks(X_rot, state->C_root.books, state->B, state->a);
    if (config.train.log_metrics) {
        LogInfo("(2)Basic encode ILS+ICM mse error after updating C_root: " + FormatFloat(pre_linkage_mse, 6));
    }
    state->icm_mse = pre_linkage_mse;

    // Julia reference: linkage_from_inner_to_outer_two_codebook_ils!
    // (train_quantizer.jl:321-337, linkage_quantizer.jl:1131-1668).
    BaseEncoding train_base;
    train_base.B = std::move(state->B);
    train_base.a = std::move(state->a);
    LinkageStructure train_linkage;
    ColMajorMatrix<float> R_full;

    if (!BuildLinkageTwoCodebook(config.train.linkage, config.hnsw,
                               X_rot, &train_base,
                               state->C_root, state->C_one,
                               &train_linkage, &R_full, error)) {
        return false;
    }
    state->B = std::move(train_base.B);
    state->a = std::move(train_base.a);
    state->train_linkage = std::move(train_linkage);
    state->R_full = std::move(R_full);

    if (config.train.log_metrics && config.train.log_linkage_pre_c1) {
        LogInfo("(3)Linkage encode analysis before updating C_one:");
        LinkageDepthStats linkage_stats = ComputeLinkageDepthStats(state->train_linkage, n);
        MseStats mse_stats = MseStatsFromRecon(X_rot, state->R_full);
        LogInfo("  Linkage Ratio: " + FormatFloat(100.0f * linkage_stats.linkage_ratio, 2) + "%");
        LogInfo("  Max depth: " + std::to_string(linkage_stats.max_depth) +
                " Mean depth: " + FormatFloat(linkage_stats.mean_depth, 2));
        LogInfo("  Linkage Error: Max=" + FormatFloat(mse_stats.max, 4) +
                ", Min=" + FormatFloat(mse_stats.min, 4) +
                ", Mean=" + FormatFloat(mse_stats.mean, 4));
    }

    // LogInfo("update one depth codebook ...");
    if (!BuildCOneFromLinkage(config, X_rot, state->R_full, state->train_linkage,
                            state->B, state->a,
                            state->C_root, &state->C_one, &state->R_in)) {
        if (error) {
            *error = "Failed to update C_one from linkage residuals.";
        }
        return false;
    }

    try {
        state->R_full = ComputeLinkagedReconstructionMultiBook(state->C_root, state->C_one,
                                                              state->B, state->a,
                                                              state->train_linkage);
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Failed to recompute linkaged reconstruction after C_one update: ") + e.what();
        }
        return false;
    }

    if (config.train.log_metrics) {
        LinkageDepthStats linkage_stats = ComputeLinkageDepthStats(state->train_linkage, n);
        MseStats mse_stats = MseStatsFromRecon(X_rot, state->R_full);
        LogInfo("(3)Linkage encode analysis after updating C_one:");
        LogInfo("  Linkage Ratio: " + FormatFloat(100.0f * linkage_stats.linkage_ratio, 2) + "%");
        LogInfo("  Max depth: " + std::to_string(linkage_stats.max_depth) +
                " Mean depth: " + FormatFloat(linkage_stats.mean_depth, 2));
        LogInfo("  Linkage Error: Max=" + FormatFloat(mse_stats.max, 4) +
                ", Min=" + FormatFloat(mse_stats.min, 4) +
                ", Mean=" + FormatFloat(mse_stats.mean, 4));
        state->linkage_mse_mean = mse_stats.mean;
        state->linkage_ratio = linkage_stats.linkage_ratio;
        state->linkage_max_depth = linkage_stats.max_depth;
        state->linkage_mean_depth = linkage_stats.mean_depth;
    }
    return true;
}


bool TrainQuantizer(const Config& config,
                    const Dataset& dataset,
                    KernelProvider* kernels,
                    StreamKernelProvider* stream_kernels,
                    TrainResult* result,
                    std::string* error) {
    (void)kernels;
    if (!result) {
        if (error) {
            *error = "TrainResult output is null.";
        }
        return false;
    }

    const int d = dataset.Xt.rows;
    const int n = dataset.Xt.cols;
    if (d <= 0 || n <= 0) {
        if (error) {
            *error = "Empty training dataset.";
        }
        return false;
    }

    TrainState state;
    const bool virtual_mode = config.virtual_cfg.enabled;
    // Julia reference:
    // - non-virtual: train_quantizer_and_linkage_beam! (train_quantizer.jl:118-232)
    // - virtual: train_quantizer_and_linkage_virtual_beam! (train_quantizer.jl:587-742)
    LogInfo("============== Baseline training with R = I (no global rotation) ==============");
    if (!(virtual_mode ? virtual_train::TrainQuantizerAndLinkageVirtualBeam(config, dataset.Xt, stream_kernels, &state, error)
                       : TrainQuantizerAndLinkageBeam(config, dataset.Xt, stream_kernels, &state, error))) {
        return false;
    }

    // Julia reference: train_with_global_rotation_beam! OPQ iterations
    // (train_quantizer.jl:35-84).
    ColMajorMatrix<float> R(d, d);
    std::fill(R.data.begin(), R.data.end(), 0.0f);
    for (int i = 0; i < d; ++i) {
        R(i, i) = 1.0f;
    }

    const int max_iters = std::max(0, config.train.max_R_iters);
    float final_linkaged_mse_rot = -1.0f;
    for (int iter = 0; iter < max_iters; ++iter) {
        LogInfo("============== Global R iteration " + std::to_string(iter + 1) + " / " +
                std::to_string(max_iters) + " ==============");
        const ColMajorMatrix<float>* X_in = &dataset.Xt;
        ColMajorMatrix<float> X_rot;
        if (config.train.use_opq_rotation) {
            X_rot = ColMajorMatrix<float>(d, n);
            ScopedBlasThreads blas_scope(OmpMaxThreads());
            Gemm(false, false, 1.0f, R, dataset.Xt, 0.0f, &X_rot);
            X_in = &X_rot;
        }
        if (!(virtual_mode ? virtual_train::LinkageRefineOnRotatedDataVirtualBeam(config, *X_in, &state, error)
                           : LinkageRefineOnRotatedDataBeam(config, *X_in, &state, error))) {
            return false;
        }

        if (config.train.use_opq_rotation) {
            if (!UpdateOpqRotation(state.R_full, dataset.Xt, &R)) {
                LogWarn("OPQ rotation update failed; keeping previous R.");
            }
        } else {
            LogInfo("[R-iter " + std::to_string(iter + 1) +
                    "] skip UpdateOpqRotation (train.use_opq_rotation=false)");
        }
    }

    result->C_root = state.C_root;
    result->C_one = state.C_one;
    result->train_linkage = state.train_linkage;
    result->R = R;
    if (virtual_mode) {
        result->is_bad_cluster = state.is_bad_cluster;
    } else {
        result->is_bad_cluster.clear();
    }
    result->init_mse = state.init_mse;
    result->beam_mse = state.beam_mse;
    result->icm_mse = state.icm_mse;
    result->linkage_mse_mean = state.linkage_mse_mean;
    result->linkage_ratio = state.linkage_ratio;
    result->linkage_max_depth = state.linkage_max_depth;
    result->linkage_mean_depth = state.linkage_mean_depth;
    result->R_iters = max_iters;
    result->final_linkaged_mse_rot = final_linkaged_mse_rot;
    return true;
}
