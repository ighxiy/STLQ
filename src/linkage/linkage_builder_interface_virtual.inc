bool ProcessGoodClustersVirtual(const LinkageBuildConfig& linkage_cfg,
                                          const HnswConfig& hnsw_cfg,
                                          const ColMajorMatrix<float>& X,
                                          std::vector<std::vector<int>> cols_by_cluster,
                                          ColMajorMatrix<FullCode>* B,
                                          ColMajorMatrix<float>* a,
                                          const CodebookPack& C_root,
                                          const CodebookPack& C_one,
                                          LinkageStructure* linkage,
                                          ColMajorMatrix<float>* R_full,
                                          std::string* error) {
    if (!linkage || !B || !a) {
        if (error) {
            *error = "LinkageStructure/B/a output is null.";
        }
        return false;
    }
    linkage->clusters.clear();
    linkage->max_depth = 0;

    const int n = B->cols;
    if (n <= 0 || X.cols != n || a->cols != n) {
        if (error) {
            *error = "Linkage build expects X columns to match encoded base size.";
        }
        return false;
    }
    if (C_root.books.empty() || C_one.books.empty()) {
        if (error) {
            *error = "C_root/C_one are empty.";
        }
        return false;
    }

    Precomp pre_root;
    if (!BuildPrecomp(C_root, &pre_root)) {
        if (error) {
            *error = "Failed to build precomp for C_root.";
        }
        return false;
    }
    Precomp pre_one;
    if (!BuildPrecomp(C_one, &pre_one)) {
        if (error) {
            *error = "Failed to build precomp for C_one.";
        }
        return false;
    }

    const int h0 = C_root.books.front().cols;
    if (h0 <= 0) {
        if (error) {
            *error = "C_root has invalid h0.";
        }
        return false;
    }
    if (static_cast<int>(cols_by_cluster.size()) < h0) {
        if (error) {
            *error = "ProcessGoodClustersVirtual: cols_by_cluster too small for C_root h0.";
        }
        return false;
    }
    if (h0 == 0) {
        if (R_full) {
            *R_full = ReconstructAll(C_root.books, *B, *a);
        }
        return true;
    }

    if (linkage_cfg.use_ils && linkage_cfg.ils_rounds > 0 && linkage_cfg.ils_perturb_layers > 0) {
        LinkageFromInnerToOuterTwoCodebookImplCore<true>(linkage_cfg, hnsw_cfg, X, cols_by_cluster,
                                                       C_root, pre_root, C_one, pre_one,
                                                       B, a,
                                                       linkage, R_full,
                                                       linkage_cfg.ils_rounds,
                                                       linkage_cfg.ils_perturb_layers);
    } else {
        LinkageFromInnerToOuterTwoCodebookImplCore<false>(linkage_cfg, hnsw_cfg, X, cols_by_cluster,
                                                        C_root, pre_root, C_one, pre_one,
                                                        B, a,
                                                        linkage, R_full,
                                                        0, 0);
    }
    return true;
}


bool ProcessBadClustersVirtualForTrain(const Config& config,
                               const ColMajorMatrix<float>& X_rot,
                               const std::vector<std::vector<int>>& bad_cols_by_cid,
                               const std::vector<int>& bad_cids,
                               const LinkageBuildConfig& linkage_cfg,
                               const CodebookPack& C_root,
                               const CodebookPack& C_one,
                               const Precomp& pre_root,
                               const Precomp& pre_one,
                               const ColMajorMatrix<float>& G_one_root,
                               ColMajorMatrix<FullCode>* B,
                               ColMajorMatrix<float>* a,
                               ColMajorMatrix<float>* R_full_shared,  // shared reconstruction buffer (d x n)
                               std::vector<BadClusterCache>* bad_cache,
                               std::string* error) {
    if (!B || !a || !R_full_shared || !bad_cache) {
        if (error) {
            *error = "ProcessBadClustersVirtualForTrain: null output.";
        }
        return false;
    }
    const int d = X_rot.rows;
    if (d <= 0) {
        if (error) {
            *error = "ProcessBadClustersVirtualForTrain: empty X_rot.";
        }
        return false;
    }

    bad_cache->assign(bad_cids.size(), BadClusterCache{});
    if (bad_cids.empty()) {
        return true;
    }

    std::atomic<int> done{0};
    std::atomic<bool> ok{true};
    std::string first_error;
    std::mutex err_mu;

#pragma omp parallel
    {
        std::vector<std::uint32_t> knn_ids_flat;
        std::vector<float> knn_dists_flat;
        std::string local_error;

#pragma omp for schedule(dynamic)
        for (int idx = 0; idx < static_cast<int>(bad_cids.size()); ++idx) {
            if (!ok.load(std::memory_order_relaxed)) {
                continue;
            }
            const int cid = bad_cids[static_cast<std::size_t>(idx)];
            const std::vector<int>& cols = bad_cols_by_cid[static_cast<std::size_t>(cid)];
            const int n_real = static_cast<int>(cols.size());
            if (n_real <= 1) {
                int dd = done.fetch_add(1, std::memory_order_relaxed) + 1;
                if (dd % 10 == 0 || dd == static_cast<int>(bad_cids.size())) {
                    PrintClusterProgress(dd, static_cast<int>(bad_cids.size()));
                }
                continue;
            }

            const int mult_bad = std::max(1, config.hnsw.candidate_multiplier_bad);
            int knn_real = std::max(std::max(1, config.virtual_cfg.umap_knn_k),
                                    mult_bad * std::max(1, linkage_cfg.knn_k));
            knn_real = std::min(knn_real, n_real - 1);
            const int knn_umap = std::min(std::max(1, config.virtual_cfg.umap_knn_k), knn_real);

            knn_ids_flat.clear();
            knn_dists_flat.clear();
            local_error.clear();
            if (!BuildClusterKnnHnswImpl(X_rot, cols, config.hnsw, knn_real,
                                         &knn_ids_flat, &knn_dists_flat, &local_error)) {
                std::lock_guard<std::mutex> guard(err_mu);
                ok.store(false, std::memory_order_relaxed);
                if (first_error.empty()) {
                    first_error = local_error;
                }
                continue;
            }

        const int k_virtual = ResolveVirtualCount(n_real, config.virtual_cfg);
        ColMajorMatrix<float> X_virt;
        ColMajorMatrix<FullCode> B_virt;
        ColMajorMatrix<float> a_virt;
        if (k_virtual > 0) {
            const std::uint32_t seed =
                static_cast<std::uint32_t>(config.train.seed + 1337 * cid + 17);
            if (!AddVirtualRootsUmapReencodeClusterFromKnnTables(config.virtual_cfg,
                                                                C_root,
                                                                pre_root,
                                                                nullptr,
                                                                nullptr,
                                                                X_rot,
                                                                cols,
                                                                knn_ids_flat,
                                                                knn_dists_flat,
                                                                knn_real,
                                                                knn_umap,
                                                                cid,
                                                                k_virtual,
                                                                config.train.ils_iters,
                                                                config.train.icm_iters,
                                                                config.train.perturb_k,
                                                                seed,
                                                                &X_virt,
                                                                &B_virt,
                                                                &a_virt,
                                                                &local_error)) {
                std::lock_guard<std::mutex> guard(err_mu);
                ok.store(false, std::memory_order_relaxed);
                if (first_error.empty()) {
                    first_error = local_error;
                }
                continue;
                                                                }
        }

            // Release dist table before linkageing.
            knn_dists_flat.clear();

        LinkageStructure linkage_bad;
        BadClusterCache cache;
        cache.cluster_id = cid;
        cache.n_real = n_real;
        cache.n_virt = B_virt.cols;
        cache.cols_global = cols;
        cache.R_virt = ColMajorMatrix<float>(d, B_virt.cols);

        try {
            if (linkage_cfg.use_ils && linkage_cfg.ils_rounds > 0 && linkage_cfg.ils_perturb_layers > 0) {
                LinkageMultiCenterTwoCodebookImplCore<true>(linkage_cfg, cid, config.hnsw,
#if defined(STLQ_ENABLE_CUDA)
                                                          &config.runtime, nullptr,
#else
                                                          nullptr,
#endif
                                                          X_rot, cols,
                                                          B_virt.data.data(),
                                                          a_virt.data.data(),
                                                          B_virt.cols,
                                                          C_root, pre_root,
                                                          C_one, pre_one,
                                                          G_one_root,
                                                          knn_ids_flat, knn_real,
                                                          B, a,
                                                          R_full_shared, &cache.R_virt,
                                                          &linkage_bad);
            } else {
                LinkageMultiCenterTwoCodebookImplCore<false>(linkage_cfg, cid, config.hnsw,
#if defined(STLQ_ENABLE_CUDA)
                                                           &config.runtime, nullptr,
#else
                                                           nullptr,
#endif
                                                           X_rot, cols,
                                                           B_virt.data.data(),
                                                           a_virt.data.data(),
                                                           B_virt.cols,
                                                           C_root, pre_root,
                                                           C_one, pre_one,
                                                           G_one_root,
                                                           knn_ids_flat, knn_real,
                                                           B, a,
                                                           R_full_shared, &cache.R_virt,
                                                           &linkage_bad);
            }
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> guard(err_mu);
            ok.store(false, std::memory_order_relaxed);
            if (first_error.empty()) {
                first_error = std::string("Multi-center linkage (split) failed: ") + e.what();
            }
            continue;
        }

            knn_ids_flat.clear();

        cache.linkage_aug = std::move(linkage_bad);
        (*bad_cache)[static_cast<std::size_t>(idx)] = std::move(cache);

            int dd = done.fetch_add(1, std::memory_order_relaxed) + 1;
            if (dd % 10 == 0 || dd == static_cast<int>(bad_cids.size())) {
                PrintClusterProgress(dd, static_cast<int>(bad_cids.size()));
            }
        }
    }

    if (!ok.load(std::memory_order_relaxed)) {
        if (error) {
            *error = first_error.empty() ? "Bad cluster processing failed." : first_error;
        }
        return false;
    }
    return true;
}



bool BuildLinkageTwoCodebookVirtual(const LinkageBuildConfig& linkage_cfg,
                                 const HnswConfig& hnsw_cfg,
                                 const VirtualConfig& vcfg,
                                 const ColMajorMatrix<float>& X_real,
                                 BaseEncoding* base_real,
                                 const VirtualEncoding& virt,
                                 const CodebookPack& C_root,
                                 const CodebookPack& C_one,
                                 const std::vector<bool>& is_bad_cluster,
                                 const BadClusterKnnCache* bad_knn_cache,
                                 LinkageStructure* linkage,
                                 ColMajorMatrix<float>* R_full,
                                 std::string* error) {
    (void)vcfg;
    if (!linkage || !base_real) {
        if (error) {
            *error = "LinkageStructure/BaseEncoding output is null.";
        }
        return false;
    }
    linkage->clusters.clear();
    linkage->max_depth = 0;

    const int n_real = base_real->B.cols;
    if (n_real <= 0 || X_real.cols != n_real || base_real->a.cols != n_real) {
        if (error) {
            *error = "Linkage build expects X columns to match encoded base size.";
        }
        return false;
    }
    if (C_root.books.empty() || C_one.books.empty()) {
        if (error) {
            *error = "C_root/C_one are empty.";
        }
        return false;
    }
    const int h0 = C_root.books.front().cols;
    if (h0 <= 0) {
        if (error) {
            *error = "C_root has invalid h0.";
        }
        return false;
    }
    if (static_cast<int>(is_bad_cluster.size()) != h0) {
        if (error) {
            *error = "BuildLinkageTwoCodebookVirtual: is_bad_cluster size mismatch.";
        }
        return false;
    }
    if (virt.n_virtual() > 0) {
        if (virt.m != base_real->B.rows || virt.nlist != h0 ||
            static_cast<int>(virt.B_by_cluster.size()) != h0 ||
            static_cast<int>(virt.a_by_cluster.size()) != h0) {
            if (error) {
                *error = "BuildLinkageTwoCodebookVirtual: virtual encoding shape mismatch.";
            }
            return false;
        }
    }

    // Build per-cluster indices (real vs virtual).
    std::vector<std::vector<int>> good_cols(static_cast<std::size_t>(h0));
    std::vector<std::vector<int>> bad_real_cols(static_cast<std::size_t>(h0));

    for (int i = 0; i < n_real; ++i) {
        const int cid = static_cast<int>(base_real->B(0, i));
        if (cid < 0 || cid >= h0) {
            continue;
        }
        if (is_bad_cluster[static_cast<std::size_t>(cid)]) {
            bad_real_cols[static_cast<std::size_t>(cid)].push_back(i);
        } else {
            good_cols[static_cast<std::size_t>(cid)].push_back(i);
        }
    }

    // Good clusters: standard inner-to-outer linkageing on real points.
    LinkageStructure linkage_good;
    ColMajorMatrix<float> R_full_shared;
    if (!ProcessGoodClustersVirtual(linkage_cfg, hnsw_cfg, X_real, std::move(good_cols),
                                    &base_real->B, &base_real->a,
                                    C_root, C_one,
                                    &linkage_good, &R_full_shared,
                                    error)) {
        return false;
    }

    linkage->clusters = std::move(linkage_good.clusters);
    linkage->max_depth = linkage_good.max_depth;

    // Precompute cross-gram once for bad clusters.
    Precomp pre_root;
    if (!BuildPrecomp(C_root, &pre_root)) {
        if (error) {
            *error = "BuildLinkageTwoCodebookVirtual: failed to build precomp for C_root.";
        }
        return false;
    }
    Precomp pre_one;
    if (!BuildPrecomp(C_one, &pre_one)) {
        if (error) {
            *error = "BuildLinkageTwoCodebookVirtual: failed to build precomp for C_one.";
        }
        return false;
    }
    ColMajorMatrix<float> G_one_root = BuildOneRootCrossGram(pre_one, pre_root);

    std::vector<int> bad_cids;
    bad_cids.reserve(static_cast<std::size_t>(h0));
    for (int cid = 0; cid < h0; ++cid) {
        if (is_bad_cluster[static_cast<std::size_t>(cid)] &&
            !bad_real_cols[static_cast<std::size_t>(cid)].empty()) {
            bad_cids.push_back(cid);
        }
    }

    if (!bad_cids.empty()) {
        std::vector<LinkageStructure::Cluster> bad_clusters(
            static_cast<std::size_t>(bad_cids.size()));
        std::atomic<bool> ok{true};
        std::atomic<int> done{0};
        std::string first_error;
        std::mutex err_mu;

        #pragma omp parallel for schedule(dynamic)
        for (int idx = 0; idx < static_cast<int>(bad_cids.size()); ++idx) {
            if (!ok.load(std::memory_order_relaxed)) {
                continue;
            }
            const int cid = bad_cids[static_cast<std::size_t>(idx)];
            const auto& cols_real = bad_real_cols[static_cast<std::size_t>(cid)];
            const int n_real_local = static_cast<int>(cols_real.size());
            const int n_virt_local = virt.n_virtual(cid);
            if (n_real_local <= 1) {
                int dd = done.fetch_add(1, std::memory_order_relaxed) + 1;
                if (dd % 10 == 0 || dd == static_cast<int>(bad_cids.size())) {
                    PrintClusterProgress(dd, static_cast<int>(bad_cids.size()));
                }
                continue;
            }

            const std::vector<std::uint32_t>* knn_ids_flat = nullptr;
            int knn_real = 0;
            std::vector<std::uint32_t> knn_ids_fallback;
            std::string local_error;
            if (bad_knn_cache && bad_knn_cache->nlist == h0 &&
                static_cast<int>(bad_knn_cache->knn_k_by_cluster.size()) == h0) {
                knn_real = bad_knn_cache->knn_k_by_cluster[static_cast<std::size_t>(cid)];
                if (knn_real > 0) {
                    knn_ids_flat = &bad_knn_cache->knn_ids_by_cluster[static_cast<std::size_t>(cid)];
                }
            }
            if (!knn_ids_flat || knn_real <= 0) {
                // Cache is ephemeral and may be absent when reusing base artifacts that already include
                // virtual nodes (thus skipping AddVirtualNodes). Fall back to building the cluster-local
                // kNN table on-the-fly using the same k heuristic as AddVirtualNodes.
                const int mult_bad = std::max(1, hnsw_cfg.candidate_multiplier_bad);
                int knn_fallback = std::max(std::max(1, vcfg.umap_knn_k),
                                            mult_bad * std::max(1, linkage_cfg.knn_k));
                knn_fallback = std::min(knn_fallback, n_real_local - 1);
                local_error.clear();
                if (!BuildClusterKnnHnswImpl(X_real, cols_real, hnsw_cfg,
                                             knn_fallback,
                                             &knn_ids_fallback, nullptr, &local_error)) {
                    std::lock_guard<std::mutex> guard(err_mu);
                    ok.store(false, std::memory_order_relaxed);
                    if (first_error.empty()) {
                        first_error = local_error.empty()
                                          ? "BuildLinkageTwoCodebookVirtual: failed to build bad-cluster kNN fallback."
                                          : local_error;
                    }
                    int dd = done.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (dd % 10 == 0 || dd == static_cast<int>(bad_cids.size())) {
                        PrintClusterProgress(dd, static_cast<int>(bad_cids.size()));
                    }
                    continue;
                }
                knn_real = knn_fallback;
                knn_ids_flat = &knn_ids_fallback;
            }

            ColMajorMatrix<float> R_virt(X_real.rows, n_virt_local);
            LinkageStructure linkage_bad;
            const FullCode* B_virt_ptr =
                virt.B_by_cluster[static_cast<std::size_t>(cid)].data.data();
            const float* a_virt_ptr =
                virt.a_by_cluster[static_cast<std::size_t>(cid)].data.data();
            try {
                if (linkage_cfg.use_ils && linkage_cfg.ils_rounds > 0 && linkage_cfg.ils_perturb_layers > 0) {
                    LinkageMultiCenterTwoCodebookImplCore<true>(linkage_cfg, cid, hnsw_cfg,
                                                              nullptr,
#if defined(STLQ_ENABLE_CUDA)
                                                              nullptr,
#endif
                                                              X_real, cols_real,
                                                              B_virt_ptr, a_virt_ptr, n_virt_local,
                                                              C_root, pre_root,
                                                              C_one, pre_one,
                                                              G_one_root,
                                                              *knn_ids_flat, knn_real,
                                                              &base_real->B, &base_real->a,
                                                              &R_full_shared, &R_virt,
                                                              &linkage_bad);
                } else {
                    LinkageMultiCenterTwoCodebookImplCore<false>(linkage_cfg, cid, hnsw_cfg,
                                                               nullptr,
#if defined(STLQ_ENABLE_CUDA)
                                                               nullptr,
#endif
                                                               X_real, cols_real,
                                                               B_virt_ptr, a_virt_ptr, n_virt_local,
                                                               C_root, pre_root,
                                                               C_one, pre_one,
                                                               G_one_root,
                                                               *knn_ids_flat, knn_real,
                                                               &base_real->B, &base_real->a,
                                                               &R_full_shared, &R_virt,
                                                               &linkage_bad);
                }
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> guard(err_mu);
                ok.store(false, std::memory_order_relaxed);
                if (first_error.empty()) {
                    first_error = std::string("Multi-center linkage (split) failed: ") + e.what();
                }
                int dd = done.fetch_add(1, std::memory_order_relaxed) + 1;
                if (dd % 10 == 0 || dd == static_cast<int>(bad_cids.size())) {
                    PrintClusterProgress(dd, static_cast<int>(bad_cids.size()));
                }
                continue;
            }

            if (linkage_bad.clusters.empty()) {
                LogWarn("BuildLinkageTwoCodebookVirtual[cid=" + std::to_string(cid) + "]: linkage build succeeded"
                        " but produced no clusters for n_real=" + std::to_string(n_real_local) +
                        "; skipping this bad cluster.");
                continue;
            }
            auto cluster = std::move(linkage_bad.clusters.front());
            cluster.n_real = n_real_local;
            cluster.n_virtual = n_virt_local;
            // `linkage_bad` is built in cluster-local indexing. After prompt13b, `cluster.indices`
            // contains real-only entries (size==n_real_local), and virtual nodes are not mapped into indices.
            if (static_cast<int>(cluster.indices.size()) != n_real_local) {
                std::lock_guard<std::mutex> guard(err_mu);
                ok.store(false, std::memory_order_relaxed);
                if (first_error.empty()) {
                    first_error = "BuildLinkageTwoCodebookVirtual: bad cluster indices size mismatch.";
                }
                continue;
            }
            bool local_ok = true;
            for (int real_idx = 0; real_idx < n_real_local; ++real_idx) {
                const int local_in_cluster = cluster.indices[static_cast<std::size_t>(real_idx)];
                if (local_in_cluster < 0 || local_in_cluster >= n_real_local) {
                    local_ok = false;
                    break;
                }
                cluster.indices[static_cast<std::size_t>(real_idx)] =
                    cols_real[static_cast<std::size_t>(local_in_cluster)];
            }
            if (!local_ok) {
                std::lock_guard<std::mutex> guard(err_mu);
                ok.store(false, std::memory_order_relaxed);
                if (first_error.empty()) {
                    first_error = "BuildLinkageTwoCodebookVirtual: bad cluster indices out of range.";
                }
                int dd = done.fetch_add(1, std::memory_order_relaxed) + 1;
                if (dd % 10 == 0 || dd == static_cast<int>(bad_cids.size())) {
                    PrintClusterProgress(dd, static_cast<int>(bad_cids.size()));
                }
                continue;
            }
            bad_clusters[static_cast<std::size_t>(idx)] = std::move(cluster);

            int dd = done.fetch_add(1, std::memory_order_relaxed) + 1;
            if (dd % 10 == 0 || dd == static_cast<int>(bad_cids.size())) {
                PrintClusterProgress(dd, static_cast<int>(bad_cids.size()));
            }
        }

        if (!ok.load(std::memory_order_relaxed)) {
            if (error) {
                *error = first_error.empty()
                             ? "BuildLinkageTwoCodebookVirtual: bad cluster processing failed."
                             : first_error;
            }
            return false;
        }

        for (auto& cluster : bad_clusters) {
            if (cluster.indices.empty()) {
                continue;
            }
            int max_depth = static_cast<int>(cluster.depth_offsets.size()) - 2;
            linkage->max_depth = std::max(linkage->max_depth, max_depth);
            linkage->clusters.push_back(std::move(cluster));
        }
    }

    if (R_full) {
        *R_full = std::move(R_full_shared);
    }
    return true;
}
