bool BuildLinkageOneInit(const LinkageBuildConfig& linkage_cfg,
                const HnswConfig& hnsw_cfg,
                const ColMajorMatrix<float>& X,
                BaseEncoding* base,
                const CodebookPack& C_root,
                LinkageStructure* linkage,
                ColMajorMatrix<float>* R_full,
                std::string* error) {
    if (!linkage || !base) {
        if (error) {
            *error = "LinkageStructure/BaseEncoding output is null.";
        }
        return false;
    }
    linkage->clusters.clear();
    linkage->max_depth = 0;

    const int n = base->B.cols;
    if (n <= 0 || X.cols != n) {
        if (error) {
            *error = "Linkage build expects X columns to match encoded base size.";
        }
        return false;
    }
    if (C_root.books.empty()) {
        if (error) {
            *error = "C_root is empty.";
        }
        return false;
    }


    Precomp precomp;
    if (!BuildPrecomp(C_root, &precomp)) {
        if (error) {
            *error = "Failed to build precomp for linkage build.";
        }
        return false;
    }

    if (!IsSupportedTrainReferencePolicy(linkage_cfg.reference_policy)) {
        if (error) {
            *error = "Unsupported train linkage reference policy: " +
                     linkage_cfg.reference_policy;
        }
        return false;
    }
    if (IsUnrestrictedReferenceForestPolicy(linkage_cfg.reference_policy)) {
        const int h0 = C_root.books.front().cols;
        std::vector<std::vector<int>> clusters(static_cast<std::size_t>(h0));
        for (int i = 0; i < n; ++i) {
            const int code = static_cast<int>(base->B(0, i));
            clusters[static_cast<std::size_t>(code)].push_back(i);
        }
        std::vector<LinkageStructure::Cluster> cluster_out(static_cast<std::size_t>(h0));
        std::vector<std::uint8_t> cluster_used(static_cast<std::size_t>(h0), 0);
        std::atomic<bool> ok{true};
        std::string first_error;
        std::mutex error_mutex;
        std::atomic<std::uint64_t> cycle_cuts{0};
        std::atomic<std::uint64_t> depth_cuts{0};

#pragma omp parallel for schedule(dynamic)
        for (int code = 0; code < h0; ++code) {
            const auto& globals = clusters[static_cast<std::size_t>(code)];
            const int count = static_cast<int>(globals.size());
            if (count == 0 || !ok.load(std::memory_order_relaxed)) continue;
            ColMajorMatrix<float> X_local(X.rows, count);
            ColMajorMatrix<FullCode> B_local(base->B.rows, count);
            ColMajorMatrix<float> a_local(base->a.rows, count);
            for (int local = 0; local < count; ++local) {
                const int global = globals[static_cast<std::size_t>(local)];
                std::memcpy(X_local.Col(local), X.Col(global),
                            sizeof(float) * static_cast<std::size_t>(X.rows));
                std::memcpy(B_local.Col(local), base->B.Col(global),
                            sizeof(FullCode) * static_cast<std::size_t>(base->B.rows));
                std::memcpy(a_local.Col(local), base->a.Col(global),
                            sizeof(float) * static_cast<std::size_t>(base->a.rows));
            }
            LinkageStructure::Cluster built;
            ReferenceForest forest;
            int ls_failures = 0;
            std::string local_error;
            const bool use_ils = linkage_cfg.use_ils && linkage_cfg.ils_rounds > 0 &&
                                 linkage_cfg.ils_perturb_layers > 0;
            const bool built_ok =
                use_ils
                    ? LinkageTwoCodebookReferenceForestOneCluster<true>(
                          linkage_cfg, hnsw_cfg, code, X_local,
                          C_root, precomp, C_root, precomp, precomp.G,
                          &B_local, &a_local, &built, &ls_failures,
                          nullptr, nullptr, &forest, &local_error)
                    : LinkageTwoCodebookReferenceForestOneCluster<false>(
                          linkage_cfg, hnsw_cfg, code, X_local,
                          C_root, precomp, C_root, precomp, precomp.G,
                          &B_local, &a_local, &built, &ls_failures,
                          nullptr, nullptr, &forest, &local_error);
            if (!built_ok) {
                ok.store(false, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (first_error.empty()) first_error = local_error;
                continue;
            }
            cycle_cuts.fetch_add(static_cast<std::uint64_t>(forest.cycle_cuts),
                                 std::memory_order_relaxed);
            depth_cuts.fetch_add(static_cast<std::uint64_t>(forest.depth_cuts),
                                 std::memory_order_relaxed);
            for (int local = 0; local < count; ++local) {
                const int global = globals[static_cast<std::size_t>(local)];
                std::memcpy(base->B.Col(global), B_local.Col(local),
                            sizeof(FullCode) * static_cast<std::size_t>(base->B.rows));
                std::memcpy(base->a.Col(global), a_local.Col(local),
                            sizeof(float) * static_cast<std::size_t>(base->a.rows));
            }
            for (int& local : built.indices) {
                local = globals[static_cast<std::size_t>(local)];
            }
            cluster_out[static_cast<std::size_t>(code)] = std::move(built);
            cluster_used[static_cast<std::size_t>(code)] = 1;
        }
        if (!ok.load(std::memory_order_relaxed)) {
            if (error) *error = first_error.empty() ? "Reference-forest train linkage failed." : first_error;
            return false;
        }
        linkage->clusters.clear();
        linkage->max_depth = 0;
        for (int code = 0; code < h0; ++code) {
            if (!cluster_used[static_cast<std::size_t>(code)]) continue;
            linkage->max_depth = std::max(
                linkage->max_depth,
                static_cast<int>(cluster_out[static_cast<std::size_t>(code)].depth_offsets.size()) - 2);
            linkage->clusters.push_back(std::move(cluster_out[static_cast<std::size_t>(code)]));
        }
        if (R_full) {
            *R_full = ComputeLinkagedReconstructionMultiBook(C_root, C_root,
                                                             base->B, base->a,
                                                             *linkage);
        }
        LogInfo("Reference forest train (" + linkage_cfg.reference_policy +
                "): cycle_cuts=" + std::to_string(cycle_cuts.load()) +
                ", depth_cuts=" + std::to_string(depth_cuts.load()));
        return true;
    }

    if (linkage_cfg.use_ils && linkage_cfg.ils_rounds > 0 && linkage_cfg.ils_perturb_layers > 0) {
        LinkageFromInnerToOuterOneCodebookImpl<true>(linkage_cfg, hnsw_cfg, X, C_root, precomp,
                                                   &base->B, &base->a,
                                                   linkage, R_full,
                                                   linkage_cfg.ils_rounds,
                                                   linkage_cfg.ils_perturb_layers);
    } else {
        LinkageFromInnerToOuterOneCodebookImpl<false>(linkage_cfg, hnsw_cfg, X, C_root, precomp,
                                                    &base->B, &base->a,
                                                    linkage, R_full,
                                                    0, 0);
    }

    return true;
}

bool BuildLinkageTwoCodebook(const LinkageBuildConfig& linkage_cfg,
                           const HnswConfig& hnsw_cfg,
                           const ColMajorMatrix<float>& X,
                           BaseEncoding* base,
                           const CodebookPack& C_root,
                           const CodebookPack& C_one,
                           LinkageStructure* linkage,
                           ColMajorMatrix<float>* R_full,
                           std::string* error) {
    if (!linkage || !base) {
        if (error) {
            *error = "LinkageStructure/BaseEncoding output is null.";
        }
        return false;
    }
    linkage->clusters.clear();
    linkage->max_depth = 0;

    const int n = base->B.cols;
    if (n <= 0 || X.cols != n) {
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
    // Julia reference: cluster_indices from first layer codes (linkage_quantizer.jl:1170-1185).
    std::vector<std::vector<int>> clusters(static_cast<std::size_t>(h0));
    for (int i = 0; i < n; ++i) {
        int code = static_cast<int>((base->B)(0, i));
        // if (code < 0 || code >= h0) continue;
        clusters[static_cast<std::size_t>(code)].push_back(i);
    }

    if (!IsSupportedReferencePolicy(linkage_cfg.reference_policy)) {
        if (error) {
            *error = "Unsupported base linkage reference policy: " +
                     linkage_cfg.reference_policy;
        }
        return false;
    }
    if (IsUnrestrictedReferenceForestPolicy(linkage_cfg.reference_policy)) {
        ColMajorMatrix<float> G_one_root;
        {
            ScopedBlasThreads blas_scope(std::max(1, omp_get_max_threads()));
            G_one_root = BuildOneRootCrossGram(pre_one, pre_root);
        }
        std::vector<LinkageStructure::Cluster> cluster_out(static_cast<std::size_t>(h0));
        std::vector<std::uint8_t> cluster_used(static_cast<std::size_t>(h0), 0);
        std::atomic<bool> ok{true};
        std::string first_error;
        std::mutex error_mutex;
        std::atomic<std::uint64_t> cycle_cuts{0};
        std::atomic<std::uint64_t> depth_cuts{0};

#pragma omp parallel for schedule(dynamic)
        for (int code = 0; code < h0; ++code) {
            const auto& globals = clusters[static_cast<std::size_t>(code)];
            const int count = static_cast<int>(globals.size());
            if (count == 0 || !ok.load(std::memory_order_relaxed)) continue;

            ColMajorMatrix<float> X_local(X.rows, count);
            ColMajorMatrix<FullCode> B_local(base->B.rows, count);
            ColMajorMatrix<float> a_local(base->a.rows, count);
            for (int local = 0; local < count; ++local) {
                const int global = globals[static_cast<std::size_t>(local)];
                std::memcpy(X_local.Col(local), X.Col(global),
                            sizeof(float) * static_cast<std::size_t>(X.rows));
                std::memcpy(B_local.Col(local), base->B.Col(global),
                            sizeof(FullCode) * static_cast<std::size_t>(base->B.rows));
                std::memcpy(a_local.Col(local), base->a.Col(global),
                            sizeof(float) * static_cast<std::size_t>(base->a.rows));
            }

            LinkageStructure::Cluster built;
            ReferenceForest forest;
            int ls_failures = 0;
            std::string local_error;
            const bool use_ils = linkage_cfg.use_ils && linkage_cfg.ils_rounds > 0 &&
                                 linkage_cfg.ils_perturb_layers > 0;
            const bool built_ok =
                use_ils
                    ? LinkageTwoCodebookReferenceForestOneCluster<true>(
                          linkage_cfg, hnsw_cfg, code, X_local,
                          C_root, pre_root, C_one, pre_one, G_one_root,
                          &B_local, &a_local, &built, &ls_failures,
                          nullptr, nullptr, &forest, &local_error)
                    : LinkageTwoCodebookReferenceForestOneCluster<false>(
                          linkage_cfg, hnsw_cfg, code, X_local,
                          C_root, pre_root, C_one, pre_one, G_one_root,
                          &B_local, &a_local, &built, &ls_failures,
                          nullptr, nullptr, &forest, &local_error);
            if (!built_ok) {
                ok.store(false, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (first_error.empty()) first_error = local_error;
                continue;
            }
            cycle_cuts.fetch_add(static_cast<std::uint64_t>(forest.cycle_cuts),
                                 std::memory_order_relaxed);
            depth_cuts.fetch_add(static_cast<std::uint64_t>(forest.depth_cuts),
                                 std::memory_order_relaxed);
            for (int local = 0; local < count; ++local) {
                const int global = globals[static_cast<std::size_t>(local)];
                std::memcpy(base->B.Col(global), B_local.Col(local),
                            sizeof(FullCode) * static_cast<std::size_t>(base->B.rows));
                std::memcpy(base->a.Col(global), a_local.Col(local),
                            sizeof(float) * static_cast<std::size_t>(base->a.rows));
            }
            for (int& local : built.indices) {
                local = globals[static_cast<std::size_t>(local)];
            }
            cluster_out[static_cast<std::size_t>(code)] = std::move(built);
            cluster_used[static_cast<std::size_t>(code)] = 1;
        }
        if (!ok.load(std::memory_order_relaxed)) {
            if (error) *error = first_error.empty() ? "Reference-forest linkage failed." : first_error;
            return false;
        }

        linkage->clusters.clear();
        linkage->max_depth = 0;
        for (int code = 0; code < h0; ++code) {
            if (!cluster_used[static_cast<std::size_t>(code)]) continue;
            linkage->max_depth = std::max(
                linkage->max_depth,
                static_cast<int>(cluster_out[static_cast<std::size_t>(code)].depth_offsets.size()) - 2);
            linkage->clusters.push_back(std::move(cluster_out[static_cast<std::size_t>(code)]));
        }
        if (R_full) {
            *R_full = ComputeLinkagedReconstructionMultiBook(C_root, C_one,
                                                             base->B, base->a,
                                                             *linkage);
        }
        LogInfo("Reference forest (" + linkage_cfg.reference_policy +
                "): cycle_cuts=" + std::to_string(cycle_cuts.load()) +
                ", depth_cuts=" + std::to_string(depth_cuts.load()));
        return true;
    }

    if (linkage_cfg.use_ils && linkage_cfg.ils_rounds > 0 && linkage_cfg.ils_perturb_layers > 0) {
        LinkageFromInnerToOuterTwoCodebookImplCore<true>(linkage_cfg, hnsw_cfg, X, clusters,
                                                 C_root, pre_root, C_one, pre_one,
                                                 &base->B, &base->a,
                                                 linkage, R_full,
                                                 linkage_cfg.ils_rounds, linkage_cfg.ils_perturb_layers);
    } else {
        LinkageFromInnerToOuterTwoCodebookImplCore<false>(linkage_cfg, hnsw_cfg, X, clusters,
                                         C_root, pre_root, C_one, pre_one,
                                         &base->B, &base->a,
                                         linkage, R_full,
                                         0, 0);
    }

    return true;
}

std::vector<int> BuildGlobalParent(const LinkageStructure& linkage, int n_total) {
    std::vector<int> parent(static_cast<std::size_t>(n_total), 0);
    const std::size_t nclusters = linkage.clusters.size();
#pragma omp parallel for schedule(static)
    for (int ci = 0; ci < static_cast<int>(nclusters); ++ci) {
        const auto& cluster = linkage.clusters[static_cast<std::size_t>(ci)];
        const std::size_t csize = cluster.indices.size();
        for (std::size_t i = 0; i < csize; ++i) {
            int g = cluster.indices[i];
            // if (g >= static_cast<Index>(n_total)) continue;
            int plocal = cluster.parent_local[i];
            if (plocal == 0) {
                parent[static_cast<std::size_t>(g)] = 0;
            } else {
                int gparent = cluster.indices[static_cast<std::size_t>(plocal - 1)];
                parent[static_cast<std::size_t>(g)] = gparent + 1;
            }
        }
    }
    return parent;
}

std::vector<int> BuildGlobalParentVirtual(const LinkageStructure& linkage,
                                         int n_base_real,
                                         const VirtualEncoding& virt) {
    const int n_total = n_base_real + virt.n_virtual();
    std::vector<int> parent(static_cast<std::size_t>(n_total), 0);

    // Prefix offsets for virtual global ids (by cluster id).
    const int nlist = virt.nlist;
    std::vector<int> virt_prefix(static_cast<std::size_t>(std::max(0, nlist) + 1), 0);
    for (int cid = 0; cid < nlist; ++cid) {
        virt_prefix[static_cast<std::size_t>(cid + 1)] =
            virt_prefix[static_cast<std::size_t>(cid)] + virt.n_virtual(cid);
    }

    const std::size_t nclusters = linkage.clusters.size();
#pragma omp parallel for schedule(static)
    for (int ci = 0; ci < static_cast<int>(nclusters); ++ci) {
        const auto& cluster = linkage.clusters[static_cast<std::size_t>(ci)];
        const int cid = cluster.cluster_id;
        const int nvirt = cluster.n_virtual;
        const int nreal = cluster.n_real;
        if (nreal <= 0 || cluster.indices.empty()) {
            continue;
        }
        if (static_cast<int>(cluster.indices.size()) != nreal) {
            continue;
        }
        if (static_cast<int>(cluster.parent_local.size()) < nvirt + nreal) {
            continue;
        }

        const int vbase = (cid >= 0 && cid < nlist) ? virt_prefix[static_cast<std::size_t>(cid)] : 0;

        for (int real_idx = 0; real_idx < nreal; ++real_idx) {
            const int g = cluster.indices[static_cast<std::size_t>(real_idx)];
            if (g < 0 || g >= n_base_real) {
                continue;
            }
            const int local_real = nvirt + real_idx;
            const int plocal1 = cluster.parent_local[static_cast<std::size_t>(local_real)];
            if (plocal1 == 0) {
                parent[static_cast<std::size_t>(g)] = 0;
                continue;
            }
            const int plocal0 = plocal1 - 1;
            int gparent = -1;
            if (plocal0 < nvirt) {
                // Parent is a virtual root in this cluster.
                gparent = n_base_real + vbase + plocal0;
            } else {
                const int parent_real_idx = plocal0 - nvirt;
                if (parent_real_idx < 0 || parent_real_idx >= nreal) {
                    continue;
                }
                gparent = cluster.indices[static_cast<std::size_t>(parent_real_idx)];
            }
            if (gparent >= 0 && gparent < n_total) {
                parent[static_cast<std::size_t>(g)] = gparent + 1;
            }
        }
    }
    return parent;
}
