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
