template <bool UseIls>
void LinkageMultiCenterTwoCodebookImplCore(const LinkageBuildConfig& linkage_cfg,
                                                        int cluster_id,
                                                        const HnswConfig& hnsw_cfg,
                                                        const RuntimeConfig* runtime_cfg,
#if defined(STLQ_ENABLE_CUDA)
                                                        CudaStreamKernelsPool* cuda_pool,
#endif
                                                        const ColMajorMatrix<float>& X_real,
                                                        const std::vector<int>& cols_real,
                                                        const FullCode* B_virtual,   // (m x n_virt), col-major
                                                        const float* a_virtual,  // (m x n_virt), col-major
                                                        int n_virt,
                                                        const CodebookPack& C_root,
                                                        const Precomp& pre_root,
                                                        const CodebookPack& C_one,
                                                        const Precomp& pre_one,
                                                        const ColMajorMatrix<float>& G_one_root,
                                                        const std::vector<std::uint32_t>& knn_ids_flat,
                                                        int knn_real,
                                                        ColMajorMatrix<FullCode>* B_real,
                                                        ColMajorMatrix<float>* a_real,
                                                        ColMajorMatrix<float>* R_full_shared,
                                                        ColMajorMatrix<float>* R_virtual,
                                                        LinkageStructure* linkage) {
    using stlq::linkage::LinkageBuildProfileEnabled;
    using stlq::linkage::LinkageBuildWallNowS;
    using stlq::linkage::GetLinkageBuildProfileTls;
    const int d = X_real.rows;
    const int n_real = static_cast<int>(cols_real.size());
    const int n_total = n_real + n_virt;
    const int real_base = n_virt;
    const int m = pre_one.m;
    const int H = pre_one.H;
    const bool can_cpu_fallback = !pre_one.G.data.empty();
    [[maybe_unused]] const bool use_large_root_eval = !can_cpu_fallback;
    (void)runtime_cfg;

    if (!linkage || !B_real || !a_real || !R_full_shared || !R_virtual) {
        return;
    }
    if (d <= 0 || n_real <= 0 || n_virt < 0) {
        linkage->clusters.clear();
        linkage->max_depth = 0;
        return;
    }
    if (n_virt > 0 && (!B_virtual || !a_virtual)) {
        throw std::runtime_error("LinkageMultiCenterTwoCodebookImplCore: virtual pointers are null.");
    }
    if (R_full_shared->rows != d) {
        throw std::runtime_error("LinkageMultiCenterTwoCodebookImplCore: R_full_shared shape mismatch.");
    }
    if (R_virtual->rows != d || R_virtual->cols != n_virt) {
        throw std::runtime_error("LinkageMultiCenterTwoCodebookImplCore: R_virtual shape mismatch.");
    }

    const int knn_k = std::max(1, linkage_cfg.knn_k);
    const int max_depth = std::max(1, linkage_cfg.max_depth);
    const int icm_round = std::max(0, linkage_cfg.icm_round);

    knn_real = std::min(std::max(1, knn_real), std::max(1, n_real - 1));
    if (static_cast<int>(knn_ids_flat.size()) != n_real * knn_real) {
        throw std::runtime_error("LinkageMultiCenterTwoCodebookImplCore: knn_ids_flat shape mismatch.");
    }

    auto get_bcol = [&](int idx) -> const FullCode* {
        if (idx < n_virt) {
            return B_virtual + static_cast<std::size_t>(idx) * static_cast<std::size_t>(m);
        }
        const int real_idx = idx - real_base;
        if (real_idx >= 0 && real_idx < n_real) {
            const int gi = cols_real[static_cast<std::size_t>(real_idx)];
            return B_real->Col(gi);
        }
        throw std::runtime_error("LinkageMultiCenterTwoCodebookImplCore: get_bcol idx out of range.");
    };
    auto get_acol = [&](int idx) -> const float* {
        if (idx < n_virt) {
            return a_virtual + static_cast<std::size_t>(idx) * static_cast<std::size_t>(m);
        }
        const int real_idx = idx - real_base;
        if (real_idx >= 0 && real_idx < n_real) {
            const int gi = cols_real[static_cast<std::size_t>(real_idx)];
            return a_real->Col(gi);
        }
        throw std::runtime_error("LinkageMultiCenterTwoCodebookImplCore: get_acol idx out of range.");
    };
    auto get_rcol = [&](int idx) -> const float* {
        if (idx < n_virt) {
            return R_virtual->Col(idx);
        }
        const int real_idx = idx - real_base;
        if (real_idx >= 0 && real_idx < n_real) {
            const int gi = cols_real[static_cast<std::size_t>(real_idx)];
            return R_full_shared->Col(gi);
        }
        throw std::runtime_error("LinkageMultiCenterTwoCodebookImplCore: get_rcol idx out of range.");
    };

    // Only reconstruct virtual roots (real columns already exist in R_full_shared from earlier ReconstructAll).
    // Virtual roots are encoded with a constant routing/root code for this IVF cluster.
    // When h0_root > 256, layer-0 codes in B_virtual are dummies (FullCode=u8 can't hold the value).
    // Always use `cluster_id` for layer 0 to keep reconstruction correct regardless of h0 size.
    for (int v = 0; v < n_virt; ++v) {
        const FullCode* bcol = B_virtual + static_cast<std::size_t>(v) * static_cast<std::size_t>(m);
        const float* acol = a_virtual + static_cast<std::size_t>(v) * static_cast<std::size_t>(m);
        float* dst = R_virtual->Col(v);
        std::fill(dst, dst + d, 0.0f);
        for (int l = 0; l < m; ++l) {
            const int code = (l == 0)
                                 ? cluster_id
                                 : static_cast<int>(bcol[static_cast<std::size_t>(l)]);
            const float coeff = acol[static_cast<std::size_t>(l)];
            const float* center = C_root.books[l].Col(code);
            #pragma omp simd
            for (int r = 0; r < d; ++r) {
                dst[r] += coeff * center[r];
            }
        }
    }

    std::vector<float> base_mse(static_cast<std::size_t>(n_real), 0.0f);
    for (int i = 0; i < n_real; ++i) {
        const float* x = X_real.Col(cols_real[static_cast<std::size_t>(i)]);
        const float* r = R_full_shared->Col(cols_real[static_cast<std::size_t>(i)]);
        float s = 0.0f;
        #pragma omp simd reduction(+:s)
        for (int j = 0; j < d; ++j) {
            float diff = x[j] - r[j];
            s += diff * diff;
        }
        base_mse[static_cast<std::size_t>(i)] = s;
    }

    std::vector<int> parent(static_cast<std::size_t>(n_total), 0);
    std::vector<int> depths(static_cast<std::size_t>(n_total), 0);

    LinkageTls tls = MakeLinkageTls(d, m, H);

    tls.order_idx.resize(static_cast<std::size_t>(n_real));
    for (int i = 0; i < n_real; ++i) {
        tls.order_idx[static_cast<std::size_t>(i)] = real_base + i;
    }
    std::mt19937 rng(static_cast<std::mt19937::result_type>(linkage_cfg.seed + 997 * cluster_id));
    std::shuffle(tls.order_idx.begin(), tls.order_idx.end(), rng);

    tls.processed.assign(static_cast<std::size_t>(n_real), 0);
    const int Kp = std::min(std::max(4 * knn_k, knn_k + 16), knn_real);

    ClusterHnswIndex virt_hnsw;
    std::vector<int> virt_ids;
    std::vector<int> virt_tmp;
    if (n_virt >= 2) {
        virt_ids.resize(static_cast<std::size_t>(n_virt));
        std::iota(virt_ids.begin(), virt_ids.end(), 0);
        const int efc = std::max(1, std::min(knn_k, n_virt));
        virt_hnsw.Build(*R_virtual, virt_ids, hnsw_cfg, efc);
        virt_tmp.reserve(static_cast<std::size_t>(std::min(n_virt, knn_k)));
    }

    std::uniform_int_distribution<int> layer_dist(0, std::max(0, m - 1));
    std::vector<std::uniform_int_distribution<int>> code_dists;
    if constexpr (UseIls) {
        code_dists.reserve(static_cast<std::size_t>(m));
        for (int layer = 0; layer < m; ++layer) {
            const int h = pre_one.h_vec[layer];
            code_dists.emplace_back(0, std::max(0, h - 2));
        }
    }

    int ls_failures = 0;

#if defined(STLQ_ENABLE_CUDA)
    const bool cuda_linkage_enabled =
        runtime_cfg && runtime_cfg->use_cuda && cuda_pool;
    CudaCtx* cuda_ctx = nullptr;
    struct CudaLease {
        CudaStreamKernelsPool* pool = nullptr;
        CudaCtx* ctx = nullptr;
        ~CudaLease() {
            if (pool && ctx) {
                pool->Release(ctx);
            }
        }
    } cuda_lease;
    if (cuda_linkage_enabled) {
        // Multi-center (bad-cluster) path: do not silently fall back to CPU just because the ctx pool is
        // temporarily exhausted. When enabled, honor the same "wait for ctx" semantics as the good/init paths.
        cuda_ctx = (runtime_cfg && runtime_cfg->cuda_linkage_wait_for_ctx) ? cuda_pool->Acquire()
                                                                         : cuda_pool->TryAcquire();
        cuda_lease.pool = cuda_pool;
        cuda_lease.ctx = cuda_ctx;
    }

    bool device_rfull_ready = false;
    if (cuda_ctx && runtime_cfg && runtime_cfg->cuda_linkage_use_device_rfull) {
        std::string up_err;
        if (n_virt > 0) {
            if (CudaLinkageUploadClusterRfullTwoParts(cuda_ctx,
                                                    R_virtual->data.data(), n_virt,
                                                    R_full_shared->data.data(), n_real,
                                                    d, &up_err)) {
                device_rfull_ready = true;
            }
        } else {
            if (CudaLinkageUploadClusterRfull(cuda_ctx,
                                            R_full_shared->data.data(),
                                            d, n_real, &up_err)) {
                device_rfull_ready = true;
            }
        }
        if (!device_rfull_ready) {
            LogWarn(std::string("CudaLinkageUploadClusterRfull(2parts) failed, fallback to pinned_Rp: ") + up_err);
        }
    }

    float* pinned_Rp = nullptr;
    int* pinned_cand = nullptr;
    int pinned_kp_cap = 0;
    auto ensure_pinned = [&](int need_kp) {
        if (!cuda_ctx) return;
        if (need_kp <= pinned_kp_cap) return;
        if (pinned_Rp) {
            cudaFreeHost(pinned_Rp);
            pinned_Rp = nullptr;
        }
        if (pinned_cand) {
            cudaFreeHost(pinned_cand);
            pinned_cand = nullptr;
        }
        cudaError_t st = cudaMallocHost(&pinned_Rp,
                                        sizeof(float) * static_cast<std::size_t>(d) *
                                            static_cast<std::size_t>(need_kp));
        if (st != cudaSuccess) {
            LogWarn(std::string("cudaMallocHost(pinned_Rp) failed: ") + cudaGetErrorString(st));
            pinned_Rp = nullptr;
            pinned_kp_cap = 0;
            return;
        }
        st = cudaMallocHost(&pinned_cand, sizeof(int) * static_cast<std::size_t>(need_kp));
        if (st != cudaSuccess) {
            LogWarn(std::string("cudaMallocHost(pinned_cand) failed: ") + cudaGetErrorString(st));
            cudaFreeHost(pinned_Rp);
            pinned_Rp = nullptr;
            pinned_cand = nullptr;
            pinned_kp_cap = 0;
            return;
        }
        pinned_kp_cap = need_kp;
    };

    // Batched device R_full updates: accumulate per-node commits and flush before GPU evals.
    std::vector<int> pending_rfull_cols;
    std::vector<float> pending_rfull_cols_data;  // packed columns, size = pending_rfull_cols.size() * d
    pending_rfull_cols.reserve(1024);
    pending_rfull_cols_data.reserve(static_cast<std::size_t>(1024) * static_cast<std::size_t>(d));

    const auto PushPendingRfullUpdate = [&](int col, const float* r_col) {
        pending_rfull_cols.push_back(col);
        const std::size_t old = pending_rfull_cols_data.size();
        pending_rfull_cols_data.resize(old + static_cast<std::size_t>(d));
        std::memcpy(pending_rfull_cols_data.data() + old, r_col, sizeof(float) * static_cast<std::size_t>(d));
    };
    const auto FlushPendingRfullUpdates = [&]() -> bool {
        if (pending_rfull_cols.empty()) return true;
        if (!cuda_ctx || !device_rfull_ready) {
            pending_rfull_cols.clear();
            pending_rfull_cols_data.clear();
            return false;
        }
        const double t0_up = LinkageBuildProfileEnabled() ? LinkageBuildWallNowS() : 0.0;
        constexpr int kMaxColsPerFlush = 2048;
        std::size_t off = 0;
        std::uint64_t n_batches = 0;
        while (off < pending_rfull_cols.size()) {
            const int take =
                static_cast<int>(std::min<std::size_t>(pending_rfull_cols.size() - off,
                                                      static_cast<std::size_t>(kMaxColsPerFlush)));
            std::string up_err;
            const int* cols = pending_rfull_cols.data() + off;
            const float* data = pending_rfull_cols_data.data() + off * static_cast<std::size_t>(d);
            if (!CudaLinkageUpdateClusterRfullColumnsBatch(cuda_ctx, cols, data, d, take, &up_err)) {
                LogWarn(std::string("CudaLinkageUpdateClusterRfullColumnsBatch failed, disable device_rfull: ") + up_err);
                device_rfull_ready = false;
                pending_rfull_cols.clear();
                pending_rfull_cols_data.clear();
                return false;
            }
            off += static_cast<std::size_t>(take);
            n_batches += 1;
        }
        if (LinkageBuildProfileEnabled()) {
            auto* prof = GetLinkageBuildProfileTls();
            prof->rfull_update_s += LinkageBuildWallNowS() - t0_up;
            prof->rfull_update_calls += static_cast<std::uint64_t>(pending_rfull_cols.size());
            prof->rfull_update_batches += n_batches;
        }
        pending_rfull_cols.clear();
        pending_rfull_cols_data.clear();
        return true;
    };
#endif

    int legacy_ord_start = 0;
    bool use_legacy = true;

#if defined(STLQ_ENABLE_CUDA)
    // When we do not have full-precomp for C_one (meta-only `pre_one`), CPU candidate evaluation is unavailable.
    // In that case we must use the GPU bad-window batching path (large-root/split-root evaluator) for correctness,
    // even if the optional `cuda_linkage_bad_window_batch` knob is off.
    const bool want_bad_window_batch =
        cuda_ctx && runtime_cfg && device_rfull_ready &&
        (runtime_cfg->cuda_linkage_bad_window_batch || use_large_root_eval);
    if (use_large_root_eval && !want_bad_window_batch) {
        throw std::runtime_error(
            "LinkageMultiCenterTwoCodebookImplCore: pre_one has no G (CPU fallback unavailable), but CUDA bad-window batching is unavailable.");
    }
    if (want_bad_window_batch) {
        use_legacy = false;

        const int window_nodes_max = std::max(1, runtime_cfg->cuda_linkage_bad_window_nodes);
        const int window_npairs_max = std::max(1, runtime_cfg->cuda_linkage_bad_window_max_npairs);

        const int kInfRank = std::numeric_limits<int>::max() / 4;
        std::vector<int> rank(static_cast<std::size_t>(n_total), kInfRank);
        for (int v = 0; v < n_virt; ++v) {
            rank[static_cast<std::size_t>(v)] = -1;
        }
        for (int ord = 0; ord < n_real; ++ord) {
            const int li = tls.order_idx[static_cast<std::size_t>(ord)];
            rank[static_cast<std::size_t>(li)] = ord;
        }

        std::vector<int> win_li;
        std::vector<float> win_X;  // d×B in col-major (node-major contiguous)
        std::vector<int> win_frozen_offsets;  // B+1
        std::vector<int> win_frozen_cand;     // Npairs
        std::vector<int> win_pair_node;       // Npairs
        std::vector<std::vector<int>> win_dynamic_cand;

        std::vector<int> win_best_parent_frozen;
        std::vector<float> win_best_cost_frozen;
        std::vector<FullCode> win_best_B_frozen;
        std::vector<float> win_best_a_frozen;

        // Bad-window same-layer dynamic: incremental many-nodes flush (semantics-preserving).
        std::vector<int> win_inv_offsets;
        std::vector<int> win_inv_child;
        std::vector<int> win_inv_parent_local;
        // Window-local same-layer dynamic candidate lists (flat, fixed-stride to avoid allocator churn).
        std::vector<int> win_dyn_list_flat;
        std::vector<int> win_dyn_len;
        int win_dyn_stride = 0;
        std::vector<int> win_dyn_done;
        std::vector<int> win_dyn_best_parent;
        std::vector<float> win_dyn_best_cost;
        std::vector<FullCode> win_dyn_best_B;
        std::vector<float> win_dyn_best_a;
        std::vector<int> win_dyn_flush_offsets;
        std::vector<int> win_dyn_flush_cand;
        std::vector<int> win_dyn_flush_pair_node;
        std::vector<float> win_dyn_flush_X;
        std::vector<int> win_dyn_flush_sel_nodes;
        std::vector<int> win_dyn_flush_sel_take;
        std::vector<int> win_dyn_flush_best_parent;
        std::vector<float> win_dyn_flush_best_cost;
        std::vector<FullCode> win_dyn_flush_best_B;
        std::vector<float> win_dyn_flush_best_a;
        std::vector<int> win_dyn_cpu_tail;

        win_li.reserve(static_cast<std::size_t>(window_nodes_max));
        win_frozen_offsets.reserve(static_cast<std::size_t>(window_nodes_max + 1));
        win_dynamic_cand.reserve(static_cast<std::size_t>(window_nodes_max));
        // Keep per-node vectors alive across windows to avoid repeated heap alloc/free.
        if (static_cast<int>(win_dynamic_cand.size()) < window_nodes_max) {
            win_dynamic_cand.resize(static_cast<std::size_t>(window_nodes_max));
        }

        auto fill_virtual_candidates = [&](const float* xi, int virtual_knn_k, std::vector<int>* out) {
            out->clear();
            if (virtual_knn_k <= 0) return;
            if (n_virt == 1) {
                out->push_back(0);
                return;
            }
            if (!virt_hnsw.IsReady()) return;
            const int k_search = std::min(n_virt, virtual_knn_k);
            virt_hnsw.SetEf(k_search);
            virt_tmp.clear();
            virt_hnsw.SearchLabelsInto(xi, k_search, &virt_tmp);
            out->reserve(static_cast<std::size_t>(k_search));
            for (int label : virt_tmp) {
                out->push_back(label);
            }
        };

        std::vector<int> virt_cand;

        int w0 = 0;
        while (w0 < n_real) {
            win_li.clear();
            win_X.clear();
            win_frozen_offsets.clear();
            win_frozen_offsets.push_back(0);
            win_frozen_cand.clear();
            win_pair_node.clear();

            int Bwin = 0;
            int frozen_npairs = 0;

            std::vector<int> frozen;
            for (int ord = w0; ord < n_real && Bwin < window_nodes_max; ++ord) {
                const double t0_build = LinkageBuildProfileEnabled() ? LinkageBuildWallNowS() : 0.0;
                const int li = tls.order_idx[static_cast<std::size_t>(ord)];
                const int real_idx = li - real_base;
                const float* xi = X_real.Col(cols_real[static_cast<std::size_t>(real_idx)]);

                const int virtual_knn_k = std::min(n_virt, knn_k);
                const int cap = knn_k + virtual_knn_k;

                frozen.clear();
                if (static_cast<int>(frozen.capacity()) < cap) {
                    frozen.reserve(static_cast<std::size_t>(cap));
                }
                std::vector<int>& dyn = win_dynamic_cand[static_cast<std::size_t>(Bwin)];
                dyn.clear();
                if (static_cast<int>(dyn.capacity()) < cap) {
                    dyn.reserve(static_cast<std::size_t>(cap));
                }

                fill_virtual_candidates(xi, virtual_knn_k, &virt_cand);
                int n_total_cand = 0;
                for (int v : virt_cand) {
                    if (n_total_cand >= cap) break;
                    const int new_depth = depths[static_cast<std::size_t>(v)] + 1;
                    if (new_depth <= max_depth) {
                        frozen.push_back(v);
                    }
                    n_total_cand += 1;
                }

                if (n_total_cand < cap) {
                    const std::uint32_t* nbrs =
                        knn_ids_flat.data() + static_cast<std::size_t>(real_idx) * knn_real;
                    for (int t = 0; t < Kp; ++t) {
                        const int nb = static_cast<int>(nbrs[t]);
                        const int p_li = real_base + nb;
                        const int prank = rank[static_cast<std::size_t>(p_li)];
                        if (prank >= ord) {
                            continue;
                        }
                        if (n_total_cand >= cap) {
                            break;
                        }
                        const int new_depth = depths[static_cast<std::size_t>(p_li)] + 1;
                        if (new_depth <= max_depth) {
                            if (prank < w0) {
                                frozen.push_back(p_li);
                            } else {
                                dyn.push_back(p_li);
                            }
                        }
                        n_total_cand += 1;
                    }
                }

                if (Bwin > 0 && frozen_npairs + static_cast<int>(frozen.size()) > window_npairs_max) {
                    break;
                }

                win_li.push_back(li);
                win_frozen_offsets.push_back(win_frozen_offsets.back() + static_cast<int>(frozen.size()));
                for (int p_idx : frozen) {
                    win_frozen_cand.push_back(p_idx);
                    win_pair_node.push_back(Bwin);
                }
                for (int r = 0; r < d; ++r) {
                    win_X.push_back(xi[r]);
                }
                frozen_npairs += static_cast<int>(frozen.size());
                Bwin += 1;
                if (LinkageBuildProfileEnabled()) {
                    GetLinkageBuildProfileTls()->cand_build_cpu_s += LinkageBuildWallNowS() - t0_build;
                }
            }

            // Evaluate frozen candidates in one GPU batch.
            win_best_parent_frozen.assign(static_cast<std::size_t>(Bwin), -1);
            win_best_cost_frozen.assign(static_cast<std::size_t>(Bwin), std::numeric_limits<float>::infinity());
            win_best_B_frozen.assign(static_cast<std::size_t>(Bwin) * static_cast<std::size_t>(m), 0);
            win_best_a_frozen.assign(static_cast<std::size_t>(Bwin) * static_cast<std::size_t>(m), 0.0f);
            if (frozen_npairs > 0) {
                int ils_iters_gpu = 0;
                int perturb_k_gpu = 0;
                if constexpr (UseIls) {
                    ils_iters_gpu = std::max(0, linkage_cfg.ils_rounds);
                    perturb_k_gpu = std::min(std::max(0, linkage_cfg.ils_perturb_layers), m);
                }
                const std::uint32_t seed_gpu =
                    static_cast<std::uint32_t>(linkage_cfg.seed + 997 * cluster_id);
                std::vector<std::uint64_t> node_sample_id_base(static_cast<std::size_t>(Bwin), 0);
                for (int i = 0; i < Bwin; ++i) {
                    const std::uint32_t node_local_u32 =
                        static_cast<std::uint32_t>(win_li[static_cast<std::size_t>(i)]);
                    node_sample_id_base[static_cast<std::size_t>(i)] =
                        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(cluster_id)) << 32) |
                        static_cast<std::uint64_t>(node_local_u32);
                }

                std::string gpu_err;
                const double t0_gpu = LinkageBuildProfileEnabled() ? LinkageBuildWallNowS() : 0.0;
                if (!FlushPendingRfullUpdates()) {
                    legacy_ord_start = w0;
                    use_legacy = true;
                    break;
                }
                const bool ok_gpu =
                    use_large_root_eval
                        ? EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRoot(
                              cuda_ctx, C_one,
                              win_X.data(), d, Bwin,
                              win_frozen_cand.data(),
                              win_pair_node.data(),
                              win_frozen_offsets.data(),
                              frozen_npairs,
                              icm_round,
                              ils_iters_gpu,
                              perturb_k_gpu,
                              seed_gpu,
                              node_sample_id_base.data(),
                              win_best_parent_frozen.data(),
                              win_best_cost_frozen.data(),
                              win_best_B_frozen.data(),
                              win_best_a_frozen.data(),
                              &gpu_err)
                        : EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBase(
                              cuda_ctx, pre_one,
                              win_X.data(), d, Bwin,
                              win_frozen_cand.data(),
                              win_pair_node.data(),
                              win_frozen_offsets.data(),
                              frozen_npairs,
                              icm_round,
                              ils_iters_gpu,
                              perturb_k_gpu,
                              seed_gpu,
                              node_sample_id_base.data(),
                              win_best_parent_frozen.data(),
                              win_best_cost_frozen.data(),
                              win_best_B_frozen.data(),
                              win_best_a_frozen.data(),
                              &gpu_err);
                if (!ok_gpu) {
                    if (use_large_root_eval) {
                        throw std::runtime_error(std::string("CUDA bad-window frozen batch failed (CPU fallback unavailable): ") + gpu_err);
                    }
                    LogWarn(std::string("CUDA bad-window frozen batch failed; falling back to legacy: ") + gpu_err);
                    legacy_ord_start = w0;
                    use_legacy = true;
                    break;
                }
                if (LinkageBuildProfileEnabled()) {
                    auto* prof = GetLinkageBuildProfileTls();
                    prof->gpu_wait_s += LinkageBuildWallNowS() - t0_gpu;
                    prof->gpu_calls_many_nodes += 1;
                }
            }

            // Bad-window dynamic: incremental window-local same-layer many-nodes flush.
            const bool win_dyn_enabled =
                (runtime_cfg && runtime_cfg->cuda_linkage_same_dynamic_min_pairs > 0 && Bwin > 1);
	            bool win_dyn_failed = false;
	            int win_dyn_min_pairs = 0;
	            int win_dyn_max_pairs = 0;
	            int win_dyn_soft_nodes_cap = 0;
	            int win_dyn_tiny_forced_node_off = -1;
	            if (win_dyn_enabled) {
                win_dyn_min_pairs = std::max(1, runtime_cfg->cuda_linkage_same_dynamic_min_pairs);

                // Max pairs cap: prefer dedicated same-dynamic cap, otherwise use the existing bad-window max.
                win_dyn_max_pairs = runtime_cfg->cuda_linkage_same_dynamic_window_max_pairs;
                if (win_dyn_max_pairs <= 0) {
                    win_dyn_max_pairs = window_npairs_max;
                }
                win_dyn_max_pairs = std::max(1, win_dyn_max_pairs);

                // Soft cap for number of nodes in one flush (secondary; max-pairs is the primary safety).
                win_dyn_soft_nodes_cap = runtime_cfg->cuda_linkage_same_dynamic_pending_nodes;
                if (win_dyn_soft_nodes_cap <= 0) win_dyn_soft_nodes_cap = Bwin;
                win_dyn_soft_nodes_cap = std::max(1, std::min(Bwin, win_dyn_soft_nodes_cap));

                std::vector<int> inv_counts(static_cast<std::size_t>(Bwin), 0);
                for (int child_off = 0; child_off < Bwin; ++child_off) {
                    const auto& dyn = win_dynamic_cand[static_cast<std::size_t>(child_off)];
                    for (int p_li : dyn) {
                        const int prank = rank[static_cast<std::size_t>(p_li)];
                        const int parent_off = prank - w0;
                        if (parent_off < 0 || parent_off >= child_off || parent_off >= Bwin) continue;
                        inv_counts[static_cast<std::size_t>(parent_off)] += 1;
                    }
                }

                win_inv_offsets.assign(static_cast<std::size_t>(Bwin) + 1u, 0);
                win_inv_offsets[0] = 0;
                for (int r = 0; r < Bwin; ++r) {
                    win_inv_offsets[static_cast<std::size_t>(r + 1)] =
                        win_inv_offsets[static_cast<std::size_t>(r)] + inv_counts[static_cast<std::size_t>(r)];
                }
                const int total_edges = win_inv_offsets[static_cast<std::size_t>(Bwin)];
                win_inv_child.resize(static_cast<std::size_t>(total_edges));
                win_inv_parent_local.resize(static_cast<std::size_t>(total_edges));
                std::vector<int> inv_cursor = win_inv_offsets;
                for (int child_off = 0; child_off < Bwin; ++child_off) {
                    const auto& dyn = win_dynamic_cand[static_cast<std::size_t>(child_off)];
                    for (int p_li : dyn) {
                        const int prank = rank[static_cast<std::size_t>(p_li)];
                        const int parent_off = prank - w0;
                        if (parent_off < 0 || parent_off >= child_off || parent_off >= Bwin) continue;
                        const int pos = inv_cursor[static_cast<std::size_t>(parent_off)]++;
                        win_inv_child[static_cast<std::size_t>(pos)] = child_off;
                        win_inv_parent_local[static_cast<std::size_t>(pos)] = p_li;
                    }
                }

                win_dyn_stride = Bwin;
                win_dyn_list_flat.resize(static_cast<std::size_t>(Bwin) *
                                         static_cast<std::size_t>(win_dyn_stride));
                win_dyn_len.assign(static_cast<std::size_t>(Bwin), 0);
                win_dyn_done.assign(static_cast<std::size_t>(Bwin), 0);
                win_dyn_best_parent.assign(static_cast<std::size_t>(Bwin), -1);
                win_dyn_best_cost.assign(static_cast<std::size_t>(Bwin),
                                         std::numeric_limits<float>::infinity());
	                win_dyn_best_B.assign(static_cast<std::size_t>(Bwin) * static_cast<std::size_t>(m), 0);
	                win_dyn_best_a.assign(static_cast<std::size_t>(Bwin) * static_cast<std::size_t>(m), 0.0f);
	                win_dyn_cpu_tail.clear();
	                win_dyn_tiny_forced_node_off = -1;
	            }

            const auto FlushBadWinDynManyNodes = [&](int start_off, int force_node_off) -> int {
                if (!win_dyn_enabled) return 0;
                if (start_off < 0) start_off = 0;
                if (start_off >= Bwin) return 0;
                if (force_node_off >= 0) {
                    force_node_off = std::max(0, std::min(Bwin - 1, force_node_off));
                    start_off = std::min(start_off, force_node_off);
                }
                if (!cuda_ctx || !device_rfull_ready) return 0;

                win_dyn_flush_sel_nodes.clear();
                win_dyn_flush_sel_take.clear();

                int total_pairs = 0;
                for (int k = start_off; k < Bwin; ++k) {
                    const int have = win_dyn_len[static_cast<std::size_t>(k)];
                    const int done = win_dyn_done[static_cast<std::size_t>(k)];
                    const int avail = std::max(0, have - done);
                    if (avail <= 0) continue;

                    const int rem = win_dyn_max_pairs - total_pairs;
                    if (rem <= 0) break;
                    const int take = std::min(avail, rem);
                    if (take <= 0) break;

                    win_dyn_flush_sel_nodes.push_back(k);
                    win_dyn_flush_sel_take.push_back(take);
                    total_pairs += take;

                    const bool enough_pairs = (total_pairs >= win_dyn_min_pairs);
                    const bool hit_soft_cap =
                        (win_dyn_soft_nodes_cap > 0 &&
                         static_cast<int>(win_dyn_flush_sel_nodes.size()) >= win_dyn_soft_nodes_cap);
                    if (enough_pairs && hit_soft_cap) break;
                }

                const int slice_len = static_cast<int>(win_dyn_flush_sel_nodes.size());
                if (slice_len <= 0 || total_pairs <= 0) return 0;
                // For opportunistic preflush we keep a minimum-pairs threshold to avoid too many tiny GPU calls.
                // But when `force_node_off` is set (commit-time), we prefer finishing the remaining dynamic work
                // on GPU (even if it is small) to avoid falling back to CPU for the tail.
                if (total_pairs < win_dyn_min_pairs && force_node_off < 0) {
                    return 0;
                }

                // Stage-3f: tiny CPU / big GPU hybrid for commit-time forced flush.
                // Forced flush slices can be extremely small (e.g. total_pairs ~ 1..16), where a GPU
                // many-nodes launch/sync can cost more than the math. When enabled, skip issuing a GPU
                // many-nodes call and let commit-time CPU tail evaluation handle the remainder.
                const bool forced = (force_node_off >= 0);
                bool tiny_cpu_enabled = false;
                int tiny_cpu_max_pairs = 0;
                int tiny_cpu_max_nodes = 0;
                if (forced && runtime_cfg) {
                    tiny_cpu_enabled = runtime_cfg->cuda_linkage_same_dynamic_tiny_cpu;
                    tiny_cpu_max_pairs = runtime_cfg->cuda_linkage_same_dynamic_tiny_cpu_max_pairs;
                    tiny_cpu_max_nodes = runtime_cfg->cuda_linkage_same_dynamic_tiny_cpu_max_nodes;
                    if (tiny_cpu_max_pairs <= 0) tiny_cpu_enabled = false;
                    if (tiny_cpu_max_nodes < 0) tiny_cpu_max_nodes = 0;
                }
                const bool use_tiny_cpu =
                    forced && tiny_cpu_enabled && (total_pairs <= tiny_cpu_max_pairs) &&
                    (tiny_cpu_max_nodes <= 0 || slice_len <= tiny_cpu_max_nodes);
                if (use_tiny_cpu) {
                    win_dyn_tiny_forced_node_off = force_node_off;
                    if (LinkageBuildProfileEnabled()) {
                        if (auto* prof = GetLinkageBuildProfileTls()) {
                            prof->bad_dyn_window_tiny_cpu_calls += 1;
                            prof->bad_dyn_window_tiny_cpu_nodes_sum += static_cast<std::uint64_t>(slice_len);
                            prof->bad_dyn_window_tiny_cpu_pairs_sum +=
                                static_cast<std::uint64_t>(std::max(0, total_pairs));
                            // Sanity guard: total_pairs is accumulated from non-negative `take` and should never be negative.
                            // If this triggers, it strongly suggests memory corruption or a stale binary after changing
                            // LinkageBuildProfileTls layout.
                            if (total_pairs < 0) {
                                LogWarn("bad_dyn_window_tiny_cpu: total_pairs < 0 (corruption suspected). "
                                        "total_pairs=" +
                                        std::to_string(total_pairs) + " slice_len=" +
                                        std::to_string(slice_len) + " start_off=" +
                                        std::to_string(start_off) + " force_node_off=" +
                                        std::to_string(force_node_off) + " win_dyn_max_pairs=" +
                                        std::to_string(win_dyn_max_pairs) + " tiny_cpu_max_pairs=" +
                                        std::to_string(tiny_cpu_max_pairs));
                            }
                        }
                    }
                    return 0;
                }

                win_dyn_flush_offsets.assign(static_cast<std::size_t>(slice_len) + 1u, 0);
                win_dyn_flush_cand.resize(static_cast<std::size_t>(total_pairs));
                win_dyn_flush_pair_node.resize(static_cast<std::size_t>(total_pairs));
                win_dyn_flush_X.resize(static_cast<std::size_t>(slice_len) * static_cast<std::size_t>(d));
                int pos = 0;
                win_dyn_flush_offsets[0] = 0;
                for (int k = 0; k < slice_len; ++k) {
                    const int kk = win_dyn_flush_sel_nodes[static_cast<std::size_t>(k)];
                    const int take = win_dyn_flush_sel_take[static_cast<std::size_t>(k)];
                    const int* lst =
                        win_dyn_list_flat.data() +
                        static_cast<std::size_t>(kk) * static_cast<std::size_t>(win_dyn_stride);
                    const int done = win_dyn_done[static_cast<std::size_t>(kk)];
                    const float* src_x =
                        win_X.data() + static_cast<std::size_t>(kk) * static_cast<std::size_t>(d);
                    float* dst_x =
                        win_dyn_flush_X.data() + static_cast<std::size_t>(k) * static_cast<std::size_t>(d);
                    std::memcpy(dst_x, src_x, sizeof(float) * static_cast<std::size_t>(d));
                    for (int t = 0; t < take; ++t) {
                        const int p_li = lst[done + t];
                        win_dyn_flush_cand[static_cast<std::size_t>(pos)] = p_li;
                        win_dyn_flush_pair_node[static_cast<std::size_t>(pos)] = k;
                        ++pos;
                    }
                    win_dyn_flush_offsets[static_cast<std::size_t>(k + 1)] = pos;
                }

                win_dyn_flush_best_parent.assign(static_cast<std::size_t>(slice_len), -1);
                win_dyn_flush_best_cost.assign(static_cast<std::size_t>(slice_len),
                                               std::numeric_limits<float>::infinity());
                win_dyn_flush_best_B.assign(static_cast<std::size_t>(slice_len) * static_cast<std::size_t>(m), 0);
                win_dyn_flush_best_a.assign(static_cast<std::size_t>(slice_len) * static_cast<std::size_t>(m), 0.0f);

                int ils_iters_gpu = 0;
                int perturb_k_gpu = 0;
                if constexpr (UseIls) {
                    ils_iters_gpu = std::max(0, linkage_cfg.ils_rounds);
                    perturb_k_gpu = std::min(std::max(0, linkage_cfg.ils_perturb_layers), m);
                }
                const std::uint32_t seed_gpu =
                    static_cast<std::uint32_t>(linkage_cfg.seed + 997 * cluster_id);
                std::vector<std::uint64_t> node_sample_id_base(static_cast<std::size_t>(slice_len), 0);
                for (int k = 0; k < slice_len; ++k) {
                    const int kk = win_dyn_flush_sel_nodes[static_cast<std::size_t>(k)];
                    const std::uint32_t node_local_u32 =
                        static_cast<std::uint32_t>(win_li[static_cast<std::size_t>(kk)]);
                    node_sample_id_base[static_cast<std::size_t>(k)] =
                        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(cluster_id)) << 32) |
                        static_cast<std::uint64_t>(node_local_u32);
                }

                std::string gpu_err;
                const double t0_gpu = LinkageBuildProfileEnabled() ? LinkageBuildWallNowS() : 0.0;
                if (!FlushPendingRfullUpdates()) {
                    win_dyn_failed = true;
                    return 0;
                }
                const bool ok_gpu =
                    use_large_root_eval
                        ? EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRoot(
                              cuda_ctx, C_one,
                              win_dyn_flush_X.data(), d, slice_len,
                              win_dyn_flush_cand.data(),
                              win_dyn_flush_pair_node.data(),
                              win_dyn_flush_offsets.data(),
                              total_pairs,
                              icm_round,
                              ils_iters_gpu,
                              perturb_k_gpu,
                              seed_gpu,
                              node_sample_id_base.data(),
                              win_dyn_flush_best_parent.data(),
                              win_dyn_flush_best_cost.data(),
                              win_dyn_flush_best_B.data(),
                              win_dyn_flush_best_a.data(),
                              &gpu_err)
                        : EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBase(
                              cuda_ctx, pre_one,
                              win_dyn_flush_X.data(), d, slice_len,
                              win_dyn_flush_cand.data(),
                              win_dyn_flush_pair_node.data(),
                              win_dyn_flush_offsets.data(),
                              total_pairs,
                              icm_round,
                              ils_iters_gpu,
                              perturb_k_gpu,
                              seed_gpu,
                              node_sample_id_base.data(),
                              win_dyn_flush_best_parent.data(),
                              win_dyn_flush_best_cost.data(),
                              win_dyn_flush_best_B.data(),
                              win_dyn_flush_best_a.data(),
                              &gpu_err);
                if (!ok_gpu) {
                    LogWarn(std::string("CUDA bad-window window-dynamic eval failed; disable bad-window dyn batching: ") +
                            gpu_err);
                    win_dyn_failed = true;
                    return 0;
                }
                if (LinkageBuildProfileEnabled()) {
                    auto* prof = GetLinkageBuildProfileTls();
                    prof->gpu_wait_s += LinkageBuildWallNowS() - t0_gpu;
                    prof->gpu_calls_many_nodes += 1;
                }

                for (int k = 0; k < slice_len; ++k) {
                    const int idx = win_dyn_flush_sel_nodes[static_cast<std::size_t>(k)];
                    const float c = win_dyn_flush_best_cost[static_cast<std::size_t>(k)];
                    if (c < win_dyn_best_cost[static_cast<std::size_t>(idx)]) {
                        win_dyn_best_cost[static_cast<std::size_t>(idx)] = c;
                        win_dyn_best_parent[static_cast<std::size_t>(idx)] =
                            win_dyn_flush_best_parent[static_cast<std::size_t>(k)];
                        const std::size_t dst_base =
                            static_cast<std::size_t>(idx) * static_cast<std::size_t>(m);
                        const std::size_t src_base =
                            static_cast<std::size_t>(k) * static_cast<std::size_t>(m);
                        std::memcpy(win_dyn_best_B.data() + dst_base,
                                    win_dyn_flush_best_B.data() + src_base,
                                    sizeof(FullCode) * static_cast<std::size_t>(m));
                        std::memcpy(win_dyn_best_a.data() + dst_base,
                                    win_dyn_flush_best_a.data() + src_base,
                                    sizeof(float) * static_cast<std::size_t>(m));
                    }
                    const int take = win_dyn_flush_sel_take[static_cast<std::size_t>(k)];
                    win_dyn_done[static_cast<std::size_t>(idx)] += take;
                }
                return slice_len;
            };

            const auto BadWinDynMaybePreflushFrom = [&](int start_off) {
                if (!win_dyn_enabled || win_dyn_failed) return;
                if (start_off < 0) start_off = 0;
                if (start_off > Bwin) start_off = Bwin;
                if (runtime_cfg && runtime_cfg->cuda_linkage_same_dynamic_window_batch) {
                    (void)FlushBadWinDynManyNodes(start_off, /*force_node_off=*/-1);
                } else {
                    for (;;) {
                        const int got = FlushBadWinDynManyNodes(start_off, /*force_node_off=*/-1);
                        if (got <= 0) break;
                    }
                }
            };

            const auto BadWinDynEnsureCompleteForNode = [&](int node_off) {
                if (!win_dyn_enabled || win_dyn_failed) return;
                if (node_off < 0 || node_off >= Bwin) return;
                for (;;) {
                    const int have = win_dyn_len[static_cast<std::size_t>(node_off)];
                    const int done = win_dyn_done[static_cast<std::size_t>(node_off)];
                    if (done >= have) break;
                    if (FlushBadWinDynManyNodes(node_off, /*force_node_off=*/node_off) <= 0) break;
                }
            };

            // Commit sequentially within the window.
            for (int i = 0; i < Bwin; ++i) {
                const int li = win_li[static_cast<std::size_t>(i)];
                const int real_idx = li - real_base;
                const float* xi = win_X.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(d);

                float bestE2 = base_mse[static_cast<std::size_t>(real_idx)];
                int best_parent = -1;
                bool did_tiny_dyn_window_forced_skip = false;

                const float frozen_cost = win_best_cost_frozen[static_cast<std::size_t>(i)];
                const int frozen_parent = win_best_parent_frozen[static_cast<std::size_t>(i)];
                if (frozen_parent >= 0 && std::isfinite(frozen_cost) && frozen_cost < bestE2) {
                    bestE2 = frozen_cost;
                    best_parent = frozen_parent;
                    tls.B_best.assign(static_cast<std::size_t>(m), 0);
                    tls.a_best.assign(static_cast<std::size_t>(m), 0.0f);
                    const std::size_t base = static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
                    for (int l = 0; l < m; ++l) {
                        tls.B_best[static_cast<std::size_t>(l)] =
                            win_best_B_frozen[base + static_cast<std::size_t>(l)];
                        tls.a_best[static_cast<std::size_t>(l)] =
                            win_best_a_frozen[base + static_cast<std::size_t>(l)];
                    }
                }

                // Dynamic candidates are window-local; evaluate serially after earlier commits.
                std::vector<int>& dyn = win_dynamic_cand[static_cast<std::size_t>(i)];
                if (win_dyn_enabled && !win_dyn_failed) {
                    dyn.clear();
                    win_dyn_cpu_tail.clear();

                    const int have = win_dyn_len[static_cast<std::size_t>(i)];
                    const int done = win_dyn_done[static_cast<std::size_t>(i)];
                    if (done < have) {
                        win_dyn_tiny_forced_node_off = -1;
                        BadWinDynEnsureCompleteForNode(i);
                        did_tiny_dyn_window_forced_skip = (win_dyn_tiny_forced_node_off == i);
                        win_dyn_tiny_forced_node_off = -1;
                        const int done2 = win_dyn_done[static_cast<std::size_t>(i)];
                        if (done2 < have) {
                            win_dyn_cpu_tail.clear();
                            win_dyn_cpu_tail.reserve(static_cast<std::size_t>(have - done2));
                            const int* lst =
                                win_dyn_list_flat.data() +
                                static_cast<std::size_t>(i) * static_cast<std::size_t>(win_dyn_stride);
                            for (int t = done2; t < have; ++t) {
                                win_dyn_cpu_tail.push_back(lst[t]);
                            }
                            win_dyn_done[static_cast<std::size_t>(i)] = have;
                        }
                    }

                    const float dyn_cost = win_dyn_best_cost[static_cast<std::size_t>(i)];
                    const int dyn_parent = win_dyn_best_parent[static_cast<std::size_t>(i)];
                    if (dyn_parent >= 0 && dyn_cost < bestE2) {
                        bestE2 = dyn_cost;
                        best_parent = dyn_parent;
                        tls.B_best.assign(static_cast<std::size_t>(m), 0);
                        tls.a_best.assign(static_cast<std::size_t>(m), 0.0f);
                        const std::size_t base =
                            static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
                        std::memcpy(tls.B_best.data(),
                                    win_dyn_best_B.data() + base,
                                    sizeof(FullCode) * static_cast<std::size_t>(m));
                        std::memcpy(tls.a_best.data(),
                                    win_dyn_best_a.data() + base,
                                    sizeof(float) * static_cast<std::size_t>(m));
                    }

                    if (!win_dyn_cpu_tail.empty()) {
                        dyn = win_dyn_cpu_tail;
                    }
                }

                if (!dyn.empty()) {
                    // (Optional Stage-3b enhancement) if dynamic list is not tiny, evaluate it in a single-node GPU batch.
                    bool used_gpu_dyn = false;
                    const int Kp_dyn = static_cast<int>(dyn.size());
                    const int min_kp = std::max(4, std::max(0, runtime_cfg->cuda_linkage_min_kp));
                    const int max_kp = runtime_cfg->cuda_linkage_max_kp;
                    if (!win_dyn_enabled && Kp_dyn >= min_kp && (max_kp <= 0 || Kp_dyn <= max_kp)) {
                        int ils_iters_gpu = 0;
                        int perturb_k_gpu = 0;
                        if constexpr (UseIls) {
                            ils_iters_gpu = std::max(0, linkage_cfg.ils_rounds);
                            perturb_k_gpu = std::min(std::max(0, linkage_cfg.ils_perturb_layers), m);
                        }
                        const std::uint32_t seed_gpu =
                            static_cast<std::uint32_t>(linkage_cfg.seed + 997 * cluster_id);
                        const std::uint64_t sample_id_offset =
                            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(cluster_id)) << 32) |
                            static_cast<std::uint32_t>(li);

                        int best_parent_gpu = -1;
                        float best_cost_gpu = std::numeric_limits<float>::infinity();
                        std::string gpu_err;
                        const double t0_gpu = LinkageBuildProfileEnabled() ? LinkageBuildWallNowS() : 0.0;
                        if (!FlushPendingRfullUpdates() ||
                            !EvaluateParentCandidatesBatchCudaDeviceRfull(cuda_ctx, pre_one,
                                                                          xi, d,
                                                                          dyn.data(), Kp_dyn,
                                                                          icm_round,
                                                                          ils_iters_gpu,
                                                                          perturb_k_gpu,
                                                                          seed_gpu,
                                                                          sample_id_offset,
                                                                          &best_parent_gpu, &best_cost_gpu,
                                                                          &tls.B_init, &tls.a_init,
                                                                          &gpu_err)) {
                            LogWarn(std::string("CUDA bad-window dynamic eval failed; falling back to CPU: ") + gpu_err);
                        } else {
                            used_gpu_dyn = true;
                            if (best_parent_gpu >= 0 && best_cost_gpu < bestE2) {
                                bestE2 = best_cost_gpu;
                                best_parent = best_parent_gpu;
                                tls.B_best = tls.B_init;
                                tls.a_best = tls.a_init;
                            }
                        }
                        if (LinkageBuildProfileEnabled()) {
                            auto* prof = GetLinkageBuildProfileTls();
                            prof->gpu_wait_s += LinkageBuildWallNowS() - t0_gpu;
                            prof->gpu_calls_batch += 1;
                        }
                    }

                    if (!used_gpu_dyn) {
                        const double t0_cpu = LinkageBuildProfileEnabled() ? LinkageBuildWallNowS() : 0.0;
                        const bool have_one_root_cross_gram = !G_one_root.data.empty();
                        if (icm_round > 0 && have_one_root_cross_gram) {
                            ComputeXcBase(pre_one, xi, &tls.xC_base_buf);
                        }
                        for (int p_idx : dyn) {
                            const int new_depth = depths[static_cast<std::size_t>(p_idx)] + 1;
                            if (new_depth > max_depth) {
                                continue;
                            }

                            const float* rp = get_rcol(p_idx);
                            float re_norm2 = 0.0f;
                            #pragma omp simd
                            for (int r = 0; r < d; ++r) {
                                const float v = xi[r] - rp[r];
                                tls.re[static_cast<std::size_t>(r)] = v;
                            }
                            const float* re_ptr = tls.re.data();
                            #pragma omp simd reduction(+:re_norm2)
                            for (int r = 0; r < d; ++r) {
                                const float v = re_ptr[r];
                                re_norm2 += v * v;
                            }

                            const double t0_q = LinkageBuildProfileEnabled() ? LinkageBuildWallNowS() : 0.0;
                            QuantizeWithKmeansAbsNoNormalSingle(pre_one, tls.re,
                                                                &tls.B_init, &tls.a_init,
                                                                &tls.rC_buf, &tls.available);
                            if (LinkageBuildProfileEnabled()) {
                                GetLinkageBuildProfileTls()->cpu_quantize_s += LinkageBuildWallNowS() - t0_q;
                            }

                            if (icm_round > 0) {
                                if (have_one_root_cross_gram) {
                                    BuildXcOneForParentSingleGetCols(pre_root, pre_one, G_one_root,
                                                                     tls.xC_base_buf,
                                                                     get_bcol, get_acol,
                                                                     parent, depths,
                                                                     p_idx, &tls.xC_buf,
                                                                     cluster_id);
                                } else {
                                    ComputeXcFromResidual(pre_one, tls.re, &tls.xC_buf);
                                }
                                const double t0_ls = LinkageBuildProfileEnabled() ? LinkageBuildWallNowS() : 0.0;
                                if (!SolveLeastSquaresVector(pre_one, tls.xC_buf, tls.B_init,
                                                             &tls.A, &tls.b, &tls.work,
                                                             tls.a_init.data())) {
                                    ++ls_failures;
                                    continue;
                                }
                                if (LinkageBuildProfileEnabled()) {
                                    GetLinkageBuildProfileTls()->cpu_ls_s += LinkageBuildWallNowS() - t0_ls;
                                }
                                const double t0_icm = LinkageBuildProfileEnabled() ? LinkageBuildWallNowS() : 0.0;
                                DynamicIcmEncodingSingleForResidual(pre_one, tls.re, tls.xC_buf,
                                                                    icm_round,
                                                                    &tls.B_init, &tls.a_init,
                                                                    &tls.a_bak, &tls.rC_buf,
                                                                    &tls.A, &tls.b, &tls.work,
                                                                    &rng, &ls_failures);
                                if (LinkageBuildProfileEnabled()) {
                                    GetLinkageBuildProfileTls()->cpu_icm_s += LinkageBuildWallNowS() - t0_icm;
                                }
                            } else {
                                tls.xC_buf = tls.rC_buf;
                                for (int l = 0; l < m; ++l) {
                                    const float alpha = tls.a_init[static_cast<std::size_t>(l)];
                                    if (alpha == 0.0f) continue;
                                    const int flat = pre_one.offsets[l] +
                                                     static_cast<int>(tls.B_init[static_cast<std::size_t>(l)]);
                                    const float* Gcol = pre_one.G.Col(flat);
                                    float* dst = tls.xC_buf.data();
                                    #pragma omp simd
                                    for (int q = 0; q < H; ++q) {
                                        dst[q] += alpha * Gcol[q];
                                    }
                                }
                                const double t0_ls = LinkageBuildProfileEnabled() ? LinkageBuildWallNowS() : 0.0;
                                if (!SolveLeastSquaresVectorFromResidual(pre_one, tls.re, tls.B_init,
                                                                         &tls.A, &tls.b, &tls.work,
                                                                         tls.a_init.data())) {
                                    ++ls_failures;
                                    continue;
                                }
                                if (LinkageBuildProfileEnabled()) {
                                    GetLinkageBuildProfileTls()->cpu_ls_s += LinkageBuildWallNowS() - t0_ls;
                                }
                            }

                            const double t0_cost = LinkageBuildProfileEnabled() ? LinkageBuildWallNowS() : 0.0;
                            float E2_best = SampleCostFromXc(pre_one, tls.xC_buf, tls.B_init, tls.a_init, re_norm2);
                            if (LinkageBuildProfileEnabled()) {
                                GetLinkageBuildProfileTls()->cpu_cost_s += LinkageBuildWallNowS() - t0_cost;
                            }

                            if constexpr (UseIls) {
                                const int ils_rounds = std::max(0, linkage_cfg.ils_rounds);
                                const int ils_k = std::min(std::max(0, linkage_cfg.ils_perturb_layers), m);
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

                                    const float E2_try =
                                        SampleCostFromXc(pre_one, tls.xC_buf, tls.B_try, tls.a_try, re_norm2);

                                    if (E2_try < E2_best) {
                                        E2_best = E2_try;
                                        tls.B_init = tls.B_try;
                                        tls.a_init = tls.a_try;
                                    }
                                }
                            }

                            if (E2_best < bestE2) {
                                bestE2 = E2_best;
                                best_parent = p_idx;
                                tls.B_best = tls.B_init;
                                tls.a_best = tls.a_init;
                            }
                        }
                        if (LinkageBuildProfileEnabled()) {
                            const double dt = LinkageBuildWallNowS() - t0_cpu;
                            auto* prof = GetLinkageBuildProfileTls();
                            prof->cpu_fallback_s += dt;
                            if (did_tiny_dyn_window_forced_skip) {
                                // NOTE: subset of `cpu_fallback_s` (do not sum).
                                prof->bad_dyn_window_tiny_cpu_s += dt;
                            }
                        }
                    }
                }

                if (best_parent >= 0) {
                    const double t0_commit = LinkageBuildProfileEnabled() ? LinkageBuildWallNowS() : 0.0;
                    parent[static_cast<std::size_t>(li)] = best_parent + 1;
                    const int gi = cols_real[static_cast<std::size_t>(real_idx)];
                    FullCode* bdst = B_real->Col(gi);
                    float* adst = a_real->Col(gi);
                    for (int l = 0; l < m; ++l) {
                        bdst[static_cast<std::size_t>(l)] = tls.B_best[static_cast<std::size_t>(l)];
                        adst[static_cast<std::size_t>(l)] = tls.a_best[static_cast<std::size_t>(l)];
                    }
                    depths[static_cast<std::size_t>(li)] = depths[static_cast<std::size_t>(best_parent)] + 1;

                    ReconstructNode(C_one.books, tls.B_best, tls.a_best, &tls.R_self);
                    const float* rp = get_rcol(best_parent);
                    float* rdst = R_full_shared->Col(gi);
                    const float* rself_best = tls.R_self.data();
                    #pragma omp simd
                    for (int r = 0; r < d; ++r) {
                        rdst[r] = rp[r] + rself_best[r];
                    }
                    if (LinkageBuildProfileEnabled()) {
                        GetLinkageBuildProfileTls()->commit_cpu_s += LinkageBuildWallNowS() - t0_commit;
                    }
    #if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                    if (cuda_ctx && device_rfull_ready) {
                        PushPendingRfullUpdate(li, rdst);
                    }
    #endif
                }

                // After processing this node, it becomes an eligible same-layer parent for later nodes in the window.
                // We append its dynamic edges (depth-filtered) and opportunistically flush to GPU.
                if (win_dyn_enabled && !win_dyn_failed) {
                    const int parent_off = i;
                    const int beg = win_inv_offsets[static_cast<std::size_t>(parent_off)];
                    const int end = win_inv_offsets[static_cast<std::size_t>(parent_off + 1)];
                    const int p_depth = depths[static_cast<std::size_t>(li)];
                    for (int e = beg; e < end; ++e) {
                        const int child_off = win_inv_child[static_cast<std::size_t>(e)];
                        const int p_li = win_inv_parent_local[static_cast<std::size_t>(e)];
                        if (p_depth + 1 > max_depth) {
                            continue;
                        }
                        const std::size_t c = static_cast<std::size_t>(child_off);
                        int len = win_dyn_len[c];
                        if (len < win_dyn_stride) {
                            win_dyn_list_flat[c * static_cast<std::size_t>(win_dyn_stride) +
                                              static_cast<std::size_t>(len)] = p_li;
                            win_dyn_len[c] = len + 1;
                        }
                    }
                    BadWinDynMaybePreflushFrom(parent_off + 1);
                }
            }

            w0 += Bwin;
        }
    }
#endif

    if (use_legacy) {
        tls.processed.assign(static_cast<std::size_t>(n_real), 0);
        for (int ord = 0; ord < legacy_ord_start; ++ord) {
            const int li = tls.order_idx[static_cast<std::size_t>(ord)];
            const int real_idx = li - real_base;
            tls.processed[static_cast<std::size_t>(real_idx)] = 1;
        }

    for (int ord = legacy_ord_start; ord < n_real; ++ord) {
        const int li = tls.order_idx[static_cast<std::size_t>(ord)];
        const int real_idx = li - real_base;
        const float* xi = X_real.Col(cols_real[static_cast<std::size_t>(real_idx)]);

        tls.candidates.clear();

        const int virtual_knn_k = std::min(n_virt, knn_k);
            if (virtual_knn_k > 0) {
                // Bad-cluster multi-center: virtual nodes are depth-0 roots used only as parent candidates.
                // Use the same in-cluster HNSW design as the standard linkage builders (no brute-force scan).
                if (n_virt == 1) {
                tls.candidates.push_back(0);
                } else if (virt_hnsw.IsReady()) {
                    const int k_search = std::min(n_virt, virtual_knn_k);
                    virt_hnsw.SetEf(k_search);
                    virt_tmp.clear();
                virt_hnsw.SearchLabelsInto(xi, k_search, &virt_tmp);
                for (int label : virt_tmp) {
                    tls.candidates.push_back(label);
                }
            }
        }

        const int cap = knn_k + virtual_knn_k;
        const std::uint32_t* nbrs =
            knn_ids_flat.data() + static_cast<std::size_t>(real_idx) * knn_real;
        for (int t = 0; t < Kp; ++t) {
            const int nb = static_cast<int>(nbrs[t]);
            if (!tls.processed[static_cast<std::size_t>(nb)]) {
                continue;
            }
            tls.candidates.push_back(real_base + nb);
            if (static_cast<int>(tls.candidates.size()) >= cap) {
                break;
            }
        }

        tls.processed[static_cast<std::size_t>(real_idx)] = 1;
        if (tls.candidates.empty()) {
            continue;
        }

        float bestE2 = base_mse[static_cast<std::size_t>(real_idx)];
        int best_parent = -1;

        bool used_gpu = false;
#if defined(STLQ_ENABLE_CUDA)
        if (cuda_ctx && runtime_cfg) {
            int keep = 0;
            for (int p_idx : tls.candidates) {
                const int new_depth = depths[static_cast<std::size_t>(p_idx)] + 1;
                if (new_depth > max_depth) continue;
                tls.candidates[static_cast<std::size_t>(keep++)] = p_idx;
            }
            tls.candidates.resize(static_cast<std::size_t>(keep));

            const int Kp_use = static_cast<int>(tls.candidates.size());
            const int min_kp = std::max(1, runtime_cfg->cuda_linkage_min_kp);
            const int max_kp = runtime_cfg->cuda_linkage_max_kp;
            if (Kp_use >= min_kp && (max_kp <= 0 || Kp_use <= max_kp)) {
                int ils_iters_gpu = 0;
                int perturb_k_gpu = 0;
                if constexpr (UseIls) {
                    ils_iters_gpu = std::max(0, linkage_cfg.ils_rounds);
                    perturb_k_gpu = std::min(std::max(0, linkage_cfg.ils_perturb_layers), m);
                }
                const std::uint32_t seed_gpu =
                    static_cast<std::uint32_t>(linkage_cfg.seed + 997 * cluster_id);
                const std::uint64_t sample_id_offset =
                    (static_cast<std::uint64_t>(static_cast<std::uint32_t>(cluster_id)) << 32) |
                    static_cast<std::uint32_t>(li);
                int best_parent_gpu = -1;
                float best_cost_gpu = std::numeric_limits<float>::infinity();
                std::string gpu_err;

                if (device_rfull_ready) {
                    if (!FlushPendingRfullUpdates() ||
                        !EvaluateParentCandidatesBatchCudaDeviceRfull(cuda_ctx, pre_one,
                                                                      xi, d,
                                                                      tls.candidates.data(), Kp_use,
                                                                      icm_round,
                                                                      ils_iters_gpu,
                                                                      perturb_k_gpu,
                                                                      seed_gpu,
                                                                      sample_id_offset,
                                                                      &best_parent_gpu, &best_cost_gpu,
                                                                      &tls.B_best, &tls.a_best,
                                                                      &gpu_err)) {
                        LogWarn(std::string("CUDA device_rfull candidate eval failed: ") + gpu_err);
                        device_rfull_ready = false;
                    } else {
                        used_gpu = true;
                    }
                }
                if (!used_gpu) {
                    ensure_pinned(Kp_use);
                    if (pinned_Rp && pinned_cand) {
                        for (int j = 0; j < Kp_use; ++j) {
                            const int p_idx = tls.candidates[static_cast<std::size_t>(j)];
                            pinned_cand[j] = p_idx;
                            const float* rp = get_rcol(p_idx);
                            std::memcpy(pinned_Rp + static_cast<std::size_t>(j) * static_cast<std::size_t>(d),
                                        rp, sizeof(float) * static_cast<std::size_t>(d));
                        }
                        if (!EvaluateParentCandidatesBatchCuda(cuda_ctx, pre_one,
                                                               xi, d,
                                                               pinned_Rp, pinned_cand,
                                                               Kp_use, icm_round,
                                                               ils_iters_gpu,
                                                               perturb_k_gpu,
                                                               seed_gpu,
                                                               sample_id_offset,
                                                               &best_parent_gpu, &best_cost_gpu,
                                                               &tls.B_best, &tls.a_best,
                                                               &gpu_err)) {
                            LogWarn(std::string("GPU candidate eval failed; falling back to CPU: ") + gpu_err);
                            used_gpu = false;
                        } else {
                            used_gpu = true;
                        }
                    }
                }
                if (used_gpu) {
                    if (best_parent_gpu >= 0 && best_cost_gpu < bestE2) {
                        bestE2 = best_cost_gpu;
                        best_parent = best_parent_gpu;
                    }
                }
            }
        }
#endif

        if (!used_gpu) {
            const bool have_one_root_cross_gram = !G_one_root.data.empty();
            if (icm_round > 0 && have_one_root_cross_gram) {
                ComputeXcBase(pre_one, xi, &tls.xC_base_buf);
            }

            for (int p_idx : tls.candidates) {
                int new_depth = depths[static_cast<std::size_t>(p_idx)] + 1;
                if (new_depth > max_depth) {
                    continue;
                }

                const float* rp = get_rcol(p_idx);
                float re_norm2 = 0.0f;
                #pragma omp simd
                for (int r = 0; r < d; ++r) {
                    const float v = xi[r] - rp[r];
                    tls.re[static_cast<std::size_t>(r)] = v;
                }
                const float* re_ptr = tls.re.data();
                #pragma omp simd reduction(+:re_norm2)
                for (int r = 0; r < d; ++r) {
                    const float v = re_ptr[r];
                    re_norm2 += v * v;
                }

                QuantizeWithKmeansAbsNoNormalSingle(pre_one, tls.re,
                                                    &tls.B_init, &tls.a_init,
                                                    &tls.rC_buf, &tls.available);

                if (icm_round > 0) {
                    const bool have_one_root_cross_gram = !G_one_root.data.empty();
                    if (have_one_root_cross_gram) {
                        BuildXcOneForParentSingleGetCols(pre_root, pre_one, G_one_root,
                                                         tls.xC_base_buf,
                                                         get_bcol, get_acol,
                                                         parent, depths,
                                                         p_idx, &tls.xC_buf,
                                                         cluster_id);
                    } else {
                        ComputeXcFromResidual(pre_one, tls.re, &tls.xC_buf);
                    }
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
                    tls.xC_buf = tls.rC_buf;
                    for (int l = 0; l < m; ++l) {
                        const float alpha = tls.a_init[static_cast<std::size_t>(l)];
                        if (alpha == 0.0f) continue;
                        const int flat = pre_one.offsets[l] + static_cast<int>(tls.B_init[static_cast<std::size_t>(l)]);
                        const float* Gcol = pre_one.G.Col(flat);
                        float* dst = tls.xC_buf.data();
                        #pragma omp simd
                        for (int q = 0; q < H; ++q) {
                            dst[q] += alpha * Gcol[q];
                        }
                    }
                    if (!SolveLeastSquaresVectorFromResidual(pre_one, tls.re, tls.B_init,
                                                             &tls.A, &tls.b, &tls.work,
                                                             tls.a_init.data())) {
                        ++ls_failures;
                        continue;
                    }
                }

                float E2_best = SampleCostFromXc(pre_one, tls.xC_buf, tls.B_init, tls.a_init, re_norm2);

                if constexpr (UseIls) {
                    const int ils_rounds = std::max(0, linkage_cfg.ils_rounds);
                    const int ils_k = std::min(std::max(0, linkage_cfg.ils_perturb_layers), m);
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

                        const float E2_try =
                            SampleCostFromXc(pre_one, tls.xC_buf, tls.B_try, tls.a_try, re_norm2);

                        if (E2_try < E2_best) {
                            E2_best = E2_try;
                            tls.B_init = tls.B_try;
                            tls.a_init = tls.a_try;
                        }
                    }
                }

                if (E2_best < bestE2) {
                    bestE2 = E2_best;
                    best_parent = p_idx;
                    tls.B_best = tls.B_init;
                    tls.a_best = tls.a_init;
                }
            }
        }

        if (best_parent >= 0) {
            parent[static_cast<std::size_t>(li)] = best_parent + 1;
            const int gi = cols_real[static_cast<std::size_t>(real_idx)];
            FullCode* bdst = B_real->Col(gi);
            float* adst = a_real->Col(gi);
            for (int l = 0; l < m; ++l) {
                bdst[static_cast<std::size_t>(l)] = tls.B_best[static_cast<std::size_t>(l)];
                adst[static_cast<std::size_t>(l)] = tls.a_best[static_cast<std::size_t>(l)];
            }
            depths[static_cast<std::size_t>(li)] = depths[static_cast<std::size_t>(best_parent)] + 1;

            ReconstructNode(C_one.books, tls.B_best, tls.a_best, &tls.R_self);
            const float* rp = get_rcol(best_parent);
            float* rdst = R_full_shared->Col(gi);
            const float* rself_best = tls.R_self.data();
            #pragma omp simd
            for (int r = 0; r < d; ++r) {
                rdst[r] = rp[r] + rself_best[r];
            }
    #if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
            if (cuda_ctx && device_rfull_ready) {
                PushPendingRfullUpdate(li, rdst);
            }
    #endif
        }
    }
    }

#if defined(STLQ_ENABLE_CUDA)
    // No further GPU candidate-eval calls happen after this point for this cluster. Any remaining pending
    // device R_full updates are therefore unnecessary; drop them to avoid a potentially huge tail flush.
    pending_rfull_cols.clear();
    pending_rfull_cols_data.clear();
    if (pinned_Rp) {
        cudaFreeHost(pinned_Rp);
    }
    if (pinned_cand) {
        cudaFreeHost(pinned_cand);
    }
#endif

    LinkageStructure::Cluster cluster;
    cluster.cluster_id = cluster_id;
    cluster.n_real = n_real;
    cluster.n_virtual = n_virt;
    cluster.indices.resize(static_cast<std::size_t>(n_real));
    std::iota(cluster.indices.begin(), cluster.indices.end(), 0);
    cluster.parent_local = std::move(parent);
    FinalizeLinkageClusterDepthOrderInPlaceFromDepth(&cluster, depths.data());

    linkage->clusters.clear();
    linkage->clusters.push_back(std::move(cluster));
    linkage->max_depth = static_cast<int>(linkage->clusters.front().depth_offsets.size()) - 2;

    if (ls_failures > 0) {
        LogWarn("Multi-center linkage LS failures: " + std::to_string(ls_failures));
    }
}
