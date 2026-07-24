        struct MseSumStats
        {
            double sum = 0.0;
            double min = std::numeric_limits<double>::infinity();
            double max = 0.0;
        };

        MseSumStats MseSumStatsFromReconLocal(const ColMajorMatrix<float>& X,
                                              const ColMajorMatrix<float>& R_full) {
            MseSumStats out;
            const int d = X.rows;
            const int n = X.cols;
            if (d <= 0 || n <= 0 || R_full.rows != d || R_full.cols != n) {
                return out;
            }

            double sum = 0.0;
            double mn = std::numeric_limits<double>::infinity();
            double mx = 0.0;

#pragma omp parallel default(none) if (!omp_in_parallel()) shared(X, R_full, sum, mn, mx) firstprivate(d, n)
            {
                double sum_local = 0.0;
                double mn_local = std::numeric_limits<double>::infinity();
                double mx_local = 0.0;

#pragma omp for schedule(static)
                for (int i = 0; i < n; ++i) {
                    const float* xi = X.Col(i);
                    const float* ri = R_full.Col(i);
                    double mse = 0.0;
#pragma omp simd reduction(+:mse)
                    for (int r = 0; r < d; ++r) {
                        const double diff = static_cast<double>(xi[r]) - static_cast<double>(ri[r]);
                        mse += diff * diff;
                    }
                    sum_local += mse;
                    mn_local = std::min(mn_local, mse);
                    mx_local = std::max(mx_local, mse);
                }

#pragma omp atomic
                sum += sum_local;
#pragma omp critical
                {
                    mn = std::min(mn, mn_local);
                    mx = std::max(mx, mx_local);
                }
            }

            out.sum = sum;
            out.min = mn;
            out.max = mx;
            return out;
        }

        void ComputeCosineAnglesOneCluster(const ColMajorMatrix<float>& X,
                                           const float* center,
                                           std::vector<float>* cos_angles) {
            const int d = X.rows;
            const int n = X.cols;
            cos_angles->assign(static_cast<std::size_t>(n), 0.0f);
            if (d <= 0 || n <= 0) {
                return;
            }

            float cnorm = 0.0f;
#pragma omp simd reduction(+:cnorm)
            for (int r = 0; r < d; ++r) {
                cnorm += center[r] * center[r];
            }
            cnorm = std::sqrt(std::max(cnorm, 1e-12f));

#pragma omp parallel for default(none) if (!omp_in_parallel()) schedule(static) shared(X, center, cos_angles) firstprivate(d, n, cnorm)
            for (int i = 0; i < n; ++i) {
                const float* x = X.Col(i);
                float xnorm2 = 0.0f;
                float dot = 0.0f;
#pragma omp simd reduction(+:xnorm2,dot)
                for (int r = 0; r < d; ++r) {
                    const float v = x[r];
                    xnorm2 += v * v;
                    dot += v * center[r];
                }
                const float denom = std::max(cnorm * std::sqrt(std::max(xnorm2, 1e-12f)), 1e-12f);
                (*cos_angles)[static_cast<std::size_t>(i)] = dot / denom;
            }
        }

        // Reconstruct all vectors using C_root with a fixed routing/root code for layer0.
        // This is required when h0_root can exceed 256 (root codes do not fit in `FullCode` which is now u8), and more
        // generally avoids relying on `B(0,i)` to carry the IVF cluster id in large-scale pipelines.
        void ReconstructAllIntoFixedRoot(const std::vector<ColMajorMatrix<float>>& C_root,
                                         int forced_root_code,
                                         const ColMajorMatrix<FullCode>& B,
                                         const ColMajorMatrix<float>& a,
                                         ColMajorMatrix<float>* out) {
            if (!out || C_root.empty()) {
                return;
            }
            const int d = C_root.front().rows;
            const int n = B.cols;
            const int m = static_cast<int>(C_root.size());
            out->rows = d;
            out->cols = n;
            const std::size_t need = static_cast<std::size_t>(d) * static_cast<std::size_t>(n);
            if (out->data.size() < need) {
                out->data.resize(need);
            }
            if (d <= 0 || n <= 0 || forced_root_code < 0 || forced_root_code >= C_root.front().cols) {
                return;
            }
            const float* c0 = C_root[0].Col(forced_root_code);

#pragma omp parallel for default(none) if (!omp_in_parallel()) schedule(static) shared(out, a, B, C_root, c0) firstprivate(d, n, m)
            for (int i = 0; i < n; ++i) {
                float* dst = out->Col(i);
                const float a0 = a(0, i);
#pragma omp simd
                for (int r = 0; r < d; ++r) {
                    dst[r] = a0 * c0[r];
                }
                for (int l = 1; l < m; ++l) {
                    const int code = static_cast<int>(B(l, i));
                    const float al = a(l, i);
                    const float* col = C_root[static_cast<std::size_t>(l)].Col(code);
#pragma omp simd
                    for (int r = 0; r < d; ++r) {
                        dst[r] += al * col[r];
                    }
                }
            }
        }

        template <bool UseIls>
        void LinkageTwoCodebookInnerToOuterOneCluster(const LinkageBuildConfig& linkage_cfg,
                                                      const HnswConfig& hnsw_cfg,
                                                      const RuntimeConfig* runtime_cfg,
#if defined(STLQ_ENABLE_CUDA)
                                                      CudaStreamKernelsPool* cuda_pool,
#endif
                                                      int forced_root_code,
                                                      const ColMajorMatrix<float>& X, // d×n_local
                                                      const CodebookPack& C_root,
                                                      const Precomp& pre_root,
                                                      const CodebookPack& C_one,
                                                      const Precomp& pre_one,
                                                      const ColMajorMatrix<float>& G_one_root,
                                                      ColMajorMatrix<FullCode>* B, // m×n_local (updated in-place)
                                                      ColMajorMatrix<float>* a, // m×n_local
                                                      LinkageStructure::Cluster* out_cluster,
                                                      int* ls_failures_out,
                                                      MseStats* linkaged_mse_out,
                                                      double* linkaged_mse_sum_out) {
            const int d = X.rows;
            const int n = X.cols;
            const int m = pre_one.m;
            const int H = pre_one.H;

            const int num_layers = std::max(1, linkage_cfg.num_layers);
            const int knn_k = std::max(1, linkage_cfg.knn_k);
            const int depth_k = std::max(1, linkage_cfg.depth_k);
            const int max_depth = std::max(1, linkage_cfg.max_depth);
            const int icm_round = std::max(0, linkage_cfg.icm_round);
            const auto root_percentile = static_cast<float>(linkage_cfg.root_percentile);
            const int ils_rounds = std::max(0, linkage_cfg.ils_rounds);
            const int ils_k = std::min(std::max(0, linkage_cfg.ils_perturb_layers), m);

            if (!out_cluster || !B || !a) {
                return;
            }
            out_cluster->cluster_id = forced_root_code;
            out_cluster->n_real = n;
            out_cluster->n_virtual = 0;
            out_cluster->indices.resize(static_cast<std::size_t>(n));
            std::iota(out_cluster->indices.begin(), out_cluster->indices.end(), 0);
            out_cluster->parent_local.assign(static_cast<std::size_t>(n), 0);
            out_cluster->depth_offsets.clear();

            if (ls_failures_out) {
                *ls_failures_out = 0;
            }
            if (d <= 0 || n <= 0) {
                out_cluster->depth_offsets.assign(2, 0);
                return;
            }

            // Partition cluster into angle-based layers.
            std::vector<float> cos_angles;
            ComputeCosineAnglesOneCluster(X, C_root.books.front().Col(forced_root_code), &cos_angles);
            std::vector<int> cluster_indices(static_cast<std::size_t>(n), 0);
            std::iota(cluster_indices.begin(), cluster_indices.end(), 0);

            std::vector<std::vector<int>> layers_local;
            PartitionClusterLayers(cos_angles, cluster_indices, num_layers, root_percentile, &layers_local);

            std::vector<int> layer_of_point(static_cast<std::size_t>(n), 0);
            for (int layer = 0; layer < num_layers; ++layer) {
                for (int local_idx : layers_local[static_cast<std::size_t>(layer)]) {
                    layer_of_point[static_cast<std::size_t>(local_idx)] = layer;
                }
            }

            // Initial reconstruction from base encoding (C_root).
            ColMajorMatrix<float> R_full;
            const bool need_root_u32 = (!C_root.books.empty() && C_root.books.front().cols > 256);
            if (!need_root_u32) {
                R_full = ReconstructAll(C_root.books, *B, *a);
            }
            else {
                // When h0_root > 256, `FullCode` (u8) cannot represent the forced root code (cid) in row0.
                // The correct semantics here is constant root=forced_root_code for all samples in this cluster.
                // Keep the reconstruction exact by using `forced_root_code` for layer0 and `B(l, i)` for l>=1.
                R_full.rows = d;
                R_full.cols = n;
                R_full.data.assign(static_cast<std::size_t>(d) * static_cast<std::size_t>(n), 0.0f);
                const float* c0 = C_root.books[0].Col(forced_root_code);
                for (int i = 0; i < n; ++i) {
                    float* dst = R_full.Col(i);
                    const float a0 = (*a)(0, i);
#pragma omp simd
                    for (int r = 0; r < d; ++r) {
                        dst[r] = a0 * c0[r];
                    }
                    for (int l = 1; l < m; ++l) {
                        const int code = static_cast<int>((*B)(l, i));
                        const float al = (*a)(l, i);
                        const float* col = C_root.books[static_cast<std::size_t>(l)].Col(code);
#pragma omp simd
                        for (int r = 0; r < d; ++r) {
                            dst[r] += al * col[r];
                        }
                    }
                }
            }
            std::vector<float> base_mse;
            SquaredErrorsFromRecon(X, R_full, &base_mse);

            LinkageTls tls = MakeLinkageTls(d, m, H);
            std::mt19937 rng(static_cast<std::mt19937::result_type>(linkage_cfg.seed + 997 * forced_root_code));

            std::vector<int> parent_local(static_cast<std::size_t>(n), 0);
            std::vector<int> depth_local(static_cast<std::size_t>(n), 0);

            std::vector<FullCode*> B_cols(static_cast<std::size_t>(n), nullptr);
            std::vector<float*> a_cols(static_cast<std::size_t>(n), nullptr);
            for (int i = 0; i < n; ++i) {
                B_cols[static_cast<std::size_t>(i)] = B->Col(i);
                a_cols[static_cast<std::size_t>(i)] = a->Col(i);
            }

            const int candidate_multiplier = std::max(1, hnsw_cfg.candidate_multiplier_good);
            ClusterHnswIndex hnsw_index;
            const int desired_ef_construction = std::max(1, depth_k * candidate_multiplier * knn_k);
            const int ef_construction = std::max(1, std::min(desired_ef_construction, n));
            {
                const double t0 =
                    stlq::linkage::LinkageBuildProfileEnabled() ? stlq::linkage::LinkageBuildWallNowS() : 0.0;
                hnsw_index.Build(X, cluster_indices, hnsw_cfg, ef_construction);
                if (stlq::linkage::LinkageBuildProfileEnabled()) {
                    if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                        prof->good_hnsw_build_cpu_s += stlq::linkage::LinkageBuildWallNowS() - t0;
                    }
                }
            }

            int ls_failures = 0;
            // In large pipeline we always omit full-precomp for C_root. For C_one we may or may not have full-precomp:
            // - when Precomp::G is present, we can use the existing (full-precomp) CUDA evaluators and CPU tiny paths.
            // - when Precomp::G is missing (meta-only), we must use the large-root/split-root CUDA evaluators.
            const bool use_large_root_eval = pre_one.G.data.empty();

            // Stage-3 (good clusters): batch inner-candidates across many nodes in the same layer.
            [[maybe_unused]] bool batch_inner_enabled_cfg = false;
            [[maybe_unused]] int batch_nodes_max = 1;
            [[maybe_unused]] int batch_pairs_max = std::numeric_limits<int>::max();
            std::vector<int> batch_node_local;
            std::vector<float> batch_X;
            std::vector<int> batch_inner_offsets;
            std::vector<int> batch_inner_cand;
            std::vector<int> batch_pair_node;
            std::vector<int> batch_same_offsets;
            std::vector<int> batch_same_cand;
            std::vector<int> batch_best_parent_inner;
            std::vector<float> batch_best_cost_inner;
            std::vector<FullCode> batch_best_B_inner;
            std::vector<float> batch_best_a_inner;
            std::vector<int> batch_same_frozen_offsets;
            std::vector<int> batch_same_frozen_cand;
            std::vector<int> batch_same_frozen_pair_node;
            std::vector<int> batch_best_parent_same_frozen;
            std::vector<float> batch_best_cost_same_frozen;
            std::vector<FullCode> batch_best_B_same_frozen;
            std::vector<float> batch_best_a_same_frozen;
            std::vector<std::uint64_t> batch_node_sample_id_base;

            CudaCtx* cuda_ctx = nullptr;
            bool device_rfull_ready = false;

#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
            const bool cuda_linkage_enabled =
                runtime_cfg && runtime_cfg->use_cuda && cuda_pool;
            struct CudaLease
            {
                CudaStreamKernelsPool* pool = nullptr;
                CudaCtx* ctx = nullptr;

                ~CudaLease() {
                    if (pool && ctx) {
                        pool->Release(ctx);
                    }
                }
            } cuda_lease;
            if (cuda_linkage_enabled) {
                // In large-root (meta-only precomp) mode we cannot fall back to CPU candidate evaluation.
                // Force acquiring a CUDA context even if `cuda_linkage_wait_for_ctx` is false.
                const bool must_have_ctx = use_large_root_eval;
                cuda_ctx = (must_have_ctx || (runtime_cfg && runtime_cfg->cuda_linkage_wait_for_ctx))
                               ? cuda_pool->Acquire()
                               : cuda_pool->TryAcquire();
                cuda_lease.pool = cuda_pool;
                cuda_lease.ctx = cuda_ctx;
            }

            if (cuda_ctx && runtime_cfg && runtime_cfg->cuda_linkage_use_device_rfull) {
                std::string up_err;
                if (CudaLinkageUploadClusterRfull(cuda_ctx, R_full.data.data(), d, n, &up_err)) {
                    device_rfull_ready = true;
                }
                else {
                    LogWarn(std::string("CudaLinkageUploadClusterRfull failed, fallback to pinned_Rp: ") + up_err);
                    device_rfull_ready = false;
                }
            }

            if (cuda_ctx && runtime_cfg && device_rfull_ready) {
                batch_inner_enabled_cfg = runtime_cfg->cuda_linkage_batch_inner_enabled;
                batch_nodes_max = std::max(1, runtime_cfg->cuda_linkage_batch_nodes);
                // Same-layer dynamic window batching works best when the per-layer batch is at least as large
                // as the configured window size; otherwise we end up flushing tiny windows too frequently.
                if (runtime_cfg->cuda_linkage_same_dynamic_min_pairs > 0) {
                    batch_nodes_max = std::max(batch_nodes_max,
                                               std::max(1, runtime_cfg->cuda_linkage_same_dynamic_nodes));
                }
                batch_pairs_max = (runtime_cfg->cuda_linkage_batch_max_pairs > 0)
                                      ? runtime_cfg->cuda_linkage_batch_max_pairs
                                      : std::numeric_limits<int>::max();
            }

            float* pinned_Rp = nullptr; // d×cap_kp (col-major), host pinned
            int* pinned_cand = nullptr; // cap_kp, host pinned
            int pinned_kp_cap = 0;
            auto ensure_pinned = [&](int need_kp)
            {
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
#endif

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

                const int inner_lo = std::max(0, layer_idx - depth_k);
                const int inner_hi = layer_idx - 1;

                tls.local_pos.resize(static_cast<std::size_t>(n));
                std::fill(tls.local_pos.begin(), tls.local_pos.end(), -1);
                for (std::size_t pos = 0; pos < current_layer.size(); ++pos) {
                    const int local_idx = current_layer[pos];
                    tls.local_pos[static_cast<std::size_t>(local_idx)] = static_cast<int>(pos);
                }

#if defined(STLQ_ENABLE_CUDA)
                if (cuda_ctx && runtime_cfg && device_rfull_ready && batch_inner_enabled_cfg) {
                    // Per-layer map: local node -> rank within the current batch window (only valid inside FlushBatchAndCommit).
                    // Kept across flushes to avoid clearing O(n) each time; we reset touched entries at the end of each flush.
                    std::vector<int> batch_rank_by_local(static_cast<std::size_t>(n), -1);

                    // Batched device R_full updates: accumulate per-node commits and only flush before GPU evals.
                    std::vector<int> pending_rfull_cols;
                    std::vector<float> pending_rfull_cols_data; // packed columns, size = pending_rfull_cols.size() * d
                    pending_rfull_cols.clear();
                    pending_rfull_cols_data.clear();
                    pending_rfull_cols.reserve(1024);
                    pending_rfull_cols_data.reserve(static_cast<std::size_t>(1024) * static_cast<std::size_t>(d));

                    const auto PushPendingRfullUpdate = [&](int col, const float* r_col)
                    {
                        pending_rfull_cols.push_back(col);
                        const std::size_t old = pending_rfull_cols_data.size();
                        pending_rfull_cols_data.resize(old + static_cast<std::size_t>(d));
                        std::memcpy(pending_rfull_cols_data.data() + old, r_col,
                                    sizeof(float) * static_cast<std::size_t>(d));
                    };
                    const auto FlushPendingRfullUpdates = [&]() -> bool
                    {
                        if (pending_rfull_cols.empty()) return true;
                        if (!cuda_ctx || !device_rfull_ready) {
                            pending_rfull_cols.clear();
                            pending_rfull_cols_data.clear();
                            return false;
                        }
                        const double t0_up =
                            stlq::linkage::LinkageBuildProfileEnabled() ? stlq::linkage::LinkageBuildWallNowS() : 0.0;
                        constexpr int kMaxColsPerFlush = 2048;
                        std::size_t off = 0;
                        std::uint64_t n_batches = 0;
                        while (off < pending_rfull_cols.size()) {
                            const int take =
                                static_cast<int>(std::min<std::size_t>(pending_rfull_cols.size() - off,
                                                                       static_cast<std::size_t>(kMaxColsPerFlush)));
                            std::string up_err;
                            const int* cols = pending_rfull_cols.data() + off;
                            const float* data =
                                pending_rfull_cols_data.data() + off * static_cast<std::size_t>(d);
                            if (!CudaLinkageUpdateClusterRfullColumnsBatch(cuda_ctx, cols, data, d, take, &up_err)) {
                                LogWarn(std::string(
                                        "CudaLinkageUpdateClusterRfullColumnsBatch failed, disable device_rfull: ") +
                                    up_err);
                                device_rfull_ready = false;
                                batch_inner_enabled_cfg = false;
                                pending_rfull_cols.clear();
                                pending_rfull_cols_data.clear();
                                return false;
                            }
                            off += static_cast<std::size_t>(take);
                            n_batches += 1;
                        }
                        if (stlq::linkage::LinkageBuildProfileEnabled()) {
                            if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                                const double dt = stlq::linkage::LinkageBuildWallNowS() - t0_up;
                                const auto cols_n =
                                    static_cast<std::uint64_t>(pending_rfull_cols.size());
                                prof->good_rfull_update_s += dt;
                                prof->good_rfull_update_calls += cols_n;
                                prof->good_rfull_update_batches += n_batches;
                                prof->rfull_update_s += dt;
                                prof->rfull_update_calls += cols_n;
                                prof->rfull_update_batches += n_batches;
                            }
                        }
                        pending_rfull_cols.clear();
                        pending_rfull_cols_data.clear();
                        return true;
                    };

                    batch_node_local.clear();
                    batch_X.clear();
                    batch_inner_offsets.clear();
                    batch_inner_offsets.push_back(0);
                    batch_inner_cand.clear();
                    batch_pair_node.clear();
                    batch_same_offsets.clear();
                    batch_same_offsets.push_back(0);
                    batch_same_cand.clear();
                    batch_same_frozen_offsets.clear();
                    batch_same_frozen_offsets.push_back(0);
                    batch_same_frozen_cand.clear();
                    batch_same_frozen_pair_node.clear();

                    auto FlushBatchAndCommit = [&]()
                    {
                        const int Bbatch = static_cast<int>(batch_node_local.size());
                        if (Bbatch <= 0) {
                            return;
                        }
                        // Stage-3e profiling-only: many-nodes batch quality attribution (good clusters).
                        constexpr int kGoodMnSrcInner = 0;
                        constexpr int kGoodMnSrcSameFrozen = 1;
                        constexpr int kGoodMnSrcSameDynBefore = 2;
                        constexpr int kGoodMnSrcSameDynWindow = 3;
                        const auto BucketPairs12 = [](int npairs) -> int
                        {
                            if (npairs <= 0) return 0; // 0
                            if (npairs <= 127) return 1; // 1-127
                            if (npairs <= 255) return 2; // 128-255
                            if (npairs <= 511) return 3; // 256-511
                            if (npairs <= 1023) return 4; // 512-1023
                            if (npairs <= 2047) return 5; // 1024-2047
                            if (npairs <= 4095) return 6; // 2048-4095
                            if (npairs <= 8191) return 7; // 4096-8191
                            if (npairs <= 16383) return 8; // 8192-16383
                            if (npairs <= 32767) return 9; // 16384-32767
                            if (npairs <= 65535) return 10; // 32768-65535
                            return 11; // 65536+
                        };
                        const auto BucketNodes12 = [](int nodes) -> int
                        {
                            if (nodes <= 0) return 0; // 0
                            if (nodes == 1) return 1; // 1
                            if (nodes == 2) return 2; // 2
                            if (nodes <= 4) return 3; // 3-4
                            if (nodes <= 8) return 4; // 5-8
                            if (nodes <= 16) return 5; // 9-16
                            if (nodes <= 32) return 6; // 17-32
                            if (nodes <= 64) return 7; // 33-64
                            if (nodes <= 128) return 8; // 65-128
                            if (nodes <= 256) return 9; // 129-256
                            if (nodes <= 512) return 10; // 257-512
                            return 11; // 513+
                        };
                        const auto UpdateGoodManyNodesBatchQuality =
                            [&](stlq::linkage::LinkageBuildProfileTls* prof,
                                int src,
                                int nodes,
                                int npairs)
                        {
                            if (!prof) return;
                            if (src < 0 || src >= 4) return;
                            prof->good_mn_calls_by_src[src] += 1;
                            prof->good_mn_nodes_sum_by_src[src] += static_cast<std::uint64_t>(std::max(0, nodes));
                            prof->good_mn_pairs_sum_by_src[src] += static_cast<std::uint64_t>(std::max(0, npairs));
                            const int b_pairs = BucketPairs12(npairs);
                            const int b_nodes = BucketNodes12(nodes);
                            prof->good_mn_npairs_hist[b_pairs] += 1;
                            prof->good_mn_nodes_hist[b_nodes] += 1;
                            prof->good_mn_npairs_max = std::max(prof->good_mn_npairs_max, npairs);
                            prof->good_mn_nodes_max = std::max(prof->good_mn_nodes_max, nodes);
                        };
                        batch_node_sample_id_base.resize(static_cast<std::size_t>(Bbatch));
                        for (int i = 0; i < Bbatch; ++i) {
                            const auto node_local_u32 =
                                static_cast<std::uint32_t>(batch_node_local[static_cast<std::size_t>(i)]);
                            batch_node_sample_id_base[static_cast<std::size_t>(i)] =
                                (static_cast<std::uint64_t>(static_cast<std::uint32_t>(forced_root_code)) << 32) |
                                static_cast<std::uint64_t>(node_local_u32);
                        }
                        const bool async_eval_enabled =
                            (runtime_cfg && runtime_cfg->cuda_linkage_eval_async_pinned_mb > 0);
                        struct AsyncEvalJob
                        {
                            bool in_flight = false;
                            bool large_root = false;
                            int src = -1;
                            int nodes = 0;
                            int npairs = 0;
                            int* best_parent = nullptr;
                            float* best_cost = nullptr;
                            FullCode* best_B = nullptr;
                            float* best_a = nullptr;
                        } async_job;
                        auto FinishAsyncEvalJob = [&]() -> bool
                        {
                            if (!async_job.in_flight) return true;
                            std::string finish_err;
                            const double t0_wait =
                                stlq::linkage::LinkageBuildProfileEnabled()
                                    ? stlq::linkage::LinkageBuildWallNowS()
                                    : 0.0;
                            const bool ok =
                                async_job.large_root
                                    ? EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExFinish(
                                        cuda_ctx,
                                        async_job.best_parent,
                                        async_job.best_cost,
                                        async_job.best_B,
                                        async_job.best_a,
                                        &finish_err)
                                    : EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseExFinish(
                                        cuda_ctx,
                                        async_job.best_parent,
                                        async_job.best_cost,
                                        async_job.best_B,
                                        async_job.best_a,
                                        &finish_err);
                            if (!ok) {
                                LogWarn(std::string(
                                        "CUDA async many-nodes eval finish failed; disable batching for this cluster: ")
                                    +
                                    finish_err);
                                async_job.in_flight = false;
                                return false;
                            }
                            if (stlq::linkage::LinkageBuildProfileEnabled()) {
                                if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                                    const double dt = stlq::linkage::LinkageBuildWallNowS() - t0_wait;
                                    prof->good_gpu_wait_s += dt;
                                    if (async_job.src >= 0 && async_job.src < 4) {
                                        prof->good_gpu_wait_mn_s_by_src[async_job.src] += dt;
                                        prof->good_gpu_wait_mn_calls_by_src[async_job.src] += 1;
                                    }
                                    prof->good_gpu_calls_many_nodes += 1;
                                    UpdateGoodManyNodesBatchQuality(
                                        prof, async_job.src, /*nodes=*/async_job.nodes, /*npairs=*/async_job.npairs);
                                }
                            }
                            async_job.in_flight = false;
                            return true;
                        };
                        for (int i = 0; i < Bbatch; ++i) {
                            batch_rank_by_local[static_cast<std::size_t>(batch_node_local[static_cast<std::size_t>(i)])]
                                = i;
                        }
                        const int Npairs_inner = static_cast<int>(batch_inner_cand.size());

                        // IMPORTANT: inner candidates are required for correctness. A previous batched implementation
                        // skipped GPU evaluation when total pairs < `batch_pairs_min`, which silently dropped inner parents
                        // for small batches (especially SIFTSMALL / early layers) and caused large linkage-ratio regressions.
                        //
                        // Fix: always evaluate inner candidates (when present). We keep the batching structure for performance,
                        // and do not fall back to per-node CUDA here (it explodes GPU-call count and can be slower than CPU).
                        bool have_batch_inner = (Npairs_inner > 0);
                        if (have_batch_inner) {
                            batch_best_parent_inner.assign(static_cast<std::size_t>(Bbatch), -1);
                            batch_best_cost_inner.assign(static_cast<std::size_t>(Bbatch),
                                                         std::numeric_limits<float>::infinity());
                            batch_best_B_inner.assign(static_cast<std::size_t>(Bbatch) * static_cast<std::size_t>(m),
                                                      0);
                            batch_best_a_inner.assign(static_cast<std::size_t>(Bbatch) * static_cast<std::size_t>(m),
                                                      0.0f);
                        }

                        if (have_batch_inner) {
                            const auto seed_gpu =
                                static_cast<std::uint32_t>(linkage_cfg.seed + 997 * forced_root_code);
                            int ils_iters_gpu = 0;
                            int perturb_k_gpu = 0;
                            if constexpr (UseIls) {
                                ils_iters_gpu = ils_rounds;
                                perturb_k_gpu = ils_k;
                            }
                            std::string batch_err;
                            bool enqueued = false;
                            if (async_eval_enabled) {
                                enqueued =
                                    use_large_root_eval
                                        ? EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExEnqueue(
                                            cuda_ctx,
                                            C_one,
                                            batch_X.data(),
                                            d,
                                            Bbatch,
                                            batch_inner_cand.data(),
                                            batch_pair_node.data(),
                                            batch_inner_offsets.data(),
                                            Npairs_inner,
                                            icm_round,
                                            ils_iters_gpu,
                                            perturb_k_gpu,
                                            seed_gpu,
                                            batch_node_sample_id_base.data(),
                                            /*pair_t_override=*/nullptr,
                                            &batch_err)
                                        : EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseExEnqueue(
                                            cuda_ctx,
                                            pre_one,
                                            batch_X.data(),
                                            d,
                                            Bbatch,
                                            batch_inner_cand.data(),
                                            batch_pair_node.data(),
                                            batch_inner_offsets.data(),
                                            Npairs_inner,
                                            icm_round,
                                            ils_iters_gpu,
                                            perturb_k_gpu,
                                            seed_gpu,
                                            batch_node_sample_id_base.data(),
                                            /*pair_t_override=*/nullptr,
                                            &batch_err);
                                if (enqueued) {
                                    async_job.in_flight = true;
                                    async_job.large_root = use_large_root_eval;
                                    async_job.src = kGoodMnSrcInner;
                                    async_job.nodes = Bbatch;
                                    async_job.npairs = Npairs_inner;
                                    async_job.best_parent = batch_best_parent_inner.data();
                                    async_job.best_cost = batch_best_cost_inner.data();
                                    async_job.best_B = batch_best_B_inner.data();
                                    async_job.best_a = batch_best_a_inner.data();
                                }
                            }
                            if (!enqueued) {
                                const double t0_gpu =
                                    stlq::linkage::LinkageBuildProfileEnabled()
                                        ? stlq::linkage::LinkageBuildWallNowS()
                                        : 0.0;
                                const bool ok_gpu =
                                    use_large_root_eval
                                        ? EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRoot(
                                            cuda_ctx,
                                            C_one,
                                            batch_X.data(),
                                            d,
                                            Bbatch,
                                            batch_inner_cand.data(),
                                            batch_pair_node.data(),
                                            batch_inner_offsets.data(),
                                            Npairs_inner,
                                            icm_round,
                                            ils_iters_gpu,
                                            perturb_k_gpu,
                                            seed_gpu,
                                            batch_node_sample_id_base.data(),
                                            batch_best_parent_inner.data(),
                                            batch_best_cost_inner.data(),
                                            batch_best_B_inner.data(),
                                            batch_best_a_inner.data(),
                                            &batch_err)
                                        : EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBase(
                                            cuda_ctx,
                                            pre_one,
                                            batch_X.data(),
                                            d,
                                            Bbatch,
                                            batch_inner_cand.data(),
                                            batch_pair_node.data(),
                                            batch_inner_offsets.data(),
                                            Npairs_inner,
                                            icm_round,
                                            ils_iters_gpu,
                                            perturb_k_gpu,
                                            seed_gpu,
                                            batch_node_sample_id_base.data(),
                                            batch_best_parent_inner.data(),
                                            batch_best_cost_inner.data(),
                                            batch_best_B_inner.data(),
                                            batch_best_a_inner.data(),
                                            &batch_err);
                                if (!ok_gpu) {
                                    LogWarn(std::string(
                                            "CUDA batch inner-candidates eval failed; disable batching for this cluster: ")
                                        +
                                        batch_err);
                                    batch_inner_enabled_cfg = false;
                                    have_batch_inner = false;
                                }
                                if (stlq::linkage::LinkageBuildProfileEnabled()) {
                                    if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                                        const double dt = stlq::linkage::LinkageBuildWallNowS() - t0_gpu;
                                        prof->good_gpu_wait_s += dt;
                                        prof->good_gpu_wait_mn_s_by_src[kGoodMnSrcInner] += dt;
                                        prof->good_gpu_wait_mn_calls_by_src[kGoodMnSrcInner] += 1;
                                        prof->good_gpu_calls_many_nodes += 1;
                                        UpdateGoodManyNodesBatchQuality(
                                            prof, kGoodMnSrcInner, /*nodes=*/Bbatch, /*npairs=*/Npairs_inner);
                                    }
                                }
                            }
                        }

                        batch_same_frozen_offsets.clear();
                        batch_same_frozen_offsets.push_back(0);
                        batch_same_frozen_cand.clear();
                        batch_same_frozen_pair_node.clear();
                        for (int i = 0; i < Bbatch; ++i) {
                            const int same_start = batch_same_offsets[static_cast<std::size_t>(i)];
                            const int same_stop = batch_same_offsets[static_cast<std::size_t>(i + 1)];
                            for (int j = same_start; j < same_stop; ++j) {
                                const int p_local = batch_same_cand[static_cast<std::size_t>(j)];
                                const int rank = batch_rank_by_local[static_cast<std::size_t>(p_local)];
                                if (rank >= 0) {
                                    // dynamic (within this batch window)
                                    continue; // dynamic (within this batch)
                                }
                                const int new_depth = depth_local[static_cast<std::size_t>(p_local)] + 1;
                                if (new_depth > max_depth) {
                                    continue;
                                }
                                batch_same_frozen_cand.push_back(p_local);
                                batch_same_frozen_pair_node.push_back(i);
                            }
                            batch_same_frozen_offsets.push_back(static_cast<int>(batch_same_frozen_cand.size()));
                        }

                        const int Npairs_same_frozen = static_cast<int>(batch_same_frozen_cand.size());
                        if (!FinishAsyncEvalJob()) {
                            batch_inner_enabled_cfg = false;
                            have_batch_inner = false;
                        }
                        bool have_batch_same_frozen = false;
                        if (Npairs_same_frozen > 0) {
                            if (!FlushPendingRfullUpdates()) {
                                batch_inner_enabled_cfg = false;
                                have_batch_inner = false;
                            }
                            batch_best_parent_same_frozen.assign(static_cast<std::size_t>(Bbatch), -1);
                            batch_best_cost_same_frozen.assign(static_cast<std::size_t>(Bbatch),
                                                               std::numeric_limits<float>::infinity());
                            batch_best_B_same_frozen.assign(
                                static_cast<std::size_t>(Bbatch) * static_cast<std::size_t>(m), 0);
                            batch_best_a_same_frozen.assign(
                                static_cast<std::size_t>(Bbatch) * static_cast<std::size_t>(m), 0.0f);

                            const auto seed_gpu =
                                static_cast<std::uint32_t>(linkage_cfg.seed + 997 * forced_root_code);
                            int ils_iters_gpu = 0;
                            int perturb_k_gpu = 0;
                            if constexpr (UseIls) {
                                ils_iters_gpu = ils_rounds;
                                perturb_k_gpu = ils_k;
                            }
                            std::string batch_err;
                            bool enqueued = false;
                            if (async_eval_enabled) {
                                enqueued =
                                    use_large_root_eval
                                        ? EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExEnqueue(
                                            cuda_ctx,
                                            C_one,
                                            batch_X.data(),
                                            d,
                                            Bbatch,
                                            batch_same_frozen_cand.data(),
                                            batch_same_frozen_pair_node.data(),
                                            batch_same_frozen_offsets.data(),
                                            Npairs_same_frozen,
                                            icm_round,
                                            ils_iters_gpu,
                                            perturb_k_gpu,
                                            seed_gpu,
                                            batch_node_sample_id_base.data(),
                                            /*pair_t_override=*/nullptr,
                                            &batch_err)
                                        : EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseExEnqueue(
                                            cuda_ctx,
                                            pre_one,
                                            batch_X.data(),
                                            d,
                                            Bbatch,
                                            batch_same_frozen_cand.data(),
                                            batch_same_frozen_pair_node.data(),
                                            batch_same_frozen_offsets.data(),
                                            Npairs_same_frozen,
                                            icm_round,
                                            ils_iters_gpu,
                                            perturb_k_gpu,
                                            seed_gpu,
                                            batch_node_sample_id_base.data(),
                                            /*pair_t_override=*/nullptr,
                                            &batch_err);
                                if (enqueued) {
                                    async_job.in_flight = true;
                                    async_job.large_root = use_large_root_eval;
                                    async_job.src = kGoodMnSrcSameFrozen;
                                    async_job.nodes = Bbatch;
                                    async_job.npairs = Npairs_same_frozen;
                                    async_job.best_parent = batch_best_parent_same_frozen.data();
                                    async_job.best_cost = batch_best_cost_same_frozen.data();
                                    async_job.best_B = batch_best_B_same_frozen.data();
                                    async_job.best_a = batch_best_a_same_frozen.data();
                                    have_batch_same_frozen = true;
                                }
                            }
                            if (!enqueued) {
                                const double t0_gpu =
                                    stlq::linkage::LinkageBuildProfileEnabled()
                                        ? stlq::linkage::LinkageBuildWallNowS()
                                        : 0.0;
                                const bool ok_gpu =
                                    use_large_root_eval
                                        ? EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRoot(
                                            cuda_ctx,
                                            C_one,
                                            batch_X.data(),
                                            d,
                                            Bbatch,
                                            batch_same_frozen_cand.data(),
                                            batch_same_frozen_pair_node.data(),
                                            batch_same_frozen_offsets.data(),
                                            Npairs_same_frozen,
                                            icm_round,
                                            ils_iters_gpu,
                                            perturb_k_gpu,
                                            seed_gpu,
                                            batch_node_sample_id_base.data(),
                                            batch_best_parent_same_frozen.data(),
                                            batch_best_cost_same_frozen.data(),
                                            batch_best_B_same_frozen.data(),
                                            batch_best_a_same_frozen.data(),
                                            &batch_err)
                                        : EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBase(
                                            cuda_ctx,
                                            pre_one,
                                            batch_X.data(),
                                            d,
                                            Bbatch,
                                            batch_same_frozen_cand.data(),
                                            batch_same_frozen_pair_node.data(),
                                            batch_same_frozen_offsets.data(),
                                            Npairs_same_frozen,
                                            icm_round,
                                            ils_iters_gpu,
                                            perturb_k_gpu,
                                            seed_gpu,
                                            batch_node_sample_id_base.data(),
                                            batch_best_parent_same_frozen.data(),
                                            batch_best_cost_same_frozen.data(),
                                            batch_best_B_same_frozen.data(),
                                            batch_best_a_same_frozen.data(),
                                            &batch_err);
                                if (!ok_gpu) {
                                    LogWarn(std::string(
                                            "CUDA batch same-layer frozen eval failed; disable batching for this cluster: ")
                                        +
                                        batch_err);
                                    batch_inner_enabled_cfg = false;
                                    have_batch_same_frozen = false;
                                }
                                else {
                                    have_batch_same_frozen = true;
                                }
                                if (stlq::linkage::LinkageBuildProfileEnabled()) {
                                    if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                                        const double dt = stlq::linkage::LinkageBuildWallNowS() - t0_gpu;
                                        prof->good_gpu_wait_s += dt;
                                        prof->good_gpu_wait_mn_s_by_src[kGoodMnSrcSameFrozen] += dt;
                                        prof->good_gpu_wait_mn_calls_by_src[kGoodMnSrcSameFrozen] += 1;
                                        prof->good_gpu_calls_many_nodes += 1;
                                        UpdateGoodManyNodesBatchQuality(
                                            prof, kGoodMnSrcSameFrozen, /*nodes=*/Bbatch, /*npairs=*/
                                            Npairs_same_frozen);
                                    }
                                }
                            }
                        }

                        // Same-layer dynamic batching (good clusters, stage-3c):
                        // Evaluate "dynamic parents that are already committed before the current block start" as a many-nodes batch,
                        // then commit sequentially within the block for true intra-block dynamic candidates.
                        //
                        // NOTE: This preserves commit-order semantics (same-layer dependencies) while allowing Kp_dyn<=7 to still benefit
                        // from GPU batching when the batch window is large enough.
                        int dyn_block_nodes = std::max(1, std::min(64, std::max(1, Bbatch / 2)));
                        if (runtime_cfg) {
                            dyn_block_nodes = std::max(
                                1, std::min(Bbatch, runtime_cfg->cuda_linkage_same_dynamic_nodes));
                        }
                        bool same_dyn_batch_failed = false;
                        std::vector<int> dyn_before_offsets;
                        std::vector<int> dyn_before_counts;
                        std::vector<int> dyn_before_cand;
                        std::vector<int> dyn_before_pair_node;
                        std::vector<int> dyn_before_best_parent;
                        std::vector<float> dyn_before_best_cost;
                        std::vector<FullCode> dyn_before_best_B;
                        std::vector<float> dyn_before_best_a;
                        bool same_dyn_window_failed = false;
                        std::vector<int> win_inv_offsets;
                        std::vector<int> win_inv_child;
                        std::vector<int> win_inv_parent_local;
                        // Window-local same-layer dynamic candidate lists.
                        // Store per-node lists in a fixed-stride flat buffer to avoid allocator churn:
                        // max per node is <= Bblk (parents within this block), and append order is preserved.
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
                        std::vector<int> win_dyn_flush_pair_t;
                        std::vector<float> win_dyn_flush_X;
                        std::vector<std::uint64_t> win_dyn_flush_node_sample_id_base;
                        std::vector<int> win_dyn_flush_sel_nodes;
                        std::vector<int> win_dyn_flush_sel_take;
                        std::vector<int> win_dyn_flush_best_parent;
                        std::vector<float> win_dyn_flush_best_cost;
                        std::vector<FullCode> win_dyn_flush_best_B;
                        std::vector<float> win_dyn_flush_best_a;
                        std::vector<int> win_dyn_cpu_tail;
                        struct WinDynAsyncState
                        {
                            bool in_flight = false;
                            bool large_root = false;
                            int slice_len = 0;
                            int total_pairs = 0;
                            std::vector<int> sel_nodes;
                            std::vector<int> sel_take;
                        } win_dyn_async;
                        // Stage-3f: when a commit-time forced flush is deemed "tiny", we skip issuing a GPU
                        // many-nodes call and let commit-time CPU tail finish the node. Track which node
                        // triggered the last tiny forced flush so we can label profiling correctly.
                        int win_dyn_tiny_forced_node_off = -1;

                        int block_start = 0;
                        while (block_start < Bbatch) {
                            const int block_end = std::min(Bbatch, block_start + dyn_block_nodes);
                            const int Bblk = block_end - block_start;
                            bool have_dyn_before = false;
                            bool win_dyn_enabled = false;
                            int win_min_pairs = 0;
                            int win_max_pairs = 0;
                            int win_pending_nodes = 0;
                            int win_preflush_target_pairs = 0;
                            int win_preflush_min_pairs = 0;

#if defined(STLQ_ENABLE_CUDA)
                            if (cuda_ctx && runtime_cfg && device_rfull_ready && !same_dyn_batch_failed && block_start >
                                0 &&
                                Bblk > 0) {
                                const int max_pairs = std::max(0, runtime_cfg->cuda_linkage_batch_max_pairs);

                                dyn_before_counts.assign(static_cast<std::size_t>(Bblk), 0);
                                int Npairs_dyn_before = 0;
                                for (int ii = 0; ii < Bblk; ++ii) {
                                    const int i = block_start + ii;
                                    const int same_start = batch_same_offsets[static_cast<std::size_t>(i)];
                                    const int same_stop = batch_same_offsets[static_cast<std::size_t>(i + 1)];
                                    int cnt = 0;
                                    for (int j = same_start; j < same_stop; ++j) {
                                        const int p_local = batch_same_cand[static_cast<std::size_t>(j)];
                                        const int rank = batch_rank_by_local[static_cast<std::size_t>(p_local)];
                                        if (rank < 0 || rank >= block_start) {
                                            continue;
                                        }
                                        const int new_depth = depth_local[static_cast<std::size_t>(p_local)] + 1;
                                        if (new_depth > max_depth) {
                                            continue;
                                        }
                                        ++cnt;
                                    }
                                    dyn_before_counts[static_cast<std::size_t>(ii)] = cnt;
                                    Npairs_dyn_before += cnt;
                                }

                                if (Npairs_dyn_before > 0 && (max_pairs <= 0 || Npairs_dyn_before <= max_pairs)) {
                                    dyn_before_offsets.assign(static_cast<std::size_t>(Bblk) + 1u, 0);
                                    dyn_before_offsets[0] = 0;
                                    for (int ii = 0; ii < Bblk; ++ii) {
                                        dyn_before_offsets[static_cast<std::size_t>(ii + 1)] =
                                            dyn_before_offsets[static_cast<std::size_t>(ii)] +
                                            dyn_before_counts[static_cast<std::size_t>(ii)];
                                    }

                                    dyn_before_cand.resize(static_cast<std::size_t>(Npairs_dyn_before));
                                    dyn_before_pair_node.resize(static_cast<std::size_t>(Npairs_dyn_before));
                                    for (int ii = 0; ii < Bblk; ++ii) {
                                        const int i = block_start + ii;
                                        const int same_start = batch_same_offsets[static_cast<std::size_t>(i)];
                                        const int same_stop = batch_same_offsets[static_cast<std::size_t>(i + 1)];
                                        int pos = dyn_before_offsets[static_cast<std::size_t>(ii)];
                                        for (int j = same_start; j < same_stop; ++j) {
                                            const int p_local = batch_same_cand[static_cast<std::size_t>(j)];
                                            const int rank = batch_rank_by_local[static_cast<std::size_t>(p_local)];
                                            if (rank < 0 || rank >= block_start) {
                                                continue;
                                            }
                                            const int new_depth = depth_local[static_cast<std::size_t>(p_local)] + 1;
                                            if (new_depth > max_depth) {
                                                continue;
                                            }
                                            dyn_before_cand[static_cast<std::size_t>(pos)] = p_local;
                                            dyn_before_pair_node[static_cast<std::size_t>(pos)] = ii;
                                            ++pos;
                                        }
                                    }

                                    dyn_before_best_parent.assign(static_cast<std::size_t>(Bblk), -1);
                                    dyn_before_best_cost.assign(static_cast<std::size_t>(Bblk),
                                                                std::numeric_limits<float>::infinity());
                                    dyn_before_best_B.assign(
                                        static_cast<std::size_t>(Bblk) * static_cast<std::size_t>(m), 0);
                                    dyn_before_best_a.assign(
                                        static_cast<std::size_t>(Bblk) * static_cast<std::size_t>(m), 0.0f);
                                    std::vector<std::uint64_t> dyn_before_node_sample_id_base(
                                        static_cast<std::size_t>(Bblk), 0);
                                    for (int ii = 0; ii < Bblk; ++ii) {
                                        const auto node_local_u32 =
                                            static_cast<std::uint32_t>(batch_node_local[static_cast<std::size_t>(
                                                block_start + ii)]);
                                        dyn_before_node_sample_id_base[static_cast<std::size_t>(ii)] =
                                            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(forced_root_code)) <<
                                                32) |
                                            static_cast<std::uint64_t>(node_local_u32);
                                    }
                                    bool dyn_before_eval_ready = FinishAsyncEvalJob();
                                    if (!dyn_before_eval_ready) {
                                        batch_inner_enabled_cfg = false;
                                        have_batch_same_frozen = false;
                                    }
                                    if (dyn_before_eval_ready && !FlushPendingRfullUpdates()) {
                                        dyn_before_eval_ready = false;
                                        same_dyn_batch_failed = true;
                                        have_dyn_before = false;
                                    }

                                    const auto seed_gpu =
                                        static_cast<std::uint32_t>(linkage_cfg.seed + 997 * forced_root_code);
                                    int ils_iters_gpu = 0;
                                    int perturb_k_gpu = 0;
                                    if constexpr (UseIls) {
                                        ils_iters_gpu = ils_rounds;
                                        perturb_k_gpu = ils_k;
                                    }
                                    std::string batch_err;
                                    bool enqueued = false;
                                    if (dyn_before_eval_ready && async_eval_enabled) {
                                        enqueued =
                                            use_large_root_eval
                                                ? EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExEnqueue(
                                                    cuda_ctx,
                                                    C_one,
                                                    batch_X.data() + static_cast<std::size_t>(block_start) * static_cast
                                                    <std::size_t>(d),
                                                    d,
                                                    Bblk,
                                                    dyn_before_cand.data(),
                                                    dyn_before_pair_node.data(),
                                                    dyn_before_offsets.data(),
                                                    Npairs_dyn_before,
                                                    icm_round,
                                                    ils_iters_gpu,
                                                    perturb_k_gpu,
                                                    seed_gpu,
                                                    dyn_before_node_sample_id_base.data(),
                                                    /*pair_t_override=*/nullptr,
                                                    &batch_err)
                                                : EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseExEnqueue(
                                                    cuda_ctx,
                                                    pre_one,
                                                    batch_X.data() + static_cast<std::size_t>(block_start) * static_cast
                                                    <std::size_t>(d),
                                                    d,
                                                    Bblk,
                                                    dyn_before_cand.data(),
                                                    dyn_before_pair_node.data(),
                                                    dyn_before_offsets.data(),
                                                    Npairs_dyn_before,
                                                    icm_round,
                                                    ils_iters_gpu,
                                                    perturb_k_gpu,
                                                    seed_gpu,
                                                    dyn_before_node_sample_id_base.data(),
                                                    /*pair_t_override=*/nullptr,
                                                    &batch_err);
                                        if (enqueued) {
                                            async_job.in_flight = true;
                                            async_job.large_root = use_large_root_eval;
                                            async_job.src = kGoodMnSrcSameDynBefore;
                                            async_job.nodes = Bblk;
                                            async_job.npairs = Npairs_dyn_before;
                                            async_job.best_parent = dyn_before_best_parent.data();
                                            async_job.best_cost = dyn_before_best_cost.data();
                                            async_job.best_B = dyn_before_best_B.data();
                                            async_job.best_a = dyn_before_best_a.data();
                                            have_dyn_before = true;
                                        }
                                    }
                                    if (dyn_before_eval_ready && !enqueued) {
                                        const double t0_gpu =
                                            stlq::linkage::LinkageBuildProfileEnabled()
                                                ? stlq::linkage::LinkageBuildWallNowS()
                                                : 0.0;
                                        const bool ok_gpu =
                                            use_large_root_eval
                                                ? EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRoot(
                                                    cuda_ctx,
                                                    C_one,
                                                    batch_X.data() + static_cast<std::size_t>(block_start) * static_cast
                                                    <std::size_t>(d),
                                                    d,
                                                    Bblk,
                                                    dyn_before_cand.data(),
                                                    dyn_before_pair_node.data(),
                                                    dyn_before_offsets.data(),
                                                    Npairs_dyn_before,
                                                    icm_round,
                                                    ils_iters_gpu,
                                                    perturb_k_gpu,
                                                    seed_gpu,
                                                    dyn_before_node_sample_id_base.data(),
                                                    dyn_before_best_parent.data(),
                                                    dyn_before_best_cost.data(),
                                                    dyn_before_best_B.data(),
                                                    dyn_before_best_a.data(),
                                                    &batch_err)
                                                : EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBase(
                                                    cuda_ctx,
                                                    pre_one,
                                                    batch_X.data() + static_cast<std::size_t>(block_start) * static_cast
                                                    <std::size_t>(d),
                                                    d,
                                                    Bblk,
                                                    dyn_before_cand.data(),
                                                    dyn_before_pair_node.data(),
                                                    dyn_before_offsets.data(),
                                                    Npairs_dyn_before,
                                                    icm_round,
                                                    ils_iters_gpu,
                                                    perturb_k_gpu,
                                                    seed_gpu,
                                                    dyn_before_node_sample_id_base.data(),
                                                    dyn_before_best_parent.data(),
                                                    dyn_before_best_cost.data(),
                                                    dyn_before_best_B.data(),
                                                    dyn_before_best_a.data(),
                                                    &batch_err);
                                        if (!ok_gpu) {
                                            LogWarn(std::string(
                                                    "CUDA good same-layer dyn-before eval failed; falling back to CPU for the rest of this cluster: ")
                                                +
                                                batch_err);
                                            same_dyn_batch_failed = true;
                                            have_dyn_before = false;
                                        }
                                        else {
                                            have_dyn_before = true;
                                        }
                                        if (stlq::linkage::LinkageBuildProfileEnabled()) {
                                            if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                                                const double dt = stlq::linkage::LinkageBuildWallNowS() - t0_gpu;
                                                prof->good_gpu_wait_s += dt;
                                                prof->good_gpu_wait_mn_s_by_src[kGoodMnSrcSameDynBefore] += dt;
                                                prof->good_gpu_wait_mn_calls_by_src[kGoodMnSrcSameDynBefore] += 1;
                                                prof->good_gpu_calls_many_nodes += 1;
                                                UpdateGoodManyNodesBatchQuality(
                                                    prof, kGoodMnSrcSameDynBefore, /*nodes=*/Bblk, /*npairs=*/
                                                    Npairs_dyn_before);
                                            }
                                        }
                                    }
                                }
                            }
#endif

#if defined(STLQ_ENABLE_CUDA)
                            // NOTE: `runtime.cuda_linkage_same_dynamic_window_batch` is treated as deprecated.
                            // The window-batched same-layer dynamic evaluation is always enabled when the
                            // remaining prerequisites are met. Keep the config key for compatibility.
                            if (cuda_ctx && runtime_cfg && device_rfull_ready && !same_dyn_batch_failed && !
                                same_dyn_window_failed &&
                                runtime_cfg->cuda_linkage_same_dynamic_min_pairs > 0 && Bblk > 1) {
                                win_dyn_enabled = true;
                                win_min_pairs = runtime_cfg->cuda_linkage_same_dynamic_min_pairs;
                                win_max_pairs = runtime_cfg->cuda_linkage_same_dynamic_window_max_pairs;
                                if (win_max_pairs <= 0) {
                                    win_max_pairs = runtime_cfg->cuda_linkage_batch_max_pairs;
                                }
                                // If the user disables both caps (<=0), still enforce a safe per-call bound to avoid
                                // enormous many-nodes launches (OOM risk) while preserving semantics via repeated flushes.
                                if (win_max_pairs <= 0) {
                                    win_max_pairs = 8192;
                                }
                                win_pending_nodes = runtime_cfg->cuda_linkage_same_dynamic_pending_nodes;
                                if (win_pending_nodes <= 0) {
                                    win_pending_nodes = Bblk;
                                }
                                win_pending_nodes = std::max(1, std::min(Bblk, win_pending_nodes));
                                // Stage-3e: opportunistic preflush target for window-local dynamic.
                                // Prefer the dedicated `runtime.cuda_linkage_same_dynamic_preflush_pairs` when provided,
                                // otherwise fall back to the global `runtime.cuda_linkage_batch_min_pairs`.
                                //
                                // IMPORTANT:
                                // - This is an *opportunistic* scheduler knob; it may be smaller than `win_min_pairs`.
                                //   We still guard against extremely tiny preflush GPU calls separately.
                                // - Always clamp to the theoretical block cap (and max-pairs cap) to keep the scheduler live.
                                const int block_pairs_cap = (Bblk > 0) ? (Bblk * Bblk) : 0;
                                if (runtime_cfg->cuda_linkage_same_dynamic_preflush_pairs > 0) {
                                    // Explicit preflush target (enables preflush).
                                    win_preflush_target_pairs =
                                        std::max(1, runtime_cfg->cuda_linkage_same_dynamic_preflush_pairs);
                                }
                                else if (runtime_cfg->cuda_linkage_same_dynamic_preflush_pairs < 0) {
                                    // Legacy behavior: preflush target falls back to global batch min-pairs (or window min-pairs).
                                    if (runtime_cfg->cuda_linkage_batch_min_pairs > 0) {
                                        win_preflush_target_pairs =
                                            std::max(1, runtime_cfg->cuda_linkage_batch_min_pairs);
                                    }
                                    else {
                                        win_preflush_target_pairs = std::max(1, win_min_pairs);
                                    }
                                }
                                else {
                                    // Disabled (unless nodes-trigger is enabled).
                                    win_preflush_target_pairs = 0;
                                }
                                if (win_preflush_target_pairs > 0) {
                                    if (block_pairs_cap > 0) {
                                        win_preflush_target_pairs =
                                            std::min(win_preflush_target_pairs, block_pairs_cap);
                                    }
                                    if (win_max_pairs > 0) {
                                        win_preflush_target_pairs =
                                            std::min(win_preflush_target_pairs, win_max_pairs);
                                    }
                                    win_preflush_target_pairs = std::max(1, win_preflush_target_pairs);
                                    // Stage-3e: opportunistic preflush should not go below the actual same-layer
                                    // window batching floor. Forced tiny tails are handled separately by tiny-CPU.
                                    win_preflush_min_pairs =
                                        std::max(win_min_pairs, win_preflush_target_pairs);
                                }
                                else {
                                    win_preflush_min_pairs = 0;
                                }

                                // Build inverse adjacency within this block:
                                // for each parent rank r in [block_start, block_end), store the (child_off, parent_local)
                                // edges where r < child_rank and the parent is a same-layer candidate of that child.
                                std::vector<int> inv_counts(static_cast<std::size_t>(Bblk), 0);
                                for (int child_off = 0; child_off < Bblk; ++child_off) {
                                    const int i = block_start + child_off;
                                    const int same_start = batch_same_offsets[static_cast<std::size_t>(i)];
                                    const int same_stop = batch_same_offsets[static_cast<std::size_t>(i + 1)];
                                    for (int j = same_start; j < same_stop; ++j) {
                                        const int p_local = batch_same_cand[static_cast<std::size_t>(j)];
                                        const int rank = batch_rank_by_local[static_cast<std::size_t>(p_local)];
                                        if (rank < block_start || rank >= i || rank >= block_end) {
                                            continue;
                                        }
                                        const int parent_off = rank - block_start;
                                        inv_counts[static_cast<std::size_t>(parent_off)] += 1;
                                    }
                                }

                                win_inv_offsets.assign(static_cast<std::size_t>(Bblk) + 1u, 0);
                                win_inv_offsets[0] = 0;
                                for (int r = 0; r < Bblk; ++r) {
                                    win_inv_offsets[static_cast<std::size_t>(r + 1)] =
                                        win_inv_offsets[static_cast<std::size_t>(r)] + inv_counts[static_cast<
                                            std::size_t>(r)];
                                }
                                const int total_edges = win_inv_offsets[static_cast<std::size_t>(Bblk)];
                                win_inv_child.resize(static_cast<std::size_t>(total_edges));
                                win_inv_parent_local.resize(static_cast<std::size_t>(total_edges));
                                std::vector<int> inv_cursor = win_inv_offsets;

                                for (int child_off = 0; child_off < Bblk; ++child_off) {
                                    const int i = block_start + child_off;
                                    const int same_start = batch_same_offsets[static_cast<std::size_t>(i)];
                                    const int same_stop = batch_same_offsets[static_cast<std::size_t>(i + 1)];
                                    for (int j = same_start; j < same_stop; ++j) {
                                        const int p_local = batch_same_cand[static_cast<std::size_t>(j)];
                                        const int rank = batch_rank_by_local[static_cast<std::size_t>(p_local)];
                                        if (rank < block_start || rank >= i || rank >= block_end) {
                                            continue;
                                        }
                                        const int parent_off = rank - block_start;
                                        const int pos = inv_cursor[static_cast<std::size_t>(parent_off)]++;
                                        win_inv_child[static_cast<std::size_t>(pos)] = child_off;
                                        win_inv_parent_local[static_cast<std::size_t>(pos)] = p_local;
                                    }
                                }

                                win_dyn_stride = Bblk;
                                win_dyn_list_flat.resize(static_cast<std::size_t>(Bblk) *
                                    static_cast<std::size_t>(win_dyn_stride));
                                win_dyn_len.assign(static_cast<std::size_t>(Bblk), 0);
                                win_dyn_done.assign(static_cast<std::size_t>(Bblk), 0);
                                win_dyn_best_parent.assign(static_cast<std::size_t>(Bblk), -1);
                                win_dyn_best_cost.assign(static_cast<std::size_t>(Bblk),
                                                         std::numeric_limits<float>::infinity());
                                win_dyn_best_B.assign(static_cast<std::size_t>(Bblk) * static_cast<std::size_t>(m), 0);
                                win_dyn_best_a.assign(static_cast<std::size_t>(Bblk) * static_cast<std::size_t>(m),
                                                      0.0f);
                                win_dyn_cpu_tail.clear();
                            }
#endif

                            const int skip_before_rank = have_dyn_before ? block_start : -1;

                            const auto WinDynFinishAsyncIfAny = [&]() -> bool
                            {
#if defined(STLQ_ENABLE_CUDA)
                                if (!win_dyn_async.in_flight) return true;
                                std::string finish_err;
                                const double t0_wait =
                                    stlq::linkage::LinkageBuildProfileEnabled()
                                        ? stlq::linkage::LinkageBuildWallNowS()
                                        : 0.0;
                                const bool ok =
                                    win_dyn_async.large_root
                                        ? EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExFinish(
                                            cuda_ctx,
                                            win_dyn_flush_best_parent.data(),
                                            win_dyn_flush_best_cost.data(),
                                            win_dyn_flush_best_B.data(),
                                            win_dyn_flush_best_a.data(),
                                            &finish_err)
                                        : EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseExFinish(
                                            cuda_ctx,
                                            win_dyn_flush_best_parent.data(),
                                            win_dyn_flush_best_cost.data(),
                                            win_dyn_flush_best_B.data(),
                                            win_dyn_flush_best_a.data(),
                                            &finish_err);
                                if (!ok) {
                                    LogWarn(std::string(
                                            "CUDA good same-layer window-dynamic async finish failed; disable window batching for this cluster: ")
                                        +
                                        finish_err);
                                    win_dyn_async.in_flight = false;
                                    same_dyn_window_failed = true;
                                    win_dyn_enabled = false;
                                    return false;
                                }
                                if (stlq::linkage::LinkageBuildProfileEnabled()) {
                                    if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                                        const double dt = stlq::linkage::LinkageBuildWallNowS() - t0_wait;
                                        prof->good_gpu_wait_s += dt;
                                        prof->good_gpu_wait_mn_s_by_src[kGoodMnSrcSameDynWindow] += dt;
                                        prof->good_gpu_wait_mn_calls_by_src[kGoodMnSrcSameDynWindow] += 1;
                                        prof->good_gpu_calls_many_nodes += 1;
                                        UpdateGoodManyNodesBatchQuality(prof,
                                                                        kGoodMnSrcSameDynWindow,
                                                                        /*nodes=*/win_dyn_async.slice_len,
                                                                        /*npairs=*/win_dyn_async.total_pairs);
                                        prof->good_dyn_window_preflush_calls += 1;
                                        prof->good_dyn_window_preflush_nodes_sum +=
                                            static_cast<std::uint64_t>(std::max(0, win_dyn_async.slice_len));
                                        prof->good_dyn_window_preflush_pairs_sum +=
                                            static_cast<std::uint64_t>(std::max(0, win_dyn_async.total_pairs));
                                    }
                                }
                                for (int k = 0; k < win_dyn_async.slice_len; ++k) {
                                    const int idx = win_dyn_async.sel_nodes[static_cast<std::size_t>(k)];
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
                                    win_dyn_done[static_cast<std::size_t>(idx)] +=
                                        win_dyn_async.sel_take[static_cast<std::size_t>(k)];
                                }
                                win_dyn_async.in_flight = false;
                                return true;
#else
		                        return true;
#endif
                            };

                            const auto FlushWindowDynamicManyNodes = [&](int start_off, int force_node_off) -> int
                            {
#if defined(STLQ_ENABLE_CUDA)
                                if (!win_dyn_enabled || same_dyn_window_failed) {
                                    return 0;
                                }
                                if (start_off < 0) start_off = 0;
                                if (start_off >= Bblk) {
                                    return 0;
                                }
                                if (force_node_off >= 0) {
                                    force_node_off = std::max(0, std::min(Bblk - 1, force_node_off));
                                    start_off = std::min(start_off, force_node_off);
                                }
                                // Ensure any in-flight opportunistic preflush is applied before issuing another flush.
                                if (win_dyn_async.in_flight) {
                                    if (!WinDynFinishAsyncIfAny()) return 0;
                                }
                                if (!(cuda_ctx && runtime_cfg && device_rfull_ready)) {
                                    same_dyn_window_failed = true;
                                    win_dyn_enabled = false;
                                    return 0;
                                }

                                // Stage-3e: forced flush should not be cut short by the soft cap.
                                const int soft_nodes_cap =
                                    (force_node_off >= 0) ? std::numeric_limits<int>::max() : win_pending_nodes;
                                const int cap_threshold_pairs =
                                    (force_node_off >= 0)
                                        ? std::numeric_limits<int>::max()
                                        : std::max(1, win_preflush_target_pairs);
                                win_dyn_flush_sel_nodes.clear();
                                win_dyn_flush_sel_take.clear();
                                int total_pairs = 0;
                                for (int k = start_off; k < Bblk; ++k) {
                                    const int have = win_dyn_len[static_cast<std::size_t>(k)];
                                    const int done = win_dyn_done[static_cast<std::size_t>(k)];
                                    const int avail = std::max(0, have - done);
                                    if (avail <= 0) {
                                        // Keep scanning; future nodes may have pending work.
                                        continue;
                                    }
                                    int take = avail;
                                    if (win_max_pairs > 0) {
                                        const int rem = win_max_pairs - total_pairs;
                                        if (rem <= 0) break;
                                        take = std::min(take, rem);
                                        if (take <= 0) break;
                                    }
                                    win_dyn_flush_sel_nodes.push_back(k);
                                    win_dyn_flush_sel_take.push_back(take);
                                    total_pairs += take;
                                    const bool hit_soft_cap =
                                    (soft_nodes_cap > 0 &&
                                        static_cast<int>(win_dyn_flush_sel_nodes.size()) >= soft_nodes_cap);
                                    // Stage-3e: only enforce soft-cap once we hit the opportunistic target.
                                    // Otherwise keep accumulating to avoid many tiny many-nodes calls.
                                    const bool enough_for_cap = (total_pairs >= cap_threshold_pairs);
                                    if (enough_for_cap && hit_soft_cap) break;
                                }
                                const int slice_len = static_cast<int>(win_dyn_flush_sel_nodes.size());
                                if (slice_len <= 0 || total_pairs <= 0) {
                                    return 0;
                                }
                                if (total_pairs < win_preflush_min_pairs && force_node_off < 0) {
                                    // Opportunistic preflush: avoid tiny GPU calls.
                                    return 0;
                                }

                                win_dyn_flush_offsets.assign(static_cast<std::size_t>(slice_len) + 1u, 0);
                                win_dyn_flush_cand.resize(static_cast<std::size_t>(total_pairs));
                                win_dyn_flush_pair_node.resize(static_cast<std::size_t>(total_pairs));
                                win_dyn_flush_pair_t.resize(static_cast<std::size_t>(total_pairs));
                                win_dyn_flush_X.resize(static_cast<std::size_t>(slice_len) *
                                    static_cast<std::size_t>(d));
                                win_dyn_flush_node_sample_id_base.assign(static_cast<std::size_t>(slice_len), 0);
                                int pos = 0;
                                win_dyn_flush_offsets[0] = 0;
                                for (int k = 0; k < slice_len; ++k) {
                                    const int kk = win_dyn_flush_sel_nodes[static_cast<std::size_t>(k)];
                                    const int take = win_dyn_flush_sel_take[static_cast<std::size_t>(k)];
                                    const int* lst =
                                        win_dyn_list_flat.data() +
                                        static_cast<std::size_t>(kk) * static_cast<std::size_t>(win_dyn_stride);
                                    const int done = win_dyn_done[static_cast<std::size_t>(kk)];
                                    // Pack X for this node into a compact B-block.
                                    const float* src_x =
                                        batch_X.data() + static_cast<std::size_t>(block_start + kk) *
                                        static_cast<std::size_t>(d);
                                    float* dst_x =
                                        win_dyn_flush_X.data() + static_cast<std::size_t>(k) *
                                        static_cast<std::size_t>(d);
                                    std::memcpy(dst_x, src_x, sizeof(float) * static_cast<std::size_t>(d));
                                    const auto node_local_u32 =
                                        static_cast<std::uint32_t>(batch_node_local[static_cast<std::size_t>(block_start
                                            + kk)]);
                                    win_dyn_flush_node_sample_id_base[static_cast<std::size_t>(k)] =
                                        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(forced_root_code)) << 32)
                                        |
                                        static_cast<std::uint64_t>(node_local_u32);
                                    for (int t = 0; t < take; ++t) {
                                        const int p_local = lst[done + t];
                                        win_dyn_flush_cand[static_cast<std::size_t>(pos)] = p_local;
                                        win_dyn_flush_pair_node[static_cast<std::size_t>(pos)] = k;
                                        win_dyn_flush_pair_t[static_cast<std::size_t>(pos)] = done + t;
                                        ++pos;
                                    }
                                    win_dyn_flush_offsets[static_cast<std::size_t>(k + 1)] = pos;
                                }

                                // Stage-3f(1): tiny CPU / big GPU hybrid for commit-time forced flush.
                                // Forced flush slices can be extremely small (e.g. total_pairs ~ 1..16), where a GPU
                                // many-nodes launch/sync can cost more than the math. When enabled, evaluate such tiny
                                // forced flushes on CPU using the existing per-parent evaluator semantics, and update
                                // win_dyn_best_* / win_dyn_done the same way as the GPU path would.
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
                                    // Tiny forced flush: avoid issuing a GPU many-nodes call.
                                    // Let `WinDynEnsureCompleteForNode(...)` fall back to the existing per-node CPU tail
                                    // evaluation during commit, which preserves commit-order semantics and avoids mixing
                                    // partial window-dynamic state across nodes.
                                    win_dyn_tiny_forced_node_off = force_node_off;
                                    if (stlq::linkage::LinkageBuildProfileEnabled()) {
                                        if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                                            prof->good_dyn_window_tiny_cpu_calls += 1;
                                            prof->good_dyn_window_tiny_cpu_nodes_sum +=
                                                static_cast<std::uint64_t>(slice_len);
                                            prof->good_dyn_window_tiny_cpu_pairs_sum +=
                                                static_cast<std::uint64_t>(std::max(0, total_pairs));
                                            if (total_pairs < 0) {
                                                LogWarn(
                                                    "good_dyn_window_tiny_cpu: total_pairs < 0 (corruption suspected). "
                                                    "total_pairs=" +
                                                    std::to_string(total_pairs) + " slice_len=" +
                                                    std::to_string(slice_len) + " win_max_pairs=" +
                                                    std::to_string(win_max_pairs) + " tiny_cpu_max_pairs=" +
                                                    std::to_string(tiny_cpu_max_pairs));
                                            }
                                        }
                                    }
                                    return 0;
                                }

                                if (!FlushPendingRfullUpdates()) {
                                    same_dyn_window_failed = true;
                                    win_dyn_enabled = false;
                                    return 0;
                                }

                                win_dyn_flush_best_parent.assign(static_cast<std::size_t>(slice_len), -1);
                                win_dyn_flush_best_cost.assign(static_cast<std::size_t>(slice_len),
                                                               std::numeric_limits<float>::infinity());
                                win_dyn_flush_best_B.assign(
                                    static_cast<std::size_t>(slice_len) * static_cast<std::size_t>(m), 0);
                                win_dyn_flush_best_a.assign(
                                    static_cast<std::size_t>(slice_len) * static_cast<std::size_t>(m), 0.0f);

                                const auto seed_gpu =
                                    static_cast<std::uint32_t>(linkage_cfg.seed + 997 * forced_root_code);
                                int ils_iters_gpu = 0;
                                int perturb_k_gpu = 0;
                                if constexpr (UseIls) {
                                    ils_iters_gpu = ils_rounds;
                                    perturb_k_gpu = ils_k;
                                }
                                std::string batch_err;
                                // Opportunistic preflush: enqueue and overlap with CPU work; commit-time flush stays synchronous.
                                if (force_node_off < 0 && async_eval_enabled) {
                                    if (!FinishAsyncEvalJob()) {
                                        same_dyn_window_failed = true;
                                        win_dyn_enabled = false;
                                        return 0;
                                    }
                                    bool enqueued = false;
                                    if (use_large_root_eval) {
                                        enqueued =
                                            EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExEnqueue(
                                                cuda_ctx,
                                                C_one,
                                                win_dyn_flush_X.data(),
                                                d,
                                                slice_len,
                                                win_dyn_flush_cand.data(),
                                                win_dyn_flush_pair_node.data(),
                                                win_dyn_flush_offsets.data(),
                                                total_pairs,
                                                icm_round,
                                                ils_iters_gpu,
                                                perturb_k_gpu,
                                                seed_gpu,
                                                win_dyn_flush_node_sample_id_base.data(),
                                                win_dyn_flush_pair_t.data(),
                                                &batch_err);
                                    }
                                    else {
                                        enqueued =
                                            EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseExEnqueue(
                                                cuda_ctx,
                                                pre_one,
                                                win_dyn_flush_X.data(),
                                                d,
                                                slice_len,
                                                win_dyn_flush_cand.data(),
                                                win_dyn_flush_pair_node.data(),
                                                win_dyn_flush_offsets.data(),
                                                total_pairs,
                                                icm_round,
                                                ils_iters_gpu,
                                                perturb_k_gpu,
                                                seed_gpu,
                                                win_dyn_flush_node_sample_id_base.data(),
                                                win_dyn_flush_pair_t.data(),
                                                &batch_err);
                                    }
                                    if (enqueued) {
                                        win_dyn_async.in_flight = true;
                                        win_dyn_async.large_root = use_large_root_eval;
                                        win_dyn_async.slice_len = slice_len;
                                        win_dyn_async.total_pairs = total_pairs;
                                        win_dyn_async.sel_nodes = win_dyn_flush_sel_nodes;
                                        win_dyn_async.sel_take = win_dyn_flush_sel_take;
                                        return 0;
                                    }
                                }

                                const double t0_gpu =
                                    stlq::linkage::LinkageBuildProfileEnabled()
                                        ? stlq::linkage::LinkageBuildWallNowS()
                                        : 0.0;
                                const bool ok_gpu =
                                    use_large_root_eval
                                        ? EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootEx(
                                            cuda_ctx,
                                            C_one,
                                            win_dyn_flush_X.data(),
                                            d,
                                            slice_len,
                                            win_dyn_flush_cand.data(),
                                            win_dyn_flush_pair_node.data(),
                                            win_dyn_flush_offsets.data(),
                                            total_pairs,
                                            icm_round,
                                            ils_iters_gpu,
                                            perturb_k_gpu,
                                            seed_gpu,
                                            win_dyn_flush_node_sample_id_base.data(),
                                            win_dyn_flush_pair_t.data(),
                                            win_dyn_flush_best_parent.data(),
                                            win_dyn_flush_best_cost.data(),
                                            win_dyn_flush_best_B.data(),
                                            win_dyn_flush_best_a.data(),
                                            &batch_err)
                                        : EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseEx(
                                            cuda_ctx,
                                            pre_one,
                                            win_dyn_flush_X.data(),
                                            d,
                                            slice_len,
                                            win_dyn_flush_cand.data(),
                                            win_dyn_flush_pair_node.data(),
                                            win_dyn_flush_offsets.data(),
                                            total_pairs,
                                            icm_round,
                                            ils_iters_gpu,
                                            perturb_k_gpu,
                                            seed_gpu,
                                            win_dyn_flush_node_sample_id_base.data(),
                                            win_dyn_flush_pair_t.data(),
                                            win_dyn_flush_best_parent.data(),
                                            win_dyn_flush_best_cost.data(),
                                            win_dyn_flush_best_B.data(),
                                            win_dyn_flush_best_a.data(),
                                            &batch_err);
                                if (!ok_gpu) {
                                    LogWarn(std::string(
                                            "CUDA good same-layer window-dynamic eval failed; disable window batching for this cluster: ")
                                        +
                                        batch_err);
                                    same_dyn_window_failed = true;
                                    win_dyn_enabled = false;
                                    return 0;
                                }
                                if (stlq::linkage::LinkageBuildProfileEnabled()) {
                                    if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                                        const double dt = stlq::linkage::LinkageBuildWallNowS() - t0_gpu;
                                        prof->good_gpu_wait_s += dt;
                                        prof->good_gpu_wait_mn_s_by_src[kGoodMnSrcSameDynWindow] += dt;
                                        prof->good_gpu_wait_mn_calls_by_src[kGoodMnSrcSameDynWindow] += 1;
                                        prof->good_gpu_calls_many_nodes += 1;
                                        UpdateGoodManyNodesBatchQuality(prof, kGoodMnSrcSameDynWindow, /*nodes=*/
                                                                        slice_len, /*npairs=*/total_pairs);
                                        if (forced) {
                                            prof->good_dyn_window_forced_calls += 1;
                                            prof->good_dyn_window_forced_nodes_sum +=
                                                static_cast<std::uint64_t>(std::max(0, slice_len));
                                            prof->good_dyn_window_forced_pairs_sum +=
                                                static_cast<std::uint64_t>(std::max(0, total_pairs));
                                        }
                                        else {
                                            prof->good_dyn_window_preflush_calls += 1;
                                            prof->good_dyn_window_preflush_nodes_sum +=
                                                static_cast<std::uint64_t>(std::max(0, slice_len));
                                            prof->good_dyn_window_preflush_pairs_sum +=
                                                static_cast<std::uint64_t>(std::max(0, total_pairs));
                                        }
                                    }
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
#else
	                        (void)start_off;
	                        (void)force_node_off;
	                        return 0;
#endif
                            };

                            // Same-layer dynamic "pending + incremental flush" scheduler:
                            // - After each commit, newly-eligible same-layer edges are appended into the per-node list
                            //   stored in `win_dyn_list_flat`/`win_dyn_len`.
                            // - We opportunistically flush many-nodes batches starting from the earliest uncommitted node
                            //   with pending (have-done) work. This allows small-Kp nodes to be evaluated earlier,
                            //   so by the time a node is committed, most/all of its dynamic candidates are already processed.
                            struct WinDynPending
                            {
                                int pairs = 0;
                                int nodes = 0;
                            };
                            const auto WinDynCountPendingFrom = [&](int start_off, int stop_at_pairs,
                                                                    int stop_at_nodes) -> WinDynPending
                            {
                                WinDynPending out;
                                if (start_off < 0) start_off = 0;
                                if (start_off >= Bblk) return out;
                                const bool count_pairs = (stop_at_pairs > 0);
                                if (count_pairs) stop_at_pairs = std::max(1, stop_at_pairs);
                                stop_at_nodes = std::max(1, stop_at_nodes);
                                for (int k = start_off; k < Bblk; ++k) {
                                    const int have = win_dyn_len[static_cast<std::size_t>(k)];
                                    const int done = win_dyn_done[static_cast<std::size_t>(k)];
                                    if (have > done) {
                                        out.nodes += 1;
                                        if (count_pairs) {
                                            out.pairs += (have - done);
                                            if (out.pairs >= stop_at_pairs) return out; // early stop
                                        }
                                        if (out.nodes >= stop_at_nodes) return out; // early stop
                                    }
                                }
                                return out;
                            };
                            const auto WinDynMaybePreflushFrom = [&](int start_off)
                            {
#if defined(STLQ_ENABLE_CUDA)
                                if (!win_dyn_enabled || same_dyn_window_failed) return;
                                if (win_dyn_async.in_flight) return;
                                // Opportunistic preflush is only meant to overlap GPU eval with other CPU work.
                                // When async pinned staging is disabled, preflush becomes a synchronous extra GPU call
                                // and can easily slow down sparse workloads (adds calls without reducing forced flushes).
                                if (!async_eval_enabled) return;
                                const bool preflush_pairs_enabled = (win_preflush_target_pairs > 0);
                                const bool preflush_nodes_enabled =
                                    (runtime_cfg && runtime_cfg->cuda_linkage_same_dynamic_preflush_nodes > 0);
                                if (!preflush_pairs_enabled && !preflush_nodes_enabled) return;
                                if (start_off < 0) start_off = 0;
                                if (start_off >= Bblk) return;
                                const int target_pairs =
                                    preflush_pairs_enabled ? std::max(1, win_preflush_target_pairs) : 0;
                                // Pair-based triggering can be too strict when same-layer dynamic is extremely sparse
                                // (e.g. ~1 parent per node). In that case we still want to preflush once we have enough
                                // *nodes with pending work* to form a meaningful many-nodes launch.
                                // Nodes-based triggering is opt-in (disabled by default).
                                int trigger_nodes = std::numeric_limits<int>::max();
                                if (preflush_nodes_enabled) {
                                    trigger_nodes =
                                        std::max(1, std::min(win_pending_nodes,
                                                             runtime_cfg->cuda_linkage_same_dynamic_preflush_nodes));
                                }
                                const auto pending = WinDynCountPendingFrom(start_off, target_pairs, trigger_nodes);
                                if (preflush_pairs_enabled && pending.pairs >= target_pairs) {
                                    // ok
                                }
                                else if (preflush_nodes_enabled && pending.nodes >= trigger_nodes) {
                                    // ok
                                }
                                else {
                                    return;
                                }
                                // Opportunistic preflush: single-shot (no draining). Keep remaining pending work to
                                // accumulate into larger batches, and rely on commit-time forced flush to ensure
                                // correctness for a specific node.
                                (void)FlushWindowDynamicManyNodes(start_off, /*force_node_off=*/-1);
#else
			                        (void)start_off;
#endif
                            };
                            const auto WinDynEnsureCompleteForNode = [&](int node_off)
                            {
#if defined(STLQ_ENABLE_CUDA)
                                if (!win_dyn_enabled || same_dyn_window_failed) return;
                                if (node_off < 0 || node_off >= Bblk) return;
                                if (!WinDynFinishAsyncIfAny()) return;
                                // Try to flush starting at this node until either it becomes complete,
                                // or we can't reach min_pairs and must leave the remainder for CPU tail.
                                for (;;) {
                                    const int have = win_dyn_len[static_cast<std::size_t>(node_off)];
                                    const int done = win_dyn_done[static_cast<std::size_t>(node_off)];
                                    if (done >= have) break;
                                    if (FlushWindowDynamicManyNodes(node_off, /*force_node_off=*/node_off) <= 0) break;
                                }
#else
	                        (void)node_off;
#endif
                            };

                            if (!FinishAsyncEvalJob()) {
                                same_dyn_batch_failed = true;
                                have_dyn_before = false;
                                win_dyn_enabled = false;
                            }

                            for (int i = block_start; i < block_end; ++i) {
                                const int local_idx = batch_node_local[static_cast<std::size_t>(i)];
                                const float* xi = batch_X.data() + static_cast<std::size_t>(i) * static_cast<
                                    std::size_t>(d);

                                float bestE2 = base_mse[static_cast<std::size_t>(local_idx)];
                                int best_parent_local = -1;
                                bool did_tiny_dyn_window_forced_skip = false;

                                if (have_batch_inner) {
                                    const float inner_cost = batch_best_cost_inner[static_cast<std::size_t>(i)];
                                    const int inner_parent = batch_best_parent_inner[static_cast<std::size_t>(i)];
                                    if (inner_parent >= 0 && inner_cost < bestE2) {
                                        bestE2 = inner_cost;
                                        best_parent_local = inner_parent;
                                        tls.B_best.assign(static_cast<std::size_t>(m), 0);
                                        tls.a_best.assign(static_cast<std::size_t>(m), 0.0f);
                                        const std::size_t base =
                                            static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
                                        std::memcpy(tls.B_best.data(),
                                                    batch_best_B_inner.data() + base,
                                                    sizeof(FullCode) * static_cast<std::size_t>(m));
                                        std::memcpy(tls.a_best.data(),
                                                    batch_best_a_inner.data() + base,
                                                    sizeof(float) * static_cast<std::size_t>(m));
                                    }
                                }

                                if (have_batch_same_frozen) {
                                    const float frozen_cost = batch_best_cost_same_frozen[static_cast<std::size_t>(i)];
                                    const int frozen_parent = batch_best_parent_same_frozen[static_cast<std::size_t>(
                                        i)];
                                    if (frozen_parent >= 0 && frozen_cost < bestE2) {
                                        bestE2 = frozen_cost;
                                        best_parent_local = frozen_parent;
                                        tls.B_best.assign(static_cast<std::size_t>(m), 0);
                                        tls.a_best.assign(static_cast<std::size_t>(m), 0.0f);
                                        const std::size_t base =
                                            static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
                                        std::memcpy(tls.B_best.data(),
                                                    batch_best_B_same_frozen.data() + base,
                                                    sizeof(FullCode) * static_cast<std::size_t>(m));
                                        std::memcpy(tls.a_best.data(),
                                                    batch_best_a_same_frozen.data() + base,
                                                    sizeof(float) * static_cast<std::size_t>(m));
                                    }
                                }

                                if (have_dyn_before) {
                                    const int bi = i - block_start;
                                    const float dyn_cost = dyn_before_best_cost[static_cast<std::size_t>(bi)];
                                    const int dyn_parent = dyn_before_best_parent[static_cast<std::size_t>(bi)];
                                    if (dyn_parent >= 0 && dyn_cost < bestE2) {
                                        bestE2 = dyn_cost;
                                        best_parent_local = dyn_parent;
                                        tls.B_best.assign(static_cast<std::size_t>(m), 0);
                                        tls.a_best.assign(static_cast<std::size_t>(m), 0.0f);
                                        const std::size_t base =
                                            static_cast<std::size_t>(bi) * static_cast<std::size_t>(m);
                                        std::memcpy(tls.B_best.data(),
                                                    dyn_before_best_B.data() + base,
                                                    sizeof(FullCode) * static_cast<std::size_t>(m));
                                        std::memcpy(tls.a_best.data(),
                                                    dyn_before_best_a.data() + base,
                                                    sizeof(float) * static_cast<std::size_t>(m));
                                    }
                                }

                                // Same-layer candidates are evaluated serially during commit (depth filtering uses updated depths).
                                const int same_start = batch_same_offsets[static_cast<std::size_t>(i)];
                                const int same_stop = batch_same_offsets[static_cast<std::size_t>(i + 1)];

                                tls.candidates.clear();
                                tls.candidates.reserve(static_cast<std::size_t>(std::max(0, same_stop - same_start)));
                                int skipped_before_cnt = 0;
                                for (int j = same_start; j < same_stop; ++j) {
                                    const int p_local = batch_same_cand[static_cast<std::size_t>(j)];
                                    const int rank = batch_rank_by_local[static_cast<std::size_t>(p_local)];
                                    if (rank >= 0) {
                                        if (rank >= i) {
                                            continue;
                                        }
                                        if (skip_before_rank >= 0 && rank < skip_before_rank) {
                                            // Already included in dyn-before many-nodes evaluation for this block.
                                            ++skipped_before_cnt;
                                            continue;
                                        }
                                        if (win_dyn_enabled && rank >= block_start && rank < block_end) {
                                            // Window-local same-layer dynamic is handled by incremental many-nodes flush.
                                            continue;
                                        }
                                        const int new_depth = depth_local[static_cast<std::size_t>(p_local)] + 1;
                                        if (new_depth > max_depth) {
                                            continue;
                                        }
                                        tls.candidates.push_back(p_local);
                                        continue;
                                    }
                                    if (!have_batch_same_frozen) {
                                        const int new_depth = depth_local[static_cast<std::size_t>(p_local)] + 1;
                                        if (new_depth > max_depth) {
                                            continue;
                                        }
                                        tls.candidates.push_back(p_local); // frozen: no frozen-GPU eval, keep on CPU
                                    }
                                }

                                const int ii = i - block_start;
                                const bool node_has_win_dyn =
                                (win_dyn_enabled && ii >= 0 && ii < Bblk &&
                                    win_dyn_len[static_cast<std::size_t>(ii)] > 0);

                                if (stlq::linkage::LinkageBuildProfileEnabled()) {
                                    if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                                        // Report total same-layer dynamic candidate count for this node, including:
                                        // - `skipped_before_cnt`: handled by dyn-before many-nodes
                                        // - `tls.candidates`: remaining serial same-layer candidates outside the window batch
                                        // - window-local pending list length (`win_dyn_len[ii]`): handled by window-dynamic flush
                                        const int win_kp = node_has_win_dyn
                                                               ? win_dyn_len[static_cast<std::size_t>(ii)]
                                                               : 0;
                                        const int Kp_dyn =
                                            static_cast<int>(tls.candidates.size()) + skipped_before_cnt + std::max(
                                                0, win_kp);
                                        const auto bucket = [](int kp) -> int
                                        {
                                            if (kp <= 0) return 0;
                                            if (kp == 1) return 1;
                                            if (kp == 2) return 2;
                                            if (kp == 3) return 3;
                                            if (kp == 4) return 4;
                                            if (kp <= 7) return 5;
                                            if (kp <= 15) return 6;
                                            if (kp <= 31) return 7;
                                            if (kp <= 63) return 8;
                                            if (kp <= 127) return 9;
                                            if (kp <= 255) return 10;
                                            return 11;
                                        };
                                        prof->good_dyn_kp_hist[bucket(Kp_dyn)] += 1;
                                        prof->good_dyn_kp_max = std::max(prof->good_dyn_kp_max, Kp_dyn);
                                    }
                                }

                                // Important: even if `tls.candidates` becomes empty after filtering, we must still
                                // process window-local dynamic candidates (if any). Otherwise we'd silently drop
                                // all same-layer parents that live inside this window.
                                win_dyn_cpu_tail.clear();
                                if (node_has_win_dyn) {
                                    const int have = win_dyn_len[static_cast<std::size_t>(ii)];
                                    const int done = win_dyn_done[static_cast<std::size_t>(ii)];
                                    if (done < have) {
                                        // Ensure we have evaluated all currently-eligible dynamic parents for this node.
                                        win_dyn_tiny_forced_node_off = -1;
                                        WinDynEnsureCompleteForNode(ii);
                                        did_tiny_dyn_window_forced_skip = (win_dyn_tiny_forced_node_off == ii);
                                        win_dyn_tiny_forced_node_off = -1;
                                        const int done2 = win_dyn_done[static_cast<std::size_t>(ii)];
                                        if (done2 < have) {
                                            // Not enough pairs to flush on GPU; evaluate the remaining tail on CPU for this node.
                                            win_dyn_cpu_tail.reserve(static_cast<std::size_t>(have - done2));
                                            const int* lst =
                                                win_dyn_list_flat.data() +
                                                static_cast<std::size_t>(ii) * static_cast<std::size_t>(win_dyn_stride);
                                            for (int t = done2; t < have; ++t) {
                                                win_dyn_cpu_tail.push_back(lst[t]);
                                            }
                                            win_dyn_done[static_cast<std::size_t>(ii)] = have;
                                        }
                                    }
                                    const float dyn_cost = win_dyn_best_cost[static_cast<std::size_t>(ii)];
                                    const int dyn_parent = win_dyn_best_parent[static_cast<std::size_t>(ii)];
                                    if (dyn_parent >= 0 && dyn_cost < bestE2) {
                                        bestE2 = dyn_cost;
                                        best_parent_local = dyn_parent;
                                        tls.B_best.assign(static_cast<std::size_t>(m), 0);
                                        tls.a_best.assign(static_cast<std::size_t>(m), 0.0f);
                                        const std::size_t base =
                                            static_cast<std::size_t>(ii) * static_cast<std::size_t>(m);
                                        std::memcpy(tls.B_best.data(),
                                                    win_dyn_best_B.data() + base,
                                                    sizeof(FullCode) * static_cast<std::size_t>(m));
                                        std::memcpy(tls.a_best.data(),
                                                    win_dyn_best_a.data() + base,
                                                    sizeof(float) * static_cast<std::size_t>(m));
                                    }
                                    if (!win_dyn_cpu_tail.empty()) {
                                        tls.candidates.insert(tls.candidates.end(),
                                                              win_dyn_cpu_tail.begin(),
                                                              win_dyn_cpu_tail.end());
                                    }
                                }

                                if (!tls.candidates.empty()) {
                                    bool used_gpu_dyn_same = false;
                                    bool gpu_attempted_dyn_same = false;
                                    bool gpu_failed_dyn_same = false;

#if defined(STLQ_ENABLE_CUDA)
                                    // Dynamic same-layer candidates are already committed earlier in this batch (rank < i).
                                    // Evaluate them as a single-node GPU batch when the list isn't tiny to avoid per-parent CPU work.
                                    if (!win_dyn_enabled && cuda_ctx && runtime_cfg && device_rfull_ready) {
                                        const int Kp_dyn = static_cast<int>(tls.candidates.size());
                                        int min_kp_cfg = runtime_cfg->cuda_linkage_min_kp;
                                        if (runtime_cfg->cuda_linkage_same_dynamic_min_kp > 0) {
                                            min_kp_cfg = runtime_cfg->cuda_linkage_same_dynamic_min_kp;
                                        }
                                        const int min_kp =
                                            use_large_root_eval ? 1 : std::max(4, std::max(0, min_kp_cfg));
                                        const int max_kp = runtime_cfg->cuda_linkage_max_kp;
                                        if (Kp_dyn >= min_kp && (max_kp <= 0 || Kp_dyn <= max_kp)) {
                                            int ils_iters_gpu = 0;
                                            int perturb_k_gpu = 0;
                                            if constexpr (UseIls) {
                                                ils_iters_gpu = ils_rounds;
                                                perturb_k_gpu = ils_k;
                                            }
                                            const auto seed_gpu =
                                                static_cast<std::uint32_t>(linkage_cfg.seed + 997 * forced_root_code);
                                            const std::uint64_t sample_id_offset =
                                                (static_cast<std::uint64_t>(static_cast<std::uint32_t>(
                                                    forced_root_code)) << 32) |
                                                static_cast<std::uint32_t>(local_idx);
                                            int best_parent_gpu = -1;
                                            float best_cost_gpu = std::numeric_limits<float>::infinity();
                                            std::string gpu_err;
                                            const double t0_gpu =
                                                stlq::linkage::LinkageBuildProfileEnabled()
                                                    ? stlq::linkage::LinkageBuildWallNowS()
                                                    : 0.0;
                                            gpu_attempted_dyn_same = true;
                                            if (!FlushPendingRfullUpdates()) {
                                                gpu_failed_dyn_same = true;
                                            }
                                            bool ok_gpu = false;
                                            if (device_rfull_ready && use_large_root_eval) {
                                                thread_local std::vector<int> pair_node_one;
                                                pair_node_one.assign(static_cast<std::size_t>(Kp_dyn), 0);
                                                const int offsets_one[2] = {0, Kp_dyn};
                                                const std::uint64_t node_sample_id_base = sample_id_offset;
                                                tls.B_try.assign(static_cast<std::size_t>(m), 0);
                                                tls.a_try.assign(static_cast<std::size_t>(m), 0.0f);
                                                ok_gpu =
                                                    EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRoot(
                                                        cuda_ctx,
                                                        C_one,
                                                        xi,
                                                        d,
                                                        /*B=*/1,
                                                        tls.candidates.data(),
                                                        pair_node_one.data(),
                                                        offsets_one,
                                                        Kp_dyn,
                                                        icm_round,
                                                        ils_iters_gpu,
                                                        perturb_k_gpu,
                                                        seed_gpu,
                                                        &node_sample_id_base,
                                                        &best_parent_gpu,
                                                        &best_cost_gpu,
                                                        tls.B_try.data(),
                                                        tls.a_try.data(),
                                                        &gpu_err);
                                            }
                                            else if (device_rfull_ready) {
                                                ok_gpu = EvaluateParentCandidatesBatchCudaDeviceRfull(
                                                    cuda_ctx, pre_one,
                                                    xi, d,
                                                    tls.candidates.data(), Kp_dyn,
                                                    icm_round,
                                                    ils_iters_gpu,
                                                    perturb_k_gpu,
                                                    seed_gpu,
                                                    sample_id_offset,
                                                    &best_parent_gpu, &best_cost_gpu,
                                                    &tls.B_try, &tls.a_try,
                                                    &gpu_err);
                                            }
                                            if (!ok_gpu) {
                                                LogWarn(std::string(
                                                        "GPU same-layer dynamic eval failed; falling back to CPU: ") +
                                                    gpu_err);
                                                gpu_failed_dyn_same = true;
                                            }
                                            else {
                                                used_gpu_dyn_same = true;
                                                if (best_parent_gpu >= 0 && best_cost_gpu < bestE2) {
                                                    bestE2 = best_cost_gpu;
                                                    best_parent_local = best_parent_gpu;
                                                    tls.B_best = tls.B_try;
                                                    tls.a_best = tls.a_try;
                                                }
                                            }
                                            if (stlq::linkage::LinkageBuildProfileEnabled()) {
                                                if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                                                    const double dt = stlq::linkage::LinkageBuildWallNowS() - t0_gpu;
                                                    prof->good_gpu_wait_s += dt;
                                                    prof->good_gpu_wait_batch_s += dt;
                                                    prof->good_gpu_calls_batch += 1;
                                                }
                                            }
                                        }
                                    }
#endif
                                    if (!used_gpu_dyn_same) {
                                        if (stlq::linkage::LinkageBuildProfileEnabled()) {
                                            if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                                                prof->good_cpu_fallback_nodes += 1;
                                                prof->good_cpu_fallback_same_layer_present += 1;
                                                const bool has_cuda = (cuda_ctx && runtime_cfg);
                                                if (!has_cuda) {
                                                    prof->good_cpu_fallback_no_cuda += 1;
                                                }
                                                else if (!device_rfull_ready) {
                                                    prof->good_cpu_fallback_device_rfull_disabled += 1;
                                                }
                                                else {
                                                    if (did_tiny_dyn_window_forced_skip) {
                                                        // Stage-3f: intentionally skipped a tiny forced window flush on GPU.
                                                        prof->good_cpu_fallback_dyn_window_tiny_cpu += 1;
                                                    }
                                                    else {
                                                        const int Kp_dyn = static_cast<int>(tls.candidates.size());
                                                        int min_kp_cfg = runtime_cfg->cuda_linkage_min_kp;
                                                        if (runtime_cfg->cuda_linkage_same_dynamic_min_kp > 0) {
                                                            min_kp_cfg = runtime_cfg->cuda_linkage_same_dynamic_min_kp;
                                                        }
                                                        const int min_kp = std::max(4, std::max(0, min_kp_cfg));
                                                        const int max_kp = (runtime_cfg)
                                                                               ? runtime_cfg->cuda_linkage_max_kp
                                                                               : 0;
                                                        if (Kp_dyn < min_kp) {
                                                            prof->good_cpu_fallback_kp_too_small += 1;
                                                        }
                                                        else if (max_kp > 0 && Kp_dyn > max_kp) {
                                                            prof->good_cpu_fallback_kp_too_large += 1;
                                                        }
                                                        else if (gpu_failed_dyn_same || gpu_attempted_dyn_same) {
                                                            prof->good_cpu_fallback_gpu_error += 1;
                                                        }
                                                        else {
                                                            prof->good_cpu_fallback_other += 1;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        if (use_large_root_eval) {
                                            // Meta-only precomp: CPU eval path is unavailable. Keep the best inner result.
                                        }
                                        else {
                                            const double t0_cpu = stlq::linkage::LinkageBuildProfileEnabled()
                                                                      ? stlq::linkage::LinkageBuildWallNowS()
                                                                      : 0.0;
                                            const bool have_one_root_cross_gram = !G_one_root.data.empty();
                                            if (icm_round > 0 && have_one_root_cross_gram) {
                                                ComputeXcBase(pre_one, xi, &tls.xC_base_buf);
                                            }

                                            const bool do_ils = (m > 1 && ils_rounds > 0 && ils_k > 0);
                                            std::uniform_int_distribution<int> layer_dist(0, std::max(0, m - 1));
                                            std::vector<std::uniform_int_distribution<int>> code_dists;
                                            if constexpr (UseIls) {
                                                if (do_ils) {
                                                    code_dists.reserve(static_cast<std::size_t>(m));
                                                    for (int layer = 0; layer < m; ++layer) {
                                                        const int h = pre_one.h_vec[layer];
                                                        code_dists.emplace_back(0, std::max(0, h - 2));
                                                    }
                                                }
                                            }
                                            for (int p_local : tls.candidates) {
                                                const float* rp = R_full.Col(p_local);
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
                                                    if (have_one_root_cross_gram) {
                                                        BuildXcOneForParentSingleCols(pre_root, pre_one, G_one_root,
                                                            tls.xC_base_buf,
                                                            B_cols.data(), a_cols.data(),
                                                            parent_local, depth_local,
                                                            p_local,
                                                            &tls.xC_buf,
                                                            forced_root_code);
                                                    }
                                                    else {
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
                                                }
                                                else {
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
                                                    if (!SolveLeastSquaresVectorFromResidual(
                                                        pre_one, tls.re, tls.B_init,
                                                        &tls.A, &tls.b, &tls.work,
                                                        tls.a_init.data())) {
                                                        ++ls_failures;
                                                        continue;
                                                    }
                                                }

                                                float E2_best = SampleCostFromXc(
                                                    pre_one, tls.xC_buf, tls.B_init, tls.a_init, re_norm2);

                                                if constexpr (UseIls) {
                                                    if (do_ils) {
                                                        for (int il = 0; il < ils_rounds; ++il) {
                                                            for (int t = 0; t < ils_k; ++t) {
                                                                const int layer = layer_dist(rng);
                                                                const int h = pre_one.h_vec[layer];
                                                                const int old_code =
                                                                    static_cast<int>(tls.B_init[static_cast<std::size_t>
                                                                        (layer)]);
                                                                int new_code = old_code;
                                                                if (h > 1) {
                                                                    int raw = code_dists[static_cast<std::size_t>(
                                                                        layer)](rng);
                                                                    new_code = raw;
                                                                    if (new_code >= old_code) {
                                                                        new_code += 1;
                                                                    }
                                                                }
                                                                const int startf = pre_one.offsets[layer];
                                                                tls.pert_layers[static_cast<std::size_t>(t)] = layer;
                                                                tls.new_flat[static_cast<std::size_t>(t)] = startf +
                                                                    new_code;
                                                            }

                                                            tls.B_try = tls.B_init;
                                                            for (int t = 0; t < ils_k; ++t) {
                                                                const int layer = tls.pert_layers[static_cast<
                                                                    std::size_t>(t)];
                                                                const int flat = tls.new_flat[static_cast<std::size_t>(
                                                                    t)];
                                                                const int code = flat - pre_one.offsets[layer];
                                                                tls.B_try[static_cast<std::size_t>(layer)] = static_cast
                                                                    <Code>(code);
                                                            }

                                                            tls.a_try = tls.a_init;
                                                            if (icm_round > 0) {
                                                                if (!SolveLeastSquaresVector(
                                                                    pre_one, tls.xC_buf, tls.B_try,
                                                                    &tls.A, &tls.b, &tls.work,
                                                                    tls.a_try.data())) {
                                                                    ++ls_failures;
                                                                    continue;
                                                                }
                                                                DynamicIcmEncodingSingleForResidual(
                                                                    pre_one, tls.re, tls.xC_buf,
                                                                    icm_round,
                                                                    &tls.B_try, &tls.a_try,
                                                                    &tls.a_bak, &tls.rC_buf,
                                                                    &tls.A, &tls.b, &tls.work,
                                                                    &rng, &ls_failures);
                                                            }
                                                            else {
                                                                if (!SolveLeastSquaresVectorFromResidual(
                                                                    pre_one, tls.re, tls.B_try,
                                                                    &tls.A, &tls.b, &tls.work,
                                                                    tls.a_try.data())) {
                                                                    ++ls_failures;
                                                                    continue;
                                                                }
                                                            }

                                                            const float E2_try =
                                                                SampleCostFromXc(
                                                                    pre_one, tls.xC_buf, tls.B_try, tls.a_try,
                                                                    re_norm2);

                                                            if (E2_try < E2_best) {
                                                                E2_best = E2_try;
                                                                tls.B_init = tls.B_try;
                                                                tls.a_init = tls.a_try;
                                                            }
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
                                            if (stlq::linkage::LinkageBuildProfileEnabled()) {
                                                if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                                                    const double dt = stlq::linkage::LinkageBuildWallNowS() - t0_cpu;
                                                    prof->good_cpu_fallback_s += dt;
                                                    if (did_tiny_dyn_window_forced_skip) {
                                                        // NOTE: subset of `good_cpu_fallback_s` (do not sum).
                                                        prof->good_dyn_window_tiny_cpu_s += dt;
                                                    }
                                                }
                                            }
                                        }
                                    } // !use_large_root_eval
                                }

                                if (best_parent_local >= 0) {
                                    const double t0_commit = stlq::linkage::LinkageBuildProfileEnabled()
                                                                 ? stlq::linkage::LinkageBuildWallNowS()
                                                                 : 0.0;
                                    for (int l = 0; l < m; ++l) {
                                        (*B)(l, local_idx) = tls.B_best[static_cast<std::size_t>(l)];
                                        (*a)(l, local_idx) = tls.a_best[static_cast<std::size_t>(l)];
                                    }
                                    parent_local[static_cast<std::size_t>(local_idx)] = best_parent_local + 1;
                                    depth_local[static_cast<std::size_t>(local_idx)] =
                                        depth_local[static_cast<std::size_t>(best_parent_local)] + 1;

                                    ReconstructNode(C_one.books, tls.B_best, tls.a_best, &tls.R_self);
                                    const float* rp = R_full.Col(best_parent_local);
                                    float* rdst = R_full.Col(local_idx);
                                    const float* rself_best = tls.R_self.data();
#pragma omp simd
                                    for (int r = 0; r < d; ++r) {
                                        rdst[r] = rp[r] + rself_best[r];
                                    }
                                    if (stlq::linkage::LinkageBuildProfileEnabled()) {
                                        if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                                            prof->good_commit_cpu_s += stlq::linkage::LinkageBuildWallNowS() -
                                                t0_commit;
                                        }
                                    }
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                                    if (cuda_ctx && device_rfull_ready) {
                                        PushPendingRfullUpdate(local_idx, R_full.Col(local_idx));
                                    }
#endif

                                    // After committing this node, it becomes an eligible same-layer parent for later nodes in the block.
                                    if (win_dyn_enabled && !same_dyn_window_failed) {
                                        const int parent_off = i - block_start;
                                        const int beg = win_inv_offsets[static_cast<std::size_t>(parent_off)];
                                        const int end = win_inv_offsets[static_cast<std::size_t>(parent_off + 1)];
                                        const int p_depth = depth_local[static_cast<std::size_t>(local_idx)];
                                        for (int e = beg; e < end; ++e) {
                                            const int child_off = win_inv_child[static_cast<std::size_t>(e)];
                                            const int p_local = win_inv_parent_local[static_cast<std::size_t>(e)];
                                            // Depth filter uses committed parent depth.
                                            if (p_depth + 1 > max_depth) {
                                                continue;
                                            }
                                            const auto c = static_cast<std::size_t>(child_off);
                                            int len = win_dyn_len[c];
                                            if (len < win_dyn_stride) {
                                                win_dyn_list_flat[c * static_cast<std::size_t>(win_dyn_stride) +
                                                    static_cast<std::size_t>(len)] = p_local;
                                                win_dyn_len[c] = len + 1;
                                            }
                                        }
                                        // Opportunistically flush for upcoming nodes after new edges become eligible.
                                        WinDynMaybePreflushFrom(parent_off + 1);
                                    }
                                }
                            }
                            (void)WinDynFinishAsyncIfAny();
                            block_start = block_end;
                        }

                        for (int i = 0; i < Bbatch; ++i) {
                            batch_rank_by_local[static_cast<std::size_t>(batch_node_local[static_cast<std::size_t>(i)])]
                                = -1;
                        }

                        batch_node_local.clear();
                        batch_X.clear();
                        batch_inner_offsets.clear();
                        batch_inner_offsets.push_back(0);
                        batch_inner_cand.clear();
                        batch_pair_node.clear();
                        batch_same_offsets.clear();
                        batch_same_offsets.push_back(0);
                        batch_same_cand.clear();
                        batch_same_frozen_offsets.clear();
                        batch_same_frozen_offsets.push_back(0);
                        batch_same_frozen_cand.clear();
                        batch_same_frozen_pair_node.clear();
                        // Stage-3f: reserve to reduce allocator churn during cand packing.
                        batch_node_local.reserve(static_cast<std::size_t>(std::max(1, batch_nodes_max)));
                        batch_X.reserve(static_cast<std::size_t>(std::max(1, batch_nodes_max)) *
                            static_cast<std::size_t>(d));
                        batch_inner_offsets.reserve(static_cast<std::size_t>(std::max(1, batch_nodes_max)) + 1u);
                        batch_same_offsets.reserve(static_cast<std::size_t>(std::max(1, batch_nodes_max)) + 1u);
                        {
                            const int est_inner_kp = std::max(0, (inner_hi - inner_lo + 1) * knn_k);
                            const int est_same_kp = std::max(0, knn_k);
                            const std::size_t cap_inner =
                                static_cast<std::size_t>(std::max(1, batch_nodes_max)) *
                                static_cast<std::size_t>(est_inner_kp);
                            const std::size_t cap_same =
                                static_cast<std::size_t>(std::max(1, batch_nodes_max)) *
                                static_cast<std::size_t>(est_same_kp);
                            batch_inner_cand.reserve(cap_inner);
                            batch_pair_node.reserve(cap_inner);
                            batch_same_cand.reserve(cap_same);
                        }
                    };

                    for (int pos_in_layer : tls.order_idx) {
                        const double t0_build = stlq::linkage::LinkageBuildProfileEnabled()
                                                    ? stlq::linkage::LinkageBuildWallNowS()
                                                    : 0.0;
                        const int local_idx = current_layer[static_cast<std::size_t>(pos_in_layer)];
                        const float* xi = X.Col(local_idx);

                        tls.candidates_inner.clear();
                        tls.candidates_same.clear();
                        const bool prof_enabled = stlq::linkage::LinkageBuildProfileEnabled();
                        const double t0_query = prof_enabled ? stlq::linkage::LinkageBuildWallNowS() : 0.0;
                        QueryCandidatesHnswSplit(hnsw_index,
                                                 xi,
                                                 cluster_indices,
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
                                                 &tls.candidates_inner,
                                                 &tls.candidates_same);
                        const double dt_query = prof_enabled ? (stlq::linkage::LinkageBuildWallNowS() - t0_query) : 0.0;

                        tls.processed[static_cast<std::size_t>(pos_in_layer)] = 1;
                        if (tls.candidates_inner.empty() && tls.candidates_same.empty()) {
                            continue;
                        }
                        if (prof_enabled) {
                            if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                                prof->good_hnsw_query_cpu_s += dt_query;
                            }
                        }

                        const double t0_pack = prof_enabled ? stlq::linkage::LinkageBuildWallNowS() : 0.0;
                        const int node_in_batch = static_cast<int>(batch_node_local.size());
                        batch_node_local.push_back(local_idx);
                        const std::size_t old_sz = batch_X.size();
                        batch_X.resize(old_sz + static_cast<std::size_t>(d));
                        std::memcpy(batch_X.data() + old_sz, xi, sizeof(float) * static_cast<std::size_t>(d));

                        for (int p_local : tls.candidates_inner) {
                            // inner candidate: depth is already final (inner layers committed before this layer starts)
                            const int new_depth = depth_local[static_cast<std::size_t>(p_local)] + 1;
                            if (new_depth > max_depth) {
                                continue;
                            }
                            batch_inner_cand.push_back(p_local);
                            batch_pair_node.push_back(node_in_batch);
                        }
                        for (int p_local : tls.candidates_same) {
                            batch_same_cand.push_back(p_local);
                        }
                        batch_inner_offsets.push_back(static_cast<int>(batch_inner_cand.size()));
                        batch_same_offsets.push_back(static_cast<int>(batch_same_cand.size()));
                        if (prof_enabled) {
                            if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                                prof->good_cand_pack_cpu_s += stlq::linkage::LinkageBuildWallNowS() - t0_pack;
                            }
                        }

                        if (prof_enabled) {
                            if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                                prof->good_cand_build_cpu_s += stlq::linkage::LinkageBuildWallNowS() - t0_build;
                            }
                        }

                        if (static_cast<int>(batch_node_local.size()) >= batch_nodes_max ||
                            static_cast<int>(batch_inner_cand.size()) >= batch_pairs_max) {
                            FlushBatchAndCommit();
                            if (!(cuda_ctx && runtime_cfg && device_rfull_ready && batch_inner_enabled_cfg)) {
                                break;
                            }
                        }
                    }

                    FlushBatchAndCommit();
                    // IMPORTANT: this batching branch runs per-layer. The device-side R_full is reused by subsequent layers
                    // (which may select parents from nodes committed in earlier layers). Therefore we must flush any remaining
                    // pending updates before leaving the layer, otherwise later GPU evals can observe stale R_full and change
                    // the linkage structure (hurting recall).
                    (void)FlushPendingRfullUpdates();
                    continue;
                }
#endif

                for (int pos_in_layer : tls.order_idx) {
                    const double t0_build = stlq::linkage::LinkageBuildProfileEnabled()
                                                ? stlq::linkage::LinkageBuildWallNowS()
                                                : 0.0;
                    const int local_idx = current_layer[static_cast<std::size_t>(pos_in_layer)];
                    const float* xi = X.Col(local_idx);

                    tls.candidates.clear();
                    QueryCandidatesHnsw(hnsw_index,
                                        xi,
                                        cluster_indices,
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

                    tls.processed[static_cast<std::size_t>(pos_in_layer)] = 1;
                    if (tls.candidates.empty()) {
                        continue;
                    }
                    if (stlq::linkage::LinkageBuildProfileEnabled()) {
                        if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                            prof->good_cand_build_cpu_s += stlq::linkage::LinkageBuildWallNowS() - t0_build;
                        }
                    }

                    float bestE2 = base_mse[static_cast<std::size_t>(local_idx)];
                    int best_parent_local = -1;

                    bool used_gpu = false;
                    bool same_layer_present = false;
                    bool gpu_attempted = false;
                    bool gpu_failed = false;
                    int Kp_filtered = static_cast<int>(tls.candidates.size());
                    int min_kp = 0;
                    int max_kp = 0;
#if defined(STLQ_ENABLE_CUDA)
                    if (cuda_ctx && runtime_cfg) {
                        // Filter depth violations (keep order).
                        int keep = 0;
                        for (int p_local : tls.candidates) {
                            const int new_depth = depth_local[static_cast<std::size_t>(p_local)] + 1;
                            if (new_depth > max_depth) continue;
                            if (stlq::linkage::LinkageBuildProfileEnabled() &&
                                layer_of_point[static_cast<std::size_t>(p_local)] == layer_idx) {
                                same_layer_present = true;
                            }
                            tls.candidates[static_cast<std::size_t>(keep++)] = p_local;
                        }
                        tls.candidates.resize(static_cast<std::size_t>(keep));

                        Kp_filtered = static_cast<int>(tls.candidates.size());
                        min_kp = use_large_root_eval ? 1 : std::max(1, runtime_cfg->cuda_linkage_min_kp);
                        max_kp = runtime_cfg->cuda_linkage_max_kp;
                        const int Kp = Kp_filtered;
                        if (Kp >= min_kp && (max_kp <= 0 || Kp <= max_kp)) {
                            int best_parent = -1;
                            float best_cost = std::numeric_limits<float>::infinity();
                            int ils_iters_gpu = 0;
                            int perturb_k_gpu = 0;
                            if constexpr (UseIls) {
                                ils_iters_gpu = ils_rounds;
                                perturb_k_gpu = ils_k;
                            }
                            const auto seed_gpu =
                                static_cast<std::uint32_t>(linkage_cfg.seed + 997 * forced_root_code);
                            const std::uint64_t sample_id_offset =
                                (static_cast<std::uint64_t>(static_cast<std::uint32_t>(forced_root_code)) << 32) |
                                static_cast<std::uint32_t>(local_idx);
                            std::string gpu_err;
                            if (device_rfull_ready) {
                                const double t0_gpu = stlq::linkage::LinkageBuildProfileEnabled()
                                                          ? stlq::linkage::LinkageBuildWallNowS()
                                                          : 0.0;
                                gpu_attempted = true;
                                bool ok_gpu = false;
                                if (use_large_root_eval) {
                                    thread_local std::vector<int> pair_node_one;
                                    pair_node_one.assign(static_cast<std::size_t>(Kp), 0);
                                    const int offsets_one[2] = {0, Kp};
                                    const std::uint64_t node_sample_id_base = sample_id_offset;
                                    tls.B_best.assign(static_cast<std::size_t>(m), 0);
                                    tls.a_best.assign(static_cast<std::size_t>(m), 0.0f);
                                    ok_gpu =
                                        EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRoot(
                                            cuda_ctx,
                                            C_one,
                                            xi,
                                            d,
                                            /*B=*/1,
                                            tls.candidates.data(),
                                            pair_node_one.data(),
                                            offsets_one,
                                            Kp,
                                            icm_round,
                                            ils_iters_gpu,
                                            perturb_k_gpu,
                                            seed_gpu,
                                            &node_sample_id_base,
                                            &best_parent,
                                            &best_cost,
                                            tls.B_best.data(),
                                            tls.a_best.data(),
                                            &gpu_err);
                                }
                                else {
                                    ok_gpu = EvaluateParentCandidatesBatchCudaDeviceRfull(
                                        cuda_ctx, pre_one,
                                        xi, d,
                                        tls.candidates.data(), Kp,
                                        icm_round,
                                        ils_iters_gpu,
                                        perturb_k_gpu,
                                        seed_gpu,
                                        sample_id_offset,
                                        &best_parent, &best_cost,
                                        &tls.B_best, &tls.a_best,
                                        &gpu_err);
                                }
                                if (!ok_gpu) {
                                    LogWarn(std::string("CUDA device_rfull candidate eval failed: ") + gpu_err);
                                    device_rfull_ready = false;
                                    gpu_failed = true;
                                }
                                else {
                                    used_gpu = true;
                                }
                                if (stlq::linkage::LinkageBuildProfileEnabled()) {
                                    if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                                        const double dt = stlq::linkage::LinkageBuildWallNowS() - t0_gpu;
                                        prof->good_gpu_wait_s += dt;
                                        prof->good_gpu_wait_batch_s += dt;
                                        prof->good_gpu_calls_batch += 1;
                                    }
                                }
                            }
                            if (!used_gpu) {
                                ensure_pinned(Kp);
                                if (pinned_Rp && pinned_cand) {
                                    for (int j = 0; j < Kp; ++j) {
                                        const int p_local = tls.candidates[static_cast<std::size_t>(j)];
                                        pinned_cand[j] = p_local;
                                        const float* rp = R_full.Col(p_local);
                                        std::memcpy(
                                            pinned_Rp + static_cast<std::size_t>(j) * static_cast<std::size_t>(d),
                                            rp, sizeof(float) * static_cast<std::size_t>(d));
                                    }
                                    const double t0_gpu_pinned =
                                        stlq::linkage::LinkageBuildProfileEnabled()
                                            ? stlq::linkage::LinkageBuildWallNowS()
                                            : 0.0;
                                    gpu_attempted = true;
                                    if (!EvaluateParentCandidatesBatchCuda(cuda_ctx, pre_one,
                                                                           xi, d,
                                                                           pinned_Rp, pinned_cand,
                                                                           Kp, icm_round,
                                                                           ils_iters_gpu,
                                                                           perturb_k_gpu,
                                                                           seed_gpu,
                                                                           sample_id_offset,
                                                                           &best_parent, &best_cost,
                                                                           &tls.B_best, &tls.a_best,
                                                                           &gpu_err)) {
                                        LogWarn(
                                            std::string("GPU candidate eval failed; falling back to CPU: ") + gpu_err);
                                        used_gpu = false;
                                        gpu_failed = true;
                                    }
                                    else {
                                        used_gpu = true;
                                    }
                                    if (stlq::linkage::LinkageBuildProfileEnabled()) {
                                        if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                                            const double dt = stlq::linkage::LinkageBuildWallNowS() - t0_gpu_pinned;
                                            prof->good_gpu_wait_s += dt;
                                            prof->good_gpu_wait_single_s += dt;
                                            prof->good_gpu_calls_single += 1;
                                        }
                                    }
                                }
                            }
                            if (used_gpu) {
                                if (best_parent >= 0 && best_cost < bestE2) {
                                    bestE2 = best_cost;
                                    best_parent_local = best_parent;
                                }
                            }
                        }
                    }
#endif

                    if (!used_gpu && !use_large_root_eval) {
                        if (stlq::linkage::LinkageBuildProfileEnabled()) {
                            if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                                prof->good_cpu_fallback_nodes += 1;
                                if (same_layer_present) {
                                    prof->good_cpu_fallback_same_layer_present += 1;
                                }
                                const bool has_cuda = (cuda_ctx && runtime_cfg);
                                if (!has_cuda) {
                                    prof->good_cpu_fallback_no_cuda += 1;
                                }
                                else if (Kp_filtered < min_kp) {
                                    prof->good_cpu_fallback_kp_too_small += 1;
                                }
                                else if (max_kp > 0 && Kp_filtered > max_kp) {
                                    prof->good_cpu_fallback_kp_too_large += 1;
                                }
                                else if (gpu_failed) {
                                    prof->good_cpu_fallback_gpu_error += 1;
                                }
                                else if (!device_rfull_ready) {
                                    prof->good_cpu_fallback_device_rfull_disabled += 1;
                                }
                                else if (gpu_attempted && !used_gpu) {
                                    prof->good_cpu_fallback_gpu_error += 1;
                                }
                                else {
                                    prof->good_cpu_fallback_other += 1;
                                }
                            }
                        }
                        const double t0_cpu = stlq::linkage::LinkageBuildProfileEnabled()
                                                  ? stlq::linkage::LinkageBuildWallNowS()
                                                  : 0.0;
                        const bool have_one_root_cross_gram = !G_one_root.data.empty();
                        if (icm_round > 0 && have_one_root_cross_gram) {
                            ComputeXcBase(pre_one, xi, &tls.xC_base_buf);
                        }

                        const bool do_ils = (m > 1 && ils_rounds > 0 && ils_k > 0);
                        std::uniform_int_distribution<int> layer_dist(0, std::max(0, m - 1));
                        std::vector<std::uniform_int_distribution<int>> code_dists;
                        if constexpr (UseIls) {
                            if (do_ils) {
                                code_dists.reserve(static_cast<std::size_t>(m));
                                for (int layer = 0; layer < m; ++layer) {
                                    const int h = pre_one.h_vec[layer];
                                    code_dists.emplace_back(0, std::max(0, h - 2));
                                }
                            }
                        }

                        for (int p_local : tls.candidates) {
                            const int new_depth = depth_local[static_cast<std::size_t>(p_local)] + 1;
                            if (new_depth > max_depth) {
                                continue;
                            }

                            const float* rp = R_full.Col(p_local);
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
                                if (have_one_root_cross_gram) {
                                    BuildXcOneForParentSingleCols(pre_root, pre_one, G_one_root,
                                                                  tls.xC_base_buf,
                                                                  B_cols.data(), a_cols.data(),
                                                                  parent_local, depth_local,
                                                                  p_local,
                                                                  &tls.xC_buf,
                                                                  forced_root_code);
                                }
                                else {
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
                            }
                            else {
                                tls.xC_buf = tls.rC_buf;
                                for (int l = 0; l < m; ++l) {
                                    const float alpha = tls.a_init[static_cast<std::size_t>(l)];
                                    if (alpha == 0.0f) continue;
                                    const int flat = pre_one.offsets[l] + static_cast<int>(tls.B_init[static_cast<
                                        std::size_t>(l)]);
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
                                if (do_ils) {
                                    for (int il = 0; il < ils_rounds; ++il) {
                                        for (int t = 0; t < ils_k; ++t) {
                                            const int layer = layer_dist(rng);
                                            const int h = pre_one.h_vec[layer];
                                            const int old_code = static_cast<int>(tls.B_init[static_cast<std::size_t>(
                                                layer)]);
                                            int new_code = old_code;
                                            if (h > 1) {
                                                int raw = code_dists[static_cast<std::size_t>(layer)](rng);
                                                new_code = raw;
                                                if (new_code >= old_code) {
                                                    new_code += 1;
                                                }
                                            }
                                            const int startf = pre_one.offsets[layer];
                                            tls.pert_layers[static_cast<std::size_t>(t)] = layer;
                                            tls.new_flat[static_cast<std::size_t>(t)] = startf + new_code;
                                        }

                                        tls.B_try = tls.B_init;
                                        for (int t = 0; t < ils_k; ++t) {
                                            const int layer = tls.pert_layers[static_cast<std::size_t>(t)];
                                            const int flat = tls.new_flat[static_cast<std::size_t>(t)];
                                            const int code = flat - pre_one.offsets[layer];
                                            tls.B_try[static_cast<std::size_t>(layer)] = static_cast<Code>(code);
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
                                        }
                                        else {
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
                            }

                            if (E2_best < bestE2) {
                                bestE2 = E2_best;
                                best_parent_local = p_local;
                                tls.B_best = tls.B_init;
                                tls.a_best = tls.a_init;
                            }
                        }
                        if (stlq::linkage::LinkageBuildProfileEnabled()) {
                            if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                                prof->good_cpu_fallback_s += stlq::linkage::LinkageBuildWallNowS() - t0_cpu;
                            }
                        }
                    }

                    if (best_parent_local >= 0) {
                        const double t0_commit = stlq::linkage::LinkageBuildProfileEnabled()
                                                     ? stlq::linkage::LinkageBuildWallNowS()
                                                     : 0.0;
                        for (int l = 0; l < m; ++l) {
                            (*B)(l, local_idx) = tls.B_best[static_cast<std::size_t>(l)];
                            (*a)(l, local_idx) = tls.a_best[static_cast<std::size_t>(l)];
                        }
                        parent_local[static_cast<std::size_t>(local_idx)] = best_parent_local + 1;
                        depth_local[static_cast<std::size_t>(local_idx)] =
                            depth_local[static_cast<std::size_t>(best_parent_local)] + 1;

                        ReconstructNode(C_one.books, tls.B_best, tls.a_best, &tls.R_self);
                        const float* rp = R_full.Col(best_parent_local);
                        float* rdst = R_full.Col(local_idx);
                        const float* rself_best = tls.R_self.data();
#pragma omp simd
                        for (int r = 0; r < d; ++r) {
                            rdst[r] = rp[r] + rself_best[r];
                        }
                        if (stlq::linkage::LinkageBuildProfileEnabled()) {
                            if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                                prof->good_commit_cpu_s += stlq::linkage::LinkageBuildWallNowS() - t0_commit;
                            }
                        }
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                        if (cuda_ctx && device_rfull_ready) {
                            const double t0_up = stlq::linkage::LinkageBuildProfileEnabled()
                                                     ? stlq::linkage::LinkageBuildWallNowS()
                                                     : 0.0;
                            std::string up_col_err;
                            if (!CudaLinkageUpdateClusterRfullColumn(cuda_ctx, local_idx, R_full.Col(local_idx), d,
                                                                     &up_col_err)) {
                                LogWarn(std::string(
                                    "CudaLinkageUpdateClusterRfullColumn failed, disable device_rfull: ") + up_col_err);
                                device_rfull_ready = false;
                            }
                            if (stlq::linkage::LinkageBuildProfileEnabled()) {
                                if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                                    const double dt = stlq::linkage::LinkageBuildWallNowS() - t0_up;
                                    prof->good_rfull_update_s += dt;
                                    prof->good_rfull_update_calls += 1;
                                    prof->good_rfull_update_batches += 1;
                                    prof->rfull_update_s += dt;
                                    prof->rfull_update_calls += 1;
                                    prof->rfull_update_batches += 1;
                                }
                            }
                        }
#endif
                    }
                }
            }

#if defined(STLQ_ENABLE_CUDA)
            if (pinned_Rp) {
                cudaFreeHost(pinned_Rp);
            }
            if (pinned_cand) {
                cudaFreeHost(pinned_cand);
            }
#endif

            out_cluster->parent_local = std::move(parent_local);
            FinalizeLinkageClusterDepthOrderInPlaceFromDepth(out_cluster, depth_local.data());

            if (linkaged_mse_out || linkaged_mse_sum_out) {
                const MseSumStats mse = MseSumStatsFromReconLocal(X, R_full);
                if (linkaged_mse_out) {
                    MseStats s;
                    if (n > 0 && std::isfinite(mse.min)) {
                        s.mean = static_cast<float>(mse.sum / static_cast<double>(n));
                        s.min = static_cast<float>(mse.min);
                        s.max = static_cast<float>(mse.max);
                    }
                    *linkaged_mse_out = s;
                }
                if (linkaged_mse_sum_out) {
                    *linkaged_mse_sum_out = mse.sum;
                }
            }
            if (ls_failures_out) {
                *ls_failures_out = ls_failures;
            }
        }

        template <bool UseIls>
        void LinkageOneCodebookInnerToOuterOneCluster(const LinkageBuildConfig& linkage_cfg,
                                                      const HnswConfig& hnsw_cfg,
                                                      const RuntimeConfig* runtime_cfg,
#if defined(STLQ_ENABLE_CUDA)
                                                      CudaStreamKernelsPool* cuda_pool,
#endif
                                                      int forced_root_code,
                                                      const ColMajorMatrix<float>& X, // d×n_local
                                                      const CodebookPack& C_root,
                                                      const Precomp& pre_root,
                                                      ColMajorMatrix<FullCode>* B, // m×n_local (updated in-place)
                                                      std::vector<RootCode>* code0_root_inout,
                                                      // n_local (only used when h0_root>256)
                                                      ColMajorMatrix<float>* a, // m×n_local
                                                      LinkageStructure::Cluster* out_cluster,
                                                      int* ls_failures_out,
                                                      MseStats* linkaged_mse_out,
                                                      double* linkaged_mse_sum_out) {
            const int d = X.rows;
            const int n = X.cols;
            const int m = pre_root.m;
            const int H = pre_root.H;
            const bool can_cpu_fallback = !pre_root.G.data.empty();
            const bool need_root_u32 = (!C_root.books.empty() && C_root.books.front().cols > 256);
            [[maybe_unused]] const bool allow_large_root_tiny_cpu =
                need_root_u32 && runtime_cfg && runtime_cfg->cuda_linkage_init_large_root_tiny_cpu;
            if (need_root_u32) {
                if (!code0_root_inout || static_cast<int>(code0_root_inout->size()) != n) {
                    throw std::runtime_error("Init-linkage: code0_root buffer missing/mismatched for h0_root>256.");
                }
                if (can_cpu_fallback) {
                    throw std::runtime_error(
                        "Init-linkage: h0_root>256 requires the large-root CUDA path; CPU fallback is not supported.");
                }
            }

            const int num_layers = std::max(1, linkage_cfg.num_layers);
            const int knn_k = std::max(1, linkage_cfg.knn_k);
            const int depth_k = std::max(1, linkage_cfg.depth_k);
            const int max_depth = std::max(1, linkage_cfg.max_depth);
            const int icm_round = std::max(0, linkage_cfg.icm_round);
            const auto root_percentile = static_cast<float>(linkage_cfg.root_percentile);
            const int ils_rounds = std::max(0, linkage_cfg.ils_rounds);
            const int ils_k = std::min(std::max(0, linkage_cfg.ils_perturb_layers), m);

            if (!out_cluster || !B || !a) {
                return;
            }

            const auto ReconstructNodeU32Root = [&](RootCode code0,
                                                    const std::vector<FullCode>& B_node,
                                                    const std::vector<float>& a_node,
                                                    std::vector<float>* out)
            {
                out->assign(static_cast<std::size_t>(d), 0.0f);
                for (int l = 0; l < m; ++l) {
                    const int code = (l == 0)
                                         ? static_cast<int>(code0)
                                         : static_cast<int>(B_node[static_cast<std::size_t>(l)]);
                    const float coeff = a_node[static_cast<std::size_t>(l)];
                    const float* col = C_root.books[static_cast<std::size_t>(l)].Col(code);
                    float* dst = out->data();
#pragma omp simd
                    for (int r = 0; r < d; ++r) {
                        dst[r] += coeff * col[r];
                    }
                }
            };
            out_cluster->cluster_id = forced_root_code;
            out_cluster->n_real = n;
            out_cluster->n_virtual = 0;
            out_cluster->indices.resize(static_cast<std::size_t>(n));
            std::iota(out_cluster->indices.begin(), out_cluster->indices.end(), 0);
            out_cluster->parent_local.assign(static_cast<std::size_t>(n), 0);
            out_cluster->depth_offsets.clear();

            if (ls_failures_out) {
                *ls_failures_out = 0;
            }
            if (d <= 0 || n <= 0) {
                out_cluster->depth_offsets.assign(2, 0);
                return;
            }

            // Partition cluster into angle-based layers.
            std::vector<float> cos_angles;
            ComputeCosineAnglesOneCluster(X, C_root.books.front().Col(forced_root_code), &cos_angles);
            std::vector<int> cluster_indices(static_cast<std::size_t>(n), 0);
            std::iota(cluster_indices.begin(), cluster_indices.end(), 0);

            std::vector<std::vector<int>> layers_local;
            PartitionClusterLayers(cos_angles, cluster_indices, num_layers, root_percentile, &layers_local);
            std::vector<int> layer_of_point(static_cast<std::size_t>(n), 0);
            for (int layer = 0; layer < num_layers; ++layer) {
                for (int local_idx : layers_local[static_cast<std::size_t>(layer)]) {
                    layer_of_point[static_cast<std::size_t>(local_idx)] = layer;
                }
            }

            // Initial reconstruction from base encoding (C_root).
            ColMajorMatrix<float> R_full;
            if (!need_root_u32) {
                R_full = ReconstructAll(C_root.books, *B, *a);
            }
            else {
                R_full = ColMajorMatrix<float>(d, n);
#pragma omp parallel default(none) shared(R_full, code0_root_inout, B, a, C_root, d, n, m)
                {
#pragma omp for schedule(static)
                    for (int i = 0; i < n; ++i) {
                        float* dst = R_full.Col(i);
                        for (int l = 0; l < m; ++l) {
                            const int code =
                                (l == 0)
                                    ? static_cast<int>((*code0_root_inout)[static_cast<std::size_t>(i)])
                                    : static_cast<int>((*B)(l, i));
                            const float coeff = (*a)(l, i);
                            const float* col = C_root.books[static_cast<std::size_t>(l)].Col(code);
#pragma omp simd
                            for (int r = 0; r < d; ++r) {
                                dst[r] += coeff * col[r];
                            }
                        }
                    }
                }
            }
            std::vector<float> base_mse;
            SquaredErrorsFromRecon(X, R_full, &base_mse);

            LinkageTls tls = MakeLinkageTls(d, m, H);
            std::mt19937 rng(static_cast<std::mt19937::result_type>(linkage_cfg.seed + 997 * forced_root_code));

            std::vector<int> parent_local(static_cast<std::size_t>(n), 0);
            std::vector<int> depth_local(static_cast<std::size_t>(n), 0);

            const int candidate_multiplier = std::max(1, hnsw_cfg.candidate_multiplier_good);
            ClusterHnswIndex hnsw_index;
            const int desired_ef_construction = std::max(1, depth_k * candidate_multiplier * knn_k);
            const int ef_construction = std::max(1, std::min(desired_ef_construction, n));
            {
                const double t0 =
                    stlq::linkage::LinkageBuildProfileEnabled() ? stlq::linkage::LinkageBuildWallNowS() : 0.0;
                hnsw_index.Build(X, cluster_indices, hnsw_cfg, ef_construction);
                if (stlq::linkage::LinkageBuildProfileEnabled()) {
                    if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                        prof->good_hnsw_build_cpu_s += stlq::linkage::LinkageBuildWallNowS() - t0;
                    }
                }
            }

            int ls_failures = 0;

            [[maybe_unused]] CudaCtx* cuda_ctx = nullptr;
            [[maybe_unused]] bool device_rfull_ready = false;

#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
            const bool cuda_linkage_enabled =
                runtime_cfg && runtime_cfg->use_cuda && cuda_pool;
            struct CudaLease
            {
                CudaStreamKernelsPool* pool = nullptr;
                CudaCtx* ctx = nullptr;

                ~CudaLease() {
                    if (pool && ctx) {
                        pool->Release(ctx);
                    }
                }
            } cuda_lease;
            if (cuda_linkage_enabled) {
                cuda_ctx = (!can_cpu_fallback || (runtime_cfg && runtime_cfg->cuda_linkage_wait_for_ctx))
                               ? cuda_pool->Acquire()
                               : cuda_pool->TryAcquire();
                cuda_lease.pool = cuda_pool;
                cuda_lease.ctx = cuda_ctx;
            }
            if (cuda_linkage_enabled && !cuda_ctx && !can_cpu_fallback) {
                throw std::runtime_error(
                    "Init-linkage: CUDA ctx unavailable and CPU fallback disabled (missing full-precomp). "
                    "Set runtime.cuda_linkage_wait_for_ctx=true and/or increase runtime.cuda_pool_size.");
            }

            if (cuda_ctx && runtime_cfg && runtime_cfg->cuda_linkage_use_device_rfull) {
                std::string up_err;
                if (CudaLinkageUploadClusterRfull(cuda_ctx, R_full.data.data(), d, n, &up_err)) {
                    device_rfull_ready = true;
                }
                else {
                    if (!can_cpu_fallback) {
                        throw std::runtime_error(
                            std::string(
                                "CudaLinkageUploadClusterRfull failed (init-linkage) and CPU fallback disabled: ") +
                            up_err);
                    }
                    LogWarn(
                        std::string("CudaLinkageUploadClusterRfull failed (init-linkage), fallback to CPU: ") + up_err);
                    device_rfull_ready = false;
                }
            }
            else if (cuda_ctx && runtime_cfg && !runtime_cfg->cuda_linkage_use_device_rfull && !can_cpu_fallback) {
                throw std::runtime_error(
                    "Init-linkage: runtime.cuda_linkage_use_device_rfull is disabled but CPU fallback is disabled (missing full-precomp).");
            }

            bool batch_inner_enabled_cfg = false;
            int batch_nodes_max = 1;
            int batch_pairs_max = std::numeric_limits<int>::max();
            if (cuda_ctx && runtime_cfg && device_rfull_ready) {
                batch_inner_enabled_cfg = runtime_cfg->cuda_linkage_batch_inner_enabled;
                batch_nodes_max = std::max(1, runtime_cfg->cuda_linkage_batch_nodes);
                batch_pairs_max = (runtime_cfg->cuda_linkage_batch_max_pairs > 0)
                                      ? runtime_cfg->cuda_linkage_batch_max_pairs
                                      : std::numeric_limits<int>::max();
            }

            std::vector<int> inner_best_parent_local;
            std::vector<float> inner_best_cost;
            std::vector<int> inner_best_t;
            std::vector<RootCode> inner_best_code0;
            std::vector<FullCode> inner_best_B;
            std::vector<float> inner_best_a;
            std::vector<std::uint8_t> inner_best_valid;
            if (cuda_ctx && runtime_cfg && device_rfull_ready && batch_inner_enabled_cfg) {
                inner_best_parent_local.assign(static_cast<std::size_t>(n), -1);
                inner_best_cost.assign(static_cast<std::size_t>(n), std::numeric_limits<float>::infinity());
                inner_best_t.assign(static_cast<std::size_t>(n), std::numeric_limits<int>::max());
                if (need_root_u32) {
                    inner_best_code0.assign(static_cast<std::size_t>(n), static_cast<RootCode>(forced_root_code));
                }
                else {
                    inner_best_code0.clear();
                }
                inner_best_B.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(m), 0);
                inner_best_a.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(m), 0.0f);
                inner_best_valid.assign(static_cast<std::size_t>(n), 0u);
            }

            std::vector<int> batch_node_local;
            std::vector<float> batch_X;
            std::vector<int> batch_inner_offsets;
            std::vector<int> batch_inner_cand;
            std::vector<int> batch_inner_t;
            std::vector<int> batch_pair_node;
            std::vector<int> batch_same_offsets;
            std::vector<int> batch_same_cand;
            std::vector<int> batch_same_t;
            std::vector<int> batch_rank_by_local;
            std::vector<int> batch_best_parent_inner;
            std::vector<float> batch_best_cost_inner;
            std::vector<FullCode> batch_best_B_inner;
            std::vector<std::uint32_t> batch_best_B_inner_u32;
            std::vector<float> batch_best_a_inner;
            std::vector<std::uint64_t> batch_node_sample_id_base;

            // Same-layer dynamic window scheduler (batched).
            std::vector<int> win_inv_offsets;
            std::vector<int> win_inv_child;
            std::vector<int> win_inv_parent_local;
            std::vector<int> win_inv_pos;
            std::vector<int> win_dyn_list_parent;
            std::vector<int> win_dyn_list_pos;
            std::vector<int> win_dyn_len;
            std::vector<int> win_dyn_done;
            std::vector<int> win_dyn_best_parent;
            std::vector<float> win_dyn_best_cost;
            std::vector<int> win_dyn_best_pos;
            std::vector<RootCode> win_dyn_best_code0;
            std::vector<FullCode> win_dyn_best_B;
            std::vector<float> win_dyn_best_a;
            std::vector<int> win_dyn_flush_offsets;
            std::vector<int> win_dyn_flush_cand;
            std::vector<int> win_dyn_flush_pos;
            std::vector<int> win_dyn_flush_pair_node;
            std::vector<float> win_dyn_flush_X;
            std::vector<std::uint64_t> win_dyn_flush_node_sample_id_base;
            std::vector<int> win_dyn_flush_sel_nodes;
            std::vector<int> win_dyn_flush_sel_take;
            std::vector<int> win_dyn_flush_best_parent;
            std::vector<float> win_dyn_flush_best_cost;
            std::vector<RootCode> win_dyn_flush_best_code0;
            std::vector<std::uint32_t> win_dyn_flush_best_B_u32;
            std::vector<FullCode> win_dyn_flush_best_B;
            std::vector<float> win_dyn_flush_best_a;

            Precomp pre_fixed_root_tiny_cpu;
            bool pre_fixed_root_tiny_cpu_ready = false;
            auto EnsureFixedRootTinyCpuPrecomp = [&]() -> const Precomp& {
                if (pre_fixed_root_tiny_cpu_ready) {
                    return pre_fixed_root_tiny_cpu;
                }
                if (!allow_large_root_tiny_cpu) {
                    throw std::runtime_error("Init-linkage: fixed-root tiny-CPU precomp requested but disabled.");
                }
                std::string perr;
                if (!BuildPrecompInitLinkageFixedRootTinyCpu(C_root, forced_root_code,
                                                             &pre_fixed_root_tiny_cpu, &perr)) {
                    throw std::runtime_error(
                        perr.empty()
                            ? "Init-linkage: failed BuildPrecompInitLinkageFixedRootTinyCpu."
                            : (std::string("Init-linkage: failed BuildPrecompInitLinkageFixedRootTinyCpu: ") + perr));
                }
                pre_fixed_root_tiny_cpu_ready = true;
                return pre_fixed_root_tiny_cpu;
            };

            // Init-linkage dyn_before: batch-evaluate frozen (prior-block) same-layer candidates
            // in a single GPU many-nodes call, analogous to the iteration-linkage's dyn_before (stage-3c).
            bool have_init_dyn_before = false;
            std::vector<int> init_dyn_before_counts;
            std::vector<int> init_dyn_before_offsets;
            std::vector<int> init_dyn_before_cand;
            std::vector<int> init_dyn_before_pair_node;
            std::vector<int> init_dyn_before_pos;
            std::vector<std::uint64_t> init_dyn_before_sample_id;
            std::vector<int> init_dyn_before_best_parent;
            std::vector<float> init_dyn_before_best_cost;
            std::vector<int> init_dyn_before_best_pos;
            std::vector<FullCode> init_dyn_before_best_B;
            std::vector<std::uint32_t> init_dyn_before_best_B_u32;
            std::vector<float> init_dyn_before_best_a;
#endif

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

                const int inner_lo = std::max(0, layer_idx - depth_k);
                const int inner_hi = layer_idx - 1;

                tls.local_pos.resize(static_cast<std::size_t>(n));
                std::fill(tls.local_pos.begin(), tls.local_pos.end(), -1);
                for (std::size_t pos = 0; pos < current_layer.size(); ++pos) {
                    const int local_idx = current_layer[pos];
                    tls.local_pos[static_cast<std::size_t>(local_idx)] = static_cast<int>(pos);
                }

#if defined(STLQ_ENABLE_CUDA)
                // Map pos_in_layer -> rank in commit order (tls.order_idx).
                static thread_local std::vector<int> layer_rank_by_pos_in_layer;
                layer_rank_by_pos_in_layer.assign(current_layer.size(), -1);
                for (int r = 0; r < static_cast<int>(tls.order_idx.size()); ++r) {
                    const int pos_in_layer = tls.order_idx[static_cast<std::size_t>(r)];
                    if (pos_in_layer >= 0 && static_cast<std::size_t>(pos_in_layer) < layer_rank_by_pos_in_layer.
                        size()) {
                        layer_rank_by_pos_in_layer[static_cast<std::size_t>(pos_in_layer)] = r;
                    }
                }

                int layer_kprime = 0;
                std::vector<int> layer_rel_flat;
                std::vector<int> layer_rel_len;
                bool inner_async_in_flight = false;
                int inner_async_ils_iters_gpu = 0;
                int inner_async_perturb_k_gpu = 0;
                std::uint32_t inner_async_seed_gpu = 0;
                std::vector<int> inner_async_node_local;
                std::vector<int> inner_async_offsets;
                std::vector<int> inner_async_cand;
                std::vector<int> inner_async_t;
                std::vector<int> inner_async_best_parent;
                std::vector<float> inner_async_best_cost;
                std::vector<FullCode> inner_async_best_B;
                std::vector<std::uint32_t> inner_async_best_B_u32;
                std::vector<float> inner_async_best_a;

                auto finish_inner_async = [&]() -> bool
                {
                    if (!inner_async_in_flight) {
                        return true;
                    }
                    const int Bbatch = static_cast<int>(inner_async_node_local.size());
                    if (Bbatch <= 0) {
                        inner_async_in_flight = false;
                        inner_async_node_local.clear();
                        inner_async_offsets.clear();
                        inner_async_cand.clear();
                        inner_async_t.clear();
                        return true;
                    }
                    inner_async_best_parent.resize(static_cast<std::size_t>(Bbatch));
                    inner_async_best_cost.resize(static_cast<std::size_t>(Bbatch));
                    if (need_root_u32) {
                        inner_async_best_B_u32.resize(static_cast<std::size_t>(Bbatch) * static_cast<std::size_t>(m));
                    }
                    else {
                        inner_async_best_B.resize(static_cast<std::size_t>(Bbatch) * static_cast<std::size_t>(m));
                    }
                    inner_async_best_a.resize(static_cast<std::size_t>(Bbatch) * static_cast<std::size_t>(m));
                    std::string gpu_err;
                    const bool ok_finish =
                        need_root_u32
                            ? EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExFinishU32(
                                cuda_ctx,
                                inner_async_best_parent.data(),
                                inner_async_best_cost.data(),
                                inner_async_best_B_u32.data(),
                                inner_async_best_a.data(),
                                &gpu_err)
                            : EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExFinish(
                                cuda_ctx,
                                inner_async_best_parent.data(),
                                inner_async_best_cost.data(),
                                inner_async_best_B.data(),
                                inner_async_best_a.data(),
                                &gpu_err);
                    if (!ok_finish) {
                        if (!can_cpu_fallback) {
                            throw std::runtime_error(
                                std::string("CUDA init-linkage inner eval failed and CPU fallback disabled: ") +
                                gpu_err);
                        }
                        LogWarn(std::string("CUDA init-linkage inner eval failed; retry sync: ") + gpu_err);
                        inner_async_in_flight = false;
                        // Retry the exact same batch synchronously (inputs are still available in the async state).
                        const int Npairs = static_cast<int>(inner_async_cand.size());
                        static thread_local std::vector<float> retry_X;
                        static thread_local std::vector<int> retry_pair_node;
                        static thread_local std::vector<std::uint64_t> retry_node_base;
                        retry_X.resize(static_cast<std::size_t>(Bbatch) * static_cast<std::size_t>(d));
                        retry_pair_node.resize(static_cast<std::size_t>(Npairs));
                        retry_node_base.resize(static_cast<std::size_t>(Bbatch));
                        for (int i = 0; i < Bbatch; ++i) {
                            const int local_idx = inner_async_node_local[static_cast<std::size_t>(i)];
                            const float* xi = X.Col(local_idx);
                            std::memcpy(retry_X.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(d),
                                        xi, sizeof(float) * static_cast<std::size_t>(d));
                            retry_node_base[static_cast<std::size_t>(i)] =
                                (static_cast<std::uint64_t>(static_cast<std::uint32_t>(forced_root_code)) << 32) |
                                static_cast<std::uint64_t>(static_cast<std::uint32_t>(local_idx));
                            const int off0 = inner_async_offsets[static_cast<std::size_t>(i)];
                            const int off1 = inner_async_offsets[static_cast<std::size_t>(i + 1)];
                            for (int p = off0; p < off1; ++p) {
                                retry_pair_node[static_cast<std::size_t>(p)] = i;
                            }
                        }
                        std::string retry_err;
                        const bool ok_retry =
                            need_root_u32
                                ? EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExU32(
                                    cuda_ctx, C_root,
                                    retry_X.data(), d, Bbatch,
                                    inner_async_cand.data(),
                                    retry_pair_node.data(),
                                    inner_async_offsets.data(),
                                    Npairs,
                                    icm_round,
                                    inner_async_ils_iters_gpu,
                                    inner_async_perturb_k_gpu,
                                    inner_async_seed_gpu,
                                    retry_node_base.data(),
                                    inner_async_t.data(),
                                    inner_async_best_parent.data(),
                                    inner_async_best_cost.data(),
                                    inner_async_best_B_u32.data(),
                                    inner_async_best_a.data(),
                                    &retry_err)
                                : EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootEx(
                                    cuda_ctx, C_root,
                                    retry_X.data(), d, Bbatch,
                                    inner_async_cand.data(),
                                    retry_pair_node.data(),
                                    inner_async_offsets.data(),
                                    Npairs,
                                    icm_round,
                                    inner_async_ils_iters_gpu,
                                    inner_async_perturb_k_gpu,
                                    inner_async_seed_gpu,
                                    retry_node_base.data(),
                                    inner_async_t.data(),
                                    inner_async_best_parent.data(),
                                    inner_async_best_cost.data(),
                                    inner_async_best_B.data(),
                                    inner_async_best_a.data(),
                                    &retry_err);
                        if (!ok_retry) {
                            throw std::runtime_error(
                                std::string("CUDA init-linkage inner eval retry failed: ") + retry_err);
                        }
                    }

                    for (int i = 0; i < Bbatch; ++i) {
                        const int local_idx = inner_async_node_local[static_cast<std::size_t>(i)];
                        inner_best_parent_local[static_cast<std::size_t>(local_idx)] =
                            inner_async_best_parent[static_cast<std::size_t>(i)];
                        inner_best_cost[static_cast<std::size_t>(local_idx)] =
                            inner_async_best_cost[static_cast<std::size_t>(i)];
                        // Recover best t for tie-breaking/ILS equivalence.
                        {
                            const int p = inner_async_best_parent[static_cast<std::size_t>(i)];
                            const int off0 = inner_async_offsets[static_cast<std::size_t>(i)];
                            const int off1 = inner_async_offsets[static_cast<std::size_t>(i + 1)];
                            int best_t_i = std::numeric_limits<int>::max();
                            for (int u = off0; u < off1; ++u) {
                                if (inner_async_cand[static_cast<std::size_t>(u)] == p) {
                                    best_t_i = inner_async_t[static_cast<std::size_t>(u)];
                                    break;
                                }
                            }
                            inner_best_t[static_cast<std::size_t>(local_idx)] = best_t_i;
                        }
                        if (!need_root_u32) {
                            std::memcpy(inner_best_B.data() +
                                        static_cast<std::size_t>(local_idx) * static_cast<std::size_t>(m),
                                        inner_async_best_B.data() +
                                        static_cast<std::size_t>(i) * static_cast<std::size_t>(m),
                                        sizeof(FullCode) * static_cast<std::size_t>(m));
                        }
                        else {
                            const std::size_t base_src = static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
                            inner_best_code0[static_cast<std::size_t>(local_idx)] =
                                static_cast<RootCode>(inner_async_best_B_u32[base_src + 0]);
                            const std::size_t base_dst =
                                static_cast<std::size_t>(local_idx) * static_cast<std::size_t>(m);
                            inner_best_B[base_dst + 0] = static_cast<FullCode>(0);
                            for (int l = 1; l < m; ++l) {
                                inner_best_B[base_dst + static_cast<std::size_t>(l)] =
                                    static_cast<FullCode>(inner_async_best_B_u32[base_src + static_cast<std::size_t>(
                                        l)]);
                            }
                        }
                        std::memcpy(inner_best_a.data() +
                                    static_cast<std::size_t>(local_idx) * static_cast<std::size_t>(m),
                                    inner_async_best_a.data() +
                                    static_cast<std::size_t>(i) * static_cast<std::size_t>(m),
                                    sizeof(float) * static_cast<std::size_t>(m));
                        inner_best_valid[static_cast<std::size_t>(local_idx)] = 1u;
                    }

                    inner_async_in_flight = false;
                    inner_async_node_local.clear();
                    inner_async_offsets.clear();
                    inner_async_cand.clear();
                    inner_async_t.clear();
                    return true;
                };

                // Filter a precomputed HNSW search result list into (inner, same-layer) candidates.
                // This mirrors QueryCandidatesHnswSplit's filtering, but avoids re-running the HNSW search.
                const auto filter_split_from_rel = [&](const int* rel_list,
                                                       int rel_count,
                                                       int query_local,
                                                       std::vector<int>* out_inner,
                                                       std::vector<int>* out_same)
                {
                    if (out_inner) out_inner->clear();
                    if (out_same) out_same->clear();
                    if (!rel_list || rel_count <= 0) {
                        return;
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
                            per_layer_budget[static_cast<std::size_t>(lp)] = std::max(1, knn_k);
                        }
                        if (layer_idx >= 0 && layer_idx < num_layers_use) {
                            per_layer_budget[static_cast<std::size_t>(layer_idx)] = std::max(1, knn_k);
                        }
                    }

                    const int depth_span = std::max(0, inner_hi - inner_lo + 1);
                    const int want_total = (depth_span + 1) * std::max(1, knn_k);

                    for (int j = 0; j < rel_count; ++j) {
                        const int c_rel = rel_list[j];
                        if (c_rel < 0 || c_rel >= n) {
                            continue;
                        }
                        if (c_rel == query_local) {
                            continue;
                        }
                        const int p_global = cluster_indices[static_cast<std::size_t>(c_rel)];
                        if (p_global < 0 || p_global >= n) {
                            continue;
                        }
                        const int lp = layer_of_point[static_cast<std::size_t>(p_global)];
                        if (lp < inner_lo || lp > layer_idx) {
                            continue;
                        }
                        if (lp == layer_idx) {
                            const int pos_in_layer = tls.local_pos[static_cast<std::size_t>(c_rel)];
                            if (pos_in_layer < 0 || !tls.processed[static_cast<std::size_t>(pos_in_layer)]) {
                                continue;
                            }
                            if (lp >= 0 && lp < num_layers_use &&
                                per_layer_budget[static_cast<std::size_t>(lp)] > 0) {
                                per_layer_budget[static_cast<std::size_t>(lp)] -= 1;
                                if (out_same) out_same->push_back(c_rel);
                            }
                        }
                        else {
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
                };

                // Split the precomputed HNSW rel list into (inner, same-layer) candidates, while also assigning a
                // per-candidate stable index `t` in the combined filtered list order. This `t` is used to build
                // per-pair sample IDs so ILS perturbation remains identical to the sequential per-node path even
                // when we evaluate subsets of candidates in multiple flushes.
                //
                // NOTE: same-layer legality uses `query_rank` (commit order) rather than `tls.processed`, because
                // we prebuild batches ahead of commit in init-linkage.
                static thread_local std::vector<int> candidates_inner_t;
                static thread_local std::vector<int> candidates_same_t;
                const auto filter_split_from_rel_by_rank_with_t = [&](const int* rel_list,
                                                                      int rel_count,
                                                                      int query_local,
                                                                      int query_rank,
                                                                      std::vector<int>* out_inner,
                                                                      std::vector<int>* out_inner_t,
                                                                      std::vector<int>* out_same,
                                                                      std::vector<int>* out_same_t)
                {
                    if (out_inner) out_inner->clear();
                    if (out_same) out_same->clear();
                    if (out_inner_t) out_inner_t->clear();
                    if (out_same_t) out_same_t->clear();
                    if (!rel_list || rel_count <= 0) return;
                    const bool want_inner = (out_inner != nullptr);
                    const bool want_same = (out_same != nullptr);
                    if (!want_inner && !want_same) return;

                    static thread_local std::vector<int> per_layer_budget;
                    const int num_layers_use = std::max(0, num_layers);
                    if (static_cast<int>(per_layer_budget.size()) < num_layers_use) {
                        per_layer_budget.resize(static_cast<std::size_t>(num_layers_use), 0);
                    }
                    const int knn_k_use = std::max(1, knn_k);
                    const int depth_span = std::max(0, inner_hi - inner_lo + 1);
                    int want_total = 0;
                    if (want_inner && want_same) {
                        want_total = (depth_span + 1) * knn_k_use;
                    }
                    else if (want_inner) {
                        want_total = depth_span * knn_k_use;
                    }
                    else {
                        // want_same only
                        want_total = knn_k_use;
                    }
                    if (want_total <= 0) {
                        return;
                    }

                    if (num_layers_use > 0) {
                        std::fill_n(per_layer_budget.data(), static_cast<std::size_t>(num_layers_use), 0);
                        if (want_inner) {
                            const int lo = std::max(0, inner_lo);
                            const int hi = std::min(num_layers_use - 1, inner_hi);
                            for (int lp = lo; lp <= hi; ++lp) {
                                per_layer_budget[static_cast<std::size_t>(lp)] = knn_k_use;
                            }
                        }
                        if (want_same && layer_idx >= 0 && layer_idx < num_layers_use) {
                            per_layer_budget[static_cast<std::size_t>(layer_idx)] = knn_k_use;
                        }
                    }

                    int t = 0;
                    for (int j = 0; j < rel_count; ++j) {
                        const int c_rel = rel_list[j];
                        if (c_rel < 0 || c_rel >= n) continue;
                        if (c_rel == query_local) continue;
                        const int p_global = cluster_indices[static_cast<std::size_t>(c_rel)];
                        if (p_global < 0 || p_global >= n) continue;
                        const int lp = layer_of_point[static_cast<std::size_t>(p_global)];
                        if (lp < inner_lo || lp > layer_idx) continue;

                        if (lp == layer_idx) {
                            const int pos_in_layer = tls.local_pos[static_cast<std::size_t>(c_rel)];
                            if (pos_in_layer < 0 ||
                                static_cast<std::size_t>(pos_in_layer) >= layer_rank_by_pos_in_layer.size()) {
                                continue;
                            }
                            const int pr = layer_rank_by_pos_in_layer[static_cast<std::size_t>(pos_in_layer)];
                            if (pr < 0 || pr >= query_rank) {
                                continue; // not yet committed at `query_rank`
                            }
                        }

                        if (lp < 0 || lp >= num_layers_use) continue;
                        if (per_layer_budget[static_cast<std::size_t>(lp)] <= 0) continue;
                        per_layer_budget[static_cast<std::size_t>(lp)] -= 1;

                        if (lp == layer_idx) {
                            if (out_same) out_same->push_back(c_rel);
                            if (out_same_t) out_same_t->push_back(t);
                        }
                        else {
                            if (out_inner) out_inner->push_back(c_rel);
                            if (out_inner_t) out_inner_t->push_back(t);
                        }
                        t += 1;

                        const std::size_t got =
                            (out_inner ? out_inner->size() : 0u) + (out_same ? out_same->size() : 0u);
                        if (static_cast<int>(got) >= want_total) break;
                    }
                };

                const bool want_layer_rel =
                (cuda_ctx && runtime_cfg && device_rfull_ready &&
                    ((batch_inner_enabled_cfg && !inner_best_valid.empty()) ||
                        (runtime_cfg->cuda_linkage_same_dynamic_min_pairs > 0 &&
                            runtime_cfg->cuda_linkage_same_dynamic_nodes > 1)));
                if (want_layer_rel) {
                    const int csize = static_cast<int>(cluster_indices.size());
                    const int max_possible = std::max(0, csize - 1);
                    if (max_possible > 0) {
                        const int depth_span = std::max(0, inner_hi - inner_lo + 1);
                        const int want_total = (depth_span + 1) * std::max(1, knn_k);
                        int kprime = std::max(32, want_total * candidate_multiplier);
                        kprime = std::min(kprime, max_possible);
                        kprime = std::max(kprime, 1);
                        layer_kprime = kprime;

                        // Precompute HNSW search results for this layer once, then re-filter cheaply per node.
                        hnsw_index.SetEf(kprime);
                        layer_rel_flat.resize(static_cast<std::size_t>(current_layer.size()) *
                            static_cast<std::size_t>(kprime));
                        layer_rel_len.resize(current_layer.size(), 0);
                        static thread_local std::vector<int> candidates;
                        if (candidates.capacity() < static_cast<std::size_t>(kprime)) {
                            candidates.reserve(static_cast<std::size_t>(kprime));
                        }
                        for (std::size_t pos = 0; pos < current_layer.size(); ++pos) {
                            const int local_idx = current_layer[pos];
                            const float* xi = X.Col(local_idx);
                            candidates.clear();
                            try {
                                hnsw_index.SearchLabelsInto(xi, kprime, &candidates);
                            }
                            catch (const std::exception& e) {
                                LogWarn(std::string("HNSW search failed (init-linkage gpu): ") + e.what());
                                layer_rel_flat.clear();
                                layer_kprime = 0;
                                break;
                            }
                            const std::size_t base = pos * static_cast<std::size_t>(kprime);
                            const int got = static_cast<int>(candidates.size());
                            const int ncopy = std::max(0, std::min(kprime, got));
                            layer_rel_len[pos] = got;
                            for (int j = 0; j < ncopy; ++j) {
                                int lab = candidates[static_cast<std::size_t>(j)];
                                if (lab < 0 || lab >= n) {
                                    lab = local_idx;
                                }
                                layer_rel_flat[base + static_cast<std::size_t>(j)] = lab;
                            }
                        }

                        if (batch_inner_enabled_cfg && !inner_best_valid.empty() && !layer_rel_flat.empty()) {
                            // Batch-evaluate inner candidates (cross-layer only) on GPU.
                            batch_node_local.clear();
                            batch_X.clear();
                            batch_inner_offsets.clear();
                            batch_inner_offsets.push_back(0);
                            batch_inner_cand.clear();
                            batch_inner_t.clear();
                            batch_pair_node.clear();
                            batch_node_sample_id_base.clear();

                            batch_node_local.reserve(static_cast<std::size_t>(std::max(1, batch_nodes_max)));
                            batch_X.reserve(static_cast<std::size_t>(std::max(1, batch_nodes_max)) *
                                static_cast<std::size_t>(d));
                            batch_inner_offsets.reserve(static_cast<std::size_t>(std::max(1, batch_nodes_max)) + 1u);

                            const bool do_ils = (m > 1 && ils_rounds > 0 && ils_k > 0);
                            const int ils_iters_gpu = (UseIls && do_ils) ? ils_rounds : 0;
                            const int perturb_k_gpu = (UseIls && do_ils) ? ils_k : 0;
                            const auto seed_gpu =
                                static_cast<std::uint32_t>(linkage_cfg.seed + 997 * forced_root_code);

                            auto flush_inner_batch = [&](bool allow_async) -> bool
                            {
                                const int Bbatch = static_cast<int>(batch_node_local.size());
                                if (Bbatch <= 0) {
                                    return true;
                                }
                                const int Npairs = static_cast<int>(batch_inner_cand.size());
                                if (Npairs == 0) {
                                    // Nothing to evaluate for this batch (all nodes had zero inner candidates).
                                    // Keep semantics: these nodes will rely on same-layer candidates (if any) or remain unlinkaged.
                                    batch_node_local.clear();
                                    batch_X.clear();
                                    batch_inner_offsets.clear();
                                    batch_inner_offsets.push_back(0);
                                    batch_inner_cand.clear();
                                    batch_inner_t.clear();
                                    batch_pair_node.clear();
                                    batch_node_sample_id_base.clear();
                                    return true;
                                }
                                batch_best_parent_inner.resize(static_cast<std::size_t>(Bbatch));
                                batch_best_cost_inner.resize(static_cast<std::size_t>(Bbatch));
                                if (need_root_u32) {
                                    batch_best_B_inner_u32.resize(
                                        static_cast<std::size_t>(Bbatch) * static_cast<std::size_t>(m));
                                }
                                else {
                                    batch_best_B_inner.resize(
                                        static_cast<std::size_t>(Bbatch) * static_cast<std::size_t>(m));
                                }
                                batch_best_a_inner.resize(
                                    static_cast<std::size_t>(Bbatch) * static_cast<std::size_t>(m));

                                std::string gpu_err;
                                bool ok_gpu = false;
                                if (allow_async && runtime_cfg &&
                                    runtime_cfg->cuda_linkage_eval_async_pinned_mb > 0) {
                                    if (!finish_inner_async()) {
                                        return false;
                                    }
                                    ok_gpu =
                                        need_root_u32
                                            ? EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExEnqueueU32(
                                                cuda_ctx, C_root,
                                                batch_X.data(), d, Bbatch,
                                                batch_inner_cand.data(),
                                                batch_pair_node.data(),
                                                batch_inner_offsets.data(),
                                                Npairs,
                                                icm_round,
                                                ils_iters_gpu,
                                                perturb_k_gpu,
                                                seed_gpu,
                                                batch_node_sample_id_base.data(),
                                                batch_inner_t.data(),
                                                &gpu_err)
                                            : EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExEnqueue(
                                                cuda_ctx, C_root,
                                                batch_X.data(), d, Bbatch,
                                                batch_inner_cand.data(),
                                                batch_pair_node.data(),
                                                batch_inner_offsets.data(),
                                                Npairs,
                                                icm_round,
                                                ils_iters_gpu,
                                                perturb_k_gpu,
                                                seed_gpu,
                                                batch_node_sample_id_base.data(),
                                                batch_inner_t.data(),
                                                &gpu_err);
                                    if (ok_gpu) {
                                        inner_async_in_flight = true;
                                        inner_async_ils_iters_gpu = ils_iters_gpu;
                                        inner_async_perturb_k_gpu = perturb_k_gpu;
                                        inner_async_seed_gpu = seed_gpu;
                                        inner_async_node_local = std::move(batch_node_local);
                                        inner_async_offsets = std::move(batch_inner_offsets);
                                        inner_async_cand = std::move(batch_inner_cand);
                                        inner_async_t = std::move(batch_inner_t);
                                        batch_node_local.clear();
                                        batch_X.clear();
                                        batch_inner_offsets.clear();
                                        batch_inner_offsets.push_back(0);
                                        batch_inner_cand.clear();
                                        batch_inner_t.clear();
                                        batch_pair_node.clear();
                                        batch_node_sample_id_base.clear();
                                        return true;
                                    }
                                }
                                ok_gpu =
                                    need_root_u32
                                        ? EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExU32(
                                            cuda_ctx, C_root,
                                            batch_X.data(), d, Bbatch,
                                            batch_inner_cand.data(),
                                            batch_pair_node.data(),
                                            batch_inner_offsets.data(),
                                            Npairs,
                                            icm_round,
                                            ils_iters_gpu,
                                            perturb_k_gpu,
                                            seed_gpu,
                                            batch_node_sample_id_base.data(),
                                            batch_inner_t.data(),
                                            batch_best_parent_inner.data(),
                                            batch_best_cost_inner.data(),
                                            batch_best_B_inner_u32.data(),
                                            batch_best_a_inner.data(),
                                            &gpu_err)
                                        : EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootEx(
                                            cuda_ctx, C_root,
                                            batch_X.data(), d, Bbatch,
                                            batch_inner_cand.data(),
                                            batch_pair_node.data(),
                                            batch_inner_offsets.data(),
                                            Npairs,
                                            icm_round,
                                            ils_iters_gpu,
                                            perturb_k_gpu,
                                            seed_gpu,
                                            batch_node_sample_id_base.data(),
                                            batch_inner_t.data(),
                                            batch_best_parent_inner.data(),
                                            batch_best_cost_inner.data(),
                                            batch_best_B_inner.data(),
                                            batch_best_a_inner.data(),
                                            &gpu_err);
                                if (!ok_gpu) {
                                    if (!can_cpu_fallback) {
                                        throw std::runtime_error(
                                            std::string(
                                                "CUDA init-linkage inner eval failed and CPU fallback disabled: ") +
                                            gpu_err);
                                    }
                                    LogWarn(
                                        std::string(
                                            "CUDA init-linkage inner eval failed, fallback to CPU: ") + gpu_err);
                                    return false;
                                }

                                for (int i = 0; i < Bbatch; ++i) {
                                    const int local_idx = batch_node_local[static_cast<std::size_t>(i)];
                                    inner_best_parent_local[static_cast<std::size_t>(local_idx)] =
                                        batch_best_parent_inner[static_cast<std::size_t>(i)];
                                    inner_best_cost[static_cast<std::size_t>(local_idx)] =
                                        batch_best_cost_inner[static_cast<std::size_t>(i)];
                                    // Recover best t for tie-breaking/ILS equivalence.
                                    {
                                        const int p = batch_best_parent_inner[static_cast<std::size_t>(i)];
                                        const int off0 = batch_inner_offsets[static_cast<std::size_t>(i)];
                                        const int off1 = batch_inner_offsets[static_cast<std::size_t>(i + 1)];
                                        int best_t_i = std::numeric_limits<int>::max();
                                        for (int u = off0; u < off1; ++u) {
                                            if (batch_inner_cand[static_cast<std::size_t>(u)] == p) {
                                                best_t_i = batch_inner_t[static_cast<std::size_t>(u)];
                                                break;
                                            }
                                        }
                                        inner_best_t[static_cast<std::size_t>(local_idx)] = best_t_i;
                                    }
                                    if (!need_root_u32) {
                                        std::memcpy(inner_best_B.data() +
                                                    static_cast<std::size_t>(local_idx) * static_cast<std::size_t>(m),
                                                    batch_best_B_inner.data() +
                                                    static_cast<std::size_t>(i) * static_cast<std::size_t>(m),
                                                    sizeof(FullCode) * static_cast<std::size_t>(m));
                                    }
                                    else {
                                        const std::size_t base_src =
                                            static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
                                        inner_best_code0[static_cast<std::size_t>(local_idx)] =
                                            static_cast<RootCode>(batch_best_B_inner_u32[base_src + 0]);
                                        const std::size_t base_dst =
                                            static_cast<std::size_t>(local_idx) * static_cast<std::size_t>(m);
                                        inner_best_B[base_dst + 0] = static_cast<FullCode>(0);
                                        for (int l = 1; l < m; ++l) {
                                            inner_best_B[base_dst + static_cast<std::size_t>(l)] =
                                                static_cast<FullCode>(batch_best_B_inner_u32[base_src + static_cast<
                                                    std::size_t>(l)]);
                                        }
                                    }
                                    std::memcpy(inner_best_a.data() +
                                                static_cast<std::size_t>(local_idx) * static_cast<std::size_t>(m),
                                                batch_best_a_inner.data() +
                                                static_cast<std::size_t>(i) * static_cast<std::size_t>(m),
                                                sizeof(float) * static_cast<std::size_t>(m));
                                    inner_best_valid[static_cast<std::size_t>(local_idx)] = 1u;
                                }

                                batch_node_local.clear();
                                batch_X.clear();
                                batch_inner_offsets.clear();
                                batch_inner_offsets.push_back(0);
                                batch_inner_cand.clear();
                                batch_inner_t.clear();
                                batch_pair_node.clear();
                                batch_node_sample_id_base.clear();
                                return true;
                            };

                            tls.candidates_inner.clear();
                            tls.candidates_same.clear();
                            for (std::size_t pos = 0; pos < current_layer.size(); ++pos) {
                                const int local_idx = current_layer[pos];
                                const float* xi = X.Col(local_idx);

                                const int* rel =
                                    layer_rel_flat.data() + pos * static_cast<std::size_t>(kprime);
                                const int rel_count =
                                    (pos < layer_rel_len.size()) ? layer_rel_len[pos] : 0;
                                const int query_rank =
                                    (pos < layer_rank_by_pos_in_layer.size())
                                        ? layer_rank_by_pos_in_layer[pos]
                                        : -1;
                                candidates_inner_t.clear();
                                filter_split_from_rel_by_rank_with_t(rel,
                                                                     rel_count,
                                                                     local_idx,
                                                                     std::max(0, query_rank),
                                                                     &tls.candidates_inner,
                                                                     &candidates_inner_t,
                                                                     /*out_same=*/nullptr,
                                                                     /*out_same_t=*/nullptr);

                                // Apply depth filter (inner candidates only).
                                int inner_keep = 0;
                                for (std::size_t u = 0; u < tls.candidates_inner.size(); ++u) {
                                    const int p_local = tls.candidates_inner[u];
                                    const int new_depth =
                                        depth_local[static_cast<std::size_t>(p_local)] + 1;
                                    if (new_depth > max_depth) {
                                        continue;
                                    }
                                    tls.candidates_inner[static_cast<std::size_t>(inner_keep)] = p_local;
                                    candidates_inner_t[static_cast<std::size_t>(inner_keep)] =
                                        candidates_inner_t[u];
                                    inner_keep += 1;
                                }
                                tls.candidates_inner.resize(static_cast<std::size_t>(inner_keep));
                                candidates_inner_t.resize(static_cast<std::size_t>(inner_keep));

                                const int new_pairs = static_cast<int>(tls.candidates_inner.size());
                                const bool need_flush =
                                    (!batch_node_local.empty()) &&
                                    (static_cast<int>(batch_node_local.size()) >= batch_nodes_max ||
                                        static_cast<int>(batch_inner_cand.size()) + new_pairs > batch_pairs_max);
                                if (need_flush) {
                                    if (!flush_inner_batch(/*allow_async=*/false)) {
                                        layer_rel_flat.clear();
                                        layer_kprime = 0;
                                        break;
                                    }
                                }

                                const int node_idx_in_batch = static_cast<int>(batch_node_local.size());
                                batch_node_local.push_back(local_idx);
                                batch_node_sample_id_base.push_back(
                                    (static_cast<std::uint64_t>(static_cast<std::uint32_t>(forced_root_code)) << 32) |
                                    static_cast<std::uint64_t>(static_cast<std::uint32_t>(local_idx)));
                                batch_X.insert(batch_X.end(), xi, xi + d);

                                for (int j = 0; j < inner_keep; ++j) {
                                    const int p_local = tls.candidates_inner[static_cast<std::size_t>(j)];
                                    batch_pair_node.push_back(node_idx_in_batch);
                                    batch_inner_cand.push_back(p_local);
                                    batch_inner_t.push_back(candidates_inner_t[static_cast<std::size_t>(j)]);
                                }
                                batch_inner_offsets.push_back(static_cast<int>(batch_inner_cand.size()));
                            }

                            if (layer_kprime > 0 && !layer_rel_flat.empty()) {
                                if (!flush_inner_batch(/*allow_async=*/true)) {
                                    layer_rel_flat.clear();
                                    layer_kprime = 0;
                                }
                            }
                        }
                    }
                }
#endif

#if defined(STLQ_ENABLE_CUDA)
                const bool init_win_dyn_enabled =
                (cuda_ctx && runtime_cfg && device_rfull_ready &&
                    runtime_cfg->cuda_linkage_same_dynamic_min_pairs > 0 &&
                    runtime_cfg->cuda_linkage_same_dynamic_nodes > 1 &&
                    layer_kprime > 0 && !layer_rel_flat.empty());
                if (init_win_dyn_enabled) {
                    const bool do_ils = (m > 1 && ils_rounds > 0 && ils_k > 0);
                    const int ils_iters_gpu = (UseIls && do_ils) ? ils_rounds : 0;
                    const int perturb_k_gpu = (UseIls && do_ils) ? ils_k : 0;
                    const auto seed_gpu =
                        static_cast<std::uint32_t>(linkage_cfg.seed + 997 * forced_root_code);

                    const int dyn_block_nodes =
                        std::max(2, std::min(static_cast<int>(tls.order_idx.size()),
                                             std::max(2, runtime_cfg->cuda_linkage_same_dynamic_nodes)));
                    const int win_min_pairs = std::max(1, runtime_cfg->cuda_linkage_same_dynamic_min_pairs);
                    const int win_max_pairs =
                        (runtime_cfg->cuda_linkage_same_dynamic_window_max_pairs > 0)
                            ? runtime_cfg->cuda_linkage_same_dynamic_window_max_pairs
                            : std::numeric_limits<int>::max();
                    int win_pending_nodes = runtime_cfg->cuda_linkage_same_dynamic_pending_nodes;
                    if (win_pending_nodes <= 0) {
                        win_pending_nodes = dyn_block_nodes;
                    }
                    win_pending_nodes = std::max(1, std::min(dyn_block_nodes, win_pending_nodes));

                    // Batched device R_full updates: in init-linkage we commit nodes sequentially, but we don't need to
                    // push every committed column to device immediately. We only need device R_full to be current
                    // before enqueuing/launching any GPU candidate-eval that may read newly-committed parents, and
                    // before moving to the next layer (inner candidates may reference the previous layer).
                    std::vector<int> pending_rfull_cols;
                    std::vector<float> pending_rfull_cols_data; // packed columns, size = pending_rfull_cols.size() * d
                    pending_rfull_cols.clear();
                    pending_rfull_cols_data.clear();
                    pending_rfull_cols.reserve(256);
                    pending_rfull_cols_data.reserve(static_cast<std::size_t>(256) * static_cast<std::size_t>(d));

                    const auto PushPendingRfullUpdate = [&](int col, const float* r_col)
                    {
                        pending_rfull_cols.push_back(col);
                        const std::size_t old = pending_rfull_cols_data.size();
                        pending_rfull_cols_data.resize(old + static_cast<std::size_t>(d));
                        std::memcpy(pending_rfull_cols_data.data() + old,
                                    r_col,
                                    sizeof(float) * static_cast<std::size_t>(d));
                    };
                    const auto FlushPendingRfullUpdates = [&]() -> bool
                    {
                        if (pending_rfull_cols.empty()) return true;
                        if (!cuda_ctx || !runtime_cfg || !device_rfull_ready) {
                            pending_rfull_cols.clear();
                            pending_rfull_cols_data.clear();
                            return false;
                        }
                        const double t0_up =
                            stlq::linkage::LinkageBuildProfileEnabled() ? stlq::linkage::LinkageBuildWallNowS() : 0.0;
                        constexpr int kMaxColsPerFlush = 2048;
                        std::size_t off = 0;
                        std::uint64_t n_batches = 0;
                        while (off < pending_rfull_cols.size()) {
                            const int take =
                                static_cast<int>(std::min<std::size_t>(pending_rfull_cols.size() - off,
                                                                       static_cast<std::size_t>(kMaxColsPerFlush)));
                            std::string up_err;
                            const int* cols = pending_rfull_cols.data() + off;
                            const float* data =
                                pending_rfull_cols_data.data() + off * static_cast<std::size_t>(d);
                            if (!CudaLinkageUpdateClusterRfullColumnsBatch(cuda_ctx, cols, data, d, take, &up_err)) {
                                if (!can_cpu_fallback) {
                                    throw std::runtime_error(
                                        std::string(
                                            "CudaLinkageUpdateClusterRfullColumnsBatch failed (init-linkage) and CPU fallback disabled: ")
                                        +
                                        up_err);
                                }
                                LogWarn(std::string(
                                        "CudaLinkageUpdateClusterRfullColumnsBatch failed (init-linkage), disable device_rfull: ")
                                    +
                                    up_err);
                                device_rfull_ready = false;
                                pending_rfull_cols.clear();
                                pending_rfull_cols_data.clear();
                                return false;
                            }
                            off += static_cast<std::size_t>(take);
                            n_batches += 1;
                        }
                        if (stlq::linkage::LinkageBuildProfileEnabled()) {
                            if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                                const double dt = stlq::linkage::LinkageBuildWallNowS() - t0_up;
                                const auto cols_n =
                                    static_cast<std::uint64_t>(pending_rfull_cols.size());
                                prof->rfull_update_s += dt;
                                prof->rfull_update_calls += cols_n;
                                prof->rfull_update_batches += n_batches;
                            }
                        }
                        pending_rfull_cols.clear();
                        pending_rfull_cols_data.clear();
                        return true;
                    };

                    if (static_cast<int>(batch_rank_by_local.size()) < n) {
                        batch_rank_by_local.assign(static_cast<std::size_t>(n), -1);
                    }

                    auto reset_blk = [&](int Bblk)
                    {
                        batch_node_local.clear();
                        batch_X.clear();
                        batch_same_offsets.clear();
                        batch_same_offsets.push_back(0);
                        batch_same_cand.clear();
                        batch_same_t.clear();
                        batch_node_sample_id_base.clear();

                        batch_node_local.reserve(static_cast<std::size_t>(Bblk));
                        batch_X.reserve(static_cast<std::size_t>(Bblk) * static_cast<std::size_t>(d));
                        batch_same_offsets.reserve(static_cast<std::size_t>(Bblk) + 1u);
                        batch_node_sample_id_base.reserve(static_cast<std::size_t>(Bblk));
                        batch_same_cand.reserve(static_cast<std::size_t>(Bblk) *
                            static_cast<std::size_t>(std::max(1, knn_k)));
                        batch_same_t.reserve(static_cast<std::size_t>(Bblk) *
                            static_cast<std::size_t>(std::max(1, knn_k)));
                    };

                    int block_start = 0;
                    while (block_start < static_cast<int>(tls.order_idx.size())) {
                        const int block_end =
                            std::min(static_cast<int>(tls.order_idx.size()), block_start + dyn_block_nodes);
                        const int Bblk = block_end - block_start;
                        if (Bblk <= 0) break;
                        reset_blk(Bblk);

                        // Assign ranks for nodes in this block (local_idx -> [0,Bblk)).
                        for (int i = block_start; i < block_end; ++i) {
                            const int pos_in_layer = tls.order_idx[static_cast<std::size_t>(i)];
                            const int local_idx = current_layer[static_cast<std::size_t>(pos_in_layer)];
                            batch_rank_by_local[static_cast<std::size_t>(local_idx)] = i - block_start;
                        }

                        // Prebuild same-layer candidate lists (in commit order).
                        for (int i = block_start; i < block_end; ++i) {
                            const int pos_in_layer = tls.order_idx[static_cast<std::size_t>(i)];
                            const int local_idx = current_layer[static_cast<std::size_t>(pos_in_layer)];
                            const float* xi = X.Col(local_idx);

                            tls.candidates_same.clear();
                            candidates_same_t.clear();
                            const std::size_t base =
                                static_cast<std::size_t>(pos_in_layer) * static_cast<std::size_t>(layer_kprime);
                            const int rel_len =
                                (static_cast<std::size_t>(pos_in_layer) < layer_rel_len.size())
                                    ? layer_rel_len[static_cast<std::size_t>(pos_in_layer)]
                                    : 0;
                            const int rel_count = std::max(0, std::min(layer_kprime, rel_len));
                            if (rel_count > 0 &&
                                base + static_cast<std::size_t>(layer_kprime) <= layer_rel_flat.size()) {
                                const int* rel = layer_rel_flat.data() + base;
                                filter_split_from_rel_by_rank_with_t(rel,
                                                                     rel_count,
                                                                     local_idx,
                                                                     /*query_rank=*/i,
                                                                     /*out_inner=*/nullptr,
                                                                     /*out_inner_t=*/nullptr,
                                                                     &tls.candidates_same,
                                                                     &candidates_same_t);
                            }

                            batch_node_local.push_back(local_idx);
                            batch_node_sample_id_base.push_back(
                                (static_cast<std::uint64_t>(static_cast<std::uint32_t>(forced_root_code)) << 32) |
                                static_cast<std::uint64_t>(static_cast<std::uint32_t>(local_idx)));
                            batch_X.insert(batch_X.end(), xi, xi + d);

                            const std::size_t same_n = tls.candidates_same.size();
                            for (std::size_t j = 0; j < same_n; ++j) {
                                batch_same_cand.push_back(tls.candidates_same[j]);
                                batch_same_t.push_back(candidates_same_t[j]);
                            }
                            batch_same_offsets.push_back(static_cast<int>(batch_same_cand.size()));
                        }

                        // Dynamic window batching for same-layer candidates (init-linkage, exact semantics).
                        // We build the exact same candidate list order (HNSW rel order + filtering), but
                        // evaluate eligible parents incrementally as they become committed. Tie-breaking on
                        // equal cost uses the candidate order index (`pos`), matching the sequential logic.
                        win_dyn_best_parent.assign(static_cast<std::size_t>(Bblk), -1);
                        win_dyn_best_cost.assign(static_cast<std::size_t>(Bblk),
                                                 std::numeric_limits<float>::infinity());
                        win_dyn_best_pos.assign(static_cast<std::size_t>(Bblk), std::numeric_limits<int>::max());
                        if (need_root_u32) {
                            win_dyn_best_code0.assign(static_cast<std::size_t>(Bblk),
                                                      static_cast<RootCode>(forced_root_code));
                        }
                        else {
                            win_dyn_best_code0.clear();
                        }
                        win_dyn_best_B.assign(static_cast<std::size_t>(Bblk) * static_cast<std::size_t>(m), 0);
                        win_dyn_best_a.assign(static_cast<std::size_t>(Bblk) * static_cast<std::size_t>(m), 0.0f);

                        // Build reverse edges for within-block parents: parent_off -> [(child_off, parent_local, pos)].
                        win_inv_offsets.assign(static_cast<std::size_t>(Bblk) + 1u, 0);
                        {
                            std::vector<int> inv_counts(static_cast<std::size_t>(Bblk), 0);
                            for (int child_off = 0; child_off < Bblk; ++child_off) {
                                const int same_start = batch_same_offsets[static_cast<std::size_t>(child_off)];
                                const int same_stop = batch_same_offsets[static_cast<std::size_t>(child_off + 1)];
                                for (int j = same_start; j < same_stop; ++j) {
                                    const int p_local = batch_same_cand[static_cast<std::size_t>(j)];
                                    const int rank = batch_rank_by_local[static_cast<std::size_t>(p_local)];
                                    if (rank >= 0 && rank < Bblk) {
                                        inv_counts[static_cast<std::size_t>(rank)] += 1;
                                    }
                                }
                            }
                            int sum = 0;
                            for (int i = 0; i < Bblk; ++i) {
                                win_inv_offsets[static_cast<std::size_t>(i)] = sum;
                                sum += inv_counts[static_cast<std::size_t>(i)];
                            }
                            win_inv_offsets[static_cast<std::size_t>(Bblk)] = sum;
                            win_inv_child.assign(static_cast<std::size_t>(sum), 0);
                            win_inv_parent_local.assign(static_cast<std::size_t>(sum), 0);
                            win_inv_pos.assign(static_cast<std::size_t>(sum), 0);
                            std::vector<int> cursor(static_cast<std::size_t>(Bblk), 0);

                            for (int child_off = 0; child_off < Bblk; ++child_off) {
                                const int same_start = batch_same_offsets[static_cast<std::size_t>(child_off)];
                                const int same_stop = batch_same_offsets[static_cast<std::size_t>(child_off + 1)];
                                for (int j = same_start; j < same_stop; ++j) {
                                    const int p_local = batch_same_cand[static_cast<std::size_t>(j)];
                                    const int pos = batch_same_t[static_cast<std::size_t>(j)];
                                    const int rank = batch_rank_by_local[static_cast<std::size_t>(p_local)];
                                    if (rank < 0 || rank >= Bblk) {
                                        continue;
                                    }
                                    const int off =
                                        win_inv_offsets[static_cast<std::size_t>(rank)] +
                                        cursor[static_cast<std::size_t>(rank)]++;
                                    win_inv_child[static_cast<std::size_t>(off)] = child_off;
                                    win_inv_parent_local[static_cast<std::size_t>(off)] = p_local;
                                    win_inv_pos[static_cast<std::size_t>(off)] = pos;
                                }
                            }
                        }

                        // Per-node pending list of enabled same-layer parents (includes frozen + dynamic).
                        int pending_stride = 1;
                        for (int i = 0; i < Bblk; ++i) {
                            const int have = batch_same_offsets[static_cast<std::size_t>(i + 1)] -
                                batch_same_offsets[static_cast<std::size_t>(i)];
                            pending_stride = std::max(pending_stride, have);
                        }
                        win_dyn_list_parent.assign(
                            static_cast<std::size_t>(Bblk) * static_cast<std::size_t>(pending_stride), -1);
                        win_dyn_list_pos.assign(
                            static_cast<std::size_t>(Bblk) * static_cast<std::size_t>(pending_stride),
                            std::numeric_limits<int>::max());
                        win_dyn_len.assign(static_cast<std::size_t>(Bblk), 0);
                        win_dyn_done.assign(static_cast<std::size_t>(Bblk), 0);

                        // ---- dyn_before: batch-evaluate frozen (prior-block) same-layer candidates ----
                        // Analogous to the iteration-linkage's dyn_before (stage-3c).  When block_start > 0,
                        // all same-layer candidates whose parent was committed in a prior block are already
                        // on device R_full.  We collect them into a single many-nodes batch, evaluate in one
                        // GPU call, and merge results during the commit loop. This avoids them trickling
                        // through the window preflush/flush path one node at a time.
                        have_init_dyn_before = false;
                        if (block_start > 0) {
                            init_dyn_before_counts.assign(static_cast<std::size_t>(Bblk), 0);
                            int Npairs_frozen = 0;
                            for (int ii = 0; ii < Bblk; ++ii) {
                                const int same_start = batch_same_offsets[static_cast<std::size_t>(ii)];
                                const int same_stop = batch_same_offsets[static_cast<std::size_t>(ii + 1)];
                                int cnt = 0;
                                for (int j = same_start; j < same_stop; ++j) {
                                    const int p_local = batch_same_cand[static_cast<std::size_t>(j)];
                                    const int rank = batch_rank_by_local[static_cast<std::size_t>(p_local)];
                                    if (rank >= 0) continue;
                                    if (depth_local[static_cast<std::size_t>(p_local)] + 1 > max_depth) continue;
                                    ++cnt;
                                }
                                init_dyn_before_counts[static_cast<std::size_t>(ii)] = cnt;
                                Npairs_frozen += cnt;
                            }
                            if (Npairs_frozen > 0) {
                                // Build offsets + flat candidate arrays.
                                init_dyn_before_offsets.assign(static_cast<std::size_t>(Bblk) + 1u, 0);
                                for (int ii = 0; ii < Bblk; ++ii) {
                                    init_dyn_before_offsets[static_cast<std::size_t>(ii + 1)] =
                                        init_dyn_before_offsets[static_cast<std::size_t>(ii)] +
                                        init_dyn_before_counts[static_cast<std::size_t>(ii)];
                                }
                                init_dyn_before_cand.resize(static_cast<std::size_t>(Npairs_frozen));
                                init_dyn_before_pair_node.resize(static_cast<std::size_t>(Npairs_frozen));
                                init_dyn_before_pos.resize(static_cast<std::size_t>(Npairs_frozen));
                                for (int ii = 0; ii < Bblk; ++ii) {
                                    const int same_start = batch_same_offsets[static_cast<std::size_t>(ii)];
                                    const int same_stop = batch_same_offsets[static_cast<std::size_t>(ii + 1)];
                                    int fpos = init_dyn_before_offsets[static_cast<std::size_t>(ii)];
                                    for (int j = same_start; j < same_stop; ++j) {
                                        const int p_local = batch_same_cand[static_cast<std::size_t>(j)];
                                        const int rank = batch_rank_by_local[static_cast<std::size_t>(p_local)];
                                        if (rank >= 0) continue;
                                        if (depth_local[static_cast<std::size_t>(p_local)] + 1 > max_depth) continue;
                                        init_dyn_before_cand[static_cast<std::size_t>(fpos)] = p_local;
                                        init_dyn_before_pair_node[static_cast<std::size_t>(fpos)] = ii;
                                        init_dyn_before_pos[static_cast<std::size_t>(fpos)] = batch_same_t[static_cast<
                                            std::size_t>(j)];
                                        ++fpos;
                                    }
                                }
                                // Output arrays.
                                init_dyn_before_best_parent.assign(static_cast<std::size_t>(Bblk), -1);
                                init_dyn_before_best_cost.assign(static_cast<std::size_t>(Bblk),
                                                                 std::numeric_limits<float>::infinity());
                                init_dyn_before_best_pos.assign(static_cast<std::size_t>(Bblk),
                                                                std::numeric_limits<int>::max());
                                init_dyn_before_best_B.assign(
                                    static_cast<std::size_t>(Bblk) * static_cast<std::size_t>(m), 0);
                                init_dyn_before_best_a.assign(
                                    static_cast<std::size_t>(Bblk) * static_cast<std::size_t>(m), 0.0f);
                                if (need_root_u32) {
                                    init_dyn_before_best_B_u32.assign(
                                        static_cast<std::size_t>(Bblk) * static_cast<std::size_t>(m), 0u);
                                }
                                init_dyn_before_sample_id.resize(static_cast<std::size_t>(Bblk));
                                for (int ii = 0; ii < Bblk; ++ii) {
                                    init_dyn_before_sample_id[static_cast<std::size_t>(ii)] =
                                        batch_node_sample_id_base[static_cast<std::size_t>(ii)];
                                }

                                // Flush pending device R_full updates so frozen parents are visible to the encoder.
                                if (!pending_rfull_cols.empty()) {
                                    if (!FlushPendingRfullUpdates()) {
                                        if (!can_cpu_fallback) {
                                            throw std::runtime_error(
                                                "Init-linkage dyn_before: failed to flush pending device R_full updates and CPU fallback is disabled.");
                                        }
                                        // Fall through: dyn_before disabled, frozen parents seed into win_dyn_list below.
                                        goto skip_init_dyn_before;
                                    }
                                }

                                {
                                    std::string batch_err;
                                    const double t0_db =
                                        stlq::linkage::LinkageBuildProfileEnabled()
                                            ? stlq::linkage::LinkageBuildWallNowS()
                                            : 0.0;
                                    const bool ok_gpu =
                                        need_root_u32
                                            ? EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExU32(
                                                cuda_ctx, C_root,
                                                batch_X.data(), d, Bblk,
                                                init_dyn_before_cand.data(),
                                                init_dyn_before_pair_node.data(),
                                                init_dyn_before_offsets.data(),
                                                Npairs_frozen,
                                                icm_round,
                                                ils_iters_gpu,
                                                perturb_k_gpu,
                                                seed_gpu,
                                                init_dyn_before_sample_id.data(),
                                                /*pair_t_override=*/init_dyn_before_pos.data(),
                                                init_dyn_before_best_parent.data(),
                                                init_dyn_before_best_cost.data(),
                                                init_dyn_before_best_B_u32.data(),
                                                init_dyn_before_best_a.data(),
                                                &batch_err)
                                            : EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootEx(
                                                cuda_ctx, C_root,
                                                batch_X.data(), d, Bblk,
                                                init_dyn_before_cand.data(),
                                                init_dyn_before_pair_node.data(),
                                                init_dyn_before_offsets.data(),
                                                Npairs_frozen,
                                                icm_round,
                                                ils_iters_gpu,
                                                perturb_k_gpu,
                                                seed_gpu,
                                                init_dyn_before_sample_id.data(),
                                                /*pair_t_override=*/init_dyn_before_pos.data(),
                                                init_dyn_before_best_parent.data(),
                                                init_dyn_before_best_cost.data(),
                                                init_dyn_before_best_B.data(),
                                                init_dyn_before_best_a.data(),
                                                &batch_err);
                                    if (ok_gpu) {
                                        have_init_dyn_before = true;
                                        // Resolve best pos (candidate order) for tie-breaking.
                                        for (int ii = 0; ii < Bblk; ++ii) {
                                            const int p = init_dyn_before_best_parent[static_cast<std::size_t>(ii)];
                                            if (p >= 0) {
                                                const int off0 = init_dyn_before_offsets[static_cast<std::size_t>(ii)];
                                                const int off1 = init_dyn_before_offsets[static_cast<std::size_t>(ii +
                                                    1)];
                                                for (int t = off0; t < off1; ++t) {
                                                    if (init_dyn_before_cand[static_cast<std::size_t>(t)] == p) {
                                                        init_dyn_before_best_pos[static_cast<std::size_t>(ii)] =
                                                            init_dyn_before_pos[static_cast<std::size_t>(t)];
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                        if (stlq::linkage::LinkageBuildProfileEnabled()) {
                                            if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                                                const double dt = stlq::linkage::LinkageBuildWallNowS() - t0_db;
                                                prof->good_gpu_wait_s += dt;
                                                prof->good_gpu_calls_many_nodes += 1;
                                            }
                                        }
                                    }
                                    else {
                                        LogWarn(std::string(
                                                "CUDA init-linkage dyn-before eval failed; frozen parents will use window path: ")
                                            +
                                            batch_err);
                                    }
                                }
                            }
                        }
                    skip_init_dyn_before:

                        // Seed frozen parents (rank < 0) into the window pending list — but only if dyn_before
                        // did not handle them (either block_start==0 or dyn_before failed).
                        if (!have_init_dyn_before) {
                            for (int child_off = 0; child_off < Bblk; ++child_off) {
                                const int same_start = batch_same_offsets[static_cast<std::size_t>(child_off)];
                                const int same_stop = batch_same_offsets[static_cast<std::size_t>(child_off + 1)];
                                for (int j = same_start; j < same_stop; ++j) {
                                    const int p_local = batch_same_cand[static_cast<std::size_t>(j)];
                                    const int pos = batch_same_t[static_cast<std::size_t>(j)];
                                    const int rank = batch_rank_by_local[static_cast<std::size_t>(p_local)];
                                    if (rank >= 0) {
                                        continue;
                                    }
                                    const int p_depth = depth_local[static_cast<std::size_t>(p_local)];
                                    if (p_depth + 1 > max_depth) {
                                        continue;
                                    }
                                    int& have = win_dyn_len[static_cast<std::size_t>(child_off)];
                                    const std::size_t idx =
                                        static_cast<std::size_t>(child_off) * static_cast<std::size_t>(pending_stride) +
                                        static_cast<std::size_t>(have);
                                    win_dyn_list_parent[idx] = p_local;
                                    win_dyn_list_pos[idx] = pos;
                                    have += 1;
                                }
                            }
                        } // end if (!have_init_dyn_before) — frozen parent seeding

                        const bool async_eval_enabled =
                            (runtime_cfg && runtime_cfg->cuda_linkage_eval_async_pinned_mb > 0);
                        struct WinDynAsyncState
                        {
                            bool in_flight = false;
                            int slice_len = 0;
                            int total_pairs = 0;
                        } win_dyn_async;
                        int win_dyn_tiny_forced_node_off = -1;
                        const auto WinDynFinishAsyncIfAny = [&]() -> bool
                        {
                            if (!win_dyn_async.in_flight) return true;
                            std::string gpu_err;
                            const bool ok_finish =
                                need_root_u32
                                    ? EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExFinishU32(
                                        cuda_ctx,
                                        win_dyn_flush_best_parent.data(),
                                        win_dyn_flush_best_cost.data(),
                                        win_dyn_flush_best_B_u32.data(),
                                        win_dyn_flush_best_a.data(),
                                        &gpu_err)
                                    : EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExFinish(
                                        cuda_ctx,
                                        win_dyn_flush_best_parent.data(),
                                        win_dyn_flush_best_cost.data(),
                                        win_dyn_flush_best_B.data(),
                                        win_dyn_flush_best_a.data(),
                                        &gpu_err);
                            if (!ok_finish) {
                                if (!can_cpu_fallback) {
                                    throw std::runtime_error(
                                        std::string(
                                            "CUDA init-linkage same-layer window async finish failed and CPU fallback disabled: ")
                                        +
                                        gpu_err);
                                }
                                LogWarn(std::string(
                                        "CUDA init-linkage same-layer window async finish failed, fallback to CPU: ") +
                                    gpu_err);
                                win_dyn_async.in_flight = false;
                                return false;
                            }

                            for (int s = 0; s < win_dyn_async.slice_len; ++s) {
                                const int kk = win_dyn_flush_sel_nodes[static_cast<std::size_t>(s)];
                                const float c = win_dyn_flush_best_cost[static_cast<std::size_t>(s)];
                                const int p = win_dyn_flush_best_parent[static_cast<std::size_t>(s)];
                                if (p >= 0) {
                                    int p_pos = std::numeric_limits<int>::max();
                                    const int off0 = win_dyn_flush_offsets[static_cast<std::size_t>(s)];
                                    const int off1 = win_dyn_flush_offsets[static_cast<std::size_t>(s + 1)];
                                    for (int t = off0; t < off1; ++t) {
                                        if (win_dyn_flush_cand[static_cast<std::size_t>(t)] == p) {
                                            p_pos = win_dyn_flush_pos[static_cast<std::size_t>(t)];
                                            break;
                                        }
                                    }

                                    const float prev_c = win_dyn_best_cost[static_cast<std::size_t>(kk)];
                                    const int prev_pos = win_dyn_best_pos[static_cast<std::size_t>(kk)];
                                    if (c < prev_c || (c == prev_c && p_pos < prev_pos)) {
                                        win_dyn_best_cost[static_cast<std::size_t>(kk)] = c;
                                        win_dyn_best_parent[static_cast<std::size_t>(kk)] = p;
                                        win_dyn_best_pos[static_cast<std::size_t>(kk)] = p_pos;
                                        const std::size_t dst = static_cast<std::size_t>(kk) * static_cast<std::size_t>(
                                            m);
                                        const std::size_t src = static_cast<std::size_t>(s) * static_cast<std::size_t>(
                                            m);
                                        if (!need_root_u32) {
                                            std::memcpy(win_dyn_best_B.data() + dst,
                                                        win_dyn_flush_best_B.data() + src,
                                                        sizeof(FullCode) * static_cast<std::size_t>(m));
                                        }
                                        else {
                                            win_dyn_best_code0[static_cast<std::size_t>(kk)] =
                                                static_cast<RootCode>(win_dyn_flush_best_B_u32[src + 0]);
                                            win_dyn_best_B[dst + 0] = static_cast<FullCode>(0);
                                            for (int l = 1; l < m; ++l) {
                                                win_dyn_best_B[dst + static_cast<std::size_t>(l)] =
                                                    static_cast<FullCode>(win_dyn_flush_best_B_u32[src + static_cast<
                                                        std::size_t>(l)]);
                                            }
                                        }
                                        std::memcpy(win_dyn_best_a.data() + dst,
                                                    win_dyn_flush_best_a.data() + src,
                                                    sizeof(float) * static_cast<std::size_t>(m));
                                    }
                                }
                            }
                            for (int s = 0; s < win_dyn_async.slice_len; ++s) {
                                const int kk = win_dyn_flush_sel_nodes[static_cast<std::size_t>(s)];
                                win_dyn_done[static_cast<std::size_t>(kk)] +=
                                    win_dyn_flush_sel_take[static_cast<std::size_t>(s)];
                            }
                            win_dyn_async.in_flight = false;
                            return true;
                        };

                        const auto FlushWindowDynamicManyNodes = [&](int start_off, int force_node_off) -> int
                        {
                            if (start_off < 0) start_off = 0;
                            if (start_off >= Bblk) return 0;
                            if (win_dyn_async.in_flight) {
                                if (!WinDynFinishAsyncIfAny()) return 0;
                            }

                            win_dyn_flush_sel_nodes.clear();
                            win_dyn_flush_sel_take.clear();
                            int total_pairs = 0;

                            const int soft_nodes_cap =
                                (force_node_off >= 0) ? std::numeric_limits<int>::max() : win_pending_nodes;

                            for (int k = start_off; k < Bblk; ++k) {
                                const int have = win_dyn_len[static_cast<std::size_t>(k)];
                                const int done = win_dyn_done[static_cast<std::size_t>(k)];
                                if (have <= done) continue;
                                const int remain = have - done;
                                const int can_take = std::min(remain, std::max(0, win_max_pairs - total_pairs));
                                if (can_take <= 0) break;
                                win_dyn_flush_sel_nodes.push_back(k);
                                win_dyn_flush_sel_take.push_back(can_take);
                                total_pairs += can_take;
                                if (total_pairs >= win_max_pairs ||
                                    static_cast<int>(win_dyn_flush_sel_nodes.size()) >= soft_nodes_cap) {
                                    break;
                                }
                            }

                            if (total_pairs <= 0) return 0;
                            if (force_node_off < 0 && total_pairs < win_min_pairs) {
                                return 0;
                            }

                            // Stage-3f(1) init-linkage parity: for tiny *forced* flushes, avoid launching a tiny
                            // many-nodes GPU evaluation. Instead, let the commit loop evaluate the remaining
                            // tail on CPU for the forced node (semantics preserved, faster wall time).
                            {
                                const bool forced = (force_node_off >= 0);
                                bool tiny_cpu_enabled = false;
                                int tiny_cpu_max_pairs = 0;
                                int tiny_cpu_max_nodes = 0;
                                if (runtime_cfg) {
                                    tiny_cpu_enabled = runtime_cfg->cuda_linkage_same_dynamic_tiny_cpu;
                                    tiny_cpu_max_pairs = runtime_cfg->cuda_linkage_same_dynamic_tiny_cpu_max_pairs;
                                    tiny_cpu_max_nodes = runtime_cfg->cuda_linkage_same_dynamic_tiny_cpu_max_nodes;
                                    if (tiny_cpu_max_pairs <= 0) tiny_cpu_enabled = false;
                                    if (tiny_cpu_max_nodes < 0) tiny_cpu_max_nodes = 0;
                                }
                                const int slice_len = static_cast<int>(win_dyn_flush_sel_nodes.size());
                                const bool use_tiny_cpu =
                                    forced && tiny_cpu_enabled && (total_pairs <= tiny_cpu_max_pairs) &&
                                    (tiny_cpu_max_nodes <= 0 || slice_len <= tiny_cpu_max_nodes);
                                const bool tiny_cpu_available =
                                    can_cpu_fallback || allow_large_root_tiny_cpu;
                                if (use_tiny_cpu && tiny_cpu_available) {
                                    win_dyn_tiny_forced_node_off = force_node_off;
                                    if (stlq::linkage::LinkageBuildProfileEnabled()) {
                                        if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                                            prof->good_dyn_window_tiny_cpu_calls += 1;
                                            prof->good_dyn_window_tiny_cpu_nodes_sum +=
                                                static_cast<std::uint64_t>(slice_len);
                                            prof->good_dyn_window_tiny_cpu_pairs_sum +=
                                                static_cast<std::uint64_t>(std::max(0, total_pairs));
                                        }
                                    }
                                    return 0;
                                }
                            }

                            const int slice_len = static_cast<int>(win_dyn_flush_sel_nodes.size());
                            win_dyn_flush_offsets.assign(static_cast<std::size_t>(slice_len) + 1u, 0);
                            win_dyn_flush_cand.resize(static_cast<std::size_t>(total_pairs));
                            win_dyn_flush_pos.resize(static_cast<std::size_t>(total_pairs));
                            win_dyn_flush_pair_node.resize(static_cast<std::size_t>(total_pairs));
                            win_dyn_flush_X.resize(static_cast<std::size_t>(slice_len) * static_cast<std::size_t>(d));
                            win_dyn_flush_node_sample_id_base.assign(static_cast<std::size_t>(slice_len), 0);

                            int pos = 0;
                            win_dyn_flush_offsets[0] = 0;
                            for (int s = 0; s < slice_len; ++s) {
                                const int kk = win_dyn_flush_sel_nodes[static_cast<std::size_t>(s)];
                                const int take = win_dyn_flush_sel_take[static_cast<std::size_t>(s)];
                                const int done = win_dyn_done[static_cast<std::size_t>(kk)];
                                const int have = win_dyn_len[static_cast<std::size_t>(kk)];

                                const float* xi = batch_X.data() + static_cast<std::size_t>(kk) * static_cast<
                                    std::size_t>(d);
                                std::memcpy(
                                    win_dyn_flush_X.data() + static_cast<std::size_t>(s) * static_cast<std::size_t>(d),
                                    xi, sizeof(float) * static_cast<std::size_t>(d));
                                win_dyn_flush_node_sample_id_base[static_cast<std::size_t>(s)] =
                                    batch_node_sample_id_base[static_cast<std::size_t>(kk)];

                                const std::size_t row0 =
                                    static_cast<std::size_t>(kk) * static_cast<std::size_t>(pending_stride);

                                // Keep per-node candidate order stable for exact semantics: if two candidates
                                // have equal cost, sequential logic keeps the earlier one in the original HNSW
                                // filtered list. We therefore sort only the remaining enabled segment by `pos`.
                                for (int a = done + 1; a < have; ++a) {
                                    const std::size_t ia = row0 + static_cast<std::size_t>(a);
                                    int key_parent = win_dyn_list_parent[ia];
                                    int key_pos = win_dyn_list_pos[ia];
                                    int b = a - 1;
                                    while (b >= done) {
                                        const std::size_t ib = row0 + static_cast<std::size_t>(b);
                                        if (win_dyn_list_pos[ib] <= key_pos) {
                                            break;
                                        }
                                        const std::size_t ib1 = row0 + static_cast<std::size_t>(b + 1);
                                        win_dyn_list_parent[ib1] = win_dyn_list_parent[ib];
                                        win_dyn_list_pos[ib1] = win_dyn_list_pos[ib];
                                        --b;
                                    }
                                    const std::size_t ins = row0 + static_cast<std::size_t>(b + 1);
                                    win_dyn_list_parent[ins] = key_parent;
                                    win_dyn_list_pos[ins] = key_pos;
                                }
                                for (int j = 0; j < take; ++j) {
                                    const std::size_t src = row0 + static_cast<std::size_t>(done + j);
                                    const int p_local = win_dyn_list_parent[src];
                                    const int p_pos = win_dyn_list_pos[src];
                                    win_dyn_flush_cand[static_cast<std::size_t>(pos)] = p_local;
                                    win_dyn_flush_pos[static_cast<std::size_t>(pos)] = p_pos;
                                    win_dyn_flush_pair_node[static_cast<std::size_t>(pos)] = s;
                                    ++pos;
                                }
                                win_dyn_flush_offsets[static_cast<std::size_t>(s + 1)] = pos;
                            }

                            win_dyn_flush_best_parent.assign(static_cast<std::size_t>(slice_len), -1);
                            win_dyn_flush_best_cost.assign(static_cast<std::size_t>(slice_len),
                                                           std::numeric_limits<float>::infinity());
                            win_dyn_flush_best_B.assign(
                                static_cast<std::size_t>(slice_len) * static_cast<std::size_t>(m), 0);
                            if (need_root_u32) {
                                win_dyn_flush_best_code0.assign(static_cast<std::size_t>(slice_len),
                                                                static_cast<RootCode>(forced_root_code));
                                win_dyn_flush_best_B_u32.assign(
                                    static_cast<std::size_t>(slice_len) * static_cast<std::size_t>(m), 0u);
                            }
                            else {
                                win_dyn_flush_best_code0.clear();
                                win_dyn_flush_best_B_u32.clear();
                            }
                            win_dyn_flush_best_a.assign(
                                static_cast<std::size_t>(slice_len) * static_cast<std::size_t>(m), 0.0f);

                            std::string batch_err;
                            if (!pending_rfull_cols.empty() && !FlushPendingRfullUpdates()) {
                                if (!can_cpu_fallback) {
                                    throw std::runtime_error(
                                        "Init-linkage: failed to flush pending device R_full updates and CPU fallback is disabled.");
                                }
                                return 0;
                            }
                            if (force_node_off < 0 && async_eval_enabled) {
                                const bool ok_enqueue =
                                    need_root_u32
                                        ? EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExEnqueueU32(
                                            cuda_ctx, C_root,
                                            win_dyn_flush_X.data(), d, slice_len,
                                            win_dyn_flush_cand.data(),
                                            win_dyn_flush_pair_node.data(),
                                            win_dyn_flush_offsets.data(),
                                            total_pairs,
                                            icm_round,
                                            ils_iters_gpu,
                                            perturb_k_gpu,
                                            seed_gpu,
                                            win_dyn_flush_node_sample_id_base.data(),
                                            /*pair_t_override=*/win_dyn_flush_pos.data(),
                                            &batch_err)
                                        : EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExEnqueue(
                                            cuda_ctx, C_root,
                                            win_dyn_flush_X.data(), d, slice_len,
                                            win_dyn_flush_cand.data(),
                                            win_dyn_flush_pair_node.data(),
                                            win_dyn_flush_offsets.data(),
                                            total_pairs,
                                            icm_round,
                                            ils_iters_gpu,
                                            perturb_k_gpu,
                                            seed_gpu,
                                            win_dyn_flush_node_sample_id_base.data(),
                                            /*pair_t_override=*/win_dyn_flush_pos.data(),
                                            &batch_err);
                                if (ok_enqueue) {
                                    win_dyn_async.in_flight = true;
                                    win_dyn_async.slice_len = slice_len;
                                    win_dyn_async.total_pairs = total_pairs;
                                    return 0;
                                }
                            }
                            const bool ok_gpu =
                                need_root_u32
                                    ? EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExU32(
                                        cuda_ctx, C_root,
                                        win_dyn_flush_X.data(), d, slice_len,
                                        win_dyn_flush_cand.data(),
                                        win_dyn_flush_pair_node.data(),
                                        win_dyn_flush_offsets.data(),
                                        total_pairs,
                                        icm_round,
                                        ils_iters_gpu,
                                        perturb_k_gpu,
                                        seed_gpu,
                                        win_dyn_flush_node_sample_id_base.data(),
                                        /*pair_t_override=*/win_dyn_flush_pos.data(),
                                        win_dyn_flush_best_parent.data(),
                                        win_dyn_flush_best_cost.data(),
                                        win_dyn_flush_best_B_u32.data(),
                                        win_dyn_flush_best_a.data(),
                                        &batch_err)
                                    : EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootEx(
                                        cuda_ctx, C_root,
                                        win_dyn_flush_X.data(), d, slice_len,
                                        win_dyn_flush_cand.data(),
                                        win_dyn_flush_pair_node.data(),
                                        win_dyn_flush_offsets.data(),
                                        total_pairs,
                                        icm_round,
                                        ils_iters_gpu,
                                        perturb_k_gpu,
                                        seed_gpu,
                                        win_dyn_flush_node_sample_id_base.data(),
                                        /*pair_t_override=*/win_dyn_flush_pos.data(),
                                        win_dyn_flush_best_parent.data(),
                                        win_dyn_flush_best_cost.data(),
                                        win_dyn_flush_best_B.data(),
                                        win_dyn_flush_best_a.data(),
                                        &batch_err);
                            if (!ok_gpu) {
                                if (!can_cpu_fallback) {
                                    throw std::runtime_error(
                                        std::string(
                                            "CUDA init-linkage same-layer window eval failed and CPU fallback disabled: ")
                                        + batch_err);
                                }
                                LogWarn(std::string(
                                    "CUDA init-linkage same-layer window eval failed, fallback to CPU: ") + batch_err);
                                return 0;
                            }

                            for (int s = 0; s < slice_len; ++s) {
                                const int kk = win_dyn_flush_sel_nodes[static_cast<std::size_t>(s)];
                                const float c = win_dyn_flush_best_cost[static_cast<std::size_t>(s)];
                                const int p = win_dyn_flush_best_parent[static_cast<std::size_t>(s)];
                                if (p >= 0) {
                                    int p_pos = std::numeric_limits<int>::max();
                                    const int off0 = win_dyn_flush_offsets[static_cast<std::size_t>(s)];
                                    const int off1 = win_dyn_flush_offsets[static_cast<std::size_t>(s + 1)];
                                    for (int t = off0; t < off1; ++t) {
                                        if (win_dyn_flush_cand[static_cast<std::size_t>(t)] == p) {
                                            p_pos = win_dyn_flush_pos[static_cast<std::size_t>(t)];
                                            break;
                                        }
                                    }

                                    const float prev_c = win_dyn_best_cost[static_cast<std::size_t>(kk)];
                                    const int prev_pos = win_dyn_best_pos[static_cast<std::size_t>(kk)];
                                    if (c < prev_c || (c == prev_c && p_pos < prev_pos)) {
                                        win_dyn_best_cost[static_cast<std::size_t>(kk)] = c;
                                        win_dyn_best_parent[static_cast<std::size_t>(kk)] = p;
                                        win_dyn_best_pos[static_cast<std::size_t>(kk)] = p_pos;
                                        const std::size_t dst = static_cast<std::size_t>(kk) * static_cast<std::size_t>(
                                            m);
                                        const std::size_t src = static_cast<std::size_t>(s) * static_cast<std::size_t>(
                                            m);
                                        if (!need_root_u32) {
                                            std::memcpy(win_dyn_best_B.data() + dst,
                                                        win_dyn_flush_best_B.data() + src,
                                                        sizeof(FullCode) * static_cast<std::size_t>(m));
                                        }
                                        else {
                                            win_dyn_best_code0[static_cast<std::size_t>(kk)] =
                                                static_cast<RootCode>(win_dyn_flush_best_B_u32[src + 0]);
                                            win_dyn_best_B[dst + 0] = static_cast<FullCode>(0);
                                            for (int l = 1; l < m; ++l) {
                                                win_dyn_best_B[dst + static_cast<std::size_t>(l)] =
                                                    static_cast<FullCode>(win_dyn_flush_best_B_u32[src + static_cast<
                                                        std::size_t>(l)]);
                                            }
                                        }
                                        std::memcpy(win_dyn_best_a.data() + dst,
                                                    win_dyn_flush_best_a.data() + src,
                                                    sizeof(float) * static_cast<std::size_t>(m));
                                    }
                                }
                            }

                            // Consume.
                            for (int s = 0; s < slice_len; ++s) {
                                const int kk = win_dyn_flush_sel_nodes[static_cast<std::size_t>(s)];
                                const int take = win_dyn_flush_sel_take[static_cast<std::size_t>(s)];
                                win_dyn_done[static_cast<std::size_t>(kk)] += take;
                            }
                            return total_pairs;
                        };

                        const auto WinDynMaybePreflushFrom = [&](int start_off)
                        {
                            if (win_dyn_async.in_flight) return;
                            // Note: the previous `if (!async_eval_enabled) return;` gate is removed.
                            // Synchronous lazy preflush is now also active; the threshold below
                            // ensures cascade GPU calls are batched rather than fired per-commit.
                            const bool preflush_nodes_enabled =
                                (runtime_cfg && runtime_cfg->cuda_linkage_same_dynamic_preflush_nodes > 0);
                            int target_pairs = 0;
                            if (runtime_cfg) {
                                if (runtime_cfg->cuda_linkage_same_dynamic_preflush_pairs > 0) {
                                    target_pairs =
                                        std::max(1, runtime_cfg->cuda_linkage_same_dynamic_preflush_pairs);
                                }
                                else if (runtime_cfg->cuda_linkage_same_dynamic_preflush_pairs < 0) {
                                    if (runtime_cfg->cuda_linkage_batch_min_pairs > 0) {
                                        target_pairs = std::max(1, runtime_cfg->cuda_linkage_batch_min_pairs);
                                    }
                                    else {
                                        target_pairs = std::max(1, win_min_pairs);
                                    }
                                }
                                else {
                                    // preflush_pairs==0: in async mode skip (will be triggered explicitly);
                                    // in sync mode use cuda_linkage_batch_nodes as coarser threshold so
                                    // cascade additions from multiple commits accumulate before flush.
                                    if (!async_eval_enabled) {
                                        target_pairs = std::max(
                                            win_min_pairs,
                                            (runtime_cfg
                                                 ? std::max(1, runtime_cfg->cuda_linkage_batch_nodes)
                                                 : win_min_pairs));
                                    }
                                }
                            }
                            const bool preflush_pairs_enabled = (target_pairs > 0);
                            if (!preflush_pairs_enabled && !preflush_nodes_enabled) return;
                            if (start_off < 0) start_off = 0;
                            if (start_off >= Bblk) return;

                            int trigger_nodes = std::numeric_limits<int>::max();
                            if (preflush_nodes_enabled) {
                                // Init-linkage window size is Bblk; use it as a natural cap.
                                trigger_nodes = std::max(
                                    1, std::min(Bblk, runtime_cfg->cuda_linkage_same_dynamic_preflush_nodes));
                            }

                            int pending_pairs = 0;
                            int pending_nodes = 0;
                            for (int k = start_off; k < Bblk; ++k) {
                                const int have = win_dyn_len[static_cast<std::size_t>(k)];
                                const int done = win_dyn_done[static_cast<std::size_t>(k)];
                                if (have > done) {
                                    pending_nodes += 1;
                                    if (preflush_pairs_enabled) {
                                        pending_pairs += (have - done);
                                    }
                                    if ((preflush_pairs_enabled && pending_pairs >= target_pairs) ||
                                        (preflush_nodes_enabled && pending_nodes >= trigger_nodes)) {
                                        break;
                                    }
                                }
                            }
                            if (preflush_pairs_enabled && pending_pairs >= target_pairs) {
                                // ok
                            }
                            else if (preflush_nodes_enabled && pending_nodes >= trigger_nodes) {
                                // ok
                            }
                            else {
                                return;
                            }

                            (void)FlushWindowDynamicManyNodes(start_off, /*force_node_off=*/-1);
                        };

                        const auto WinDynEnsureCompleteForNode = [&](int node_off)
                        {
                            if (node_off < 0 || node_off >= Bblk) return;
                            if (!WinDynFinishAsyncIfAny()) return;
                            for (;;) {
                                const int have = win_dyn_len[static_cast<std::size_t>(node_off)];
                                const int done = win_dyn_done[static_cast<std::size_t>(node_off)];
                                if (done >= have) break;
                                if (FlushWindowDynamicManyNodes(node_off, /*force_node_off=*/node_off) <= 0) break;
                            }
                        };

#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                        // Inner GPU batch may be enqueued async; complete it before using the same CUDA ctx
                        // for same-layer window evaluation, and before consulting `inner_best_*`.
                        if (!finish_inner_async()) {
                            layer_rel_flat.clear();
                            layer_kprime = 0;
                            break;
                        }
#endif

                        // Preflush frozen candidates.
                        WinDynMaybePreflushFrom(/*start_off=*/0);

                        // Commit order within the block, with dynamic window batching.
                        for (int i = 0; i < Bblk; ++i) {
                            const int local_idx = batch_node_local[static_cast<std::size_t>(i)];

                            float bestE2 = base_mse[static_cast<std::size_t>(local_idx)];
                            int best_parent_local = -1;
                            int best_pos = std::numeric_limits<int>::max();
                            RootCode best_code0 =
                                need_root_u32
                                    ? (*code0_root_inout)[static_cast<std::size_t>(local_idx)]
                                    : static_cast<RootCode>((*B)(0, local_idx));

                            if (!inner_best_valid.empty() &&
                                inner_best_valid[static_cast<std::size_t>(local_idx)] != 0u) {
                                const int p_inner = inner_best_parent_local[static_cast<std::size_t>(local_idx)];
                                const float c_inner = inner_best_cost[static_cast<std::size_t>(local_idx)];
                                const int pos_inner = inner_best_t[static_cast<std::size_t>(local_idx)];
                                if (p_inner >= 0 &&
                                    (c_inner < bestE2 ||
                                        (best_parent_local >= 0 && c_inner == bestE2 && pos_inner < best_pos))) {
                                    bestE2 = c_inner;
                                    best_parent_local = p_inner;
                                    best_pos = pos_inner;
                                    tls.B_best.resize(static_cast<std::size_t>(m));
                                    tls.a_best.resize(static_cast<std::size_t>(m));
                                    std::memcpy(tls.B_best.data(),
                                                inner_best_B.data() +
                                                static_cast<std::size_t>(local_idx) * static_cast<std::size_t>(m),
                                                sizeof(FullCode) * static_cast<std::size_t>(m));
                                    std::memcpy(tls.a_best.data(),
                                                inner_best_a.data() +
                                                static_cast<std::size_t>(local_idx) * static_cast<std::size_t>(m),
                                                sizeof(float) * static_cast<std::size_t>(m));
                                    if (need_root_u32 && !inner_best_code0.empty()) {
                                        best_code0 = inner_best_code0[static_cast<std::size_t>(local_idx)];
                                    }
                                    else if (!need_root_u32 && !tls.B_best.empty()) {
                                        best_code0 = static_cast<RootCode>(tls.B_best[0]);
                                    }
                                }
                            }

                            // Merge dyn_before results (frozen same-layer candidates, batch-evaluated before commit loop).
                            if (have_init_dyn_before) {
                                const int p_db = init_dyn_before_best_parent[static_cast<std::size_t>(i)];
                                const float c_db = init_dyn_before_best_cost[static_cast<std::size_t>(i)];
                                const int pos_db = init_dyn_before_best_pos[static_cast<std::size_t>(i)];
                                if (p_db >= 0 &&
                                    (c_db < bestE2 ||
                                        (best_parent_local >= 0 && c_db == bestE2 && pos_db < best_pos))) {
                                    bestE2 = c_db;
                                    best_parent_local = p_db;
                                    best_pos = pos_db;
                                    tls.B_best.resize(static_cast<std::size_t>(m));
                                    tls.a_best.resize(static_cast<std::size_t>(m));
                                    const std::size_t src =
                                        static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
                                    if (need_root_u32) {
                                        best_code0 =
                                            static_cast<RootCode>(init_dyn_before_best_B_u32[src]);
                                        tls.B_best[0] = static_cast<FullCode>(0);
                                        for (std::size_t l = 1; l < static_cast<std::size_t>(m); ++l) {
                                            tls.B_best[l] = static_cast<FullCode>(
                                                init_dyn_before_best_B_u32[src + l]);
                                        }
                                    }
                                    else {
                                        std::memcpy(tls.B_best.data(),
                                                    init_dyn_before_best_B.data() + src,
                                                    sizeof(FullCode) * static_cast<std::size_t>(m));
                                        best_code0 = static_cast<RootCode>(tls.B_best[0]);
                                    }
                                    std::memcpy(tls.a_best.data(),
                                                init_dyn_before_best_a.data() + src,
                                                sizeof(float) * static_cast<std::size_t>(m));
                                }
                            }

                            WinDynEnsureCompleteForNode(i);
                            {
                                const int have = win_dyn_len[static_cast<std::size_t>(i)];
                                const int done = win_dyn_done[static_cast<std::size_t>(i)];
                                if (done < have) {
                                    const bool did_tiny_forced_skip = (win_dyn_tiny_forced_node_off == i);
                                    if (did_tiny_forced_skip) {
                                        // Clear the marker so it doesn't affect later nodes.
                                        win_dyn_tiny_forced_node_off = -1;
                                    }
                                    // Not enough pairs to flush on GPU (or intentionally skipped a tiny forced flush).
                                    // Evaluate the remaining tail on CPU for this node.
                                    const float* xi =
                                        batch_X.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(d);
                                    const Precomp* pre_tail = &pre_root;
                                    if (!can_cpu_fallback) {
                                        if (!allow_large_root_tiny_cpu) {
                                            throw std::runtime_error(
                                                "Init-linkage: same-layer window tail requires CPU fallback (full-precomp). "
                                                "Enable runtime.cuda_linkage_init_large_root_tiny_cpu for large-root init-linkage.");
                                        }
                                        pre_tail = &EnsureFixedRootTinyCpuPrecomp();
                                    }
                                    else {
                                        ComputeXcBase(pre_root, xi, &tls.xC_base_buf);
                                    }

                                    // Use a deterministic per-node RNG so this optimization does not perturb
                                    // the global RNG stream for other nodes.
                                    std::mt19937 rng_tail(static_cast<std::mt19937::result_type>(
                                        static_cast<std::uint32_t>(linkage_cfg.seed + 997 * forced_root_code) ^
                                        (static_cast<std::uint32_t>(local_idx) * 0x9E3779B9u) ^
                                        0xA7F1C3D5u));

                                    const int ils_k_eff = ils_k;
                                    const bool do_ils = (m > 1 && ils_rounds > 0 && ils_k_eff > 0);
                                    const int ils_layer_lo = (!can_cpu_fallback) ? 1 : 0;
                                    std::uniform_int_distribution<int> layer_dist(
                                        ils_layer_lo, std::max(ils_layer_lo, m - 1));
                                    std::vector<std::uniform_int_distribution<int>> code_dists;
                                    if constexpr (UseIls) {
                                        if (do_ils) {
                                            code_dists.reserve(static_cast<std::size_t>(m));
                                            for (int layer = 0; layer < m; ++layer) {
                                                const int h = pre_tail->h_vec[layer];
                                                code_dists.emplace_back(0, std::max(0, h - 2));
                                            }
                                        }
                                    }

                                    // Start from the current best (possibly from earlier GPU flushes).
                                    int best_parent = win_dyn_best_parent[static_cast<std::size_t>(i)];
                                    float best_cost = win_dyn_best_cost[static_cast<std::size_t>(i)];
                                    int best_pos = win_dyn_best_pos[static_cast<std::size_t>(i)];

                                    const std::size_t row0 =
                                        static_cast<std::size_t>(i) * static_cast<std::size_t>(pending_stride);
                                    for (int t = done; t < have; ++t) {
                                        const int p_local =
                                            win_dyn_list_parent[row0 + static_cast<std::size_t>(t)];
                                        const int pos =
                                            win_dyn_list_pos[row0 + static_cast<std::size_t>(t)];
                                        const int new_depth =
                                            depth_local[static_cast<std::size_t>(p_local)] + 1;
                                        if (new_depth > max_depth) {
                                            continue;
                                        }

                                        const float* rp = R_full.Col(p_local);
                                        float re_norm2 = 0.0f;
#pragma omp simd reduction(+:re_norm2)
                                        for (int r = 0; r < d; ++r) {
                                            const float v = xi[r] - rp[r];
                                            tls.re[static_cast<std::size_t>(r)] = v;
                                            re_norm2 += v * v;
                                        }

                                        if (can_cpu_fallback) {
                                            BuildXcForParentSingleLocal(pre_root,
                                                                        tls.xC_base_buf,
                                                                        *B,
                                                                        *a,
                                                                        cluster_indices,
                                                                        parent_local,
                                                                        p_local,
                                                                        &tls.xC_buf);
                                        }
                                        else {
                                            // Large-root fixed-root CPU tail: compute xC from residual directly.
                                            ComputeXcFromResidual(*pre_tail, tls.re, &tls.xC_buf);
                                        }

                                        QuantizeWithKmeansAbsNoNormalSingle(*pre_tail,
                                                                            tls.re,
                                                                            &tls.B_init,
                                                                            &tls.a_init,
                                                                            &tls.rC_buf,
                                                                            &tls.available);

                                        if (!SolveLeastSquaresVector(*pre_tail,
                                                                     tls.xC_buf,
                                                                     tls.B_init,
                                                                     &tls.A,
                                                                     &tls.b,
                                                                     &tls.work,
                                                                     tls.a_init.data())) {
                                            ++ls_failures;
                                            continue;
                                        }
                                        if (icm_round > 0) {
                                            DynamicIcmEncodingSingleForResidual(*pre_tail,
                                                tls.re,
                                                tls.xC_buf,
                                                icm_round,
                                                &tls.B_init,
                                                &tls.a_init,
                                                &tls.a_bak,
                                                &tls.rC_buf,
                                                &tls.A,
                                                &tls.b,
                                                &tls.work,
                                                &rng_tail,
                                                &ls_failures);
                                        }

                                        float e2_best =
                                            SampleCostFromXc(*pre_tail, tls.xC_buf, tls.B_init, tls.a_init, re_norm2);

                                        if constexpr (UseIls) {
                                            if (do_ils) {
                                                for (int il = 0; il < ils_rounds; ++il) {
                                                    for (int u = 0; u < ils_k_eff; ++u) {
                                                        const int layer = layer_dist(rng_tail);
                                                        const int h = pre_tail->h_vec[layer];
                                                        const int old_code =
                                                            static_cast<int>(tls.B_init[static_cast<std::size_t>(
                                                                layer)]);
                                                        int new_code = old_code;
                                                        if (h > 1) {
                                                            int raw = code_dists[static_cast<std::size_t>(layer)](
                                                                rng_tail);
                                                            new_code = raw;
                                                            if (new_code >= old_code) {
                                                                new_code += 1;
                                                            }
                                                        }
                                                        const int startf = pre_tail->offsets[layer];
                                                        tls.pert_layers[static_cast<std::size_t>(u)] = layer;
                                                        tls.new_flat[static_cast<std::size_t>(u)] = startf + new_code;
                                                    }

                                                    tls.B_try = tls.B_init;
                                                    for (int u = 0; u < ils_k_eff; ++u) {
                                                        const int layer = tls.pert_layers[static_cast<std::size_t>(u)];
                                                        const int flat = tls.new_flat[static_cast<std::size_t>(u)];
                                                        const int code = flat - pre_tail->offsets[layer];
                                                        tls.B_try[static_cast<std::size_t>(layer)] =
                                                            static_cast<FullCode>(code);
                                                    }
                                                    tls.a_try = tls.a_init;
                                                    if (!SolveLeastSquaresVector(*pre_tail,
                                                        tls.xC_buf,
                                                        tls.B_try,
                                                        &tls.A,
                                                        &tls.b,
                                                        &tls.work,
                                                        tls.a_try.data())) {
                                                        ++ls_failures;
                                                        continue;
                                                    }
                                                    if (icm_round > 0) {
                                                        DynamicIcmEncodingSingleForResidual(*pre_tail,
                                                            tls.re,
                                                            tls.xC_buf,
                                                            icm_round,
                                                            &tls.B_try,
                                                            &tls.a_try,
                                                            &tls.a_bak,
                                                            &tls.rC_buf,
                                                            &tls.A,
                                                            &tls.b,
                                                            &tls.work,
                                                            &rng_tail,
                                                            &ls_failures);
                                                    }
                                                    const float e2_try = SampleCostFromXc(
                                                        *pre_tail, tls.xC_buf, tls.B_try, tls.a_try, re_norm2);
                                                    if (e2_try < e2_best) {
                                                        e2_best = e2_try;
                                                        tls.B_init = tls.B_try;
                                                        tls.a_init = tls.a_try;
                                                    }
                                                }
                                            }
                                        }

                                        const bool better =
                                            (e2_best < best_cost) ||
                                            (best_parent >= 0 && e2_best == best_cost && pos < best_pos);
                                        if (better) {
                                            best_cost = e2_best;
                                            best_parent = p_local;
                                            best_pos = pos;
                                            win_dyn_best_parent[static_cast<std::size_t>(i)] = best_parent;
                                            win_dyn_best_cost[static_cast<std::size_t>(i)] = best_cost;
                                            win_dyn_best_pos[static_cast<std::size_t>(i)] = best_pos;
                                            const std::size_t dst =
                                                static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
                                            std::memcpy(win_dyn_best_B.data() + dst,
                                                        tls.B_init.data(),
                                                        sizeof(FullCode) * static_cast<std::size_t>(m));
                                            std::memcpy(win_dyn_best_a.data() + dst,
                                                        tls.a_init.data(),
                                                        sizeof(float) * static_cast<std::size_t>(m));
                                        }
                                    }

                                    win_dyn_done[static_cast<std::size_t>(i)] = have;

                                    if (did_tiny_forced_skip) {
                                        if (stlq::linkage::LinkageBuildProfileEnabled()) {
                                            if (auto* prof = stlq::linkage::GetLinkageBuildProfileTls()) {
                                                prof->good_cpu_fallback_dyn_window_tiny_cpu += 1;
                                            }
                                        }
                                    }
                                }
                            }
                            {
                                const int p_d = win_dyn_best_parent[static_cast<std::size_t>(i)];
                                const float c_d = win_dyn_best_cost[static_cast<std::size_t>(i)];
                                const int pos_d = win_dyn_best_pos[static_cast<std::size_t>(i)];
                                if (p_d >= 0 &&
                                    (c_d < bestE2 ||
                                        (best_parent_local >= 0 && c_d == bestE2 && pos_d < best_pos))) {
                                    bestE2 = c_d;
                                    best_parent_local = p_d;
                                    best_pos = pos_d;
                                    tls.B_best.resize(static_cast<std::size_t>(m));
                                    tls.a_best.resize(static_cast<std::size_t>(m));
                                    const std::size_t src = static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
                                    std::memcpy(tls.B_best.data(),
                                                win_dyn_best_B.data() + src,
                                                sizeof(FullCode) * static_cast<std::size_t>(m));
                                    std::memcpy(tls.a_best.data(),
                                                win_dyn_best_a.data() + src,
                                                sizeof(float) * static_cast<std::size_t>(m));
                                    if (need_root_u32 && !win_dyn_best_code0.empty()) {
                                        best_code0 = win_dyn_best_code0[static_cast<std::size_t>(i)];
                                    }
                                    else if (!need_root_u32 && !tls.B_best.empty()) {
                                        best_code0 = static_cast<RootCode>(tls.B_best[0]);
                                    }
                                }
                            }

                            if (best_parent_local >= 0) {
                                for (int l = 0; l < m; ++l) {
                                    if (need_root_u32 && l == 0) {
                                        (*B)(0, local_idx) = static_cast<FullCode>(0);
                                    }
                                    else {
                                        (*B)(l, local_idx) = tls.B_best[static_cast<std::size_t>(l)];
                                    }
                                    (*a)(l, local_idx) = tls.a_best[static_cast<std::size_t>(l)];
                                }
                                if (need_root_u32) {
                                    (*code0_root_inout)[static_cast<std::size_t>(local_idx)] = best_code0;
                                }
                                parent_local[static_cast<std::size_t>(local_idx)] = best_parent_local + 1;
                                depth_local[static_cast<std::size_t>(local_idx)] =
                                    depth_local[static_cast<std::size_t>(best_parent_local)] + 1;

                                if (!need_root_u32) {
                                    ReconstructNode(C_root.books, tls.B_best, tls.a_best, &tls.R_self);
                                }
                                else {
                                    ReconstructNodeU32Root(best_code0, tls.B_best, tls.a_best, &tls.R_self);
                                }
                                const float* rp = R_full.Col(best_parent_local);
                                float* rdst = R_full.Col(local_idx);
                                const float* rself_best = tls.R_self.data();
#pragma omp simd
                                for (int r = 0; r < d; ++r) {
                                    rdst[r] = rp[r] + rself_best[r];
                                }

#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                                if (cuda_ctx && runtime_cfg && device_rfull_ready) {
                                    PushPendingRfullUpdate(local_idx, rdst);
                                }
#endif
                            }

                            // After committing this node (regardless of whether it linkaged), it becomes an eligible
                            // same-layer parent for later nodes, matching the sequential `processed[pos]=1` behavior.
                            // If the node did not linkage, its depth stays 0 and its reconstruction in R_full stays as-is.
                            const int beg = win_inv_offsets[static_cast<std::size_t>(i)];
                            const int end = win_inv_offsets[static_cast<std::size_t>(i + 1)];
                            const int p_depth = depth_local[static_cast<std::size_t>(local_idx)];
                            if (p_depth + 1 <= max_depth) {
                                for (int e = beg; e < end; ++e) {
                                    const int child_off = win_inv_child[static_cast<std::size_t>(e)];
                                    const int p_local = win_inv_parent_local[static_cast<std::size_t>(e)];
                                    const int p_pos = win_inv_pos[static_cast<std::size_t>(e)];
                                    int& have = win_dyn_len[static_cast<std::size_t>(child_off)];
                                    const std::size_t idx =
                                        static_cast<std::size_t>(child_off) * static_cast<std::size_t>(pending_stride) +
                                        static_cast<std::size_t>(have);
                                    win_dyn_list_parent[idx] = p_local;
                                    win_dyn_list_pos[idx] = p_pos;
                                    have += 1;
                                }
                            }
                            WinDynMaybePreflushFrom(i + 1);
                        }
                        (void)WinDynFinishAsyncIfAny();
                        if (!pending_rfull_cols.empty() && !FlushPendingRfullUpdates()) {
                            if (!can_cpu_fallback) {
                                throw std::runtime_error(
                                    "Init-linkage: failed to flush pending device R_full updates at end-of-block and CPU fallback is disabled.");
                            }
                        }

                        // Clear ranks for this block.
                        for (int i = block_start; i < block_end; ++i) {
                            const int pos_in_layer = tls.order_idx[static_cast<std::size_t>(i)];
                            const int local_idx = current_layer[static_cast<std::size_t>(pos_in_layer)];
                            batch_rank_by_local[static_cast<std::size_t>(local_idx)] = -1;
                        }

                        block_start = block_end;
                    }
                }
                else
#endif
                {
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                    // Inner GPU batch may be enqueued async; complete it before per-node processing.
                    if (!finish_inner_async()) {
                        layer_rel_flat.clear();
                        layer_kprime = 0;
                    }
#endif
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                    // -- Fallthrough-path R_full batch accumulator ------------------------------------
                    // Replaces per-commit CudaLinkageUpdateClusterRfullColumn with a deferred batch
                    // that is flushed only when a subsequent same-layer GPU eval has a dependency on
                    // a recently-committed column.  Reduces GPU syncs from 2N to ~N (eval) + 1 (bulk).
                    std::vector<int> fl_rfull_pending_cols;
                    std::vector<float> fl_rfull_pending_data;
                    std::vector<bool> fl_rfull_is_pending(static_cast<std::size_t>(n), false);
                    fl_rfull_pending_cols.reserve(256);
                    fl_rfull_pending_data.reserve(static_cast<std::size_t>(256) * static_cast<std::size_t>(d));
                    const auto fl_rfull_push = [&](int col, const float* r_col)
                    {
                        fl_rfull_pending_cols.push_back(col);
                        fl_rfull_is_pending[static_cast<std::size_t>(col)] = true;
                        const std::size_t fl_old = fl_rfull_pending_data.size();
                        fl_rfull_pending_data.resize(fl_old + static_cast<std::size_t>(d));
                        std::memcpy(fl_rfull_pending_data.data() + fl_old, r_col,
                                    sizeof(float) * static_cast<std::size_t>(d));
                    };
                    const auto fl_rfull_flush = [&]() -> bool
                    {
                        if (fl_rfull_pending_cols.empty()) return true;
                        if (!cuda_ctx || !runtime_cfg || !device_rfull_ready) {
                            for (int c : fl_rfull_pending_cols)
                                fl_rfull_is_pending[static_cast<std::size_t>(c)] =
                                    false;
                            fl_rfull_pending_cols.clear();
                            fl_rfull_pending_data.clear();
                            return false;
                        }
                        constexpr int kFlMax = 2048;
                        std::size_t fl_off = 0;
                        while (fl_off < fl_rfull_pending_cols.size()) {
                            const int fl_take = static_cast<int>(std::min<std::size_t>(
                                fl_rfull_pending_cols.size() - fl_off, static_cast<std::size_t>(kFlMax)));
                            std::string up_err;
                            if (!CudaLinkageUpdateClusterRfullColumnsBatch(
                                cuda_ctx,
                                fl_rfull_pending_cols.data() + fl_off,
                                fl_rfull_pending_data.data() + fl_off * static_cast<std::size_t>(d),
                                d, fl_take, &up_err)) {
                                if (!can_cpu_fallback) {
                                    throw std::runtime_error(
                                        std::string("CudaLinkageUpdateClusterRfullColumnsBatch failed "
                                            "(init-linkage-fallthrough) and CPU fallback disabled: ") + up_err);
                                }
                                LogWarn(std::string("CudaLinkageUpdateClusterRfullColumnsBatch failed "
                                    "(init-linkage-fallthrough), disabling device R_full: ") + up_err);
                                device_rfull_ready = false;
                                for (int c : fl_rfull_pending_cols)
                                    fl_rfull_is_pending[static_cast<std::size_t>(c)] =
                                        false;
                                fl_rfull_pending_cols.clear();
                                fl_rfull_pending_data.clear();
                                return false;
                            }
                            fl_off += static_cast<std::size_t>(fl_take);
                        }
                        for (int c : fl_rfull_pending_cols) fl_rfull_is_pending[static_cast<std::size_t>(c)] = false;
                        fl_rfull_pending_cols.clear();
                        fl_rfull_pending_data.clear();
                        return true;
                    };
                    // ---------------------------------------------------------------------------------
#endif
                    for (int pos_in_layer : tls.order_idx) {
                        const int local_idx = current_layer[static_cast<std::size_t>(pos_in_layer)];
                        const float* xi = X.Col(local_idx);

#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                        bool use_gpu_layer = (layer_kprime > 0 && !layer_rel_flat.empty());
                        if (use_gpu_layer) {
                            const std::size_t base =
                                static_cast<std::size_t>(pos_in_layer) * static_cast<std::size_t>(std::max(
                                    1, layer_kprime));
                            if (layer_kprime <= 0 || base + static_cast<std::size_t>(layer_kprime) > layer_rel_flat.
                                size()) {
                                use_gpu_layer = false;
                            }
                        }
                        if (use_gpu_layer) {
                            tls.candidates_inner.clear();
                            tls.candidates_same.clear();
                            const int* rel =
                                layer_rel_flat.data() + static_cast<std::size_t>(pos_in_layer) *
                                static_cast<std::size_t>(layer_kprime);
                            const int rel_count =
                                (static_cast<std::size_t>(pos_in_layer) < layer_rel_len.size())
                                    ? layer_rel_len[static_cast<std::size_t>(pos_in_layer)]
                                    : 0;
                            filter_split_from_rel(rel, rel_count, local_idx,
                                                  /*out_inner=*/nullptr,
                                                  &tls.candidates_same);
                        }
                        else
#endif
                        {
                            QueryCandidatesHnsw(hnsw_index,
                                                xi,
                                                cluster_indices,
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
                        }

                        tls.processed[static_cast<std::size_t>(pos_in_layer)] = 1;

#if defined(STLQ_ENABLE_CUDA)
                        if (use_gpu_layer && !device_rfull_ready) {
                            // GPU path is unavailable mid-layer (e.g. device_rfull disabled); fall back to CPU
                            // candidates for this node to avoid using stale tls.candidates.
                            QueryCandidatesHnsw(hnsw_index,
                                                xi,
                                                cluster_indices,
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
                        }
#endif

                        float bestE2 = base_mse[static_cast<std::size_t>(local_idx)];
                        int best_parent_local = -1;
                        RootCode best_code0 =
                            need_root_u32
                                ? (*code0_root_inout)[static_cast<std::size_t>(local_idx)]
                                : static_cast<RootCode>((*B)(0, local_idx));

#if defined(STLQ_ENABLE_CUDA)
                        if (use_gpu_layer && !inner_best_valid.empty() &&
                            inner_best_valid[static_cast<std::size_t>(local_idx)] != 0u) {
                            const int p_inner = inner_best_parent_local[static_cast<std::size_t>(local_idx)];
                            const float c_inner = inner_best_cost[static_cast<std::size_t>(local_idx)];
                            if (p_inner >= 0 && c_inner < bestE2) {
                                bestE2 = c_inner;
                                best_parent_local = p_inner;
                                tls.B_best.resize(static_cast<std::size_t>(m));
                                tls.a_best.resize(static_cast<std::size_t>(m));
                                std::memcpy(tls.B_best.data(),
                                            inner_best_B.data() +
                                            static_cast<std::size_t>(local_idx) * static_cast<std::size_t>(m),
                                            sizeof(FullCode) * static_cast<std::size_t>(m));
                                std::memcpy(tls.a_best.data(),
                                            inner_best_a.data() +
                                            static_cast<std::size_t>(local_idx) * static_cast<std::size_t>(m),
                                            sizeof(float) * static_cast<std::size_t>(m));
                                if (need_root_u32 && !inner_best_code0.empty()) {
                                    best_code0 = inner_best_code0[static_cast<std::size_t>(local_idx)];
                                }
                                else if (!need_root_u32 && !tls.B_best.empty()) {
                                    best_code0 = static_cast<RootCode>(tls.B_best[0]);
                                }
                            }
                        }
#endif

#if defined(STLQ_ENABLE_CUDA)
                        if (use_gpu_layer && cuda_ctx && runtime_cfg && device_rfull_ready) {
                            // Same-layer candidates must be evaluated after each commit (sequential),
                            // but can still use the GPU device-R_full gather path.
                            tls.candidates_same.erase(
                                std::remove_if(tls.candidates_same.begin(), tls.candidates_same.end(),
                                               [&](int p_local)
                                               {
                                                   const int new_depth = depth_local[static_cast<std::size_t>(p_local)]
                                                       + 1;
                                                   return new_depth > max_depth;
                                               }),
                                tls.candidates_same.end());

                            if (!tls.candidates_same.empty()) {
                                // Flush pending device R_full updates if any candidate's reconstruction
                                // was committed (but not yet pushed to device) in this layer pass.
                                for (int _fc : tls.candidates_same) {
                                    if (fl_rfull_is_pending[static_cast<std::size_t>(_fc)]) {
                                        (void)fl_rfull_flush();
                                        break;
                                    }
                                }
                                const int Kp = static_cast<int>(tls.candidates_same.size());
                                static thread_local std::vector<int> pair_node_single;
                                pair_node_single.assign(static_cast<std::size_t>(Kp), 0); // pair_node all 0 (B=1)
                                int offsets[2] = {0, Kp};
                                std::uint64_t node_base[1] = {
                                    (static_cast<std::uint64_t>(static_cast<std::uint32_t>(forced_root_code)) << 32) |
                                    static_cast<std::uint64_t>(static_cast<std::uint32_t>(local_idx))
                                };
                                int best_parent_gpu[1] = {-1};
                                float best_cost_gpu[1] = {std::numeric_limits<float>::infinity()};
                                tls.B_try.resize(static_cast<std::size_t>(m));
                                tls.a_try.resize(static_cast<std::size_t>(m));

                                std::string gpu_err;
                                bool ok_gpu = false;
                                RootCode code0_try = best_code0;
                                if (!need_root_u32) {
                                    ok_gpu =
                                        EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRoot(
                                            cuda_ctx, C_root,
                                            xi, d, /*B=*/1,
                                            tls.candidates_same.data(),
                                            pair_node_single.data(),
                                            offsets,
                                            Kp,
                                            icm_round,
                                            /*ils_iters=*/
                                            (UseIls && (m > 1 && ils_rounds > 0 && ils_k > 0)) ? ils_rounds : 0,
                                            /*perturb_k=*/
                                            (UseIls && (m > 1 && ils_rounds > 0 && ils_k > 0)) ? ils_k : 0,
                                            static_cast<std::uint32_t>(linkage_cfg.seed + 997 * forced_root_code),
                                            node_base,
                                            best_parent_gpu,
                                            best_cost_gpu,
                                            tls.B_try.data(),
                                            tls.a_try.data(),
                                            &gpu_err);
                                    if (ok_gpu && !tls.B_try.empty()) {
                                        code0_try = static_cast<RootCode>(tls.B_try[0]);
                                    }
                                }
                                else {
                                    static thread_local std::vector<std::uint32_t> best_B_u32;
                                    best_B_u32.resize(static_cast<std::size_t>(m));
                                    ok_gpu =
                                        EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootU32(
                                            cuda_ctx, C_root,
                                            xi, d, /*B=*/1,
                                            tls.candidates_same.data(),
                                            pair_node_single.data(),
                                            offsets,
                                            Kp,
                                            icm_round,
                                            /*ils_iters=*/
                                            (UseIls && (m > 1 && ils_rounds > 0 && ils_k > 0)) ? ils_rounds : 0,
                                            /*perturb_k=*/
                                            (UseIls && (m > 1 && ils_rounds > 0 && ils_k > 0)) ? ils_k : 0,
                                            static_cast<std::uint32_t>(linkage_cfg.seed + 997 * forced_root_code),
                                            node_base,
                                            best_parent_gpu,
                                            best_cost_gpu,
                                            best_B_u32.data(),
                                            tls.a_try.data(),
                                            &gpu_err);
                                    if (ok_gpu) {
                                        code0_try = static_cast<RootCode>(best_B_u32[0]);
                                        tls.B_try[0] = static_cast<FullCode>(0);
                                        for (int l = 1; l < m; ++l) {
                                            tls.B_try[static_cast<std::size_t>(l)] =
                                                static_cast<FullCode>(best_B_u32[static_cast<std::size_t>(l)]);
                                        }
                                    }
                                }
                                if (ok_gpu) {
                                    if (best_parent_gpu[0] >= 0 && best_cost_gpu[0] < bestE2) {
                                        bestE2 = best_cost_gpu[0];
                                        best_parent_local = best_parent_gpu[0];
                                        tls.B_best.assign(tls.B_try.begin(), tls.B_try.end());
                                        tls.a_best.assign(tls.a_try.begin(), tls.a_try.end());
                                        best_code0 = code0_try;
                                    }
                                }
                                else {
                                    if (!can_cpu_fallback) {
                                        throw std::runtime_error(
                                            std::string(
                                                "CUDA init-linkage same-layer eval failed and CPU fallback disabled: ")
                                            + gpu_err);
                                    }
                                    LogWarn(
                                        std::string("CUDA init-linkage same-layer eval failed, fallback to CPU: ") +
                                        gpu_err);
                                    device_rfull_ready = false;
                                    QueryCandidatesHnsw(hnsw_index,
                                                        xi,
                                                        cluster_indices,
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
                                }
                            }
                        }
#endif

                        // Init-linkage ICM/ILS policy: allow perturbing any layer (including layer0).
                        const int ils_k_eff = ils_k;
                        const bool do_ils = (m > 1 && ils_rounds > 0 && ils_k_eff > 0);
                        std::uniform_int_distribution<int> layer_dist(0, std::max(0, m - 1));
                        std::vector<std::uniform_int_distribution<int>> code_dists;
                        if constexpr (UseIls) {
                            if (do_ils) {
                                code_dists.reserve(static_cast<std::size_t>(m));
                                for (int layer = 0; layer < m; ++layer) {
                                    const int h = pre_root.h_vec[layer];
                                    code_dists.emplace_back(0, std::max(0, h - 2));
                                }
                            }
                        }

#if defined(STLQ_ENABLE_CUDA)
                        if (use_gpu_layer && device_rfull_ready) {
                            // Candidate evaluation already done on GPU (inner batch + per-node same-layer).
                        }
                        else
#endif
                        {
                            if (!tls.candidates.empty()) {
                                ComputeXcBase(pre_root, xi, &tls.xC_base_buf);
                            }
                            for (int p_local : tls.candidates) {
                                const int new_depth = depth_local[static_cast<std::size_t>(p_local)] + 1;
                                if (new_depth > max_depth) {
                                    continue;
                                }

                                const float* rp = R_full.Col(p_local);
                                float re_norm2 = 0.0f;
#pragma omp simd reduction(+:re_norm2)
                                for (int r = 0; r < d; ++r) {
                                    const float v = xi[r] - rp[r];
                                    tls.re[static_cast<std::size_t>(r)] = v;
                                    re_norm2 += v * v;
                                }

                                BuildXcForParentSingleLocal(pre_root,
                                                            tls.xC_base_buf,
                                                            *B,
                                                            *a,
                                                            cluster_indices,
                                                            parent_local,
                                                            p_local,
                                                            &tls.xC_buf);

                                QuantizeWithKmeansAbsNoNormalSingle(pre_root, tls.re,
                                                                    &tls.B_init,
                                                                    &tls.a_init,
                                                                    &tls.rC_buf,
                                                                    &tls.available);

                                if (!SolveLeastSquaresVector(pre_root, tls.xC_buf, tls.B_init,
                                                             &tls.A, &tls.b, &tls.work,
                                                             tls.a_init.data())) {
                                    ++ls_failures;
                                    continue;
                                }
                                if (icm_round > 0) {
                                    DynamicIcmEncodingSingleForResidual(pre_root, tls.re, tls.xC_buf,
                                                                        icm_round,
                                                                        &tls.B_init, &tls.a_init,
                                                                        &tls.a_bak, &tls.rC_buf,
                                                                        &tls.A, &tls.b, &tls.work,
                                                                        &rng, &ls_failures);
                                }

                                float E2_best =
                                    SampleCostFromXc(pre_root, tls.xC_buf, tls.B_init, tls.a_init, re_norm2);

                                if constexpr (UseIls) {
                                    if (do_ils) {
                                        for (int il = 0; il < ils_rounds; ++il) {
                                            for (int t = 0; t < ils_k_eff; ++t) {
                                                const int layer = layer_dist(rng);
                                                const int h = pre_root.h_vec[layer];
                                                const int old_code = static_cast<int>(tls.B_init[static_cast<
                                                    std::size_t>(layer)]);
                                                int new_code = old_code;
                                                if (h > 1) {
                                                    int raw = code_dists[static_cast<std::size_t>(layer)](rng);
                                                    new_code = raw;
                                                    if (new_code >= old_code) {
                                                        new_code += 1;
                                                    }
                                                }
                                                const int startf = pre_root.offsets[layer];
                                                tls.pert_layers[static_cast<std::size_t>(t)] = layer;
                                                tls.new_flat[static_cast<std::size_t>(t)] = startf + new_code;
                                            }

                                            tls.B_try = tls.B_init;
                                            for (int t = 0; t < ils_k_eff; ++t) {
                                                const int layer = tls.pert_layers[static_cast<std::size_t>(t)];
                                                const int flat = tls.new_flat[static_cast<std::size_t>(t)];
                                                const int code = flat - pre_root.offsets[layer];
                                                tls.B_try[static_cast<std::size_t>(layer)] = static_cast<FullCode>(
                                                    code);
                                            }
                                            tls.a_try = tls.a_init;
                                            if (!SolveLeastSquaresVector(pre_root, tls.xC_buf, tls.B_try,
                                                                         &tls.A, &tls.b, &tls.work,
                                                                         tls.a_try.data())) {
                                                ++ls_failures;
                                                continue;
                                            }
                                            if (icm_round > 0) {
                                                DynamicIcmEncodingSingleForResidual(pre_root, tls.re, tls.xC_buf,
                                                    icm_round,
                                                    &tls.B_try, &tls.a_try,
                                                    &tls.a_bak, &tls.rC_buf,
                                                    &tls.A, &tls.b, &tls.work,
                                                    &rng, &ls_failures);
                                            }

                                            float E2_try = SampleCostFromXc(
                                                pre_root, tls.xC_buf, tls.B_try, tls.a_try, re_norm2);
                                            if (E2_try < E2_best) {
                                                E2_best = E2_try;
                                                tls.B_init = tls.B_try;
                                                tls.a_init = tls.a_try;
                                            }
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
                        }

                        if (best_parent_local >= 0) {
                            for (int l = 0; l < m; ++l) {
                                if (need_root_u32 && l == 0) {
                                    (*B)(0, local_idx) = static_cast<FullCode>(0);
                                }
                                else {
                                    (*B)(l, local_idx) = tls.B_best[static_cast<std::size_t>(l)];
                                }
                                (*a)(l, local_idx) = tls.a_best[static_cast<std::size_t>(l)];
                            }
                            if (need_root_u32) {
                                (*code0_root_inout)[static_cast<std::size_t>(local_idx)] = best_code0;
                            }
                            parent_local[static_cast<std::size_t>(local_idx)] = best_parent_local + 1;
                            depth_local[static_cast<std::size_t>(local_idx)] =
                                depth_local[static_cast<std::size_t>(best_parent_local)] + 1;

                            if (!need_root_u32) {
                                ReconstructNode(C_root.books, tls.B_best, tls.a_best, &tls.R_self);
                            }
                            else {
                                ReconstructNodeU32Root(best_code0, tls.B_best, tls.a_best, &tls.R_self);
                            }
                            const float* rp = R_full.Col(best_parent_local);
                            float* rdst = R_full.Col(local_idx);
                            const float* rself_best = tls.R_self.data();
#pragma omp simd
                            for (int r = 0; r < d; ++r) {
                                rdst[r] = rp[r] + rself_best[r];
                            }

#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                            // Defer device R_full update to fl_rfull_flush(), which runs just before any
                            // same-layer GPU eval that reads this column.  Eliminates per-commit GPU sync.
                            if (cuda_ctx && runtime_cfg && device_rfull_ready) {
                                fl_rfull_push(local_idx, rdst);
                            }
#endif
                        }
                    }
                    // Flush any remaining pending device R_full updates at the end of this layer.
#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
                    (void)fl_rfull_flush();
#endif
                }
            }

            out_cluster->parent_local = std::move(parent_local);
            FinalizeLinkageClusterDepthOrderInPlaceFromDepth(out_cluster, depth_local.data());

            if (linkaged_mse_out || linkaged_mse_sum_out) {
                const MseSumStats mse = MseSumStatsFromReconLocal(X, R_full);
                if (linkaged_mse_out) {
                    MseStats s;
                    if (n > 0 && std::isfinite(mse.min)) {
                        s.mean = static_cast<float>(mse.sum / static_cast<double>(n));
                        s.min = static_cast<float>(mse.min);
                        s.max = static_cast<float>(mse.max);
                    }
                    *linkaged_mse_out = s;
                }
                if (linkaged_mse_sum_out) {
                    *linkaged_mse_sum_out = mse.sum;
                }
            }
            if (ls_failures_out) {
                *ls_failures_out = ls_failures;
            }
        }
