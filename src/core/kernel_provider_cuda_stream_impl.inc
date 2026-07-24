namespace stlq {

struct CudaStreamKernels::Impl {
    int device = 0;
    bool allow_tf32 = false;
    cudaStream_t stream = nullptr;
    cudaStream_t stream_copy = nullptr;
    cublasHandle_t handle = nullptr;

    // cuBLAS workspace can materially affect GEMM algorithm selection/perf.
    // Keep this reasonably sized for RTX 4090-class GPUs; if allocation fails we fall back.
    DeviceBuffer cublas_workspace;

    // Cache common matrices on device (codebooks, rotations).
    CachedDeviceMatF32 A_cache;
    CachedDeviceMatF32 B_cache;
    CachedDeviceMatF32 R_cache;
    // Dedicated cache for large, frequently reused matrices (e.g. k-means X).
    CachedDeviceMatF32 X_cache;

    // Reused per-call buffers.
    DeviceBuffer dB;
    DeviceBuffer du8;
    DeviceBuffer dtmp;
    DeviceBuffer d_assign;
    DeviceBuffer d_best;

    // K-means streaming scratch buffers.
    DeviceBuffer kmeans_xnorm;
    int kmeans_xnorm_rows = 0;
    int kmeans_xnorm_cols = 0;
    int kmeans_xnorm_ld = 0;
    DeviceBuffer kmeans_norm2;  // float[cols] for current (non-slot) normalized matrix
    HostPinnedBuffer kmeans_hpin;
    // K-means streaming ping-pong buffers for overlap (2-slot).
    DeviceBuffer kmeans_xnorm_slot[2];
    int kmeans_slot_rows[2] = {0, 0};
    int kmeans_slot_cols[2] = {0, 0};
    int kmeans_slot_ld[2] = {0, 0};
    DeviceBuffer kmeans_norm2_slot[2];    // float[cols] original ||x||^2 per slot
    DeviceBuffer rvq_hatnorm2_slot[2];    // float[cols] residual_hat ||r_hat||^2 after prior projections
    HostPinnedBuffer kmeans_hpin_slot[2];
    DeviceBuffer kmeans_u8_slot[2];
    int kmeans_u8_slot_rows[2] = {0, 0};
    int kmeans_u8_slot_cols[2] = {0, 0};
    int kmeans_u8_slot_ld[2] = {0, 0};
    HostPinnedBuffer kmeans_hpin_u8_slot[2];

    // kmeans|| seeding scratch (GPU-resident init_samples).
    DeviceBuffer kmeansll_sample;  // float32[d * sample_cols] column-major
    int kmeansll_sample_rows = 0;
    int kmeansll_sample_cols = 0;
    int kmeansll_sample_ld = 0;

    DeviceBuffer kmeansll_best_dot;  // float32[sample_cols]
    int kmeansll_best_dot_n = 0;
    DeviceBuffer kmeansll_d2;        // float32[sample_cols]
    DeviceBuffer kmeansll_flags;     // uint8[sample_cols]
    DeviceBuffer kmeansll_indices;   // int32[sample_cols] (iota 0..n-1)
    int kmeansll_indices_n = 0;
    DeviceBuffer kmeansll_selected;      // int32[sample_cols]
    DeviceBuffer kmeansll_num_selected;  // int32[1]
    DeviceBuffer kmeansll_phi;           // float32[1]
    int kmeansll_last_selected = 0;
    HostPinnedBuffer kmeansll_selected_hpin;

    DeviceBuffer kmeansll_idxbuf;   // int32[m] staging for gather/finalize
    int kmeansll_idxbuf_cap = 0;
    HostPinnedBuffer kmeansll_idxbuf_hpin;

    DeviceBuffer kmeansll_gather;  // float32[d * m] gather output
    int kmeansll_gather_rows = 0;
    int kmeansll_gather_cols = 0;
    int kmeansll_gather_ld = 0;

    DeviceBuffer kmeansll_cub_tmp;

    // RVQ init residual reconstruction scratch (Prompt-06).
    DeviceBuffer rvq_codebooks;  // concatenated float32 codebooks for layers [0,rvq_upto_layer)
    int rvq_codebooks_rows = 0;
    int rvq_upto_layer = 0;
    std::vector<std::size_t> rvq_cb_offsets_f;  // float offsets into rvq_codebooks
    std::vector<int> rvq_cb_K;                  // K per prior layer
    DeviceBuffer rvq_meta;                      // int[4 * upto_layer]: cb_offsets_f, cb_K, code_offsets_bytes, code_bytes

    // Codes for the current block (per upload slot). Stored as a packed byte buffer with fixed
    // capacity = rvq_codes_capacity_cols[slot]. Layer offsets are in bytes.
    DeviceBuffer rvq_codes_slot[2];
    int rvq_codes_capacity_cols[2] = {0, 0};
    int rvq_codes_offsets_capacity_cols = 0;
    int rvq_codes_upto_layer = 0;
    std::vector<int> rvq_code_bytes_per_code;
    std::vector<std::size_t> rvq_code_offsets_bytes;

    // copy->compute readiness events for slot ping-pong (disable timing).
    cudaEvent_t kmeans_ready_ev[2] = {nullptr, nullptr};
    // compute->copy "slot free" events for slot ping-pong (disable timing).
    cudaEvent_t kmeans_free_ev[2] = {nullptr, nullptr};
    bool kmeans_ready_inited = false;
    // Optional pinned host mirror for the full X (float32). This avoids repeated pageable staging costs
    // during streaming k-means when X cannot be cached on device.
    HostPinnedBuffer kmeans_X_pinned;
    const float* kmeans_X_pinned_host_ptr = nullptr;
    int kmeans_X_pinned_rows = 0;
    int kmeans_X_pinned_cols = 0;
    std::uint64_t kmeans_X_pinned_stamp = 0;
    // Optional full X_norm cache (device) for k-means assignment/update.
    DeviceBuffer kmeans_xnorm_full;
    const float* kmeans_xnorm_full_host_ptr = nullptr;
    int kmeans_xnorm_full_rows = 0;
    int kmeans_xnorm_full_cols = 0;
    std::uint64_t kmeans_xnorm_full_stamp = 0;

    // Optional partial X_norm cache (device) for streaming k-means (prefix window).
    DeviceBuffer kmeans_xnorm_partial;
    int kmeans_xnorm_partial_rows = 0;
    int kmeans_xnorm_partial_cols = 0;
    int kmeans_xnorm_partial_ld = 0;
    DeviceBuffer kmeans_norm2_partial;  // float[partial_cols] base ||x||^2
    int kmeans_norm2_partial_cols = 0;

    DeviceBuffer kmeans_sum_x;
    DeviceBuffer kmeans_sum_w;
    int kmeans_sum_rows = 0;
    int kmeans_sum_k = 0;
    int kmeans_sum_ld = 0;

    DeviceBuffer kmeans_assign;
    int kmeans_assign_cols = 0;
    DeviceBuffer kmeans_weights;
    int kmeans_weights_cols = 0;

    // K-means cost histogram (uint32 bins).
    DeviceBuffer kmeans_hist_u32;
    int kmeans_hist_bins = 0;
    HostPinnedBuffer kmeans_hist_hpin;
    cudaEvent_t kmeans_hist_ready_ev = nullptr;  // recorded on compute stream
    cudaEvent_t kmeans_hist_done_ev = nullptr;   // recorded on copy stream
    bool kmeans_hist_events_ready = false;
    std::uint32_t* kmeans_hist_host_dst = nullptr;
    int kmeans_hist_host_bins = 0;
    bool kmeans_hist_pending = false;

    // K-means global anneal weights (device-resident) and scalar totals.
    DeviceBuffer kmeans_weights_all;
    std::int64_t kmeans_weights_all_n = 0;
    DeviceBuffer kmeans_anneal_totals;       // double[2]: (total_w, total_cost)
    HostPinnedBuffer kmeans_anneal_totals_hpin;
    cudaEvent_t kmeans_anneal_ready_ev = nullptr;  // recorded on compute stream
    cudaEvent_t kmeans_anneal_done_ev = nullptr;   // recorded on copy stream
    bool kmeans_anneal_events_ready = false;
    bool kmeans_anneal_pending = false;

    // K-means metric totals (device-only bestdot metrics).
    DeviceBuffer kmeans_metric_totals;       // double[2]: (mse_sum, mse_count)
    HostPinnedBuffer kmeans_metric_totals_hpin;
    cudaEvent_t kmeans_metric_ready_ev = nullptr;  // recorded on compute stream
    cudaEvent_t kmeans_metric_done_ev = nullptr;   // recorded on copy stream
    bool kmeans_metric_events_ready = false;
    bool kmeans_metric_pending = false;

    int kmeans_active_slot = 0;
    bool kmeans_collect_metrics = true;
    bool kmeans_timing_enabled = false;
    double kmeans_timing_upload_h2d_s = 0.0;
    double kmeans_timing_upload_kernel_s = 0.0;
    double kmeans_timing_rvq_project_s = 0.0;
    double kmeans_timing_assign_update_compute_s = 0.0;
    double kmeans_timing_emit_codes_d2h_s = 0.0;
    double kmeans_timing_wait_copy_s = 0.0;     // copy stream wait for slot free
    double kmeans_timing_wait_compute_s = 0.0;  // compute stream wait for slot ready

    cudaEvent_t kmeans_upload_ev0[2]{};
    cudaEvent_t kmeans_upload_ev1[2]{};
    cudaEvent_t kmeans_upload_ev2[2]{};
    bool kmeans_upload_pending[2]{false, false};

    cudaEvent_t kmeans_rvq_ev0[2]{};
    cudaEvent_t kmeans_rvq_ev1[2]{};
    bool kmeans_rvq_pending[2]{false, false};

    cudaEvent_t kmeans_compute_ev0[2]{};
    cudaEvent_t kmeans_compute_ev1[2]{};
    bool kmeans_compute_pending[2]{false, false};
    bool kmeans_compute_started[2]{false, false};

    cudaEvent_t kmeans_assign_d2h_ev0[2]{};
    bool kmeans_assign_d2h_pending[2]{false, false};

    cudaEvent_t kmeans_wait_copy_ev0[2]{};
    cudaEvent_t kmeans_wait_copy_ev1[2]{};
    bool kmeans_wait_copy_pending[2]{false, false};

    cudaEvent_t kmeans_wait_compute_ev0[2]{};
    cudaEvent_t kmeans_wait_compute_ev1[2]{};
    bool kmeans_wait_compute_pending[2]{false, false};

    void EnsureKmeansTimingEvents() {
        for (int s = 0; s < 2; ++s) {
            if (!kmeans_upload_ev0[s]) {
                ThrowIf(cudaEventCreate(&kmeans_upload_ev0[s]), "cudaEventCreate (kmeans_upload_ev0)");
                ThrowIf(cudaEventCreate(&kmeans_upload_ev1[s]), "cudaEventCreate (kmeans_upload_ev1)");
                ThrowIf(cudaEventCreate(&kmeans_upload_ev2[s]), "cudaEventCreate (kmeans_upload_ev2)");
            }
            if (!kmeans_rvq_ev0[s]) {
                ThrowIf(cudaEventCreate(&kmeans_rvq_ev0[s]), "cudaEventCreate (kmeans_rvq_ev0)");
                ThrowIf(cudaEventCreate(&kmeans_rvq_ev1[s]), "cudaEventCreate (kmeans_rvq_ev1)");
            }
            if (!kmeans_compute_ev0[s]) {
                ThrowIf(cudaEventCreate(&kmeans_compute_ev0[s]), "cudaEventCreate (kmeans_compute_ev0)");
                ThrowIf(cudaEventCreate(&kmeans_compute_ev1[s]), "cudaEventCreate (kmeans_compute_ev1)");
            }
            if (!kmeans_assign_d2h_ev0[s]) {
                ThrowIf(cudaEventCreate(&kmeans_assign_d2h_ev0[s]), "cudaEventCreate (kmeans_assign_d2h_ev0)");
            }
            if (!kmeans_wait_copy_ev0[s]) {
                ThrowIf(cudaEventCreate(&kmeans_wait_copy_ev0[s]), "cudaEventCreate (kmeans_wait_copy_ev0)");
                ThrowIf(cudaEventCreate(&kmeans_wait_copy_ev1[s]), "cudaEventCreate (kmeans_wait_copy_ev1)");
            }
            if (!kmeans_wait_compute_ev0[s]) {
                ThrowIf(cudaEventCreate(&kmeans_wait_compute_ev0[s]), "cudaEventCreate (kmeans_wait_compute_ev0)");
                ThrowIf(cudaEventCreate(&kmeans_wait_compute_ev1[s]), "cudaEventCreate (kmeans_wait_compute_ev1)");
            }
        }
    }

    void HarvestKmeansUploadTimingSlot(int slot, bool force_sync) {
        if (!kmeans_upload_pending[slot]) return;
        const cudaError_t q = force_sync ? cudaEventSynchronize(kmeans_upload_ev2[slot])
                                         : cudaEventQuery(kmeans_upload_ev2[slot]);
        if (!force_sync && q == cudaErrorNotReady) return;
        ThrowIf(q, "HarvestKmeansUploadTimingSlot: upload ev2");
        float ms_h2d = 0.0f;
        float ms_ker = 0.0f;
        ThrowIf(cudaEventElapsedTime(&ms_h2d, kmeans_upload_ev0[slot], kmeans_upload_ev1[slot]),
                "cudaEventElapsedTime (kmeans upload h2d)");
        ThrowIf(cudaEventElapsedTime(&ms_ker, kmeans_upload_ev1[slot], kmeans_upload_ev2[slot]),
                "cudaEventElapsedTime (kmeans upload kernel)");
        kmeans_timing_upload_h2d_s += static_cast<double>(ms_h2d) * 1e-3;
        kmeans_timing_upload_kernel_s += static_cast<double>(ms_ker) * 1e-3;
        kmeans_upload_pending[slot] = false;
    }

    void HarvestKmeansRvqTimingSlot(int slot, bool force_sync) {
        if (!kmeans_rvq_pending[slot]) return;
        const cudaError_t q = force_sync ? cudaEventSynchronize(kmeans_rvq_ev1[slot])
                                         : cudaEventQuery(kmeans_rvq_ev1[slot]);
        if (!force_sync && q == cudaErrorNotReady) return;
        ThrowIf(q, "HarvestKmeansRvqTimingSlot: rvq ev1");
        float ms = 0.0f;
        ThrowIf(cudaEventElapsedTime(&ms, kmeans_rvq_ev0[slot], kmeans_rvq_ev1[slot]),
                "cudaEventElapsedTime (kmeans rvq project)");
        kmeans_timing_rvq_project_s += static_cast<double>(ms) * 1e-3;
        kmeans_rvq_pending[slot] = false;
    }

    void HarvestKmeansComputeTimingSlot(int slot, bool force_sync) {
        if (!kmeans_compute_pending[slot]) return;
        const cudaError_t q = force_sync ? cudaEventSynchronize(kmeans_compute_ev1[slot])
                                         : cudaEventQuery(kmeans_compute_ev1[slot]);
        if (!force_sync && q == cudaErrorNotReady) return;
        ThrowIf(q, "HarvestKmeansComputeTimingSlot: compute ev1");
        float ms = 0.0f;
        ThrowIf(cudaEventElapsedTime(&ms, kmeans_compute_ev0[slot], kmeans_compute_ev1[slot]),
                "cudaEventElapsedTime (kmeans compute section)");
        kmeans_timing_assign_update_compute_s += static_cast<double>(ms) * 1e-3;
        kmeans_compute_pending[slot] = false;
        kmeans_compute_started[slot] = false;
    }

    void HarvestKmeansWaitCopyTimingSlot(int slot, bool force_sync) {
        if (!kmeans_wait_copy_pending[slot]) return;
        const cudaError_t q = force_sync ? cudaEventSynchronize(kmeans_wait_copy_ev1[slot])
                                         : cudaEventQuery(kmeans_wait_copy_ev1[slot]);
        if (!force_sync && q == cudaErrorNotReady) return;
        ThrowIf(q, "HarvestKmeansWaitCopyTimingSlot: wait_copy ev1");
        float ms = 0.0f;
        ThrowIf(cudaEventElapsedTime(&ms, kmeans_wait_copy_ev0[slot], kmeans_wait_copy_ev1[slot]),
                "cudaEventElapsedTime (kmeans wait copy)");
        kmeans_timing_wait_copy_s += static_cast<double>(ms) * 1e-3;
        kmeans_wait_copy_pending[slot] = false;
    }

    void HarvestKmeansWaitComputeTimingSlot(int slot, bool force_sync) {
        if (!kmeans_wait_compute_pending[slot]) return;
        const cudaError_t q = force_sync ? cudaEventSynchronize(kmeans_wait_compute_ev1[slot])
                                         : cudaEventQuery(kmeans_wait_compute_ev1[slot]);
        if (!force_sync && q == cudaErrorNotReady) return;
        ThrowIf(q, "HarvestKmeansWaitComputeTimingSlot: wait_compute ev1");
        float ms = 0.0f;
        ThrowIf(cudaEventElapsedTime(&ms, kmeans_wait_compute_ev0[slot], kmeans_wait_compute_ev1[slot]),
                "cudaEventElapsedTime (kmeans wait compute)");
        kmeans_timing_wait_compute_s += static_cast<double>(ms) * 1e-3;
        kmeans_wait_compute_pending[slot] = false;
    }

    void HarvestKmeansAssignD2HTimingSlot(int slot) {
        if (!kmeans_assign_d2h_pending[slot]) return;
        if (!kmeans_blk_assign_pending[slot]) {
            kmeans_assign_d2h_pending[slot] = false;
            return;
        }
        // Only safe to compute once the done event has been synchronized by Sync().
        float ms = 0.0f;
        ThrowIf(cudaEventElapsedTime(&ms, kmeans_assign_d2h_ev0[slot], kmeans_blk_assign_done_ev_timing[slot]),
                "cudaEventElapsedTime (kmeans assign D2H)");
        kmeans_timing_emit_codes_d2h_s += static_cast<double>(ms) * 1e-3;
        kmeans_assign_d2h_pending[slot] = false;
    }

    // K-means outlier mask (device->host fallback when global weights are unavailable).
    DeviceBuffer kmeans_outlier_mask;        // uint8[cols]
    int kmeans_outlier_mask_cols = 0;
    HostPinnedBuffer kmeans_outlier_mask_hpin;
    cudaEvent_t kmeans_mask_ready_ev = nullptr;  // recorded on compute stream
    cudaEvent_t kmeans_mask_done_ev = nullptr;   // recorded on copy stream
    bool kmeans_mask_events_ready = false;
    bool kmeans_mask_pending = false;
    int kmeans_mask_bytes = 0;

    // K-means segmented-reduce accumulation scratch (no atomics).
    DeviceBuffer kmeans_sort_keys_in;
    DeviceBuffer kmeans_sort_vals_in;
    DeviceBuffer kmeans_sort_keys_out;
    DeviceBuffer kmeans_sort_vals_out;
    DeviceBuffer kmeans_unique_keys;       // int32[k]
    DeviceBuffer kmeans_counts;            // int32[k]
    DeviceBuffer kmeans_num_segments;      // int32[1]
    DeviceBuffer kmeans_seg_offsets;       // int32[k]
    DeviceBuffer kmeans_cub_tmp;
    int kmeans_sort_capacity_cols = 0;
    int kmeans_segment_capacity_k = 0;

    // K-means hier2 assignment scratch buffers.
    DeviceBuffer kmeans_hier2_topk_idx;
    DeviceBuffer kmeans_hier2_topk_val;

    // Per-block argmax outputs for k-means streaming (device). This avoids per-tile D2H.
    DeviceBuffer kmeans_blk_assign;
    DeviceBuffer kmeans_blk_best;
    HostPinnedBuffer kmeans_blk_assign_hpin;
    HostPinnedBuffer kmeans_blk_best_hpin;
    int kmeans_blk_cols = 0;

    // Async per-block assignments export (Prompt-10): stage to a per-slot device buffer and copy on stream_copy.
    DeviceBuffer kmeans_blk_assign_stage[2];
    HostPinnedBuffer kmeans_blk_assign_hpin_slot[2];
    cudaEvent_t kmeans_blk_assign_ready_ev[2] = {nullptr, nullptr};  // recorded on compute stream
    cudaEvent_t kmeans_blk_assign_done_ev[2] = {nullptr, nullptr};   // recorded on copy stream
    cudaEvent_t kmeans_blk_assign_done_ev_timing[2] = {nullptr, nullptr};  // recorded on copy stream (timing-enabled)
    bool kmeans_blk_assign_events_ready = false;
    bool kmeans_blk_assign_pending[2] = {false, false};
    int kmeans_blk_assign_slot_cols[2] = {0, 0};

    // Optional profiling events (created lazily).
    cudaEvent_t ev0 = nullptr;
    cudaEvent_t ev1 = nullptr;
    cudaEvent_t ev2 = nullptr;
    bool events_ready = false;

    // Cache the most-recent output matrix on device, keyed by the host matrix pointer.
    // This lets us avoid re-uploading intermediates like Xblk when we compute xC right after rotation.
    DeviceBuffer d_cached;
    const float* cached_host_ptr = nullptr;
    int cached_rows = 0;
    int cached_cols = 0;
    std::uint64_t cached_host_stamp = 0;
    DeviceBuffer d_out;

    void EnsureKmeansSegmentScratch(int cols, int k) {
        cols = std::max(1, cols);
        k = std::max(1, k);
        if (cols > kmeans_sort_capacity_cols) {
            kmeans_sort_keys_in.Ensure(static_cast<std::size_t>(cols) * sizeof(int));
            kmeans_sort_vals_in.Ensure(static_cast<std::size_t>(cols) * sizeof(int));
            kmeans_sort_keys_out.Ensure(static_cast<std::size_t>(cols) * sizeof(int));
            kmeans_sort_vals_out.Ensure(static_cast<std::size_t>(cols) * sizeof(int));
            kmeans_sort_capacity_cols = cols;
        }
        if (k > kmeans_segment_capacity_k) {
            kmeans_unique_keys.Ensure(static_cast<std::size_t>(k) * sizeof(int));
            kmeans_counts.Ensure(static_cast<std::size_t>(k) * sizeof(int));
            kmeans_seg_offsets.Ensure(static_cast<std::size_t>(k) * sizeof(int));
            kmeans_segment_capacity_k = k;
        }
        kmeans_num_segments.Ensure(sizeof(int));
    }

    void GemmF32(cublasOperation_t opA,
                cublasOperation_t opB,
                int m,
                int n,
                int k,
                const float* alpha,
                const float* A,
                int lda,
                const float* B,
                int ldb,
                const float* beta,
                float* C,
                int ldc,
                const char* what) {
        if (allow_tf32) {
            ThrowIf(cublasGemmEx(handle,
                                 opA,
                                 opB,
                                 m,
                                 n,
                                 k,
                                 alpha,
                                 A,
                                 CUDA_R_32F,
                                 lda,
                                 B,
                                 CUDA_R_32F,
                                 ldb,
                                 beta,
                                 C,
                                 CUDA_R_32F,
                                 ldc,
                                 CUBLAS_COMPUTE_32F_FAST_TF32,
                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                    what);
        } else {
            ThrowIf(cublasSgemm(handle,
                                opA,
                                opB,
                                m,
                                n,
                                k,
                                alpha,
                                A,
                                lda,
                                B,
                                ldb,
                                beta,
                                C,
                                ldc),
                    what);
        }
    }

    explicit Impl(int dev, bool tf32, std::size_t cublas_workspace_mb)
        : device(dev), allow_tf32(tf32) {
        ThrowIf(cudaSetDevice(device), "cudaSetDevice");
        ThrowIf(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
        ThrowIf(cudaStreamCreateWithFlags(&stream_copy, cudaStreamNonBlocking), "cudaStreamCreate (copy)");
        ThrowIf(cublasCreate(&handle), "cublasCreate");
        ThrowIf(cublasSetStream(handle, stream), "cublasSetStream");
        if (allow_tf32) {
            ThrowIf(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH), "cublasSetMathMode(TF32)");
        } else {
            ThrowIf(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH), "cublasSetMathMode(DEFAULT)");
        }
        // Reproducibility vs throughput:
        // - TF32 on: prefer speed; allow atomic-based GEMM algorithms.
        // - TF32 off ("strict" default): prefer determinism; disallow atomic-based GEMM algorithms.
        ThrowIf(cublasSetAtomicsMode(handle,
                                     allow_tf32 ? CUBLAS_ATOMICS_ALLOWED : CUBLAS_ATOMICS_NOT_ALLOWED),
                "cublasSetAtomicsMode");

        // Provide an explicit workspace to stabilize GEMM algorithm selection/perf.
        // This is intentionally a fixed size (configured by the caller) to avoid run-to-run
        // variability due to allocator fragmentation or changing fallback heuristics.
        if (cublas_workspace_mb > 0) {
            try {
                const std::size_t bytes = cublas_workspace_mb * 1024ull * 1024ull;
                cublas_workspace.Ensure(bytes);
                ThrowIf(cublasSetWorkspace(handle, cublas_workspace.ptr, cublas_workspace.bytes),
                        "cublasSetWorkspace");
            } catch (...) {
                // If allocation fails, continue without an explicit workspace.
                cublas_workspace.Reset();
            }
        }
    }

    ~Impl() {
        if (kmeans_ready_ev[0]) {
            cudaEventDestroy(kmeans_ready_ev[0]);
            kmeans_ready_ev[0] = nullptr;
        }
        if (kmeans_ready_ev[1]) {
            cudaEventDestroy(kmeans_ready_ev[1]);
            kmeans_ready_ev[1] = nullptr;
        }
        if (kmeans_free_ev[0]) {
            cudaEventDestroy(kmeans_free_ev[0]);
            kmeans_free_ev[0] = nullptr;
        }
        if (kmeans_free_ev[1]) {
            cudaEventDestroy(kmeans_free_ev[1]);
            kmeans_free_ev[1] = nullptr;
        }
        kmeans_ready_inited = false;
        if (kmeans_hist_ready_ev) {
            cudaEventDestroy(kmeans_hist_ready_ev);
            kmeans_hist_ready_ev = nullptr;
        }
        if (kmeans_hist_done_ev) {
            cudaEventDestroy(kmeans_hist_done_ev);
            kmeans_hist_done_ev = nullptr;
        }
        kmeans_hist_events_ready = false;
        if (kmeans_anneal_ready_ev) {
            cudaEventDestroy(kmeans_anneal_ready_ev);
            kmeans_anneal_ready_ev = nullptr;
        }
        if (kmeans_anneal_done_ev) {
            cudaEventDestroy(kmeans_anneal_done_ev);
            kmeans_anneal_done_ev = nullptr;
        }
        kmeans_anneal_events_ready = false;
        if (kmeans_metric_ready_ev) {
            cudaEventDestroy(kmeans_metric_ready_ev);
            kmeans_metric_ready_ev = nullptr;
        }
        if (kmeans_metric_done_ev) {
            cudaEventDestroy(kmeans_metric_done_ev);
            kmeans_metric_done_ev = nullptr;
        }
        kmeans_metric_events_ready = false;
        if (kmeans_mask_ready_ev) {
            cudaEventDestroy(kmeans_mask_ready_ev);
            kmeans_mask_ready_ev = nullptr;
        }
        if (kmeans_mask_done_ev) {
            cudaEventDestroy(kmeans_mask_done_ev);
            kmeans_mask_done_ev = nullptr;
        }
        kmeans_mask_events_ready = false;
        for (int s = 0; s < 2; ++s) {
            if (kmeans_blk_assign_ready_ev[s]) {
                cudaEventDestroy(kmeans_blk_assign_ready_ev[s]);
                kmeans_blk_assign_ready_ev[s] = nullptr;
            }
            if (kmeans_blk_assign_done_ev[s]) {
                cudaEventDestroy(kmeans_blk_assign_done_ev[s]);
                kmeans_blk_assign_done_ev[s] = nullptr;
            }
            if (kmeans_blk_assign_done_ev_timing[s]) {
                cudaEventDestroy(kmeans_blk_assign_done_ev_timing[s]);
                kmeans_blk_assign_done_ev_timing[s] = nullptr;
            }
        }
        kmeans_blk_assign_events_ready = false;
        if (stream_copy) {
            cudaStreamDestroy(stream_copy);
            stream_copy = nullptr;
        }
        if (stream) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
        if (handle) {
            cublasDestroy(handle);
            handle = nullptr;
        }
    }

    void EnsureKmeansReadyEvents() {
        if (kmeans_ready_inited) return;
        ThrowIf(cudaEventCreateWithFlags(&kmeans_ready_ev[0], cudaEventDisableTiming),
                "cudaEventCreate kmeans_ready_ev[0]");
        ThrowIf(cudaEventCreateWithFlags(&kmeans_ready_ev[1], cudaEventDisableTiming),
                "cudaEventCreate kmeans_ready_ev[1]");
        ThrowIf(cudaEventCreateWithFlags(&kmeans_free_ev[0], cudaEventDisableTiming),
                "cudaEventCreate kmeans_free_ev[0]");
        ThrowIf(cudaEventCreateWithFlags(&kmeans_free_ev[1], cudaEventDisableTiming),
                "cudaEventCreate kmeans_free_ev[1]");
        // Initially, both slots are free. Record the events on the compute stream so
        // the copy stream can wait on them without any extra sync.
        ThrowIf(cudaEventRecord(kmeans_free_ev[0], stream), "cudaEventRecord kmeans_free_ev[0]");
        ThrowIf(cudaEventRecord(kmeans_free_ev[1], stream), "cudaEventRecord kmeans_free_ev[1]");
        kmeans_ready_inited = true;
    }

    void EnsureKmeansHistEvents() {
        if (kmeans_hist_events_ready) return;
        ThrowIf(cudaEventCreateWithFlags(&kmeans_hist_ready_ev, cudaEventDisableTiming),
                "cudaEventCreate kmeans_hist_ready_ev");
        ThrowIf(cudaEventCreateWithFlags(&kmeans_hist_done_ev, cudaEventDisableTiming),
                "cudaEventCreate kmeans_hist_done_ev");
        kmeans_hist_events_ready = true;
    }

    void EnsureKmeansAnnealEvents() {
        if (kmeans_anneal_events_ready) return;
        ThrowIf(cudaEventCreateWithFlags(&kmeans_anneal_ready_ev, cudaEventDisableTiming),
                "cudaEventCreate kmeans_anneal_ready_ev");
        ThrowIf(cudaEventCreateWithFlags(&kmeans_anneal_done_ev, cudaEventDisableTiming),
                "cudaEventCreate kmeans_anneal_done_ev");
        kmeans_anneal_events_ready = true;
    }

    void EnsureKmeansMetricEvents() {
        if (kmeans_metric_events_ready) return;
        ThrowIf(cudaEventCreateWithFlags(&kmeans_metric_ready_ev, cudaEventDisableTiming),
                "cudaEventCreate kmeans_metric_ready_ev");
        ThrowIf(cudaEventCreateWithFlags(&kmeans_metric_done_ev, cudaEventDisableTiming),
                "cudaEventCreate kmeans_metric_done_ev");
        kmeans_metric_events_ready = true;
    }

    void EnsureKmeansMaskEvents() {
        if (kmeans_mask_events_ready) return;
        ThrowIf(cudaEventCreateWithFlags(&kmeans_mask_ready_ev, cudaEventDisableTiming),
                "cudaEventCreate kmeans_mask_ready_ev");
        ThrowIf(cudaEventCreateWithFlags(&kmeans_mask_done_ev, cudaEventDisableTiming),
                "cudaEventCreate kmeans_mask_done_ev");
        kmeans_mask_events_ready = true;
    }

    void EnsureKmeansBlockAssignEvents() {
        if (kmeans_blk_assign_events_ready) return;
        for (int s = 0; s < 2; ++s) {
            ThrowIf(cudaEventCreateWithFlags(&kmeans_blk_assign_ready_ev[s], cudaEventDisableTiming),
                    "cudaEventCreate kmeans_blk_assign_ready_ev");
            ThrowIf(cudaEventCreateWithFlags(&kmeans_blk_assign_done_ev[s], cudaEventDisableTiming),
                    "cudaEventCreate kmeans_blk_assign_done_ev");
            ThrowIf(cudaEventCreate(&kmeans_blk_assign_done_ev_timing[s]),
                    "cudaEventCreate kmeans_blk_assign_done_ev_timing");
        }
        kmeans_blk_assign_events_ready = true;
    }

    void Gemm(bool transA, bool transB,
              float alpha,
              const ColMajorMatrix<float>& A,
              const ColMajorMatrix<float>& B,
              float beta,
              ColMajorMatrix<float>* C) {
        if (!C) {
            throw std::runtime_error("CudaStreamKernels::Gemm: output is null.");
        }
        if (A.rows <= 0 || A.cols <= 0 || B.rows <= 0 || B.cols <= 0) {
            C->rows = 0;
            C->cols = 0;
            C->data.clear();
            return;
        }

        const int m = transA ? A.cols : A.rows;
        const int kA = transA ? A.rows : A.cols;
        const int kB = transB ? B.cols : B.rows;
        const int n = transB ? B.rows : B.cols;
        if (kA != kB) {
            throw std::runtime_error("CudaStreamKernels::Gemm: dimension mismatch.");
        }

        C->rows = m;
        C->cols = n;
        C->data.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));

        // Upload A (cached).
        A_cache.UploadIfNeeded(A, stream);

        const float* dB_ptr = nullptr;
        const std::uint64_t B_stamp = CachedDeviceMatF32::Stamp(B);
        if (cached_host_ptr == B.data.data() && cached_rows == B.rows && cached_cols == B.cols &&
            cached_host_stamp == B_stamp && d_cached.ptr) {
            dB_ptr = static_cast<const float*>(d_cached.ptr);
        } else {
            const std::size_t nb = static_cast<std::size_t>(B.rows) * static_cast<std::size_t>(B.cols);
            dB.Ensure(nb * sizeof(float));
            ThrowIf(cudaMemcpyAsync(dB.ptr, B.data.data(), nb * sizeof(float),
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync B H2D");
            dB_ptr = static_cast<const float*>(dB.ptr);
        }

        // Allocate output on device (avoid aliasing the cached input buffer).
        {
            const std::size_t nc = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
            d_out.Ensure(nc * sizeof(float));
            if (beta != 0.0f) {
                ThrowIf(cudaMemcpyAsync(d_out.ptr, C->data.data(), nc * sizeof(float),
                                        cudaMemcpyHostToDevice, stream),
                        "cudaMemcpyAsync C H2D (beta!=0)");
            }
        }

        const cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
        const cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
        const int lda = A.rows;
        const int ldb = B.rows;
        const int ldc = m;

        GemmF32(opA, opB, m, n, kA,
                &alpha,
                A_cache.dptr(),
                lda,
                dB_ptr,
                ldb,
                &beta,
                static_cast<float*>(d_out.ptr),
                ldc,
                allow_tf32 ? "cublasGemmEx(TF32)" : "cublasSgemm");

        // Download.
        {
            const std::size_t nc = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
        ThrowIf(cudaStreamSynchronize(stream), "cudaStreamSynchronize (Gemm)");
        ThrowIf(cudaMemcpy(C->data.data(), d_out.ptr, nc * sizeof(float), cudaMemcpyDeviceToHost),
                "cudaMemcpy C D2H");
        }

        // Update cached output (swap to reuse allocations without extra copies).
        std::swap(d_cached.ptr, d_out.ptr);
        std::swap(d_cached.bytes, d_out.bytes);
        cached_host_ptr = C->data.data();
        cached_rows = C->rows;
        cached_cols = C->cols;
        cached_host_stamp = CachedDeviceMatF32::Stamp(*C);
    }

    void GemmDevice(bool transA, bool transB,
                    float alpha,
                    const ColMajorMatrix<float>& A,
                    const ColMajorMatrix<float>& B,
                    float beta,
                    DeviceMatF32View* out) {
        if (!out) {
            throw std::runtime_error("CudaStreamKernels::GemmDevice: output view is null.");
        }
        out->ptr = nullptr;
        out->rows = 0;
        out->cols = 0;
        out->ld = 0;

        if (A.rows <= 0 || A.cols <= 0 || B.rows <= 0 || B.cols <= 0) {
            return;
        }

        const int m = transA ? A.cols : A.rows;
        const int kA = transA ? A.rows : A.cols;
        const int kB = transB ? B.cols : B.rows;
        const int n = transB ? B.rows : B.cols;
        if (kA != kB) {
            throw std::runtime_error("CudaStreamKernels::GemmDevice: dimension mismatch.");
        }

        // Upload A (cached).
        A_cache.UploadIfNeeded(A, stream);

        // Upload B (cached). This is critical for k-means assignment where X is reused across iterations.
        B_cache.UploadIfNeeded(B, stream);
        const float* dB_ptr = B_cache.dptr();

        // Allocate output on device.
        const std::size_t nc = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
        d_out.Ensure(nc * sizeof(float));

        const cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
        const cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
        const int lda = A.rows;
        const int ldb = B.rows;
        const int ldc = m;

        GemmF32(opA, opB, m, n, kA,
                &alpha,
                A_cache.dptr(),
                lda,
                dB_ptr,
                ldb,
                &beta,
                static_cast<float*>(d_out.ptr),
                ldc,
                allow_tf32 ? "cublasGemmEx(TF32) (GemmDevice)" : "cublasSgemm (GemmDevice)");

        // Publish view as the cached device output (swap to reuse buffers).
        std::swap(d_cached.ptr, d_out.ptr);
        std::swap(d_cached.bytes, d_out.bytes);

        cached_host_ptr = nullptr;
        cached_rows = m;
        cached_cols = n;
        cached_host_stamp = 0;

        out->ptr = static_cast<const float*>(d_cached.ptr);
        out->rows = m;
        out->cols = n;
        out->ld = m;
    }

    void GemmDeviceHostPtrB(bool transA, bool transB,
                            float alpha,
                            const ColMajorMatrix<float>& A,
                            const float* B,
                            int ldB,
                            int rowsB,
                            int colsB,
                            float beta,
                            DeviceMatF32View* out) {
        if (!out) {
            throw std::runtime_error("CudaStreamKernels::GemmDeviceHostPtrB: output view is null.");
        }
        out->ptr = nullptr;
        out->rows = 0;
        out->cols = 0;
        out->ld = 0;

        if (A.rows <= 0 || A.cols <= 0 || !B || rowsB <= 0 || colsB <= 0) {
            return;
        }

        const int m = transA ? A.cols : A.rows;
        const int kA = transA ? A.rows : A.cols;
        const int kB = transB ? colsB : rowsB;
        const int n = transB ? rowsB : colsB;
        if (kA != kB) {
            throw std::runtime_error("CudaStreamKernels::GemmDeviceHostPtrB: dimension mismatch.");
        }
        if (ldB < rowsB) {
            throw std::runtime_error("CudaStreamKernels::GemmDeviceHostPtrB: invalid ldB.");
        }

        // Upload A (cached).
        A_cache.UploadIfNeeded(A, stream);

        // Upload B tile (no caching: callers pass different pointers per tile).
        const std::size_t nb = static_cast<std::size_t>(rowsB) * static_cast<std::size_t>(colsB);
        dB.Ensure(nb * sizeof(float));
        if (ldB == rowsB) {
            ThrowIf(cudaMemcpyAsync(dB.ptr, B, nb * sizeof(float),
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync B H2D (GemmDeviceHostPtrB)");
        } else {
            ThrowIf(cudaMemcpy2DAsync(dB.ptr,
                                      static_cast<std::size_t>(rowsB) * sizeof(float),
                                      B,
                                      static_cast<std::size_t>(ldB) * sizeof(float),
                                      static_cast<std::size_t>(rowsB) * sizeof(float),
                                      static_cast<std::size_t>(colsB),
                                      cudaMemcpyHostToDevice,
                                      stream),
                    "cudaMemcpy2DAsync B H2D (GemmDeviceHostPtrB)");
        }
        const float* dB_ptr = static_cast<const float*>(dB.ptr);

        // Allocate output on device.
        const std::size_t nc = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
        d_out.Ensure(nc * sizeof(float));

        const cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
        const cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
        const int lda = A.rows;
        const int ldb = rowsB;
        const int ldc = m;

        GemmF32(opA, opB, m, n, kA,
                &alpha,
                A_cache.dptr(),
                lda,
                dB_ptr,
                ldb,
                &beta,
                static_cast<float*>(d_out.ptr),
                ldc,
                allow_tf32 ? "cublasGemmEx(TF32) (GemmDeviceHostPtrB)" : "cublasSgemm (GemmDeviceHostPtrB)");

        // Publish view as the cached device output (swap to reuse buffers).
        std::swap(d_cached.ptr, d_out.ptr);
        std::swap(d_cached.bytes, d_out.bytes);
        cached_host_ptr = nullptr;
        cached_rows = m;
        cached_cols = n;
        cached_host_stamp = 0;

        out->ptr = static_cast<const float*>(d_cached.ptr);
        out->rows = m;
        out->cols = n;
        out->ld = m;
    }

    bool EnsureDeviceF32(const ColMajorMatrix<float>& host,
                         DeviceMatF32View* out) {
        if (out) {
            out->ptr = nullptr;
            out->rows = 0;
            out->cols = 0;
            out->ld = 0;
        }
        if (host.rows <= 0 || host.cols <= 0 || host.data.empty()) {
            return false;
        }
        try {
            X_cache.UploadIfNeeded(host, stream);
            if (out) {
                out->ptr = X_cache.dptr();
                out->rows = host.rows;
                out->cols = host.cols;
                out->ld = host.rows;
            }
            return true;
        } catch (...) {
            // If we cannot keep a persistent device copy (e.g., OOM), caller can fall back to tiling H2D.
            if (out) {
                out->ptr = nullptr;
                out->rows = 0;
                out->cols = 0;
                out->ld = 0;
            }
            return false;
        }
    }

    void GemmDeviceDevicePtrB(bool transA, bool transB,
                              float alpha,
                              const ColMajorMatrix<float>& A,
                              const float* dB_ptr,
                              int ldB,
                              int rowsB,
                              int colsB,
                              float beta,
                              DeviceMatF32View* out) {
        if (!out) {
            throw std::runtime_error("CudaStreamKernels::GemmDeviceDevicePtrB: output view is null.");
        }
        out->ptr = nullptr;
        out->rows = 0;
        out->cols = 0;
        out->ld = 0;

        if (A.rows <= 0 || A.cols <= 0 || !dB_ptr || rowsB <= 0 || colsB <= 0) {
            return;
        }
        if (ldB < rowsB) {
            throw std::runtime_error("CudaStreamKernels::GemmDeviceDevicePtrB: invalid ldB.");
        }

        const int m = transA ? A.cols : A.rows;
        const int kA = transA ? A.rows : A.cols;
        const int kB = transB ? colsB : rowsB;
        const int n = transB ? rowsB : colsB;
        if (kA != kB) {
            throw std::runtime_error("CudaStreamKernels::GemmDeviceDevicePtrB: dimension mismatch.");
        }

        // Upload A (cached).
        A_cache.UploadIfNeeded(A, stream);

        // Allocate output on device.
        const std::size_t nc = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
        d_out.Ensure(nc * sizeof(float));

        const cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
        const cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
        const int lda = A.rows;
        const int ldb = ldB;
        const int ldc = m;

        GemmF32(opA, opB, m, n, kA,
                &alpha,
                A_cache.dptr(),
                lda,
                dB_ptr,
                ldb,
                &beta,
                static_cast<float*>(d_out.ptr),
                ldc,
                allow_tf32 ? "cublasGemmEx(TF32) (GemmDeviceDevicePtrB)" : "cublasSgemm (GemmDeviceDevicePtrB)");

        // Publish view as the cached device output (swap to reuse buffers).
        std::swap(d_cached.ptr, d_out.ptr);
        std::swap(d_cached.bytes, d_out.bytes);
        cached_host_ptr = nullptr;
        cached_rows = m;
        cached_cols = n;
        cached_host_stamp = 0;

        out->ptr = static_cast<const float*>(d_cached.ptr);
        out->rows = m;
        out->cols = n;
        out->ld = m;
    }

    void DownloadF32(const DeviceMatF32View& view,
                     ColMajorMatrix<float>* out) {
        if (!out) {
            throw std::runtime_error("CudaStreamKernels::DownloadF32: output is null.");
        }
        if (!view.ptr || view.rows <= 0 || view.cols <= 0 || view.ld < view.rows) {
            out->rows = 0;
            out->cols = 0;
            out->data.clear();
            return;
        }
        out->rows = view.rows;
        out->cols = view.cols;
        out->data.resize(static_cast<std::size_t>(view.rows) * static_cast<std::size_t>(view.cols));
        ThrowIf(cudaStreamSynchronize(stream), "cudaStreamSynchronize (DownloadF32)");
        ThrowIf(cudaMemcpy(out->data.data(),
                           view.ptr,
                           sizeof(float) * static_cast<std::size_t>(view.rows) * static_cast<std::size_t>(view.cols),
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy DownloadF32 D2H");
    }

    void ArgmaxColsF32(const DeviceMatF32View& scores,
                       std::vector<int>* assignments,
                       std::vector<float>* best_values) {
        if (!assignments || !best_values) {
            throw std::runtime_error("CudaStreamKernels::ArgmaxColsF32: output vectors must be non-null.");
        }
        if (!scores.ptr || scores.rows <= 0 || scores.cols <= 0 || scores.ld < scores.rows) {
            assignments->clear();
            best_values->clear();
            return;
        }

        const int rows = scores.rows;
        const int cols = scores.cols;
        // Performance: avoid reinitializing output buffers every call; cudaMemcpy fully overwrites.
        // (Typical use is fixed-size tiles; only the last tile may differ.)
        assignments->resize(static_cast<std::size_t>(cols));
        best_values->resize(static_cast<std::size_t>(cols));

        d_assign.Ensure(static_cast<std::size_t>(cols) * sizeof(int));
        d_best.Ensure(static_cast<std::size_t>(cols) * sizeof(float));

        const dim3 block(256, 1, 1);
        const dim3 grid(static_cast<unsigned int>(cols), 1, 1);
        ArgmaxColsKernel<<<grid, block, 0, stream>>>(scores.ptr, scores.ld, rows, cols,
                                          static_cast<int*>(d_assign.ptr),
                                          static_cast<float*>(d_best.ptr));
        ThrowIf(cudaGetLastError(), "ArgmaxColsKernel launch");

        // D2H copies must wait for work on our non-default stream.
        ThrowIf(cudaStreamSynchronize(stream), "cudaStreamSynchronize (ArgmaxColsF32)");
        ThrowIf(cudaMemcpy(assignments->data(), d_assign.ptr,
                           static_cast<std::size_t>(cols) * sizeof(int),
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy argmax idx D2H");
        ThrowIf(cudaMemcpy(best_values->data(), d_best.ptr,
                           static_cast<std::size_t>(cols) * sizeof(float),
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy argmax val D2H");
    }

    void ArgmaxColsF32Device(const DeviceMatF32View& scores,
                             DeviceVecI32View* d_assign_out,
                             DeviceVecF32View* d_best_out) {
        if (d_assign_out) {
            d_assign_out->ptr = nullptr;
            d_assign_out->n = 0;
        }
        if (d_best_out) {
            d_best_out->ptr = nullptr;
            d_best_out->n = 0;
        }
        if (!scores.ptr || scores.rows <= 0 || scores.cols <= 0 || scores.ld < scores.rows) {
            return;
        }

        const int rows = scores.rows;
        const int cols = scores.cols;
        d_assign.Ensure(static_cast<std::size_t>(cols) * sizeof(int));
        d_best.Ensure(static_cast<std::size_t>(cols) * sizeof(float));

        const dim3 block(256, 1, 1);
        const dim3 grid(static_cast<unsigned int>(cols), 1, 1);
        ArgmaxColsKernel<<<grid, block, 0, stream>>>(scores.ptr, scores.ld, rows, cols,
                                                     static_cast<int*>(d_assign.ptr),
                                                     static_cast<float*>(d_best.ptr));
        ThrowIf(cudaGetLastError(), "ArgmaxColsKernel launch (device)");

        if (d_assign_out) {
            d_assign_out->ptr = static_cast<const int*>(d_assign.ptr);
            d_assign_out->n = cols;
        }
        if (d_best_out) {
            d_best_out->ptr = static_cast<const float*>(d_best.ptr);
            d_best_out->n = cols;
        }
    }

    void Sync() {
        if (stream_copy) {
            ThrowIf(cudaStreamSynchronize(stream_copy), "cudaStreamSynchronize (copy stream)");
        }
        ThrowIf(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }

    void EnsureEvents() {
        if (events_ready) return;
        ThrowIf(cudaEventCreateWithFlags(&ev0, cudaEventDefault), "cudaEventCreate ev0");
        ThrowIf(cudaEventCreateWithFlags(&ev1, cudaEventDefault), "cudaEventCreate ev1");
        ThrowIf(cudaEventCreateWithFlags(&ev2, cudaEventDefault), "cudaEventCreate ev2");
        events_ready = true;
    }

    DeviceMatF32View KmeansUploadAndNormalizeHostPtrB(const float* B,
                                                      int ldB,
                                                      int rowsB,
                                                      int colsB) {
        return KmeansUploadAndNormalizeHostPtrBTimed(B, ldB, rowsB, colsB,
                                                     /*stage_to_pinned=*/false,
                                                     /*stage_s=*/nullptr,
                                                     /*h2d_s=*/nullptr,
                                                     /*kernel_s=*/nullptr);
    }

    DeviceMatF32View KmeansUploadAndNormalizeHostPtrBTimed(const float* B,
                                                           int ldB,
                                                           int rowsB,
                                                           int colsB,
                                                           bool stage_to_pinned,
                                                           double* stage_s,
                                                           double* h2d_s,
                                                           double* kernel_s) {
        if (stage_s) *stage_s = 0.0;
        if (h2d_s) *h2d_s = 0.0;
        if (kernel_s) *kernel_s = 0.0;
        if (!B || rowsB <= 0 || colsB <= 0 || ldB < rowsB) {
            return DeviceMatF32View{};
        }
        const std::size_t n = static_cast<std::size_t>(rowsB) * static_cast<std::size_t>(colsB);
        kmeans_xnorm.Ensure(n * sizeof(float));
        kmeans_xnorm_rows = rowsB;
        kmeans_xnorm_cols = colsB;
        kmeans_xnorm_ld = rowsB;
        if (kmeans_collect_metrics) {
            kmeans_norm2.Ensure(static_cast<std::size_t>(colsB) * sizeof(float));
        }

        const bool do_prof = (h2d_s != nullptr) || (kernel_s != nullptr);
        if (do_prof) EnsureEvents();
        if (do_prof) ThrowIf(cudaEventRecord(ev0, stream), "cudaEventRecord ev0");

        const float* src = B;
        if (stage_to_pinned) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            // Use write-combined pinned memory to maximize H2D bandwidth (we only write on host, then DMA to GPU).
            kmeans_hpin.EnsureWithFlags(n * sizeof(float),
                                        cudaHostAllocPortable | cudaHostAllocWriteCombined);
            float* dst = static_cast<float*>(kmeans_hpin.ptr);
            if (ldB == rowsB) {
                std::memcpy(dst, B, n * sizeof(float));
            } else {
                for (int c = 0; c < colsB; ++c) {
                    std::memcpy(dst + static_cast<std::size_t>(c) * static_cast<std::size_t>(rowsB),
                                B + static_cast<std::size_t>(c) * static_cast<std::size_t>(ldB),
                                static_cast<std::size_t>(rowsB) * sizeof(float));
                }
            }
            if (stage_s) {
                const auto t1 = std::chrono::high_resolution_clock::now();
                *stage_s = std::chrono::duration<double>(t1 - t0).count();
            }
            src = dst;
        }

        ThrowIf(cudaMemcpyAsync(kmeans_xnorm.ptr,
                                src,
                                n * sizeof(float),
                                cudaMemcpyHostToDevice,
                                stream),
                "cudaMemcpyAsync (kmeans_xnorm H2D)");

        if (do_prof) ThrowIf(cudaEventRecord(ev1, stream), "cudaEventRecord ev1");

        // Normalize with multiple warps per block to reduce launch overhead for large `colsB`.
        constexpr int kWarpsPerBlock = 8;
        const dim3 block(32 * kWarpsPerBlock, 1, 1);
        const dim3 grid(static_cast<unsigned int>((colsB + kWarpsPerBlock - 1) / kWarpsPerBlock), 1, 1);
        if (kmeans_collect_metrics) {
            NormalizeColsWarpKernelWithNorm2<kWarpsPerBlock><<<grid, block, 0, stream>>>(
                static_cast<float*>(kmeans_xnorm.ptr), kmeans_xnorm_ld, rowsB, colsB,
                static_cast<float*>(kmeans_norm2.ptr));
            ThrowIf(cudaGetLastError(), "NormalizeColsWarpKernel launch (with norm2)");
        } else {
            NormalizeColsWarpKernel<kWarpsPerBlock><<<grid, block, 0, stream>>>(
                static_cast<float*>(kmeans_xnorm.ptr), kmeans_xnorm_ld, rowsB, colsB);
            ThrowIf(cudaGetLastError(), "NormalizeColsWarpKernel launch");
        }

        if (do_prof) ThrowIf(cudaEventRecord(ev2, stream), "cudaEventRecord ev2");

        if (do_prof) {
            ThrowIf(cudaEventSynchronize(ev2), "cudaEventSynchronize ev2");
            float ms_h2d = 0.0f;
            float ms_ker = 0.0f;
            ThrowIf(cudaEventElapsedTime(&ms_h2d, ev0, ev1), "cudaEventElapsedTime h2d");
            ThrowIf(cudaEventElapsedTime(&ms_ker, ev1, ev2), "cudaEventElapsedTime kernel");
            if (h2d_s) *h2d_s = static_cast<double>(ms_h2d) * 1e-3;
            if (kernel_s) *kernel_s = static_cast<double>(ms_ker) * 1e-3;
        }

        DeviceMatF32View out;
        out.ptr = static_cast<const float*>(kmeans_xnorm.ptr);
        out.rows = rowsB;
        out.cols = colsB;
        out.ld = kmeans_xnorm_ld;
        kmeans_active_slot = -1;
        return out;
    }

    DeviceMatF32View KmeansUploadAndNormalizeHostPtrBAsync(int slot,
                                                           const float* B,
                                                           int ldB,
                                                           int rowsB,
                                                           int colsB,
                                                           bool stage_to_pinned,
                                                           double* stage_s,
                                                           double* h2d_s,
                                                           double* kernel_s) {
        if (stage_s) *stage_s = 0.0;
        if (h2d_s) *h2d_s = 0.0;
        if (kernel_s) *kernel_s = 0.0;
        if (slot != 0 && slot != 1) {
            throw std::runtime_error("KmeansUploadAndNormalizeHostPtrBAsync: slot must be 0 or 1.");
        }
        if (!B || rowsB <= 0 || colsB <= 0 || ldB < rowsB) {
            return DeviceMatF32View{};
        }
        EnsureKmeansReadyEvents();
        // Ensure we don't reuse this slot while the compute stream is still reading it.
        if (kmeans_timing_enabled) {
            EnsureKmeansTimingEvents();
            HarvestKmeansWaitCopyTimingSlot(slot, /*force_sync=*/false);
            ThrowIf(cudaEventRecord(kmeans_wait_copy_ev0[slot], stream_copy),
                    "cudaEventRecord (kmeans_wait_copy_ev0)");
        }
        ThrowIf(cudaStreamWaitEvent(stream_copy, kmeans_free_ev[slot], 0),
                "cudaStreamWaitEvent (kmeans_free_ev)");
        if (kmeans_timing_enabled) {
            ThrowIf(cudaEventRecord(kmeans_wait_copy_ev1[slot], stream_copy),
                    "cudaEventRecord (kmeans_wait_copy_ev1)");
            kmeans_wait_copy_pending[slot] = true;
        }

        const std::size_t n = static_cast<std::size_t>(rowsB) * static_cast<std::size_t>(colsB);
        kmeans_xnorm_slot[slot].Ensure(n * sizeof(float));
        kmeans_slot_rows[slot] = rowsB;
        kmeans_slot_cols[slot] = colsB;
        kmeans_slot_ld[slot] = rowsB;
        if (kmeans_collect_metrics) {
            kmeans_norm2_slot[slot].Ensure(static_cast<std::size_t>(colsB) * sizeof(float));
            rvq_hatnorm2_slot[slot].Ensure(static_cast<std::size_t>(colsB) * sizeof(float));
        }

        const float* src = B;
        if (stage_to_pinned) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            kmeans_hpin_slot[slot].EnsureWithFlags(n * sizeof(float),
                                                   cudaHostAllocPortable | cudaHostAllocWriteCombined);
            float* dst = static_cast<float*>(kmeans_hpin_slot[slot].ptr);
            if (ldB == rowsB) {
                std::memcpy(dst, B, n * sizeof(float));
            } else {
                for (int c = 0; c < colsB; ++c) {
                    std::memcpy(dst + static_cast<std::size_t>(c) * static_cast<std::size_t>(rowsB),
                                B + static_cast<std::size_t>(c) * static_cast<std::size_t>(ldB),
                                static_cast<std::size_t>(rowsB) * sizeof(float));
                }
            }
            if (stage_s) {
                const auto t1 = std::chrono::high_resolution_clock::now();
                *stage_s = std::chrono::duration<double>(t1 - t0).count();
            }
            src = dst;
        }

        const bool deferred_time = kmeans_timing_enabled;
        const bool want_time = (!deferred_time) && ((h2d_s != nullptr) || (kernel_s != nullptr));
        if (deferred_time) {
            EnsureKmeansTimingEvents();
            HarvestKmeansUploadTimingSlot(slot, /*force_sync=*/false);
            ThrowIf(cudaEventRecord(kmeans_upload_ev0[slot], stream_copy), "cudaEventRecord (kmeans_upload_ev0 copy)");
        } else if (want_time) {
            EnsureEvents();
            ThrowIf(cudaEventRecord(ev0, stream_copy), "cudaEventRecord ev0 (copy)");
        }

        ThrowIf(cudaMemcpyAsync(kmeans_xnorm_slot[slot].ptr,
                                src,
                                n * sizeof(float),
                                cudaMemcpyHostToDevice,
                                stream_copy),
                "cudaMemcpyAsync (kmeans_xnorm_slot H2D)");

        if (deferred_time) {
            ThrowIf(cudaEventRecord(kmeans_upload_ev1[slot], stream_copy), "cudaEventRecord (kmeans_upload_ev1 copy)");
        } else if (want_time) {
            ThrowIf(cudaEventRecord(ev1, stream_copy), "cudaEventRecord ev1 (copy)");
        }

        constexpr int kWarpsPerBlock = 8;
        const dim3 block(32 * kWarpsPerBlock, 1, 1);
        const dim3 grid(static_cast<unsigned int>((colsB + kWarpsPerBlock - 1) / kWarpsPerBlock), 1, 1);
        if (kmeans_collect_metrics) {
            NormalizeColsWarpKernelWithNorm2<kWarpsPerBlock><<<grid, block, 0, stream_copy>>>(
                static_cast<float*>(kmeans_xnorm_slot[slot].ptr), kmeans_slot_ld[slot], rowsB, colsB,
                static_cast<float*>(kmeans_norm2_slot[slot].ptr));
            ThrowIf(cudaGetLastError(), "NormalizeColsWarpKernel launch (async, with norm2)");

            const int threads = 256;
            const int blocks = (colsB + threads - 1) / threads;
            FillF32Kernel<<<blocks, threads, 0, stream_copy>>>(
                static_cast<float*>(rvq_hatnorm2_slot[slot].ptr), colsB, 1.0f);
            ThrowIf(cudaGetLastError(), "FillF32Kernel launch (rvq_hatnorm2 init)");
        } else {
            NormalizeColsWarpKernel<kWarpsPerBlock><<<grid, block, 0, stream_copy>>>(
                static_cast<float*>(kmeans_xnorm_slot[slot].ptr), kmeans_slot_ld[slot], rowsB, colsB);
            ThrowIf(cudaGetLastError(), "NormalizeColsWarpKernel launch (async)");
        }

        if (deferred_time) {
            ThrowIf(cudaEventRecord(kmeans_upload_ev2[slot], stream_copy), "cudaEventRecord (kmeans_upload_ev2 copy)");
            kmeans_upload_pending[slot] = true;
        } else if (want_time) {
            ThrowIf(cudaEventRecord(ev2, stream_copy), "cudaEventRecord ev2 (copy)");
        }
        ThrowIf(cudaEventRecord(kmeans_ready_ev[slot], stream_copy), "cudaEventRecord kmeans_ready_ev (copy)");

        if (want_time) {
            ThrowIf(cudaEventSynchronize(ev2), "cudaEventSynchronize ev2 (copy)");
            float ms_h2d = 0.0f;
            float ms_ker = 0.0f;
            ThrowIf(cudaEventElapsedTime(&ms_h2d, ev0, ev1), "cudaEventElapsedTime h2d (copy)");
            ThrowIf(cudaEventElapsedTime(&ms_ker, ev1, ev2), "cudaEventElapsedTime kernel (copy)");
            if (h2d_s) *h2d_s = static_cast<double>(ms_h2d) * 1e-3;
            if (kernel_s) *kernel_s = static_cast<double>(ms_ker) * 1e-3;
        }

        DeviceMatF32View out;
        out.ptr = static_cast<const float*>(kmeans_xnorm_slot[slot].ptr);
        out.rows = rowsB;
        out.cols = colsB;
        out.ld = kmeans_slot_ld[slot];
        return out;
    }

    float* KmeansGetPinnedStageBuffer(int slot, int rowsB, int colsB) {
        if (slot != 0 && slot != 1) {
            throw std::runtime_error("KmeansGetPinnedStageBuffer: slot must be 0 or 1.");
        }
        if (rowsB <= 0 || colsB <= 0) {
            return nullptr;
        }
        const std::size_t n = static_cast<std::size_t>(rowsB) * static_cast<std::size_t>(colsB);
        kmeans_hpin_slot[slot].EnsureWithFlags(n * sizeof(float),
                                              cudaHostAllocPortable | cudaHostAllocWriteCombined);
        return static_cast<float*>(kmeans_hpin_slot[slot].ptr);
    }

    std::uint8_t* KmeansGetPinnedStageBufferU8(int slot, int rowsB, int colsB) {
        if (slot != 0 && slot != 1) {
            throw std::runtime_error("KmeansGetPinnedStageBufferU8: slot must be 0 or 1.");
        }
        if (rowsB <= 0 || colsB <= 0) {
            return nullptr;
        }
        const std::size_t n = static_cast<std::size_t>(rowsB) * static_cast<std::size_t>(colsB);
        kmeans_hpin_u8_slot[slot].EnsureWithFlags(n * sizeof(std::uint8_t),
                                                 cudaHostAllocPortable | cudaHostAllocWriteCombined);
        return static_cast<std::uint8_t*>(kmeans_hpin_u8_slot[slot].ptr);
    }

    DeviceMatF32View KmeansUploadAndNormalizePinnedAsync(int slot,
                                                        const float* pinned_B,
                                                        int ldB,
                                                        int rowsB,
                                                        int colsB,
                                                        double* h2d_s,
                                                        double* kernel_s) {
        if (h2d_s) *h2d_s = 0.0;
        if (kernel_s) *kernel_s = 0.0;
        if (slot != 0 && slot != 1) {
            throw std::runtime_error("KmeansUploadAndNormalizePinnedAsync: slot must be 0 or 1.");
        }
        if (!pinned_B || rowsB <= 0 || colsB <= 0 || ldB < rowsB) {
            return DeviceMatF32View{};
        }
        EnsureKmeansReadyEvents();
        // Ensure we don't reuse this slot while the compute stream is still reading it.
        if (kmeans_timing_enabled) {
            EnsureKmeansTimingEvents();
            HarvestKmeansWaitCopyTimingSlot(slot, /*force_sync=*/false);
            ThrowIf(cudaEventRecord(kmeans_wait_copy_ev0[slot], stream_copy),
                    "cudaEventRecord (kmeans_wait_copy_ev0 pinned)");
        }
        ThrowIf(cudaStreamWaitEvent(stream_copy, kmeans_free_ev[slot], 0),
                "cudaStreamWaitEvent (kmeans_free_ev pinned)");
        if (kmeans_timing_enabled) {
            ThrowIf(cudaEventRecord(kmeans_wait_copy_ev1[slot], stream_copy),
                    "cudaEventRecord (kmeans_wait_copy_ev1 pinned)");
            kmeans_wait_copy_pending[slot] = true;
        }

        const std::size_t n = static_cast<std::size_t>(rowsB) * static_cast<std::size_t>(colsB);
        kmeans_xnorm_slot[slot].Ensure(n * sizeof(float));
        kmeans_slot_rows[slot] = rowsB;
        kmeans_slot_cols[slot] = colsB;
        kmeans_slot_ld[slot] = rowsB;
        if (kmeans_collect_metrics) {
            kmeans_norm2_slot[slot].Ensure(static_cast<std::size_t>(colsB) * sizeof(float));
            rvq_hatnorm2_slot[slot].Ensure(static_cast<std::size_t>(colsB) * sizeof(float));
        }

        const bool deferred_time = kmeans_timing_enabled;
        const bool want_time = (!deferred_time) && ((h2d_s != nullptr) || (kernel_s != nullptr));
        if (deferred_time) {
            EnsureKmeansTimingEvents();
            HarvestKmeansUploadTimingSlot(slot, /*force_sync=*/false);
            ThrowIf(cudaEventRecord(kmeans_upload_ev0[slot], stream_copy), "cudaEventRecord (kmeans_upload_ev0 pinned)");
        } else if (want_time) {
            EnsureEvents();
            ThrowIf(cudaEventRecord(ev0, stream_copy), "cudaEventRecord ev0 (pinned)");
        }

        if (ldB == rowsB) {
            ThrowIf(cudaMemcpyAsync(kmeans_xnorm_slot[slot].ptr,
                                    pinned_B,
                                    n * sizeof(float),
                                    cudaMemcpyHostToDevice,
                                    stream_copy),
                    "cudaMemcpyAsync (kmeans_xnorm_slot H2D pinned)");
        } else {
            // Pack on device via strided copies: fall back to host repack into the internal pinned buffer.
            float* dst = static_cast<float*>(kmeans_hpin_slot[slot].ptr);
            for (int c = 0; c < colsB; ++c) {
                std::memcpy(dst + static_cast<std::size_t>(c) * static_cast<std::size_t>(rowsB),
                            pinned_B + static_cast<std::size_t>(c) * static_cast<std::size_t>(ldB),
                            static_cast<std::size_t>(rowsB) * sizeof(float));
            }
            ThrowIf(cudaMemcpyAsync(kmeans_xnorm_slot[slot].ptr,
                                    dst,
                                    n * sizeof(float),
                                    cudaMemcpyHostToDevice,
                                    stream_copy),
                    "cudaMemcpyAsync (kmeans_xnorm_slot H2D pinned repack)");
        }

        if (deferred_time) {
            ThrowIf(cudaEventRecord(kmeans_upload_ev1[slot], stream_copy), "cudaEventRecord (kmeans_upload_ev1 pinned)");
        } else if (want_time) {
            ThrowIf(cudaEventRecord(ev1, stream_copy), "cudaEventRecord ev1 (pinned)");
        }

        constexpr int kWarpsPerBlock = 8;
        const dim3 block(32 * kWarpsPerBlock, 1, 1);
        const dim3 grid(static_cast<unsigned int>((colsB + kWarpsPerBlock - 1) / kWarpsPerBlock), 1, 1);
        if (kmeans_collect_metrics) {
            NormalizeColsWarpKernelWithNorm2<kWarpsPerBlock><<<grid, block, 0, stream_copy>>>(
                static_cast<float*>(kmeans_xnorm_slot[slot].ptr), kmeans_slot_ld[slot], rowsB, colsB,
                static_cast<float*>(kmeans_norm2_slot[slot].ptr));
            ThrowIf(cudaGetLastError(), "NormalizeColsWarpKernel launch (pinned, with norm2)");

            const int threads = 256;
            const int blocks = (colsB + threads - 1) / threads;
            FillF32Kernel<<<blocks, threads, 0, stream_copy>>>(
                static_cast<float*>(rvq_hatnorm2_slot[slot].ptr), colsB, 1.0f);
            ThrowIf(cudaGetLastError(), "FillF32Kernel launch (rvq_hatnorm2 init pinned)");
        } else {
            NormalizeColsWarpKernel<kWarpsPerBlock><<<grid, block, 0, stream_copy>>>(
                static_cast<float*>(kmeans_xnorm_slot[slot].ptr), kmeans_slot_ld[slot], rowsB, colsB);
            ThrowIf(cudaGetLastError(), "NormalizeColsWarpKernel launch (pinned)");
        }

        if (deferred_time) {
            ThrowIf(cudaEventRecord(kmeans_upload_ev2[slot], stream_copy), "cudaEventRecord (kmeans_upload_ev2 pinned)");
            kmeans_upload_pending[slot] = true;
        } else if (want_time) {
            ThrowIf(cudaEventRecord(ev2, stream_copy), "cudaEventRecord ev2 (pinned)");
        }
        ThrowIf(cudaEventRecord(kmeans_ready_ev[slot], stream_copy),
                "cudaEventRecord kmeans_ready_ev (pinned)");

        if (want_time) {
            ThrowIf(cudaEventSynchronize(ev2), "cudaEventSynchronize ev2 (pinned)");
            float ms_h2d = 0.0f;
            float ms_ker = 0.0f;
            ThrowIf(cudaEventElapsedTime(&ms_h2d, ev0, ev1), "cudaEventElapsedTime h2d (pinned)");
            ThrowIf(cudaEventElapsedTime(&ms_ker, ev1, ev2), "cudaEventElapsedTime kernel (pinned)");
            if (h2d_s) *h2d_s = static_cast<double>(ms_h2d) * 1e-3;
            if (kernel_s) *kernel_s = static_cast<double>(ms_ker) * 1e-3;
        }

        DeviceMatF32View out;
        out.ptr = static_cast<const float*>(kmeans_xnorm_slot[slot].ptr);
        out.rows = rowsB;
        out.cols = colsB;
        out.ld = kmeans_slot_ld[slot];
        return out;
    }

    DeviceMatF32View KmeansUploadU8AndNormalizePinnedAsync(int slot,
                                                           const std::uint8_t* pinned_B,
                                                           int ldB,
                                                           int rowsB,
                                                           int colsB,
                                                           double* h2d_s,
                                                           double* kernel_s) {
        if (h2d_s) *h2d_s = 0.0;
        if (kernel_s) *kernel_s = 0.0;
        if (slot != 0 && slot != 1) {
            throw std::runtime_error("KmeansUploadU8AndNormalizePinnedAsync: slot must be 0 or 1.");
        }
        if (!pinned_B || rowsB <= 0 || colsB <= 0 || ldB < rowsB) {
            return DeviceMatF32View{};
        }
        EnsureKmeansReadyEvents();
        // Ensure we don't reuse this slot while the compute stream is still reading it.
        if (kmeans_timing_enabled) {
            EnsureKmeansTimingEvents();
            HarvestKmeansWaitCopyTimingSlot(slot, /*force_sync=*/false);
            ThrowIf(cudaEventRecord(kmeans_wait_copy_ev0[slot], stream_copy),
                    "cudaEventRecord (kmeans_wait_copy_ev0 u8 pinned)");
        }
        ThrowIf(cudaStreamWaitEvent(stream_copy, kmeans_free_ev[slot], 0),
                "cudaStreamWaitEvent (kmeans_free_ev u8 pinned)");
        if (kmeans_timing_enabled) {
            ThrowIf(cudaEventRecord(kmeans_wait_copy_ev1[slot], stream_copy),
                    "cudaEventRecord (kmeans_wait_copy_ev1 u8 pinned)");
            kmeans_wait_copy_pending[slot] = true;
        }

        const std::size_t n = static_cast<std::size_t>(rowsB) * static_cast<std::size_t>(colsB);
        const std::size_t nbytes_u8 = n * sizeof(std::uint8_t);
        kmeans_u8_slot[slot].Ensure(nbytes_u8);
        kmeans_u8_slot_rows[slot] = rowsB;
        kmeans_u8_slot_cols[slot] = colsB;
        kmeans_u8_slot_ld[slot] = rowsB;

        kmeans_xnorm_slot[slot].Ensure(n * sizeof(float));
        kmeans_slot_rows[slot] = rowsB;
        kmeans_slot_cols[slot] = colsB;
        kmeans_slot_ld[slot] = rowsB;
        if (kmeans_collect_metrics) {
            kmeans_norm2_slot[slot].Ensure(static_cast<std::size_t>(colsB) * sizeof(float));
            rvq_hatnorm2_slot[slot].Ensure(static_cast<std::size_t>(colsB) * sizeof(float));
        }

        const bool deferred_time = kmeans_timing_enabled;
        const bool want_time = (!deferred_time) && ((h2d_s != nullptr) || (kernel_s != nullptr));
        if (deferred_time) {
            EnsureKmeansTimingEvents();
            HarvestKmeansUploadTimingSlot(slot, /*force_sync=*/false);
            ThrowIf(cudaEventRecord(kmeans_upload_ev0[slot], stream_copy), "cudaEventRecord (kmeans_upload_ev0 u8 pinned)");
        } else if (want_time) {
            EnsureEvents();
            ThrowIf(cudaEventRecord(ev0, stream_copy), "cudaEventRecord ev0 (u8 pinned)");
        }

        if (ldB == rowsB) {
            ThrowIf(cudaMemcpyAsync(kmeans_u8_slot[slot].ptr,
                                    pinned_B,
                                    nbytes_u8,
                                    cudaMemcpyHostToDevice,
                                    stream_copy),
                    "cudaMemcpyAsync (kmeans_u8_slot H2D pinned)");
        } else {
            // Repack into internal pinned u8 buffer with ld == rowsB.
            kmeans_hpin_u8_slot[slot].EnsureWithFlags(nbytes_u8,
                                                     cudaHostAllocPortable | cudaHostAllocWriteCombined);
            std::uint8_t* dst = static_cast<std::uint8_t*>(kmeans_hpin_u8_slot[slot].ptr);
            for (int c = 0; c < colsB; ++c) {
                std::memcpy(dst + static_cast<std::size_t>(c) * static_cast<std::size_t>(rowsB),
                            pinned_B + static_cast<std::size_t>(c) * static_cast<std::size_t>(ldB),
                            static_cast<std::size_t>(rowsB) * sizeof(std::uint8_t));
            }
            ThrowIf(cudaMemcpyAsync(kmeans_u8_slot[slot].ptr,
                                    dst,
                                    nbytes_u8,
                                    cudaMemcpyHostToDevice,
                                    stream_copy),
                    "cudaMemcpyAsync (kmeans_u8_slot H2D pinned repack)");
        }

        if (deferred_time) {
            ThrowIf(cudaEventRecord(kmeans_upload_ev1[slot], stream_copy), "cudaEventRecord (kmeans_upload_ev1 u8 pinned)");
        } else if (want_time) {
            ThrowIf(cudaEventRecord(ev1, stream_copy), "cudaEventRecord ev1 (u8 pinned)");
        }

        constexpr int kWarpsPerBlock = 8;
        const dim3 block(32 * kWarpsPerBlock, 1, 1);
        const dim3 grid(static_cast<unsigned int>((colsB + kWarpsPerBlock - 1) / kWarpsPerBlock), 1, 1);
        if (kmeans_collect_metrics) {
            NormalizeColsWarpKernelU8WithNorm2<kWarpsPerBlock><<<grid, block, 0, stream_copy>>>(
                static_cast<const std::uint8_t*>(kmeans_u8_slot[slot].ptr), kmeans_u8_slot_ld[slot],
                static_cast<float*>(kmeans_xnorm_slot[slot].ptr), kmeans_slot_ld[slot],
                rowsB, colsB,
                static_cast<float*>(kmeans_norm2_slot[slot].ptr));
            ThrowIf(cudaGetLastError(), "NormalizeColsWarpKernelU8 launch (u8 pinned, with norm2)");

            const int threads = 256;
            const int blocks = (colsB + threads - 1) / threads;
            FillF32Kernel<<<blocks, threads, 0, stream_copy>>>(
                static_cast<float*>(rvq_hatnorm2_slot[slot].ptr), colsB, 1.0f);
            ThrowIf(cudaGetLastError(), "FillF32Kernel launch (rvq_hatnorm2 init u8 pinned)");
        } else {
            NormalizeColsWarpKernelU8<kWarpsPerBlock><<<grid, block, 0, stream_copy>>>(
                static_cast<const std::uint8_t*>(kmeans_u8_slot[slot].ptr), kmeans_u8_slot_ld[slot],
                static_cast<float*>(kmeans_xnorm_slot[slot].ptr), kmeans_slot_ld[slot],
                rowsB, colsB);
            ThrowIf(cudaGetLastError(), "NormalizeColsWarpKernelU8 launch (u8 pinned)");
        }

        if (deferred_time) {
            ThrowIf(cudaEventRecord(kmeans_upload_ev2[slot], stream_copy), "cudaEventRecord (kmeans_upload_ev2 u8 pinned)");
            kmeans_upload_pending[slot] = true;
        } else if (want_time) {
            ThrowIf(cudaEventRecord(ev2, stream_copy), "cudaEventRecord ev2 (u8 pinned)");
        }

        // Signal compute stream that the slot is ready.
        ThrowIf(cudaEventRecord(kmeans_ready_ev[slot], stream_copy), "cudaEventRecord kmeans_ready_ev[u8]");

        if (want_time) {
            ThrowIf(cudaEventSynchronize(ev2), "cudaEventSynchronize ev2 (u8 pinned)");
            float ms_h2d = 0.0f;
            float ms_ker = 0.0f;
            ThrowIf(cudaEventElapsedTime(&ms_h2d, ev0, ev1), "cudaEventElapsedTime h2d (u8 pinned)");
            ThrowIf(cudaEventElapsedTime(&ms_ker, ev1, ev2), "cudaEventElapsedTime kernel (u8 pinned)");
            if (h2d_s) *h2d_s = static_cast<double>(ms_h2d) * 1e-3;
            if (kernel_s) *kernel_s = static_cast<double>(ms_ker) * 1e-3;
        }

        DeviceMatF32View out;
        out.ptr = static_cast<const float*>(kmeans_xnorm_slot[slot].ptr);
        out.rows = rowsB;
        out.cols = colsB;
        out.ld = kmeans_slot_ld[slot];
        return out;
    }

    DeviceMatF32View KmeansUploadU8AndNormalizeHostPtrBAsync(int slot,
                                                             const std::uint8_t* B,
                                                             int ldB,
                                                             int rowsB,
                                                             int colsB,
                                                             bool stage_to_pinned,
                                                             double* stage_s,
                                                             double* h2d_s,
                                                             double* kernel_s) {
        if (stage_s) *stage_s = 0.0;
        if (!B || rowsB <= 0 || colsB <= 0 || ldB < rowsB) {
            if (h2d_s) *h2d_s = 0.0;
            if (kernel_s) *kernel_s = 0.0;
            return DeviceMatF32View{};
        }

        const std::uint8_t* src = B;
        int ld_src = ldB;
        if (stage_to_pinned) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            std::uint8_t* dst = KmeansGetPinnedStageBufferU8(slot, rowsB, colsB);
            if (!dst) {
                throw std::runtime_error("KmeansUploadU8AndNormalizeHostPtrBAsync: failed to get pinned stage buffer.");
            }
            if (ldB == rowsB) {
                std::memcpy(dst, B, static_cast<std::size_t>(rowsB) * static_cast<std::size_t>(colsB));
            } else {
                for (int c = 0; c < colsB; ++c) {
                    std::memcpy(dst + static_cast<std::size_t>(c) * static_cast<std::size_t>(rowsB),
                                B + static_cast<std::size_t>(c) * static_cast<std::size_t>(ldB),
                                static_cast<std::size_t>(rowsB));
                }
            }
            if (stage_s) {
                const auto t1 = std::chrono::high_resolution_clock::now();
                *stage_s = std::chrono::duration<double>(t1 - t0).count();
            }
            src = dst;
            ld_src = rowsB;
        }

        return KmeansUploadU8AndNormalizePinnedAsync(slot, src, ld_src, rowsB, colsB, h2d_s, kernel_s);
    }

    void KmeansComputeWaitForUpload(int slot) {
        if (slot != 0 && slot != 1) {
            throw std::runtime_error("KmeansComputeWaitForUpload: slot must be 0 or 1.");
        }
        EnsureKmeansReadyEvents();
        if (kmeans_timing_enabled) {
            EnsureKmeansTimingEvents();
            HarvestKmeansWaitComputeTimingSlot(slot, /*force_sync=*/false);
            ThrowIf(cudaEventRecord(kmeans_wait_compute_ev0[slot], stream),
                    "cudaEventRecord (kmeans_wait_compute_ev0)");
        }
        ThrowIf(cudaStreamWaitEvent(stream, kmeans_ready_ev[slot], 0),
                "cudaStreamWaitEvent (kmeans_ready_ev)");
        if (kmeans_timing_enabled) {
            ThrowIf(cudaEventRecord(kmeans_wait_compute_ev1[slot], stream),
                    "cudaEventRecord (kmeans_wait_compute_ev1)");
            kmeans_wait_compute_pending[slot] = true;
        }
        kmeans_active_slot = slot;
    }

    void KmeansMarkUploadSlotConsumed(int slot) {
        if (slot != 0 && slot != 1) {
            throw std::runtime_error("KmeansMarkUploadSlotConsumed: slot must be 0 or 1.");
        }
        EnsureKmeansReadyEvents();
        // Record on compute stream so the copy stream can wait before reusing this slot.
        ThrowIf(cudaEventRecord(kmeans_free_ev[slot], stream),
                "cudaEventRecord (kmeans_free_ev)");
    }

    void KmeansTimingEnable(bool enable) { kmeans_timing_enabled = enable; }

    void KmeansTimingReset() {
        kmeans_timing_upload_h2d_s = 0.0;
        kmeans_timing_upload_kernel_s = 0.0;
        kmeans_timing_rvq_project_s = 0.0;
        kmeans_timing_assign_update_compute_s = 0.0;
        kmeans_timing_emit_codes_d2h_s = 0.0;
        kmeans_timing_wait_copy_s = 0.0;
        kmeans_timing_wait_compute_s = 0.0;
        kmeans_upload_pending[0] = kmeans_upload_pending[1] = false;
        kmeans_rvq_pending[0] = kmeans_rvq_pending[1] = false;
        kmeans_compute_pending[0] = kmeans_compute_pending[1] = false;
        kmeans_compute_started[0] = kmeans_compute_started[1] = false;
        kmeans_assign_d2h_pending[0] = kmeans_assign_d2h_pending[1] = false;
        kmeans_wait_copy_pending[0] = kmeans_wait_copy_pending[1] = false;
        kmeans_wait_compute_pending[0] = kmeans_wait_compute_pending[1] = false;
    }

    void KmeansTimingComputeSectionBegin(int slot) {
        if (slot < 0 || slot >= 2) return;
        if (!kmeans_timing_enabled) return;
        EnsureKmeansTimingEvents();
        HarvestKmeansComputeTimingSlot(slot, /*force_sync=*/false);
        ThrowIf(cudaEventRecord(kmeans_compute_ev0[slot], stream), "cudaEventRecord (kmeans_compute_ev0)");
        kmeans_compute_started[slot] = true;
    }

    void KmeansTimingComputeSectionEnd(int slot) {
        if (slot < 0 || slot >= 2) return;
        if (!kmeans_timing_enabled) return;
        if (!kmeans_compute_started[slot]) return;
        EnsureKmeansTimingEvents();
        ThrowIf(cudaEventRecord(kmeans_compute_ev1[slot], stream), "cudaEventRecord (kmeans_compute_ev1)");
        kmeans_compute_pending[slot] = true;
    }

    void KmeansTimingFlush(double* out_upload_h2d_s,
                           double* out_upload_kernel_s,
                           double* out_rvq_project_s,
                           double* out_assign_update_compute_s,
                           double* out_emit_codes_d2h_s,
                           double* out_wait_copy_s,
                           double* out_wait_compute_s) {
        if (!kmeans_timing_enabled) {
            if (out_upload_h2d_s) *out_upload_h2d_s = 0.0;
            if (out_upload_kernel_s) *out_upload_kernel_s = 0.0;
            if (out_rvq_project_s) *out_rvq_project_s = 0.0;
            if (out_assign_update_compute_s) *out_assign_update_compute_s = 0.0;
            if (out_emit_codes_d2h_s) *out_emit_codes_d2h_s = 0.0;
            if (out_wait_copy_s) *out_wait_copy_s = 0.0;
            if (out_wait_compute_s) *out_wait_compute_s = 0.0;
            return;
        }
        EnsureKmeansTimingEvents();
        for (int s = 0; s < 2; ++s) {
            HarvestKmeansUploadTimingSlot(s, /*force_sync=*/true);
            HarvestKmeansRvqTimingSlot(s, /*force_sync=*/true);
            HarvestKmeansComputeTimingSlot(s, /*force_sync=*/true);
            HarvestKmeansWaitCopyTimingSlot(s, /*force_sync=*/true);
            HarvestKmeansWaitComputeTimingSlot(s, /*force_sync=*/true);
        }
        if (out_upload_h2d_s) *out_upload_h2d_s = kmeans_timing_upload_h2d_s;
        if (out_upload_kernel_s) *out_upload_kernel_s = kmeans_timing_upload_kernel_s;
        if (out_rvq_project_s) *out_rvq_project_s = kmeans_timing_rvq_project_s;
        if (out_assign_update_compute_s) *out_assign_update_compute_s = kmeans_timing_assign_update_compute_s;
        if (out_emit_codes_d2h_s) *out_emit_codes_d2h_s = kmeans_timing_emit_codes_d2h_s;
        if (out_wait_copy_s) *out_wait_copy_s = kmeans_timing_wait_copy_s;
        if (out_wait_compute_s) *out_wait_compute_s = kmeans_timing_wait_compute_s;
    }

    // -------- kmeans|| (kmeansll) seeding helpers --------
    DeviceMatF32View KmeansllEnsureSampleBuffer(int rows, int cols) {
        if (rows <= 0 || cols <= 0) return DeviceMatF32View{};
        const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
        kmeansll_sample.Ensure(n * sizeof(float));
        kmeansll_sample_rows = rows;
        kmeansll_sample_cols = cols;
        kmeansll_sample_ld = rows;
        DeviceMatF32View out;
        out.ptr = static_cast<const float*>(kmeansll_sample.ptr);
        out.rows = rows;
        out.cols = cols;
        out.ld = rows;
        return out;
    }

    void KmeansllReleaseSampleCache() {
        kmeansll_sample.Reset();
        kmeansll_sample_rows = 0;
        kmeansll_sample_cols = 0;
        kmeansll_sample_ld = 0;
        kmeansll_best_dot.Reset();
        kmeansll_best_dot_n = 0;
        kmeansll_d2.Reset();
        kmeansll_flags.Reset();
        kmeansll_indices.Reset();
        kmeansll_indices_n = 0;
        kmeansll_selected.Reset();
        kmeansll_num_selected.Reset();
        kmeansll_phi.Reset();
        kmeansll_last_selected = 0;
        kmeansll_selected_hpin.Reset();
        kmeansll_idxbuf.Reset();
        kmeansll_idxbuf_cap = 0;
        kmeansll_idxbuf_hpin.Reset();
        kmeansll_gather.Reset();
        kmeansll_gather_rows = 0;
        kmeansll_gather_cols = 0;
        kmeansll_gather_ld = 0;
        kmeansll_cub_tmp.Reset();
    }

    DeviceMatF32View KmeansllSampleView() const {
        if (!kmeansll_sample.ptr || kmeansll_sample_rows <= 0 || kmeansll_sample_cols <= 0) {
            return DeviceMatF32View{};
        }
        DeviceMatF32View out;
        out.ptr = static_cast<const float*>(kmeansll_sample.ptr);
        out.rows = kmeansll_sample_rows;
        out.cols = kmeansll_sample_cols;
        out.ld = kmeansll_sample_ld;
        return out;
    }

    void KmeansllCopyActiveSlotToSample(int slot, int dst_col0, int cols) {
        if (slot != 0 && slot != 1) throw std::runtime_error("KmeansllCopyActiveSlotToSample: slot must be 0 or 1.");
        if (cols <= 0) return;
        if (!kmeansll_sample.ptr || kmeansll_sample_rows <= 0 || kmeansll_sample_cols <= 0) {
            throw std::runtime_error("KmeansllCopyActiveSlotToSample: sample buffer not initialized.");
        }
        if (dst_col0 < 0 || dst_col0 + cols > kmeansll_sample_cols) {
            throw std::runtime_error("KmeansllCopyActiveSlotToSample: dst range out of bounds.");
        }
        if (!kmeans_xnorm_slot[slot].ptr || kmeans_slot_rows[slot] != kmeansll_sample_rows || kmeans_slot_cols[slot] < cols) {
            throw std::runtime_error("KmeansllCopyActiveSlotToSample: slot buffer not ready/mismatched.");
        }
        float* dst = static_cast<float*>(kmeansll_sample.ptr) +
                     static_cast<std::size_t>(dst_col0) * static_cast<std::size_t>(kmeansll_sample_ld);
        const float* src = static_cast<const float*>(kmeans_xnorm_slot[slot].ptr);
        const std::size_t bytes =
            sizeof(float) * static_cast<std::size_t>(kmeansll_sample_rows) * static_cast<std::size_t>(cols);
        ThrowIf(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream),
                "cudaMemcpyAsync (kmeansll sample D2D)");
    }

    DeviceVecF32View KmeansllEnsureBestDot(int n) {
        if (n <= 0) return DeviceVecF32View{};
        kmeansll_best_dot.Ensure(static_cast<std::size_t>(n) * sizeof(float));
        kmeansll_best_dot_n = n;
        // Keep d2/flags/indices/selected buffers sized consistently with n.
        kmeansll_d2.Ensure(static_cast<std::size_t>(n) * sizeof(float));
        kmeansll_flags.Ensure(static_cast<std::size_t>(n) * sizeof(std::uint8_t));
        kmeansll_selected.Ensure(static_cast<std::size_t>(n) * sizeof(int));
        kmeansll_num_selected.Ensure(sizeof(int));
        kmeansll_phi.Ensure(sizeof(float));
        if (kmeansll_indices_n != n) {
            kmeansll_indices.Ensure(static_cast<std::size_t>(n) * sizeof(int));
            kmeansll_indices_n = n;
            const dim3 block(256, 1, 1);
            const dim3 grid(static_cast<unsigned int>(std::max(1, (n + 255) / 256)), 1, 1);
            IotaI32Kernel<<<grid, block, 0, stream>>>(static_cast<int*>(kmeansll_indices.ptr), n);
            ThrowIf(cudaGetLastError(), "IotaI32Kernel launch (kmeansll)");
        }
        DeviceVecF32View out;
        out.ptr = static_cast<const float*>(kmeansll_best_dot.ptr);
        out.n = n;
        return out;
    }

    void KmeansllSetBestDot(float value) {
        if (kmeansll_best_dot_n <= 0 || !kmeansll_best_dot.ptr) return;
        const dim3 block(256, 1, 1);
        const dim3 grid(static_cast<unsigned int>(std::max(1, (kmeansll_best_dot_n + 255) / 256)), 1, 1);
        SetF32Kernel<<<grid, block, 0, stream>>>(static_cast<float*>(kmeansll_best_dot.ptr), kmeansll_best_dot_n, value);
        ThrowIf(cudaGetLastError(), "SetF32Kernel launch (kmeansll best_dot)");
    }

    void KmeansllUpdateBestDotMaxOffset(const DeviceVecF32View& tile_best, int offset) {
        if (kmeansll_best_dot_n <= 0 || !kmeansll_best_dot.ptr) {
            throw std::runtime_error("KmeansllUpdateBestDotMaxOffset: best_dot not initialized.");
        }
        if (!tile_best.ptr || tile_best.n <= 0) return;
        if (offset < 0 || offset + tile_best.n > kmeansll_best_dot_n) {
            throw std::runtime_error("KmeansllUpdateBestDotMaxOffset: offset out of range.");
        }
        const dim3 block(256, 1, 1);
        const dim3 grid(static_cast<unsigned int>(std::max(1, (tile_best.n + 255) / 256)), 1, 1);
        MaxOffsetKernel<<<grid, block, 0, stream>>>(
            static_cast<float*>(kmeansll_best_dot.ptr),
            tile_best.ptr,
            tile_best.n,
            offset);
        ThrowIf(cudaGetLastError(), "MaxOffsetKernel launch (kmeansll best_dot)");
    }

    float KmeansllComputePhiFromBestDot() {
        if (kmeansll_best_dot_n <= 0 || !kmeansll_best_dot.ptr) return 0.0f;
        const int n = kmeansll_best_dot_n;
        const dim3 block(256, 1, 1);
        const dim3 grid(static_cast<unsigned int>(std::max(1, (n + 255) / 256)), 1, 1);
        KmeansllComputeD2Kernel<<<grid, block, 0, stream>>>(
            static_cast<const float*>(kmeansll_best_dot.ptr),
            static_cast<float*>(kmeansll_d2.ptr),
            n);
        ThrowIf(cudaGetLastError(), "KmeansllComputeD2Kernel launch");

        std::size_t tmp_bytes = 0;
        cub::DeviceReduce::Sum(nullptr, tmp_bytes,
                               static_cast<const float*>(kmeansll_d2.ptr),
                               static_cast<float*>(kmeansll_phi.ptr),
                               n, stream);
        kmeansll_cub_tmp.Ensure(tmp_bytes);
        cub::DeviceReduce::Sum(kmeansll_cub_tmp.ptr, tmp_bytes,
                               static_cast<const float*>(kmeansll_d2.ptr),
                               static_cast<float*>(kmeansll_phi.ptr),
                               n, stream);
        ThrowIf(cudaGetLastError(), "cub::DeviceReduce::Sum (kmeansll phi)");
        float host_phi = 0.0f;
        ThrowIf(cudaMemcpyAsync(&host_phi, kmeansll_phi.ptr, sizeof(float), cudaMemcpyDeviceToHost, stream),
                "cudaMemcpyAsync (kmeansll phi D2H)");
        ThrowIf(cudaStreamSynchronize(stream), "cudaStreamSynchronize (kmeansll phi D2H)");
        return host_phi;
    }

    DeviceVecI32View KmeansllSampleCandidates(float oversample_l,
                                             float phi,
                                             std::uint64_t seed,
                                             int max_out,
                                             int* out_count_host) {
        if (out_count_host) *out_count_host = 0;
        kmeansll_last_selected = 0;
        if (kmeansll_best_dot_n <= 0 || !kmeansll_d2.ptr) return DeviceVecI32View{};
        const int n = kmeansll_best_dot_n;
        if (!(phi > 0.0f) || !(oversample_l >= 1.0f)) return DeviceVecI32View{};
        if (max_out == 0) return DeviceVecI32View{};
        const float inv_phi = 1.0f / phi;

        const dim3 block(256, 1, 1);
        const dim3 grid(static_cast<unsigned int>(std::max(1, (n + 255) / 256)), 1, 1);
        KmeansllSampleFlagsKernel<<<grid, block, 0, stream>>>(
            static_cast<const float*>(kmeansll_d2.ptr),
            static_cast<std::uint8_t*>(kmeansll_flags.ptr),
            n,
            inv_phi,
            oversample_l,
            seed);
        ThrowIf(cudaGetLastError(), "KmeansllSampleFlagsKernel launch");

        std::size_t tmp_bytes = 0;
        cub::DeviceSelect::Flagged(nullptr, tmp_bytes,
                                  static_cast<const int*>(kmeansll_indices.ptr),
                                  static_cast<const std::uint8_t*>(kmeansll_flags.ptr),
                                  static_cast<int*>(kmeansll_selected.ptr),
                                  static_cast<int*>(kmeansll_num_selected.ptr),
                                  n, stream);
        kmeansll_cub_tmp.Ensure(tmp_bytes);
        cub::DeviceSelect::Flagged(kmeansll_cub_tmp.ptr, tmp_bytes,
                                  static_cast<const int*>(kmeansll_indices.ptr),
                                  static_cast<const std::uint8_t*>(kmeansll_flags.ptr),
                                  static_cast<int*>(kmeansll_selected.ptr),
                                  static_cast<int*>(kmeansll_num_selected.ptr),
                                  n, stream);
        ThrowIf(cudaGetLastError(), "cub::DeviceSelect::Flagged (kmeansll)");

        int host_count = 0;
        ThrowIf(cudaMemcpyAsync(&host_count, kmeansll_num_selected.ptr, sizeof(int), cudaMemcpyDeviceToHost, stream),
                "cudaMemcpyAsync (kmeansll num_selected D2H)");
        ThrowIf(cudaStreamSynchronize(stream), "cudaStreamSynchronize (kmeansll num_selected D2H)");
        if (max_out > 0) {
            host_count = std::min(host_count, max_out);
        }
        kmeansll_last_selected = std::max(0, host_count);
        if (out_count_host) *out_count_host = kmeansll_last_selected;

        DeviceVecI32View out;
        out.ptr = static_cast<const int*>(kmeansll_selected.ptr);
        out.n = kmeansll_last_selected;
        return out;
    }

    void KmeansllDownloadLastSampled(std::vector<int>* out) {
        if (!out) return;
        out->clear();
        if (kmeansll_last_selected <= 0 || !kmeansll_selected.ptr) return;
        const std::size_t bytes = static_cast<std::size_t>(kmeansll_last_selected) * sizeof(int);
        kmeansll_selected_hpin.EnsureWithFlags(bytes, cudaHostAllocPortable);
        ThrowIf(cudaMemcpyAsync(kmeansll_selected_hpin.ptr,
                                kmeansll_selected.ptr,
                                bytes,
                                cudaMemcpyDeviceToHost,
                                stream_copy),
                "cudaMemcpyAsync (kmeansll selected D2H)");
        ThrowIf(cudaStreamSynchronize(stream_copy), "cudaStreamSynchronize (kmeansll selected D2H)");
        out->resize(static_cast<std::size_t>(kmeansll_last_selected));
        std::memcpy(out->data(), kmeansll_selected_hpin.ptr, bytes);
    }

    DeviceVecI32View KmeansllUploadIndicesHost(const int* idx_host, int m) {
        if (!idx_host || m <= 0) return DeviceVecI32View{};
        const std::size_t bytes = static_cast<std::size_t>(m) * sizeof(int);
        if (kmeansll_idxbuf_cap < m) {
            kmeansll_idxbuf.Ensure(bytes);
            kmeansll_idxbuf_cap = m;
        }
        ThrowIf(cudaMemcpyAsync(kmeansll_idxbuf.ptr, idx_host, bytes, cudaMemcpyHostToDevice, stream),
                "cudaMemcpyAsync (kmeansll idx H2D)");
        DeviceVecI32View out;
        out.ptr = static_cast<const int*>(kmeansll_idxbuf.ptr);
        out.n = m;
        return out;
    }

    DeviceMatF32View KmeansllGatherColumns(const DeviceMatF32View& X,
                                          const DeviceVecI32View& idx,
                                          int m) {
        if (!X.ptr || X.rows <= 0 || X.cols <= 0 || X.ld < X.rows) {
            throw std::runtime_error("KmeansllGatherColumns: invalid X view.");
        }
        if (!idx.ptr || idx.n < m || m <= 0) return DeviceMatF32View{};
        const int d = X.rows;
        const std::size_t n = static_cast<std::size_t>(d) * static_cast<std::size_t>(m);
        kmeansll_gather.Ensure(n * sizeof(float));
        kmeansll_gather_rows = d;
        kmeansll_gather_cols = m;
        kmeansll_gather_ld = d;

        const dim3 block(256, 1, 1);
        const std::int64_t grid_x64 =
            std::min<std::int64_t>((static_cast<std::int64_t>(n) + static_cast<std::int64_t>(block.x) - 1) /
                                       static_cast<std::int64_t>(block.x),
                                   65535);
        const dim3 grid(static_cast<unsigned int>(std::max<std::int64_t>(1, grid_x64)), 1, 1);
        GatherColsKernel1D<<<grid, block, 0, stream>>>(
            static_cast<float*>(kmeansll_gather.ptr),
            kmeansll_gather_ld,
            X.ptr,
            X.ld,
            idx.ptr,
            d,
            m);
        ThrowIf(cudaGetLastError(), "GatherColsKernel1D launch (kmeansll)");

        DeviceMatF32View out;
        out.ptr = static_cast<const float*>(kmeansll_gather.ptr);
        out.rows = d;
        out.cols = m;
        out.ld = kmeansll_gather_ld;
        return out;
    }

    void RvqSetPriorCodebooks(const std::vector<ColMajorMatrix<float>>& codebooks,
                              int upto_layer) {
        upto_layer = std::min(std::max(0, upto_layer), static_cast<int>(codebooks.size()));
        if (upto_layer <= 0) {
            rvq_codebooks.Reset();
            rvq_codebooks_rows = 0;
            rvq_upto_layer = 0;
            rvq_cb_offsets_f.clear();
            rvq_cb_K.clear();
            return;
        }
        const int d = codebooks[0].rows;
        if (d <= 0) {
            throw std::runtime_error("RvqSetPriorCodebooks: invalid d.");
        }
        std::size_t total_f = 0;
        rvq_cb_offsets_f.assign(static_cast<std::size_t>(upto_layer), 0);
        rvq_cb_K.assign(static_cast<std::size_t>(upto_layer), 0);
        for (int l = 0; l < upto_layer; ++l) {
            const auto& C = codebooks[static_cast<std::size_t>(l)];
            if (C.rows != d || C.cols <= 0 || C.data.empty()) {
                throw std::runtime_error("RvqSetPriorCodebooks: invalid codebook shape/data.");
            }
            rvq_cb_offsets_f[static_cast<std::size_t>(l)] = total_f;
            rvq_cb_K[static_cast<std::size_t>(l)] = C.cols;
            total_f += static_cast<std::size_t>(d) * static_cast<std::size_t>(C.cols);
        }
        rvq_codebooks_rows = d;
        rvq_upto_layer = upto_layer;

        rvq_codebooks.Ensure(total_f * sizeof(float));
        // Upload on the copy stream so subsequent slot uploads/projections (also on copy stream) are ordered.
        float* dst = static_cast<float*>(rvq_codebooks.ptr);
        for (int l = 0; l < upto_layer; ++l) {
            const auto& C = codebooks[static_cast<std::size_t>(l)];
            const std::size_t off = rvq_cb_offsets_f[static_cast<std::size_t>(l)];
            ThrowIf(cudaMemcpyAsync(dst + off,
                                    C.data.data(),
                                    static_cast<std::size_t>(d) * static_cast<std::size_t>(C.cols) * sizeof(float),
                                    cudaMemcpyHostToDevice,
                                    stream_copy),
                    "cudaMemcpyAsync (rvq codebook H2D)");
        }
    }

    void EnsureRvqCodesLayout(int upto_layer, int capacity_cols, const int* code_bytes_per_code) {
        if (upto_layer <= 0) return;
        if (capacity_cols <= 0) throw std::runtime_error("EnsureRvqCodesLayout: invalid capacity_cols.");
        if (!code_bytes_per_code) throw std::runtime_error("EnsureRvqCodesLayout: code_bytes_per_code is null.");

        bool bytes_changed = false;
        if (rvq_codes_upto_layer != upto_layer) {
            bytes_changed = true;
        }
        if (static_cast<int>(rvq_code_bytes_per_code.size()) != upto_layer) {
            bytes_changed = true;
        }
        if (!bytes_changed) {
            for (int l = 0; l < upto_layer; ++l) {
                if (rvq_code_bytes_per_code[static_cast<std::size_t>(l)] != code_bytes_per_code[l]) {
                    bytes_changed = true;
                    break;
                }
            }
        }
        if (bytes_changed) {
            rvq_code_bytes_per_code.assign(static_cast<std::size_t>(upto_layer), 1);
            for (int l = 0; l < upto_layer; ++l) {
                const int b = code_bytes_per_code[l];
                if (!(b == 1 || b == 2 || b == 4)) {
                    throw std::runtime_error("EnsureRvqCodesLayout: bytes_per_code must be 1/2/4.");
                }
                rvq_code_bytes_per_code[static_cast<std::size_t>(l)] = b;
            }
        }

        const bool layout_ok =
            (!bytes_changed) &&
            (rvq_codes_offsets_capacity_cols == capacity_cols) &&
            (static_cast<int>(rvq_code_offsets_bytes.size()) == upto_layer);
        if (!layout_ok) {
            rvq_codes_offsets_capacity_cols = capacity_cols;
            rvq_codes_upto_layer = upto_layer;
            rvq_code_offsets_bytes.assign(static_cast<std::size_t>(upto_layer), 0);
            std::size_t off = 0;
            for (int l = 0; l < upto_layer; ++l) {
                rvq_code_offsets_bytes[static_cast<std::size_t>(l)] = off;
                off += static_cast<std::size_t>(capacity_cols) *
                       static_cast<std::size_t>(rvq_code_bytes_per_code[static_cast<std::size_t>(l)]);
            }
            // Ensure both slots have enough space for the new layout/capacity.
            for (int s = 0; s < 2; ++s) {
                rvq_codes_slot[s].Ensure(off);
                rvq_codes_capacity_cols[s] = std::max(rvq_codes_capacity_cols[s], capacity_cols);
            }
        } else {
            // Ensure each slot buffer is large enough for this capacity.
            std::size_t total_bytes = 0;
            for (int l = 0; l < upto_layer; ++l) {
                total_bytes += static_cast<std::size_t>(capacity_cols) *
                               static_cast<std::size_t>(rvq_code_bytes_per_code[static_cast<std::size_t>(l)]);
            }
            for (int s = 0; s < 2; ++s) {
                if (capacity_cols > rvq_codes_capacity_cols[s]) {
                    rvq_codes_slot[s].Ensure(total_bytes);
                    rvq_codes_capacity_cols[s] = capacity_cols;
                }
            }
        }
    }

    void EnsureRvqMetaOnDevice(int upto_layer, const int* code_bytes_per_code) {
        if (upto_layer <= 0) return;
        if (!code_bytes_per_code) {
            throw std::runtime_error("EnsureRvqMetaOnDevice: code_bytes_per_code is null.");
        }
        const int meta_elems = 4 * upto_layer;
        if (meta_elems > 128) {
            throw std::runtime_error("EnsureRvqMetaOnDevice: upto_layer too large.");
        }
        rvq_meta.Ensure(static_cast<std::size_t>(meta_elems) * sizeof(int));
        std::array<int, 128> meta{};
        for (int l = 0; l < upto_layer; ++l) {
            meta[static_cast<std::size_t>(l)] = static_cast<int>(rvq_cb_offsets_f[static_cast<std::size_t>(l)]);
            meta[static_cast<std::size_t>(upto_layer + l)] = rvq_cb_K[static_cast<std::size_t>(l)];
            meta[static_cast<std::size_t>(2 * upto_layer + l)] =
                static_cast<int>(rvq_code_offsets_bytes[static_cast<std::size_t>(l)]);
            meta[static_cast<std::size_t>(3 * upto_layer + l)] = code_bytes_per_code[l];
        }
        ThrowIf(cudaMemcpyAsync(rvq_meta.ptr,
                                meta.data(),
                                static_cast<std::size_t>(meta_elems) * sizeof(int),
                                cudaMemcpyHostToDevice,
                                stream_copy),
                "cudaMemcpyAsync (rvq meta H2D)");
    }

    void RvqStageCodesBlockInCopyStreamAfterUpload(int slot,
                                                   int cols,
                                                   int upto_layer,
                                                   const void* const* codes_host,
                                                   const int* code_bytes_per_code) {
        if (slot != 0 && slot != 1) {
            throw std::runtime_error("RvqStageCodesBlockInCopyStreamAfterUpload: slot must be 0 or 1.");
        }
        if (upto_layer <= 0 || cols <= 0) {
            return;
        }
        if (!codes_host || !code_bytes_per_code) {
            throw std::runtime_error("RvqStageCodesBlockInCopyStreamAfterUpload: null codes args.");
        }
        if (upto_layer > rvq_upto_layer || upto_layer > static_cast<int>(rvq_cb_offsets_f.size())) {
            throw std::runtime_error("RvqStageCodesBlockInCopyStreamAfterUpload: codebooks not prepared.");
        }

        EnsureRvqCodesLayout(upto_layer, /*capacity_cols=*/cols, code_bytes_per_code);
        EnsureRvqMetaOnDevice(upto_layer, code_bytes_per_code);

        std::uint8_t* codes_dst = static_cast<std::uint8_t*>(rvq_codes_slot[slot].ptr);
        for (int l = 0; l < upto_layer; ++l) {
            const void* src = codes_host[l];
            if (!src) {
                throw std::runtime_error("RvqStageCodesBlockInCopyStreamAfterUpload: codes_host contains null.");
            }
            const int b = code_bytes_per_code[l];
            const std::size_t bytes = static_cast<std::size_t>(cols) * static_cast<std::size_t>(b);
            const std::size_t off = rvq_code_offsets_bytes[static_cast<std::size_t>(l)];
            ThrowIf(cudaMemcpyAsync(codes_dst + off, src, bytes, cudaMemcpyHostToDevice, stream_copy),
                    "cudaMemcpyAsync (rvq codes H2D)");
        }

        EnsureKmeansReadyEvents();
        ThrowIf(cudaEventRecord(kmeans_ready_ev[slot], stream_copy),
                "cudaEventRecord kmeans_ready_ev (rvq stage)");
    }

    void RvqProjectXnormBlockOnComputeStream(int slot,
                                            const DeviceMatF32View& Xnorm_slot,
                                            int cols,
                                            int upto_layer) {
        if (slot != 0 && slot != 1) {
            throw std::runtime_error("RvqProjectXnormBlockOnComputeStream: slot must be 0 or 1.");
        }
        if (upto_layer <= 0 || cols <= 0) {
            return;
        }
        if (!Xnorm_slot.ptr || Xnorm_slot.rows <= 0 || Xnorm_slot.ld < Xnorm_slot.rows) {
            throw std::runtime_error("RvqProjectXnormBlockOnComputeStream: invalid Xnorm_slot.");
        }
        if (upto_layer > rvq_upto_layer || upto_layer > static_cast<int>(rvq_cb_offsets_f.size())) {
            throw std::runtime_error("RvqProjectXnormBlockOnComputeStream: codebooks not prepared.");
        }
        if (Xnorm_slot.rows != rvq_codebooks_rows) {
            throw std::runtime_error("RvqProjectXnormBlockOnComputeStream: d mismatch.");
        }
        if (!rvq_codes_slot[slot].ptr || rvq_codes_capacity_cols[slot] < cols) {
            throw std::runtime_error("RvqProjectXnormBlockOnComputeStream: codes not staged for this block.");
        }

        constexpr int kWarpsPerBlock = 8;
        const dim3 block(32 * kWarpsPerBlock, 1, 1);
        const dim3 grid(static_cast<unsigned int>((cols + kWarpsPerBlock - 1) / kWarpsPerBlock), 1, 1);
        float* Xptr = const_cast<float*>(Xnorm_slot.ptr);
        const float* Cbase = static_cast<const float*>(rvq_codebooks.ptr);

        float* hatnorm2_out = nullptr;
        if (kmeans_collect_metrics) {
            rvq_hatnorm2_slot[slot].Ensure(static_cast<std::size_t>(cols) * sizeof(float));
            hatnorm2_out = static_cast<float*>(rvq_hatnorm2_slot[slot].ptr);
        }
        const int* cb_offsets = static_cast<const int*>(rvq_meta.ptr);
        const int* cb_K = cb_offsets + upto_layer;
        const int* d_code_offsets = cb_K + upto_layer;
        const int* d_code_bytes = d_code_offsets + upto_layer;
        if (kmeans_timing_enabled) {
            EnsureKmeansTimingEvents();
            HarvestKmeansRvqTimingSlot(slot, /*force_sync=*/false);
            ThrowIf(cudaEventRecord(kmeans_rvq_ev0[slot], stream), "cudaEventRecord (kmeans_rvq_ev0)");
        }

        RvqProjectAndNormalizeColsWarpKernel<kWarpsPerBlock><<<grid, block, 0, stream>>>(
            Xptr, Xnorm_slot.ld, Xnorm_slot.rows, cols,
            Cbase, rvq_codebooks_rows,
            cb_offsets, cb_K,
            static_cast<const std::uint8_t*>(rvq_codes_slot[slot].ptr),
            d_code_offsets, d_code_bytes,
            upto_layer,
            hatnorm2_out);
        ThrowIf(cudaGetLastError(), "RvqProjectAndNormalizeColsWarpKernel launch (compute)");

        if (kmeans_timing_enabled) {
            ThrowIf(cudaEventRecord(kmeans_rvq_ev1[slot], stream), "cudaEventRecord (kmeans_rvq_ev1)");
            kmeans_rvq_pending[slot] = true;
        }
    }

    void RvqProjectXnormBlockInCopyStreamAfterUpload(int slot,
                                                     const DeviceMatF32View& Xnorm_slot,
                                                     int cols,
                                                     int upto_layer,
                                                     const void* const* codes_host,
                                                     const int* code_bytes_per_code) {
        if (slot != 0 && slot != 1) {
            throw std::runtime_error("RvqProjectXnormBlockInCopyStreamAfterUpload: slot must be 0 or 1.");
        }
        if (upto_layer <= 0 || cols <= 0) {
            return;
        }
        if (!Xnorm_slot.ptr || Xnorm_slot.rows <= 0 || Xnorm_slot.ld < Xnorm_slot.rows) {
            throw std::runtime_error("RvqProjectXnormBlockInCopyStreamAfterUpload: invalid Xnorm_slot.");
        }
        if (!codes_host || !code_bytes_per_code) {
            throw std::runtime_error("RvqProjectXnormBlockInCopyStreamAfterUpload: null codes args.");
        }
        if (upto_layer > rvq_upto_layer || upto_layer > static_cast<int>(rvq_cb_offsets_f.size())) {
            throw std::runtime_error("RvqProjectXnormBlockInCopyStreamAfterUpload: codebooks not prepared.");
        }
        if (Xnorm_slot.rows != rvq_codebooks_rows) {
            throw std::runtime_error("RvqProjectXnormBlockInCopyStreamAfterUpload: d mismatch.");
        }

        // Stage codes/meta first, then run projection on the copy stream (legacy path).
        RvqStageCodesBlockInCopyStreamAfterUpload(slot, cols, upto_layer, codes_host, code_bytes_per_code);

        constexpr int kWarpsPerBlock = 8;
        const dim3 block(32 * kWarpsPerBlock, 1, 1);
        const dim3 grid(static_cast<unsigned int>((cols + kWarpsPerBlock - 1) / kWarpsPerBlock), 1, 1);
        float* Xptr = const_cast<float*>(Xnorm_slot.ptr);
        const float* Cbase = static_cast<const float*>(rvq_codebooks.ptr);
        float* hatnorm2_out = nullptr;
        if (kmeans_collect_metrics) {
            rvq_hatnorm2_slot[slot].Ensure(static_cast<std::size_t>(cols) * sizeof(float));
            hatnorm2_out = static_cast<float*>(rvq_hatnorm2_slot[slot].ptr);
        }
        const int* cb_offsets = static_cast<const int*>(rvq_meta.ptr);
        const int* cb_K = cb_offsets + upto_layer;
        const int* code_offsets = cb_K + upto_layer;
        const int* code_bytes = code_offsets + upto_layer;
        std::uint8_t* codes_dst = static_cast<std::uint8_t*>(rvq_codes_slot[slot].ptr);
        RvqProjectAndNormalizeColsWarpKernel<kWarpsPerBlock><<<grid, block, 0, stream_copy>>>(
            Xptr, Xnorm_slot.ld, Xnorm_slot.rows, cols,
            Cbase, rvq_codebooks_rows,
            cb_offsets, cb_K,
            codes_dst,
            code_offsets, code_bytes,
            upto_layer,
            hatnorm2_out);
        ThrowIf(cudaGetLastError(), "RvqProjectAndNormalizeColsWarpKernel launch (copy)");

        // Extend the slot-ready event so the compute stream waits for residual build too.
        EnsureKmeansReadyEvents();
        ThrowIf(cudaEventRecord(kmeans_ready_ev[slot], stream_copy),
                "cudaEventRecord kmeans_ready_ev (rvq)");
    }

    bool KmeansTryCacheXnormOnDevice(const ColMajorMatrix<float>& X,
                                     DeviceMatF32View* out_full) {
        if (out_full) {
            out_full->ptr = nullptr;
            out_full->rows = 0;
            out_full->cols = 0;
            out_full->ld = 0;
        }
        if (X.rows <= 0 || X.cols <= 0 || X.data.empty()) return false;
        const std::uint64_t stamp = CachedDeviceMatF32::Stamp(X);
        if (kmeans_xnorm_full_host_ptr == X.data.data() &&
            kmeans_xnorm_full_rows == X.rows &&
            kmeans_xnorm_full_cols == X.cols &&
            kmeans_xnorm_full_stamp == stamp &&
            kmeans_xnorm_full.ptr) {
            if (out_full) {
                out_full->ptr = static_cast<const float*>(kmeans_xnorm_full.ptr);
                out_full->rows = X.rows;
                out_full->cols = X.cols;
                out_full->ld = X.rows;
            }
            return true;
        }

        std::size_t free_b = 0;
        std::size_t total_b = 0;
        ThrowIf(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
        const std::size_t need_b =
            static_cast<std::size_t>(X.rows) * static_cast<std::size_t>(X.cols) * sizeof(float);
        // Keep a safety margin so later allocations (scores, sums, etc.) won't OOM.
        const std::size_t margin = static_cast<std::size_t>(0.35 * static_cast<double>(free_b));
        if (need_b + margin > free_b) {
            return false;
        }

        kmeans_xnorm_full.Ensure(need_b);
        ThrowIf(cudaMemcpyAsync(kmeans_xnorm_full.ptr,
                                X.data.data(),
                                need_b,
                                cudaMemcpyHostToDevice,
                                stream),
                "cudaMemcpyAsync (kmeans_xnorm_full H2D)");
        // Normalize all columns in-place once.
        constexpr int kWarpsPerBlock = 8;
        const dim3 block(32 * kWarpsPerBlock, 1, 1);
        const dim3 grid(static_cast<unsigned int>((X.cols + kWarpsPerBlock - 1) / kWarpsPerBlock), 1, 1);
        NormalizeColsWarpKernel<kWarpsPerBlock><<<grid, block, 0, stream>>>(
            static_cast<float*>(kmeans_xnorm_full.ptr), X.rows, X.rows, X.cols);
        ThrowIf(cudaGetLastError(), "NormalizeColsWarpKernel launch (kmeans_xnorm_full)");
        ThrowIf(cudaStreamSynchronize(stream), "cudaStreamSynchronize (kmeans_xnorm_full)");

        kmeans_xnorm_full_host_ptr = X.data.data();
        kmeans_xnorm_full_rows = X.rows;
        kmeans_xnorm_full_cols = X.cols;
        kmeans_xnorm_full_stamp = stamp;

        if (out_full) {
            out_full->ptr = static_cast<const float*>(kmeans_xnorm_full.ptr);
            out_full->rows = X.rows;
            out_full->cols = X.cols;
            out_full->ld = X.rows;
        }
        return true;
    }

    bool KmeansTryPinHostX(const ColMajorMatrix<float>& X, const float** out_pinned_base) {
        if (out_pinned_base) *out_pinned_base = nullptr;
        if (X.rows <= 0 || X.cols <= 0 || X.data.empty()) return false;
        const std::uint64_t stamp = CachedDeviceMatF32::Stamp(X);
        if (kmeans_X_pinned_host_ptr == X.data.data() &&
            kmeans_X_pinned_rows == X.rows &&
            kmeans_X_pinned_cols == X.cols &&
            kmeans_X_pinned_stamp == stamp &&
            kmeans_X_pinned.ptr) {
            if (out_pinned_base) *out_pinned_base = static_cast<const float*>(kmeans_X_pinned.ptr);
            return true;
        }
        // Safety cap: do not try to pin extremely large host matrices (can harm OS memory management).
        constexpr std::size_t kMaxPinnedBytes = 4ull * 1024ull * 1024ull * 1024ull;  // 4 GiB
        const std::size_t need_b =
            static_cast<std::size_t>(X.rows) * static_cast<std::size_t>(X.cols) * sizeof(float);
        if (need_b > kMaxPinnedBytes) {
            return false;
        }
        try {
            // Write-combined pinned memory is optimized for H2D bandwidth (we only write once, then DMA to GPU).
            kmeans_X_pinned.EnsureWithFlags(need_b, cudaHostAllocPortable | cudaHostAllocWriteCombined);
        } catch (...) {
            return false;
        }
        std::memcpy(kmeans_X_pinned.ptr, X.data.data(), need_b);
        kmeans_X_pinned_host_ptr = X.data.data();
        kmeans_X_pinned_rows = X.rows;
        kmeans_X_pinned_cols = X.cols;
        kmeans_X_pinned_stamp = stamp;
        if (out_pinned_base) *out_pinned_base = static_cast<const float*>(kmeans_X_pinned.ptr);
        return true;
    }

    void KmeansReleaseXnormCache() {
        // Full X_norm cache.
        kmeans_xnorm_full.Reset();
        kmeans_xnorm_full_host_ptr = nullptr;
        kmeans_xnorm_full_rows = 0;
        kmeans_xnorm_full_cols = 0;
        kmeans_xnorm_full_stamp = 0;

        // Per-block normalize scratch.
        kmeans_xnorm.Reset();
        kmeans_xnorm_rows = 0;
        kmeans_xnorm_cols = 0;
        kmeans_xnorm_ld = 0;
        kmeans_hpin.Reset();
        // Overlap slot buffers.
        kmeans_xnorm_slot[0].Reset();
        kmeans_xnorm_slot[1].Reset();
        kmeans_slot_rows[0] = 0;
        kmeans_slot_rows[1] = 0;
        kmeans_slot_cols[0] = 0;
        kmeans_slot_cols[1] = 0;
        kmeans_slot_ld[0] = 0;
        kmeans_slot_ld[1] = 0;
        kmeans_hpin_slot[0].Reset();
        kmeans_hpin_slot[1].Reset();

        // Accumulation scratch.
        kmeans_sum_x.Reset();
        kmeans_sum_w.Reset();
        kmeans_sum_rows = 0;
        kmeans_sum_k = 0;
        kmeans_sum_ld = 0;
        kmeans_assign.Reset();
        kmeans_assign_cols = 0;
        kmeans_weights.Reset();
        kmeans_weights_cols = 0;

        // Cost histogram scratch.
        kmeans_hist_u32.Reset();
        kmeans_hist_bins = 0;
        kmeans_hist_hpin.Reset();
        kmeans_hist_host_dst = nullptr;
        kmeans_hist_host_bins = 0;
        kmeans_hist_pending = false;

        // Global weights/totals and metric/mask scratch.
        kmeans_weights_all.Reset();
        kmeans_weights_all_n = 0;
        kmeans_anneal_totals.Reset();
        kmeans_anneal_totals_hpin.Reset();
        kmeans_anneal_pending = false;
        kmeans_metric_totals.Reset();
        kmeans_metric_totals_hpin.Reset();
        kmeans_metric_pending = false;
        kmeans_outlier_mask.Reset();
        kmeans_outlier_mask_cols = 0;
        kmeans_outlier_mask_hpin.Reset();
        kmeans_mask_pending = false;
        kmeans_mask_bytes = 0;

        // Per-block argmax buffers.
        kmeans_blk_assign.Reset();
        kmeans_blk_best.Reset();
        kmeans_blk_assign_hpin.Reset();
        kmeans_blk_best_hpin.Reset();
        kmeans_blk_cols = 0;
        kmeans_blk_assign_stage[0].Reset();
        kmeans_blk_assign_stage[1].Reset();
        kmeans_blk_assign_hpin_slot[0].Reset();
        kmeans_blk_assign_hpin_slot[1].Reset();
        kmeans_blk_assign_pending[0] = false;
        kmeans_blk_assign_pending[1] = false;
        kmeans_blk_assign_slot_cols[0] = 0;
        kmeans_blk_assign_slot_cols[1] = 0;

        // Pinned host mirror.
        kmeans_X_pinned.Reset();
        kmeans_X_pinned_host_ptr = nullptr;
        kmeans_X_pinned_rows = 0;
        kmeans_X_pinned_cols = 0;
        kmeans_X_pinned_stamp = 0;

        // Generic GEMM / argmax scratch grown by large-K kmeans passes.
        dB.Reset();
        du8.Reset();
        dtmp.Reset();
        d_assign.Reset();
        d_best.Reset();
        d_cached.Reset();
        cached_host_ptr = nullptr;
        cached_rows = 0;
        cached_cols = 0;
        cached_host_stamp = 0;
        d_out.Reset();

        // Segmented-reduce / hier2 scratch.
        kmeans_sort_keys_in.Reset();
        kmeans_sort_vals_in.Reset();
        kmeans_sort_keys_out.Reset();
        kmeans_sort_vals_out.Reset();
        kmeans_unique_keys.Reset();
        kmeans_counts.Reset();
        kmeans_num_segments.Reset();
        kmeans_seg_offsets.Reset();
        kmeans_cub_tmp.Reset();
        kmeans_sort_capacity_cols = 0;
        kmeans_segment_capacity_k = 0;
        kmeans_hier2_topk_idx.Reset();
        kmeans_hier2_topk_val.Reset();

        // RVQ residual projection scratch used by residual-layer streaming kmeans.
        rvq_codebooks.Reset();
        rvq_codebooks_rows = 0;
        rvq_upto_layer = 0;
        rvq_cb_offsets_f.clear();
        rvq_cb_K.clear();
        rvq_meta.Reset();
        for (int s = 0; s < 2; ++s) {
            rvq_codes_slot[s].Reset();
            rvq_codes_capacity_cols[s] = 0;
        }
        rvq_codes_offsets_capacity_cols = 0;
        rvq_codes_upto_layer = 0;
        rvq_code_bytes_per_code.clear();
        rvq_code_offsets_bytes.clear();

        // Device matrix caches can retain large stale uploads across RVQ layers.
        A_cache.buf.Reset();
        A_cache.host_ptr = nullptr;
        A_cache.rows = 0;
        A_cache.cols = 0;
        A_cache.host_stamp = 0;
        B_cache.buf.Reset();
        B_cache.host_ptr = nullptr;
        B_cache.rows = 0;
        B_cache.cols = 0;
        B_cache.host_stamp = 0;
        R_cache.buf.Reset();
        R_cache.host_ptr = nullptr;
        R_cache.rows = 0;
        R_cache.cols = 0;
        R_cache.host_stamp = 0;
        X_cache.buf.Reset();
        X_cache.host_ptr = nullptr;
        X_cache.rows = 0;
        X_cache.cols = 0;
        X_cache.host_stamp = 0;
    }

    DeviceMatF32View KmeansEnsurePartialXnormCache(int rows, int cols) {
        if (rows <= 0 || cols <= 0) return DeviceMatF32View{};
        const std::size_t bytes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) * sizeof(float);
        kmeans_xnorm_partial.Ensure(bytes);
        kmeans_xnorm_partial_rows = rows;
        kmeans_xnorm_partial_cols = cols;
        kmeans_xnorm_partial_ld = rows;
        kmeans_norm2_partial.Ensure(static_cast<std::size_t>(cols) * sizeof(float));
        kmeans_norm2_partial_cols = cols;
        DeviceMatF32View out;
        out.ptr = static_cast<const float*>(kmeans_xnorm_partial.ptr);
        out.rows = rows;
        out.cols = cols;
        out.ld = rows;
        return out;
    }

    DeviceMatF32View KmeansPartialXnormCacheView() const {
        if (!kmeans_xnorm_partial.ptr || kmeans_xnorm_partial_rows <= 0 || kmeans_xnorm_partial_cols <= 0) {
            return DeviceMatF32View{};
        }
        DeviceMatF32View out;
        out.ptr = static_cast<const float*>(kmeans_xnorm_partial.ptr);
        out.rows = kmeans_xnorm_partial_rows;
        out.cols = kmeans_xnorm_partial_cols;
        out.ld = kmeans_xnorm_partial_ld;
        return out;
    }

    void KmeansCopyActiveSlotToPartialCache(int slot, int dst_col0, int cols) {
        if (slot != 0 && slot != 1) throw std::runtime_error("KmeansCopyActiveSlotToPartialCache: slot must be 0 or 1.");
        if (cols <= 0) return;
        if (!kmeans_xnorm_partial.ptr || kmeans_xnorm_partial_rows <= 0 || kmeans_xnorm_partial_cols <= 0) {
            throw std::runtime_error("KmeansCopyActiveSlotToPartialCache: partial cache not initialized.");
        }
        if (dst_col0 < 0 || dst_col0 + cols > kmeans_xnorm_partial_cols) {
            throw std::runtime_error("KmeansCopyActiveSlotToPartialCache: dst range out of bounds.");
        }
        if (!kmeans_xnorm_slot[slot].ptr || kmeans_slot_rows[slot] != kmeans_xnorm_partial_rows || kmeans_slot_cols[slot] < cols) {
            throw std::runtime_error("KmeansCopyActiveSlotToPartialCache: slot buffer not ready/mismatched.");
        }
        float* dst = static_cast<float*>(kmeans_xnorm_partial.ptr) +
                     static_cast<std::size_t>(dst_col0) * static_cast<std::size_t>(kmeans_xnorm_partial_ld);
        const float* src = static_cast<const float*>(kmeans_xnorm_slot[slot].ptr);
        const std::size_t bytes =
            sizeof(float) * static_cast<std::size_t>(kmeans_xnorm_partial_rows) * static_cast<std::size_t>(cols);
        ThrowIf(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream_copy),
                "cudaMemcpyAsync (kmeans partial xnorm D2D)");

        if (kmeans_collect_metrics) {
            if (!kmeans_norm2_partial.ptr || kmeans_norm2_partial_cols < dst_col0 + cols) {
                throw std::runtime_error("KmeansCopyActiveSlotToPartialCache: norm2 partial not initialized/mismatched.");
            }
            if (!kmeans_norm2_slot[slot].ptr || kmeans_slot_cols[slot] < cols) {
                throw std::runtime_error("KmeansCopyActiveSlotToPartialCache: norm2 slot missing/mismatched.");
            }
            float* dstn = static_cast<float*>(kmeans_norm2_partial.ptr) + dst_col0;
            const float* srcn = static_cast<const float*>(kmeans_norm2_slot[slot].ptr);
            ThrowIf(cudaMemcpyAsync(dstn, srcn, static_cast<std::size_t>(cols) * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream_copy),
                    "cudaMemcpyAsync (kmeans partial norm2 D2D)");
        }
    }

    DeviceMatF32View KmeansCopyPartialCacheToSlot(int slot, int src_col0, int cols) {
        if (slot != 0 && slot != 1) throw std::runtime_error("KmeansCopyPartialCacheToSlot: slot must be 0 or 1.");
        if (cols <= 0) return DeviceMatF32View{};
        if (!kmeans_xnorm_partial.ptr || kmeans_xnorm_partial_rows <= 0 || kmeans_xnorm_partial_cols <= 0) {
            throw std::runtime_error("KmeansCopyPartialCacheToSlot: partial cache not initialized.");
        }
        if (src_col0 < 0 || src_col0 + cols > kmeans_xnorm_partial_cols) {
            throw std::runtime_error("KmeansCopyPartialCacheToSlot: src range out of bounds.");
        }
        EnsureKmeansReadyEvents();
        // Ensure we don't reuse this slot while the compute stream is still reading it.
        ThrowIf(cudaStreamWaitEvent(stream_copy, kmeans_free_ev[slot], 0),
                "cudaStreamWaitEvent (kmeans_free_ev partial)");

        const int rows = kmeans_xnorm_partial_rows;
        
        // Optimization: For Layer 0 (rvq_upto_layer <= 0), we can avoid the D2D copy
        // and just point the slot view directly into the partial cache.
        // For Layer > 0, we MUST copy because the slot will be modified in-place (residual projection).
        if (rvq_upto_layer <= 0) {
            // "Zero-copy" path: point directly to cache.
            const float* src = static_cast<const float*>(kmeans_xnorm_partial.ptr) +
                               static_cast<std::size_t>(src_col0) * static_cast<std::size_t>(kmeans_xnorm_partial_ld);
            
            // We still need to record the ready event to satisfy the ping-pong protocol
            // (compute stream waiting for "upload" to finish, which here is just a no-op).
            ThrowIf(cudaEventRecord(kmeans_ready_ev[slot], stream_copy),
                    "cudaEventRecord kmeans_ready_ev (partial cache zero-copy)");

            // We still copy norm2 because it's small and kernels expect it in kmeans_norm2_slot.
            if (kmeans_collect_metrics) {
                kmeans_norm2_slot[slot].Ensure(static_cast<std::size_t>(cols) * sizeof(float));
                const float* srcn = static_cast<const float*>(kmeans_norm2_partial.ptr) + src_col0;
                float* dstn = static_cast<float*>(kmeans_norm2_slot[slot].ptr);
                ThrowIf(cudaMemcpyAsync(dstn, srcn, static_cast<std::size_t>(cols) * sizeof(float),
                                        cudaMemcpyDeviceToDevice, stream_copy),
                        "cudaMemcpyAsync (kmeans partial norm2 -> slot)");
            }
            
            DeviceMatF32View out;
            out.ptr = src;
            out.rows = rows;
            out.cols = cols;
            out.ld = kmeans_xnorm_partial_ld;
            return out;
        }

        // Standard path: Copy to slot (scratchpad).
        const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
        kmeans_xnorm_slot[slot].Ensure(n * sizeof(float));
        kmeans_slot_rows[slot] = rows;
        kmeans_slot_cols[slot] = cols;
        kmeans_slot_ld[slot] = rows;
        if (kmeans_collect_metrics) {
            kmeans_norm2_slot[slot].Ensure(static_cast<std::size_t>(cols) * sizeof(float));
            rvq_hatnorm2_slot[slot].Ensure(static_cast<std::size_t>(cols) * sizeof(float));
        }

        const float* src = static_cast<const float*>(kmeans_xnorm_partial.ptr) +
                           static_cast<std::size_t>(src_col0) * static_cast<std::size_t>(kmeans_xnorm_partial_ld);
        float* dst = static_cast<float*>(kmeans_xnorm_slot[slot].ptr);
        const std::size_t bytes = n * sizeof(float);
        ThrowIf(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream_copy),
                "cudaMemcpyAsync (kmeans partial xnorm -> slot)");

        if (kmeans_collect_metrics) {
            const float* srcn = static_cast<const float*>(kmeans_norm2_partial.ptr) + src_col0;
            float* dstn = static_cast<float*>(kmeans_norm2_slot[slot].ptr);
            ThrowIf(cudaMemcpyAsync(dstn, srcn, static_cast<std::size_t>(cols) * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream_copy),
                    "cudaMemcpyAsync (kmeans partial norm2 -> slot)");
        }

        // Record the usual slot-ready event so compute stream can wait via KmeansComputeWaitForUpload(slot).
        ThrowIf(cudaEventRecord(kmeans_ready_ev[slot], stream_copy),
                "cudaEventRecord kmeans_ready_ev (partial cache)");
        DeviceMatF32View out;
        out.ptr = static_cast<const float*>(kmeans_xnorm_slot[slot].ptr);
        out.rows = rows;
        out.cols = cols;
        out.ld = rows;
        return out;
    }

    void KmeansSetActiveSlot(int slot) { kmeans_active_slot = slot; }

    void KmeansReleasePartialXnormCache() {
        kmeans_xnorm_partial.Reset();
        kmeans_xnorm_partial_rows = 0;
        kmeans_xnorm_partial_cols = 0;
        kmeans_xnorm_partial_ld = 0;
        kmeans_norm2_partial.Reset();
        kmeans_norm2_partial_cols = 0;
    }

    void KmeansEnsureBlockArgmax(int cols) {
        if (cols <= 0) {
            kmeans_blk_cols = 0;
            return;
        }
        kmeans_blk_assign.Ensure(static_cast<std::size_t>(cols) * sizeof(int));
        kmeans_blk_best.Ensure(static_cast<std::size_t>(cols) * sizeof(float));
        kmeans_blk_cols = cols;
    }

    void KmeansCopyTileArgmaxToBlock(const DeviceVecI32View& tile_assign,
                                     const DeviceVecF32View& tile_best,
                                     int col_base) {
        if (!tile_assign.ptr || !tile_best.ptr || tile_assign.n <= 0 || tile_best.n != tile_assign.n) {
            throw std::runtime_error("KmeansCopyTileArgmaxToBlock: invalid tile views.");
        }
        if (kmeans_blk_cols <= 0 || !kmeans_blk_assign.ptr || !kmeans_blk_best.ptr) {
            throw std::runtime_error("KmeansCopyTileArgmaxToBlock: block buffers not initialized.");
        }
        if (col_base < 0 || col_base + tile_assign.n > kmeans_blk_cols) {
            throw std::runtime_error("KmeansCopyTileArgmaxToBlock: tile range out of bounds.");
        }
        ThrowIf(cudaMemcpyAsync(static_cast<char*>(kmeans_blk_assign.ptr) +
                                    static_cast<std::size_t>(col_base) * sizeof(int),
                                tile_assign.ptr,
                                static_cast<std::size_t>(tile_assign.n) * sizeof(int),
                                cudaMemcpyDeviceToDevice,
                                stream),
                "cudaMemcpyAsync (kmeans_blk_assign D2D)");
        ThrowIf(cudaMemcpyAsync(static_cast<char*>(kmeans_blk_best.ptr) +
                                    static_cast<std::size_t>(col_base) * sizeof(float),
                                tile_best.ptr,
                                static_cast<std::size_t>(tile_best.n) * sizeof(float),
                                cudaMemcpyDeviceToDevice,
                                stream),
                "cudaMemcpyAsync (kmeans_blk_best D2D)");
    }

    DeviceVecI32View KmeansBlockAssignView() const {
        DeviceVecI32View out;
        out.ptr = static_cast<const int*>(kmeans_blk_assign.ptr);
        out.n = kmeans_blk_cols;
        return out;
    }

    DeviceVecF32View KmeansBlockBestView() const {
        DeviceVecF32View out;
        out.ptr = static_cast<const float*>(kmeans_blk_best.ptr);
        out.n = kmeans_blk_cols;
        return out;
    }

    void KmeansDownloadBlockArgmax(std::vector<int>* assign,
                                   std::vector<float>* best) {
        if (!assign || !best) {
            throw std::runtime_error("KmeansDownloadBlockArgmax: outputs must be non-null.");
        }
        assign->clear();
        best->clear();
        if (kmeans_blk_cols <= 0 || !kmeans_blk_assign.ptr || !kmeans_blk_best.ptr) {
            return;
        }
        const int cols = kmeans_blk_cols;
        assign->resize(static_cast<std::size_t>(cols));
        best->resize(static_cast<std::size_t>(cols));

        kmeans_blk_assign_hpin.EnsureWithFlags(static_cast<std::size_t>(cols) * sizeof(int),
                                               cudaHostAllocPortable);
        kmeans_blk_best_hpin.EnsureWithFlags(static_cast<std::size_t>(cols) * sizeof(float),
                                             cudaHostAllocPortable);

        ThrowIf(cudaMemcpyAsync(kmeans_blk_assign_hpin.ptr,
                                kmeans_blk_assign.ptr,
                                static_cast<std::size_t>(cols) * sizeof(int),
                                cudaMemcpyDeviceToHost,
                                stream),
                "cudaMemcpyAsync (kmeans_blk_assign D2H)");
        ThrowIf(cudaMemcpyAsync(kmeans_blk_best_hpin.ptr,
                                kmeans_blk_best.ptr,
                                static_cast<std::size_t>(cols) * sizeof(float),
                                cudaMemcpyDeviceToHost,
                                stream),
                "cudaMemcpyAsync (kmeans_blk_best D2H)");
        ThrowIf(cudaStreamSynchronize(stream), "cudaStreamSynchronize (KmeansDownloadBlockArgmax)");
        std::memcpy(assign->data(), kmeans_blk_assign_hpin.ptr,
                    static_cast<std::size_t>(cols) * sizeof(int));
        std::memcpy(best->data(), kmeans_blk_best_hpin.ptr,
                    static_cast<std::size_t>(cols) * sizeof(float));
    }

    void KmeansDownloadBlockBestDot(std::vector<float>* best) {
        if (!best) {
            throw std::runtime_error("KmeansDownloadBlockBestDot: output must be non-null.");
        }
        best->clear();
        if (kmeans_blk_cols <= 0 || !kmeans_blk_best.ptr) {
            return;
        }
        const int cols = kmeans_blk_cols;
        best->resize(static_cast<std::size_t>(cols));

        kmeans_blk_best_hpin.EnsureWithFlags(static_cast<std::size_t>(cols) * sizeof(float),
                                             cudaHostAllocPortable);
        ThrowIf(cudaMemcpyAsync(kmeans_blk_best_hpin.ptr,
                                kmeans_blk_best.ptr,
                                static_cast<std::size_t>(cols) * sizeof(float),
                                cudaMemcpyDeviceToHost,
                                stream),
                "cudaMemcpyAsync (kmeans_blk_best D2H)");
        ThrowIf(cudaStreamSynchronize(stream), "cudaStreamSynchronize (KmeansDownloadBlockBestDot)");
        std::memcpy(best->data(), kmeans_blk_best_hpin.ptr,
                    static_cast<std::size_t>(cols) * sizeof(float));
    }

    void KmeansDownloadBlockAssignAsync(int slot) {
        if (slot < 0 || slot >= 2) {
            throw std::runtime_error("KmeansDownloadBlockAssignAsync: slot out of range.");
        }
        if (kmeans_blk_assign_pending[slot]) {
            throw std::runtime_error("KmeansDownloadBlockAssignAsync: pending async copy; call Sync(slot) first.");
        }
        if (kmeans_blk_cols <= 0 || !kmeans_blk_assign.ptr) {
            return;
        }
        EnsureKmeansBlockAssignEvents();

        const int cols = kmeans_blk_cols;
        const std::size_t bytes = static_cast<std::size_t>(cols) * sizeof(int);
        kmeans_blk_assign_stage[slot].Ensure(bytes);
        kmeans_blk_assign_hpin_slot[slot].EnsureWithFlags(bytes, cudaHostAllocPortable);

        // Stage the current block assignments into a per-slot device buffer on the compute stream.
        // This prevents races because `kmeans_blk_assign` is reused for subsequent blocks.
        ThrowIf(cudaMemcpyAsync(kmeans_blk_assign_stage[slot].ptr,
                                kmeans_blk_assign.ptr,
                                bytes,
                                cudaMemcpyDeviceToDevice,
                                stream),
                "cudaMemcpyAsync (kmeans_blk_assign D2D stage)");
        ThrowIf(cudaEventRecord(kmeans_blk_assign_ready_ev[slot], stream),
                "cudaEventRecord (kmeans_blk_assign_ready_ev)");

        // Copy staged assignments to host on the copy stream.
        ThrowIf(cudaStreamWaitEvent(stream_copy, kmeans_blk_assign_ready_ev[slot], 0),
                "cudaStreamWaitEvent (kmeans_blk_assign_ready_ev)");
        if (kmeans_timing_enabled) {
            EnsureKmeansTimingEvents();
            ThrowIf(cudaEventRecord(kmeans_assign_d2h_ev0[slot], stream_copy),
                    "cudaEventRecord (kmeans_assign_d2h_ev0)");
            kmeans_assign_d2h_pending[slot] = true;
        }
        ThrowIf(cudaMemcpyAsync(kmeans_blk_assign_hpin_slot[slot].ptr,
                                kmeans_blk_assign_stage[slot].ptr,
                                bytes,
                                cudaMemcpyDeviceToHost,
                                stream_copy),
                "cudaMemcpyAsync (kmeans_blk_assign D2H async)");
        if (kmeans_timing_enabled && kmeans_assign_d2h_pending[slot]) {
            ThrowIf(cudaEventRecord(kmeans_blk_assign_done_ev_timing[slot], stream_copy),
                    "cudaEventRecord (kmeans_blk_assign_done_ev_timing)");
        }
        ThrowIf(cudaEventRecord(kmeans_blk_assign_done_ev[slot], stream_copy),
                "cudaEventRecord (kmeans_blk_assign_done_ev)");

        kmeans_blk_assign_slot_cols[slot] = cols;
        kmeans_blk_assign_pending[slot] = true;
    }

    void KmeansDownloadBlockAssignSync(int slot, const int** out_assign_i32, int* out_cols) {
        if (slot < 0 || slot >= 2) {
            throw std::runtime_error("KmeansDownloadBlockAssignSync: slot out of range.");
        }
        if (out_assign_i32) *out_assign_i32 = nullptr;
        if (out_cols) *out_cols = 0;
        if (!kmeans_blk_assign_pending[slot]) {
            return;
        }
        ThrowIf(cudaEventSynchronize(kmeans_blk_assign_done_ev[slot]),
                "cudaEventSynchronize (kmeans_blk_assign_done_ev)");
        if (kmeans_timing_enabled && kmeans_assign_d2h_pending[slot]) {
            HarvestKmeansAssignD2HTimingSlot(slot);
        }
        if (out_assign_i32) {
            *out_assign_i32 = static_cast<const int*>(kmeans_blk_assign_hpin_slot[slot].ptr);
        }
        if (out_cols) {
            *out_cols = kmeans_blk_assign_slot_cols[slot];
        }
        kmeans_blk_assign_slot_cols[slot] = 0;
        kmeans_blk_assign_pending[slot] = false;
    }

    DeviceVecF32View KmeansUploadBlockWeights(const std::vector<float>& weights) {
        if (kmeans_blk_cols <= 0) {
            throw std::runtime_error("KmeansUploadBlockWeights: block buffers not initialized.");
        }
        const int cols = kmeans_blk_cols;
        if (static_cast<int>(weights.size()) != cols) {
            throw std::runtime_error("KmeansUploadBlockWeights: weights size mismatch.");
        }
        kmeans_weights.Ensure(static_cast<std::size_t>(cols) * sizeof(float));
        kmeans_weights_cols = cols;
        ThrowIf(cudaMemcpyAsync(kmeans_weights.ptr,
                                weights.data(),
                                static_cast<std::size_t>(cols) * sizeof(float),
                                cudaMemcpyHostToDevice,
                                stream),
                "cudaMemcpyAsync (kmeans_weights H2D block)");
        DeviceVecF32View out;
        out.ptr = static_cast<const float*>(kmeans_weights.ptr);
        out.n = cols;
        return out;
    }

    bool KmeansInitAnnealWeightsAll(std::int64_t n, float initial_weight) {
        if (n <= 0) {
            kmeans_weights_all.Reset();
            kmeans_weights_all_n = 0;
            return true;
        }
        const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
        if (kmeans_weights_all_n != n || !kmeans_weights_all.ptr || kmeans_weights_all.bytes < bytes) {
            try {
                kmeans_weights_all.Ensure(bytes);
            } catch (...) {
                kmeans_weights_all.Reset();
                kmeans_weights_all_n = 0;
                return false;
            }
            kmeans_weights_all_n = n;
        }
        float* d_w = static_cast<float*>(kmeans_weights_all.ptr);
        const dim3 block(256, 1, 1);
        const std::int64_t grid_x64 =
            std::min<std::int64_t>((n + static_cast<std::int64_t>(block.x) - 1) / static_cast<std::int64_t>(block.x),
                                   65535);
        const dim3 grid(static_cast<unsigned int>(std::max<std::int64_t>(1, grid_x64)), 1, 1);
        FillF32Kernel<<<grid, block, 0, stream>>>(d_w, n, initial_weight);
        ThrowIf(cudaGetLastError(), "FillF32Kernel launch (KmeansInitAnnealWeightsAll)");
        return true;
    }

    DeviceVecF32View KmeansAnnealWeightsBlockView(std::int64_t col0, int cols) const {
        if (!kmeans_weights_all.ptr || kmeans_weights_all_n <= 0) {
            throw std::runtime_error("KmeansAnnealWeightsBlockView: global weights not initialized.");
        }
        if (col0 < 0 || cols < 0 || col0 + static_cast<std::int64_t>(cols) > kmeans_weights_all_n) {
            throw std::runtime_error("KmeansAnnealWeightsBlockView: range out of bounds.");
        }
        DeviceVecF32View out;
        out.ptr = static_cast<const float*>(kmeans_weights_all.ptr) + col0;
        out.n = cols;
        return out;
    }

    void KmeansResetAnnealTotals() {
        const std::size_t bytes = 2 * sizeof(double);
        if (!kmeans_anneal_totals.ptr || kmeans_anneal_totals.bytes < bytes) {
            kmeans_anneal_totals.Reset();
            kmeans_anneal_totals.Ensure(bytes);
        }
        ThrowIf(cudaMemsetAsync(kmeans_anneal_totals.ptr, 0, bytes, stream),
                "cudaMemsetAsync (kmeans_anneal_totals)");
    }

    void KmeansAnnealUpdateWeightsAndTotalsFromBlockBest(std::int64_t col0,
                                                         int cols,
                                                         float threshold,
                                                         float annealed_factor,
                                                         float min_weight) {
        if (!kmeans_weights_all.ptr || kmeans_weights_all_n <= 0) {
            throw std::runtime_error("KmeansAnnealUpdateWeightsAndTotalsFromBlockBest: global weights not initialized.");
        }
        if (kmeans_blk_cols != cols || cols <= 0 || !kmeans_blk_best.ptr) {
            throw std::runtime_error("KmeansAnnealUpdateWeightsAndTotalsFromBlockBest: block buffers not initialized/mismatched.");
        }
        if (col0 < 0 || col0 + static_cast<std::int64_t>(cols) > kmeans_weights_all_n) {
            throw std::runtime_error("KmeansAnnealUpdateWeightsAndTotalsFromBlockBest: block range out of bounds.");
        }
        if (!kmeans_anneal_totals.ptr) {
            KmeansResetAnnealTotals();
        }

        const dim3 block(256, 1, 1);
        const std::int64_t grid_x64 =
            std::min<std::int64_t>((static_cast<std::int64_t>(cols) + static_cast<std::int64_t>(block.x) - 1) /
                                       static_cast<std::int64_t>(block.x),
                                   4096);
        const dim3 grid(static_cast<unsigned int>(std::max<std::int64_t>(1, grid_x64)), 1, 1);
        KmeansAnnealUpdateWeightsAndTotalsFromBestDotKernel<<<grid, block, 0, stream>>>(
            static_cast<const float*>(kmeans_blk_best.ptr),
            static_cast<float*>(kmeans_weights_all.ptr),
            col0,
            cols,
            threshold,
            annealed_factor,
            min_weight,
            static_cast<double*>(kmeans_anneal_totals.ptr));
        ThrowIf(cudaGetLastError(), "KmeansAnnealUpdateWeightsAndTotalsFromBestDotKernel launch");
    }

    void KmeansAccumulateAnnealTotalsFromBlockBestDotAndWeights(const DeviceVecF32View& d_weights) {
        if (!d_weights.ptr || d_weights.n <= 0) return;
        if (kmeans_blk_cols != d_weights.n || !kmeans_blk_best.ptr) {
            throw std::runtime_error("KmeansAccumulateAnnealTotalsFromBlockBestDotAndWeights: block buffers not initialized/mismatched.");
        }
        if (!kmeans_anneal_totals.ptr) {
            KmeansResetAnnealTotals();
        }
        const int cols = d_weights.n;
        const dim3 block(256, 1, 1);
        const std::int64_t grid_x64 =
            std::min<std::int64_t>((static_cast<std::int64_t>(cols) + static_cast<std::int64_t>(block.x) - 1) /
                                       static_cast<std::int64_t>(block.x),
                                   4096);
        const dim3 grid(static_cast<unsigned int>(std::max<std::int64_t>(1, grid_x64)), 1, 1);
        KmeansAccumulateAnnealTotalsFromBestDotAndWeightsKernel<<<grid, block, 0, stream>>>(
            static_cast<const float*>(kmeans_blk_best.ptr),
            d_weights.ptr,
            cols,
            static_cast<double*>(kmeans_anneal_totals.ptr));
        ThrowIf(cudaGetLastError(), "KmeansAccumulateAnnealTotalsFromBestDotAndWeightsKernel launch");
    }

    void KmeansDownloadAnnealTotals(double* total_w, double* total_cost) {
        KmeansDownloadAnnealTotalsAsync();
        KmeansSyncAnnealTotalsDownload(total_w, total_cost);
    }

    void KmeansDownloadAnnealTotalsAsync() {
        if (kmeans_anneal_pending) {
            throw std::runtime_error("KmeansDownloadAnnealTotalsAsync: pending async copy; call Sync() first.");
        }
        if (!kmeans_anneal_totals.ptr) return;
        EnsureKmeansAnnealEvents();
        kmeans_anneal_totals_hpin.EnsureWithFlags(2 * sizeof(double), cudaHostAllocPortable);
        ThrowIf(cudaEventRecord(kmeans_anneal_ready_ev, stream), "cudaEventRecord (kmeans_anneal_ready_ev)");
        ThrowIf(cudaStreamWaitEvent(stream_copy, kmeans_anneal_ready_ev, 0),
                "cudaStreamWaitEvent (kmeans_anneal_ready_ev)");
        ThrowIf(cudaMemcpyAsync(kmeans_anneal_totals_hpin.ptr,
                                kmeans_anneal_totals.ptr,
                                2 * sizeof(double),
                                cudaMemcpyDeviceToHost,
                                stream_copy),
                "cudaMemcpyAsync (kmeans_anneal_totals D2H async)");
        ThrowIf(cudaEventRecord(kmeans_anneal_done_ev, stream_copy), "cudaEventRecord (kmeans_anneal_done_ev)");
        kmeans_anneal_pending = true;
    }

    void KmeansSyncAnnealTotalsDownload(double* total_w, double* total_cost) {
        if (!kmeans_anneal_pending) return;
        ThrowIf(cudaEventSynchronize(kmeans_anneal_done_ev), "cudaEventSynchronize (kmeans_anneal_done_ev)");
        const double* h = static_cast<const double*>(kmeans_anneal_totals_hpin.ptr);
        if (total_w) *total_w = h[0];
        if (total_cost) *total_cost = h[1];
        kmeans_anneal_pending = false;
    }

    void KmeansResetMetricTotals() {
        const std::size_t bytes = 2 * sizeof(double);
        if (!kmeans_metric_totals.ptr || kmeans_metric_totals.bytes < bytes) {
            kmeans_metric_totals.Reset();
            kmeans_metric_totals.Ensure(bytes);
        }
        ThrowIf(cudaMemsetAsync(kmeans_metric_totals.ptr, 0, bytes, stream),
                "cudaMemsetAsync (kmeans_metric_totals)");
    }

    void KmeansSetCollectMetrics(bool enable) { kmeans_collect_metrics = enable; }

    void KmeansAccumulateMseFromBlockBestDot(int cols) {
        if (cols <= 0) return;
        if (kmeans_blk_cols != cols || !kmeans_blk_best.ptr) {
            throw std::runtime_error("KmeansAccumulateMseFromBlockBestDot: block buffers not initialized/mismatched.");
        }
        if (!kmeans_metric_totals.ptr) {
            KmeansResetMetricTotals();
        }
        const float* base_norm2 = nullptr;
        const float* hatnorm2 = nullptr;
        if (kmeans_active_slot >= 0) {
            if (kmeans_norm2_slot[kmeans_active_slot].ptr &&
                kmeans_slot_cols[kmeans_active_slot] == cols) {
                base_norm2 = static_cast<const float*>(kmeans_norm2_slot[kmeans_active_slot].ptr);
            }
            if (rvq_hatnorm2_slot[kmeans_active_slot].ptr &&
                kmeans_slot_cols[kmeans_active_slot] == cols) {
                hatnorm2 = static_cast<const float*>(rvq_hatnorm2_slot[kmeans_active_slot].ptr);
            }
        }
        if (!base_norm2 && kmeans_norm2.ptr && kmeans_xnorm_cols == cols) {
            base_norm2 = static_cast<const float*>(kmeans_norm2.ptr);
        }
        const dim3 block(256, 1, 1);
        const std::int64_t grid_x64 =
            std::min<std::int64_t>((static_cast<std::int64_t>(cols) + static_cast<std::int64_t>(block.x) - 1) /
                                       static_cast<std::int64_t>(block.x),
                                   4096);
        const dim3 grid(static_cast<unsigned int>(std::max<std::int64_t>(1, grid_x64)), 1, 1);
        KmeansAccumulateMseFromBestDotKernel<<<grid, block, 0, stream>>>(
            static_cast<const float*>(kmeans_blk_best.ptr),
            base_norm2,
            hatnorm2,
            cols,
            static_cast<double*>(kmeans_metric_totals.ptr));
        ThrowIf(cudaGetLastError(), "KmeansAccumulateMseFromBestDotKernel launch");
    }

    void KmeansDownloadMetricTotals(double* mse_sum, double* mse_count) {
        KmeansDownloadMetricTotalsAsync();
        KmeansSyncMetricTotalsDownload(mse_sum, mse_count);
    }

    void KmeansDownloadMetricTotalsAsync() {
        if (kmeans_metric_pending) {
            throw std::runtime_error("KmeansDownloadMetricTotalsAsync: pending async copy; call Sync() first.");
        }
        if (!kmeans_metric_totals.ptr) return;
        EnsureKmeansMetricEvents();
        kmeans_metric_totals_hpin.EnsureWithFlags(2 * sizeof(double), cudaHostAllocPortable);
        ThrowIf(cudaEventRecord(kmeans_metric_ready_ev, stream), "cudaEventRecord (kmeans_metric_ready_ev)");
        ThrowIf(cudaStreamWaitEvent(stream_copy, kmeans_metric_ready_ev, 0),
                "cudaStreamWaitEvent (kmeans_metric_ready_ev)");
        ThrowIf(cudaMemcpyAsync(kmeans_metric_totals_hpin.ptr,
                                kmeans_metric_totals.ptr,
                                2 * sizeof(double),
                                cudaMemcpyDeviceToHost,
                                stream_copy),
                "cudaMemcpyAsync (kmeans_metric_totals D2H async)");
        ThrowIf(cudaEventRecord(kmeans_metric_done_ev, stream_copy), "cudaEventRecord (kmeans_metric_done_ev)");
        kmeans_metric_pending = true;
    }

    void KmeansSyncMetricTotalsDownload(double* mse_sum, double* mse_count) {
        if (!kmeans_metric_pending) return;
        ThrowIf(cudaEventSynchronize(kmeans_metric_done_ev), "cudaEventSynchronize (kmeans_metric_done_ev)");
        const double* h = static_cast<const double*>(kmeans_metric_totals_hpin.ptr);
        if (mse_sum) *mse_sum = h[0];
        if (mse_count) *mse_count = h[1];
        kmeans_metric_pending = false;
    }

    void KmeansComputeOutlierMaskFromBlockBest(int cols, float threshold) {
        if (cols <= 0) return;
        if (kmeans_blk_cols != cols || !kmeans_blk_best.ptr) {
            throw std::runtime_error("KmeansComputeOutlierMaskFromBlockBest: block buffers not initialized/mismatched.");
        }
        kmeans_outlier_mask.Ensure(static_cast<std::size_t>(cols) * sizeof(std::uint8_t));
        kmeans_outlier_mask_cols = cols;
        const dim3 block(256, 1, 1);
        const std::int64_t grid_x64 =
            std::min<std::int64_t>((static_cast<std::int64_t>(cols) + static_cast<std::int64_t>(block.x) - 1) /
                                       static_cast<std::int64_t>(block.x),
                                   4096);
        const dim3 grid(static_cast<unsigned int>(std::max<std::int64_t>(1, grid_x64)), 1, 1);
        OutlierMaskFromBestDotKernel<<<grid, block, 0, stream>>>(
            static_cast<const float*>(kmeans_blk_best.ptr),
            cols,
            threshold,
            static_cast<std::uint8_t*>(kmeans_outlier_mask.ptr));
        ThrowIf(cudaGetLastError(), "OutlierMaskFromBestDotKernel launch");
    }

    void KmeansDownloadOutlierMaskAsync(int cols) {
        if (kmeans_mask_pending) {
            throw std::runtime_error("KmeansDownloadOutlierMaskAsync: pending async copy; call Sync() first.");
        }
        if (cols <= 0) return;
        if (kmeans_outlier_mask_cols != cols || !kmeans_outlier_mask.ptr) {
            throw std::runtime_error("KmeansDownloadOutlierMaskAsync: mask buffer not initialized/mismatched.");
        }
        EnsureKmeansMaskEvents();
        kmeans_outlier_mask_hpin.EnsureWithFlags(static_cast<std::size_t>(cols), cudaHostAllocPortable);
        ThrowIf(cudaEventRecord(kmeans_mask_ready_ev, stream), "cudaEventRecord (kmeans_mask_ready_ev)");
        ThrowIf(cudaStreamWaitEvent(stream_copy, kmeans_mask_ready_ev, 0),
                "cudaStreamWaitEvent (kmeans_mask_ready_ev)");
        ThrowIf(cudaMemcpyAsync(kmeans_outlier_mask_hpin.ptr,
                                kmeans_outlier_mask.ptr,
                                static_cast<std::size_t>(cols),
                                cudaMemcpyDeviceToHost,
                                stream_copy),
                "cudaMemcpyAsync (kmeans_outlier_mask D2H async)");
        ThrowIf(cudaEventRecord(kmeans_mask_done_ev, stream_copy), "cudaEventRecord (kmeans_mask_done_ev)");
        kmeans_mask_pending = true;
        kmeans_mask_bytes = cols;
    }

    void KmeansSyncOutlierMaskDownload(std::uint8_t** host_ptr, int* host_bytes) {
        if (!kmeans_mask_pending) {
            if (host_ptr) *host_ptr = nullptr;
            if (host_bytes) *host_bytes = 0;
            return;
        }
        ThrowIf(cudaEventSynchronize(kmeans_mask_done_ev), "cudaEventSynchronize (kmeans_mask_done_ev)");
        if (host_ptr) *host_ptr = static_cast<std::uint8_t*>(kmeans_outlier_mask_hpin.ptr);
        if (host_bytes) *host_bytes = kmeans_mask_bytes;
        kmeans_mask_pending = false;
    }

    void KmeansResetCostHistogram(int bins) {
        bins = std::max(1, bins);
        const std::size_t bytes = static_cast<std::size_t>(bins) * sizeof(std::uint32_t);
        if (kmeans_hist_bins != bins || !kmeans_hist_u32.ptr || kmeans_hist_u32.bytes < bytes) {
            kmeans_hist_u32.Reset();
            kmeans_hist_u32.Ensure(bytes);
            kmeans_hist_bins = bins;
        }
        ThrowIf(cudaMemsetAsync(kmeans_hist_u32.ptr, 0, bytes, stream),
                "cudaMemsetAsync (kmeans_hist_u32)");
    }

    void KmeansAccumulateCostHistogramFromBestDot(const DeviceVecF32View& d_best_dot,
                                                  int bins) {
        if (!d_best_dot.ptr || d_best_dot.n <= 0) return;
        bins = std::max(1, bins);
        if (kmeans_hist_bins != bins || !kmeans_hist_u32.ptr) {
            // Reset also allocates if needed.
            KmeansResetCostHistogram(bins);
        }
        const int n = d_best_dot.n;
        const dim3 block(256, 1, 1);
        const std::int64_t grid_x64 =
            std::min<std::int64_t>((static_cast<std::int64_t>(n) + static_cast<std::int64_t>(block.x) - 1) /
                                       static_cast<std::int64_t>(block.x),
                                   4096);
        const dim3 grid(static_cast<unsigned int>(std::max<std::int64_t>(1, grid_x64)), 1, 1);
        KmeansHistCostFromBestDotWarpAggKernel<<<grid, block, 0, stream>>>(
            d_best_dot.ptr, n, static_cast<std::uint32_t*>(kmeans_hist_u32.ptr), bins);
        ThrowIf(cudaGetLastError(), "KmeansHistCostFromBestDotWarpAggKernel launch");
    }

    void KmeansDownloadCostHistogram(std::vector<std::uint32_t>* hist_u32) {
        if (!hist_u32) return;
        hist_u32->clear();
        if (kmeans_hist_bins <= 0 || !kmeans_hist_u32.ptr) return;
        const int bins = kmeans_hist_bins;
        hist_u32->resize(static_cast<std::size_t>(bins));
        KmeansDownloadCostHistogramAsync(hist_u32->data(), bins);
        KmeansDownloadCostHistogramSync();
    }

    void KmeansDownloadCostHistogramAsync(std::uint32_t* host_dst, int bins) {
        if (!host_dst) {
            throw std::runtime_error("KmeansDownloadCostHistogramAsync: host_dst is null.");
        }
        if (kmeans_hist_pending) {
            throw std::runtime_error("KmeansDownloadCostHistogramAsync: pending async copy; call Sync() first.");
        }
        if (kmeans_hist_bins <= 0 || !kmeans_hist_u32.ptr) {
            return;
        }
        bins = std::max(1, bins);
        if (bins != kmeans_hist_bins) {
            throw std::runtime_error("KmeansDownloadCostHistogramAsync: bins mismatch.");
        }
        EnsureKmeansHistEvents();

        const std::size_t bytes = static_cast<std::size_t>(bins) * sizeof(std::uint32_t);
        kmeans_hist_hpin.EnsureWithFlags(bytes, cudaHostAllocPortable);

        // Make the copy stream wait for all prior histogram updates on the compute stream.
        ThrowIf(cudaEventRecord(kmeans_hist_ready_ev, stream), "cudaEventRecord (kmeans_hist_ready_ev)");
        ThrowIf(cudaStreamWaitEvent(stream_copy, kmeans_hist_ready_ev, 0),
                "cudaStreamWaitEvent (kmeans_hist_ready_ev)");
        ThrowIf(cudaMemcpyAsync(kmeans_hist_hpin.ptr,
                                kmeans_hist_u32.ptr,
                                bytes,
                                cudaMemcpyDeviceToHost,
                                stream_copy),
                "cudaMemcpyAsync (kmeans_hist_u32 D2H async)");
        ThrowIf(cudaEventRecord(kmeans_hist_done_ev, stream_copy), "cudaEventRecord (kmeans_hist_done_ev)");

        kmeans_hist_host_dst = host_dst;
        kmeans_hist_host_bins = bins;
        kmeans_hist_pending = true;
    }

    void KmeansDownloadCostHistogramSync() {
        if (!kmeans_hist_pending) return;
        ThrowIf(cudaEventSynchronize(kmeans_hist_done_ev), "cudaEventSynchronize (kmeans_hist_done_ev)");
        const std::size_t bytes =
            static_cast<std::size_t>(std::max(1, kmeans_hist_host_bins)) * sizeof(std::uint32_t);
        std::memcpy(kmeans_hist_host_dst, kmeans_hist_hpin.ptr, bytes);
        kmeans_hist_host_dst = nullptr;
        kmeans_hist_host_bins = 0;
        kmeans_hist_pending = false;
    }

    void KmeansResetSums(int d, int k) {
        if (d <= 0 || k <= 0) return;
        const std::size_t n = static_cast<std::size_t>(d) * static_cast<std::size_t>(k);
        kmeans_sum_x.Ensure(n * sizeof(float));
        kmeans_sum_w.Ensure(static_cast<std::size_t>(k) * sizeof(float));
        kmeans_sum_rows = d;
        kmeans_sum_k = k;
        kmeans_sum_ld = d;
        ThrowIf(cudaMemsetAsync(kmeans_sum_x.ptr, 0, n * sizeof(float), stream),
                "cudaMemsetAsync (kmeans_sum_x)");
        ThrowIf(cudaMemsetAsync(kmeans_sum_w.ptr, 0, static_cast<std::size_t>(k) * sizeof(float), stream),
                "cudaMemsetAsync (kmeans_sum_w)");
    }

    void KmeansAccumulateUnitWeights(const DeviceMatF32View& Xnorm,
                                     const std::vector<int>& assignments) {
        if (!Xnorm.ptr || Xnorm.rows <= 0 || Xnorm.cols <= 0 || Xnorm.ld < Xnorm.rows) {
            return;
        }
        if (kmeans_sum_rows != Xnorm.rows || kmeans_sum_ld != Xnorm.rows) {
            throw std::runtime_error("KmeansAccumulateUnitWeights: sums not initialized for this d.");
        }
        if (static_cast<int>(assignments.size()) != Xnorm.cols) {
            throw std::runtime_error("KmeansAccumulateUnitWeights: assignments size mismatch.");
        }
        const int cols = Xnorm.cols;
        kmeans_assign.Ensure(static_cast<std::size_t>(cols) * sizeof(int));
        kmeans_assign_cols = cols;
        ThrowIf(cudaMemcpyAsync(kmeans_assign.ptr,
                                assignments.data(),
                                static_cast<std::size_t>(cols) * sizeof(int),
                                cudaMemcpyHostToDevice,
                                stream),
                "cudaMemcpyAsync (kmeans_assign H2D)");

        const dim3 block(256, 1, 1);
        const std::int64_t total = static_cast<std::int64_t>(Xnorm.rows) * static_cast<std::int64_t>(Xnorm.cols);
        const std::int64_t grid_x64 =
            std::min<std::int64_t>((total + static_cast<std::int64_t>(block.x) - 1) / static_cast<std::int64_t>(block.x),
                                   65535);
        const dim3 grid(static_cast<unsigned int>(grid_x64), 1, 1);
        KmeansAccumulateUnitWeightsKernel1D<<<grid, block, 0, stream>>>(
            Xnorm.ptr,
            Xnorm.ld,
            Xnorm.rows,
            Xnorm.cols,
            static_cast<const int*>(kmeans_assign.ptr),
            static_cast<float*>(kmeans_sum_x.ptr),
            kmeans_sum_ld,
            static_cast<float*>(kmeans_sum_w.ptr));
        ThrowIf(cudaGetLastError(), "KmeansAccumulateUnitWeightsKernel1D launch");
    }

    void KmeansAccumulateUnitWeightsDeviceAssign(const DeviceMatF32View& Xnorm,
                                                 const DeviceVecI32View& d_assign_view) {
        if (!Xnorm.ptr || Xnorm.rows <= 0 || Xnorm.cols <= 0 || Xnorm.ld < Xnorm.rows) {
            return;
        }
        if (!d_assign_view.ptr || d_assign_view.n != Xnorm.cols) {
            throw std::runtime_error("KmeansAccumulateUnitWeightsDeviceAssign: assignments size mismatch.");
        }
        if (kmeans_sum_rows != Xnorm.rows || kmeans_sum_ld != Xnorm.rows) {
            throw std::runtime_error("KmeansAccumulateUnitWeightsDeviceAssign: sums not initialized for this d.");
        }

        const dim3 block(256, 1, 1);
        const std::int64_t total = static_cast<std::int64_t>(Xnorm.rows) * static_cast<std::int64_t>(Xnorm.cols);
        const std::int64_t grid_x64 =
            std::min<std::int64_t>((total + static_cast<std::int64_t>(block.x) - 1) / static_cast<std::int64_t>(block.x),
                                   65535);
        const dim3 grid(static_cast<unsigned int>(grid_x64), 1, 1);
        KmeansAccumulateUnitWeightsKernel1D<<<grid, block, 0, stream>>>(
            Xnorm.ptr,
            Xnorm.ld,
            Xnorm.rows,
            Xnorm.cols,
            d_assign_view.ptr,
            static_cast<float*>(kmeans_sum_x.ptr),
            kmeans_sum_ld,
            static_cast<float*>(kmeans_sum_w.ptr));
        ThrowIf(cudaGetLastError(), "KmeansAccumulateUnitWeightsKernel1D launch (device assign)");
    }

    void KmeansAccumulateDeviceAssignNoAtomicImpl(const DeviceMatF32View& Xnorm,
                                                 const DeviceVecI32View& d_assign_view,
                                                 const float* d_w_or_null) {
        if (!Xnorm.ptr || Xnorm.rows <= 0 || Xnorm.cols <= 0 || Xnorm.ld < Xnorm.rows) {
            return;
        }
        if (!d_assign_view.ptr || d_assign_view.n != Xnorm.cols) {
            throw std::runtime_error("KmeansAccumulateDeviceAssignNoAtomicImpl: assignment size mismatch.");
        }
        if (d_w_or_null) {
            // d_w_or_null is assumed length == Xnorm.cols.
        }
        if (kmeans_sum_rows != Xnorm.rows || kmeans_sum_ld != Xnorm.rows) {
            throw std::runtime_error("KmeansAccumulateDeviceAssignNoAtomicImpl: sums not initialized for this d.");
        }
        const int cols = Xnorm.cols;
        const int k = kmeans_sum_k;
        EnsureKmeansSegmentScratch(cols, k);

        int* d_keys_in = static_cast<int*>(kmeans_sort_keys_in.ptr);
        int* d_vals_in = static_cast<int*>(kmeans_sort_vals_in.ptr);
        int* d_keys_out = static_cast<int*>(kmeans_sort_keys_out.ptr);
        int* d_vals_out = static_cast<int*>(kmeans_sort_vals_out.ptr);
        int* d_unique = static_cast<int*>(kmeans_unique_keys.ptr);
        int* d_counts = static_cast<int*>(kmeans_counts.ptr);
        int* d_offsets = static_cast<int*>(kmeans_seg_offsets.ptr);
        int* d_num_segments = static_cast<int*>(kmeans_num_segments.ptr);

        // Fill sort inputs.
        {
            const dim3 block(256, 1, 1);
            const dim3 grid(std::min(65535, (cols + 255) / 256), 1, 1);
            FillKeysValsKernel<<<grid, block, 0, stream>>>(cols, d_assign_view.ptr, d_keys_in, d_vals_in);
            ThrowIf(cudaGetLastError(), "FillKeysValsKernel launch");
        }

        // Zero counts so that scan over [0,k) is valid even though only the first num_segments entries are written.
        ThrowIf(cudaMemsetAsync(d_counts, 0, static_cast<std::size_t>(k) * sizeof(int), stream),
                "cudaMemsetAsync (kmeans_counts)");
        ThrowIf(cudaMemsetAsync(d_offsets, 0, static_cast<std::size_t>(k) * sizeof(int), stream),
                "cudaMemsetAsync (kmeans_seg_offsets)");

        // CUB temp storage sizing (reused buffer).
        std::size_t tmp_sort = 0;
        std::size_t tmp_rle = 0;
        std::size_t tmp_scan = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, tmp_sort,
                                        d_keys_in, d_keys_out,
                                        d_vals_in, d_vals_out,
                                        cols,
                                        /*begin_bit=*/0, /*end_bit=*/32,
                                        stream);
        cub::DeviceRunLengthEncode::Encode(nullptr, tmp_rle,
                                           d_keys_out,
                                           d_unique,
                                           d_counts,
                                           d_num_segments,
                                           cols,
                                           stream);
        cub::DeviceScan::ExclusiveSum(nullptr, tmp_scan,
                                      d_counts, d_offsets,
                                      k,
                                      stream);
        const std::size_t tmp_bytes = std::max(tmp_sort, std::max(tmp_rle, tmp_scan));
        if (tmp_bytes > 0) {
            kmeans_cub_tmp.Ensure(tmp_bytes);
        }

        // Sort by assignment.
        cub::DeviceRadixSort::SortPairs(kmeans_cub_tmp.ptr, tmp_sort,
                                        d_keys_in, d_keys_out,
                                        d_vals_in, d_vals_out,
                                        cols,
                                        /*begin_bit=*/0, /*end_bit=*/32,
                                        stream);
        // Run-length encode to segments (unique keys + counts).
        cub::DeviceRunLengthEncode::Encode(kmeans_cub_tmp.ptr, tmp_rle,
                                           d_keys_out,
                                           d_unique,
                                           d_counts,
                                           d_num_segments,
                                           cols,
                                           stream);
        // Exclusive scan counts -> segment offsets (scan over capacity k; counts beyond num_segments are zero).
        cub::DeviceScan::ExclusiveSum(kmeans_cub_tmp.ptr, tmp_scan,
                                      d_counts, d_offsets,
                                      k,
                                      stream);

        // Reduce per segment (one block per potential segment slot; early-exit for seg>=num_segments).
        {
            const dim3 block(256, 1, 1);
            const dim3 grid(static_cast<unsigned int>(k), 1, 1);
            ReduceSegmentsAccumulateKernel<<<grid, block, 0, stream>>>(
                Xnorm.ptr, Xnorm.ld, Xnorm.rows, cols,
                d_vals_out,
                d_unique,
                d_offsets,
                d_counts,
                d_num_segments,
                d_w_or_null,
                static_cast<float*>(kmeans_sum_x.ptr),
                kmeans_sum_ld,
                static_cast<float*>(kmeans_sum_w.ptr));
            ThrowIf(cudaGetLastError(), "ReduceSegmentsAccumulateKernel launch");
        }
    }

    void KmeansAccumulateUnitWeightsDeviceAssignNoAtomic(const DeviceMatF32View& Xnorm,
                                                         const DeviceVecI32View& d_assign_view) {
        KmeansAccumulateDeviceAssignNoAtomicImpl(Xnorm, d_assign_view, /*d_w_or_null=*/nullptr);
    }

    void KmeansAccumulateWeights(const DeviceMatF32View& Xnorm,
                                 const std::vector<int>& assignments,
                                 const std::vector<float>& weights) {
        if (!Xnorm.ptr || Xnorm.rows <= 0 || Xnorm.cols <= 0 || Xnorm.ld < Xnorm.rows) {
            return;
        }
        if (kmeans_sum_rows != Xnorm.rows || kmeans_sum_ld != Xnorm.rows) {
            throw std::runtime_error("KmeansAccumulateWeights: sums not initialized for this d.");
        }
        if (static_cast<int>(assignments.size()) != Xnorm.cols ||
            static_cast<int>(weights.size()) != Xnorm.cols) {
            throw std::runtime_error("KmeansAccumulateWeights: assignments/weights size mismatch.");
        }
        const int cols = Xnorm.cols;
        kmeans_assign.Ensure(static_cast<std::size_t>(cols) * sizeof(int));
        kmeans_assign_cols = cols;
        kmeans_weights.Ensure(static_cast<std::size_t>(cols) * sizeof(float));
        kmeans_weights_cols = cols;
        ThrowIf(cudaMemcpyAsync(kmeans_assign.ptr,
                                assignments.data(),
                                static_cast<std::size_t>(cols) * sizeof(int),
                                cudaMemcpyHostToDevice,
                                stream),
                "cudaMemcpyAsync (kmeans_assign H2D)");
        ThrowIf(cudaMemcpyAsync(kmeans_weights.ptr,
                                weights.data(),
                                static_cast<std::size_t>(cols) * sizeof(float),
                                cudaMemcpyHostToDevice,
                                stream),
                "cudaMemcpyAsync (kmeans_weights H2D)");

        const dim3 block(256, 1, 1);
        const std::int64_t total = static_cast<std::int64_t>(Xnorm.rows) * static_cast<std::int64_t>(Xnorm.cols);
        const std::int64_t grid_x64 =
            std::min<std::int64_t>((total + static_cast<std::int64_t>(block.x) - 1) / static_cast<std::int64_t>(block.x),
                                   65535);
        const dim3 grid(static_cast<unsigned int>(grid_x64), 1, 1);
        KmeansAccumulateWeightsKernel1D<<<grid, block, 0, stream>>>(
            Xnorm.ptr,
            Xnorm.ld,
            Xnorm.rows,
            Xnorm.cols,
            static_cast<const int*>(kmeans_assign.ptr),
            static_cast<const float*>(kmeans_weights.ptr),
            static_cast<float*>(kmeans_sum_x.ptr),
            kmeans_sum_ld,
            static_cast<float*>(kmeans_sum_w.ptr));
        ThrowIf(cudaGetLastError(), "KmeansAccumulateWeightsKernel1D launch");
    }

    void KmeansAccumulateWeightsDeviceAssign(const DeviceMatF32View& Xnorm,
                                             const DeviceVecI32View& d_assign_view,
                                             const DeviceVecF32View& d_weights_view) {
        if (!Xnorm.ptr || Xnorm.rows <= 0 || Xnorm.cols <= 0 || Xnorm.ld < Xnorm.rows) {
            return;
        }
        if (!d_assign_view.ptr || d_assign_view.n != Xnorm.cols ||
            !d_weights_view.ptr || d_weights_view.n != Xnorm.cols) {
            throw std::runtime_error("KmeansAccumulateWeightsDeviceAssign: assignments/weights size mismatch.");
        }
        if (kmeans_sum_rows != Xnorm.rows || kmeans_sum_ld != Xnorm.rows) {
            throw std::runtime_error("KmeansAccumulateWeightsDeviceAssign: sums not initialized for this d.");
        }

        const dim3 block(256, 1, 1);
        const std::int64_t total = static_cast<std::int64_t>(Xnorm.rows) * static_cast<std::int64_t>(Xnorm.cols);
        const std::int64_t grid_x64 =
            std::min<std::int64_t>((total + static_cast<std::int64_t>(block.x) - 1) / static_cast<std::int64_t>(block.x),
                                   65535);
        const dim3 grid(static_cast<unsigned int>(grid_x64), 1, 1);
        KmeansAccumulateWeightsKernel1D<<<grid, block, 0, stream>>>(
            Xnorm.ptr,
            Xnorm.ld,
            Xnorm.rows,
            Xnorm.cols,
            d_assign_view.ptr,
            d_weights_view.ptr,
            static_cast<float*>(kmeans_sum_x.ptr),
            kmeans_sum_ld,
            static_cast<float*>(kmeans_sum_w.ptr));
        ThrowIf(cudaGetLastError(), "KmeansAccumulateWeightsKernel1D launch (device assign)");
    }

    void KmeansAccumulateWeightsDeviceAssignNoAtomic(const DeviceMatF32View& Xnorm,
                                                     const DeviceVecI32View& d_assign_view,
                                                     const DeviceVecF32View& d_weights_view) {
        if (!d_weights_view.ptr || d_weights_view.n != Xnorm.cols) {
            throw std::runtime_error("KmeansAccumulateWeightsDeviceAssignNoAtomic: weights size mismatch.");
        }
        KmeansAccumulateDeviceAssignNoAtomicImpl(Xnorm, d_assign_view, d_weights_view.ptr);
    }

    void KmeansDownloadSums(ColMajorMatrix<float>* sum_x,
                            std::vector<float>* sum_w) {
        if (!sum_x || !sum_w) return;
        if (!kmeans_sum_x.ptr || !kmeans_sum_w.ptr || kmeans_sum_rows <= 0 || kmeans_sum_k <= 0) {
            sum_x->rows = 0;
            sum_x->cols = 0;
            sum_x->data.clear();
            sum_w->clear();
            return;
        }
        const int d = kmeans_sum_rows;
        const int k = kmeans_sum_k;
        sum_x->rows = d;
        sum_x->cols = k;
        sum_x->data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(k));
        sum_w->assign(static_cast<std::size_t>(k), 0.0f);
        const std::size_t nx = static_cast<std::size_t>(d) * static_cast<std::size_t>(k) * sizeof(float);
        ThrowIf(cudaMemcpyAsync(sum_x->data.data(),
                                kmeans_sum_x.ptr,
                                nx,
                                cudaMemcpyDeviceToHost,
                                stream),
                "cudaMemcpyAsync (kmeans_sum_x D2H)");
        ThrowIf(cudaMemcpyAsync(sum_w->data(),
                                kmeans_sum_w.ptr,
                                static_cast<std::size_t>(k) * sizeof(float),
                                cudaMemcpyDeviceToHost,
                                stream),
                "cudaMemcpyAsync (kmeans_sum_w D2H)");
        ThrowIf(cudaStreamSynchronize(stream), "cudaStreamSynchronize (KmeansDownloadSums)");
    }

    void ConvertU8ToF32(const ColMajorMatrix<std::uint8_t>& in,
                        ColMajorMatrix<float>* out) {
        if (!out) {
            throw std::runtime_error("CudaStreamKernels::ConvertU8ToF32: output is null.");
        }
        const int d = in.rows;
        const int n = in.cols;
        out->rows = d;
        out->cols = n;
        out->data.resize(static_cast<std::size_t>(d) * static_cast<std::size_t>(n));
        if (d <= 0 || n <= 0) {
            return;
        }

        const std::size_t nu8 = static_cast<std::size_t>(d) * static_cast<std::size_t>(n);
        du8.Ensure(nu8 * sizeof(std::uint8_t));
        dtmp.Ensure(nu8 * sizeof(float));
        ThrowIf(cudaMemcpyAsync(du8.ptr, in.data.data(), nu8 * sizeof(std::uint8_t),
                                cudaMemcpyHostToDevice, stream),
                "cudaMemcpyAsync u8 H2D");

        const dim3 block(256, 1, 1);
        const std::int64_t total = static_cast<std::int64_t>(d) * static_cast<std::int64_t>(n);
        const std::int64_t grid_x64 =
            std::min<std::int64_t>((total + static_cast<std::int64_t>(block.x) - 1) / static_cast<std::int64_t>(block.x),
                                   65535);
        const dim3 grid(static_cast<unsigned int>(grid_x64), 1, 1);
        U8ToF32Kernel1D<<<grid, block, 0, stream>>>(static_cast<const std::uint8_t*>(du8.ptr),
                                         d,
                                         static_cast<float*>(dtmp.ptr),
                                         d,
                                         d,
                                         n);
        ThrowIf(cudaGetLastError(), "U8ToF32Kernel launch");

        ThrowIf(cudaStreamSynchronize(stream), "cudaStreamSynchronize (ConvertU8ToF32)");
        ThrowIf(cudaMemcpy(out->data.data(), dtmp.ptr, nu8 * sizeof(float), cudaMemcpyDeviceToHost),
                "cudaMemcpy f32 D2H");
    }

    void ConvertU8ToF32AndRotatePtr(const std::uint8_t* in,
                                    int ld_in,
                                    int rows,
                                    int cols,
                                    const ColMajorMatrix<float>& R,
                                    ColMajorMatrix<float>* out) {
        if (!out) {
            throw std::runtime_error("CudaStreamKernels::ConvertU8ToF32AndRotatePtr: output is null.");
        }
        if (!in || rows <= 0 || cols <= 0 || R.rows <= 0 || R.cols <= 0) {
            out->rows = 0;
            out->cols = 0;
            out->data.clear();
            return;
        }
        if (R.cols != rows) {
            throw std::runtime_error("CudaStreamKernels::ConvertU8ToF32AndRotatePtr: dim mismatch (R.cols != rows).");
        }

        out->rows = R.rows;
        out->cols = cols;
        out->data.resize(static_cast<std::size_t>(out->rows) * static_cast<std::size_t>(cols));

        // Upload R (cached).
        R_cache.UploadIfNeeded(R, stream);

        // Upload u8 -> tmp float on device.
        const std::size_t nu8 = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
        du8.Ensure(nu8 * sizeof(std::uint8_t));
        dtmp.Ensure(nu8 * sizeof(float));

        // Input is column-major with stride ld_in; fast path assumes ld_in == rows (contiguous tile).
        // If ld_in differs, upload a packed tile via 2D copy (still no intermediate host memcpy).
        if (ld_in == rows) {
            ThrowIf(cudaMemcpyAsync(du8.ptr, in, nu8 * sizeof(std::uint8_t),
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync u8 H2D (ptr)");
        } else {
            ThrowIf(cudaMemcpy2DAsync(du8.ptr,
                                      static_cast<std::size_t>(rows) * sizeof(std::uint8_t),
                                      in,
                                      static_cast<std::size_t>(ld_in) * sizeof(std::uint8_t),
                                      static_cast<std::size_t>(rows) * sizeof(std::uint8_t),
                                      static_cast<std::size_t>(cols),
                                      cudaMemcpyHostToDevice,
                                      stream),
                    "cudaMemcpy2DAsync u8 H2D (ptr)");
        }

        const dim3 block(256, 1, 1);
        const std::int64_t total = static_cast<std::int64_t>(rows) * static_cast<std::int64_t>(cols);
        const std::int64_t grid_x64 =
            std::min<std::int64_t>((total + static_cast<std::int64_t>(block.x) - 1) / static_cast<std::int64_t>(block.x),
                                   65535);
        const dim3 grid(static_cast<unsigned int>(grid_x64), 1, 1);
        U8ToF32Kernel1D<<<grid, block, 0, stream>>>(static_cast<const std::uint8_t*>(du8.ptr),
                                         rows,
                                         static_cast<float*>(dtmp.ptr),
                                         rows,
                                         rows,
                                         cols);
        ThrowIf(cudaGetLastError(), "U8ToF32Kernel launch (ptr)");

        // out = R * tmp
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const int m = R.rows;
        const int k = R.cols;
        d_cached.Ensure(static_cast<std::size_t>(m) * static_cast<std::size_t>(cols) * sizeof(float));

        GemmF32(CUBLAS_OP_N, CUBLAS_OP_N, m, cols, k,
                &alpha,
                R_cache.dptr(),
                R.rows,
                static_cast<const float*>(dtmp.ptr),
                rows,
                &beta,
                static_cast<float*>(d_cached.ptr),
                m,
                allow_tf32 ? "cublasGemmEx(TF32) (rotate ptr)" : "cublasSgemm (rotate ptr)");

        ThrowIf(cudaStreamSynchronize(stream), "cudaStreamSynchronize (rotate ptr)");
        ThrowIf(cudaMemcpy(out->data.data(), d_cached.ptr,
                           static_cast<std::size_t>(m) * static_cast<std::size_t>(cols) * sizeof(float),
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy rotated D2H (ptr)");

        cached_host_ptr = out->data.data();
        cached_rows = out->rows;
        cached_cols = out->cols;
        cached_host_stamp = CachedDeviceMatF32::Stamp(*out);
    }

    void ConvertU8ToF32AndRotate(const ColMajorMatrix<std::uint8_t>& in,
                                 const ColMajorMatrix<float>& R,
                                 ColMajorMatrix<float>* out) {
        if (!out) {
            throw std::runtime_error("CudaStreamKernels::ConvertU8ToF32AndRotate: output is null.");
        }
        if (R.rows <= 0 || R.cols <= 0) {
            out->rows = 0;
            out->cols = 0;
            out->data.clear();
            return;
        }
        const int d = in.rows;
        const int n = in.cols;
        if (d != R.cols) {
            throw std::runtime_error("CudaStreamKernels::ConvertU8ToF32AndRotate: dim mismatch (R.cols != d).");
        }

        out->rows = R.rows;
        out->cols = n;
        out->data.resize(static_cast<std::size_t>(out->rows) * static_cast<std::size_t>(n));
        if (d <= 0 || n <= 0) {
            return;
        }

        // Upload R (cached).
        R_cache.UploadIfNeeded(R, stream);

        // Upload u8 -> tmp float on device.
        const std::size_t nu8 = static_cast<std::size_t>(d) * static_cast<std::size_t>(n);
        du8.Ensure(nu8 * sizeof(std::uint8_t));
        dtmp.Ensure(nu8 * sizeof(float));
        ThrowIf(cudaMemcpyAsync(du8.ptr, in.data.data(), nu8 * sizeof(std::uint8_t),
                                cudaMemcpyHostToDevice, stream),
                "cudaMemcpyAsync u8 H2D");
        const dim3 block(256, 1, 1);
        const int total = d * n;
        const dim3 grid((total + static_cast<int>(block.x) - 1) / static_cast<int>(block.x), 1, 1);
        U8ToF32Kernel1D<<<grid, block, 0, stream>>>(static_cast<const std::uint8_t*>(du8.ptr),
                                         d,
                                         static_cast<float*>(dtmp.ptr),
                                         d,
                                         d,
                                         n);
        ThrowIf(cudaGetLastError(), "U8ToF32Kernel launch");

        // out = R * tmp
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const int m = R.rows;
        const int k = R.cols;
        const int cols = n;
        d_cached.Ensure(static_cast<std::size_t>(m) * static_cast<std::size_t>(cols) * sizeof(float));

        GemmF32(CUBLAS_OP_N, CUBLAS_OP_N, m, cols, k,
                &alpha,
                R_cache.dptr(),
                R.rows,
                static_cast<const float*>(dtmp.ptr),
                d,
                &beta,
                static_cast<float*>(d_cached.ptr),
                m,
                allow_tf32 ? "cublasGemmEx(TF32) (rotate)" : "cublasSgemm (rotate)");

        ThrowIf(cudaStreamSynchronize(stream), "cudaStreamSynchronize (rotate)");
        ThrowIf(cudaMemcpy(out->data.data(), d_cached.ptr,
                           static_cast<std::size_t>(m) * static_cast<std::size_t>(cols) * sizeof(float),
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy rotated D2H");

        cached_host_ptr = out->data.data();
        cached_rows = out->rows;
        cached_cols = out->cols;
        cached_host_stamp = CachedDeviceMatF32::Stamp(*out);
    }
};

}  // namespace stlq
