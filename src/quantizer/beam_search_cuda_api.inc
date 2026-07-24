void BeamSearchPrefixLSLargeRootCuda(const ColMajorMatrix<float>& X,
                                     const PrecompLargeRoot& pre,
                                     const ColMajorMatrix<float>& xC_small,
                                     int H_beam,
                                     bool allow_tf32,
                                     std::vector<std::uint32_t>* cluster_id,
                                     ColMajorMatrix<Code>* B_small,
                                     BeamSearchLargeRootTiming* timing) {
    if (!cluster_id || !B_small) return;
    const int n = X.cols;
    const int d = X.rows;
    const int m = pre.m;
    const int h0 = pre.h_vec.empty() ? 0 : pre.h_vec[0];
    const int Hs = pre.H_small;
    if (!pre.ready || n <= 0 || d <= 0 || m <= 0 || h0 <= 0 || xC_small.rows != Hs || xC_small.cols != n) {
        return;
    }
    if (H_beam <= 0) return;
    if (H_beam > 2) {
        // Keep it simple for now; CPU path covers larger beams.
        BeamSearchPrefixLSLargeRoot(X, pre, xC_small, H_beam, cluster_id, B_small, timing);
        return;
    }

    cluster_id->resize(static_cast<std::size_t>(n));
    B_small->rows = std::max(0, m - 1);
    B_small->cols = n;
    B_small->data.resize(static_cast<std::size_t>(B_small->rows) * static_cast<std::size_t>(n));

    const int q_block = std::min(8192, std::max(1, n));
    const int r_block = 1024;

    static thread_local CudaRootBeamSession sess;
    sess.EnsureHandle(allow_tf32);
    sess.EnsurePre(pre);

    std::vector<float> root_errs;
    std::vector<int> root_codes;
    std::vector<float> root_a0;
    std::vector<float> root_r00;

    for (int q0 = 0; q0 < n; q0 += q_block) {
        const int qlen = std::min(q_block, n - q0);
        const int rlen = std::min(r_block, h0);
        sess.EnsureBatch(qlen, rlen);

        // Upload X block.
        const float* Xptr = X.data.data() + static_cast<std::size_t>(q0) * static_cast<std::size_t>(d);
        ThrowIf(cudaMemcpy(sess.d_X.ptr,
                           Xptr,
                           sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(qlen),
                           cudaMemcpyHostToDevice),
                "cudaMemcpy X H2D");

        // norm_x2
        if (timing) {
            Timer t;
            Norm2ColsKernel<<<qlen, 256>>>(sess.d_X.ptr, d, d, qlen, sess.d_normx2.ptr);
            ThrowIf(cudaGetLastError(), "Norm2ColsKernel");
            ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(norm2)");
            timing->norm_x2 += t.ElapsedSeconds();
        } else {
            Norm2ColsKernel<<<qlen, 256>>>(sess.d_X.ptr, d, d, qlen, sess.d_normx2.ptr);
            ThrowIf(cudaGetLastError(), "Norm2ColsKernel");
        }

        // init best arrays
        // NOTE: Use -1 for codes (not 0) to avoid code 0 being a phantom second-best candidate.
        // If code 0 has zero norm the kernel skips it, leaving the initial value in place; -1 is
        // an explicit sentinel that the expand step treats as "no valid candidate", whereas 0 was
        // silently treated as a valid root code causing a consistently degenerate cluster-0 result.
        // (Same semantics as BeamSearchPrefixLSLargeRootCudaH2/H4 variants.)
        std::vector<float> init_inf(static_cast<std::size_t>(qlen), std::numeric_limits<float>::infinity());
        std::vector<int> init_neg1(static_cast<std::size_t>(qlen), -1);
        std::vector<float> init_zero_f(static_cast<std::size_t>(qlen), 0.0f);
        ThrowIf(cudaMemcpy(sess.d_best_err0.ptr, init_inf.data(), sizeof(float) * init_inf.size(), cudaMemcpyHostToDevice),
                "init best_err0");
        ThrowIf(cudaMemcpy(sess.d_best_err1.ptr, init_inf.data(), sizeof(float) * init_inf.size(), cudaMemcpyHostToDevice),
                "init best_err1");
        ThrowIf(cudaMemcpy(sess.d_best_code0.ptr, init_neg1.data(), sizeof(int) * init_neg1.size(), cudaMemcpyHostToDevice),
                "init best_code0");
        ThrowIf(cudaMemcpy(sess.d_best_code1.ptr, init_neg1.data(), sizeof(int) * init_neg1.size(), cudaMemcpyHostToDevice),
                "init best_code1");
        ThrowIf(cudaMemcpy(sess.d_best_a0.ptr, init_zero_f.data(), sizeof(float) * init_zero_f.size(), cudaMemcpyHostToDevice),
                "init best_a0");
        ThrowIf(cudaMemcpy(sess.d_best_a1.ptr, init_zero_f.data(), sizeof(float) * init_zero_f.size(), cudaMemcpyHostToDevice),
                "init best_a1");
        ThrowIf(cudaMemcpy(sess.d_best_r00.ptr, init_zero_f.data(), sizeof(float) * init_zero_f.size(), cudaMemcpyHostToDevice),
                "init best_r00");
        ThrowIf(cudaMemcpy(sess.d_best_r01.ptr, init_zero_f.data(), sizeof(float) * init_zero_f.size(), cudaMemcpyHostToDevice),
                "init best_r01");

        // Root candidates in chunks.
        for (int r0 = 0; r0 < h0; r0 += r_block) {
            const int rc = std::min(r_block, h0 - r0);
            // scores = C0_chunk' * X
            const float alpha = 1.0f;
            const float beta = 0.0f;
            const float* dA = sess.d_C0.ptr + static_cast<std::size_t>(r0) * static_cast<std::size_t>(d);
            const float* dB = sess.d_X.ptr;
            float* dC = sess.d_scores.ptr;
            if (timing) {
                Timer t;
                ThrowIf(cublasSgemm(sess.handle,
                                    CUBLAS_OP_T,
                                    CUBLAS_OP_N,
                                    rc,
                                    qlen,
                                    d,
                                    &alpha,
                                    dA,
                                    d,
                                    dB,
                                    d,
                                    &beta,
                                    dC,
                                    rc),
                        "cublasSgemm(C0'X)");
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(root_gemm)");
                timing->root_gemm += t.ElapsedSeconds();
            } else {
                ThrowIf(cublasSgemm(sess.handle,
                                    CUBLAS_OP_T,
                                    CUBLAS_OP_N,
                                    rc,
                                    qlen,
                                    d,
                                    &alpha,
                                    dA,
                                    d,
                                    dB,
                                    d,
                                    &beta,
                                    dC,
                                    rc),
                        "cublasSgemm(C0'X)");
            }

            if (timing) {
                Timer t;
                UpdateTop2FromScoresKernel<<<qlen, 256>>>(sess.d_scores.ptr, rc, rc, r0, sess.d_norm0.ptr,
                                                         sess.d_normx2.ptr, qlen,
                                                         sess.d_best_err0.ptr, sess.d_best_code0.ptr,
                                                         sess.d_best_a0.ptr, sess.d_best_r00.ptr,
                                                         sess.d_best_err1.ptr, sess.d_best_code1.ptr,
                                                         sess.d_best_a1.ptr, sess.d_best_r01.ptr);
                ThrowIf(cudaGetLastError(), "UpdateTop2FromScoresKernel");
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(root_update)");
                timing->root_update += t.ElapsedSeconds();
            } else {
                UpdateTop2FromScoresKernel<<<qlen, 256>>>(sess.d_scores.ptr, rc, rc, r0, sess.d_norm0.ptr,
                                                         sess.d_normx2.ptr, qlen,
                                                         sess.d_best_err0.ptr, sess.d_best_code0.ptr,
                                                         sess.d_best_a0.ptr, sess.d_best_r00.ptr,
                                                         sess.d_best_err1.ptr, sess.d_best_code1.ptr,
                                                         sess.d_best_a1.ptr, sess.d_best_r01.ptr);
                ThrowIf(cudaGetLastError(), "UpdateTop2FromScoresKernel");
            }
        }

        root_errs.resize(static_cast<std::size_t>(qlen) * static_cast<std::size_t>(H_beam));
        root_codes.resize(static_cast<std::size_t>(qlen) * static_cast<std::size_t>(H_beam));
        root_a0.resize(static_cast<std::size_t>(qlen) * static_cast<std::size_t>(H_beam));
        root_r00.resize(static_cast<std::size_t>(qlen) * static_cast<std::size_t>(H_beam));

        // Download best arrays.
        std::vector<float> h_e0(static_cast<std::size_t>(qlen));
        std::vector<int> h_c0(static_cast<std::size_t>(qlen));
        std::vector<float> h_a0(static_cast<std::size_t>(qlen));
        std::vector<float> h_r0(static_cast<std::size_t>(qlen));
        std::vector<float> h_e1(static_cast<std::size_t>(qlen));
        std::vector<int> h_c1(static_cast<std::size_t>(qlen));
        std::vector<float> h_a1(static_cast<std::size_t>(qlen));
        std::vector<float> h_r1(static_cast<std::size_t>(qlen));

        ThrowIf(cudaMemcpy(h_e0.data(), sess.d_best_err0.ptr, sizeof(float) * h_e0.size(), cudaMemcpyDeviceToHost),
                "Memcpy best_err0 D2H");
        ThrowIf(cudaMemcpy(h_c0.data(), sess.d_best_code0.ptr, sizeof(int) * h_c0.size(), cudaMemcpyDeviceToHost),
                "Memcpy best_code0 D2H");
        ThrowIf(cudaMemcpy(h_a0.data(), sess.d_best_a0.ptr, sizeof(float) * h_a0.size(), cudaMemcpyDeviceToHost),
                "Memcpy best_a0 D2H");
        ThrowIf(cudaMemcpy(h_r0.data(), sess.d_best_r00.ptr, sizeof(float) * h_r0.size(), cudaMemcpyDeviceToHost),
                "Memcpy best_r00 D2H");
        ThrowIf(cudaMemcpy(h_e1.data(), sess.d_best_err1.ptr, sizeof(float) * h_e1.size(), cudaMemcpyDeviceToHost),
                "Memcpy best_err1 D2H");
        ThrowIf(cudaMemcpy(h_c1.data(), sess.d_best_code1.ptr, sizeof(int) * h_c1.size(), cudaMemcpyDeviceToHost),
                "Memcpy best_code1 D2H");
        ThrowIf(cudaMemcpy(h_a1.data(), sess.d_best_a1.ptr, sizeof(float) * h_a1.size(), cudaMemcpyDeviceToHost),
                "Memcpy best_a1 D2H");
        ThrowIf(cudaMemcpy(h_r1.data(), sess.d_best_r01.ptr, sizeof(float) * h_r1.size(), cudaMemcpyDeviceToHost),
                "Memcpy best_r01 D2H");

        for (int j = 0; j < qlen; ++j) {
            root_errs[static_cast<std::size_t>(j) * H_beam + 0] = h_e0[static_cast<std::size_t>(j)];
            root_codes[static_cast<std::size_t>(j) * H_beam + 0] = h_c0[static_cast<std::size_t>(j)];
            root_a0[static_cast<std::size_t>(j) * H_beam + 0] = h_a0[static_cast<std::size_t>(j)];
            root_r00[static_cast<std::size_t>(j) * H_beam + 0] = h_r0[static_cast<std::size_t>(j)];
            if (H_beam > 1) {
                root_errs[static_cast<std::size_t>(j) * H_beam + 1] = h_e1[static_cast<std::size_t>(j)];
                root_codes[static_cast<std::size_t>(j) * H_beam + 1] = h_c1[static_cast<std::size_t>(j)];
                root_a0[static_cast<std::size_t>(j) * H_beam + 1] = h_a1[static_cast<std::size_t>(j)];
                root_r00[static_cast<std::size_t>(j) * H_beam + 1] = h_r1[static_cast<std::size_t>(j)];
            }
        }

        Timer t_expand;
        BeamSearchPrefixLSLargeRootExpandFromRootCandidates(
            pre, xC_small, q0, qlen, H_beam,
            root_errs.data(),
            root_codes.data(),
            root_a0.data(),
            root_r00.data(),
            cluster_id, B_small);
        if (timing) timing->expand += t_expand.ElapsedSeconds();
    }
}

void BeamSearchPrefixLSLargeRootCudaH2(const ColMajorMatrix<float>& X,
                                       const PrecompLargeRoot& pre,
                                       const ColMajorMatrix<float>& xC_small,
                                       bool allow_tf32,
                                       std::vector<std::uint32_t>* cluster_id,
                                       ColMajorMatrix<Code>* B_small,
                                       BeamSearchLargeRootTiming* timing) {
    if (!cluster_id || !B_small) return;
    const int n = X.cols;
    const int d = X.rows;
    const int m = pre.m;
    const int h0 = pre.h_vec.empty() ? 0 : pre.h_vec[0];
    const int Hs = pre.H_small;
    if (!pre.ready || n <= 0 || d <= 0 || m <= 0 || h0 <= 0 || Hs <= 0) {
        return;
    }

    cluster_id->resize(static_cast<std::size_t>(n));
    B_small->rows = std::max(0, m - 1);
    B_small->cols = n;
    B_small->data.resize(static_cast<std::size_t>(B_small->rows) * static_cast<std::size_t>(n));

    const int q_block = std::min(8192, std::max(1, n));
    const int r_block = 1024;

    static thread_local CudaRootBeamSession root_sess;
    static thread_local CudaExpandH2Session expand_sess;
    root_sess.EnsureHandle(allow_tf32);
    root_sess.EnsurePre(pre);
    expand_sess.EnsurePre(pre);

    auto swap_buf = [](auto& a, auto& b) {
        std::swap(a.ptr, b.ptr);
        std::swap(a.n, b.n);
    };

    for (int q0 = 0; q0 < n; q0 += q_block) {
        const int qlen = std::min(q_block, n - q0);
        const int rlen = std::min(r_block, h0);
        root_sess.EnsureBatch(qlen, rlen);
        expand_sess.EnsureBatch(qlen);

        // Upload X block.
        const float* Xptr = X.data.data() + static_cast<std::size_t>(q0) * static_cast<std::size_t>(d);
        ThrowIf(cudaMemcpy(root_sess.d_X.ptr,
                           Xptr,
                           sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(qlen),
                           cudaMemcpyHostToDevice),
                "cudaMemcpy X H2D");

        // norm_x2
        if (timing) {
            Timer t;
            Norm2ColsKernel<<<qlen, 256>>>(root_sess.d_X.ptr, d, d, qlen, root_sess.d_normx2.ptr);
            ThrowIf(cudaGetLastError(), "Norm2ColsKernel");
            ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(norm_x2)");
            timing->norm_x2 += t.ElapsedSeconds();
        } else {
            Norm2ColsKernel<<<qlen, 256>>>(root_sess.d_X.ptr, d, d, qlen, root_sess.d_normx2.ptr);
            ThrowIf(cudaGetLastError(), "Norm2ColsKernel");
        }

        // Init best arrays.
        {
            const float inf = std::numeric_limits<float>::infinity();
            std::vector<float> init_err(qlen, inf);
            std::vector<int> init_code(qlen, -1);
            std::vector<float> init_a(qlen, 0.0f);
            std::vector<float> init_r(qlen, 0.0f);
            ThrowIf(cudaMemcpy(root_sess.d_best_err0.ptr, init_err.data(), sizeof(float) * init_err.size(),
                               cudaMemcpyHostToDevice),
                    "Memcpy init best_err0");
            ThrowIf(cudaMemcpy(root_sess.d_best_code0.ptr, init_code.data(), sizeof(int) * init_code.size(),
                               cudaMemcpyHostToDevice),
                    "Memcpy init best_code0");
            ThrowIf(cudaMemcpy(root_sess.d_best_a0.ptr, init_a.data(), sizeof(float) * init_a.size(),
                               cudaMemcpyHostToDevice),
                    "Memcpy init best_a0");
            ThrowIf(cudaMemcpy(root_sess.d_best_r00.ptr, init_r.data(), sizeof(float) * init_r.size(),
                               cudaMemcpyHostToDevice),
                    "Memcpy init best_r00");
            ThrowIf(cudaMemcpy(root_sess.d_best_err1.ptr, init_err.data(), sizeof(float) * init_err.size(),
                               cudaMemcpyHostToDevice),
                    "Memcpy init best_err1");
            ThrowIf(cudaMemcpy(root_sess.d_best_code1.ptr, init_code.data(), sizeof(int) * init_code.size(),
                               cudaMemcpyHostToDevice),
                    "Memcpy init best_code1");
            ThrowIf(cudaMemcpy(root_sess.d_best_a1.ptr, init_a.data(), sizeof(float) * init_a.size(),
                               cudaMemcpyHostToDevice),
                    "Memcpy init best_a1");
            ThrowIf(cudaMemcpy(root_sess.d_best_r01.ptr, init_r.data(), sizeof(float) * init_r.size(),
                               cudaMemcpyHostToDevice),
                    "Memcpy init best_r01");
        }

        // Root GEMM in chunks of r_block.
        const int rc = rlen;
        for (int r0 = 0; r0 < h0; r0 += rc) {
            const int rcur = std::min(rc, h0 - r0);
            if (rcur <= 0) break;

            // scores = C0(:,r0:r0+rcur-1)^T * X
            const float alpha = 1.0f;
            const float beta = 0.0f;
            const int lda = d;
            const int ldb = d;
            const int ldc = rcur;
            const float* A = root_sess.d_C0.ptr + static_cast<std::size_t>(r0) * static_cast<std::size_t>(d);
            const float* B = root_sess.d_X.ptr;
            float* C = root_sess.d_scores.ptr;

            if (timing) {
                Timer t;
                ThrowIf(cublasSgemm(root_sess.handle,
                                    CUBLAS_OP_T,
                                    CUBLAS_OP_N,
                                    rcur,
                                    qlen,
                                    d,
                                    &alpha,
                                    A,
                                    lda,
                                    B,
                                    ldb,
                                    &beta,
                                    C,
                                    ldc),
                        "cublasSgemm(root)");
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(root_gemm)");
                timing->root_gemm += t.ElapsedSeconds();
            } else {
                ThrowIf(cublasSgemm(root_sess.handle,
                                    CUBLAS_OP_T,
                                    CUBLAS_OP_N,
                                    rcur,
                                    qlen,
                                    d,
                                    &alpha,
                                    A,
                                    lda,
                                    B,
                                    ldb,
                                    &beta,
                                    C,
                                    ldc),
                        "cublasSgemm(root)");
            }

            // Update top2.
            if (timing) {
                Timer t;
                UpdateTop2FromScoresKernel<<<qlen, 256>>>(root_sess.d_scores.ptr, rcur, rcur, r0,
                                                         root_sess.d_norm0.ptr,
                                                         root_sess.d_normx2.ptr, qlen,
                                                         root_sess.d_best_err0.ptr, root_sess.d_best_code0.ptr,
                                                         root_sess.d_best_a0.ptr, root_sess.d_best_r00.ptr,
                                                         root_sess.d_best_err1.ptr, root_sess.d_best_code1.ptr,
                                                         root_sess.d_best_a1.ptr, root_sess.d_best_r01.ptr);
                ThrowIf(cudaGetLastError(), "UpdateTop2FromScoresKernel");
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(root_update)");
                timing->root_update += t.ElapsedSeconds();
            } else {
                UpdateTop2FromScoresKernel<<<qlen, 256>>>(root_sess.d_scores.ptr, rcur, rcur, r0,
                                                         root_sess.d_norm0.ptr,
                                                         root_sess.d_normx2.ptr, qlen,
                                                         root_sess.d_best_err0.ptr, root_sess.d_best_code0.ptr,
                                                         root_sess.d_best_a0.ptr, root_sess.d_best_r00.ptr,
                                                         root_sess.d_best_err1.ptr, root_sess.d_best_code1.ptr,
                                                         root_sess.d_best_a1.ptr, root_sess.d_best_r01.ptr);
                ThrowIf(cudaGetLastError(), "UpdateTop2FromScoresKernel");
            }
        }

        // Upload xC_small tile to device (Hs×qlen).
        {
            const float* xCptr = xC_small.data.data() + static_cast<std::size_t>(q0) * static_cast<std::size_t>(Hs);
            ThrowIf(cudaMemcpy(expand_sess.d_xC.ptr, xCptr,
                               sizeof(float) * static_cast<std::size_t>(Hs) * static_cast<std::size_t>(qlen),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy xC_small H2D");
        }

        // Init beam2 state from root top2.
        {
            const dim3 block(256, 1, 1);
            const dim3 grid((qlen + static_cast<int>(block.x) - 1) / static_cast<int>(block.x), 1, 1);
            InitBeam2FromRootTop2Kernel<<<grid, block>>>(
                m, qlen,
                root_sess.d_best_err0.ptr, root_sess.d_best_code0.ptr, root_sess.d_best_a0.ptr, root_sess.d_best_r00.ptr,
                root_sess.d_best_err1.ptr, root_sess.d_best_code1.ptr, root_sess.d_best_a1.ptr, root_sess.d_best_r01.ptr,
                expand_sess.d_root_curr.ptr,
                expand_sess.d_codes_curr.ptr,
                expand_sess.d_err_curr.ptr,
                expand_sess.d_a_curr.ptr,
                expand_sess.d_R_curr.ptr);
            ThrowIf(cudaGetLastError(), "InitBeam2FromRootTop2Kernel");
            if (timing) {
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(init_h2)");
            }
        }

        // Expand layers 1..m-1.
        Timer t_expand;
        const dim3 block(128, 1, 1);
        const dim3 grid((qlen + static_cast<int>(block.x) - 1) / static_cast<int>(block.x), 1, 1);
        for (int layer = 1; layer < m; ++layer) {
            ExpandLayerBeam2Kernel<<<grid, block>>>(
                m, Hs, h0, layer, qlen,
                expand_sess.d_Gs.ptr,
                expand_sess.d_G0S.ptr,
                expand_sess.d_small_offsets.ptr,
                expand_sess.d_hvec.ptr,
                expand_sess.d_xC.ptr,
                expand_sess.d_root_curr.ptr,
                expand_sess.d_codes_curr.ptr,
                expand_sess.d_err_curr.ptr,
                expand_sess.d_a_curr.ptr,
                expand_sess.d_R_curr.ptr,
                expand_sess.d_root_next.ptr,
                expand_sess.d_codes_next.ptr,
                expand_sess.d_err_next.ptr,
                expand_sess.d_a_next.ptr,
                expand_sess.d_R_next.ptr);
            ThrowIf(cudaGetLastError(), "ExpandLayerBeam2Kernel");
            if (timing) {
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(expand_layer)");
            }
            // swap ping-pong (host-side swap of pointers)
            swap_buf(expand_sess.d_root_curr, expand_sess.d_root_next);
            swap_buf(expand_sess.d_codes_curr, expand_sess.d_codes_next);
            swap_buf(expand_sess.d_err_curr, expand_sess.d_err_next);
            swap_buf(expand_sess.d_a_curr, expand_sess.d_a_next);
            swap_buf(expand_sess.d_R_curr, expand_sess.d_R_next);
        }
        if (timing) {
            ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(expand_all)");
            timing->expand += t_expand.ElapsedSeconds();
        }

        // Finalize (best beam per sample).
        {
            const dim3 block2(256, 1, 1);
            const dim3 grid2((qlen + static_cast<int>(block2.x) - 1) / static_cast<int>(block2.x), 1, 1);
            FinalizeBestBeam2Kernel<<<grid2, block2>>>(
                m, qlen,
                expand_sess.d_root_curr.ptr,
                expand_sess.d_codes_curr.ptr,
                expand_sess.d_err_curr.ptr,
                expand_sess.d_cluster_id.ptr,
                expand_sess.d_B_small.ptr);
            ThrowIf(cudaGetLastError(), "FinalizeBestBeam2Kernel");
            if (timing) {
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(finalize)");
            }
        }

        // Download outputs into host.
        ThrowIf(cudaMemcpy(cluster_id->data() + static_cast<std::size_t>(q0),
                           expand_sess.d_cluster_id.ptr,
                           sizeof(std::uint32_t) * static_cast<std::size_t>(qlen),
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy cluster_id D2H");
        if (B_small->rows > 0) {
            Code* dst = B_small->data.data() + static_cast<std::size_t>(q0) * static_cast<std::size_t>(B_small->rows);
            ThrowIf(cudaMemcpy(dst,
                               expand_sess.d_B_small.ptr,
                               sizeof(Code) * static_cast<std::size_t>(B_small->rows) * static_cast<std::size_t>(qlen),
                               cudaMemcpyDeviceToHost),
                    "cudaMemcpy B_small D2H");
        }
    }
}

static void BeamSearchPrefixLSLargeRootCudaH2DeviceXcCore(const ColMajorMatrix<float>& X,
                                                          const PrecompLargeRoot& pre,
                                                          const DeviceMatF32View& xC_small_dev,
                                                          bool allow_tf32,
                                                          std::vector<std::uint32_t>* cluster_id,
                                                          ColMajorMatrix<Code>* B_small,
                                                          std::vector<float>* best_err,
                                                          BeamSearchLargeRootTiming* timing) {
    // Ensure all prior device work (e.g. GemmDevice on a non-blocking stream) has completed
    // before the beam session reads xC_small_dev / X_dev on the default (NULL) stream.
    // Without this, cudaStreamNonBlocking streams do NOT synchronize with stream 0,
    // causing a data race that produces catastrophic errors (especially on Windows).
    ThrowIf(cudaDeviceSynchronize(), "cudaDeviceSynchronize (H2DeviceXcCore entry)");

    if (!cluster_id || !B_small) return;
    const int n = X.cols;
    const int d = X.rows;
    const int m = pre.m;
    const int h0 = pre.h_vec.empty() ? 0 : pre.h_vec[0];
    const int Hs = pre.H_small;
    if (!pre.ready || n <= 0 || d <= 0 || m <= 0 || h0 <= 0 || Hs <= 0) {
        return;
    }
    if (!xC_small_dev.ptr || xC_small_dev.rows != Hs || xC_small_dev.cols != n || xC_small_dev.ld != Hs) {
        throw std::runtime_error("BeamSearchPrefixLSLargeRootCudaH2DeviceXc: invalid device xC_small view.");
    }

    cluster_id->resize(static_cast<std::size_t>(n));
    B_small->rows = std::max(0, m - 1);
    B_small->cols = n;
    B_small->data.resize(static_cast<std::size_t>(B_small->rows) * static_cast<std::size_t>(n));
    if (best_err) {
        best_err->resize(static_cast<std::size_t>(n));
    }

    const int q_block = std::min(8192, std::max(1, n));
    const int r_block = 1024;

    static thread_local CudaRootBeamSession root_sess;
    static thread_local CudaExpandH2Session expand_sess;
    root_sess.EnsureHandle(allow_tf32);
    root_sess.EnsurePre(pre);
    expand_sess.EnsurePre(pre);

    auto swap_buf = [](auto& a, auto& b) {
        std::swap(a.ptr, b.ptr);
        std::swap(a.n, b.n);
    };

    for (int q0 = 0; q0 < n; q0 += q_block) {
        const int qlen = std::min(q_block, n - q0);
        const int rlen = std::min(r_block, h0);
        root_sess.EnsureBatch(qlen, rlen);
        expand_sess.EnsureBatch(qlen);

        // Upload X block.
        const float* Xptr = X.data.data() + static_cast<std::size_t>(q0) * static_cast<std::size_t>(d);
        ThrowIf(cudaMemcpy(root_sess.d_X.ptr,
                           Xptr,
                           sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(qlen),
                           cudaMemcpyHostToDevice),
                "cudaMemcpy X H2D");

        // norm_x2
        if (timing) {
            Timer t;
            Norm2ColsKernel<<<qlen, 256>>>(root_sess.d_X.ptr, d, d, qlen, root_sess.d_normx2.ptr);
            ThrowIf(cudaGetLastError(), "Norm2ColsKernel");
            ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(norm_x2)");
            timing->norm_x2 += t.ElapsedSeconds();
        } else {
            Norm2ColsKernel<<<qlen, 256>>>(root_sess.d_X.ptr, d, d, qlen, root_sess.d_normx2.ptr);
            ThrowIf(cudaGetLastError(), "Norm2ColsKernel");
        }

        // Init best arrays.
        {
            const float inf = std::numeric_limits<float>::infinity();
            std::vector<float> init_err(qlen, inf);
            std::vector<int> init_code(qlen, -1);
            std::vector<float> init_a(qlen, 0.0f);
            std::vector<float> init_r(qlen, 0.0f);
            ThrowIf(cudaMemcpy(root_sess.d_best_err0.ptr, init_err.data(), sizeof(float) * init_err.size(),
                               cudaMemcpyHostToDevice),
                    "Memcpy init best_err0");
            ThrowIf(cudaMemcpy(root_sess.d_best_code0.ptr, init_code.data(), sizeof(int) * init_code.size(),
                               cudaMemcpyHostToDevice),
                    "Memcpy init best_code0");
            ThrowIf(cudaMemcpy(root_sess.d_best_a0.ptr, init_a.data(), sizeof(float) * init_a.size(),
                               cudaMemcpyHostToDevice),
                    "Memcpy init best_a0");
            ThrowIf(cudaMemcpy(root_sess.d_best_r00.ptr, init_r.data(), sizeof(float) * init_r.size(),
                               cudaMemcpyHostToDevice),
                    "Memcpy init best_r00");
            ThrowIf(cudaMemcpy(root_sess.d_best_err1.ptr, init_err.data(), sizeof(float) * init_err.size(),
                               cudaMemcpyHostToDevice),
                    "Memcpy init best_err1");
            ThrowIf(cudaMemcpy(root_sess.d_best_code1.ptr, init_code.data(), sizeof(int) * init_code.size(),
                               cudaMemcpyHostToDevice),
                    "Memcpy init best_code1");
            ThrowIf(cudaMemcpy(root_sess.d_best_a1.ptr, init_a.data(), sizeof(float) * init_a.size(),
                               cudaMemcpyHostToDevice),
                    "Memcpy init best_a1");
            ThrowIf(cudaMemcpy(root_sess.d_best_r01.ptr, init_r.data(), sizeof(float) * init_r.size(),
                               cudaMemcpyHostToDevice),
                    "Memcpy init best_r01");
        }

        // Root GEMM in chunks of r_block.
        const int rc = rlen;
        for (int r0 = 0; r0 < h0; r0 += rc) {
            const int rcur = std::min(rc, h0 - r0);
            if (rcur <= 0) break;

            const float alpha = 1.0f;
            const float beta = 0.0f;
            const int lda = d;
            const int ldb = d;
            const int ldc = rcur;
            const float* A = root_sess.d_C0.ptr + static_cast<std::size_t>(r0) * static_cast<std::size_t>(d);
            const float* B = root_sess.d_X.ptr;
            float* C = root_sess.d_scores.ptr;

            if (timing) {
                Timer t;
                ThrowIf(cublasSgemm(root_sess.handle,
                                    CUBLAS_OP_T,
                                    CUBLAS_OP_N,
                                    rcur,
                                    qlen,
                                    d,
                                    &alpha,
                                    A,
                                    lda,
                                    B,
                                    ldb,
                                    &beta,
                                    C,
                                    ldc),
                        "cublasSgemm(root)");
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(root_gemm)");
                timing->root_gemm += t.ElapsedSeconds();
            } else {
                ThrowIf(cublasSgemm(root_sess.handle,
                                    CUBLAS_OP_T,
                                    CUBLAS_OP_N,
                                    rcur,
                                    qlen,
                                    d,
                                    &alpha,
                                    A,
                                    lda,
                                    B,
                                    ldb,
                                    &beta,
                                    C,
                                    ldc),
                        "cublasSgemm(root)");
            }

            if (timing) {
                Timer t;
                UpdateTop2FromScoresKernel<<<qlen, 256>>>(root_sess.d_scores.ptr, rcur, rcur, r0,
                                                         root_sess.d_norm0.ptr,
                                                         root_sess.d_normx2.ptr, qlen,
                                                         root_sess.d_best_err0.ptr, root_sess.d_best_code0.ptr,
                                                         root_sess.d_best_a0.ptr, root_sess.d_best_r00.ptr,
                                                         root_sess.d_best_err1.ptr, root_sess.d_best_code1.ptr,
                                                         root_sess.d_best_a1.ptr, root_sess.d_best_r01.ptr);
                ThrowIf(cudaGetLastError(), "UpdateTop2FromScoresKernel");
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(root_update)");
                timing->root_update += t.ElapsedSeconds();
            } else {
                UpdateTop2FromScoresKernel<<<qlen, 256>>>(root_sess.d_scores.ptr, rcur, rcur, r0,
                                                         root_sess.d_norm0.ptr,
                                                         root_sess.d_normx2.ptr, qlen,
                                                         root_sess.d_best_err0.ptr, root_sess.d_best_code0.ptr,
                                                         root_sess.d_best_a0.ptr, root_sess.d_best_r00.ptr,
                                                         root_sess.d_best_err1.ptr, root_sess.d_best_code1.ptr,
                                                         root_sess.d_best_a1.ptr, root_sess.d_best_r01.ptr);
                ThrowIf(cudaGetLastError(), "UpdateTop2FromScoresKernel");
            }
        }

        // Init beam2 state from root top2.
        {
            const dim3 block(256, 1, 1);
            const dim3 grid((qlen + static_cast<int>(block.x) - 1) / static_cast<int>(block.x), 1, 1);
            InitBeam2FromRootTop2Kernel<<<grid, block>>>(
                m, qlen,
                root_sess.d_best_err0.ptr, root_sess.d_best_code0.ptr, root_sess.d_best_a0.ptr, root_sess.d_best_r00.ptr,
                root_sess.d_best_err1.ptr, root_sess.d_best_code1.ptr, root_sess.d_best_a1.ptr, root_sess.d_best_r01.ptr,
                expand_sess.d_root_curr.ptr,
                expand_sess.d_codes_curr.ptr,
                expand_sess.d_err_curr.ptr,
                expand_sess.d_a_curr.ptr,
                expand_sess.d_R_curr.ptr);
            ThrowIf(cudaGetLastError(), "InitBeam2FromRootTop2Kernel");
            if (timing) {
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(init_h2)");
            }
        }

        // Expand layers 1..m-1 using the device xC pointer for the current q0 block.
        Timer t_expand;
        const dim3 block(128, 1, 1);
        const dim3 grid((qlen + static_cast<int>(block.x) - 1) / static_cast<int>(block.x), 1, 1);
        const float* xC_ptr = xC_small_dev.ptr + static_cast<std::size_t>(q0) * static_cast<std::size_t>(Hs);
        for (int layer = 1; layer < m; ++layer) {
            ExpandLayerBeam2Kernel<<<grid, block>>>(
                m, Hs, h0, layer, qlen,
                expand_sess.d_Gs.ptr,
                expand_sess.d_G0S.ptr,
                expand_sess.d_small_offsets.ptr,
                expand_sess.d_hvec.ptr,
                xC_ptr,
                expand_sess.d_root_curr.ptr,
                expand_sess.d_codes_curr.ptr,
                expand_sess.d_err_curr.ptr,
                expand_sess.d_a_curr.ptr,
                expand_sess.d_R_curr.ptr,
                expand_sess.d_root_next.ptr,
                expand_sess.d_codes_next.ptr,
                expand_sess.d_err_next.ptr,
                expand_sess.d_a_next.ptr,
                expand_sess.d_R_next.ptr);
            ThrowIf(cudaGetLastError(), "ExpandLayerBeam2Kernel");
            if (timing) {
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(expand_layer)");
            }
            swap_buf(expand_sess.d_root_curr, expand_sess.d_root_next);
            swap_buf(expand_sess.d_codes_curr, expand_sess.d_codes_next);
            swap_buf(expand_sess.d_err_curr, expand_sess.d_err_next);
            swap_buf(expand_sess.d_a_curr, expand_sess.d_a_next);
            swap_buf(expand_sess.d_R_curr, expand_sess.d_R_next);
        }
        if (timing) {
            ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(expand_all)");
            timing->expand += t_expand.ElapsedSeconds();
        }

        // Finalize (best beam per sample).
        {
            const dim3 block2(256, 1, 1);
            const dim3 grid2((qlen + static_cast<int>(block2.x) - 1) / static_cast<int>(block2.x), 1, 1);
            FinalizeBestBeam2Kernel<<<grid2, block2>>>(
                m, qlen,
                expand_sess.d_root_curr.ptr,
                expand_sess.d_codes_curr.ptr,
                expand_sess.d_err_curr.ptr,
                expand_sess.d_cluster_id.ptr,
                expand_sess.d_B_small.ptr);
            ThrowIf(cudaGetLastError(), "FinalizeBestBeam2Kernel");
            if (timing) {
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(finalize)");
            }
        }

        if (best_err) {
            std::vector<float> err_all(static_cast<std::size_t>(qlen) * 2);
            ThrowIf(cudaMemcpy(err_all.data(),
                               expand_sess.d_err_curr.ptr,
                               sizeof(float) * err_all.size(),
                               cudaMemcpyDeviceToHost),
                    "cudaMemcpy err_curr D2H");
            for (int j = 0; j < qlen; ++j) {
                const float e0 = err_all[static_cast<std::size_t>(j) * 2 + 0];
                const float e1 = err_all[static_cast<std::size_t>(j) * 2 + 1];
                float e = (e1 < e0) ? e1 : e0;  // tie => beam0
                if (e < 0.0f) e = 0.0f;
                (*best_err)[static_cast<std::size_t>(q0 + j)] = e;
            }
        }

        // Download outputs into host.
        ThrowIf(cudaMemcpy(cluster_id->data() + static_cast<std::size_t>(q0),
                           expand_sess.d_cluster_id.ptr,
                           sizeof(std::uint32_t) * static_cast<std::size_t>(qlen),
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy cluster_id D2H");
        if (B_small->rows > 0) {
            Code* dst = B_small->data.data() + static_cast<std::size_t>(q0) * static_cast<std::size_t>(B_small->rows);
            ThrowIf(cudaMemcpy(dst,
                               expand_sess.d_B_small.ptr,
                               sizeof(Code) * static_cast<std::size_t>(B_small->rows) * static_cast<std::size_t>(qlen),
                               cudaMemcpyDeviceToHost),
                    "cudaMemcpy B_small D2H");
        }
    }
}

void BeamSearchPrefixLSLargeRootCudaH2DeviceXc(const ColMajorMatrix<float>& X,
                                               const PrecompLargeRoot& pre,
                                               const DeviceMatF32View& xC_small_dev,
                                               bool allow_tf32,
                                               std::vector<std::uint32_t>* cluster_id,
                                               ColMajorMatrix<Code>* B_small,
                                               BeamSearchLargeRootTiming* timing) {
    BeamSearchPrefixLSLargeRootCudaH2DeviceXcCore(X, pre, xC_small_dev, allow_tf32, cluster_id, B_small,
                                                 /*best_err=*/nullptr, timing);
}

void BeamSearchPrefixLSLargeRootCudaH2DeviceXcWithBestErr(const ColMajorMatrix<float>& X,
                                                          const PrecompLargeRoot& pre,
                                                          const DeviceMatF32View& xC_small_dev,
                                                          bool allow_tf32,
                                                          std::vector<std::uint32_t>* cluster_id,
                                                          ColMajorMatrix<Code>* B_small,
                                                          std::vector<float>* best_err,
                                                          BeamSearchLargeRootTiming* timing) {
    BeamSearchPrefixLSLargeRootCudaH2DeviceXcCore(X, pre, xC_small_dev, allow_tf32, cluster_id, B_small,
                                                 best_err, timing);
}

static void BeamSearchPrefixLSLargeRootCudaH2DeviceXcDeviceXCore(const DeviceMatF32View& X_dev,
                                                                 const PrecompLargeRoot& pre,
                                                                 const DeviceMatF32View& xC_small_dev,
                                                                 bool allow_tf32,
                                                                 std::vector<std::uint32_t>* cluster_id,
                                                                 ColMajorMatrix<Code>* B_small,
                                                                 std::vector<float>* best_err,
                                                                 BeamSearchLargeRootTiming* timing) {
    // Synchronize: xC_small_dev / X_dev may have been produced on a non-blocking stream.
    ThrowIf(cudaDeviceSynchronize(), "cudaDeviceSynchronize (H2DeviceXcDeviceXCore entry)");

    if (!cluster_id || !B_small) return;
    const int n = X_dev.cols;
    const int d = X_dev.rows;
    const int m = pre.m;
    const int h0 = pre.h_vec.empty() ? 0 : pre.h_vec[0];
    const int Hs = pre.H_small;
    if (!pre.ready || n <= 0 || d <= 0 || m <= 0 || h0 <= 0 || Hs <= 0) return;
    if (!X_dev.ptr || X_dev.ld < d) {
        throw std::runtime_error("BeamSearchPrefixLSLargeRootCudaH2DeviceXcDeviceXCore: invalid X_dev view.");
    }
    if (!xC_small_dev.ptr || xC_small_dev.rows != Hs || xC_small_dev.cols != n || xC_small_dev.ld != Hs) {
        throw std::runtime_error("BeamSearchPrefixLSLargeRootCudaH2DeviceXcDeviceXCore: invalid device xC_small view.");
    }

    cluster_id->resize(static_cast<std::size_t>(n));
    B_small->rows = std::max(0, m - 1);
    B_small->cols = n;
    B_small->data.resize(static_cast<std::size_t>(B_small->rows) * static_cast<std::size_t>(n));
    if (best_err) best_err->resize(static_cast<std::size_t>(n));

    const int q_block = std::min(8192, std::max(1, n));
    const int r_block = 1024;

    static thread_local CudaRootBeamSession root_sess;
    static thread_local CudaExpandH2Session expand_sess;
    root_sess.EnsureHandle(allow_tf32);
    root_sess.EnsurePre(pre);
    expand_sess.EnsurePre(pre);

    auto swap_buf = [](auto& a, auto& b) {
        std::swap(a.ptr, b.ptr);
        std::swap(a.n, b.n);
    };

    for (int q0 = 0; q0 < n; q0 += q_block) {
        const int qlen = std::min(q_block, n - q0);
        const int rlen = std::min(r_block, h0);
        root_sess.EnsureBatch(qlen, rlen);
        expand_sess.EnsureBatch(qlen);

        const float* Xptr = X_dev.ptr + static_cast<std::size_t>(q0) * static_cast<std::size_t>(X_dev.ld);

        // norm_x2
        if (timing) {
            Timer t;
            Norm2ColsKernel<<<qlen, 256>>>(Xptr, X_dev.ld, d, qlen, root_sess.d_normx2.ptr);
            ThrowIf(cudaGetLastError(), "Norm2ColsKernel");
            ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(norm_x2)");
            timing->norm_x2 += t.ElapsedSeconds();
        } else {
            Norm2ColsKernel<<<qlen, 256>>>(Xptr, X_dev.ld, d, qlen, root_sess.d_normx2.ptr);
            ThrowIf(cudaGetLastError(), "Norm2ColsKernel");
        }

        // Init best arrays.
        {
            const float inf = std::numeric_limits<float>::infinity();
            std::vector<float> init_err(qlen, inf);
            std::vector<int> init_code(qlen, -1);
            std::vector<float> init_a(qlen, 0.0f);
            std::vector<float> init_r(qlen, 0.0f);
            ThrowIf(cudaMemcpy(root_sess.d_best_err0.ptr, init_err.data(), sizeof(float) * init_err.size(),
                               cudaMemcpyHostToDevice),
                    "Memcpy init best_err0");
            ThrowIf(cudaMemcpy(root_sess.d_best_code0.ptr, init_code.data(), sizeof(int) * init_code.size(),
                               cudaMemcpyHostToDevice),
                    "Memcpy init best_code0");
            ThrowIf(cudaMemcpy(root_sess.d_best_a0.ptr, init_a.data(), sizeof(float) * init_a.size(),
                               cudaMemcpyHostToDevice),
                    "Memcpy init best_a0");
            ThrowIf(cudaMemcpy(root_sess.d_best_r00.ptr, init_r.data(), sizeof(float) * init_r.size(),
                               cudaMemcpyHostToDevice),
                    "Memcpy init best_r00");
            ThrowIf(cudaMemcpy(root_sess.d_best_err1.ptr, init_err.data(), sizeof(float) * init_err.size(),
                               cudaMemcpyHostToDevice),
                    "Memcpy init best_err1");
            ThrowIf(cudaMemcpy(root_sess.d_best_code1.ptr, init_code.data(), sizeof(int) * init_code.size(),
                               cudaMemcpyHostToDevice),
                    "Memcpy init best_code1");
            ThrowIf(cudaMemcpy(root_sess.d_best_a1.ptr, init_a.data(), sizeof(float) * init_a.size(),
                               cudaMemcpyHostToDevice),
                    "Memcpy init best_a1");
            ThrowIf(cudaMemcpy(root_sess.d_best_r01.ptr, init_r.data(), sizeof(float) * init_r.size(),
                               cudaMemcpyHostToDevice),
                    "Memcpy init best_r01");
        }

        // Root GEMM in chunks of r_block.
        const int rc = rlen;
        for (int r0 = 0; r0 < h0; r0 += rc) {
            const int rcur = std::min(rc, h0 - r0);
            if (rcur <= 0) break;

            const float alpha = 1.0f;
            const float beta = 0.0f;
            const int lda = d;
            const int ldb = X_dev.ld;
            const int ldc = rcur;
            const float* A = root_sess.d_C0.ptr + static_cast<std::size_t>(r0) * static_cast<std::size_t>(d);
            const float* B = Xptr;
            float* C = root_sess.d_scores.ptr;

            if (timing) {
                Timer t;
                ThrowIf(cublasSgemm(root_sess.handle,
                                    CUBLAS_OP_T,
                                    CUBLAS_OP_N,
                                    rcur,
                                    qlen,
                                    d,
                                    &alpha,
                                    A,
                                    lda,
                                    B,
                                    ldb,
                                    &beta,
                                    C,
                                    ldc),
                        "cublasSgemm(root)");
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(root_gemm)");
                timing->root_gemm += t.ElapsedSeconds();
            } else {
                ThrowIf(cublasSgemm(root_sess.handle,
                                    CUBLAS_OP_T,
                                    CUBLAS_OP_N,
                                    rcur,
                                    qlen,
                                    d,
                                    &alpha,
                                    A,
                                    lda,
                                    B,
                                    ldb,
                                    &beta,
                                    C,
                                    ldc),
                        "cublasSgemm(root)");
            }

            if (timing) {
                Timer t;
                UpdateTop2FromScoresKernel<<<qlen, 256>>>(root_sess.d_scores.ptr, rcur, rcur, r0,
                                                         root_sess.d_norm0.ptr,
                                                         root_sess.d_normx2.ptr, qlen,
                                                         root_sess.d_best_err0.ptr, root_sess.d_best_code0.ptr,
                                                         root_sess.d_best_a0.ptr, root_sess.d_best_r00.ptr,
                                                         root_sess.d_best_err1.ptr, root_sess.d_best_code1.ptr,
                                                         root_sess.d_best_a1.ptr, root_sess.d_best_r01.ptr);
                ThrowIf(cudaGetLastError(), "UpdateTop2FromScoresKernel");
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(root_update)");
                timing->root_update += t.ElapsedSeconds();
            } else {
                UpdateTop2FromScoresKernel<<<qlen, 256>>>(root_sess.d_scores.ptr, rcur, rcur, r0,
                                                         root_sess.d_norm0.ptr,
                                                         root_sess.d_normx2.ptr, qlen,
                                                         root_sess.d_best_err0.ptr, root_sess.d_best_code0.ptr,
                                                         root_sess.d_best_a0.ptr, root_sess.d_best_r00.ptr,
                                                         root_sess.d_best_err1.ptr, root_sess.d_best_code1.ptr,
                                                         root_sess.d_best_a1.ptr, root_sess.d_best_r01.ptr);
                ThrowIf(cudaGetLastError(), "UpdateTop2FromScoresKernel");
            }
        }

        // Init beam2 state from root top2.
        {
            const dim3 block(256, 1, 1);
            const dim3 grid((qlen + static_cast<int>(block.x) - 1) / static_cast<int>(block.x), 1, 1);
            InitBeam2FromRootTop2Kernel<<<grid, block>>>(
                m, qlen,
                root_sess.d_best_err0.ptr, root_sess.d_best_code0.ptr, root_sess.d_best_a0.ptr, root_sess.d_best_r00.ptr,
                root_sess.d_best_err1.ptr, root_sess.d_best_code1.ptr, root_sess.d_best_a1.ptr, root_sess.d_best_r01.ptr,
                expand_sess.d_root_curr.ptr,
                expand_sess.d_codes_curr.ptr,
                expand_sess.d_err_curr.ptr,
                expand_sess.d_a_curr.ptr,
                expand_sess.d_R_curr.ptr);
            ThrowIf(cudaGetLastError(), "InitBeam2FromRootTop2Kernel");
            if (timing) {
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(init_h2)");
            }
        }

        // Expand layers 1..m-1 using the device xC pointer for the current q0 block.
        Timer t_expand;
        const dim3 block(128, 1, 1);
        const dim3 grid((qlen + static_cast<int>(block.x) - 1) / static_cast<int>(block.x), 1, 1);
        const float* xC_ptr = xC_small_dev.ptr + static_cast<std::size_t>(q0) * static_cast<std::size_t>(Hs);
        for (int layer = 1; layer < m; ++layer) {
            ExpandLayerBeam2Kernel<<<grid, block>>>(
                m, Hs, h0, layer, qlen,
                expand_sess.d_Gs.ptr,
                expand_sess.d_G0S.ptr,
                expand_sess.d_small_offsets.ptr,
                expand_sess.d_hvec.ptr,
                xC_ptr,
                expand_sess.d_root_curr.ptr,
                expand_sess.d_codes_curr.ptr,
                expand_sess.d_err_curr.ptr,
                expand_sess.d_a_curr.ptr,
                expand_sess.d_R_curr.ptr,
                expand_sess.d_root_next.ptr,
                expand_sess.d_codes_next.ptr,
                expand_sess.d_err_next.ptr,
                expand_sess.d_a_next.ptr,
                expand_sess.d_R_next.ptr);
            ThrowIf(cudaGetLastError(), "ExpandLayerBeam2Kernel");
            if (timing) {
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(expand_layer)");
            }
            swap_buf(expand_sess.d_root_curr, expand_sess.d_root_next);
            swap_buf(expand_sess.d_codes_curr, expand_sess.d_codes_next);
            swap_buf(expand_sess.d_err_curr, expand_sess.d_err_next);
            swap_buf(expand_sess.d_a_curr, expand_sess.d_a_next);
            swap_buf(expand_sess.d_R_curr, expand_sess.d_R_next);
        }
        if (timing) {
            ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(expand_all)");
            timing->expand += t_expand.ElapsedSeconds();
        }

        // Finalize (best beam per sample).
        {
            const dim3 block2(256, 1, 1);
            const dim3 grid2((qlen + static_cast<int>(block2.x) - 1) / static_cast<int>(block2.x), 1, 1);
            FinalizeBestBeam2Kernel<<<grid2, block2>>>(
                m, qlen,
                expand_sess.d_root_curr.ptr,
                expand_sess.d_codes_curr.ptr,
                expand_sess.d_err_curr.ptr,
                expand_sess.d_cluster_id.ptr,
                expand_sess.d_B_small.ptr);
            ThrowIf(cudaGetLastError(), "FinalizeBestBeam2Kernel");
            if (timing) {
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(finalize)");
            }
        }

        if (best_err) {
            std::vector<float> err_all(static_cast<std::size_t>(qlen) * 2);
            ThrowIf(cudaMemcpy(err_all.data(),
                               expand_sess.d_err_curr.ptr,
                               sizeof(float) * err_all.size(),
                               cudaMemcpyDeviceToHost),
                    "cudaMemcpy err_curr D2H");
            for (int j = 0; j < qlen; ++j) {
                const float e0 = err_all[static_cast<std::size_t>(j) * 2 + 0];
                const float e1 = err_all[static_cast<std::size_t>(j) * 2 + 1];
                float e = (e1 < e0) ? e1 : e0;  // tie => beam0
                if (e < 0.0f) e = 0.0f;
                (*best_err)[static_cast<std::size_t>(q0 + j)] = e;
            }
        }

        // Download outputs into host.
        ThrowIf(cudaMemcpy(cluster_id->data() + static_cast<std::size_t>(q0),
                           expand_sess.d_cluster_id.ptr,
                           sizeof(std::uint32_t) * static_cast<std::size_t>(qlen),
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy cluster_id D2H");
        if (B_small->rows > 0) {
            Code* dst = B_small->data.data() + static_cast<std::size_t>(q0) * static_cast<std::size_t>(B_small->rows);
            ThrowIf(cudaMemcpy(dst,
                               expand_sess.d_B_small.ptr,
                               sizeof(Code) * static_cast<std::size_t>(B_small->rows) * static_cast<std::size_t>(qlen),
                               cudaMemcpyDeviceToHost),
                    "cudaMemcpy B_small D2H");
        }
    }
}

void BeamSearchPrefixLSLargeRootCudaH2DeviceXcDeviceX(const DeviceMatF32View& X_dev,
                                                      const PrecompLargeRoot& pre,
                                                      const DeviceMatF32View& xC_small_dev,
                                                      bool allow_tf32,
                                                      std::vector<std::uint32_t>* cluster_id,
                                                      ColMajorMatrix<Code>* B_small,
                                                      BeamSearchLargeRootTiming* timing) {
    BeamSearchPrefixLSLargeRootCudaH2DeviceXcDeviceXCore(X_dev, pre, xC_small_dev, allow_tf32, cluster_id, B_small,
                                                        /*best_err=*/nullptr, timing);
}

void BeamSearchPrefixLSLargeRootCudaH2DeviceXcDeviceXWithBestErr(const DeviceMatF32View& X_dev,
                                                                 const PrecompLargeRoot& pre,
                                                                 const DeviceMatF32View& xC_small_dev,
                                                                 bool allow_tf32,
                                                                 std::vector<std::uint32_t>* cluster_id,
                                                                 ColMajorMatrix<Code>* B_small,
                                                                 std::vector<float>* best_err,
                                                                 BeamSearchLargeRootTiming* timing) {
    BeamSearchPrefixLSLargeRootCudaH2DeviceXcDeviceXCore(X_dev, pre, xC_small_dev, allow_tf32, cluster_id, B_small,
                                                        best_err, timing);
}

void BeamSearchPrefixLSLargeRootCudaH4(const ColMajorMatrix<float>& X,
                                       const PrecompLargeRoot& pre,
                                       const ColMajorMatrix<float>& xC_small,
                                       bool allow_tf32,
                                       std::vector<std::uint32_t>* cluster_id,
                                       ColMajorMatrix<Code>* B_small,
                                       BeamSearchLargeRootTiming* timing) {
    if (!cluster_id || !B_small) return;
    const int n = X.cols;
    const int d = X.rows;
    const int m = pre.m;
    const int h0 = pre.h_vec.empty() ? 0 : pre.h_vec[0];
    const int Hs = pre.H_small;
    if (!pre.ready || n <= 0 || d <= 0 || m <= 0 || h0 <= 0 || Hs <= 0) {
        return;
    }
    cluster_id->resize(static_cast<std::size_t>(n));
    B_small->rows = std::max(0, m - 1);
    B_small->cols = n;
    B_small->data.resize(static_cast<std::size_t>(B_small->rows) * static_cast<std::size_t>(n));

    const int q_block = std::min(8192, std::max(1, n));
    const int r_block = 1024;

    static thread_local CudaRootBeamSession root_sess;
    static thread_local CudaExpandH4Session expand_sess;
    root_sess.EnsureHandle(allow_tf32);
    root_sess.EnsurePre(pre);
    expand_sess.EnsurePre(pre);

    auto swap_buf = [](auto& a, auto& b) {
        std::swap(a.ptr, b.ptr);
        std::swap(a.n, b.n);
    };

    for (int q0 = 0; q0 < n; q0 += q_block) {
        const int qlen = std::min(q_block, n - q0);
        const int rlen = std::min(r_block, h0);
        root_sess.EnsureBatch(qlen, rlen);
        expand_sess.EnsureBatch(qlen);

        const float* Xptr = X.data.data() + static_cast<std::size_t>(q0) * static_cast<std::size_t>(d);
        ThrowIf(cudaMemcpy(root_sess.d_X.ptr,
                           Xptr,
                           sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(qlen),
                           cudaMemcpyHostToDevice),
                "cudaMemcpy X H2D");

        if (timing) {
            Timer t;
            Norm2ColsKernel<<<qlen, 256>>>(root_sess.d_X.ptr, d, d, qlen, root_sess.d_normx2.ptr);
            ThrowIf(cudaGetLastError(), "Norm2ColsKernel");
            ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(norm_x2)");
            timing->norm_x2 += t.ElapsedSeconds();
        } else {
            Norm2ColsKernel<<<qlen, 256>>>(root_sess.d_X.ptr, d, d, qlen, root_sess.d_normx2.ptr);
            ThrowIf(cudaGetLastError(), "Norm2ColsKernel");
        }

        // Init best arrays to inf/-1 on device.
        {
            const float inf = std::numeric_limits<float>::infinity();
            std::vector<float> init_err(qlen, inf);
            std::vector<int> init_code(qlen, -1);
            std::vector<float> init_a(qlen, 0.0f);
            std::vector<float> init_r(qlen, 0.0f);

            auto H2D_f = [&](DeviceBuf<float>& dst, const std::vector<float>& src, const char* what) {
                ThrowIf(cudaMemcpy(dst.ptr, src.data(), sizeof(float) * src.size(), cudaMemcpyHostToDevice), what);
            };
            auto H2D_i = [&](DeviceBuf<int>& dst, const std::vector<int>& src, const char* what) {
                ThrowIf(cudaMemcpy(dst.ptr, src.data(), sizeof(int) * src.size(), cudaMemcpyHostToDevice), what);
            };

            H2D_f(root_sess.d_best_err0, init_err, "Memcpy init best_err0");
            H2D_i(root_sess.d_best_code0, init_code, "Memcpy init best_code0");
            H2D_f(root_sess.d_best_a0, init_a, "Memcpy init best_a0");
            H2D_f(root_sess.d_best_r00, init_r, "Memcpy init best_r00");

            H2D_f(root_sess.d_best_err1, init_err, "Memcpy init best_err1");
            H2D_i(root_sess.d_best_code1, init_code, "Memcpy init best_code1");
            H2D_f(root_sess.d_best_a1, init_a, "Memcpy init best_a1");
            H2D_f(root_sess.d_best_r01, init_r, "Memcpy init best_r01");

            H2D_f(root_sess.d_best_err2, init_err, "Memcpy init best_err2");
            H2D_i(root_sess.d_best_code2, init_code, "Memcpy init best_code2");
            H2D_f(root_sess.d_best_a2, init_a, "Memcpy init best_a2");
            H2D_f(root_sess.d_best_r02, init_r, "Memcpy init best_r02");

            H2D_f(root_sess.d_best_err3, init_err, "Memcpy init best_err3");
            H2D_i(root_sess.d_best_code3, init_code, "Memcpy init best_code3");
            H2D_f(root_sess.d_best_a3, init_a, "Memcpy init best_a3");
            H2D_f(root_sess.d_best_r03, init_r, "Memcpy init best_r03");
        }

        // Root GEMM in chunks.
        const int rc = rlen;
        for (int r0 = 0; r0 < h0; r0 += rc) {
            const int rcur = std::min(rc, h0 - r0);
            if (rcur <= 0) break;

            const float alpha = 1.0f;
            const float beta = 0.0f;
            const int lda = d;
            const int ldb = d;
            const int ldc = rcur;
            const float* A = root_sess.d_C0.ptr + static_cast<std::size_t>(r0) * static_cast<std::size_t>(d);
            const float* B = root_sess.d_X.ptr;
            float* C = root_sess.d_scores.ptr;

            if (timing) {
                Timer t;
                ThrowIf(cublasSgemm(root_sess.handle,
                                    CUBLAS_OP_T,
                                    CUBLAS_OP_N,
                                    rcur,
                                    qlen,
                                    d,
                                    &alpha,
                                    A,
                                    lda,
                                    B,
                                    ldb,
                                    &beta,
                                    C,
                                    ldc),
                        "cublasSgemm(root)");
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(root_gemm)");
                timing->root_gemm += t.ElapsedSeconds();
            } else {
                ThrowIf(cublasSgemm(root_sess.handle,
                                    CUBLAS_OP_T,
                                    CUBLAS_OP_N,
                                    rcur,
                                    qlen,
                                    d,
                                    &alpha,
                                    A,
                                    lda,
                                    B,
                                    ldb,
                                    &beta,
                                    C,
                                    ldc),
                        "cublasSgemm(root)");
            }

            if (timing) {
                Timer t;
                UpdateTop4FromScoresKernel<<<qlen, 256>>>(root_sess.d_scores.ptr, rcur, rcur, r0,
                                                         root_sess.d_norm0.ptr,
                                                         root_sess.d_normx2.ptr, qlen,
                                                         root_sess.d_best_err0.ptr, root_sess.d_best_code0.ptr,
                                                         root_sess.d_best_a0.ptr, root_sess.d_best_r00.ptr,
                                                         root_sess.d_best_err1.ptr, root_sess.d_best_code1.ptr,
                                                         root_sess.d_best_a1.ptr, root_sess.d_best_r01.ptr,
                                                         root_sess.d_best_err2.ptr, root_sess.d_best_code2.ptr,
                                                         root_sess.d_best_a2.ptr, root_sess.d_best_r02.ptr,
                                                         root_sess.d_best_err3.ptr, root_sess.d_best_code3.ptr,
                                                         root_sess.d_best_a3.ptr, root_sess.d_best_r03.ptr);
                ThrowIf(cudaGetLastError(), "UpdateTop4FromScoresKernel");
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(root_update)");
                timing->root_update += t.ElapsedSeconds();
            } else {
                UpdateTop4FromScoresKernel<<<qlen, 256>>>(root_sess.d_scores.ptr, rcur, rcur, r0,
                                                         root_sess.d_norm0.ptr,
                                                         root_sess.d_normx2.ptr, qlen,
                                                         root_sess.d_best_err0.ptr, root_sess.d_best_code0.ptr,
                                                         root_sess.d_best_a0.ptr, root_sess.d_best_r00.ptr,
                                                         root_sess.d_best_err1.ptr, root_sess.d_best_code1.ptr,
                                                         root_sess.d_best_a1.ptr, root_sess.d_best_r01.ptr,
                                                         root_sess.d_best_err2.ptr, root_sess.d_best_code2.ptr,
                                                         root_sess.d_best_a2.ptr, root_sess.d_best_r02.ptr,
                                                         root_sess.d_best_err3.ptr, root_sess.d_best_code3.ptr,
                                                         root_sess.d_best_a3.ptr, root_sess.d_best_r03.ptr);
                ThrowIf(cudaGetLastError(), "UpdateTop4FromScoresKernel");
            }
        }

        // Upload xC_small tile to device.
        {
            const float* xCptr = xC_small.data.data() + static_cast<std::size_t>(q0) * static_cast<std::size_t>(Hs);
            ThrowIf(cudaMemcpy(expand_sess.d_xC.ptr, xCptr,
                               sizeof(float) * static_cast<std::size_t>(Hs) * static_cast<std::size_t>(qlen),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy xC_small H2D (H4)");
        }

        // Init beam4 state.
        {
            const dim3 block(256, 1, 1);
            const dim3 grid((qlen + static_cast<int>(block.x) - 1) / static_cast<int>(block.x), 1, 1);
            InitBeam4FromRootTop4Kernel<<<grid, block>>>(
                m, qlen,
                root_sess.d_best_err0.ptr, root_sess.d_best_code0.ptr, root_sess.d_best_a0.ptr, root_sess.d_best_r00.ptr,
                root_sess.d_best_err1.ptr, root_sess.d_best_code1.ptr, root_sess.d_best_a1.ptr, root_sess.d_best_r01.ptr,
                root_sess.d_best_err2.ptr, root_sess.d_best_code2.ptr, root_sess.d_best_a2.ptr, root_sess.d_best_r02.ptr,
                root_sess.d_best_err3.ptr, root_sess.d_best_code3.ptr, root_sess.d_best_a3.ptr, root_sess.d_best_r03.ptr,
                expand_sess.d_root_curr.ptr,
                expand_sess.d_codes_curr.ptr,
                expand_sess.d_err_curr.ptr,
                expand_sess.d_a_curr.ptr,
                expand_sess.d_R_curr.ptr);
            ThrowIf(cudaGetLastError(), "InitBeam4FromRootTop4Kernel");
            if (timing) {
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(init_h4)");
            }
        }

        // Expand layers 1..m-1.
        Timer t_expand;
        const dim3 block(96, 1, 1);
        const dim3 grid((qlen + static_cast<int>(block.x) - 1) / static_cast<int>(block.x), 1, 1);
        for (int layer = 1; layer < m; ++layer) {
            ExpandLayerBeam4Kernel<<<grid, block>>>(
                m, Hs, h0, layer, qlen,
                expand_sess.d_Gs.ptr,
                expand_sess.d_G0S.ptr,
                expand_sess.d_small_offsets.ptr,
                expand_sess.d_hvec.ptr,
                expand_sess.d_xC.ptr,
                expand_sess.d_root_curr.ptr,
                expand_sess.d_codes_curr.ptr,
                expand_sess.d_err_curr.ptr,
                expand_sess.d_a_curr.ptr,
                expand_sess.d_R_curr.ptr,
                expand_sess.d_root_next.ptr,
                expand_sess.d_codes_next.ptr,
                expand_sess.d_err_next.ptr,
                expand_sess.d_a_next.ptr,
                expand_sess.d_R_next.ptr);
            ThrowIf(cudaGetLastError(), "ExpandLayerBeam4Kernel");
            if (timing) {
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(expand4_layer)");
            }
            swap_buf(expand_sess.d_root_curr, expand_sess.d_root_next);
            swap_buf(expand_sess.d_codes_curr, expand_sess.d_codes_next);
            swap_buf(expand_sess.d_err_curr, expand_sess.d_err_next);
            swap_buf(expand_sess.d_a_curr, expand_sess.d_a_next);
            swap_buf(expand_sess.d_R_curr, expand_sess.d_R_next);
        }
        if (timing) {
            ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(expand4_all)");
            timing->expand += t_expand.ElapsedSeconds();
        }

        // Finalize.
        {
            const dim3 block2(256, 1, 1);
            const dim3 grid2((qlen + static_cast<int>(block2.x) - 1) / static_cast<int>(block2.x), 1, 1);
            FinalizeBestBeam4Kernel<<<grid2, block2>>>(
                m, qlen,
                expand_sess.d_root_curr.ptr,
                expand_sess.d_codes_curr.ptr,
                expand_sess.d_err_curr.ptr,
                expand_sess.d_cluster_id.ptr,
                expand_sess.d_B_small.ptr);
            ThrowIf(cudaGetLastError(), "FinalizeBestBeam4Kernel");
            if (timing) {
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(finalize4)");
            }
        }

        ThrowIf(cudaMemcpy(cluster_id->data() + static_cast<std::size_t>(q0),
                           expand_sess.d_cluster_id.ptr,
                           sizeof(std::uint32_t) * static_cast<std::size_t>(qlen),
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy cluster_id D2H (H4)");
        if (B_small->rows > 0) {
            Code* dst = B_small->data.data() + static_cast<std::size_t>(q0) * static_cast<std::size_t>(B_small->rows);
            ThrowIf(cudaMemcpy(dst,
                               expand_sess.d_B_small.ptr,
                               sizeof(Code) * static_cast<std::size_t>(B_small->rows) * static_cast<std::size_t>(qlen),
                               cudaMemcpyDeviceToHost),
                    "cudaMemcpy B_small D2H (H4)");
        }
    }
}

static void BeamSearchPrefixLSLargeRootCudaH4DeviceXcCore(const ColMajorMatrix<float>& X,
                                               const PrecompLargeRoot& pre,
                                               const DeviceMatF32View& xC_small_dev,
                                               bool allow_tf32,
                                               std::vector<std::uint32_t>* cluster_id,
                                               ColMajorMatrix<Code>* B_small,
                                               std::vector<float>* best_err,
                                               BeamSearchLargeRootTiming* timing) {
    ThrowIf(cudaDeviceSynchronize(), "cudaDeviceSynchronize (H4DeviceXcCore entry)");

    if (!cluster_id || !B_small) return;
    const int n = X.cols;
    const int d = X.rows;
    const int m = pre.m;
    const int h0 = pre.h_vec.empty() ? 0 : pre.h_vec[0];
    const int Hs = pre.H_small;
    if (!pre.ready || n <= 0 || d <= 0 || m <= 0 || h0 <= 0 || Hs <= 0) {
        return;
    }
    if (!xC_small_dev.ptr || xC_small_dev.rows != Hs || xC_small_dev.cols != n || xC_small_dev.ld != Hs) {
        throw std::runtime_error("BeamSearchPrefixLSLargeRootCudaH4DeviceXc: invalid device xC_small view.");
    }

    cluster_id->resize(static_cast<std::size_t>(n));
    B_small->rows = std::max(0, m - 1);
    B_small->cols = n;
    B_small->data.resize(static_cast<std::size_t>(B_small->rows) * static_cast<std::size_t>(n));
    if (best_err) {
        best_err->resize(static_cast<std::size_t>(n));
    }

    const int q_block = std::min(8192, std::max(1, n));
    const int r_block = 1024;

    static thread_local CudaRootBeamSession root_sess;
    static thread_local CudaExpandH4Session expand_sess;
    root_sess.EnsureHandle(allow_tf32);
    root_sess.EnsurePre(pre);
    expand_sess.EnsurePre(pre);

    auto swap_buf = [](auto& a, auto& b) {
        std::swap(a.ptr, b.ptr);
        std::swap(a.n, b.n);
    };

    for (int q0 = 0; q0 < n; q0 += q_block) {
        const int qlen = std::min(q_block, n - q0);
        const int rlen = std::min(r_block, h0);
        root_sess.EnsureBatch(qlen, rlen);
        expand_sess.EnsureBatch(qlen);

        const float* Xptr = X.data.data() + static_cast<std::size_t>(q0) * static_cast<std::size_t>(d);
        ThrowIf(cudaMemcpy(root_sess.d_X.ptr,
                           Xptr,
                           sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(qlen),
                           cudaMemcpyHostToDevice),
                "cudaMemcpy X H2D (H4)");

        if (timing) {
            Timer t;
            Norm2ColsKernel<<<qlen, 256>>>(root_sess.d_X.ptr, d, d, qlen, root_sess.d_normx2.ptr);
            ThrowIf(cudaGetLastError(), "Norm2ColsKernel");
            ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(norm_x2)");
            timing->norm_x2 += t.ElapsedSeconds();
        } else {
            Norm2ColsKernel<<<qlen, 256>>>(root_sess.d_X.ptr, d, d, qlen, root_sess.d_normx2.ptr);
            ThrowIf(cudaGetLastError(), "Norm2ColsKernel");
        }

        // Init best arrays to inf/-1 on device.
        {
            const float inf = std::numeric_limits<float>::infinity();
            std::vector<float> init_err(qlen, inf);
            std::vector<int> init_code(qlen, -1);
            std::vector<float> init_a(qlen, 0.0f);
            std::vector<float> init_r(qlen, 0.0f);

            auto H2D_f = [&](DeviceBuf<float>& dst, const std::vector<float>& src, const char* what) {
                ThrowIf(cudaMemcpy(dst.ptr, src.data(), sizeof(float) * src.size(), cudaMemcpyHostToDevice), what);
            };
            auto H2D_i = [&](DeviceBuf<int>& dst, const std::vector<int>& src, const char* what) {
                ThrowIf(cudaMemcpy(dst.ptr, src.data(), sizeof(int) * src.size(), cudaMemcpyHostToDevice), what);
            };

            H2D_f(root_sess.d_best_err0, init_err, "Memcpy init best_err0 (H4)");
            H2D_i(root_sess.d_best_code0, init_code, "Memcpy init best_code0 (H4)");
            H2D_f(root_sess.d_best_a0, init_a, "Memcpy init best_a0 (H4)");
            H2D_f(root_sess.d_best_r00, init_r, "Memcpy init best_r00 (H4)");

            H2D_f(root_sess.d_best_err1, init_err, "Memcpy init best_err1 (H4)");
            H2D_i(root_sess.d_best_code1, init_code, "Memcpy init best_code1 (H4)");
            H2D_f(root_sess.d_best_a1, init_a, "Memcpy init best_a1 (H4)");
            H2D_f(root_sess.d_best_r01, init_r, "Memcpy init best_r01 (H4)");

            H2D_f(root_sess.d_best_err2, init_err, "Memcpy init best_err2 (H4)");
            H2D_i(root_sess.d_best_code2, init_code, "Memcpy init best_code2 (H4)");
            H2D_f(root_sess.d_best_a2, init_a, "Memcpy init best_a2 (H4)");
            H2D_f(root_sess.d_best_r02, init_r, "Memcpy init best_r02 (H4)");

            H2D_f(root_sess.d_best_err3, init_err, "Memcpy init best_err3 (H4)");
            H2D_i(root_sess.d_best_code3, init_code, "Memcpy init best_code3 (H4)");
            H2D_f(root_sess.d_best_a3, init_a, "Memcpy init best_a3 (H4)");
            H2D_f(root_sess.d_best_r03, init_r, "Memcpy init best_r03 (H4)");
        }

        // Root GEMM in chunks.
        const int rc = rlen;
        for (int r0 = 0; r0 < h0; r0 += rc) {
            const int rcur = std::min(rc, h0 - r0);
            if (rcur <= 0) break;

            const float alpha = 1.0f;
            const float beta = 0.0f;
            const int lda = d;
            const int ldb = d;
            const int ldc = rcur;
            const float* A = root_sess.d_C0.ptr + static_cast<std::size_t>(r0) * static_cast<std::size_t>(d);
            const float* B = root_sess.d_X.ptr;
            float* C = root_sess.d_scores.ptr;

            if (timing) {
                Timer t;
                ThrowIf(cublasSgemm(root_sess.handle,
                                    CUBLAS_OP_T,
                                    CUBLAS_OP_N,
                                    rcur,
                                    qlen,
                                    d,
                                    &alpha,
                                    A,
                                    lda,
                                    B,
                                    ldb,
                                    &beta,
                                    C,
                                    ldc),
                        "cublasSgemm(root H4)");
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(root_gemm H4)");
                timing->root_gemm += t.ElapsedSeconds();
            } else {
                ThrowIf(cublasSgemm(root_sess.handle,
                                    CUBLAS_OP_T,
                                    CUBLAS_OP_N,
                                    rcur,
                                    qlen,
                                    d,
                                    &alpha,
                                    A,
                                    lda,
                                    B,
                                    ldb,
                                    &beta,
                                    C,
                                    ldc),
                        "cublasSgemm(root H4)");
            }

            if (timing) {
                Timer t;
                UpdateTop4FromScoresKernel<<<qlen, 256>>>(root_sess.d_scores.ptr, rcur, rcur, r0,
                                                         root_sess.d_norm0.ptr,
                                                         root_sess.d_normx2.ptr, qlen,
                                                         root_sess.d_best_err0.ptr, root_sess.d_best_code0.ptr,
                                                         root_sess.d_best_a0.ptr, root_sess.d_best_r00.ptr,
                                                         root_sess.d_best_err1.ptr, root_sess.d_best_code1.ptr,
                                                         root_sess.d_best_a1.ptr, root_sess.d_best_r01.ptr,
                                                         root_sess.d_best_err2.ptr, root_sess.d_best_code2.ptr,
                                                         root_sess.d_best_a2.ptr, root_sess.d_best_r02.ptr,
                                                         root_sess.d_best_err3.ptr, root_sess.d_best_code3.ptr,
                                                         root_sess.d_best_a3.ptr, root_sess.d_best_r03.ptr);
                ThrowIf(cudaGetLastError(), "UpdateTop4FromScoresKernel");
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(root_update H4)");
                timing->root_update += t.ElapsedSeconds();
            } else {
                UpdateTop4FromScoresKernel<<<qlen, 256>>>(root_sess.d_scores.ptr, rcur, rcur, r0,
                                                         root_sess.d_norm0.ptr,
                                                         root_sess.d_normx2.ptr, qlen,
                                                         root_sess.d_best_err0.ptr, root_sess.d_best_code0.ptr,
                                                         root_sess.d_best_a0.ptr, root_sess.d_best_r00.ptr,
                                                         root_sess.d_best_err1.ptr, root_sess.d_best_code1.ptr,
                                                         root_sess.d_best_a1.ptr, root_sess.d_best_r01.ptr,
                                                         root_sess.d_best_err2.ptr, root_sess.d_best_code2.ptr,
                                                         root_sess.d_best_a2.ptr, root_sess.d_best_r02.ptr,
                                                         root_sess.d_best_err3.ptr, root_sess.d_best_code3.ptr,
                                                         root_sess.d_best_a3.ptr, root_sess.d_best_r03.ptr);
                ThrowIf(cudaGetLastError(), "UpdateTop4FromScoresKernel");
            }
        }

        // Init beam4 state.
        {
            const dim3 block(256, 1, 1);
            const dim3 grid((qlen + static_cast<int>(block.x) - 1) / static_cast<int>(block.x), 1, 1);
            InitBeam4FromRootTop4Kernel<<<grid, block>>>(
                m, qlen,
                root_sess.d_best_err0.ptr, root_sess.d_best_code0.ptr, root_sess.d_best_a0.ptr, root_sess.d_best_r00.ptr,
                root_sess.d_best_err1.ptr, root_sess.d_best_code1.ptr, root_sess.d_best_a1.ptr, root_sess.d_best_r01.ptr,
                root_sess.d_best_err2.ptr, root_sess.d_best_code2.ptr, root_sess.d_best_a2.ptr, root_sess.d_best_r02.ptr,
                root_sess.d_best_err3.ptr, root_sess.d_best_code3.ptr, root_sess.d_best_a3.ptr, root_sess.d_best_r03.ptr,
                expand_sess.d_root_curr.ptr,
                expand_sess.d_codes_curr.ptr,
                expand_sess.d_err_curr.ptr,
                expand_sess.d_a_curr.ptr,
                expand_sess.d_R_curr.ptr);
            ThrowIf(cudaGetLastError(), "InitBeam4FromRootTop4Kernel");
            if (timing) {
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(init_h4)");
            }
        }

        // Expand using device xC pointer.
        Timer t_expand;
        const dim3 block(96, 1, 1);
        const dim3 grid((qlen + static_cast<int>(block.x) - 1) / static_cast<int>(block.x), 1, 1);
        const float* xC_ptr = xC_small_dev.ptr + static_cast<std::size_t>(q0) * static_cast<std::size_t>(Hs);
        for (int layer = 1; layer < m; ++layer) {
            ExpandLayerBeam4Kernel<<<grid, block>>>(
                m, Hs, h0, layer, qlen,
                expand_sess.d_Gs.ptr,
                expand_sess.d_G0S.ptr,
                expand_sess.d_small_offsets.ptr,
                expand_sess.d_hvec.ptr,
                xC_ptr,
                expand_sess.d_root_curr.ptr,
                expand_sess.d_codes_curr.ptr,
                expand_sess.d_err_curr.ptr,
                expand_sess.d_a_curr.ptr,
                expand_sess.d_R_curr.ptr,
                expand_sess.d_root_next.ptr,
                expand_sess.d_codes_next.ptr,
                expand_sess.d_err_next.ptr,
                expand_sess.d_a_next.ptr,
                expand_sess.d_R_next.ptr);
            ThrowIf(cudaGetLastError(), "ExpandLayerBeam4Kernel");
            if (timing) {
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(expand4_layer)");
            }
            swap_buf(expand_sess.d_root_curr, expand_sess.d_root_next);
            swap_buf(expand_sess.d_codes_curr, expand_sess.d_codes_next);
            swap_buf(expand_sess.d_err_curr, expand_sess.d_err_next);
            swap_buf(expand_sess.d_a_curr, expand_sess.d_a_next);
            swap_buf(expand_sess.d_R_curr, expand_sess.d_R_next);
        }
        if (timing) {
            ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(expand4_all)");
            timing->expand += t_expand.ElapsedSeconds();
        }

        // Finalize.
        {
            const dim3 block2(256, 1, 1);
            const dim3 grid2((qlen + static_cast<int>(block2.x) - 1) / static_cast<int>(block2.x), 1, 1);
            FinalizeBestBeam4Kernel<<<grid2, block2>>>(
                m, qlen,
                expand_sess.d_root_curr.ptr,
                expand_sess.d_codes_curr.ptr,
                expand_sess.d_err_curr.ptr,
                expand_sess.d_cluster_id.ptr,
                expand_sess.d_B_small.ptr);
            ThrowIf(cudaGetLastError(), "FinalizeBestBeam4Kernel");
            if (timing) {
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(finalize4)");
            }
        }

        if (best_err) {
            std::vector<float> err_all(static_cast<std::size_t>(qlen) * 4);
            ThrowIf(cudaMemcpy(err_all.data(),
                               expand_sess.d_err_curr.ptr,
                               sizeof(float) * err_all.size(),
                               cudaMemcpyDeviceToHost),
                    "cudaMemcpy err_curr D2H (H4)");
            for (int j = 0; j < qlen; ++j) {
                float be = err_all[static_cast<std::size_t>(j) * 4 + 0];
                // tie => earliest
                for (int b = 1; b < 4; ++b) {
                    const float e = err_all[static_cast<std::size_t>(j) * 4 + static_cast<std::size_t>(b)];
                    if (e < be) be = e;
                }
                // 最终防御：best_err 不允许为负数（避免 root 阶段出现数值/未初始化导致的负值传播）
                if (be < 0.0f) be = 0.0f;
                (*best_err)[static_cast<std::size_t>(q0 + j)] = be;
            }
        }

        ThrowIf(cudaMemcpy(cluster_id->data() + static_cast<std::size_t>(q0),
                           expand_sess.d_cluster_id.ptr,
                           sizeof(std::uint32_t) * static_cast<std::size_t>(qlen),
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy cluster_id D2H (H4)");
        if (B_small->rows > 0) {
            Code* dst = B_small->data.data() + static_cast<std::size_t>(q0) * static_cast<std::size_t>(B_small->rows);
            ThrowIf(cudaMemcpy(dst,
                               expand_sess.d_B_small.ptr,
                               sizeof(Code) * static_cast<std::size_t>(B_small->rows) * static_cast<std::size_t>(qlen),
                               cudaMemcpyDeviceToHost),
                    "cudaMemcpy B_small D2H (H4)");
        }
    }
}

void BeamSearchPrefixLSLargeRootCudaH4DeviceXc(const ColMajorMatrix<float>& X,
                                               const PrecompLargeRoot& pre,
                                               const DeviceMatF32View& xC_small_dev,
                                               bool allow_tf32,
                                               std::vector<std::uint32_t>* cluster_id,
                                               ColMajorMatrix<Code>* B_small,
                                               BeamSearchLargeRootTiming* timing) {
    BeamSearchPrefixLSLargeRootCudaH4DeviceXcCore(X, pre, xC_small_dev, allow_tf32, cluster_id, B_small,
                                                 /*best_err=*/nullptr, timing);
}

void BeamSearchPrefixLSLargeRootCudaH4DeviceXcWithBestErr(const ColMajorMatrix<float>& X,
                                                          const PrecompLargeRoot& pre,
                                                          const DeviceMatF32View& xC_small_dev,
                                                          bool allow_tf32,
                                                          std::vector<std::uint32_t>* cluster_id,
                                                          ColMajorMatrix<Code>* B_small,
                                                          std::vector<float>* best_err,
                                                          BeamSearchLargeRootTiming* timing) {
    BeamSearchPrefixLSLargeRootCudaH4DeviceXcCore(X, pre, xC_small_dev, allow_tf32, cluster_id, B_small,
                                                 best_err, timing);
}

static void BeamSearchPrefixLSLargeRootCudaH4DeviceXcDeviceXCore(const DeviceMatF32View& X_dev,
                                                                 const PrecompLargeRoot& pre,
                                                                 const DeviceMatF32View& xC_small_dev,
                                                                 bool allow_tf32,
                                                                 std::vector<std::uint32_t>* cluster_id,
                                                                 ColMajorMatrix<Code>* B_small,
                                                                 std::vector<float>* best_err,
                                                                 BeamSearchLargeRootTiming* timing) {
    ThrowIf(cudaDeviceSynchronize(), "cudaDeviceSynchronize (H4DeviceXcDeviceXCore entry)");

    if (!cluster_id || !B_small) return;
    const int n = X_dev.cols;
    const int d = X_dev.rows;
    const int m = pre.m;
    const int h0 = pre.h_vec.empty() ? 0 : pre.h_vec[0];
    const int Hs = pre.H_small;
    if (!pre.ready || n <= 0 || d <= 0 || m <= 0 || h0 <= 0 || Hs <= 0) return;
    if (!X_dev.ptr || X_dev.ld < d) {
        throw std::runtime_error("BeamSearchPrefixLSLargeRootCudaH4DeviceXcDeviceXCore: invalid X_dev view.");
    }
    if (!xC_small_dev.ptr || xC_small_dev.rows != Hs || xC_small_dev.cols != n || xC_small_dev.ld != Hs) {
        throw std::runtime_error("BeamSearchPrefixLSLargeRootCudaH4DeviceXcDeviceXCore: invalid device xC_small view.");
    }

    cluster_id->resize(static_cast<std::size_t>(n));
    B_small->rows = std::max(0, m - 1);
    B_small->cols = n;
    B_small->data.resize(static_cast<std::size_t>(B_small->rows) * static_cast<std::size_t>(n));
    if (best_err) best_err->resize(static_cast<std::size_t>(n));

    const int q_block = std::min(8192, std::max(1, n));
    const int r_block = 1024;

    static thread_local CudaRootBeamSession root_sess;
    static thread_local CudaExpandH4Session expand_sess;
    root_sess.EnsureHandle(allow_tf32);
    root_sess.EnsurePre(pre);
    expand_sess.EnsurePre(pre);

    auto swap_buf = [](auto& a, auto& b) {
        std::swap(a.ptr, b.ptr);
        std::swap(a.n, b.n);
    };

    for (int q0 = 0; q0 < n; q0 += q_block) {
        const int qlen = std::min(q_block, n - q0);
        const int rlen = std::min(r_block, h0);
        root_sess.EnsureBatch(qlen, rlen);
        expand_sess.EnsureBatch(qlen);

        const float* Xptr = X_dev.ptr + static_cast<std::size_t>(q0) * static_cast<std::size_t>(X_dev.ld);

        if (timing) {
            Timer t;
            Norm2ColsKernel<<<qlen, 256>>>(Xptr, X_dev.ld, d, qlen, root_sess.d_normx2.ptr);
            ThrowIf(cudaGetLastError(), "Norm2ColsKernel");
            ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(norm_x2)");
            timing->norm_x2 += t.ElapsedSeconds();
        } else {
            Norm2ColsKernel<<<qlen, 256>>>(Xptr, X_dev.ld, d, qlen, root_sess.d_normx2.ptr);
            ThrowIf(cudaGetLastError(), "Norm2ColsKernel");
        }

        // Init best arrays to inf/-1 on device.
        {
            const float inf = std::numeric_limits<float>::infinity();
            std::vector<float> init_err(qlen, inf);
            std::vector<int> init_code(qlen, -1);
            std::vector<float> init_a(qlen, 0.0f);
            std::vector<float> init_r(qlen, 0.0f);

            auto H2D_f = [&](DeviceBuf<float>& dst, const std::vector<float>& src, const char* what) {
                ThrowIf(cudaMemcpy(dst.ptr, src.data(), sizeof(float) * src.size(), cudaMemcpyHostToDevice), what);
            };
            auto H2D_i = [&](DeviceBuf<int>& dst, const std::vector<int>& src, const char* what) {
                ThrowIf(cudaMemcpy(dst.ptr, src.data(), sizeof(int) * src.size(), cudaMemcpyHostToDevice), what);
            };

            H2D_f(root_sess.d_best_err0, init_err, "Memcpy init best_err0 (H4 devX)");
            H2D_i(root_sess.d_best_code0, init_code, "Memcpy init best_code0 (H4 devX)");
            H2D_f(root_sess.d_best_a0, init_a, "Memcpy init best_a0 (H4 devX)");
            H2D_f(root_sess.d_best_r00, init_r, "Memcpy init best_r00 (H4 devX)");

            H2D_f(root_sess.d_best_err1, init_err, "Memcpy init best_err1 (H4 devX)");
            H2D_i(root_sess.d_best_code1, init_code, "Memcpy init best_code1 (H4 devX)");
            H2D_f(root_sess.d_best_a1, init_a, "Memcpy init best_a1 (H4 devX)");
            H2D_f(root_sess.d_best_r01, init_r, "Memcpy init best_r01 (H4 devX)");

            H2D_f(root_sess.d_best_err2, init_err, "Memcpy init best_err2 (H4 devX)");
            H2D_i(root_sess.d_best_code2, init_code, "Memcpy init best_code2 (H4 devX)");
            H2D_f(root_sess.d_best_a2, init_a, "Memcpy init best_a2 (H4 devX)");
            H2D_f(root_sess.d_best_r02, init_r, "Memcpy init best_r02 (H4 devX)");

            H2D_f(root_sess.d_best_err3, init_err, "Memcpy init best_err3 (H4 devX)");
            H2D_i(root_sess.d_best_code3, init_code, "Memcpy init best_code3 (H4 devX)");
            H2D_f(root_sess.d_best_a3, init_a, "Memcpy init best_a3 (H4 devX)");
            H2D_f(root_sess.d_best_r03, init_r, "Memcpy init best_r03 (H4 devX)");
        }

        // Root GEMM in chunks of r_block.
        const int rc = rlen;
        for (int r0 = 0; r0 < h0; r0 += rc) {
            const int rcur = std::min(rc, h0 - r0);
            if (rcur <= 0) break;

            const float alpha = 1.0f;
            const float beta = 0.0f;
            const int lda = d;
            const int ldb = X_dev.ld;
            const int ldc = rcur;
            const float* A = root_sess.d_C0.ptr + static_cast<std::size_t>(r0) * static_cast<std::size_t>(d);
            const float* B = Xptr;
            float* C = root_sess.d_scores.ptr;

            if (timing) {
                Timer t;
                ThrowIf(cublasSgemm(root_sess.handle,
                                    CUBLAS_OP_T,
                                    CUBLAS_OP_N,
                                    rcur,
                                    qlen,
                                    d,
                                    &alpha,
                                    A,
                                    lda,
                                    B,
                                    ldb,
                                    &beta,
                                    C,
                                    ldc),
                        "cublasSgemm(root H4)");
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(root_gemm H4)");
                timing->root_gemm += t.ElapsedSeconds();
            } else {
                ThrowIf(cublasSgemm(root_sess.handle,
                                    CUBLAS_OP_T,
                                    CUBLAS_OP_N,
                                    rcur,
                                    qlen,
                                    d,
                                    &alpha,
                                    A,
                                    lda,
                                    B,
                                    ldb,
                                    &beta,
                                    C,
                                    ldc),
                        "cublasSgemm(root H4)");
            }

            if (timing) {
                Timer t;
                UpdateTop4FromScoresKernel<<<qlen, 256>>>(root_sess.d_scores.ptr, rcur, rcur, r0,
                                                         root_sess.d_norm0.ptr,
                                                         root_sess.d_normx2.ptr, qlen,
                                                         root_sess.d_best_err0.ptr, root_sess.d_best_code0.ptr,
                                                         root_sess.d_best_a0.ptr, root_sess.d_best_r00.ptr,
                                                         root_sess.d_best_err1.ptr, root_sess.d_best_code1.ptr,
                                                         root_sess.d_best_a1.ptr, root_sess.d_best_r01.ptr,
                                                         root_sess.d_best_err2.ptr, root_sess.d_best_code2.ptr,
                                                         root_sess.d_best_a2.ptr, root_sess.d_best_r02.ptr,
                                                         root_sess.d_best_err3.ptr, root_sess.d_best_code3.ptr,
                                                         root_sess.d_best_a3.ptr, root_sess.d_best_r03.ptr);
                ThrowIf(cudaGetLastError(), "UpdateTop4FromScoresKernel");
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(root_update H4)");
                timing->root_update += t.ElapsedSeconds();
            } else {
                UpdateTop4FromScoresKernel<<<qlen, 256>>>(root_sess.d_scores.ptr, rcur, rcur, r0,
                                                         root_sess.d_norm0.ptr,
                                                         root_sess.d_normx2.ptr, qlen,
                                                         root_sess.d_best_err0.ptr, root_sess.d_best_code0.ptr,
                                                         root_sess.d_best_a0.ptr, root_sess.d_best_r00.ptr,
                                                         root_sess.d_best_err1.ptr, root_sess.d_best_code1.ptr,
                                                         root_sess.d_best_a1.ptr, root_sess.d_best_r01.ptr,
                                                         root_sess.d_best_err2.ptr, root_sess.d_best_code2.ptr,
                                                         root_sess.d_best_a2.ptr, root_sess.d_best_r02.ptr,
                                                         root_sess.d_best_err3.ptr, root_sess.d_best_code3.ptr,
                                                         root_sess.d_best_a3.ptr, root_sess.d_best_r03.ptr);
                ThrowIf(cudaGetLastError(), "UpdateTop4FromScoresKernel");
            }
        }

        // Init beam4 state from root top4.
        {
            const dim3 block(256, 1, 1);
            const dim3 grid((qlen + static_cast<int>(block.x) - 1) / static_cast<int>(block.x), 1, 1);
            InitBeam4FromRootTop4Kernel<<<grid, block>>>(
                m, qlen,
                root_sess.d_best_err0.ptr, root_sess.d_best_code0.ptr, root_sess.d_best_a0.ptr, root_sess.d_best_r00.ptr,
                root_sess.d_best_err1.ptr, root_sess.d_best_code1.ptr, root_sess.d_best_a1.ptr, root_sess.d_best_r01.ptr,
                root_sess.d_best_err2.ptr, root_sess.d_best_code2.ptr, root_sess.d_best_a2.ptr, root_sess.d_best_r02.ptr,
                root_sess.d_best_err3.ptr, root_sess.d_best_code3.ptr, root_sess.d_best_a3.ptr, root_sess.d_best_r03.ptr,
                expand_sess.d_root_curr.ptr,
                expand_sess.d_codes_curr.ptr,
                expand_sess.d_err_curr.ptr,
                expand_sess.d_a_curr.ptr,
                expand_sess.d_R_curr.ptr);
            ThrowIf(cudaGetLastError(), "InitBeam4FromRootTop4Kernel");
            if (timing) {
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(init_h4)");
            }
        }

        // Expand layers 1..m-1 using the device xC pointer for the current q0 block.
        Timer t_expand;
        const dim3 block(128, 1, 1);
        const dim3 grid((qlen + static_cast<int>(block.x) - 1) / static_cast<int>(block.x), 1, 1);
        const float* xC_ptr = xC_small_dev.ptr + static_cast<std::size_t>(q0) * static_cast<std::size_t>(Hs);
        for (int layer = 1; layer < m; ++layer) {
            ExpandLayerBeam4Kernel<<<grid, block>>>(
                m, Hs, h0, layer, qlen,
                expand_sess.d_Gs.ptr,
                expand_sess.d_G0S.ptr,
                expand_sess.d_small_offsets.ptr,
                expand_sess.d_hvec.ptr,
                xC_ptr,
                expand_sess.d_root_curr.ptr,
                expand_sess.d_codes_curr.ptr,
                expand_sess.d_err_curr.ptr,
                expand_sess.d_a_curr.ptr,
                expand_sess.d_R_curr.ptr,
                expand_sess.d_root_next.ptr,
                expand_sess.d_codes_next.ptr,
                expand_sess.d_err_next.ptr,
                expand_sess.d_a_next.ptr,
                expand_sess.d_R_next.ptr);
            ThrowIf(cudaGetLastError(), "ExpandLayerBeam4Kernel");
            if (timing) {
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(expand4_layer)");
            }
            swap_buf(expand_sess.d_root_curr, expand_sess.d_root_next);
            swap_buf(expand_sess.d_codes_curr, expand_sess.d_codes_next);
            swap_buf(expand_sess.d_err_curr, expand_sess.d_err_next);
            swap_buf(expand_sess.d_a_curr, expand_sess.d_a_next);
            swap_buf(expand_sess.d_R_curr, expand_sess.d_R_next);
        }
        if (timing) {
            ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(expand4_all)");
            timing->expand += t_expand.ElapsedSeconds();
        }

        // Finalize.
        {
            const dim3 block2(256, 1, 1);
            const dim3 grid2((qlen + static_cast<int>(block2.x) - 1) / static_cast<int>(block2.x), 1, 1);
            FinalizeBestBeam4Kernel<<<grid2, block2>>>(
                m, qlen,
                expand_sess.d_root_curr.ptr,
                expand_sess.d_codes_curr.ptr,
                expand_sess.d_err_curr.ptr,
                expand_sess.d_cluster_id.ptr,
                expand_sess.d_B_small.ptr);
            ThrowIf(cudaGetLastError(), "FinalizeBestBeam4Kernel");
            if (timing) {
                ThrowIf(cudaStreamSynchronize(nullptr), "cudaStreamSync(finalize4)");
            }
        }

        if (best_err) {
            std::vector<float> err_all(static_cast<std::size_t>(qlen) * 4);
            ThrowIf(cudaMemcpy(err_all.data(),
                               expand_sess.d_err_curr.ptr,
                               sizeof(float) * err_all.size(),
                               cudaMemcpyDeviceToHost),
                    "cudaMemcpy err_curr D2H (H4)");
            for (int j = 0; j < qlen; ++j) {
                float be = err_all[static_cast<std::size_t>(j) * 4 + 0];
                for (int b = 1; b < 4; ++b) {
                    const float e = err_all[static_cast<std::size_t>(j) * 4 + static_cast<std::size_t>(b)];
                    if (e < be) be = e;
                }
                (*best_err)[static_cast<std::size_t>(q0 + j)] = be;
            }
        }

        ThrowIf(cudaMemcpy(cluster_id->data() + static_cast<std::size_t>(q0),
                           expand_sess.d_cluster_id.ptr,
                           sizeof(std::uint32_t) * static_cast<std::size_t>(qlen),
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy cluster_id D2H (H4)");
        if (B_small->rows > 0) {
            Code* dst = B_small->data.data() + static_cast<std::size_t>(q0) * static_cast<std::size_t>(B_small->rows);
            ThrowIf(cudaMemcpy(dst,
                               expand_sess.d_B_small.ptr,
                               sizeof(Code) * static_cast<std::size_t>(B_small->rows) * static_cast<std::size_t>(qlen),
                               cudaMemcpyDeviceToHost),
                    "cudaMemcpy B_small D2H (H4)");
        }
    }
}

void BeamSearchPrefixLSLargeRootCudaH4DeviceXcDeviceX(const DeviceMatF32View& X_dev,
                                                      const PrecompLargeRoot& pre,
                                                      const DeviceMatF32View& xC_small_dev,
                                                      bool allow_tf32,
                                                      std::vector<std::uint32_t>* cluster_id,
                                                      ColMajorMatrix<Code>* B_small,
                                                      BeamSearchLargeRootTiming* timing) {
    BeamSearchPrefixLSLargeRootCudaH4DeviceXcDeviceXCore(X_dev, pre, xC_small_dev, allow_tf32, cluster_id, B_small,
                                                        /*best_err=*/nullptr, timing);
}

void BeamSearchPrefixLSLargeRootCudaH4DeviceXcDeviceXWithBestErr(const DeviceMatF32View& X_dev,
                                                                 const PrecompLargeRoot& pre,
                                                                 const DeviceMatF32View& xC_small_dev,
                                                                 bool allow_tf32,
                                                                 std::vector<std::uint32_t>* cluster_id,
                                                                 ColMajorMatrix<Code>* B_small,
                                                                 std::vector<float>* best_err,
                                                                 BeamSearchLargeRootTiming* timing) {
    BeamSearchPrefixLSLargeRootCudaH4DeviceXcDeviceXCore(X_dev, pre, xC_small_dev, allow_tf32, cluster_id, B_small,
                                                        best_err, timing);
}

}  // namespace stlq
