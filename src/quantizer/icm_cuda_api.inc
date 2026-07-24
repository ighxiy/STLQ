void DynamicIcmWithIlsNoAbsNoNormalCuda(const ColMajorMatrix<float>& X,
                                       const Precomp& precomp,
                                       const ColMajorMatrix<float>& xC,
                                       int icm_iters,
                                       int ils_iters,
                                       int perturb_k,
                                       std::uint32_t seed,
                                       ColMajorMatrix<FullCode>* B,
                                       ColMajorMatrix<float>* a,
                                       std::vector<float>* X_norm2,
                                       std::vector<float>* cost,
                                       bool print_progress,
                                       std::uint64_t sample_id_offset,
                                       CudaIcmTiming* timing) {
    try {
        DynamicIcmWithIlsCudaImpl<false>(X, precomp, xC, icm_iters, ils_iters, perturb_k,
                                         seed, sample_id_offset, B, a, X_norm2, cost, print_progress, timing);
    } catch (const std::exception& e) {
        LogError(std::string("DynamicIcmWithIlsNoAbsNoNormalCuda failed: ") + e.what() + " (fallback to CPU)");
        if (!B || !a || !X_norm2) return;
        DynamicIcmWithIlsNoAbsNoNormal(X, precomp, xC, icm_iters, ils_iters, perturb_k,
                                       seed, B, a, X_norm2, cost, print_progress, sample_id_offset);
    }
}

void DynamicIcmWithIlsAbsNoNormalCuda(const ColMajorMatrix<float>& X,
                                     const Precomp& precomp,
                                     const ColMajorMatrix<float>& xC,
                                     int icm_iters,
                                     int ils_iters,
                                     int perturb_k,
                                     std::uint32_t seed,
                                     ColMajorMatrix<FullCode>* B,
                                     ColMajorMatrix<float>* a,
                                     std::vector<float>* X_norm2,
                                     std::vector<float>* cost,
                                     bool print_progress,
                                     std::uint64_t sample_id_offset,
                                     CudaIcmTiming* timing) {
    try {
        DynamicIcmWithIlsCudaImpl<true>(X, precomp, xC, icm_iters, ils_iters, perturb_k,
                                        seed, sample_id_offset, B, a, X_norm2, cost, print_progress, timing);
    } catch (const std::exception& e) {
        LogError(std::string("DynamicIcmWithIlsAbsNoNormalCuda failed: ") + e.what() + " (fallback to CPU)");
        if (!B || !a || !X_norm2) return;
        DynamicIcmWithIlsAbsNoNormal(X, precomp, xC, icm_iters, ils_iters, perturb_k,
                                     seed, B, a, X_norm2, cost, print_progress, sample_id_offset);
    }
}

void DynamicIcmWithIlsNoAbsNoNormalCudaDeviceXc(const ColMajorMatrix<float>& X,
                                               const Precomp& precomp,
                                               const DeviceMatF32View& xC_dev,
                                               int icm_iters,
                                               int ils_iters,
                                               int perturb_k,
                                               std::uint32_t seed,
                                               ColMajorMatrix<FullCode>* B,
                                               ColMajorMatrix<float>* a,
                                               std::vector<float>* X_norm2,
                                               std::vector<float>* cost,
                                               bool print_progress,
                                               std::uint64_t sample_id_offset,
                                               CudaIcmTiming* timing) {
    try {
        DynamicIcmWithIlsCudaImplDeviceXc<false>(X, precomp, xC_dev, icm_iters, ils_iters, perturb_k,
                                                seed, sample_id_offset, B, a, X_norm2, cost, print_progress, timing);
    } catch (const std::exception& e) {
        LogError(std::string("DynamicIcmWithIlsNoAbsNoNormalCudaDeviceXc failed: ") + e.what() +
                 " (fallback to CPU via D2H xC)");
        if (!B || !a || !X_norm2) return;
        if (!xC_dev.ptr) {
            LogError("DynamicIcmWithIlsNoAbsNoNormalCudaDeviceXc: xC_dev is null; cannot fallback.");
            return;
        }
        ColMajorMatrix<float> xC;
        xC.rows = precomp.H;
        xC.cols = B->cols;
        xC.data.resize(static_cast<std::size_t>(xC.rows) * static_cast<std::size_t>(xC.cols));
        const cudaError_t st = cudaMemcpy(xC.data.data(), xC_dev.ptr,
                                          sizeof(float) * xC.data.size(),
                                          cudaMemcpyDeviceToHost);
        if (st != cudaSuccess) {
            LogError(std::string("Memcpy xC_dev D2H (fallback) failed: ") + cudaGetErrorString(st));
            return;
        }
        DynamicIcmWithIlsNoAbsNoNormal(X, precomp, xC, icm_iters, ils_iters, perturb_k,
                                       seed, B, a, X_norm2, cost, print_progress, sample_id_offset);
    }
}

void DynamicIcmWithIlsAbsNoNormalCudaDeviceXc(const ColMajorMatrix<float>& X,
                                             const Precomp& precomp,
                                             const DeviceMatF32View& xC_dev,
                                             int icm_iters,
                                             int ils_iters,
                                             int perturb_k,
                                             std::uint32_t seed,
                                             ColMajorMatrix<FullCode>* B,
                                             ColMajorMatrix<float>* a,
                                             std::vector<float>* X_norm2,
                                             std::vector<float>* cost,
                                             bool print_progress,
                                             std::uint64_t sample_id_offset,
                                             CudaIcmTiming* timing) {
    try {
        DynamicIcmWithIlsCudaImplDeviceXc<true>(X, precomp, xC_dev, icm_iters, ils_iters, perturb_k,
                                               seed, sample_id_offset, B, a, X_norm2, cost, print_progress, timing);
    } catch (const std::exception& e) {
        LogError(std::string("DynamicIcmWithIlsAbsNoNormalCudaDeviceXc failed: ") + e.what() +
                 " (fallback to CPU via D2H xC)");
        if (!B || !a || !X_norm2) return;
        if (!xC_dev.ptr) {
            LogError("DynamicIcmWithIlsAbsNoNormalCudaDeviceXc: xC_dev is null; cannot fallback.");
            return;
        }
        ColMajorMatrix<float> xC;
        xC.rows = precomp.H;
        xC.cols = B->cols;
        xC.data.resize(static_cast<std::size_t>(xC.rows) * static_cast<std::size_t>(xC.cols));
        const cudaError_t st = cudaMemcpy(xC.data.data(), xC_dev.ptr,
                                          sizeof(float) * xC.data.size(),
                                          cudaMemcpyDeviceToHost);
        if (st != cudaSuccess) {
            LogError(std::string("Memcpy xC_dev D2H (fallback) failed: ") + cudaGetErrorString(st));
            return;
        }
        DynamicIcmWithIlsAbsNoNormal(X, precomp, xC, icm_iters, ils_iters, perturb_k,
                                     seed, B, a, X_norm2, cost, print_progress, sample_id_offset);
    }
}

void DynamicIcmWithIlsNoAbsNoNormalLargeRootCuda(const ColMajorMatrix<float>& X,
                                                const PrecompLargeRoot& pre,
                                                const ColMajorMatrix<float>& xC_small,
                                                int icm_iters,
                                                int ils_iters,
                                                int perturb_k,
                                                std::uint32_t seed,
                                                const std::vector<std::uint32_t>& cluster_id,
                                                ColMajorMatrix<Code>* B_small,
                                                ColMajorMatrix<float>* a,
                                                std::vector<float>* X_norm2,
                                                std::vector<float>* cost,
                                                bool print_progress,
                                                std::uint64_t sample_id_offset,
                                                CudaIcmTiming* timing) {
    try {
        DynamicIcmWithIlsLargeRootCudaImpl<false>(X, pre, xC_small, icm_iters, ils_iters, perturb_k,
                                                 seed, sample_id_offset, cluster_id, B_small, a, X_norm2, cost,
                                                 print_progress, timing);
    } catch (const std::exception& e) {
        LogError(std::string("DynamicIcmWithIlsNoAbsNoNormalLargeRootCuda failed: ") + e.what() + " (fallback to CPU)");
        if (!B_small || !a || !X_norm2) return;
        DynamicIcmWithIlsNoAbsNoNormalLargeRoot(X, pre, xC_small, icm_iters, ils_iters, perturb_k,
                                                seed, cluster_id, B_small, a, X_norm2, cost,
                                                print_progress, sample_id_offset);
    }
}

void DynamicIcmWithIlsAbsNoNormalLargeRootCuda(const ColMajorMatrix<float>& X,
                                              const PrecompLargeRoot& pre,
                                              const ColMajorMatrix<float>& xC_small,
                                              int icm_iters,
                                              int ils_iters,
                                              int perturb_k,
                                              std::uint32_t seed,
                                              const std::vector<std::uint32_t>& cluster_id,
                                              ColMajorMatrix<Code>* B_small,
                                              ColMajorMatrix<float>* a,
                                              std::vector<float>* X_norm2,
                                              std::vector<float>* cost,
                                              bool print_progress,
                                              std::uint64_t sample_id_offset,
                                              CudaIcmTiming* timing) {
    try {
        DynamicIcmWithIlsLargeRootCudaImpl<true>(X, pre, xC_small, icm_iters, ils_iters, perturb_k,
                                                seed, sample_id_offset, cluster_id, B_small, a, X_norm2, cost,
                                                print_progress, timing);
    } catch (const std::exception& e) {
        LogError(std::string("DynamicIcmWithIlsAbsNoNormalLargeRootCuda failed: ") + e.what() + " (fallback to CPU)");
        if (!B_small || !a || !X_norm2) return;
        DynamicIcmWithIlsAbsNoNormalLargeRoot(X, pre, xC_small, icm_iters, ils_iters, perturb_k,
                                              seed, cluster_id, B_small, a, X_norm2, cost,
                                              print_progress, sample_id_offset);
    }
}

template <bool UseAbs>
void DynamicIcmWithIlsLargeRootCudaImplDeviceXc(const ColMajorMatrix<float>& X,
                                               const PrecompLargeRoot& pre,
                                               const DeviceMatF32View& xC_small_dev,
                                               int icm_iters,
                                               int ils_iters,
                                               int perturb_k,
                                               std::uint32_t seed,
                                               std::uint64_t sample_id_offset,
                                               const std::vector<std::uint32_t>& cluster_id,
                                               ColMajorMatrix<Code>* B_small,
                                               ColMajorMatrix<float>* a,
                                               std::vector<float>* X_norm2,
                                               std::vector<float>* cost,
                                               bool print_progress,
                                               CudaIcmTiming* timing) {
    if (!B_small || !a || !X_norm2) {
        return;
    }
    static thread_local std::vector<float> cost_scratch;
    std::vector<float>* cost_out = cost ? cost : &cost_scratch;
    const int n = B_small->cols;
    if (n <= 0) {
        return;
    }
    if (print_progress) {
        throw std::runtime_error("DynamicIcmWithIlsLargeRootCudaImplDeviceXc: print_progress is not supported.");
    }
    if (!xC_small_dev.ptr || xC_small_dev.rows != pre.H_small || xC_small_dev.cols != n ||
        xC_small_dev.ld != pre.H_small) {
        throw std::runtime_error("DynamicIcmWithIlsLargeRootCudaImplDeviceXc: invalid xC_small_dev view.");
    }
    if (pre.m <= 1 || pre.m > kMaxM) {
        throw std::runtime_error("DynamicIcmWithIlsLargeRootCudaImplDeviceXc: unsupported m for CUDA device-xC path.");
    }
    if (static_cast<int>(X_norm2->size()) != n) {
        ComputeXNorm2(X, X_norm2);
    }

    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));

    auto DotScalar = [](const float* a, const float* b, int d) -> float {
        float s = 0.0f;
        for (int i = 0; i < d; ++i) {
            s += a[i] * b[i];
        }
        return s;
    };

    static thread_local LargeRootIcmCudaSession session;
    session.EnsurePrecomp(pre);
    session.EnsureBatch(n, /*need_xc_buffer=*/false);
    static thread_local std::vector<float> xC0_host;
    xC0_host.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const std::uint32_t cid = cluster_id[static_cast<std::size_t>(i)];
        if (cid >= static_cast<std::uint32_t>(pre.h_vec.empty() ? 0 : pre.h_vec[0])) {
            xC0_host[static_cast<std::size_t>(i)] = 0.0f;
            continue;
        }
        const float* x = X.data.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(pre.d);
        const float* c0 = pre.C0.data.data() + static_cast<std::size_t>(cid) * static_cast<std::size_t>(pre.d);
        xC0_host[static_cast<std::size_t>(i)] = DotScalar(c0, x, pre.d);
    }

    if (timing) {
        Timer tt;
        session.UploadFixedDeviceXc(xC0_host, xC_small_dev, *X_norm2, cluster_id);
        timing->upload_fixed += tt.ElapsedSeconds();
    } else {
        session.UploadFixedDeviceXc(xC0_host, xC_small_dev, *X_norm2, cluster_id);
    }

    a->rows = pre.m;
    a->cols = n;
    a->data.resize(static_cast<std::size_t>(pre.m) * static_cast<std::size_t>(n));
    cost_out->resize(static_cast<std::size_t>(n));

    if (timing) {
        Timer tt;
        session.UploadB(B_small->data.data());
        timing->upload_B += tt.ElapsedSeconds();
        timing->icm_init += session.ev.MeasureSeconds([&] {
            session.RunIcmDevice(/*use_abs=*/UseAbs,
                                 icm_iters,
                                 session.d_B.ptr,
                                 session.d_a.ptr,
                                 session.d_cost.ptr,
                                 &rng);
        });
    } else {
        session.UploadB(B_small->data.data());
        session.RunIcmDevice(/*use_abs=*/UseAbs,
                             icm_iters,
                             session.d_B.ptr,
                             session.d_a.ptr,
                             session.d_cost.ptr,
                             &rng);
    }

    if (ils_iters <= 0 || perturb_k <= 0 || pre.m <= 1) {
        if (timing) {
            Timer tt;
            if (cost) {
                session.DownloadAll(B_small->data.data(), a->data.data(), cost_out->data());
            } else {
                CudaCheck(cudaMemcpy(B_small->data.data(), session.d_B.ptr,
                                     sizeof(Code) * static_cast<std::size_t>(pre.m - 1) * static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "Memcpy B_small back (no cost)");
                CudaCheck(cudaMemcpy(a->data.data(), session.d_a.ptr,
                                     sizeof(float) * static_cast<std::size_t>(pre.m) * static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "Memcpy a back (no cost)");
            }
            timing->download += tt.ElapsedSeconds();
        } else {
            if (cost) {
                session.DownloadAll(B_small->data.data(), a->data.data(), cost_out->data());
            } else {
                CudaCheck(cudaMemcpy(B_small->data.data(), session.d_B.ptr,
                                     sizeof(Code) * static_cast<std::size_t>(pre.m - 1) * static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "Memcpy B_small back (no cost)");
                CudaCheck(cudaMemcpy(a->data.data(), session.d_a.ptr,
                                     sizeof(float) * static_cast<std::size_t>(pre.m) * static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "Memcpy a back (no cost)");
            }
        }
        return;
    }

    const int ksel = std::min(perturb_k, pre.m);
    const int threads = 128;
    const int blocks = (n + threads - 1) / threads;

    for (int outer = 0; outer < ils_iters; ++outer) {
        if (timing) {
            timing->ils_copy_perturb += session.ev.MeasureSeconds([&] {
                if (pre.m == 5) {
                    CopyAndPerturbSmallCodesFixed<5><<<blocks, threads>>>(session.d_hvec.ptr,
                                                                         seed,
                                                                         outer + 1,
                                                                         sample_id_offset,
                                                                         ksel,
                                                                         session.d_B.ptr,
                                                                         session.d_B_cand.ptr,
                                                                         n);
                } else if (pre.m == 10) {
                    CopyAndPerturbSmallCodesFixed<10><<<blocks, threads>>>(session.d_hvec.ptr,
                                                                          seed,
                                                                          outer + 1,
                                                                          sample_id_offset,
                                                                          ksel,
                                                                          session.d_B.ptr,
                                                                          session.d_B_cand.ptr,
                                                                          n);
                } else {
                    CopyAndPerturbSmallCodes<<<blocks, threads>>>(pre.m,
                                                                 session.d_hvec.ptr,
                                                                 seed,
                                                                 outer + 1,
                                                                 sample_id_offset,
                                                                 ksel,
                                                                 session.d_B.ptr,
                                                                 session.d_B_cand.ptr,
                                                                 n);
                }
                CudaCheck(cudaGetLastError(), "Launch CopyAndPerturbSmallCodes");
            });
            timing->ils_icm += session.ev.MeasureSeconds([&] {
                session.RunIcmDevice(/*use_abs=*/UseAbs,
                                     icm_iters,
                                     session.d_B_cand.ptr,
                                     session.d_a_cand.ptr,
                                     session.d_cost_cand.ptr,
                                     &rng);
            });
            timing->ils_accept += session.ev.MeasureSeconds([&] {
                if (pre.m == 5) {
                    AcceptIfBetterSmallFixed<5><<<blocks, threads>>>(session.d_B_cand.ptr,
                                                                    session.d_a_cand.ptr,
                                                                    session.d_cost_cand.ptr,
                                                                    session.d_B.ptr,
                                                                    session.d_a.ptr,
                                                                    session.d_cost.ptr,
                                                                    n);
                } else if (pre.m == 10) {
                    AcceptIfBetterSmallFixed<10><<<blocks, threads>>>(session.d_B_cand.ptr,
                                                                     session.d_a_cand.ptr,
                                                                     session.d_cost_cand.ptr,
                                                                     session.d_B.ptr,
                                                                     session.d_a.ptr,
                                                                     session.d_cost.ptr,
                                                                     n);
                } else {
                    AcceptIfBetterSmall<<<blocks, threads>>>(pre.m,
                                                            session.d_B_cand.ptr,
                                                            session.d_a_cand.ptr,
                                                            session.d_cost_cand.ptr,
                                                            session.d_B.ptr,
                                                            session.d_a.ptr,
                                                            session.d_cost.ptr,
                                                            n);
                }
                CudaCheck(cudaGetLastError(), "Launch AcceptIfBetterSmall");
            });
        } else {
            if (pre.m == 5) {
                CopyAndPerturbSmallCodesFixed<5><<<blocks, threads>>>(session.d_hvec.ptr,
                                                                     seed,
                                                                     outer + 1,
                                                                     sample_id_offset,
                                                                     ksel,
                                                                     session.d_B.ptr,
                                                                     session.d_B_cand.ptr,
                                                                     n);
            } else if (pre.m == 10) {
                CopyAndPerturbSmallCodesFixed<10><<<blocks, threads>>>(session.d_hvec.ptr,
                                                                      seed,
                                                                      outer + 1,
                                                                      sample_id_offset,
                                                                      ksel,
                                                                      session.d_B.ptr,
                                                                      session.d_B_cand.ptr,
                                                                      n);
            } else {
                CopyAndPerturbSmallCodes<<<blocks, threads>>>(pre.m,
                                                             session.d_hvec.ptr,
                                                             seed,
                                                             outer + 1,
                                                             sample_id_offset,
                                                             ksel,
                                                             session.d_B.ptr,
                                                             session.d_B_cand.ptr,
                                                             n);
            }

            session.RunIcmDevice(/*use_abs=*/UseAbs,
                                 icm_iters,
                                 session.d_B_cand.ptr,
                                 session.d_a_cand.ptr,
                                 session.d_cost_cand.ptr,
                                 &rng);

            if (pre.m == 5) {
                AcceptIfBetterSmallFixed<5><<<blocks, threads>>>(session.d_B_cand.ptr,
                                                                session.d_a_cand.ptr,
                                                                session.d_cost_cand.ptr,
                                                                session.d_B.ptr,
                                                                session.d_a.ptr,
                                                                session.d_cost.ptr,
                                                                n);
            } else if (pre.m == 10) {
                AcceptIfBetterSmallFixed<10><<<blocks, threads>>>(session.d_B_cand.ptr,
                                                                 session.d_a_cand.ptr,
                                                                 session.d_cost_cand.ptr,
                                                                 session.d_B.ptr,
                                                                 session.d_a.ptr,
                                                                 session.d_cost.ptr,
                                                                 n);
            } else {
                AcceptIfBetterSmall<<<blocks, threads>>>(pre.m,
                                                        session.d_B_cand.ptr,
                                                        session.d_a_cand.ptr,
                                                        session.d_cost_cand.ptr,
                                                        session.d_B.ptr,
                                                        session.d_a.ptr,
                                                        session.d_cost.ptr,
                                                        n);
            }
        }
    }

    if (timing) {
        Timer tt;
        if (cost) {
            session.DownloadAll(B_small->data.data(), a->data.data(), cost_out->data());
        } else {
            CudaCheck(cudaMemcpy(B_small->data.data(), session.d_B.ptr,
                                 sizeof(Code) * static_cast<std::size_t>(pre.m - 1) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      "Memcpy B_small back (no cost)");
            CudaCheck(cudaMemcpy(a->data.data(), session.d_a.ptr,
                                 sizeof(float) * static_cast<std::size_t>(pre.m) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      "Memcpy a back (no cost)");
        }
        timing->download += tt.ElapsedSeconds();
    } else {
        if (cost) {
            session.DownloadAll(B_small->data.data(), a->data.data(), cost_out->data());
        } else {
            CudaCheck(cudaMemcpy(B_small->data.data(), session.d_B.ptr,
                                 sizeof(Code) * static_cast<std::size_t>(pre.m - 1) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      "Memcpy B_small back (no cost)");
            CudaCheck(cudaMemcpy(a->data.data(), session.d_a.ptr,
                                 sizeof(float) * static_cast<std::size_t>(pre.m) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      "Memcpy a back (no cost)");
        }
    }
}

void DynamicIcmWithIlsNoAbsNoNormalLargeRootCudaDeviceXc(const ColMajorMatrix<float>& X,
                                                        const PrecompLargeRoot& pre,
                                                        const DeviceMatF32View& xC_small_dev,
                                                        int icm_iters,
                                                        int ils_iters,
                                                        int perturb_k,
                                                        std::uint32_t seed,
                                                        const std::vector<std::uint32_t>& cluster_id,
                                                        ColMajorMatrix<Code>* B_small,
                                                        ColMajorMatrix<float>* a,
                                                        std::vector<float>* X_norm2,
                                                        std::vector<float>* cost,
                                                        bool print_progress,
                                                        std::uint64_t sample_id_offset,
                                                        CudaIcmTiming* timing) {
    try {
        DynamicIcmWithIlsLargeRootCudaImplDeviceXc<false>(X, pre, xC_small_dev, icm_iters, ils_iters, perturb_k,
                                                         seed, sample_id_offset, cluster_id, B_small, a, X_norm2, cost,
                                                         print_progress, timing);
    } catch (const std::exception& e) {
        LogError(std::string("DynamicIcmWithIlsNoAbsNoNormalLargeRootCudaDeviceXc failed: ") + e.what() +
                 " (fallback to CPU via D2H xC_small)");
        if (!B_small || !a || !X_norm2) return;
        if (!xC_small_dev.ptr) {
            LogError("DynamicIcmWithIlsNoAbsNoNormalLargeRootCudaDeviceXc: xC_small_dev is null; cannot fallback.");
            return;
        }
        ColMajorMatrix<float> xC_small;
        xC_small.rows = pre.H_small;
        xC_small.cols = B_small->cols;
        xC_small.data.resize(static_cast<std::size_t>(xC_small.rows) * static_cast<std::size_t>(xC_small.cols));
        const cudaError_t st = cudaMemcpy(xC_small.data.data(), xC_small_dev.ptr,
                                          sizeof(float) * xC_small.data.size(),
                                          cudaMemcpyDeviceToHost);
        if (st != cudaSuccess) {
            LogError(std::string("Memcpy xC_small_dev D2H (fallback) failed: ") + cudaGetErrorString(st));
            return;
        }
        DynamicIcmWithIlsNoAbsNoNormalLargeRoot(X, pre, xC_small, icm_iters, ils_iters, perturb_k,
                                                seed, cluster_id, B_small, a, X_norm2, cost,
                                                print_progress, sample_id_offset);
    }
}

void DynamicIcmWithIlsAbsNoNormalLargeRootCudaDeviceXc(const ColMajorMatrix<float>& X,
                                                      const PrecompLargeRoot& pre,
                                                      const DeviceMatF32View& xC_small_dev,
                                                      int icm_iters,
                                                      int ils_iters,
                                                      int perturb_k,
                                                      std::uint32_t seed,
                                                      const std::vector<std::uint32_t>& cluster_id,
                                                      ColMajorMatrix<Code>* B_small,
                                                      ColMajorMatrix<float>* a,
                                                      std::vector<float>* X_norm2,
                                                      std::vector<float>* cost,
                                                      bool print_progress,
                                                      std::uint64_t sample_id_offset,
                                                      CudaIcmTiming* timing) {
    try {
        DynamicIcmWithIlsLargeRootCudaImplDeviceXc<true>(X, pre, xC_small_dev, icm_iters, ils_iters, perturb_k,
                                                        seed, sample_id_offset, cluster_id, B_small, a, X_norm2, cost,
                                                        print_progress, timing);
    } catch (const std::exception& e) {
        LogError(std::string("DynamicIcmWithIlsAbsNoNormalLargeRootCudaDeviceXc failed: ") + e.what() +
                 " (fallback to CPU via D2H xC_small)");
        if (!B_small || !a || !X_norm2) return;
        if (!xC_small_dev.ptr) {
            LogError("DynamicIcmWithIlsAbsNoNormalLargeRootCudaDeviceXc: xC_small_dev is null; cannot fallback.");
            return;
        }
        ColMajorMatrix<float> xC_small;
        xC_small.rows = pre.H_small;
        xC_small.cols = B_small->cols;
        xC_small.data.resize(static_cast<std::size_t>(xC_small.rows) * static_cast<std::size_t>(xC_small.cols));
        const cudaError_t st = cudaMemcpy(xC_small.data.data(), xC_small_dev.ptr,
                                          sizeof(float) * xC_small.data.size(),
                                          cudaMemcpyDeviceToHost);
        if (st != cudaSuccess) {
            LogError(std::string("Memcpy xC_small_dev D2H (fallback) failed: ") + cudaGetErrorString(st));
            return;
        }
        DynamicIcmWithIlsAbsNoNormalLargeRoot(X, pre, xC_small, icm_iters, ils_iters, perturb_k,
                                              seed, cluster_id, B_small, a, X_norm2, cost,
                                              print_progress, sample_id_offset);
    }
}

template <bool UseAbs>
static void DynamicIcmWithIlsLargeRootCudaImplDeviceXcDeviceX(const DeviceMatF32View& X_dev,
                                                              const PrecompLargeRoot& pre,
                                                              const DeviceMatF32View& xC_small_dev,
                                                              int icm_iters,
                                                              int ils_iters,
                                                              int perturb_k,
                                                              std::uint32_t seed,
                                                              std::uint64_t sample_id_offset,
                                                              const std::vector<std::uint32_t>& cluster_id,
                                                              ColMajorMatrix<Code>* B_small,
                                                              ColMajorMatrix<float>* a,
                                                              std::vector<float>* X_norm2,
                                                              std::vector<float>* cost,
                                                              bool print_progress,
                                                              CudaIcmTiming* timing) {
    if (!B_small || !a || !X_norm2) {
        return;
    }
    static thread_local std::vector<float> cost_scratch;
    std::vector<float>* cost_out = cost ? cost : &cost_scratch;
    const int n = B_small->cols;
    if (n <= 0) {
        return;
    }
    if (print_progress) {
        throw std::runtime_error("DynamicIcmWithIlsLargeRootCudaImplDeviceXcDeviceX: print_progress is not supported.");
    }
    if (!X_dev.ptr || X_dev.rows != pre.d || X_dev.cols != n || X_dev.ld != pre.d) {
        throw std::runtime_error("DynamicIcmWithIlsLargeRootCudaImplDeviceXcDeviceX: invalid X_dev view.");
    }
    if (!xC_small_dev.ptr || xC_small_dev.rows != pre.H_small || xC_small_dev.cols != n ||
        xC_small_dev.ld != pre.H_small) {
        throw std::runtime_error("DynamicIcmWithIlsLargeRootCudaImplDeviceXcDeviceX: invalid xC_small_dev view.");
    }
    if (pre.m <= 1 || pre.m > kMaxM) {
        throw std::runtime_error("DynamicIcmWithIlsLargeRootCudaImplDeviceXcDeviceX: unsupported m for CUDA device-xC path.");
    }
    if (static_cast<int>(X_norm2->size()) != n) {
        throw std::runtime_error("DynamicIcmWithIlsLargeRootCudaImplDeviceXcDeviceX: X_norm2 must be precomputed.");
    }

    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));

    static thread_local LargeRootIcmCudaSession session;
    session.EnsurePrecomp(pre);
    session.EnsureBatch(n, /*need_xc_buffer=*/false);

    if (timing) {
        Timer tt;
        session.UploadFixedDeviceXcNoXc0(xC_small_dev, *X_norm2, cluster_id);
        timing->upload_fixed += tt.ElapsedSeconds();
    } else {
        session.UploadFixedDeviceXcNoXc0(xC_small_dev, *X_norm2, cluster_id);
    }

    // Compute xC0 on device (avoid CPU dot + H2D copy).
    {
        const int threads = 256;
        const int blocks = (n + threads - 1) / threads;
        const int h0 = pre.h_vec.empty() ? 0 : pre.h_vec[0];
        ComputeXc0FromXAndC0<<<blocks, threads>>>(pre.d, h0,
                                                  X_dev.ptr, X_dev.ld,
                                                  session.d_C0.ptr, pre.d,
                                                  session.d_cid.ptr,
                                                  session.d_xC0.ptr,
                                                  n);
        CudaCheck(cudaGetLastError(), "Launch ComputeXc0FromXAndC0");
    }

    a->rows = pre.m;
    a->cols = n;
    a->data.resize(static_cast<std::size_t>(pre.m) * static_cast<std::size_t>(n));
    cost_out->resize(static_cast<std::size_t>(n));

    if (timing) {
        Timer tt;
        session.UploadB(B_small->data.data());
        timing->upload_B += tt.ElapsedSeconds();
        timing->icm_init += session.ev.MeasureSeconds([&] {
            session.RunIcmDevice(/*use_abs=*/UseAbs,
                                 icm_iters,
                                 session.d_B.ptr,
                                 session.d_a.ptr,
                                 session.d_cost.ptr,
                                 &rng);
        });
    } else {
        session.UploadB(B_small->data.data());
        session.RunIcmDevice(/*use_abs=*/UseAbs,
                             icm_iters,
                             session.d_B.ptr,
                             session.d_a.ptr,
                             session.d_cost.ptr,
                             &rng);
    }

    if (ils_iters > 0 && perturb_k > 0 && pre.m > 1) {
        const int ksel = std::min(perturb_k, pre.m);
        const int threads = 128;
        const int blocks = (n + threads - 1) / threads;
        for (int outer = 0; outer < ils_iters; ++outer) {
            if (timing) {
                timing->ils_copy_perturb += session.ev.MeasureSeconds([&] {
                    if (pre.m == 5) {
                        CopyAndPerturbSmallCodesFixed<5><<<blocks, threads>>>(session.d_hvec.ptr,
                                                                             seed, outer + 1, sample_id_offset, ksel,
                                                                             session.d_B.ptr, session.d_B_cand.ptr, n);
                    } else if (pre.m == 10) {
                        CopyAndPerturbSmallCodesFixed<10><<<blocks, threads>>>(session.d_hvec.ptr,
                                                                              seed, outer + 1, sample_id_offset, ksel,
                                                                              session.d_B.ptr, session.d_B_cand.ptr, n);
                    } else {
                        CopyAndPerturbSmallCodes<<<blocks, threads>>>(pre.m, session.d_hvec.ptr,
                                                                     seed, outer + 1, sample_id_offset, ksel,
                                                                     session.d_B.ptr, session.d_B_cand.ptr, n);
                    }
                    CudaCheck(cudaGetLastError(), "Launch CopyAndPerturbSmallCodes");
                });
                timing->ils_icm += session.ev.MeasureSeconds([&] {
                    session.RunIcmDevice(/*use_abs=*/UseAbs, icm_iters,
                                         session.d_B_cand.ptr, session.d_a_cand.ptr, session.d_cost_cand.ptr, &rng);
                });
                timing->ils_accept += session.ev.MeasureSeconds([&] {
                    if (pre.m == 5) {
                        AcceptIfBetterSmallFixed<5><<<blocks, threads>>>(session.d_B_cand.ptr, session.d_a_cand.ptr,
                                                                        session.d_cost_cand.ptr,
                                                                        session.d_B.ptr, session.d_a.ptr, session.d_cost.ptr, n);
                    } else if (pre.m == 10) {
                        AcceptIfBetterSmallFixed<10><<<blocks, threads>>>(session.d_B_cand.ptr, session.d_a_cand.ptr,
                                                                         session.d_cost_cand.ptr,
                                                                         session.d_B.ptr, session.d_a.ptr, session.d_cost.ptr, n);
                    } else {
                        AcceptIfBetterSmall<<<blocks, threads>>>(pre.m, session.d_B_cand.ptr, session.d_a_cand.ptr,
                                                                session.d_cost_cand.ptr,
                                                                session.d_B.ptr, session.d_a.ptr, session.d_cost.ptr, n);
                    }
                    CudaCheck(cudaGetLastError(), "Launch AcceptIfBetterSmall");
                });
            } else {
                if (pre.m == 5) {
                    CopyAndPerturbSmallCodesFixed<5><<<blocks, threads>>>(session.d_hvec.ptr,
                                                                         seed, outer + 1, sample_id_offset, ksel,
                                                                         session.d_B.ptr, session.d_B_cand.ptr, n);
                } else if (pre.m == 10) {
                    CopyAndPerturbSmallCodesFixed<10><<<blocks, threads>>>(session.d_hvec.ptr,
                                                                          seed, outer + 1, sample_id_offset, ksel,
                                                                          session.d_B.ptr, session.d_B_cand.ptr, n);
                } else {
                    CopyAndPerturbSmallCodes<<<blocks, threads>>>(pre.m, session.d_hvec.ptr,
                                                                 seed, outer + 1, sample_id_offset, ksel,
                                                                 session.d_B.ptr, session.d_B_cand.ptr, n);
                }
                session.RunIcmDevice(/*use_abs=*/UseAbs, icm_iters,
                                     session.d_B_cand.ptr, session.d_a_cand.ptr, session.d_cost_cand.ptr, &rng);
                if (pre.m == 5) {
                    AcceptIfBetterSmallFixed<5><<<blocks, threads>>>(session.d_B_cand.ptr, session.d_a_cand.ptr,
                                                                    session.d_cost_cand.ptr,
                                                                    session.d_B.ptr, session.d_a.ptr, session.d_cost.ptr, n);
                } else if (pre.m == 10) {
                    AcceptIfBetterSmallFixed<10><<<blocks, threads>>>(session.d_B_cand.ptr, session.d_a_cand.ptr,
                                                                     session.d_cost_cand.ptr,
                                                                     session.d_B.ptr, session.d_a.ptr, session.d_cost.ptr, n);
                } else {
                    AcceptIfBetterSmall<<<blocks, threads>>>(pre.m, session.d_B_cand.ptr, session.d_a_cand.ptr,
                                                            session.d_cost_cand.ptr,
                                                            session.d_B.ptr, session.d_a.ptr, session.d_cost.ptr, n);
                }
            }
        }
    }

    if (timing) {
        Timer tt;
        if (cost) {
            session.DownloadAll(B_small->data.data(), a->data.data(), cost_out->data());
        } else {
            CudaCheck(cudaMemcpy(B_small->data.data(), session.d_B.ptr,
                                 sizeof(Code) * static_cast<std::size_t>(pre.m - 1) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      "Memcpy B_small back (no cost)");
            CudaCheck(cudaMemcpy(a->data.data(), session.d_a.ptr,
                                 sizeof(float) * static_cast<std::size_t>(pre.m) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      "Memcpy a back (no cost)");
        }
        timing->download += tt.ElapsedSeconds();
    } else {
        if (cost) {
            session.DownloadAll(B_small->data.data(), a->data.data(), cost_out->data());
        } else {
            CudaCheck(cudaMemcpy(B_small->data.data(), session.d_B.ptr,
                                 sizeof(Code) * static_cast<std::size_t>(pre.m - 1) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      "Memcpy B_small back (no cost)");
            CudaCheck(cudaMemcpy(a->data.data(), session.d_a.ptr,
                                 sizeof(float) * static_cast<std::size_t>(pre.m) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      "Memcpy a back (no cost)");
        }
    }
}

void DynamicIcmWithIlsNoAbsNoNormalLargeRootCudaDeviceXcDeviceX(const DeviceMatF32View& X_dev,
                                                               const PrecompLargeRoot& pre,
                                                               const DeviceMatF32View& xC_small_dev,
                                                               int icm_iters,
                                                               int ils_iters,
                                                               int perturb_k,
                                                               std::uint32_t seed,
                                                               const std::vector<std::uint32_t>& cluster_id,
                                                               ColMajorMatrix<Code>* B_small,
                                                               ColMajorMatrix<float>* a,
                                                               std::vector<float>* X_norm2,
                                                               std::vector<float>* cost,
                                                               bool print_progress,
                                                               std::uint64_t sample_id_offset,
                                                               CudaIcmTiming* timing) {
    try {
        DynamicIcmWithIlsLargeRootCudaImplDeviceXcDeviceX<false>(X_dev, pre, xC_small_dev,
                                                                icm_iters, ils_iters, perturb_k,
                                                                seed, sample_id_offset, cluster_id,
                                                                B_small, a, X_norm2, cost,
                                                                print_progress, timing);
    } catch (const std::exception& e) {
        LogError(std::string("DynamicIcmWithIlsNoAbsNoNormalLargeRootCudaDeviceXcDeviceX failed: ") + e.what() +
                 " (fallback to CPU via D2H X + xC_small)");
        if (!X_dev.ptr || !xC_small_dev.ptr || !B_small) {
            LogError("DynamicIcmWithIlsNoAbsNoNormalLargeRootCudaDeviceXcDeviceX: missing inputs; cannot fallback.");
            return;
        }
        ColMajorMatrix<float> X;
        X.rows = pre.d;
        X.cols = B_small->cols;
        X.data.resize(static_cast<std::size_t>(X.rows) * static_cast<std::size_t>(X.cols));
        {
            const cudaError_t st = cudaMemcpy(X.data.data(), X_dev.ptr,
                                              sizeof(float) * X.data.size(),
                                              cudaMemcpyDeviceToHost);
            if (st != cudaSuccess) {
                LogError(std::string("Memcpy X_dev D2H (fallback) failed: ") + cudaGetErrorString(st));
                return;
            }
        }
        ColMajorMatrix<float> xC_small;
        xC_small.rows = pre.H_small;
        xC_small.cols = B_small->cols;
        xC_small.data.resize(static_cast<std::size_t>(xC_small.rows) * static_cast<std::size_t>(xC_small.cols));
        {
            const cudaError_t st = cudaMemcpy(xC_small.data.data(), xC_small_dev.ptr,
                                              sizeof(float) * xC_small.data.size(),
                                              cudaMemcpyDeviceToHost);
            if (st != cudaSuccess) {
                LogError(std::string("Memcpy xC_small_dev D2H (fallback) failed: ") + cudaGetErrorString(st));
                return;
            }
        }
        DynamicIcmWithIlsNoAbsNoNormalLargeRoot(X, pre, xC_small, icm_iters, ils_iters, perturb_k,
                                                seed, cluster_id, B_small, a, X_norm2, cost,
                                                print_progress, sample_id_offset);
    }
}

void DynamicIcmWithIlsAbsNoNormalLargeRootCudaDeviceXcDeviceX(const DeviceMatF32View& X_dev,
                                                             const PrecompLargeRoot& pre,
                                                             const DeviceMatF32View& xC_small_dev,
                                                             int icm_iters,
                                                             int ils_iters,
                                                             int perturb_k,
                                                             std::uint32_t seed,
                                                             const std::vector<std::uint32_t>& cluster_id,
                                                             ColMajorMatrix<Code>* B_small,
                                                             ColMajorMatrix<float>* a,
                                                             std::vector<float>* X_norm2,
                                                             std::vector<float>* cost,
                                                             bool print_progress,
                                                             std::uint64_t sample_id_offset,
                                                             CudaIcmTiming* timing) {
    try {
        DynamicIcmWithIlsLargeRootCudaImplDeviceXcDeviceX<true>(X_dev, pre, xC_small_dev,
                                                               icm_iters, ils_iters, perturb_k,
                                                               seed, sample_id_offset, cluster_id,
                                                               B_small, a, X_norm2, cost,
                                                               print_progress, timing);
    } catch (const std::exception& e) {
        LogError(std::string("DynamicIcmWithIlsAbsNoNormalLargeRootCudaDeviceXcDeviceX failed: ") + e.what() +
                 " (fallback to CPU via D2H X + xC_small)");
        if (!X_dev.ptr || !xC_small_dev.ptr || !B_small) {
            LogError("DynamicIcmWithIlsAbsNoNormalLargeRootCudaDeviceXcDeviceX: missing inputs; cannot fallback.");
            return;
        }
        ColMajorMatrix<float> X;
        X.rows = pre.d;
        X.cols = B_small->cols;
        X.data.resize(static_cast<std::size_t>(X.rows) * static_cast<std::size_t>(X.cols));
        {
            const cudaError_t st = cudaMemcpy(X.data.data(), X_dev.ptr,
                                              sizeof(float) * X.data.size(),
                                              cudaMemcpyDeviceToHost);
            if (st != cudaSuccess) {
                LogError(std::string("Memcpy X_dev D2H (fallback) failed: ") + cudaGetErrorString(st));
                return;
            }
        }
        ColMajorMatrix<float> xC_small;
        xC_small.rows = pre.H_small;
        xC_small.cols = B_small->cols;
        xC_small.data.resize(static_cast<std::size_t>(xC_small.rows) * static_cast<std::size_t>(xC_small.cols));
        {
            const cudaError_t st = cudaMemcpy(xC_small.data.data(), xC_small_dev.ptr,
                                              sizeof(float) * xC_small.data.size(),
                                              cudaMemcpyDeviceToHost);
            if (st != cudaSuccess) {
                LogError(std::string("Memcpy xC_small_dev D2H (fallback) failed: ") + cudaGetErrorString(st));
                return;
            }
        }
        DynamicIcmWithIlsAbsNoNormalLargeRoot(X, pre, xC_small, icm_iters, ils_iters, perturb_k,
                                              seed, cluster_id, B_small, a, X_norm2, cost,
                                              print_progress, sample_id_offset);
    }
}

}  // namespace stlq
