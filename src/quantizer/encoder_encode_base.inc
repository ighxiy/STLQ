bool EncodeBase(const Config& config,
                const Dataset& dataset,
                const CodebookPack& codebooks,
                const Precomp& precomp,
                KernelProvider* kernels,
                BaseEncoding* base,
                float* out_beam_mse,
                float* out_final_mse,
                std::string* error) {
    (void)kernels;
    if (!base) {
        if (error) {
            *error = "BaseEncoding output is null.";
        }
        return false;
    }

    const int d = dataset.Xb.rows;
    const int n = dataset.Xb.cols;
    if (d <= 0 || n <= 0) {
        if (error) {
            *error = "Empty base dataset.";
        }
        return false;
    }
    if (precomp.d != d || precomp.m != static_cast<int>(codebooks.books.size())) {
        if (error) {
            *error = "Precomp dimensions do not match codebooks or dataset.";
        }
        return false;
    }

    ColMajorMatrix<float> xC(precomp.H, n);
    {
        ScopedBlasThreads blas_scope(OmpMaxThreads());
        Gemm(true, false, 1.0f, precomp.C_all, dataset.Xb, 0.0f, &xC);
    }

    const int H_beam = std::max(1, config.base.encode.H_beam);
    BeamSearchPrefixLS(dataset.Xb, precomp, xC, H_beam, &base->B);

    // Julia reference: compute LS coeffs after beam search, then report base MSE.
    // (demos/main.jl:134-140).
    SolveLeastSquaresAll(precomp, xC, base->B, &base->a);
    float beam_mse = MeanMseFromPrecomp(dataset.Xb, precomp, xC, base->B, base->a);
    LogInfo("BaseSet beam search mse error: " + FormatFloat(beam_mse, 6));
    if (out_beam_mse) {
        *out_beam_mse = beam_mse;
    }

    std::vector<float> X_norm2;
    std::vector<float> base_cost;
    [[maybe_unused]] const bool want_cuda_icm = config.runtime.use_cuda;
    if (config.base.encode.use_abs) {
        // Julia reference: dynamic_icm_with_ils_abs_nonormal!
        // (encode_abs.jl:379-512, demos/main.jl:126-131).
#if defined(STLQ_ENABLE_CUDA)
        if (want_cuda_icm) {
            DynamicIcmWithIlsAbsNoNormalCuda(dataset.Xb, precomp, xC,
                                             config.base.encode.icm_iters,
                                             config.base.encode.ils_iters,
                                             config.base.encode.perturb_k,
                                             static_cast<std::uint32_t>(config.base.encode.seed),
                                             &base->B, &base->a, &X_norm2, &base_cost,
                                             /*print_progress=*/false);
        } else
#endif
        {
            DynamicIcmWithIlsAbsNoNormal(dataset.Xb, precomp, xC,
                                         config.base.encode.icm_iters,
                                         config.base.encode.ils_iters,
                                         config.base.encode.perturb_k,
                                         static_cast<std::uint32_t>(config.base.encode.seed),
                                         &base->B, &base->a, &X_norm2, &base_cost);
        }
    } else {
        // Julia reference: dynamic_icm_with_ils_noabs_nonormal!
        // (encode_noabs.jl:8-112).
#if defined(STLQ_ENABLE_CUDA)
        if (want_cuda_icm) {
            DynamicIcmWithIlsNoAbsNoNormalCuda(dataset.Xb, precomp, xC,
                                               config.base.encode.icm_iters,
                                               config.base.encode.ils_iters,
                                               config.base.encode.perturb_k,
                                               static_cast<std::uint32_t>(config.base.encode.seed),
                                               &base->B, &base->a, &X_norm2, &base_cost,
                                               /*print_progress=*/false);
        } else
#endif
        {
            DynamicIcmWithIlsNoAbsNoNormal(dataset.Xb, precomp, xC,
                                           config.base.encode.icm_iters,
                                           config.base.encode.ils_iters,
                                           config.base.encode.perturb_k,
                                           static_cast<std::uint32_t>(config.base.encode.seed),
                                           &base->B, &base->a, &X_norm2, &base_cost);
        }
    }

    const float final_mse = MeanMseFromPrecomp(dataset.Xb, precomp, xC, base->B, base->a);
    LogInfo("BaseSet ILS+ICM mse error: " + FormatFloat(final_mse, 6));
    if (out_final_mse) {
        *out_final_mse = final_mse;
    }
    return true;
}
