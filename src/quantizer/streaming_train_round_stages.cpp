#include "stlq/quantizer/streaming_train_round_stages.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>

#include "stlq/pipeline/app_utils.h"
#include "stlq/linkage/linkage_streaming_builders.h"
#include "stlq/linkage/linkage_summary.h"
#include "stlq/io/base_store.h"
#include "stlq/io/linkage_list_store.h"
#include "stlq/common/logger.h"
#include "stlq/quantizer/encode_base_streaming.h"
#include "stlq/quantizer/precomp_large_root.h"
#include "stlq/quantizer/update_codebooks_streaming.h"
#include "stlq/quantizer/streaming_train_io.h"
#include "stlq/common/timer.h"

#if defined(STLQ_ENABLE_CUDA)
#include "stlq/linkage/linkage_encode_cuda.h"
#include "stlq/core/kernel_provider_cuda_stream.h"
#include <cuda_runtime.h>
#endif

namespace stlq {

namespace {

#if defined(STLQ_ENABLE_CUDA)
void LogCudaVramIfEnabled(const Config& cfg, const char* label) {
    if (!cfg.runtime.use_cuda) return;
    if (!cfg.large.profile_timing) return;
    std::size_t free_b = 0;
    std::size_t total_b = 0;
    const cudaError_t st = cudaMemGetInfo(&free_b, &total_b);
    if (st != cudaSuccess) {
        LogWarn(std::string("cudaMemGetInfo failed (") + label + "): " + cudaGetErrorString(st));
        return;
    }
    int device = 0;
    (void)cudaGetDevice(&device);
    const std::size_t used_b = (total_b >= free_b) ? (total_b - free_b) : 0;
    const auto used_mib = static_cast<std::uint64_t>(used_b / (1024ull * 1024ull));
    const auto free_mib = static_cast<std::uint64_t>(free_b / (1024ull * 1024ull));
    const auto total_mib = static_cast<std::uint64_t>(total_b / (1024ull * 1024ull));
    LogInfo(std::string("CUDA VRAM (") + label + "): device=" + std::to_string(device) +
            " used_mib=" + std::to_string(used_mib) +
            " free_mib=" + std::to_string(free_mib) +
            " total_mib=" + std::to_string(total_mib));
}
#endif

}  // namespace

bool RunStreamingTrainBasicEncodeStage(const Config& config,
                                       TrainResult& result,
                                       const StreamingTrainWorkspacePaths& paths,
                                       std::uint64_t ntrain,
                                       int d,
                                       bool is_global_round,
                                       StreamKernelProvider* stream_kernels,
                                       std::string* error) {
    const stlq::Timer round_prep_wall;

#if defined(STLQ_ENABLE_CUDA)
    if (config.runtime.use_cuda) {
        // Avoid unbounded VRAM growth across rounds when codebooks/precomps are rebuilt (build_tag changes).
        // Safe here because the previous round is complete and all CUDA work is synchronized by design.
        ClearCudaLinkageEncodeSharedPrecompCaches(/*device=*/-1);
    }
#endif

    // Large pipeline: use large-root precomp unconditionally (do not maintain full-precomp here).
    double precomp_croot_wall_sec = 0.0;
    PrecompLargeRoot pre_lr;
    {
        const stlq::Timer precomp_wall;
        std::string local_err;
        PrecompLargeRootBuildOptions pre_opts;
        pre_opts.build_g0s_transpose = config.runtime.precomp_large_root_g0s_transpose;
        pre_opts.g0s_transpose_max_mb = config.runtime.precomp_large_root_g0s_transpose_max_mb;
        if (!BuildPrecompLargeRoot(result.C_root, stream_kernels, pre_opts, &pre_lr, &local_err)) {
            if (error) {
                *error =
                    "TrainQuantizerStreamingLarge: BuildPrecompLargeRoot(C_root) failed (large pipeline does not use full-precomp): " +
                    (local_err.empty() ? "unknown error" : local_err);
            }
            return false;
        }
        precomp_croot_wall_sec = precomp_wall.ElapsedSeconds();
    }
    PrecompLargeRoot* pre_lr_ptr = &pre_lr;

    // Dummy precomp for large-root path (EncodeBaseStreaming will not use it).
    Precomp pre_root;
    pre_root.H = 0;
    pre_root.d = d;
    pre_root.m = config.model.m;
    pre_root.h_vec = config.model.h_vec;

    // Train basic pass (Prompt 05): write train_basic.
    Config enc_cfg = MakeTrainBasicEncodeConfig(config, ntrain);

    float train_beam_mse = -1.0f;
    float train_final_mse = -1.0f;
    {
        const stlq::Timer basic_open_wall;
        // LogInfo("Train streaming basic pass: writing train_basic to " + paths.train_basic_dir.string());

        StreamingTrainInput basic_input;
        if (!OpenStreamingTrainInput(enc_cfg, &basic_input, error)) {
            return false;
        }
        io::BaseBasicStoreConfig store_cfg =
            MakeTrainBasicStoreConfig(config, paths.train_basic_dir, d, basic_input.train_is_u8);

        io::BaseBasicWriter writer;
        if (!writer.Open(store_cfg, error)) {
            return false;
        }
        if (is_global_round) {
            // One-line breakdown to help separate "prep" (alloc/open/precomp) vs actual compute.
            const double basic_open_sec = basic_open_wall.ElapsedSeconds();
            const double prep_total_sec = round_prep_wall.ElapsedSeconds();
            LogInfo("Train round prep wall time(s): precomp_croot=" +
                    FormatFloat(static_cast<float>(precomp_croot_wall_sec), 3) +
                    " basic_open=" + FormatFloat(static_cast<float>(basic_open_sec), 3) +
                    " total=" + FormatFloat(static_cast<float>(prep_total_sec), 3));
        }
        float* out_beam_ptr = config.train.log_metrics ? &train_beam_mse : nullptr;
        float* out_final_ptr = config.train.log_metrics ? &train_final_mse : nullptr;
        const bool ok =
            basic_input.train_is_u8 ? EncodeBasicStreaming(enc_cfg, basic_input.reader.bvecs, result.R, result.C_root,
                                                           pre_root, pre_lr_ptr, stream_kernels,
                                                           &writer, out_beam_ptr, out_final_ptr, error)
                  : (basic_input.train_format == TrainFormat::kFbin)
                    ? EncodeBaseStreamingF32(enc_cfg, basic_input.reader.fbin, result.R, result.C_root,
                                             pre_root, pre_lr_ptr, stream_kernels,
                                             &writer, out_beam_ptr, out_final_ptr, error)
                    : EncodeBasicStreamingF32(enc_cfg, basic_input.reader.fvecs, result.R, result.C_root,
                                              pre_root, pre_lr_ptr, stream_kernels,
                                              &writer, out_beam_ptr, out_final_ptr, error);
        if (!ok) {
            return false;
        }
        if (!writer.Close(error)) {
            return false;
        }
    }
#if defined(STLQ_ENABLE_CUDA)
    LogCudaVramIfEnabled(config, "after train basic encode");
#endif
    return true;
}

bool RunStreamingTrainUpdateCRootStage(const Config& config,
                                       TrainResult& result,
                                       const StreamingTrainWorkspacePaths& paths,
                                       StreamKernelProvider* stream_kernels,
                                       std::string* error) {
    // Update C_root from train_basic (Prompt 09).
    StreamingTrainInput train_input;
    if (!OpenStreamingTrainInput(config, &train_input, error)) {
        return false;
    }
    float mse_after_croot = 0.0f;
    float* mse_after_ptr = config.train.log_metrics ? &mse_after_croot : nullptr;
    double mse_wall_sec = 0.0;
    double* mse_wall_ptr = config.train.log_metrics ? &mse_wall_sec : nullptr;
    const stlq::Timer update_croot_wall;
    if (!UpdateCRootFromTrainBasicStreamingExactLS(config,
                                                   paths.train_basic_dir.string(),
                                                   result.R,
                                                   &train_input.reader,
                                                   stream_kernels,
                                                   &result.C_root,
                                                   mse_after_ptr,
                                                   mse_wall_ptr,
                                                   error)) {
        return false;
    }
    const double total_sec = update_croot_wall.ElapsedSeconds();
    if (config.train.log_metrics) {
        LogInfo("(3)Train basic MSE after updating C_root: " + FormatFloat(mse_after_croot, 6));
        LogInfo("Train update C_root MSE wall time(s): " + FormatFloat(static_cast<float>(mse_wall_sec), 3));
        const double solve_sec = std::max(0.0, total_sec - mse_wall_sec);
        LogInfo("Train update C_root solve wall time(s): " + FormatFloat(static_cast<float>(solve_sec), 3));
        // Note: total ~= solve + mse (we intentionally do not print total here to avoid duplicate/confusing timers).
    } else {
        LogInfo("Train update C_root wall time(s): " + FormatFloat(static_cast<float>(total_sec), 3));
    }
#if defined(STLQ_ENABLE_CUDA)
    LogCudaVramIfEnabled(config, "after updating C_root");
#endif
    return true;
}

bool BuildStreamingTrainIvfAndList(const Config& config,
                                   const StreamingTrainWorkspacePaths& paths,
                                   int nlist,
                                   io::IvfListsReader* ivf,
                                   io::BaseListReader* train_list,
                                   std::string* error) {
    if (!ivf || !train_list) {
        if (error) *error = "BuildStreamingTrainIvfAndList: output reader is null.";
        return false;
    }

    // Build train_ivf and train_list from train_basic (Prompt 06).
    const std::string cluster_id_path = (paths.train_basic_dir / "cluster_id.u32").string();
    io::IvfListsPaths ivf_paths;
    ivf_paths.offsets_u64 = (paths.train_ivf_dir / "ivf_offsets.u64").string();
    ivf_paths.ids_u32 = (paths.train_ivf_dir / "ivf_ids.u32").string();
    const std::filesystem::path ivf_tmp =
        std::filesystem::path(config.large.tmp_dir) / config.dataset.name / config.io.pre_fix / "train_ivf_tmp";

    if (!io::BuildIvfListsFromClusterIdFile(cluster_id_path,
                                            nlist,
                                            std::max(1, config.large.cluster_bucket_size),
                                            ivf_paths,
                                            ivf_tmp.string(),
                                            /*keep_tmp=*/false,
                                            error)) {
        return false;
    }

    if (!ivf->Open(ivf_paths, error)) {
        return false;
    }

    const std::filesystem::path list_tmp =
        std::filesystem::path(config.large.tmp_dir) / config.dataset.name / config.io.pre_fix / "train_list_tmp";
    // LogInfo("Train streaming: building train_list (list-order store) ...");
    if (!io::BuildBaseListStoreFromBasicBuckets(paths.train_basic_dir.string(),
                                                *ivf,
                                                paths.train_list_dir.string(),
                                                list_tmp.string(),
                                                /*keep_tmp=*/false,
                                                /*profile_timing=*/config.large.profile_timing,
                                                error)) {
        return false;
    }

    if (!train_list->Open(paths.train_list_dir.string(), error)) {
        return false;
    }
    return true;
}

bool OpenStreamingTrainIvfAndList(const StreamingTrainWorkspacePaths& paths,
                                  io::IvfListsReader* ivf,
                                  io::BaseListReader* train_list,
                                  std::string* error) {
    if (!ivf || !train_list) {
        if (error) *error = "OpenStreamingTrainIvfAndList: output reader is null.";
        return false;
    }
    io::IvfListsPaths ivf_paths;
    ivf_paths.offsets_u64 = (paths.train_ivf_dir / "ivf_offsets.u64").string();
    ivf_paths.ids_u32 = (paths.train_ivf_dir / "ivf_ids.u32").string();
    if (!ivf->Open(ivf_paths, error)) {
        return false;
    }
    if (!train_list->Open(paths.train_list_dir.string(), error)) {
        return false;
    }
    return true;
}

void PrepareStreamingTrainLinkageRoundContext(const Config& config,
                                            const StreamingTrainWorkspacePaths& paths,
                                            const io::IvfListsReader& ivf,
                                            const StreamingTrainInput& train_input,
                                            bool allow_random_fallback,
                                            StreamKernelProvider* stream_kernels,
                                            bool* logged_linkage_stage_cpu_notice,
                                            StreamingTrainLinkageRoundContext* context) {
    if (!context) return;

    // NOTE: CudaStreamKernels is not thread-safe (one cublasHandle + shared scratch buffers).
    // The linkage stages are cluster-parallel (OpenMP) and may call GEMM from multiple threads.
    // For correctness, force CPU stream kernels for linkage build / analysis / updates for now.
    context->linkage_stream_kernels = stream_kernels;
    if (stream_kernels && stream_kernels->IsGpu()) {
        context->linkage_stream_kernels = &context->cpu_linkage_kernels;
        if (logged_linkage_stage_cpu_notice && !*logged_linkage_stage_cpu_notice) {
            LogInfo("CUDA enabled: using CPU stream kernels for linkage stages (thread-safety); basic encoding stays on GPU.");
            *logged_linkage_stage_cpu_notice = true;
        }
    }

    // Linkage build (Prompt 07): build train_linkage_list from train_list + raw (via buckets or fallback).
    context->linkage_store_cfg =
        MakeTrainLinkageListStoreConfig(config, paths.train_linkage_list_dir, ivf.nlist());

    context->fallback_reader = &train_input.reader;
    context->allow_random_fallback = allow_random_fallback;
    context->linkage_cfg = MakeTrainLinkageConfig(config);
}

bool RunStreamingTrainIterationTailStage(const Config& config,
                                         TrainResult& result,
                                         const StreamingTrainWorkspacePaths& paths,
                                         int global_iter,
                                         bool do_opq_update,
                                         io::BaseListReader& train_list,
                                         io::IvfListsReader& ivf,
                                         StreamKernelProvider* rvq_stream_kernels,
                                         StreamKernelProvider* linkage_stream_kernels,
                                         const io::DatasetVectorReader* fallback_reader,
                                         bool allow_random_fallback,
                                         const io::LinkageListStoreConfig& linkage_store_cfg,
                                         const Config& linkage_cfg,
                                         std::string* error) {
#if defined(STLQ_ENABLE_CUDA)
    // Global-iteration linkage build uses a CUDA ctx pool and may be VRAM-hungry (e.g. large h0_root).
    // Force-release RVQ/basic CUDA cached buffers before entering the linkage stage to avoid VRAM stacking.
    if (config.runtime.use_cuda) if (auto* cuda = dynamic_cast<CudaStreamKernels*>(rvq_stream_kernels)) {
        cuda->ResetCudaState();
    }
    LogCudaVramIfEnabled(config, "before iteration-linkage");
#endif
    const stlq::Timer iter_linkage_wall;
    const stlq::VirtualUmapReencodeEncodeConfig umap_reencode_cfg{
        .ils_iters = config.train.ils_iters,
        .icm_iters = config.train.icm_iters,
        .perturb_k = config.train.perturb_k,
        .seed = static_cast<std::uint32_t>(config.train.seed),
    };
    if (!BuildLinkageTwoCodebookVirtualStreamingByCluster(linkage_cfg,
                                                        result,
                                                        umap_reencode_cfg,
                                                        train_list,
                                                        ivf,
                                                        fallback_reader,
                                                        allow_random_fallback,
                                                        linkage_stream_kernels,
                                                        linkage_store_cfg,
                                                        /*base_depth_stats_out=*/nullptr,
                                                        /*base_linkage_mse_out=*/nullptr,
                                                        error)) {
        return false;
    }
    LogInfo(iter_linkage_wall.ReportSeconds("Iteration-linkage wall time(s)"));

    // Update C_one from train_linkage_list (Prompt 08).
    io::LinkageListReader linkage_list;
    if (!linkage_list.Open(paths.train_linkage_list_dir.string(), error)) {
        return false;
    }

    if (config.train.log_metrics && config.train.log_linkage_pre_c1) {
        LinkageDepthStats stats_pre;
        MseStats mse_pre;
        const stlq::Timer iter_pre_mse_wall;
        if (!AnalyzeTrainLinkageListAfterCOneUpdateStreaming(config,
                                                           result,
                                                           train_list,
                                                           ivf,
                                                           linkage_list,
                                                           /*code0_one_bytes_are_root_codes=*/false,
                                                           fallback_reader,
                                                           allow_random_fallback,
                                                           linkage_stream_kernels,
                                                           &stats_pre,
                                                           &mse_pre,
                                                           /*opq_M_out=*/nullptr,
                                                           error)) {
            return false;
        }
        LogInfo(iter_pre_mse_wall.ReportSeconds("Iteration-linkage linkage-summary+MSE analysis (pre C_one update) wall time(s)"));
        LogInfo("Train Linkage encode analysis before updating C_one:");
        LogInfo("  Linkage Ratio: " + FormatFloat(100.0f * stats_pre.linkage_ratio, 2) + "%");
        LogInfo("  Max depth: " + std::to_string(stats_pre.max_depth) +
                " Mean depth: " + FormatFloat(stats_pre.mean_depth, 2));
        LogInfo("  Linkage Error: Max=" + FormatFloat(mse_pre.max, 4) +
                ", Min=" + FormatFloat(mse_pre.min, 4) +
                ", Mean=" + FormatFloat(mse_pre.mean, 4));
    }

    //LogInfo("Train streaming: updating C_one from train_linkage_list (depth>0 only) ...");
    const stlq::Timer iter_update_c1_wall;
    if (!UpdateCOneFromTrainLinkageListStreaming(config,
                                               result,
                                               train_list,
                                               ivf,
                                               linkage_list,
                                               fallback_reader,
                                               allow_random_fallback,
                                               linkage_stream_kernels,
                                               &result.C_one,
                                               error)) {
        return false;
    }
    LogInfo(iter_update_c1_wall.ReportSeconds("Iteration-linkage update C_one wall time(s)"));

    // is_bad_cluster is computed only when virtual nodes are enabled.
    // Fallback: if the init-stage mask is unavailable, derive it from the two-codebook linkage_list.
    if (config.virtual_cfg.enabled && global_iter == 0 && result.is_bad_cluster.empty()) {
        double baseline = 0.0;
        if (!ComputeBadClusterMaskFromLinkageListStreaming(config,
                                                         result,
                                                         train_list,
                                                         ivf,
                                                         linkage_list,
                                                         fallback_reader,
                                                         allow_random_fallback,
                                                         linkage_stream_kernels,
                                                         &result.is_bad_cluster,
                                                         &baseline,
                                                         error)) {
            LogWarn("Failed to compute is_bad_cluster; virtual node weighting disabled.");
            result.is_bad_cluster.clear();
        }
    }
    ColMajorMatrix<float> opq_M_from_post_analysis;
    bool have_opq_M_from_post_analysis = false;
    if (config.train.log_metrics) {
        LinkageDepthStats stats;
        MseStats mse;
        ColMajorMatrix<float>* iter_opq_M_ptr =
            (do_opq_update && config.train.use_opq_rotation) ? &opq_M_from_post_analysis : nullptr;
        const stlq::Timer iter_post_mse_wall;
        if (!AnalyzeTrainLinkageListAfterCOneUpdateStreaming(config,
                                                           result,
                                                           train_list,
                                                           ivf,
                                                           linkage_list,
                                                           /*code0_one_bytes_are_root_codes=*/false,
                                                           fallback_reader,
                                                           allow_random_fallback,
                                                           linkage_stream_kernels,
                                                           &stats,
                                                           &mse,
                                                           iter_opq_M_ptr,
                                                           error)) {
            return false;
        }
        have_opq_M_from_post_analysis = (iter_opq_M_ptr && !opq_M_from_post_analysis.data.empty());
        LogInfo(iter_post_mse_wall.ReportSeconds("Iteration-linkage linkage-summary+MSE analysis (post C_one update) wall time(s)"));
        LogInfo("Train Linkage encode analysis after updating C_one:");
        LogInfo("  Linkage Ratio: " + FormatFloat(100.0f * stats.linkage_ratio, 2) + "%");
        LogInfo("  Max depth: " + std::to_string(stats.max_depth) +
                " Mean depth: " + FormatFloat(stats.mean_depth, 2));
        LogInfo("  Linkage Error: Max=" + FormatFloat(mse.max, 4) +
                ", Min=" + FormatFloat(mse.min, 4) +
                ", Mean=" + FormatFloat(mse.mean, 4));
    }

    // OPQ update (Prompt 10): optional.
    if (do_opq_update && config.train.use_opq_rotation) {
        const stlq::Timer opq_wall;
        //LogInfo("Train streaming: updating OPQ rotation R from train_linkage_list ...");
        if (config.train.log_metrics && have_opq_M_from_post_analysis) {
            // Fast path: OPQ M was accumulated during the post-C_one analysis.
            if (!UpdateOpqRotationFromCrossCov(opq_M_from_post_analysis, &result.R, error)) {
                return false;
            }
        } else if (!UpdateOpqRotationStreamingByCluster(config,
                                                        result,
                                                        train_list,
                                                        ivf,
                                                        linkage_list,
                                                        fallback_reader,
                                                        allow_random_fallback,
                                                        linkage_stream_kernels,
                                                        &result.R,
                                                        error)) {
            return false;
        }
        LogInfo("Iteration-tail wall time(s): opq_update=" +
                FormatFloat(static_cast<float>(opq_wall.ElapsedSeconds()), 3));
    } else if (do_opq_update && !config.train.use_opq_rotation) {
        LogInfo("Train streaming: use_opq_rotation=false; keeping R unchanged.");
    }
    return true;
}

}  // namespace stlq
