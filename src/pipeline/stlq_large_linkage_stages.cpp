#include "stlq/pipeline/stlq_large_linkage_stages.h"

#include "stlq/pipeline/app_utils.h"
#include "stlq/pipeline/large_store_hash.h"
#include "stlq/pipeline/large_store_lifecycle.h"
#include "stlq/pipeline/main_virtual_helpers.h"
#include "stlq/linkage/linkage_streaming_builders.h"
#include "stlq/core/blas.h"
#include "stlq/core/kernel_provider_cuda_stream.h"
#include "stlq/eval/recall_disk.h"
#include "stlq/io/dataset_io.h"
#include "stlq/io/result_io.h"
#include "stlq/common/logger.h"
#include "stlq/quantizer/encode_base_streaming.h"
#include "stlq/quantizer/reconstruct.h"
#include "stlq/common/timer.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <utility>

namespace stlq {

int PrepareLargeLinkageListStateStage(const stlq::Config& config,
                                    const stlq::TrainResult& train_result,
                                    const LargeRunState& run_state,
                                    const LargeIvfStageState& ivf_state,
                                    const LargeBaseListStageState& base_list_state,
                                    LargeLinkageListStageState& linkage_list_state) {
    using namespace stlq;
    linkage_list_state.linkage_list_dir =
        (std::filesystem::path(run_state.effective_run_root) / "linkage_list").string();
    if (config.io.linkage_file_set && !config.io.linkage_file.empty()) {
        std::error_code ec;
        const std::filesystem::path explicit_linkage_path(config.io.linkage_file);
        if (std::filesystem::exists(explicit_linkage_path, ec) && !ec &&
            std::filesystem::is_directory(explicit_linkage_path, ec) && !ec) {
            linkage_list_state.linkage_list_dir = explicit_linkage_path.string();
        }
    }
    linkage_list_state.linkage_list_meta =
        (std::filesystem::path(linkage_list_state.linkage_list_dir) / "meta.bin").string();
    linkage_list_state.linkage_list_hash_path = app::StoreHashPath(linkage_list_state.linkage_list_dir);
    const std::uint64_t base_list_hash_for_linkage =
        base_list_state.need_base_list ? base_list_state.base_list_hash : run_state.base_list_hash_identity;
    const int linkage_nlist_for_hash =
        ivf_state.need_ivf ? ivf_state.ivf.nlist() : run_state.expected_nlist;
    linkage_list_state.linkage_list_hash =
        app::ComputeLinkageListStoreHash(config, train_result, base_list_hash_for_linkage, linkage_nlist_for_hash);
    linkage_list_state.done_without_linkage_eval = false;
    return 0;
}

int RunLargeLinkageListBuildStage(const stlq::Config& config,
                                const stlq::TrainResult& train_result,
                                const LargeBaseReaderState& base_reader,
                                const LargeRunState& run_state,
                                const LargeBaseBasicStageState& base_basic_state,
                                LargeBaseListStageState& base_list_state,
                                LargeIvfStageState& ivf_state,
                                stlq::StreamKernelProvider* stream_kernels,
                                bool do_eval_linkage,
                                LargeLinkageListStageState& linkage_list_state,
                                std::string& error) {
    using namespace stlq;
    if (config.base.linkage.enabled) {
        if (!base_list_state.need_base_list) {
            LogError("Linkage build requires base_list store.");
            return 1;
        }

        io::LinkageListStoreConfig linkage_store_cfg;
        linkage_store_cfg.dir = linkage_list_state.linkage_list_dir;
        linkage_store_cfg.nlist = ivf_state.ivf.nlist();
        linkage_store_cfg.m_codes = std::max(0, config.model.m - 1);
        // code0_one (C_one[0]) is stored as uint8 in the large-scale pipeline.
        linkage_store_cfg.code0_width_bytes = 1;
        // BaseSet linkage_list: prefer LOUDS-only parent storage by default to save disk (SIFT1B ~4GB).
        linkage_store_cfg.store_parent_u32 = config.large.linkage_store_parent_u32;
        linkage_store_cfg.store_parent_louds = true;
        linkage_store_cfg.enable_checkpoint = (config.large.enabled && config.large.linkage_checkpoint);
        // Always store float coefficients for reference recall — unless the user explicitly opted out
        // via large.base_linkage_store_coeffs_f32=false (requires coeff codec enabled for int8 eval).
        // If large.linkage_coeff_codec.enabled, we will additionally store the compressed coeff codec.
        linkage_store_cfg.store_coeffs_f32 = config.large.base_linkage_store_coeffs_f32;

        LogInfo("Building virtual-mode linkage_list (streaming by cluster)...");
        {
            const bool want_linkage_ckpt = config.large.enabled && config.large.linkage_checkpoint;
            const std::filesystem::path ckpt_done_path =
                std::filesystem::path(linkage_list_state.linkage_list_dir) / "ckpt_linkage_done.u8";
            const bool has_ckpt = want_linkage_ckpt && std::filesystem::exists(ckpt_done_path);

            app::StoreLifecyclePrepareOptions lifecycle_opts;
            lifecycle_opts.label = "linkage_list";
            lifecycle_opts.dir = linkage_list_state.linkage_list_dir;
            lifecycle_opts.protect_existing_outputs = config.large.protect_existing_outputs;
            lifecycle_opts.preserve_existing = has_ckpt;
            lifecycle_opts.verify_hash_when_preserving = has_ckpt;
            lifecycle_opts.hash_path = linkage_list_state.linkage_list_hash_path;
            lifecycle_opts.expected_hash = linkage_list_state.linkage_list_hash;
            lifecycle_opts.hash_mismatch_message =
                "linkage_list store hash mismatch for checkpoint resume (likely incompatible outputs): " +
                linkage_list_state.linkage_list_dir;
            lifecycle_opts.preserve_log_message =
                "Linkage checkpoint enabled: preserving existing linkage_list outputs for resume: " +
                linkage_list_state.linkage_list_dir;
            if (!app::PrepareStoreForRebuildOrResume(lifecycle_opts, &error)) {
                LogError(error);
                return 1;
            }
        }
        Timer linkage_timer;
#if defined(STLQ_ENABLE_CUDA)
        // BaseSet linkage build uses a CUDA ctx pool and can be VRAM-hungry for large h0_root.
        // Force-release base.encode CUDA cached buffers before entering the linkage stage to avoid VRAM stacking.
        if (config.runtime.use_cuda) if (auto* cuda = dynamic_cast<CudaStreamKernels*>(stream_kernels)) {
            cuda->ResetCudaState();
        }
#endif
        // NOTE: CudaStreamKernels is not thread-safe (one cublasHandle + shared scratch buffers).
        // The linkage stages are cluster-parallel (OpenMP) and may call GEMM from multiple threads.
        // For correctness, force CPU stream kernels for linkage build / analysis for now.
        CpuStreamKernels cpu_linkage_kernels;
        StreamKernelProvider* linkage_stream_kernels = stream_kernels;
        if (stream_kernels && stream_kernels->IsGpu()) {
            linkage_stream_kernels = &cpu_linkage_kernels;
            LogInfo("CUDA enabled: using CPU stream kernels for base.linkage build (thread-safety); base.encode stays on GPU.");
        }
        if (!BuildLinkageTwoCodebookVirtualStreamingByCluster(config,
                                                            train_result,
                                                            stlq::VirtualUmapReencodeEncodeConfig{
                                                                .ils_iters = config.base.encode.ils_iters,
                                                                .icm_iters = config.base.encode.icm_iters,
                                                                .perturb_k = config.base.encode.perturb_k,
                                                                .seed = static_cast<std::uint32_t>(config.base.encode.seed),
                                                            },
                                                            base_list_state.base_list,
                                                            ivf_state.ivf,
                                                            &base_reader.reader,
                                                            /*allow_random_fallback=*/true,
                                                            linkage_stream_kernels,
                                                            linkage_store_cfg,
                                                            &linkage_list_state.base_linkage_depth,
                                                            &linkage_list_state.base_linkage_mse_stats,
                                                            &error)) {
            LogError(error);
            return 1;
        }
        if (!app::WriteExpectedStoreHash("linkage_list",
                                         linkage_list_state.linkage_list_hash_path,
                                         linkage_list_state.linkage_list_hash,
                                         &error)) {
            LogError(error);
            return 1;
        }
        if (config.large.linkage_coeff_codec.enabled) {
            const std::filesystem::path coeff_meta =
                std::filesystem::path(linkage_list_state.linkage_list_dir) / "coeff_meta.bin";
            if (std::filesystem::exists(coeff_meta)) {
                const std::string coeff_hash_path = app::CodecHashPath(linkage_list_state.linkage_list_dir);
                const std::uint64_t coeff_hash =
                    app::ComputeLinkageCoeffCodecHash(config, linkage_list_state.linkage_list_hash);
                if (!app::WriteExpectedStoreHash("linkage coeff codec",
                                                 coeff_hash_path,
                                                 coeff_hash,
                                                 &error)) {
                    LogError(error);
                    return 1;
                }
            }
        }
        LogInfo(linkage_timer.ReportSeconds("Linkage streaming finished in"));
        LogInfo("Linkage store dir: " + linkage_list_state.linkage_list_dir);
        LogInfo("BaseSet linkage summary:");
        {
            // In virtual-mode linkage build, virtual nodes are appended per-cluster and stored in linkage_list.
            // Report their total count for storage budgeting and coeff codec "virt theory" comparisons.
            io::LinkageListReader linkage_reader;
            std::string local_err;
            if (!linkage_reader.Open(linkage_list_state.linkage_list_dir, &local_err)) {
                LogWarn("Failed to open linkage_list to query total virtual nodes: " + local_err);
            } else {
                LogInfo("  Total virtual nodes: " + std::to_string(linkage_reader.total_virtual()));
            }
        }
        LogInfo("  Linkage Ratio: " + FormatFloat(100.0f * linkage_list_state.base_linkage_depth.linkage_ratio, 2) + "%");
        LogInfo("  Max depth: " + std::to_string(linkage_list_state.base_linkage_depth.max_depth) +
                " Mean depth: " + FormatFloat(linkage_list_state.base_linkage_depth.mean_depth, 2));
        LogInfo("  Linkage Error: Max=" + FormatFloat(linkage_list_state.base_linkage_mse_stats.max, 4) +
                ", Min=" + FormatFloat(linkage_list_state.base_linkage_mse_stats.min, 4) +
                ", Mean=" + FormatFloat(linkage_list_state.base_linkage_mse_stats.mean, 4));

        // Stage 2 cleanup: after linkage_list (+codec) is ready, apply retention policy.
        CleanupAfterLinkageListReady(config,
                                   std::filesystem::path(base_basic_state.out_dir),
                                   std::filesystem::path(base_list_state.base_list_dir),
                                   std::filesystem::path(linkage_list_state.linkage_list_dir));
    } else if (do_eval_linkage) {
        if (!std::filesystem::exists(linkage_list_state.linkage_list_meta)) {
            LogError("base.linkage.enabled=false but eval.linkage.use_ivf_disk=true and linkage_list store is missing: " +
                     linkage_list_state.linkage_list_meta);
            return 1;
        }
        app::StoreReuseCheckOptions reuse_opts;
        reuse_opts.label = "linkage_list";
        reuse_opts.dir = linkage_list_state.linkage_list_dir;
        reuse_opts.hash_path = linkage_list_state.linkage_list_hash_path;
        reuse_opts.expected_hash = linkage_list_state.linkage_list_hash;
        reuse_opts.eval_only = run_state.eval_only;
        reuse_opts.error_on_non_eval_hash_mismatch = true;
        reuse_opts.allow_missing_hash_reuse = true;
        reuse_opts.non_eval_hash_mismatch_error =
            "linkage_list store hash mismatch (likely stale outputs): " +
            linkage_list_state.linkage_list_dir +
            " ; enable base.linkage to rebuild or change io.pre_fix.";
        reuse_opts.eval_only_hash_mismatch_warning =
            "linkage_list store hash mismatch in eval-only mode; proceeding with existing store: " +
            linkage_list_state.linkage_list_dir;
        app::StoreReuseDecision reuse_decision;
        if (!app::CheckExistingStoreReuse(reuse_opts, &reuse_decision, &error)) {
            LogError(error);
            return 1;
        }
        if (reuse_decision.adopted_previous_hash) {
            linkage_list_state.linkage_list_hash = reuse_decision.effective_hash;
        }
        LogInfo("base.linkage.enabled=false: reusing linkage_list store: " + linkage_list_state.linkage_list_dir);
    } else {
        LogInfo("base.linkage.enabled=false: skipping linkage build and linkage disk recall.");
        linkage_list_state.done_without_linkage_eval = true;
    }
    return 0;
}


}  // namespace stlq
