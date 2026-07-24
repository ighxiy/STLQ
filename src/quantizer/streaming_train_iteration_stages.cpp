#include "stlq/quantizer/streaming_train_iteration_stages.h"

#include <algorithm>

#include "stlq/io/result_io.h"
#include "stlq/common/logger.h"
#include "stlq/quantizer/streaming_train_init_stages.h"
#include "stlq/quantizer/streaming_train_round_stages.h"

namespace stlq {

StreamingTrainIterationPlan MakeStreamingTrainIterationPlan(const Config& config,
                                                            std::uint64_t ntrain,
                                                            int resume_base_R_iters) {
    StreamingTrainIterationPlan plan;
    plan.resume_base_R_iters = resume_base_R_iters;
    plan.additional_iters = std::max(0, config.train.max_R_iters);
    plan.display_total_iters = resume_base_R_iters + plan.additional_iters;
    plan.final_R_iters = resume_base_R_iters + plan.additional_iters;
    plan.allow_random_fallback = (!config.large.write_vector_bucket && ntrain <= 1000000ull);
    return plan;
}

bool RunStreamingTrainRoundStage(const Config& config,
                                 TrainResult& result,
                                 const StreamingTrainWorkspacePaths& paths,
                                 const StreamingTrainIterationPlan& plan,
                                 int global_iter,
                                 bool do_init_linkage_and_c1,
                                 bool do_opq_update,
                                 int d,
                                 int nlist,
                                 std::uint64_t ntrain,
                                 const StreamingTrainInput& train_input,
                                 StreamKernelProvider* stream_kernels,
                                 bool* logged_linkage_stage_cpu_notice,
                                 bool* early_exit_after_ckpt_init_basic,
                                 std::string* error) {
    const bool is_global_round = (global_iter >= 0);
    if (is_global_round) {
        LogInfo("============== Global R iteration " + std::to_string(global_iter + 1) + " / " +
                std::to_string(plan.display_total_iters) + " ==============");
    }

    if (!RunStreamingTrainBasicEncodeStage(config,
                                           result,
                                           paths,
                                           ntrain,
                                           d,
                                           is_global_round,
                                           stream_kernels,
                                           error)) {
        return false;
    }

    if (!RunStreamingTrainUpdateCRootStage(config,
                                           result,
                                           paths,
                                           stream_kernels,
                                           error)) {
        return false;
    }

    io::IvfListsReader ivf;
    io::BaseListReader train_list;
    if (!BuildStreamingTrainIvfAndList(config, paths, nlist, &ivf, &train_list, error)) {
        return false;
    }

    StreamingTrainLinkageRoundContext linkage_context;
    PrepareStreamingTrainLinkageRoundContext(config,
                                           paths,
                                           ivf,
                                           train_input,
                                           plan.allow_random_fallback,
                                           stream_kernels,
                                           logged_linkage_stage_cpu_notice,
                                           &linkage_context);

    // Prompt 11 init stage (baseline only): build init linkage_list using only C_root, then
    // compute bad cluster mask and do the first exact-LS update for C_one.
    if (do_init_linkage_and_c1) {
        if (config.train.ckpt_after_init_basic) {
            result.R_iters = plan.resume_base_R_iters;
            std::string local_err;
            const std::string ckpt_path = io::MakeUniqueDatedPath(config.io.train_file, &local_err);
            if (ckpt_path.empty()) {
                if (error) *error = "Pre-init-linkage checkpoint path alloc failed: " + local_err;
                return false;
            }
            LogInfo("Saving pre-init-linkage checkpoint to: " + ckpt_path + " (C_root+R only)");
            if (!io::SaveTrainResultsPreInitLinkage(ckpt_path, config, result, paths.exp_root_run.string(), &local_err)) {
                if (error) *error = "Failed to save pre-init-linkage checkpoint: " + local_err;
                return false;
            }
            if (config.train.exit_after_ckpt_init_basic) {
                LogInfo("Train: exit_after_ckpt_init_basic=1 (stop before init-linkage).");
                result.checkpoint_stage = 1;
                if (early_exit_after_ckpt_init_basic) {
                    *early_exit_after_ckpt_init_basic = true;
                }
                return true;
            }
        }
        if (!RunStreamingTrainInitLinkageAndUpdateC1(config,
                                                   result,
                                                   paths,
                                                   train_list,
                                                   ivf,
                                                   stream_kernels,
                                                   linkage_context.linkage_stream_kernels,
                                                   linkage_context.fallback_reader,
                                                   linkage_context.allow_random_fallback,
                                                   linkage_context.linkage_store_cfg,
                                                   linkage_context.linkage_cfg,
                                                   error)) {
            return false;
        }
    }
    if (is_global_round) {
        if (!RunStreamingTrainIterationTailStage(config,
                                                 result,
                                                 paths,
                                                 global_iter,
                                                 do_opq_update,
                                                 train_list,
                                                 ivf,
                                                 stream_kernels,
                                                 linkage_context.linkage_stream_kernels,
                                                 linkage_context.fallback_reader,
                                                 linkage_context.allow_random_fallback,
                                                 linkage_context.linkage_store_cfg,
                                                 linkage_context.linkage_cfg,
                                                 error)) {
            return false;
        }
    }
    return true;
}

bool RunStreamingTrainInitCheckpointStage(const Config& config,
                                          TrainResult& result,
                                          const StreamingTrainIterationPlan& plan,
                                          std::string* error) {
    // Optional: write an init checkpoint at R_iters=0 so that init results are resumable even if the run
    // stops before the first periodic R checkpoint (e.g. every_R=5).
    // This does not change training semantics/results.
    if (config.train.ckpt.enabled && config.train.ckpt.every_R > 0) {
        // Avoid writing two identical checkpoints when there are no R iterations and the caller
        // will also write the final train H5 via io.save_train.
        if (!(config.io.save_train && plan.final_R_iters == plan.resume_base_R_iters)) {
            result.R_iters = plan.resume_base_R_iters;  // init stage completes with R_iters unchanged
            std::string local_err;
            const std::string ckpt_path = io::MakeUniqueDatedPath(config.io.train_file, &local_err);
            if (ckpt_path.empty()) {
                if (error) *error = "Init checkpoint path alloc failed: " + local_err;
                return false;
            }
            LogInfo("Saving init checkpoint to: " + ckpt_path + " (R_iters=" +
                    std::to_string(result.R_iters) + ")");
            if (!io::SaveTrainResults(ckpt_path, config, result, &local_err)) {
                if (error) *error = "Failed to save init checkpoint: " + local_err;
                return false;
            }
        }
    }
    return true;
}

bool RunStreamingTrainResumePreInitLinkageStage(const Config& config,
                                              TrainResult& result,
                                              const StreamingTrainWorkspacePaths& paths,
                                              const StreamingTrainIterationPlan& plan,
                                              const StreamingTrainInput& train_input,
                                              StreamKernelProvider* stream_kernels,
                                              bool* logged_linkage_stage_cpu_notice,
                                              std::string* error) {
    io::IvfListsReader ivf;
    io::BaseListReader train_list;
    if (!OpenStreamingTrainIvfAndList(paths, &ivf, &train_list, error)) {
        return false;
    }

    StreamingTrainLinkageRoundContext linkage_context;
    PrepareStreamingTrainLinkageRoundContext(config,
                                           paths,
                                           ivf,
                                           train_input,
                                           plan.allow_random_fallback,
                                           stream_kernels,
                                           logged_linkage_stage_cpu_notice,
                                           &linkage_context);

    if (!RunStreamingTrainInitLinkageAndUpdateC1(config,
                                               result,
                                               paths,
                                               train_list,
                                               ivf,
                                               stream_kernels,
                                               linkage_context.linkage_stream_kernels,
                                               linkage_context.fallback_reader,
                                               linkage_context.allow_random_fallback,
                                               linkage_context.linkage_store_cfg,
                                               linkage_context.linkage_cfg,
                                               error)) {
        return false;
    }
    // The loaded checkpoint was "pre-init-linkage" (C_root+R only). After init-linkage, we now have a full
    // in-memory training state (including C_one), so clear the stage marker.
    result.checkpoint_stage = 0;
    return true;
}

bool RunStreamingTrainGlobalIterationLoopStage(const Config& config,
                                               TrainResult& result,
                                               const StreamingTrainWorkspacePaths& paths,
                                               const StreamingTrainIterationPlan& plan,
                                               int d,
                                               int nlist,
                                               std::uint64_t ntrain,
                                               const StreamingTrainInput& train_input,
                                               StreamKernelProvider* stream_kernels,
                                               bool* logged_linkage_stage_cpu_notice,
                                               std::string* error) {
    for (int iter = 0; iter < plan.additional_iters; ++iter) {
        if (!RunStreamingTrainRoundStage(config,
                                         result,
                                         paths,
                                         plan,
                                         /*global_iter=*/plan.resume_base_R_iters + iter,
                                         /*do_init_linkage_and_c1=*/false,
                                         /*do_opq_update=*/true,
                                         d,
                                         nlist,
                                         ntrain,
                                         train_input,
                                         stream_kernels,
                                         logged_linkage_stage_cpu_notice,
                                         /*early_exit_after_ckpt_init_basic=*/nullptr,
                                         error)) {
            return false;
        }

        // Keep R-iteration progress up-to-date so that periodic checkpoints record the correct value.
        result.R_iters = plan.resume_base_R_iters + iter + 1;

        if (config.train.ckpt.enabled && config.train.ckpt.every_R > 0) {
            const int cur_R_iters = result.R_iters;
            if (cur_R_iters > 0 && (cur_R_iters % std::max(1, config.train.ckpt.every_R) == 0)) {
                // Avoid writing two identical "final" checkpoints when:
                // - periodic ckpt triggers at the last iteration, and
                // - the caller will also write the final train H5 via io.save_train.
                if (config.io.save_train && cur_R_iters == plan.final_R_iters) {
                    continue;
                }
                std::string local_err;
                const std::string ckpt_path = io::MakeUniqueDatedPath(config.io.train_file, &local_err);
                if (ckpt_path.empty()) {
                    if (error) *error = "Train checkpoint path alloc failed: " + local_err;
                    return false;
                }
                LogInfo("Saving train checkpoint to: " + ckpt_path + " (R_iters=" + std::to_string(cur_R_iters) + ")");
                if (!io::SaveTrainResults(ckpt_path, config, result, &local_err)) {
                    if (error) *error = "Failed to save train checkpoint: " + local_err;
                    return false;
                }
            }
        }
    }
    result.R_iters = plan.resume_base_R_iters + plan.additional_iters;
    return true;
}

}  // namespace stlq
