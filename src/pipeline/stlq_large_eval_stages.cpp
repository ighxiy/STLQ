#include "stlq/pipeline/stlq_large_eval_stages.h"

#include "stlq/pipeline/large_store_hash.h"
#include "stlq/pipeline/large_store_lifecycle.h"
#include "stlq/pipeline/main_virtual_helpers.h"
#include "stlq/pipeline/stlq_large_eval_execute.h"
#include "stlq/pipeline/stlq_large_eval_plan.h"
#include "stlq/linkage/linkage_streaming_builders.h"
#include "stlq/io/linkage_list_store.h"
#include "stlq/common/logger.h"

#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

namespace stlq {

int RunLargeLinkageDiskEvalStage(const Config& config,
                               const TrainResult& train_result,
                               EvalMetricMode eval_metric_mode,
                               const LargeRunState& run_state,
                               const LargeDiskEvalQueryState& eval_query_state,
                               const LargeLinkageListStageState& linkage_list_state,
                               const std::vector<int>& linkage_nprobe_batch_cfg,
                               const std::vector<int>& linkage_ef_search_batch_cfg,
                               std::string& error) {
    const std::string& linkage_list_dir = linkage_list_state.linkage_list_dir;
    const std::uint64_t linkage_list_hash = linkage_list_state.linkage_list_hash;
    io::LinkageListReader linkage_list;
    if (!linkage_list.Open(linkage_list_dir, &error)) {
        LogError(error);
        return 1;
    }

    LargeLinkageDiskEvalPlan eval_plan;
    const int eval_plan_stage =
        PrepareLargeLinkageDiskEvalPlanStage(config,
                                           linkage_list_dir,
                                           linkage_list,
                                           linkage_nprobe_batch_cfg,
                                           linkage_ef_search_batch_cfg,
                                           &eval_plan,
                                           error);
    if (eval_plan_stage != 0) {
        return eval_plan_stage;
    }

    // Report bit budgets before entering eval paths. If coeff codec gets rebuilt below,
    // we will log the refreshed budget again after rebuild.
    const app::LinkageLOUDSIndexBitBudget louds_index_budget = app::LogLinkageBitBudgetSummary(config, linkage_list);
    LogLargeLinkageProbeEvalPlan(eval_plan.probe);

    for (const LargeLinkageCoeffEvalRun& run : eval_plan.coeff_runs) {
        if (run.use_coeff_codec) {
            if (!config.large.linkage_coeff_codec.enabled) {
                LogWarn("eval.linkage.coeff_mode=int8/both but large.linkage_coeff_codec.enabled=false; proceeding with int8 recall anyway (eval requested).");
            }

            const std::filesystem::path& coeff_meta = eval_plan.coeff.coeff_meta;
            const std::string coeff_hash_path = app::CodecHashPath(linkage_list_dir);
            const std::uint64_t expected_coeff_hash = app::ComputeLinkageCoeffCodecHash(config, linkage_list_hash);

            app::CodecStoreLifecycleOptions codec_opts;
            codec_opts.label = "coeff codec";
            codec_opts.store_dir = linkage_list_dir;
            codec_opts.codec_meta_path = coeff_meta.string();
            codec_opts.hash_path = coeff_hash_path;
            codec_opts.expected_hash = expected_coeff_hash;
            codec_opts.has_codec_store = eval_plan.coeff.has_coeff_codec;
            codec_opts.protect_existing_outputs = config.large.protect_existing_outputs;
            codec_opts.has_rebuild_source = eval_plan.coeff.has_float_coeffs;
            codec_opts.rebuild_source_why_not = eval_plan.coeff.float_coeffs_why_not;

            app::CodecStoreLifecycleDecision codec_decision;
            if (!app::PrepareCodecStoreForEval(codec_opts, &codec_decision, &error)) {
                LogError(error);
                return 1;
            }

            bool coeff_codec_rebuilt = false;
            if (codec_decision.needs_rebuild) {
                LogInfo("Rebuilding linkage coeff codec from float linkage_list (no linkage rebuild)...");
                if (!RebuildLinkageCoeffCodecFromLinkageListVirtual(config, train_result, linkage_list, &error)) {
                    LogWarn(error);
                } else {
                    coeff_codec_rebuilt = true;
                    if (!app::WriteExpectedStoreHash("linkage coeff codec",
                                                     coeff_hash_path,
                                                     expected_coeff_hash,
                                                     &error)) {
                        LogWarn(error);
                    }
                }
            }

            if (coeff_codec_rebuilt) {
                app::LogLinkageBitBudgetSummary(config, linkage_list);
            }

            const bool has_coeff_codec_now = std::filesystem::exists(coeff_meta);
            if (!has_coeff_codec_now) {
                LogWarn("eval.linkage.coeff_mode=int8/both but coeff codec store is missing: " +
                        coeff_meta.string());
                return 1;
            }
        }

        RunLargeLinkageDiskEvalBatchStage(config,
                                        train_result,
                                        eval_metric_mode,
                                        run_state,
                                        eval_query_state,
                                        linkage_list,
                                        louds_index_budget,
                                        eval_plan.probe,
                                        run.use_coeff_codec,
                                        run.label,
                                        run.archive_label,
                                        error);
    }
    return 0;
}

}  // namespace stlq
