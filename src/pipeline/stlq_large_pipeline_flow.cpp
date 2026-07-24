#include "stlq/pipeline/stlq_large_pipeline_flow.h"

#include "stlq/pipeline/app_utils.h"
#include "stlq/pipeline/stlq_large_base_flow.h"
#include "stlq/pipeline/stlq_large_linkage_stages.h"
#include "stlq/pipeline/stlq_large_eval_stages.h"

int RunLargePipelineFlow(stlq::Config& config,
                         const std::filesystem::path& large_root,
                         const std::filesystem::path& large_tmp,
                         stlq::StreamKernelProvider* stream_kernels,
                         stlq::TrainResult& train_result,
                         stlq::EvalMetricMode eval_metric_mode,
                         const std::string& frozen_original_cfg,
                         const std::filesystem::path& frozen_original_cfg_name,
                         const std::vector<int>& linkage_nprobe_batch_cfg,
                         const std::vector<int>& linkage_ef_search_batch_cfg,
                         std::string& error) {
    using namespace stlq;

    LargeBaseFlowState base_flow_state;
    const int base_flow_stage = RunLargeBaseFlowStage(config,
                                                      train_result,
                                                      eval_metric_mode,
                                                      stream_kernels,
                                                      frozen_original_cfg,
                                                      frozen_original_cfg_name,
                                                      base_flow_state,
                                                      error);
    if (base_flow_stage != 0) {
        return base_flow_stage;
    }

    LargeLinkageListStageState linkage_list_state;
    const int linkage_list_state_stage = PrepareLargeLinkageListStateStage(config,
                                                                       train_result,
                                                                       base_flow_state.run_state,
                                                                       base_flow_state.ivf_state,
                                                                       base_flow_state.base_list_state,
                                                                       linkage_list_state);
    if (linkage_list_state_stage != 0) {
        return linkage_list_state_stage;
    }

    const int linkage_list_stage = RunLargeLinkageListBuildStage(config,
                                                             train_result,
                                                             base_flow_state.base_reader,
                                                             base_flow_state.run_state,
                                                             base_flow_state.base_basic_state,
                                                             base_flow_state.base_list_state,
                                                             base_flow_state.ivf_state,
                                                             stream_kernels,
                                                             base_flow_state.do_eval_linkage,
                                                             linkage_list_state,
                                                             error);
    if (linkage_list_stage != 0) {
        return linkage_list_stage;
    }
    if (linkage_list_state.done_without_linkage_eval) {
        return 0;
    }

    if (base_flow_state.do_eval_linkage) {
        const int linkage_eval_stage = RunLargeLinkageDiskEvalStage(config,
                                                                train_result,
                                                                eval_metric_mode,
                                                                base_flow_state.run_state,
                                                                base_flow_state.eval_query_state,
                                                                linkage_list_state,
                                                                linkage_nprobe_batch_cfg,
                                                                linkage_ef_search_batch_cfg,
                                                                error);
        if (linkage_eval_stage != 0) {
            return linkage_eval_stage;
        }
    }

    CleanupEmptyLargeWorkspaceDirs(large_root, large_tmp);
    return 0;
}
