#include "stlq/pipeline/stlq_large_base_flow.h"

namespace stlq {

int RunLargeBaseFlowStage(Config& config,
                          TrainResult& train_result,
                          EvalMetricMode eval_metric_mode,
                          StreamKernelProvider* stream_kernels,
                          const std::string& frozen_original_cfg,
                          const std::filesystem::path& frozen_original_cfg_name,
                          LargeBaseFlowState& base_flow_state,
                          std::string& error) {
    Precomp pre_root;
    bool pre_root_ready = false;

    const int base_reader_stage = OpenLargeBaseReaderStage(config, base_flow_state.base_reader, error);
    if (base_reader_stage != 0) {
        return base_reader_stage;
    }

    const int run_state_stage =
        PrepareLargeRunStateStage(config, train_result, base_flow_state.base_reader, base_flow_state.run_state);
    if (run_state_stage != 0) {
        return run_state_stage;
    }

    const int run_artifacts_stage = WriteLargeRunArtifactsStage(config,
                                                                train_result,
                                                                base_flow_state.base_reader,
                                                                base_flow_state.run_state,
                                                                frozen_original_cfg,
                                                                frozen_original_cfg_name,
                                                                error);
    if (run_artifacts_stage != 0) {
        return run_artifacts_stage;
    }

    const int guardrails_stage = ValidateLargeRunGuardrailsStage(config);
    if (guardrails_stage != 0) {
        return guardrails_stage;
    }

    PrecompLargeRoot pre_lr;
    PrecompLargeRoot* pre_lr_ptr = nullptr;
    const int precomp_stage =
        BuildLargePrecompRootStage(config, train_result, stream_kernels, pre_lr, pre_lr_ptr, error);
    if (precomp_stage != 0) {
        return precomp_stage;
    }

    const int base_basic_stage = RunLargeBaseBasicStage(config,
                                                        train_result,
                                                        base_flow_state.base_reader,
                                                        base_flow_state.run_state,
                                                        pre_root,
                                                        pre_root_ready,
                                                        pre_lr_ptr,
                                                        stream_kernels,
                                                        base_flow_state.base_basic_state,
                                                        error);
    if (base_basic_stage != 0) {
        return base_basic_stage;
    }

    base_flow_state.do_eval_base = (config.eval.base_enabled && config.eval.base_use_ivf);
    base_flow_state.do_eval_linkage = (config.eval.linkage_enabled && config.eval.linkage_use_ivf_disk);

    const int ivf_stage = RunLargeIvfStage(config,
                                           base_flow_state.run_state,
                                           base_flow_state.base_basic_state,
                                           base_flow_state.do_eval_base,
                                           base_flow_state.ivf_state,
                                           error);
    if (ivf_stage != 0) {
        return ivf_stage;
    }

    const int base_list_stage = RunLargeBaseListStage(config,
                                                      base_flow_state.run_state,
                                                      base_flow_state.base_basic_state,
                                                      base_flow_state.ivf_state,
                                                      base_flow_state.do_eval_base,
                                                      base_flow_state.base_list_state,
                                                      error);
    if (base_list_stage != 0) {
        return base_list_stage;
    }

    const int base_list_cleanup_stage =
        RunLargeBaseListCleanupStage(config, base_flow_state.base_basic_state, base_flow_state.base_list_state);
    if (base_list_cleanup_stage != 0) {
        return base_list_cleanup_stage;
    }

    const int eval_query_stage = PrepareLargeDiskEvalQueriesStage(config,
                                                                  train_result,
                                                                  eval_metric_mode,
                                                                  base_flow_state.do_eval_base,
                                                                  base_flow_state.do_eval_linkage,
                                                                  base_flow_state.eval_query_state,
                                                                  error);
    if (eval_query_stage != 0) {
        return eval_query_stage;
    }

    if (base_flow_state.do_eval_base) {
        const int base_eval_stage = RunLargeBaseDiskEvalStage(config,
                                                              train_result,
                                                              eval_metric_mode,
                                                              base_flow_state.ivf_state.ivf,
                                                              base_flow_state.base_list_state.base_list,
                                                              base_flow_state.eval_query_state,
                                                              error);
        if (base_eval_stage != 0) {
            return base_eval_stage;
        }
    }

    return 0;
}

}  // namespace stlq
