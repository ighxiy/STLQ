#pragma once

#include "stlq/pipeline/stlq_large_base_stages.h"
#include "stlq/common/config.h"
#include "stlq/eval/eval_metrics.h"
#include "stlq/quantizer/encoder.h"

#include <filesystem>
#include <string>

namespace stlq {

class StreamKernelProvider;

struct LargeBaseFlowState {
    LargeBaseReaderState base_reader;
    LargeRunState run_state;
    LargeBaseBasicStageState base_basic_state;
    LargeIvfStageState ivf_state;
    LargeBaseListStageState base_list_state;
    LargeDiskEvalQueryState eval_query_state;
    bool do_eval_base = false;
    bool do_eval_linkage = false;
};

int RunLargeBaseFlowStage(Config& config,
                          TrainResult& train_result,
                          EvalMetricMode eval_metric_mode,
                          StreamKernelProvider* stream_kernels,
                          const std::string& frozen_original_cfg,
                          const std::filesystem::path& frozen_original_cfg_name,
                          LargeBaseFlowState& base_flow_state,
                          std::string& error);

}  // namespace stlq
