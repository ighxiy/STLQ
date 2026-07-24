#pragma once

#include "stlq/pipeline/stlq_large_linkage_stages.h"
#include "stlq/common/config.h"
#include "stlq/eval/eval_metrics.h"
#include "stlq/quantizer/encoder.h"

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
                               std::string& error);

}  // namespace stlq
