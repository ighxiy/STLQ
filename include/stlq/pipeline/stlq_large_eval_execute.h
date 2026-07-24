#pragma once

#include "stlq/pipeline/main_virtual_helpers.h"
#include "stlq/pipeline/stlq_large_base_stages.h"
#include "stlq/pipeline/stlq_large_eval_plan.h"
#include "stlq/io/linkage_list_store.h"

#include <string>

namespace stlq {

void RunLargeLinkageDiskEvalBatchStage(const Config& config,
                                     const TrainResult& train_result,
                                     EvalMetricMode eval_metric_mode,
                                     const LargeRunState& run_state,
                                     const LargeDiskEvalQueryState& eval_query_state,
                                     io::LinkageListReader& linkage_list,
                                     const app::LinkageLOUDSIndexBitBudget& louds_index_budget,
                                     const LargeLinkageProbeEvalPlan& probe_eval_plan,
                                     bool use_coeff_codec,
                                     const std::string& label,
                                     const std::string& archive_label,
                                     std::string& error);

}  // namespace stlq
