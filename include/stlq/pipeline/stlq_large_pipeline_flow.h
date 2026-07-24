#pragma once

#include "stlq/common/config.h"
#include "stlq/eval/eval_metrics.h"
#include "stlq/quantizer/encoder.h"

#include <filesystem>
#include <string>
#include <vector>

namespace stlq {
class StreamKernelProvider;
}  // namespace stlq

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
                         std::string& error);
