#pragma once

#include "stlq/common/config.h"
#include "stlq/core/kernels.h"
#include "stlq/eval/eval_metrics.h"
#include "stlq/quantizer/encoder.h"

#include <filesystem>
#include <string>

namespace stlq {

int RunNonLargePipelineFlow(Config& config,
                            const std::filesystem::path& large_root,
                            const std::filesystem::path& large_tmp,
                            KernelProvider* kernels,
                            TrainResult& train_result,
                            EvalMetricMode eval_metric_mode,
                            std::string& error);

}  // namespace stlq
