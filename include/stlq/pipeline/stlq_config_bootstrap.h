#pragma once

#include "stlq/common/config.h"
#include "stlq/eval/eval_metrics.h"

#include <filesystem>
#include <string>
#include <vector>

namespace stlq {

struct MainConfigBootstrapState {
    Config config = DefaultConfig(true);
    EvalMetricMode eval_metric_mode = EvalMetricMode::kTop1Recall;
    std::vector<int> linkage_nprobe_batch_cfg;
    std::vector<int> linkage_ef_search_batch_cfg;
    std::string dump_path;
    std::string frozen_original_cfg;
    std::filesystem::path frozen_original_cfg_name;
};

int PrepareMainConfigBootstrapStage(int argc,
                                    char** argv,
                                    MainConfigBootstrapState& bootstrap,
                                    std::string& error);

}  // namespace stlq
