#pragma once

#include "stlq/pipeline/stlq_large_base_stages.h"
#include "stlq/linkage/linkage_summary.h"
#include "stlq/common/config.h"
#include "stlq/core/kernel_provider_cpu.h"

#include <string>

namespace stlq {

struct LargeLinkageListStageState {
    std::string linkage_list_dir;
    std::string linkage_list_meta;
    std::string linkage_list_hash_path;
    std::uint64_t linkage_list_hash = 0;
    LinkageDepthStats base_linkage_depth;
    MseStats base_linkage_mse_stats;
    bool done_without_linkage_eval = false;
};

int PrepareLargeLinkageListStateStage(const Config& config, const TrainResult& train_result, const LargeRunState& run_state, const LargeIvfStageState& ivf_state, const LargeBaseListStageState& base_list_state, LargeLinkageListStageState& linkage_list_state);
int RunLargeLinkageListBuildStage(const Config& config, const TrainResult& train_result, const LargeBaseReaderState& base_reader, const LargeRunState& run_state, const LargeBaseBasicStageState& base_basic_state, LargeBaseListStageState& base_list_state, LargeIvfStageState& ivf_state, StreamKernelProvider* stream_kernels, bool do_eval_linkage, LargeLinkageListStageState& linkage_list_state, std::string& error);

}  // namespace stlq
