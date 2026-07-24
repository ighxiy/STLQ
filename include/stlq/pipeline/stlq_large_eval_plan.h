#pragma once

#include "stlq/common/config.h"
#include "stlq/io/linkage_list_store.h"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace stlq {

struct LargeLinkageCoeffEvalPlan {
    std::filesystem::path coeff_meta;
    std::string mode;
    std::string float_coeffs_why_not;
    bool has_float_coeffs = false;
    bool has_coeff_codec = false;
    bool run_float = false;
    bool run_int8 = false;
};

struct LargeLinkageCoeffEvalRun {
    bool use_coeff_codec = false;
    std::string label;
    std::string archive_label;
};

struct LargeLinkageProbeEvalCase {
    int nprobe = 1;
    int ef_search = 1;
};

struct LargeLinkageProbeEvalPlan {
    std::vector<int> linkage_nprobe_batch;
    std::vector<int> linkage_ef_search_batch;
    std::vector<LargeLinkageProbeEvalCase> cases;
    bool hnsw_probe_mode = false;
    bool reuse_provider_across_nprobes = false;
    std::size_t linkage_eval_combo_count = 0;
};

struct LargeLinkageDiskEvalPlan {
    LargeLinkageCoeffEvalPlan coeff;
    LargeLinkageProbeEvalPlan probe;
    std::vector<LargeLinkageCoeffEvalRun> coeff_runs;
};

int PrepareLargeLinkageCoeffEvalPlanStage(const Config& config,
                                        const std::string& linkage_list_dir,
                                        io::LinkageListReader& linkage_list,
                                        LargeLinkageCoeffEvalPlan& coeff_eval_plan,
                                        std::string& error);

LargeLinkageProbeEvalPlan PrepareLargeLinkageProbeEvalPlanStage(
    const Config& config,
    const std::vector<int>& linkage_nprobe_batch_cfg,
    const std::vector<int>& linkage_ef_search_batch_cfg);

int PrepareLargeLinkageDiskEvalPlanStage(const Config& config,
                                       const std::string& linkage_list_dir,
                                       io::LinkageListReader& linkage_list,
                                       const std::vector<int>& linkage_nprobe_batch_cfg,
                                       const std::vector<int>& linkage_ef_search_batch_cfg,
                                       LargeLinkageDiskEvalPlan* plan,
                                       std::string& error);

void LogLargeLinkageProbeEvalPlan(const LargeLinkageProbeEvalPlan& plan);

}  // namespace stlq
