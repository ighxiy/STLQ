#pragma once

#include "stlq/common/config.h"

#include <filesystem>
#include <string>
#include <vector>

namespace stlq {

struct StlqRunOptions {
    std::string dump_config_path;
    std::string frozen_original_config_text;
    std::filesystem::path frozen_original_config_name;
    std::vector<int> linkage_nprobe_batch;
    std::vector<int> linkage_ef_search_batch;
};

struct StlqRunResult {
    int exit_code = 0;
    std::string error;
    Config effective_config;
};

enum class StlqRunMode {
    // Preserve the stage switches already present in Config.
    kUseConfig = 0,
    // Enable train, base encode, and linkage build. Eval switches remain Config-owned.
    kFullPipeline,
    // Train/save/load only, then stop before base/linkage/eval stages.
    kTrainOnly,
    // Load existing train result and build base artifacts only.
    kBuildBaseOnly,
    // Load existing train/base artifacts and build linkage artifacts only.
    kBuildLinkageOnly,
    // Force eval-only semantics, matching advanced.eval_only=true.
    kEvalOnly,
};

struct StlqRunRequest {
    Config config;
    StlqRunMode mode = StlqRunMode::kUseConfig;
    StlqRunOptions options;
};

// Library facade matching the command-line program behavior.
// Parses argv using the same --config / --set / --dump-config contract as stlq_main.
int RunStlq(int argc, char** argv, StlqRunResult* result = nullptr);

// Library facade for callers that already own a Config object.
// The config is copied intentionally: the returned effective_config records normalized defaults
// and eval-only overrides without mutating the caller's object.
int RunStlq(Config config,
            const StlqRunOptions& options = StlqRunOptions{},
            StlqRunResult* result = nullptr);

// Library facade for callers that want a stage-oriented request without manually
// editing train/base/linkage/eval switches in Config.
int RunStlq(StlqRunRequest request, StlqRunResult* result = nullptr);

}  // namespace stlq
