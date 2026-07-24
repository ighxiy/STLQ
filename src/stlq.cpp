#include "stlq/stlq.h"

#include "stlq/pipeline/app_utils.h"
#include "stlq/pipeline/main_virtual_helpers.h"
#include "stlq/pipeline/stlq_config_bootstrap.h"
#include "stlq/pipeline/stlq_entry_stages.h"
#include "stlq/pipeline/stlq_large_pipeline_flow.h"
#include "stlq/pipeline/stlq_non_large_pipeline_flow.h"
#include "stlq/pipeline/stlq_train_quantizer_flow.h"
#include "stlq/common/logger.h"
#if defined(STLQ_ENABLE_HDF5) && STLQ_ENABLE_HDF5
#include "stlq/io/hdf5_io.h"
#endif

#include <filesystem>
#include <string>

namespace stlq {
namespace {

void FillResult(StlqRunResult* result, int exit_code, const std::string& error, const Config& config) {
    if (!result) {
        return;
    }
    result->exit_code = exit_code;
    result->error = error;
    result->effective_config = config;
}

void ApplyRunModeToConfig(StlqRunMode mode, Config& config) {
    switch (mode) {
    case StlqRunMode::kUseConfig:
        break;
    case StlqRunMode::kFullPipeline:
        config.advanced.eval_only = false;
        config.train.enabled = true;
        config.base.encode.enabled = true;
        config.base.linkage.enabled = true;
        break;
    case StlqRunMode::kTrainOnly:
        config.advanced.eval_only = false;
        config.train.enabled = true;
        config.base.encode.enabled = false;
        config.base.linkage.enabled = false;
        config.eval.base_enabled = false;
        config.eval.linkage_enabled = false;
        break;
    case StlqRunMode::kBuildBaseOnly:
        config.advanced.eval_only = false;
        config.train.enabled = false;
        config.base.encode.enabled = true;
        config.base.linkage.enabled = false;
        config.eval.base_enabled = false;
        config.eval.linkage_enabled = false;
        break;
    case StlqRunMode::kBuildLinkageOnly:
        config.advanced.eval_only = false;
        config.train.enabled = false;
        config.base.encode.enabled = false;
        config.base.linkage.enabled = true;
        config.eval.base_enabled = false;
        config.eval.linkage_enabled = false;
        break;
    case StlqRunMode::kEvalOnly:
        config.advanced.eval_only = true;
        break;
    }
}

int RunPreparedConfig(Config& config,
                      const StlqRunOptions& options,
                      EvalMetricMode eval_metric_mode,
                      StlqRunResult* result,
                      std::string& error) {
    ApplyMainRuntimeStage(config);

    if (!options.dump_config_path.empty()) {
        if (!SaveConfigSnapshot(config, options.dump_config_path, &error)) {
            LogError(error);
            FillResult(result, 1, error, config);
            return 1;
        }
    }

    std::filesystem::path large_root;
    std::filesystem::path large_tmp;
    const int workspace_stage =
        PrepareLargeWorkspaceStage(config, large_root, large_tmp, error);
    if (workspace_stage != kMainStageContinue) {
        FillResult(result, workspace_stage, error, config);
        return workspace_stage;
    }

    KernelRuntimeState kernel_runtime;
    const int kernel_stage = PrepareKernelRuntimeStage(config, kernel_runtime);
    if (kernel_stage != kMainStageContinue) {
        FillResult(result, kernel_stage, error, config);
        return kernel_stage;
    }

    TrainResult train_result;
    const int train_stage = RunTrainOrLoadQuantizerStage(config,
                                                         large_root,
                                                         kernel_runtime.kernels,
                                                         kernel_runtime.stream_kernels,
                                                         train_result,
                                                         error);
    if (train_stage != kMainStageContinue) {
        FillResult(result, train_stage, error, config);
        return train_stage;
    }

    if (config.model.h0_one <= 0 || config.model.h0_one > 256) {
        error = "model.h0_one must be in [1,256] (code0_one is stored as uint8).";
        LogError(error);
        FillResult(result, 1, error, config);
        return 1;
    }

    int exit_code = 0;
    if (config.large.enabled) {
        exit_code = RunLargePipelineFlow(config,
                                         large_root,
                                         large_tmp,
                                         kernel_runtime.stream_kernels,
                                         train_result,
                                         eval_metric_mode,
                                         options.frozen_original_config_text,
                                         options.frozen_original_config_name,
                                         options.linkage_nprobe_batch,
                                         options.linkage_ef_search_batch,
                                         error);
    } else {
        exit_code = RunNonLargePipelineFlow(config,
                                            large_root,
                                            large_tmp,
                                            &kernel_runtime.kernels,
                                            train_result,
                                            eval_metric_mode,
                                            error);
    }

    FillResult(result, exit_code, error, config);
    return exit_code;
}

void NormalizeConfigForDirectRun(Config& config) {
    NormalizeHVec(&config);
    if (config.advanced.eval_only) {
        config.train.enabled = false;
        config.base.encode.enabled = false;
        config.base.linkage.enabled = false;
        LogInfo("advanced.eval_only=true: forcing train.enabled=false, base.encode.enabled=false, base.linkage.enabled=false");
    }
    NormalizeHVec(&config);
    ApplyDatasetDefaults(&config);
    NormalizeIoPrefixWithM(&config);

#if defined(STLQ_ENABLE_HDF5) && STLQ_ENABLE_HDF5
    io::SetDefaultHdf5MatrixLayout(config.io.hdf5_layout == "cxx"
                                       ? io::Hdf5MatrixLayout::kCxx
                                       : io::Hdf5MatrixLayout::kJulia);
#endif

    LogInfo("Config: dataset.name=" + config.dataset.name);
    LogInfo("Config: io.train_file=" + config.io.train_file);
    LogInfo("Config: io.base_file=" + config.io.base_file);
    LogInfo("Config: io.linkage_file=" + config.io.linkage_file);
    LogInfo("Config: io.load_date=\"" + config.io.load_date +
            "\" io.load_seq=\"" + config.io.load_seq + "\"");
    LogInfo("Config: large.enabled=" + std::string(config.large.enabled ? "true" : "false"));
}

}  // namespace

int RunStlq(int argc, char** argv, StlqRunResult* result) {
    std::string error;
    MainConfigBootstrapState bootstrap;
    const int config_stage = PrepareMainConfigBootstrapStage(argc, argv, bootstrap, error);
    if (config_stage != kMainStageContinue) {
        FillResult(result, config_stage, error, bootstrap.config);
        return config_stage;
    }

    StlqRunOptions options;
    options.dump_config_path = bootstrap.dump_path;
    options.frozen_original_config_text = bootstrap.frozen_original_cfg;
    options.frozen_original_config_name = bootstrap.frozen_original_cfg_name;
    options.linkage_nprobe_batch = bootstrap.linkage_nprobe_batch_cfg;
    options.linkage_ef_search_batch = bootstrap.linkage_ef_search_batch_cfg;

    Config& config = bootstrap.config;
    const int exit_code = RunPreparedConfig(config, options, bootstrap.eval_metric_mode, result, error);
    if (result) {
        result->effective_config = config;
    }
    return exit_code;
}

int RunStlq(Config config, const StlqRunOptions& options, StlqRunResult* result) {
    std::string error;
    app::InstallCrashTraceIfEnabled();
    EvalMetricMode eval_metric_mode = EvalMetricMode::kTop1Recall;
    if (!ParseEvalMetricMode(config.eval.metric_mode, &eval_metric_mode)) {
        error = "Invalid eval.metric_mode=" + std::to_string(config.eval.metric_mode) +
                " (expected 0, 1, or 2).";
        LogError(error);
        FillResult(result, 1, error, config);
        return 1;
    }

    NormalizeConfigForDirectRun(config);
    return RunPreparedConfig(config, options, eval_metric_mode, result, error);
}

int RunStlq(StlqRunRequest request, StlqRunResult* result) {
    ApplyRunModeToConfig(request.mode, request.config);
    return RunStlq(request.config, request.options, result);
}

}  // namespace stlq
