#include "stlq/pipeline/stlq_config_bootstrap.h"

#include "stlq/pipeline/app_utils.h"
#include "stlq/pipeline/main_virtual_helpers.h"
#include "stlq/pipeline/stlq_entry_stages.h"
#if defined(STLQ_ENABLE_HDF5) && STLQ_ENABLE_HDF5
#include "stlq/io/hdf5_io.h"
#endif
#include "stlq/common/logger.h"

#include <filesystem>
#include <fstream>
#include <sstream>

namespace stlq {

int PrepareMainConfigBootstrapStage(int argc,
                                    char** argv,
                                    MainConfigBootstrapState& bootstrap,
                                    std::string& error) {
    bootstrap = MainConfigBootstrapState{};
    NormalizeHVec(&bootstrap.config);

    app::InstallCrashTraceIfEnabled();

    if (!ParseArgs(argc,
                   argv,
                   &bootstrap.config,
                   &error,
                   &bootstrap.dump_path,
                   &bootstrap.linkage_nprobe_batch_cfg,
                   &bootstrap.linkage_ef_search_batch_cfg)) {
        LogError(error);
        return ReturnFromMainStage(1);
    }
    if (!ParseEvalMetricMode(bootstrap.config.eval.metric_mode, &bootstrap.eval_metric_mode)) {
        LogError("Invalid eval.metric_mode=" + std::to_string(bootstrap.config.eval.metric_mode) +
                 " (expected 0, 1, or 2).");
        return ReturnFromMainStage(1);
    }
    if (!bootstrap.config.io.config_file.empty()) {
        const std::filesystem::path cfg_path(bootstrap.config.io.config_file);
        std::ifstream in(cfg_path, std::ios::binary);
        if (!in.is_open()) {
            LogWarn("Failed to open original config file for early backup freeze: " + cfg_path.string());
        } else {
            std::ostringstream oss;
            oss << in.rdbuf();
            if (!in.bad()) {
                bootstrap.frozen_original_cfg = oss.str();
                bootstrap.frozen_original_cfg_name =
                    cfg_path.filename().empty() ? std::filesystem::path("original_config.cfg")
                                                : cfg_path.filename();
            } else {
                LogWarn("Failed to read original config file for early backup freeze: " + cfg_path.string());
            }
        }
    }
    if (bootstrap.config.advanced.eval_only) {
        bootstrap.config.train.enabled = false;
        bootstrap.config.base.encode.enabled = false;
        bootstrap.config.base.linkage.enabled = false;
        LogInfo("advanced.eval_only=true: forcing train.enabled=false, base.encode.enabled=false, base.linkage.enabled=false");
    }
    NormalizeHVec(&bootstrap.config);
    ApplyDatasetDefaults(&bootstrap.config);
    NormalizeIoPrefixWithM(&bootstrap.config);

#if defined(STLQ_ENABLE_HDF5) && STLQ_ENABLE_HDF5
    io::SetDefaultHdf5MatrixLayout(bootstrap.config.io.hdf5_layout == "cxx"
                                       ? io::Hdf5MatrixLayout::kCxx
                                       : io::Hdf5MatrixLayout::kJulia);
#endif

    LogInfo("Config: dataset.name=" + bootstrap.config.dataset.name);
    LogInfo("Config: io.train_file=" + bootstrap.config.io.train_file);
    LogInfo("Config: io.base_file=" + bootstrap.config.io.base_file);
    LogInfo("Config: io.linkage_file=" + bootstrap.config.io.linkage_file);
    LogInfo("Config: io.load_date=\"" + bootstrap.config.io.load_date +
            "\" io.load_seq=\"" + bootstrap.config.io.load_seq + "\"");
    LogInfo("Config: large.enabled=" + std::string(bootstrap.config.large.enabled ? "true" : "false"));
    return ContinueMainStage();
}

}  // namespace stlq
