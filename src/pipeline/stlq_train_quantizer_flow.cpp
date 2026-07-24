#include "stlq/pipeline/stlq_train_quantizer_flow.h"

#include "stlq/pipeline/app_utils.h"
#include "stlq/pipeline/stlq_entry_stages.h"
#include "stlq/io/dataset_io.h"
#include "stlq/io/result_io.h"
#include "stlq/common/logger.h"
#include "stlq/common/timer.h"

#include <filesystem>

namespace stlq {

int RunTrainOrLoadQuantizerStage(Config& config,
                                 const std::filesystem::path& large_root,
                                 CpuKernels& kernels,
                                 StreamKernelProvider* stream_kernels,
                                 TrainResult& train_result,
                                 std::string& error) {
    if (config.train.enabled) {
        LogInfo("Training quantizer...");
        Timer train_timer;
        if (config.large.enabled) {
            if (!TrainQuantizerStreamingLarge(config, large_root, &kernels, stream_kernels, &train_result, &error)) {
                LogError(error);
                return ReturnFromMainStage(1);
            }
        } else {
            Dataset train_dataset;
            LogInfo("Loading train set: " + config.dataset.name);
            Timer load_timer;
            if (!io::LoadTrainSet(config.dataset, &train_dataset.Xt, &error)) {
                LogError(error);
                return ReturnFromMainStage(1);
            }
            LogInfo("Loaded Xt=" + DescribeMatrix(train_dataset.Xt) +
                    " in " + load_timer.ReportSeconds("time"));
            if (!TrainQuantizer(config, train_dataset, &kernels, stream_kernels, &train_result, &error)) {
                LogError(error);
                return ReturnFromMainStage(1);
            }
            train_dataset.Xt = {};
        }
        LogInfo(train_timer.ReportSeconds("Training finished in"));

        if (config.train.exit_after_rvq_init) {
            LogInfo("Exit: train.exit_after_rvq_init=1");
            return ReturnFromMainStage(0);
        }

        if (config.train.init_enabled &&
            config.train.ckpt_after_init_basic &&
            config.train.exit_after_ckpt_init_basic &&
            train_result.checkpoint_stage == 1) {
            LogInfo("Exit: pre-init-linkage checkpoint requested (checkpoint_stage=1).");
            return ReturnFromMainStage(0);
        }

        if (config.io.save_train) {
            std::string train_path = io::MakeUniqueDatedPath(config.io.train_file, &error);
            if (train_path.empty()) {
                LogError(error);
                return ReturnFromMainStage(1);
            }
            LogInfo("Saving train results to: " + train_path);
            Timer save_timer;
            if (!io::SaveTrainResults(train_path, config, train_result, &error)) {
                LogError(error);
                return ReturnFromMainStage(1);
            }
            train_result.saved_train_h5 = train_path;
            LogInfo(save_timer.ReportSeconds("Train results saved in"));
        }
    } else {
        LogInfo("Training disabled; loading train results...");
        std::string train_load_date = config.io.load_date;
        std::string train_load_seq = config.io.load_seq;
        if (config.io.train_file_set && !config.io.train_file.empty()) {
            std::error_code ec;
            const std::filesystem::path train_path(config.io.train_file);
            const bool train_path_exists = std::filesystem::exists(train_path, ec);
            if (train_path_exists && !ec) {
                train_load_date = "raw";
                train_load_seq.clear();
            }
        }
        if (!io::LoadTrainResults(config.io.train_file,
                                  train_load_date,
                                  train_load_seq,
                                  config.model.m,
                                  &train_result,
                                  &error)) {
            LogError(error);
            return ReturnFromMainStage(1);
        }
        {
            std::string local_err;
            const std::string resolved =
                io::MakeDatedPathForLoad(config.io.train_file, train_load_date, train_load_seq, &local_err);
            if (!resolved.empty()) {
                train_result.loaded_train_h5 = resolved;
            }
        }
    }
    return ContinueMainStage();
}

}  // namespace stlq
