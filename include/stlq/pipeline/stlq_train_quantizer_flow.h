#pragma once

#include "stlq/common/config.h"
#include "stlq/core/kernels_cpu.h"
#include "stlq/quantizer/encoder.h"

#include <filesystem>
#include <string>

namespace stlq {

int RunTrainOrLoadQuantizerStage(Config& config,
                                 const std::filesystem::path& large_root,
                                 CpuKernels& kernels,
                                 StreamKernelProvider* stream_kernels,
                                 TrainResult& train_result,
                                 std::string& error);

}  // namespace stlq
