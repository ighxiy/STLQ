#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "stlq/common/config.h"
#include "stlq/io/col_block_reader.h"
#include "stlq/io/dataset_reader_factory.h"
#include "stlq/quantizer/spkmeans.h"

namespace stlq {

enum class TrainFormat { kBvecs, kFvecs, kFbin };

struct StreamingTrainInput {
    io::DatasetVectorReader reader;
    TrainFormat train_format = TrainFormat::kFvecs;
    bool train_is_u8 = false;
    int d = 0;
    std::uint64_t ntrain = 0;
};

bool OpenTrainReaderAuto(const Config& cfg,
                         io::DatasetVectorReader* reader,
                         std::string* err);

bool OpenStreamingTrainInput(const Config& config,
                             StreamingTrainInput* input,
                             std::string* error);

std::unique_ptr<io::IColBlockReader> MakeTrainStreamingColBlockReader(const StreamingTrainInput& train_input,
                                                                      const KmeansConfig& kcfg,
                                                                      std::uint64_t ntrain,
                                                                      std::string* error);

}  // namespace stlq
