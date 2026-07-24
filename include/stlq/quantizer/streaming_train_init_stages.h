#pragma once

#include <string>

#include "stlq/common/config.h"
#include "stlq/io/linkage_list_store.h"
#include "stlq/quantizer/streaming_train_config.h"
#include "stlq/quantizer/streaming_train_io.h"
#include "stlq/common/types.h"

namespace stlq {

class StreamKernelProvider;
class KernelProvider;

namespace io {
class BaseListReader;
class IvfListsReader;
}  // namespace io

bool RunStreamingTrainRvqInitStage(const Config& config,
                                   int d,
                                   std::uint64_t ntrain,
                                   const StreamingTrainInput& train_input,
                                   KernelProvider* kernels,
                                   StreamKernelProvider* stream_kernels,
                                   TrainResult& result,
                                   std::string* error);

bool RunStreamingTrainInitLinkageAndUpdateC1(const Config& config,
                                          TrainResult& result,
                                          const StreamingTrainWorkspacePaths& train_paths,
                                          io::BaseListReader& train_list,
                                          io::IvfListsReader& ivf,
                                          StreamKernelProvider* rvq_stream_kernels,
                                          StreamKernelProvider* linkage_stream_kernels,
                                          const io::DatasetVectorReader* fallback_reader,
                                          bool allow_random_fallback,
                                          const io::LinkageListStoreConfig& linkage_store_cfg,
                                          const Config& linkage_cfg,
                                          std::string* error);

}  // namespace stlq
