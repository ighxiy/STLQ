#pragma once

#include <cstdint>
#include <string>

#include "stlq/common/config.h"
#include "stlq/core/kernel_provider_cpu.h"
#include "stlq/io/base_list_store.h"
#include "stlq/io/linkage_list_store.h"
#include "stlq/io/ivf_lists.h"
#include "stlq/quantizer/streaming_train_config.h"
#include "stlq/quantizer/streaming_train_io.h"
#include "stlq/common/types.h"

namespace stlq {

class StreamKernelProvider;

struct StreamingTrainLinkageRoundContext {
    CpuStreamKernels cpu_linkage_kernels;
    StreamKernelProvider* linkage_stream_kernels = nullptr;
    const io::DatasetVectorReader* fallback_reader = nullptr;
    bool allow_random_fallback = false;
    io::LinkageListStoreConfig linkage_store_cfg;
    Config linkage_cfg;
};

bool RunStreamingTrainBasicEncodeStage(const Config& config,
                                       TrainResult& result,
                                       const StreamingTrainWorkspacePaths& paths,
                                       std::uint64_t ntrain,
                                       int d,
                                       bool is_global_round,
                                       StreamKernelProvider* stream_kernels,
                                       std::string* error);

bool RunStreamingTrainUpdateCRootStage(const Config& config,
                                       TrainResult& result,
                                       const StreamingTrainWorkspacePaths& paths,
                                       StreamKernelProvider* stream_kernels,
                                       std::string* error);

bool BuildStreamingTrainIvfAndList(const Config& config,
                                   const StreamingTrainWorkspacePaths& paths,
                                   int nlist,
                                   io::IvfListsReader* ivf,
                                   io::BaseListReader* train_list,
                                   std::string* error);

bool OpenStreamingTrainIvfAndList(const StreamingTrainWorkspacePaths& paths,
                                  io::IvfListsReader* ivf,
                                  io::BaseListReader* train_list,
                                  std::string* error);

void PrepareStreamingTrainLinkageRoundContext(const Config& config,
                                            const StreamingTrainWorkspacePaths& paths,
                                            const io::IvfListsReader& ivf,
                                            const StreamingTrainInput& train_input,
                                            bool allow_random_fallback,
                                            StreamKernelProvider* stream_kernels,
                                            bool* logged_linkage_stage_cpu_notice,
                                            StreamingTrainLinkageRoundContext* context);

bool RunStreamingTrainIterationTailStage(const Config& config,
                                         TrainResult& result,
                                         const StreamingTrainWorkspacePaths& paths,
                                         int global_iter,
                                         bool do_opq_update,
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
