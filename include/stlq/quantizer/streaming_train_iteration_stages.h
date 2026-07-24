#pragma once

#include <cstdint>
#include <string>

#include "stlq/common/config.h"
#include "stlq/quantizer/streaming_train_config.h"
#include "stlq/quantizer/streaming_train_io.h"
#include "stlq/common/types.h"

namespace stlq {

class StreamKernelProvider;

struct StreamingTrainIterationPlan {
    int resume_base_R_iters = 0;
    int additional_iters = 0;
    int display_total_iters = 0;
    int final_R_iters = 0;
    bool allow_random_fallback = false;
};

StreamingTrainIterationPlan MakeStreamingTrainIterationPlan(const Config& config,
                                                            std::uint64_t ntrain,
                                                            int resume_base_R_iters);

bool RunStreamingTrainRoundStage(const Config& config,
                                 TrainResult& result,
                                 const StreamingTrainWorkspacePaths& paths,
                                 const StreamingTrainIterationPlan& plan,
                                 int global_iter,
                                 bool do_init_linkage_and_c1,
                                 bool do_opq_update,
                                 int d,
                                 int nlist,
                                 std::uint64_t ntrain,
                                 const StreamingTrainInput& train_input,
                                 StreamKernelProvider* stream_kernels,
                                 bool* logged_linkage_stage_cpu_notice,
                                 bool* early_exit_after_ckpt_init_basic,
                                 std::string* error);

bool RunStreamingTrainInitCheckpointStage(const Config& config,
                                          TrainResult& result,
                                          const StreamingTrainIterationPlan& plan,
                                          std::string* error);

bool RunStreamingTrainResumePreInitLinkageStage(const Config& config,
                                              TrainResult& result,
                                              const StreamingTrainWorkspacePaths& paths,
                                              const StreamingTrainIterationPlan& plan,
                                              const StreamingTrainInput& train_input,
                                              StreamKernelProvider* stream_kernels,
                                              bool* logged_linkage_stage_cpu_notice,
                                              std::string* error);

bool RunStreamingTrainGlobalIterationLoopStage(const Config& config,
                                               TrainResult& result,
                                               const StreamingTrainWorkspacePaths& paths,
                                               const StreamingTrainIterationPlan& plan,
                                               int d,
                                               int nlist,
                                               std::uint64_t ntrain,
                                               const StreamingTrainInput& train_input,
                                               StreamKernelProvider* stream_kernels,
                                               bool* logged_linkage_stage_cpu_notice,
                                               std::string* error);

}  // namespace stlq
