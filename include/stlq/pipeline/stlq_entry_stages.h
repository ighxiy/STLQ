#pragma once

#include "stlq/common/config.h"
#include "stlq/core/kernel_provider_cpu.h"
#include "stlq/core/kernels_cpu.h"

#include <filesystem>
#include <memory>
#include <string>

namespace stlq {

class CudaStreamKernels;

constexpr int kMainStageContinue = -1;

struct KernelRuntimeState {
    KernelRuntimeState();
    ~KernelRuntimeState();

    KernelRuntimeState(KernelRuntimeState&&) noexcept;
    KernelRuntimeState& operator=(KernelRuntimeState&&) noexcept;
    KernelRuntimeState(const KernelRuntimeState&) = delete;
    KernelRuntimeState& operator=(const KernelRuntimeState&) = delete;

    CpuKernels kernels;
    CpuStreamKernels cpu_stream_kernels;
    std::unique_ptr<CudaStreamKernels> cuda_stream_kernels;
    StreamKernelProvider* stream_kernels = &cpu_stream_kernels;
};

int ContinueMainStage();
int ReturnFromMainStage(int exit_code);

void ApplyMainRuntimeStage(const Config& config);

int PrepareLargeWorkspaceStage(const Config& config,
                               std::filesystem::path& large_root,
                               std::filesystem::path& large_tmp,
                               std::string& error);

int PrepareKernelRuntimeStage(const Config& config,
                              KernelRuntimeState& kernel_runtime);

}  // namespace stlq
