#include "stlq/pipeline/stlq_entry_stages.h"

#include "stlq/pipeline/app_utils.h"
#include "stlq/common/config.h"
#include "stlq/core/blas.h"
#include "stlq/core/kernel_provider_cuda_stream.h"
#include "stlq/core/threading.h"
#include "stlq/common/logger.h"

#include <algorithm>
#include <filesystem>

#include <omp.h>

namespace stlq {

KernelRuntimeState::KernelRuntimeState() = default;
KernelRuntimeState::~KernelRuntimeState() = default;
KernelRuntimeState::KernelRuntimeState(KernelRuntimeState&&) noexcept = default;
KernelRuntimeState& KernelRuntimeState::operator=(KernelRuntimeState&&) noexcept = default;

int ContinueMainStage() {
    return kMainStageContinue;
}

int ReturnFromMainStage(int exit_code) {
    return exit_code;
}

void ApplyMainRuntimeStage(const Config& config) {
    if (config.runtime.omp_threads > 0) {
        omp_set_num_threads(config.runtime.omp_threads);
    }
    omp_set_dynamic(0);
#if defined(_OPENMP) && (_OPENMP >= 200805)
    // OpenMP 3.0+: keep nested parallelism off (predictable performance).
    omp_set_max_active_levels(1);
#else
    // MSVC's legacy OpenMP (2.0) does not provide omp_set_max_active_levels.
    omp_set_nested(0);
#endif
    BlasSetThreads(1);
    SetOmpDefaultThreads(std::max(1, omp_get_max_threads()));
    LogInfo("Runtime: omp_max_threads=" + std::to_string(omp_get_max_threads()) +
            " blas_threads=1");
}

int PrepareLargeWorkspaceStage(const Config& config,
                               std::filesystem::path& large_root,
                               std::filesystem::path& large_tmp,
                               std::string& error) {
    if (!PrepareLargeWorkspaceDirs(config, &large_root, &large_tmp, &error)) {
        LogError(error);
        return ReturnFromMainStage(1);
    }
    return ContinueMainStage();
}

int PrepareKernelRuntimeStage(const Config& config,
                              KernelRuntimeState& kernel_runtime) {
    kernel_runtime.stream_kernels = &kernel_runtime.cpu_stream_kernels;
    if (config.runtime.use_cuda) {
#if defined(STLQ_ENABLE_CUDA)
        kernel_runtime.cuda_stream_kernels = std::make_unique<CudaStreamKernels>(
            config.runtime.cuda_device,
            EffectiveCudaAllowTf32(config.runtime),
            config.runtime.cuda_cublas_workspace_mb);
        kernel_runtime.stream_kernels = kernel_runtime.cuda_stream_kernels.get();
#else
        LogError("runtime.use_cuda=true but this binary was built without STLQ_ENABLE_CUDA.");
        return ReturnFromMainStage(1);
#endif
    }
    return ContinueMainStage();
}

}  // namespace stlq
