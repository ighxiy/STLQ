#pragma once

#include <filesystem>
#include <string>

#include "stlq/common/config.h"
#include "stlq/core/kernels.h"
#include "stlq/common/types.h"

namespace stlq {

class StreamKernelProvider;

struct TrainState {
    CodebookPack C_root;
    CodebookPack C_one;
    ColMajorMatrix<FullCode> B;
    ColMajorMatrix<float> a;
    ColMajorMatrix<float> R_full;
    ColMajorMatrix<float> R_in;
    LinkageStructure train_linkage;
    std::vector<bool> is_bad_cluster;

    // Training summary metrics (used for logging + train-result metadata).
    float init_mse = -1.0f;
    float beam_mse = -1.0f;
    float icm_mse = -1.0f;
    float linkage_mse_mean = -1.0f;
    float linkage_ratio = -1.0f;
    int linkage_max_depth = -1;
    float linkage_mean_depth = -1.0f;
};

bool BuildPrecomp(const CodebookPack& codebooks, Precomp* precomp);

// Julia reference: train_quantizer_and_linkage_beam! (train_quantizer.jl:118-232).
bool TrainQuantizerAndLinkageBeam(const Config& config,
                                const ColMajorMatrix<float>& X_rot,
                                StreamKernelProvider* stream_kernels,
                                TrainState* state,
                                std::string* error);

// Julia reference: linkage_refine_on_rotated_data_beam! (train_quantizer.jl:247-360).
bool LinkageRefineOnRotatedDataBeam(const Config& config,
                                  const ColMajorMatrix<float>& X_rot,
                                  TrainState* state,
                                  std::string* error);

bool TrainQuantizer(const Config& config,
                    const Dataset& dataset,
                    KernelProvider* kernels,
                    StreamKernelProvider* stream_kernels,
                    TrainResult* result,
                    std::string* error);

// Large-scale out-of-core training entry (used only when large.enabled=true).
// This function must not load the full train matrix into memory.
// Prompts 01-12 progressively extend this implementation.
bool TrainQuantizerStreamingLarge(const Config& config,
                                 const std::filesystem::path& exp_root,
                                 KernelProvider* kernels,
                                 StreamKernelProvider* stream_kernels,
                                 TrainResult* result,
                                 std::string* error);

bool EncodeBase(const Config& config,
                const Dataset& dataset,
                const CodebookPack& codebooks,
                const Precomp& precomp,
                KernelProvider* kernels,
                BaseEncoding* base,
                float* out_beam_mse,
                float* out_final_mse,
                std::string* error);

}  // namespace stlq
