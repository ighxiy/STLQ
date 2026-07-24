#pragma once

#include <memory>
#include <string>
#include <vector>

#include "stlq/eval/linkage_cluster_provider.h"
#include "stlq/eval/norm2_lut.h"
#include "stlq/quantizer/codebook_meta.h"
#include "stlq/common/types.h"

namespace stlq::eval::cuda {

// Experimental CUDA norm2 provider for disk IVF linkage recall.
// This is an eval-only optimization; callers must keep CPU semantics identical.
class LinkageNormProviderLookupCuda final : public IClusterNormProvider {
public:
    struct Stats {
        bool prof_enabled = false;
        bool used_gpu = false;
        bool used_gpu_lut = false;
        int fallback_reason = 0; // 0=none, 2=too_large, 3=unsupported, 4=gpu_error
        double cpu_c0dot_sec = 0.0;
        double h2d_sec = 0.0;
        double kernel_sec = 0.0;
        double d2h_sec = 0.0;
        double lut_kmeans_sec = 0.0;
    };

    LinkageNormProviderLookupCuda(const ColMajorMatrix<float>& C_root0,
                                const CodebookMeta& meta_root_small,
                                const CodebookMeta& meta_one,
                                int gpu_max_nc,
                                bool profile_breakdown);
    ~LinkageNormProviderLookupCuda() override;

    bool ComputeNorm2(const ClusterView& cv,
                      std::vector<float>* out_r_norm2,
                      std::string* err) override;
    bool ComputeNorm2Lut(const ClusterView& cv,
                         int requested_centers,
                         int max_iter,
                         Norm2Lut* out,
                         std::string* err);
    bool ComputeNorm2LutFromHost(const float* norm2,
                                 int n,
                                 int requested_centers,
                                 int max_iter,
                                 Norm2Lut* out,
                                 std::string* err) override;
    bool IsThreadSafeForParallelPrecompute() const override { return false; }
    std::unique_ptr<IClusterNormProvider> CloneForParallelPrecompute() const override;

    const Stats& last_stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace stlq::eval::cuda
