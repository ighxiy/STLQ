#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>

#include "stlq/core/kernel_provider_cuda_stream.h"
#include "stlq/io/col_block_reader.h"
#include "stlq/quantizer/spkmeans.h"

namespace {

stlq::ColMajorMatrix<float> MakeDeterministicData(int d, int n, std::uint32_t seed) {
    stlq::ColMajorMatrix<float> X(d, n);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int j = 0; j < n; ++j) {
        float* col = X.Col(j);
        for (int r = 0; r < d; ++r) {
            col[r] = dist(rng);
        }
    }
    return X;
}

void RemoveTreeBestEffort(const std::filesystem::path& p) {
    std::error_code ec;
    std::filesystem::remove_all(p, ec);
}

}  // namespace

int main() {
    constexpr int d = 16;
    constexpr int n = 5000;
    constexpr int k = 64;
    constexpr std::uint32_t seed = 123;

    stlq::ColMajorMatrix<float> X = MakeDeterministicData(d, n, seed);

    stlq::CudaStreamKernels cuda(/*device=*/0, /*allow_tf32=*/true, /*cublas_workspace_mb=*/256);

    auto Run = [&](const stlq::KmeansConfig& cfg, std::uint32_t run_seed, stlq::KmeansTiming* timing) {
        std::mt19937 rng(run_seed);
        std::string err;
        std::unique_ptr<stlq::io::IColBlockReader> reader = stlq::io::MakeMatrixColBlockReader(X);
        stlq::ColMajorMatrix<float> centers =
            stlq::SphericalKmeansCentersOnlyStreamingReader(reader.get(),
                                                              k,
                                                              cfg,
                                                              &rng,
                                                              &cuda,
                                                              /*profile_timing=*/false,
                                                              timing,
                                                              &err);
        if (!err.empty()) {
            throw std::runtime_error(err);
        }
        if (centers.cols != k || centers.rows != d) {
            throw std::runtime_error("Unexpected centers shape.");
        }
        if (!timing || !std::isfinite(timing->mse_proxy) || timing->mse_proxy < 0.0) {
            throw std::runtime_error("mse_proxy was not computed (expected CUDA path).");
        }
        return centers;
    };

    const std::filesystem::path tmp_base = std::filesystem::path("./tmp_opt") / "test_kmeans_anneal_regression";
    RemoveTreeBestEffort(tmp_base);
    std::filesystem::create_directories(tmp_base);

    stlq::KmeansConfig cfg;
    cfg.max_iters = 4;
    cfg.tol = 1e-6f;
    cfg.init_method = "random";
    cfg.init_samples = 512;
    cfg.initial_weight = 1.0f;
    cfg.min_weight = 0.1f;
    cfg.outlier_quantile = 0.95f;
    cfg.cost_threshold = 0.8f;
    cfg.annealing_factor = 0.5f;
    cfg.warmup_iters = 1;
    cfg.quantile_bins = 2048;
    cfg.tmp_dir = (tmp_base / "tmp").string();
    cfg.streaming = true;
    cfg.cache_xnorm_device = false;
    cfg.pin_host_x = true;
    cfg.bvecs_use_u8 = false;
    cfg.block_cols = 512;
    cfg.anneal_no_spill = true;
    cfg.anneal_weights_device = true;
    cfg.force_disable_device_weights = false;

    stlq::KmeansTiming t_twopass{};
    stlq::KmeansConfig cfg_twopass = cfg;
    cfg_twopass.anneal_mode = stlq::KmeansAnnealMode::kTwoPass;
    Run(cfg_twopass, /*run_seed=*/seed, &t_twopass);

    stlq::KmeansTiming t_onepass{};
    stlq::KmeansConfig cfg_onepass = cfg;
    cfg_onepass.anneal_mode = stlq::KmeansAnnealMode::kOnePass;
    Run(cfg_onepass, /*run_seed=*/seed, &t_onepass);

    if (!std::isfinite(t_twopass.mse_proxy) || !std::isfinite(t_onepass.mse_proxy)) {
        throw std::runtime_error("mse_proxy not finite.");
    }
    const double diff = std::abs(t_twopass.mse_proxy - t_onepass.mse_proxy);
    if (diff > 0.25) {
        throw std::runtime_error("onepass vs twopass mse_proxy diverged too much: diff=" + std::to_string(diff));
    }

    // Fallback path (simulate OOM/unavailable device weights).
    stlq::KmeansTiming t_fallback{};
    stlq::KmeansConfig cfg_fallback = cfg;
    cfg_fallback.anneal_mode = stlq::KmeansAnnealMode::kTwoPass;
    cfg_fallback.force_disable_device_weights = true;
    cfg_fallback.tmp_dir = (tmp_base / "tmp_fallback").string();
    Run(cfg_fallback, /*run_seed=*/seed, &t_fallback);

    RemoveTreeBestEffort(tmp_base);
    return 0;
}
