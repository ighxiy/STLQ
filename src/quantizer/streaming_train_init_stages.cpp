#include "stlq/quantizer/streaming_train_init_stages.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <random>
#include <string>
#include <utility>

#include "stlq/pipeline/app_utils.h"
#include "stlq/linkage/linkage_streaming_builders.h"
#include "stlq/linkage/linkage_summary.h"
#include "stlq/core/kernel_provider.h"
#include "stlq/core/threading.h"
#include "stlq/io/base_list_store.h"
#include "stlq/io/bvecs_reader.h"
#include "stlq/io/col_block_reader.h"
#include "stlq/io/dataset_io.h"
#include "stlq/io/fbin_reader.h"
#include "stlq/io/fvecs_reader.h"
#include "stlq/io/ivf_lists.h"
#include "stlq/common/logger.h"
#include "stlq/quantizer/encoder.h"
#include "stlq/quantizer/spkmeans.h"
#include "stlq/quantizer/streaming_train_io.h"
#include "stlq/quantizer/update_codebooks_streaming.h"
#include "stlq/quantizer/rvq_layer_store.h"
#include "stlq/common/timer.h"

#if defined(STLQ_ENABLE_CUDA)
#include "stlq/linkage/linkage_encode_cuda.h"
#include "stlq/core/kernel_provider_cuda_stream.h"
#include <cuda_runtime.h>
#endif

namespace stlq {

namespace {

inline std::string ToLowerCopy(std::string s) {
    for (char& ch : s) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return s;
}

#if defined(STLQ_ENABLE_CUDA)
void LogCudaVramIfEnabled(const Config& cfg, const char* label) {
    if (!cfg.runtime.use_cuda) return;
    if (!cfg.large.profile_timing) return;
    std::size_t free_b = 0;
    std::size_t total_b = 0;
    const cudaError_t st = cudaMemGetInfo(&free_b, &total_b);
    if (st != cudaSuccess) {
        LogWarn(std::string("cudaMemGetInfo failed (") + label + "): " + cudaGetErrorString(st));
        return;
    }
    int device = 0;
    (void)cudaGetDevice(&device);
    const std::size_t used_b = (total_b >= free_b) ? (total_b - free_b) : 0;
    const auto used_mib = static_cast<std::uint64_t>(used_b / (1024ull * 1024ull));
    const auto free_mib = static_cast<std::uint64_t>(free_b / (1024ull * 1024ull));
    const auto total_mib = static_cast<std::uint64_t>(total_b / (1024ull * 1024ull));
    LogInfo(std::string("CUDA VRAM (") + label + "): device=" + std::to_string(device) +
            " used_mib=" + std::to_string(used_mib) +
            " free_mib=" + std::to_string(free_mib) +
            " total_mib=" + std::to_string(total_mib));
}
#endif

enum class InitLinkageMode {
    kLegacyRootOnly = 0,
    kFastInit = 1,
    kHybrid = 2,   // Legacy path but with hybrid ILS: first round VarRoot, subsequent rounds ConstRoot.
};

bool ParseInitLinkageMode(const std::string& s_in, InitLinkageMode* out, std::string* err) {
    if (!out) return false;
    const std::string s = ToLowerCopy(s_in);
    if (s.empty() || s == "legacy_root_only" || s == "legacy") {
        *out = InitLinkageMode::kLegacyRootOnly;
        return true;
    }
    if (s == "fast_init" || s == "fast") {
        *out = InitLinkageMode::kFastInit;
        return true;
    }
    if (s == "hybrid") {
        *out = InitLinkageMode::kHybrid;
        return true;
    }
    if (err) {
        *err = "invalid train.init_linkage_mode=\"" + s_in +
               "\" (expected legacy_root_only|fast_init|hybrid).";
    }
    return false;
}

}  // namespace

bool RunStreamingTrainRvqInitStage(const Config& config,
                                   int d,
                                   std::uint64_t ntrain,
                                   const StreamingTrainInput& train_input,
                                   KernelProvider* kernels,
                                   StreamKernelProvider* stream_kernels,
                                   TrainResult& result,
                                   std::string* error) {
    // Init stage: full streaming RVQ init from the dataset reader (C_root only).
    {
        LogInfo("============== Baseline training with R = I (no global rotation) ==============");
        LogInfo("(1)RVQ-like init by Spherical kmeans with annealing factor...");
        LogInfo("Train streaming init: using dataset reader (out-of-core).");

        // When CUDA stream kernels are available, do fully streaming RVQ init directly from the dataset reader.
        // Otherwise (CPU-only run), fall back to the older "reservoir sample + in-memory init" path:
        // - This keeps large training usable without CUDA (useful for debugging / CPU-only machines).
        // - It is NOT fully streaming and is typically less stable/performant than the GPU init.
        if (!stream_kernels || !stream_kernels->IsGpu()) {
            const int init_samples = std::max(1, config.train.init_samples);
            const int K = static_cast<int>(std::min<std::uint64_t>(ntrain, static_cast<std::uint64_t>(init_samples)));
            LogWarn("Train streaming init: CUDA stream kernels unavailable; falling back to reservoir-sampled CPU init "
                    "(K=" + std::to_string(K) + ", ntrain=" + std::to_string(ntrain) + ")");

            ColMajorMatrix<float> Xt_init;
            if (!io::ReservoirSampleTrainToF32(config, static_cast<std::int64_t>(ntrain), K, &Xt_init, error)) {
                return false;
            }

            Dataset init_ds;
            init_ds.Xt = std::move(Xt_init);

            Config init_cfg = config;
            init_cfg.train.max_R_iters = 0;
            init_cfg.dataset.ntrain = init_ds.Xt.cols;
            init_cfg.dataset.ntrain_set = true;

            if (!TrainQuantizer(init_cfg, init_ds, kernels, stream_kernels, &result, error)) {
                return false;
            }

            // Match the streaming training expectations: C_one is initialized later, and rotation starts as identity.
            result.C_one = {};
            result.R = ColMajorMatrix<float>(d, d);
            std::fill(result.R.data.begin(), result.R.data.end(), 0.0f);
            for (int i = 0; i < d; ++i) {
                result.R(i, i) = 1.0f;
            }
            result.train_linkage = {};
            result.is_bad_cluster.clear();
            result.init_mse = -1.0f;
            result.beam_mse = -1.0f;
            result.icm_mse = -1.0f;
            result.linkage_mse_mean = -1.0f;
            result.linkage_ratio = -1.0f;
            result.linkage_max_depth = -1;
            result.linkage_mean_depth = -1.0f;
            result.R_iters = 0;
        } else {
            if (!config.train.kmeans_streaming) {
                if (error) *error = "TrainQuantizerStreamingLarge: requires train.kmeans_streaming=true.";
                return false;
            }

            #if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
            auto* cuda = dynamic_cast<CudaStreamKernels*>(stream_kernels);
            if (!cuda) {
                if (error) *error = "TrainQuantizerStreamingLarge: CUDA kernels type mismatch.";
                return false;
            }
            #else
            (void)stream_kernels;
            if (error) *error = "TrainQuantizerStreamingLarge: requires STLQ_ENABLE_CUDA build.";
            return false;
            #endif

        KmeansConfig kcfg = MakeStreamingTrainKmeansConfig(config, train_input.train_is_u8);

        std::string err;
        auto reader = MakeTrainStreamingColBlockReader(train_input,
                                                       kcfg,
                                                       ntrain,
                                                       error);
        if (!reader) return false;

        LogInfo("Train streaming init: dataset-reader RVQ ntrain=" + std::to_string(ntrain) +
                " m=" + std::to_string(config.model.m) +
                " block_cols=" + std::to_string(kcfg.block_cols) +
                " bvecs_u8=" + std::to_string(kcfg.bvecs_use_u8 ? 1 : 0));

        Timer t_init_wall;
        std::mt19937 rng(static_cast<std::mt19937::result_type>(config.train.seed));
        KmeansTiming timing{};
        double last_mse = -1.0;

        std::vector<ColMajorMatrix<float>> codebooks;
        codebooks.assign(static_cast<std::size_t>(config.model.m), ColMajorMatrix<float>());

        RvqInitCodesInMemory codes_store;
        if (!codes_store.Init(config.model.h_vec, reader->n(), &err)) {
            if (error) *error = "TrainQuantizerStreamingLarge: codes_store.Init failed: " + err;
            return false;
        }

        // Per-layer streaming RVQ init: centers-only kmeans + in-train codes emission (Prompt-10).
        for (int layer = 0; layer < config.model.m; ++layer) {
            const int k = config.model.h_vec[static_cast<std::size_t>(layer)];
            const bool need_layer_codes = (layer + 1 < config.model.m);
            ColMajorMatrix<float> centers;
            if (layer == 0) {
                centers = need_layer_codes
                              ? SphericalKmeansCentersOnlyStreamingReaderEmitCodes(
                                    reader.get(),
                                    k,
                                    kcfg,
                                    &rng,
                                    stream_kernels,
                                    &codes_store,
                                    /*out_layer=*/layer,
                                    /*profile_timing=*/config.large.profile_timing,
                                    &timing,
                                    &err)
                              : SphericalKmeansCentersOnlyStreamingReader(
                                    reader.get(),
                                    k,
                                    kcfg,
                                    &rng,
                                    stream_kernels,
                                    /*profile_timing=*/config.large.profile_timing,
                                    &timing,
                                    &err);
            } else {
                centers = need_layer_codes
                              ? SphericalKmeansCentersOnlyStreamingReaderRvqResidualGpuEmitCodes(
                                    reader.get(),
                                    k,
                                    kcfg,
                                    &rng,
                                    stream_kernels,
                                    &codes_store,
                                    &codebooks,
                                    /*upto_layer=*/layer,
                                    &codes_store,
                                    /*out_layer=*/layer,
                                    /*profile_timing=*/config.large.profile_timing,
                                    &timing,
                                    &err)
                              : SphericalKmeansCentersOnlyStreamingReaderRvqResidualGpu(
                                    reader.get(),
                                    k,
                                    kcfg,
                                    &rng,
                                    stream_kernels,
                                    &codes_store,
                                    &codebooks,
                                    layer,
                                    /*profile_timing=*/config.large.profile_timing,
                                    &timing,
                                    &err);
            }
            if (!err.empty()) {
                if (error) *error = "TrainQuantizerStreamingLarge: kmeans failed: " + err;
                return false;
            }
            if (centers.rows != d || centers.cols != k) {
                if (error) *error = "TrainQuantizerStreamingLarge: invalid centers shape from kmeans.";
                return false;
            }
            codebooks[static_cast<std::size_t>(layer)] = centers;
            last_mse = timing.mse_proxy;
        #if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
            if (layer + 1 < config.model.m) {
                if (auto* cuda = dynamic_cast<CudaStreamKernels*>(stream_kernels)) {
                    //LogInfo("CUDA: resetting stream-kernels state between RVQ init layers.");
                    cuda->ResetCudaState();
                }
            }
        #endif
        }
        LogInfo("Train streaming init wall time(s): " + FormatFloat(static_cast<float>(t_init_wall.ElapsedSeconds()), 3));
        if (!config.large.profile_timing) {
            LogInfo("RVQ kmeans diag: overlap_p1=" + std::to_string(timing.used_overlap_pass1) +
                    " overlap_p2=" + std::to_string(timing.used_overlap_pass2) +
                    " xnorm_dev_used=" + std::to_string(timing.used_device_xnorm_cache) +
                    " x_pin_used=" + std::to_string(timing.used_pinned_host_x));
        }
        if (config.large.profile_timing) {
            const double norm_total_s =
                ((timing.normalize_stage_s + timing.normalize_h2d_s + timing.normalize_kernel_s) > 0.0)
                    ? (timing.normalize_stage_s + timing.normalize_h2d_s + timing.normalize_kernel_s)
                    : timing.normalize_s;
            const double accounted =
                timing.reader_io_s +
                norm_total_s +
                timing.init_centers_s +
                timing.assign_cpu_s +
                timing.assign_gemm_s +
                timing.assign_argmax_s +
                timing.rvq_project_s +
                timing.assign_update_compute_s +
                timing.emit_codes_d2h_s +
                timing.wait_copy_s +
                timing.wait_compute_s +
                timing.weights_s +
                timing.update_s +
                timing.accumulate_s +
                timing.finalize_s +
                timing.spill_s +
                timing.update_read_weights_s;
            const double other = std::max(0.0, timing.total_s - accounted);
            LogInfo("RVQ kmeans timing(gpu, s): total=" + FormatFloat(static_cast<float>(timing.total_s), 4) +
                    " kmeans_streaming=" + std::to_string(kcfg.streaming ? 1 : 0) +
                    " cache_xnorm_dev=" + std::to_string(kcfg.cache_xnorm_device ? 1 : 0) +
                    " pin_host_x=" + std::to_string(kcfg.pin_host_x ? 1 : 0) +
                    " bvecs_u8=" + std::to_string(kcfg.bvecs_use_u8 ? 1 : 0) +
                    " tf32_eff=" + std::to_string(EffectiveCudaAllowTf32(config.runtime) ? 1 : 0) +
                    " ws_mb=" + std::to_string(config.runtime.cuda_cublas_workspace_mb) +
                    " overlap_p1=" + std::to_string(timing.used_overlap_pass1) +
                    " overlap_p2=" + std::to_string(timing.used_overlap_pass2) +
                    " hier2_en=" + std::to_string(timing.hier2_enable) +
                    " hier2_thr=" + std::to_string(timing.hier2_threshold) +
                    " hier2_used=" + std::to_string(timing.used_hier2) +
                    " hier2_K=" + std::to_string(timing.hier2_K) +
                    " hier2_K1=" + std::to_string(timing.hier2_K1) +
                    " hier2_K2=" + std::to_string(timing.hier2_K2) +
                    " hier2_topL=" + std::to_string(timing.hier2_topL) +
                    " anneal_en=" + std::to_string(timing.anneal_enabled) +
                    " anneal_no_spill=" + std::to_string(timing.anneal_no_spill) +
                    " io=" + FormatFloat(static_cast<float>(timing.reader_io_s), 4) +
                    " io_file=" + FormatFloat(static_cast<float>(timing.reader_file_s), 4) +
                    " io_unpack=" + FormatFloat(static_cast<float>(timing.reader_unpack_s), 4) +
                    " norm=" + FormatFloat(static_cast<float>(norm_total_s), 4) +
                    " norm_stage=" + FormatFloat(static_cast<float>(timing.normalize_stage_s), 4) +
                    " norm_h2d=" + FormatFloat(static_cast<float>(timing.normalize_h2d_s), 4) +
                    " norm_ker=" + FormatFloat(static_cast<float>(timing.normalize_kernel_s), 4) +
                    " proj=" + FormatFloat(static_cast<float>(timing.rvq_project_s), 4) +
                    " assign_upd=" + FormatFloat(static_cast<float>(timing.assign_update_compute_s), 4) +
                    " codes_d2h=" + FormatFloat(static_cast<float>(timing.emit_codes_d2h_s), 4) +
                    " wait_copy=" + FormatFloat(static_cast<float>(timing.wait_copy_s), 4) +
                    " wait_compute=" + FormatFloat(static_cast<float>(timing.wait_compute_s), 4) +
                    " init=" + FormatFloat(static_cast<float>(timing.init_centers_s), 4) +
                    " init_sample=" + FormatFloat(static_cast<float>(timing.init_sample_s), 4) +
                    " init_pick=" + FormatFloat(static_cast<float>(timing.init_pick_s), 4) +
                    " assign_cpu=" + FormatFloat(static_cast<float>(timing.assign_cpu_s), 4) +
                    " assign_gemm=" + FormatFloat(static_cast<float>(timing.assign_gemm_s), 4) +
                    " assign_argmax=" + FormatFloat(static_cast<float>(timing.assign_argmax_s), 4) +
                    " weights=" + FormatFloat(static_cast<float>(timing.weights_s), 4) +
                    " update=" + FormatFloat(static_cast<float>(timing.update_s), 4) +
                    " acc=" + FormatFloat(static_cast<float>(timing.accumulate_s), 4) +
                    " fin=" + FormatFloat(static_cast<float>(timing.finalize_s), 4) +
                    " spill=" + FormatFloat(static_cast<float>(timing.spill_s), 4) +
                    " upd_wread=" + FormatFloat(static_cast<float>(timing.update_read_weights_s), 4) +
                    " other=" + FormatFloat(static_cast<float>(other), 4));
        }
        if (config.train.log_metrics) {
            if (last_mse < 0.0) {
                LogInfo("RVQ-like Init mse error: unavailable (missing GPU metric totals)");
            } else {
                LogInfo("RVQ-like Init mse error: " +
                        FormatFloat(static_cast<float>(last_mse), 6));
            }
        }

#ifndef NDEBUG
        if (config.large.profile_timing) {
            if (auto* s = dynamic_cast<io::IColBlockReaderDebugStats*>(reader.get())) {
                const auto st = s->DebugStats();
                LogInfo("RVQ reader debug: read_calls=" + std::to_string(st.read_calls) +
                        " seek_calls=" + std::to_string(st.seek_calls) +
                        " seq_seek_skips=" + std::to_string(st.sequential_seek_skips) +
                        " bytes_read=" + std::to_string(st.bytes_read) +
                        " last_off=" + std::to_string(st.last_offset_bytes) +
                        " last_bytes=" + std::to_string(st.last_read_bytes));
            }
        }
#endif

        // Publish C_root.
        result.C_root.d = d;
        result.C_root.h_vec = config.model.h_vec;
        result.C_root.books = std::move(codebooks);

        // Note: C_one is initialized later from the streaming train_basic + init-linkage stage
        // (beam/ILS/ICM -> Update C_root -> build init-linkage -> exact-LS update for C_one).
        result.C_one = {};

        // Rotation init: identity.
        result.R = ColMajorMatrix<float>(d, d);
        std::fill(result.R.data.begin(), result.R.data.end(), 0.0f);
        for (int i = 0; i < d; ++i) {
            result.R(i, i) = 1.0f;
        }
        result.train_linkage = {};
        result.is_bad_cluster.clear();
        result.init_mse = -1.0f;
        result.beam_mse = -1.0f;
        result.icm_mse = -1.0f;
        result.linkage_mse_mean = -1.0f;
        result.linkage_ratio = -1.0f;
        result.linkage_max_depth = -1;
        result.linkage_mean_depth = -1.0f;
        result.R_iters = 0;

#if defined(STLQ_ENABLE_CUDA) && STLQ_ENABLE_CUDA
        // Phase boundary: streaming RVQ/kmeans may allocate very large CUDA buffers (e.g. device cache for X_norm).
        // These are not reused by the subsequent basic encode stage, and keeping them can cause CUDA OOM due to
        // tight VRAM or allocator fragmentation (especially when h0_root is large and basic needs big precomps).
        if (kcfg.device_cache_mb > 0 || kcfg.cache_xnorm_device) {
            //LogInfo("CUDA: resetting stream-kernels state after RVQ init (release kmeans device cache).");
            cuda->ResetCudaState();
        }
#endif
        }
    }
    return true;
}

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
                                          std::string* error) {
#if defined(STLQ_ENABLE_CUDA)
    // Init-linkage uses its own CUDA ctx pool and can be extremely VRAM-hungry (e.g. h0_root=65536).
    // `rvq_stream_kernels` (CudaStreamKernels) is only used by RVQ/basic stages and is not thread-safe for linkage.
    // Force-release its cached device buffers here so init-linkage has maximum free VRAM.
    if (auto* cuda = dynamic_cast<CudaStreamKernels*>(rvq_stream_kernels)) {
        cuda->ResetCudaState();
    }
    LogCudaVramIfEnabled(config, "before init-linkage");
#endif
    const stlq::Timer init_linkage_wall;
    io::LinkageListStoreConfig init_linkage_store_cfg = linkage_store_cfg;
    init_linkage_store_cfg.dir = train_paths.train_linkage_init_list_dir.string();
    // Legacy init-linkage stores layer0 as C_root (routing) codes in `code0_one`.
    // For large IVF (nlist), this must auto-adapt to u8/u16/u32.
    // Fast init-linkage builds a regular two-codebook linkage_list, so `code0_one` is always C_one[0] (u8).
    const auto legacy_code0_width_bytes = [&]() -> int {
        const int h0_root = ivf.nlist();
        if (h0_root <= 256) return 1;
        if (h0_root <= 65536) return 2;
        return 4;
    };

    Config init_linkage_cfg = linkage_cfg;
    init_linkage_cfg.virtual_cfg.enabled = false;
    // Init-linkage-specific ICM/ILS overrides (performance-only).
    if (config.train.init_linkage_icm_round >= 0) {
        init_linkage_cfg.base.linkage.icm_round = config.train.init_linkage_icm_round;
    }
    if (config.train.init_linkage_ils_rounds >= 0) {
        init_linkage_cfg.base.linkage.ils_rounds = config.train.init_linkage_ils_rounds;
    }

    InitLinkageMode init_mode = InitLinkageMode::kLegacyRootOnly;
    {
        std::string mode_err;
        if (!ParseInitLinkageMode(config.train.init_linkage_mode, &init_mode, &mode_err)) {
            if (error) *error = "TrainQuantizerStreamingLarge: " + mode_err;
            return false;
        }
    }

    if (init_mode == InitLinkageMode::kFastInit) {
        // Seed C_one from C_root[0] using a small-k spherical k-means over root centroids.
        // This avoids the huge-h0 legacy init-linkage path and makes init-linkage isomorphic to iteration-linkage.
        if (result.C_one.books.empty()) {
            const int d = train_list.meta().d;
            const int m = config.model.m;
            const int h0_one = std::max(1, config.model.h0_one);
            const int h0_root = result.C_root.books.empty() ? 0 : result.C_root.books.front().cols;
            if (d <= 0 || m <= 1) {
                if (error) *error = "fast_init: invalid d/m.";
                return false;
            }
            if (h0_one > 256) {
                if (error) *error = "fast_init requires model.h0_one<=256 (code0_one stored as uint8).";
                return false;
            }
            if (result.C_root.books.empty() || static_cast<int>(result.C_root.books.size()) != m) {
                if (error) *error = "fast_init requires a full C_root (m books).";
                return false;
            }
            if (h0_root <= 0) {
                if (error) *error = "fast_init: invalid h0_root.";
                return false;
            }

            CodebookPack seed;
            seed.d = d;
            seed.h_vec = config.model.h_vec;
            if (static_cast<int>(seed.h_vec.size()) != m) {
                if (error) *error = "fast_init: model.h_vec size mismatch.";
                return false;
            }
            seed.h_vec[0] = h0_one;
            seed.books.resize(static_cast<std::size_t>(m));

            // Layer0: k-means map C_root[0] -> C_one[0] (or copy when sizes match).
            if (h0_one == h0_root) {
                seed.books[0] = result.C_root.books[0];
            } else {
                ScopedOmpThreads omp_scope(1);  // keep seeding stable across OMP settings
                KmeansConfig kcfg;
                kcfg.max_iters = 30;
                kcfg.tol = 1e-6f;
                kcfg.init_method = "kmeans++";
                kcfg.init_samples = std::max(1, h0_root);
                // Stable non-adaptive weights.
                kcfg.initial_weight = 1.0f;
                kcfg.min_weight = 1.0f;
                kcfg.outlier_quantile = 1.0f;
                kcfg.cost_threshold = 1e9f;
                kcfg.annealing_factor = 0.0f;
                kcfg.warmup_iters = 0;
                std::mt19937 rng(static_cast<std::mt19937::result_type>(config.train.seed ^ 0xF4171C01u));
                const KmeansResult km = SphericalKmeans(result.C_root.books.front(), h0_one, kcfg, &rng,
                                                       /*stream_kernels=*/nullptr,
                                                       /*profile_timing=*/false,
                                                       /*timing=*/nullptr);
                if (km.centers.cols != h0_one || km.centers.rows != d) {
                    if (error) *error = "fast_init: unexpected kmeans centers shape.";
                    return false;
                }
                seed.books[0] = km.centers;
            }

            // Small layers: copy from C_root (cheap seed; will be refined by the first C_one update).
            for (int l = 1; l < m; ++l) {
                seed.books[static_cast<std::size_t>(l)] = result.C_root.books[static_cast<std::size_t>(l)];
            }
            result.C_one = std::move(seed);
        }

        // Init-linkage writes a linkage_list that matches the iteration-stage store format:
        // - code0_one is always uint8 (C_one[0])
        // - small layers are uint8 (Code)
        init_linkage_store_cfg.code0_width_bytes = 1;

        const stlq::VirtualUmapReencodeEncodeConfig umap_reencode_cfg{
            .ils_iters = config.train.ils_iters,
            .icm_iters = config.train.icm_iters,
            .perturb_k = config.train.perturb_k,
            .seed = static_cast<std::uint32_t>(config.train.seed),
        };
        if (!BuildLinkageTwoCodebookVirtualStreamingByCluster(init_linkage_cfg,
                                                            result,
                                                            umap_reencode_cfg,
                                                            train_list,
                                                            ivf,
                                                            fallback_reader,
                                                            allow_random_fallback,
                                                            linkage_stream_kernels,
                                                            init_linkage_store_cfg,
                                                            /*base_depth_stats_out=*/nullptr,
                                                            /*base_linkage_mse_out=*/nullptr,
                                                            error)) {
            return false;
        }
    } else {
        // kLegacyRootOnly or kHybrid: build single-codebook init-linkage using C_root.
        // Hybrid: set VarRoot ILS rounds = 1 so that only the first GPU ILS round
        // runs full root ICM (expensive tiled GEMM over h0), subsequent rounds freeze layer0.
        [[maybe_unused]] const bool hybrid = (init_mode == InitLinkageMode::kHybrid);
#if defined(STLQ_ENABLE_CUDA)
        if (hybrid) {
            SetCudaLinkageEncodeHybridVarRootIlsRounds(config.train.init_linkage_hybrid_varroot_rounds);
        }
#endif
        init_linkage_store_cfg.code0_width_bytes = legacy_code0_width_bytes();
        const bool ok = BuildLinkageOneCodebookVirtualInitStreamingByCluster(init_linkage_cfg,
                                                                result,
                                                                train_list,
                                                                ivf,
                                                                fallback_reader,
                                                                allow_random_fallback,
                                                                linkage_stream_kernels,
                                                                init_linkage_store_cfg,
                                                                /*base_depth_stats_out=*/nullptr,
                                                                /*base_linkage_mse_out=*/nullptr,
                                                                error);
#if defined(STLQ_ENABLE_CUDA)
        if (hybrid) {
            ResetCudaLinkageEncodeHybridVarRootIlsRounds();
        }
#endif
        if (!ok) {
            return false;
        }
    }
    LogInfo("Init-linkage wall time(s): " + FormatFloat(static_cast<float>(init_linkage_wall.ElapsedSeconds()), 3));
#if defined(STLQ_ENABLE_CUDA)
    LogCudaVramIfEnabled(config, "after init-linkage");
#endif

    io::LinkageListReader init_linkage_list;
    if (!init_linkage_list.Open(train_paths.train_linkage_init_list_dir.string(), error)) {
        return false;
    }

    double baseline = 0.0;
    const stlq::Timer init_badmask_wall;
    if (!config.virtual_cfg.enabled) {
        result.is_bad_cluster.clear();
    } else if (init_mode == InitLinkageMode::kFastInit) {
        if (!ComputeBadClusterMaskFromLinkageListStreaming(config,
                                                         result,
                                                         train_list,
                                                         ivf,
                                                         init_linkage_list,
                                                         fallback_reader,
                                                         allow_random_fallback,
                                                         linkage_stream_kernels,
                                                         &result.is_bad_cluster,
                                                         &baseline,
                                                         error)) {
            LogWarn("Failed to compute is_bad_cluster from init linkage_list; virtual node weighting disabled.");
            result.is_bad_cluster.clear();
        } else {
            LogInfo(init_badmask_wall.ReportSeconds("Init-linkage bad-mask+MSE (C_root only) wall time(s)"));
        }
    } else {
        if (!ComputeBadClusterMaskFromInitLinkageListStreaming(config,
                                                             result,
                                                             train_list,
                                                             ivf,
                                                             init_linkage_list,
                                                             fallback_reader,
                                                             allow_random_fallback,
                                                             linkage_stream_kernels,
                                                             &result.is_bad_cluster,
                                                             &baseline,
                                                             error)) {
            LogWarn("Failed to compute is_bad_cluster from init linkage_list; virtual node weighting disabled.");
            result.is_bad_cluster.clear();
        } else {
            LogInfo(init_badmask_wall.ReportSeconds("Init-linkage bad-mask+MSE (C_root only) wall time(s)"));
        }
    }

    const stlq::Timer init_update_c1_wall;
    if (init_mode == InitLinkageMode::kFastInit) {
        if (!UpdateCOneFromTrainLinkageListStreaming(config,
                                                   result,
                                                   train_list,
                                                   ivf,
                                                   init_linkage_list,
                                                   fallback_reader,
                                                   allow_random_fallback,
                                                   linkage_stream_kernels,
                                                   &result.C_one,
                                                   error)) {
            return false;
        }
    } else {
        if (!UpdateCOneFromInitLinkageListStreamingExactLS(config,
                                                         result,
                                                         train_list,
                                                         ivf,
                                                         init_linkage_list,
                                                         fallback_reader,
                                                         allow_random_fallback,
                                                         linkage_stream_kernels,
                                                         &result.C_one,
                                                         error)) {
            return false;
        }
    }
    LogInfo(init_update_c1_wall.ReportSeconds("Init-linkage update C_one wall time(s)"));

    if (config.train.log_metrics) {
        LinkageDepthStats stats;
        MseStats mse;
        const stlq::Timer init_mse_wall;
        if (!AnalyzeTrainLinkageListAfterCOneUpdateStreaming(config,
                                                          result,
                                                          train_list,
                                                          ivf,
                                                          init_linkage_list,
                                                          /*code0_one_bytes_are_root_codes=*/(init_mode != InitLinkageMode::kFastInit),
                                                          fallback_reader,
                                                          allow_random_fallback,
                                                          linkage_stream_kernels,
                                                          &stats,
                                                          &mse,
                                                          /*opq_M_out=*/nullptr,
                                                          error)) {
            return false;
        }
        LogInfo(init_mse_wall.ReportSeconds("Init-linkage linkage-summary+MSE analysis wall time(s)"));
        if (init_mode == InitLinkageMode::kFastInit) {
            LogInfo("Train init-linkage encode analysis after updating C_one:");
        } else {
            LogInfo("Train init-linkage encode analysis after updating C_one "
                    "(note: init-linkage stores layer0 as C_root codes; if h0_one<h0_root "
                    "this analysis maps root->one, so treat as a reference metric):");
        }
        LogInfo("  Linkage Ratio: " + FormatFloat(100.0f * stats.linkage_ratio, 2) + "%");
        LogInfo("  Max depth: " + std::to_string(stats.max_depth) +
                " Mean depth: " + FormatFloat(stats.mean_depth, 2));
        LogInfo("  Linkage Error: Max=" + FormatFloat(mse.max, 4) +
                ", Min=" + FormatFloat(mse.min, 4) +
                ", Mean=" + FormatFloat(mse.mean, 4));
    }
    return true;
}

}  // namespace stlq
