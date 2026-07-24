#include "stlq/quantizer/streaming_train_config.h"

#include <algorithm>
#include <cstdint>
#include <system_error>

namespace stlq {

KmeansConfig MakeStreamingTrainKmeansConfig(const Config& config, bool train_is_u8) {
    KmeansConfig kcfg;
    kcfg.max_iters = std::max(1, config.train.kmeans_iters);
    kcfg.tol = static_cast<float>(config.train.kmeans_tol);
    kcfg.init_method = config.train.kmeans_init;
    kcfg.init_samples = std::max(1, config.train.init_samples);
    kcfg.kmeansll_gpu_enable = config.train.kmeansll_gpu_enable;
    kcfg.kmeansll_rounds = std::min(16, std::max(1, config.train.kmeansll_rounds));
    kcfg.kmeansll_oversample = static_cast<float>(std::min(16.0, std::max(1.0, config.train.kmeansll_oversample)));
    kcfg.kmeansll_candidate_cap = std::max(0, config.train.kmeansll_candidate_cap);
    kcfg.kmeansll_seed = config.train.kmeansll_seed;
    kcfg.kmeansll_hier_enable = config.train.kmeansll_hier_enable;
    kcfg.kmeansll_hier_threshold = std::max(1, config.train.kmeansll_hier_threshold);
    kcfg.kmeansll_hier_fixed_k1 = config.train.kmeansll_hier_fixed_k1;
    kcfg.initial_weight = static_cast<float>(config.train.kmeans_initial_weight);
    kcfg.min_weight = static_cast<float>(config.train.kmeans_min_weight);
    kcfg.outlier_quantile = static_cast<float>(config.train.kmeans_outlier_quantile);
    kcfg.cost_threshold = static_cast<float>(config.train.kmeans_cost_threshold);
    kcfg.annealing_factor = static_cast<float>(config.train.kmeans_annealing_factor);
    kcfg.warmup_iters = std::max(0, config.train.kmeans_warmup_iters);
    kcfg.quantile_bins = std::max(1, config.train.kmeans_quantile_bins);
    kcfg.anneal_no_spill = config.train.kmeans_anneal_no_spill;
    kcfg.anneal_mode = (config.train.kmeans_anneal_mode == "twopass") ? KmeansAnnealMode::kTwoPass
                                                                      : KmeansAnnealMode::kOnePass;
    kcfg.anneal_weights_device = config.train.kmeans_anneal_weights_device;
    kcfg.force_disable_device_weights = config.train.kmeans_force_disable_device_weights;
    kcfg.streaming = true;
    kcfg.cache_xnorm_device = config.train.kmeans_cache_xnorm_device;
    kcfg.device_cache_mb = std::max(0, config.train.kmeans_device_cache_mb);
    kcfg.pin_host_x = config.train.kmeans_pin_host_x;
    kcfg.bvecs_use_u8 = train_is_u8 && config.train.kmeans_bvecs_use_u8;
    kcfg.block_cols = std::max(1, config.large.train_block);
    kcfg.tmp_dir = !config.train.kmeans_tmp_dir.empty() ? config.train.kmeans_tmp_dir : config.large.tmp_dir;
    kcfg.collect_metrics = config.train.log_metrics;
    kcfg.large_k_threshold = std::max(1, config.train.kmeans_large_k_threshold);
    kcfg.large_k_hier2_enable = config.train.kmeans_large_k_hier2_enable;
    kcfg.large_k_fixed_k1 = std::max(0, config.train.kmeans_large_k_fixed_k1);
    kcfg.large_k_k1_min = std::max(1, config.train.kmeans_large_k_k1_min);
    kcfg.large_k_k1_max = std::max(kcfg.large_k_k1_min, config.train.kmeans_large_k_k1_max);
    kcfg.large_k_k1_pow2 = config.train.kmeans_large_k_k1_pow2;
    kcfg.large_k_top_coarse = std::max(1, config.train.kmeans_large_k_top_coarse);
    kcfg.large_k_hier2_train = config.train.kmeans_large_k_hier2_train;
    kcfg.large_k_hier2_encode = config.train.kmeans_large_k_hier2_encode;
    return kcfg;
}

Config MakeTrainBasicEncodeConfig(const Config& config, std::uint64_t ntrain) {
    Config enc_cfg = config;
    enc_cfg.large.base_block = std::max(1, config.large.train_block);
    enc_cfg.dataset.nbase_set = true;
    enc_cfg.dataset.nbase = static_cast<int>(ntrain);
    enc_cfg.base.encode.use_abs = false;
    enc_cfg.base.encode.icm_iters = std::max(0, config.train.icm_iters);
    enc_cfg.base.encode.ils_iters = std::max(0, config.train.ils_iters);
    enc_cfg.base.encode.perturb_k = std::max(0, config.train.perturb_k);
    enc_cfg.base.encode.seed = config.train.seed;
    return enc_cfg;
}

io::BaseBasicStoreConfig MakeTrainBasicStoreConfig(const Config& config,
                                                   const std::filesystem::path& train_basic_dir,
                                                   int d,
                                                   bool is_u8) {
    io::BaseBasicStoreConfig store_cfg;
    store_cfg.dir = train_basic_dir.string();
    store_cfg.d = d;
    store_cfg.m = config.model.m;
    store_cfg.h_vec = config.model.h_vec;
    store_cfg.shard_size = std::max(1, config.large.base_shard_size);
    store_cfg.bucket_size = std::max(1, config.large.cluster_bucket_size);
    store_cfg.bucket_flush_mb = std::max(1, config.large.bucket_flush_mb);
    store_cfg.write_vector_bucket = config.large.write_vector_bucket;
    store_cfg.write_basic_to_bucket = config.large.write_basic_to_bucket;
    if (store_cfg.write_vector_bucket) {
        store_cfg.raw_bytes_per_vec = is_u8 ? d : (d * static_cast<int>(sizeof(float)));
    }
    return store_cfg;
}

io::LinkageListStoreConfig MakeTrainLinkageListStoreConfig(const Config& config,
                                                       const std::filesystem::path& train_linkage_list_dir,
                                                       int nlist) {
    io::LinkageListStoreConfig linkage_store_cfg;
    linkage_store_cfg.dir = train_linkage_list_dir.string();
    linkage_store_cfg.nlist = nlist;
    linkage_store_cfg.m_codes = std::max(0, config.model.m - 1);
    // code0_one (C_one[0]) is stored as uint8 in the large-scale pipeline.
    linkage_store_cfg.code0_width_bytes = 1;
    // Training-stage linkage_list must keep float coeffs for updating C_one.
    linkage_store_cfg.store_coeffs_f32 = true;
    // Training/iteration linkage_list does not need parent LOUDS (baseset disk eval only).
    linkage_store_cfg.store_parent_louds = false;
    linkage_store_cfg.store_parent_u32 = true;
    return linkage_store_cfg;
}

Config MakeTrainLinkageConfig(const Config& config) {
    Config linkage_cfg = config;
    linkage_cfg.base.linkage = config.train.linkage;
    // Do not enable coeff codec for train_linkage_list: UpdateCOneFromTrainLinkageListStreaming reads float coeffs.
    linkage_cfg.large.linkage_coeff_codec.enabled = false;
    // Training-stage linkage_list is heavily reused by update stages; keep parent.u32 for speed.
    linkage_cfg.large.linkage_store_parent_u32 = true;
    return linkage_cfg;
}

StreamingTrainWorkspacePaths ResolveStreamingTrainWorkspacePaths(const Config& config,
                                                                 const std::filesystem::path& exp_root,
                                                                 const TrainResult& result) {
    StreamingTrainWorkspacePaths paths;

    // Default: use the computed large workspace root for this run.
    //
    // When resuming from a pre-init-linkage checkpoint, prefer meta/train/exp_root_hint
    // so that init-linkage can reuse the already-built train_basic/ivf/list without rebuilding.
    paths.exp_root_run = exp_root;
    if (!config.train.init_enabled && !result.exp_root_hint.empty()) {
        paths.exp_root_run = std::filesystem::path(result.exp_root_hint);
    }

    paths.train_basic_dir = paths.exp_root_run / "train_basic";
    paths.train_ivf_dir = paths.exp_root_run / "train_ivf";
    paths.train_list_dir = paths.exp_root_run / "train_list";
    paths.train_linkage_list_dir = paths.exp_root_run / "train_linkage_list";
    paths.train_linkage_init_list_dir = paths.exp_root_run / "train_linkage_init_list";
    return paths;
}

void EnsureStreamingTrainWorkspaceDirs(const StreamingTrainWorkspacePaths& paths) {
    std::error_code ec;
    std::filesystem::create_directories(paths.train_basic_dir, ec);
    std::filesystem::create_directories(paths.train_ivf_dir, ec);
    std::filesystem::create_directories(paths.train_list_dir, ec);
    std::filesystem::create_directories(paths.train_linkage_list_dir, ec);
    std::filesystem::create_directories(paths.train_linkage_init_list_dir, ec);
}

}  // namespace stlq
