#include "stlq/common/config.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <unordered_set>
#include <sstream>

namespace stlq
{
    namespace
    {
        std::string Trim(const std::string& input) {
            std::size_t start = 0;
            while (start < input.size() && std::isspace(static_cast<unsigned char>(input[start]))) {
                ++start;
            }
            std::size_t end = input.size();
            while (end > start && std::isspace(static_cast<unsigned char>(input[end - 1]))) {
                --end;
            }
            return input.substr(start, end - start);
        }

        std::string StripQuotes(const std::string& input) {
            if (input.size() >= 2 && ((input.front() == '"' && input.back() == '"') ||
                (input.front() == '\'' && input.back() == '\''))) {
                return input.substr(1, input.size() - 2);
            }
            return input;
        }

        bool ParseBool(const std::string& value, bool* out) {
            std::string v = value;
            std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return std::tolower(c); });
            if (v == "true" || v == "1") {
                *out = true;
                return true;
            }
            if (v == "false" || v == "0") {
                *out = false;
                return true;
            }
            return false;
        }

        bool ParseInt(const std::string& value, int* out) {
            try {
                std::size_t idx = 0;
                int parsed = std::stoi(value, &idx);
                if (idx != value.size()) {
                    return false;
                }
                *out = parsed;
                return true;
            }
            catch (...) {
                return false;
            }
        }

        bool ParseFloat(const std::string& value, float* out) {
            try {
                std::size_t idx = 0;
                float parsed = std::stof(value, &idx);
                if (idx != value.size()) {
                    return false;
                }
                *out = parsed;
                return true;
            }
            catch (...) {
                return false;
            }
        }

        bool ParseDouble(const std::string& value, double* out) {
            try {
                std::size_t idx = 0;
                double parsed = std::stod(value, &idx);
                if (idx != value.size()) {
                    return false;
                }
                *out = parsed;
                return true;
            }
            catch (...) {
                return false;
            }
        }

        bool ParseUInt64(const std::string& value, std::uint64_t* out) {
            try {
                std::size_t idx = 0;
                unsigned long long parsed = std::stoull(value, &idx, 0);
                if (idx != value.size()) {
                    return false;
                }
                *out = static_cast<std::uint64_t>(parsed);
                return true;
            }
            catch (...) {
                return false;
            }
        }

        bool ParseIntList(const std::string& value, std::vector<int>* out, bool allow_empty) {
            std::string v = Trim(value);
            if (v.empty()) {
                return false;
            }
            if (v.front() == '[' && v.back() == ']') {
                v = v.substr(1, v.size() - 2);
            }
            std::vector<int> result;
            std::stringstream ss(v);
            std::string item;
            while (std::getline(ss, item, ',')) {
                item = Trim(item);
                if (item.empty()) {
                    continue;
                }
                int parsed = 0;
                if (!ParseInt(item, &parsed)) {
                    return false;
                }
                result.push_back(parsed);
            }
            if (result.empty()) {
                if (!allow_empty) {
                    return false;
                }
                out->clear();
                return true;
            }
            *out = std::move(result);
            return true;
        }

        bool ParseIntList(const std::string& value, std::vector<int>* out) {
            return ParseIntList(value, out, /*allow_empty=*/false);
        }

        bool ParseDoubleList(const std::string& value, std::vector<double>* out, bool allow_empty) {
            std::string v = Trim(value);
            if (v.empty()) {
                return false;
            }
            if (v.front() == '[' && v.back() == ']') {
                v = v.substr(1, v.size() - 2);
            }
            std::vector<double> result;
            std::stringstream ss(v);
            std::string item;
            while (std::getline(ss, item, ',')) {
                item = Trim(item);
                if (item.empty()) {
                    continue;
                }
                double parsed = 0.0;
                if (!ParseDouble(item, &parsed)) {
                    return false;
                }
                result.push_back(parsed);
            }
            if (result.empty()) {
                if (!allow_empty) {
                    return false;
                }
                out->clear();
                return true;
            }
            *out = std::move(result);
            return true;
        }

        bool ParseDoubleList(const std::string& value, std::vector<double>* out) {
            return ParseDoubleList(value, out, /*allow_empty=*/false);
        }

        bool ParseKeyValueLine(const std::string& line, std::string* key, std::string* value) {
            auto pos = line.find('=');
            if (pos == std::string::npos) {
                return false;
            }
            *key = Trim(line.substr(0, pos));
            *value = Trim(line.substr(pos + 1));
            return !key->empty();
        }

        std::string DefaultTrainResultExtension(const IOConfig& io) {
            if (io.result_format == "bin") {
                return ".stlqbin";
            }
            return ".h5";
        }

        std::string DefaultLegacyResultExtension(const IOConfig& io) {
            return io.result_format == "bin" ? ".stlqbin" : ".h5";
        }

        std::string DatasetDefaultsHint(const std::string& name, Config* config) {
            if (name == "SIFTSMALL") {
                if (!config->dataset.ntrain_set) {
                    config->dataset.ntrain = 25000;
                }
                if (!config->dataset.nquery_set) {
                    config->dataset.nquery = 100;
                }
                if (!config->dataset.nbase_set) {
                    config->dataset.nbase = 10000;
                }
                if (!config->dataset.k_set) {
                    config->dataset.k = 100;
                }
                return "SIFTSMALL";
            }
            if (name == "SIFT1M" || name == "Deep1M" || name == "DEEP1M" || name == "Convnet1M") {
                if (!config->dataset.ntrain_set) {
                    config->dataset.ntrain = 100000;
                }
                if (!config->dataset.nquery_set) {
                    config->dataset.nquery = 10000;
                }
                if (!config->dataset.nbase_set) {
                    config->dataset.nbase = 1000000;
                }
                if (!config->dataset.k_set) {
                    config->dataset.k = 100;
                }
                return "SIFT1M";
            }
            if (name == "GIST1M") {
                if (!config->dataset.ntrain_set) {
                    config->dataset.ntrain = 100000;
                }
                if (!config->dataset.nquery_set) {
                    config->dataset.nquery = 1000;
                }
                if (!config->dataset.nbase_set) {
                    config->dataset.nbase = 1000000;
                }
                if (!config->dataset.k_set) {
                    config->dataset.k = 100;
                }
                return "GIST1M";
            }
            if (name == "MSONG") {
                if (!config->dataset.ntrain_set) {
                    config->dataset.ntrain = 100000;
                }
                if (!config->dataset.nquery_set) {
                    config->dataset.nquery = 1000;
                }
                if (!config->dataset.nbase_set) {
                    config->dataset.nbase = 994185;
                }
                if (!config->dataset.k_set) {
                    config->dataset.k = 100;
                }
                return "MSONG";
            }
            if (name == "GLOVE100") {
                if (!config->dataset.ntrain_set) {
                    config->dataset.ntrain = 100000;
                }
                if (!config->dataset.nquery_set) {
                    config->dataset.nquery = 10000;
                }
                if (!config->dataset.nbase_set) {
                    config->dataset.nbase = 1183514;
                }
                if (!config->dataset.k_set) {
                    config->dataset.k = 100;
                }
                return "GLOVE100";
            }
            return "custom";
        }
    } // namespace

    void FinalizeLinkageCoeffCodecConfig(Config* config) {
        auto& ccfg = config->large.linkage_coeff_codec;
        // Normalize granularity.
        std::string g = ccfg.granularity;
        std::transform(g.begin(), g.end(), g.begin(), [](unsigned char c) { return std::tolower(c); });
        if (g == "per_cluster" || g == "cluster" || g == "all" || g == "all_layers") {
            ccfg.granularity = "cluster";
        }
        else if (g == "per_layer" || g == "layer") {
            ccfg.granularity = "layer";
        }
        else {
            // Keep user value (will be validated at use sites).
            ccfg.granularity = g;
        }

        const int m = std::max(1, config->model.m);
        if (ccfg.bits_per_layer.empty()) {
            ccfg.bits_per_layer.assign(1, 7);
        }
        // Normalize bits_per_layer length to exactly m.
        if (ccfg.bits_per_layer.size() == 1 && m > 1) {
            ccfg.bits_per_layer.assign(static_cast<std::size_t>(m), ccfg.bits_per_layer.front());
        }
        else if (ccfg.bits_per_layer.size() == 2 && m > 2) {
            const int b0 = ccfg.bits_per_layer[0];
            const int b1 = ccfg.bits_per_layer[1];
            ccfg.bits_per_layer.assign(static_cast<std::size_t>(m), b1);
            ccfg.bits_per_layer[0] = b0;
        }
        else if (ccfg.bits_per_layer.size() < static_cast<std::size_t>(m)) {
            ccfg.bits_per_layer.resize(static_cast<std::size_t>(m), ccfg.bits_per_layer.back());
        }
        else if (ccfg.bits_per_layer.size() > static_cast<std::size_t>(m)) {
            ccfg.bits_per_layer.resize(static_cast<std::size_t>(m));
        }
    }

    ResolvedCleanupFlags ResolveCleanupFlags(const LargeScaleConfig& large) {
        ResolvedCleanupFlags f;

        if (!large.cleanup.enabled) {
            // Everything kept.
            return f;
        }

        // Apply preset defaults.
        // NOTE: Presets are intentionally *progressive* (monotonic in deletion):
        //   none < dev_all < eval_both < eval_float_min < eval_int8_min
        // so that later presets never delete less than earlier ones.
        const std::string& preset = large.cleanup.preset;

        auto apply_dev_all = [&]()
        {
            // Minimal cleanup: only delete base_basic payload (bucket/codes_shard/coeffs_shard).
            f.keep_base_basic = false;
            // Also drop cluster_id.u32 by default to save disk once base_list + ivf CSR are ready.
            // Users can keep it explicitly when they want to rebuild ivf CSR without re-encoding.
            f.keep_base_basic_cluster_id = false;
        };
        auto apply_eval_both = [&]()
        {
            // Supports both float and int8 linkage eval.
            // Deletes base_basic payload and the derived base_list store (including raw vectors).
            apply_dev_all();
            // Drop cluster_id.u32 to save disk; disk linkage recall uses linkage_list only.
            f.keep_base_basic_cluster_id = false;
            // Drop IVF CSR lists by default for linkage-only eval presets.
            f.keep_base_basic_ivf_lists = false;
            f.keep_base_list = false;
            f.keep_base_list_raw = false;
            f.keep_linkage_list_f32 = true;
            f.keep_linkage_parent_u32 = true;
        };
        auto apply_eval_float_min = [&]()
        {
            // Minimum footprint for float-only linkage eval.
            // Same as eval_both but allows dropping parent.u32 when LOUDS exists.
            apply_eval_both();
            f.keep_linkage_parent_u32 = false;
        };
        auto apply_eval_int8_min = [&]()
        {
            // Minimum footprint for int8-only linkage eval (requires coeff codec).
            // Same as eval_float_min but also deletes linkage_list/*.f32 coeff payload.
            apply_eval_float_min();
            f.keep_linkage_list_f32 = false;
            f.keep_linkage_parent_u32 = false;
        };

        if (preset == "none") {
            // No deletion — all kept (same as disabled).
        }
        else if (preset == "dev_all") {
            apply_dev_all();
        }
        else if (preset == "eval_both") {
            apply_eval_both();
        }
        else if (preset == "eval_float_min") {
            apply_eval_float_min();
        }
        else if (preset == "eval_int8_min") {
            apply_eval_int8_min();
        }
        else if (preset == "custom") {
            // All true by default; user must provide explicit keep_* flags to delete.
        }
        // else: unknown preset, treat as all-keep for safety.

        // Override with explicit flags (-1 = use preset default, 0 = delete, 1 = keep).
        const auto& c = large.cleanup;
        auto apply = [](bool& flag, int explicit_val)
        {
            if (explicit_val == 0) flag = false;
            else if (explicit_val == 1) flag = true;
            // -1: keep preset default.
        };
        apply(f.keep_base_basic, c.keep_base_basic);
        apply(f.keep_base_basic_cluster_id, c.keep_base_basic_cluster_id);
        apply(f.keep_base_basic_ivf_lists, c.keep_base_basic_ivf_lists);
        apply(f.keep_base_list, c.keep_base_list);
        apply(f.keep_base_list_raw, c.keep_base_list_raw);
        apply(f.keep_linkage_list_f32, c.keep_linkage_list_f32);
        apply(f.keep_linkage_parent_u32, c.keep_linkage_parent_u32);

        return f;
    }

    Config DefaultConfig(bool virtual_mode) {
        Config config;
        config.model.h_vec = {256, 256, 256, 256, 256};
        config.model.h_vec_set = false;
        config.virtual_cfg.enabled = virtual_mode;
        if (virtual_mode) {
            config.io.pre_fix = "umap-reg";
            config.train.max_R_iters = 60;
            config.train.ils_iters = 60;
            config.base.encode.ils_iters = 200;
            config.base.linkage.num_layers = 12;
            config.base.linkage.knn_k = 20;
            config.base.linkage.depth_k = 15;
            config.train.linkage.num_layers = 12;
            config.train.linkage.knn_k = 20;
            config.train.linkage.depth_k = 15;
            config.train.linkage.root_percentile = 0.010;
            config.virtual_cfg.virtual_ratio = 0.10;
            config.virtual_cfg.good_fraction = 0.6;
            config.virtual_cfg.alpha_bad = 0.3f;
        }
        ApplyDatasetDefaults(&config);
        FinalizeLinkageCoeffCodecConfig(&config);
        return config;
    }

    void NormalizeModelHVec(Config* config) {
        if (!config) {
            return;
        }
        const int m = std::max(1, config->model.m);
        config->model.m = m;
        if (!config->model.h_vec_set) {
            config->model.h_vec.assign(static_cast<std::size_t>(m), 256);
            config->model.h_vec[0] = config->model.h0;
            return;
        }
        if (config->model.h_vec.empty()) {
            config->model.h_vec.assign(static_cast<std::size_t>(m), 256);
            config->model.h_vec[0] = config->model.h0;
            return;
        }
        if (static_cast<int>(config->model.h_vec.size()) < m) {
            const int last = config->model.h_vec.back();
            config->model.h_vec.resize(static_cast<std::size_t>(m), last);
        } else if (static_cast<int>(config->model.h_vec.size()) > m) {
            config->model.h_vec.resize(static_cast<std::size_t>(m));
        }
        config->model.h0 = config->model.h_vec.front();
    }

    void ApplyDatasetDefaults(Config* config) {
        DatasetDefaultsHint(config->dataset.name, config);

        if (!config->io.train_file_set) {
            config->io.train_file = "./results/" + config->dataset.name + "/" + config->io.pre_fix + "/" +
                config->io.pre_fix + "_train" + DefaultTrainResultExtension(config->io);
        }
        if (!config->io.base_file_set) {
            config->io.base_file = "./results/" + config->dataset.name + "/" + config->io.pre_fix + "/" +
                config->io.pre_fix + "_base" + DefaultLegacyResultExtension(config->io);
        }
        if (!config->io.linkage_file_set) {
            config->io.linkage_file = "./results/" + config->dataset.name + "/" + config->io.pre_fix + "/" +
                config->io.pre_fix + "_linkage" + DefaultLegacyResultExtension(config->io);
        }
    }

    bool LoadConfigFile(const std::string& path, Config* config, std::string* error,
                        std::vector<int>* linkage_nprobe_batch,
                        std::vector<int>* linkage_ef_search_batch) {
        std::ifstream file(path);
        if (!file.is_open()) {
            if (error) {
                *error = "Failed to open config file: " + path;
            }
            return false;
        }
        std::string line;
        int line_number = 0;
        while (std::getline(file, line)) {
            ++line_number;
            auto comment_pos = line.find('#');
            if (comment_pos != std::string::npos) {
                line = line.substr(0, comment_pos);
            }
            comment_pos = line.find("//");
            if (comment_pos != std::string::npos) {
                line = line.substr(0, comment_pos);
            }
            line = Trim(line);
            if (line.empty()) {
                continue;
            }
            std::string key;
            std::string value;
            if (!ParseKeyValueLine(line, &key, &value)) {
                if (error) {
                    *error = "Invalid line " + std::to_string(line_number) + ": " + line;
                }
                return false;
            }
            if (!ApplyOverride(key, value, config, error, linkage_nprobe_batch, linkage_ef_search_batch)) {
                if (error && error->empty()) {
                    *error = "Invalid config entry at line " + std::to_string(line_number);
                }
                return false;
            }
        }
        ApplyDatasetDefaults(config);
        NormalizeModelHVec(config);
        FinalizeLinkageCoeffCodecConfig(config);
        return true;
    }

    bool ApplyOverride(const std::string& key, const std::string& value, Config* config, std::string* error,
                       std::vector<int>* linkage_nprobe_batch,
                       std::vector<int>* linkage_ef_search_batch) {
        const std::string v = StripQuotes(value);
        if (key == "runtime.omp_threads") {
            return ParseInt(v, &config->runtime.omp_threads);
        }
        if (key == "runtime.use_cuda") {
            return ParseBool(v, &config->runtime.use_cuda);
        }
        if (key == "runtime.cuda_device") {
            return ParseInt(v, &config->runtime.cuda_device);
        }
        if (key == "runtime.cuda_mode") {
            config->runtime.cuda_mode = v;
            return true;
        }
        if (key == "runtime.cuda_allow_tf32") {
            return ParseBool(v, &config->runtime.cuda_allow_tf32);
        }
        if (key == "runtime.cuda_pool_size") {
            return ParseInt(v, &config->runtime.cuda_pool_size);
        }
        if (key == "runtime.cuda_pool_size_init_linkage") {
            return ParseInt(v, &config->runtime.cuda_pool_size_init_linkage);
        }
        if (key == "runtime.cuda_cublas_workspace_mb") {
            return ParseInt(v, &config->runtime.cuda_cublas_workspace_mb);
        }
        if (key == "runtime.cuda_linkage_min_kp") {
            return ParseInt(v, &config->runtime.cuda_linkage_min_kp);
        }
        if (key == "runtime.cuda_linkage_max_kp") {
            return ParseInt(v, &config->runtime.cuda_linkage_max_kp);
        }
        if (key == "runtime.cuda_linkage_use_device_rfull") {
            return ParseBool(v, &config->runtime.cuda_linkage_use_device_rfull);
        }
        if (key == "runtime.cuda_linkage_batch_inner_enabled") {
            return ParseBool(v, &config->runtime.cuda_linkage_batch_inner_enabled);
        }
        if (key == "runtime.cuda_linkage_batch_nodes") {
            return ParseInt(v, &config->runtime.cuda_linkage_batch_nodes);
        }
        if (key == "runtime.cuda_linkage_batch_max_pairs") {
            return ParseInt(v, &config->runtime.cuda_linkage_batch_max_pairs);
        }
        if (key == "runtime.cuda_linkage_batch_max_pairs_init_linkage") {
            return ParseInt(v, &config->runtime.cuda_linkage_batch_max_pairs_init_linkage);
        }
        if (key == "runtime.cuda_linkage_batch_min_pairs") {
            return ParseInt(v, &config->runtime.cuda_linkage_batch_min_pairs);
        }
        if (key == "runtime.linkage_async_io") {
            return ParseBool(v, &config->runtime.linkage_async_io);
        }
        if (key == "runtime.linkage_async_io_depth") {
            return ParseInt(v, &config->runtime.linkage_async_io_depth);
        }
        if (key == "runtime.basic_async_io") {
            return ParseBool(v, &config->runtime.basic_async_io);
        }
        if (key == "runtime.basic_async_io_depth") {
            return ParseInt(v, &config->runtime.basic_async_io_depth);
        }
        if (key == "runtime.basic_async_io_mb") {
            return ParseInt(v, &config->runtime.basic_async_io_mb);
        }
        if (key == "runtime.basic_async_write") {
            return ParseBool(v, &config->runtime.basic_async_write);
        }
        if (key == "runtime.basic_async_write_depth") {
            return ParseInt(v, &config->runtime.basic_async_write_depth);
        }
        if (key == "runtime.basic_async_write_mb") {
            return ParseInt(v, &config->runtime.basic_async_write_mb);
        }
        if (key == "runtime.precomp_large_root_g0s_transpose") {
            return ParseBool(v, &config->runtime.precomp_large_root_g0s_transpose);
        }
        if (key == "runtime.precomp_large_root_g0s_transpose_max_mb") {
            return ParseInt(v, &config->runtime.precomp_large_root_g0s_transpose_max_mb);
        }
        if (key == "runtime.c_one_update_shards") {
            return ParseInt(v, &config->runtime.c_one_update_shards);
        }
        if (key == "runtime.c_one_update_shards_init_linkage") {
            return ParseInt(v, &config->runtime.c_one_update_shards_init_linkage);
        }
        if (key == "runtime.basic_hybrid_enable") {
            return ParseBool(v, &config->runtime.basic_hybrid_enable);
        }
        if (key == "runtime.basic_hybrid_cpu_stride") {
            return ParseInt(v, &config->runtime.basic_hybrid_cpu_stride);
        }
        if (key == "runtime.basic_hybrid_cpu_threads") {
            return ParseInt(v, &config->runtime.basic_hybrid_cpu_threads);
        }
        if (key == "runtime.basic_hybrid_reorder_depth") {
            return ParseInt(v, &config->runtime.basic_hybrid_reorder_depth);
        }
        if (key == "runtime.basic_hybrid_inflight_mb") {
            return ParseInt(v, &config->runtime.basic_hybrid_inflight_mb);
        }
        if (key == "runtime.cuda_linkage_min_kp" ||
            key == "runtime.cuda_linkage_single_gpu_min_candidates") {
            return ParseInt(v, &config->runtime.cuda_linkage_min_kp);
        }
        if (key == "runtime.cuda_linkage_max_kp" ||
            key == "runtime.cuda_linkage_single_gpu_max_candidates") {
            return ParseInt(v, &config->runtime.cuda_linkage_max_kp);
        }
        if (key == "runtime.cuda_linkage_batch_inner_enabled" ||
            key == "runtime.cuda_linkage_inner_many_nodes_enable") {
            return ParseBool(v, &config->runtime.cuda_linkage_batch_inner_enabled);
        }
        if (key == "runtime.cuda_linkage_batch_nodes" ||
            key == "runtime.cuda_linkage_inner_many_nodes_target_nodes") {
            return ParseInt(v, &config->runtime.cuda_linkage_batch_nodes);
        }
        if (key == "runtime.cuda_linkage_batch_max_pairs" ||
            key == "runtime.cuda_linkage_many_nodes_max_pairs") {
            return ParseInt(v, &config->runtime.cuda_linkage_batch_max_pairs);
        }
        if (key == "runtime.cuda_linkage_batch_max_pairs_init_linkage" ||
            key == "runtime.cuda_linkage_many_nodes_max_pairs_init_linkage") {
            return ParseInt(v, &config->runtime.cuda_linkage_batch_max_pairs_init_linkage);
        }
        if (key == "runtime.cuda_linkage_batch_min_pairs" ||
            key == "runtime.cuda_linkage_many_nodes_preflush_target_pairs") {
            return ParseInt(v, &config->runtime.cuda_linkage_batch_min_pairs);
        }
        if (key == "runtime.cuda_linkage_same_dynamic_min_pairs" ||
            key == "runtime.cuda_linkage_same_layer_window_min_pairs") {
            return ParseInt(v, &config->runtime.cuda_linkage_same_dynamic_min_pairs);
        }
        if (key == "runtime.cuda_linkage_same_dynamic_tiny_cpu" ||
            key == "runtime.cuda_linkage_same_layer_tiny_forced_cpu_enable") {
            return ParseBool(v, &config->runtime.cuda_linkage_same_dynamic_tiny_cpu);
        }
        if (key == "runtime.cuda_linkage_init_large_root_tiny_cpu" ||
            key == "runtime.cuda_linkage_init_same_layer_tiny_forced_cpu_enable") {
            return ParseBool(v, &config->runtime.cuda_linkage_init_large_root_tiny_cpu);
        }
        if (key == "runtime.cuda_linkage_same_dynamic_tiny_cpu_max_pairs" ||
            key == "runtime.cuda_linkage_same_layer_tiny_forced_cpu_max_pairs") {
            return ParseInt(v, &config->runtime.cuda_linkage_same_dynamic_tiny_cpu_max_pairs);
        }
        if (key == "runtime.cuda_linkage_same_dynamic_tiny_cpu_max_nodes" ||
            key == "runtime.cuda_linkage_same_layer_tiny_forced_cpu_max_nodes") {
            return ParseInt(v, &config->runtime.cuda_linkage_same_dynamic_tiny_cpu_max_nodes);
        }
        if (key == "runtime.cuda_linkage_same_dynamic_nodes" ||
            key == "runtime.cuda_linkage_same_layer_block_nodes") {
            return ParseInt(v, &config->runtime.cuda_linkage_same_dynamic_nodes);
        }
        if (key == "runtime.cuda_linkage_same_dynamic_min_kp" ||
            key == "runtime.cuda_linkage_same_layer_single_gpu_min_candidates") {
            return ParseInt(v, &config->runtime.cuda_linkage_same_dynamic_min_kp);
        }
        if (key == "runtime.cuda_linkage_same_dynamic_window_batch" ||
            key == "runtime.cuda_linkage_same_layer_preflush_single_shot") {
            return ParseBool(v, &config->runtime.cuda_linkage_same_dynamic_window_batch);
        }
        if (key == "runtime.cuda_linkage_same_dynamic_window_max_pairs" ||
            key == "runtime.cuda_linkage_same_layer_window_max_pairs") {
            return ParseInt(v, &config->runtime.cuda_linkage_same_dynamic_window_max_pairs);
        }
        if (key == "runtime.cuda_linkage_same_dynamic_pending_nodes" ||
            key == "runtime.cuda_linkage_same_layer_window_max_pending_nodes") {
            return ParseInt(v, &config->runtime.cuda_linkage_same_dynamic_pending_nodes);
        }
        if (key == "runtime.cuda_linkage_same_dynamic_preflush_pairs" ||
            key == "runtime.cuda_linkage_same_layer_preflush_target_pairs") {
            return ParseInt(v, &config->runtime.cuda_linkage_same_dynamic_preflush_pairs);
        }
        if (key == "runtime.cuda_linkage_same_dynamic_preflush_nodes" ||
            key == "runtime.cuda_linkage_same_layer_preflush_target_nodes") {
            return ParseInt(v, &config->runtime.cuda_linkage_same_dynamic_preflush_nodes);
        }
        if (key == "runtime.cuda_linkage_bad_window_batch" ||
            key == "runtime.cuda_linkage_bad_frozen_window_enable") {
            return ParseBool(v, &config->runtime.cuda_linkage_bad_window_batch);
        }
        if (key == "runtime.cuda_linkage_bad_window_nodes" ||
            key == "runtime.cuda_linkage_bad_frozen_window_nodes") {
            return ParseInt(v, &config->runtime.cuda_linkage_bad_window_nodes);
        }
        if (key == "runtime.cuda_linkage_bad_window_max_npairs" ||
            key == "runtime.cuda_linkage_bad_frozen_window_max_pairs") {
            return ParseInt(v, &config->runtime.cuda_linkage_bad_window_max_npairs);
        }
        if (key == "runtime.cuda_linkage_chunk_max_pairs") {
            return ParseInt(v, &config->runtime.cuda_linkage_chunk_max_pairs);
        }
        if (key == "runtime.cuda_linkage_chunk_max_pairs_init_linkage") {
            return ParseInt(v, &config->runtime.cuda_linkage_chunk_max_pairs_init_linkage);
        }
        if (key == "runtime.cuda_linkage_mem_budget_mb") {
            return ParseInt(v, &config->runtime.cuda_linkage_mem_budget_mb);
        }
        if (key == "runtime.cuda_linkage_mem_budget_mb_init_linkage") {
            return ParseInt(v, &config->runtime.cuda_linkage_mem_budget_mb_init_linkage);
        }
        if (key == "runtime.cuda_linkage_eval_async_pinned_mb") {
            return ParseInt(v, &config->runtime.cuda_linkage_eval_async_pinned_mb);
        }
        if (key == "runtime.cuda_linkage_eval_async_pinned_mb_init_linkage") {
            return ParseInt(v, &config->runtime.cuda_linkage_eval_async_pinned_mb_init_linkage);
        }
        if (key == "runtime.cuda_linkage_large_root_xc0_chunk_mb") {
            return ParseInt(v, &config->runtime.cuda_linkage_large_root_xc0_chunk_mb);
        }
        if (key == "runtime.cuda_linkage_large_root_xc0_chunk_mb_init_linkage") {
            return ParseInt(v, &config->runtime.cuda_linkage_large_root_xc0_chunk_mb_init_linkage);
        }
        if (key == "runtime.cuda_linkage_wait_for_ctx") {
            return ParseBool(v, &config->runtime.cuda_linkage_wait_for_ctx);
        }
        if (key == "eval.linkage.parent_louds_enable" || key == "eval.linkage_parent_louds_enable") {
            return ParseBool(v, &config->eval.linkage_parent_louds_enable);
        }
        if (key == "eval.linkage.parent_louds_native_eval") {
            return ParseBool(v, &config->eval.linkage_parent_louds_native_eval);
        }
        if (key == "eval.linkage.parent_louds_select_stride" || key == "eval.parent_louds_select_stride") {
            return ParseInt(v, &config->eval.parent_louds_select_stride);
        }
        if (key == "eval.linkage.parent_louds_rank_words_per_super_log2" || key ==
            "eval.parent_louds_rank_words_per_super_log2") {
            return ParseInt(v, &config->eval.parent_louds_rank_words_per_super_log2);
        }
        if (key == "eval.linkage.parent_louds_build_indices" || key == "eval.parent_louds_build_indices") {
            return ParseBool(v, &config->eval.parent_louds_build_indices);
        }
        if (key == "eval.linkage.parent_adaptive_u16_cache") {
            return ParseBool(v, &config->eval.linkage_parent_adaptive_u16_cache);
        }
        if (key == "eval.linkage.preload_clusters_io_threads") {
            return ParseInt(v, &config->eval.linkage_preload_clusters_io_threads);
        }
        if (key == "eval.linkage.louds_huffman_bench_times") {
            if (!ParseInt(v, &config->eval.linkage_louds_huffman_bench_times)) return false;
            if (config->eval.linkage_louds_huffman_bench_times < 1)
                config->eval.linkage_louds_huffman_bench_times = 1;
            return true;
        }
        if (key == "dataset.name") {
            config->dataset.name = v;
            ApplyDatasetDefaults(config);
            return true;
        }
        if (key == "dataset.data_root") {
            config->dataset.data_root = v;
            return true;
        }
        if (key == "dataset.train_path" || key == "dataset.learn_path") {
            config->dataset.train_path = v;
            return true;
        }
        if (key == "dataset.base_path") {
            config->dataset.base_path = v;
            return true;
        }
        if (key == "dataset.query_path") {
            config->dataset.query_path = v;
            return true;
        }
        if (key == "dataset.groundtruth_path" || key == "dataset.gt_path") {
            config->dataset.groundtruth_path = v;
            return true;
        }
        if (key == "dataset.groundtruth_add1") {
            bool flag = false;
            if (ParseBool(v, &flag)) {
                config->dataset.groundtruth_add1 = flag ? 1 : 0;
                return true;
            }
            int parsed = 0;
            if (ParseInt(v, &parsed)) {
                config->dataset.groundtruth_add1 = parsed != 0 ? 1 : 0;
                return true;
            }
            return false;
        }
        if (key == "dataset.ntrain") {
            if (!ParseInt(v, &config->dataset.ntrain)) {
                return false;
            }
            config->dataset.ntrain_set = true;
            return true;
        }
        if (key == "dataset.ntrain_set") {
            bool flag = false;
            if (ParseBool(v, &flag)) {
                config->dataset.ntrain_set = flag;
                return true;
            }
            int parsed = 0;
            if (ParseInt(v, &parsed)) {
                config->dataset.ntrain_set = (parsed != 0);
                return true;
            }
            return false;
        }
        if (key == "dataset.nbase") {
            if (!ParseInt(v, &config->dataset.nbase)) {
                return false;
            }
            config->dataset.nbase_set = true;
            return true;
        }
        if (key == "dataset.nbase_set") {
            bool flag = false;
            if (ParseBool(v, &flag)) {
                config->dataset.nbase_set = flag;
                return true;
            }
            int parsed = 0;
            if (ParseInt(v, &parsed)) {
                config->dataset.nbase_set = (parsed != 0);
                return true;
            }
            return false;
        }
        if (key == "dataset.nquery") {
            if (!ParseInt(v, &config->dataset.nquery)) {
                return false;
            }
            config->dataset.nquery_set = true;
            return true;
        }
        if (key == "dataset.nquery_set") {
            bool flag = false;
            if (ParseBool(v, &flag)) {
                config->dataset.nquery_set = flag;
                return true;
            }
            int parsed = 0;
            if (ParseInt(v, &parsed)) {
                config->dataset.nquery_set = (parsed != 0);
                return true;
            }
            return false;
        }
        if (key == "dataset.k") {
            if (!ParseInt(v, &config->dataset.k)) {
                return false;
            }
            config->dataset.k_set = true;
            return true;
        }
        if (key == "dataset.k_set") {
            bool flag = false;
            if (ParseBool(v, &flag)) {
                config->dataset.k_set = flag;
                return true;
            }
            int parsed = 0;
            if (ParseInt(v, &parsed)) {
                config->dataset.k_set = (parsed != 0);
                return true;
            }
            return false;
        }
        if (key == "model.m") {
            const bool ok = ParseInt(v, &config->model.m);
            if (ok) {
                FinalizeLinkageCoeffCodecConfig(config);
            }
            return ok;
        }
        if (key == "model.h0") {
            return ParseInt(v, &config->model.h0);
        }
        if (key == "model.h0_one") {
            return ParseInt(v, &config->model.h0_one);
        }
        if (key == "model.h_vec") {
            const bool ok = ParseIntList(v, &config->model.h_vec);
            if (ok) {
                config->model.h_vec_set = true;
            }
            return ok;
        }
        if (key == "advanced.eval_only") {
            return ParseBool(v, &config->advanced.eval_only);
        }
        if (key == "train.enabled") {
            return ParseBool(v, &config->train.enabled);
        }
        if (key == "train.init_enabled") {
            return ParseBool(v, &config->train.init_enabled);
        }
        if (key == "train.ckpt.enabled") {
            return ParseBool(v, &config->train.ckpt.enabled);
        }
        if (key == "train.ckpt.every_R") {
            return ParseInt(v, &config->train.ckpt.every_R);
        }
        if (key == "train.use_opq_rotation") {
            return ParseBool(v, &config->train.use_opq_rotation);
        }
        if (key == "train.log_metrics") {
            return ParseBool(v, &config->train.log_metrics);
        }
        if (key == "train.log_linkage_pre_c1") {
            return ParseBool(v, &config->train.log_linkage_pre_c1);
        }
        if (key == "train.exit_after_rvq_init") {
            return ParseBool(v, &config->train.exit_after_rvq_init);
        }
        if (key == "train.ckpt_after_init_basic") {
            return ParseBool(v, &config->train.ckpt_after_init_basic);
        }
        if (key == "train.exit_after_ckpt_init_basic") {
            return ParseBool(v, &config->train.exit_after_ckpt_init_basic);
        }
        if (key == "train.ils_iters") {
            return ParseInt(v, &config->train.ils_iters);
        }
        if (key == "train.icm_iters") {
            return ParseInt(v, &config->train.icm_iters);
        }
        if (key == "train.perturb_k") {
            return ParseInt(v, &config->train.perturb_k);
        }
        if (key == "train.max_R_iters") {
            return ParseInt(v, &config->train.max_R_iters);
        }
        if (key == "train.kmeans_iters") {
            return ParseInt(v, &config->train.kmeans_iters);
        }
        if (key == "train.kmeans_tol") {
            return ParseDouble(v, &config->train.kmeans_tol);
        }
        if (key == "train.kmeans_init") {
            config->train.kmeans_init = v;
            return true;
        }
        if (key == "train.init_samples") {
            return ParseInt(v, &config->train.init_samples);
        }
        if (key == "train.kmeans_initial_weight") {
            return ParseDouble(v, &config->train.kmeans_initial_weight);
        }
        if (key == "train.kmeans_min_weight") {
            return ParseDouble(v, &config->train.kmeans_min_weight);
        }
        if (key == "train.kmeans_outlier_quantile") {
            return ParseDouble(v, &config->train.kmeans_outlier_quantile);
        }
        if (key == "train.kmeans_cost_threshold") {
            return ParseDouble(v, &config->train.kmeans_cost_threshold);
        }
        if (key == "train.kmeans_annealing_factor") {
            return ParseDouble(v, &config->train.kmeans_annealing_factor);
        }
        if (key == "train.kmeans_warmup_iters") {
            return ParseInt(v, &config->train.kmeans_warmup_iters);
        }
        if (key == "train.kmeans_quantile_bins") {
            return ParseInt(v, &config->train.kmeans_quantile_bins);
        }
        if (key == "train.kmeans_anneal_no_spill") {
            return ParseBool(v, &config->train.kmeans_anneal_no_spill);
        }
        if (key == "train.kmeans_anneal_mode") {
            config->train.kmeans_anneal_mode = StripQuotes(v);
            return true;
        }
        if (key == "train.kmeans_anneal_weights_device") {
            return ParseBool(v, &config->train.kmeans_anneal_weights_device);
        }
        if (key == "train.kmeans_force_disable_device_weights") {
            return ParseBool(v, &config->train.kmeans_force_disable_device_weights);
        }
        if (key == "train.kmeans_streaming") {
            return ParseBool(v, &config->train.kmeans_streaming);
        }
        if (key == "train.kmeans_cache_xnorm_device") {
            return ParseBool(v, &config->train.kmeans_cache_xnorm_device);
        }
        if (key == "train.kmeans_device_cache_mb") {
            return ParseInt(v, &config->train.kmeans_device_cache_mb);
        }
        if (key == "train.kmeans_pin_host_x") {
            return ParseBool(v, &config->train.kmeans_pin_host_x);
        }
        if (key == "train.kmeans_bvecs_use_u8") {
            return ParseBool(v, &config->train.kmeans_bvecs_use_u8);
        }

        if (key == "train.kmeans_tmp_dir") {
            config->train.kmeans_tmp_dir = StripQuotes(v);
            return true;
        }

        if (key == "train.kmeansll_gpu_enable") {
            return ParseBool(v, &config->train.kmeansll_gpu_enable);
        }
        if (key == "train.kmeansll_rounds") {
            return ParseInt(v, &config->train.kmeansll_rounds);
        }
        if (key == "train.kmeansll_oversample") {
            return ParseDouble(v, &config->train.kmeansll_oversample);
        }
        if (key == "train.kmeansll_candidate_cap") {
            return ParseInt(v, &config->train.kmeansll_candidate_cap);
        }
        if (key == "train.kmeansll_seed") {
            return ParseUInt64(v, &config->train.kmeansll_seed);
        }
        if (key == "train.kmeansll_hier_enable") {
            return ParseBool(v, &config->train.kmeansll_hier_enable);
        }
        if (key == "train.kmeansll_hier_threshold") {
            return ParseInt(v, &config->train.kmeansll_hier_threshold);
        }
        if (key == "train.kmeansll_hier_fixed_k1") {
            return ParseInt(v, &config->train.kmeansll_hier_fixed_k1);
        }
        if (key == "train.kmeans_large_k_threshold") {
            return ParseInt(v, &config->train.kmeans_large_k_threshold);
        }
        if (key == "train.kmeans_large_k_hier2_enable") {
            return ParseBool(v, &config->train.kmeans_large_k_hier2_enable);
        }
        if (key == "train.kmeans_large_k_fixed_k1") {
            return ParseInt(v, &config->train.kmeans_large_k_fixed_k1);
        }
        if (key == "train.kmeans_large_k_k1_min") {
            return ParseInt(v, &config->train.kmeans_large_k_k1_min);
        }
        if (key == "train.kmeans_large_k_k1_max") {
            return ParseInt(v, &config->train.kmeans_large_k_k1_max);
        }
        if (key == "train.kmeans_large_k_k1_pow2") {
            return ParseBool(v, &config->train.kmeans_large_k_k1_pow2);
        }
        if (key == "train.kmeans_large_k_top_coarse") {
            return ParseInt(v, &config->train.kmeans_large_k_top_coarse);
        }
        if (key == "train.kmeans_large_k_hier2_train") {
            return ParseBool(v, &config->train.kmeans_large_k_hier2_train);
        }
        if (key == "train.kmeans_large_k_hier2_encode") {
            return ParseBool(v, &config->train.kmeans_large_k_hier2_encode);
        }
        if (key == "train.save_init_codes") {
            return ParseBool(v, &config->train.save_init_codes);
        }
        if (key == "train.init_codes_path") {
            config->train.init_codes_path = StripQuotes(v);
            return true;
        }
        if (key == "train.encode_only_after_layer") {
            return ParseBool(v, &config->train.encode_only_after_layer);
        }

        if (key == "train.seed") {
            return ParseInt(v, &config->train.seed);
        }
        // Train-time linkage build parameters (preferred: train.linkage.*).
        if (key == "train.linkage.root_percentile") {
            return ParseDouble(v, &config->train.linkage.root_percentile);
        }
        if (key == "train.linkage.num_layers") {
            return ParseInt(v, &config->train.linkage.num_layers);
        }
        if (key == "train.linkage.max_depth") {
            return ParseInt(v, &config->train.linkage.max_depth);
        }
        if (key == "train.linkage.knn_k") {
            return ParseInt(v, &config->train.linkage.knn_k);
        }
        if (key == "train.linkage.depth_k") {
            return ParseInt(v, &config->train.linkage.depth_k);
        }
        if (key == "train.linkage.icm_round") {
            return ParseInt(v, &config->train.linkage.icm_round);
        }
        if (key == "train.linkage.use_ils") {
            return ParseBool(v, &config->train.linkage.use_ils);
        }
        if (key == "train.linkage.ils_rounds") {
            return ParseInt(v, &config->train.linkage.ils_rounds);
        }
        if (key == "train.linkage.ils_perturb_layers") {
            return ParseInt(v, &config->train.linkage.ils_perturb_layers);
        }
        if (key == "train.init_linkage_icm_round") {
            return ParseInt(v, &config->train.init_linkage_icm_round);
        }
        if (key == "train.init_linkage_ils_rounds") {
            return ParseInt(v, &config->train.init_linkage_ils_rounds);
        }
        if (key == "train.init_linkage_hybrid_varroot_rounds") {
            return ParseInt(v, &config->train.init_linkage_hybrid_varroot_rounds);
        }
        if (key == "train.init_linkage_mode") {
            config->train.init_linkage_mode = StripQuotes(v);
            return true;
        }
        if (key == "train.linkage.seed") {
            return ParseInt(v, &config->train.linkage.seed);
        }

        // Base-stage encoding.
        if (key == "base.encode.enabled") {
            return ParseBool(v, &config->base.encode.enabled);
        }
        if (key == "base.encode.use_abs") {
            return ParseBool(v, &config->base.encode.use_abs);
        }
        if (key == "base.encode.H_beam") {
            return ParseInt(v, &config->base.encode.H_beam);
        }
        if (key == "base.encode.ils_iters") {
            return ParseInt(v, &config->base.encode.ils_iters);
        }
        if (key == "base.encode.icm_iters") {
            return ParseInt(v, &config->base.encode.icm_iters);
        }
        if (key == "base.encode.perturb_k") {
            return ParseInt(v, &config->base.encode.perturb_k);
        }
        if (key == "base.encode.hnorms") {
            return ParseInt(v, &config->base.encode.hnorms);
        }
        if (key == "base.encode.seed") {
            return ParseInt(v, &config->base.encode.seed);
        }

        // Base-stage linkage build.
        if (key == "base.linkage.enabled") {
            return ParseBool(v, &config->base.linkage.enabled);
        }
        if (key == "base.linkage.root_percentile") {
            return ParseDouble(v, &config->base.linkage.root_percentile);
        }
        if (key == "base.linkage.num_layers") {
            return ParseInt(v, &config->base.linkage.num_layers);
        }
        if (key == "base.linkage.max_depth") {
            return ParseInt(v, &config->base.linkage.max_depth);
        }
        if (key == "base.linkage.knn_k") {
            return ParseInt(v, &config->base.linkage.knn_k);
        }
        if (key == "base.linkage.depth_k") {
            return ParseInt(v, &config->base.linkage.depth_k);
        }
        if (key == "base.linkage.icm_round") {
            return ParseInt(v, &config->base.linkage.icm_round);
        }
        if (key == "base.linkage.use_ils") {
            return ParseBool(v, &config->base.linkage.use_ils);
        }
        if (key == "base.linkage.ils_rounds") {
            return ParseInt(v, &config->base.linkage.ils_rounds);
        }
        if (key == "base.linkage.ils_perturb_layers") {
            return ParseInt(v, &config->base.linkage.ils_perturb_layers);
        }
        if (key == "base.linkage.seed") {
            return ParseInt(v, &config->base.linkage.seed);
        }
        if (key == "hnsw.M") {
            return ParseInt(v, &config->hnsw.M);
        }
        if (key == "hnsw.candidate_multiplier_good") {
            return ParseInt(v, &config->hnsw.candidate_multiplier_good);
        }
        if (key == "hnsw.candidate_multiplier_bad") {
            return ParseInt(v, &config->hnsw.candidate_multiplier_bad);
        }
        if (key == "hnsw.ef_construction_cap") {
            return ParseInt(v, &config->hnsw.ef_construction_cap);
        }
        if (key == "virtual.enabled") {
            return ParseBool(v, &config->virtual_cfg.enabled);
        }
        if (key == "virtual.virtual_ratio") {
            return ParseDouble(v, &config->virtual_cfg.virtual_ratio);
        }
        if (key == "virtual.good_fraction") {
            return ParseDouble(v, &config->virtual_cfg.good_fraction);
        }
        if (key == "virtual.min_virtual") {
            return ParseInt(v, &config->virtual_cfg.min_virtual);
        }
        if (key == "virtual.max_virtual") {
            return ParseInt(v, &config->virtual_cfg.max_virtual);
        }
        if (key == "virtual.alpha_bad") {
            return ParseFloat(v, &config->virtual_cfg.alpha_bad);
        }
        if (key == "virtual.use_fixed_virtual_per_cluster") {
            return ParseBool(v, &config->virtual_cfg.use_fixed_virtual_per_cluster);
        }
        if (key == "virtual.fixed_virtual_per_cluster") {
            return ParseInt(v, &config->virtual_cfg.fixed_virtual_per_cluster);
        }
        if (key == "virtual.umap_knn_k") {
            return ParseInt(v, &config->virtual_cfg.umap_knn_k);
        }
        if (key == "virtual.local_connectivity") {
            return ParseInt(v, &config->virtual_cfg.local_connectivity);
        }
        if (key == "virtual.overlap_thr") {
            return ParseDouble(v, &config->virtual_cfg.overlap_thr);
        }
        if (key == "virtual.prefer_peaks") {
            return ParseBool(v, &config->virtual_cfg.prefer_peaks);
        }
        if (key == "virtual.anchor_neighbor_k") {
            return ParseInt(v, &config->virtual_cfg.anchor_neighbor_k);
        }
        if (key == "io.pre_fix") {
            config->io.pre_fix = v;
            ApplyDatasetDefaults(config);
            return true;
        }
        if (key == "io.config_file") {
            config->io.config_file = StripQuotes(v);
            return true;
        }
        if (key == "io.hdf5_layout") {
            if (v != "julia" && v != "cxx") {
                if (error) {
                    *error = "Invalid io.hdf5_layout: " + v + R"( (expected "julia" or "cxx"))";
                }
                return false;
            }
            config->io.hdf5_layout = v;
            return true;
        }
        if (key == "io.result_format") {
            if (v != "auto" && v != "hdf5" && v != "bin") {
                if (error) {
                    *error = R"(Invalid io.result_format: )" + v + R"( (expected "auto", "hdf5", or "bin"))";
                }
                return false;
            }
            config->io.result_format = v;
            ApplyDatasetDefaults(config);
            return true;
        }
        if (key == "io.index_dtype") {
            std::string dtype = v;
            if (dtype == "int32") dtype = "int";
            if (dtype == "uint32") dtype = "uint";
            if (dtype != "int" && dtype != "uint") {
                if (error) {
                    *error = "Invalid io.index_dtype: " + v + R"( (expected "int" or "uint"))";
                }
                return false;
            }
            config->io.index_dtype = dtype;
            return true;
        }
        if (key == "io.train_file") {
            config->io.train_file = v;
            config->io.train_file_set = true;
            return true;
        }
        if (key == "io.base_file") {
            config->io.base_file = v;
            config->io.base_file_set = true;
            return true;
        }
        if (key == "io.linkage_file") {
            config->io.linkage_file = v;
            config->io.linkage_file_set = true;
            return true;
        }
        if (key == "io.save_train") {
            return ParseBool(v, &config->io.save_train);
        }
        if (key == "io.save_base") {
            return ParseBool(v, &config->io.save_base);
        }
        if (key == "io.save_linkage") {
            return ParseBool(v, &config->io.save_linkage);
        }
        if (key == "io.load_date") {
            config->io.load_date = v;
            return true;
        }
        if (key == "io.load_seq") {
            config->io.load_seq = v;
            return true;
        }

        // Large-scale (out-of-core) config.
        if (key == "large.enabled") {
            return ParseBool(v, &config->large.enabled);
        }
        if (key == "large.train_format") {
            config->large.train_format = v;
            return true;
        }
        if (key == "large.base_format") {
            config->large.base_format = v;
            return true;
        }
        if (key == "large.query_format") {
            config->large.query_format = v;
            return true;
        }
        if (key == "large.gt_format") {
            config->large.gt_format = v;
            return true;
        }
        if (key == "large.output_dir") {
            config->large.output_dir = v;
            return true;
        }
        if (key == "large.tmp_dir") {
            config->large.tmp_dir = v;
            return true;
        }
        if (key == "large.train_block") {
            return ParseInt(v, &config->large.train_block);
        }
        if (key == "large.base_block") {
            return ParseInt(v, &config->large.base_block);
        }
        if (key == "large.base_shard_size") {
            return ParseInt(v, &config->large.base_shard_size);
        }
        if (key == "large.cluster_bucket_size") {
            return ParseInt(v, &config->large.cluster_bucket_size);
        }
        if (key == "large.bucket_flush_mb") {
            return ParseInt(v, &config->large.bucket_flush_mb);
        }
        if (key == "large.write_vector_bucket") {
            return ParseBool(v, &config->large.write_vector_bucket);
        }
        if (key == "large.write_basic_to_bucket") {
            return ParseBool(v, &config->large.write_basic_to_bucket);
        }
        if (key == "large.base_linkage_store_coeffs_f32") {
            return ParseBool(v, &config->large.base_linkage_store_coeffs_f32);
        }
        if (key == "large.cleanup.enabled") {
            return ParseBool(v, &config->large.cleanup.enabled);
        }
        if (key == "large.cleanup.preset") {
            config->large.cleanup.preset = v;
            return true;
        }
        if (key == "large.cleanup.keep_base_basic") {
            return ParseInt(v, &config->large.cleanup.keep_base_basic);
        }
        if (key == "large.cleanup.keep_base_basic_cluster_id") {
            return ParseInt(v, &config->large.cleanup.keep_base_basic_cluster_id);
        }
        if (key == "large.cleanup.keep_base_basic_ivf_lists") {
            return ParseInt(v, &config->large.cleanup.keep_base_basic_ivf_lists);
        }
        if (key == "large.cleanup.keep_base_list") {
            return ParseInt(v, &config->large.cleanup.keep_base_list);
        }
        if (key == "large.cleanup.keep_base_list_raw") {
            return ParseInt(v, &config->large.cleanup.keep_base_list_raw);
        }
        if (key == "large.cleanup.keep_linkage_list_f32") {
            return ParseInt(v, &config->large.cleanup.keep_linkage_list_f32);
        }
        if (key == "large.cleanup.keep_linkage_parent_u32") {
            return ParseInt(v, &config->large.cleanup.keep_linkage_parent_u32);
        }
        if (key == "large.linkage_store_parent_u32") {
            return ParseBool(v, &config->large.linkage_store_parent_u32);
        }
        if (key == "large.linkage_checkpoint") {
            return ParseBool(v, &config->large.linkage_checkpoint);
        }
        if (key == "large.base_basic_checkpoint") {
            return ParseBool(v, &config->large.base_basic_checkpoint);
        }
        if (key == "large.profile_timing") {
            return ParseBool(v, &config->large.profile_timing);
        }
        if (key == "large.archive_log") {
            return ParseBool(v, &config->large.archive_log);
        }
        if (key == "large.protect_existing_outputs") {
            return ParseBool(v, &config->large.protect_existing_outputs);
        }
        if (key == "large.linkage_coeff_codec.enabled") {
            return ParseBool(v, &config->large.linkage_coeff_codec.enabled);
        }
        if (key == "large.linkage_coeff_codec.granularity") {
            config->large.linkage_coeff_codec.granularity = v;
            return true;
        }
        if (key == "large.linkage_coeff_codec.bits_per_layer") {
            if (!ParseIntList(v, &config->large.linkage_coeff_codec.bits_per_layer)) {
                return false;
            }
            FinalizeLinkageCoeffCodecConfig(config);
            return true;
        }
        if (key == "large.linkage_coeff_codec.p_first_candidates") {
            return ParseDoubleList(v, &config->large.linkage_coeff_codec.p_first_candidates);
        }
        if (key == "large.linkage_coeff_codec.p_rest_candidates") {
            return ParseDoubleList(v, &config->large.linkage_coeff_codec.p_rest_candidates);
        }
        if (key == "large.linkage_coeff_codec.use_weighted_quantile") {
            return ParseBool(v, &config->large.linkage_coeff_codec.use_weighted_quantile);
        }
        if (key == "large.linkage_coeff_codec.allow_clip") {
            return ParseBool(v, &config->large.linkage_coeff_codec.allow_clip);
        }
        if (key == "large.linkage_coeff_codec.fit_scale") {
            return ParseBool(v, &config->large.linkage_coeff_codec.fit_scale);
        }
        if (key == "large.linkage_coeff_codec.q_refine_sweeps") {
            return ParseInt(v, &config->large.linkage_coeff_codec.q_refine_sweeps);
        }
        if (key == "large.linkage_coeff_codec.q_refine_max_layer") {
            return ParseInt(v, &config->large.linkage_coeff_codec.q_refine_max_layer);
        }
        if (key == "large.linkage_coeff_codec.q_refine_step_limit") {
            return ParseInt(v, &config->large.linkage_coeff_codec.q_refine_step_limit);
        }

        // Evaluation.
        if (key == "eval.base.enabled") {
            return ParseBool(v, &config->eval.base_enabled);
        }
        if (key == "eval.metric_mode") {
            return ParseInt(v, &config->eval.metric_mode);
        }
        if (key == "eval.metric_gt_topks") {
            // metric_gt_topks is optional; allow an explicit empty list ("[]").
            return ParseIntList(v, &config->eval.metric_gt_topks, /*allow_empty=*/true);
        }
        if (key == "eval.base.use_ivf") {
            return ParseBool(v, &config->eval.base_use_ivf);
        }
        if (key == "eval.base.nprobe") {
            return ParseInt(v, &config->eval.base_nprobe);
        }
        if (key == "eval.base.warmup") {
            return ParseInt(v, &config->eval.base_warmup);
        }
        if (key == "eval.base.repeat") {
            return ParseInt(v, &config->eval.base_repeat);
        }
        if (key == "eval.linkage.enabled") {
            return ParseBool(v, &config->eval.linkage_enabled);
        }
        if (key == "eval.linkage.use_ivf_disk") {
            return ParseBool(v, &config->eval.linkage_use_ivf_disk);
        }
        if (key == "eval.linkage.nprobe") {
            int scalar = 0;
            if (ParseInt(v, &scalar)) {
                config->eval.linkage_nprobe = scalar;
                if (linkage_nprobe_batch) linkage_nprobe_batch->clear();
                return true;
            }
            std::vector<int> probes;
            if (!ParseIntList(v, &probes)) {
                return false;
            }
            probes.erase(std::remove_if(probes.begin(), probes.end(),
                                        [](int x) { return x <= 0; }),
                         probes.end());
            if (probes.empty()) {
                return false;
            }
            config->eval.linkage_nprobe = probes.front();
            if (linkage_nprobe_batch) {
                *linkage_nprobe_batch = std::move(probes);
            }
            return true;
        }
        if (key == "eval.linkage.query_block") {
            return ParseInt(v, &config->eval.linkage_query_block);
        }
        if (key == "eval.linkage.warmup") {
            return ParseInt(v, &config->eval.linkage_warmup);
        }
        if (key == "eval.linkage.repeat") {
            return ParseInt(v, &config->eval.linkage_repeat);
        }
        if (key == "eval.bench.quiet") {
            return ParseBool(v, &config->eval.bench_quiet);
        }
        // NOTE: legacy disk linkage recall toggles were removed; lookup path is always used.
        if (key == "eval.linkage.coeff_mode") {
            config->eval.linkage_coeff_mode = StripQuotes(v);
            return true;
        }
        if (key == "eval.linkage.gpu_scan_enable") {
            return ParseBool(v, &config->eval.linkage_gpu_scan_enable);
        }
        if (key == "eval.linkage.gpu_scan_tile256") {
            return ParseBool(v, &config->eval.linkage_gpu_scan_tile256);
        }
        if (key == "eval.linkage.gpu_scan_max_nc") {
            return ParseInt(v, &config->eval.linkage_gpu_scan_max_nc);
        }
        if (key == "eval.linkage.gpu_scan_cache_mb") {
            return ParseInt(v, &config->eval.linkage_gpu_scan_cache_mb);
        }
        if (key == "eval.linkage.gpu_norm_enable") {
            return ParseBool(v, &config->eval.linkage_gpu_norm_enable);
        }
        if (key == "eval.disk_norm2_mode") {
            const std::string mode = StripQuotes(v);
            if (mode == "float" || mode == "lut" || mode == "lut_sqrt" ||
                mode == "lut_log1p" || mode == "lut_tail_weighted" ||
                mode == "lut_piecewise" || mode == "lut_cluster") {
                config->eval.disk_norm2_mode = mode;
                return true;
            }
            return false;
        }
        if (key == "eval.disk_norm2_lut_kmeans_niter") {
            return ParseInt(v, &config->eval.disk_norm2_lut_kmeans_niter);
        }
        if (key == "eval.disk_norm2_lut_log_alpha") {
            return ParseDouble(v, &config->eval.disk_norm2_lut_log_alpha);
        }
        if (key == "eval.disk_norm2_lut_piecewise_p1") {
            return ParseDouble(v, &config->eval.disk_norm2_lut_piecewise_p1);
        }
        if (key == "eval.disk_norm2_lut_piecewise_p2") {
            return ParseDouble(v, &config->eval.disk_norm2_lut_piecewise_p2);
        }
        if (key == "eval.disk_norm2_lut_piecewise_count_weight") {
            return ParseDouble(v, &config->eval.disk_norm2_lut_piecewise_count_weight);
        }
        if (key == "eval.disk_norm2_lut_piecewise_range_weight") {
            return ParseDouble(v, &config->eval.disk_norm2_lut_piecewise_range_weight);
        }
        if (key == "eval.linkage.ivf_probe_mode") {
            config->eval.linkage_ivf_probe_mode = StripQuotes(v);
            return true;
        }
        if (key == "eval.linkage.ivf_hier2_top_coarse") {
            return ParseInt(v, &config->eval.linkage_ivf_hier2_top_coarse);
        }
        if (key == "eval.linkage.ivf_hier2_kmeans_niter") {
            return ParseInt(v, &config->eval.linkage_ivf_hier2_kmeans_niter);
        }
        if (key == "eval.linkage.ivf_hnsw_M") {
            return ParseInt(v, &config->eval.linkage_ivf_hnsw_M);
        }
        if (key == "eval.linkage.ivf_hnsw_ef_construction") {
            return ParseInt(v, &config->eval.linkage_ivf_hnsw_ef_construction);
        }
        if (key == "eval.linkage.ivf_hnsw_ef_search") {
            int scalar = 0;
            if (ParseInt(v, &scalar)) {
                config->eval.linkage_ivf_hnsw_ef_search = scalar;
                if (linkage_ef_search_batch) linkage_ef_search_batch->clear();
                return true;
            }
            std::vector<int> ef_searches;
            if (!ParseIntList(v, &ef_searches)) {
                return false;
            }
            ef_searches.erase(std::remove_if(ef_searches.begin(), ef_searches.end(),
                                             [](int x) { return x <= 0; }),
                              ef_searches.end());
            if (ef_searches.empty()) {
                return false;
            }
            config->eval.linkage_ivf_hnsw_ef_search = ef_searches.front();
            if (linkage_ef_search_batch) {
                *linkage_ef_search_batch = std::move(ef_searches);
            }
            return true;
        }
        if (key == "eval.linkage.norm2_store_float") {
            return ParseBool(v, &config->eval.linkage_norm2_store_float);
        }
        if (key == "eval.linkage.norm2_store_int8") {
            return ParseBool(v, &config->eval.linkage_norm2_store_int8);
        }
        if (key == "eval.linkage.archive_eval_result") {
            return ParseBool(v, &config->eval.linkage_archive_eval_result);
        }

        if (error) {
            *error = "Unknown config key: " + key;
        }
        return false;
    }

    bool ParseArgs(int argc, char** argv, Config* config, std::string* error, std::string* dump_path,
                   std::vector<int>* linkage_nprobe_batch,
                   std::vector<int>* linkage_ef_search_batch) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--config" && i + 1 < argc) {
                std::string config_path = argv[++i];
                if (!LoadConfigFile(config_path, config, error, linkage_nprobe_batch, linkage_ef_search_batch)) {
                    return false;
                }
                config->io.config_file = config_path;
                continue;
            }
            if (arg == "--set" && i + 1 < argc) {
                std::string kv = argv[++i];
                std::string key;
                std::string value;
                if (!ParseKeyValueLine(kv, &key, &value)) {
                    if (error) {
                        *error = "Invalid override: " + kv;
                    }
                    return false;
                }
                if (!ApplyOverride(key, value, config, error, linkage_nprobe_batch, linkage_ef_search_batch)) {
                    return false;
                }
                continue;
            }
            if (arg == "--dump-config" && i + 1 < argc) {
                if (dump_path) {
                    *dump_path = argv[++i];
                }
                continue;
            }
            if (error) {
                *error = "Unknown argument: " + arg;
            }
            return false;
        }
        return true;
    }

    std::uint64_t Fnv1a64(const void* data, std::size_t len) {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        std::uint64_t h = 1469598103934665603ull;
        for (std::size_t i = 0; i < len; ++i) {
            h ^= static_cast<std::uint64_t>(bytes[i]);
            h *= 1099511628211ull;
        }
        return h;
    }

    std::uint64_t Fnv1a64WithSeed(const void* data, std::size_t len, std::uint64_t seed) {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        std::uint64_t h = seed;
        for (std::size_t i = 0; i < len; ++i) {
            h ^= static_cast<std::uint64_t>(bytes[i]);
            h *= 1099511628211ull;
        }
        return h;
    }

    void AppendConfigSnapshotFull(std::ostream& out, const Config& config) {
        out << "dataset.name = \"" << config.dataset.name << "\"\n";
        out << "dataset.data_root = \"" << config.dataset.data_root << "\"\n";
        out << "dataset.train_path = \"" << config.dataset.train_path << "\"\n";
        out << "dataset.base_path = \"" << config.dataset.base_path << "\"\n";
        out << "dataset.query_path = \"" << config.dataset.query_path << "\"\n";
        out << "dataset.groundtruth_path = \"" << config.dataset.groundtruth_path << "\"\n";
        out << "dataset.groundtruth_add1 = " << (config.dataset.groundtruth_add1 != 0 ? "true" : "false") << "\n";
        out << "dataset.ntrain_set = " << (config.dataset.ntrain_set ? "true" : "false") << "\n";
        out << "dataset.nbase_set = " << (config.dataset.nbase_set ? "true" : "false") << "\n";
        out << "dataset.nquery_set = " << (config.dataset.nquery_set ? "true" : "false") << "\n";
        out << "dataset.k_set = " << (config.dataset.k_set ? "true" : "false") << "\n";
        out << "dataset.ntrain = " << config.dataset.ntrain << "\n";
        out << "dataset.nbase = " << config.dataset.nbase << "\n";
        out << "dataset.nquery = " << config.dataset.nquery << "\n";
        out << "dataset.k = " << config.dataset.k << "\n";

        out << "model.m = " << config.model.m << "\n";
        out << "model.h0 = " << config.model.h0 << "\n";
        out << "model.h0_one = " << config.model.h0_one << "\n";
        out << "model.h_vec = [";
        for (std::size_t i = 0; i < config.model.h_vec.size(); ++i) {
            if (i > 0) {
                out << ", ";
            }
            out << config.model.h_vec[i];
        }
        out << "]\n";

        out << "runtime.omp_threads = " << config.runtime.omp_threads << "\n";
        out << "runtime.use_cuda = " << (config.runtime.use_cuda ? "true" : "false") << "\n";
        out << "runtime.cuda_device = " << config.runtime.cuda_device << "\n";
        out << "runtime.cuda_mode = \"" << config.runtime.cuda_mode << "\"\n";
        out << "runtime.cuda_allow_tf32 = " << (config.runtime.cuda_allow_tf32 ? "true" : "false") << "\n";
        out << "runtime.cuda_pool_size = " << config.runtime.cuda_pool_size << "\n";
        out << "runtime.cuda_pool_size_init_linkage = " << config.runtime.cuda_pool_size_init_linkage << "\n";
        out << "runtime.cuda_cublas_workspace_mb = " << config.runtime.cuda_cublas_workspace_mb << "\n";
        out << "runtime.cuda_linkage_use_device_rfull = " << (config.runtime.cuda_linkage_use_device_rfull
                                                                  ? "true"
                                                                  : "false") << "\n";
        out << "runtime.linkage_async_io = " << (config.runtime.linkage_async_io ? "true" : "false") << "\n";
        out << "runtime.linkage_async_io_depth = " << config.runtime.linkage_async_io_depth << "\n";
        out << "runtime.basic_async_io = " << (config.runtime.basic_async_io ? "true" : "false") << "\n";
        out << "runtime.basic_async_io_depth = " << config.runtime.basic_async_io_depth << "\n";
        out << "runtime.basic_async_io_mb = " << config.runtime.basic_async_io_mb << "\n";
        out << "runtime.basic_async_write = " << (config.runtime.basic_async_write ? "true" : "false") << "\n";
        out << "runtime.basic_async_write_depth = " << config.runtime.basic_async_write_depth << "\n";
        out << "runtime.basic_async_write_mb = " << config.runtime.basic_async_write_mb << "\n";
        out << "runtime.precomp_large_root_g0s_transpose = "
            << (config.runtime.precomp_large_root_g0s_transpose ? "true" : "false") << "\n";
        out << "runtime.precomp_large_root_g0s_transpose_max_mb = "
            << config.runtime.precomp_large_root_g0s_transpose_max_mb << "\n";
        out << "runtime.c_one_update_shards = " << config.runtime.c_one_update_shards << "\n";
        out << "runtime.c_one_update_shards_init_linkage = " << config.runtime.c_one_update_shards_init_linkage << "\n";
        out << "runtime.basic_hybrid_enable = " << (config.runtime.basic_hybrid_enable ? "true" : "false") << "\n";
        out << "runtime.basic_hybrid_cpu_stride = " << config.runtime.basic_hybrid_cpu_stride << "\n";
        out << "runtime.basic_hybrid_cpu_threads = " << config.runtime.basic_hybrid_cpu_threads << "\n";
        out << "runtime.basic_hybrid_reorder_depth = " << config.runtime.basic_hybrid_reorder_depth << "\n";
        out << "runtime.basic_hybrid_inflight_mb = " << config.runtime.basic_hybrid_inflight_mb << "\n";
        out << "runtime.cuda_linkage_single_gpu_min_candidates = " << config.runtime.cuda_linkage_min_kp << "\n";
        out << "runtime.cuda_linkage_single_gpu_max_candidates = " << config.runtime.cuda_linkage_max_kp << "\n";
        out << "runtime.cuda_linkage_inner_many_nodes_enable = "
            << (config.runtime.cuda_linkage_batch_inner_enabled ? "true" : "false") << "\n";
        out << "runtime.cuda_linkage_inner_many_nodes_target_nodes = " << config.runtime.cuda_linkage_batch_nodes <<
            "\n";
        out << "runtime.cuda_linkage_many_nodes_max_pairs = " << config.runtime.cuda_linkage_batch_max_pairs << "\n";
        out << "runtime.cuda_linkage_many_nodes_max_pairs_init_linkage = "
            << config.runtime.cuda_linkage_batch_max_pairs_init_linkage << "\n";
        out << "runtime.cuda_linkage_many_nodes_preflush_target_pairs = " << config.runtime.cuda_linkage_batch_min_pairs
            << "\n";
        out << "runtime.cuda_linkage_same_layer_window_min_pairs = " << config.runtime.
                                                                               cuda_linkage_same_dynamic_min_pairs <<
            "\n";
        out << "runtime.cuda_linkage_same_layer_tiny_forced_cpu_enable = "
            << (config.runtime.cuda_linkage_same_dynamic_tiny_cpu ? "true" : "false") << "\n";
        out << "runtime.cuda_linkage_init_same_layer_tiny_forced_cpu_enable = "
            << (config.runtime.cuda_linkage_init_large_root_tiny_cpu ? "true" : "false") << "\n";
        out << "runtime.cuda_linkage_same_layer_tiny_forced_cpu_max_pairs = "
            << config.runtime.cuda_linkage_same_dynamic_tiny_cpu_max_pairs << "\n";
        out << "runtime.cuda_linkage_same_layer_tiny_forced_cpu_max_nodes = "
            << config.runtime.cuda_linkage_same_dynamic_tiny_cpu_max_nodes << "\n";
        out << "runtime.cuda_linkage_same_layer_block_nodes = " << config.runtime.cuda_linkage_same_dynamic_nodes <<
            "\n";
        out << "runtime.cuda_linkage_same_layer_single_gpu_min_candidates = " << config.runtime.
            cuda_linkage_same_dynamic_min_kp << "\n";
        out << "runtime.cuda_linkage_same_layer_preflush_single_shot = "
            << (config.runtime.cuda_linkage_same_dynamic_window_batch ? "true" : "false") << "\n";
        out << "runtime.cuda_linkage_same_layer_window_max_pairs = " << config.runtime.
                                                                               cuda_linkage_same_dynamic_window_max_pairs
            << "\n";
        out << "runtime.cuda_linkage_same_layer_window_max_pending_nodes = " << config.runtime.
            cuda_linkage_same_dynamic_pending_nodes << "\n";
        out << "runtime.cuda_linkage_same_layer_preflush_target_pairs = " << config.runtime.
            cuda_linkage_same_dynamic_preflush_pairs << "\n";
        out << "runtime.cuda_linkage_same_layer_preflush_target_nodes = " << config.runtime.
            cuda_linkage_same_dynamic_preflush_nodes << "\n";
        out << "runtime.cuda_linkage_bad_frozen_window_enable = "
            << (config.runtime.cuda_linkage_bad_window_batch ? "true" : "false") << "\n";
        out << "runtime.cuda_linkage_bad_frozen_window_nodes = " << config.runtime.cuda_linkage_bad_window_nodes <<
            "\n";
        out << "runtime.cuda_linkage_bad_frozen_window_max_pairs = " << config.runtime.
                                                                               cuda_linkage_bad_window_max_npairs <<
            "\n";
        out << "runtime.cuda_linkage_chunk_max_pairs = " << config.runtime.cuda_linkage_chunk_max_pairs << "\n";
        out << "runtime.cuda_linkage_chunk_max_pairs_init_linkage = " << config.runtime.
            cuda_linkage_chunk_max_pairs_init_linkage << "\n";
        out << "runtime.cuda_linkage_mem_budget_mb = " << config.runtime.cuda_linkage_mem_budget_mb << "\n";
        out << "runtime.cuda_linkage_mem_budget_mb_init_linkage = " << config.runtime.
                                                                              cuda_linkage_mem_budget_mb_init_linkage <<
            "\n";
        out << "runtime.cuda_linkage_eval_async_pinned_mb = " << config.runtime.cuda_linkage_eval_async_pinned_mb <<
            "\n";
        out << "runtime.cuda_linkage_eval_async_pinned_mb_init_linkage = " << config.runtime.
            cuda_linkage_eval_async_pinned_mb_init_linkage << "\n";
        out << "runtime.cuda_linkage_large_root_xc0_chunk_mb = " << config.runtime.cuda_linkage_large_root_xc0_chunk_mb
            << "\n";
        out << "runtime.cuda_linkage_large_root_xc0_chunk_mb_init_linkage = " << config.runtime.
            cuda_linkage_large_root_xc0_chunk_mb_init_linkage << "\n";
        out << "runtime.cuda_linkage_wait_for_ctx = " << (config.runtime.cuda_linkage_wait_for_ctx ? "true" : "false")
            << "\n";

        out << "advanced.eval_only = " << (config.advanced.eval_only ? "true" : "false") << "\n";
        out << "train.enabled = " << (config.train.enabled ? "true" : "false") << "\n";
        out << "train.init_enabled = " << (config.train.init_enabled ? "true" : "false") << "\n";
        out << "train.ckpt.enabled = " << (config.train.ckpt.enabled ? "true" : "false") << "\n";
        out << "train.ckpt.every_R = " << config.train.ckpt.every_R << "\n";
        out << "train.use_opq_rotation = " << (config.train.use_opq_rotation ? "true" : "false") << "\n";
        out << "train.log_metrics = " << (config.train.log_metrics ? "true" : "false") << "\n";
        out << "train.log_linkage_pre_c1 = " << (config.train.log_linkage_pre_c1 ? "true" : "false") << "\n";
        out << "train.exit_after_rvq_init = " << (config.train.exit_after_rvq_init ? "true" : "false") << "\n";
        out << "train.ckpt_after_init_basic = " << (config.train.ckpt_after_init_basic ? "true" : "false") << "\n";
        out << "train.exit_after_ckpt_init_basic = " << (config.train.exit_after_ckpt_init_basic ? "true" : "false") <<
            "\n";
        out << "train.ils_iters = " << config.train.ils_iters << "\n";
        out << "train.icm_iters = " << config.train.icm_iters << "\n";
        out << "train.perturb_k = " << config.train.perturb_k << "\n";
        out << "train.max_R_iters = " << config.train.max_R_iters << "\n";
        out << "train.kmeans_iters = " << config.train.kmeans_iters << "\n";
        out << "train.kmeans_tol = " << config.train.kmeans_tol << "\n";
        out << "train.kmeans_init = \"" << config.train.kmeans_init << "\"\n";
        out << "train.init_samples = " << config.train.init_samples << "\n";
        out << "train.kmeans_initial_weight = " << config.train.kmeans_initial_weight << "\n";
        out << "train.kmeans_min_weight = " << config.train.kmeans_min_weight << "\n";
        out << "train.kmeans_outlier_quantile = " << config.train.kmeans_outlier_quantile << "\n";
        out << "train.kmeans_cost_threshold = " << config.train.kmeans_cost_threshold << "\n";
        out << "train.kmeans_annealing_factor = " << config.train.kmeans_annealing_factor << "\n";
        out << "train.kmeans_warmup_iters = " << config.train.kmeans_warmup_iters << "\n";
        out << "train.kmeans_quantile_bins = " << config.train.kmeans_quantile_bins << "\n";
        out << "train.kmeans_anneal_no_spill = " << (config.train.kmeans_anneal_no_spill ? "true" : "false") << "\n";
        out << "train.kmeans_anneal_mode = \"" << config.train.kmeans_anneal_mode << "\"\n";
        out << "train.kmeans_anneal_weights_device = " << (config.train.kmeans_anneal_weights_device ? "true" : "false")
            << "\n";
        out << "train.kmeans_force_disable_device_weights = "
            << (config.train.kmeans_force_disable_device_weights ? "true" : "false") << "\n";
        out << "train.kmeans_streaming = " << (config.train.kmeans_streaming ? "true" : "false") << "\n";
        out << "train.kmeans_cache_xnorm_device = " << (config.train.kmeans_cache_xnorm_device ? "true" : "false") <<
            "\n";
        out << "train.kmeans_device_cache_mb = " << config.train.kmeans_device_cache_mb << "\n";
        out << "train.kmeans_pin_host_x = " << (config.train.kmeans_pin_host_x ? "true" : "false") << "\n";
        out << "train.kmeans_bvecs_use_u8 = " << (config.train.kmeans_bvecs_use_u8 ? "true" : "false") << "\n";
        out << "train.kmeans_tmp_dir = \"" << config.train.kmeans_tmp_dir << "\"\n";
        out << "train.kmeansll_gpu_enable = " << (config.train.kmeansll_gpu_enable ? "true" : "false") << "\n";
        out << "train.kmeansll_rounds = " << config.train.kmeansll_rounds << "\n";
        out << "train.kmeansll_oversample = " << config.train.kmeansll_oversample << "\n";
        out << "train.kmeansll_candidate_cap = " << config.train.kmeansll_candidate_cap << "\n";
        out << "train.kmeansll_seed = " << config.train.kmeansll_seed << "\n";
        out << "train.kmeansll_hier_enable = " << (config.train.kmeansll_hier_enable ? "true" : "false") << "\n";
        out << "train.kmeansll_hier_threshold = " << config.train.kmeansll_hier_threshold << "\n";
        out << "train.kmeansll_hier_fixed_k1 = " << config.train.kmeansll_hier_fixed_k1 << "\n";
        out << "train.kmeans_large_k_threshold = " << config.train.kmeans_large_k_threshold << "\n";
        out << "train.kmeans_large_k_hier2_enable = " << (config.train.kmeans_large_k_hier2_enable ? "true" : "false")
            << "\n";
        out << "train.kmeans_large_k_fixed_k1 = " << config.train.kmeans_large_k_fixed_k1 << "\n";
        out << "train.kmeans_large_k_k1_min = " << config.train.kmeans_large_k_k1_min << "\n";
        out << "train.kmeans_large_k_k1_max = " << config.train.kmeans_large_k_k1_max << "\n";
        out << "train.kmeans_large_k_k1_pow2 = " << (config.train.kmeans_large_k_k1_pow2 ? "true" : "false") << "\n";
        out << "train.kmeans_large_k_top_coarse = " << config.train.kmeans_large_k_top_coarse << "\n";
        out << "train.kmeans_large_k_hier2_train = " << (config.train.kmeans_large_k_hier2_train ? "true" : "false") <<
            "\n";
        out << "train.kmeans_large_k_hier2_encode = " << (config.train.kmeans_large_k_hier2_encode ? "true" : "false")
            << "\n";
        out << "train.save_init_codes = " << (config.train.save_init_codes ? "true" : "false") << "\n";
        out << "train.init_codes_path = \"" << config.train.init_codes_path << "\"\n";
        out << "train.encode_only_after_layer = " << (config.train.encode_only_after_layer ? "true" : "false") << "\n";

        out << "train.seed = " << config.train.seed << "\n";
        out << "train.linkage.root_percentile = " << config.train.linkage.root_percentile << "\n";
        out << "train.linkage.num_layers = " << config.train.linkage.num_layers << "\n";
        out << "train.linkage.max_depth = " << config.train.linkage.max_depth << "\n";
        out << "train.linkage.knn_k = " << config.train.linkage.knn_k << "\n";
        out << "train.linkage.depth_k = " << config.train.linkage.depth_k << "\n";
        out << "train.linkage.icm_round = " << config.train.linkage.icm_round << "\n";
        out << "train.linkage.use_ils = " << (config.train.linkage.use_ils ? "true" : "false") << "\n";
        out << "train.linkage.ils_rounds = " << config.train.linkage.ils_rounds << "\n";
        out << "train.linkage.ils_perturb_layers = " << config.train.linkage.ils_perturb_layers << "\n";
        out << "train.linkage.seed = " << config.train.linkage.seed << "\n";
        out << "train.init_linkage_icm_round = " << config.train.init_linkage_icm_round << "\n";
        out << "train.init_linkage_ils_rounds = " << config.train.init_linkage_ils_rounds << "\n";
        out << "train.init_linkage_hybrid_varroot_rounds = " << config.train.init_linkage_hybrid_varroot_rounds << "\n";
        out << "train.init_linkage_mode = \"" << config.train.init_linkage_mode << "\"\n";

        out << "base.encode.enabled = " << (config.base.encode.enabled ? "true" : "false") << "\n";
        out << "base.encode.use_abs = " << (config.base.encode.use_abs ? "true" : "false") << "\n";
        out << "base.encode.H_beam = " << config.base.encode.H_beam << "\n";
        out << "base.encode.ils_iters = " << config.base.encode.ils_iters << "\n";
        out << "base.encode.icm_iters = " << config.base.encode.icm_iters << "\n";
        out << "base.encode.perturb_k = " << config.base.encode.perturb_k << "\n";
        out << "base.encode.hnorms = " << config.base.encode.hnorms << "\n";
        out << "base.encode.seed = " << config.base.encode.seed << "\n";

        out << "base.linkage.enabled = " << (config.base.linkage.enabled ? "true" : "false") << "\n";
        out << "base.linkage.root_percentile = " << config.base.linkage.root_percentile << "\n";
        out << "base.linkage.num_layers = " << config.base.linkage.num_layers << "\n";
        out << "base.linkage.max_depth = " << config.base.linkage.max_depth << "\n";
        out << "base.linkage.knn_k = " << config.base.linkage.knn_k << "\n";
        out << "base.linkage.depth_k = " << config.base.linkage.depth_k << "\n";
        out << "base.linkage.icm_round = " << config.base.linkage.icm_round << "\n";
        out << "base.linkage.use_ils = " << (config.base.linkage.use_ils ? "true" : "false") << "\n";
        out << "base.linkage.ils_rounds = " << config.base.linkage.ils_rounds << "\n";
        out << "base.linkage.ils_perturb_layers = " << config.base.linkage.ils_perturb_layers << "\n";
        out << "base.linkage.seed = " << config.base.linkage.seed << "\n";
        out << "hnsw.M = " << config.hnsw.M << "\n";
        out << "hnsw.candidate_multiplier_good = " << config.hnsw.candidate_multiplier_good << "\n";
        out << "hnsw.candidate_multiplier_bad = " << config.hnsw.candidate_multiplier_bad << "\n";
        out << "hnsw.ef_construction_cap = " << config.hnsw.ef_construction_cap << "\n";

        out << "virtual.enabled = " << (config.virtual_cfg.enabled ? "true" : "false") << "\n";
        out << "virtual.virtual_ratio = " << config.virtual_cfg.virtual_ratio << "\n";
        out << "virtual.good_fraction = " << config.virtual_cfg.good_fraction << "\n";
        out << "virtual.min_virtual = " << config.virtual_cfg.min_virtual << "\n";
        out << "virtual.max_virtual = " << config.virtual_cfg.max_virtual << "\n";
        out << "virtual.alpha_bad = " << config.virtual_cfg.alpha_bad << "\n";
        out << "virtual.use_fixed_virtual_per_cluster = "
            << (config.virtual_cfg.use_fixed_virtual_per_cluster ? "true" : "false") << "\n";
        out << "virtual.fixed_virtual_per_cluster = " << config.virtual_cfg.fixed_virtual_per_cluster << "\n";
        out << "virtual.umap_knn_k = " << config.virtual_cfg.umap_knn_k << "\n";
        out << "virtual.local_connectivity = " << config.virtual_cfg.local_connectivity << "\n";
        out << "virtual.overlap_thr = " << config.virtual_cfg.overlap_thr << "\n";
        out << "virtual.prefer_peaks = " << (config.virtual_cfg.prefer_peaks ? "true" : "false") << "\n";
        out << "virtual.anchor_neighbor_k = " << config.virtual_cfg.anchor_neighbor_k << "\n";

        out << "io.pre_fix = \"" << config.io.pre_fix << "\"\n";
        out << "io.config_file = \"" << config.io.config_file << "\"\n";
        out << "io.train_file = \"" << config.io.train_file << "\"\n";
        out << "io.base_file = \"" << config.io.base_file << "\"\n";
        out << "io.linkage_file = \"" << config.io.linkage_file << "\"\n";
        out << "io.save_train = " << (config.io.save_train ? "true" : "false") << "\n";
        out << "io.save_base = " << (config.io.save_base ? "true" : "false") << "\n";
        out << "io.save_linkage = " << (config.io.save_linkage ? "true" : "false") << "\n";
        out << "io.result_format = \"" << config.io.result_format << "\"\n";
        out << "io.hdf5_layout = \"" << config.io.hdf5_layout << "\"\n";
        out << "io.index_dtype = \"" << config.io.index_dtype << "\"\n";
        out << "io.load_date = \"" << config.io.load_date << "\"\n";
        out << "io.load_seq = \"" << config.io.load_seq << "\"\n";

        out << "large.enabled = " << (config.large.enabled ? "true" : "false") << "\n";
        out << "large.train_format = \"" << config.large.train_format << "\"\n";
        out << "large.base_format = \"" << config.large.base_format << "\"\n";
        out << "large.query_format = \"" << config.large.query_format << "\"\n";
        out << "large.gt_format = \"" << config.large.gt_format << "\"\n";
        out << "large.output_dir = \"" << config.large.output_dir << "\"\n";
        out << "large.tmp_dir = \"" << config.large.tmp_dir << "\"\n";
        out << "large.train_block = " << config.large.train_block << "\n";
        out << "large.base_block = " << config.large.base_block << "\n";
        out << "large.base_shard_size = " << config.large.base_shard_size << "\n";
        out << "large.cluster_bucket_size = " << config.large.cluster_bucket_size << "\n";
        out << "large.bucket_flush_mb = " << config.large.bucket_flush_mb << "\n";
        out << "large.write_vector_bucket = " << (config.large.write_vector_bucket ? "true" : "false") << "\n";
        out << "large.write_basic_to_bucket = " << (config.large.write_basic_to_bucket ? "true" : "false") << "\n";
        out << "large.base_linkage_store_coeffs_f32 = "
            << (config.large.base_linkage_store_coeffs_f32 ? "true" : "false") << "\n";
        out << "large.cleanup.enabled = " << (config.large.cleanup.enabled ? "true" : "false") << "\n";
        out << "large.cleanup.preset = \"" << config.large.cleanup.preset << "\"\n";
        out << "large.cleanup.keep_base_basic = " << config.large.cleanup.keep_base_basic << "\n";
        out << "large.cleanup.keep_base_basic_cluster_id = " << config.large.cleanup.keep_base_basic_cluster_id << "\n";
        out << "large.cleanup.keep_base_basic_ivf_lists = " << config.large.cleanup.keep_base_basic_ivf_lists << "\n";
        out << "large.cleanup.keep_base_list = " << config.large.cleanup.keep_base_list << "\n";
        out << "large.cleanup.keep_base_list_raw = " << config.large.cleanup.keep_base_list_raw << "\n";
        out << "large.cleanup.keep_linkage_list_f32 = " << config.large.cleanup.keep_linkage_list_f32 << "\n";
        out << "large.cleanup.keep_linkage_parent_u32 = " << config.large.cleanup.keep_linkage_parent_u32 << "\n";
        out << "large.linkage_checkpoint = " << (config.large.linkage_checkpoint ? "true" : "false") << "\n";
        out << "large.base_basic_checkpoint = " << (config.large.base_basic_checkpoint ? "true" : "false") << "\n";
        out << "large.profile_timing = " << (config.large.profile_timing ? "true" : "false") << "\n";
        out << "large.archive_log = " << (config.large.archive_log ? "true" : "false") << "\n";
        out << "large.protect_existing_outputs = "
            << (config.large.protect_existing_outputs ? "true" : "false") << "\n";
        out << "large.linkage_store_parent_u32 = "
            << (config.large.linkage_store_parent_u32 ? "true" : "false") << "\n";
        out << "large.linkage_coeff_codec.enabled = " << (config.large.linkage_coeff_codec.enabled ? "true" : "false")
            << "\n";
        out << "large.linkage_coeff_codec.granularity = \"" << config.large.linkage_coeff_codec.granularity << "\"\n";
        out << "large.linkage_coeff_codec.bits_per_layer = [";
        for (std::size_t i = 0; i < config.large.linkage_coeff_codec.bits_per_layer.size(); ++i) {
            if (i) out << ", ";
            out << config.large.linkage_coeff_codec.bits_per_layer[i];
        }
        out << "]\n";
        out << "large.linkage_coeff_codec.p_first_candidates = [";
        for (std::size_t i = 0; i < config.large.linkage_coeff_codec.p_first_candidates.size(); ++i) {
            if (i) out << ", ";
            out << config.large.linkage_coeff_codec.p_first_candidates[i];
        }
        out << "]\n";
        out << "large.linkage_coeff_codec.p_rest_candidates = [";
        for (std::size_t i = 0; i < config.large.linkage_coeff_codec.p_rest_candidates.size(); ++i) {
            if (i) out << ", ";
            out << config.large.linkage_coeff_codec.p_rest_candidates[i];
        }
        out << "]\n";
        out << "large.linkage_coeff_codec.use_weighted_quantile = "
            << (config.large.linkage_coeff_codec.use_weighted_quantile ? "true" : "false") << "\n";
        out << "large.linkage_coeff_codec.allow_clip = "
            << (config.large.linkage_coeff_codec.allow_clip ? "true" : "false") << "\n";
        out << "large.linkage_coeff_codec.fit_scale = "
            << (config.large.linkage_coeff_codec.fit_scale ? "true" : "false") << "\n";
        out << "large.linkage_coeff_codec.q_refine_sweeps = " << config.large.linkage_coeff_codec.q_refine_sweeps <<
            "\n";
        out << "large.linkage_coeff_codec.q_refine_max_layer = " << config.large.linkage_coeff_codec.q_refine_max_layer
            << "\n";
        out << "large.linkage_coeff_codec.q_refine_step_limit = " << config.large.linkage_coeff_codec.
                                                                            q_refine_step_limit << "\n";

        out << "eval.base.enabled = " << (config.eval.base_enabled ? "true" : "false") << "\n";
        out << "eval.metric_mode = " << config.eval.metric_mode << "\n";
        out << "eval.metric_gt_topks = [";
        for (std::size_t i = 0; i < config.eval.metric_gt_topks.size(); ++i) {
            if (i) out << ", ";
            out << config.eval.metric_gt_topks[i];
        }
        out << "]\n";
        out << "eval.base.use_ivf = " << (config.eval.base_use_ivf ? "true" : "false") << "\n";
        out << "eval.base.nprobe = " << config.eval.base_nprobe << "\n";
        out << "eval.base.warmup = " << config.eval.base_warmup << "\n";
        out << "eval.base.repeat = " << config.eval.base_repeat << "\n";
        out << "eval.linkage.enabled = " << (config.eval.linkage_enabled ? "true" : "false") << "\n";
        out << "eval.linkage.use_ivf_disk = " << (config.eval.linkage_use_ivf_disk ? "true" : "false") << "\n";
        out << "eval.linkage.nprobe = " << config.eval.linkage_nprobe << "\n";
        out << "eval.linkage.query_block = " << config.eval.linkage_query_block << "\n";
        out << "eval.linkage.warmup = " << config.eval.linkage_warmup << "\n";
        out << "eval.linkage.repeat = " << config.eval.linkage_repeat << "\n";
        // NOTE: legacy disk linkage recall toggles were removed; lookup path is always used.
        out << "eval.linkage.coeff_mode = \"" << config.eval.linkage_coeff_mode << "\"\n";
        out << "eval.bench.quiet = " << (config.eval.bench_quiet ? "true" : "false") << "\n";
        out << "eval.linkage.parent_louds_enable = " << (config.eval.linkage_parent_louds_enable ? "true" : "false") <<
            "\n";
        out << "eval.linkage.parent_louds_native_eval = " << (config.eval.linkage_parent_louds_native_eval
                                                                  ? "true"
                                                                  : "false") << "\n";
        out << "eval.linkage.parent_louds_select_stride = " << config.eval.parent_louds_select_stride << "\n";
        out << "eval.linkage.parent_louds_rank_words_per_super_log2 = " << config.eval.
            parent_louds_rank_words_per_super_log2 << "\n";
        out << "eval.linkage.parent_louds_build_indices = " << (config.eval.parent_louds_build_indices
                                                                    ? "true"
                                                                    : "false") << "\n";
        out << "eval.linkage.parent_adaptive_u16_cache = " << (config.eval.linkage_parent_adaptive_u16_cache
                                                                   ? "true"
                                                                   : "false") << "\n";
        out << "eval.linkage.preload_clusters_io_threads = " << config.eval.linkage_preload_clusters_io_threads << "\n";
        out << "eval.linkage.louds_huffman_bench_times = " << config.eval.linkage_louds_huffman_bench_times << "\n";
        out << "eval.linkage.gpu_scan_enable = " << (config.eval.linkage_gpu_scan_enable ? "true" : "false") << "\n";
        out << "eval.linkage.gpu_scan_tile256 = " << (config.eval.linkage_gpu_scan_tile256 ? "true" : "false") << "\n";
        out << "eval.linkage.gpu_scan_max_nc = " << config.eval.linkage_gpu_scan_max_nc << "\n";
        out << "eval.linkage.gpu_scan_cache_mb = " << config.eval.linkage_gpu_scan_cache_mb << "\n";
        out << "eval.linkage.gpu_norm_enable = " << (config.eval.linkage_gpu_norm_enable ? "true" : "false") << "\n";
        out << "eval.disk_norm2_mode = \"" << config.eval.disk_norm2_mode << "\"\n";
        out << "eval.disk_norm2_lut_kmeans_niter = " << config.eval.disk_norm2_lut_kmeans_niter << "\n";
        out << "eval.disk_norm2_lut_log_alpha = " << config.eval.disk_norm2_lut_log_alpha << "\n";
        out << "eval.disk_norm2_lut_piecewise_p1 = " << config.eval.disk_norm2_lut_piecewise_p1 << "\n";
        out << "eval.disk_norm2_lut_piecewise_p2 = " << config.eval.disk_norm2_lut_piecewise_p2 << "\n";
        out << "eval.disk_norm2_lut_piecewise_count_weight = " << config.eval.disk_norm2_lut_piecewise_count_weight <<
            "\n";
        out << "eval.disk_norm2_lut_piecewise_range_weight = " << config.eval.disk_norm2_lut_piecewise_range_weight <<
            "\n";
        out << "eval.linkage.ivf_probe_mode = \"" << config.eval.linkage_ivf_probe_mode << "\"\n";
        out << "eval.linkage.ivf_hier2_top_coarse = " << config.eval.linkage_ivf_hier2_top_coarse << "\n";
        out << "eval.linkage.ivf_hier2_kmeans_niter = " << config.eval.linkage_ivf_hier2_kmeans_niter << "\n";
        out << "eval.linkage.ivf_hnsw_M = " << config.eval.linkage_ivf_hnsw_M << "\n";
        out << "eval.linkage.ivf_hnsw_ef_construction = " << config.eval.linkage_ivf_hnsw_ef_construction << "\n";
        out << "eval.linkage.ivf_hnsw_ef_search = " << config.eval.linkage_ivf_hnsw_ef_search << "\n";
        out << "eval.linkage.norm2_store_float = " << (config.eval.linkage_norm2_store_float ? "true" : "false") <<
            "\n";
        out << "eval.linkage.norm2_store_int8 = " << (config.eval.linkage_norm2_store_int8 ? "true" : "false") << "\n";
        out << "eval.linkage.archive_eval_result = " << (config.eval.linkage_archive_eval_result ? "true" : "false") <<
            "\n";
    }

    namespace
    {
        bool StartsWith(const std::string& s, const char* prefix) {
            const std::size_t n = std::char_traits<char>::length(prefix);
            return s.size() >= n && s.compare(0, n, prefix) == 0;
        }

        bool IsHashRelevantKey(const Config& config, const std::string& key) {
            // This is intentionally *exact*: it includes only keys that are directly used in large-scale
            // store hash computations (base_basic/base_list/linkage_list/coeff codec) in
            // `stlq_gpu/src/app/large_store_hash.cpp`.
            //
            // NOTE: store hashes also depend on non-config inputs (e.g. resolved base path/dtype/ntotal cap)
            // and on trained artifacts (TrainResult rotation/codebooks). Those are logged separately.

            static const std::unordered_set<std::string> kKeys = {
                // ComputeBaseBasicStoreHash inputs (Config portion).
                "dataset.name",
                "dataset.nbase_set",
                "dataset.nbase",
                "model.m",
                "model.h_vec",
                "large.base_shard_size",
                "large.cluster_bucket_size",
                "large.bucket_flush_mb",
                "large.write_vector_bucket",
                "large.write_basic_to_bucket",
                "base.encode.use_abs",
                "base.encode.H_beam",
                "base.encode.ils_iters",
                "base.encode.icm_iters",
                "base.encode.perturb_k",
                "base.encode.hnorms",
                "base.encode.seed",

                // ComputeBaseListStoreHash additional inputs.
                "large.write_basic_to_bucket",
                "large.write_vector_bucket",

                // ComputeLinkageListStoreHash inputs (Config portion).
                "model.h0_one",
                "base.linkage.root_percentile",
                "base.linkage.num_layers",
                "base.linkage.max_depth",
                "base.linkage.knn_k",
                "base.linkage.depth_k",
                "base.linkage.icm_round",
                "base.linkage.use_ils",
                "base.linkage.ils_rounds",
                "base.linkage.ils_perturb_layers",
                "base.linkage.seed",
                "virtual.enabled",
                "virtual.virtual_ratio",
                "virtual.good_fraction",
                "virtual.min_virtual",
                "virtual.max_virtual",
                "virtual.alpha_bad",
                "virtual.use_fixed_virtual_per_cluster",
                "virtual.fixed_virtual_per_cluster",
                "virtual.umap_knn_k",
                "virtual.local_connectivity",
                "virtual.overlap_thr",
                "virtual.prefer_peaks",
                "virtual.anchor_neighbor_k",
                "hnsw.M",
                "hnsw.candidate_multiplier_good",
                "hnsw.candidate_multiplier_bad",

                // ComputeLinkageCoeffCodecHash inputs (Config portion).
                "large.linkage_coeff_codec.granularity",
                "large.linkage_coeff_codec.bits_per_layer",
                "large.linkage_coeff_codec.p_first_candidates",
                "large.linkage_coeff_codec.p_rest_candidates",
                "large.linkage_coeff_codec.use_weighted_quantile",
                "large.linkage_coeff_codec.allow_clip",
                "large.linkage_coeff_codec.fit_scale",
                "large.linkage_coeff_codec.q_refine_sweeps",
                "large.linkage_coeff_codec.q_refine_max_layer",
                "large.linkage_coeff_codec.q_refine_step_limit",
            };

            if (key == "hnsw.ef_construction_cap") {
                return config.hnsw.ef_construction_cap > 0;
            }
            return kKeys.find(key) != kKeys.end();
        }

        bool IsPotentialHashRelevantKey(const std::string& key) {
            // "Potentially hash-relevant" means: does not appear directly in store hash inputs, but can
            // affect *derived* inputs that do (e.g. TrainResult R/codebooks, resolved dataset file paths,
            // large pipeline I/O formats).

            // eval.* is recall-only; not part of any store identity.
            if (StartsWith(key, "eval.")) return false;

            // Operational paths and safety toggles (do not affect artifact contents).
            if (StartsWith(key, "advanced.")) return false;
            if (key == "large.output_dir") return false;
            if (key == "large.tmp_dir") return false;
            if (key == "large.profile_timing") return false;
            if (key == "large.archive_log") return false;
            if (key == "large.protect_existing_outputs") return false;
            // Cleanup / retention policy: purely operational, never affects store contents or hashes.
            if (StartsWith(key, "large.cleanup.")) return false;
            if (key == "large.linkage_store_parent_u32") return false;

            // These dataset fields are evaluation-related.
            if (key == "dataset.groundtruth_add1") return false;
            // Dataset root path relocation should not be treated as hash-relevant; store identity is
            // keyed by dataset.name + train artifacts + store config.
            if (key == "dataset.data_root") return false;
            if (key == "dataset.nquery_set") return false;
            if (key == "dataset.nquery") return false;
            if (key == "dataset.k_set") return false;
            if (key == "dataset.k") return false;

            // Enable/log toggles generally do not affect artifacts.
            if (key == "train.log_metrics") return false;
            if (key == "train.log_linkage_pre_c1") return false;
            if (key == "train.save_init_codes") return false;
            if (key == "train.init_codes_path") return false;
            if (key == "train.init_enabled") return false;
            if (key == "train.ckpt_after_init_basic") return false;
            if (key == "train.exit_after_ckpt_init_basic") return false;
            if (StartsWith(key, "train.ckpt.")) return false;
            // Init-linkage performance tuning only; does not affect store artifact content.
            // NOTE: init-linkage knobs can change the trained artifacts when `train.init_enabled=true`,
            // so treat them as potentially hash-relevant (record in snapshots for reproducibility).

            if (key == "base.encode.enabled") return false;
            if (key == "base.linkage.enabled") return false;
            if (key == "train.enabled") return false;

            if (key == "io.save_train") return false;
            if (key == "io.save_base") return false;
            if (key == "io.save_linkage") return false;
            if (key == "io.config_file") return false;

            // Likely-to-matter upstream knobs.
            return StartsWith(key, "train.") ||
                StartsWith(key, "runtime.") ||
                StartsWith(key, "dataset.") ||
                StartsWith(key, "io.") ||
                StartsWith(key, "large.");
        }

        void AppendConfigSnapshotPartitioned(std::ostream& out, const Config& config) {
            std::ostringstream oss;
            AppendConfigSnapshotFull(oss, config);
            const std::string full = oss.str();

            std::vector<std::string> lines;
            {
                std::stringstream ss(full);
                std::string line;
                while (std::getline(ss, line)) {
                    lines.push_back(line);
                }
            }

            auto parse_key = [](const std::string& line) -> std::string
            {
                const std::size_t pos = line.find(" = ");
                if (pos == std::string::npos) return {};
                return line.substr(0, pos);
            };

            out << "# ==================== HASH/IDENTITY-RELEVANT CONFIG ====================\n";
            out << "# NOTE: this section includes only keys directly used by large store hash computations.\n";
            out << "# Store hashes also depend on derived runtime inputs (e.g. resolved base path/dtype/ntotal cap)\n";
            out << "# and on trained artifacts (TrainResult rotation/codebooks). See run_root/store_hashes.txt.\n";
            for (const auto& line : lines) {
                const std::string key = parse_key(line);
                if (!key.empty() && IsHashRelevantKey(config, key)) {
                    out << line << "\n";
                }
            }

            out << "\n# ==================== POTENTIALLY HASH-RELEVANT CONFIG ====================\n";
            out << "# NOTE: these keys are not used directly in large store hashes, but can affect derived inputs\n";
            out << "# (e.g. TrainResult) or I/O resolution that will change store hashes in practice.\n";
            for (const auto& line : lines) {
                const std::string key = parse_key(line);
                if (!key.empty() && !IsHashRelevantKey(config, key) && IsPotentialHashRelevantKey(key)) {
                    out << line << "\n";
                }
            }

            out << "\n# ==================== OTHER CONFIG (COMPLEMENT) ====================\n";
            for (const auto& line : lines) {
                const std::string key = parse_key(line);
                if (key.empty() || (!IsHashRelevantKey(config, key) && !IsPotentialHashRelevantKey(key))) {
                    out << line << "\n";
                }
            }
        }
    } // namespace

    void AppendConfigSnapshot(std::ostream& out, const Config& config) {
        AppendConfigSnapshotPartitioned(out, config);
    }

    bool SaveConfigSnapshot(const Config& config, const std::string& path, std::string* error) {
        std::ofstream file(path);
        if (!file.is_open()) {
            if (error) {
                *error = "Failed to write config snapshot: " + path;
            }
            return false;
        }

        AppendConfigSnapshot(file, config);
        return true;
    }

    std::uint64_t ComputeConfigHash(const Config& config) {
        std::ostringstream oss;
        // Use the full unpartitioned snapshot for hashing to keep the hash stable w.r.t. presentation changes.
        AppendConfigSnapshotFull(oss, config);
        const std::string snapshot = oss.str();
        return Fnv1a64(snapshot.data(), snapshot.size());
    }

    std::uint64_t ComputeTrainResumeSigU64(const Config& config) {
        // Use a config-derived signature for resume-compatibility checks.
        // Goal: catch semantic mismatches while allowing benign differences like:
        // - train.max_R_iters (we treat it as "additional iters")
        // - periodic checkpoint knobs
        // - logging/perf/paths (runtime/io/large dirs)
        //
        // We hash a filtered snapshot, line-by-line, to keep behavior transparent.
        std::ostringstream oss;
        AppendConfigSnapshotFull(oss, config);
        const std::string full = oss.str();

        auto parse_key = [](const std::string& line) -> std::string
        {
            const std::size_t pos = line.find(" = ");
            if (pos == std::string::npos) return {};
            return line.substr(0, pos);
        };
        auto include_key = [&config](const std::string& key) -> bool
        {
            if (key.empty()) return false;
            if (key == "dataset.data_root") return false;
            // Evaluation-only dataset knobs (do not affect training semantics).
            if (key == "dataset.groundtruth_add1") return false;
            if (StartsWith(key, "advanced.")) return false;
            if (key == "dataset.nbase_set") return false;
            if (key == "dataset.nbase") return false;
            if (key == "dataset.nquery_set") return false;
            if (key == "dataset.nquery") return false;
            if (key == "dataset.k_set") return false;
            if (key == "dataset.k") return false;
            if (key == "train.enabled") return false;
            if (key == "train.max_R_iters") return false;
            if (key == "train.exit_after_rvq_init") return false;
            if (key == "train.ckpt_after_init_basic") return false;
            if (key == "train.exit_after_ckpt_init_basic") return false;
            if (key == "train.log_metrics") return false;
            if (key == "train.log_linkage_pre_c1") return false;
            if (key == "train.init_enabled") return false;
            if (StartsWith(key, "train.ckpt.")) return false;
            if (StartsWith(key, "io.")) return false;
            if (StartsWith(key, "runtime.")) return false;
            if (StartsWith(key, "eval.")) return false;
            if (StartsWith(key, "base.")) return false;
            if (StartsWith(key, "large.")) return false;
            if (key == "hnsw.ef_construction_cap" && config.hnsw.ef_construction_cap <= 0) return false;
            return StartsWith(key, "dataset.") ||
                StartsWith(key, "model.") ||
                StartsWith(key, "train.") ||
                StartsWith(key, "virtual.") ||
                StartsWith(key, "hnsw.");
        };

        std::uint64_t h = 1469598103934665603ull;
        std::stringstream ss(full);
        std::string line;
        while (std::getline(ss, line)) {
            const std::string key = parse_key(line);
            if (!include_key(key)) continue;
            // Include the full line to capture values; include newline for unambiguous concatenation.
            h = Fnv1a64WithSeed(line.data(), line.size(), h);
            const char nl = '\n';
            h = Fnv1a64WithSeed(&nl, 1, h);
        }
        return h;
    }
} // namespace stlq
