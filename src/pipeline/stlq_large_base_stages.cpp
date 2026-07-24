#include "stlq/pipeline/stlq_large_base_stages.h"

#include "stlq/pipeline/app_utils.h"
#include "stlq/pipeline/large_store_hash.h"
#include "stlq/pipeline/large_store_lifecycle.h"
#include "stlq/pipeline/main_virtual_helpers.h"
#include "stlq/eval/recall_disk.h"
#include "stlq/io/dataset_io.h"
#include "stlq/io/dataset_reader_factory.h"
#include "stlq/io/result_io.h"
#include "stlq/common/logger.h"
#include "stlq/quantizer/encode_base_streaming.h"
#include "stlq/quantizer/reconstruct.h"
#include "stlq/common/timer.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <utility>

namespace stlq {

std::string FormatHexU64(std::uint64_t v) {
    std::ostringstream oss;
    oss << "0x" << std::hex << v << std::dec;
    return oss.str();
}

int OpenLargeBaseReaderStage(const stlq::Config& config,
                             LargeBaseReaderState& base_reader,
                             std::string& error) {
    using namespace stlq;
    io::DatasetVectorReader opened;
    if (!io::OpenDatasetVectorReader(config, io::DatasetRole::kBase, &opened, &error)) {
        LogError(error);
        return 1;
    }

    base_reader.reader = std::move(opened);
    base_reader.is_u8 = base_reader.reader.is_u8;
    base_reader.d = base_reader.reader.d;
    base_reader.n = base_reader.reader.n;
    base_reader.path = base_reader.reader.path;

    if (base_reader.reader.format == io::VectorFileFormat::kBvecs) {
        base_reader.format = BaseFormat::kBvecs;
    } else if (base_reader.reader.format == io::VectorFileFormat::kFbin) {
        base_reader.format = BaseFormat::kFbin;
    } else if (base_reader.reader.format == io::VectorFileFormat::kFvecs) {
        base_reader.format = BaseFormat::kFvecs;
    } else {
        LogError("OpenLargeBaseReaderStage: unsupported base reader format.");
        return 1;
    }

    // LogInfo("Large-scale base reader: " + base_reader.path + " d=" +
    //         std::to_string(base_reader.d) + " n=" + std::to_string(base_reader.n) +
    //         " dtype=" + std::string(base_reader.is_u8 ? "u8(bvecs/siftbin)" : "f32(fvecs)"));
    return 0;
}

int PrepareLargeRunStateStage(const stlq::Config& config,
                              const stlq::TrainResult& train_result,
                              const LargeBaseReaderState& base_reader,
                              LargeRunState& run_state) {
    using namespace stlq;
    run_state.expected_nlist = config.model.h_vec.empty() ? 0 : config.model.h_vec[0];
    run_state.base_ntotal_effective =
        (config.dataset.nbase_set && config.dataset.nbase > 0)
            ? std::min<std::uint64_t>(base_reader.n, static_cast<std::uint64_t>(config.dataset.nbase))
            : base_reader.n;

    run_state.store_cfg.d = base_reader.d;
    run_state.store_cfg.m = config.model.m;
    run_state.store_cfg.h_vec = config.model.h_vec;
    run_state.store_cfg.shard_size = std::max(1, config.large.base_shard_size);
    run_state.store_cfg.bucket_size = std::max(1, config.large.cluster_bucket_size);
    run_state.store_cfg.bucket_flush_mb = std::max(1, config.large.bucket_flush_mb);
    run_state.store_cfg.write_vector_bucket = config.large.write_vector_bucket;
    run_state.store_cfg.write_basic_to_bucket = config.large.write_basic_to_bucket;
    if (run_state.store_cfg.write_vector_bucket) {
        run_state.store_cfg.raw_bytes_per_vec =
            base_reader.is_u8 ? base_reader.d : (base_reader.d * static_cast<int>(sizeof(float)));
    }

    run_state.base_basic_hash =
        app::ComputeBaseBasicStoreHash(config, train_result, run_state.store_cfg, base_reader.is_u8);
    run_state.base_list_hash_identity =
        app::ComputeBaseListStoreHash(config,
                                      run_state.base_basic_hash,
                                      run_state.expected_nlist,
                                      run_state.base_ntotal_effective);
    run_state.linkage_list_hash_identity =
        app::ComputeLinkageListStoreHash(config,
                                       train_result,
                                       run_state.base_list_hash_identity,
                                       run_state.expected_nlist);
    // IMPORTANT: the run directory identity must not depend on coeff codec knobs (bits_per_layer, etc).
    // Those knobs only affect a derived, fully-regenerable artifact (the int8 coeff codec), and can be
    // rebuilt in-place from float coeffs without rebuilding linkage_list.
    run_state.run_hash_identity = run_state.linkage_list_hash_identity;
    run_state.coeff_codec_hash =
        app::ComputeLinkageCoeffCodecHash(config, run_state.linkage_list_hash_identity);
    run_state.eval_only =
        (!config.train.enabled && !config.base.encode.enabled && !config.base.linkage.enabled);

    run_state.run_tag = config.io.pre_fix + "__" + FormatHexU64(run_state.run_hash_identity);
    run_state.run_root =
        (std::filesystem::path(config.large.output_dir) / config.dataset.name / run_state.run_tag).string();
    run_state.run_tmp_root =
        (std::filesystem::path(config.large.tmp_dir) / config.dataset.name / run_state.run_tag).string();
    run_state.effective_run_root = run_state.run_root;
    run_state.effective_run_tmp_root = run_state.run_tmp_root;
    if (run_state.eval_only && config.io.linkage_file_set && !config.io.linkage_file.empty()) {
        std::error_code ec;
        const std::filesystem::path linkage_path(config.io.linkage_file);
        if (std::filesystem::exists(linkage_path, ec) && !ec &&
            std::filesystem::is_directory(linkage_path, ec) && !ec &&
            linkage_path.filename() == "linkage_list") {
            const std::filesystem::path explicit_run_root = linkage_path.parent_path();
            if (!explicit_run_root.empty()) {
                run_state.effective_run_root = explicit_run_root.string();
            }
        }
    }
    if (run_state.eval_only && !std::filesystem::exists(run_state.effective_run_root)) {
        LogError("Eval-only mode requires existing run dir: " + run_state.effective_run_root +
                 " ; enable base.encode/base.linkage to rebuild or change io.pre_fix.");
        return 1;
    }
    return 0;
}

int WriteLargeRunArtifactsStage(const stlq::Config& config,
                                const stlq::TrainResult& train_result,
                                const LargeBaseReaderState& base_reader,
                                const LargeRunState& run_state,
                                const std::string& frozen_original_cfg,
                                const std::filesystem::path& frozen_original_cfg_name,
                                std::string& error) {
    using namespace stlq;
    if (config.large.archive_log) {
        const std::filesystem::path log_dir = std::filesystem::path(run_state.effective_run_root) / "log";
        std::error_code ec;
        std::filesystem::create_directories(log_dir, ec);
        if (ec) {
            LogWarn("Failed to create log dir for runtime archive: " + log_dir.string());
        } else {
            std::string log_err;
            const std::filesystem::path log_path = app::NextRuntimeLogArchivePath(log_dir);
            if (!SetLogFile(log_path.string(), /*append=*/false, &log_err)) {
                LogWarn(log_err);
            } else {
                LogInfo("Archiving runtime log to: " + log_path.string());
            }
        }
    }

    LogInfo("Large run tag: " + run_state.run_tag +
            " (coeff_codec_hash=" + FormatHexU64(run_state.coeff_codec_hash) + ")");
    if (run_state.effective_run_root != run_state.run_root) {
        LogInfo("Eval-only: using explicit run dir from io.linkage_file: " + run_state.effective_run_root);
    }

    // Write a human-readable config snapshot alongside run outputs (text format).
    // In eval-only mode, do not overwrite the archived config backup captured during
    // the original train/encode run.
    std::error_code ec;
    std::filesystem::create_directories(run_state.effective_run_root, ec);
    if (!ec) {
        std::filesystem::create_directories(run_state.effective_run_tmp_root, ec);
    }
    if (!run_state.eval_only) {
        const std::string snap_path =
            (std::filesystem::path(run_state.effective_run_root) / "config_snapshot.txt").string();
        if (!SaveConfigSnapshot(config, snap_path, &error)) {
            LogWarn(error);
        }
        if (!frozen_original_cfg.empty()) {
            const std::filesystem::path dst_cfg_path =
                std::filesystem::path(run_state.effective_run_root) /
                (frozen_original_cfg_name.empty() ? std::filesystem::path("original_config.cfg")
                                                  : frozen_original_cfg_name);
            std::ofstream out(dst_cfg_path, std::ios::binary | std::ios::trunc);
            if (!out.is_open()) {
                LogWarn("Failed to write frozen original config file to run dir: dst=" +
                        dst_cfg_path.string());
            } else {
                out.write(frozen_original_cfg.data(),
                          static_cast<std::streamsize>(frozen_original_cfg.size()));
                if (!out) {
                    LogWarn("Failed to write frozen original config file to run dir: dst=" +
                            dst_cfg_path.string());
                }
            }
        }
    }
    // Also write a small "run state" file with dynamic information that is not part of the config
    // (e.g., the actual checkpoint R_iters loaded/produced during this run).
    const std::string state_path =
        (std::filesystem::path(run_state.effective_run_root) / "run_state.txt").string();
    std::ofstream state(state_path, std::ios::trunc);
    if (!state.is_open()) {
        LogWarn("Failed to write run state: " + state_path);
    } else {
        std::string local_err;
        const std::string resolved_train_path =
            io::MakeDatedPathForLoad(config.io.train_file, config.io.load_date, config.io.load_seq, &local_err);
        const std::string effective_train_h5 =
            (!train_result.saved_train_h5.empty()
                 ? train_result.saved_train_h5
                 : (!train_result.loaded_train_h5.empty() ? train_result.loaded_train_h5 : "<in-memory>"));
        state << "dataset.name = \"" << config.dataset.name << "\"\n";
        state << "run_tag = " << run_state.run_tag << "\n";
        state << "train.enabled = " << (config.train.enabled ? "true" : "false") << "\n";
        state << "train.init_enabled = " << (config.train.init_enabled ? "true" : "false") << "\n";
        state << "train.max_R_iters = " << config.train.max_R_iters << "\n";
        state << "train.ckpt.enabled = " << (config.train.ckpt.enabled ? "true" : "false") << "\n";
        state << "train.ckpt.every_R = " << config.train.ckpt.every_R << "\n";
        state << "io.train_file = \"" << config.io.train_file << "\"\n";
        state << "io.load_date = \"" << config.io.load_date << "\"\n";
        state << "io.load_seq = \"" << config.io.load_seq << "\"\n";
        state << "resolved_train_h5 = \"" << (resolved_train_path.empty() ? "<none>" : resolved_train_path) << "\"\n";
        if (!local_err.empty()) {
            state << "resolved_train_h5_error = \"" << local_err << "\"\n";
        }
        state << "effective_train_h5 = \"" << effective_train_h5 << "\"\n";
        state << "train_result.loaded_train_h5 = \"" << (train_result.loaded_train_h5.empty() ? "<none>" : train_result.loaded_train_h5) << "\"\n";
        state << "train_result.saved_train_h5 = \"" << (train_result.saved_train_h5.empty() ? "<none>" : train_result.saved_train_h5) << "\"\n";
        state << "train_result.R_iters = " << train_result.R_iters << "\n";
        state << "resume_sig_hex(current_config) = \"" << FormatHexU64(ComputeTrainResumeSigU64(config)) << "\"\n";
    }
    // Also write the resolved large-store hashes to simplify debugging "why did it rebuild?".
    // This file is intentionally human-readable and may be overwritten by subsequent eval-only runs.
    const std::string hashes_path =
        (std::filesystem::path(run_state.effective_run_root) / "store_hashes.txt").string();
    std::ofstream hashes(hashes_path, std::ios::trunc);
    if (!hashes.is_open()) {
        LogWarn("Failed to write store hashes: " + hashes_path);
    } else {
        hashes << "run_tag = " << run_state.run_tag << "\n";
        hashes << "run_hash_identity(linkage_list) = " << FormatHexU64(run_state.run_hash_identity) << "\n";
        hashes << "coeff_codec_hash = " << FormatHexU64(run_state.coeff_codec_hash) << "\n";
        hashes << "base_basic_hash = " << FormatHexU64(run_state.base_basic_hash) << "\n";
        hashes << "base_list_hash_identity = " << FormatHexU64(run_state.base_list_hash_identity) << "\n";
        hashes << "linkage_list_hash_identity = " << FormatHexU64(run_state.linkage_list_hash_identity) << "\n";
        hashes << "base_is_u8 = " << (base_reader.is_u8 ? "true" : "false") << "\n";
        hashes << "base_path = \"" << base_reader.path << "\"\n";
        hashes << "base_d = " << base_reader.d << "\n";
        hashes << "base_n = " << base_reader.n << "\n";
        hashes << "base_ntotal_effective = " << run_state.base_ntotal_effective << "\n";
    }
    return 0;
}

int ValidateLargeRunGuardrailsStage(const stlq::Config& config) {
    using namespace stlq;
    // Validate: base_linkage_store_coeffs_f32=false requires coeff codec enabled.
    if (!config.large.base_linkage_store_coeffs_f32 && !config.large.linkage_coeff_codec.enabled) {
        LogError("large.base_linkage_store_coeffs_f32=false requires large.linkage_coeff_codec.enabled=true "
                 "(otherwise no coefficient representation is available for linkage eval).");
        return 1;
    }

    // Guardrail: users often set large.cleanup.preset but forget to enable the cleanup engine.
    // Presets only take effect when large.cleanup.enabled=true.
    if (!config.large.cleanup.enabled && config.large.cleanup.preset != "none") {
        LogWarn("large.cleanup.preset=\"" + config.large.cleanup.preset +
                "\" but large.cleanup.enabled=false; no disk artifacts will be deleted. "
                "Set large.cleanup.enabled=1 to activate staged cleanup.");
    }
    return 0;
}

int BuildLargePrecompRootStage(const stlq::Config& config,
                               const stlq::TrainResult& train_result,
                               stlq::StreamKernelProvider* stream_kernels,
                               stlq::PrecompLargeRoot& pre_lr,
                               stlq::PrecompLargeRoot*& pre_lr_ptr,
                               std::string& error) {
    using namespace stlq;
    PrecompLargeRootBuildOptions pre_opts;
    pre_opts.build_g0s_transpose = config.runtime.precomp_large_root_g0s_transpose;
    pre_opts.g0s_transpose_max_mb = config.runtime.precomp_large_root_g0s_transpose_max_mb;
    if (!BuildPrecompLargeRoot(train_result.C_root, /*kernels=*/stream_kernels, pre_opts, &pre_lr, &error)) {
        LogError("Large pipeline requires PrecompLargeRoot for basic encoding: " + error);
        return 1;
    }
    pre_lr_ptr = &pre_lr;
    LogInfo("Using PrecompLargeRoot: h0=" + std::to_string(pre_lr.h_vec[0]) +
            " H_small=" + std::to_string(pre_lr.H_small));
    return 0;
}

int RunLargeBaseBasicStage(const stlq::Config& config,
                           const stlq::TrainResult& train_result,
                           LargeBaseReaderState& base_reader,
                           LargeRunState& run_state,
                           stlq::Precomp& pre_root,
                           bool& pre_root_ready,
                           stlq::PrecompLargeRoot* pre_lr_ptr,
                           stlq::StreamKernelProvider* stream_kernels,
                           LargeBaseBasicStageState& base_basic_state,
                           std::string& error) {
    using namespace stlq;
    base_basic_state.out_dir =
        (std::filesystem::path(run_state.effective_run_root) / "base_basic").string();
    base_basic_state.meta_path =
        (std::filesystem::path(base_basic_state.out_dir) / "meta.bin").string();
    run_state.store_cfg.dir = base_basic_state.out_dir;
    base_basic_state.hash_path = app::StoreHashPath(base_basic_state.out_dir);

    float base_beam = 0.0f;
    float base_final = 0.0f;
    base_basic_state.rebuilt = false;
    if (config.base.encode.enabled) {
        const bool enable_base_basic_ckpt = config.large.base_basic_checkpoint;
        const auto ckpt_path = app::BaseBasicCheckpointPath(base_basic_state.out_dir);
        const bool have_ckpt = enable_base_basic_ckpt && std::filesystem::exists(ckpt_path);

        app::BaseBasicCheckpoint ckpt;
        bool resume = false;
        std::uint64_t start_id = 0;
        if (have_ckpt) {
            std::string local_err;
            if (!app::ReadBaseBasicCheckpoint(ckpt_path, &ckpt, &local_err)) {
                LogError("Failed to read base_basic checkpoint: " + local_err);
                return 1;
            }
            resume = true;
            start_id = ckpt.next_id;
        }

        app::StoreLifecyclePrepareOptions lifecycle_opts;
        lifecycle_opts.label = "base_basic";
        lifecycle_opts.dir = base_basic_state.out_dir;
        lifecycle_opts.protect_existing_outputs = config.large.protect_existing_outputs;
        lifecycle_opts.preserve_existing = resume;
        if (!app::PrepareStoreForRebuildOrResume(lifecycle_opts, &error)) {
            LogError(error);
            return 1;
        }

        if (enable_base_basic_ckpt) {
            // WriteU64FileHex does not create parent directories.
            {
                std::error_code ec;
                std::filesystem::create_directories(base_basic_state.out_dir, ec);
                if (ec) {
                    LogError("Failed to create base_basic dir: " + base_basic_state.out_dir);
                    return 1;
                }
            }
            // Ensure a stable identity for resume before encoding starts.
            if (!app::WriteExpectedStoreHash("base_basic",
                                             base_basic_state.hash_path,
                                             run_state.base_basic_hash,
                                             &error)) {
                LogError(error);
                return 1;
            }
        }

        io::BaseBasicWriter writer;
        if (resume) {
            // Verify store identity matches this run.
            std::uint64_t prev_hash = 0;
            if (!app::ReadU64File(base_basic_state.hash_path, &prev_hash) ||
                prev_hash != run_state.base_basic_hash) {
                LogError("base_basic store hash mismatch for resume: " + base_basic_state.out_dir);
                return 1;
            }

            if (start_id >= static_cast<std::uint64_t>(run_state.base_ntotal_effective)) {
                LogInfo("base_basic checkpoint indicates completion (next_id=" + std::to_string(start_id) + "); skipping encode.");
            } else {
                if (!app::TruncateBaseBasicToCheckpoint(base_basic_state.out_dir, run_state.store_cfg, ckpt, &error)) {
                    LogError(error);
                    return 1;
                }
                if (!writer.OpenResume(run_state.store_cfg, start_id, ckpt.bucket_sizes, &error)) {
                    LogError(error);
                    return 1;
                }
            }
        } else {
            if (!writer.Open(run_state.store_cfg, &error)) {
                LogError(error);
                return 1;
            }

            if (enable_base_basic_ckpt) {
                app::BaseBasicCheckpoint init;
                init.nbucket = static_cast<std::uint32_t>(app::ComputeNbucketsForBasic(run_state.store_cfg));
                init.next_id = 0;
                init.bucket_sizes.assign(static_cast<std::size_t>(init.nbucket), 0ull);
                if (!app::WriteBaseBasicCheckpointAtomic(ckpt_path, init, &error)) {
                    LogError("Failed to write base_basic checkpoint: " + error);
                    return 1;
                }
            }
        }

        LogInfo("============== BaseSet Encoding ==============");
        Timer enc_timer;

        float* out_beam_ptr = config.train.log_metrics ? &base_beam : nullptr;
        float* out_final_ptr = config.train.log_metrics ? &base_final : nullptr;
        if (resume && start_id >= static_cast<std::uint64_t>(run_state.base_ntotal_effective)) {
            // No-op: store already complete.
        } else {
            StreamingBasicCommitHook hook;
            app::BaseBasicCkptCtx hook_ctx;
            const StreamingBasicCommitHook* hook_ptr = nullptr;
            if (enable_base_basic_ckpt) {
                hook_ctx.path = ckpt_path;
                hook_ctx.nbucket = static_cast<std::uint32_t>(app::ComputeNbucketsForBasic(run_state.store_cfg));
                hook.ctx = &hook_ctx;
                hook.OnCommit = &app::OnCommitBaseBasicCkpt;
                hook_ptr = &hook;
            }
            if (!pre_root_ready) {
                // Large pipeline does not use full-precomp (HxH Gram). Keep only metadata to satisfy
                // shared APIs that still take a `Precomp&`.
                pre_root = app::BuildPrecompMetaOnly(train_result.C_root);
                pre_root_ready = true;
            }

            const bool ok_encode = base_reader.is_u8
                                       ? EncodeBaseStreaming(config,
                                                             base_reader.reader.bvecs,
                                                             train_result.R,
                                                             train_result.C_root,
                                                             pre_root,
                                                             pre_lr_ptr,
                                                             stream_kernels,
                                                             &writer,
                                                             out_beam_ptr,
                                                             out_final_ptr,
                                                             &error,
                                                             start_id,
                                                             hook_ptr)
                                       : (base_reader.format == BaseFormat::kFbin)
                                         ? EncodeBaseStreamingF32(config,
                                                                  base_reader.reader.fbin,
                                                                  train_result.R,
                                                                  train_result.C_root,
                                                                  pre_root,
                                                                  pre_lr_ptr,
                                                                  stream_kernels,
                                                                  &writer,
                                                                  out_beam_ptr,
                                                                  out_final_ptr,
                                                                  &error,
                                                                  start_id,
                                                                  hook_ptr)
                                         : EncodeBaseStreamingF32(config,
                                                                  base_reader.reader.fvecs,
                                                                  train_result.R,
                                                                  train_result.C_root,
                                                                  pre_root,
                                                                  pre_lr_ptr,
                                                                  stream_kernels,
                                                                  &writer,
                                                                  out_beam_ptr,
                                                                  out_final_ptr,
                                                                  &error,
                                                                  start_id,
                                                                  hook_ptr);
            if (!ok_encode) {
                LogError(error);
                return 1;
            }
            const Timer base_close_wall;
            if (!writer.Close(&error)) {
                LogError(error);
                return 1;
            }
            const double base_close_sec = base_close_wall.ElapsedSeconds();
            if (base_close_sec > 1.0) {
                LogInfo("BaseBasicWriter Close wall time(s): " + FormatFloat(static_cast<float>(base_close_sec), 3));
            }

            if (enable_base_basic_ckpt) {
                // Finalize checkpoint (in case the last commit hook write was interrupted).
                app::BaseBasicCheckpoint final;
                final.nbucket = static_cast<std::uint32_t>(app::ComputeNbucketsForBasic(run_state.store_cfg));
                final.next_id = static_cast<std::uint64_t>(run_state.base_ntotal_effective);
                final.bucket_sizes = writer.bucket_file_bytes();
                if (final.bucket_sizes.size() == static_cast<std::size_t>(final.nbucket)) {
                    std::string local_err;
                    if (!app::WriteBaseBasicCheckpointAtomic(ckpt_path, final, &local_err)) {
                        LogWarn("Failed to write final base_basic checkpoint: " + local_err);
                    }
                }
            } else {
                const Timer base_hash_wall;
                if (!app::WriteExpectedStoreHash("base_basic",
                                                 base_basic_state.hash_path,
                                                 run_state.base_basic_hash,
                                                 &error)) {
                    LogError(error);
                    return 1;
                }
                const double base_hash_sec = base_hash_wall.ElapsedSeconds();
                if (base_hash_sec > 1.0) {
                    LogInfo("base_basic hash write wall time(s): " + FormatFloat(static_cast<float>(base_hash_sec), 3));
                }
            }

            base_basic_state.rebuilt = true;
            LogInfo(enc_timer.ReportSeconds("Large-scale base encoding finished in"));
        }
        // LogInfo("Large-scale base store dir: " + out_dir);
    } else {
        const bool do_eval_base = (config.eval.base_enabled && config.eval.base_use_ivf);
        const bool need_base_basic_store = (config.base.linkage.enabled || do_eval_base);
        if (!std::filesystem::exists(base_basic_state.meta_path)) {
            if (need_base_basic_store) {
                LogError("large.enabled=true & base.encode.enabled=false, but base_basic store is missing: " +
                         base_basic_state.meta_path);
                return 1;
            }
            LogInfo("large.enabled=true & base.encode.enabled=false: base_basic store is missing but not needed for this run; continuing without base_basic.");
        } else {
            app::StoreReuseCheckOptions reuse_opts;
            reuse_opts.label = "base_basic";
            reuse_opts.dir = base_basic_state.out_dir;
            reuse_opts.hash_path = base_basic_state.hash_path;
            reuse_opts.expected_hash = run_state.base_basic_hash;
            reuse_opts.eval_only = run_state.eval_only;
            reuse_opts.error_on_non_eval_hash_mismatch = true;
            reuse_opts.allow_missing_hash_reuse = true;
            reuse_opts.non_eval_hash_mismatch_error =
                "base_basic store hash mismatch (likely stale outputs): " + base_basic_state.out_dir +
                " ; enable base.encode to rebuild or change io.pre_fix.";
            reuse_opts.eval_only_hash_mismatch_warning =
                "base_basic store hash mismatch in eval-only mode; proceeding with existing store: " +
                base_basic_state.out_dir;
            app::StoreReuseDecision reuse_decision;
            if (!app::CheckExistingStoreReuse(reuse_opts, &reuse_decision, &error)) {
                LogError(error);
                return 1;
            }
            if (reuse_decision.adopted_previous_hash) {
                // Use the on-disk hash as the store identity for downstream derived hashes.
                run_state.base_basic_hash = reuse_decision.effective_hash;
            }
            LogInfo("large.enabled=true & base.encode.enabled=false: reusing base_basic store dir: " +
                    base_basic_state.out_dir);
        }
    }
    return 0;
}

int RunLargeIvfStage(const stlq::Config& config,
                     const LargeRunState& run_state,
                     const LargeBaseBasicStageState& base_basic_state,
                     bool do_eval_base,
                     LargeIvfStageState& ivf_state,
                     std::string& error) {
    using namespace stlq;
    // Disk linkage recall uses `linkage_list` only; it does NOT require IVF CSR lists on disk.
    // IVF CSR lists are still required for base IVF eval and for building linkage_list.
    ivf_state.need_ivf = (config.base.linkage.enabled || do_eval_base);
    if (ivf_state.need_ivf) {
        io::IvfListsPaths ivf_paths;
        ivf_paths.offsets_u64 = (std::filesystem::path(base_basic_state.out_dir) / "ivf_offsets.u64").string();
        ivf_paths.ids_u32 = (std::filesystem::path(base_basic_state.out_dir) / "ivf_ids.u32").string();

        bool need_rebuild_ivf = true;
        if (std::filesystem::exists(ivf_paths.offsets_u64) && std::filesystem::exists(ivf_paths.ids_u32)) {
            if (ivf_state.ivf.Open(ivf_paths, &error)) {
                if (ivf_state.ivf.nlist() == run_state.expected_nlist &&
                    ivf_state.ivf.ntotal() == run_state.base_ntotal_effective) {
                    need_rebuild_ivf = false;
                    // LogInfo("Reusing IVF lists from: " + ivf_paths.offsets_u64 + " and " + ivf_paths.ids_u32);
                }
            }
        }
        if (base_basic_state.rebuilt) {
            need_rebuild_ivf = true;
        }

        if (need_rebuild_ivf) {
            const auto cluster_id_u32 = (std::filesystem::path(base_basic_state.out_dir) / "cluster_id.u32");
            if (!std::filesystem::exists(cluster_id_u32)) {
                LogError("IVF lists are missing/stale and cluster_id.u32 is unavailable for rebuild: " +
                         base_basic_state.out_dir);
                return 1;
            }
            const std::string cluster_id_path = cluster_id_u32.string();
            {
                std::error_code ec;
                const std::uint64_t bytes =
                    static_cast<std::uint64_t>(std::filesystem::file_size(cluster_id_path, ec));
                if (ec || bytes == 0 || (bytes % sizeof(std::uint32_t)) != 0) {
                    LogError("Invalid cluster_id file for IVF build: " + cluster_id_path);
                    return 1;
                }
                const std::uint64_t nbase_cluster_id = bytes / sizeof(std::uint32_t);
                if (nbase_cluster_id != run_state.base_ntotal_effective) {
                    LogError("cluster_id length mismatch for IVF build: expected " +
                             std::to_string(run_state.base_ntotal_effective) + ", got " +
                             std::to_string(nbase_cluster_id));
                    return 1;
                }
            }
            LogInfo("Building IVF lists (CSR) from base_basic/cluster_id.u32 ...");
            const std::string tmp_dir =
                (std::filesystem::path(run_state.effective_run_tmp_root) / "ivf_tmp").string();
            Timer ivf_timer;
            if (!io::BuildIvfListsFromClusterIdFile(cluster_id_path,
                                                   run_state.expected_nlist,
                                                   std::max(1, config.large.cluster_bucket_size),
                                                   ivf_paths,
                                                   tmp_dir,
                                                   /*keep_tmp=*/false,
                                                   &error)) {
                LogError(error);
                return 1;
            }
            LogInfo(ivf_timer.ReportSeconds("IVF lists built in"));
        }

        if (need_rebuild_ivf) {
            if (!ivf_state.ivf.Open(ivf_paths, &error)) {
                LogError(error);
                return 1;
            }
        }
        if (!ivf_state.base_basic.Open(base_basic_state.out_dir, &error)) {
            LogError(error);
            return 1;
        }
    }
    return 0;
}

int RunLargeBaseListStage(const stlq::Config& config,
                          const LargeRunState& run_state,
                          const LargeBaseBasicStageState& base_basic_state,
                          LargeIvfStageState& ivf_state,
                          bool do_eval_base,
                          LargeBaseListStageState& base_list_state,
                          std::string& error) {
    using namespace stlq;
    // NOTE: disk IVF linkage eval uses `linkage_list` directly and does not require `base_list`.
    // This is important for cleanup presets like `eval_both`, which intentionally delete `base_list/`.
    base_list_state.need_base_list = (do_eval_base || config.base.linkage.enabled);
    if (base_list_state.need_base_list) {
        base_list_state.base_list_dir =
            (std::filesystem::path(run_state.effective_run_root) / "base_list").string();
        const std::string base_list_tmp =
            (std::filesystem::path(run_state.effective_run_tmp_root) / "base_list_tmp").string();
        const std::string base_list_meta =
            (std::filesystem::path(base_list_state.base_list_dir) / "meta.bin").string();
        const std::string base_list_hash_path = app::StoreHashPath(base_list_state.base_list_dir);
        base_list_state.base_list_hash =
            app::ComputeBaseListStoreHash(config,
                                          run_state.base_basic_hash,
                                          ivf_state.ivf.nlist(),
                                          ivf_state.ivf.ntotal());
        bool need_rebuild_base_list = true;
        if (std::filesystem::exists(base_list_meta)) {
            if (base_list_state.base_list.Open(base_list_state.base_list_dir, &error)) {
                const bool metadata_fallback_ok =
                    base_list_state.base_list.meta().ntotal == ivf_state.ivf.ntotal() &&
                    base_list_state.base_list.meta().m == config.model.m &&
                    base_list_state.base_list.meta().m_codes == std::max(0, config.model.m - 1);
                app::StoreReuseCheckOptions reuse_opts;
                reuse_opts.label = "base_list";
                reuse_opts.dir = base_list_state.base_list_dir;
                reuse_opts.hash_path = base_list_hash_path;
                reuse_opts.expected_hash = base_list_state.base_list_hash;
                reuse_opts.eval_only = run_state.eval_only;
                reuse_opts.allow_missing_hash_reuse = metadata_fallback_ok;
                reuse_opts.eval_only_hash_mismatch_warning =
                    "base_list store hash mismatch in eval-only mode; proceeding with existing store: " +
                    base_list_state.base_list_dir;
                app::StoreReuseDecision reuse_decision;
                if (!app::CheckExistingStoreReuse(reuse_opts, &reuse_decision, &error)) {
                    LogError(error);
                    return 1;
                }
                need_rebuild_base_list = reuse_decision.rebuild;
                if (reuse_decision.adopted_previous_hash) {
                    base_list_state.base_list_hash = reuse_decision.effective_hash;
                }
            }
        }

        if (base_basic_state.rebuilt) {
            // base_list is derived from base_basic buckets; rebuild it only when we rebuilt base_basic
            // in this run (or when base_list is missing / incompatible). Otherwise, reusing base_list
            // is both correct and required when base_basic buckets were auto-cleaned.
            need_rebuild_base_list = true;
        }
        if (need_rebuild_base_list) {
            LogInfo("Building list-order base store...");
            app::StoreLifecyclePrepareOptions lifecycle_opts;
            lifecycle_opts.label = "base_list";
            lifecycle_opts.dir = base_list_state.base_list_dir;
            lifecycle_opts.extra_remove_dirs.push_back(base_list_tmp);
            lifecycle_opts.protect_existing_outputs = config.large.protect_existing_outputs;
            if (!app::PrepareStoreForRebuildOrResume(lifecycle_opts, &error)) {
                LogError(error);
                return 1;
            }
            Timer list_timer;
            if (!io::BuildBaseListStoreFromBasicBuckets(base_basic_state.out_dir,
                                                       ivf_state.ivf,
                                                       base_list_state.base_list_dir,
                                                       base_list_tmp,
                                                       /*keep_tmp=*/false,
                                                       /*profile_timing=*/config.large.profile_timing,
                                                       &error)) {
                LogError(error);
                return 1;
            }
            if (!app::WriteExpectedStoreHash("base_list",
                                             base_list_hash_path,
                                             base_list_state.base_list_hash,
                                             &error)) {
                LogError(error);
                return 1;
            }
            LogInfo(list_timer.ReportSeconds("Base list-order store built in"));
        }
        if (need_rebuild_base_list) {
            if (!base_list_state.base_list.Open(base_list_state.base_list_dir, &error)) {
                LogError(error);
                return 1;
            }
        }
    }
    return 0;
}

bool BuildLargeEvalMetricsForIndices(const stlq::Config& config,
                                     stlq::EvalMetricMode eval_metric_mode,
                                     const stlq::ColMajorMatrix<int>& query_gt_topk,
                                     const std::vector<int>& gt_first,
                                     const stlq::ColMajorMatrix<int>& indices,
                                     stlq::EvalMetricBundle* metrics_out) {
    using namespace stlq;
    if (!metrics_out) return false;
    const ColMajorMatrix<int>* gt_topk_ptr =
        NeedsFullGroundtruth(eval_metric_mode) ? &query_gt_topk : nullptr;
    std::string metric_err;
    if (!ComputeEvalMetricBundle(gt_first, gt_topk_ptr, indices, eval_metric_mode,
                                 config.eval.metric_gt_topks, metrics_out, &metric_err)) {
        LogWarn(metric_err);
        return false;
    }
    return true;
}

int PrepareLargeDiskEvalQueriesStage(const stlq::Config& config,
                                     const stlq::TrainResult& train_result,
                                     stlq::EvalMetricMode eval_metric_mode,
                                     bool do_eval_base,
                                     bool do_eval_linkage,
                                     LargeDiskEvalQueryState& eval_query_state,
                                     std::string& error) {
    using namespace stlq;
    Dataset query_dataset_raw;
    auto LoadDiskEvalQueriesRaw = [&](Dataset* out) -> bool {
        if (!out) {
            LogError("LoadDiskEvalQueriesRaw: output dataset is null.");
            return false;
        }
        if (out->Xq.cols > 0) {
            return true;
        }
        LogInfo("Loading query set for disk eval: " + config.dataset.name);
        Timer qtimer;
        if (!io::LoadQuerySet(config, &out->Xq, &out->gt, &error)) {
            LogError(error);
            return false;
        }
        LogInfo("Loaded Xq=" + DescribeMatrix(out->Xq) +
                " in " + qtimer.ReportSeconds("time"));
        return true;
    };
    if (do_eval_base || do_eval_linkage) {
        if (!LoadDiskEvalQueriesRaw(&query_dataset_raw)) {
            return 1;
        }
        if (NeedsFullGroundtruth(eval_metric_mode)) {
            if (!io::LoadGroundtruthTopK(config,
                                         query_dataset_raw.Xq.cols,
                                         std::max(1, config.dataset.k),
                                         &eval_query_state.query_gt_topk,
                                         &error)) {
                LogError(error);
                return 1;
            }
        }
    }
    if (do_eval_linkage) {
        if (do_eval_base) {
            eval_query_state.query_dataset = query_dataset_raw;
        } else {
            eval_query_state.query_dataset = std::move(query_dataset_raw);
        }
        eval_query_state.linkage_qt_rotate_sec =
            app::RotateQueriesForEvalWithWarmup(train_result.R, &eval_query_state.query_dataset.Xq);
    }
    if (do_eval_base) {
        eval_query_state.query_dataset_base = std::move(query_dataset_raw);
        eval_query_state.base_qt_rotate_sec =
            app::RotateQueriesForEvalWithWarmup(train_result.R, &eval_query_state.query_dataset_base.Xq);
    }
    return 0;
}

int RunLargeBaseDiskEvalStage(const stlq::Config& config,
                              const stlq::TrainResult& train_result,
                              stlq::EvalMetricMode eval_metric_mode,
                              const stlq::io::IvfListsReader& ivf,
                              const stlq::io::BaseListReader& base_list,
                              LargeDiskEvalQueryState& eval_query_state,
                              std::string& error) {
    using namespace stlq;
    const int warmup = std::max(0, config.eval.base_warmup);
    const int repeat = std::max(1, config.eval.base_repeat);
    const bool bench = (warmup > 0 || repeat > 1);

    if (!bench) {
        RecallResult base_recall;
        stlq::DiskIvfEvalTiming timing{};
        timing.qt_rotate_wall_sec = eval_query_state.base_qt_rotate_sec;
        LogInfo("========= Evaluating base recall (disk IVF) =========");
        LogInfo(std::string("Base recall mode: disk-ivf (nprobe=") +
                std::to_string(std::max(1, config.eval.base_nprobe)) + ")");
        Timer eval_timer;
        if (!EvaluateRecallBaseIvfFromDiskTimed(config,
                                                eval_query_state.query_dataset_base,
                                                ivf,
                                                base_list,
                                                train_result,
                                                &base_recall,
                                                &timing,
                                                &error)) {
            LogWarn(error);
        } else {
            EvalMetricBundle metrics;
            if (BuildLargeEvalMetricsForIndices(config,
                                                eval_metric_mode,
                                                eval_query_state.query_gt_topk,
                                                eval_query_state.query_dataset_base.gt,
                                                base_recall.indices,
                                                &metrics)) {
                PrintEvalMetrics(metrics);
            }
            if (config.eval.bench_quiet) {
                const int nq = eval_query_state.query_dataset_base.Xq.cols;
                const double qps_wall = (timing.core_wall_sec > 0.0)
                    ? (static_cast<double>(nq) / timing.core_wall_sec)
                    : 0.0;
                LogInfo("Base disk recall QPS: qps_wall=" + std::to_string(qps_wall));
            }
        }
        LogInfo(eval_timer.ReportSeconds("Base disk recall finished in"));
        return 0;
    }

    RecallResult base_recall;
    stlq::DiskIvfEvalTiming timing{};
    timing.qt_rotate_wall_sec = eval_query_state.base_qt_rotate_sec;
    std::vector<double> wall_secs;
    wall_secs.reserve(static_cast<std::size_t>(repeat));

    LogInfo("========= Evaluating base recall (disk IVF) =========");
    LogInfo(std::string("Base recall mode: disk-ivf (nprobe=") +
            std::to_string(std::max(1, config.eval.base_nprobe)) + ")");

    // Warmup runs (not reported).
    if (bench && warmup > 0) {
        Config cfg_run = config;
        cfg_run.eval.bench_quiet = true;
        for (int i = 0; i < warmup; ++i) {
            if (!EvaluateRecallBaseIvfFromDiskTimed(cfg_run,
                                                   eval_query_state.query_dataset_base,
                                                   ivf,
                                                   base_list,
                                                   train_result,
                                                   &base_recall,
                                                   &timing,
                                                   &error)) {
                LogWarn(error);
                break;
            }
        }
    }

    // Timed repeats.
    for (int i = 0; i < repeat; ++i) {
        Config cfg_run = config;
        cfg_run.eval.bench_quiet = config.eval.bench_quiet;
        if (!EvaluateRecallBaseIvfFromDiskTimed(cfg_run,
                                               eval_query_state.query_dataset_base,
                                               ivf,
                                               base_list,
                                               train_result,
                                               &base_recall,
                                               &timing,
                                               &error)) {
            LogWarn(error);
            break;
        }
        wall_secs.push_back(timing.core_wall_sec);

        if (cfg_run.eval.bench_quiet) {
            const int nq = eval_query_state.query_dataset_base.Xq.cols;
            const double qps_wall = (timing.core_wall_sec > 0.0) ? (static_cast<double>(nq) / timing.core_wall_sec) : 0.0;
            LogInfo("Base disk recall QPS: run=" + std::to_string(i) +
                    " qps_wall=" + std::to_string(qps_wall));
        }
    }

    if (!wall_secs.empty()) {
        EvalMetricBundle metrics;
        if (BuildLargeEvalMetricsForIndices(config,
                                            eval_metric_mode,
                                            eval_query_state.query_gt_topk,
                                            eval_query_state.query_dataset_base.gt,
                                            base_recall.indices,
                                            &metrics)) {
            PrintEvalMetrics(metrics);
        }
    }
    if (bench && !wall_secs.empty()) {
        const int nq = eval_query_state.query_dataset_base.Xq.cols;
        const double med_wall = app::Median(wall_secs);
        const double med_qps_wall = (med_wall > 0.0) ? (static_cast<double>(nq) / med_wall) : 0.0;
        double sum_wall = 0.0;
        for (double v : wall_secs) sum_wall += v;
        const double avg_wall = sum_wall / static_cast<double>(wall_secs.size());
        const double avg_qps_wall = (avg_wall > 0.0) ? (static_cast<double>(nq) / avg_wall) : 0.0;
        if (config.eval.bench_quiet) {
            LogInfo("Base disk recall QPS summary: nq=" + std::to_string(nq) +
                    " repeat=" + std::to_string(repeat) +
                    " median_qps_wall=" + std::to_string(med_qps_wall) +
                    " avg_qps_wall=" + std::to_string(avg_qps_wall));
        } else {
            LogInfo("Base disk recall bench summary: nq=" + std::to_string(nq) +
                    " repeat=" + std::to_string(repeat) +
                    " | median_wall_sec=" + std::to_string(med_wall) +
                    " median_qps_wall=" + std::to_string(med_qps_wall) +
                    " | avg_wall_sec=" + std::to_string(avg_wall) +
                    " avg_qps_wall=" + std::to_string(avg_qps_wall));
        }
    }
    return 0;
}

int RunLargeBaseListCleanupStage(const stlq::Config& config,
                                 const LargeBaseBasicStageState& base_basic_state,
                                 const LargeBaseListStageState& base_list_state) {
    if (base_list_state.need_base_list) {
        stlq::CleanupAfterBaseListReady(config,
                                          std::filesystem::path(base_basic_state.out_dir),
                                          std::filesystem::path(base_list_state.base_list_dir));
    }
    return 0;
}


}  // namespace stlq
