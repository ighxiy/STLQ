#include "stlq/quantizer/encoder.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

#include "stlq/pipeline/app_utils.h"
#include "stlq/core/blas.h"
#include "stlq/core/kernel_provider.h"
#include "stlq/core/kernel_provider_cpu.h"
#include "stlq/core/kernels_cpu.h"
#include "stlq/core/lapack.h"
#include "stlq/core/threading.h"
#include "stlq/io/base_list_store.h"
#include "stlq/io/base_store.h"
#include "stlq/io/col_block_reader.h"
#include "stlq/io/dataset_io.h"
#include "stlq/io/fbin_reader.h"
#include "stlq/io/ivf_lists.h"
#include "stlq/io/result_io.h"
#include "stlq/linkage/linkage_builder.h"
#include "stlq/linkage/linkage_streaming_builders.h"
#include "stlq/linkage/linkage_summary.h"
#include "stlq/common/logger.h"
#include "stlq/quantizer/spkmeans.h"
#include "stlq/quantizer/encode_base_streaming.h"
#include "stlq/quantizer/precomp_large_root.h"
#include "stlq/quantizer/streaming_train_config.h"
#include "stlq/quantizer/streaming_train_init_stages.h"
#include "stlq/quantizer/streaming_train_iteration_stages.h"
#include "stlq/quantizer/streaming_train_io.h"
#include "stlq/quantizer/update_codebooks_streaming.h"
#include "stlq/quantizer/rvq_layer_store.h"
#include "stlq/common/timer.h"

#if defined(STLQ_ENABLE_CUDA)
#include "stlq/core/kernel_provider_cuda_stream.h"
#include "stlq/linkage/linkage_encode_cuda.h"
#include <cuda_runtime.h>
#endif

namespace stlq
{
    namespace encoder_internal
    {
#include "encoder_internal.inc"
    } // namespace encoder_internal

    namespace
    {
#if defined(STLQ_ENABLE_CUDA)
        bool EncodeLayerCodesToInitStore(int layer,
                                         const ColMajorMatrix<float>& centers,
                                         io::IColBlockReader* x_reader,
                                         const std::vector<ColMajorMatrix<float>>* codebooks,
                                         RvqInitCodesInMemory* codes_store,
                                         CudaStreamKernels* cuda,
                                         const KmeansConfig& kcfg,
                                         bool profile_timing,
                                         KmeansTiming* timing,
                                         std::string* err) {
            if (!x_reader) {
                if (err) *err = "EncodeLayerCodesToInitStore: x_reader is null.";
                return false;
            }
            if (!codes_store) {
                if (err) *err = "EncodeLayerCodesToInitStore: codes_store is null.";
                return false;
            }
            if (!cuda) {
                if (err) *err = "EncodeLayerCodesToInitStore requires CUDA.";
                return false;
            }
            if (centers.rows <= 0 || centers.cols <= 0) {
                if (err) *err = "EncodeLayerCodesToInitStore: empty centers.";
                return false;
            }
            if (x_reader->d() != centers.rows) {
                if (err) *err = "EncodeLayerCodesToInitStore: dim mismatch.";
                return false;
            }
            const std::int64_t n64 = x_reader->n();
            if (n64 <= 0 || codes_store->n() != n64) {
                if (err) *err = "EncodeLayerCodesToInitStore: invalid n/codes_store.";
                return false;
            }

            const int block_cols = (kcfg.block_cols > 0) ? kcfg.block_cols : 200000;
            if (!x_reader->Reset(err)) return false;

            if (layer > 0) {
                if (!codebooks) {
                    if (err) *err = "EncodeLayerCodesToInitStore: codebooks is null for layer>0.";
                    return false;
                }
                cuda->RvqSetPriorCodebooks(*codebooks, layer);
            }

            std::vector<int> assign_blk;
            std::vector<float> best_blk;

            io::ColBlock blk;
            while (true) {
                // Prefer direct-into when available (fvecs and bvecs-u8 readers support this).
                const int d = x_reader->d();
                const io::ColBlockDType dt = x_reader->dtype();
                std::int64_t col0 = 0;
                int cols = 0;
                bool used_direct = false;
                if (dt == io::ColBlockDType::kU8) {
                    std::uint8_t* pinned = cuda->KmeansGetPinnedStageBufferU8(/*slot=*/0, d, block_cols);
                    const std::size_t bytes =
                        sizeof(std::uint8_t) * static_cast<std::size_t>(d) * static_cast<std::size_t>(block_cols);
                    std::string local_err;
                    const bool ok = x_reader->ReadNextInto(block_cols, pinned, bytes, &col0, &cols, &local_err);
                    if (ok) {
                        used_direct = true;
                        if (cols == 0) break;
                        DeviceMatF32View Xd = cuda->KmeansUploadU8AndNormalizePinnedAsync(
                            /*slot=*/0,
                                     pinned,
                                     /*ldB=*/d,
                                     /*rowsB=*/d,
                                     /*colsB=*/cols,
                                     /*h2d_s=*/nullptr,
                                     /*kernel_s=*/nullptr);
                        if (layer > 0) {
                            std::vector<const void*> codes_ptr(static_cast<std::size_t>(layer), nullptr);
                            std::vector<int> code_bytes(static_cast<std::size_t>(layer), 0);
                            for (int l = 0; l < layer; ++l) {
                                std::string span_err;
                                const RvqCodeSpan span = codes_store->SpanBlock(l, col0, cols, &span_err);
                                if (!span_err.empty()) {
                                    if (err) *err = "EncodeLayerCodesToInitStore: SpanBlock failed: " + span_err;
                                    return false;
                                }
                                codes_ptr[static_cast<std::size_t>(l)] = span.ptr;
                                code_bytes[static_cast<std::size_t>(l)] = RvqCodeBytes(span.dtype);
                            }
                            cuda->RvqStageCodesBlockInCopyStreamAfterUpload(/*slot=*/0,
                                cols,
                                layer,
                                codes_ptr.data(),
                                code_bytes.data());
                        }
                        cuda->KmeansComputeWaitForUpload(/*slot=*/0);
                        if (layer > 0) {
                            cuda->RvqProjectXnormBlockOnComputeStream(/*slot=*/0, Xd, cols, layer);
                        }

                        DeviceMatF32View scores{};
                        Timer t_gemm;
                        cuda->GemmDeviceDevicePtrB(/*transA=*/true, /*transB=*/false,
                                                              /*alpha=*/1.0f,
                                                              centers,
                                                              Xd.ptr,
                                                              /*ldB=*/Xd.ld,
                                                              /*rowsB=*/Xd.rows,
                                                              /*colsB=*/Xd.cols,
                                                              /*beta=*/0.0f,
                                                              &scores);
                        if (profile_timing && timing) {
                            cuda->SyncCompute();
                            timing->assign_gemm_s += t_gemm.ElapsedSeconds();
                        }

                        Timer t_arg;
                        cuda->ArgmaxColsF32(scores, &assign_blk, &best_blk);
                        if (profile_timing && timing) {
                            cuda->SyncCompute();
                            timing->assign_argmax_s += t_arg.ElapsedSeconds();
                        }
                        if (!codes_store->WriteBlockFromI32(layer, col0, cols, assign_blk.data(), err)) {
                            return false;
                        }
                        cuda->KmeansMarkUploadSlotConsumed(/*slot=*/0);
                        continue;
                    }
                    if (!local_err.empty()) {
                        if (err) *err = local_err;
                        return false;
                    }
                }
                else {
                    float* pinned = cuda->KmeansGetPinnedStageBuffer(/*slot=*/0, d, block_cols);
                    const std::size_t bytes =
                        sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(block_cols);
                    std::string local_err;
                    const bool ok = x_reader->ReadNextInto(block_cols, pinned, bytes, &col0, &cols, &local_err);
                    if (ok) {
                        used_direct = true;
                        if (cols == 0) break;
                        DeviceMatF32View Xd = cuda->KmeansUploadAndNormalizePinnedAsync(
                            /*slot=*/0,
                                     pinned,
                                     /*ldB=*/d,
                                     /*rowsB=*/d,
                                     /*colsB=*/cols,
                                     /*h2d_s=*/nullptr,
                                     /*kernel_s=*/nullptr);
                        if (layer > 0) {
                            std::vector<const void*> codes_ptr(static_cast<std::size_t>(layer), nullptr);
                            std::vector<int> code_bytes(static_cast<std::size_t>(layer), 0);
                            for (int l = 0; l < layer; ++l) {
                                std::string span_err;
                                const RvqCodeSpan span = codes_store->SpanBlock(l, col0, cols, &span_err);
                                if (!span_err.empty()) {
                                    if (err) *err = "EncodeLayerCodesToInitStore: SpanBlock failed: " + span_err;
                                    return false;
                                }
                                codes_ptr[static_cast<std::size_t>(l)] = span.ptr;
                                code_bytes[static_cast<std::size_t>(l)] = RvqCodeBytes(span.dtype);
                            }
                            cuda->RvqStageCodesBlockInCopyStreamAfterUpload(/*slot=*/0,
                                cols,
                                layer,
                                codes_ptr.data(),
                                code_bytes.data());
                        }
                        cuda->KmeansComputeWaitForUpload(/*slot=*/0);
                        if (layer > 0) {
                            cuda->RvqProjectXnormBlockOnComputeStream(/*slot=*/0, Xd, cols, layer);
                        }

                        DeviceMatF32View scores{};
                        Timer t_gemm;
                        cuda->GemmDeviceDevicePtrB(/*transA=*/true, /*transB=*/false,
                                                              /*alpha=*/1.0f,
                                                              centers,
                                                              Xd.ptr,
                                                              /*ldB=*/Xd.ld,
                                                              /*rowsB=*/Xd.rows,
                                                              /*colsB=*/Xd.cols,
                                                              /*beta=*/0.0f,
                                                              &scores);
                        if (profile_timing && timing) {
                            cuda->SyncCompute();
                            timing->assign_gemm_s += t_gemm.ElapsedSeconds();
                        }
                        Timer t_arg;
                        cuda->ArgmaxColsF32(scores, &assign_blk, &best_blk);
                        if (profile_timing && timing) {
                            cuda->SyncCompute();
                            timing->assign_argmax_s += t_arg.ElapsedSeconds();
                        }
                        if (!codes_store->WriteBlockFromI32(layer, col0, cols, assign_blk.data(), err)) {
                            return false;
                        }
                        cuda->KmeansMarkUploadSlotConsumed(/*slot=*/0);
                        continue;
                    }
                    if (!local_err.empty()) {
                        if (err) *err = local_err;
                        return false;
                    }
                }

                if (used_direct) {
                    // handled above
                    continue;
                }

                // Fallback: reader does not support ReadNextInto (should be rare). ReadNext and memcpy into pinned.
                if (!x_reader->ReadNext(block_cols, &blk, err)) return false;
                cols = (blk.dtype == io::ColBlockDType::kU8) ? blk.X_u8.cols : blk.X_f32.cols;
                if (cols == 0) break;
                col0 = blk.col0;

                DeviceMatF32View Xd{};
                if (blk.dtype == io::ColBlockDType::kU8) {
                    std::uint8_t* pinned = cuda->KmeansGetPinnedStageBufferU8(/*slot=*/0, d, cols);
                    std::memcpy(pinned,
                                blk.X_u8.data.data(),
                                sizeof(std::uint8_t) * static_cast<std::size_t>(d) * static_cast<std::size_t>(cols));
                    Xd = cuda->KmeansUploadU8AndNormalizePinnedAsync(
                        /*slot=*/0, pinned, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                                 /*h2d_s=*/nullptr, /*kernel_s=*/nullptr);
                }
                else {
                    float* pinned = cuda->KmeansGetPinnedStageBuffer(/*slot=*/0, d, cols);
                    std::memcpy(pinned,
                                blk.X_f32.data.data(),
                                sizeof(float) * static_cast<std::size_t>(d) * static_cast<std::size_t>(cols));
                    Xd = cuda->KmeansUploadAndNormalizePinnedAsync(
                        /*slot=*/0, pinned, /*ldB=*/d, /*rowsB=*/d, /*colsB=*/cols,
                                 /*h2d_s=*/nullptr, /*kernel_s=*/nullptr);
                }
                if (layer > 0) {
                    std::vector<const void*> codes_ptr(static_cast<std::size_t>(layer), nullptr);
                    std::vector<int> code_bytes(static_cast<std::size_t>(layer), 0);
                    for (int l = 0; l < layer; ++l) {
                        std::string span_err;
                        const RvqCodeSpan span = codes_store->SpanBlock(l, col0, cols, &span_err);
                        if (!span_err.empty()) {
                            if (err) *err = "EncodeLayerCodesToInitStore: SpanBlock failed: " + span_err;
                            return false;
                        }
                        codes_ptr[static_cast<std::size_t>(l)] = span.ptr;
                        code_bytes[static_cast<std::size_t>(l)] = RvqCodeBytes(span.dtype);
                    }
                    cuda->RvqStageCodesBlockInCopyStreamAfterUpload(/*slot=*/0,
                                                                             cols,
                                                                             layer,
                                                                             codes_ptr.data(),
                                                                             code_bytes.data());
                }
                cuda->KmeansComputeWaitForUpload(/*slot=*/0);
                if (layer > 0) {
                    cuda->RvqProjectXnormBlockOnComputeStream(/*slot=*/0, Xd, cols, layer);
                }
                DeviceMatF32View scores{};
                cuda->GemmDeviceDevicePtrB(/*transA=*/true, /*transB=*/false,
                                                      /*alpha=*/1.0f,
                                                      centers,
                                                      Xd.ptr,
                                                      /*ldB=*/Xd.ld,
                                                      /*rowsB=*/Xd.rows,
                                                      /*colsB=*/Xd.cols,
                                                      /*beta=*/0.0f,
                                                      &scores);
                cuda->ArgmaxColsF32(scores, &assign_blk, &best_blk);
                if (!codes_store->WriteBlockFromI32(layer, col0, cols, assign_blk.data(), err)) {
                    return false;
                }
                cuda->KmeansMarkUploadSlotConsumed(/*slot=*/0);
            }

            return true;
        }
#else
[[maybe_unused]] static bool EncodeLayerCodesToInitStore(int /*layer*/,
                                       const ColMajorMatrix<float>& /*centers*/,
                                       io::IColBlockReader* /*x_reader*/,
                                       const std::vector<ColMajorMatrix<float>>* /*codebooks*/,
                                       RvqInitCodesInMemory* /*codes_store*/,
                                       void* /*cuda*/,
                                       const KmeansConfig& /*kcfg*/,
                                       bool /*profile_timing*/,
                                       KmeansTiming* /*timing*/,
                                       std::string* err) {
    if (err) *err = "EncodeLayerCodesToInitStore requires STLQ_ENABLE_CUDA build.";
    return false;
}
#endif
    } // namespace

    bool TrainQuantizerStreamingLarge(const Config& config,
                                      const std::filesystem::path& exp_root,
                                      KernelProvider* kernels,
                                      StreamKernelProvider* stream_kernels,
                                      TrainResult* result,
                                      std::string* error) {
        (void)kernels;
        if (!result) {
            if (error) *error = "TrainQuantizerStreamingLarge: TrainResult output is null.";
            return false;
        }
        if (!config.large.enabled) {
            if (error) *error = "TrainQuantizerStreamingLarge requires large.enabled=true.";
            return false;
        }
        if (config.model.h0_one <= 0 || config.model.h0_one > 256) {
            if (error)
                *error =
                    "TrainQuantizerStreamingLarge: model.h0_one must be in [1,256] (code0_one stored as uint8).";
            return false;
        }

        StreamingTrainInput train_input;
        if (!OpenStreamingTrainInput(config, &train_input, error)) {
            return false;
        }
        const int d = train_input.d;
        const std::uint64_t ntrain = train_input.ntrain;

        int resume_base_R_iters = 0;
        bool resume_need_init_linkage = false;
        if (!config.train.init_enabled) {
            // Resume mode: skip the entire init stage (RVQ + basic init + init-linkage) and continue OPQ iterations
            // from an existing train checkpoint.
            std::string train_load_date = config.io.load_date;
            std::string train_load_seq = config.io.load_seq;
            if (config.io.train_file_set && !config.io.train_file.empty()) {
                std::error_code ec;
                const std::filesystem::path train_path(config.io.train_file);
                const bool train_path_exists = std::filesystem::exists(train_path, ec);
                if (train_path_exists && !ec) {
                    train_load_date = "raw";
                    train_load_seq.clear();
                }
            }
            {
                std::string local_err;
                const std::string resolved =
                    io::MakeDatedPathForLoad(config.io.train_file, train_load_date, train_load_seq, &local_err);
                if (!resolved.empty()) {
                    result->loaded_train_h5 = resolved;
                }
            }
            if (!io::LoadTrainResults(config.io.train_file,
                                      train_load_date,
                                      train_load_seq,
                                      config.model.m,
                                      result,
                                      error)) {
                return false;
            }
            if (result->R.rows != d || result->R.cols != d) {
                if (error) *error = "TrainQuantizerStreamingLarge(resume): checkpoint R shape mismatch.";
                return false;
            }
            if (result->C_root.d != d || result->C_one.d != d) {
                if (error) *error = "TrainQuantizerStreamingLarge(resume): checkpoint codebook dim mismatch.";
                return false;
            }
            auto want_h_vec_one = config.model.h_vec;
            if (!want_h_vec_one.empty()) {
                want_h_vec_one[0] = config.model.h0_one;
            }
            if (result->checkpoint_stage == 0 &&
                (result->C_root.h_vec != config.model.h_vec || result->C_one.h_vec != want_h_vec_one)) {
                auto fmt_hvec = [](const std::vector<int>& v) -> std::string
                {
                    std::ostringstream oss;
                    oss << "[";
                    for (std::size_t i = 0; i < v.size(); ++i) {
                        if (i) oss << ", ";
                        oss << v[i];
                    }
                    oss << "]";
                    return oss.str();
                };
                if (error) {
                    std::ostringstream oss;
                    oss << "TrainQuantizerStreamingLarge(resume): checkpoint h_vec mismatch.\n"
                        << "  checkpoint.C_root.h_vec = " << fmt_hvec(result->C_root.h_vec) << "\n"
                        << "  checkpoint.C_one.h_vec  = " << fmt_hvec(result->C_one.h_vec) << "\n"
                        << "  config.model.h_vec      = " << fmt_hvec(config.model.h_vec) << "\n"
                        << "  want_h_vec_one          = " << fmt_hvec(want_h_vec_one) << " (model.h0_one override)";
                    *error = oss.str();
                }
                return false;
            }
            if (result->checkpoint_stage == 1 && result->C_root.h_vec != config.model.h_vec) {
                if (error)
                    *error =
                        "TrainQuantizerStreamingLarge(resume): pre-init-linkage checkpoint C_root.h_vec mismatch.";
                return false;
            }
            resume_base_R_iters = std::max(0, result->R_iters);
            resume_need_init_linkage = (result->checkpoint_stage == 1);
            const std::uint64_t want_sig = ComputeTrainResumeSigU64(config);
            if (result->resume_sig_u64 != 0 && want_sig != result->resume_sig_u64) {
                if (error) {
                    std::ostringstream oss;
                    oss << "TrainQuantizerStreamingLarge(resume): checkpoint config signature mismatch.\n"
                        << "  checkpoint.meta/train/resume_sig_hex = 0x" << std::hex << result->resume_sig_u64 <<
                        std::dec << "\n"
                        << "  current ComputeTrainResumeSigU64(config) = 0x" << std::hex << want_sig << std::dec << "\n"
                        << "Hint: ensure train/model/virtual/hnsw/dataset semantic knobs match; "
                        "train.max_R_iters/train.init_enabled/runtime/io/large knobs are ignored by this signature.";
                    *error = oss.str();
                }
                return false;
            }
            LogInfo("Train streaming: init disabled; resuming from checkpoint (stage=" +
                std::to_string(result->checkpoint_stage) + ", R_iters=" +
                std::to_string(resume_base_R_iters) + ").");
        }
        else {
            if (!RunStreamingTrainRvqInitStage(config,
                                               d,
                                               ntrain,
                                               train_input,
                                               kernels,
                                               stream_kernels,
                                               *result,
                                               error)) {
                return false;
            }
        }
        result->final_linkaged_mse_rot = -1.0f;

        if (config.train.exit_after_rvq_init) {
            LogInfo("Train: exit_after_rvq_init=1 (stop after RVQ init).");
            return true;
        }

        const StreamingTrainWorkspacePaths train_paths =
            ResolveStreamingTrainWorkspacePaths(config, exp_root, *result);
        EnsureStreamingTrainWorkspaceDirs(train_paths);
        const std::filesystem::path& exp_root_run = train_paths.exp_root_run;

        // Baseline stage (R=I): basic -> update C_root -> build ivf/list -> init-linkage -> init C_one -> linkage -> update C_one.
        // Global R iterations (OPQ): repeat the same loop, with optional rotation updates.
        const int nlist = config.model.h_vec.empty() ? 0 : config.model.h_vec[0];
        if (nlist <= 0) {
            if (error) *error = "TrainQuantizerStreamingLarge: invalid h_vec[0] (nlist).";
            return false;
        }
        const StreamingTrainIterationPlan iteration_plan =
            MakeStreamingTrainIterationPlan(config, ntrain, resume_base_R_iters);

        bool logged_linkage_stage_cpu_notice = false;
        bool early_exit_after_ckpt_init_basic = false;

        if (config.train.init_enabled) {
            // Baseline is a standalone stage and is not counted in train.max_R_iters.
            LogInfo("(2)Train basic + init C_one (R=I)...");
            if (!RunStreamingTrainRoundStage(config,
                                             *result,
                                             train_paths,
                                             iteration_plan,
                                             /*global_iter=*/-1,
                                             /*do_init_linkage_and_c1=*/true,
                                             /*do_opq_update=*/false,
                                             d,
                                             nlist,
                                             ntrain,
                                             train_input,
                                             stream_kernels,
                                             &logged_linkage_stage_cpu_notice,
                                             &early_exit_after_ckpt_init_basic,
                                             error)) {
                return false;
            }
            if (early_exit_after_ckpt_init_basic) {
                // Profiling-only checkpoint: keep intermediates intact and stop before init-linkage.
                return true;
            }

            if (!RunStreamingTrainInitCheckpointStage(config, *result, iteration_plan, error)) {
                return false;
            }
        }

        // Resume from a pre-init-linkage checkpoint (C_root+R only): run init-linkage now to (re)build C_one.
        if (!config.train.init_enabled && resume_need_init_linkage) {
            if (!RunStreamingTrainResumePreInitLinkageStage(config,
                                                            *result,
                                                            train_paths,
                                                            iteration_plan,
                                                            train_input,
                                                            stream_kernels,
                                                            &logged_linkage_stage_cpu_notice,
                                                            error)) {
                return false;
            }
        }

        if (!RunStreamingTrainGlobalIterationLoopStage(config,
                                                       *result,
                                                       train_paths,
                                                       iteration_plan,
                                                       d,
                                                       nlist,
                                                       ntrain,
                                                       train_input,
                                                       stream_kernels,
                                                       &logged_linkage_stage_cpu_notice,
                                                       error)) {
            return false;
        }

        // Optional: delete large training intermediates (train_basic/train_list/etc) after a successful run.
        // These are always regenerable; the canonical artifact is the saved train result (h5) if enabled.
        CleanupTrainIntermediates(config, exp_root_run);

        return true;
    }
} // namespace stlq
